"""评测运行器：data 层守住口径，agent 层守住问答行为。

用法：
    uv run python scripts/run_eval.py                    # data 层：不花 token，改一次口径就该跑一次
    uv run python scripts/run_eval.py --mode agent       # agent 层：真跑 Agent，会花 token
    uv run python scripts/run_eval.py --mode agent --kind nearby,aggregate --limit 6
    uv run python scripts/run_eval.py --mode agent --only nearby-002,refusal-001

data 层与 eval/ground_truth.json 逐字段精确比对：任何 SQL / 别名 / 口径文件 / 数据重载引起的
变化都会显形，并在报告里区分「口径文件变了」与「期望值变了」。
agent 层的判定是确定性的（工具是否调对、数字能否溯源、关键数字是否落到答案里、该拒绝的是否拒绝），
refusal / tool_error 两类用关键词做代理判定，不是语义判定。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import eval as evalset  # noqa: E402
from eval import executor  # noqa: E402

from agent import report  # noqa: E402
from agent.config import Settings  # noqa: E402
from agent.loop import run  # noqa: E402
from agent.mcp_hub import MCPHub  # noqa: E402

DEFAULT_OUTDIR = ROOT / "eval" / "runs"


def select(cases: list[dict], args) -> list[dict]:
    out = cases
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        out = [c for c in out if c["id"] in wanted]
        missing = wanted - {c["id"] for c in out}
        if missing:
            raise SystemExit(f"题库里没有这些用例：{sorted(missing)}")
    if args.kind:
        kinds = {x.strip() for x in args.kind.split(",") if x.strip()}
        out = [c for c in out if c["kind"] in kinds]
    if args.limit:
        out = out[:args.limit]
    if not out:
        raise SystemExit("筛选后没有用例，检查 --only / --kind / --limit")
    return out


def fingerprint_diff(frozen: dict) -> list[str]:
    from scripts.build_eval import fingerprint  # noqa: PLC0415

    now = fingerprint()
    diffs = []
    for key, was in (frozen or {}).items():
        if now.get(key) != was:
            diffs.append(f"{key}: 冻结时 {was} → 现在 {now.get(key)}")
    return diffs


async def run_data(cases: list[dict], truth: dict, quiet: bool) -> int:
    rows, problems, skipped = [], [], []
    async with MCPHub() as hub:
        for case in cases:
            if case["kind"] == "refusal":
                # 拒答是 Agent 行为，data 层没有可比的期望值，跳过而不是假装通过
                skipped.append(case["id"])
                if not quiet:
                    print(f"  [SKIP] {case['id']:<14} refusal    只由 agent 层判定")
                continue
            result = await executor.run_case(hub, case)
            expect = (truth["cases"].get(case["id"]) or {}).get("expected") or {}
            actual = evalset.project(result["tool"], result["payload"]) if result["tool"] else {}
            mismatches = evalset.compare(expect, actual)
            if case.get("expect_error"):
                if result["ok"]:
                    mismatches.append("期望报错，实际调用成功")
                elif case["expect_error"] not in (result["error"] or ""):
                    mismatches.append(f"错误信息里没有「{case['expect_error']}」"
                                      f"（实际：{result['error']}）")
            elif not result["ok"]:
                mismatches.append(f"调用失败：{result['error']}")
            passed = not mismatches
            rows.append({"id": case["id"], "kind": case["kind"], "passed": passed,
                         "problems": mismatches})
            problems.extend(f"{case['id']}: {m}" for m in mismatches)
            if not quiet:
                detail = mismatches[0] if mismatches else "与冻结值一致"
                print(f"  [{'PASS' if passed else 'FAIL'}] {case['id']:<14} {case['kind']:<10} "
                      f"{detail}", flush=True)

    passed = sum(1 for r in rows if r["passed"])
    tail = f"，另有 {len(skipped)} 条只由 agent 层判定（{', '.join(skipped)}）" if skipped else ""
    print(f"\ndata 层：{passed}/{len(rows)} 通过{tail}")
    if problems:
        print(f"\n不一致 {len(problems)} 处：")
        for p in problems:
            print(f"  - {p}")
    diffs = fingerprint_diff(truth.get("fingerprint") or {})
    if diffs:
        print("\n口径所在文件在冻结之后有变动（不一致可能就是它引起的）：")
        for d in diffs:
            print(f"  - {d}")
    return 0 if passed == len(rows) else 2


def write_case_artifacts(run_result, checks, case_dir: Path) -> None:
    from agent.loop import build_visual_hints  # noqa: PLC0415

    case_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (case_dir / "report.md").write_text(
        report.render(run_result, checks, str(case_dir), ts), encoding="utf-8")
    (case_dir / "trace.json").write_text(
        json.dumps(run_result.to_dict(), ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    (case_dir / "visual_hints.json").write_text(
        json.dumps(build_visual_hints(run_result), ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")


async def run_agent(cases: list[dict], truth: dict, args) -> int:
    settings = Settings()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.outdir) / f"agent_{ts}"
    rows, tokens = [], 0

    async with MCPHub() as hub:
        for case in cases:
            def on_event(invocation):
                mark = "OK " if invocation.ok else "ERR"
                print(f"    [{mark}] {invocation.name}({json.dumps(invocation.arguments, ensure_ascii=False)[:90]})")

            run_result = await run(case["question"], hub, settings,
                                   max_steps=args.max_steps, on_event=on_event)
            checks = report.checks_for(run_result)
            expected = (truth["cases"].get(case["id"]) or {}).get("expected") or {}
            scores = evalset.score_answer(case, expected, run_result, run_result.answer, checks)
            passed = all(ok for _, ok, _ in scores)
            tokens += int((run_result.usage or {}).get("total_tokens") or 0)
            write_case_artifacts(run_result, checks, run_dir / case["id"])
            rows.append({
                "id": case["id"], "kind": case["kind"], "question": case["question"],
                "passed": passed, "answer": run_result.answer,
                "stopped": run_result.stopped, "steps": run_result.steps,
                "usage": run_result.usage, "checks": scores,
            })
            print(f"  [{'PASS' if passed else 'FAIL'}] {case['id']:<14} {case['kind']:<10} "
                  f"{' '.join(f'{n}={ok}' for n, ok, _ in scores)}", flush=True)
            if not passed:
                for name, ok, detail in scores:
                    if not ok:
                        print(f"        × {name}: {detail}")

    summary = evalset.summarize_scores(rows)
    summary.update({"generated": ts, "mode": "agent", "tokens_total": tokens,
                    "run_dir": str(run_dir), "max_steps": args.max_steps})
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps({"summary": summary, "cases": rows}, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")

    lines = ["# Agent 层评测", "",
             f"- 时间（UTC）：{ts}",
             f"- 用例：{summary['passed']}/{summary['total']} 通过",
             f"- 逐项通过率：{json.dumps(summary['by_check'], ensure_ascii=False)}",
             f"- token 合计：{tokens}",
             f"- 逐题产物：{run_dir}",
             "", "| 用例 | 类别 | 结果 | 未通过项 |", "|---|---|---|---|"]
    for row in rows:
        bad = "；".join(f"{n}: {d}" for n, ok, d in row["checks"] if not ok) or "-"
        lines.append(f"| {row['id']} | {row['kind']} | {'通过' if row['passed'] else '未通过'} | {bad} |")
    lines += ["", "> refusal / tool_error 两类用关键词做代理判定，不是语义判定。"]
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nagent 层：{summary['passed']}/{summary['total']} 通过，token 合计 {tokens}")
    print(f"逐项通过率：{json.dumps(summary['by_check'], ensure_ascii=False)}")
    if summary["failed_ids"]:
        print(f"未通过：{summary['failed_ids']}")
    print(f"产物目录：{run_dir}")
    return 0 if summary["failed"] == 0 else 2


async def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["data", "agent"], default="data")
    ap.add_argument("--only", default=None, help="逗号分隔的用例 id")
    ap.add_argument("--kind", default=None, help="逗号分隔的问法类别")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    truth = evalset.load_ground_truth()
    if truth is None:
        raise SystemExit("缺少 eval/ground_truth.json，先跑：uv run python scripts/build_eval.py")
    cases = select(evalset.load_cases(), args)
    print(f"模式 {args.mode}，选中 {len(cases)} 条用例"
          f"（{', '.join(sorted({c['kind'] for c in cases}))}）\n")
    if args.mode == "data":
        return await run_data(cases, truth, args.quiet)
    return await run_agent(cases, truth, args)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
