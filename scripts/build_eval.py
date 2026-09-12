"""生成评测期望值：跑真实 MCP 工具，把结果冻结进 eval/ground_truth.json。

用法：uv run python scripts/build_eval.py
任何口径改动（SQL / 别名 / 口径文件 / 数据重载）后都应重新生成，并在提交信息里说明差异。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import eval as evalset  # noqa: E402
from eval import executor  # noqa: E402

from agent.mcp_hub import MCPHub  # noqa: E402

FINGERPRINT_FILES = [
    "servers/geo_compute/query.py",
    "servers/geo_compute/server.py",
    "servers/geo_knowledge/poi_scope.json",
    "servers/geo_knowledge/categories/aliases.json",
]


def fingerprint() -> dict:
    """冻结时把「口径所在的那几个文件」的哈希与数据文件状态记下来。

    这样 run_eval 报错时能立刻分清是代码/口径变了，还是数据换了。
    """
    out = {}
    for rel in FINGERPRINT_FILES:
        p = ROOT / rel
        out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else None
    db = ROOT / "data" / "processed" / "geo.duckdb"
    out["data/processed/geo.duckdb"] = (
        f"{db.stat().st_size}:{int(db.stat().st_mtime)}" if db.exists() else None)
    return out


async def build() -> int:
    cases = evalset.load_cases()
    rows: dict[str, dict] = {}
    problems: list[str] = []
    async with MCPHub() as hub:
        for case in cases:
            result = await executor.run_case(hub, case)
            expected = evalset.project(result["tool"], result["payload"]) if result["tool"] else {}
            row = {
                "kind": case["kind"],
                "tool": result["tool"],
                "args": evalset.plainify(result["args"]),
                "ok": result["ok"],
                "error": result["error"],
                "center": evalset.plainify(result["center"]),
                "expected": expected,
            }
            rows[case["id"]] = row

            if case.get("expect_error"):
                if result["ok"]:
                    problems.append(f"{case['id']}: 期望报错却成功了")
                elif case["expect_error"] not in (result["error"] or ""):
                    problems.append(f"{case['id']}: 错误信息里没有「{case['expect_error']}」"
                                    f"（实际：{result['error']}）")
            elif case["kind"] != "refusal" and not result["ok"]:
                problems.append(f"{case['id']}: 调用失败 {result['error']}")

            cross = case.get("cross_check")
            if cross and not expected.get("error"):
                got = expected.get("count_total")
                if got != cross["value"]:
                    problems.append(f"{case['id']}: 与 {cross['source']} 记录的 {cross['value']} "
                                    f"不一致，实际 {got}")
                    row["cross_check"] = {"expected": cross["value"], "actual": got,
                                          "source": cross["source"], "passed": False}
                else:
                    row["cross_check"] = {"expected": cross["value"], "actual": got,
                                          "source": cross["source"], "passed": True}
            print(f"  {'OK ' if result['ok'] else 'ERR'} {case['id']:<14} "
                  f"{json.dumps(expected.get('count_total', expected.get('planar_distance_m', '')),
                                ensure_ascii=False)}", flush=True)

    payload = {
        "version": "1.0",
        "generated": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "generator": "scripts/build_eval.py",
        "fingerprint": fingerprint(),
        "cases": rows,
    }
    evalset.GROUND_TRUTH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print(f"\n已写入 {evalset.GROUND_TRUTH.relative_to(ROOT)}：{len(rows)} 条")
    if problems:
        print(f"\n需要人过目的 {len(problems)} 处：")
        for p in problems:
            print(f"  [CHECK] {p}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(build()))
