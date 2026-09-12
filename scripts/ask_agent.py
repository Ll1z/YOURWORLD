"""Agent 入口：自然语言提问 → 选 MCP 工具 → 出答案 + 空间自检 + 可复现产物。

用法：
    uv run python scripts/ask_agent.py "东城区有哪些医院在 1 公里内"
    uv run python scripts/ask_agent.py "五个区各有多少家便利店" --max-steps 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone

from agent import selfcheck
from agent.config import Settings
from agent.loop import AgentRun, build_visual_hints, run
from agent.mcp_hub import MCPHub


def _selfcheck(run_result: AgentRun) -> list[selfcheck.Check]:
    last = next((i for i in reversed(run_result.invocations)
                 if i.ok and i.name == "query_nearby"), None)
    checks: list[selfcheck.Check] = []
    if last is None:
        checks.append(selfcheck.Check("crs", True, "跳过：本次运行没有 query_nearby 结果"))
        checks.append(selfcheck.Check("units", True, "跳过：本次运行没有 query_nearby 结果"))
        checks.append(selfcheck.Check("geometry", True, "跳过：本次运行没有 query_nearby 结果"))
        checks.append(selfcheck.Check("magnitude", True, "跳过：本次运行没有 query_nearby 结果"))
    else:
        checks.extend(selfcheck.run_all(last.result))

    pool = [run_result.context, {"question": run_result.question}, *run_result.tool_results()]
    checks.append(selfcheck.check_grounding(run_result.answer, pool))
    return checks


def _render_report(run_result: AgentRun, checks: list[selfcheck.Check], run_dir: str, ts: str) -> str:
    lines = []
    lines.append("# GeoAnalyst 问答报告")
    lines.append("")
    lines.append(f"- 生成时间（UTC）：{ts}")
    lines.append(f"- 问题：{run_result.question}")
    lines.append(f"- 循环：# {run_result.steps} 步，结束方式 {run_result.stopped}"
                 f"（final = 模型给出最终回答，max_steps = 触顶）")
    lines.append(f"- 数值溯源修复轮次：{run_result.repair_rounds}"
                 f"（回答里出现无法溯源的数字时会被打回重写，最多 2 轮）")
    lines.append(f"- token 用量：{run_result.usage}")
    lines.append("")
    lines.append("## 结论（模型叙述）")
    lines.append("")
    lines.append(run_result.answer.strip() or "（模型未给出回答）")
    lines.append("")
    lines.append("## 空间自检")
    lines.append("")
    lines.append(selfcheck.render(checks))
    lines.append("")
    lines.append("## 工具调用轨迹")
    lines.append("")
    lines.append("| # | Server | 工具 | 参数 | 结果 | 耗时(s) |")
    lines.append("|---|---|---|---|---|---|")
    for k, inv in enumerate(run_result.invocations, 1):
        args = json.dumps(inv.arguments, ensure_ascii=False)
        outcome = "成功" if inv.ok else f"失败：{inv.error}"
        if inv.ok:
            outcome += f"（{json.dumps(inv.result, ensure_ascii=False, default=str)[:120]}…）"
        lines.append(f"| {k} | {inv.server} | {inv.name} | `{args}` | {outcome} | {inv.elapsed_s} |")
    if not run_result.invocations:
        lines.append("| - | - | - | - | 本次未调用任何工具 | - |")
    lines.append("")
    lines.append("## 产物")
    lines.append("")
    lines.append(f"- `{os.path.join(run_dir, 'report.md')}`：本报告")
    lines.append(f"- `{os.path.join(run_dir, 'trace.json')}`：完整轨迹（含每次工具调用的参数与结果摘要）")
    lines.append(f"- `{os.path.join(run_dir, 'visual_hints.json')}`：Cesium 前端预留字段（camera 飞向查询点 + markers 标记附近内容），Stage 1 只落盘不渲染")
    lines.append("")
    lines.append("## 硬性规则声明")
    lines.append("")
    lines.append("- 本报告所有数值均来自 MCP 工具返回，模型不得生成数字；`grounding` 检查列出无法溯源的数字。")
    lines.append("- 数据来源：OpenStreetMap 北京省级切片 2026-09-11（ODbL 1.0），行政边界见 `servers/geo_catalog/cards/cn-bj-adm5-wgs84.json`。")
    lines.append("- CRS：存储 EPSG:4326，距离与缓冲区 EPSG:32650（UTM 50N）；面积口径为大地线面积。")
    lines.append("- 覆盖范围：北京五区（东城 / 西城 / 朝阳 / 丰台 / 海淀）。")
    return "\n".join(lines) + "\n"


async def main() -> int:
    # Windows 下 stdout 被重定向到文件时默认走 ANSI 代码页，报告里的 m² 会直接抛 UnicodeEncodeError
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("question", help="自然语言问题")
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--outdir", default=os.path.join("data", "processed", "results"))
    ap.add_argument("--quiet", action="store_true", help="不实时打印工具调用")
    args = ap.parse_args()

    settings = Settings()

    def on_event(inv):
        if not args.quiet:
            mark = "OK " if inv.ok else "ERR"
            print(f"  [{mark}] {inv.server}.{inv.name}({json.dumps(inv.arguments, ensure_ascii=False)}) "
                  f"{inv.elapsed_s}s", flush=True)

    async with MCPHub() as hub:
        if not args.quiet:
            print(f"已连接 {len(hub.sessions)} 个 MCP Server，聚合 {len(hub.tools)} 个工具："
                  f"{', '.join(sorted(hub.tools))}")
        run_result = await run(args.question, hub, settings, max_steps=args.max_steps,
                               on_event=on_event)

    checks = _selfcheck(run_result)
    hints = build_visual_hints(run_result)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(args.outdir, f"agent_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    report = _render_report(run_result, checks, run_dir, ts)
    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(run_dir, "trace.json"), "w", encoding="utf-8") as f:
        json.dump(run_result.to_dict(), f, ensure_ascii=False, indent=1, default=str)
    with open(os.path.join(run_dir, "visual_hints.json"), "w", encoding="utf-8") as f:
        json.dump(hints, f, ensure_ascii=False, indent=1, default=str)

    print()
    print(report)
    print(f"产物目录: {run_dir}")
    return 0 if all(c.passed for c in checks) else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
