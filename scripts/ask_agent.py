"""Agent 入口：自然语言提问 → 选 MCP 工具 → 出答案 + 空间自检 + 可复现产物。

用法：
    uv run python scripts/ask_agent.py "东城区有哪些医院在 1 公里内"
    uv run python scripts/ask_agent.py "五个区各有多少家便利店" --max-steps 4

产物与自检逻辑在 agent/report.py，与 web/server.py 共用同一份实现。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from agent import report
from agent.config import Settings
from agent.loop import run
from agent.mcp_hub import MCPHub


async def main() -> int:
    # Windows 下 stdout 被重定向到文件时默认走 ANSI 代码页，报告里的 m² 会直接抛 UnicodeEncodeError
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("question", help="自然语言问题")
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--outdir", default=report.DEFAULT_OUTDIR)
    ap.add_argument("--quiet", action="store_true", help="不实时打印工具调用")
    args = ap.parse_args()

    settings = Settings()

    def on_event(invocation):
        if not args.quiet:
            mark = "OK " if invocation.ok else "ERR"
            print(f"  [{mark}] {invocation.server}.{invocation.name}"
                  f"({json.dumps(invocation.arguments, ensure_ascii=False)}) "
                  f"{invocation.elapsed_s}s", flush=True)

    async with MCPHub() as hub:
        if not args.quiet:
            print(f"已连接 {len(hub.sessions)} 个 MCP Server，聚合 {len(hub.tools)} 个工具："
                  f"{', '.join(sorted(hub.tools))}")
        run_result = await run(args.question, hub, settings,
                               max_steps=args.max_steps, on_event=on_event)

    artifacts = report.persist(run_result, args.outdir)
    print()
    print(artifacts.report_text)
    print(f"产物目录: {artifacts.run_dir}")
    return 0 if artifacts.passed else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))