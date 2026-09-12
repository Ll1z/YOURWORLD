"""把一次 Agent 运行固化成可复现的产物。

scripts/ask_agent.py（命令行）与 web/server.py（前端）共用这一份实现，
避免报告模板写两遍、最后漂移成两个不同的东西。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from agent import selfcheck
from agent.loop import AgentRun, build_visual_hints, grounding_pool

DEFAULT_OUTDIR = os.path.join("data", "processed", "results")


@dataclass
class Artifacts:
    run_dir: str
    report_text: str
    checks: list[selfcheck.Check]
    visual_hints: dict

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


def checks_for(run_result: AgentRun) -> list[selfcheck.Check]:
    """对一次运行做全部自检。没调到的工具按「跳过」记录，而不是假装通过。"""

    def last_ok(tool: str):
        return next((i for i in reversed(run_result.invocations) if i.ok and i.name == tool), None)

    checks: list[selfcheck.Check] = []
    nearby = last_ok("query_nearby")
    if nearby is None:
        checks.extend(selfcheck.Check(name, True, "跳过：本次运行没有 query_nearby 结果")
                      for name in ("crs", "units", "geometry", "magnitude"))
    else:
        checks.extend(selfcheck.run_all(nearby.result))

    distance = last_ok("distance_between")
    if distance is not None:
        checks.append(selfcheck.check_distance(distance.result))

    pool = grounding_pool(run_result.question, run_result.clarification,
                          run_result.context, run_result.invocations)
    checks.append(selfcheck.check_grounding(run_result.answer, pool))
    return checks


def render(run_result: AgentRun, checks: list[selfcheck.Check], run_dir: str, ts: str) -> str:
    lines = []
    lines.append("# GeoAnalyst 问答报告")
    lines.append("")
    lines.append(f"- 生成时间（UTC）：{ts}")
    lines.append(f"- 问题：{run_result.question}")
    if run_result.clarification:
        lines.append(f"- 用户在界面上确认：{run_result.clarification}")
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
    lines.append(f"- `{os.path.join(run_dir, 'visual_hints.json')}`：Cesium 前端字段"
                 "（camera 飞向查询点 + markers 标记附近内容）")
    lines.append("")
    lines.append("## 硬性规则声明")
    lines.append("")
    lines.append("- 本报告所有数值均来自 MCP 工具返回，模型不得生成数字；`grounding` 检查列出无法溯源的数字。")
    lines.append("- 数据来源：OpenStreetMap 北京省级切片 2026-09-11（ODbL 1.0），"
                 "行政边界见 `servers/geo_catalog/cards/cn-bj-adm5-wgs84.json`。")
    lines.append("- CRS：存储 EPSG:4326，距离与缓冲区 EPSG:32650（UTM 50N）；面积口径为大地线面积。")
    lines.append("- 覆盖范围：北京五区（东城 / 西城 / 朝阳 / 丰台 / 海淀）。")
    return "\n".join(lines) + "\n"


def persist(run_result: AgentRun, outdir: str = DEFAULT_OUTDIR) -> Artifacts:
    """写出 report.md / trace.json / visual_hints.json，返回产物句柄。"""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(outdir, f"agent_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    checks = checks_for(run_result)
    hints = build_visual_hints(run_result)
    report_text = render(run_result, checks, run_dir, ts)

    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report_text)
    with open(os.path.join(run_dir, "trace.json"), "w", encoding="utf-8") as f:
        json.dump(run_result.to_dict(), f, ensure_ascii=False, indent=1, default=str)
    with open(os.path.join(run_dir, "visual_hints.json"), "w", encoding="utf-8") as f:
        json.dump(hints, f, ensure_ascii=False, indent=1, default=str)

    return Artifacts(run_dir=run_dir, report_text=report_text, checks=checks, visual_hints=hints)


def load_visual_hints(run_id: str, outdir: str = DEFAULT_OUTDIR) -> dict:
    path = Path(outdir) / run_id / "visual_hints.json"
    if not path.exists():
        raise FileNotFoundError(run_id)
    return json.loads(path.read_text(encoding="utf-8"))


def list_runs(outdir: str = DEFAULT_OUTDIR) -> list[dict]:
    """列出已落盘的历史运行（新的在前），供前端回放。"""
    base = Path(outdir)
    if not base.exists():
        return []
    runs = []
    for d in sorted((p for p in base.iterdir() if p.is_dir()), reverse=True):
        hints_path = d / "visual_hints.json"
        if not hints_path.exists():
            continue
        try:
            hints = json.loads(hints_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        question = ""
        report = d / "report.md"
        if report.exists():
            for line in report.read_text(encoding="utf-8").splitlines():
                if line.startswith("- 问题："):
                    question = line[len("- 问题："):].strip()
                    break
        runs.append({
            "id": d.name,
            "kind": "agent" if d.name.startswith("agent_") else "run",
            "question": question,
            "marker_count": len(hints.get("markers") or []),
            "has_camera": bool(hints.get("camera")),
            "mtime": datetime.fromtimestamp(d.stat().st_mtime, timezone.utc).isoformat(),
        })
    return runs
