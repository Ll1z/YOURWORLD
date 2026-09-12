"""Agent 裸循环：自然语言提问 → 选工具 → 出答案。

刻意不引入 LangChain / LangGraph：Stage 1 要看清循环本身。
循环结构就是标准的 OpenAI 兼容 function calling 三段式：
把 Resource 上下文与工具表交给模型 → 模型请求工具 → 执行并回喂结果 → 直到模型给最终回答。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from openai import OpenAI

from agent.config import Settings
from agent.mcp_hub import MCPHub

SYSTEM_PROMPT = """你是 GeoAnalyst，一个地理空间分析 Agent。当前数据只覆盖北京五区：东城区、西城区、朝阳区、丰台区、海淀区。

硬性规则，任何情况下都不得违背：
1. 所有数值（数量、距离、面积、排名）只能来自工具返回。禁止自己计算、估算或凭常识填写。
2. 先给结论，再给依据，依据至少包含：数据来源、CRS、口径、方法与局限。
3. 涉及「有哪些 / 有多少」时，必须同时考虑点层与面层——只用点层会漏掉一半以上的医院。
4. 结论一律表述为「OSM 数据显示」，因为 OSM 的完备性取决于志愿者测绘。
5. 距离是 UTM 平面下的直线距离，不是路网可达距离，不得据此下可达性结论。
6. 工具报错时最多换一次参数重试；仍失败就如实说明失败原因，不要绕过。
7. 现有工具答不了的问题，直接说明缺什么数据或工具，不要编造。

请用中文回答。"""


def _context_block(context: dict) -> str:
    parts = ["以下是从 MCP Resource 预加载的上下文（可用工具与表结构、数据目录、计数口径、坐标系口径）："]
    for uri, payload in context.items():
        parts.append(f"\n### {uri}\n```json\n{json.dumps(payload, ensure_ascii=False, indent=1)}\n```")
    return "\n".join(parts)


@dataclass
class Invocation:
    step: int
    server: str
    name: str
    arguments: dict
    ok: bool
    error: str | None
    elapsed_s: float
    result: dict


@dataclass
class AgentRun:
    question: str
    answer: str
    stopped: str
    steps: int
    repair_rounds: int = 0
    invocations: list[Invocation] = field(default_factory=list)
    context: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)

    def tool_results(self) -> list[dict]:
        return [i.result for i in self.invocations]

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "stopped": self.stopped,
            "steps": self.steps,
            "repair_rounds": self.repair_rounds,
            "usage": self.usage,
            "invocations": [
                {**asdict(i), "result": _truncate(i.result)} for i in self.invocations
            ],
            "context_uris": sorted(self.context),
        }


def _truncate(payload: Any, limit: int = 4000) -> Any:
    """轨迹里只保留结果摘要，避免 trace 文件被几何 WKT 撑爆。"""
    text = json.dumps(payload, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return payload
    return {"_truncated": True, "_original_chars": len(text), "head": text[:limit]}


async def run(
    question: str,
    hub: MCPHub,
    settings: Settings,
    max_steps: int = 6,
    on_event=None,
) -> AgentRun:
    client = OpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.openai_base_url,
        timeout=settings.request_timeout_s,
    )
    context = await hub.context()
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + _context_block(context)},
        {"role": "user", "content": question},
    ]
    tools = hub.openai_tools()
    invocations: list[Invocation] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    answer, stopped, steps = "", "max_steps", 0

    for step in range(1, max_steps + 1):
        steps = step
        response = client.chat.completions.create(
            model=settings.deepseek_model,
            messages=messages,
            tools=tools,
            temperature=settings.temperature,
        )
        if response.usage:
            for key in usage:
                usage[key] += getattr(response.usage, key, 0) or 0

        message = response.choices[0].message
        if not message.tool_calls:
            answer, stopped = message.content or "", "final"
            break

        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [tc.model_dump() for tc in message.tool_calls],
        })

        for call in message.tool_calls:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            started = time.perf_counter()
            payload, ok, error = await hub.call(call.function.name, arguments)
            invocation = Invocation(
                step=step,
                server=hub.server_of.get(call.function.name, "?"),
                name=call.function.name,
                arguments=arguments,
                ok=ok,
                error=error,
                elapsed_s=round(time.perf_counter() - started, 3),
                result=payload,
            )
            invocations.append(invocation)
            if on_event:
                on_event(invocation)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps({"ok": ok, "error": error, "result": payload},
                                      ensure_ascii=False, default=str)[:20000],
            })

    repair_rounds = 0
    if stopped == "final" and answer:
        pool = [context, {"question": question}, *[i.result for i in invocations]]
        answer, repair_rounds = _repair_ungrounded(client, settings, messages, answer, pool)

    return AgentRun(question=question, answer=answer, stopped=stopped, steps=steps,
                    repair_rounds=repair_rounds, invocations=invocations,
                    context=context, usage=usage)


def _repair_ungrounded(
    client: OpenAI,
    settings: Settings,
    messages: list[dict],
    answer: str,
    pool: list,
    max_rounds: int = 2,
) -> tuple[str, int]:
    """自检反馈：回答里出现无法溯源的数字时，把它打回去重写。

    这是「数值必须由代码算出」这条硬性规则的执行机制——只在提示词里禁止是不够的，
    实测模型会在叙述里自行估算两个设施之间的距离。
    """
    from agent.selfcheck import check_grounding

    for round_no in range(1, max_rounds + 1):
        check = check_grounding(answer, pool)
        if check.passed:
            return answer, round_no - 1
        messages.append({"role": "assistant", "content": answer})
        messages.append({
            "role": "user",
            "content": (
                f"上一版回答未通过数值溯源自检：{check.detail}\n"
                "请重写回答，只使用工具返回中出现过的数字。若确实需要一个工具没有返回的数字，"
                "就明确写「该数值需要额外计算，本次未计算」，不要自行估算或换算。"
            ),
        })
        response = client.chat.completions.create(
            model=settings.deepseek_model,
            messages=messages,
            temperature=settings.temperature,
        )
        answer = response.choices[0].message.content or answer
    return answer, max_rounds


def build_visual_hints(run_result: AgentRun) -> dict:
    """为 Cesium 前端预留的结构化提示。Stage 1 只落盘，不渲染。"""
    last = next((i for i in reversed(run_result.invocations)
                 if i.ok and i.name == "query_nearby" and i.result.get("hits") is not None), None)
    if last is None:
        return {"camera": None, "markers": [], "note": "本次运行没有产生可定位的空间结果（未调用 query_nearby）"}
    payload = last.result
    return {
        "camera": {
            "lon": payload["center_lon"],
            "lat": payload["center_lat"],
            "height_m": max(payload["radius_m"] * 4, 2000.0),
            "heading_deg": 0.0,
            "pitch_deg": -60.0,
        },
        "radius_m": payload["radius_m"],
        "markers": [
            {
                "id": f"{h['layer']}/{h['osm_id']}",
                "lon": h["lon"],
                "lat": h["lat"],
                "label": h.get("name") or "(无名)",
                "category": h["category_value"],
                "dist_m": h["dist_m"],
                "area_m2": h.get("area_m2"),
            }
            for h in payload["hits"]
        ],
        "note": "为 Cesium 前端预留：camera 飞向该点，markers 标记附近内容。Stage 1 不渲染。",
    }
