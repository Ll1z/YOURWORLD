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

from agent import telemetry
from agent.config import Settings
from agent.mcp_hub import GROUNDING_EXCLUDE, MCPHub

SYSTEM_PROMPT = """你是 GeoAnalyst，一个地理空间分析 Agent。当前数据只覆盖北京五区：东城区、西城区、朝阳区、丰台区、海淀区。

硬性规则，任何情况下都不得违背：
1. 所有数值（数量、距离、面积、排名）只能来自工具返回。禁止自己计算、估算或凭常识填写。表格与列表不要加序号列或编号，序号也是一种凭空生成的数字。
2. 先给结论，再给依据，依据至少包含：数据来源、CRS、口径、方法与局限。
3. 半径查询必须先有一个明确、可命名的中心，和路径规划要先选起点是一个道理。顺序是：用户在问题里给了地点就先用 find_places 把它解析成坐标；只给了区名加半径就先追问以哪里为中心。禁止用行政区的几何代表点当圆心。
4. 涉及「有哪些 / 有多少」时，必须同时考虑点层与面层——只用点层会漏掉一半以上的医院。
5. 结论一律表述为「OSM 数据显示」，因为 OSM 的完备性取决于志愿者测绘。
6. 距离是 UTM 平面下的直线距离，不是路网可达距离，不得据此下可达性结论。
7. 「A 和 B 相距多远」必须调 distance_between 计算，不要拿坐标自行估算。
8. 工具报错时，只允许做同义修正（半径、坐标写法、地名写法）后重试一次；参数含义一旦改变就不再是「重试」，而是换了一个问题。仍失败就如实说明失败原因，不要绕过。
9. 查什么类别只能来自三处：compute://categories 的类别清单、knowledge://categories/aliases 的中文说法、用户原话里的类别词。工具报「不认识的类别」时，改用它给出的近似建议或向用户澄清，禁止换成另一个类别去凑答案——「马甸桥 10 公里内有哪些高校」答成一堆医院，就是这么来的。
10. 某个类别 0 命中就是 0：如实说「OSM 数据里没有」，并说清查的是哪个类别，不要用别的类别替代，也不要把 0 说成「工具没能给出结果」。
11. 现有工具答不了的问题，直接说明缺什么数据或工具，不要编造：例如地铁站、火车站只在 anchor 层，不参与半径检索，问「附近有哪些地铁站」目前没有数据支持。
12. 现成工具拼不出来的分析（自定义缓冲区、按距离分箱、多表关联、导出中间结果），可以用 run_python 在受限沙箱里写代码算。但顺序不能反：能用 find_places / query_nearby / summarize_poi / distance_between 回答的，一律先用工具——工具的口径全项目唯一，脚本里的口径是你临时写的。沙箱里的 con 就是工具用的那份库，优先在脚本里复用它，不要把数值硬编码进代码。
13. 沙箱脚本失败会连 traceback 一起返回：照着 traceback 改，改完重跑；同一个错误连续两次没修好就停下来如实说明，不要换个说法糊过去。沙箱输出同样是工具返回，答案里的数字必须能在其中找到出处。沙箱默认 30 秒超时、1024 MB 内存上限，长循环自己先分片。
14. 问的是「数据本身」时——某项口径怎么定的、这个中文说法对应哪个 OSM 标签、某份数据有哪些已知坑、某个类别为什么查不到——用 search_knowledge 检索知识库（覆盖数据卡、口径文件、中文别名、坐标系定义与已验证的经验条目），命中项带 source_uri，要看全文就按它读对应 Resource；只问「该用哪份数据」时用 search_datasets。这两个都是检索：检索不到就如实说没查到，不要改用关键词猜。

请用中文回答。"""

# 用户在界面上点名了中心点（或端点）时，把这句话追加到提问后面。
# 不绕过 Agent 直接查库：中心点的选择要留在轨迹里，才能解释「为什么是这几个结果」。
CLARIFICATION_TEMPLATE = ("[用户在界面上确认了{note}。直接用它，"
                          "不要再解析地名、也不要换成别的点。]")


def _context_block(context: dict) -> str:
    parts = ["以下是从 MCP Resource 预加载的上下文（可用工具与表结构、可查类别与中文别名、"
             "数据目录、计数口径、坐标系口径、知识库索引构成）："]
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
    clarification: str | None = None
    invocations: list[Invocation] = field(default_factory=list)
    context: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    spans: list[dict] = field(default_factory=list)
    span_warnings: list[str] = field(default_factory=list)

    def tool_results(self) -> list[dict]:
        return [i.result for i in self.invocations]

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "stopped": self.stopped,
            "steps": self.steps,
            "repair_rounds": self.repair_rounds,
            "clarification": self.clarification,
            "usage": self.usage,
            "invocations": [
                {**asdict(i), "result": _truncate(i.result)} for i in self.invocations
            ],
            "context_uris": sorted(self.context),
            "spans": self.spans,
            "span_warnings": self.span_warnings,
        }


def _truncate(payload: Any, limit: int = 4000) -> Any:
    """轨迹里只保留结果摘要，避免 trace 文件被几何 WKT 撑爆。"""
    text = json.dumps(payload, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return payload
    return {"_truncated": True, "_original_chars": len(text), "head": text[:limit]}


# 回喂给模型的工具结果上限。超长时必须显式说明被截断：静默截断会让模型以为
# 自己拿到了全部明细，进而说出「完整明细已由工具返回」这种不成立的话。
TOOL_RESULT_LIMIT = 20000


def _tool_content(ok: bool, error: str | None, payload: Any) -> str:
    body = json.dumps({"ok": ok, "error": error, "result": payload},
                      ensure_ascii=False, default=str)
    if len(body) <= TOOL_RESULT_LIMIT:
        return body
    return json.dumps({
        "ok": ok,
        "error": error,
        "truncated": True,
        "original_chars": len(body),
        "note": f"结果过长，只回喂前 {TOOL_RESULT_LIMIT} 个字符。计数类字段（count_total、"
                "count_point、count_area 与 categories 里的 count）是完整的，明细列表不是。"
                "回答时要说明「明细未全部列出」，不要写成「完整明细已由工具返回」。",
        "head": body[:TOOL_RESULT_LIMIT],
    }, ensure_ascii=False)


def grounding_pool(question: str, clarification: str | None, context: dict,
                   invocations: list["Invocation"]) -> list:
    """数值溯源的可信来源：预加载上下文 + 用户输入 + 全部工具返回。

    用户在界面上点选的中心点与原始提问同属输入，模型引用它不算凭空造数；
    漏了它就会把「用户自己给的坐标」判成幻觉。
    目录型清单（类别目录、数据集清单）不进池子：它们的数字在描述别的对象，
    留在里面会让任意小整数都能找到「出处」。
    """
    evidence = {uri: value for uri, value in context.items()
                if uri not in GROUNDING_EXCLUDE}
    return [evidence, {"question": question}, {"clarification": clarification},
            *[i.result for i in invocations]]


async def run(
    question: str,
    hub: MCPHub,
    settings: Settings,
    max_steps: int = 6,
    on_event=None,
    clarification: str | None = None,
    tracer: telemetry.Tracer | None = None,
) -> AgentRun:
    """一次提问的入口：起根 span → 跑循环 → 把 span 树挂回结果。

    整段循环包在根 span 里，chat / execute_tool 才会挂在它下面，
    trace 里能直接看出「哪次工具调用属于哪一轮、哪一轮开始跑偏」。
    """
    settings.apply_otel_env()
    tracer = tracer or telemetry.Tracer()
    with tracer.span(
        "invoke_agent GeoAnalyst",
        telemetry.INVOKE_AGENT,
        **{
            telemetry.AGENT_NAME: "GeoAnalyst",
            telemetry.SYSTEM: "deepseek",
            telemetry.REQUEST_MODEL: settings.deepseek_model,
            telemetry.CONVERSATION_ID: tracer.run_id,
            "geo.question": question,
            "geo.clarification": clarification,
        },
    ) as span:
        result = await _drive(question, hub, settings, max_steps, on_event, clarification, tracer)
        span.set(**{
            telemetry.INPUT_TOKENS: result.usage.get("prompt_tokens", 0),
            telemetry.OUTPUT_TOKENS: result.usage.get("completion_tokens", 0),
            telemetry.STOPPED: result.stopped,
            "geo.steps": result.steps,
            "geo.repair_rounds": result.repair_rounds,
        })
    tracer.flush()
    result.spans = tracer.spans
    result.span_warnings = tracer.warnings
    return result


async def _drive(
    question: str,
    hub: MCPHub,
    settings: Settings,
    max_steps: int,
    on_event,
    clarification: str | None,
    tracer: telemetry.Tracer,
) -> AgentRun:
    client = OpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.openai_base_url,
        timeout=settings.request_timeout_s,
    )
    context = await hub.context()
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + _context_block(context)},
        {"role": "user", "content": question + (f"\n\n{CLARIFICATION_TEMPLATE.format(note=clarification)}"
                                                if clarification else "")},
    ]
    tools = hub.openai_tools()
    invocations: list[Invocation] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    answer, stopped, steps = "", "max_steps", 0

    for step in range(1, max_steps + 1):
        steps = step
        with tracer.span(
            f"chat {settings.deepseek_model}",
            telemetry.CHAT,
            **{
                telemetry.SYSTEM: "deepseek",
                telemetry.REQUEST_MODEL: settings.deepseek_model,
                telemetry.STEP: step,
                "geo.message_count": len(messages),
            },
        ) as span:
            response = client.chat.completions.create(
                model=settings.deepseek_model,
                messages=messages,
                tools=tools,
                temperature=settings.temperature,
            )
            if response.usage:
                for key in usage:
                    usage[key] += getattr(response.usage, key, 0) or 0
                span.set(**{
                    telemetry.INPUT_TOKENS: getattr(response.usage, "prompt_tokens", 0) or 0,
                    telemetry.OUTPUT_TOKENS: getattr(response.usage, "completion_tokens", 0) or 0,
                })
            span.set(**{
                telemetry.RESPONSE_MODEL: response.model,
                "geo.tool_calls": len(response.choices[0].message.tool_calls or []),
            })

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
            with tracer.span(
                f"execute_tool {call.function.name}",
                telemetry.EXECUTE_TOOL,
                **{
                    telemetry.TOOL_NAME: call.function.name,
                    telemetry.TOOL_CALL_ID: call.id,
                    telemetry.TOOL_SERVER: hub.server_of.get(call.function.name, "?"),
                    telemetry.STEP: step,
                    "geo.tool.arguments": json.dumps(arguments, ensure_ascii=False),
                },
            ) as span:
                payload, ok, error = await hub.call(call.function.name, arguments)
                span.set(**{telemetry.TOOL_OK: ok, "geo.tool.error": error})
                if not ok:
                    span.fail(RuntimeError(str(error or "工具返回失败")))
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
                "content": _tool_content(ok, error, payload),
            })

    repair_rounds = 0
    if stopped == "final" and answer:
        pool = grounding_pool(question, clarification, context, invocations)
        answer, repair_rounds = _repair_ungrounded(client, settings, messages, answer, pool, tracer)

    return AgentRun(question=question, answer=answer, stopped=stopped, steps=steps,
                    repair_rounds=repair_rounds, clarification=clarification,
                    invocations=invocations,
                    context=context, usage=usage)


def _repair_ungrounded(
    client: OpenAI,
    settings: Settings,
    messages: list[dict],
    answer: str,
    pool: list,
    tracer: telemetry.Tracer,
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
                f"上一版回答未通过数值溯源自检：有 {len(check.offenders)} 个数字在工具返回中找不到出处。\n"
                "请重写回答，只使用工具返回中出现过的数字；那几个数字直接删掉，"
                "不要在回答里复述、罗列或解释它们本身——列出来一样算没通过。"
                "常见的漏网之鱼是表格序号列与列表编号：它们不是数据，直接去掉编号即可。"
                "若确实需要一个工具没有返回的数字，"
                "就明确写「该数值需要额外计算，本次未计算」，不要自行估算或换算。"
            ),
        })
        with tracer.span(
            f"chat {settings.deepseek_model}",
            telemetry.CHAT,
            **{
                telemetry.SYSTEM: "deepseek",
                telemetry.REQUEST_MODEL: settings.deepseek_model,
                telemetry.REPAIR_ROUND: round_no,
                # 具体是哪几个数字没溯源上，要留在轨迹里——经验库蒸馏靠它归纳
                # 「哪类问法会让模型自己编数」，只留个数等于把材料扔了
                "geo.offenders": [str(o) for o in check.offenders[:10]],
                "geo.offender_count": len(check.offenders),
            },
        ) as span:
            response = client.chat.completions.create(
                model=settings.deepseek_model,
                messages=messages,
                temperature=settings.temperature,
            )
            answer = response.choices[0].message.content or answer
            if response.usage:
                span.set(**{
                    telemetry.INPUT_TOKENS: getattr(response.usage, "prompt_tokens", 0) or 0,
                    telemetry.OUTPUT_TOKENS: getattr(response.usage, "completion_tokens", 0) or 0,
                })
    return answer, max_rounds


HINT_VERSION = 3


def build_visual_hints(run_result: AgentRun) -> dict:
    """为 Cesium 前端生成结构化提示：相机飞向哪里、标记什么、连线哪两点。

    后端只描述「看什么」，不描述「怎么画」——Cesium 版本升级不该改到 Python。
    三种可定位结果按特异性排序：两点距离 > 半径查询 > 地名候选。
    """

    def last_ok(tool: str):
        return next((i for i in reversed(run_result.invocations) if i.ok and i.name == tool), None)

    # 地名候选单独列出来：前端把它渲染成可点的中心点确认列表
    places = last_ok("find_places")
    found = candidate_list(places.result) if places is not None else []

    hints = {
        "version": HINT_VERSION,
        "question": run_result.question,
        "kind": "none",
        "camera": None,
        "markers": [],
        "candidates": found,
        "path": None,
        "radius_m": None,
        "note": "本次运行没有产生可定位的空间结果",
    }

    distance = last_ok("distance_between")
    if distance is not None and distance.result.get("a") and distance.result.get("b"):
        a, b = distance.result["a"], distance.result["b"]
        # 相机取两端点中点。这只决定「看向哪里」，不参与任何对外报告的数字，
        # 因此用经纬度均值即可，公里尺度上与椭球面中点的差异远小于相机精度。
        hints.update({
            "kind": "distance",
            "camera": {
                "lon": (a["lon"] + b["lon"]) / 2.0,
                "lat": (a["lat"] + b["lat"]) / 2.0,
                "height_m": max(float(distance.result.get("geodesic_distance_m") or 0.0) * 3.0,
                                1500.0),
                "heading_deg": 0.0,
                "pitch_deg": -60.0,
            },
            "path": [[a["lon"], a["lat"]], [b["lon"], b["lat"]]],
            "markers": [_endpoint_marker("a", a), _endpoint_marker("b", b)],
            "note": "camera 取两端点中点俯视，path 是两点连线，markers 标出两端点。",
        })
        return hints

    nearby = last_ok("query_nearby")
    if nearby is not None and nearby.result.get("hits") is not None:
        payload = nearby.result
        hints.update({
            "kind": "nearby",
            "camera": {
                "lon": payload["center_lon"],
                "lat": payload["center_lat"],
                "height_m": max(payload["radius_m"] * 4.0, 2000.0),
                "heading_deg": 0.0,
                "pitch_deg": -60.0,
            },
            "radius_m": payload["radius_m"],
            "markers": [
                {
                    "id": "center",
                    "lon": payload["center_lon"],
                    "lat": payload["center_lat"],
                    "label": "查询中心",
                    "kind": "center",
                    "source": payload.get("center_source"),
                },
                *[
                    {
                        "id": f"{h['layer']}/{h['osm_id']}",
                        "lon": h["lon"],
                        "lat": h["lat"],
                        "label": h.get("name") or "(无名)",
                        "kind": h["layer"],
                        "category": h["category_value"],
                        "dist_m": h["dist_m"],
                        "area_m2": h.get("area_m2"),
                    }
                    for h in payload["hits"]
                ],
            ],
            "note": "camera 飞向查询中心，markers 标记半径内命中，radius_m 供前端画范围圈。",
        })
        return hints

    if found:
        top = found[0]
        hints.update({
            "kind": "places",
            "camera": {
                "lon": top["lon"],
                "lat": top["lat"],
                "height_m": 4000.0,
                "heading_deg": 0.0,
                "pitch_deg": -60.0,
            },
            "markers": [
                {
                    "id": c["ref"],
                    "lon": c["lon"],
                    "lat": c["lat"],
                    "label": c["label"],
                    "kind": c["layer"],
                    "ref": c["ref"],
                    "category": c["category"],
                    "district": c["district"],
                    "match_score": c["match_score"],
                }
                for c in found
            ],
            "note": "camera 飞向匹配度最高的候选，markers 列出全部候选，供人工确认中心点。",
        })
        return hints

    hints["note"] = ("本次运行没有产生可定位的空间结果"
                     "（未调用 query_nearby / distance_between / find_places）")
    return hints


def _records(payload: Any) -> list[dict]:
    """取工具返回里的记录列表。

    MCP 对返回 list 的工具（find_places / list_districts）会包一层 {"result": [...]}，
    返回单个模型实例的工具则是扁平字典，这里把两种形状归一。
    """
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        for key in ("hits", "result"):
            value = payload.get(key)
            if isinstance(value, list):
                return [r for r in value if isinstance(r, dict)]
    return []


def candidate_list(payload: Any) -> list[dict]:
    """地名候选：前端把它渲染成可点的中心点确认列表。

    候选随每次运行一起落盘，所以事后回看也能知道当时在几个同名地点里选了哪一个。
    """
    return [
        {
            "ref": r["ref"],
            "label": r.get("name") or "(无名)",
            "layer": r["layer"],
            "category": r.get("category_value"),
            "district": r.get("district"),
            "lon": r["lon"],
            "lat": r["lat"],
            "match_score": r.get("match_score"),
        }
        for r in _records(payload)
    ]


def _endpoint_marker(tag: str, endpoint: dict) -> dict:
    return {
        "id": f"endpoint-{tag}",
        "lon": endpoint["lon"],
        "lat": endpoint["lat"],
        "label": endpoint.get("label") or tag,
        "kind": "endpoint",
        "ref": endpoint.get("ref"),
        "layer": endpoint.get("layer"),
        "category": endpoint.get("category_value"),
        "district": endpoint.get("district"),
    }
