"""多 Agent 协作：规划者 → 执行者 → 复核者。

分工与边界（这是本文件唯一需要记住的事）：
- 规划者（planner）：动手前先把问题拆成子问题，可以先查经验库（search_knowledge/search_datasets），
  但**只产出计划，不产出任何数值**。它的输出是给执行者看的提示，不是给用户看的答案。
- 执行者（executor）：就是 agent/loop.py 那条裸循环，带上计划去跑。答案仍然只由工具返回的数构成。
- 复核者（critic）：**只判断，不改写**。它拿不到也不许产出数值，只回答「这个答案能不能发」，
  而且是零工具、纯阅读。判 rework 的标准是封闭的四条，不是「我觉得不够好」——
  否则复核者会变成一个随机的答案改写器，比没有还糟。

为什么值得多花两次模型调用：确定性自检器（agent/selfcheck.py）能查数字有没有出处、
CRS 有没有声明，但查不了「问高校答成医院」这种语义错位。复核者补的正是这一块。

复核不通过时最多重做一次，重做完不再迭代：无限自我批评会把一次问答拖成一场会议。
"""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from agent import loop, telemetry
from agent.config import Settings
from agent.mcp_hub import MCPHub

# 规划者只许用检索类工具：它要查的是「这类问题以前踩过什么坑」，
# 不是「数据库里有多少家医院」——后者是执行者的活，规划者算出来的数字没人能验收
PLANNER_TOOLS = ("search_knowledge", "search_datasets")

PLANNER_SYSTEM = """你是 GeoAnalyst 的规划者。用户会给你一个地理空间问题，你负责在动手前把它拆清楚。

你可以调用检索工具查项目的知识库与经验库（数据口径、中文别名、已知坑、以前踩过的坑），
但你没有查询数据的能力，也不要给出任何数字——数量、距离、面积都留给执行者用工具算。

完成后只输出一个 JSON 对象，不要写别的文字：
{
  "restated": "把问题重述成一句可核对的话",
  "sub_questions": [
    {"id": "q1", "question": "子问题原话", "tool": "建议的工具名", "center": "半径查询的中心地名，没有就写 null"}
  ],
  "pitfalls": ["这类问题容易踩的坑，来自检索或常识"],
  "acceptance": ["回答里必须出现什么才算答到"]
}

规则：
- 一句话的问题就拆成一个子问题，不要为了显得细致而硬拆。
- 问题里的地名如果有多个同名候选，在 pitfalls 里写出来，让执行者自己选并说明。
- 问题没给半径查询的中心时，在 pitfalls 里写明「需要先确认中心，不许拿行政区几何代表点当圆心」。"""

CRITIC_SYSTEM = """你是 GeoAnalyst 的复核者。给你一个问题、当时的执行计划、最终回答，以及工具调用的结果摘要。

你只做判断：这个回答能不能直接发给用户。你不改写回答，不产出任何数字，也不要建议具体措辞。

只有下面四条能判 rework：
1. 答非所问：回答的主语或范围与问题不符（问高校答成医院、问数量答成清单）
2. 结论没有工具结果支撑：回答里的关键事实来自叙述而不是工具返回
3. 该说明的没说明：不说数据来源 / 不说 CRS / 0 结果没说清是「数据里没有」还是「没查到」
4. 问题含多个子问题，有子问题没被回答

不要因为措辞、格式、篇幅、风格判 rework。没有上面四条就必须判 pass。

只输出 JSON：
{"verdict": "pass" 或 "rework", "problems": ["命中的第几条 + 具体是什么"], "missing": ["没被回答的子问题"]}"""


def _client(settings: Settings) -> OpenAI:
    return OpenAI(api_key=settings.deepseek_api_key,
                  base_url=settings.openai_base_url,
                  timeout=settings.request_timeout_s)


def _extract_json(text: str | None) -> dict:
    """从模型输出里抠出第一个 JSON 对象。抠不出来就返回空字典，由调用方兜底。"""
    raw = (text or "").strip()
    start = raw.find("{")
    if start < 0:
        return {}
    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start:i + 1])
                except json.JSONDecodeError:
                    return {}
    return {}


async def plan(question: str, hub: MCPHub, settings: Settings,
               context_block: str, tracer: telemetry.Tracer,
               max_steps: int = 3) -> dict:
    client = _client(settings)
    tools = [t for t in hub.openai_tools() if t["function"]["name"] in PLANNER_TOOLS]
    messages: list[dict] = [
        {"role": "system", "content": PLANNER_SYSTEM + "\n\n" + context_block},
        {"role": "user", "content": question},
    ]
    for step in range(1, max_steps + 1):
        with tracer.span("invoke_agent GeoAnalyst.planner", telemetry.INVOKE_AGENT,
                         **{telemetry.AGENT_NAME: "planner", telemetry.STEP: step}) as span:
            response = client.chat.completions.create(model=settings.deepseek_model,
                                                      messages=messages, tools=tools,
                                                      temperature=settings.temperature)
            message = response.choices[0].message
            span.set(**{"geo.tool_calls": len(message.tool_calls or [])})
        if not message.tool_calls:
            return _extract_json(message.content)
        messages.append({"role": "assistant", "content": message.content or "",
                         "tool_calls": [tc.model_dump() for tc in message.tool_calls]})
        for call in message.tool_calls:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            payload, ok, error = await hub.call(call.function.name, arguments)
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": loop._tool_content(ok, error, payload)})
    return {}


def _summarize(invocations: list) -> list[dict]:
    out = []
    for inv in invocations:
        item = {"tool": inv.name, "arguments": inv.arguments, "ok": inv.ok, "error": inv.error}
        if inv.ok:
            body = json.dumps(inv.result, ensure_ascii=False, default=str)
            item["result"] = body[:900]
        out.append(item)
    return out


async def review(question: str, plan_doc: dict, answer: str,
                 invocations: list, settings: Settings,
                 tracer: telemetry.Tracer) -> dict:
    client = _client(settings)
    brief = {
        "question": question,
        "plan": plan_doc or "（规划者没有给出计划）",
        "answer": answer,
        "tool_calls": _summarize(invocations),
    }
    with tracer.span("invoke_agent GeoAnalyst.critic", telemetry.INVOKE_AGENT,
                     **{telemetry.AGENT_NAME: "critic"}) as span:
        response = client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[{"role": "system", "content": CRITIC_SYSTEM},
                      {"role": "user", "content": json.dumps(brief, ensure_ascii=False)}],
            temperature=settings.temperature,
        )
        span.set(**{telemetry.INPUT_TOKENS: getattr(response.usage, "prompt_tokens", 0) or 0,
                    telemetry.OUTPUT_TOKENS: getattr(response.usage, "completion_tokens", 0) or 0})
    return _extract_json(response.choices[0].message.content)


def _problems(verdict: dict) -> int:
    return len(verdict.get("problems") or []) + len(verdict.get("missing") or [])


def _should_replace(candidate_answer: str, candidate_verdict: dict, base_verdict: dict) -> bool:
    """重做要不要顶替上一版：先看有没有回答，再看复核指出的问题是不是更少。

    只在「更少」时顶替，是为了别用一次更差的重做换掉已经合格的答案——
    复核者的意见是线索，不是命令；它说重做，不等于重做的结果就一定更好。
    """
    if not candidate_answer.strip():
        return False
    return _problems(candidate_verdict) < _problems(base_verdict)


def _plan_block(plan_doc: dict) -> str:
    if not plan_doc:
        return ""
    lines = ["【规划者给出的计划】这是动手前另一位模型定的计划，按它执行；与用户原话冲突时以用户原话为准。"]
    if plan_doc.get("restated"):
        lines.append(f"- 问题重述：{plan_doc['restated']}")
    for i, sub in enumerate(plan_doc.get("sub_questions") or [], 1):
        center = sub.get("center")
        lines.append(f"- 子问题{i}：{sub.get('question')}"
                     f"（建议工具 {sub.get('tool') or '未指定'}"
                     f"{'，中心 ' + center if center else ''}）")
    for pitfall in plan_doc.get("pitfalls") or []:
        lines.append(f"- 注意：{pitfall}")
    if plan_doc.get("acceptance"):
        lines.append(f"- 回答必须包含：{'；'.join(plan_doc['acceptance'])}")
    return "\n".join(lines)


def _rework_block(verdict: dict) -> str:
    lines = ["【复核者的意见】上一版回答被打回，只针对下面这些点重做，其余部分保持原样："]
    for item in verdict.get("problems") or []:
        lines.append(f"- {item}")
    for item in verdict.get("missing") or []:
        lines.append(f"- 没回答到的子问题：{item}")
    lines.append("重做时数字仍只能来自工具返回；工具没返回的数字就不写。")
    return "\n".join(lines)


async def run_team(question: str, hub: MCPHub, settings: Settings, max_steps: int = 6,
                   on_event=None, clarification: str | None = None,
                   tracer: telemetry.Tracer | None = None) -> loop.AgentRun:
    """三角色协作跑一次问答：规划 → 执行 → 复核（不通过则重做一次再复核）。

    hub 由调用方建好并持有（MCP 会话绑定在创建它的 loop 上），这里只借用，不接管生命周期。
    """
    settings.apply_otel_env()
    tracer = tracer or telemetry.Tracer()
    context_block = loop._context_block(await hub.context())

    with tracer.span("invoke_agent GeoAnalyst.team", telemetry.INVOKE_AGENT,
                     **{telemetry.AGENT_NAME: "GeoAnalyst.team",
                        telemetry.REQUEST_MODEL: settings.deepseek_model,
                        telemetry.CONVERSATION_ID: tracer.run_id,
                        "geo.question": question}) as span:
        plan_doc = await plan(question, hub, settings, context_block, tracer)
        extra = _plan_block(plan_doc)
        if not plan_doc:
            span.set(**{"geo.plan": "规划者未给出可用计划，按单循环执行"})

        result = await loop.run(question, hub, settings, max_steps=max_steps, on_event=on_event,
                                clarification=clarification, tracer=tracer, extra_system=extra)
        verdict = await review(question, plan_doc, result.answer, result.invocations, settings, tracer)

        rework = 0
        # 只重做一次：无限自我批评会把一次问答拖成一场会议
        if verdict.get("verdict") == "rework" and (verdict.get("problems") or verdict.get("missing")):
            rework = 1
            retry = await loop.run(question, hub, settings, max_steps=max_steps, on_event=on_event,
                                   clarification=clarification, tracer=tracer,
                                   extra_system=extra + "\n\n" + _rework_block(verdict))
            retry_verdict = await review(question, plan_doc, retry.answer, retry.invocations,
                                         settings, tracer)
            if _should_replace(retry.answer, retry_verdict, verdict):
                result, verdict = retry, retry_verdict
            else:
                # 重做不一定更好。用空答案或问题更多的版本换掉上一版，
                # 是这套协作最蠢的失败方式。
                verdict = {**verdict, "note": "重做那一版没有更好（空答案或复核问题没减少），已保留上一版"}

        span.set(**{"geo.rework": rework, "geo.verdict": verdict.get("verdict"),
                    telemetry.STOPPED: result.stopped})

    tracer.flush()
    result.spans = tracer.spans
    result.span_warnings = tracer.warnings
    result.plan = plan_doc
    result.verdict = verdict
    result.rework = rework
    return result
