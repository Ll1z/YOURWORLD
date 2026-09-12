"""观测层：把一次 Agent 运行变成一棵 OTel span 树。

口径：
- span 命名与属性遵循 OpenTelemetry GenAI 语义约定（gen_ai.*）：一次提问一个
  invoke_agent 根 span，每次模型调用一个 chat 子 span，每次工具调用一个 execute_tool 子 span。
- 出口有两个，各自独立开关：
  1) 本地 spans.jsonl —— 永远写进本次运行的产物目录，离线可查、可回放；
  2) OTLP/HTTP —— 设了 OTEL_EXPORTER_OTLP_ENDPOINT 才启用，Langfuse 走这个口
     （v3 的 OTLP 入口 + Basic Auth 头都从环境变量读，密钥不进代码）。
- 观测不是跑通的前提：没装 SDK、出口配错，都只记一条告警，不打断问答。
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

# OTel GenAI 语义约定里的属性名写成字符串常量，而不是 import 常量模块：
# 那些常量挂在 incubating 命名空间下、会随版本挪位置，属性名本身反而稳定。
SYSTEM = "gen_ai.system"
OPERATION = "gen_ai.operation.name"
REQUEST_MODEL = "gen_ai.request.model"
RESPONSE_MODEL = "gen_ai.response.model"
INPUT_TOKENS = "gen_ai.usage.input_tokens"
OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
TOOL_NAME = "gen_ai.tool.name"
TOOL_CALL_ID = "gen_ai.tool.call.id"
AGENT_NAME = "gen_ai.agent.name"
CONVERSATION_ID = "gen_ai.conversation.id"

INVOKE_AGENT = "invoke_agent"
CHAT = "chat"
EXECUTE_TOOL = "execute_tool"

# 项目自己的属性统一用 geo. 前缀，避免和语义约定撞名
STEP = "geo.step"
REPAIR_ROUND = "geo.repair_round"
TOOL_SERVER = "geo.tool.server"
TOOL_OK = "geo.tool.ok"
STOPPED = "geo.stopped"

try:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        SimpleSpanProcessor,
        SpanExporter,
        SpanExportResult,
    )
    from opentelemetry.trace import Status, StatusCode

    HAS_OTEL = True
except ImportError:  # 没装 SDK 时退化成「只记耗时」的本地记录
    HAS_OTEL = False


def _jsonable(value: Any) -> Any:
    """OTel 属性只认标量与其序列，其余一律转成 JSON 字符串。"""
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)) and all(
        isinstance(v, (bool, int, float, str)) for v in value
    ):
        return list(value)
    return json.dumps(value, ensure_ascii=False, default=str)


def _iso(ns: int | None) -> str | None:
    if not ns:
        return None
    return datetime.fromtimestamp(ns / 1e9, timezone.utc).isoformat()


def _span_to_dict(span: Any) -> dict:
    ctx = span.get_span_context()
    parent = span.parent
    attributes = {k: v for k, v in (span.attributes or {}).items()}
    return {
        "name": span.name,
        "trace_id": format(ctx.trace_id, "032x"),
        "span_id": format(ctx.span_id, "016x"),
        "parent_span_id": format(parent.span_id, "016x") if parent else None,
        "start_time": _iso(span.start_time),
        "end_time": _iso(span.end_time),
        "duration_ms": round((span.end_time - span.start_time) / 1e6, 1) if span.end_time else None,
        "status": "error" if span.status.status_code is StatusCode.ERROR else "ok",
        "attributes": attributes,
        "events": [event.name for event in (span.events or [])],
    }


if HAS_OTEL:

    class _ListExporter(SpanExporter):
        """把 span 收进内存列表，运行结束后由 report.py 写成 spans.jsonl。"""

        def __init__(self, sink: list[dict]):
            self._sink = sink

        def export(self, spans) -> Any:
            self._sink.extend(_span_to_dict(span) for span in spans)
            return SpanExportResult.SUCCESS

        def shutdown(self) -> None:
            return None


class SpanHandle:
    """span 的句柄。没装 SDK 时它就是一块什么都不写的占位。"""

    def __init__(self, otel_span: Any, attributes: dict):
        self.attributes = {k: v for k, v in attributes.items() if v is not None}
        self.status = "ok"
        self.elapsed_ms: float | None = None
        self._span = otel_span
        self.set(**self.attributes)

    def set(self, **attributes: Any) -> None:
        self.attributes.update({k: v for k, v in attributes.items() if v is not None})
        if self._span is None:
            return
        for key, value in attributes.items():
            if value is not None:
                self._span.set_attribute(key, _jsonable(value))

    def fail(self, exc: BaseException) -> None:
        self.status = "error"
        if self._span is None:
            return
        self._span.set_status(Status(StatusCode.ERROR, str(exc)[:200]))


class Tracer:
    """一次运行的 span 容器，用法与 OTel 一致：`with tracer.span(...) as s:`。"""

    def __init__(self, run_id: str | None = None):
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.spans: list[dict] = []
        self.warnings: list[str] = []
        self._provider = None
        self._tracer = None
        if not HAS_OTEL:
            self.warnings.append("未安装 opentelemetry-sdk，只记本地耗时，不产出 span 树")
            return
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(_ListExporter(self.spans)))
        self._add_otlp(provider)
        self._provider = provider
        self._tracer = provider.get_tracer("geoanalyst", "0.2.0")

    def _add_otlp(self, provider: Any) -> None:
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if not endpoint:
            return
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        except Exception as exc:  # noqa: BLE001 - 出口配错不能影响问答
            self.warnings.append(
                f"OTLP 出口不可用（{type(exc).__name__}: {exc}），本次只写本地 spans.jsonl"
            )

    @contextmanager
    def span(self, name: str, operation: str, **attributes: Any) -> Iterator[SpanHandle]:
        if self._tracer is None:
            handle = SpanHandle(None, {OPERATION: operation, **attributes})
            started = time.perf_counter()
            try:
                yield handle
            finally:
                handle.elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
                self.spans.append({
                    "name": name,
                    "trace_id": None,
                    "span_id": None,
                    "parent_span_id": None,
                    "duration_ms": handle.elapsed_ms,
                    "status": handle.status,
                    "attributes": handle.attributes,
                    "events": [],
                })
            return

        with self._tracer.start_as_current_span(name) as otel_span:
            handle = SpanHandle(otel_span, {OPERATION: operation, **attributes})
            try:
                yield handle
            except Exception as exc:
                handle.fail(exc)
                raise

    def flush(self) -> None:
        if self._provider is not None:
            self._provider.force_flush()
            self._provider.shutdown()


def summarize(spans: list[dict]) -> dict:
    """给报告用的小结：各类 span 条数、耗时与失败数。"""

    def bucket(operation: str) -> dict:
        rows = [s for s in spans if (s.get("attributes") or {}).get(OPERATION) == operation]
        return {
            "count": len(rows),
            "duration_ms": round(sum(s.get("duration_ms") or 0.0 for s in rows), 1),
            "errors": sum(1 for s in rows if s.get("status") == "error"),
        }

    return {
        "span_total": len(spans),
        "invoke_agent": bucket(INVOKE_AGENT),
        "chat": bucket(CHAT),
        "execute_tool": bucket(EXECUTE_TOOL),
        "errors": sum(1 for s in spans if s.get("status") == "error"),
        "otlp": bool(os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")),
    }
