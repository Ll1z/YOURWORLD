"""GeoAnalyst 的 Web 薄壳：把 agent/report.py 的产物暴露成 HTTP，前端只消费 JSON。

设计取舍：
- 后端不含任何地图逻辑，只负责「提问 → 跑 Agent → 落盘 → 回吐产物」。
  渲染全部在前端，Cesium 升级不会牵动 Python。
- Agent 跑在独占的后台事件循环里（AgentWorker）。这是两条硬约束逼出来的：
  MCP stdio 会话绑定在创建它的 loop 上，而 OpenAI SDK 是同步阻塞的。
  直接跑在 web 的 loop 上，一次提问会把静态页面一起卡住几十秒。
  顺带的好处是串行执行，并发提问不会互相踩会话状态。
- 天地图 Key 只注入 index.html，由浏览器直连天地图。服务端代理会被拒：
  Key 类型是「浏览器端」，服务端请求返回 403 + code 301012。

运行：
    uv run python web/server.py --port 8000
    uv run uvicorn web.server:app --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import contextlib
import json
import re
import threading
from pathlib import Path
from typing import Any, Callable

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from agent import report
from agent.config import Settings
from agent.loop import AgentRun, run
from agent.mcp_hub import MCPHub

INDEX_HTML = Path(__file__).resolve().parent / "index.html"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

settings = Settings()


class AgentWorker:
    """独占一个后台事件循环跑 Agent，web 的 loop 只负责收发 HTTP。

    MCP 会话不能跨 loop 使用，所以 hub 在这一侧创建并常驻；提问串行排队在这里，
    既避免了共享会话的并发问题，也让静态页面在 Agent 跑动时保持响应。
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._hub: MCPHub | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._boot_error: str | None = None

    def start(self, timeout: float = 60.0) -> None:
        self._thread = threading.Thread(target=self._main, name="agent-worker", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise RuntimeError(f"Agent 工作线程 {timeout:.0f}s 内未就绪")

    def _main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._serve())
        finally:
            loop.close()

    async def _serve(self) -> None:
        self._stop = asyncio.Event()
        try:
            async with MCPHub() as hub:
                self._hub = hub
                self._ready.set()
                await self._stop.wait()
        except Exception as e:  # noqa: BLE001 - 启动失败要能回报给页面，而不是静默死掉
            self._boot_error = f"{type(e).__name__}: {e}"
            self._ready.set()

    @property
    def boot_error(self) -> str | None:
        return self._boot_error

    def tools(self) -> list[str]:
        return sorted(self._hub.tools) if self._hub is not None else []

    def servers(self) -> list[str]:
        return sorted(self._hub.sessions) if self._hub is not None else []

    def submit(self, job: Callable[[MCPHub], Any]) -> concurrent.futures.Future:
        if self._loop is None or self._hub is None:
            raise RuntimeError(self._boot_error or "Agent 工作线程尚未就绪")
        return asyncio.run_coroutine_threadsafe(job(self._hub), self._loop)

    def stop(self) -> None:
        if self._loop is None:
            return
        if self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread is not None:
            self._thread.join(timeout=10)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    max_steps: int = Field(default=6, ge=1, le=12)


def _ask_job(question: str, max_steps: int) -> Callable[[MCPHub], Any]:
    async def job(hub: MCPHub) -> tuple[AgentRun, report.Artifacts]:
        run_result = await run(question, hub, settings, max_steps=max_steps)
        return run_result, await asyncio.to_thread(report.persist, run_result)

    return job


def _preview(payload: Any, limit: int = 240) -> str:
    text = json.dumps(payload, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "…"


def _payload(run_result: AgentRun, artifacts: report.Artifacts) -> dict:
    return {
        "run_id": Path(artifacts.run_dir).name,
        "run_dir": artifacts.run_dir,
        "question": run_result.question,
        "answer": run_result.answer,
        "passed": artifacts.passed,
        "steps": run_result.steps,
        "stopped": run_result.stopped,
        "repair_rounds": run_result.repair_rounds,
        "usage": run_result.usage,
        "checks": [
            {"name": c.name, "passed": c.passed, "detail": c.detail} for c in artifacts.checks
        ],
        "invocations": [
            {
                "step": i.step,
                "server": i.server,
                "name": i.name,
                "arguments": i.arguments,
                "ok": i.ok,
                "error": i.error,
                "elapsed_s": i.elapsed_s,
                "result_preview": _preview(i.result),
            }
            for i in run_result.invocations
        ],
        "visual_hints": artifacts.visual_hints,
        "report": artifacts.report_text,
    }


def _require_run_id(run_id: str) -> str:
    # 运行目录名直接来自 URL，先挡住 ../ 这类越界读取
    if not RUN_ID_RE.match(run_id):
        raise HTTPException(400, "run_id 只允许字母、数字、下划线与连字符")
    return run_id


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    worker = AgentWorker()
    try:
        worker.start()
        app.state.worker = worker
        print(f"[web] Agent 就绪：{len(worker.servers())} 个 MCP Server，"
              f"{len(worker.tools())} 个工具", flush=True)
    except Exception as e:  # noqa: BLE001 - 页面仍要能打开并显示失败原因
        app.state.worker = None
        app.state.worker_error = f"{type(e).__name__}: {e}"
        print(f"[web] Agent 启动失败：{app.state.worker_error}", flush=True)
    try:
        yield
    finally:
        if app.state.worker is not None:
            app.state.worker.stop()


app = FastAPI(title="GeoAnalyst", version="0.2.0", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """注入天地图 Key 与 Cesium CDN 地址。Key 只在响应里出现，不落盘、不入库。"""
    html = INDEX_HTML.read_text(encoding="utf-8")
    html = _inject(html, "TIANDITU_KEY", settings.tianditu_key)
    html = _inject(html, "CESIUM_BASE", settings.cesium_base_url)
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


def _inject(html: str, key: str, value: str) -> str:
    """把配置值原样替换进页面。

    占位符同时出现在 HTML 属性与 JS 字符串字面量里，所以这里注入的是原始值，
    由下面这条校验兜住注入风险——配置项本来就只该是 URL 或十六进制 Key。
    """
    if any(ch in value for ch in "\"'<>&"):
        raise HTTPException(500, f"配置项 {key} 含引号或尖括号，拒绝注入页面")
    return html.replace("{{" + key + "}}", value)


@app.get("/api/health")
async def health() -> dict:
    worker: AgentWorker | None = app.state.worker
    return {
        "ok": worker is not None,
        "error": getattr(app.state, "worker_error", None),
        "servers": worker.servers() if worker else [],
        "tools": worker.tools() if worker else [],
        "model": settings.deepseek_model,
        "tianditu_key_configured": bool(settings.tianditu_key),
    }


@app.post("/api/ask")
async def ask(req: AskRequest) -> dict:
    worker: AgentWorker | None = app.state.worker
    if worker is None:
        raise HTTPException(503, f"Agent 未就绪：{getattr(app.state, 'worker_error', '未知原因')}")
    try:
        future = worker.submit(_ask_job(req.question, req.max_steps))
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    try:
        run_result, artifacts = await asyncio.wrap_future(future)
    except Exception as e:  # noqa: BLE001 - 失败要以 JSON 回给前端，不能只留一行栈
        raise HTTPException(500, f"{type(e).__name__}: {e}")
    return _payload(run_result, artifacts)


@app.get("/api/runs")
async def runs() -> list[dict]:
    return await asyncio.to_thread(report.list_runs)


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    _require_run_id(run_id)
    try:
        hints = await asyncio.to_thread(report.load_visual_hints, run_id)
    except FileNotFoundError:
        raise HTTPException(404, f"找不到运行记录 {run_id}")
    report_path = Path(report.DEFAULT_OUTDIR) / run_id / "report.md"
    text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    return {"run_id": run_id, "visual_hints": hints, "report": text}


@app.get("/api/runs/{run_id}/report.md", response_class=PlainTextResponse)
async def get_run_report(run_id: str) -> PlainTextResponse:
    _require_run_id(run_id)
    path = Path(report.DEFAULT_OUTDIR) / run_id / "report.md"
    if not path.exists():
        raise HTTPException(404, f"找不到报告 {run_id}")
    return PlainTextResponse(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="GeoAnalyst Web 服务")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
