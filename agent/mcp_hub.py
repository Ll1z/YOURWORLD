"""把三个 MCP Server 聚合成一个工具集，供 Agent 循环调用。

设计取舍：
- 三个 Server 各自独立进程，走 stdio；Agent 侧只看到一张统一工具表。
  Server 之间有明确分工，进程隔离保证一个 Server 崩了不会拖垮另两个。
- 只预加载少量 Resource 注入 system prompt（Resource 承载上下文，Tool 承载动作），
  避免把整套数据卡塞进上下文。
- 工具名跨 Server 唯一，因此对外暴露裸名，同时保留 server_of 映射用于追踪与观测。
"""

from __future__ import annotations

import json
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]

SERVERS: dict[str, str] = {
    "geo_catalog": "geo_catalog.server",
    "geo_compute": "geo_compute.server",
    "geo_knowledge": "geo_knowledge.server",
}

CONTEXT_RESOURCES: list[tuple[str, str]] = [
    ("geo_compute", "compute://schema"),
    ("geo_compute", "compute://categories"),
    ("geo_catalog", "catalog://datasets"),
    ("geo_knowledge", "knowledge://scope/poi_scope"),
    ("geo_knowledge", "knowledge://scope/anchor_scope"),
    ("geo_knowledge", "knowledge://categories/aliases"),
    ("geo_knowledge", "knowledge://coords/systems"),
]

# 溯源池不采信的资源：目录型清单里的数字在描述「别的类别有多少条」「一共几个数据集」，
# 与本次提问无关。留在池子里会让任意小整数都能找到「出处」——实测「4 家咖啡馆」这种
# 编造的计数，就是被 compute://categories 里的 4 兜住的。
GROUNDING_EXCLUDE = ("compute://categories", "catalog://datasets")


def _as_json(result: Any) -> dict:
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        return structured
    parts = [getattr(c, "text", None) or "" for c in (getattr(result, "content", None) or [])]
    body = "\n".join(p for p in parts if p)
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"text": body}


class MCPHub:
    """聚合三个 stdio MCP Server，用法：`async with MCPHub() as hub: ...`"""

    def __init__(self, servers: dict[str, str] | None = None):
        self.servers = dict(servers or SERVERS)
        self.sessions: dict[str, ClientSession] = {}
        self.tools: dict[str, Any] = {}
        self.server_of: dict[str, str] = {}
        self._stack: AsyncExitStack | None = None

    async def __aenter__(self) -> "MCPHub":
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        for name, module in self.servers.items():
            params = StdioServerParameters(
                command=sys.executable,
                args=["-m", module],
                cwd=str(ROOT),
                env={"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
            )
            read, write = await self._stack.enter_async_context(stdio_client(params))
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self.sessions[name] = session
        await self._index()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        assert self._stack is not None
        await self._stack.__aexit__(*exc)

    async def _index(self) -> None:
        for name, session in self.sessions.items():
            for tool in (await session.list_tools()).tools:
                if tool.name in self.tools:
                    raise RuntimeError(f"工具名跨 Server 冲突: {tool.name}")
                self.tools[tool.name] = tool
                self.server_of[tool.name] = name

    def openai_tools(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema,
                },
            }
            for tool in self.tools.values()
        ]

    async def call(self, name: str, arguments: dict | None = None) -> tuple[dict, bool, str | None]:
        """调用一个工具，返回 (结构化结果, 是否成功, 错误信息)。"""
        server = self.server_of.get(name)
        if server is None:
            return {}, False, f"未知工具: {name}"
        try:
            result = await self.sessions[server].call_tool(name, arguments or {})
        except Exception as e:  # noqa: BLE001 - 工具异常要原样回喂给模型，不能中断循环
            return {}, False, f"{type(e).__name__}: {e}"
        payload = _as_json(result)
        if result.is_error:
            return payload, False, str(payload)[:400]
        return payload, True, None

    async def context(self) -> dict[str, Any]:
        """读取预加载的上下文资源，注入 system prompt。"""
        out: dict[str, Any] = {}
        for server, uri in CONTEXT_RESOURCES:
            try:
                res = await self.sessions[server].read_resource(uri)
                text = "\n".join(getattr(c, "text", "") or "" for c in res.contents)
                out[uri] = json.loads(text)
            except Exception as e:  # noqa: BLE001 - 资源读不到不应中断启动
                out[uri] = {"error": f"{type(e).__name__}: {e}"}
        return out
