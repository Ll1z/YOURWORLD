"""MCP Server 冒烟测试：用 stdio 拉起三个 Server，校验 tools / resources / templates 全部可用。

用法：uv run python scripts/mcp_smoke_test.py
退出码非 0 表示有未通过的用例。

注意（mcp 2.x 实测）：
  - ClientSession 进入上下文后需显式 await session.initialize()，否则服务端拒绝后续请求
  - 返回模型字段为 snake_case：server_info / protocol_version / resource_templates / structured_content
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]
TIMEOUT = 60.0

CASES = [
    {
        "name": "geo_compute",
        "module": "geo_compute.server",
        "tools": [
            ("list_districts", {}),
            ("list_districts", {"name": "东城区"}),
            ("summarize_poi", {"district": "东城区", "preset": "medical"}),
            ("query_nearby", {"district": "东城区", "radius_m": 1000, "preset": "medical"}),
            ("query_nearby", {"lon": 116.4074, "lat": 39.9042, "radius_m": 800, "preset": "convenience"}),
        ],
        "resources": ["compute://schema"],
        "templates": [],
    },
    {
        "name": "geo_catalog",
        "module": "geo_catalog.server",
        "tools": [
            ("list_datasets", {}),
            ("search_datasets", {"query": "坐标系 偏移"}),
        ],
        "resources": ["catalog://datasets", "catalog://dataset/osm-beijing-full"],
        "templates": ["catalog://dataset/{dataset_id}"],
    },
    {
        "name": "geo_knowledge",
        "module": "geo_knowledge.server",
        "tools": [
            ("convert_coordinates", {"lon": 116.4074, "lat": 39.9042,
                                     "from_crs": "WGS84", "to_crs": "GCJ-02"}),
            ("convert_coordinates", {"lon": 116.412, "lat": 39.906,
                                     "from_crs": "GCJ-02", "to_crs": "WGS84"}),
        ],
        "resources": ["knowledge://scopes", "knowledge://scope/poi_scope",
                      "knowledge://coords/systems"],
        "templates": ["knowledge://scope/{scope_id}"],
    },
]


def _leaves(exc: BaseException) -> list[BaseException]:
    if isinstance(exc, BaseExceptionGroup):
        out: list[BaseException] = []
        for sub in exc.exceptions:
            out.extend(_leaves(sub))
        return out
    return [exc]


async def run_case(case: dict) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", case["module"]],
        cwd=str(ROOT),
        env={"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            results.append(("initialize", True,
                            f"{init.server_info.name} {init.server_info.version} "
                            f"proto={init.protocol_version}"))

            tools = (await session.list_tools()).tools
            results.append(("tools/list", len(tools) > 0, ",".join(t.name for t in tools)))

            resources = (await session.list_resources()).resources
            results.append(("resources/list", len(resources) > 0,
                            ",".join(str(r.uri) for r in resources)))

            templates = (await session.list_resource_templates()).resource_templates
            got = sorted(t.uri_template for t in templates)
            want = sorted(case["templates"])
            results.append(("resources/templates", got == want,
                            f"got {got} want {want}"))

            for uri in case["resources"]:
                try:
                    res = await asyncio.wait_for(session.read_resource(uri), TIMEOUT)
                    chars = sum(len(getattr(c, "text", "") or "") for c in res.contents)
                    results.append((f"read {uri}", chars > 2, f"{chars} chars"))
                except Exception as e:  # noqa: BLE001
                    results.append((f"read {uri}", False, f"{type(e).__name__}: {e}"))

            for name, args in case["tools"]:
                label = f"call {name}({json.dumps(args, ensure_ascii=False)})"
                try:
                    res = await asyncio.wait_for(session.call_tool(name, args), TIMEOUT)
                    payload = res.structured_content if res.structured_content is not None else res.content
                    detail = json.dumps(payload, ensure_ascii=False, default=str)
                    results.append((label, not res.is_error, detail[:220]))
                except Exception as e:  # noqa: BLE001
                    results.append((label, False, f"{type(e).__name__}: {e}"))
    return results


async def main() -> int:
    failed = 0
    for case in CASES:
        print(f"\n===== {case['name']} ({case['module']}) =====")
        try:
            results = await asyncio.wait_for(run_case(case), TIMEOUT * 4)
        except BaseException as e:  # noqa: BLE001
            for leaf in _leaves(e):
                print(f"  [FAIL] 通信异常: {type(leaf).__name__}: {leaf}")
                failed += 1
            continue
        for label, ok, detail in results:
            if not ok:
                failed += 1
            print(f"  [{'PASS' if ok else 'FAIL'}] {label} :: {detail}")
    print(f"\n合计失败用例: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
