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
            ("summarize_poi", {"district": "东城区",
                               "categories": ["amenity=university", "amenity=college"]}),
            ("summarize_poi", {"categories": ["高校"]}),
            ("find_places", {"name": "北京协和医院"}),
            ("find_places", {"name": "中关村", "district": "海淀区", "limit": 5}),
            ("find_places", {"name": "王府井", "limit": 3}),
            ("query_nearby", {"lon": 116.4074, "lat": 39.9042, "radius_m": 1000, "preset": "medical"}),
            ("query_nearby", {"lon": 116.4074, "lat": 39.9042, "radius_m": 800, "preset": "convenience"}),
            ("query_nearby", {"lon": 116.3758342, "lat": 39.9668285, "radius_m": 2000,
                              "categories": ["高校"]}),
            ("query_nearby", {"lon": 116.4074, "lat": 39.9042, "radius_m": 1000,
                              "categories": ["银行", "park"]}),
            ("query_nearby", {"lon": 116.3758342, "lat": 39.9668285, "radius_m": 10000,
                              "categories": ["地铁站"], "include_anchor": True}),
            ("query_nearby", {"lon": 116.3758342, "lat": 39.9668285, "radius_m": 2000,
                              "categories": ["高校"], "include_anchor": True}),
            ("distance_between", {"a_ref": "poi_area/19126182", "b_ref": "poi_point/13888669701"}),
            ("distance_between", {"a_ref": "anchor/5196349280", "b_ref": "anchor/6617849503"}),
            ("distance_between", {"a_lon": 116.4074, "a_lat": 39.9042,
                                  "b_lon": 116.4171, "b_lat": 39.9103}),
            ("run_python", {"code": "RESULT = con.execute('select count(*) from districts').fetchone()[0]",
                            "purpose": "冒烟：沙箱工具端到端可用"}),
        ],
        "resources": ["compute://schema", "compute://categories"],
        "templates": [],
        "expect_error": [
            ("query_nearby", {"district": "东城区", "radius_m": 1000}, "lon"),
            ("distance_between", {"a_ref": "poi_area/19126182"}, "b 端点信息不足"),
            ("query_nearby", {"lon": 116.4074, "lat": 39.9042, "radius_m": 1000}, "必须指定要查什么类别"),
            ("summarize_poi", {"district": "东城区"}, "必须指定要查什么类别"),
            ("query_nearby", {"lon": 116.4074, "lat": 39.9042, "preset": "不存在的口径"}, "未知口径预设"),
            ("query_nearby", {"lon": 116.4074, "lat": 39.9042, "categories": ["医院院"]}, "不认识的类别"),
            ("query_nearby", {"lon": 116.4074, "lat": 39.9042, "categories": ["地铁站"]}, "只在 anchor 层"),
            ("query_nearby", {"lon": 116.4074, "lat": 39.9042,
                              "categories": ["amenity=不存在的值"]}, "本库没有 amenity=不存在的值"),
            ("query_nearby", {"lon": 116.4074, "lat": 39.9042, "categories": ["医院"],
                              "preset": "medical"}, "只能给一个"),
            ("distance_between", {"a_ref": "poi_point/不存在", "b_lon": 116.4, "b_lat": 39.9}, "库中找不到"),
        ],
    },
    {
        "name": "geo_catalog",
        "module": "geo_catalog.server",
        "tools": [
            ("list_datasets", {}),
            ("search_datasets", {"query": "坐标系 偏移"}),
            ("search_datasets", {"query": "哪份数据有 GCJ-02 偏移坑"}),
            ("search_knowledge", {"query": "地铁站为什么查不到"}),
            ("search_knowledge", {"query": "面积用哪种算法算"}),
            ("search_knowledge", {"query": "医院", "kinds": ["dataset_pitfall"], "limit": 2}),
            ("search_knowledge", {"query": "半径查询为什么不能拿行政区几何代表点当圆心",
                                  "kinds": ["experience"]}),
        ],
        "resources": ["catalog://datasets", "catalog://dataset/osm-beijing-full",
                      "catalog://knowledge"],
        "templates": ["catalog://dataset/{dataset_id}"],
        "expect_resource_error": [("catalog://dataset/not-registered", "未登记的数据集")],
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
                      "knowledge://scope/anchor_scope", "knowledge://coords/systems",
                      "knowledge://categories/aliases", "knowledge://experience"],
        "templates": ["knowledge://scope/{scope_id}"],
        "expect_resource_error": [("knowledge://scope/not-registered", "未登记的口径")],
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
            for name, args, expect in case.get("expect_error", []):
                label = f"expect_error {name}({json.dumps(args, ensure_ascii=False)})"
                try:
                    res = await asyncio.wait_for(session.call_tool(name, args), TIMEOUT)
                    text = json.dumps(res.content, ensure_ascii=False, default=str)
                    results.append((label, bool(res.is_error) and expect in text, text[:200]))
                except Exception as e:  # noqa: BLE001
                    results.append((label, expect in str(e), f"{type(e).__name__}: {e}"))

            for uri, expect in case.get("expect_resource_error", []):
                label = f"expect_resource_error {uri}"
                try:
                    await asyncio.wait_for(session.read_resource(uri), TIMEOUT)
                    results.append((label, False, "本应报错，却读取成功了"))
                except Exception as e:  # noqa: BLE001
                    results.append((label, expect in str(e), f"{type(e).__name__}: {str(e)[:160]}"))

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
