"""评测执行器：按 cases.json 调真实 MCP 工具。

期望值一律由工具算出，不手写；中心点也走 find_places 解析，
以便同时守住「钉住的 ref 仍在候选里」这件事（anchor 数据漂移会当场显形）。
"""

from __future__ import annotations

from agent.mcp_hub import MCPHub


async def resolve_center(hub: MCPHub, case: dict) -> dict:
    center = case.get("center")
    if not center:
        return {}
    payload, ok, error = await hub.call("find_places", {"name": center["place"], "limit": 5})
    if not ok:
        return {"_error": f"find_places({center['place']}) 调用失败：{error}"}
    hits = payload["result"] if isinstance(payload, dict) and "result" in payload else payload
    match = next((h for h in hits if h["ref"] == center["ref"]), None)
    if match is None:
        return {"_error": f"find_places({center['place']}) 的候选里没有 {center['ref']}，"
                          f"实际候选 {[h['ref'] for h in hits]}"}
    return {"ref": match["ref"], "name": match["name"], "lon": match["lon"], "lat": match["lat"],
            "district": match.get("district"), "match_score": match["match_score"]}


def build_args(case: dict, center: dict) -> dict:
    args = dict(case.get("args") or {})
    ref = args.pop("center_ref", None)
    if ref:
        if center.get("_error"):
            raise ValueError(center["_error"])
        if center.get("ref") != ref:
            raise ValueError(f"中心解析不一致：钉住 {ref}，实际解析出 {center.get('ref')}")
        args["lon"], args["lat"] = center["lon"], center["lat"]
    return args


async def run_case(hub: MCPHub, case: dict) -> dict:
    """执行一条用例。refusal 用例没有工具调用，原样返回以便 agent 层判定。"""
    if not case.get("tool"):
        return {"tool": None, "args": {}, "ok": False,
                "error": "该用例只由 agent 层判定", "payload": {}, "center": {}}
    center = await resolve_center(hub, case)
    try:
        args = build_args(case, center)
    except ValueError as e:
        return {"tool": case["tool"], "args": {}, "ok": False, "error": str(e),
                "payload": {}, "center": center}
    payload, ok, error = await hub.call(case["tool"], args)
    if not ok:
        # 失败时 MCP 侧给的是文本错误，不是结构化结果；统一成 {"error": ...} 便于投影
        payload = {"error": error}
    return {"tool": case["tool"], "args": args, "ok": ok, "error": error,
            "payload": payload, "center": center}
