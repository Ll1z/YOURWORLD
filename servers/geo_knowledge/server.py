"""geo-knowledge：标准、术语与口径的 MCP Server。

分工原则（见 AGENTS.md 硬性规则）：口径与标准作为 Resource 承载上下文，坐标转换作为 Tool 承载动作。
"""

import json
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceNotFoundError

from geo_knowledge.coords.gcj02 import gcj02_to_wgs84, wgs84_to_gcj02

ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE = ROOT / "servers" / "geo_knowledge"

mcp = MCPServer(
    name="geo-knowledge",
    version="0.2.0",
    description="地理空间分析的标准、术语与分析口径，外加程序性记忆（经验库）。"
                "回答「这个数按什么口径算」「这个坑踩过没有」这类问题。",
)


@mcp.resource("knowledge://scopes", name="scopes", description="所有已登记口径的清单")
def list_scopes() -> dict:
    """列出全部口径文件的 id、名称与适用范围。"""
    items = []
    for p in sorted(KNOWLEDGE.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if not d.get("id") or not d.get("name"):
            continue  # 这个目录不只放口径文件，缺 id/name 的不是口径
        items.append({"id": d.get("id"), "name": d.get("name"), "version": d.get("version"),
                      "applies_to": d.get("applies_to", []), "uri": f"knowledge://scope/{d.get('id')}"})
    p = KNOWLEDGE / "coords" / "systems.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    items.append({"id": d.get("id"), "name": d.get("name"), "version": d.get("version"),
                  "applies_to": ["坐标系与度量单位"], "uri": "knowledge://coords/systems"})
    return {"count": len(items), "items": items}


@mcp.resource("knowledge://scope/{scope_id}", name="scope", description="按 id 读取口径定义全文")
def get_scope(scope_id: str) -> dict:
    """读取指定口径的完整定义，包含默认取值、证据与已知局限。"""
    p = next((q for q in KNOWLEDGE.glob("*.json") if q.stem == scope_id), None)
    if p is None:
        raise ResourceNotFoundError(f"未登记的口径: {scope_id}")
    scope = json.loads(p.read_text(encoding="utf-8"))
    if not scope.get("id"):
        raise ResourceNotFoundError(f"未登记的口径: {scope_id}")
    return scope


@mcp.resource("knowledge://categories/aliases", name="category-aliases",
              description="POI 类别的中文别名表（含只在 anchor 层的别名），以及库中确实没有的类别及原因")
def get_category_aliases() -> dict:
    """中文说法 → OSM 类别的映射，由 geo_compute 在解析 categories 参数时展开。

    别名属于术语层（「高校」对应哪些 OSM 值、OSM 的 school 不区分中小学），
    因此放在 knowledge 侧，而不是散落进查询代码。
    """
    return json.loads((KNOWLEDGE / "categories" / "aliases.json").read_text(encoding="utf-8"))


@mcp.resource("knowledge://coords/systems", name="coordinate-systems",
              description="坐标系与面积/距离度量口径")
def get_coord_systems() -> dict:
    """读取坐标系与度量口径：存储 CRS、面积与距离的计算方式、已知坐标系偏移。"""
    return json.loads((KNOWLEDGE / "coords" / "systems.json").read_text(encoding="utf-8"))


@mcp.resource("knowledge://experience", name="experience",
              description="程序性记忆：已确认的经验条目（踩过的坑与当时的处理）")
def get_experience() -> dict:
    """经验库：lessons 是已确认的（每条都能追到一次实测或一份口径文件），candidates 还没确认。

    candidates 由 scripts/distill_experience.py 从运行轨迹里抽出来，只有出现次数与例子、
    没有 statement，不要当结论用；确认之后才提升为 lesson。
    """
    return json.loads((KNOWLEDGE / "experience" / "lessons.json").read_text(encoding="utf-8"))


@mcp.tool()
def convert_coordinates(lon: float, lat: float, from_crs: str, to_crs: str) -> dict:
    """在坐标系之间转换一个点。

    支持 GCJ-02（火星坐标，DataV/高德/腾讯使用）与任意 EPSG 之间的转换。
    from_crs / to_crs 取 'GCJ-02'、'WGS84'、'EPSG:4326'、'EPSG:32650' 等形式。
    涉及 GCJ-02 时会走加密坐标反解，不涉及则走 pyproj。
    """
    src, dst = from_crs.upper(), to_crs.upper()
    if src == "WGS84":
        src = "EPSG:4326"
    if dst == "WGS84":
        dst = "EPSG:4326"

    if src == "GCJ-02" and dst == "EPSG:4326":
        x, y = gcj02_to_wgs84(lon, lat)
    elif src == "EPSG:4326" and dst == "GCJ-02":
        x, y = wgs84_to_gcj02(lon, lat)
    elif src == dst:
        x, y = lon, lat
    else:
        from pyproj import Transformer
        t = Transformer.from_crs(src, dst, always_xy=True)
        x, y = t.transform(lon, lat)
    return {"lon": float(x), "lat": float(y), "from_crs": from_crs, "to_crs": to_crs}


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
