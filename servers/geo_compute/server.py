"""geo-compute：空间计算 MCP Server。

分工原则（见 AGENTS.md 硬性规则）：动作与计算作为 Tool，表结构作为 Resource。
所有数值均由 DuckDB / pyproj 算出，LLM 不参与数字生成。
"""

from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, Field

from geo_compute import query

mcp = MCPServer(
    name="geo-compute",
    version="0.1.0",
    description="空间查询与统计。数据为北京五区的 OSM POI 与行政边界，"
                "坐标系 EPSG:4326，距离在 UTM 50N 下按米计算。",
)


class PoiHit(BaseModel):
    layer: str = Field(description="poi_point 或 poi_area")
    osm_id: str
    name: str | None = None
    category_key: str
    category_value: str
    district: str | None = None
    lon: float
    lat: float
    dist_m: float = Field(description="到查询中心的直线距离，米")
    area_m2: float | None = Field(default=None, description="仅面层：大地线面积，平方米")


class NearbyResult(BaseModel):
    center_lon: float
    center_lat: float
    center_source: str
    radius_m: float
    district: str | None
    preset: str
    categories: list[str]
    count_total: int
    count_point: int
    count_area: int
    hits: list[PoiHit]
    suspected_duplicates: list[dict] = Field(
        default_factory=list,
        description="相距 150 米内的点面配对，疑似同一设施被点面双挂。按口径保留不合并。",
    )
    crs_note: str
    data_source: str
    caveats: list[str]


@mcp.resource("compute://schema", name="schema", description="可用数据表与字段说明")
def get_schema() -> dict:
    """列出 geo_compute 可用的表与字段，以及数据覆盖范围。"""
    return {
        "database": str(query.DB.relative_to(query.ROOT)),
        "tables": {
            "districts": {
                "desc": "五区行政边界",
                "fields": ["adcode", "name", "source", "area_km2", "geom"],
            },
            "poi_point": {
                "desc": "点状 POI（OSM node）",
                "fields": ["osm_type", "osm_id", "name", "category_key", "category_value",
                           "district", "lon", "lat", "x_utm", "y_utm", "geom"],
            },
            "poi_area": {
                "desc": "面状 POI（OSM way/relation），lon/lat 为其代表点",
                "fields": ["osm_type", "osm_id", "name", "category_key", "category_value",
                           "district", "lon", "lat", "x_utm", "y_utm", "area_m2", "geom"],
            },
        },
        "covered_districts": ["东城区", "西城区", "朝阳区", "丰台区", "海淀区"],
        "presets": query.presets(),
        "note": "仅覆盖上述五区，district 为 null 的记录在五区之外。",
    }


@mcp.tool()
def query_nearby(lon: float | None = None, lat: float | None = None, radius_m: float = 1000.0,
                 district: str | None = None, preset: str = "medical") -> NearbyResult:
    """查询一个点周围指定半径内有哪些设施，返回按距离升序的明细。

    参数：
      lon/lat    查询中心（WGS84 经纬度）。省略时必须给出 district，将用该区边界的代表点作中心。
      radius_m   半径，米，直线距离。
      district   限定区名（东城区/西城区/朝阳区/丰台区/海淀区），省略则不限。
      preset     类别口径预设，见 compute://schema，常用 medical（医疗）与 convenience（便利店/超市）。

    距离在 EPSG:32650（UTM 50N）平面下计算；面层设施以其代表点参与距离计算。
    结果中的 suspected_duplicates 是相距 150 米内的点面配对，疑似同一设施被点面双挂。
    """
    center_source = "调用方给定"
    if lon is None or lat is None:
        if not district:
            raise ValueError("需给出 lon/lat，或给出 district 以使用该区代表点作中心")
        lon, lat = query.district_point_on_surface(district)
        center_source = f"{district}边界的 point_on_surface（自动选取）"

    df, _, _ = query.nearby(lon, lat, radius_m, district, preset)
    dups = query.find_duplicates(df)
    hits = [PoiHit(**{k: r.get(k) for k in PoiHit.model_fields}) for r in df.to_dict("records")]

    return NearbyResult(
        center_lon=lon, center_lat=lat, center_source=center_source,
        radius_m=radius_m, district=district, preset=preset,
        categories=query.resolve_categories(preset),
        count_total=len(df),
        count_point=int((df["layer"] == "poi_point").sum()),
        count_area=int((df["layer"] == "poi_area").sum()),
        hits=hits,
        suspected_duplicates=dups,
        crs_note="存储 CRS EPSG:4326；距离计算 CRS EPSG:32650（UTM 50N，中央经线 117 度）",
        data_source="OpenStreetMap 北京省级切片 2026-09-11（ODbL 1.0）",
        caveats=[
            "直线距离，非路网可达距离，不能用于可达性结论",
            "OSM 完备性取决于志愿者测绘，结论应表述为「OSM 数据显示」",
            "面层以代表点参与距离计算，与设施实际入口可能有偏差",
        ],
    )


@mcp.tool()
def summarize_poi(district: str | None = None, preset: str = "medical") -> dict:
    """按类别与行政区统计设施数量。

    参数：
      district  限定区名，省略则统计全部五区。
      preset    类别口径预设，常用 medical（医疗）与 convenience（便利店/超市）。

    计数采用 poi_scope 口径：点层与面层取并集、按 osm_id 唯一、不做几何去重。
    因此同一设施若被点面双挂会被计两次，实测噪声约 2.6%。
    """
    return query.summarize(district, preset)


@mcp.tool()
def list_districts(name: str | None = None) -> list[dict]:
    """列出五区基本信息：行政区划代码、面积、边界来源与范围。

    参数 name 可指定单个区名。面积口径为大地线面积（平方公里）。
    注意 source 字段：西城区边界来自 DataV 纠偏补齐，其余四区取自 OSM。
    """
    return query.district_info(name)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
