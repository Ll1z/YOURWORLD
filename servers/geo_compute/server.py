"""geo-compute：空间计算 MCP Server。

分工原则（见 AGENTS.md 硬性规则）：动作与计算作为 Tool，表结构作为 Resource。
所有数值均由 DuckDB / pyproj 算出，LLM 不参与数字生成。

中心点口径：半径查询的中心必须是明确、可命名的锚点（坐标或地名解析结果），
不允许用行政区的几何代表点当圆心，理由见 geo_knowledge 的 anchor_scope。
"""

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

from geo_compute import query

mcp = MCPServer(
    name="geo-compute",
    version="0.2.0",
    description="空间查询与统计。数据为北京五区的 OSM POI 与行政边界，"
                "坐标系 EPSG:4326，距离在 UTM 50N 下按米计算。",
)

CRS_NOTE = "存储 CRS EPSG:4326；距离计算 CRS EPSG:32650（UTM 50N，中央经线 117 度）"
DATA_SOURCE = "OpenStreetMap 北京省级切片 2026-09-11（ODbL 1.0）"

NO_CENTER_HINT = (
    "半径查询必须给出明确的查询中心：请传 lon/lat，"
    "或先用 find_places 把地名（如「王府井」「中关村地铁站」）解析成坐标再调用。"
    "若你想知道的是某个区的总量而不需要中心，请改用 summarize_poi。"
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


class PlaceHit(BaseModel):
    ref: str = Field(description="形如 anchor/12345 或 poi_area/6789，"
                                 "可直接传给 distance_between 的 a_ref / b_ref")
    layer: str
    osm_type: str | None = None
    osm_id: str
    name: str | None = None
    category_key: str | None = None
    category_value: str | None = None
    district: str | None = None
    lon: float
    lat: float
    area_m2: float | None = None
    match_score: int = Field(description="3 = 名称完全相同，2 = 前缀命中，1 = 名称包含。"
                                         "同分时 anchor 层优先，place 类优先，名称更短的优先")


class EndpointInfo(BaseModel):
    label: str
    source: str = Field(description="「调用方给定坐标」或「按引用解析」")
    lon: float
    lat: float
    ref: str | None = None
    layer: str | None = None
    category_value: str | None = None
    district: str | None = None


class DistanceResult(BaseModel):
    a: EndpointInfo
    b: EndpointInfo
    planar_distance_m: float = Field(description="UTM 50N 平面距离，本项目口径的距离，米")
    geodesic_distance_m: float = Field(description="椭球面大地线距离，米，用于互相印证")
    planar_vs_geodesic_pct: float = Field(description="两种距离的相对偏差百分比，应远小于 1")
    bearing_deg: float = Field(description="从 a 指向 b 的方位角，正北为 0，顺时针")
    crs_note: str
    caveats: list[str]


@mcp.resource("compute://schema", name="schema", description="可用数据表、字段与中心点口径")
def get_schema() -> dict:
    """列出 geo_compute 可用的表与字段、数据覆盖范围，以及半径查询的中心点口径。"""
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
            "anchor": {
                "desc": "定位锚点：带名称的地名与车站，供 find_places 解析查询中心用。"
                        "与 POI 层的区别是它回答「这个地方叫什么、在哪」，而不是「这里有什么设施」",
                "fields": ["osm_type", "osm_id", "name", "category_key", "category_value",
                           "district", "lon", "lat", "geom"],
                "category_key_values": ["place", "railway", "highway", "public_transport"],
                "examples": ["中关村（neighbourhood）", "王府井（station）", "西直门（station）"],
            },
        },
        "covered_districts": ["东城区", "西城区", "朝阳区", "丰台区", "海淀区"],
        "presets": query.presets(),
        "primary_key": "(osm_type, osm_id)——way 与 relation 的编号空间独立",
        "center_rule": {
            "rule": "半径查询的中心必须明确、可命名，禁止用行政区几何代表点当圆心",
            "how": "传 lon/lat，或先调 find_places 解析地名后再传解析出的坐标",
            "why": "point_on_surface 是几何产物，可能落在无人区，会产出无意义的 0 结果",
            "resource": "knowledge://scope/anchor_scope",
        },
        "note": "仅覆盖上述五区，district 为 null 的记录在五区之外。",
    }


@mcp.tool()
def find_places(name: str, district: str | None = None, limit: int = 10) -> list[PlaceHit]:
    """按名称检索可作为查询锚点的地名，返回候选点及其坐标。

    用于把「王府井」「中关村地铁站」「北京协和医院」这类地名解析成经纬度。
    匹配优先级：名称完全相同 > 名称前缀命中 > 名称包含，同档内名称更短的排前面。
    返回多条候选时，请选最符合提问语境的一条，并在回答里说明你用的是哪一个。

    参数：
      name      地名关键词
      district  限定区名，可选，用于消歧
      limit     最多返回条数，默认 10
    """
    df, _, _ = query.find_places(name, district, limit)
    return [PlaceHit(**query.plain(r)) for r in df.to_dict("records")]


@mcp.tool()
def query_nearby(lon: float, lat: float, radius_m: float = 1000.0,
                 district: str | None = None, preset: str = "medical") -> NearbyResult:
    """查询一个点周围指定半径内有哪些设施，返回按距离升序的明细。

    参数：
      lon/lat    查询中心（WGS84 经纬度），必填。手上只有地名时先用 find_places 解析。
      radius_m   半径，米，直线距离。
      district   限定区名（东城区/西城区/朝阳区/丰台区/海淀区），省略则不限。
      preset     类别口径预设，见 compute://schema，常用 medical（医疗）与 convenience（便利店/超市）。

    中心点必须是明确位置，本工具不会替你猜中心。
    距离在 EPSG:32650（UTM 50N）平面下计算；面层设施以其代表点参与距离计算。
    结果中的 suspected_duplicates 是相距 150 米内的点面配对，疑似同一设施被点面双挂。
    """
    if lon is None or lat is None:
        raise ToolError(NO_CENTER_HINT)

    try:
        df, _, _ = query.nearby(lon, lat, radius_m, district, preset)
        categories = query.resolve_categories(preset)
    except ValueError as e:
        raise ToolError(str(e)) from e
    dups = query.find_duplicates(df)
    hits = [PoiHit(**{k: r.get(k) for k in PoiHit.model_fields}) for r in df.to_dict("records")]

    return NearbyResult(
        center_lon=lon, center_lat=lat, center_source="调用方给定",
        radius_m=radius_m, district=district, preset=preset,
        categories=categories,
        count_total=len(df),
        count_point=int((df["layer"] == "poi_point").sum()),
        count_area=int((df["layer"] == "poi_area").sum()),
        hits=hits,
        suspected_duplicates=dups,
        crs_note=CRS_NOTE,
        data_source=DATA_SOURCE,
        caveats=[
            "直线距离，非路网可达距离，不能用于可达性结论",
            "OSM 完备性取决于志愿者测绘，结论应表述为「OSM 数据显示」",
            "面层以代表点参与距离计算，与设施实际入口可能有偏差",
        ],
    )


@mcp.tool()
def distance_between(a_ref: str | None = None, a_lon: float | None = None,
                     a_lat: float | None = None, b_ref: str | None = None,
                     b_lon: float | None = None, b_lat: float | None = None) -> DistanceResult:
    """计算两个点之间的距离与方位角。每个点用「引用」或「坐标」二选一给出。

    参数（a、b 两端各自给一种）：
      a_ref / b_ref   形如 "poi_point/12345" 或 "poi_area/6789" 的库内引用。
                      query_nearby 与 find_places 返回的 ref 字段可直接用。
      a_lon/a_lat 等  直接给 WGS84 经纬度。

    同时返回 UTM 平面距离（本项目口径）与椭球面大地线距离，两者应互相印证；
    planar_vs_geodesic_pct 就是这两种算法的相对偏差，正常在 0.01 以内。

    典型用途：回答「A 和 B 相距多远」。这类问题必须用本工具算，不要凭坐标自行估算。
    """
    def build(label: str, ref: str | None, lon: float | None, lat: float | None) -> EndpointInfo:
        if ref:
            try:
                record = query.resolve_ref(ref)
            except ValueError as e:
                raise ToolError(str(e)) from e
            return EndpointInfo(
                label=record.get("name") or ref, source="按引用解析", lon=record["lon"],
                lat=record["lat"], ref=ref, layer=record["layer"],
                category_value=record.get("category_value"), district=record.get("district"),
            )
        if lon is None or lat is None:
            raise ToolError(f"{label} 端点信息不足：请给出 ref，或同时给出 lon 与 lat")
        return EndpointInfo(label=label, source="调用方给定坐标", lon=lon, lat=lat)

    a = build("a", a_ref, a_lon, a_lat)
    b = build("b", b_ref, b_lon, b_lat)
    metrics = query.distance_between(a.model_dump(), b.model_dump())

    return DistanceResult(
        a=a, b=b, **metrics, crs_note=CRS_NOTE,
        caveats=[
            "直线距离，非路网可达距离",
            "面层设施以其代表点参与计算，与设施实际入口可能有偏差",
            "平面距离与大地线距离的偏差由 UTM 投影变形引起，北京范围内应远小于 1%",
        ],
    )


@mcp.tool()
def summarize_poi(district: str | None = None, preset: str = "medical") -> dict:
    """按类别与行政区统计设施数量。不需要中心点，用于回答「某区有多少 X」。

    参数：
      district  限定区名，省略则统计全部五区。
      preset    类别口径预设，常用 medical（医疗）与 convenience（便利店/超市）。

    计数采用 poi_scope 口径：点层与面层取并集、按 (osm_type, osm_id) 唯一、不做几何去重。
    因此同一设施若被点面双挂会被计两次，实测噪声约 2.6%。
    """
    try:
        return query.summarize(district, preset)
    except ValueError as e:
        raise ToolError(str(e)) from e


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
