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
from geo_compute.sandbox import LocalProcessSandbox

mcp = MCPServer(
    name="geo-compute",
    version="0.5.0",
    description="空间查询与统计。数据为北京五区的 OSM POI 与行政边界，"
                "坐标系 EPSG:4326，距离在 UTM 50N 下按米计算；"
                "另有受限沙箱 run_python 供工具覆盖不到的计算使用。",
)

CRS_NOTE = "存储 CRS EPSG:4326；距离计算 CRS EPSG:32650（UTM 50N，中央经线 117 度）"
DATA_SOURCE = "OpenStreetMap 北京省级切片 2026-09-11（ODbL 1.0）"
# compute://categories 只列 >= 该条数的类别，避免上下文被长尾撑爆（校验仍用全量清单）
CATALOG_MIN_COUNT = 5

# 沙箱后端只在 geo_compute.sandbox 里换（L1 受限子进程 → L2 Docker），工具签名不动
SANDBOX = LocalProcessSandbox()

NO_CENTER_HINT = (
    "半径查询必须给出明确的查询中心：请传 lon/lat，"
    "或先用 find_places 把地名（如「王府井」「中关村地铁站」）解析成坐标再调用。"
    "若你想知道的是某个区的总量而不需要中心，请改用 summarize_poi。"
)

ANCHOR_ONLY_HINT = (
    "{labels} 只在 anchor 层（地铁站、火车站、公交站、地名这类定位锚点），"
    "默认的半径查询只覆盖 poi_point / poi_area 两层。要查它们请把 include_anchor 设为 true；"
    "anchor 的命中会单独计在 count_anchor，不会混进 count_point / count_area。"
)


class PoiHit(BaseModel):
    layer: str = Field(description="poi_point / poi_area，include_anchor=true 时还会有 anchor")
    osm_id: str
    name: str | None = None
    category_key: str
    category_value: str
    district: str | None = None
    lon: float
    lat: float
    dist_m: float = Field(description="到查询中心的直线距离，米")
    area_m2: float | None = Field(default=None, description="仅面层：大地线面积，平方米")


class CategoryTally(BaseModel):
    key: str | None = Field(default=None,
                            description="标签键；null 表示不限键（同值的 amenity/office 都算）")
    value: str
    label: str = Field(description="展示用标签，如 amenity=university")
    count: int = Field(description="命中数；0 表示这类设施在查询范围内确实没有")
    matched: bool


class NearbyResult(BaseModel):
    center_lon: float
    center_lat: float
    center_source: str
    radius_m: float
    district: str | None
    preset: str | None = Field(default=None,
                               description="本次使用的预设名；用 categories 自由类别时为 null")
    categories: list[CategoryTally] = Field(
        description="每个请求类别的命中数明细，含 0 命中——0 是结论本身，不是工具失败")
    count_total: int = Field(
        description="命中总数，等于 count_point + count_area + count_anchor；"
                    "默认 include_anchor=false 时就是 POI 两层之和。明细被 limit 截断时这个数仍然完整")
    count_point: int
    count_area: int
    count_anchor: int = Field(
        default=0,
        description="anchor 层（地铁站、火车站、公交站、地名）的命中数，只可能出现在 "
                    "include_anchor=true 的查询里；与 POI 两层分开计数，不混进设施口径")
    count_returned: int = Field(description="本次返回的明细条数")
    returned_point: int
    returned_area: int
    returned_anchor: int = Field(default=0, description="本次返回的明细里 anchor 层的条数")
    hits_truncated: bool = Field(
        description="true 表示明细被 limit 截断：计数完整、列表不全。"
                    "此时不要用沙箱去捞全量，调大 limit 重查即可")
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
                        "与 POI 层的区别是它回答「这个地方叫什么、在哪」，而不是「这里有什么设施」。"
                        "默认不参与半径查询；要查站点类设施就给 query_nearby 传 include_anchor=true",
                "fields": ["osm_type", "osm_id", "name", "category_key", "category_value",
                           "district", "lon", "lat", "x_utm", "y_utm", "geom"],
                "category_key_values": ["place", "railway", "highway", "public_transport"],
                "examples": ["中关村（neighbourhood）", "王府井（station）", "西直门（station）"],
            },
        },
        "covered_districts": ["东城区", "西城区", "朝阳区", "丰台区", "海淀区"],
        "presets": query.presets(),
        "free_categories": "categories 参数可自由指定库里存在的任何类别值（469 个 key×value 组合），"
                           "不限于上面几个 presets；清单见 compute://categories，"
                           "中文说法见 knowledge://categories/aliases",
        "category_rule": {
            "rule": "类别参数没有默认值；不认识、拼错的类别直接报错并给近似建议",
            "how": "传 categories（自由类别，中英文皆可）或 preset（常用组合），二选一",
            "why": "给默认类别会把「想查高校却拿到医院」变成静默替换，看起来像正常结果",
            "resource": "compute://categories",
        },
        "primary_key": "(osm_type, osm_id)——way 与 relation 的编号空间独立",
        "center_rule": {
            "rule": "半径查询的中心必须明确、可命名，禁止用行政区几何代表点当圆心",
            "how": "传 lon/lat，或先调 find_places 解析地名后再传解析出的坐标",
            "why": "point_on_surface 是几何产物，可能落在无人区，会产出无意义的 0 结果",
            "resource": "knowledge://scope/anchor_scope",
        },
        "note": "仅覆盖上述五区，district 为 null 的记录在五区之外。",
    }


@mcp.resource("compute://categories", name="categories",
              description="可查询的 POI 类别清单：库内真实存在的 key×value 与计数")
def get_categories() -> dict:
    """列出库里真实存在的 POI 类别，按标签键分组。

    只列出现次数 >= CATALOG_MIN_COUNT 的值（5 条，覆盖 99.3% 的记录），避免上下文被长尾
    撑爆；查询与校验用的是全量清单，长尾值照样能查，拼错才会报错。
    计数是全库口径（五区内外都算），不是某个半径内的计数。
    anchor_layer_only 单列只在 anchor 层的值（地铁站、公交站等），它们不计入下面两个总数，
    要 query_nearby 传 include_anchor=true 才查得到。
    """
    inv = query.category_inventory()
    anc = query.anchor_inventory()
    shown = inv[inv["n"] >= CATALOG_MIN_COUNT]
    by_key: dict[str, dict[str, int]] = {}
    for key, value, n in zip(shown["category_key"], shown["category_value"], shown["n"]):
        by_key.setdefault(key, {})[value] = int(n)
    anchor_only: dict[str, dict[str, int]] = {}
    for key, value, n in zip(anc["category_key"], anc["category_value"], anc["n"]):
        anchor_only.setdefault(key, {})[value] = int(n)
    return {
        "how_to_use": [
            "query_nearby / summarize_poi 的 categories 直接收下面的值，或写 'key=value' 精确限定标签键",
            "中文说法（高校 / 药店 / 公园…）见 knowledge://categories/aliases，由服务端展开，不需要自己猜标签",
            "category_value 是主分类，一个设施只归一个 key；同名值出现在多个 key 下时，裸值写法会全部计入",
            "认不出来的类别会直接报错并给近似建议，不会静默返回 0 条",
        ],
        "anchor_layer_only": {
            "how_to_use": "这些 (key, value) 只存在于 anchor 层，上面的 by_key 里没有它们；"
                          "query_nearby 默认查不到，要查就传 include_anchor=true，"
                          "命中单独计在 count_anchor。summarize_poi 不覆盖 anchor 层。",
            "values": anchor_only,
        },
        "total_combinations": int(len(inv)),
        "total_records": int(inv["n"].sum()),
        "shown_min_count": CATALOG_MIN_COUNT,
        "omitted_values": int(len(inv) - len(shown)),
        "not_available": sorted(query.load_aliases().get("not_available") or {}),
        "presets": query.presets(),
        "by_key": by_key,
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
                 district: str | None = None, categories: list[str] | None = None,
                 preset: str | None = None, limit: int = 50,
                 include_anchor: bool = False) -> NearbyResult:
    """查询一个点周围指定半径内有哪些设施，返回按距离升序的明细。

    参数：
      lon/lat    查询中心（WGS84 经纬度），必填。手上只有地名时先用 find_places 解析。
      radius_m   半径，米，直线距离。
      district   限定区名（东城区/西城区/朝阳区/丰台区/海淀区），省略则不限。
      categories 要查什么类别，自由填写，三种写法可混用：
                   OSM 值    "university"（不限标签键）、"amenity=university"（限定键）
                   中文说法  "高校"、"药店"、"公园"（映射见 knowledge://categories/aliases）
                   多个类别  ["高校", "银行"]
                 可查的值清单见 compute://categories。
      preset     常用组合的快捷方式（见 compute://schema 的 presets），与 categories 二选一。
      limit      明细最多回多少条，默认 50；传 0 表示不限。count_total 与逐类别计数
                 永远是全量，被截断时 hits_truncated 为 true。一次大半径查询可能命中上万条，
                 全塞回来只会把结果撑爆、让真正有用的部分看不见。
      include_anchor
                 是否把 anchor 层（地铁站、火车站、公交站、地名等定位锚点）也算进来。
                 默认 false，只查 poi_point / poi_area 两层——anchor 回答「在哪」而不是
                 「有什么」，混进设施统计会凭空多出几百个地名节点。问地铁站这类设施时传 true，
                 命中单独记在 count_anchor / returned_anchor，不混进 count_point / count_area。

    categories 与 preset 都不给会直接报错：本工具没有默认类别。默认一个类别会让
    「想查高校却拿到医院」变成静默替换，看起来像正常结果。
    中心点必须是明确位置，本工具不会替你猜中心。
    距离在 EPSG:32650（UTM 50N）平面下计算；面层设施以其代表点参与距离计算。
    结果中的 suspected_duplicates 是相距 150 米内的点面配对，疑似同一设施被点面双挂。
    """
    if lon is None or lat is None:
        raise ToolError(NO_CENTER_HINT)

    try:
        specs = query.resolve_categories(categories, preset)
        if not include_anchor:
            # anchor 专属类别在 POI 两层里必然是 0 条。这个 0 不是「附近没有」，
            # 是「查错层了」，必须报出来，不能让调用方拿个 0 回去当初结论。
            only_anchor = query.missing_from_poi(specs)
            if only_anchor:
                raise ToolError(ANCHOR_ONLY_HINT.format(
                    labels="、".join(s.label for s in only_anchor)))
        df, _, _ = query.nearby(lon, lat, radius_m, district, specs,
                                include_anchor=include_anchor)
    except ValueError as e:
        raise ToolError(str(e)) from e
    dups = query.find_duplicates(df)
    # 计数在截断前算完，明细按距离取前 limit 条：
    # 实测一次 10 公里半径的高校查询返回 158 条、37920 字符，回喂时被截到 20000，
    # 模型拿不到完整列表就跑去沙箱里捞，连着几步都耗在那儿。
    count_total, count_point, count_area, count_anchor = len(df), \
        int((df["layer"] == "poi_point").sum()), int((df["layer"] == "poi_area").sum()), \
        int((df["layer"] == "anchor").sum())
    shown = df if limit <= 0 else df.head(int(limit))
    hits = [PoiHit(**{k: r.get(k) for k in PoiHit.model_fields}) for r in shown.to_dict("records")]

    return NearbyResult(
        center_lon=lon, center_lat=lat, center_source="调用方给定",
        radius_m=radius_m, district=district, preset=preset,
        categories=[CategoryTally(**t) for t in query.category_tally(df, specs)],
        count_total=count_total,
        count_point=count_point,
        count_area=count_area,
        count_anchor=count_anchor,
        count_returned=len(shown),
        returned_point=int((shown["layer"] == "poi_point").sum()),
        returned_area=int((shown["layer"] == "poi_area").sum()),
        returned_anchor=int((shown["layer"] == "anchor").sum()),
        hits_truncated=len(shown) < count_total,
        hits=hits,
        suspected_duplicates=dups,
        crs_note=CRS_NOTE,
        data_source=DATA_SOURCE,
        caveats=[
            "直线距离，非路网可达距离，不能用于可达性结论",
            "OSM 完备性取决于志愿者测绘，结论应表述为「OSM 数据显示」",
            "面层以代表点参与距离计算，与设施实际入口可能有偏差",
            "categories 里 count=0 的类别就是「确实没有」，不要换成别的类别来替代",
            "hits 只是最近 limit 条；hits_truncated 为 true 时完整列表要用更大的 limit 重查，"
            "不要用 run_python 绕过去捞全量明细——那等于在脚本里重写一遍查询口径",
            "默认只覆盖 poi_point / poi_area 两层；地铁站、火车站、公交站只在 anchor 层，"
            "要查就给 include_anchor 传 true，命中单独计在 count_anchor",
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
def summarize_poi(district: str | None = None, categories: list[str] | None = None,
                  preset: str | None = None) -> dict:
    """按类别与行政区统计设施数量。不需要中心点，用于回答「某区有多少 X」。

    参数：
      district  限定区名，省略则统计全部五区。
      categories 要统计的类别，写法与 query_nearby 相同（自由类别 / key=value / 中文说法），
                 可给多个，清单见 compute://categories，中文映射见 knowledge://categories/aliases。
      preset    常用组合的快捷方式，与 categories 二选一。

    两个都不给会直接报错，本工具没有默认类别。
    返回里的 category_tally 是每个请求类别的计数，count=0 就是「这一类没有」。
    计数采用 poi_scope 口径：点层与面层取并集、按 (osm_type, osm_id) 唯一、不做几何去重。
    因此同一设施若被点面双挂会被计两次，实测噪声约 2.6%。
    """
    try:
        specs = query.resolve_categories(categories, preset)
        out = query.summarize(district, specs)
    except ValueError as e:
        raise ToolError(str(e)) from e
    return {**out, "preset": preset, "crs_note": CRS_NOTE,
            "caveats": [
                "OSM 完备性取决于志愿者测绘，结论应表述为「OSM 数据显示」",
                "category_tally 里 count=0 的类别就是「没有」，不要换成别的类别来替代",
                "五区外的 POI district 为 null，统计不加 district 时不计入",
            ]}


@mcp.tool()
def list_districts(name: str | None = None) -> list[dict]:
    """列出五区基本信息：行政区划代码、面积、边界来源与范围。

    参数 name 可指定单个区名。面积口径为大地线面积（平方公里）。
    注意 source 字段：西城区边界来自 DataV 纠偏补齐，其余四区取自 OSM。
    """
    return query.district_info(name)


@mcp.tool()
def run_python(code: str, purpose: str = "", timeout_s: float = 30.0,
               memory_mb: int = 1024) -> dict:
    """在受限沙箱里跑一段 Python，用于固定工具没封装过的计算。

    该用它：一次性的空间加工——自定义缓冲区、按距离分箱、多表关联算指标、
    导出中间结果。
    不该用它：能用 query_nearby / summarize_poi / distance_between / find_places
    回答的，一律先用工具。工具的口径全项目唯一，代码里的口径是你临时写的。

    预置环境：con（只读 DuckDB 连接，与工具同一份库）、query（查询口径模块）、
    pd / np / gpd / shapely / pyproj，以及 RUN_DIR（唯一可写目录）、DATA_DIR、DB。
    想把结构化结果带回来，把值赋给 RESULT 变量；写进 RUN_DIR 的文件会作为产物回报。

    围栏：禁网、禁起子进程；默认 30 秒超时、1024 MB 内存上限；读取项目内文件
    只放行 data/ 与运行目录。失败会连 traceback 一起返回，照着它改再跑一次。
    """
    return SANDBOX.run(code, purpose=purpose, timeout_s=timeout_s,
                       memory_mb=memory_mb).as_dict()


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
