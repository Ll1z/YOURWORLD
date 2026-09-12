"""geo_compute 的查询核心。

供 MCP Tool 与命令行脚本共用，保证 SQL 与口径只有一处定义。

中心点口径（重要）：
    半径查询的中心必须是一个明确、可命名的位置（坐标或地名锚点），
    和路径规划要先选起点是一个道理。禁止用行政区的几何代表点当圆心——
    point_on_surface 只是几何产物，可能落在无人区，会让「某区 X 米内有什么」
    这类问题产生看似合理实则无意义的 0 结果。

类别口径（重要）：
    可查的类别不是几个预设，而是库里真实存在的 (category_key, category_value) 全集；
    预设只是常用组合的快捷方式。因此类别参数一律不给默认值——有默认值就会出现
    「想查高校、拿到医院」这种无从察觉的替换，宁可直接报错。中文说法（「高校」）
    由 geo_knowledge/categories/aliases.json 展开成 OSM 值，展开处只有这一份实现。

图层口径（重要）：
    anchor 层（地铁站、火车站、公交站、地名等定位锚点）默认不参与半径查询——它回答
    「这个地方叫什么、在哪」，不回答「这里有什么设施」，混进设施统计会凭空多出几百个
    地名节点。要查就显式传 include_anchor=True，并且分层计数，不把两层的数混成一个。
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import shapely
from pyproj import Geod, Transformer

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "processed" / "geo.duckdb"
SCOPE_FILE = ROOT / "servers" / "geo_knowledge" / "poi_scope.json"
ALIAS_FILE = ROOT / "servers" / "geo_knowledge" / "categories" / "aliases.json"
UTM = "EPSG:32650"
DUP_M = 150.0
# 各图层的字段列表：只有 poi_area 有面积列
_TABLE_COLUMNS = {
    "poi_point": ("osm_type", "osm_id", "name", "category_key", "category_value",
                  "district", "lon", "lat", "NULL::DOUBLE AS area_m2"),
    "poi_area": ("osm_type", "osm_id", "name", "category_key", "category_value",
                 "district", "lon", "lat", "area_m2"),
    "anchor": ("osm_type", "osm_id", "name", "category_key", "category_value",
               "district", "lon", "lat", "NULL::DOUBLE AS area_m2"),
}
LAYERS = tuple(_TABLE_COLUMNS)

TO_UTM = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)
GEOD = Geod(ellps="WGS84")

_DIST = "sqrt((x_utm - q.x) * (x_utm - q.x) + (y_utm - q.y) * (y_utm - q.y))"

_POI_COLUMNS = ("osm_type", "osm_id", "name", "category_key", "category_value",
                "district", "lon", "lat")

NEARBY_SQL = f"""
WITH q AS (SELECT $qx AS x, $qy AS y)
SELECT 'poi_point' AS layer, osm_id, name, category_key, category_value, district, lon, lat,
       {_DIST} AS dist_m, NULL::DOUBLE AS area_m2, ST_AsText(geom) AS wkt
FROM poi_point, q
WHERE {{D}}{{CAT}} AND {_DIST} <= $radius
UNION ALL
SELECT 'poi_area' AS layer, osm_id, name, category_key, category_value, district, lon, lat,
       {_DIST} AS dist_m, area_m2, ST_AsText(geom) AS wkt
FROM poi_area, q
WHERE {{D}}{{CAT}} AND {_DIST} <= $radius
ORDER BY dist_m
"""

# anchor 层的半径查询：只有调用方明确要求查站点类设施时才拼上这一段。
# 默认不查是刻意的——anchor 层是「定位锚点」（地名、站点、路口），把它混进设施统计会
# 让「附近有多少家便利店」这类问题悄悄多出几百个地名节点。要查就明说，并且分开计数。
_ANCHOR_UNION = f"""
UNION ALL
SELECT 'anchor' AS layer, osm_id, name, category_key, category_value, district, lon, lat,
       {_DIST} AS dist_m, NULL::DOUBLE AS area_m2, ST_AsText(geom) AS wkt
FROM anchor, q
WHERE {{D}}{{CAT}} AND {_DIST} <= $radius
"""

NEARBY_SQL_ANCHOR = NEARBY_SQL.replace("ORDER BY dist_m", _ANCHOR_UNION + "ORDER BY dist_m")

SUMMARY_SQL = """
SELECT district, category_key, category_value, count(*) AS n FROM (
  SELECT district, category_key, category_value FROM poi_point WHERE {D}{CAT}
  UNION ALL
  SELECT district, category_key, category_value FROM poi_area  WHERE {D}{CAT}
) GROUP BY 1, 2, 3 ORDER BY 1, 4 DESC
"""

_PLACE_POOL = "\n  UNION ALL\n".join(
    f"  SELECT '{table}' AS layer, {', '.join(cols)}\n  FROM {table} "
    f"WHERE {{D}}name IS NOT NULL AND name <> ''"
    for table, cols in _TABLE_COLUMNS.items()
)

PLACE_SQL = f"""
WITH q AS (SELECT $kw AS kw),
pool AS (
{_PLACE_POOL}
)
SELECT pool.*,
       CASE WHEN name = q.kw THEN 3 WHEN name LIKE q.kw || '%' THEN 2 ELSE 1 END AS match_score
FROM pool, q
WHERE name = q.kw OR name LIKE '%' || q.kw || '%'
ORDER BY match_score DESC,
         CASE pool.layer WHEN 'anchor' THEN 0 WHEN 'poi_area' THEN 1 ELSE 2 END,
         CASE pool.category_key WHEN 'place' THEN 0 WHEN 'railway' THEN 1 ELSE 2 END,
         length(name) ASC, name
LIMIT $limit
"""


def load_scope() -> dict:
    return json.loads(SCOPE_FILE.read_text(encoding="utf-8"))


def presets() -> dict[str, list[str]]:
    return load_scope()["category_matching"]["presets"]


@dataclass(frozen=True)
class CategorySpec:
    """一条类别口径：value 是 OSM 的 category_value，key 是标签键。

    key 为 None 表示不限标签键——value='university' 会同时命中 amenity=university
    与 office=university，因为数据里它们指同一种设施。
    """

    value: str
    key: str | None = None

    @property
    def label(self) -> str:
        return f"{self.key}={self.value}" if self.key else self.value

    def to_dict(self) -> dict:
        return {"key": self.key, "value": self.value, "label": self.label}


NO_CATEGORY_HINT = (
    "必须指定要查什么类别：传 categories（自由类别，如 ['高校']、['amenity=university']、"
    "['bank']），或传 preset（常用组合，见 compute://schema）。本工具不设默认类别——"
    "给默认值会出现「想查 A 却拿到 B」而无人察觉。可查的类别清单见 compute://categories。"
)

_CACHE: dict[str, object] = {}


def load_aliases() -> dict:
    """读取类别别名表（由 geo_knowledge 维护，geo_compute 只消费）。"""
    if "aliases" not in _CACHE:
        _CACHE["aliases"] = json.loads(ALIAS_FILE.read_text(encoding="utf-8"))
    return _CACHE["aliases"]  # type: ignore[return-value]


def category_inventory() -> pd.DataFrame:
    """全库可查的 (category_key, category_value) 与计数，进程内只查一次。"""
    if "inventory" not in _CACHE:
        con = connect()
        try:
            df = con.execute("""
                SELECT category_key, category_value,
                       count(*) FILTER (WHERE layer = 'poi_point') AS n_point,
                       count(*) FILTER (WHERE layer = 'poi_area')  AS n_area,
                       count(*) AS n
                FROM (SELECT 'poi_point' AS layer, category_key, category_value FROM poi_point
                      UNION ALL
                      SELECT 'poi_area' AS layer, category_key, category_value FROM poi_area)
                GROUP BY 1, 2 ORDER BY n DESC, category_key, category_value
            """).df()
        finally:
            con.close()
        _CACHE["inventory"] = df
    return _CACHE["inventory"]  # type: ignore[return-value]


def anchor_inventory() -> pd.DataFrame:
    """anchor 层的 (category_key, category_value) 与计数，进程内只查一次。

    单独一份是刻意的：compute://categories 与 summarize_poi 的口径都是 POI 两层，
    把 anchor 混进去会让「可查类别」看起来比实际宽——railway=station 这类值只在
    query_nearby(include_anchor=True) 里查得到。它的用处是让类别校验认识这些值，
    并把「这项只在 anchor 层」明确报给调用方，而不是静默返回一个 0。
    """
    if "anchor_inventory" not in _CACHE:
        con = connect()
        try:
            df = con.execute("""
                SELECT category_key, category_value, count(*) AS n
                FROM anchor GROUP BY 1, 2 ORDER BY n DESC, category_key, category_value
            """).df()
        finally:
            con.close()
        _CACHE["anchor_inventory"] = df
    return _CACHE["anchor_inventory"]  # type: ignore[return-value]


def _known_keys() -> list[str]:
    return sorted(set(category_inventory()["category_key"])
                  | set(anchor_inventory()["category_key"]))


def _values_of_key(key: str) -> list[str]:
    out: set[str] = set()
    for inv in (category_inventory(), anchor_inventory()):
        out |= set(inv.loc[inv["category_key"] == key, "category_value"])
    return sorted(out)


def _known_values() -> set[str]:
    return (set(category_inventory()["category_value"])
            | set(anchor_inventory()["category_value"]))


def missing_from_poi(specs: list[CategorySpec]) -> list[CategorySpec]:
    """筛出「POI 两层里一条都没有」的类别口径，典型是只在 anchor 层的 railway=station。

    这类口径在默认的半径查询里必然得到 0，而这个 0 不是「附近没有」，是「查错层了」。
    调用方据此提示 include_anchor=True——两个意思不能混成同一个 0。
    """
    inv = category_inventory()
    pairs = set(zip(inv["category_key"], inv["category_value"]))
    values = set(inv["category_value"])
    return [s for s in specs
            if not ((s.key, s.value) in pairs if s.key else s.value in values)]


def _parse_osm_tag(text: str) -> CategorySpec:
    """解析 'key=value' 形式的类别实参，并校验它在本库真实存在。"""
    key, _, value = text.partition("=")
    key, value = key.strip().lower(), value.strip().lower()
    if not key or not value:
        raise ValueError(f"类别 {text!r} 写法不对，应形如 amenity=university，或直接给 university")
    if key not in _known_keys():
        raise ValueError(f"本库没有标签键 {key!r}；可用标签键: {_known_keys()}")
    if value not in _values_of_key(key):
        near = difflib.get_close_matches(value, _values_of_key(key), n=5, cutoff=0.6)
        tail = (f"；{key} 下最接近的是 {near}" if near
                else f"；{key} 下的可查值见 compute://categories")
        raise ValueError(f"本库没有 {key}={value} 这类设施{tail}")
    return CategorySpec(value, key)


def expand_category(raw: str) -> list[CategorySpec]:
    """把一个类别说法展开成 CategorySpec 列表：先认别名（中文），再认裸 OSM 值。

    认不出来就报错并给近似建议。拼错必须显式失败——静默返回 0 条会把
    「没有这类设施」和「你查错类别了」混成同一个结论。
    """
    text = str(raw).strip()
    if not text:
        raise ValueError("类别不能是空字符串")
    if "=" in text:
        return [_parse_osm_tag(text)]
    aliases = load_aliases()["aliases"]
    entry = aliases.get(text) or aliases.get(text.lower())
    if entry is not None:
        specs = [_parse_osm_tag(item) for item in entry["categories"]]
        if not specs:
            raise ValueError(f"别名表里的 {text!r} 没有配任何类别，需修 aliases.json")
        return specs
    unavailable = load_aliases().get("not_available") or {}
    note = unavailable.get(text) or unavailable.get(text.lower())
    if note is not None:
        raise ValueError(f"{text} 不在可查询范围内：{note}")
    if text.lower() in _known_values():
        return [CategorySpec(text.lower())]
    near_alias = difflib.get_close_matches(text, list(aliases), n=5, cutoff=0.5)
    near_value = difflib.get_close_matches(text.lower(), sorted(_known_values()), n=5, cutoff=0.7)
    hints = [f"别名里接近的有 {near_alias}"] if near_alias else []
    if near_value:
        hints.append(f"OSM 值里接近的有 {near_value}")
    raise ValueError(f"不认识的类别 {text!r}。" + "；".join(hints) + "。"
                     "完整清单见 compute://categories")


def resolve_categories(categories: list[str] | None = None,
                       preset: str | None = None) -> list[CategorySpec]:
    """把调用方的类别说法统一解析成 CategorySpec 列表（去重、保序）。"""
    if categories and preset:
        raise ValueError("categories 与 preset 只能给一个，不能同时给")
    if preset:
        combos = presets().get(preset)
        if not combos:
            raise ValueError(f"未知口径预设 {preset!r}；可用预设: {sorted(presets())}")
        specs = [s for combo in combos for s in expand_category(combo)]
    elif categories:
        specs = [s for raw in categories for s in expand_category(raw)]
    else:
        raise ValueError(NO_CATEGORY_HINT)
    seen: set[tuple] = set()
    out: list[CategorySpec] = []
    for spec in specs:
        if (spec.key, spec.value) not in seen:
            seen.add((spec.key, spec.value))
            out.append(spec)
    return out


def category_predicate(specs: list[CategorySpec]) -> tuple[str, dict]:
    """把类别口径编译成 SQL 谓词与绑定参数。值一律走绑定参数，不做字符串拼接。"""
    params: dict[str, str] = {}
    any_key = sorted({s.value for s in specs if s.key is None})
    pairs = [s for s in specs if s.key is not None]
    clauses = []
    if any_key:
        names = []
        for i, value in enumerate(any_key):
            params[f"cv{i}"] = value
            names.append(f"$cv{i}")
        clauses.append(f"category_value IN ({', '.join(names)})")
    if pairs:
        sub = []
        for i, spec in enumerate(pairs):
            params[f"ck{i}"], params[f"cx{i}"] = spec.key, spec.value
            sub.append(f"(category_key = $ck{i} AND category_value = $cx{i})")
        clauses.append(f"({' OR '.join(sub)})")
    if not clauses:
        raise ValueError("没有任何类别条件，拒绝执行全表扫描")
    return f"({' OR '.join(clauses)})", params


def category_tally(df: pd.DataFrame, specs: list[CategorySpec],
                   weight: str | None = None) -> list[dict]:
    """每个请求类别的命中数（含 0 命中）。0 是真实结论，要能一眼看出是哪一个为 0。

    df 是明细时按行数计；df 是汇总结果时把 weight 指向计数列（summarize 用 'n'），
    否则数的是聚合后的行数而不是设施数。
    """
    if df.empty:
        counts, pairs = {}, {}
    elif weight:
        counts = df.groupby("category_value")[weight].sum().to_dict()
        pairs = df.groupby(["category_key", "category_value"])[weight].sum().to_dict()
    else:
        counts = df.groupby("category_value").size().to_dict()
        pairs = df.groupby(["category_key", "category_value"]).size().to_dict()
    out = []
    for spec in specs:
        n = int(pairs.get((spec.key, spec.value), 0) if spec.key else counts.get(spec.value, 0))
        out.append({**spec.to_dict(), "count": n, "matched": n > 0})
    return out


def connect(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(DB), read_only=read_only)
    con.execute("LOAD spatial")
    return con


def normalize_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """把 DuckDB 返回的 NaN 归一为 None。

    NULL 经 Arrow 转 pandas 会变成 float('nan')；nan 在 Python 里是真值且不是字符串，
    既污染文本列，也会让下游 Pydantic 的 str | None 校验直接失败。
    """
    for col in ("name", "district", "category_key", "category_value", "osm_id", "osm_type"):
        if col in df.columns:
            df[col] = df[col].astype(object).where(df[col].notna(), None)
    if "area_m2" in df.columns:
        df["area_m2"] = df["area_m2"].astype(object).where(df["area_m2"].notna(), None)
    return df


def plain(record: dict) -> dict:
    """把 numpy 标量转成原生 Python，便于直接进 JSON / pydantic。"""
    out = {}
    for key, value in record.items():
        if isinstance(value, np.integer):
            out[key] = int(value)
        elif isinstance(value, np.floating):
            out[key] = None if pd.isna(value) else float(value)
        elif isinstance(value, float) and pd.isna(value):
            out[key] = None
        else:
            out[key] = value
    return out


def nearby(lon: float, lat: float, radius_m: float = 1000.0,
           district: str | None = None,
           categories: list[CategorySpec] | None = None,
           include_anchor: bool = False):
    """返回 (结果 DataFrame, 实际执行 SQL, 绑定参数)。中心必须由调用方给定。

    categories 是 resolve_categories 的产物；本函数不认识 preset，也没有默认类别。
    include_anchor=True 时把 anchor 层（地名、地铁站、火车站、公交站等锚点）也纳入半径查询；
    默认 False，层与层的计数必须分开，不能混成一个数。
    """
    pred, cat_params = category_predicate(list(categories or []))
    con = connect()
    try:
        sql = (NEARBY_SQL_ANCHOR if include_anchor else NEARBY_SQL).replace("{CAT}", pred)
        params = {"radius": float(radius_m), **cat_params}
        if district:
            sql = sql.replace("{D}", "district = $district AND ")
            params["district"] = district
        else:
            sql = sql.replace("{D}", "")
        qx, qy = TO_UTM.transform(lon, lat)
        params["qx"], params["qy"] = qx, qy
        df = con.execute(sql, params).df()
    finally:
        con.close()
    df["dist_m"] = df["dist_m"].round(1)
    return normalize_nulls(df.sort_values("dist_m").reset_index(drop=True)), sql, params


def find_places(name: str, district: str | None = None, limit: int = 10):
    """按名称检索可作锚点的地名（精确 > 前缀 > 包含），返回 (DataFrame, SQL, 参数)。"""
    con = connect()
    try:
        sql = PLACE_SQL
        params: dict = {"kw": name, "limit": int(limit)}
        if district:
            sql = sql.replace("{D}", "district = $district AND ")
            params["district"] = district
        else:
            sql = sql.replace("{D}", "")
        df = con.execute(sql, params).df()
    finally:
        con.close()
    df = normalize_nulls(df)
    df["ref"] = df["layer"] + "/" + df["osm_id"].astype(str)
    return df, sql, params


def resolve_ref(ref: str) -> dict:
    """把 'poi_point/7591774161' 这样的引用解析成一条记录（含坐标）。"""
    layer, _, osm_id = ref.partition("/")
    if layer not in LAYERS or not osm_id:
        raise ValueError(
            f"引用格式应为 <layer>/<osm_id>，layer 取 {' 或 '.join(LAYERS)}；实际收到 {ref!r}"
        )
    con = connect()
    try:
        df = con.execute(
            f"SELECT {', '.join(_TABLE_COLUMNS[layer])} FROM {layer} WHERE osm_id = $i",
            {"i": osm_id},
        ).df()
    finally:
        con.close()
    if df.empty:
        raise ValueError(f"库中找不到 {ref}")
    record = plain(normalize_nulls(df).iloc[0].to_dict())
    record.update({"ref": ref, "layer": layer})
    return record


def distance_between(a: dict, b: dict) -> dict:
    """两点距离。同时给出 UTM 平面距离与椭球面大地线距离，两者互为印证。"""
    ax, ay = TO_UTM.transform(a["lon"], a["lat"])
    bx, by = TO_UTM.transform(b["lon"], b["lat"])
    planar = float(np.hypot(bx - ax, by - ay))
    azimuth, _, geodesic = GEOD.inv(a["lon"], a["lat"], b["lon"], b["lat"])
    gap_pct = (abs(planar - geodesic) / geodesic * 100.0) if geodesic > 0 else 0.0
    return {
        "planar_distance_m": round(planar, 2),
        "geodesic_distance_m": round(float(geodesic), 2),
        "planar_vs_geodesic_pct": round(gap_pct, 4),
        "bearing_deg": round(float(azimuth) % 360.0, 2),
    }


def find_duplicates(df: pd.DataFrame) -> list[dict]:
    """标记结果内相距 <= DUP_M 的点面配对：疑似同一设施被点面双挂。只提示不合并。

    只认 poi_area 作面层：调用了 include_anchor 时结果里还有 anchor 行，
    把它们当成「面」会凭空造出一堆不存在的点面双挂。
    """
    if df.empty:
        return []
    px, py = TO_UTM.transform(df["lon"].values, df["lat"].values)
    is_pt = (df["layer"] == "poi_point").values
    is_area = (df["layer"] == "poi_area").values
    pt_idx, ag_idx = np.where(is_pt)[0], np.where(is_area)[0]
    if len(pt_idx) == 0 or len(ag_idx) == 0:
        return []
    pg = shapely.points(px[pt_idx], py[pt_idx])
    nn = shapely.STRtree(pg).nearest(shapely.points(px[ag_idx], py[ag_idx]))
    dd = shapely.distance(pg[nn], shapely.points(px[ag_idx], py[ag_idx]))
    return [
        {"area_name": str(df["name"].iloc[i] or ""), "area_osm_id": str(df["osm_id"].iloc[i]),
         "point_name": str(df["name"].iloc[pt_idx[nn[k]]] or ""),
         "point_osm_id": str(df["osm_id"].iloc[pt_idx[nn[k]]]), "gap_m": round(float(dd[k]), 1)}
        for k, i in enumerate(ag_idx) if dd[k] <= DUP_M
    ]


def summarize(district: str | None = None,
              categories: list[CategorySpec] | None = None) -> dict:
    """按类别与行政区统计数量。categories 是 resolve_categories 的产物。"""
    specs = list(categories or [])
    pred, cat_params = category_predicate(specs)
    sql = SUMMARY_SQL.replace("{CAT}", pred)
    con = connect()
    try:
        if district:
            sql = sql.replace("{D}", "district = $district AND ")
            df = con.execute(sql, {"district": district, **cat_params}).df()
        else:
            sql = sql.replace("{D}", "district IS NOT NULL AND ")
            df = con.execute(sql, cat_params).df()
    finally:
        con.close()
    labels = {s.value: s.label for s in specs}
    labels.update({(s.key, s.value): s.label for s in specs if s.key})
    df = df.assign(label=[labels.get((k, v)) or labels.get(v) or v
                          for k, v in zip(df["category_key"], df["category_value"])])
    return {
        "district": district,
        "categories": [s.label for s in specs],
        "category_tally": category_tally(df, specs, weight="n"),
        "count_total": int(df["n"].sum()) if not df.empty else 0,
        "by_district": df.groupby("district")["n"].sum().sort_values(ascending=False).to_dict(),
        "by_category": df.groupby("label")["n"].sum().sort_values(ascending=False).to_dict(),
        "detail": df.to_dict("records"),
    }


def district_info(name: str | None = None) -> list[dict]:
    con = connect()
    try:
        sql = """
        SELECT adcode, name, source, area_km2,
               ST_XMin(geom) AS min_lon, ST_YMin(geom) AS min_lat,
               ST_XMax(geom) AS max_lon, ST_YMax(geom) AS max_lat
        FROM districts """
        if name:
            sql += "WHERE name = $name "
        sql += "ORDER BY adcode"
        return con.execute(sql, {"name": name} if name else {}).df().to_dict("records")
    finally:
        con.close()
