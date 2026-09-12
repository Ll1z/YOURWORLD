"""geo_compute 的查询核心。

供 MCP Tool 与命令行脚本共用，保证 SQL 与口径只有一处定义。

中心点口径（重要）：
    半径查询的中心必须是一个明确、可命名的位置（坐标或地名锚点），
    和路径规划要先选起点是一个道理。禁止用行政区的几何代表点当圆心——
    point_on_surface 只是几何产物，可能落在无人区，会让「某区 X 米内有什么」
    这类问题产生看似合理实则无意义的 0 结果。
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import shapely
from pyproj import Geod, Transformer

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "processed" / "geo.duckdb"
SCOPE_FILE = ROOT / "servers" / "geo_knowledge" / "poi_scope.json"
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
WHERE {{D}}category_value IN ({{CAT}}) AND {_DIST} <= $radius
UNION ALL
SELECT 'poi_area' AS layer, osm_id, name, category_key, category_value, district, lon, lat,
       {_DIST} AS dist_m, area_m2, ST_AsText(geom) AS wkt
FROM poi_area, q
WHERE {{D}}category_value IN ({{CAT}}) AND {_DIST} <= $radius
ORDER BY dist_m
"""

SUMMARY_SQL = """
SELECT district, category_value, count(*) AS n FROM (
  SELECT district, category_value FROM poi_point WHERE {D}category_value IN ({CAT})
  UNION ALL
  SELECT district, category_value FROM poi_area  WHERE {D}category_value IN ({CAT})
) GROUP BY 1, 2 ORDER BY 1, 3 DESC
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


def resolve_categories(preset: str) -> list[str]:
    cats = presets().get(preset)
    if not cats:
        raise ValueError(f"未知口径预设 {preset}；可用: {sorted(presets())}")
    return cats


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
           district: str | None = None, preset: str = "medical"):
    """返回 (结果 DataFrame, 实际执行 SQL, 绑定参数)。中心必须由调用方给定。"""
    cats = resolve_categories(preset)
    con = connect()
    try:
        sql = NEARBY_SQL.replace("{CAT}", ",".join(f"'{c}'" for c in cats))
        params = {"radius": float(radius_m)}
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
    """标记结果内相距 <= DUP_M 的点面配对：疑似同一设施被点面双挂。只提示不合并。"""
    if df.empty or df["layer"].nunique() < 2:
        return []
    px, py = TO_UTM.transform(df["lon"].values, df["lat"].values)
    is_pt = (df["layer"] == "poi_point").values
    pt_idx, ag_idx = np.where(is_pt)[0], np.where(~is_pt)[0]
    pg = shapely.points(px[pt_idx], py[pt_idx])
    nn = shapely.STRtree(pg).nearest(shapely.points(px[ag_idx], py[ag_idx]))
    dd = shapely.distance(pg[nn], shapely.points(px[ag_idx], py[ag_idx]))
    return [
        {"area_name": str(df["name"].iloc[i] or ""), "area_osm_id": str(df["osm_id"].iloc[i]),
         "point_name": str(df["name"].iloc[pt_idx[nn[k]]] or ""),
         "point_osm_id": str(df["osm_id"].iloc[pt_idx[nn[k]]]), "gap_m": round(float(dd[k]), 1)}
        for k, i in enumerate(ag_idx) if dd[k] <= DUP_M
    ]


def summarize(district: str | None = None, preset: str = "medical") -> dict:
    cats = resolve_categories(preset)
    sql = SUMMARY_SQL.replace("{CAT}", ",".join(f"'{c}'" for c in cats))
    con = connect()
    try:
        if district:
            sql = sql.replace("{D}", "district = $district AND ")
            df = con.execute(sql, {"district": district}).df()
        else:
            sql = sql.replace("{D}", "district IS NOT NULL AND ")
            df = con.execute(sql).df()
    finally:
        con.close()
    return {
        "district": district, "preset": preset, "categories": cats,
        "by_district": df.groupby("district")["n"].sum().sort_values(ascending=False).to_dict(),
        "by_category": df.groupby("category_value")["n"].sum().sort_values(ascending=False).to_dict(),
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
