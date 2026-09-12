"""geo-compute 的查询核心。

供 MCP Tool 与命令行脚本共用，保证 SQL 与口径只有一处定义。
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "processed" / "geo.duckdb"
SCOPE_FILE = ROOT / "servers" / "geo_knowledge" / "poi_scope.json"
UTM = "EPSG:32650"
DUP_M = 150.0
TO_UTM = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)

_DIST = "sqrt((x_utm - q.x) * (x_utm - q.x) + (y_utm - q.y) * (y_utm - q.y))"

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


def district_point_on_surface(name: str, con=None) -> tuple[float, float]:
    own = con is None
    con = con or connect()
    try:
        row = con.execute(
            "SELECT ST_X(p) AS lon, ST_Y(p) AS lat FROM "
            "(SELECT ST_PointOnSurface(geom) AS p FROM districts WHERE name = $name)",
            {"name": name},
        ).df()
    finally:
        if own:
            con.close()
    if row.empty:
        raise ValueError(f"未找到区: {name}")
    return float(row["lon"].iloc[0]), float(row["lat"].iloc[0])


def nearby(lon: float, lat: float, radius_m: float = 1000.0,
           district: str | None = None, preset: str = "medical"):
    """返回 (结果 DataFrame, 实际执行 SQL, 绑定参数)。"""
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


def normalize_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """把 DuckDB 返回的 NaN 归一为 None。

    NULL 经 Arrow 转 pandas 会变成 float('nan')；nan 在 Python 里是真值且不是字符串，
    既污染文本列，也会让下游 Pydantic 的 str | None 校验直接失败。
    """
    for col in ("name", "district", "category_key", "category_value", "osm_id"):
        if col in df.columns:
            df[col] = df[col].astype(object).where(df[col].notna(), None)
    if "area_m2" in df.columns:
        df["area_m2"] = df["area_m2"].astype(object).where(df["area_m2"].notna(), None)
    return df


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
