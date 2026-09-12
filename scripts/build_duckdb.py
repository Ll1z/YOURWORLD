"""把边界与 POI 装载进 DuckDB，并建立空间索引。

坐标策略：几何以 WGS84(EPSG:4326) 为唯一真身入库；距离类计算所需的 UTM 50N(EPSG:32650)
坐标作为派生列 x_utm / y_utm 一并落库，避免查询时反复做投影变换。
"""

import os
import warnings
import geopandas as gpd
import duckdb
import numpy as np
import pandas as pd
import pyogrio
import shapely
from pyproj import Transformer

warnings.filterwarnings("ignore")

DB = r"data\processed\geo.duckdb"
T = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)

if os.path.exists(DB):
    os.remove(DB)
con = duckdb.connect(DB)
con.execute("INSTALL spatial")
con.execute("LOAD spatial")


def load(gpkg, layer):
    gdf = pyogrio.read_dataframe(gpkg, layer=layer, on_invalid="ignore")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].reset_index(drop=True)
    return gdf


print("=== districts ===")
d = load(r"data\processed\districts_wgs84.gpkg", "districts")
d["wkt"] = shapely.to_wkt(d.geometry.values, rounding_precision=8)
con.register("src", d[["adcode", "name", "source", "area_km2", "wkt"]])
con.execute("CREATE OR REPLACE TABLE districts AS SELECT adcode, name, source, area_km2, ST_GeomFromText(wkt) AS geom FROM src")
print(f"  {len(d)} 行")

print("=== poi_point ===")
p = load(r"data\processed\poi_point.gpkg", "poi_point")
p["lon"] = p.geometry.x
p["lat"] = p.geometry.y
p["x_utm"], p["y_utm"] = T.transform(p["lon"].values, p["lat"].values)
p["wkt"] = shapely.to_wkt(p.geometry.values, rounding_precision=8)
con.register("src", p[["osm_type", "osm_id", "name", "category_key", "category_value",
                       "district", "lon", "lat", "x_utm", "y_utm", "wkt"]])
con.execute("""
CREATE OR REPLACE TABLE poi_point AS
SELECT osm_type, osm_id, name, category_key, category_value, district,
       lon, lat, x_utm, y_utm, ST_GeomFromText(wkt) AS geom
FROM src
""")
print(f"  {len(p):,} 行")

print("=== poi_area ===")
a = load(r"data\processed\poi_area.gpkg", "poi_area")
a["lon"], a["lat"] = a["rep_x"], a["rep_y"]
a["x_utm"], a["y_utm"] = T.transform(a["lon"].values, a["lat"].values)
a["wkt"] = shapely.to_wkt(a.geometry.values, rounding_precision=8)
con.register("src", a[["osm_type", "osm_id", "name", "category_key", "category_value",
                       "district", "lon", "lat", "x_utm", "y_utm", "area_m2", "wkt"]])
con.execute("""
CREATE OR REPLACE TABLE poi_area AS
SELECT osm_type, osm_id, name, category_key, category_value, district,
       lon, lat, x_utm, y_utm, area_m2, ST_GeomFromText(wkt) AS geom
FROM src
""")
print(f"  {len(a):,} 行")

print("=== anchor ===")
n = load(r"data\processed\anchors.gpkg", "anchor")
n["wkt"] = shapely.to_wkt(n.geometry.values, rounding_precision=8)
con.register("src", n[["osm_type", "osm_id", "name", "category_key", "category_value",
                       "district", "lon", "lat", "wkt"]])
con.execute("""
CREATE OR REPLACE TABLE anchor AS
SELECT osm_type, osm_id, name, category_key, category_value, district,
       lon, lat, ST_GeomFromText(wkt) AS geom
FROM src
""")
print(f"  {len(n):,} 行")

print("\n=== 建立空间索引 ===")
for tbl, col in (("districts", "geom"), ("poi_point", "geom"), ("poi_area", "geom"), ("anchor", "geom")):
    try:
        con.execute(f"CREATE INDEX idx_{tbl}_rtree ON {tbl} USING RTREE ({col})")
        print(f"  {tbl}: R-tree 已建立")
    except Exception as e:
        print(f"  {tbl}: R-tree 失败 ({type(e).__name__}: {str(e)[:80]})")
for tbl in ("poi_point", "poi_area"):
    con.execute(f"CREATE INDEX idx_{tbl}_utm ON {tbl} (x_utm, y_utm)")
    print(f"  {tbl}: x_utm/y_utm 索引已建立")

print("\n=== 校验 ===")
print(con.execute("""
SELECT 'districts' AS t, count(*) AS n FROM districts
UNION ALL SELECT 'poi_point', count(*) FROM poi_point
UNION ALL SELECT 'poi_area', count(*) FROM poi_area
UNION ALL SELECT 'anchor', count(*) FROM anchor
""").df().to_string(index=False))
print("\n五区医疗类计数（点面并集）：")
print(con.execute("""
SELECT district, count(*) AS n FROM (
  SELECT district FROM poi_point WHERE district IS NOT NULL AND category_value IN ('hospital','clinic','doctors','pharmacy')
  UNION ALL
  SELECT district FROM poi_area  WHERE district IS NOT NULL AND category_value IN ('hospital','clinic','doctors','pharmacy')
) GROUP BY district ORDER BY n DESC
""").df().to_string(index=False))
con.close()
print(f"\n=== 输出 ===\n  {DB}")
