"""构建北京五区的 WGS84 行政边界：四区取自 OSM，西城区由 DataV 纠偏补齐，并与街道 dissolve 交叉验证。"""

import json
import math
import warnings
import numpy as np
import geopandas as gpd
import pandas as pd
import pyogrio
import shapely
from pyproj import Geod
from shapely.geometry import shape
from shapely.ops import unary_union

from geo_knowledge.coords.gcj02 import gcj02_to_wgs84

warnings.filterwarnings("ignore")

PBF = r"data\raw\osmf-beijing-full.osm.pbf"
DATAV = r"data\raw\beijing_datav_full.json"
OUT_GPKG = r"data\processed\districts_wgs84.gpkg"
GEOD = Geod(ellps="WGS84")

DISTRICTS = {"东城区": "110101", "西城区": "110102", "朝阳区": "110105",
             "丰台区": "110106", "海淀区": "110108"}
FROM_OSM = ("东城区", "朝阳区", "丰台区", "海淀区")
XICHENG_STREETS = ["德胜街道", "什刹海街道", "新街口街道", "月坛街道", "金融街街道", "西长安街街道",
                   "展览路街道", "广安门内街道", "广安门外街道", "牛街街道", "白纸坊街道", "椿树街道",
                   "大栅栏街道", "天桥街道", "陶然亭街道"]


def to_wgs84(geom):
    return shapely.transform(geom, lambda c: np.column_stack(gcj02_to_wgs84(c[:, 0], c[:, 1])))


def area_m2(geom):
    a, _ = GEOD.geometry_area_perimeter(geom)
    return abs(a)


datav = {f["properties"]["name"]: shape(f["geometry"])
         for f in json.load(open(DATAV, encoding="utf-8"))["features"] if f.get("geometry")}

inv = pyogrio.read_dataframe(PBF, layer="multipolygons",
                             columns=["osm_id", "name", "admin_level", "boundary"],
                             read_geometry=False)
lvl6 = inv[(inv["admin_level"] == "6") & (inv["boundary"].notna())]
lvl8 = inv[(inv["admin_level"] == "8") & (inv["boundary"].notna())]


def osm_geom(name, table):
    oid = table[table["name"] == name]["osm_id"].iloc[0]
    return pyogrio.read_dataframe(PBF, sql=f"SELECT * FROM multipolygons WHERE osm_id = '{oid}'").geometry.iloc[0]


records, geoms = [], {}
for n in FROM_OSM:
    g = osm_geom(n, lvl6)
    geoms[n] = g
    records.append({"adcode": DISTRICTS[n], "name": n, "source": "OSM(直接)",
                    "area_km2": round(area_m2(g) / 1e6, 3), "note": ""})

print("=== A. 街道 dissolve 重建西城区（独立来源）===")
streets = [osm_geom(s, lvl8) for s in XICHENG_STREETS if s in set(lvl8["name"].astype(str))]
missing_streets = [s for s in XICHENG_STREETS if s not in set(lvl8["name"].astype(str))]
print(f"  命中街道 {len(streets)}/{len(XICHENG_STREETS)}，缺失: {missing_streets or '无'}")
xc_dissolve = unary_union(streets)
print(f"  dissolve 面积 {area_m2(xc_dissolve)/1e6:.2f} km²")

print("\n=== B. DataV + GCJ-02 纠偏 得到西城区 ===")
xc_fixed = to_wgs84(datav["西城区"])
print(f"  纠偏后面积 {area_m2(xc_fixed)/1e6:.2f} km²")

print("\n=== C. 交叉验证（A vs B）===")
inter = xc_fixed.intersection(xc_dissolve)
union = xc_fixed.union(xc_dissolve)
print(f"  IoU = {inter.area/union.area:.4f}")
print(f"  面积差 = {(area_m2(xc_fixed)-area_m2(xc_dissolve))/1e6:+.2f} km²")
diff = xc_fixed.difference(xc_dissolve)
print(f"  B 比 A 多出的面积 = {area_m2(diff)/1e6:.2f} km²")
if not diff.is_empty:
    c = diff.centroid
    print(f"  多出部分质心 = ({c.x:.5f}, {c.y:.5f})")
pts = pyogrio.read_dataframe(PBF, layer="points", columns=["osm_id", "name"])
bz = pts[pts["name"] == "白纸坊街道"]
if len(bz):
    p = bz.geometry.iloc[0]
    print(f"  白纸坊街道 POI 位于 ({p.x:.5f}, {p.y:.5f})；是否落在「多出部分」内: "
          f"{'是' if diff.contains(p) else '否'}")

geoms["西城区"] = xc_fixed
records.append({"adcode": DISTRICTS["西城区"], "name": "西城区", "source": "DataV+GCJ02纠偏",
                "area_km2": round(area_m2(xc_fixed) / 1e6, 3),
                "note": f"OSM 缺失该区边界(relation 568660 几何损坏); 街道dissolve交叉验证 IoU={inter.area/union.area:.3f}"})

gdf = gpd.GeoDataFrame(records, geometry=[geoms[r["name"]] for r in records], crs="EPSG:4326")
gdf = gdf.sort_values("adcode").reset_index(drop=True)
gdf.to_file(OUT_GPKG, layer="districts", driver="GPKG")
gdf.to_file(OUT_GPKG.replace(".gpkg", ".geojson"), driver="GeoJSON")
print(f"\n=== 输出 ===\n  已写入 {OUT_GPKG}")
print(gdf[["adcode", "name", "source", "area_km2"]].to_string(index=False))
print(f"  五区合计 {gdf['area_km2'].sum():.1f} km²")