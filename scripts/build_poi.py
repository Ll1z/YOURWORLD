"""从 OSM 切片抽取 POI，分层输出点层与面层，并标注所属区。

设计要点（对应方案 C）：
- 点层与面层分开保存，口径由查询层决定，不在数据层做隐式合并
- 每个 POI 一行，主分类按 PRIORITY 顺序取第一个命中的键，原始标签完整保留在 other_tags
- 面层额外给出代表点（rep_x/rep_y）与大地线面积（area_m2），避免查询时重复计算
"""

import re
import warnings
import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely
from pyproj import Geod

warnings.filterwarnings("ignore")

PBF = r"data\raw\osmf-beijing-full.osm.pbf"
DISTRICTS = r"data\processed\districts_wgs84.gpkg"
OUT_POINT = r"data\processed\poi_point.gpkg"
OUT_AREA = r"data\processed\poi_area.gpkg"

PRIORITY = ("amenity", "healthcare", "shop", "tourism", "office", "leisure", "craft", "historic", "sport")
HSTORE = re.compile(r'"((?:[^"\\]|\\.)*)"\s*=>\s*(?:"((?:[^"\\]|\\.)*)"|([^,]*))')
GEOD = Geod(ellps="WGS84")


def parse_hstore(s):
    if not isinstance(s, str) or not s:
        return {}
    out = {}
    for k, vq, vu in HSTORE.findall(s):
        v = vq if vq != "" else vu.strip()
        out[k.replace('\\"', '"').replace("\\\\", "\\")] = v.replace('\\"', '"').replace("\\\\", "\\")
    return out


def pick_category(tags):
    for key in PRIORITY:
        v = tags.get(key)
        if v:
            return key, v
    return None, None


def build(tags_series, extra_df, osm_type):
    cats = tags_series.map(pick_category)
    keys = np.array([c[0] for c in cats], dtype=object)
    vals = np.array([c[1] for c in cats], dtype=object)
    keep = pd.notna(keys)
    out = pd.DataFrame({
        "osm_type": osm_type,
        "osm_id": extra_df["osm_id"].astype(str),
        "name": extra_df["name"],
        "category_key": keys,
        "category_value": vals,
        "other_tags": tags_series,
    })
    return out, keep


print("=== 点层 ===")
pts = pyogrio.read_dataframe(PBF, layer="points", on_invalid="ignore")
pt_tags = pts["other_tags"].map(parse_hstore)
for col in ("place", "man_made", "highway", "barrier"):
    if col in pts.columns:
        for i, v in pts[col].items():
            if isinstance(v, str) and v:
                pt_tags.at[i].setdefault(col, v)
pt_df, pt_keep = build(pt_tags, pts, "node")
print(f"  points 总数 {len(pts):,}，其中含 POI 标签 {pt_keep.sum():,}")
pt_gdf = gpd.GeoDataFrame(pt_df[pt_keep].reset_index(drop=True),
                          geometry=pts.geometry[pt_keep].reset_index(drop=True), crs="EPSG:4326")

del pts, pt_tags, pt_df

print("\n=== 面层 ===")
mp = pyogrio.read_dataframe(PBF, layer="multipolygons", on_invalid="ignore")
mp_tags = mp["other_tags"].map(parse_hstore)
for col in ("amenity", "shop", "tourism", "office", "leisure", "craft", "historic", "sport", "man_made", "place"):
    if col in mp.columns:
        for i, v in mp[col].items():
            if isinstance(v, str) and v:
                mp_tags.at[i].setdefault(col, v)
mp_df, mp_keep = build(mp_tags, mp, mp["type"].fillna("way").astype(str))
print(f"  multipolygons 总数 {len(mp):,}，其中含 POI 标签 {mp_keep.sum():,}")
mp_gdf = gpd.GeoDataFrame(mp_df[mp_keep].reset_index(drop=True),
                          geometry=mp.geometry[mp_keep].reset_index(drop=True), crs="EPSG:4326")
mp_gdf = mp_gdf[mp_gdf.geometry.notna() & ~mp_gdf.geometry.is_empty].reset_index(drop=True)
print(f"  去掉空几何后 {len(mp_gdf):,}")
invalid = ~shapely.is_valid(mp_gdf.geometry.values)
if invalid.any():
    print(f"  几何无效 {int(invalid.sum())} 个，已剔除。示例 osm_id: "
          f"{mp_gdf.loc[invalid, 'osm_id'].head(5).tolist()}")
    mp_gdf = mp_gdf[~invalid].reset_index(drop=True)

del mp, mp_tags, mp_df

print("\n=== 标注所属区 ===")
dist = pyogrio.read_dataframe(DISTRICTS, layer="districts")[["adcode", "name", "geometry"]]
dist = dist.rename(columns={"name": "district"})
for gdf in (pt_gdf, mp_gdf):
    j = gpd.sjoin(gdf, dist, how="left", predicate="intersects")
    j = j[~j.index.duplicated(keep="first")]
    gdf["district"] = j["district"].reindex(gdf.index)
    if gdf is mp_gdf:
        rp = shapely.point_on_surface(mp_gdf.geometry.values)
        gdf["rep_x"] = shapely.get_x(rp)
        gdf["rep_y"] = shapely.get_y(rp)
        areas = [abs(GEOD.geometry_area_perimeter(g)[0]) for g in mp_gdf.geometry]
        gdf["area_m2"] = np.round(areas, 1)

pt_gdf.to_file(OUT_POINT, layer="poi_point", driver="GPKG")
mp_gdf.to_file(OUT_AREA, layer="poi_area", driver="GPKG")

print("\n=== 分布统计 ===")
for label, gdf in (("点层", pt_gdf), ("面层", mp_gdf)):
    print(f"\n{label}: {len(gdf):,}")
    print("  按区:")
    print(gdf["district"].fillna("(五区外)").value_counts().to_string())
    print("  按主分类 top8:")
    print(gdf["category_key"].value_counts().head(8).to_string())

print("\n=== 与 Stage 1 验收相关的 POI ===")
for label, gdf in (("点层", pt_gdf), ("面层", mp_gdf)):
    med = gdf[(gdf["category_key"].isin(["amenity", "healthcare"])) &
              (gdf["category_value"].isin(["hospital", "clinic", "doctors", "pharmacy"]))]
    conv = gdf[gdf["category_value"].isin(["convenience", "supermarket"])]
    print(f"  {label}: 医疗类 {len(med)}，便利店/超市类 {len(conv)}")
    if len(med):
        print("    医疗类分布:", med["category_value"].value_counts().to_dict())

print(f"\n=== 输出 ===\n  {OUT_POINT}\n  {OUT_AREA}")