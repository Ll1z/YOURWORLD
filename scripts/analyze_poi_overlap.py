"""量化点层与面层的重复情况，为「1 公里内有哪些医院」这类问题确定计数口径。

使用 shapely.STRtree（R-tree）做邻近查询，与架构决策一致，不引入 scipy。
"""

import warnings
import geopandas as gpd
import numpy as np
import pyogrio
import shapely
from pyproj import Transformer

warnings.filterwarnings("ignore")
MED = ("hospital", "clinic", "doctors", "pharmacy")
HOSP = ("hospital", "clinic", "doctors")
NEAR_M = 150.0
T = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)

pt = pyogrio.read_dataframe(r"data\processed\poi_point.gpkg", layer="poi_point")
ar = pyogrio.read_dataframe(r"data\processed\poi_area.gpkg", layer="poi_area")

px, py = T.transform(pt.geometry.x.values, pt.geometry.y.values)
rx, ry = T.transform(ar["rep_x"].values, ar["rep_y"].values)
pt_utm = shapely.points(px, py)
ar_utm = shapely.points(rx, ry)

print("=== 医疗类 POI 分层计数 ===")
for label, gdf in (("点层", pt), ("面层", ar)):
    m = gdf[gdf["category_value"].isin(MED)]
    in5 = m[m["district"].notna()]
    print(f"  {label}: 全域 {len(m):,}  五区内 {len(in5):,}  {in5['category_value'].value_counts().to_dict()}")

print(f"\n=== 点面重复量化（五区内医疗类，阈值 {NEAR_M:.0f} m）===")
P = ((pt["category_value"].isin(MED)) & pt["district"].notna()).values
A = ((ar["category_value"].isin(MED)) & ar["district"].notna()).values
p_geom, p_meta = pt_utm[P], pt.loc[P].reset_index(drop=True)
a_geom, a_meta = ar_utm[A], ar.loc[A].reset_index(drop=True)
print(f"  比对规模：点 {len(a_geom)} 个面 对 {len(p_geom)} 个点")

tree = shapely.STRtree(p_geom)
idx = tree.nearest(a_geom)
d = shapely.distance(p_geom[idx], a_geom)
hit = d <= NEAR_M
print(f"  面中 {NEAR_M:.0f} m 内有同区医疗点的比例: {int(hit.sum())}/{len(a_geom)} = {hit.mean()*100:.1f}%")
print(f"  距离中位数 {np.median(d[hit]):.0f} m，最大 {d[hit].max():.0f} m")

same = 0
for i in np.where(hit)[0]:
    an = str(a_meta["name"].iloc[i] or "").strip()
    pn = str(p_meta["name"].iloc[idx[i]] or "").strip()
    if an and pn and (an == pn or an in pn or pn in an):
        same += 1
print(f"  其中名称相符 {same}，不符 {int(hit.sum()) - same}")

print("\n=== 口径对结果的影响（东城区医院类：hospital/clinic/doctors）===")
def count_hosp(meta, mask, valid=None):
    m = mask.copy()
    if valid is not None:
        m = m & valid
    return int(((meta["district"] == "东城区").values & m).sum())

n_pt = count_hosp(p_meta, p_meta["category_value"].isin(HOSP).values)
n_ar = count_hosp(a_meta, a_meta["category_value"].isin(HOSP).values)
a_hosp = a_meta["category_value"].isin(HOSP).values
n_naive = n_pt + count_hosp(a_meta, a_hosp)
n_dedup = n_pt + count_hosp(a_meta, a_hosp, ~hit)
print(f"  仅点层           : {n_pt}")
print(f"  仅面层           : {n_ar}")
print(f"  点+面朴素合并    : {n_naive}")
print(f"  点+面去重(150 m) : {n_dedup}")
print(f"  重复占比         : {(n_naive - n_dedup) / n_naive * 100:.0f}%")

print("\n=== 全域五区内医疗类，三种口径对比 ===")
for label, m in (("点+面朴素合并", None), ("点+面去重", True)):
    pass
tot_pt = int((pt["category_value"].isin(MED) & pt["district"].notna()).sum())
tot_ar = int((ar["category_value"].isin(MED) & ar["district"].notna()).sum())
tot_dup = int(((ar["category_value"].isin(MED)) & ar["district"].notna()).values[ (ar["district"].notna()).values ].sum()) if False else None
print(f"  仅点层 {tot_pt} / 仅面层 {tot_ar} / 朴素合并 {tot_pt + tot_ar} / 去重 {tot_pt + int((~hit).sum())}")