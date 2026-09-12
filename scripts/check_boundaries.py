"""检查 OSM 提取的区级行政边界完整性，并实测与 DataV 边界的偏移。"""

import json
import math
import sys
import warnings
import pyogrio
import pandas as pd
from shapely.geometry import shape

warnings.filterwarnings("ignore")
DISTRICTS = ("东城区", "西城区", "海淀区", "朝阳区", "丰台区")
PBF = sys.argv[1] if len(sys.argv) > 1 else r"data\raw\osmf-beijing-full.osm.pbf"
print(f"### {PBF}\n")

pts = pyogrio.read_dataframe(PBF, layer="points")
bp = pts.total_bounds
print(f"points: {len(pts):,} 个, 范围=({bp[0]:.4f}, {bp[1]:.4f}, {bp[2]:.4f}, {bp[3]:.4f})")
del pts

inv = pyogrio.read_dataframe(PBF, layer="multipolygons",
                             columns=["osm_id", "name", "admin_level", "boundary"],
                             read_geometry=False)
bnd = inv[inv["boundary"].notna()]
print(f"boundary 要素 {len(bnd)} 个, admin_level 分布: {bnd['admin_level'].value_counts().sort_index().to_dict()}")

lvl6 = bnd[bnd["admin_level"] == "6"][["osm_id", "name"]].sort_values("name")
print(f"\n=== 全部 admin_level=6 边界 ({len(lvl6)}) ===")
print(lvl6.to_string(index=False))

print("\n=== 含 '西城' 字样的所有 boundary 要素 ===")
sub = bnd[bnd["name"].astype(str).str.contains("西城", na=False)]
print(sub[["osm_id", "name", "admin_level"]].to_string(index=False) if len(sub) else "  无")

print("\n=== 取五区几何（OGR SQL）===")
geoms = {}
for n in DISTRICTS:
    row = lvl6[lvl6["name"] == n]
    if not len(row):
        print(f"  [缺] {n}")
        continue
    oid = row["osm_id"].iloc[0]
    try:
        g = pyogrio.read_dataframe(
            PBF, sql=f"SELECT * FROM multipolygons WHERE osm_id = '{oid}'").geometry.iloc[0]
        if g is None or g.is_empty:
            print(f"  [几何为空] {n} (osm_id={oid})")
        else:
            geoms[n] = g
            print(f"  [有] {n}: bbox=({g.bounds[0]:.4f},{g.bounds[1]:.4f},{g.bounds[2]:.4f},{g.bounds[3]:.4f}) 面积={g.area:.5f}")
    except Exception as e:
        print(f"  [读取失败] {n}: {type(e).__name__}: {str(e)[:120]}")

d = json.load(open(r"data\raw\beijing_datav_full.json", encoding="utf-8"))
datav = {f["properties"]["name"]: shape(f["geometry"]) for f in d["features"] if f.get("geometry")}

print("\n=== 偏移实测 OSM vs DataV ===")
for n in DISTRICTS:
    if n not in geoms or n not in datav:
        continue
    go, gd = geoms[n], datav[n]
    co, cd = go.centroid, gd.centroid
    mx = (cd.x - co.x) * 111320 * math.cos(math.radians(co.y))
    my = (cd.y - co.y) * 110540
    print(f"  {n}: 偏移 东{mx:+.0f}m 北{my:+.0f}m 直线{math.hypot(mx,my):.0f}m | 面积比 {gd.area/go.area:.3f} | 重合率 {go.intersection(gd).area/go.area:.3f}")

print("\n=== OSM 有而 DataV 无 / 反之 ===")
print("  OSM 命中的区:", sorted(geoms))
all6 = set(lvl6["name"].astype(str))
print(f"  OSM 出现的区级名称({len(all6)}): {sorted(all6)}")