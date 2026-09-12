"""用有 OSM 真值的四个区校准 GCJ-02 纠偏算法。"""

import json
import math
import warnings
import numpy as np
import pyogrio
import shapely
from shapely.geometry import shape

from geo_knowledge.coords.gcj02 import gcj02_to_wgs84, wgs84_to_gcj02

warnings.filterwarnings("ignore")

PBF = r"data\raw\osmf-beijing-full.osm.pbf"
DATAV = r"data\raw\beijing_datav_full.json"
CALIB = ("东城区", "海淀区", "朝阳区", "丰台区")

print("=== 0. 算法自检：gcj2wgs(wgs2gcj(x)) 是否还原 ===")
probe = (116.3975, 39.9087)
g = tuple(float(v) for v in wgs84_to_gcj02(*probe))
w = tuple(float(v) for v in gcj02_to_wgs84(*g))
err_m = math.hypot((w[0] - probe[0]) * 111320 * math.cos(math.radians(probe[1])),
                   (w[1] - probe[1]) * 110540)
print(f"  输入 WGS84 {probe} -> GCJ {tuple(round(v, 8) for v in g)} -> 还原 {tuple(round(v, 8) for v in w)}")
print(f"  往返误差: {err_m * 100:.4f} cm")

d = json.load(open(DATAV, encoding="utf-8"))
datav = {f["properties"]["name"]: shape(f["geometry"]) for f in d["features"] if f.get("geometry")}
inv = pyogrio.read_dataframe(PBF, layer="multipolygons",
                             columns=["osm_id", "name", "admin_level", "boundary"],
                             read_geometry=False)
lvl6 = inv[(inv["admin_level"] == "6") & (inv["boundary"].notna())]

osm = {}
for n in CALIB:
    oid = lvl6[lvl6["name"] == n]["osm_id"].iloc[0]
    osm[n] = pyogrio.read_dataframe(PBF, sql=f"SELECT * FROM multipolygons WHERE osm_id = '{oid}'").geometry.iloc[0]


def to_wgs84(geom):
    return shapely.transform(geom, lambda c: np.column_stack(gcj02_to_wgs84(c[:, 0], c[:, 1])))


def metrics(a, b):
    inter = a.intersection(b).area
    union = a.union(b).area
    ca, cb = a.centroid, b.centroid
    mx = (ca.x - cb.x) * 111320 * math.cos(math.radians(cb.y))
    my = (ca.y - cb.y) * 110540
    return inter / union, math.hypot(mx, my)


print("\n=== 1. 纠偏前后对比（DataV 转换后 vs OSM 真值）===")
print(f"{'区':<8}{'IoU前':>8}{'IoU后':>8}{'质心残差前':>12}{'质心残差后':>12}{'顶点残差均值':>14}{'顶点残差P95':>12}")
rows = []
for n in CALIB:
    dv, os_ = datav[n], osm[n]
    iou_raw, off_raw = metrics(dv, os_)
    dv_fix = to_wgs84(dv)
    iou_fix, off_fix = metrics(dv_fix, os_)
    bnd = os_.boundary
    coords = shapely.get_coordinates(dv_fix)
    dists = shapely.distance(shapely.points(coords), bnd)
    d_mean = float(np.mean(dists)) * 111320 * math.cos(math.radians(39.9))
    d_p95 = float(np.percentile(dists, 95)) * 111320 * math.cos(math.radians(39.9))
    rows.append((n, iou_raw, iou_fix, off_raw, off_fix, d_mean, d_p95))
    print(f"{n:<8}{iou_raw:>8.3f}{iou_fix:>8.3f}{off_raw:>11.0f}m{off_fix:>11.1f}m{d_mean:>13.1f}m{d_p95:>11.1f}m")

print("\n=== 2. 结论判定 ===")
avg_iou = np.mean([r[2] for r in rows])
max_off = max(r[4] for r in rows)
max_p95 = max(r[6] for r in rows)
print(f"  纠偏后平均 IoU = {avg_iou:.4f} (1.0 为完全重合)")
print(f"  纠偏后最大质心残差 = {max_off:.1f} m")
print(f"  纠偏后最大 P95 顶点残差 = {max_p95:.1f} m")
print("  判定:", "通过，可用于西城区" if avg_iou > 0.95 and max_off < 60 else "未通过，需复查")