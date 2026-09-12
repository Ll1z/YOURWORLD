"""从 OSM 切片抽取 POI 与锚点，分层输出并标注所属区。

设计要点（对应方案 C）：
- 点层与面层分开保存，口径由查询层决定，不在数据层做隐式合并
- 每个 POI 一行，主分类按 PRIORITY 顺序取第一个命中的键，原始标签完整保留在 other_tags
- 面层额外给出代表点（rep_x/rep_y）与大地线面积（area_m2），避免查询时重复计算
- 锚点层单独输出：POI 层回答「这里有什么设施」，锚点层回答「这个地方叫什么、在哪」

身份口径（重要）：
- GDAL 的 OSM multipolygons 层把要素 id 拆成两个互斥字段：来自 relation 的填 osm_id，
  来自闭合 way 的填 osm_way_id。只读 osm_id 会丢掉绝大多数面要素的身份，必须 coalesce。
- 主键是 (osm_type, osm_id) 组合：way 与 relation 的编号空间独立，同号不代表同一要素。

锚点口径：
- 锚点是「人用来定位的参照物」，取自 place / railway / highway / public_transport 四类标签，
  且必须带名称。place=suburb 这类节点此前会被 POI 层的 PRIORITY 过滤掉，
  但「中关村」「国贸」恰恰是用户最常用的定位参照物。
- 只取点层。线状站台、面状地名不在此列。
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
OUT_ANCHOR = r"data\processed\anchors.gpkg"

PRIORITY = ("amenity", "healthcare", "shop", "tourism", "office", "leisure", "craft", "historic", "sport")
HSTORE = re.compile(r'"((?:[^"\\]|\\.)*)"\s*=>\s*(?:"((?:[^"\\]|\\.)*)"|([^,]*))')
GEOD = Geod(ellps="WGS84")

# 锚点标签白名单：按此优先级每个要素只取一个，避免同一站点被算两次
ANCHOR_KINDS = (
    ("place", frozenset({"city", "town", "suburb", "quarter", "neighbourhood",
                         "village", "hamlet", "locality"})),
    ("railway", frozenset({"station", "halt", "tram_stop", "subway_entrance"})),
    ("highway", frozenset({"bus_stop", "motorway_junction", "elevator"})),
    ("public_transport", frozenset({"station"})),
)


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


def polygon_identity(mp):
    """把 multipolygons 层的 osm_id / osm_way_id 合并成 (要素类型, 要素 id) 两个数组。"""
    is_relation = mp["osm_id"].notna()
    ids = mp["osm_id"].where(is_relation, mp["osm_way_id"]).astype("int64").astype(str)
    etype = np.where(is_relation.to_numpy(), "relation", "way").astype(object)
    return etype, ids.to_numpy()


def build(tags_series, extra_df, osm_type, osm_id=None):
    """osm_type 为逐行数组；osm_id 省略时取 extra_df["osm_id"]。"""
    cats = tags_series.map(pick_category)
    keys = np.array([c[0] for c in cats], dtype=object)
    vals = np.array([c[1] for c in cats], dtype=object)
    keep = pd.notna(keys)
    ids = extra_df["osm_id"] if osm_id is None else osm_id
    out = pd.DataFrame({
        "osm_type": osm_type,
        "osm_id": ids,
        "name": extra_df["name"],
        "category_key": keys,
        "category_value": vals,
        "other_tags": tags_series,
    })
    return out, keep


def build_anchors(tags_series, pts):
    """从点层抽取带名称的定位锚点。"""
    names = pts["name"].to_numpy()
    osm_ids = pts["osm_id"].astype(str).to_numpy()
    geometry = pts.geometry.values
    rows = []
    for i, tags in enumerate(tags_series):
        name = names[i]
        if not isinstance(name, str) or not name:
            continue
        for key, allowed in ANCHOR_KINDS:
            value = tags.get(key)
            if value and value in allowed:
                rows.append(("node", osm_ids[i], name, key, value,
                             float(geometry[i].x), float(geometry[i].y)))
                break
    return pd.DataFrame(rows, columns=["osm_type", "osm_id", "name", "category_key",
                                       "category_value", "lon", "lat"])


print("=== 点层 ===")
pts = pyogrio.read_dataframe(PBF, layer="points", on_invalid="ignore")
pt_tags = pts["other_tags"].map(parse_hstore)
for col in ("place", "man_made", "highway", "barrier"):
    if col in pts.columns:
        for i, v in pts[col].items():
            if isinstance(v, str) and v:
                pt_tags.at[i].setdefault(col, v)
pt_df, pt_keep = build(pt_tags, pts, np.full(len(pts), "node", dtype=object))
print(f"  points 总数 {len(pts):,}，其中含 POI 标签 {pt_keep.sum():,}")
pt_gdf = gpd.GeoDataFrame(pt_df[pt_keep].reset_index(drop=True),
                          geometry=pts.geometry[pt_keep].reset_index(drop=True), crs="EPSG:4326")

print("\n=== 锚点层 ===")
anchor_df = build_anchors(pt_tags, pts)
anchor_gdf = gpd.GeoDataFrame(
    anchor_df,
    geometry=gpd.points_from_xy(anchor_df["lon"], anchor_df["lat"]) if len(anchor_df) else [],
    crs="EPSG:4326",
)
print(f"  命名锚点 {len(anchor_gdf):,} 条")
if len(anchor_gdf):
    print(anchor_gdf["category_value"].value_counts().head(10).to_string())

del pts, pt_tags, pt_df

print("\n=== 面层 ===")
mp = pyogrio.read_dataframe(PBF, layer="multipolygons", on_invalid="ignore")
mp_tags = mp["other_tags"].map(parse_hstore)
for col in ("amenity", "shop", "tourism", "office", "leisure", "craft", "historic", "sport", "man_made", "place"):
    if col in mp.columns:
        for i, v in mp[col].items():
            if isinstance(v, str) and v:
                mp_tags.at[i].setdefault(col, v)
mp_etype, mp_id = polygon_identity(mp)
mp_df, mp_keep = build(mp_tags, mp, mp_etype, osm_id=mp_id)
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
for gdf in (pt_gdf, mp_gdf, anchor_gdf):
    j = gpd.sjoin(gdf, dist, how="left", predicate="intersects")
    j = j[~j.index.duplicated(keep="first")]
    gdf["district"] = j["district"].reindex(gdf.index)

rp = shapely.point_on_surface(mp_gdf.geometry.values)
mp_gdf["rep_x"] = shapely.get_x(rp)
mp_gdf["rep_y"] = shapely.get_y(rp)
mp_gdf["area_m2"] = np.round([abs(GEOD.geometry_area_perimeter(g)[0]) for g in mp_gdf.geometry], 1)

pt_gdf.to_file(OUT_POINT, layer="poi_point", driver="GPKG")
mp_gdf.to_file(OUT_AREA, layer="poi_area", driver="GPKG")
anchor_gdf.to_file(OUT_ANCHOR, layer="anchor", driver="GPKG")

print("\n=== 分布统计 ===")
for label, gdf in (("点层", pt_gdf), ("面层", mp_gdf), ("锚点层", anchor_gdf)):
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

print("\n=== 锚点层常见参照物 ===")
if len(anchor_gdf):
    for value in ("suburb", "quarter", "neighbourhood", "station", "subway_entrance", "bus_stop"):
        sub = anchor_gdf[anchor_gdf["category_value"] == value]
        sample = "、".join(str(n) for n in sub["name"].head(4))
        print(f"  {value}: {len(sub)} 条，示例 {sample or '（无）'}")

print(f"\n=== 输出 ===\n  {OUT_POINT}\n  {OUT_AREA}\n  {OUT_ANCHOR}")