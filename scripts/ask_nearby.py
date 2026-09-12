"""端到端查询：某点周围指定半径内有哪些 POI。

产出物固定包含数据来源、CRS 声明、计算方法、可复现 SQL 与结果文件，
对应 HANDOFF.md 的 Stage 1 验收标准形态。
"""

import argparse
import json
import os
from datetime import datetime, timezone

import duckdb
import geopandas as gpd
import numpy as np
import shapely
from pyproj import Transformer

DB = r"data\processed\geo.duckdb"
SCOPE = r"servers\geo_knowledge\poi_scope.json"
T = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)
DUP_M = 150.0

DIST = "sqrt((x_utm - q.x) * (x_utm - q.x) + (y_utm - q.y) * (y_utm - q.y))"

SQL = f"""
WITH q AS (SELECT $qx AS x, $qy AS y)
SELECT 'poi_point' AS layer, osm_id, name, category_key, category_value, district, lon, lat,
       {DIST} AS dist_m, NULL::DOUBLE AS area_m2, ST_AsText(geom) AS wkt
FROM poi_point, q
WHERE {{D}}category_value IN ({{CAT}}) AND {DIST} <= $radius
UNION ALL
SELECT 'poi_area' AS layer, osm_id, name, category_key, category_value, district, lon, lat,
       {DIST} AS dist_m, area_m2, ST_AsText(geom) AS wkt
FROM poi_area, q
WHERE {{D}}category_value IN ({{CAT}}) AND {DIST} <= $radius
ORDER BY dist_m
"""


def find_duplicates(df):
    """标记相距很近的点层/面层记录：可能描述同一设施，按口径不合并，仅提示。"""
    if df.empty or df["layer"].nunique() < 2:
        return []
    px, py = T.transform(df["lon"].values, df["lat"].values)
    is_pt = (df["layer"] == "poi_point").values
    pt_idx, ag_idx = np.where(is_pt)[0], np.where(~is_pt)[0]
    pg = shapely.points(px[pt_idx], py[pt_idx])
    ag = shapely.points(px[ag_idx], py[ag_idx])
    nn = shapely.STRtree(pg).nearest(ag)
    dd = shapely.distance(pg[nn], ag)
    out = []
    for k, i in enumerate(ag_idx):
        if dd[k] <= DUP_M:
            out.append((str(df["name"].iloc[i] or "(无名)"),
                        str(df["name"].iloc[pt_idx[nn[k]]] or "(无名)"), float(dd[k])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lon", type=float)
    ap.add_argument("--lat", type=float)
    ap.add_argument("--district", default=None, help="限定区名；省略则不限")
    ap.add_argument("--radius", type=float, default=1000.0, help="半径（米，直线距离）")
    ap.add_argument("--preset", default="medical", help="口径预设，见 poi_scope.json")
    ap.add_argument("--outdir", default=r"data\processed\results")
    args = ap.parse_args()

    scope = json.load(open(SCOPE, encoding="utf-8"))
    cats = scope["category_matching"]["presets"].get(args.preset)
    if not cats:
        raise SystemExit(f"未知预设 {args.preset}，可用: {list(scope['category_matching']['presets'])}")

    con = duckdb.connect(DB, read_only=True)
    con.execute("LOAD spatial")

    sql = SQL.replace("{CAT}", ",".join(f"'{c}'" for c in cats))
    params = {"radius": args.radius}
    if args.district:
        sql = sql.replace("{D}", "district = $district AND ")
        params["district"] = args.district
    else:
        sql = sql.replace("{D}", "")

    if args.lon is None or args.lat is None:
        if not args.district:
            raise SystemExit("需给出 --lon/--lat，或给出 --district 以使用该区代表点作中心")
        c = con.execute(
            "SELECT ST_X(p) AS lon, ST_Y(p) AS lat FROM "
            "(SELECT ST_PointOnSurface(geom) AS p FROM districts WHERE name = $district)",
            {"district": args.district},
        ).df().iloc[0]
        lon, lat = float(c["lon"]), float(c["lat"])
        center_src = f"{args.district}边界的 point_on_surface（脚本自动选取）"
    else:
        lon, lat = args.lon, args.lat
        center_src = "命令行给定"

    qx, qy = T.transform(lon, lat)
    params["qx"], params["qy"] = qx, qy
    df = con.execute(sql, params).df()
    con.close()

    df["dist_m"] = df["dist_m"].round(1)
    df = df.sort_values("dist_m").reset_index(drop=True)
    gdf = gpd.GeoDataFrame(df.drop(columns=["wkt"]).copy(),
                           geometry=shapely.from_wkt(df["wkt"].values), crs="EPSG:4326")

    suspect = find_duplicates(df)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(args.outdir, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    gdf.to_file(os.path.join(run_dir, "result.geojson"), driver="GeoJSON")
    gdf.drop(columns="geometry").to_csv(os.path.join(run_dir, "result.csv"), index=False, encoding="utf-8")
    with open(os.path.join(run_dir, "query.sql"), "w", encoding="utf-8") as f:
        f.write(sql + "\n\n-- 绑定参数:\n")
        for k, v in params.items():
            f.write(f"--   ${k} = {v}\n")

    n_pt = int((df["layer"] == "poi_point").sum())
    n_ar = int((df["layer"] == "poi_area").sum())
    scope_txt = args.district or "不限（全域）"
    top = df.head(8)

    lines = []
    lines.append(f"# 查询报告：{scope_txt} · 半径 {args.radius:.0f} m 内的「{args.preset}」类设施")
    lines.append("")
    lines.append(f"生成时间（UTC）：{ts}")
    lines.append("")
    lines.append("## 查询参数")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 中心点 | {lon:.6f}, {lat:.6f}（WGS84） |")
    lines.append(f"| 中心点来源 | {center_src} |")
    lines.append(f"| 半径 | {args.radius:.0f} m（直线距离） |")
    lines.append(f"| 范围限定 | {scope_txt} |")
    lines.append(f"| 类别预设 | {args.preset} = {cats} |")
    lines.append("")
    lines.append("## 数据来源")
    lines.append("")
    lines.append("- POI 与行政边界：OpenStreetMap 北京省级切片，2026-09-11（ODbL 1.0，(c) OpenStreetMap contributors）")
    lines.append("- 边界数据：cn-bj-adm5-wgs84（东城/朝阳/丰台/海淀 取自 OSM，西城区由 DataV 边界经 GCJ-02 纠偏补齐）")
    lines.append("- 数据卡片：servers/geo_catalog/cards/")
    lines.append(f"- 口径定义：servers/geo_knowledge/poi_scope.json（版本 {scope['version']}）")
    lines.append("")
    lines.append("## CRS 与单位声明")
    lines.append("")
    lines.append("- 数据存储 CRS：EPSG:4326（WGS84）")
    lines.append("- 距离计算 CRS：EPSG:32650（UTM 50N，中央经线 117 度）")
    lines.append("- 距离单位：米，直线距离（非路网可达距离）")
    lines.append("- 面积口径：大地线面积（面层 area_m2）")
    lines.append("")
    lines.append("## 计算方法")
    lines.append("")
    lines.append("1. 中心点由 EPSG:4326 经 pyproj 投影到 EPSG:32650")
    lines.append("2. 在 poi_point 与 poi_area 两表中分别按 district 与 category_value 过滤")
    lines.append(f"3. 用平面欧氏距离筛出 <= {args.radius:.0f} m 的记录")
    lines.append("4. 两层结果取并集，按 osm_id 唯一标识，不做几何去重（理由见口径文件）")
    lines.append("5. 面层以其 point_on_surface 代表点参与距离计算")
    lines.append("6. 结果按距离升序")
    lines.append("")
    lines.append("## 结果")
    lines.append("")
    lines.append(f"命中 **{len(df)}** 个（点层 {n_pt}，面层 {n_ar}）。")
    lines.append("")
    lines.append("| 距离(m) | 层 | 名称 | 类别 | 所属区 |")
    lines.append("|---|---|---|---|---|")
    for _, r in top.iterrows():
        lines.append(f"| {r['dist_m']:.0f} | {r['layer']} | {r['name'] or '(无名)'} | {r['category_value']} | {r['district']} |")
    if len(df) > len(top):
        lines.append("")
        lines.append(f"（仅列出最近 {len(top)} 个，完整结果见 result.csv）")
    stat = "、".join(f"{k} {v}" for k, v in df["category_value"].value_counts().items())
    lines.append("")
    lines.append(f"按类别统计：{stat}")
    lines.append("")
    lines.append("## 疑似重复（点面双挂）")
    lines.append("")
    if suspect:
        lines.append(f"以下点层与面层记录相距 <= {DUP_M:.0f} m，可能描述同一设施。"
                     f"按口径保留两条不合并，但计数时需注意：")
        lines.append("")
        lines.append("| 面层名称 | 点层名称 | 相距(m) |")
        lines.append("|---|---|---|")
        for a, b, d in suspect:
            lines.append(f"| {a} | {b} | {d:.0f} |")
        lines.append("")
        lines.append(f"共 {len(suspect)} 对。若本次查询目的是「计数」，建议口径改为去重后再计数。")
    else:
        lines.append("本次结果中未发现距离 <= 150 m 的点面配对。")
    lines.append("")
    lines.append("## 产物")
    lines.append("")
    lines.append("- `result.geojson`：结果几何（点层为点位，面层为多边形状）")
    lines.append("- `result.csv`：结果表格")
    lines.append("- `query.sql`：本次使用的完整 SQL 与绑定参数")
    lines.append("- `report.md`：本报告")
    lines.append("")
    lines.append("## 局限")
    lines.append("")
    lines.append("- 直线距离，未考虑路网与实际通行，不能用于可达性结论")
    lines.append("- OSM 数据完备性取决于志愿者测绘，结论应表述为「OSM 数据显示」")
    lines.append("- 面层以代表点参与距离计算，与设施实际入口可能存在偏差")
    lines.append("- 点面并集存在双挂噪声（同一设施点面同时测绘），见「疑似重复」一节")
    report = "\n".join(lines) + "\n"
    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"产物目录: {run_dir}")


if __name__ == "__main__":
    main()