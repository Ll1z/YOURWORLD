"""空间自检器：把静默错误变成显式失败。

四项检查对应 HANDOFF.md 的 Stage 1 要求：
  1. crs       —— 坐标参考系与声明是否一致，坐标是否落在合理经纬度范围
  2. units     —— 半径、距离、面积的单位是否自洽（米 vs 度是最常见的静默错误）
  3. geometry  —— 几何有效性，以及每条命中都能在库里回溯到
  4. magnitude —— 计数自洽、主键无重复、无跨区泄漏、量级有界

外加一项数值溯源检查：最终回答里的数字是否都出现在工具返回中。
这一项服务于硬性规则「数值结论必须由代码算出，LLM 不得直接生成数字」。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "processed" / "geo.duckdb"

STORAGE_CRS = "EPSG:4326"
DISTANCE_CRS = "EPSG:32650"
CHINA_BBOX = (73.0, 3.0, 135.1, 53.6)
LAYERS = ("poi_point", "poi_area")
MAX_PLAUSIBLE_AREA_M2 = 1e9


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def _in_china(lon: float, lat: float) -> bool:
    return CHINA_BBOX[0] <= lon <= CHINA_BBOX[2] and CHINA_BBOX[1] <= lat <= CHINA_BBOX[3]


def check_crs(result: dict) -> Check:
    problems: list[str] = []
    lon, lat = result.get("center_lon"), result.get("center_lat")
    if lon is None or lat is None:
        return Check("crs", False, "结果缺少查询中心经纬度")
    if not _in_china(lon, lat):
        problems.append(f"中心点 ({lon}, {lat}) 不在中国经纬度范围内，疑似投影坐标被当成经纬度使用")

    outside = [h["osm_id"] for h in result.get("hits", []) if not _in_china(h["lon"], h["lat"])]
    if outside:
        problems.append(f"{len(outside)} 条命中经纬度越界，示例 {outside[:3]}")

    note = result.get("crs_note", "")
    if STORAGE_CRS not in note or DISTANCE_CRS not in note:
        problems.append(f"结果未同时声明存储 CRS({STORAGE_CRS}) 与距离 CRS({DISTANCE_CRS})：{note!r}")

    detail = "；".join(problems) or (
        f"声明存储 {STORAGE_CRS} / 距离 {DISTANCE_CRS}；中心点与全部命中均在合理经纬度范围内"
    )
    return Check("crs", not problems, detail)


def check_units(result: dict) -> Check:
    problems: list[str] = []
    hits = result.get("hits", [])
    radius = result.get("radius_m")

    if not radius or radius <= 0:
        problems.append(f"半径应为正数（米），实际 {radius!r}")
    else:
        over = [h["osm_id"] for h in hits if h["dist_m"] > radius + 1e-6]
        if over:
            problems.append(f"{len(over)} 条命中距离超过半径，疑似单位不一致（米 vs 度）：{over[:3]}")

    bad_area = [
        h["osm_id"] for h in hits
        if h.get("area_m2") is not None and not (0 < h["area_m2"] < MAX_PLAUSIBLE_AREA_M2)
    ]
    if bad_area:
        problems.append(f"{len(bad_area)} 条面层面积不在 (0, 1e9) 平方米内：{bad_area[:3]}")

    point_with_area = [h["osm_id"] for h in hits if h["layer"] == "poi_point" and h.get("area_m2")]
    if point_with_area:
        problems.append(f"{len(point_with_area)} 条点层记录带了面积字段")

    area_without_area = [h["osm_id"] for h in hits if h["layer"] == "poi_area" and h.get("area_m2") is None]
    if area_without_area:
        problems.append(f"{len(area_without_area)} 条面层记录缺面积字段")

    detail = "；".join(problems) or (
        f"半径 {radius:g} 米；{len(hits)} 条命中距离均不超过半径，面层面积均为正且量级正常"
    )
    return Check("units", not problems, detail)


def check_geometry(result: dict) -> Check:
    if not DB.exists():
        return Check("geometry", False, f"数据库不存在：{DB}")

    problems: list[str] = []
    con = duckdb.connect(str(DB), read_only=True)
    con.execute("LOAD spatial")
    try:
        invalid = con.execute(
            "SELECT " + " + ".join(
                f"(SELECT count(*) FROM {t} WHERE NOT ST_IsValid(geom))" for t in LAYERS)
        ).fetchone()[0]
        if invalid:
            problems.append(f"库内无效几何 {invalid} 个（读入时须 on_invalid=\"ignore\"）")

        empty = con.execute(
            "SELECT " + " + ".join(
                f"(SELECT count(*) FROM {t} WHERE ST_IsEmpty(geom))" for t in LAYERS)
        ).fetchone()[0]
        if empty:
            problems.append(f"库内空几何 {empty} 个")

        missing = []
        for h in result.get("hits", []):
            layer = h.get("layer")
            if layer not in LAYERS:
                missing.append(f"非法图层 {layer!r}")
                continue
            n = con.execute(
                f"SELECT count(*) FROM {layer} WHERE osm_id = $i", {"i": h["osm_id"]}
            ).fetchone()[0]
            if n == 0:
                missing.append(f"{layer}/{h['osm_id']}")
        if missing:
            problems.append(f"{len(missing)} 条命中无法在库中回溯：{missing[:3]}")
    finally:
        con.close()

    detail = "；".join(problems) or (
        f"两图层几何全部有效且非空；{len(result.get('hits', []))} 条命中均可在库中按其 osm_id 回溯"
    )
    return Check("geometry", not problems, detail)


def check_magnitude(result: dict) -> Check:
    problems: list[str] = []
    hits = result.get("hits", [])
    total = result.get("count_total")

    if total != len(hits):
        problems.append(f"count_total={total} 与 hits 实际长度 {len(hits)} 不一致")
    if result.get("count_point", 0) + result.get("count_area", 0) != total:
        problems.append("点层计数 + 面层计数 != 总数")
    if result.get("count_point") != sum(1 for h in hits if h["layer"] == "poi_point"):
        problems.append("count_point 与 hits 中点层条数不符")
    if result.get("count_area") != sum(1 for h in hits if h["layer"] == "poi_area"):
        problems.append("count_area 与 hits 中面层条数不符")

    keys = [(h["layer"], h["osm_id"]) for h in hits]
    if len(keys) != len(set(keys)):
        problems.append("同一 (图层, osm_id) 出现多次，疑似关联查询发生笛卡尔扇出")

    dists = [h["dist_m"] for h in hits]
    if dists != sorted(dists):
        problems.append("结果未按距离升序排列")

    district = result.get("district")
    if district:
        leaked = {h["district"] for h in hits if h.get("district") and h["district"] != district}
        if leaked:
            problems.append(f"限定 {district}，但结果里出现其他区：{sorted(leaked)}")

    if total and total > 100000:
        problems.append(f"命中数 {total} 量级异常，北京五区一个点周围不可能有这个数量")

    detail = "；".join(problems) or (
        f"{total} 条命中：计数自洽、主键无重复、距离严格升序"
        + (f"、未越出 {district}" if district else "")
    )
    return Check("magnitude", not problems, detail)


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
# 千分位逗号要先去干净，否则「18,610.7」会被拆成 18 和 610.7 两个假数字
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


def _collect_numbers(obj, pool: set[float]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        pool.add(float(obj))
    elif isinstance(obj, str):
        pool.update(float(m) for m in _NUMBER.findall(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, pool)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_numbers(v, pool)


def _grounding_pool(tool_results) -> set[float]:
    """把工具返回里的数字展开成可溯源集合。

    额外加入 ×1000 / ÷1000 与取整变体：模型把「1000 米」写成「1 公里」、
    把「371.4 米」写成「371 米」都属于正常的同一事实，不该判为编造。
    用的是绝对容差而非百分比容差——百分比会让 116.4 这种经度把 120 也「兜住」。
    """
    raw: set[float] = set()
    _collect_numbers(tool_results, raw)
    pool: set[float] = set()
    for t in raw:
        pool.update({t, t / 1000.0, t * 1000.0, t / 10000.0, t * 10000.0, float(round(t))})
    return pool


def check_grounding(answer: str, tool_results) -> Check:
    """最终回答里的数字必须能在工具返回中找到出处。"""
    pool = _grounding_pool(tool_results)
    ungrounded = [
        token for token in _NUMBER.findall(_THOUSANDS.sub("", answer))
        if not any(abs(float(token) - t) <= 0.5001 for t in pool)
    ]

    if ungrounded:
        return Check(
            "grounding", False,
            f"{len(ungrounded)} 个数字无法在工具返回中溯源，疑似模型自行生成：{ungrounded[:8]}",
        )
    return Check("grounding", True, "回答中的全部数字都能在工具返回中找到出处")


def run_all(result: dict) -> list[Check]:
    return [check_crs(result), check_units(result), check_geometry(result), check_magnitude(result)]


def render(checks: list[Check]) -> str:
    lines = ["| 检查项 | 结果 | 说明 |", "|---|---|---|"]
    for c in checks:
        lines.append(f"| {c.name} | {'通过' if c.passed else '**未通过**'} | {c.detail} |")
    return "\n".join(lines)
