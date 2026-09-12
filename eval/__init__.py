"""评测集：任务集、期望值投影与判定规则。

设计要点：
  1. 期望值不手写。eval/ground_truth.json 由 scripts/build_eval.py 跑真实 MCP 工具生成并冻结；
     scripts/run_eval.py --mode data 重新跑一遍与之对比，口径 / SQL / 别名 / 数据的漂移都会显形。
  2. 判定分两层：data 层确定性、不花 token，改一次口径就该跑一次；agent 层真跑 Agent，
     判定「调没调对工具、数字能不能溯源、关键数字有没有落到答案里、该拒绝的是否拒绝」。
  3. 本模块只做投影、比较与判定，不生成任何数字——数字一律来自工具返回。
  4. refusal / tool_error 两类用关键词做代理判定，不是语义判定，报告里要如实标注。
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "eval" / "cases.json"
GROUND_TRUTH = ROOT / "eval" / "ground_truth.json"

# 0 命中的题目不能用「数字是否出现」判定，改用这些说法
ZERO_WORDS = re.compile(r"(没有|没有任何|无此类|未发现|零|0\s*(个|条|家|座|处))")
NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*")


def load_cases(path: Path | None = None) -> list[dict]:
    return json.loads((path or CASES).read_text(encoding="utf-8"))["cases"]


def load_ground_truth(path: Path | None = None) -> dict | None:
    p = path or GROUND_TRUTH
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def plainify(value):
    """把 numpy 标量与 NaN 归一成可 JSON 化的原生值（工具返回里两者都有）。"""
    if isinstance(value, dict):
        return {str(k): plainify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plainify(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return plainify(value.item())
        except (AttributeError, ValueError):
            return value
    return value


def project(tool: str, payload) -> dict:
    """把一次工具返回压成可冻结、可对比的显著字段。

    只留「换个数字就意味着口径变了」的字段：计数、逐类别计数、最近若干条的
    名称与距离、距离指标本身。完整明细不入库，避免 ground_truth 被 WKT 撑爆。
    """
    data = payload
    if isinstance(data, dict) and isinstance(data.get("result"), list):
        data = data["result"]  # MCP 对返回 list 的工具会包一层
    if isinstance(payload, dict) and payload.get("error"):
        return {"error": payload["error"]}

    if tool == "query_nearby":
        return plainify({
            "count_total": data["count_total"],
            "count_point": data["count_point"],
            "count_area": data["count_area"],
            "categories": {c["label"]: c["count"] for c in data["categories"]},
            "nearest": [[h.get("name") or "", round(h["dist_m"], 1)] for h in data["hits"][:3]],
        })
    if tool == "summarize_poi":
        return plainify({
            "count_total": data["count_total"],
            "categories": {c["label"]: c["count"] for c in data["category_tally"]},
            "by_district": data["by_district"],
            "by_category": data["by_category"],
        })
    if tool == "distance_between":
        return plainify({k: data[k] for k in ("planar_distance_m", "geodesic_distance_m",
                                             "planar_vs_geodesic_pct", "bearing_deg")})
    if tool == "find_places":
        return plainify({"hits": [[h["name"], h["ref"], h["match_score"]] for h in data[:3]]})
    return plainify({"raw": data})


def compare(expect: dict, actual: dict, path: str = "") -> list[str]:
    """逐字段精确比较（data 层用，不含容差）。返回不一致的描述列表，空列表表示一致。"""
    problems: list[str] = []
    if isinstance(expect, dict) and isinstance(actual, dict):
        for key in sorted(set(expect) | set(actual)):
            where = f"{path}.{key}" if path else key
            if key not in expect:
                problems.append(f"{where}: 多出字段 {actual[key]!r}")
            elif key not in actual:
                problems.append(f"{where}: 缺少字段（期望 {expect[key]!r}）")
            else:
                problems.extend(compare(expect[key], actual[key], where))
        return problems
    if isinstance(expect, list) and isinstance(actual, list):
        if len(expect) != len(actual):
            return [f"{path}: 长度不同（期望 {len(expect)}，实际 {len(actual)}）"]
        for i, (e, a) in enumerate(zip(expect, actual)):
            problems.extend(compare(e, a, f"{path}[{i}]"))
        return problems
    if isinstance(expect, bool) or isinstance(actual, bool):
        if expect != actual:
            problems.append(f"{path}: 期望 {expect!r}，实际 {actual!r}")
        return problems
    if isinstance(expect, (int, float)) and isinstance(actual, (int, float)):
        if expect != actual:
            problems.append(f"{path}: 期望 {expect!r}，实际 {actual!r}")
        return problems
    if expect != actual:
        problems.append(f"{path}: 期望 {expect!r}，实际 {actual!r}")
    return problems


def numbers_in(text: str) -> list[float]:
    out = []
    for raw in NUMBER_RE.findall(text or ""):
        try:
            out.append(float(raw.replace(",", "")))
        except ValueError:
            continue
    return out


def number_present(text: str, target: float, tolerance: float) -> bool:
    """答案里是否出现了目标数字（容差内）。0 命中题改用「没有」类说法判定。"""
    if target == 0:
        return bool(ZERO_WORDS.search(text or ""))
    return any(abs(n - target) <= tolerance for n in numbers_in(text))


def key_numbers(case: dict, expected: dict) -> list[tuple[str, float, float]]:
    """(标签, 目标值, 容差)。容差只在 agent 层用：同名候选点会让半径边缘的设施进出。

    「五个区各有多少家」这类题问的是分区明细，不是总数，用 breakdown 标记后
    逐区校验——只对总数会让答案明明正确却判失败。
    """
    pct = float(case.get("tolerance_pct", 0) or 0) / 100.0
    kind = case.get("kind")
    if expected.get("error"):
        return []
    if kind in ("nearby", "aggregate"):
        target = float(expected["count_total"])
        if case.get("breakdown"):
            return [(f"{d}", float(n), float(n) * pct)
                    for d, n in sorted((expected.get("by_district") or {}).items())]
        return [("总数", target, target * pct)]
    if kind == "distance":
        target = float(expected["planar_distance_m"])
        # 平面距离与大地线距离都算对：北京范围内两者偏差远小于 0.5%
        return [("距离", target, max(0.5, target * max(pct, 0.005)))]
    return []


def _alternate_hit(case: dict, expected: dict, answer: str, label: str,
                   target: float, tolerance: float) -> str | None:
    """总数或距离被换成等价说法时，接受它。

    - nearby：模型常写「点层 1 个、面层 66 个」而不是「共 67 个」，信息等价且同样能在
      工具返回里逐项溯源；只要各层读数能加回总数就算过
    - distance：写成「约 6.06 公里」与「6061 米」等价
    只在等价说法本身可验证时才认，避免把「数字碰巧出现」当成命中。
    """
    if label == "总数" and case.get("kind") == "nearby":
        parts = [v for v in (expected.get("count_point"), expected.get("count_area")) if v]
        if parts and sum(parts) == target and all(number_present(answer, p, 0) for p in parts):
            joined = " + ".join(f"{p:g}" for p in parts)
            return f"总数未整段出现，但分层读数 {joined} 全部出现（合计 {sum(parts):g}），判定通过"
    if label == "距离" and target >= 1000:
        km, km_tol = target / 1000.0, max(tolerance, 5.0) / 1000.0
        if number_present(answer, km, km_tol):
            return f"答案以公里表述 {km:g} km（容差 ±{km_tol:g}），与 {target:g} m 等价"
    return None


def score_answer(case: dict, expected: dict, run_result, answer: str,
                 checks: list) -> list[tuple[str, bool, str]]:
    """agent 层判定。检查项名字固定，便于汇总成回归线。"""
    if case.get("agent_skip"):
        return [("不判定", True, case["agent_skip"])]
    kind = case.get("kind")
    invocations = list(run_result.invocations)
    ok_names = [i.name for i in invocations if i.ok]
    results: list[tuple[str, bool, str]] = []

    if kind == "refusal":
        guessed = [i.name for i in invocations if i.name == "query_nearby" and i.ok]
        results.append(("拒答:未猜中心", not guessed,
                        "没有成功的 query_nearby 调用" if not guessed
                        else f"无中心却成功查询了 {len(guessed)} 次，中心被擅自假定"))
        words = case.get("answer_must_mention") or []
        hit = next((w for w in words if w in (answer or "")), None)
        results.append(("拒答:说明理由", hit is not None,
                        f"命中「{hit}」" if hit else f"答案里没有出现 {words} 中的任何一个"))
        return results

    if kind == "tool_error":
        words = case.get("answer_must_mention") or []
        if not words:
            return [("工具错误:已说明", False, "用例缺少 answer_must_mention，无法判定")]
        hit = next((w for w in words if w in (answer or "")), None)
        return [("工具错误:已说明", hit is not None,
                 f"命中「{hit}」" if hit else f"答案里没有出现 {words} 中的任何一个")]

    tool = case.get("tool")
    results.append(("调用正确工具", tool in ok_names,
                    f"成功调用了 {tool}" if tool in ok_names
                    else f"成功调用的工具是 {ok_names}，期望 {tool}"))

    grounding = next((c for c in checks if c.name == "grounding"), None)
    if grounding is None:
        results.append(("数值可溯源", False, "缺少 grounding 检查结果"))
    else:
        results.append(("数值可溯源", grounding.passed, grounding.detail))

    if kind == "places":
        place = (case.get("center") or {}).get("place")
        results.append(("答案含地名", bool(place) and place in (answer or ""),
                        f"答案里{'出现' if place and place in (answer or '') else '没出现'}「{place}」"))
        return results

    for label, target, tolerance in key_numbers(case, expected):
        ok = number_present(answer, target, tolerance)
        detail = (f"期望 {label} {target:g}（容差 ±{tolerance:g}）"
                  f"{'出现在' if ok else '未出现在'}答案中")
        if not ok:
            alternate = _alternate_hit(case, expected, answer, label, target, tolerance)
            if alternate:
                ok, detail = True, alternate
        results.append((f"关键数字:{label}", ok, detail))

    if kind == "nearby":
        spatial = [c for c in checks if c.name in ("crs", "units", "geometry", "magnitude")]
        bad = [c.name for c in spatial if not c.passed]
        results.append(("空间自检", not bad, "四项基础检查通过" if not bad else f"未通过：{bad}"))
    return results


def summarize_scores(rows: list[dict]) -> dict:
    total = len(rows)
    passed = sum(1 for r in rows if r["passed"])
    per_check: dict[str, list[bool]] = {}
    for row in rows:
        for name, ok, _ in row["checks"]:
            per_check.setdefault(name, []).append(ok)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "by_check": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(per_check.items())},
        "failed_ids": [r["id"] for r in rows if not r["passed"]],
    }
