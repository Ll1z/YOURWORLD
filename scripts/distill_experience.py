"""经验库蒸馏：把运行轨迹里反复出现的坑抽成候选经验。

用法：
    uv run python scripts/distill_experience.py              # 扫描轨迹 → 合并进经验库
    uv run python scripts/distill_experience.py --dry-run    # 只说要新增/累加什么，不写文件

为什么是规则抽取，而不是让模型写小结：经验条必须能追到具体哪一次运行。
模型写的教训读起来更顺，但无法验证，也累计不了出现次数——而「同一个坑踩了几次」
正是经验值的全部来源。模型该做的是**用**这些经验（search_knowledge 命中 kind=experience），
不是编它们。

抽三类：
- tool_error：工具报错后重试成功的，记下错误特征与当时的修正参数
- grounding：数值溯源打回重写的，记下无法溯源的数字与当时的工具链
- steps：跑满步数上限仍未收敛的

signature 是去重与计数的键，格式即约定：tool_error:{工具}:{错误特征}、
grounding:{工具链}、steps_exhausted:{工具链}；人工写的、机器认不出来的用 manual: 前缀。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LESSONS = ROOT / "servers" / "geo_knowledge" / "experience" / "lessons.json"

# 轨迹来源：评测跑出来的、日常问答跑出来的。两边格式一样，都是 AgentRun.to_dict()
SOURCES = (
    ("eval", ROOT / "eval" / "runs"),
    ("agent", ROOT / "data" / "processed" / "results"),
)


# MCP 把工具异常包成 {'text': "Error executing tool <名字>: <真正的原因>"}，
# 工具名已经进了 signature，这里只剥掉包装、留下原因
_WRAPPER = re.compile(r"^.*?Error executing tool\s+\S+\s*:\s*")


def _error_key(message: str) -> str:
    """错误特征：剥掉 MCP 包装，取第一句，去掉引号与括号里的具体值。

    同一个「不认识的类别 '医院院'」和「不认识的类别 '银行行'」是同一条经验，
    具体是哪个词属于例子，不属于特征。
    """
    head = re.split(r"[。；;\n]", _WRAPPER.sub("", (message or "").strip()))[0]
    head = re.sub(r"[「『\"'](.*?)[」』\"']", "", head)
    head = re.sub(r"[\[（(](.*?)[\]）)]", "", head)
    head = re.sub(r"\s+", "", head).strip("：:，,、}{")
    return head[:20] or "未知错误"


def _tool_chain(trace: dict) -> str:
    """工具链：按首次出现顺序去重，看的是「这条问法要经过哪几步」。"""
    names = [i.get("name") or "?" for i in trace.get("invocations") or []]
    return ">".join(dict.fromkeys(names)) or "无工具"


def candidates_from(trace: dict, where: str) -> list[dict]:
    invocations = trace.get("invocations") or []
    out: list[dict] = []

    for err in invocations:
        if err.get("ok"):
            continue
        name = err.get("name") or "?"
        fixed = next((i for i in invocations
                      if i.get("ok") and i.get("name") == name
                      and (i.get("step") or 0) >= (err.get("step") or 0)), None)
        out.append({
            "signature": f"tool_error:{name}:{_error_key(err.get('error') or '')}",
            "kind": "tool_error",
            "summary": f"{name} 报错，{'随后重试成功' if fixed else '本次没有恢复'}",
            "examples": [{
                "where": where,
                "detail": (err.get("error") or "")[:200],
                "fix": json.dumps(fixed["arguments"], ensure_ascii=False) if fixed else None,
            }],
        })

    offenders: list[str] = []
    for span in trace.get("spans") or []:
        values = (span.get("attributes") or {}).get("geo.offenders") or []
        # 老轨迹里这个字段存的是个数（int），只有新轨迹才有具体数字
        if isinstance(values, (list, tuple)):
            offenders.extend(str(v) for v in values)
    if offenders:
        chain = _tool_chain(trace)
        out.append({
            "signature": f"grounding:{chain}",
            "kind": "grounding",
            "summary": f"「{chain}」这条链的回答里出现过无法溯源的数字",
            "examples": [{"where": where,
                          "detail": "无法溯源的数字：" + "、".join(offenders[:10])}],
        })

    if trace.get("stopped") == "max_steps":
        chain = _tool_chain(trace)
        out.append({
            "signature": f"steps_exhausted:{chain}",
            "kind": "steps",
            "summary": f"「{chain}」跑满步数上限仍未收口",
            "examples": [{"where": where, "detail": f"步数 {trace.get('steps')}"}],
        })
    return out


def scan() -> list[dict]:
    found: list[dict] = []
    for label, base in SOURCES:
        if not base.exists():
            continue
        for path in sorted(base.glob("**/trace.json")):
            try:
                trace = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            where = f"{label}:{path.parent.relative_to(base).as_posix()}"
            found.extend(candidates_from(trace, where))
    return found


def group(found: list[dict]) -> dict[str, dict]:
    groups: dict[str, dict] = {}
    for cand in found:
        entry = groups.setdefault(cand["signature"], {
            "signature": cand["signature"], "kind": cand["kind"],
            "summary": cand["summary"], "occurrences": 0, "examples": [],
        })
        entry["occurrences"] += 1
        entry["examples"].extend(cand["examples"])
    return groups


def merge(doc: dict, groups: dict[str, dict], now: str,
          max_examples: int = 3, max_sources: int = 8) -> dict:
    """把这次的观测并进经验库。已有的累加次数，没见过的进 candidates。"""
    report = {"new": [], "updated": []}
    index = {e["signature"]: e for e in doc["lessons"] + doc["candidates"]}

    for signature, fresh in sorted(groups.items()):
        entry = index.get(signature)
        if entry is None:
            entry = {
                "signature": signature,
                "kind": fresh["kind"],
                "summary": fresh["summary"],
                "origin": "蒸馏",
                "occurrences": 0,
                "first_seen": now,
                "last_seen": now,
                "sources": [],
                "examples": [],
            }
            doc["candidates"].append(entry)
            index[signature] = entry
            report["new"].append((signature, fresh["occurrences"]))
        else:
            report["updated"].append((signature, fresh["occurrences"], entry["occurrences"]))
        entry["occurrences"] += fresh["occurrences"]
        entry["last_seen"] = now
        seen = {e.get("where") for e in entry["examples"]}
        for example in fresh["examples"]:
            if example["where"] not in seen:
                entry["examples"].append(example)
                seen.add(example["where"])
        entry["examples"] = entry["examples"][-max_examples:]
        where = [e["where"] for e in entry["examples"]]
        entry["sources"] = list(dict.fromkeys([*entry.get("sources", []), *where]))[-max_sources:]

    doc["updated_at"] = now
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="不写文件，只报告")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    doc = json.loads(LESSONS.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    found = scan()
    groups = group(found)
    report = merge(doc, groups, now)

    print(f"扫描到 {len(found)} 条观测，归并成 {len(groups)} 个特征")
    for signature, count in report["new"]:
        print(f"  [新增候选] {signature}（{count} 次）")
    for signature, count, before in report["updated"]:
        print(f"  [累加] {signature}：{before} → {before + count} 次")
    if not report["new"] and not report["updated"]:
        print("  没有新东西")

    if args.dry_run:
        print("dry-run：未写文件")
        return 0
    LESSONS.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"已写入 {LESSONS.relative_to(ROOT)}："
          f"教训 {len(doc['lessons'])} 条，候选 {len(doc['candidates'])} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
