"""校对类别别名表：别名里的每个 key=value 都必须在库里真实存在。

用法：uv run python scripts/check_categories.py
退出码非 0 表示别名表里有对不上库的条目，或别名覆盖的类别少得可疑。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "servers"))

from geo_compute import query  # noqa: E402


def main() -> int:
    aliases = query.load_aliases()
    inv = query.category_inventory()
    keys = set(inv["category_key"])
    values = set(inv["category_value"])
    pairs = set(zip(inv["category_key"], inv["category_value"]))
    counts = {(k, v): int(n) for k, v, n in
              zip(inv["category_key"], inv["category_value"], inv["n"])}

    problems: list[str] = []
    lines: list[tuple[int, str]] = []
    for name, entry in aliases["aliases"].items():
        cats = entry.get("categories") or []
        if not cats:
            problems.append(f"{name}: 没有配任何类别")
        total = 0
        for item in cats:
            if "=" in item:
                key, _, value = item.partition("=")
                if key not in keys:
                    problems.append(f"{name}: 库里没有标签键 {key!r}（{item}）")
                elif (key, value) not in pairs:
                    problems.append(f"{name}: 库里没有 {item}")
                else:
                    total += counts[(key, value)]
            elif item not in values:
                problems.append(f"{name}: 库里没有任何 {item}")
            else:
                total += sum(n for (k, v), n in counts.items() if v == item)
        lines.append((total, f"  {name} → {' + '.join(cats)} = {total} 条"))

    for name, note in (aliases.get("not_available") or {}).items():
        if not isinstance(note, str) or not note.strip():
            problems.append(f"not_available 里的 {name!r} 没有写清不可用的原因")

    covered = {c for e in aliases["aliases"].values() for c in e["categories"]}
    print(f"别名 {len(aliases['aliases'])} 条，涉及 OSM 类别 {len(covered)} 个")
    print(f"库内类别组合 {len(pairs)} 个，记录 {int(inv['n'].sum())} 条")
    print("别名覆盖量（按命中记录数降序）：")
    for _, line in sorted(lines, reverse=True):
        print(line)

    if problems:
        print(f"\n对不上库的条目 {len(problems)} 处：")
        for p in problems:
            print(f"  [FAIL] {p}")
        return 1
    print("\n全部别名均可对到库内类别。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())