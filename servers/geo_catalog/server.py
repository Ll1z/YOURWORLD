"""geo-catalog：数据目录 MCP Server。

分工原则（见 AGENTS.md 硬性规则）：数据卡片是上下文，作为 Resource 暴露；
检索动作作为 Tool。不把数据卡做成 Tool 的返回值。
"""

import json
from pathlib import Path

from mcp.server.mcpserver import MCPServer

ROOT = Path(__file__).resolve().parents[2]
CARDS = ROOT / "servers" / "geo_catalog" / "cards"

mcp = MCPServer(
    name="geo-catalog",
    version="0.1.0",
    description="数据目录：登记有哪些数据集、各自是什么坐标系、有哪些已知坑。回答「该用哪份数据」这类问题。",
)


def _cards() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(CARDS.glob("*.json"))}


def _searchable(card: dict) -> str:
    parts = [card.get("name", ""), card.get("description", "")]
    parts += [str(v) for v in (card.get("source") or {}).values()]
    parts += (card.get("known_pitfalls") or [])
    parts += list((card.get("schema") or {}).values())
    parts += list((card.get("layers") or {}).keys())
    return " ".join(parts).lower()


@mcp.resource("catalog://datasets", name="datasets", description="全部数据集的摘要清单")
def list_resources_() -> dict:
    """列出所有已登记数据集的摘要，完整数据卡请读 catalog://dataset/{dataset_id}。"""
    out = []
    for cid, c in _cards().items():
        out.append({
            "id": cid,
            "name": c.get("name"),
            "description": c.get("description"),
            "crs": (c.get("crs") or {}).get("name"),
            "pitfall_count": len(c.get("known_pitfalls") or []),
            "uri": f"catalog://dataset/{cid}",
        })
    return {"count": len(out), "datasets": out}


@mcp.resource("catalog://dataset/{dataset_id}", name="dataset-card",
              description="按 id 读取完整数据卡片")
def get_card(dataset_id: str) -> dict:
    """读取指定数据集的完整卡片，含 CRS、单位、字段 schema、样例行与全部已知坑。"""
    p = CARDS / f"{dataset_id}.json"
    if not p.exists():
        raise ValueError(f"未登记的数据集: {dataset_id}；可用 {sorted(_cards())}")
    return json.loads(p.read_text(encoding="utf-8"))


@mcp.tool()
def list_datasets() -> list[dict]:
    """列出所有已登记数据集的摘要（id、名称、坐标系、已知坑数量）。

    只返回摘要以节省上下文；需要完整的字段定义、已知坑与样例行时，
    读取对应的 catalog://dataset/{id} 资源。
    """
    return [
        {"id": cid, "name": c.get("name"), "description": c.get("description"),
         "crs": (c.get("crs") or {}).get("name"),
         "pitfall_count": len(c.get("known_pitfalls") or [])}
        for cid, c in _cards().items()
    ]


@mcp.tool()
def search_datasets(query: str) -> list[dict]:
    """按关键词检索数据集，匹配范围包括名称、描述、数据源与全部已知坑。

    用于回答「有没有医院数据」「哪份数据有坐标系问题」这类问题。
    注意：当前为关键词匹配实现，DuckDB FTS + VSS 混合检索是后续待办。
    """
    tokens = [t for t in query.lower().replace("，", " ").replace(",", " ").split() if t]
    scored = []
    for cid, c in _cards().items():
        text = _searchable(c)
        score = sum(text.count(t) for t in tokens)
        if score:
            scored.append({
                "id": cid, "name": c.get("name"), "score": score,
                "matched_pitfalls": [p for p in (c.get("known_pitfalls") or [])
                                     if any(t in p.lower() for t in tokens)],
            })
    return sorted(scored, key=lambda r: -r["score"])


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()