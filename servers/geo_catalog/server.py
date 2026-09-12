"""geo-catalog：数据目录 MCP Server。

分工原则（见 AGENTS.md 硬性规则）：数据卡片是上下文，作为 Resource 暴露；
检索动作作为 Tool。不把数据卡做成 Tool 的返回值。
"""

import json
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceNotFoundError, ToolError

from geo_catalog import index

ROOT = Path(__file__).resolve().parents[2]
CARDS = ROOT / "servers" / "geo_catalog" / "cards"

mcp = MCPServer(
    name="geo-catalog",
    version="0.2.0",
    description="数据目录与知识库：登记有哪些数据集、各自是什么坐标系、有哪些已知坑，"
                "并混合检索数据卡、口径文件、类别中文别名与坐标系定义。",
)

# search_datasets 只在这几类语料块里检索，别把别名条目混进「该用哪份数据」的答案
DATASET_KINDS = ("dataset_card", "dataset_schema", "dataset_pitfall")


def _cards() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(CARDS.glob("*.json"))}


def _search(query: str, **kwargs) -> dict:
    """索引没建好就把话说清楚，不要退回关键词匹配——静默降级等于换了一套口径。"""
    try:
        return index.search(query, **kwargs)
    except FileNotFoundError as exc:
        raise ToolError(str(exc)) from exc


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
    p = next((q for q in CARDS.glob("*.json") if q.stem == dataset_id), None)
    if p is None:
        raise ResourceNotFoundError(f"未登记的数据集: {dataset_id}；可用 {sorted(_cards())}")
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
def search_datasets(query: str, limit: int = 5) -> list[dict]:
    """检索数据集，覆盖名称、描述、数据源、字段定义与全部已知坑。

    用于回答「有没有医院数据」「哪份数据有坐标系问题」这类问题。命中项带
    source_uri，可按它读完整数据卡（catalog://dataset/{id}）。
    检索是混合式：BM25 关键词 + bge 向量，按 RRF 融合排名。
    """
    result = _search(query, limit=limit, kinds=DATASET_KINDS)
    return [{"id": hit["dataset_id"], "title": hit["title"], "kind": hit["kind"],
             "source_uri": hit["source_uri"], "score": hit["score"],
             "matched_by": hit["matched_by"], "snippet": hit["snippet"]}
            for hit in result["hits"]]


@mcp.tool()
def search_knowledge(query: str, limit: int = 5, kinds: list[str] | None = None,
                     dataset_id: str | None = None) -> dict:
    """检索知识库：数据卡、口径文件、类别中文别名、坐标系定义。

    用于回答「口径是怎么定的」「地铁站为什么查不到」「这个中文说法对应哪个 OSM 标签」
    「坐标系有哪些坑」这类问题。命中项带 source_uri，要看全文就按它读对应 Resource。
    kinds 可按语料类型过滤：dataset_card / dataset_schema / dataset_pitfall /
    scope / scope_section / alias / alias_unavailable；dataset_id 限定到某份数据。
    返回里带 retrieval 段，写明这次各召回了几条候选、融合方式与向量模型。
    """
    return _search(query, limit=limit, kinds=kinds, dataset_id=dataset_id)


@mcp.resource("catalog://knowledge", name="knowledge-index",
              description="知识库检索索引的规模与构成")
def knowledge_index() -> dict:
    """索引里有哪些语料、各多少条、建了哪些索引。检索动作请用 search_knowledge 工具。"""
    try:
        return index.stats()
    except FileNotFoundError as exc:
        raise ResourceNotFoundError(str(exc)) from exc


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
