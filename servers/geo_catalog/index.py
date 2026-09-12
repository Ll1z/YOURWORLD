"""知识库检索：DuckDB 单文件 FTS（BM25）+ VSS（向量）混合索引。

对应 AGENTS.md 的 RAG 四类索引里的第一类：数据卡片 / 文档 / 标准。四类索引各司其职，
空间对象仍然走 R-tree——绝不用 embedding 做空间过滤。

索引落在 data/processed/knowledge.duckdb 一个文件里：
  chunks     语料块（chunk_id / kind / dataset_id / crs / source_uri / title / text / tokens / vec）
  fts 索引   DuckDB fts 扩展，BM25，建在 tokens 列上
  hnsw 索引  DuckDB vss 扩展，余弦距离，建在 vec 列上

中文 FTS 的关键是分词。DuckDB 的 fts 扩展按空白与标点切词，整句中文会被当成一个词，
检索直接返回 0（实测确认）。这里用字符二元组（bigram）预分词——Lucene 的 CJKBigramFilter
就是这么做，不需要词典依赖；真想换 jieba 只需要改 tokenize 一处。

检索是 filter-then-rank：先用结构化条件（kind / dataset_id / crs）过滤，再分别取
BM25 与向量的 top-k，最后用 RRF 融合。用排名融合而不是加权求和，是因为 BM25 分数与
余弦距离不同量纲，拍一个权重不如用排名稳。
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

import duckdb

from geo_catalog.embedding import DIM, Embedder

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "processed" / "knowledge.duckdb"
CARDS = ROOT / "servers" / "geo_catalog" / "cards"
KNOWLEDGE = ROOT / "servers" / "geo_knowledge"
TABLE = "chunks"
RRF_K = 60
POOL = 20
# 余弦相似度双阈值，取值来自实测分布：相关问句的 top-1 落在 0.62-0.71，
# 完全无关的问句也有 0.46（bge 的相似度基线偏高，光看绝对值没有区分度）。
# 所以先卡绝对下限 0.45，再拉到 top-1 附近 ±0.06 —— 既不把弱问句的向量侧清零，
# 也不让明显不相关的块靠名次挤进 RRF。
MIN_SIM = 0.45
SIM_MARGIN = 0.06

_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_WORD = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> str:
    """中文按字符二元组切，ASCII 按词切，拼成空格分隔串交给 BM25。"""
    lowered = (text or "").lower()
    tokens: list[str] = []
    for run in _CJK.findall(lowered):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    tokens.extend(_WORD.findall(lowered))
    return " ".join(tokens)


def _flatten(value: Any, prefix: str = "") -> str:
    if isinstance(value, dict):
        return "\n".join(_flatten(v, f"{prefix}{k}：") for k, v in value.items())
    if isinstance(value, (list, tuple)):
        if all(not isinstance(v, (dict, list, tuple)) for v in value):
            return f"{prefix}{'；'.join(str(v) for v in value)}"
        return "\n".join(_flatten(v, prefix) for v in value)
    return f"{prefix}{value}"


def chunks() -> list[dict]:
    """把数据卡、口径文件、类别别名、坐标系定义切成语料块。

    切法按「检索单位」定：数据卡整体一条（要的是全局印象），每个已知坑单独一条
    （「哪份数据有坐标系问题」要能直接命中那一条），口径文件每个顶层小节一条。
    """
    out: list[dict] = []

    for path in sorted(CARDS.glob("*.json")):
        card = json.loads(path.read_text(encoding="utf-8"))
        cid, uri = path.stem, f"catalog://dataset/{path.stem}"
        crs = (card.get("crs") or {}).get("name")
        summary = _flatten({k: v for k, v in card.items()
                            if k in ("name", "description", "spatial", "temporal", "crs",
                                     "format", "units", "update_frequency", "source", "counts")})
        out.append(_chunk(f"{cid}#card", "dataset_card", uri, card.get("name", cid),
                          summary, dataset_id=cid, crs=crs))
        schema = _flatten(card.get("schema") or {})
        if schema:
            out.append(_chunk(f"{cid}#schema", "dataset_schema", uri,
                              f"{card.get('name', cid)} 的字段定义", schema,
                              dataset_id=cid, crs=crs))
        for index, pitfall in enumerate(card.get("known_pitfalls") or [], 1):
            out.append(_chunk(f"{cid}#pitfall{index}", "dataset_pitfall", uri,
                              f"{card.get('name', cid)} 已知坑 #{index}", str(pitfall),
                              dataset_id=cid, crs=crs))

    for name, uri_key in (("poi_scope", "poi_scope"), ("anchor_scope", "anchor_scope")):
        path = KNOWLEDGE / f"{name}.json"
        if not path.exists():
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        uri = f"knowledge://scope/{uri_key}"
        header = _flatten({k: v for k, v in doc.items() if k in ("name", "applies_to", "question", "rule")})
        out.append(_chunk(f"{name}#head", "scope", uri, doc.get("name", name), header))
        for key, value in doc.items():
            if key in ("id", "name", "version", "effective_from", "applies_to", "question", "rule"):
                continue
            out.append(_chunk(f"{name}#{key}", "scope_section", uri, f"{doc.get('name', name)} · {key}",
                              _flatten(value)))

    coords = KNOWLEDGE / "coords" / "systems.json"
    if coords.exists():
        doc = json.loads(coords.read_text(encoding="utf-8"))
        uri = "knowledge://coords/systems"
        out.append(_chunk("coords#head", "scope", uri, doc.get("name", "坐标系"),
                          _flatten({k: v for k, v in doc.items() if k in ("name", "storage_crs")})))
        for key, value in doc.items():
            if key in ("id", "name", "version", "effective_from", "storage_crs"):
                continue
            out.append(_chunk(f"coords#{key}", "scope_section", uri, f"坐标系 · {key}", _flatten(value)))

    aliases = KNOWLEDGE / "categories" / "aliases.json"
    if aliases.exists():
        doc = json.loads(aliases.read_text(encoding="utf-8"))
        uri = "knowledge://categories/aliases"
        out.append(_chunk("aliases#head", "scope", uri, doc.get("name", "类别中文别名"),
                          _flatten({k: v for k, v in doc.items() if k in ("how_to_use", "known_limitations")})))
        for alias, spec in (doc.get("aliases") or {}).items():
            text = _flatten(spec)
            out.append(_chunk(f"alias#{alias}", "alias", uri, f"中文说法「{alias}」对应的 OSM 类别", text))
        for alias, reason in (doc.get("not_available") or {}).items():
            out.append(_chunk(f"unavailable#{alias}", "alias_unavailable", uri,
                              f"「{alias}」为什么查不到", str(reason)))

    # 经验库：只收已确认的 lessons，候选（candidates）不进索引——它们还没写 statement，
    # 索引进去就会和「已验证的结论」混在同一个 top-k 里，检索质量与可信度一起下降
    experience = KNOWLEDGE / "experience" / "lessons.json"
    if experience.exists():
        doc = json.loads(experience.read_text(encoding="utf-8"))
        uri = "knowledge://experience"
        for lesson in doc.get("lessons") or []:
            out.append(_chunk(
                f"experience#{lesson['signature']}", "experience", uri,
                lesson.get("statement", lesson["signature"]),
                _flatten({"为什么": lesson.get("why"), "出处": lesson.get("evidence"),
                          "来源": lesson.get("sources"), "类型": lesson.get("kind")})))
    return out


def _chunk(chunk_id: str, kind: str, source_uri: str, title: str, text: str,
           dataset_id: str | None = None, crs: str | None = None) -> dict:
    body = (text or "").strip()
    return {"chunk_id": chunk_id, "kind": kind, "dataset_id": dataset_id, "crs": crs,
            "source_uri": source_uri, "title": title, "text": body,
            "tokens": tokenize(f"{title}\n{body}")}


def connect(db_path: Path = DB) -> duckdb.DuckDBPyConnection:
    if not Path(db_path).exists():
        raise FileNotFoundError(
            f"知识库索引不存在：{db_path}；先跑 uv run python scripts/build_knowledge_index.py")
    con = duckdb.connect(str(db_path), read_only=True)
    con.execute("INSTALL fts")
    con.execute("LOAD fts")
    con.execute("INSTALL vss")
    con.execute("LOAD vss")
    con.execute("SET hnsw_enable_experimental_persistence = true")
    return con


def build(db_path: Path = DB, *, embedder: Embedder | None = None,
          progress=print) -> dict:
    from geo_catalog.embedding import BgeEmbedder

    embedder = embedder or BgeEmbedder()
    docs = chunks()
    progress(f"语料块 {len(docs)} 条，用 {embedder.name}（{embedder.dim} 维）算向量…")
    vectors = embedder.embed([f"{d['title']}\n{d['text']}" for d in docs])

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_suffix(".building")
    if tmp.exists():
        tmp.unlink()
    con = duckdb.connect(str(tmp))
    try:
        con.execute("INSTALL fts")
        con.execute("LOAD fts")
        con.execute("INSTALL vss")
        con.execute("LOAD vss")
        con.execute("SET hnsw_enable_experimental_persistence = true")
        con.execute(f"""
            CREATE TABLE {TABLE} (
                chunk_id VARCHAR, kind VARCHAR, dataset_id VARCHAR, crs VARCHAR,
                source_uri VARCHAR, title VARCHAR, text VARCHAR, tokens VARCHAR,
                vec FLOAT[{embedder.dim}]
            )""")
        con.executemany(
            f"INSERT INTO {TABLE} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(d["chunk_id"], d["kind"], d["dataset_id"], d["crs"], d["source_uri"],
              d["title"], d["text"], d["tokens"], vec) for d, vec in zip(docs, vectors)])
        con.execute(f"PRAGMA create_fts_index('{TABLE}', 'chunk_id', 'tokens', "
                    f"stemmer='none', stopwords='none', overwrite=1)")
        con.execute(f"CREATE INDEX chunks_hnsw ON {TABLE} USING HNSW (vec) WITH (metric = 'cosine')")
        counts = dict(con.execute(f"SELECT kind, count(*) FROM {TABLE} GROUP BY 1 ORDER BY 1").fetchall())
        con.close()
    except Exception:
        con.close()
        raise
    shutil.move(str(tmp), str(db_path))
    return {"chunks": len(docs), "by_kind": counts, "dim": embedder.dim,
            "model": embedder.name, "db": str(db_path)}


def _filters(kinds: Iterable[str] | None, dataset_id: str | None,
             crs: str | None) -> tuple[str, list]:
    clauses, params = [], []
    if kinds:
        keys = list(kinds)
        clauses.append(f"kind IN ({', '.join('?' * len(keys))})")
        params.extend(keys)
    if dataset_id:
        clauses.append("dataset_id = ?")
        params.append(dataset_id)
    if crs:
        clauses.append("crs = ?")
        params.append(crs)
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


_COLUMNS = ("chunk_id", "kind", "dataset_id", "title", "source_uri", "text")


def _new_hit(row) -> dict:
    return {"chunk_id": row[0], "kind": row[1], "dataset_id": row[2], "title": row[3],
            "source_uri": row[4], "snippet": _snippet(row[5]), "score": 0.0,
            "matched_by": [], "bm25_rank": None, "dense_rank": None,
            "bm25_score": None, "dense_sim": None}


def search(query: str, limit: int = 5, *, kinds: Iterable[str] | None = None,
           dataset_id: str | None = None, crs: str | None = None,
           db_path: Path = DB, embedder: Embedder | None = None,
           min_sim: float = MIN_SIM, explain: bool = True) -> dict:
    """filter-then-rank 混合检索：结构化过滤 → BM25 + 向量 → RRF 融合。"""
    from geo_catalog.embedding import BgeEmbedder

    embedder = embedder or BgeEmbedder()
    con = connect(db_path)
    try:
        where, params = _filters(kinds, dataset_id, crs)
        cols = ", ".join(_COLUMNS)
        bm25 = con.execute(
            f"SELECT {cols}, score FROM (SELECT {cols}, "
            f"fts_main_{TABLE}.match_bm25(chunk_id, ?) AS score FROM {TABLE}{where}) sq "
            f"WHERE score IS NOT NULL ORDER BY score DESC LIMIT {POOL}",
            [tokenize(query), *params]).fetchall()
        vector = embedder.embed([query])[0]
        dense = con.execute(
            f"SELECT {cols}, sim FROM (SELECT {cols}, "
            f"array_cosine_similarity(vec, ?::FLOAT[{embedder.dim}]) AS sim FROM {TABLE}{where}) sq "
            f"ORDER BY sim DESC LIMIT {POOL}", [vector, *params]).fetchall()
        dense_floor = max(min_sim, dense[0][-1] - SIM_MARGIN) if dense else None
        if dense_floor is not None:
            dense = [row for row in dense if row[-1] >= dense_floor]
    finally:
        con.close()

    fused: dict[str, dict] = {}
    for retriever, rows in (("bm25", bm25), ("dense", dense)):
        for rank, row in enumerate(rows, 1):
            hit = fused.setdefault(row[0], _new_hit(row))
            hit["score"] += 1.0 / (RRF_K + rank)
            hit["matched_by"].append(retriever)
            hit[{"bm25": "bm25_rank", "dense": "dense_rank"}[retriever]] = rank
            hit[{"bm25": "bm25_score", "dense": "dense_sim"}[retriever]] = round(float(row[-1]), 4)

    hits = sorted(fused.values(), key=lambda h: (-h["score"], h["chunk_id"]))[:limit]
    for hit in hits:
        hit["score"] = round(hit["score"], 6)
    result = {"query": query, "returned": len(hits), "filters": {
        "kinds": list(kinds) if kinds else None, "dataset_id": dataset_id, "crs": crs},
        "hits": hits}
    if explain:
        result["retrieval"] = {"bm25_candidates": len(bm25), "dense_candidates": len(dense),
                               "dense_floor": round(dense_floor, 4) if dense_floor else None,
                               "fusion": f"RRF(k={RRF_K})", "embedding_model": embedder.name}
    return result


def _snippet(text: str, limit: int = 220) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


def stats(db_path: Path = DB) -> dict:
    con = connect(db_path)
    try:
        by_kind = dict(con.execute(f"SELECT kind, count(*) FROM {TABLE} GROUP BY 1 ORDER BY 1").fetchall())
        total = con.execute(f"SELECT count(*) FROM {TABLE}").fetchone()[0]
        indexes = [row[0] for row in con.execute(
            "SELECT index_name FROM duckdb_indexes() WHERE table_name = ?", [TABLE]).fetchall()]
    finally:
        con.close()
    return {"chunks": total, "by_kind": by_kind, "indexes": indexes, "db": str(db_path)}
