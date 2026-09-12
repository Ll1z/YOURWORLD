"""构建知识库检索索引：data/processed/knowledge.duckdb。

把四张数据卡、两份口径文件、类别别名表与坐标系定义切成语料块，用本地 bge 算向量，
建 BM25（DuckDB fts）+ HNSW（DuckDB vss）双索引。

用法：uv run python scripts/build_knowledge_index.py
首次运行需要联网下载模型（默认走 hf-mirror），之后离线可用。
数据卡、口径文件改过之后要重跑，否则检索到的是旧内容。
"""

from __future__ import annotations

import sys
import time

from geo_catalog import index


def main() -> int:
    started = time.perf_counter()
    report = index.build()
    stats = index.stats()
    print(f"索引已写入 {report['db']}")
    print(f"  语料块 {report['chunks']} 条，向量 {report['dim']} 维（{report['model']}）")
    for kind, count in stats["by_kind"].items():
        print(f"    {kind:20s} {count}")
    print(f"  索引：{', '.join(stats['indexes']) or '（无）'}")
    print(f"  耗时 {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())