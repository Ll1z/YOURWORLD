"""文本向量化：本地 bge（ONNX）后端。

选型（HANDOFF 第 8 节的待定项）：
- `fastembed` + `BAAI/bge-small-zh-v1.5`，512 维，ONNX Runtime 跑 CPU。
  语料是中文，bge 是 AGENTS.md 里定下的方向；走 ONNX 就不用装 torch，
  整套依赖约 200 MB、模型 90 MB，个人笔记本上够快。
- 升级路径：换 bge-m3（更强、多语、1024 维）或改用 GPU，只改本文件的 MODEL_NAME / DIM
  再重建索引——索引与检索都只认 Embedder 协议。

模型缓存固定在 data/models/fastembed（不入库），不放系统临时目录，
免得被清理后每次重建都重新下载。首次需要联网；huggingface.co 在国内不通，
默认走 hf-mirror，可用 HF_ENDPOINT 覆盖。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Protocol

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
DIM = 512
CACHE_DIR = ROOT / "data" / "models" / "fastembed"
BATCH = 32


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _batched(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


class BgeEmbedder:
    """懒加载：只有真要算向量时才把模型拉起来，MCP Server 启动不必等它。"""

    def __init__(self, model_name: str = MODEL_NAME, dim: int = DIM,
                 cache_dir: Path = CACHE_DIR):
        self.name = model_name
        self.dim = dim
        self.cache_dir = Path(cache_dir)
        self._model = None

    def _ensure(self):
        if self._model is None:
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            from fastembed import TextEmbedding

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._model = TextEmbedding(model_name=self.name, cache_dir=str(self.cache_dir))
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure()
        vectors: list[list[float]] = []
        for batch in _batched(texts, BATCH):
            vectors.extend([list(map(float, v)) for v in model.embed(batch)])
        return vectors