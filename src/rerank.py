"""Cross-encoder reranker with a (question, chunk) score cache.

Reranking is the slow arm (one forward pass per candidate pair), so scores are
cached per (question_id, chunk_id, model) and shared across every config that
produces the same candidates.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CACHE_DIR = Path("data/cache/rerank")
# Two sizes: the MiniLM cross-encoder (22M params) is ~10x faster than
# bge-reranker-base (278M) and is the grid default; the larger model is a
# comparison arm on the shortlisted configs.
RERANKERS = {
    "minilm-ce": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "bge-reranker-base": "BAAI/bge-reranker-base",
}
DEFAULT_MODEL = "minilm-ce"


class Reranker:
    def __init__(self, key: str = DEFAULT_MODEL, cache_dir: Path = CACHE_DIR,
                 batch_size: int = 64, max_length: int = 512):
        self.key = key
        self.model_name = RERANKERS.get(key, key)
        self.max_length = max_length
        self.batch_size = batch_size
        self.cache_path = cache_dir / (self.model_name.replace("/", "__") + ".jsonl")
        self._model = None
        self._cache: dict[str, float] = {}
        self._dirty = 0
        if self.cache_path.exists():
            with self.cache_path.open() as f:
                for line in f:
                    k, v = json.loads(line)
                    self._cache[k] = v

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            from embed import _device

            self._model = CrossEncoder(self.model_name, device=_device(), max_length=self.max_length)
        return self._model

    def score(self, question_id: str, question: str, chunk_ids: list[str],
              chunk_texts: list[str]) -> np.ndarray:
        keys = [f"{question_id}\t{cid}" for cid in chunk_ids]
        missing = [i for i, k in enumerate(keys) if k not in self._cache]
        if missing:
            pairs = [(question, chunk_texts[i]) for i in missing]
            scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
            for i, s in zip(missing, scores):
                self._cache[keys[i]] = float(s)
            self._dirty += len(missing)
            if self._dirty >= 5000:
                self.flush()
        return np.array([self._cache[k] for k in keys], dtype=np.float32)

    def flush(self) -> None:
        if not self._dirty:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("w") as f:
            for k, v in self._cache.items():
                f.write(json.dumps([k, v]) + "\n")
        self._dirty = 0
