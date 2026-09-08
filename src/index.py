"""Dense (exact cosine), lexical (BM25) indexes and rank fusion.

The corpus is tens of thousands of chunks, so the dense index is an exact
inner-product search over unit vectors (one numpy matmul, ~1 ms per query).
Approximate search would add a recall/latency knob with nothing to trade at
this scale. FAISS's flat index computes the identical result; it was dropped
because faiss-cpu and PyTorch each bundle their own OpenMP runtime on macOS
and the duplicate crashes the process (OMP Error #15) mid-evaluation.

Both indexes support *scoped* search restricted to one document's chunks,
which models the "chat with this PDF" product setting, alongside *open*
search over the whole corpus.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Ranked:
    idx: np.ndarray      # chunk row indices, best first
    score: np.ndarray


class DenseIndex:
    def __init__(self, vectors: np.ndarray, doc_ids: list[str]):
        assert vectors.dtype == np.float32
        self.vectors = np.ascontiguousarray(vectors)
        self.doc_ids = np.asarray(doc_ids)
        self._rows_by_doc: dict[str, np.ndarray] = {}

    def rows_for_doc(self, doc_id: str) -> np.ndarray:
        if doc_id not in self._rows_by_doc:
            self._rows_by_doc[doc_id] = np.flatnonzero(self.doc_ids == doc_id)
        return self._rows_by_doc[doc_id]

    def search(self, q: np.ndarray, k: int, doc_id: str | None = None) -> Ranked:
        q = q.astype(np.float32).reshape(-1)
        if doc_id is None:
            s = self.vectors @ q
            k = min(k, len(s))
            top = np.argpartition(-s, k - 1)[:k]
            order = top[np.argsort(-s[top])]
            return Ranked(order, s[order])
        rows = self.rows_for_doc(doc_id)
        s = self.vectors[rows] @ q
        order = np.argsort(-s)[:k]
        return Ranked(rows[order], s[order])


class BM25Index:
    def __init__(self, texts: list[str], doc_ids: list[str]):
        import bm25s

        self._bm25s = bm25s
        self.doc_ids = np.asarray(doc_ids)
        self._rows_by_doc: dict[str, np.ndarray] = {}
        tokens = bm25s.tokenize(texts, stopwords="en", show_progress=False)
        self.retriever = bm25s.BM25()
        self.retriever.index(tokens, show_progress=False)

    def rows_for_doc(self, doc_id: str) -> np.ndarray:
        if doc_id not in self._rows_by_doc:
            self._rows_by_doc[doc_id] = np.flatnonzero(self.doc_ids == doc_id)
        return self._rows_by_doc[doc_id]

    def search(self, query: str, k: int, doc_id: str | None = None) -> Ranked:
        toks = self._bm25s.tokenize(query, stopwords="en", return_ids=False, show_progress=False)[0]
        scores = self.retriever.get_scores(toks)
        if doc_id is None:
            order = np.argsort(-scores)[:k]
            return Ranked(order, scores[order])
        rows = self.rows_for_doc(doc_id)
        s = scores[rows]
        order = np.argsort(-s)[:k]
        return Ranked(rows[order], s[order])


def rrf(rankings: list[np.ndarray], k: int = 60, top: int | None = None) -> Ranked:
    """Reciprocal rank fusion (Cormack et al., 2009): score = sum 1/(k+rank)."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, row in enumerate(ranking):
            fused[int(row)] = fused.get(int(row), 0.0) + 1.0 / (k + rank + 1)
    items = sorted(fused.items(), key=lambda x: -x[1])
    if top is not None:
        items = items[:top]
    return Ranked(np.array([i for i, _ in items], dtype=np.int64),
                  np.array([s for _, s in items], dtype=np.float32))
