"""One entry point for every retrieval configuration.

A ``RetrievalConfig`` names one cell of the ablation grid. ``Retriever`` owns
the chunks for one chunk config plus the dense and BM25 indexes over them, and
answers ``retrieve(question)`` for any combination of hybrid / rerank / top-k.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from chunking import Chunk, ChunkConfig
from embed import Embedder
from index import BM25Index, DenseIndex, Ranked, rrf
from rerank import Reranker


@dataclass(frozen=True)
class RetrievalConfig:
    chunk: ChunkConfig = ChunkConfig()
    embedder: str = "bge-small"
    hybrid: bool = False          # RRF-fuse BM25 with dense
    rerank: bool = False          # cross-encoder over the candidate pool
    reranker: str = "minilm-ce"   # key in rerank.RERANKERS (only used if rerank)
    top_k: int = 5                # what the generator sees / what metrics use
    candidate_k: int = 50         # pool size before fusion / reranking
    # "doc":        retrieve within the question's own paper (chat-with-this-PDF)
    # "open":       whole corpus, question text as-is (ill-posed for QASPER: the
    #               question was written about one paper but never names it)
    # "open-titled": whole corpus, question prefixed with the paper title, i.e.
    #               the user names the document they are asking about
    scope: str = "doc"

    @property
    def name(self) -> str:
        bits = [self.chunk.name, self.embedder,
                "hybrid" if self.hybrid else "dense",
                f"rerank-{self.reranker}" if self.rerank else "norerank", self.scope]
        return "_".join(bits)

    def with_top_k(self, k: int) -> "RetrievalConfig":
        return replace(self, top_k=k)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int


class Retriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder,
                 reranker: Reranker | None = None, build_bm25: bool = True):
        self.chunks = chunks
        self.embedder = embedder
        self.reranker = reranker
        texts = [c.text for c in chunks]
        doc_ids = [c.doc_id for c in chunks]
        self.dense = DenseIndex(embedder.encode_passages(texts), doc_ids)
        self.bm25 = BM25Index(texts, doc_ids) if build_bm25 else None

    def candidates(self, q_vec: np.ndarray, question: str, cfg: RetrievalConfig,
                   doc_id: str | None) -> Ranked:
        scope = doc_id if cfg.scope == "doc" else None
        if cfg.scope not in ("doc", "open", "open-titled"):
            raise ValueError(f"unknown scope {cfg.scope!r}")
        k = cfg.candidate_k if (cfg.hybrid or cfg.rerank) else cfg.top_k
        dense = self.dense.search(q_vec, k, scope)
        if not cfg.hybrid:
            return dense
        assert self.bm25 is not None, "hybrid requires a BM25 index"
        lex = self.bm25.search(question, k, scope)
        return rrf([dense.idx, lex.idx], top=k)

    def retrieve(self, question_id: str, question: str, doc_id: str | None,
                 cfg: RetrievalConfig, q_vec: np.ndarray | None = None) -> list[Hit]:
        if q_vec is None:
            q_vec = self.embedder.encode_queries([question])[0]
        ranked = self.candidates(q_vec, question, cfg, doc_id)
        idx, scores = ranked.idx, ranked.score
        if cfg.rerank:
            assert self.reranker is not None, "rerank requires a Reranker"
            cids = [self.chunks[i].chunk_id for i in idx]
            texts = [self.chunks[i].text for i in idx]
            scores = self.reranker.score(question_id, question, cids, texts)
            order = np.argsort(-scores)
            idx, scores = idx[order], scores[order]
        idx, scores = idx[:cfg.top_k], scores[:cfg.top_k]
        return [Hit(self.chunks[int(i)], float(s), r) for r, (i, s) in enumerate(zip(idx, scores))]
