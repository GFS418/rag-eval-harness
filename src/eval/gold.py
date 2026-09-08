"""Map paragraph-level gold evidence to chunk-level relevance.

A chunk is *relevant* to a question if it contains at least ``threshold`` of
the tokens of some gold paragraph. Under the paragraph strategy every gold
paragraph sits whole inside one chunk, so the threshold is moot; under fixed
windows a paragraph can be split, and the threshold decides whether a chunk
with half of it "counts". The threshold is a reported sensitivity axis.
"""
from __future__ import annotations

from collections import defaultdict

from chunking import Chunk
from corpus.base import Question


def gold_chunk_ids(question: Question, chunks_by_doc: dict[str, list[Chunk]],
                   para_tokens: dict[str, int], threshold: float = 0.5) -> frozenset[str]:
    if not question.gold_para_ids:
        return frozenset()
    out = set()
    for c in chunks_by_doc.get(question.doc_id, []):
        for pid in question.gold_para_ids:
            n = para_tokens.get(pid, 0)
            if n and c.para_token_counts.get(pid, 0) / n >= threshold:
                out.add(c.chunk_id)
                break
    return frozenset(out)


def group_chunks(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        by_doc[c.doc_id].append(c)
    return dict(by_doc)
