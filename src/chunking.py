"""Chunking strategies.

Two strategies, each an ablation arm:

* ``fixed``     - sliding window of ``size`` tokens with ``overlap`` tokens,
                  over the document's paragraphs concatenated in order. Ignores
                  structure; may split a paragraph across chunks.
* ``paragraph`` - packs whole paragraphs greedily up to ``size`` tokens; never
                  splits a paragraph unless a single paragraph exceeds ``size``,
                  in which case that paragraph alone is windowed. Respects
                  structure; no overlap by construction.

Token boundaries come from the BERT WordPiece tokenizer shared by the bge / e5
embedding families, so ``size`` means the same thing for every open embedding
arm. Only the *character offsets* of tokens are used: chunk text is always a
slice of the original paragraph text, never re-joined WordPieces, so casing
and punctuation survive for BM25, the API embedding arm and the generator.

Every chunk records, for each paragraph it touches, how many of that
paragraph's tokens it contains. ``eval.gold`` uses that to decide whether a
chunk "contains" a gold paragraph under a coverage threshold, so retrieval
recall is exact rather than estimated by string overlap.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Iterable

from corpus.base import Document, Paragraph

TOKENIZER_NAME = "BAAI/bge-small-en-v1.5"  # BERT uncased WordPiece vocab

# A tokenizer returns (start, end) character offsets for each token.
Span = tuple[int, int]
Tokenizer = Callable[[str], list[Span]]


@lru_cache(maxsize=1)
def default_tokenizer() -> Tokenizer:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    tok.model_max_length = 10**9  # we only want offsets; silence the length warning

    def offsets(text: str) -> list[Span]:
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return [(s, e) for s, e in enc["offset_mapping"] if e > s]

    return offsets


def whitespace_tokenizer(text: str) -> list[Span]:
    """Deterministic stand-in for tests; no model download."""
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


@dataclass(frozen=True)
class ChunkConfig:
    strategy: str = "fixed"      # "fixed" | "paragraph"
    size: int = 256              # max tokens per chunk
    overlap: int = 0             # tokens shared with previous chunk (fixed only)

    def __post_init__(self) -> None:
        if self.strategy not in ("fixed", "paragraph"):
            raise ValueError(f"unknown strategy {self.strategy!r}")
        if self.size <= 0:
            raise ValueError("size must be positive")
        if not 0 <= self.overlap < self.size:
            raise ValueError("overlap must satisfy 0 <= overlap < size")

    @property
    def name(self) -> str:
        return f"{self.strategy}-{self.size}-{self.overlap}"


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    n_tokens: int
    # para_id -> number of that paragraph's tokens inside this chunk
    para_token_counts: dict[str, int] = field(default_factory=dict)
    section: str = ""

    def coverage(self, para: Paragraph, para_n_tokens: int) -> float:
        """Fraction of ``para``'s tokens contained in this chunk."""
        if para_n_tokens == 0:
            return 0.0
        return self.para_token_counts.get(para.para_id, 0) / para_n_tokens


# One token of the flattened document: which paragraph it belongs to and where.
_Tok = tuple[Paragraph, int, int]


def _window_to_chunk(doc: Document, window: list[_Tok], idx: int) -> Chunk:
    """Build a chunk from consecutive tokens, slicing original paragraph text."""
    parts: list[str] = []
    counts: dict[str, int] = {}
    cur: Paragraph | None = None
    start = end = 0
    for para, s, e in window:
        counts[para.para_id] = counts.get(para.para_id, 0) + 1
        if para is not cur:
            if cur is not None:
                parts.append(cur.text[start:end])
            cur, start = para, s
        end = e
    assert cur is not None
    parts.append(cur.text[start:end])
    return Chunk(
        chunk_id=f"{doc.doc_id}#{idx}",
        doc_id=doc.doc_id,
        text="\n\n".join(parts),
        n_tokens=len(window),
        para_token_counts=counts,
        section=window[0][0].section,
    )


def _windows(stream: list[_Tok], size: int, overlap: int) -> Iterable[list[_Tok]]:
    step = size - overlap
    i = 0
    while i < len(stream):
        yield stream[i:i + size]
        if i + size >= len(stream):
            break
        i += step


def _flatten(doc: Document, tokenizer: Tokenizer) -> list[list[_Tok]]:
    """Per-paragraph token lists (empty paragraphs dropped)."""
    out = []
    for p in doc.paragraphs:
        spans = tokenizer(p.text)
        if spans:
            out.append([(p, s, e) for s, e in spans])
    return out


def chunk_document(doc: Document, cfg: ChunkConfig, tokenizer: Tokenizer | None = None) -> list[Chunk]:
    tokenizer = tokenizer or default_tokenizer()
    per_para = _flatten(doc, tokenizer)
    if not per_para:
        return []
    chunks: list[Chunk] = []

    if cfg.strategy == "fixed":
        stream = [t for toks in per_para for t in toks]
        for w in _windows(stream, cfg.size, cfg.overlap):
            chunks.append(_window_to_chunk(doc, w, len(chunks)))
        return chunks

    # paragraph strategy: greedy packing, over-long paragraphs windowed alone
    buf: list[_Tok] = []
    for toks in per_para:
        if len(toks) > cfg.size:
            if buf:
                chunks.append(_window_to_chunk(doc, buf, len(chunks)))
                buf = []
            for w in _windows(toks, cfg.size, cfg.overlap):
                chunks.append(_window_to_chunk(doc, w, len(chunks)))
            continue
        if len(buf) + len(toks) > cfg.size:
            chunks.append(_window_to_chunk(doc, buf, len(chunks)))
            buf = []
        buf.extend(toks)
    if buf:
        chunks.append(_window_to_chunk(doc, buf, len(chunks)))
    return chunks


def chunk_corpus(docs: Iterable[Document], cfg: ChunkConfig,
                 tokenizer: Tokenizer | None = None) -> list[Chunk]:
    tokenizer = tokenizer or default_tokenizer()
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, cfg, tokenizer))
    return out


def paragraph_token_counts(docs: Iterable[Document],
                           tokenizer: Tokenizer | None = None) -> dict[str, int]:
    """para_id -> token count, needed to turn chunk counts into coverage fractions."""
    tokenizer = tokenizer or default_tokenizer()
    return {p.para_id: len(tokenizer(p.text)) for d in docs for p in d.paragraphs}
