"""Embedding arms with a disk cache.

Each arm is a named config so the ablation grid can refer to "bge-small" and
the README can list exactly what that meant. Open models run locally via
sentence-transformers (MPS on Apple silicon, CUDA elsewhere, CPU fallback).
An API arm (Voyage) plugs into the same interface.

Embeddings are cached under ``data/cache/emb/`` keyed by a hash of the texts,
so re-running the grid with a new top-k or reranker never re-embeds.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

CACHE_DIR = Path("data/cache/emb")


@dataclass(frozen=True)
class EmbeddingArm:
    key: str
    model_name: str
    query_prefix: str = ""
    passage_prefix: str = ""
    max_seq_length: int = 512
    kind: str = "sentence-transformers"   # or "voyage"


ARMS: dict[str, EmbeddingArm] = {
    "bge-small": EmbeddingArm(
        "bge-small", "BAAI/bge-small-en-v1.5",
        query_prefix="Represent this sentence for searching relevant passages: "),
    "bge-base": EmbeddingArm(
        "bge-base", "BAAI/bge-base-en-v1.5",
        query_prefix="Represent this sentence for searching relevant passages: "),
    "e5-base": EmbeddingArm(
        "e5-base", "intfloat/e5-base-v2",
        query_prefix="query: ", passage_prefix="passage: "),
    "minilm": EmbeddingArm(
        "minilm", "sentence-transformers/all-MiniLM-L6-v2", max_seq_length=256),
    # "voyage-3" is added by finetune / api arms later; keep the registry open.
}


def register_arm(arm: EmbeddingArm) -> None:
    ARMS[arm.key] = arm


def _register_local_arms(models_dir: Path = Path("data/models")) -> None:
    """Fine-tuned models saved by finetune/train.py register themselves."""
    for meta in models_dir.glob("*/arm.json"):
        d = json.loads(meta.read_text())
        register_arm(EmbeddingArm(d["key"], d["model_name"], d.get("query_prefix", ""),
                                  d.get("passage_prefix", ""), d.get("max_seq_length", 512)))


_register_local_arms()


def register_hub_arm(key: str, repo_id: str, base: str = "bge-small") -> None:
    """Register a fine-tuned arm served from the Hugging Face Hub (deployed app)
    unless the same key already resolves to local weights."""
    if key in ARMS and Path(ARMS[key].model_name).exists():
        return
    b = ARMS[base]
    register_arm(EmbeddingArm(key, repo_id, b.query_prefix, b.passage_prefix, b.max_seq_length))


class Embedder(Protocol):
    arm: EmbeddingArm

    def encode_passages(self, texts: list[str]) -> np.ndarray: ...
    def encode_queries(self, texts: list[str]) -> np.ndarray: ...


def _texts_hash(texts: list[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class STEmbedder:
    def __init__(self, arm: EmbeddingArm, cache_dir: Path = CACHE_DIR, batch_size: int = 64):
        self.arm = arm
        self.cache_dir = cache_dir / arm.key
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.arm.model_name, device=_device())
            self._model.max_seq_length = self.arm.max_seq_length
        return self._model

    def _encode(self, texts: list[str], prefix: str, tag: str) -> np.ndarray:
        key = f"{tag}-{_texts_hash(texts)}-{prefix!r}"
        path = self.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.npy"
        if path.exists():
            return np.load(path)
        vecs = self.model.encode(
            [prefix + t for t in texts], batch_size=self.batch_size,
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=len(texts) > 500,
        ).astype(np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vecs)
        (path.with_suffix(".json")).write_text(json.dumps({"key": key, "n": len(texts)}))
        return vecs

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.arm.passage_prefix, "passages")

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.arm.query_prefix, "queries")


def get_embedder(key: str) -> Embedder:
    arm = ARMS[key]
    if arm.kind == "sentence-transformers":
        return STEmbedder(arm)
    raise NotImplementedError(f"embedding kind {arm.kind!r} not wired yet")
