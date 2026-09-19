"""Precompute what the deployed app needs so it never chunks or embeds at
startup (Streamlit Community Cloud has no GPU and a cold-start budget):

  data/app_bundle/chunks.parquet      chunk_id, doc_id, section, text, n_tokens
  data/app_bundle/embeddings.npy      float32 [n_chunks, dim], unit vectors
  data/app_bundle/meta.json           config used

Reads configs/app.json for split / chunk config / embedder.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chunking import ChunkConfig, chunk_corpus, default_tokenizer  # noqa: E402
from corpus.qasper import load_qasper  # noqa: E402
from embed import get_embedder  # noqa: E402

OUT = Path("data/app_bundle")


def main() -> None:
    cfg = json.loads(Path("configs/app.json").read_text())
    strat, size, ov = cfg["chunk"].split("-")
    corpus = load_qasper(cfg["split"])
    chunks = chunk_corpus(corpus.documents.values(), ChunkConfig(strat, int(size), int(ov)), default_tokenizer())
    emb = get_embedder(cfg["embedder"])
    vecs = emb.encode_passages([c.text for c in chunks])
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"chunk_id": c.chunk_id, "doc_id": c.doc_id, "section": c.section, "text": c.text,
                   "n_tokens": c.n_tokens} for c in chunks]).to_parquet(OUT / "chunks.parquet", index=False)
    np.save(OUT / "embeddings.npy", vecs.astype(np.float32))
    (OUT / "meta.json").write_text(json.dumps({**{k: v for k, v in cfg.items() if not k.startswith("_")},
                                               "n_chunks": len(chunks), "dim": int(vecs.shape[1]),
                                               "model_name": emb.arm.model_name}, indent=1))
    print(f"bundle: {len(chunks)} chunks x {vecs.shape[1]} dims -> {OUT} "
          f"({sum(p.stat().st_size for p in OUT.iterdir()) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
