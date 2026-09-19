"""Upload the fine-tuned embedding model and the app bundle to one Hugging
Face model repo, so the deployed app can fetch both. Uses the token stored by
`hf auth login` (never read or printed here).

    python scripts/publish_hf.py --repo <hf-username>/bge-small-qasper-ft
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

MODEL_DIR = Path("data/models/bge-small-ft")
BUNDLE_DIR = Path("data/app_bundle")

CARD = """---
license: mit
base_model: BAAI/bge-small-en-v1.5
library_name: sentence-transformers
tags: [sentence-transformers, retrieval, qasper, fine-tuned]
---

# bge-small-en-v1.5 fine-tuned on QASPER

`BAAI/bge-small-en-v1.5` fine-tuned for two epochs with MultipleNegativesRankingLoss on
3,560 (question, gold evidence paragraph, BM25 hard negative) triples from the QASPER
**train** split. On the QASPER test split (papers disjoint from training) it improves
recall at a 1,024-token context budget by +0.149 [+0.127, +0.171] over the base model
with paragraph-packed 256-token chunks. Query prefix: `Represent this sentence for
searching relevant passages: `.

The `app_bundle/` folder holds the precomputed test-split chunks and embeddings used by
the demo app. Project, evaluation harness and full results:
https://github.com/GFS418/rag-eval-harness
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    args = ap.parse_args()
    api = HfApi()
    who = api.whoami()["name"]
    print(f"logged in as {who}; target repo {args.repo}")
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    (MODEL_DIR / "README.md").write_text(CARD)
    api.upload_folder(folder_path=str(MODEL_DIR), repo_id=args.repo, repo_type="model",
                      ignore_patterns=["checkpoints/*", "arm.json"], commit_message="fine-tuned weights")
    api.upload_folder(folder_path=str(BUNDLE_DIR), repo_id=args.repo, repo_type="model",
                      path_in_repo="app_bundle", commit_message="app bundle: test-split chunks + embeddings")
    meta = json.loads((BUNDLE_DIR / "meta.json").read_text())
    print(f"uploaded model + bundle ({meta['n_chunks']} chunks) to https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
