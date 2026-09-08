"""Contrastive fine-tuning of bge-small on the train-split triples.

Loss: MultipleNegativesRankingLoss with an explicit hard negative per anchor
(in-batch negatives + the BM25 hard negative). This is the standard recipe
for adapting a bi-encoder to a domain; the point of the arm is to measure
what it buys on held-out papers, not to invent a new method.

The bge query prefix is applied to anchors so training matches inference.
Output: data/models/<name>/ , registered as embedding arm "<name>".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from embed import ARMS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="bge-small")
    ap.add_argument("--name", default="bge-small-ft")
    ap.add_argument("--pairs", default="data/processed/finetune_pairs.jsonl")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-seq-length", type=int, default=256,
                    help="training-time truncation; paragraphs are median 91 / p90 204 tokens")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)

    arm = ARMS[args.base]
    rows = [json.loads(l) for l in Path(args.pairs).read_text().splitlines()]
    ds = Dataset.from_dict({
        "anchor": [arm.query_prefix + r["anchor"] for r in rows],
        "positive": [arm.passage_prefix + r["positive"] for r in rows],
        "negative": [arm.passage_prefix + r["negative"] for r in rows],
    }).shuffle(seed=args.seed)
    print(f"{len(ds)} triples; base={arm.model_name}")

    model = SentenceTransformer(arm.model_name)
    model.max_seq_length = args.max_seq_length
    loss = losses.MultipleNegativesRankingLoss(model)
    out_dir = Path("data/models") / args.name
    targs = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        fp16=False, bf16=False,
        logging_steps=50,
        save_strategy="no",
        seed=args.seed,
        report_to="none",
    )
    trainer = SentenceTransformerTrainer(model=model, args=targs, train_dataset=ds, loss=loss)
    trainer.train()
    model.max_seq_length = arm.max_seq_length   # inference uses the full window
    model.save(str(out_dir))
    (out_dir / "arm.json").write_text(json.dumps({
        "key": args.name, "model_name": str(out_dir), "query_prefix": arm.query_prefix,
        "passage_prefix": arm.passage_prefix, "max_seq_length": arm.max_seq_length,
        "base": args.base, "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "n_triples": len(ds), "seed": args.seed, "train_max_seq_length": args.max_seq_length}))
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
