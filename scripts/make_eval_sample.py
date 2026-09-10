"""Fixed, stratified sample of test questions for the generation/judge runs.

Generation and judging cost API money; retrieval does not. The full test split
(1,451 questions) would cost ~$60 to generate and judge across six arms, so
the paid runs use a fixed sample: ALL majority-unanswerable questions (so
abstention/hallucination on unanswerables is measured on the full set) plus a
seeded random sample of answerable questions with gold paragraphs.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from corpus.qasper import load_qasper  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-answerable", type=int, default=302)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    c = load_qasper(args.split)
    unans = [q.question_id for q in c.questions if q.unanswerable]
    ans = sorted(q.question_id for q in c.questions if not q.unanswerable and q.gold_para_ids)
    rng = random.Random(args.seed)
    picked = rng.sample(ans, min(args.n_answerable, len(ans)))
    out = {"split": args.split, "seed": args.seed, "n_unanswerable": len(unans), "n_answerable": len(picked),
           "question_ids": sorted(unans + picked)}
    path = Path("data/processed") / f"{args.split}_sample_{len(out['question_ids'])}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}: {len(unans)} unanswerable + {len(picked)} answerable")


if __name__ == "__main__":
    main()
