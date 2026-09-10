"""Paired comparison of two levels of one axis across every config where both
exist, e.g. the fine-tuned embedder vs its base, or rerank vs no rerank.

    python scripts/compare_arms.py --split test --axis embedder --a bge-small-ft --b bge-small
    python scripts/compare_arms.py --split dev  --axis rerank --a rerank-minilm-ce --b norerank

Writes reports/retrieval/<split>/compare_<axis>_<a>_vs_<b>.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eval.stats import paired_bootstrap_diff  # noqa: E402

AXES = ["chunk", "embedder", "lexical", "rerank", "scope"]
METRICS = ["recall@1024tok", "recall@5", "hit@5", "mrr"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--axis", required=True, choices=AXES)
    ap.add_argument("--a", required=True, help="treatment level")
    ap.add_argument("--b", required=True, help="baseline level")
    args = ap.parse_args()
    rep = Path("reports/retrieval") / args.split
    configs = {p.name.replace(".per_question.parquet", ""): p for p in rep.glob("*.per_question.parquet")}
    ai = AXES.index(args.axis)
    rows = []
    for name, path in sorted(configs.items()):
        parts = name.split("_")
        if parts[ai] != args.a:
            continue
        base_parts = list(parts)
        base_parts[ai] = args.b
        base = "_".join(base_parts)
        if base not in configs:
            continue
        a = pd.read_parquet(path).set_index("question_id")
        b = pd.read_parquet(configs[base]).set_index("question_id")
        idx = a.index.intersection(b.index)
        row = {"config (other axes)": "_".join(p for i, p in enumerate(parts) if i != ai), "n": len(idx)}
        for m in METRICS:
            if m not in a or m not in b:
                continue
            ci = paired_bootstrap_diff(a.loc[idx, m].values, b.loc[idx, m].values)
            row[f"{m} base"] = b.loc[idx, m].mean()
            row[f"{m} diff"] = f"{ci.mean:+.3f} [{ci.lo:+.3f}, {ci.hi:+.3f}]"
        rows.append(row)
    if not rows:
        print("no matching config pairs")
        return
    df = pd.DataFrame(rows)
    md = (f"# {args.axis}: {args.a} vs {args.b} ({args.split})\n\nPaired bootstrap over questions; "
          f"diff = {args.a} minus {args.b}, 95% CI.\n\n" + df.to_markdown(index=False, floatfmt=".3f"))
    (rep / f"compare_{args.axis}_{args.a}_vs_{args.b}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
