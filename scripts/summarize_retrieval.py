"""Turn the per-config retrieval results into an ablation table with CIs.

Reports, for one split:
  * a full table (one row per config) sorted by recall@5, with bootstrap CIs
  * marginal effects per axis: for every pair of configs that differ ONLY in
    that axis, the paired bootstrap of the difference, pooled across the
    other settings (so "hybrid vs dense" is one number with a CI, not 72)

Writes reports/retrieval/<split>/ablation_table.md and ablation_table.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eval.stats import bootstrap_mean, paired_bootstrap_diff  # noqa: E402

AXES = ["chunk", "embedder", "lexical", "rerank", "scope"]


def parse_name(name: str) -> dict:
    chunk, emb, lex, rr, scope = name.split("_")
    return {"config": name, "chunk": chunk, "embedder": emb, "lexical": lex, "rerank": rr, "scope": scope}


def load(split: str, metric: str):
    rep = Path("reports/retrieval") / split
    rows, perq = [], {}
    for js in sorted(rep.glob("*.json")):
        d = json.loads(js.read_text())
        pq = pd.read_parquet(rep / f"{d['config']}.per_question.parquet").set_index("question_id")
        perq[d["config"]] = pq
        ci = bootstrap_mean(pq[metric].values)
        rows.append({**parse_name(d["config"]), "n": d["n_questions"], "n_chunks": d["n_chunks"],
                     "ms_per_query": d["ms_per_query"], "recall@1": d["recall@1"], "recall@5": d["recall@5"],
                     "recall@10": d["recall@10"], "hit@5": d["hit@5"], "mrr": d["mrr"], "ndcg@10": d["ndcg@10"],
                     "recall@512tok": d.get("recall@512tok"), "recall@1024tok": d.get("recall@1024tok"),
                     "recall@2048tok": d.get("recall@2048tok"),
                     f"{metric}_lo": ci.lo, f"{metric}_hi": ci.hi})
    return pd.DataFrame(rows), perq


def marginal_effects(table: pd.DataFrame, perq: dict, metric: str) -> pd.DataFrame:
    out = []
    for axis in AXES:
        levels = sorted(table[axis].unique())
        if len(levels) < 2:
            continue
        others = [a for a in AXES if a != axis]
        base = levels[0]
        for lvl in levels[1:]:
            diffs = []
            for _, grp in table.groupby(others):
                a = grp[grp[axis] == lvl]
                b = grp[grp[axis] == base]
                if len(a) == 1 and len(b) == 1:
                    pa, pb = perq[a.iloc[0]["config"]], perq[b.iloc[0]["config"]]
                    common = pa.index.intersection(pb.index)
                    diffs.append(pa.loc[common, metric].values - pb.loc[common, metric].values)
            if not diffs:
                continue
            d = np.concatenate(diffs)
            ci = bootstrap_mean(d)
            out.append({"axis": axis, "level": lvl, "vs": base, "n_pairs": len(diffs),
                        "n_questions": len(d), f"mean_diff_{metric}": ci.mean, "lo": ci.lo, "hi": ci.hi,
                        "significant": (ci.lo > 0) or (ci.hi < 0)})
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--metric", default="recall@5")
    args = ap.parse_args()
    table, perq = load(args.split, args.metric)
    table = table.sort_values(args.metric, ascending=False)
    rep = Path("reports/retrieval") / args.split
    table.to_csv(rep / "ablation_table.csv", index=False)
    eff = marginal_effects(table, perq, args.metric)
    eff.to_csv(rep / "marginal_effects.csv", index=False)

    def fmt(df, cols):
        return df[cols].to_markdown(index=False, floatfmt=".3f")

    md = [f"# Retrieval ablation ({args.split}), sorted by {args.metric}\n",
          f"n = {table['n'].iloc[0]} answerable questions with gold chunks; 95% bootstrap CIs.\n",
          "## Top 15 configs\n",
          fmt(table.head(15), ["chunk", "embedder", "lexical", "scope", "recall@1", "recall@5",
                               f"{args.metric}_lo", f"{args.metric}_hi", "recall@10", "mrr",
                               "recall@512tok", "recall@1024tok", "recall@2048tok", "ms_per_query"]),
          "\n## Marginal effect of each axis (paired over all other settings)\n",
          fmt(eff, ["axis", "level", "vs", "n_pairs", "n_questions", f"mean_diff_{args.metric}", "lo", "hi", "significant"]),
          "\n## Best config per scope\n",
          fmt(table.sort_values(args.metric, ascending=False).groupby("scope").head(3),
              ["scope", "chunk", "embedder", "lexical", "recall@5", f"{args.metric}_lo", f"{args.metric}_hi", "mrr", "recall@1024tok"])]
    (rep / f"ablation_table_{args.metric.replace('@', '_')}.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
