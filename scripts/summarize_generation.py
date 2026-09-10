"""Generation / answer-quality ablation table with paired CIs vs the primary arm.

Reads reports/generation/<split>/<model>/<name>.json and .per_question.parquet.
Writes reports/generation/<split>/summary.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eval.stats import bootstrap_mean, paired_bootstrap_diff  # noqa: E402

COLS = ["abstain_rate", "token_f1", "judge_correct", "judge_correct_or_partial", "judge_faithful",
        "nli_faithful", "judge_citation_precision", "hallucination_rate", "abstain_precision", "abstain_recall",
        "retrieval_hit", "mean_input_tokens", "cost_usd_total", "judge_cost_usd"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--primary", default=None, help="run name to compare others against")
    args = ap.parse_args()
    rep = Path("reports/generation") / args.split
    runs = {}
    for js in sorted(rep.glob("*/*.json")):
        d = json.loads(js.read_text())
        d["run"] = f"{d['model']}/{d['name']}"
        d["_pq"] = js.with_name(js.stem + ".per_question.parquet")
        runs[d["run"]] = d
    if not runs:
        print("no generation reports found")
        return
    table = pd.DataFrame([{k: v for k, v in d.items() if not k.startswith("_") and k != "decomposition"} for d in runs.values()])
    keep = ["run", "mode", "n"] + [c for c in COLS if c in table]
    md = [f"# Answer-quality ablation ({args.split})\n", table[keep].to_markdown(index=False, floatfmt=".3f")]

    # error decomposition per run
    md.append("\n## Retrieval x correctness decomposition (answerable, judged)\n")
    rows = []
    for d in runs.values():
        dec = d.get("decomposition")
        if dec:
            rows.append({"run": d["run"], **dec})
    if rows:
        md.append(pd.DataFrame(rows).to_markdown(index=False))

    # paired comparisons vs primary
    primary = args.primary or next(iter(runs))
    if primary in runs:
        md.append(f"\n## Paired differences vs `{primary}` (95% bootstrap CI over the same questions)\n")
        base = pd.read_parquet(runs[primary]["_pq"]).set_index("question_id")
        rows = []
        for name, d in runs.items():
            if name == primary:
                continue
            other = pd.read_parquet(d["_pq"]).set_index("question_id")
            idx = base.index.intersection(other.index)
            row = {"run": name, "n": len(idx)}
            for m in ["judge_correct", "token_f1", "judge_faithful", "nli_faithful", "hallucinated"]:
                if m in base and m in other:
                    a = pd.to_numeric(other.loc[idx, m], errors="coerce")
                    b = pd.to_numeric(base.loc[idx, m], errors="coerce")
                    ok = a.notna() & b.notna()
                    if ok.sum() > 10:
                        ci = paired_bootstrap_diff(a[ok].values, b[ok].values)
                        row[m] = f"{ci.mean:+.3f} [{ci.lo:+.3f}, {ci.hi:+.3f}]"
            rows.append(row)
        md.append(pd.DataFrame(rows).to_markdown(index=False))

    # CIs on the headline numbers of the primary
    if primary in runs:
        base = pd.read_parquet(runs[primary]["_pq"])
        md.append(f"\n## Headline CIs for `{primary}`\n")
        for m in ["judge_correct", "judge_faithful", "nli_faithful", "hallucinated", "abstain"]:
            if m in base:
                x = pd.to_numeric(base[m], errors="coerce").dropna()
                if len(x):
                    md.append(f"- {m}: {bootstrap_mean(x.values)}")
    (rep / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
