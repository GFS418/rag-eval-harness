"""Agreement between human labels and the automatic judges (Opus 5, NLI).

Reports Cohen's kappa and raw agreement with bootstrap CIs for:
  correctness   human {correct,partial,incorrect} vs judge verdict (3-class, and
                binary correct-vs-not)
  faithfulness  human faithful vs judge faithful, and vs NLI faithful
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eval.stats import bootstrap_mean  # noqa: E402


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    labels = sorted(set(a) | set(b))
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    n = m.sum()
    po = np.trace(m) / n
    pe = (m.sum(0) * m.sum(1)).sum() / n**2
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def kappa_ci(a, b, n_boot=2000, seed=0):
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    ks = [kappa(a[i], b[i]) for i in rng.integers(0, len(a), size=(n_boot, len(a)))]
    return kappa(a, b), float(np.quantile(ks, 0.025)), float(np.quantile(ks, 0.975))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="reports/human_labels.csv")
    ap.add_argument("--judged", required=True, help="reports/generation/.../<name>.per_question.parquet")
    args = ap.parse_args()
    h = pd.read_csv(args.labels)
    j = pd.read_parquet(args.judged)
    run = Path(args.judged).name.replace(".per_question.parquet", "")
    h = h[h["run"] == run]
    df = h.merge(j, on="question_id", how="inner")
    print(f"{len(df)} labelled items matched to {run}")
    rows = []
    if "judge_verdict" in df:
        d = df.dropna(subset=["judge_verdict"])
        k, lo, hi = kappa_ci(d["human_correctness"], d["judge_verdict"])
        rows.append(("correctness 3-class: human vs Opus judge", k, lo, hi, (d["human_correctness"] == d["judge_verdict"]).mean(), len(d)))
        k, lo, hi = kappa_ci(d["human_correctness"] == "correct", d["judge_verdict"] == "correct")
        rows.append(("correct-vs-not: human vs Opus judge", k, lo, hi, ((d["human_correctness"] == "correct") == (d["judge_verdict"] == "correct")).mean(), len(d)))
    for col, label in [("judge_faithful", "Opus judge"), ("nli_faithful", "NLI")]:
        if col in df:
            d = df.dropna(subset=[col])
            k, lo, hi = kappa_ci(d["human_faithful"].astype(bool), d[col].astype(bool))
            rows.append((f"faithfulness: human vs {label}", k, lo, hi, (d["human_faithful"].astype(bool) == d[col].astype(bool)).mean(), len(d)))
    out = pd.DataFrame(rows, columns=["comparison", "kappa", "kappa_lo", "kappa_hi", "agreement", "n"])
    print(out.to_markdown(index=False, floatfmt=".3f"))
    Path("reports/judge_agreement.md").write_text(f"# Judge validation ({run})\n\n" + out.to_markdown(index=False, floatfmt=".3f"))


if __name__ == "__main__":
    main()
