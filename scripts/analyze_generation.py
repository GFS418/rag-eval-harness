"""Secondary analyses on the judged generation runs: hallucination
decomposition, correctness over all answerable questions (abstain = wrong),
NLI-vs-judge agreement, cost per query, and the memorisation check.
Writes reports/generation/<split>/analysis.md"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval.stats import bootstrap_mean  # noqa: E402
from judge_agreement import kappa_ci  # noqa: E402
from llm import price_for  # noqa: E402

MODEL_IDS = {"sonnet-5": "claude-sonnet-5", "haiku-4.5": "claude-haiku-4-5"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--primary", default="sonnet-5/rag__paragraph-256-0_bge-small-ft_dense_norerank_doc__k5")
    ap.add_argument("--closed-book", default="sonnet-5/closed-book__none__k5")
    args = ap.parse_args()
    rep = Path("reports/generation") / args.split
    dfs = {f"{p.parent.name}/{p.name.replace('.per_question.parquet', '')}": pd.read_parquet(p)
           for p in sorted(rep.glob("*/*.per_question.parquet"))}
    out = [f"# Secondary analyses ({args.split})\n"]

    out.append("## Hallucination decomposition (arms with the Opus faithfulness judge)\n")
    rows = []
    for k, d in dfs.items():
        if d.get("judge_faithful") is None or d["judge_faithful"].isna().all():
            continue
        n_un, n_an = int(d.unanswerable.sum()), int((~d.unanswerable).sum())
        ua = int((d.unanswerable & (d.abstain == False)).sum())  # noqa: E712
        uf = int(((~d.unanswerable) & (d.abstain == False) & (d.judge_faithful == False)).sum())  # noqa: E712
        oa = int(((~d.unanswerable) & (d.abstain == True)).sum())  # noqa: E712
        rows.append({"run": k, "answered an unanswerable": f"{ua}/{n_un}", "unfaithful on answerable": f"{uf}/{n_an}",
                     "over-abstained on answerable": f"{oa}/{n_an}", "hallucination rate": (ua + uf) / len(d)})
    out.append(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f"))

    out.append("\n## Correctness over ALL answerable questions (abstaining counts as not correct)\n")
    rows = []
    for k, d in dfs.items():
        a = d[~d.unanswerable]
        corr = (a.judge_correct == 1.0).fillna(False).values.astype(float)
        rows.append({"run": k, "n": len(a), "correct": str(bootstrap_mean(corr))})
    out.append(pd.DataFrame(rows).to_markdown(index=False))

    out.append("\n## NLI cross-encoder vs Opus judge on faithfulness\n")
    rows = []
    for k, d in dfs.items():
        if "judge_faithful" not in d or d["judge_faithful"].isna().all():
            continue
        d = d.dropna(subset=["judge_faithful", "nli_faithful"])
        j, n = d.judge_faithful.astype(bool).values, d.nli_faithful.astype(bool).values
        kp, lo, hi = kappa_ci(j, n)
        pos, neg = d[j].nli_min_entail.values, d[~j].nli_min_entail.values
        auc = float(np.mean([p > q for p in pos for q in neg])) if len(pos) and len(neg) else float("nan")
        rows.append({"run": k, "n": len(d), "judge faithful": j.mean(), "NLI faithful": n.mean(),
                     "kappa": f"{kp:.3f} [{lo:.3f}, {hi:.3f}]", "AUC of NLI score for judge label": auc})
    out.append(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f"))

    out.append("\n## Cost per query (Batches API prices) and prompt size\n")
    rows = []
    for k, d in dfs.items():
        pi, po = price_for(MODEL_IDS.get(k.split("/")[0], ""))
        usd = ((d.input_tokens * pi + d.output_tokens * po) / 1e6 * 0.5).mean()
        rows.append({"run": k, "usd per query": usd, "input tokens": d.input_tokens.mean(), "output tokens": d.output_tokens.mean()})
    out.append(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4f"))

    if args.primary in dfs and args.closed_book in dfs:
        d = dfs[args.primary]
        m = d[(~d.unanswerable) & (d.retrieval_hit == False) & d.judge_correct.notna()]  # noqa: E712
        cb = dfs[args.closed_book].set_index("question_id")
        cb_correct = int((cb.reindex(m.question_id).judge_correct == 1).sum())
        out.append(f"\n## Memorisation check\n\nPrimary arm: {int((m.judge_correct == 1).sum())} of {len(m)} answered "
                   f"questions where no gold chunk was retrieved were still judged correct. Closed-book got "
                   f"{cb_correct} of those same questions right. The rest were answered from non-gold passages, "
                   f"i.e. the evidence labels are not exhaustive and recall@k understates retrieval.\n")
    (rep / "analysis.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
