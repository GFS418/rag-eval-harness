"""Pick example questions for the deployed app from the judged primary-arm run
and store their answers, claims and passages, so visitors can see the system
work without any API call. Selection favours variety over flattery: mostly
judged-correct answers, plus one abstention and one judged-partial answer, so
the demo shows failure modes too. Writes data/processed/demo_answers.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from corpus.qasper import load_qasper  # noqa: E402

RUN = "rag__paragraph-256-0_bge-small-ft_dense_norerank_doc__k5"
OUT = Path("data/processed/demo_answers.json")


def main() -> None:
    gen = pd.read_parquet(f"data/processed/generation/test/sonnet-5/{RUN}.parquet")
    judged = pd.read_parquet(f"reports/generation/test/sonnet-5/{RUN}.per_question.parquet")
    chunks = pd.read_parquet("data/app_bundle/chunks.parquet").set_index("chunk_id")
    corpus = load_qasper("test")
    qs = {q.question_id: q for q in corpus.questions}
    d = gen.merge(judged.drop(columns=["abstain", "error"], errors="ignore"), on="question_id")
    d["n_claims"] = d["claims"].apply(lambda s: len(json.loads(s)) if s else 0)
    good = d[(d.judge_correct == 1.0) & (d.judge_faithful == True) & d.n_claims.between(2, 4)  # noqa: E712
             & (d.answer.str.len() < 320)].sample(frac=1, random_state=3)
    seen_docs, picks = set(), []
    for _, r in good.iterrows():
        if qs[r.question_id].doc_id in seen_docs:
            continue
        seen_docs.add(qs[r.question_id].doc_id)
        picks.append((r, "judged correct and faithful"))
        if len(picks) == 6:
            break
    abst = d[(d.abstain == True) & d.unanswerable].head(1)  # noqa: E712
    part = d[(d.judge_verdict == "partial") & d.n_claims.between(1, 3)].head(1)
    for _, r in abst.iterrows():
        picks.append((r, "unanswerable question: the model abstained"))
    for _, r in part.iterrows():
        picks.append((r, "judged only partially correct"))
    out = []
    for r, label in picks:
        q = qs[r.question_id]
        cids = json.loads(r.passages)
        out.append({
            "question_id": r.question_id, "doc_id": q.doc_id, "title": corpus.documents[q.doc_id].title,
            "question": q.text, "label": label, "reference_answers": q.reference_answers,
            "abstain": bool(r.abstain), "answer": r.answer, "claims": json.loads(r.claims) if r.claims else [],
            "passages": [{"chunk_id": c, "section": chunks.loc[c, "section"], "text": chunks.loc[c, "text"]} for c in cids],
        })
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {len(out)} demo answers -> {OUT}")
    for o in out:
        print(f"  [{o['label']}] {o['question'][:70]}")


if __name__ == "__main__":
    main()
