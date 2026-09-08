"""Build (question, positive paragraph, hard negative) triples for contrastive
fine-tuning from the QASPER *train* split only.

Why QASPER's own questions rather than LLM-synthesised ones: they are real
information-seeking questions with human-labelled evidence, they cost nothing,
and there is no synthetic-question style to confound the comparison.

Hard negatives: the highest-BM25 paragraph in the same paper that is NOT gold.
Same-paper negatives are what the scoped retrieval task actually has to
discriminate between.

Leakage guard: the paper ids of train/dev/test are pairwise disjoint (asserted
here and in tests/test_no_leakage.py) and every triple's doc_id must be a
train paper. Evaluation questions never touch training.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from corpus.qasper import load_qasper  # noqa: E402

OUT = Path("data/processed/finetune_pairs.jsonl")


def check_no_leakage(train_ids: set[str], eval_ids: set[str], pair_doc_ids: set[str]) -> None:
    overlap = train_ids & eval_ids
    if overlap:
        raise AssertionError(f"{len(overlap)} papers appear in both train and eval splits")
    stray = pair_doc_ids - train_ids
    if stray:
        raise AssertionError(f"{len(stray)} training pairs come from non-train papers")


def main() -> None:
    import bm25s

    train = load_qasper("train")
    eval_ids = set(load_qasper("dev").documents) | set(load_qasper("test").documents)
    rows = []
    for doc in train.documents.values():
        paras = [p for p in doc.paragraphs if len(p.text.split()) >= 5]
        if len(paras) < 3:
            continue
        pid_to_i = {p.para_id: i for i, p in enumerate(paras)}
        bm = bm25s.BM25()
        bm.index(bm25s.tokenize([p.text for p in paras], stopwords="en", show_progress=False), show_progress=False)
        for q in train.questions:
            if q.doc_id != doc.doc_id or q.unanswerable or not q.gold_para_ids:
                continue
            gold = [g for g in q.gold_para_ids if g in pid_to_i]
            if not gold:
                continue
            scores = bm.get_scores(bm25s.tokenize(q.text, stopwords="en", return_ids=False, show_progress=False)[0])
            order = sorted(range(len(paras)), key=lambda i: -scores[i])
            neg = next((paras[i] for i in order if paras[i].para_id not in q.gold_para_ids), None)
            if neg is None:
                continue
            for g in gold:
                rows.append({"question_id": q.question_id, "doc_id": doc.doc_id, "anchor": q.text,
                             "positive": paras[pid_to_i[g]].text, "negative": neg.text})
    check_no_leakage(set(train.documents), eval_ids, {r["doc_id"] for r in rows})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} triples from {len({r['question_id'] for r in rows})} questions "
          f"/ {len({r['doc_id'] for r in rows})} train papers -> {OUT}")


if __name__ == "__main__":
    main()
