"""Human labelling UI for validating the automatic judges.

    streamlit run scripts/label_ui.py -- --generation <parquet> [--n 100]

Samples answered (non-abstained) questions from one generation run, shows the
passages the model saw, its answer and claims, and the human references, and
records: correctness (correct / partial / incorrect) and per-claim support
(supported / unsupported). Labels append to reports/human_labels.csv.
Labels are blind to the judge's verdicts (they are not shown).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunking import chunk_corpus, default_tokenizer  # noqa: E402
from corpus.qasper import load_qasper  # noqa: E402
from run_retrieval import parse_chunk  # noqa: E402

OUT = Path("reports/human_labels.csv")


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generation", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


@st.cache_resource
def load(generation: str, split: str, n: int, seed: int):
    gen = pd.read_parquet(generation)
    name = Path(generation).stem
    corpus = load_qasper(split)
    qs = {q.question_id: q for q in corpus.questions}
    mode = gen["mode"].iloc[0]
    if mode == "rag":
        chunks = chunk_corpus(corpus.documents.values(), parse_chunk(name.split("__")[1].split("_")[0]), default_tokenizer())
        text = {c.chunk_id: c.text for c in chunks}
    else:
        text = {p.para_id: p.text for d in corpus.documents.values() for p in d.paragraphs}
    pool = gen[(gen["abstain"] == False) & gen["error"].isna()]  # noqa: E712
    pool = pool[pool["question_id"].map(lambda q: not qs[q].unanswerable)]
    sample = pool.sample(n=min(n, len(pool)), random_state=seed).reset_index(drop=True)
    return sample, qs, text, name


args = get_args()
sample, qs, text, run_name = load(args.generation, args.split, args.n, args.seed)
done = set()
if OUT.exists():
    prev = pd.read_csv(OUT)
    done = set(prev[prev["run"] == run_name]["question_id"])
todo = [i for i in range(len(sample)) if sample.loc[i, "question_id"] not in done]
st.sidebar.markdown(f"**{run_name}**\n\n{len(sample) - len(todo)} / {len(sample)} labelled")
if not todo:
    st.success("All sampled items labelled.")
    st.stop()
i = todo[0]
row = sample.loc[i]
q = qs[row["question_id"]]
claims = json.loads(row["claims"]) if row["claims"] else []

st.markdown(f"### Question\n{q.text}")
st.markdown("**Reference answers (human annotators):**")
for r in q.reference_answers:
    st.markdown(f"- {r}")
c1, c2 = st.columns(2)
with c1:
    st.markdown("### Model answer")
    st.markdown(row["answer"])
    st.markdown("**Claims**")
    for k, c in enumerate(claims, start=1):
        st.markdown(f"{k}. {c['text']}  _(cites {c['passage_numbers']})_")
with c2:
    st.markdown("### Passages the model saw")
    for k, cid in enumerate(json.loads(row["passages"]), start=1):
        with st.expander(f"[{k}]", expanded=True):
            st.write(text.get(cid, ""))

with st.form("label"):
    correctness = st.radio("Correctness vs references", ["correct", "partial", "incorrect"], horizontal=True)
    supports = [st.radio(f"Claim {k}: supported by the passages?", ["supported", "unsupported"], horizontal=True,
                         key=f"c{k}") for k in range(1, len(claims) + 1)]
    note = st.text_input("Note (optional)")
    if st.form_submit_button("Save and next"):
        rec = {"run": run_name, "question_id": row["question_id"], "human_correctness": correctness,
               "human_faithful": all(s == "supported" for s in supports),
               "human_claim_supports": json.dumps(supports), "n_claims": len(claims), "note": note}
        pd.DataFrame([rec]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        st.rerun()
