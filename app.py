"""Streamlit front end: ask a question about a QASPER paper, see the retrieved
passages with scores, and a grounded answer with per-claim citations.

The evaluation mindset is visible in the UI: every answer shows which chunks
were retrieved, which the model cited, and whether it abstained. Eight
pre-answered example questions (including an abstention and a partially
correct answer) demonstrate the system at zero API cost; typed questions call
Claude Sonnet 5 live, under a daily spend cap.

Deployed mode: the fine-tuned embedder, the test-split chunks and their
embeddings come from the Hugging Face Hub (configs/app.json -> hf_repo); the
QASPER corpus downloads from the public release. Nothing is chunked or
embedded at startup.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from chunking import Chunk, ChunkConfig  # noqa: E402
from corpus.qasper import load_qasper  # noqa: E402
from embed import get_embedder, register_hub_arm  # noqa: E402
from generate import GenerationInput, Passage, build_request, parse_answer  # noqa: E402
from index import DenseIndex  # noqa: E402
from llm import Cache, call, load_dotenv  # noqa: E402

CFG = json.loads(Path("configs/app.json").read_text())
BUNDLE = Path("data/app_bundle")
DEMO = Path("data/processed/demo_answers.json")
SPEND_FILE = Path("data/cache/app_spend.json")


# --------------------------------------------------------------------------
# resources
# --------------------------------------------------------------------------

def _bundle_dir() -> Path:
    if (BUNDLE / "embeddings.npy").exists():
        return BUNDLE
    from huggingface_hub import snapshot_download

    local = snapshot_download(CFG["hf_repo"], allow_patterns=["app_bundle/*"])
    return Path(local) / "app_bundle"


@st.cache_resource(show_spinner="Loading corpus, index and embedder (first load takes a minute)...")
def load_stack():
    register_hub_arm(CFG["embedder"], CFG["hf_repo"])
    corpus = load_qasper(CFG["split"])
    bdir = _bundle_dir()
    df = pd.read_parquet(bdir / "chunks.parquet")
    vecs = np.load(bdir / "embeddings.npy")
    chunks = [Chunk(r.chunk_id, r.doc_id, r.text, int(r.n_tokens), {}, r.section) for r in df.itertuples()]
    index = DenseIndex(vecs, df["doc_id"].tolist())
    embedder = get_embedder(CFG["embedder"])
    return corpus, chunks, index, embedder


def retrieve(question: str, doc_id: str, chunks, index, embedder, k: int):
    q = embedder.encode_queries([question])[0]
    r = index.search(q, k, doc_id)
    return [(chunks[int(i)], float(s)) for i, s in zip(r.idx, r.score)]


# --------------------------------------------------------------------------
# spend cap (per day, per process; the Anthropic console limit is the real guard)
# --------------------------------------------------------------------------

def spend_today() -> float:
    try:
        d = json.loads(SPEND_FILE.read_text())
        return float(d["usd"]) if d.get("day") == time.strftime("%Y-%m-%d") else 0.0
    except Exception:
        return 0.0


def add_spend(usd: float) -> None:
    SPEND_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPEND_FILE.write_text(json.dumps({"day": time.strftime("%Y-%m-%d"), "usd": spend_today() + usd}))


def api_key_available() -> bool:
    load_dotenv()
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def render_result(question: str, answer, passages: list[dict], scores: list[float] | None,
                  references: list[str] | None = None, note: str | None = None):
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Answer")
        if note:
            st.caption(note)
        if answer is None:
            st.info("No answer generated (retrieval only).")
        elif answer["abstain"]:
            st.warning(answer["answer"])
            st.caption("The model abstained: the retrieved passages did not support an answer.")
        else:
            st.markdown(answer["answer"])
            st.markdown("**Claims and citations**")
            for c in answer["claims"]:
                nums = ", ".join(f"[{n}]" for n in c["passage_numbers"]) or "(uncited)"
                st.markdown(f"- {c['text']} {nums}")
        if references:
            with st.expander("Human reference answers (QASPER annotators)"):
                for r in references:
                    st.markdown(f"- {r}")
    with right:
        st.subheader("Retrieved passages")
        cited = {n for c in (answer["claims"] if answer and not answer["abstain"] else []) for n in c["passage_numbers"]}
        for i, p in enumerate(passages, start=1):
            score = f" · score {scores[i - 1]:.3f}" if scores else ""
            tag = " · cited" if i in cited else ""
            with st.expander(f"[{i}] {p['section'] or 'no section'}{score}{tag}", expanded=i in cited):
                st.write(p["text"])


st.set_page_config(page_title="RAG Eval Harness", layout="wide")
st.title("Document intelligence over scientific papers")
st.caption("Retrieval-augmented QA with citations over 416 NLP papers (QASPER test split). "
           "Every answer shows its evidence; the README shows how often it is wrong.")

corpus, chunks, index, embedder = load_stack()
titles = {d.title: d.doc_id for d in corpus.documents.values()}
demos = json.loads(DEMO.read_text()) if DEMO.exists() else []

tab_demo, tab_live = st.tabs(["Example questions (pre-computed)", "Ask your own (live)"])

with tab_demo:
    st.markdown("Answers below were produced by the deployed configuration during the evaluation run and "
                "graded by the Opus 5 judge. They include an abstention and a partially correct answer on purpose.")
    labels = [f"{d['question']}  —  *{d['label']}*" for d in demos]
    choice = st.radio("Pick a question", range(len(demos)), format_func=lambda i: labels[i], label_visibility="collapsed")
    if demos:
        d = demos[choice]
        st.markdown(f"**Paper:** {d['title']}")
        render_result(d["question"], {"abstain": d["abstain"], "answer": d["answer"], "claims": d["claims"]},
                      d["passages"], None, d["reference_answers"])

with tab_live:
    col1, col2 = st.columns([2, 1])
    with col1:
        title = st.selectbox("Paper", sorted(titles))
        question = st.text_input("Question", placeholder="What datasets do they evaluate on?")
        ask = st.button("Ask", type="primary")
    with col2:
        st.markdown(f"**Retrieval:** fine-tuned bge-small, paragraph chunks, dense, top-{CFG['top_k']}  \n"
                    f"**Generator:** Claude Sonnet 5  \n"
                    f"**Today's live spend:** ${spend_today():.3f} of ${CFG['daily_cap_usd']:.2f} cap")
    if ask and question.strip():
        doc_id = titles[title]
        hits = retrieve(question, doc_id, chunks, index, embedder, CFG["top_k"])
        passages = [Passage(c.chunk_id, c.text, c.section) for c, _ in hits]
        inp = GenerationInput("app", question, title, passages, "rag")
        answer, note = None, None
        if spend_today() >= CFG["daily_cap_usd"]:
            note = "Daily spend cap reached; showing retrieval only."
        elif not api_key_available():
            try:
                has_store = len(list(st.secrets.keys())) > 0
            except Exception:
                has_store = False
            note = ("No API key configured; showing retrieval only. "
                    + ("A secrets store exists but has no ANTHROPIC_API_KEY entry." if has_store
                       else "No secrets store found (set ANTHROPIC_API_KEY in the app's Secrets settings)."))
        else:
            with st.spinner("Generating grounded answer..."):
                resp = call(build_request(inp, CFG["model"], "app"), Cache(Path("data/cache/app_llm.sqlite")))
            add_spend(resp.cost_usd)
            if resp.data:
                ga = parse_answer(inp, resp.data)
                answer = {"abstain": ga.abstain, "answer": ga.answer,
                          "claims": [{"text": c.text, "passage_numbers": c.passage_numbers} for c in ga.claims]}
            else:
                note = f"Generation failed: {resp.error}"
        render_result(question, answer, [{"section": c.section, "text": c.text} for c, _ in hits],
                      [s for _, s in hits], note=note)

st.divider()
st.markdown("**How good is this?** On 400 held-out test questions: 74% correct, 93% of answers fully supported by "
            "their passages, 13% hallucination rate (mostly answering unanswerable questions), judge validated "
            "against human labels at kappa 0.73. Full ablations with confidence intervals in the "
            "[README](https://github.com/GFS418/rag-eval-harness).")
