"""Streamlit front end: ask a question about a QASPER paper, see the retrieved
passages with scores, and a grounded answer with per-claim citations.

The evaluation mindset is visible in the UI: every answer shows which chunks
were retrieved, which the model cited, and whether it abstained. The retrieval
config is the one selected on the dev split (see README); the generator is
Claude Sonnet 5. A daily spend cap and cached demo questions keep the free
tier alive.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from chunking import ChunkConfig, chunk_corpus, default_tokenizer  # noqa: E402
from corpus.qasper import load_qasper  # noqa: E402
from embed import get_embedder  # noqa: E402
from generate import GenerationInput, Passage, build_request, parse_answer  # noqa: E402
from llm import Cache, call, load_dotenv  # noqa: E402
from rerank import Reranker  # noqa: E402
from retrieval import RetrievalConfig, Retriever  # noqa: E402

APP_CFG = json.loads(Path("configs/app.json").read_text()) if Path("configs/app.json").exists() else {
    "split": "test", "chunk": "fixed-256-0", "embedder": "bge-small", "hybrid": True,
    "rerank": True, "reranker": "minilm-ce", "top_k": 5, "model": "sonnet-5", "daily_cap_usd": 1.0}
DEMO_PATH = Path("data/processed/demo_answers.json")


@st.cache_resource(show_spinner="Loading corpus and index (first load takes a minute)...")
def load_stack():
    corpus = load_qasper(APP_CFG["split"])
    strat, size, ov = APP_CFG["chunk"].split("-")
    chunks = chunk_corpus(corpus.documents.values(), ChunkConfig(strat, int(size), int(ov)), default_tokenizer())
    reranker = Reranker(APP_CFG["reranker"]) if APP_CFG["rerank"] else None
    retriever = Retriever(chunks, get_embedder(APP_CFG["embedder"]), reranker)
    cfg = RetrievalConfig(ChunkConfig(strat, int(size), int(ov)), APP_CFG["embedder"], APP_CFG["hybrid"],
                          APP_CFG["rerank"], APP_CFG["reranker"], top_k=APP_CFG["top_k"], scope="doc")
    return corpus, retriever, cfg


def spend_today() -> float:
    return float(st.session_state.get("spend", 0.0))


st.set_page_config(page_title="RAG Eval Harness", layout="wide")
st.title("Document intelligence over scientific papers")
st.caption("Retrieval-augmented QA with citations. Every answer shows its evidence, and the "
           "README shows how often it is wrong.")

corpus, retriever, cfg = load_stack()
titles = {d.title: d.doc_id for d in corpus.documents.values()}
col1, col2 = st.columns([2, 1])
with col1:
    title = st.selectbox("Paper", sorted(titles), index=0)
    question = st.text_input("Question", placeholder="What datasets do they evaluate on?")
with col2:
    st.markdown(f"**Retrieval config**\n\n`{cfg.name}`, top-k = {cfg.top_k}\n\n**Generator:** {APP_CFG['model']}")
    st.markdown(f"Session spend: ${spend_today():.3f} / cap ${APP_CFG['daily_cap_usd']:.2f}")

if st.button("Ask", type="primary") and question.strip():
    doc_id = titles[title]
    hits = retriever.retrieve("app", question, doc_id, cfg)
    passages = [Passage(h.chunk.chunk_id, h.chunk.text, h.chunk.section) for h in hits]
    inp = GenerationInput("app", question, title, passages, "rag")
    load_dotenv()
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:          # no secrets.toml at all
        pass
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    answer = None
    if spend_today() >= APP_CFG["daily_cap_usd"]:
        st.warning("Spend cap reached for this session; showing retrieval only.")
    elif not has_key:
        st.warning("No API key configured; showing retrieval only.")
    else:
        with st.spinner("Generating grounded answer..."):
            resp = call(build_request(inp, APP_CFG["model"], "app"), Cache(Path("data/cache/app_llm.sqlite")))
        st.session_state["spend"] = spend_today() + resp.cost_usd
        if resp.data:
            answer = parse_answer(inp, resp.data)
        else:
            st.error(f"Generation failed: {resp.error}")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Answer")
        if answer is None:
            st.info("(no answer generated)")
        elif answer.abstain:
            st.warning(answer.answer)
            st.caption("The model abstained: the retrieved passages did not support an answer.")
        else:
            st.markdown(answer.answer)
            st.markdown("**Claims and citations**")
            for c in answer.claims:
                nums = ", ".join(f"[{n}]" for n in c.passage_numbers) or "(uncited)"
                st.markdown(f"- {c.text} {nums}")
            if answer.invalid_citations:
                st.caption(f"{answer.invalid_citations} citation(s) pointed outside the passage list.")
    with right:
        st.subheader("Retrieved passages")
        cited = {n for c in (answer.claims if answer else []) for n in c.passage_numbers}
        for i, h in enumerate(hits, start=1):
            tag = " · cited" if i in cited else ""
            with st.expander(f"[{i}] {h.chunk.section or 'no section'} · score {h.score:.3f}{tag}", expanded=i in cited):
                st.write(h.chunk.text)

st.divider()
st.markdown("**How good is this?** See the README for recall@k, faithfulness, hallucination rate and the "
            "ablation table with confidence intervals. The evaluation is the project; this page is the demo.")
