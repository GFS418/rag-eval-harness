"""Generate answers for one (split, retrieval config, model, mode) and save them.

Reads the rankings written by run_retrieval.py, builds one prompt per question
with the top-k chunks (rag), the gold paragraphs (oracle) or nothing
(closed-book), sends everything through the Batches API (50% cost) unless
--sync, and writes:

  data/processed/generation/<split>/<model>/<mode>__<cfg>__k<k>.parquet
     question_id, abstain, answer, claims(json), invalid_citations,
     input_tokens, output_tokens, latency_s, error
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chunking import chunk_corpus, default_tokenizer  # noqa: E402
from corpus.qasper import load_qasper  # noqa: E402
from generate import GenerationInput, Passage, build_request, parse_answer  # noqa: E402
from llm import Cache, call, run_batch  # noqa: E402
from run_retrieval import parse_chunk  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--config", required=True, help="retrieval config name (rag mode) or chunk config (oracle)")
    ap.add_argument("--model", default="sonnet-5")
    ap.add_argument("--mode", default="rag", choices=["rag", "oracle", "closed-book"])
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--include-unanswerable", action="store_true", default=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sync", action="store_true", help="sequential calls instead of a batch")
    args = ap.parse_args()

    corpus = load_qasper(args.split)
    questions = [q for q in corpus.questions if q.unanswerable or q.gold_para_ids]
    if args.limit:
        questions = questions[:args.limit]
    para_text = {p.para_id: p for d in corpus.documents.values() for p in d.paragraphs}

    inputs: list[GenerationInput] = []
    if args.mode == "rag":
        ranks = pd.read_parquet(Path("data/processed/retrieval") / args.split / f"{args.config}.parquet")
        chunk_cfg = parse_chunk(args.config.split("_")[0])
        chunks = {c.chunk_id: c for c in chunk_corpus(corpus.documents.values(), chunk_cfg, default_tokenizer())}
        top = ranks[ranks["rank"] < args.top_k].sort_values(["question_id", "rank"])
        by_q = {qid: g["chunk_id"].tolist() for qid, g in top.groupby("question_id")}
        for q in questions:
            cids = by_q.get(q.question_id)
            if cids is None:      # unanswerable questions were not retrieved for; do it now
                continue
            inputs.append(GenerationInput(q.question_id, q.text, corpus.documents[q.doc_id].title,
                                          [Passage(c, chunks[c].text, chunks[c].section) for c in cids], "rag"))
    elif args.mode == "oracle":
        for q in questions:
            if not q.gold_para_ids:
                continue
            ps = [para_text[p] for p in sorted(q.gold_para_ids)]
            inputs.append(GenerationInput(q.question_id, q.text, corpus.documents[q.doc_id].title,
                                          [Passage(p.para_id, p.text, p.section) for p in ps], "oracle"))
    else:
        for q in questions:
            inputs.append(GenerationInput(q.question_id, q.text, corpus.documents[q.doc_id].title, [], "closed-book"))

    reqs = [build_request(i, args.model, f"{args.config}-k{args.top_k}", args.effort) for i in inputs]
    print(f"{len(reqs)} requests ({args.mode}, {args.model}, {args.config})", flush=True)
    cache = Cache()
    if args.sync:
        responses = {r.custom_id: call(r, cache) for r in reqs}
    else:
        responses = run_batch(reqs, cache)

    rows = []
    for inp, req in zip(inputs, reqs):
        resp = responses[req.custom_id]
        ga = parse_answer(inp, resp.data) if resp.data else None
        rows.append({
            "question_id": inp.question_id, "mode": inp.mode, "model": args.model,
            "abstain": ga.abstain if ga else None, "answer": ga.answer if ga else None,
            "claims": json.dumps([c.__dict__ for c in ga.claims]) if ga else None,
            "invalid_citations": ga.invalid_citations if ga else None,
            "passages": json.dumps([p.chunk_id for p in inp.passages]),
            "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens,
            "latency_s": resp.latency_s, "cost_usd": resp.cost_usd, "error": resp.error,
        })
    df = pd.DataFrame(rows)
    out = Path("data/processed/generation") / args.split / args.model
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.mode}__{args.config}__k{args.top_k}.parquet"
    df.to_parquet(path, index=False)
    print(f"wrote {path}: {len(df)} rows, {df['error'].notna().sum()} errors, "
          f"abstain rate {df['abstain'].mean():.2f}, est. cost ${df['cost_usd'].sum():.2f}")


if __name__ == "__main__":
    main()
