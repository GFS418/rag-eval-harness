"""Run retrieval for a set of configs over one split and save rankings + metrics.

top_k is post-hoc: one ranking of ``candidate_k`` chunks per
(chunk config, embedder, hybrid, rerank, scope) yields every k <= candidate_k,
so the grid never re-retrieves just to change k.

Outputs
  data/processed/retrieval/<split>/<cfg>.parquet    question_id, rank, chunk_id, score
  reports/retrieval/<split>/<cfg>.per_question.parquet  per-question metrics (for bootstrap)
  reports/retrieval/<split>/<cfg>.json               means + n + timing
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chunking import ChunkConfig, chunk_corpus, default_tokenizer, paragraph_token_counts  # noqa: E402
from corpus.qasper import load_qasper  # noqa: E402
from embed import get_embedder  # noqa: E402
from eval.gold import gold_chunk_ids, group_chunks  # noqa: E402
from eval.retrieval_metrics import all_metrics  # noqa: E402
from rerank import Reranker  # noqa: E402
from retrieval import RetrievalConfig, Retriever  # noqa: E402

KS = (1, 3, 5, 10)


def query_text(q, corpus, scope: str) -> str:
    if scope == "open-titled":
        return f"In the paper \"{corpus.documents[q.doc_id].title}\": {q.text}"
    return q.text


def parse_chunk(s: str) -> ChunkConfig:
    strat, size, overlap = s.split("-")
    return ChunkConfig(strat, int(size), int(overlap))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--chunks", nargs="+", default=["fixed-256-0"])
    ap.add_argument("--embedders", nargs="+", default=["bge-small"])
    ap.add_argument("--hybrid", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--rerank", nargs="+", type=int, default=[0])
    ap.add_argument("--scopes", nargs="+", default=["doc", "open", "open-titled"])
    ap.add_argument("--candidate-k", type=int, default=50)
    ap.add_argument("--reranker", default="minilm-ce")
    ap.add_argument("--gold-threshold", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None, help="first N questions (smoke tests)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    corpus = load_qasper(args.split)
    # Retrieve for *every* question (the generator needs passages for the
    # unanswerable ones too); retrieval metrics only where gold exists.
    questions = list(corpus.questions)
    if args.limit:
        questions = questions[:args.limit]
    n_gold_q = sum(bool(q.gold_para_ids) and not q.unanswerable for q in questions)
    print(f"{corpus} -> retrieving for {len(questions)} questions; "
          f"{n_gold_q} answerable with gold paragraphs", flush=True)
    tok = default_tokenizer()
    para_tokens = paragraph_token_counts(corpus.documents.values(), tok)

    out_rank = Path("data/processed/retrieval") / args.split
    out_rep = Path("reports/retrieval") / args.split
    out_rank.mkdir(parents=True, exist_ok=True)
    out_rep.mkdir(parents=True, exist_ok=True)
    reranker = Reranker(args.reranker) if any(args.rerank) else None

    for chunk_s in args.chunks:
        ccfg = parse_chunk(chunk_s)
        chunks = chunk_corpus(corpus.documents.values(), ccfg, tok)
        by_doc = group_chunks(chunks)
        gold = {q.question_id: (frozenset() if q.unanswerable else
                                gold_chunk_ids(q, by_doc, para_tokens, args.gold_threshold))
                for q in questions}
        n_gold = sum(bool(g) for g in gold.values())
        print(f"[{ccfg.name}] {len(chunks)} chunks; {n_gold}/{n_gold_q} answerable questions have >=1 gold chunk", flush=True)
        for emb_key in args.embedders:
            embedder = get_embedder(emb_key)
            t0 = time.time()
            retriever = Retriever(chunks, embedder, reranker)
            print(f"  [{emb_key}] index built in {time.time() - t0:.0f}s", flush=True)
            for hybrid, rerank, scope in itertools.product(args.hybrid, args.rerank, args.scopes):
                cfg = RetrievalConfig(ccfg, emb_key, bool(hybrid), bool(rerank), reranker=args.reranker,
                                      top_k=args.candidate_k, candidate_k=args.candidate_k, scope=scope)
                q_texts = [query_text(q, corpus, scope) for q in questions]
                q_vecs = embedder.encode_queries(q_texts)
                rep_path = out_rep / f"{cfg.name}.json"
                if rep_path.exists() and not args.force:
                    print(f"    skip {cfg.name} (exists)", flush=True)
                    continue
                t0 = time.time()
                rows, per_q = [], []
                for q, qt, qv in zip(questions, q_texts, q_vecs):
                    hits = retriever.retrieve(q.question_id, qt, q.doc_id, cfg, qv)
                    ranked = [h.chunk.chunk_id for h in hits]
                    rows.extend((q.question_id, h.rank, h.chunk.chunk_id, h.score) for h in hits)
                    if gold[q.question_id]:
                        m = all_metrics(ranked, gold[q.question_id], KS)
                        m["question_id"] = q.question_id
                        m["n_gold"] = len(gold[q.question_id])
                        per_q.append(m)
                elapsed = time.time() - t0
                if reranker:
                    reranker.flush()
                pd.DataFrame(rows, columns=["question_id", "rank", "chunk_id", "score"]).to_parquet(
                    out_rank / f"{cfg.name}.parquet", index=False)
                df = pd.DataFrame(per_q)
                df.to_parquet(out_rep / f"{cfg.name}.per_question.parquet", index=False)
                means = {k: float(df[k].mean()) for k in df.columns if k not in ("question_id", "n_gold")}
                summary = {"config": cfg.name, "split": args.split, "n_questions": len(df),
                           "n_chunks": len(chunks), "gold_threshold": args.gold_threshold,
                           "seconds": elapsed, "ms_per_query": 1000 * elapsed / len(questions),
                           **means}
                rep_path.write_text(json.dumps(summary, indent=1))
                print(f"    {cfg.name:55s} R@5={means['recall@5']:.3f} hit@5={means['hit@5']:.3f} "
                      f"MRR={means['mrr']:.3f} nDCG@10={means['ndcg@10']:.3f}  {1000*elapsed/len(questions):.1f} ms/q", flush=True)


if __name__ == "__main__":
    main()
