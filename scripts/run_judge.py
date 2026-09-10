"""Score a generation run: lexical F1, NLI faithfulness, LLM-judge correctness
and faithfulness, abstention, hallucination rate, and the retrieval x answer
error decomposition.

Input : one parquet from run_generation.py
Output: reports/generation/<split>/<model>/<name>.per_question.parquet
        reports/generation/<split>/<model>/<name>.json  (means)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chunking import chunk_corpus, default_tokenizer, paragraph_token_counts  # noqa: E402
from corpus.qasper import load_qasper  # noqa: E402
from eval.answer_metrics import (NLIScorer, Outcome, abstention_metrics, correctness_request,  # noqa: E402
                                 faithfulness_request, summarize_faithfulness, token_f1, yes_no_match)
from eval.gold import gold_chunk_ids, group_chunks  # noqa: E402
from llm import Cache, run_batch  # noqa: E402
from run_retrieval import parse_chunk  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--generation", required=True, help="parquet from run_generation.py")
    ap.add_argument("--gold-threshold", type=float, default=0.5)
    ap.add_argument("--no-judge", action="store_true", help="skip Opus judge (lexical + NLI only)")
    ap.add_argument("--no-nli", action="store_true")
    ap.add_argument("--no-faith-judge", action="store_true", help="skip the Opus faithfulness judge (keep correctness + NLI)")
    args = ap.parse_args()

    gen = pd.read_parquet(args.generation)
    name = Path(args.generation).stem
    model = Path(args.generation).parent.name
    corpus = load_qasper(args.split)
    qs = {q.question_id: q for q in corpus.questions}
    paras = {p.para_id: p for d in corpus.documents.values() for p in d.paragraphs}

    # passage texts: chunks (rag) or paragraphs (oracle)
    mode = gen["mode"].iloc[0]
    chunk_text: dict[str, str] = {}
    gold_chunks: dict[str, frozenset[str]] = {}
    if mode == "rag":
        cfg_chunk = parse_chunk(name.split("__")[1].split("_")[0])
        tok = default_tokenizer()
        chunks = chunk_corpus(corpus.documents.values(), cfg_chunk, tok)
        chunk_text = {c.chunk_id: c.text for c in chunks}
        by_doc = group_chunks(chunks)
        pt = paragraph_token_counts(corpus.documents.values(), tok)
        gold_chunks = {qid: gold_chunk_ids(q, by_doc, pt, args.gold_threshold) for qid, q in qs.items()}
    elif mode == "oracle":
        chunk_text = {pid: p.text for pid, p in paras.items()}
        gold_chunks = {qid: q.gold_para_ids for qid, q in qs.items()}

    def passages_of(row):
        return [(cid, chunk_text.get(cid, "")) for cid in json.loads(row["passages"])]

    def claims_of(row):
        return [(c["text"], c["passage_numbers"]) for c in json.loads(row["claims"])] if row["claims"] else []

    # ---- judge requests ---------------------------------------------------
    corr_reqs, faith_reqs = {}, {}
    for _, row in gen.iterrows():
        q = qs[row["question_id"]]
        if row["error"] or row["abstain"]:
            continue
        if not q.unanswerable and q.reference_answers:
            corr_reqs[row["question_id"]] = correctness_request(
                row["question_id"], q.text, q.reference_answers, row["answer"], f"{model}|{name}")
        cl = claims_of(row)
        if cl and mode != "closed-book" and not args.no_faith_judge:
            faith_reqs[row["question_id"]] = faithfulness_request(
                row["question_id"], q.text, passages_of(row), cl, f"{model}|{name}")
    judge = {}
    if not args.no_judge:
        cache = Cache()
        judge = run_batch(list(corr_reqs.values()) + list(faith_reqs.values()), cache)
    nli = None if (args.no_nli or mode == "closed-book") else NLIScorer()

    # ---- per-question scoring ---------------------------------------------
    rows, outcomes = [], []
    for _, row in gen.iterrows():
        qid = row["question_id"]
        q = qs[qid]
        r = {"question_id": qid, "unanswerable": q.unanswerable, "answer_type": q.answer_type,
             "abstain": bool(row["abstain"]) if row["abstain"] is not None else None,
             "error": row["error"], "input_tokens": row["input_tokens"], "output_tokens": row["output_tokens"],
             "cost_usd": row["cost_usd"], "latency_s": row["latency_s"], "n_claims": len(claims_of(row))}
        if mode in ("rag", "oracle"):
            g = gold_chunks.get(qid, frozenset())
            r["retrieval_hit"] = bool(g & set(json.loads(row["passages"]))) if g else None
        if row["error"]:
            outcomes.append(Outcome(q.unanswerable, False, None))
            r["hallucinated"] = None
            rows.append(r)
            continue
        if not q.unanswerable and not row["abstain"]:
            refs = q.reference_answers
            r["token_f1"] = yes_no_match(row["answer"], refs) if q.answer_type == "yes_no" else token_f1(row["answer"], refs)
        if qid in corr_reqs and (resp := judge.get(corr_reqs[qid].custom_id)) and resp.data:
            r["judge_verdict"] = resp.data["verdict"]
            r["judge_correct"] = float(resp.data["verdict"] == "correct")
            r["judge_correct_or_partial"] = float(resp.data["verdict"] != "incorrect")
        faithful = None
        if qid in faith_reqs and (resp := judge.get(faith_reqs[qid].custom_id)) and resp.data:
            s = summarize_faithfulness(resp.data, r["n_claims"])
            r.update({"judge_faithful": s["faithful"], "judge_frac_supported": s["frac_supported"],
                      "judge_citation_precision": s["citation_precision"], "judge_any_contradicted": s["any_contradicted"]})
            faithful = s["faithful"]
        if nli is not None and not row["abstain"] and claims_of(row):
            s = nli.score_answer([t for _, t in passages_of(row)], claims_of(row))
            r.update(s)
            if faithful is None:
                faithful = s["nli_faithful"]
        o = Outcome(q.unanswerable, bool(row["abstain"]), faithful)
        outcomes.append(o)
        r["hallucinated"] = o.hallucinated
        rows.append(r)

    df = pd.DataFrame(rows)
    out = Path("reports/generation") / args.split / model
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / f"{name}.per_question.parquet", index=False)

    ans = df[~df["unanswerable"]]
    summary = {"name": name, "model": model, "split": args.split, "mode": mode, "n": len(df),
               "n_answerable": int(len(ans)), "n_unanswerable": int(df["unanswerable"].sum()),
               "errors": int(df["error"].notna().sum()),
               **abstention_metrics(outcomes),
               "cost_usd_total": float(df["cost_usd"].sum()),
               "judge_cost_usd": float(sum(r.cost_usd for r in judge.values())),
               "judge_errors": int(sum(1 for r in judge.values() if r.error)), "mean_input_tokens": float(df["input_tokens"].mean()),
               "mean_latency_s": float(df["latency_s"].mean()) if df["latency_s"].notna().any() else None}
    for col in ["token_f1", "judge_correct", "judge_correct_or_partial", "judge_faithful", "judge_citation_precision",
                "nli_faithful", "nli_frac_supported", "retrieval_hit"]:
        if col in df:
            summary[col] = float(pd.to_numeric(df[col], errors="coerce").mean())
    if "retrieval_hit" in df and "judge_correct" in df:
        dec = ans.dropna(subset=["retrieval_hit", "judge_correct"])
        summary["decomposition"] = {
            f"hit={h}|correct={c}": int(((dec["retrieval_hit"] == h) & (dec["judge_correct"] == c)).sum())
            for h in (True, False) for c in (1.0, 0.0)}
    (out / f"{name}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
