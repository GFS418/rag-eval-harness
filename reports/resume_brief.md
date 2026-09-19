# Résumé brief: RAG Evaluation Harness

Purpose of this file: give an assistant (or a human) everything needed to
describe this project on a résumé, in a cover letter, or in an interview,
without reading the codebase. Every number below is from the repo's README and
`reports/`, computed on held-out data, with confidence intervals where they
exist. Do not round them up.

## One-line description

Built and deployed a retrieval-augmented question-answering system over 1,585
NLP research papers, and, more importantly, a quantitative evaluation harness
that measures how often it is right, how often it hallucinates, and which
design choices matter, with paired bootstrap confidence intervals and a
human-validated LLM judge.

## Links

- Code: https://github.com/GFS418/rag-eval-harness (CI passing, 49 tests)
- Live demo: https://rag-eval-harness-enlpi7prtwctye2692cupf.streamlit.app/
- Fine-tuned model: https://huggingface.co/GFS418/bge-small-qasper-ft

## What the project is, in plain terms

The market is full of "chat with your PDF" demos. This project treats the RAG
pipeline as table stakes and makes the *evaluation* the deliverable. The corpus
is QASPER (arXiv NLP papers with human-labelled evidence paragraphs and ~10%
unanswerable questions), chosen because it allows exact retrieval metrics and
measurable abstention. Everything is measured on the test split, with papers
disjoint from training, and a test enforces the split.

## Skills demonstrated (map these to job requirements)

| Skill area | Evidence in the project |
|---|---|
| NLP / LLM engineering | Chunking, dense + BM25 hybrid retrieval, cross-encoder reranking, grounded generation with claim-level citations, structured outputs, prompt design, Batches API |
| Deep learning (PyTorch) | Contrastive fine-tuning of a 33M-parameter bi-encoder (`bge-small`) with BM25-mined hard negatives; sentence-transformers / HF ecosystem; model published to the HF Hub |
| Evaluation methodology | Hand-rolled recall@k / MRR / nDCG, token-budget-matched recall, faithfulness, citation precision, abstention precision/recall, hallucination rate; closed-book and oracle baselines; retrieval × correctness error decomposition |
| Statistics | Paired bootstrap CIs on every comparison; Cohen's kappa with bootstrap CI for judge validation; explicit handling of low-base-rate metrics |
| LLM-as-judge rigor | Judge validated against 100 blind human labels under a written rubric; a "free" NLI instrument tested and rejected as a negative result |
| MLOps / engineering | Cached, resumable, cost-guarded evaluation pipeline; sqlite response cache with spend ledger; pytest + GitHub Actions; leakage-guard test; hidden-input credential installer |
| Deployment | Streamlit Community Cloud app serving the fine-tuned model and precomputed index from the HF Hub; per-day spend cap; pre-computed demo answers |
| Cost awareness | Whole evaluation for $14.90 in API spend via the Batches API, with a projection guard that refused an over-budget design and forced a redesign |

## Headline results (test split)

Retrieval, 1,297 answerable questions, scoped to the paper:
- Recall at a 1,024-token context budget: **0.713 [0.692, 0.733]** with the
  fine-tuned embedder + paragraph chunks + dense search; 0.600 for the best
  off-the-shelf configuration.
- Fine-tuning lift: **+0.149 [+0.127, +0.171]** on that config; +0.070
  [+0.065, +0.075] pooled over all chunkings and scopes.

Answer quality, 400-question stratified sample (all 98 unanswerable + 302
answerable), Claude Sonnet 5 generator, Claude Opus 5 judge:
- Correct on **74.2%** [69.2, 79.1] of answerable questions (abstentions count as wrong).
- **93.2%** [90.4, 95.7] of answers fully supported by their passages; citation precision 95.6%.
- **Hallucination rate 12.8%** [9.5, 16.3], of which four-fifths is answering
  a question the paper does not answer; unsupported claims on answerable
  questions are 3%.
- Oracle (gold passages) reaches only 76.2%, so retrieval is no longer the
  bottleneck; closed-book (no retrieval) gets 17.5%.
- Claude Haiku 4.5 on the same retrieval: as correct (paired diff +0.008,
  CI straddles zero), hallucinates less (−4.0 points [−6.8, −1.5]) because it
  abstains more, at a third of the cost.

Judge validation, 100 answers labelled blind by the author under the judge's
rubric:
- Correctness kappa **0.73 [0.54, 0.88]**, 92% agreement, no correct-vs-incorrect swaps.
- Faithfulness: 98% agreement; the judge was *stricter* than the human
  (5 vs 2 unsupported claims out of 299), so same-family leniency did not appear.
- NLI cross-encoder vs human: kappa 0.01 (rejected as an instrument).

## Findings worth telling in an interview

1. **The chunk-size "winner" flipped when the comparison was done right.**
   At fixed top-5, 512-token chunks won by +0.23 recall; at an equal
   1,024-token context budget they were the *worst* chunking (−0.06) and
   paragraph-packed 256-token chunks were best. Comparing retrieval at fixed k
   rewards reading more, not retrieving better.
2. **A 33M-parameter model fine-tuned on 3,560 in-domain triples beat every
   off-the-shelf model**, including ones 3x larger, in 8 minutes on a laptop GPU.
   And once it was fine-tuned, both hybrid BM25 fusion and cross-encoder
   reranking *hurt* (−0.075 [−0.098, −0.052] for reranking): a reranker caps
   you at its own quality.
3. **A reranker that looked like +0.095 on 100 questions was +0.026 with a CI
   through zero on 880.** The project's recurring point: confidence intervals
   change conclusions.
4. **Hallucination is mostly failing to abstain**, not inventing facts. Given
   passages, the generator invented unsupported claims 3% of the time but
   answered 43% of unanswerable questions. Abstention calibration is the next
   lever.
5. **Human evidence labels are not exhaustive.** Of 32 answers judged correct
   with no gold passage retrieved, closed-book got only 4, so most came from
   real but unlabelled passages; recall@k understates retrieval.

## Honest limitations (say these if asked; they are in the README)

- Judge and generator are from the same model family; mitigated by the human
  validation, not eliminated.
- One labeller (the author) for the judge validation; kappa CI is wide at the
  low end (0.54).
- Generation metrics are on a 400-question sample, not all 1,451, to keep
  API spend under $15; headline CIs are ±5 points.
- QASPER papers predate model training cutoffs; the closed-book baseline
  bounds the memorisation contribution (17.5%).
- FAISS was replaced by exact numpy search (identical at this corpus size)
  after an OpenMP conflict with PyTorch on macOS.

## Suggested résumé bullets (pick by target role; keep numbers exact)

General data science / ML:
- Built and deployed an LLM document-intelligence (RAG) system over 1,585
  research papers with a quantitative evaluation harness: retrieval recall@k
  across a 144-configuration ablation with paired bootstrap CIs, answer
  faithfulness and hallucination rate via an LLM judge validated against
  blind human labels (κ = 0.73), and closed-book/oracle baselines; Python,
  PyTorch, sentence-transformers, Claude API, Streamlit, pytest + CI.
- Fine-tuned a bi-encoder embedding model with contrastive loss and hard
  negatives (+0.15 recall at a fixed context budget on held-out papers),
  showing that a 33M-parameter adapted model beat larger off-the-shelf models
  and that reranking on top of it hurt.

Shorter, for a one-line project entry:
- RAG system + evaluation harness over scientific papers: 74% correct, 93%
  faithful, 12.8% hallucination on held-out questions, all with CIs and a
  human-validated judge; fine-tuned embedder (+0.15 recall); live Streamlit
  app, pytest + CI. [GitHub] [Demo]

Analyst-leaning:
- Designed and ran a 144-configuration retrieval ablation and a six-arm
  generation study with paired bootstrap confidence intervals; identified a
  metric confound (recall@k vs. token-budget recall) that reversed the
  headline conclusion, and validated an LLM judge against 100 blind human
  labels (κ = 0.73).

## Numbers that should NOT be quoted without their caveat

- "93% faithful" → say "of answers, as graded by an Opus 5 judge validated
  against human labels."
- "74% correct" → say "on answerable held-out questions, abstentions counted
  as wrong."
- "+0.15 recall" → say "at a 1,024-token context budget on held-out papers,
  for the paragraph-chunk configuration."
- Any comparison → mention it is paired over the same questions.

## Timeline and cost

Built 2026-09-07 to 2026-09-18 in about two weeks of part-time work. Total API
spend $14.90. Compute: one Apple-silicon laptop (fine-tuning ~8 min; retrieval
grid ~1 hour).
