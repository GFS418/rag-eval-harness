# RAG Evaluation Harness — document intelligence over scientific papers

[![CI](https://github.com/GFS418/rag-eval-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/GFS418/rag-eval-harness/actions/workflows/ci.yml)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rag-eval-harness-enlpi7prtwctye2692cupf.streamlit.app/)
[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Hub-bge--small--qasper--ft-yellow)](https://huggingface.co/GFS418/bge-small-qasper-ft)

**▶ [Live demo](https://rag-eval-harness-enlpi7prtwctye2692cupf.streamlit.app/):** pick a
paper, ask a question → grounded answer with per-claim citations and the retrieved
passages. Eight pre-computed examples (including an abstention and a partially correct
answer) cost nothing to view; typed questions call Claude Sonnet 5 live. *(Free tier sleeps
when idle; give it a minute to wake.)*

*Retrieval-augmented question answering with citations, and a quantitative
evaluation harness that says how often it is right, how often it hallucinates,
and which design choices matter, with confidence intervals.*

**The evaluation is the project; the RAG pipeline is table stakes.**

## Results at a glance

_(filled in from `reports/` when the test-split runs complete)_

- **Retrieval (test, 1,297 questions, scoped to the paper):** recall at a 1,024-token context budget
  **0.713** [0.692, 0.733] with the fine-tuned bge-small + paragraph chunks + dense search,
  versus 0.600 for the best off-the-shelf configuration (bge-small, hybrid). recall@5 0.725, MRR 0.658.
- **Answer quality (test, 400-question stratified sample, Claude Sonnet 5):** correct on **74.2%**
  [69.2, 79.1] of answerable questions (abstentions count as wrong); **93.2%** [90.4, 95.7] of answers
  fully supported by the passages (Opus 5 judge); citation precision 95.6%; **hallucination rate 12.8%**
  [9.5, 16.3], four-fifths of it from answering questions that had no answer in the paper.
- **Baselines bracket the system:** closed-book (no retrieval) is correct on 17.5% of answerable questions;
  oracle gold context reaches 76.2%, only +2 points over live retrieval, so generation, not retrieval, is now the ceiling.
- **Cheaper is not worse:** Haiku 4.5 on the same retrieval is as correct (paired diff +0.008 [−0.039, +0.051]),
  hallucinates *less* (−4.0 points [−6.8, −1.5]) because it abstains more, at a third of the cost.
- **A negative result on instruments:** the open NLI cross-encoder agrees with the Opus judge at kappa 0.03.
  Sentence-level NLI does not transfer to paragraph-length evidence; only the validated judge is reported.
- **Fine-tuned embedding (test):** **+0.149** recall@1024 tokens over the same model off the shelf,
  paired bootstrap CI [+0.127, +0.171]; +0.070 [+0.065, +0.075] pooled over all chunkings and scopes.
- **Judge validation:** 100 answers labelled blind by a human under the judge's rubric. Correctness
  kappa **0.73** [0.54, 0.88] (92% agreement, no correct-vs-incorrect swaps); faithfulness agreement 98%,
  with the judge *stricter* than the human (5 vs 2 unsupported claims out of 299). The NLI instrument
  agrees with the human at kappa 0.01.

## What this is

A question-answering system over a corpus of 1,585 NLP research papers
(the [QASPER](https://allenai.org/data/qasper) dataset), where every answer is
grounded in retrieved passages and cites them claim by claim. Around it, an
evaluation harness that measures:

| Layer | Metric | Instrument |
|---|---|---|
| Retrieval | recall@k, hit@k, MRR, nDCG@k | exact, from human-labelled evidence paragraphs |
| Answer correctness | token-F1; judge verdict (correct / partial / incorrect) | QASPER evaluator; Claude Opus 5 judge |
| Faithfulness | every claim supported by a provided passage | NLI cross-encoder **and** Opus 5 judge, both validated against human labels |
| Citation precision | cited passage actually supports the claim | Opus 5 judge |
| Abstention | precision / recall of "cannot answer" on unanswerable questions | gold unanswerable labels (majority of annotators) |
| Hallucination rate | unsupported claim on an answerable question, or a confident answer on an unanswerable one | composite of the above |
| Cost and latency | tokens, $ at list price, ms per query | recorded per call |

Every comparison between configurations is a **paired bootstrap over questions**
(same questions, two configs), so "hybrid beats dense by +0.024 recall@5
[0.012, 0.036]" is a statement with uncertainty attached, not a point estimate.

## Why QASPER

- Questions were written by NLP practitioners who had read only the title and
  abstract; answers and **evidence paragraphs** were labelled by annotators who
  read the full paper. That gives exact retrieval gold at paragraph level.
- ~10% of questions are **unanswerable** from the paper, so abstention is measurable.
- Full text is already parsed to sections and paragraphs (no PDF parsing noise).
- Three splits by paper: train (fine-tuning pairs only), dev (config selection),
  test (reported numbers). Papers are disjoint; a test enforces it.

Eval-set construction is audited by `corpus.qasper.summarize()`; the counts
(unanswerable by majority vs by any annotator, questions whose only evidence is
a figure or table, evidence strings that could not be mapped to a paragraph)
are reported rather than hidden.

## The ablation grid

| Axis | Levels |
|---|---|
| Chunking | fixed windows of 128 / 256 / 512 tokens (±overlap); paragraph-packed 256 / 512 |
| Embedding | bge-small, bge-base, e5-base, MiniLM; **bge-small fine-tuned** on train-split questions |
| Lexical | dense only vs hybrid (BM25 + dense, reciprocal rank fusion) |
| Reranker | none vs MiniLM cross-encoder vs bge-reranker-base |
| Scope | within the paper (chat-with-this-PDF); whole corpus with the title in the query; whole corpus, bare question |
| Generator | Claude Sonnet 5 vs Claude Haiku 4.5, top-k 3 / 5 / 10 |
| Baselines | closed-book (no passages); oracle (gold paragraphs as passages) |

Retrieval metrics need no API calls, so the full grid runs; generation and
judging run on the dev-selected shortlist plus the baselines.

## Finding 1: the chunk-size "winner" depends on what you hold fixed

Selection was done on the dev split (886 answerable questions with gold
paragraphs, 144 configurations, no API calls). Sorted by **recall@5**, 512-token
fixed windows win by a mile (+0.23 over 128-token windows, paired bootstrap CI
[0.227, 0.240]). But five 512-token chunks are ~2,500 tokens, roughly half of a
typical paper, while five 128-token chunks are an eighth of it. Comparing at a
fixed *k* rewards reading more.

Held at an equal **context budget of 1,024 tokens** handed to the generator,
the ranking inverts: 512-token windows are the *worst* chunking (−0.062 vs
128-token windows, CI [−0.068, −0.057]) and paragraph-packed 256-token chunks
are the best (+0.014, CI [0.009, 0.019]). Respecting paragraph boundaries helps;
overlap hurts at a fixed budget because it spends tokens on repeated text.

| Axis (dev, doc scope pooled with open scopes) | Effect on recall@1024 tokens | 95% CI |
|---|---|---|
| hybrid (BM25 + dense, RRF) vs dense only | **+0.025** | [0.022, 0.027] |
| bge-small vs bge-base | 0.000 | [−0.003, 0.004] |
| e5-base vs bge-base | −0.007 | [−0.011, −0.003] |
| MiniLM vs bge-base | −0.016 | [−0.020, −0.012] |
| whole corpus (title in query) vs within-paper | −0.225 | [−0.230, −0.220] |
| whole corpus (bare question) vs within-paper | −0.380 | [−0.384, −0.375] |

Two practical conclusions: the 33M-parameter bge-small is as good as the
110M bge-base here, so the deployed app uses the small one; and hybrid
retrieval is a free, consistent gain. Full tables:
`reports/retrieval/dev/ablation_table_recall_5.md` and
`ablation_table_recall_1024tok.md`.

## Finding 2: fine-tuning a 33M-parameter embedder beats every off-the-shelf model

`bge-small` was fine-tuned for two epochs with a contrastive loss on 3,560
(question, gold paragraph, BM25 hard negative) triples built from QASPER's
**train** split only (papers disjoint from dev and test; a test enforces it).
No synthetic data, no API calls, ~8 minutes on a laptop GPU. Weights:
[GFS418/bge-small-qasper-ft](https://huggingface.co/GFS418/bge-small-qasper-ft) on the Hugging Face Hub.

| Setting (dev, doc scope) | recall@1024 tokens, base | fine-tuned minus base [95% CI] |
|---|---|---|
| paragraph-256, dense | 0.592 | **+0.106** [+0.083, +0.131] |
| paragraph-256, hybrid | 0.591 | +0.075 [+0.055, +0.097] |
| fixed-128, dense | 0.578 | +0.099 [+0.073, +0.126] |
| fixed-512, dense | 0.458 | +0.117 [+0.087, +0.144] |
| paragraph-256, dense, whole corpus (title in query) | 0.328 | +0.060 [+0.035, +0.088] |

The lift is consistent across chunkings and scopes and replicates on the test
split (fixed-128 dense: +0.103 [+0.082, +0.122]). A side effect: once the dense
model is fine-tuned, **hybrid fusion hurts** (0.666 vs 0.698 dense-only on
dev), because reciprocal rank fusion pulls the better ranking toward BM25's.
Hybrid is a free win for off-the-shelf embeddings and a small loss for an
adapted one. Caveat: the gain includes adaptation to QASPER's question style,
which is exactly what a production system would also get from in-domain data,
but it should not be read as a general-purpose improvement to bge-small.

## Finding 3: the reranker looked great on 100 questions and mostly vanished on 880

A first pass on a 100-question sample showed the MiniLM cross-encoder adding
+0.095 recall@5. On the full dev split the paired estimate is +0.026 recall at
a 1,024-token budget with a CI of [−0.004, +0.054], and MRR +0.044
[+0.017, +0.071]: a real but small ranking improvement, at ~150 ms per query
versus ~0.1 ms without. The larger `bge-reranker-base` was clearly *worse*
(0.533 vs 0.591) and 1 s per query. And on top of the **fine-tuned** embedder
the MiniLM reranker is significantly harmful (−0.078 [−0.107, −0.051]): a
reranker caps retrieval at the reranker's own quality, so it only helps when
it is better than the first-stage model. On the test split the pattern holds:
+0.073 [+0.051, +0.097] on top of off-the-shelf bge-small, −0.075
[−0.098, −0.052] on top of the fine-tuned one. Confidence intervals are the
whole point.

## Finding 4: retrieval stopped being the bottleneck

Generation and judging ran on a fixed stratified sample of 400 test questions
(all 98 majority-unanswerable + 302 random answerable; the paid runs cost
$14.65 in total via the Batches API). Every arm answers the same questions, so
comparisons are paired.

| Arm (top-5 passages, 400 questions) | Correct, all answerable | Faithful (judge) | Hallucination rate | Abstains | Input tok/query | $/query |
|---|---|---|---|---|---|---|
| **Sonnet 5, fine-tuned retrieval** (primary) | **0.742** [0.692, 0.791] | 0.932 | 0.128 | 0.190 | 2,163 | 0.0033 |
| Sonnet 5, off-the-shelf retrieval (bge-small hybrid) | 0.709 [0.656, 0.758] | 0.924 | 0.133 | 0.207 | 2,186 | 0.0032 |
| Haiku 4.5, fine-tuned retrieval | 0.709 [0.656, 0.758] | 0.962 | **0.087** | 0.285 | 1,574 | 0.0011 |
| Sonnet 5, 512-token chunks (recall@5 "winner") | 0.745 [0.692, 0.791] | n/a | n/a | 0.205 | 4,275 | 0.0054 |
| Sonnet 5, oracle gold paragraphs (upper bound) | 0.762 [0.712, 0.808] | n/a | n/a | 0.124 | 1,085 | 0.0021 |
| Sonnet 5, closed-book (lower bound) | 0.175 [0.132, 0.219] | n/a | n/a | 0.698 | 571 | 0.0009 |

- **Fine-tuned vs off-the-shelf retrieval:** +0.15 recall became +3.3 points of
  correctness, paired diff on judged answers −0.023 [−0.069, +0.023] for the
  off-the-shelf arm, i.e. not significant at n = 400. With top-5 hit rates of
  0.86 vs 0.79 the generator usually has enough context either way.
- **Oracle headroom is +2 points.** Perfect retrieval would not fix most
  remaining errors; they sit in generation and in the judge's tolerance.
- **Big chunks buy nothing at 2x the cost.** The recall@5 winner from Finding 1
  matches the primary on correctness (+0.004 [−0.041, +0.052]) with twice the
  prompt tokens.
- **Closed-book is not zero.** 17.5% correct with no passages is the
  memorisation floor for 2018–2020 arXiv papers. But of the 32 primary-arm
  answers judged correct *despite* no gold chunk retrieved, closed-book got only
  4 right: the rest were answered from non-gold passages, which means the
  human evidence labels are not exhaustive and recall@k understates retrieval.

## Finding 5: hallucinations are mostly failures to abstain

| Arm | Answered an unanswerable question | Unsupported claim on an answerable one | Over-abstained on an answerable one |
|---|---|---|---|
| Sonnet 5, primary | 42 / 98 | 9 / 302 | 20 / 302 |
| Sonnet 5, off-the-shelf retrieval | 43 / 98 | 10 / 302 | 28 / 302 |
| Haiku 4.5, primary retrieval | **26 / 98** | 9 / 302 | 42 / 302 |

Given passages, the generators almost never invent unsupported claims (3%).
The dominant failure is answering a question the paper does not answer: Sonnet
5 does it 43% of the time, Haiku 4.5 27%, and Haiku pays for that caution with
more refusals on answerable questions. Abstention calibration, not grounding, is
where the next gain is. Note that "unanswerable" is the majority label of 2–6
annotators who disagreed 14% of the time, so some of these 42 are contestable.

## Finding 6: the free faithfulness instrument does not work

The NLI cross-encoder (`nli-deberta-v3-base`) called 17% of primary-arm answers
fully supported; the Opus 5 judge called 93%. Cohen's kappa between them is
0.030 [0.017, 0.047]; the NLI score ranks judge-faithful above judge-unfaithful
answers with AUC 0.72, so it carries some signal but has no usable operating
point. Its rate also tracks premise length (1.6% on 512-token chunks, 42% on
short oracle paragraphs), which is a property of the model, not of the answers.
Sentence-pair NLI does not transfer to multi-paragraph evidence. The judge is
therefore the only faithfulness number reported, and it is validated against
human labels below.

## Finding 7: the judge agrees with a human, and errs on the strict side

One human (the author) labelled 100 randomly sampled primary-arm answers,
blind to the judge's verdicts, under the same rubric the judge was prompted
with (`reports/labeling_rubric.md`, including the tie-break rules that came up).
Every claim was checked against the passages the model actually saw, ~300
claims in all.

| Comparison | Cohen's kappa [95% CI] | Agreement | n |
|---|---|---|---|
| Correctness, 3-class (correct / partial / incorrect): human vs Opus 5 judge | **0.73** [0.54, 0.88] | 0.92 | 100 |
| Correctness, correct-vs-not: human vs Opus 5 judge | **0.79** [0.59, 0.93] | 0.94 | 100 |
| Faithfulness: human vs Opus 5 judge | 0.66 (CI not estimable) | 0.98 | 100 |
| Faithfulness: human vs NLI cross-encoder | 0.01 [0.00, 0.03] | 0.23 | 100 |

- **Correctness.** Human and judge produced the same marginal distribution
  (83 correct each; 10 vs 12 partial; 7 vs 5 incorrect). All eight
  disagreements are one step apart on the scale, three in each direction
  between correct and partial and two where the human said incorrect and the
  judge said partial. No answer was called correct by one and incorrect by the
  other. The headline correctness rate would be identical under human grading.
- **Faithfulness.** Unsupported claims are rare enough (2% of claims by the
  human) that kappa is unstable; the informative statement is the count. Of 299
  claims the human found 2 unsupported; the judge found those 2 plus 3 more.
  So the same-family leniency concern did not materialise here: the judge is
  the stricter grader, and the 93% faithful figure is, if anything, a lower
  bound.
- **NLI.** Confirms Finding 6 against a human rather than against the judge:
  the cross-encoder calls 79% of human-faithful answers unsupported.

Caveat: a single labeller who also built the system, 100 items, and a
low base rate of unfaithfulness. The kappa CI is wide at the low end (0.54),
which is "moderate" agreement; the point estimate is "substantial".

## Repository

```
src/corpus/        corpus adapters (QASPER; FinanceBench planned) -> Document / Question
src/chunking.py    fixed-window and paragraph-packed chunkers, token-offset based
src/embed.py       embedding arms with a disk cache
src/index.py       exact cosine search, BM25, reciprocal rank fusion
src/rerank.py      cross-encoder reranker with a score cache
src/retrieval.py   RetrievalConfig + Retriever (one entry point for the grid)
src/generate.py    grounded generation prompt, structured output, claim citations
src/llm.py         Claude client, sqlite cache, Batches API runner, cost accounting
src/eval/          gold mapping, retrieval metrics, answer metrics, bootstrap stats
src/finetune/      hard-negative triples from the train split; contrastive fine-tuning
scripts/           run_retrieval, run_generation, run_judge, summarize_retrieval,
                   label_ui (human labelling), judge_agreement (kappa)
tests/             chunking, adapter, metrics, stats, index, generation, leakage guard
app.py             Streamlit demo (deployed; model + index from the HF Hub)
```

Reproduce: `pip install -r requirements-project.txt`; QASPER v0.3 downloads itself into
`data/raw/qasper/`, then `python scripts/run_retrieval.py --split dev ...`,
`python scripts/summarize_retrieval.py`, `python scripts/run_generation.py ...`,
`python scripts/run_judge.py ...`. Every model call is cached, so re-runs are free.

## Limitations

_(to be completed with the results)_

- The judge (Opus 5) and the generator (Sonnet 5 / Haiku 4.5) are from the same
  model family; same-family leniency is a known bias. The mitigation is the
  human-label agreement check, not a cure.
- QASPER papers (2018–2020) predate model training cutoffs. The closed-book
  baseline and the "retrieval missed but answer correct" cell of the error
  decomposition quantify how much memorisation contributes.
- Evidence labels are paragraph-level; chunk relevance is derived by token
  coverage under a threshold (default 0.5; sensitivity reported).
- Exact search in numpy rather than FAISS: identical results at this corpus
  size; FAISS was removed after an OpenMP runtime conflict with PyTorch on macOS.
- Generation and judge numbers are on a 400-question stratified sample of the
  test split (all unanswerable + 302 answerable), not all 1,451, to keep API
  spend under $15. Headline CIs are ±5 points; paired comparisons are tighter.
- The judge ran at low effort with a verdict-only schema to fit the budget.
- "Retrieval hit" uses human evidence labels that are not exhaustive; several
  answers judged correct came from unlabelled passages.
- Cost figures are Batches API list prices on 2026-09-09.
