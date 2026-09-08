# RAG Evaluation Harness — document intelligence over scientific papers

*Retrieval-augmented question answering with citations, and a quantitative
evaluation harness that says how often it is right, how often it hallucinates,
and which design choices matter, with confidence intervals.*

> **Status: in progress.** Retrieval grid and fine-tuning are running; generation
> and judge results land once the API key is configured. Numbers below are
> placeholders until then.

**The evaluation is the project; the RAG pipeline is table stakes.**

## Results at a glance

_(filled in from `reports/` when the test-split runs complete)_

- **Retrieval (test, scoped to the paper):** recall@5 = TBD [CI], best config = TBD
- **Answer quality (test):** judge-correct = TBD, faithful = TBD, hallucination rate = TBD
- **Baselines:** closed-book (no retrieval) correct = TBD; oracle context correct = TBD
- **Fine-tuned embedding:** +TBD recall@5 over the off-the-shelf model (paired bootstrap CI)
- **Judge validation:** Cohen's kappa vs 100 human labels = TBD

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
app.py             Streamlit demo
```

Reproduce: `pip install -r requirements.txt`, download QASPER v0.3 into
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
