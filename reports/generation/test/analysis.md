# Secondary analyses (test)

## Hallucination decomposition (arms with the Opus faithfulness judge)

| run                                                                | answered an unanswerable   | unfaithful on answerable   | over-abstained on answerable   |   hallucination rate |
|:-------------------------------------------------------------------|:---------------------------|:---------------------------|:-------------------------------|---------------------:|
| haiku-4.5/rag__paragraph-256-0_bge-small-ft_dense_norerank_doc__k5 | 26/98                      | 9/302                      | 42/302                         |                0.087 |
| sonnet-5/rag__paragraph-256-0_bge-small-ft_dense_norerank_doc__k5  | 42/98                      | 9/302                      | 20/302                         |                0.128 |
| sonnet-5/rag__paragraph-256-0_bge-small_hybrid_norerank_doc__k5    | 43/98                      | 10/302                     | 28/302                         |                0.133 |

## Correctness over ALL answerable questions (abstaining counts as not correct)

| run                                                                |   n | correct                      |
|:-------------------------------------------------------------------|----:|:-----------------------------|
| haiku-4.5/rag__paragraph-256-0_bge-small-ft_dense_norerank_doc__k5 | 302 | 0.709 [0.656, 0.758] (n=302) |
| sonnet-5/closed-book__none__k5                                     | 302 | 0.175 [0.132, 0.219] (n=302) |
| sonnet-5/oracle__paragraph-256-0__k5                               | 302 | 0.762 [0.712, 0.808] (n=302) |
| sonnet-5/rag__fixed-512-0_bge-small_hybrid_norerank_doc__k5        | 302 | 0.745 [0.692, 0.791] (n=302) |
| sonnet-5/rag__paragraph-256-0_bge-small-ft_dense_norerank_doc__k5  | 302 | 0.742 [0.692, 0.791] (n=302) |
| sonnet-5/rag__paragraph-256-0_bge-small_hybrid_norerank_doc__k5    | 302 | 0.709 [0.656, 0.758] (n=302) |

## NLI cross-encoder vs Opus judge on faithfulness

| run                                                                |   n |   judge faithful |   NLI faithful | kappa                 |   AUC of NLI score for judge label |
|:-------------------------------------------------------------------|----:|-----------------:|---------------:|:----------------------|-----------------------------------:|
| haiku-4.5/rag__paragraph-256-0_bge-small-ft_dense_norerank_doc__k5 | 286 |            0.962 |          0.273 | 0.020 [-0.003, 0.041] |                              0.642 |
| sonnet-5/rag__paragraph-256-0_bge-small-ft_dense_norerank_doc__k5  | 324 |            0.932 |          0.173 | 0.030 [0.017, 0.047]  |                              0.723 |
| sonnet-5/rag__paragraph-256-0_bge-small_hybrid_norerank_doc__k5    | 317 |            0.924 |          0.202 | 0.032 [0.012, 0.056]  |                              0.785 |

## Cost per query (Batches API prices) and prompt size

| run                                                                |   usd per query |   input tokens |   output tokens |
|:-------------------------------------------------------------------|----------------:|---------------:|----------------:|
| haiku-4.5/rag__paragraph-256-0_bge-small-ft_dense_norerank_doc__k5 |          0.0011 |      1573.5700 |        128.3325 |
| sonnet-5/closed-book__none__k5                                     |          0.0009 |       570.9500 |         71.5225 |
| sonnet-5/oracle__paragraph-256-0__k5                               |          0.0021 |      1084.9459 |        205.4650 |
| sonnet-5/rag__fixed-512-0_bge-small_hybrid_norerank_doc__k5        |          0.0054 |      4275.4850 |        230.1275 |
| sonnet-5/rag__paragraph-256-0_bge-small-ft_dense_norerank_doc__k5  |          0.0033 |      2163.2325 |        222.5150 |
| sonnet-5/rag__paragraph-256-0_bge-small_hybrid_norerank_doc__k5    |          0.0032 |      2186.3425 |        210.0425 |

## Memorisation check

Primary arm: 21 of 32 answered questions where no gold chunk was retrieved were still judged correct. Closed-book got 4 of those same questions right. The rest were answered from non-gold passages, i.e. the evidence labels are not exhaustive and recall@k understates retrieval.
