# Judge validation (rag__paragraph-256-0_bge-small-ft_dense_norerank_doc__k5)

| comparison                               |   kappa |   kappa_lo |   kappa_hi |   agreement |   n |
|:-----------------------------------------|--------:|-----------:|-----------:|------------:|----:|
| correctness 3-class: human vs Opus judge |   0.729 |      0.535 |      0.881 |       0.920 | 100 |
| correct-vs-not: human vs Opus judge      |   0.787 |      0.589 |      0.929 |       0.940 | 100 |
| faithfulness: human vs Opus judge        |   0.658 |      0.000 |      1.000 |       0.980 | 100 |
| faithfulness: human vs NLI               |   0.011 |      0.000 |      0.031 |       0.230 | 100 |

## Disagreement analysis

Correctness (8 of 100): all one step apart on the correct/partial/incorrect
scale. human=partial, judge=correct: 3 (answers covering part of a list or one
of several methods). human=correct, judge=partial: 3 (judge penalised extra or
differently-sliced detail). human=incorrect, judge=partial: 2 (the judge gave
credit for being on-topic where the human required the specific content).
Marginals are identical for "correct" (83 each).

Faithfulness (2 of 100): both are human=faithful, judge=unfaithful, one claim
of two in each case. At claim level the human marked 2 of 299 claims
unsupported and the judge marked 5, including both of the human's. The judge
is the stricter instrument.
