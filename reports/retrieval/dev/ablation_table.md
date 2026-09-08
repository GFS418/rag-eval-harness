# Retrieval ablation (dev), sorted by recall@5

n = 100 answerable questions with gold chunks; 95% bootstrap CIs.

## Top 15 configs

| chunk       | embedder   | lexical   | scope       |   recall@1 |   recall@5 |   recall@5_lo |   recall@5_hi |   recall@10 |   mrr |   ndcg@10 |   ms_per_query |
|:------------|:-----------|:----------|:------------|-----------:|-----------:|--------------:|--------------:|------------:|------:|----------:|---------------:|
| fixed-256-0 | bge-small  | dense     | doc         |      0.272 |      0.651 |         0.571 |         0.729 |       0.833 | 0.576 |     0.601 |        134.110 |
| fixed-256-0 | bge-small  | hybrid    | doc         |      0.233 |      0.626 |         0.597 |         0.654 |       0.843 | 0.497 |     0.555 |          0.061 |
| fixed-256-0 | bge-small  | dense     | doc         |      0.204 |      0.609 |         0.579 |         0.638 |       0.846 | 0.472 |     0.537 |          0.018 |
| fixed-256-0 | bge-small  | dense     | doc         |      0.205 |      0.505 |         0.426 |         0.587 |       0.794 | 0.491 |     0.516 |        768.831 |
| fixed-128-0 | bge-small  | dense     | doc         |      0.152 |      0.433 |         0.402 |         0.462 |       0.644 | 0.379 |     0.403 |          0.027 |
| fixed-256-0 | bge-small  | hybrid    | open-titled |      0.104 |      0.366 |         0.338 |         0.394 |       0.531 | 0.290 |     0.317 |          0.354 |
| fixed-256-0 | bge-small  | dense     | open-titled |      0.109 |      0.357 |         0.329 |         0.385 |       0.501 | 0.287 |     0.308 |          0.128 |
| fixed-256-0 | bge-small  | hybrid    | open        |      0.080 |      0.193 |         0.169 |         0.217 |       0.257 | 0.173 |     0.173 |          0.302 |
| fixed-256-0 | bge-small  | dense     | open        |      0.054 |      0.148 |         0.126 |         0.171 |       0.208 | 0.130 |     0.133 |          0.146 |

## Marginal effect of each axis (paired over all other settings)

| axis    | level            | vs          |   n_pairs |   n_questions |   mean_diff_recall@5 |     lo |     hi | significant   |
|:--------|:-----------------|:------------|----------:|--------------:|---------------------:|-------:|-------:|:--------------|
| chunk   | fixed-256-0      | fixed-128-0 |         1 |           819 |                0.177 |  0.147 |  0.209 | True          |
| lexical | hybrid           | dense       |         3 |          2640 |                0.024 |  0.012 |  0.036 | True          |
| rerank  | rerank           | norerank    |         1 |           100 |               -0.052 | -0.150 |  0.043 | False         |
| rerank  | rerank-minilm-ce | norerank    |         1 |           100 |                0.095 |  0.009 |  0.181 | True          |
| scope   | open             | doc         |         2 |          1760 |               -0.447 | -0.466 | -0.424 | True          |
| scope   | open-titled      | doc         |         2 |          1760 |               -0.256 | -0.281 | -0.231 | True          |

## Best config per scope

| scope       | chunk       | embedder   | lexical   |   recall@5 |   recall@5_lo |   recall@5_hi |   mrr |
|:------------|:------------|:-----------|:----------|-----------:|--------------:|--------------:|------:|
| doc         | fixed-256-0 | bge-small  | dense     |      0.651 |         0.571 |         0.729 | 0.576 |
| doc         | fixed-256-0 | bge-small  | hybrid    |      0.626 |         0.597 |         0.654 | 0.497 |
| doc         | fixed-256-0 | bge-small  | dense     |      0.609 |         0.579 |         0.638 | 0.472 |
| open-titled | fixed-256-0 | bge-small  | hybrid    |      0.366 |         0.338 |         0.394 | 0.290 |
| open-titled | fixed-256-0 | bge-small  | dense     |      0.357 |         0.329 |         0.385 | 0.287 |
| open        | fixed-256-0 | bge-small  | hybrid    |      0.193 |         0.169 |         0.217 | 0.173 |
| open        | fixed-256-0 | bge-small  | dense     |      0.148 |         0.126 |         0.171 | 0.130 |