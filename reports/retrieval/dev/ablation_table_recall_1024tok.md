# Retrieval ablation (dev), sorted by recall@1024tok

n = 880 answerable questions with gold chunks; 95% bootstrap CIs.

## Top 15 configs

| chunk           | embedder   | lexical   | scope   |   recall@1 |   recall@5 |   recall@1024tok_lo |   recall@1024tok_hi |   recall@10 |   mrr |   recall@512tok |   recall@1024tok |   recall@2048tok |   ms_per_query |
|:----------------|:-----------|:----------|:--------|-----------:|-----------:|--------------------:|--------------------:|------------:|------:|----------------:|-----------------:|-----------------:|---------------:|
| paragraph-256-0 | bge-base   | hybrid    | doc     |      0.248 |      0.612 |               0.564 |               0.621 |       0.802 | 0.509 |           0.384 |            0.593 |            0.795 |          0.234 |
| paragraph-256-0 | bge-small  | dense     | doc     |      0.221 |      0.607 |               0.561 |               0.620 |       0.795 | 0.488 |           0.369 |            0.592 |            0.794 |          0.070 |
| paragraph-256-0 | bge-small  | hybrid    | doc     |      0.250 |      0.616 |               0.562 |               0.619 |       0.812 | 0.516 |           0.392 |            0.591 |            0.814 |          0.187 |
| paragraph-256-0 | e5-base    | hybrid    | doc     |      0.226 |      0.611 |               0.556 |               0.613 |       0.806 | 0.497 |           0.393 |            0.585 |            0.801 |          0.224 |
| paragraph-256-0 | minilm     | hybrid    | doc     |      0.232 |      0.606 |               0.555 |               0.614 |       0.809 | 0.499 |           0.381 |            0.585 |            0.808 |          0.173 |
| paragraph-256-0 | e5-base    | dense     | doc     |      0.230 |      0.593 |               0.551 |               0.609 |       0.794 | 0.494 |           0.373 |            0.581 |            0.786 |          0.076 |
| fixed-128-0     | bge-small  | hybrid    | doc     |      0.162 |      0.450 |               0.551 |               0.609 |       0.657 | 0.391 |           0.399 |            0.580 |            0.784 |          0.251 |
| fixed-128-0     | bge-small  | dense     | doc     |      0.152 |      0.433 |               0.547 |               0.605 |       0.644 | 0.379 |           0.381 |            0.578 |            0.785 |          0.068 |
| fixed-128-0     | e5-base    | hybrid    | doc     |      0.152 |      0.452 |               0.542 |               0.600 |       0.636 | 0.380 |           0.386 |            0.571 |            0.778 |          0.241 |
| fixed-128-0     | bge-base   | dense     | doc     |      0.162 |      0.450 |               0.539 |               0.599 |       0.627 | 0.400 |           0.399 |            0.569 |            0.787 |          0.089 |
| fixed-128-0     | e5-base    | dense     | doc     |      0.139 |      0.432 |               0.536 |               0.596 |       0.627 | 0.362 |           0.373 |            0.568 |            0.772 |          0.110 |
| fixed-256-0     | minilm     | hybrid    | doc     |      0.216 |      0.642 |               0.538 |               0.597 |       0.848 | 0.487 |           0.360 |            0.567 |            0.774 |          0.196 |
| fixed-128-0     | bge-base   | hybrid    | doc     |      0.164 |      0.452 |               0.536 |               0.596 |       0.639 | 0.396 |           0.393 |            0.567 |            0.793 |          0.250 |
| paragraph-256-0 | bge-base   | dense     | doc     |      0.230 |      0.574 |               0.533 |               0.593 |       0.788 | 0.484 |           0.369 |            0.564 |            0.787 |          0.110 |
| paragraph-256-0 | minilm     | dense     | doc     |      0.190 |      0.571 |               0.534 |               0.594 |       0.775 | 0.454 |           0.348 |            0.563 |            0.778 |          0.077 |

## Marginal effect of each axis (paired over all other settings)

| axis     | level           | vs          |   n_pairs |   n_questions |   mean_diff_recall@1024tok |     lo |     hi | significant   |
|:---------|:----------------|:------------|----------:|--------------:|---------------------------:|-------:|-------:|:--------------|
| chunk    | fixed-256-0     | fixed-128-0 |        24 |         19656 |                     -0.008 | -0.013 | -0.003 | True          |
| chunk    | fixed-256-32    | fixed-128-0 |        24 |         19656 |                     -0.032 | -0.038 | -0.027 | True          |
| chunk    | fixed-512-0     | fixed-128-0 |        24 |         19656 |                     -0.062 | -0.068 | -0.057 | True          |
| chunk    | paragraph-256-0 | fixed-128-0 |        24 |         19656 |                      0.014 |  0.009 |  0.019 | True          |
| chunk    | paragraph-512-0 | fixed-128-0 |        24 |         19656 |                     -0.055 | -0.061 | -0.049 | True          |
| embedder | bge-small       | bge-base    |        36 |         31386 |                      0.000 | -0.003 |  0.004 | False         |
| embedder | e5-base         | bge-base    |        36 |         31386 |                     -0.007 | -0.011 | -0.003 | True          |
| embedder | minilm          | bge-base    |        36 |         31386 |                     -0.016 | -0.020 | -0.012 | True          |
| lexical  | hybrid          | dense       |        72 |         62772 |                      0.025 |  0.022 |  0.027 | True          |
| scope    | open            | doc         |        48 |         41848 |                     -0.380 | -0.384 | -0.375 | True          |
| scope    | open-titled     | doc         |        48 |         41848 |                     -0.225 | -0.230 | -0.220 | True          |

## Best config per scope

| scope       | chunk           | embedder   | lexical   |   recall@5 |   recall@1024tok_lo |   recall@1024tok_hi |   mrr |   recall@1024tok |
|:------------|:----------------|:-----------|:----------|-----------:|--------------------:|--------------------:|------:|-----------------:|
| doc         | paragraph-256-0 | bge-base   | hybrid    |      0.612 |               0.564 |               0.621 | 0.509 |            0.593 |
| doc         | paragraph-256-0 | bge-small  | dense     |      0.607 |               0.561 |               0.620 | 0.488 |            0.592 |
| doc         | paragraph-256-0 | bge-small  | hybrid    |      0.616 |               0.562 |               0.619 | 0.516 |            0.591 |
| open-titled | paragraph-256-0 | bge-small  | dense     |      0.340 |               0.300 |               0.355 | 0.289 |            0.328 |
| open-titled | paragraph-256-0 | e5-base    | hybrid    |      0.339 |               0.296 |               0.352 | 0.277 |            0.324 |
| open-titled | paragraph-256-0 | bge-small  | hybrid    |      0.344 |               0.294 |               0.348 | 0.284 |            0.322 |
| open        | paragraph-256-0 | e5-base    | hybrid    |      0.201 |               0.168 |               0.215 | 0.180 |            0.191 |
| open        | paragraph-256-0 | bge-small  | hybrid    |      0.196 |               0.164 |               0.210 | 0.180 |            0.187 |
| open        | fixed-128-0     | bge-base   | hybrid    |      0.146 |               0.157 |               0.204 | 0.131 |            0.180 |