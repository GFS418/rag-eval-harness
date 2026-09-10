# Retrieval ablation (test), sorted by recall@1024tok

n = 1297 answerable questions with gold chunks; 95% bootstrap CIs.

## Top 15 configs

| chunk           | embedder     | lexical   | scope   |   recall@1 |   recall@5 |   recall@1024tok_lo |   recall@1024tok_hi |   recall@10 |   mrr |   recall@512tok |   recall@1024tok |   recall@2048tok |   ms_per_query |
|:----------------|:-------------|:----------|:--------|-----------:|-----------:|--------------------:|--------------------:|------------:|------:|----------------:|-----------------:|-----------------:|---------------:|
| paragraph-256-0 | bge-small-ft | dense     | doc     |      0.339 |      0.725 |               0.692 |               0.733 |       0.872 | 0.658 |           0.509 |            0.713 |            0.874 |          0.044 |
| fixed-128-0     | bge-small-ft | dense     | doc     |      0.252 |      0.598 |               0.681 |               0.722 |       0.757 | 0.561 |           0.545 |            0.701 |            0.864 |          0.062 |
| paragraph-256-0 | bge-small-ft | hybrid    | doc     |      0.303 |      0.688 |               0.647 |               0.691 |       0.862 | 0.619 |           0.463 |            0.669 |            0.863 |          0.099 |
| fixed-128-0     | bge-small-ft | hybrid    | doc     |      0.218 |      0.543 |               0.642 |               0.684 |       0.724 | 0.511 |           0.481 |            0.664 |            0.852 |          0.266 |
| paragraph-256-0 | bge-small    | hybrid    | doc     |      0.226 |      0.622 |               0.577 |               0.622 |       0.813 | 0.529 |           0.371 |            0.600 |            0.808 |          0.072 |
| fixed-128-0     | bge-small    | dense     | doc     |      0.176 |      0.467 |               0.576 |               0.622 |       0.657 | 0.442 |           0.419 |            0.599 |            0.790 |          0.041 |
| fixed-128-0     | bge-small    | hybrid    | doc     |      0.171 |      0.473 |               0.571 |               0.615 |       0.659 | 0.437 |           0.414 |            0.593 |            0.797 |          0.139 |
| fixed-512-0     | bge-small-ft | dense     | doc     |      0.377 |      0.838 |               0.545 |               0.590 |       0.978 | 0.694 |           0.377 |            0.568 |            0.783 |          0.021 |
| paragraph-256-0 | bge-small    | dense     | doc     |      0.237 |      0.585 |               0.541 |               0.587 |       0.791 | 0.532 |           0.376 |            0.564 |            0.786 |          0.037 |
| fixed-256-0     | bge-small    | hybrid    | doc     |      0.227 |      0.639 |               0.536 |               0.582 |       0.845 | 0.524 |           0.367 |            0.559 |            0.781 |          0.087 |
| fixed-256-0     | bge-small    | dense     | doc     |      0.236 |      0.619 |               0.534 |               0.580 |       0.836 | 0.528 |           0.359 |            0.558 |            0.760 |          0.028 |
| fixed-512-0     | bge-small-ft | hybrid    | doc     |      0.358 |      0.847 |               0.523 |               0.569 |       0.981 | 0.675 |           0.358 |            0.546 |            0.783 |          0.060 |
| fixed-256-32    | bge-small    | hybrid    | doc     |      0.214 |      0.588 |               0.506 |               0.550 |       0.810 | 0.546 |           0.356 |            0.529 |            0.737 |          0.069 |
| fixed-256-32    | bge-small    | dense     | doc     |      0.204 |      0.588 |               0.497 |               0.541 |       0.791 | 0.532 |           0.338 |            0.520 |            0.722 |          0.029 |
| fixed-512-0     | bge-small    | hybrid    | doc     |      0.308 |      0.801 |               0.471 |               0.519 |       0.966 | 0.621 |           0.308 |            0.495 |            0.724 |          0.062 |

## Marginal effect of each axis (paired over all other settings)

| axis     | level           | vs          |   n_pairs |   n_questions |   mean_diff_recall@1024tok |     lo |     hi | significant   |
|:---------|:----------------|:------------|----------:|--------------:|---------------------------:|-------:|-------:|:--------------|
| chunk    | fixed-256-0     | fixed-128-0 |         4 |          4964 |                     -0.003 | -0.014 |  0.008 | False         |
| chunk    | fixed-256-32    | fixed-128-0 |         4 |          4964 |                     -0.033 | -0.044 | -0.022 | True          |
| chunk    | fixed-512-0     | fixed-128-0 |         8 |          9928 |                     -0.049 | -0.058 | -0.039 | True          |
| chunk    | paragraph-256-0 | fixed-128-0 |         8 |          9928 |                      0.015 |  0.007 |  0.022 | True          |
| chunk    | paragraph-512-0 | fixed-128-0 |         4 |          4964 |                     -0.065 | -0.077 | -0.053 | True          |
| embedder | bge-small-ft    | bge-small   |        12 |         15340 |                      0.070 |  0.065 |  0.075 | True          |
| lexical  | hybrid          | dense       |        18 |         23122 |                      0.005 |  0.001 |  0.009 | True          |
| scope    | open-titled     | doc         |        18 |         23122 |                     -0.272 | -0.278 | -0.265 | True          |

## Best config per scope

| scope       | chunk           | embedder     | lexical   |   recall@5 |   recall@1024tok_lo |   recall@1024tok_hi |   mrr |   recall@1024tok |
|:------------|:----------------|:-------------|:----------|-----------:|--------------------:|--------------------:|------:|-----------------:|
| doc         | paragraph-256-0 | bge-small-ft | dense     |      0.725 |               0.692 |               0.733 | 0.658 |            0.713 |
| doc         | fixed-128-0     | bge-small-ft | dense     |      0.598 |               0.681 |               0.722 | 0.561 |            0.701 |
| doc         | paragraph-256-0 | bge-small-ft | hybrid    |      0.688 |               0.647 |               0.691 | 0.619 |            0.669 |
| open-titled | paragraph-256-0 | bge-small-ft | hybrid    |      0.380 |               0.336 |               0.381 | 0.337 |            0.359 |
| open-titled | paragraph-256-0 | bge-small-ft | dense     |      0.364 |               0.330 |               0.374 | 0.349 |            0.352 |
| open-titled | fixed-512-0     | bge-small-ft | hybrid    |      0.559 |               0.322 |               0.367 | 0.473 |            0.344 |