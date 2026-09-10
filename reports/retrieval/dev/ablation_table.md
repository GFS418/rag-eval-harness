# Retrieval ablation (dev), sorted by recall@5

n = 886 answerable questions with gold chunks; 95% bootstrap CIs.

## Top 15 configs

| chunk           | embedder   | lexical   | scope   |   recall@1 |   recall@5 |   recall@5_lo |   recall@5_hi |   recall@10 |   mrr |   ndcg@10 |   ms_per_query |
|:----------------|:-----------|:----------|:--------|-----------:|-----------:|--------------:|--------------:|------------:|------:|----------:|---------------:|
| fixed-512-0     | bge-small  | hybrid    | doc     |      0.294 |      0.809 |         0.785 |         0.832 |       0.976 | 0.585 |     0.667 |          0.048 |
| fixed-512-0     | bge-base   | hybrid    | doc     |      0.290 |      0.808 |         0.785 |         0.830 |       0.966 | 0.576 |     0.661 |          0.049 |
| fixed-512-0     | e5-base    | hybrid    | doc     |      0.270 |      0.803 |         0.781 |         0.825 |       0.973 | 0.561 |     0.650 |          0.049 |
| fixed-512-0     | minilm     | hybrid    | doc     |      0.263 |      0.792 |         0.769 |         0.816 |       0.971 | 0.551 |     0.645 |          0.048 |
| paragraph-512-0 | bge-small  | hybrid    | doc     |      0.297 |      0.776 |         0.752 |         0.800 |       0.964 | 0.578 |     0.659 |          0.050 |
| fixed-512-0     | bge-base   | dense     | doc     |      0.270 |      0.776 |         0.752 |         0.800 |       0.974 | 0.555 |     0.643 |          0.015 |
| paragraph-512-0 | bge-base   | hybrid    | doc     |      0.295 |      0.775 |         0.750 |         0.797 |       0.954 | 0.581 |     0.656 |          0.052 |
| paragraph-512-0 | e5-base    | hybrid    | doc     |      0.286 |      0.775 |         0.751 |         0.798 |       0.961 | 0.575 |     0.653 |          0.051 |
| fixed-512-0     | bge-small  | dense     | doc     |      0.271 |      0.772 |         0.748 |         0.796 |       0.973 | 0.558 |     0.646 |          0.015 |
| paragraph-512-0 | e5-base    | dense     | doc     |      0.277 |      0.771 |         0.747 |         0.794 |       0.947 | 0.560 |     0.638 |          0.022 |
| fixed-512-0     | e5-base    | dense     | doc     |      0.245 |      0.764 |         0.741 |         0.788 |       0.964 | 0.524 |     0.621 |          0.017 |
| paragraph-512-0 | bge-small  | dense     | doc     |      0.274 |      0.760 |         0.736 |         0.784 |       0.952 | 0.552 |     0.635 |          0.016 |
| paragraph-512-0 | minilm     | hybrid    | doc     |      0.286 |      0.756 |         0.730 |         0.780 |       0.948 | 0.572 |     0.644 |          0.050 |
| paragraph-512-0 | bge-base   | dense     | doc     |      0.271 |      0.751 |         0.725 |         0.774 |       0.953 | 0.554 |     0.634 |          0.023 |
| fixed-512-0     | minilm     | dense     | doc     |      0.225 |      0.746 |         0.721 |         0.773 |       0.947 | 0.508 |     0.604 |          0.015 |

## Marginal effect of each axis (paired over all other settings)

| axis     | level           | vs          |   n_pairs |   n_questions |   mean_diff_recall@5 |     lo |     hi | significant   |
|:---------|:----------------|:------------|----------:|--------------:|---------------------:|-------:|-------:|:--------------|
| chunk    | fixed-256-0     | fixed-128-0 |        24 |         19656 |                0.115 |  0.109 |  0.120 | True          |
| chunk    | fixed-256-32    | fixed-128-0 |        24 |         19656 |                0.090 |  0.084 |  0.095 | True          |
| chunk    | fixed-512-0     | fixed-128-0 |        24 |         19656 |                0.233 |  0.227 |  0.240 | True          |
| chunk    | paragraph-256-0 | fixed-128-0 |        24 |         19656 |                0.102 |  0.096 |  0.107 | True          |
| chunk    | paragraph-512-0 | fixed-128-0 |        24 |         19656 |                0.220 |  0.214 |  0.226 | True          |
| embedder | bge-small       | bge-base    |        36 |         31386 |                0.001 | -0.002 |  0.005 | False         |
| embedder | e5-base         | bge-base    |        36 |         31386 |               -0.008 | -0.011 | -0.004 | True          |
| embedder | minilm          | bge-base    |        36 |         31386 |               -0.023 | -0.027 | -0.020 | True          |
| lexical  | hybrid          | dense       |        72 |         62772 |                0.034 |  0.032 |  0.037 | True          |
| scope    | open            | doc         |        48 |         41848 |               -0.465 | -0.469 | -0.460 | True          |
| scope    | open-titled     | doc         |        48 |         41848 |               -0.252 | -0.256 | -0.247 | True          |

## Best config per scope

| scope       | chunk           | embedder   | lexical   |   recall@5 |   recall@5_lo |   recall@5_hi |   mrr |
|:------------|:----------------|:-----------|:----------|-----------:|--------------:|--------------:|------:|
| doc         | fixed-512-0     | bge-small  | hybrid    |      0.809 |         0.785 |         0.832 | 0.585 |
| doc         | fixed-512-0     | bge-base   | hybrid    |      0.808 |         0.785 |         0.830 | 0.576 |
| doc         | fixed-512-0     | e5-base    | hybrid    |      0.803 |         0.781 |         0.825 | 0.561 |
| open-titled | fixed-512-0     | bge-small  | hybrid    |      0.568 |         0.537 |         0.596 | 0.397 |
| open-titled | fixed-512-0     | bge-base   | hybrid    |      0.557 |         0.526 |         0.587 | 0.389 |
| open-titled | fixed-512-0     | e5-base    | hybrid    |      0.541 |         0.510 |         0.569 | 0.393 |
| open        | fixed-512-0     | e5-base    | hybrid    |      0.232 |         0.208 |         0.259 | 0.179 |
| open        | fixed-512-0     | bge-base   | hybrid    |      0.232 |         0.206 |         0.259 | 0.191 |
| open        | paragraph-512-0 | e5-base    | hybrid    |      0.231 |         0.205 |         0.258 | 0.193 |