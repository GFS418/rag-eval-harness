# Answer-quality ablation (test)

| run                                                                | mode        |   n |   abstain_rate |   token_f1 |   judge_correct |   judge_correct_or_partial |   judge_faithful |   nli_faithful |   judge_citation_precision |   hallucination_rate |   abstain_precision |   abstain_recall |   retrieval_hit |   mean_input_tokens |   cost_usd_total |   judge_cost_usd |
|:-------------------------------------------------------------------|:------------|----:|---------------:|-----------:|----------------:|---------------------------:|-----------------:|---------------:|---------------------------:|---------------------:|--------------------:|-----------------:|----------------:|--------------------:|-----------------:|-----------------:|
| haiku-4.5/rag__paragraph-256-0_bge-small-ft_dense_norerank_doc__k5 | rag         | 400 |          0.285 |      0.446 |           0.823 |                      0.927 |            0.962 |          0.273 |                      0.980 |                0.087 |               0.632 |            0.735 |           0.857 |            1573.570 |            0.000 |            4.458 |
| sonnet-5/closed-book__none__k5                                     | closed-book | 400 |          0.698 |      0.338 |           0.510 |                      0.788 |          nan     |        nan     |                    nan     |                0.043 |               0.290 |            0.827 |         nan     |             570.950 |            0.743 |            0.333 |
| sonnet-5/oracle__paragraph-256-0__k5                               | oracle      | 314 |          0.124 |      0.453 |           0.852 |                      0.915 |          nan     |          0.425 |                    nan     |                0.513 |               0.179 |            0.583 |           1.000 |            1084.946 |            1.327 |            0.841 |
| sonnet-5/rag__fixed-512-0_bge-small_hybrid_norerank_doc__k5        | rag         | 400 |          0.205 |      0.440 |           0.806 |                      0.910 |          nan     |          0.016 |                    nan     |                0.782 |               0.720 |            0.602 |           0.879 |            4275.485 |            4.341 |            0.907 |
| sonnet-5/rag__paragraph-256-0_bge-small-ft_dense_norerank_doc__k5  | rag         | 400 |          0.190 |      0.417 |           0.794 |                      0.911 |            0.932 |          0.173 |                      0.956 |                0.128 |               0.737 |            0.571 |           0.857 |            2163.233 |            2.621 |            5.178 |
| sonnet-5/rag__paragraph-256-0_bge-small_hybrid_norerank_doc__k5    | rag         | 400 |          0.207 |      0.413 |           0.781 |                      0.909 |            0.924 |          0.202 |                      0.953 |                0.133 |               0.663 |            0.561 |           0.793 |            2186.343 |            2.589 |            5.064 |

## Retrieval x correctness decomposition (answerable, judged)

| run                                                                |   hit=True|correct=1.0 |   hit=True|correct=0.0 |   hit=False|correct=1.0 |   hit=False|correct=0.0 |
|:-------------------------------------------------------------------|-----------------------:|-----------------------:|------------------------:|------------------------:|
| haiku-4.5/rag__paragraph-256-0_bge-small-ft_dense_norerank_doc__k5 |                    194 |                     37 |                      20 |                       9 |
| sonnet-5/oracle__paragraph-256-0__k5                               |                    230 |                     40 |                       0 |                       0 |
| sonnet-5/rag__fixed-512-0_bge-small_hybrid_norerank_doc__k5        |                    207 |                     43 |                      18 |                      11 |
| sonnet-5/rag__paragraph-256-0_bge-small-ft_dense_norerank_doc__k5  |                    203 |                     47 |                      21 |                      11 |
| sonnet-5/rag__paragraph-256-0_bge-small_hybrid_norerank_doc__k5    |                    191 |                     38 |                      23 |                      22 |

## Paired differences vs `sonnet-5/rag__paragraph-256-0_bge-small-ft_dense_norerank_doc__k5` (95% bootstrap CI over the same questions)

| run                                                                |   n | judge_correct           | token_f1                | judge_faithful          | nli_faithful            | hallucinated            |
|:-------------------------------------------------------------------|----:|:------------------------|:------------------------|:------------------------|:------------------------|:------------------------|
| haiku-4.5/rag__paragraph-256-0_bge-small-ft_dense_norerank_doc__k5 | 400 | +0.008 [-0.039, +0.051] | +0.024 [-0.001, +0.048] | +0.007 [-0.018, +0.035] | +0.085 [+0.032, +0.138] | -0.040 [-0.068, -0.015] |
| sonnet-5/closed-book__none__k5                                     | 400 | -0.296 [-0.398, -0.204] | -0.124 [-0.170, -0.082] | nan                     | nan                     | -0.085 [-0.120, -0.050] |
| sonnet-5/oracle__paragraph-256-0__k5                               | 314 | +0.031 [-0.019, +0.081] | +0.033 [+0.010, +0.058] | nan                     | +0.242 [+0.174, +0.309] | +0.459 [+0.398, +0.516] |
| sonnet-5/rag__fixed-512-0_bge-small_hybrid_norerank_doc__k5        | 400 | +0.004 [-0.041, +0.052] | +0.018 [-0.005, +0.043] | nan                     | -0.164 [-0.207, -0.120] | +0.655 [+0.600, +0.703] |
| sonnet-5/rag__paragraph-256-0_bge-small_hybrid_norerank_doc__k5    | 400 | -0.023 [-0.069, +0.023] | +0.002 [-0.021, +0.025] | +0.000 [-0.031, +0.034] | +0.027 [-0.020, +0.075] | +0.005 [-0.025, +0.035] |

## Headline CIs for `sonnet-5/rag__paragraph-256-0_bge-small-ft_dense_norerank_doc__k5`

- judge_correct: 0.794 [0.745, 0.837] (n=282)
- judge_faithful: 0.932 [0.904, 0.957] (n=324)
- nli_faithful: 0.173 [0.133, 0.216] (n=324)
- hallucinated: 0.128 [0.095, 0.163] (n=400)
- abstain: 0.190 [0.152, 0.230] (n=400)