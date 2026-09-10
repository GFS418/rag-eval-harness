import math

import pytest

from eval.retrieval_metrics import all_metrics, hit_at_k, mrr, ndcg_at_k, recall_at_k


def test_basic_values():
    ranked = ["a", "b", "c", "d"]
    gold = {"b", "d", "z"}
    assert hit_at_k(ranked, gold, 1) == 0.0
    assert hit_at_k(ranked, gold, 2) == 1.0
    assert recall_at_k(ranked, gold, 4) == pytest.approx(2 / 3)
    assert mrr(ranked, gold) == 0.5
    assert mrr(ranked, gold, k=1) == 0.0


def test_ndcg_perfect_and_worst():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == 1.0
    assert ndcg_at_k(["x", "y"], {"a"}, 2) == 0.0
    # single gold at rank 2 of 2
    assert ndcg_at_k(["x", "a"], {"a"}, 2) == pytest.approx((1 / math.log2(3)) / 1.0)


def test_empty_gold_raises():
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), 1)


def test_all_metrics_keys():
    m = all_metrics(["a"], {"a"}, ks=(1, 5))
    assert set(m) == {"mrr", "hit@1", "recall@1", "ndcg@1", "hit@5", "recall@5", "ndcg@5"}


def test_budget_metrics():
    from eval.retrieval_metrics import hit_at_budget, recall_at_budget, within_budget

    toks = {"a": 300, "b": 300, "c": 300, "d": 300}
    assert within_budget(["a", "b", "c"], toks, 650) == ["a", "b"]
    assert within_budget(["a"], toks, 10) == ["a"]  # always at least one
    assert recall_at_budget(["a", "b", "c"], {"b", "c"}, toks, 650) == 0.5
    assert hit_at_budget(["a", "b", "c"], {"c"}, toks, 650) == 0.0
    assert hit_at_budget(["a", "b", "c"], {"c"}, toks, 900) == 1.0
