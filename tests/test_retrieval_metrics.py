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
