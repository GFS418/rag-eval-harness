"""Per-question retrieval metrics. Aggregation and CIs live in ``eval.stats``.

All metrics take a ranked list of chunk ids and the gold set. Questions with an
empty gold set are the caller's problem (they are excluded upstream and
counted in the report), so an empty gold set here raises.
"""
from __future__ import annotations

import math
from collections.abc import Collection, Sequence


def hit_at_k(ranked: Sequence[str], gold: Collection[str], k: int) -> float:
    _check(gold)
    return float(any(r in gold for r in ranked[:k]))


def recall_at_k(ranked: Sequence[str], gold: Collection[str], k: int) -> float:
    _check(gold)
    return sum(r in gold for r in ranked[:k]) / len(gold)


def mrr(ranked: Sequence[str], gold: Collection[str], k: int | None = None) -> float:
    _check(gold)
    top = ranked if k is None else ranked[:k]
    for i, r in enumerate(top):
        if r in gold:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gold: Collection[str], k: int) -> float:
    _check(gold)
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(ranked[:k]) if r in gold)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal


def all_metrics(ranked: Sequence[str], gold: Collection[str],
                ks: Sequence[int] = (1, 3, 5, 10)) -> dict[str, float]:
    out: dict[str, float] = {"mrr": mrr(ranked, gold)}
    for k in ks:
        out[f"hit@{k}"] = hit_at_k(ranked, gold, k)
        out[f"recall@{k}"] = recall_at_k(ranked, gold, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranked, gold, k)
    return out


def _check(gold: Collection[str]) -> None:
    if not gold:
        raise ValueError("gold set is empty; exclude this question upstream")
