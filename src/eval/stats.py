"""Uncertainty for per-question metrics.

Every ablation comparison is a *paired* comparison: the same questions are
scored under two configs. The paired bootstrap resamples questions with
replacement and recomputes the mean difference, which respects that pairing
and is far tighter than comparing two independent CIs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    mean: float
    lo: float
    hi: float
    n: int

    def __str__(self) -> str:
        return f"{self.mean:.3f} [{self.lo:.3f}, {self.hi:.3f}] (n={self.n})"


def bootstrap_mean(x: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                   seed: int = 0) -> Interval:
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return Interval(float(x.mean()), float(np.quantile(means, alpha / 2)),
                    float(np.quantile(means, 1 - alpha / 2)), len(x))


def paired_bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 2000,
                          alpha: float = 0.05, seed: int = 0) -> Interval:
    """CI for mean(a - b) over the same questions. Interval excluding 0 => the
    configs differ at level alpha (two-sided)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have the same shape")
    return bootstrap_mean(a - b, n_boot=n_boot, alpha=alpha, seed=seed)
