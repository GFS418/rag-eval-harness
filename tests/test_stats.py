import numpy as np
import pytest

from eval.stats import bootstrap_mean, paired_bootstrap_diff


def test_bootstrap_mean_covers_true_mean():
    rng = np.random.default_rng(1)
    x = rng.normal(0.5, 0.1, size=400)
    iv = bootstrap_mean(x)
    assert iv.lo < 0.5 < iv.hi and iv.n == 400
    assert iv.mean == pytest.approx(x.mean())


def test_paired_diff_is_tighter_than_unpaired_when_correlated():
    rng = np.random.default_rng(2)
    base = rng.uniform(0, 1, size=300)
    a = base + 0.02
    b = base + rng.normal(0, 0.001, size=300)
    paired = paired_bootstrap_diff(a, b)
    assert paired.lo > 0  # detects the +0.02 improvement
    width_unpaired = (bootstrap_mean(a).hi - bootstrap_mean(a).lo)
    assert (paired.hi - paired.lo) < width_unpaired


def test_paired_shape_mismatch():
    with pytest.raises(ValueError):
        paired_bootstrap_diff(np.ones(3), np.ones(4))
