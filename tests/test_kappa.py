import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from judge_agreement import kappa  # noqa: E402


def test_kappa_perfect_and_chance():
    a = np.array(["x", "y", "x", "y"])
    assert kappa(a, a) == 1.0
    b = np.array(["x", "x", "y", "y"])
    assert kappa(a, b) == pytest.approx(0.0)


def test_kappa_below_chance_negative():
    a = np.array(["x", "y", "x", "y"])
    b = np.array(["y", "x", "y", "x"])
    assert kappa(a, b) < 0
