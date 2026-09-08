"""Leakage guard: fine-tuning data must come only from train papers, and the
three QASPER splits must be disjoint. The synthetic test always runs; the
real-data test runs whenever the raw files are present (CI has none)."""
import json
from pathlib import Path

import pytest

from finetune.make_pairs import check_no_leakage

RAW = Path("data/raw/qasper")


def test_synthetic_disjoint_ok():
    check_no_leakage({"a", "b"}, {"c"}, {"a"})


def test_synthetic_split_overlap_fails():
    with pytest.raises(AssertionError):
        check_no_leakage({"a", "b"}, {"b"}, {"a"})


def test_synthetic_stray_pair_fails():
    with pytest.raises(AssertionError):
        check_no_leakage({"a"}, {"b"}, {"a", "b"})


@pytest.mark.skipif(not (RAW / "qasper-train-v0.3.json").exists(), reason="raw data not present")
def test_real_splits_disjoint_and_pairs_from_train():
    ids = {s: set(json.loads((RAW / f"qasper-{s}-v0.3.json").read_text())) for s in ("train", "dev", "test")}
    assert not (ids["train"] & ids["dev"]) and not (ids["train"] & ids["test"]) and not (ids["dev"] & ids["test"])
    pairs = Path("data/processed/finetune_pairs.jsonl")
    if pairs.exists():
        docs = {json.loads(l)["doc_id"] for l in pairs.read_text().splitlines()}
        check_no_leakage(ids["train"], ids["dev"] | ids["test"], docs)
