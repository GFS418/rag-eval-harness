import numpy as np
import pytest

from index import rrf


def test_rrf_prefers_items_high_in_both_lists():
    a = np.array([0, 1, 2, 3])
    b = np.array([2, 0, 5, 1])
    fused = rrf([a, b], k=1)
    assert list(fused.idx[:2]) in ([0, 2], [2, 0])
    assert len(fused.idx) == 5
    assert fused.score[0] >= fused.score[-1]


def test_rrf_top_truncates():
    assert len(rrf([np.arange(10)], top=3).idx) == 3


def test_dense_index_scoped_and_open():
    from index import DenseIndex

    v = np.eye(4, dtype=np.float32)
    ix = DenseIndex(v, ["d1", "d1", "d2", "d2"])
    q = np.array([0.1, 0.5, 0.9, 0], dtype=np.float32)
    assert ix.search(q, 1).idx[0] == 2
    assert ix.search(q, 1, doc_id="d1").idx[0] == 1
    assert set(ix.search(q, 5, doc_id="d1").idx) == {0, 1}
    assert list(ix.search(q, 2).idx) == [2, 1]
    assert len(ix.search(q, 10).idx) == 4


def test_bm25_scoped():
    pytest.importorskip("bm25s")
    from index import BM25Index

    ix = BM25Index(["cats purr loudly", "dogs bark loudly", "cats climb trees"], ["d1", "d1", "d2"])
    assert ix.search("cats", 1).idx[0] in (0, 2)
    assert ix.search("cats", 1, doc_id="d2").idx[0] == 2
