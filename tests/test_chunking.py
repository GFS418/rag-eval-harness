from corpus.base import Document, Paragraph
from chunking import (Chunk, ChunkConfig, chunk_document, paragraph_token_counts,
                      whitespace_tokenizer)

TOK = whitespace_tokenizer


def make_doc(para_lengths):
    paras = [Paragraph(para_id=f"d:0:{i}", doc_id="d", section="S",
                       text=" ".join(f"w{i}_{j}" for j in range(n)))
             for i, n in enumerate(para_lengths)]
    return Document(doc_id="d", title="t", abstract="a", paragraphs=paras)


def test_config_validation():
    import pytest
    with pytest.raises(ValueError):
        ChunkConfig(strategy="nope")
    with pytest.raises(ValueError):
        ChunkConfig(size=10, overlap=10)
    with pytest.raises(ValueError):
        ChunkConfig(size=0)


def test_fixed_covers_every_token_exactly_once_without_overlap():
    doc = make_doc([10, 25, 7, 40])
    chunks = chunk_document(doc, ChunkConfig("fixed", size=16, overlap=0), TOK)
    assert sum(c.n_tokens for c in chunks) == 82
    assert all(c.n_tokens <= 16 for c in chunks)
    # per-paragraph token counts add back up to paragraph lengths
    totals = {}
    for c in chunks:
        for pid, n in c.para_token_counts.items():
            totals[pid] = totals.get(pid, 0) + n
    assert totals == paragraph_token_counts([doc], TOK)


def test_fixed_overlap_shares_tokens_between_neighbours():
    doc = make_doc([50])
    cfg = ChunkConfig("fixed", size=20, overlap=5)
    chunks = chunk_document(doc, cfg, TOK)
    # windows start at 0,15,30; the third reaches token 49 so no 4th window
    assert [c.n_tokens for c in chunks] == [20, 20, 20]
    a, b = chunks[0].text.split(), chunks[1].text.split()
    assert a[-5:] == b[:5]
    assert chunks[-1].text.split()[-1] == "w0_49"


def test_fixed_last_window_not_duplicated_when_exact_multiple():
    doc = make_doc([40])
    chunks = chunk_document(doc, ChunkConfig("fixed", size=20, overlap=0), TOK)
    assert len(chunks) == 2


def test_paragraph_strategy_never_splits_short_paragraphs():
    doc = make_doc([10, 25, 7, 12])
    chunks = chunk_document(doc, ChunkConfig("paragraph", size=30), TOK)
    for c in chunks:
        for pid, n in c.para_token_counts.items():
            assert n == len(doc.paragraphs[int(pid.split(":")[-1])].text.split())
    assert [sorted(c.para_token_counts) for c in chunks] == [["d:0:0"], ["d:0:1"], ["d:0:2", "d:0:3"]]
    # original paragraph text preserved verbatim
    assert chunks[2].text == doc.paragraphs[2].text + "\n\n" + doc.paragraphs[3].text


def test_paragraph_strategy_windows_overlong_paragraph():
    doc = make_doc([5, 70, 5])
    chunks = chunk_document(doc, ChunkConfig("paragraph", size=30), TOK)
    lengths = [c.n_tokens for c in chunks]
    assert lengths == [5, 30, 30, 10, 5]
    assert all(set(c.para_token_counts) == {"d:0:1"} for c in chunks[1:4])


def test_coverage_fraction():
    doc = make_doc([10, 10])
    chunks = chunk_document(doc, ChunkConfig("fixed", size=15, overlap=0), TOK)
    counts = paragraph_token_counts([doc], TOK)
    p1 = doc.paragraphs[1]
    assert chunks[0].coverage(p1, counts[p1.para_id]) == 0.5
    assert chunks[1].coverage(p1, counts[p1.para_id]) == 0.5
    assert chunks[0].coverage(doc.paragraphs[0], 10) == 1.0


def test_empty_document():
    doc = Document(doc_id="d", title="", abstract="", paragraphs=[])
    assert chunk_document(doc, ChunkConfig(), TOK) == []


def test_chunk_ids_unique_and_ordered():
    doc = make_doc([30, 30, 30])
    chunks = chunk_document(doc, ChunkConfig("fixed", size=20, overlap=5), TOK)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids == [f"d#{i}" for i in range(len(ids))]


def test_fixed_window_preserves_original_text_and_casing():
    p = Paragraph(para_id="d:0:0", doc_id="d", section="S",
                  text="Alpha Beta, gamma. Delta EPSILON zeta")
    doc = Document(doc_id="d", title="", abstract="", paragraphs=[p])
    chunks = chunk_document(doc, ChunkConfig("fixed", size=3, overlap=0), TOK)
    assert [c.text for c in chunks] == ["Alpha Beta, gamma.", "Delta EPSILON zeta"]


def test_fixed_window_spanning_paragraphs_joins_with_blank_line():
    doc = make_doc([2, 2])
    chunks = chunk_document(doc, ChunkConfig("fixed", size=3, overlap=0), TOK)
    assert chunks[0].text == "w0_0 w0_1\n\nw1_0"
    assert chunks[0].para_token_counts == {"d:0:0": 2, "d:0:1": 1}
