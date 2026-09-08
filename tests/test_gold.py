from chunking import ChunkConfig, chunk_document, paragraph_token_counts, whitespace_tokenizer
from corpus.base import Document, Paragraph, Question
from eval.gold import gold_chunk_ids, group_chunks


def doc():
    paras = [Paragraph(f"d:0:{i}", "d", "S", " ".join(f"w{i}_{j}" for j in range(10))) for i in range(3)]
    return Document("d", "t", "a", paras)


def q(gold):
    return Question("q", "d", "?", frozenset(gold), False, 0.0)


def test_paragraph_strategy_threshold_irrelevant():
    d = doc()
    chunks = chunk_document(d, ChunkConfig("paragraph", 15), whitespace_tokenizer)
    pt = paragraph_token_counts([d], whitespace_tokenizer)
    for thr in (0.1, 0.5, 1.0):
        assert gold_chunk_ids(q({"d:0:1"}), group_chunks(chunks), pt, thr) == {"d#1"}


def test_fixed_windows_split_paragraph_threshold_matters():
    d = doc()
    chunks = chunk_document(d, ChunkConfig("fixed", 15), whitespace_tokenizer)  # windows 0-15, 15-30
    pt = paragraph_token_counts([d], whitespace_tokenizer)
    by = group_chunks(chunks)
    # paragraph 1 (tokens 10-20) is half in each window
    assert gold_chunk_ids(q({"d:0:1"}), by, pt, 0.5) == {"d#0", "d#1"}
    assert gold_chunk_ids(q({"d:0:1"}), by, pt, 0.6) == frozenset()
    assert gold_chunk_ids(q({"d:0:0"}), by, pt, 1.0) == {"d#0"}


def test_no_gold_gives_empty():
    d = doc()
    chunks = chunk_document(d, ChunkConfig("fixed", 15), whitespace_tokenizer)
    assert gold_chunk_ids(q(set()), group_chunks(chunks), {}, 0.5) == frozenset()
