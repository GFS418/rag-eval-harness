from generate import (SCHEMA, GenerationInput, Passage, build_request, build_user_prompt,
                      parse_answer)
from llm import Request


def inp(mode="rag"):
    return GenerationInput("q1", "What dataset?", "A Paper",
                           [Passage("p#0", "We use SQuAD.", "Data"), Passage("p#1", "Results are good.")],
                           mode=mode)


def test_prompt_numbers_passages_from_one():
    u = build_user_prompt(inp())
    assert "[1] (section: Data)\nWe use SQuAD." in u
    assert "[2]\nResults are good." in u
    assert u.startswith('Paper title: "A Paper"')


def test_closed_book_has_no_passages():
    u = build_user_prompt(inp("closed-book"))
    assert "Passages" not in u and "SQuAD" not in u


def test_request_cache_key_ignores_effort_for_haiku():
    r1 = build_request(inp(), "haiku-4.5", "cfg", effort="medium")
    r2 = build_request(inp(), "haiku-4.5", "cfg", effort="high")
    assert r1.cache_key() == r2.cache_key()
    s1 = build_request(inp(), "sonnet-5", "cfg", effort="medium")
    s2 = build_request(inp(), "sonnet-5", "cfg", effort="high")
    assert s1.cache_key() != s2.cache_key()
    assert "effort" not in r1.params()["output_config"]
    assert s1.params()["output_config"]["effort"] == "medium"
    assert s1.params()["output_config"]["format"]["schema"] is SCHEMA


def test_parse_answer_resolves_and_counts_bad_citations():
    data = {"abstain": False, "answer": "SQuAD",
            "claims": [{"text": "They use SQuAD.", "citations": [1, 7]},
                       {"text": "Results good.", "citations": [2]}]}
    ga = parse_answer(inp(), data)
    assert not ga.abstain and ga.answer == "SQuAD"
    assert ga.claims[0].chunk_ids == ["p#0"] and ga.claims[1].chunk_ids == ["p#1"]
    assert ga.invalid_citations == 1


def test_parse_abstain():
    ga = parse_answer(inp(), {"abstain": True, "answer": "Cannot answer.", "claims": []})
    assert ga.abstain and ga.claims == []


def test_cache_roundtrip(tmp_path):
    from llm import Cache, Response

    c = Cache(tmp_path / "c.sqlite")
    r = Request("id", "sonnet-5", "s", "u", SCHEMA)
    assert c.get(r) is None
    c.put(r, Response("id", {"a": 1}, "end_turn", 10, 5, 0.5, None, "claude-sonnet-5"))
    got = c.get(r)
    assert got.data == {"a": 1} and got.cost_usd > 0
