import pytest

from eval.answer_metrics import (Outcome, abstention_metrics, normalize, summarize_faithfulness,
                                 token_f1, yes_no_match)


def test_normalize_strips_articles_and_punctuation():
    assert normalize("The SQuAD dataset, v1.1!") == "squad dataset v11"


def test_token_f1_max_over_refs():
    assert token_f1("SQuAD and NewsQA", ["SQuAD", "the NewsQA dataset"]) == pytest.approx(2 * (1 / 3) * 1 / (1 / 3 + 1))
    assert token_f1("SQuAD", ["SQuAD"]) == 1.0
    assert token_f1("nothing", []) == 0.0
    assert token_f1("", [""]) == 1.0


def test_yes_no_match_uses_first_word():
    assert yes_no_match("Yes, they do.", ["Yes"]) == 1.0
    assert yes_no_match("No", ["Yes"]) == 0.0
    assert yes_no_match("", ["Yes"]) == 0.0


def test_hallucination_definition():
    assert Outcome(unanswerable=True, abstain=False, faithful=True).hallucinated
    assert not Outcome(unanswerable=True, abstain=True, faithful=None).hallucinated
    assert Outcome(unanswerable=False, abstain=False, faithful=False).hallucinated
    assert not Outcome(unanswerable=False, abstain=False, faithful=True).hallucinated
    assert not Outcome(unanswerable=False, abstain=True, faithful=None).hallucinated


def test_abstention_metrics():
    outs = [Outcome(True, True, None), Outcome(True, False, True),
            Outcome(False, True, None), Outcome(False, False, True)]
    m = abstention_metrics(outs)
    assert m["abstain_precision"] == 0.5 and m["abstain_recall"] == 0.5
    assert m["hallucination_rate"] == 0.25 and m["abstain_rate"] == 0.5


def test_summarize_faithfulness_handles_missing_claims():
    data = {"claims": [{"index": 1, "support": "supported", "cited_support": True}]}
    s = summarize_faithfulness(data, n_claims=2)
    assert s["faithful"] is False and s["frac_supported"] == 0.5 and s["citation_precision"] == 0.5
    assert summarize_faithfulness({"claims": []}, 0)["faithful"] is True
