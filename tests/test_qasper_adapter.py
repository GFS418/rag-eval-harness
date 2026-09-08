import json

from corpus.qasper import load_qasper, summarize

PAPER = {
    "title": "T", "abstract": "A",
    "full_text": [
        {"section_name": "Intro", "paragraphs": ["Para zero has enough words to be a real paragraph of text.", ""]},
        {"section_name": None, "paragraphs": ["Para  one\nwith odd   whitespace and enough words to exceed forty chars."]},
    ],
    "figures_and_tables": [],
    "qas": [
        {"question": "q1?", "question_id": "q1", "nlp_background": "", "topic_background": "",
         "paper_read": "", "search_query": "", "question_writer": "w",
         "answers": [
             {"answer": {"unanswerable": False, "extractive_spans": ["zero"], "yes_no": None,
                         "free_form_answer": "", "evidence": ["Para zero has enough words to be a real paragraph of text."],
                         "highlighted_evidence": []}, "annotation_id": "a", "worker_id": "w"},
             {"answer": {"unanswerable": False, "extractive_spans": [], "yes_no": None,
                         "free_form_answer": "free", "evidence": ["Para one with odd whitespace and enough words to exceed forty chars."],
                         "highlighted_evidence": []}, "annotation_id": "b", "worker_id": "w"},
         ]},
        {"question": "q2?", "question_id": "q2", "nlp_background": "", "topic_background": "",
         "paper_read": "", "search_query": "", "question_writer": "w",
         "answers": [
             {"answer": {"unanswerable": True, "extractive_spans": [], "yes_no": None,
                         "free_form_answer": "", "evidence": [], "highlighted_evidence": []},
              "annotation_id": "c", "worker_id": "w"},
             {"answer": {"unanswerable": True, "extractive_spans": [], "yes_no": None,
                         "free_form_answer": "", "evidence": [], "highlighted_evidence": []},
              "annotation_id": "d", "worker_id": "w"},
             {"answer": {"unanswerable": False, "extractive_spans": [], "yes_no": True,
                         "free_form_answer": "", "evidence": ["FLOAT SELECTED: Table 1"], "highlighted_evidence": []},
              "annotation_id": "e", "worker_id": "w"},
         ]},
        {"question": "q3?", "question_id": "q3", "nlp_background": "", "topic_background": "",
         "paper_read": "", "search_query": "", "question_writer": "w",
         "answers": [
             {"answer": {"unanswerable": False, "extractive_spans": [], "yes_no": False,
                         "free_form_answer": "", "evidence": ["Intro", "FLOAT SELECTED: Figure 2"],
                         "highlighted_evidence": []}, "annotation_id": "f", "worker_id": "w"},
         ]},
    ],
}


def write(tmp_path):
    (tmp_path / "qasper-dev-v0.3.json").write_text(json.dumps({"1234.5678": PAPER}))
    return load_qasper("dev", tmp_path)


def test_documents_skip_empty_paragraphs_and_handle_none_section(tmp_path):
    c = write(tmp_path)
    doc = c.documents["1234.5678"]
    assert [p.para_id for p in doc.paragraphs] == ["1234.5678:0:0", "1234.5678:1:0"]
    assert doc.paragraphs[1].section == ""


def test_gold_is_union_across_annotators_with_normalised_match(tmp_path):
    c = write(tmp_path)
    q1 = c.questions[0]
    assert q1.gold_para_ids == {"1234.5678:0:0", "1234.5678:1:0"}
    assert q1.reference_answers == ["zero", "free"]
    assert not q1.unanswerable and q1.n_annotators == 2


def test_unanswerable_majority_and_fraction(tmp_path):
    q2 = write(tmp_path).questions[1]
    assert q2.unanswerable and abs(q2.unanswerable_frac - 2 / 3) < 1e-9
    assert q2.answer_type == "unanswerable"
    assert q2.reference_answers == ["Yes"]


def test_heading_evidence_not_mapped_and_float_flagged(tmp_path):
    q3 = write(tmp_path).questions[2]
    assert q3.gold_para_ids == frozenset()
    assert q3.evidence_in_float_only is False  # it had a heading string too
    assert q3.answer_type == "yes_no" and q3.reference_answers == ["No"]


def test_summary_counts(tmp_path):
    s = summarize(write(tmp_path))
    assert s["questions"] == 3 and s["unanswerable_majority"] == 1
    assert s["answerable_with_gold_paragraphs"] == 1
