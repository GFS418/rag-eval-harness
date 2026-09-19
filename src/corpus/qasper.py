"""QASPER adapter (Dasigi et al., 2021), raw v0.3 JSON release.

Schema facts that drive the design (measured on the release files):

* Textual ``evidence`` entries are whole paragraphs copied from ``full_text``.
  ~92% match a paragraph exactly; a further few % match after whitespace
  normalisation or as a sub/super-string; the remainder are section headings
  (e.g. "Datasets") or stray strings. Headings are not mapped to gold.
* Evidence beginning with ``FLOAT SELECTED`` is a figure or table caption.
  Text retrieval cannot reach it; such questions are flagged, not dropped.
* Questions have 1-6 annotators (test has >=2 for 98%). Gold evidence is the
  union across annotators; "unanswerable" is decided by majority vote, with
  the fraction kept for sensitivity analysis.

Splits: ``train`` (fine-tuning pairs only), ``dev`` (config selection),
``test`` (reported numbers). Papers are disjoint across splits.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from corpus.base import Corpus, Document, Paragraph, Question

FLOAT_PREFIX = "FLOAT SELECTED"
RELEASE_URLS = {
    "train": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz",
    "dev": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz",
    "test": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz",
}


def ensure_qasper(split: str, root: str | Path = "data/raw/qasper") -> Path:
    """Download and extract the public v0.3 release if the split file is missing
    (used by the deployed app, whose container starts without data/raw)."""
    import tarfile
    import urllib.request

    root = Path(root)
    path = root / f"qasper-{split}-v0.3.json"
    if path.exists():
        return path
    root.mkdir(parents=True, exist_ok=True)
    tgz = root / f"{split}.tgz"
    urllib.request.urlretrieve(RELEASE_URLS[split], tgz)
    with tarfile.open(tgz) as tf:
        tf.extractall(root, filter="data")
    tgz.unlink()
    if not path.exists():
        raise FileNotFoundError(f"{path} missing after extracting the release tarball")
    return path
_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s).strip().lower()


def _answer_type(ans: dict) -> str:
    if ans["unanswerable"]:
        return "unanswerable"
    if ans["yes_no"] is not None:
        return "yes_no"
    if ans["extractive_spans"]:
        return "extractive"
    return "free_form"


def _answer_text(ans: dict) -> str | None:
    t = _answer_type(ans)
    if t == "unanswerable":
        return None
    if t == "yes_no":
        return "Yes" if ans["yes_no"] else "No"
    if t == "extractive":
        return "; ".join(ans["extractive_spans"])
    return ans["free_form_answer"]


def _build_document(paper_id: str, paper: dict) -> Document:
    paras: list[Paragraph] = []
    for si, sec in enumerate(paper["full_text"]):
        for pi, text in enumerate(sec["paragraphs"]):
            if not text.strip():
                continue
            paras.append(Paragraph(
                para_id=f"{paper_id}:{si}:{pi}",
                doc_id=paper_id,
                section=sec["section_name"] or "",
                text=text,
            ))
    return Document(doc_id=paper_id, title=paper["title"],
                    abstract=paper["abstract"], paragraphs=paras)


def _map_evidence(evidence: str, doc: Document, exact: dict[str, str],
                  normed: dict[str, str]) -> set[str]:
    """Return the para_ids an evidence string maps to (possibly empty)."""
    if evidence in exact:
        return {exact[evidence]}
    ne = _norm(evidence)
    if ne in normed:
        return {normed[ne]}
    if len(ne) < 40:          # short strings are headings / labels; too ambiguous
        return set()
    hits = {p.para_id for p in doc.paragraphs
            if ne in _norm(p.text) or _norm(p.text) in ne}
    return hits


def _build_question(qa: dict, doc: Document, exact: dict[str, str],
                    normed: dict[str, str]) -> Question:
    anns = [a["answer"] for a in qa["answers"]]
    n = len(anns)
    n_unans = sum(a["unanswerable"] for a in anns)
    gold: set[str] = set()
    float_only = False
    refs: list[str] = []
    types = Counter()
    for a in anns:
        types[_answer_type(a)] += 1
        txt = _answer_text(a)
        if txt:
            refs.append(txt)
        ev = a["evidence"]
        text_ev = [e for e in ev if not e.startswith(FLOAT_PREFIX)]
        if ev and not text_ev and not a["unanswerable"]:
            float_only = True
        for e in text_ev:
            gold |= _map_evidence(e, doc, exact, normed)
    majority_unans = n_unans * 2 > n
    return Question(
        question_id=qa["question_id"],
        doc_id=doc.doc_id,
        text=qa["question"],
        gold_para_ids=frozenset(gold),
        unanswerable=majority_unans,
        unanswerable_frac=n_unans / n,
        reference_answers=refs,
        answer_type="unanswerable" if majority_unans else
        (types.most_common(1)[0][0] if types else "free_form"),
        evidence_in_float_only=float_only,
        n_annotators=n,
    )


def load_qasper(split: str, root: str | Path = "data/raw/qasper") -> Corpus:
    path = ensure_qasper(split, root)
    raw = json.loads(path.read_text())
    documents: dict[str, Document] = {}
    questions: list[Question] = []
    for paper_id, paper in raw.items():
        doc = _build_document(paper_id, paper)
        documents[paper_id] = doc
        exact = {p.text: p.para_id for p in doc.paragraphs}
        normed = {_norm(p.text): p.para_id for p in doc.paragraphs}
        for qa in paper["qas"]:
            questions.append(_build_question(qa, doc, exact, normed))
    return Corpus(name="qasper", split=split, documents=documents, questions=questions)


def summarize(corpus: Corpus) -> dict:
    """Counts the README reports so eval-set construction is auditable."""
    qs = corpus.questions
    answerable = [q for q in qs if not q.unanswerable]
    return {
        "documents": len(corpus.documents),
        "paragraphs": sum(len(d.paragraphs) for d in corpus.documents.values()),
        "questions": len(qs),
        "unanswerable_majority": sum(q.unanswerable for q in qs),
        "unanswerable_any": sum(q.unanswerable_frac > 0 for q in qs),
        "answerable": len(answerable),
        "answerable_with_gold_paragraphs": sum(bool(q.gold_para_ids) for q in answerable),
        "answerable_float_only_some_annotator": sum(q.evidence_in_float_only for q in answerable),
        "answer_types": dict(Counter(q.answer_type for q in qs)),
        "annotators_per_question": dict(Counter(q.n_annotators for q in qs)),
    }
