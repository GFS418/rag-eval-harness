"""Corpus-agnostic data model.

Every corpus adapter (QASPER, FinanceBench, ...) produces the same three
things so that chunking, retrieval and evaluation never need to know where
the documents came from:

* ``Document``  - ordered list of paragraphs, each with a stable ``para_id``.
* ``Question``  - a question over one document (``doc_id``) with gold
  evidence expressed as a set of ``para_id`` values, plus reference answers.
* ``Corpus``    - the documents and questions of one split.

Gold evidence is expressed at paragraph level because that is the granularity
human annotators label at. Mapping paragraphs to *chunks* is the job of
``eval.gold`` and depends on the chunking config.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator


@dataclass(frozen=True)
class Paragraph:
    para_id: str          # f"{doc_id}:{section_idx}:{para_idx}"
    doc_id: str
    section: str
    text: str


@dataclass
class Document:
    doc_id: str
    title: str
    abstract: str
    paragraphs: list[Paragraph]

    def __iter__(self) -> Iterator[Paragraph]:
        return iter(self.paragraphs)


@dataclass
class Question:
    question_id: str
    doc_id: str
    text: str
    # Union of textual evidence paragraphs across annotators.
    gold_para_ids: frozenset[str]
    # True iff the *majority* of annotators marked the question unanswerable.
    unanswerable: bool
    # Fraction of annotators who said unanswerable (for sensitivity checks).
    unanswerable_frac: float
    # One reference answer string per annotator who gave one.
    reference_answers: list[str] = field(default_factory=list)
    # "extractive" | "free_form" | "yes_no" | "unanswerable" (majority type).
    answer_type: str = "free_form"
    # True if at least one annotator's evidence was only a figure/table,
    # i.e. text retrieval cannot reach it. Reported separately.
    evidence_in_float_only: bool = False
    n_annotators: int = 1


@dataclass
class Corpus:
    name: str
    split: str
    documents: dict[str, Document]
    questions: list[Question]

    def paragraphs(self) -> Iterator[Paragraph]:
        for doc in self.documents.values():
            yield from doc.paragraphs

    def __repr__(self) -> str:
        return (f"Corpus({self.name}/{self.split}: {len(self.documents)} docs, "
                f"{sum(len(d.paragraphs) for d in self.documents.values())} paragraphs, "
                f"{len(self.questions)} questions)")
