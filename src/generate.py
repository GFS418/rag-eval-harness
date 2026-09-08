"""Grounded answer generation with citations.

Three modes share one prompt so they are comparable:

* ``rag``         - passages are the retrieved chunks (the system under test)
* ``oracle``      - passages are the gold evidence paragraphs (upper bound:
                    isolates generation error from retrieval error)
* ``closed-book`` - no passages; the model answers from memory (lower bound:
                    what retrieval adds, and a memorisation signal)

The model must return a structured answer: an ``abstain`` flag, a short
``answer``, and a list of ``claims`` each citing passage numbers. Claim-level
citations are what make citation precision and faithfulness measurable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from llm import Request

SYSTEM = """You answer questions about a scientific paper using only the passages provided.

Rules:
- Use ONLY the provided passages. Do not use outside knowledge.
- If the passages do not contain enough information to answer, set "abstain" to true, \
set "answer" to "Cannot answer from the provided passages." and leave "claims" empty.
- Otherwise give a concise answer (one sentence or a short list; for yes/no questions start with "Yes" or "No").
- Break the answer into claims. Every claim must cite the passage number(s) that support it. \
Do not make claims that no passage supports.
- Passage numbers are the integers in square brackets, e.g. [3]."""

SYSTEM_CLOSED_BOOK = """You answer questions about a scientific paper from your own knowledge. \
The paper's title is given. No passages are provided.

Rules:
- If you do not know the answer for this specific paper, set "abstain" to true, \
set "answer" to "I do not know." and leave "claims" empty. Do not guess.
- Otherwise give a concise answer (one sentence or a short list; for yes/no questions start with "Yes" or "No").
- Break the answer into claims. Cite nothing (use an empty citations list)."""

SCHEMA = {
    "type": "object",
    "properties": {
        "abstain": {"type": "boolean"},
        "answer": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "citations": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["text", "citations"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["abstain", "answer", "claims"],
    "additionalProperties": False,
}


@dataclass
class Passage:
    chunk_id: str
    text: str
    section: str = ""


@dataclass
class GenerationInput:
    question_id: str
    question: str
    title: str
    passages: list[Passage] = field(default_factory=list)
    mode: str = "rag"          # rag | oracle | closed-book


def build_user_prompt(inp: GenerationInput) -> str:
    head = f'Paper title: "{inp.title}"\n\nQuestion: {inp.question}\n'
    if inp.mode == "closed-book":
        return head
    parts = [head, "\nPassages:"]
    for i, p in enumerate(inp.passages, start=1):
        sec = f" (section: {p.section})" if p.section else ""
        parts.append(f"\n[{i}]{sec}\n{p.text}\n")
    return "\n".join(parts)


def build_request(inp: GenerationInput, model: str, cfg_name: str,
                  effort: str | None = "medium") -> Request:
    system = SYSTEM_CLOSED_BOOK if inp.mode == "closed-book" else SYSTEM
    return Request(
        custom_id=f"{inp.question_id}|{model}|{inp.mode}|{cfg_name}"[:64],
        model=model, system=system, user=build_user_prompt(inp),
        schema=SCHEMA, effort=effort,
    )


@dataclass
class Claim:
    text: str
    chunk_ids: list[str]          # resolved from passage numbers
    passage_numbers: list[int]    # 1-based, as the model wrote them (valid ones only)


@dataclass
class GroundedAnswer:
    question_id: str
    abstain: bool
    answer: str
    claims: list[Claim]
    invalid_citations: int        # passage numbers outside 1..len(passages)


def parse_answer(inp: GenerationInput, data: dict) -> GroundedAnswer:
    """Resolve passage numbers to chunk ids; count out-of-range citations."""
    n = len(inp.passages)
    claims, bad = [], 0
    for c in data.get("claims", []):
        ids, nums = [], []
        for k in c.get("citations", []):
            if 1 <= k <= n:
                ids.append(inp.passages[k - 1].chunk_id)
                nums.append(k)
            else:
                bad += 1
        claims.append(Claim(c.get("text", ""), ids, nums))
    return GroundedAnswer(inp.question_id, bool(data.get("abstain", False)),
                          data.get("answer", ""), claims, bad)
