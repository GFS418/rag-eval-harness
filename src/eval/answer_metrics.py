"""Answer-level metrics.

Three independent instruments, each with a stated failure mode:

* ``token_f1``      - QASPER's official lexical overlap vs reference answers.
                      Cheap and deterministic; harsh on paraphrase.
* NLI faithfulness  - an open NLI cross-encoder scores each claim against the
                      passages it cites. No API, reproducible; weak on
                      multi-sentence reasoning and numbers.
* LLM judge         - Claude Opus 5 grades correctness (vs references) and
                      per-claim support / citation precision. Strong but
                      opaque; must be validated against human labels.

Composite definitions (used by the report):

* faithful(answer)      = every claim is supported by *some* provided passage
* citation_precision    = fraction of claims whose *cited* passages support it
* hallucinated(answer)  = (answerable and not abstain and not faithful)
                          or (unanswerable and not abstain)
"""
from __future__ import annotations

import json
import re
import string
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from llm import Request

# --------------------------------------------------------------------------
# Lexical correctness (QASPER evaluator semantics)
# --------------------------------------------------------------------------

_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def _f1(pred: str, ref: str) -> float:
    p, r = normalize(pred).split(), normalize(ref).split()
    if not p or not r:
        return float(p == r)
    common = Counter(p) & Counter(r)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(r)
    return 2 * prec * rec / (prec + rec)


def token_f1(pred: str, refs: Sequence[str]) -> float:
    """Max token-F1 over reference answers (0 if no references)."""
    return max((_f1(pred, r) for r in refs), default=0.0)


def yes_no_match(pred: str, refs: Sequence[str]) -> float:
    first = normalize(pred).split()[:1]
    return float(bool(first) and any(first[0] == normalize(r) for r in refs))


# --------------------------------------------------------------------------
# Abstention / hallucination bookkeeping
# --------------------------------------------------------------------------

@dataclass
class Outcome:
    unanswerable: bool
    abstain: bool
    faithful: bool | None       # None when abstained / not judged

    @property
    def hallucinated(self) -> bool:
        if self.abstain:
            return False
        if self.unanswerable:
            return True
        return self.faithful is False


def abstention_metrics(outcomes: Sequence[Outcome]) -> dict[str, float]:
    tp = sum(o.abstain and o.unanswerable for o in outcomes)
    fp = sum(o.abstain and not o.unanswerable for o in outcomes)
    fn = sum((not o.abstain) and o.unanswerable for o in outcomes)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "abstain_precision": prec,
        "abstain_recall": rec,
        "abstain_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "abstain_rate": sum(o.abstain for o in outcomes) / max(len(outcomes), 1),
        "hallucination_rate": sum(o.hallucinated for o in outcomes) / max(len(outcomes), 1),
    }


# --------------------------------------------------------------------------
# LLM judge (Opus 5): correctness and per-claim support
# --------------------------------------------------------------------------

JUDGE_MODEL = "opus-5"

CORRECTNESS_SYSTEM = """You grade answers to questions about scientific papers.
You are given the question, one or more reference answers written by human annotators, and a candidate answer.
Grade the candidate ONLY against the references:
- "correct": conveys the same information as at least one reference (wording may differ; extra correct detail is fine).
- "partial": captures part of a reference but omits or blurs something important.
- "incorrect": contradicts the references or answers a different question.
Give a one-sentence rationale."""

CORRECTNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["correct", "partial", "incorrect"]},
        "rationale": {"type": "string"},
    },
    "required": ["verdict", "rationale"],
    "additionalProperties": False,
}

FAITHFULNESS_SYSTEM = """You audit whether claims are supported by passages.
You are given numbered passages, a question, and a list of numbered claims. Each claim lists the passage numbers it cites.
For EACH claim decide:
- "support": "supported" if some provided passage (any of them) states or directly implies the claim;
  "contradicted" if a passage says the opposite; "unsupported" if no passage backs it.
- "cited_support": true only if the claim's OWN cited passages are sufficient to support it.
Judge strictly from the passages. Outside knowledge does not count."""

FAITHFULNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "support": {"type": "string", "enum": ["supported", "unsupported", "contradicted"]},
                    "cited_support": {"type": "boolean"},
                },
                "required": ["index", "support", "cited_support"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}


def correctness_request(question_id: str, question: str, refs: Sequence[str], answer: str,
                        tag: str) -> Request:
    refs_txt = "\n".join(f"- {r}" for r in refs)
    user = f"Question: {question}\n\nReference answers:\n{refs_txt}\n\nCandidate answer: {answer}"
    return Request(custom_id=f"corr|{question_id}|{tag}"[:64], model=JUDGE_MODEL,
                   system=CORRECTNESS_SYSTEM, user=user, schema=CORRECTNESS_SCHEMA, effort="medium")


def faithfulness_request(question_id: str, question: str, passages: Sequence[tuple[str, str]],
                         claims: Sequence[tuple[str, Sequence[int]]], tag: str) -> Request:
    """passages: (chunk_id, text); claims: (text, cited passage numbers 1-based)."""
    ptxt = "\n\n".join(f"[{i}]\n{t}" for i, (_, t) in enumerate(passages, start=1))
    ctxt = "\n".join(f"Claim {i} (cites {list(c)}): {t}" for i, (t, c) in enumerate(claims, start=1))
    user = f"Passages:\n{ptxt}\n\nQuestion: {question}\n\nClaims:\n{ctxt}"
    return Request(custom_id=f"faith|{question_id}|{tag}"[:64], model=JUDGE_MODEL,
                   system=FAITHFULNESS_SYSTEM, user=user, schema=FAITHFULNESS_SCHEMA, effort="medium")


def summarize_faithfulness(data: dict, n_claims: int) -> dict[str, float | bool]:
    by_idx = {c["index"]: c for c in data.get("claims", [])}
    supports = [by_idx.get(i, {}).get("support", "unsupported") for i in range(1, n_claims + 1)]
    cited = [bool(by_idx.get(i, {}).get("cited_support", False)) for i in range(1, n_claims + 1)]
    return {
        "faithful": all(s == "supported" for s in supports) if n_claims else True,
        "frac_supported": sum(s == "supported" for s in supports) / n_claims if n_claims else 1.0,
        "any_contradicted": any(s == "contradicted" for s in supports),
        "citation_precision": sum(cited) / n_claims if n_claims else 1.0,
    }


# --------------------------------------------------------------------------
# NLI faithfulness (open model, no API)
# --------------------------------------------------------------------------

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"


class NLIScorer:
    """P(entailment) of each claim given its cited passages (or all passages)."""

    def __init__(self, model_name: str = NLI_MODEL, threshold: float = 0.5):
        self.model_name = model_name
        self.threshold = threshold
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            from embed import _device

            self._model = CrossEncoder(self.model_name, device=_device(), max_length=512)
        return self._model

    def entailment_probs(self, pairs: list[tuple[str, str]]) -> list[float]:
        import numpy as np

        logits = self.model.predict(pairs, batch_size=32, show_progress_bar=False,
                                    apply_softmax=True)
        labels = self.model.model.config.id2label
        ent = next(i for i, name in labels.items() if name.lower().startswith("entail"))
        return [float(p) for p in np.asarray(logits)[:, ent]]

    def score_answer(self, passages: Sequence[str], claims: Sequence[tuple[str, Sequence[int]]]) -> dict:
        if not claims:
            return {"nli_faithful": True, "nli_min_entail": 1.0, "nli_frac_supported": 1.0}
        pairs = []
        for text, cites in claims:
            idx = [i - 1 for i in cites if 1 <= i <= len(passages)] or list(range(len(passages)))
            premise = "\n".join(passages[i] for i in idx)
            pairs.append((premise, text))
        probs = self.entailment_probs(pairs)
        return {
            "nli_faithful": all(p >= self.threshold for p in probs),
            "nli_min_entail": min(probs),
            "nli_frac_supported": sum(p >= self.threshold for p in probs) / len(probs),
        }


def claims_from_json(s: str | None) -> list[tuple[str, list[int]]]:
    """Recover (text, passage numbers) from the parquet ``claims`` column and passage list."""
    if not s:
        return []
    return [(c["text"], c.get("passage_numbers", [])) for c in json.loads(s)]
