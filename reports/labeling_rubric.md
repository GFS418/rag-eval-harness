# Labelling rubric (human validation of the judges)

The human labeller applies the **same rubric the Opus 5 judge was prompted
with**, so that Cohen's kappa measures judge error rather than rubric
mismatch. Section 1 is the judge's prompt in plain English; section 2 records
the tie-break rules settled during labelling, applied consistently to all 100
items. Labels were made blind to the judge's verdicts.

## 1. Correctness (answer vs. reference answers)

Grade the candidate answer ONLY against the human reference answers.

- **correct** — conveys the same information as **at least one** reference.
  Wording may differ. Extra correct detail is fine.
- **partial** — captures part of a reference but omits or blurs something
  important.
- **incorrect** — contradicts the references, or answers a different question.

## 2. Claim support (each claim vs. the passages)

For each claim the model made, mark **supported** if some provided passage
states or directly implies it; **unsupported** otherwise (including when the
claim is true in the world but not in the passages). Outside knowledge does not
count. The passages the model saw are the only evidence.

## 3. Tie-break rules

1. **References are alternatives, not parts.** Each reference is one
   annotator's answer. Match one of them; do not grade against their union.
   *Example:* three annotators describe a comparison as (a) a GNN feature-based
   method, (b) a domain-knowledge difference, (c) a node-type caveat. An answer
   matching (a) in full is correct even though it says nothing about (b) or (c).
2. **Quoted-evidence references: grade against the part that answers the
   question.** Some references are pasted paper text with side remarks. Omitting
   a side remark (e.g. an explanation of *why* recall is higher) is not an
   omission.
3. **Specifics asked, vagueness given = incorrect, not partial.** Test: could
   someone holding only the model's answer produce any of the reference's
   specific content? If they would have to guess, it is incorrect. "They used
   Chinese-English and English-Japanese datasets" against a list of named
   datasets is incorrect; "NIST 2003–2008" that drops one named dataset is
   partial.
4. **Same metric, different slicing = partial.** Numbers that answer the
   question with the right metric but cannot reproduce either reference's
   figures (per-competitor vs. per-dataset) are partial.
5. **Partial list = partial.** Count the items of the matched reference the
   answer covers. Three of five methods, with a prominent one missing, is
   partial. Note which item is missing.
6. **Off-target extra material neither helps nor hurts** unless it changes the
   meaning of the answer (listing seven domains when the reference lists five
   is partial; an extra sentence describing the dataset is fine). Unsupported
   detail in extra material is caught by the claim-support labels, not here.
7. **Matching one reference while contradicting another** stays correct under
   the rubric, but check the contradicting claim's support carefully and leave
   a note ("ref 1 suggests speech present"). These items are reviewed
   individually in the disagreement analysis.
8. **Use the note field** for: references that disagree with each other,
   contestable calls, and which list item was omitted. Notes are used to
   classify judge–human disagreements.
