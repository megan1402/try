# Next experiment: candidate generation + LLM verification

Recorded 2026-11-21, after the precision measurement. This is the plan for
fixing event extraction. It is deliberately **not** a fine-tuning plan.

## Where we are

| Measurement | Result | What it means |
|---|---|---|
| Event study (41 issuers, 118 dates) | no abnormal returns | says nothing about the market |
| Precision (149 stratified events) | **0.055** [0.028, 0.104] | the detector is usually wrong when it fires |
| Recall | **not yet measured** | the missing half |

The event study is not evidence that filings do not move prices. It is evidence
that the events it measured were mostly not events. **Extraction must be fixed
before any return analysis is interpretable.**

## The four steps, in order

### 1. Establish recall

Label the 38 event-bearing passages (`make review-events`, `make recall-events`).
This gives recall *conditional on the passage mentioning an event* — a narrower
claim than corpus-wide recall, but the one that answers "does the detector miss
real events?"

**Do not proceed until this number exists.** Without it, "the detector is bad"
is a guess about which half is bad.

### 2. Score the existing zero-shot model

`app/models/zero_shot.py` has never been measured. Score it against the same
149 labelled events and the same 38 passages:

- If it beats 0.055 precision, a model is worth pursuing.
- If it does not, zero-shot NLI is also insufficient, and the answer is a
  different architecture — not a better prompt.

This is cheap: the labels exist, the model exists, only the scoring is missing.

### 3. Inspect the errors

Not aggregate numbers — actual failures. For each false positive and false
negative, record *why*:

| Failure mode | Example seen already |
|---|---|
| Accounting vocabulary | "Acquisitions, net of cash acquired and divestitures (516)" |
| Historical reference | "In July 2025, the company completed its acquisition of Hess" |
| Verb-before-role | "Following Mr. Huang's resignation, the Board appointed..." |
| Policy description | "Cybersecurity assessments for acquired companies" |
| Table row | "Deferred Acquisition Costs Prepaid and other current assets" |

The error taxonomy decides what the next architecture must handle. Guessing at
it produces a system that fails in new ways.

### 4. Prototype: high-recall candidate generator + LLM verifier

Only after 1–3. The shape:

```
passage
  -> candidate generator (high recall, low precision)
       regex + vocabulary + section heuristics
       goal: miss nothing, accept noise
  -> LLM verifier (high precision)
       "Does this passage report a <class> event that occurred in the filing
        period? Answer yes/no and quote the evidence."
       goal: reject boilerplate, historical references, hypotheticals
  -> temporal validation
       reject events whose date precedes the filing period
  -> structured event with evidence span
```

Why this shape rather than fine-tuning:

- **The failure modes are semantic, not lexical.** "Assets acquired" vs "we
  acquired Acme" is a meaning distinction. A verifier that reads the sentence
  can make it; a regex cannot.
- **No training data needed.** Fine-tuning needs thousands of labels; we have
  149. The verifier needs a good prompt and a labelled test set, which we have.
- **The candidate generator stays cheap and auditable.** Recall is its job, and
  recall is easy to measure — sample passages, check nothing was missed.
- **Temporal validation is a separate, testable step.** It is the one part that
  must be exactly right, because a wrong date corrupts the event study.

## The separation that must hold

**Extraction benchmark and return analysis stay apart.**

- Extraction is judged on precision and recall against labelled passages.
- Returns are judged on abnormal returns against a null distribution.
- A return result is only interpretable once extraction precision is high enough
  that the events are real.

Conflating them is how projects end up reporting a return "signal" that is
really a labelling artefact. The current 0.055 precision is exactly that risk
made visible.

## What would make this plan wrong

- **Recall turns out to be high** (say 0.7). Then the detector finds real events
  and the problem is precision alone — a verifier is still the right fix, but
  the candidate generator needs no work.
- **The zero-shot model scores well** (say 0.5 precision). Then step 4 is
  unnecessary; use the model directly and spend the effort on temporal
  validation.
- **The error taxonomy is dominated by one fixable pattern.** Then a rule change
  beats an LLM, and the prototype is over-engineering.

Each of these is cheap to check before building anything.

## Explicitly not doing

- **Fine-tuning.** 149 labels is not enough, and the failure modes are not yet
  characterised well enough to know what to train on.
- **More statistics on the event study.** The controls are sound; the input is
  the problem.
- **A UI, Docker, or deployment.** Nothing worth showing until extraction works.
