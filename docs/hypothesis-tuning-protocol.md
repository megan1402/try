# Hypothesis tuning protocol

How to revise the zero-shot model's hypotheses without fooling ourselves.

## The problem this guards against

The current hypotheses score **0.156 precision** at best (threshold 0.5). The
temptation is to rewrite them, re-run on the same 149 labels, and report the
improvement. That measures **overfitting to the labels**, not better detection.

The fix is a held-out set that is never looked at during tuning.

## The data

| set | items | real events | role |
|---|---|---|---|
| `events_precision.jsonl` | 149 | 10 | **tuning** |
| `events_holdout.jsonl` | ~150 | ~10 | **test — do not look until the end** |
| `passages_events.jsonl` | 38 | 2 | secondary check |

The holdout batch is sampled with `--exclude events_precision.jsonl`, so no
snippet appears in both. It is built with a different seed.

**Rule: the holdout file is not opened, scored, or inspected until the revised
hypotheses are frozen.** Not even to "check it looks right".

## Step 1 — preserve the baseline

Before changing anything, record:

- the current `HYPOTHESES` table (verbatim)
- precision and recall at every threshold on the tuning set
- the per-class confusion matrix

This is the number to beat: **0.156 precision at threshold 0.5**.

## Step 2 — analyse the errors

For each false positive on the tuning set, record *why* it fired. The categories
seen so far:

| category | example | what the hypothesis must exclude |
|---|---|---|
| Accounting vocabulary | "Acquisitions, net of cash acquired and divestitures (516)" | cash-flow table rows |
| Historical reference | "In July 2025, the company completed its acquisition of Hess" | prior-period events |
| Policy description | "Cybersecurity assessments for acquired companies" | practices, not events |
| Hypothetical | "An acquisition could harm our business" | conditional statements |
| Section heading | "Structural Changes, Acquired Brands" | table-of-contents entries |

For each false negative, record what the model *did* predict instead — usually
`none`, which tells you the `none` hypothesis is winning when it should not.

## Step 3 — revise the hypotheses

Two changes are expected to matter most:

**1. Make `none` explicit about the reporting period.** The current hypothesis is
"the text describes ordinary business results or boilerplate risk factors". It
does not mention *when*. A better one:

> "This passage does not describe a concrete corporate event occurring within
> the reporting period."

That directly targets the historical-reference failure, which is the largest
category.

**2. Make each event hypothesis require a concrete occurrence.** The current
"the company acquired another company or business" is satisfied by a mention.
A better one:

> "The company reports that it completed or agreed an acquisition of another
> company during the period."

The words "reports that it completed or agreed" exclude policy descriptions and
hypotheticals.

### Rules for revising

- **Change one thing at a time** where possible, so the effect is attributable.
- **Keep hypotheses mutually distinguishable.** If two classes become
  near-synonyms, the model cannot separate them.
- **Do not add class-specific keywords** ("subpoena", "layoffs"). That is
  re-implementing the regex inside a prompt, and it will not generalise.
- **Write down each revision and its rationale** before measuring.

## Step 4 — evaluate on the tuning set

Run the threshold sweep on `events_precision.jsonl` only. Record precision and
recall at every threshold.

If the best precision does not exceed 0.156, **stop**. The hypotheses are not the
problem, and the holdout set stays untouched for a future experiment.

## Step 5 — evaluate once on the holdout

Only if step 4 improved. Run the same sweep on `events_holdout.jsonl`, once.

Report:

- precision and recall with Wilson intervals
- macro-F1
- per-class confusion matrix
- **the gap between tuning and holdout**

A large gap means the revision overfitted. A small gap means it generalised.

## What counts as success

| tuning precision | holdout precision | reading |
|---|---|---|
| 0.30 | 0.28 | generalised — real improvement |
| 0.30 | 0.12 | overfitted — the revision learned the tuning set |
| 0.18 | 0.16 | marginal — not worth the complexity |

The holdout number is the one to report. The tuning number is a means, not a
result.

## What this cannot fix

- **12 real events is small.** Even a perfect split leaves intervals spanning
  most of [0, 1]. A difference of 0.1 will not be distinguishable from noise.
- **One annotator.** The labels themselves have unknown noise.
- **The model is 330 MB of NLI.** A better prompt may not rescue a model that
  was never trained on filing prose.

If the holdout result is inconclusive, that is the honest finding — and it
strengthens the case for fine-tuning, which needs more labels anyway.
