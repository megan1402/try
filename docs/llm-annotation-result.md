# Phase 1 result: LLM annotation (2026-11-22)

DeepSeek `deepseek-chat` annotated all 187 human-labelled passages, blind to the
human labels and to the rule detector. This is the first measurement of whether
an LLM can reproduce the annotations.

## Result

### Event versus none

| metric | value | 95% CI | n |
|---|---|---|---|
| precision | **0.153** | 0.088 – 0.253 | 72 |
| recall | **0.917** | 0.646 – 0.985 | 12 |
| specificity | 0.645 | 0.571 – 0.713 | 172 |
| accuracy | 0.663 | 0.592 – 0.727 | 184 |

### Per class

| class | support | predicted | tp | fp | precision | recall |
|---|---|---|---|---|---|---|
| acquisition | 7 | 37 | 6 | 31 | 0.162 | 0.857 |
| legal_or_regulatory | 2 | 18 | 2 | 16 | 0.111 | 1.000 |
| management_change | 3 | 1 | 1 | 0 | 1.000 | 0.333 |
| restructuring | 0 | 14 | 0 | 14 | 0.000 | — |
| divestiture | 0 | 2 | 0 | 2 | 0.000 | — |
| none | 172 | 112 | 111 | 1 | 0.991 | 0.645 |

## What this shows

**The LLM finds events the rules miss, and fires on far too many.**

- **Recall 0.917** — it caught 11 of 12 real events. The rule baseline caught 0.
  This is a genuine capability difference.
- **Precision 0.153** — but it predicted 72 events where 12 exist. It over-fires
  by roughly 6×.
- **Accuracy 0.663 is *below* the always-`none` baseline of 0.935.** The model
  is worse than a constant predictor on this data, because it fires so often.

That last point is the one to keep in mind: any report of "66% accuracy" here
would be misleading. The dataset is 93% `none`.

## The failure mode is the rubric, not the model

Reading the disagreements, the model's reasoning is coherent and cites the
rubric correctly. It fails on one specific ambiguity:

| id | passage | human | model |
|---|---|---|---|
| ev0001 | "up 10 percent... due to the acquisition of Hess" | none | acquisition |
| ev0007 | "sales from GMS, which was acquired on September 4, 2025" | none | acquisition |
| ev0011 | "On December 8, 2025, we completed the acquisition of Spirit AeroSystems" | none | acquisition |
| ev0026 | "The acquisition of Wiz is expected to close in 2026" | none | acquisition |

Three distinct issues, all of them rubric gaps:

1. **Undated or prior-period acquisitions.** The rubric says prior-period events
   are `none`, but does not say how to judge an acquisition whose date is stated
   as 2025 in a 2026 filing. The model read "in-period" from context; the human
   did not.
2. **Proposed transactions.** "Expected to close in 2026, subject to regulatory
   approvals" is not a completed acquisition. The rubric mentions hypotheticals
   but not announced-but-unclosed deals.
3. **Section headings.** "Acquisition of Silicon Labs" is a heading, not a
   sentence. The model labelled it an event.

The model is not ignoring the rubric — it is applying it where the rubric is
silent.

## What this does *not* show

- **Not a verdict on LLMs.** One model, one prompt, one rubric. A frontier model
  or a revised rubric could move these numbers substantially.
- **Not a test-set result.** These 187 labels are a development set. The rubric
  was tuned on them once already, and the rule baseline was measured on them.
- **Not a recall estimate for the corpus.** 12 real events; the interval is
  0.646–0.985.

## Comparison with the rule baseline

| | rules | DeepSeek |
|---|---|---|
| precision | 0.055 | **0.153** |
| recall | 0.000 | **0.917** |
| fires on event-free passages | 2/36 | 61/172 |

The LLM is better on both axes, but its precision is still far too low to use
directly. It is, however, a plausible **candidate generator** — which is exactly
the role `docs/next-experiment.md` proposed, and which the NLI model could not
fill.

## Recommendation

**Do not use AI labels as ground truth.** At 0.153 precision, 85% of its
positive labels would be wrong.

**Do use it as a candidate generator**, with a human or a stricter verifier
filtering its output. The recall is high enough that filtering is worthwhile:
it finds events the rules miss entirely.

**Fix the rubric before the next run.** Three rules are missing:

1. An acquisition whose stated date precedes the filing period is `none`.
2. An announced but unclosed transaction is `none`.
3. A heading or table-of-contents entry is `none`.

Those are rubric clarifications, not model tuning — and they should be applied
to the holdout run, not re-measured on this development set.

## Cost and runtime

- 187 passages, ~2 minutes wall clock
- Approximately **$0.05** at DeepSeek pricing
- 0 failures, 0 retries needed

## Provenance

Every annotation in `tests/fixtures/llm_annotations.jsonl` records the model id,
prompt version, rubric version, timestamp, source filing and
`human_verified: false`.
