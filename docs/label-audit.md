# Test partition label audit

The test partition was labelled in a separate session from the rubric. This
audit compares every positive label against the rubric in
[`methodology.md`](methodology.md) and records where they disagree.

The audit was written **after** the LLM was run, and the LLM's answers were
visible while writing it. That is a real limitation: a reviewer who has seen the
model's answer is anchored by it. The audit is therefore a list of *rubric
contradictions*, each of which can be checked against the rubric text alone, not
a claim about which label is right in a hard case.

## Why duplicates exist

`sample_events` draws from **detected events**, not from documents. A sentence
that appears verbatim in every filing a company files — "We have entered into
settlement agreements with other responsible parties on two of the orders" —
was detected once per filing and sampled once per filing.

`events_holdout.jsonl` has 150 rows but only 124 distinct texts. The test
partition has 70 rows but only 59 distinct texts.

Fixed in `3de1a88`: both samplers now keep the first occurrence of each
normalised text.

## Contradictions

### 1. Same sentence, different labels

| ids | text | labels |
|---|---|---|
| ev0138, ev0142, ev0146 | "In fiscal 2025, the Company initiated a restructuring program of its Surgery franchise…" | `none` ×3 |
| ev0139 | "Restructuring: In fiscal 2025, the company initiated a restructuring program of its Surgery franchise…" | `restructuring` |

The only difference is the `Restructuring:` heading prefix and the capitalisation
of "Company". Under the rubric's **heading** rule a heading is `none`, but the
heading here is a prefix on a sentence that states an event, not a standalone
heading. Either all four are `restructuring` or all four are `none`.

| ids | text | labels |
|---|---|---|
| ev0112 | "As previously disclosed, on February 11, 2026, we entered into a settlement agreement with the U." | `none` |
| ev0115 | "On February 11, 2026, we entered into a settlement agreement with the U." | `legal_or_regulatory` |

Same event, same date, same truncated sentence. The lead-in clause differs.

### 2. Historical events labelled as current

The rubric's **historical reference** rule: a real event from a prior period is
not an event of this filing.

| id | text | label | rubric |
|---|---|---|---|
| ev0149 | "We recognized $186 million of restructuring charges in **2024**…" | `restructuring` | `none` |
| ev0110 | "…and in **2019** we divested the majority of our 5G smartphone modem business" | `divestiture` | `none` |

### 3. Descriptive mentions labelled `none`

The rubric's **descriptive mention** rule: if the filing reports an acquisition
that occurred in the period, label it `acquisition` even when the sentence is
descriptive.

| id | text | label | rubric |
|---|---|---|---|
| ev0060 | "…with the acquisition of Silo AI Oy (Silo AI), an AI lab based in Finland." | `none` | `acquisition` |
| ev0072 | "The acquisition of Pensando and its leading distributed services platform expands our ability…" | `none` | `acquisition` |

Both are undated in the snippet, which the **undated event** rule would send to
`none`. But ev0060's context establishes the acquisition as in-period, and
ev0072's context gives the purchase consideration. These are genuinely arguable
and are the weakest entries in this audit.

### 4. Accounting statement labelled as an event

| id | text | label | rubric |
|---|---|---|---|
| ev0105 | "The excess of the net proceeds from the IPO over the net book value of the Johnson & Johnson divested interest was $2.5 billion…" | `divestiture` | `none` |

This is an accounting entry about IPO proceeds, not a report of a divestiture.
The rubric's **accounting** rule covers it.

### 5. Correctly labelled, model wrong

| id | text | label |
|---|---|---|
| ev0136 | "Restructuring and other expenses in fiscal 2026 consist of employee severance costs…" | `none` |
| ev0137 | "…we recognized $12 million in restructuring charges…" | `none` |
| ev0143 | "Restructuring expenses in the first quarter of fiscal 2026 primarily related to…" | `none` |

Undated expense lines. The rubric's **undated event** rule says `none`, and the
human said `none`. The model said `restructuring`. The model is wrong here.

## What this means for the reported numbers

The LLM scored precision 0.526 [0.317, 0.727] and recall 0.833 [0.552, 0.953] on
this partition. Of the 12 positives, at least 6 are contradicted by the rubric
above. The precision figure is therefore measuring annotator inconsistency as
much as model error, and should not be quoted as a model quality estimate.

The rubric was **not** changed in response to these numbers. The contradictions
are between the labels and the rubric as it stood before the test run.
