# Methodology

How every number in this project is produced, and what each safeguard protects
against. Read this before quoting a result.

- [1. Annotation rubric](#1-annotation-rubric) — how a passage is labelled
- [2. Sampling](#2-sampling) — how passages are chosen
- [3. Event study](#3-event-study) — how returns are measured
- [4. Statistical controls](#4-statistical-controls) — what makes a result credible
- [5. LLM annotation](#5-llm-annotation) — how AI labels are produced and kept separate

---

## 1. Annotation rubric

The rubric is the contract between every annotator — human or model. It is
embedded verbatim in the LLM prompt, so the two cannot drift apart.

### Label it as an event when

- The filing **reports** the event as having happened or being agreed:
  *"We completed the acquisition of Acme Corp."*
- The company **states** a change in outlook: *"We raised our full-year
  guidance."*
- A named role or person changed: *"Our CFO resigned effective immediately."*
- A legal outcome is reported: *"We recorded a settlement of $50 million."*

### Label it `none` when

| Category | Example | Why |
|---|---|---|
| Boilerplate | "Our acquisition strategy focuses on tuck-in deals." | A policy, not an event |
| Accounting | "Property and equipment acquired was $1.2 billion." | Capex, not a corporate action |
| Hypothetical | "An acquisition could harm our business." | Conditional, not reported |
| Historical | "In 2019 we acquired Mellanox." | Real, but not this filing's event |
| Forward-looking | "We will need to accurately forecast components." | Forecasting as an activity |
| Cross-reference | "See Item 1A for litigation risk." | Points elsewhere |
| Different class | Detected `acquisition`, actually a divestiture | Label the correct class |

### Label it `skip` when

The passage is truncated, or the event is real but the class is genuinely
arguable. Skipped items are excluded from every metric rather than scored as
errors.

### The four disambiguating rules

These exist because each one was a real disagreement between annotators.

**Historical reference.** A real event from a prior period is not an event of
this filing. The event study anchors returns on the filing date, so a 2024 event
dated 2026 would corrupt the measurement.

**Undated event.** If the passage does not state when the event occurred, and
the context does not establish it as in-period, label `none`. An undated event
cannot be anchored in time and is unusable regardless of whether it happened.

| Passage | Label | Why |
|---|---|---|
| "Sales from GMS, which was acquired on September 4, 2025" | `none` | Date precedes the period |
| "Growth in customers of our acquired fiber business" | `none` | No date, no actor |
| "In March 2026, we completed the acquisition of a wafer fab" | `acquisition` | Dated, in-period |

**Unclosed transaction.** An announced but not-yet-completed deal is `none`. The
study measures what happened after a filing; a deal that may still fail has no
outcome to measure.

| Passage | Label | Why |
|---|---|---|
| "The acquisition of Wiz is expected to close in 2026" | `none` | Not completed |
| "We entered into a definitive agreement to acquire Acme" | `none` | Agreed, not closed |
| "We completed the acquisition of Acme on 3 March 2026" | `acquisition` | Closed |

**Heading.** A heading, table-of-contents entry or table row is `none`. These
label content elsewhere; they are not statements about events.

| Passage | Label | Why |
|---|---|---|
| "Acquisition of Silicon Labs" | `none` | A heading |
| "Acquisitions, net of cash acquired and divestitures (516) (291)" | `none` | A cash-flow row |

**Descriptive mention.** If the filing reports an acquisition that occurred in
the period, label it `acquisition` even when the sentence is descriptive rather
than an announcement — *"Our acquisition of Semrush in April 2026 enhances our
offerings."*

---

## 2. Sampling

Two samples, answering different questions. Neither can answer the other's.

### Precision sample — drawn from detections

Stratified by event class, seeded, so rare classes are represented rather than
swamped by acquisitions. Answers: *of the events the detector reported, how many
are real?*

It **cannot** measure recall: the events the detector missed are absent by
construction.

### Recall sample — drawn from passages

Paragraphs sampled independently of the detector, then filtered to those
mentioning event vocabulary. Answers: *of the real events present, how many did
the detector find?*

The filter is deliberately generous — it over-includes rather than missing a
real event, because under-inclusion would bias recall upward.

### Held-out batch

A second precision sample drawn with `--exclude`, so no passage is judged twice.
This is the only clean set: the development set's rubric was tuned against it.

### Known limits

- Large-cap US issuers only, 2025–2026. No small caps, ADRs or delistings.
- One annotator, so the noise floor of the labels is unknown.
- Passage-level, not document-level: a document with one event among fifty
  paragraphs contributes one positive passage.

---

## 3. Event study

### Where the clock starts

EDGAR accepts filings after the close — `acceptance_datetime` is typically
20:00–22:00 UTC. The last price that could **not** have reacted is that
session's close, so each event is anchored on the **last session whose close
precedes the acceptance timestamp**:

```
close(t0)  ->  close(t0 + h) / close(t0) - 1        t0 = anchor close
```

A 21:00 acceptance anchors on the same session; a 10:00 one anchors on the
previous session, so the whole of the following day counts as reaction.

Two mistakes this rules out:

- anchoring on the *next* session, which discards the overnight gap where most
  of the reaction happens;
- dating the event by `filing_date` and counting a return that begins *before*
  the news — look-ahead bias.

Anchoring at or before the close does not claim you could have traded at that
close. It fixes the window in which the market digested the news.

### Clustering

Ten mega-caps file within days of each other, so 827 events can sit on 52
distinct dates. Events sharing an anchor date share a return window and are not
independent trials. Events are averaged within `(event_type, event_date)` first,
so the sample size is the number of dates.

---

## 4. Statistical controls

Three controls, each removing a specific way a result could be spurious.

**Benchmark adjustment.** A −1% median after *every* class is a statement about
the sample period, not about events:

```
abnormal_h = return_h(stock) - return_h(benchmark)
```

The benchmark is aligned to the same anchor date. A missing benchmark bar gives
`NaN`, never a silent fallback to the raw return.

**Clustering.** See above. `n_dates` is printed next to every statistic because
it, not the event count, is the effective sample size.

**Null distribution.** 1 000 seeded draws of non-filing days per ticker, with
any day within 30 calendar days of a real anchor excluded. Reports the event
median against the null's 5th/95th percentiles and a two-sided empirical
p-value — distribution-free, because these returns are fat-tailed.

### What counts as a result

A class is only called significant if **all** hold:

1. the empirical p-value is small after benchmark adjustment;
2. the sample size is the number of dates, not events;
3. the effect is not explained by the null distribution;
4. the direction is stable across horizons.

Anything else is reported as "no measurable effect in this sample".

### Intervals

Proportions use **Wilson score intervals**, not the normal approximation, which
produces bounds outside [0, 1] at small samples and near 0 or 1. A class with 5
items and 5 correct answers has a true precision plausibly anywhere from 0.57 to
1.0.

---

## 5. LLM annotation

### What the model sees

The prompt contains the rubric, the class list, filing metadata and the passage.
It **never** contains the human label, the detector's prediction, or any other
annotator's output. A test asserts that two passages with identical text but
different labels produce byte-identical prompts.

### Structured output

Responses are validated by a Pydantic schema before storage:

- `label` is an integer index into a frozen class list, so the model cannot
  invent a class;
- `event_date` accepts ISO dates only — `"April 2026"` and `"Q3"` are rejected,
  forcing `null`, because a guessed date would corrupt the event study;
- `label` and `event_type` must agree, and `is_genuine_event` must be consistent
  with the class.

### Provenance

Every stored annotation records the model id, prompt version, rubric hash,
timestamp, source filing, sampling method and `human_verified: false`.

### AI labels are never ground truth

`human_verified` stays `false` until a human overrides it. The comparison module
treats human labels as the reference, never the reverse.

### Evaluation safeguards

- Never train on evaluation examples.
- Grouped splits by company and underlying event, so passages from one filing
  never straddle train and test.
- Freeze the test set before model development.
- Do not repeatedly optimise prompts against the held-out set.
- Keep enriched and representative sampling results separate.
- Never report precision measured on an enriched sample as corpus-wide
  precision.
