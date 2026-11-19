# Precision protocol

How to label the 150-event sample, and what the resulting numbers do and do not
mean. Read this before labelling; the rubric is what makes the labels
comparable across sessions.

## What is being measured

**Precision:** of the events the detector reported, what share are real?

The sample is drawn from *detected* events, stratified by class so that rare
classes are represented. It is **not** a sample of filing text, so it cannot
measure **recall** — the events the detector missed are absent by construction.
Measuring recall requires independently sampling passages (including documents
where the detector found nothing) and labelling those. That is a separate
exercise and is not attempted here.

## The rubric

For each snippet, answer one question: **is this really a `<detected_type>`?**

### Label it as the detected class when

- The filing **reports** the event as having happened or being agreed:
  *"We completed the acquisition of Acme Corp."*
- The company **states** a change in outlook: *"We raised our full-year
  guidance."*
- A named role or person changed: *"Our CFO resigned effective immediately."*
- A legal outcome is reported: *"We recorded a settlement of $50 million."*

### Label it `none` when

| Category | Example | Why |
|---|---|---|
| **Boilerplate** | "Our acquisition strategy focuses on tuck-in deals." | Describes a policy, not an event |
| **Accounting** | "Property and equipment acquired was $1.2 billion." | Capex, not a corporate action |
| **Hypothetical** | "An acquisition could harm our business." | Conditional, not reported |
| **Historical reference** | "In 2019 we acquired Mellanox." | Real, but not this filing's event |
| **Forward-looking** | "We will need to accurately forecast components." | Forecasting as an activity |
| **Cross-reference** | "See Item 1A for litigation risk." | Points elsewhere; no event here |
| **Different class** | Detected `acquisition`, actually a divestiture | Label the *correct* class instead |

### Label it `skip` when

The snippet is genuinely ambiguous — truncated mid-sentence, or the event is
real but the class is arguable. Skipped items are excluded from every metric
rather than scored as errors.

### The historical-reference rule

A real event from a prior period is **not** an event of this filing. If the
snippet says "in fiscal 2024 we acquired X" and the filing is from 2026, label
`none` and note the year. This matters because the event study anchors returns
on the filing date; a 2024 event dated 2026 would corrupt the measurement.

## How to label

```bash
make review-precision     # one keystroke per snippet, saved after each answer
make precision            # per-class precision with Wilson intervals
```

The reviewer shows the detected class, the source filing and section, the
sentence, and surrounding context. **No proposed label is shown** — the
detector's own answer would anchor the judgement and inflate the measured
precision.

## Reading the output

```
event_type   labelled  correct  precision  ci_low  ci_high  ci_width
acquisition        60       41      0.683   0.557    0.786     0.229
```

- **`precision`** is the point estimate.
- **`ci_low`/`ci_high`** is a 95% Wilson score interval. It is not symmetric and
  stays inside [0, 1], which the textbook normal approximation does not.
- **`ci_width`** is the honest measure of how much the sample supports. A width
  of 0.23 means the true precision could plausibly be anywhere in that range.

### Rules for reporting

1. **Never quote a precision without its interval.**
2. **Never quote a class with fewer than 10 labels** as a finding. The tool
   flags these; the flag is not a formality.
3. **Never average across classes** to get a headline number. Pooled precision
   is dominated by whichever class was sampled most.
4. **Never call this accuracy.** It is precision on detected events only.

## What a good result looks like

Precision of 0.7–0.9 for `acquisition` and `legal_or_regulatory` would mean the
detector is usable and the negative event-study result is about the *market*,
not the detector. Precision near 0.3 would mean the events are mostly noise, and
the negative result says nothing about whether filings move prices.

That distinction is the whole point of this exercise: it turns "no signal" into
either "no signal in real events" or "no real events to find a signal in".

## Known limitations

- **One annotator.** No inter-annotator agreement, so the noise floor of the
  labels is unknown. A difference smaller than that floor cannot be interpreted.
- **Single rubric pass.** The rubric above was written before labelling, which
  is better than after, but it has not been piloted.
- **Large caps only.** 41 US issuers, 2025–2026.
- **Precision only.** Recall is not measured, so a detector that finds 10% of
  real events could still score 0.9 precision.
