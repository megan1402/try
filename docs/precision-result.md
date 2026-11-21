# Precision result (2026-11-21)

The first measured precision of the rule-based event detector, on a stratified
sample of 149 detected events drawn from 41 issuers and 516 filings.

## Result

| event type | labelled | correct | precision | 95% CI (Wilson) |
|---|---|---|---|---|
| acquisition | 79 | 6 | **0.076** | 0.035 – 0.156 |
| legal_or_regulatory | 19 | 2 | **0.105** | 0.029 – 0.314 |
| restructuring | 18 | 0 | **0.000** | 0.000 – 0.176 |
| divestiture | 10 | 0 | **0.000** | 0.000 – 0.278 |
| guidance_raise | 7 | 0 | 0.000 | 0.000 – 0.354 |
| earnings_beat | 5 | 0 | 0.000 | 0.000 – 0.434 |
| guidance_cut | 4 | 0 | 0.000 | 0.000 – 0.490 |
| earnings_miss | 3 | 0 | 0.000 | 0.000 – 0.561 |
| management_change | 1 | 0 | 0.000 | 0.000 – 0.793 |

**Pooled precision: 0.055 [0.028, 0.104]** over 146 scored items.

## What this means

**The detector is the bottleneck.** Of 149 events it reported, 8 were real.

This resolves the question the event study could not answer. The negative
predictive result — no abnormal returns after any event class — is **not**
evidence that filings do not move prices. It is evidence that the events being
measured were mostly not events. The study measured accounting notes, historical
references and boilerplate.

The correct statement of the project's status:

> Filing events, as detected by the current rule baseline, show no measurable
> abnormal return. But that baseline has a measured precision of 5.5%, so the
> study tested almost nothing. Whether real filing events predict returns
> remains **untested**.

## Why the number is this low

Sampling the failures shows three recurring causes:

1. **Accounting vocabulary.** "Acquisitions, net of cash acquired and
   divestitures (516) (291)" is a cash-flow table row. "Shares acquired under
   the repurchase programme" is a buyback.
2. **Historical references.** "In July 2025, the company completed its
   acquisition of Hess" is real, but it is not this filing's event. The event
   study anchors returns on the filing date, so counting it would corrupt the
   measurement.
3. **Descriptive mentions without an event.** "Cybersecurity assessments for
   acquired companies" describes a practice.

The precision filters added earlier cut the event count by 91% but did not fix
the underlying problem: **matching a keyword is not detecting an event.**

## What this does *not* say

- **Not** that the rules are worthless. `legal_or_regulatory` at 0.105 is still
  better than the 0.0 of five other classes, and the filters did remove the
  worst false positives.
- **Not** that a model would do better. That is untested. The zero-shot
  comparison in `app/models/` has never been scored against these labels.
- **Not** a recall measurement. A sample of detected events cannot contain the
  events the detector missed. Recall is unknown and could be anything.

## Method notes

- **Stratified sample**, seeded (`20260119`), so the result is reproducible with
  `make sample-events`.
- **Blind labelling**: the sample shipped with no proposed label, so the
  reviewer judged the detector rather than agreeing with it.
- **One annotator**, no inter-annotator agreement, so the noise floor is
  unknown. The rubric was revised once mid-way (the descriptive-mention rule),
  which moved acquisition precision from 0.038 to 0.076 — a reminder that the
  rubric itself is a source of variance.
- **Small classes are uninterpretable.** Five classes have fewer than 10 labels;
  their intervals span most of [0, 1] and should not be quoted.

## Consequence for the roadmap

Steps 16–17 (benchmark set, model) are now the critical path, not the product
work. The order changes:

1. **Build a rule-blind benchmark** — sample filing passages, not rule hits, so
   recall becomes measurable.
2. **Score the zero-shot model** against the same labels, to see whether a model
   clears the 5.5% floor.
3. **Only then** decide whether event extraction is viable at all.

If a model cannot beat 5.5% precision on this task, the honest conclusion is
that keyword and zero-shot approaches are insufficient, and the project needs
either a fine-tuned classifier with real training data or a different event
definition.
