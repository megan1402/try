# Recall result (2026-11-22)

Recall on event-bearing passages, measured from 38 labelled passages sampled
independently of the detector.

## Result

| event type | true positives | false negatives | recall | 95% CI |
|---|---|---|---|---|
| acquisition | 0 | 1 | **0.000** | 0.000 – 0.793 |
| management_change | 0 | 1 | **0.000** | 0.000 – 0.793 |

Confusion matrix:

| human label | detector answer | count |
|---|---|---|
| acquisition | none | 1 |
| management_change | none | 1 |
| none | acquisition | 2 |
| none | none | 34 |

**The detector found 0 of the 2 real events in the sample, and fired on 2 of the
36 passages that contained no event.**

## What this means

Combined with the precision result (0.055), the picture is now complete:

| | value | reading |
|---|---|---|
| Precision | 0.055 | usually wrong when it fires |
| Recall | 0.000 (0/2) | missed both real events in the sample |

**The rule baseline is not working, in either direction.** It is not a detector
that needs tuning; it is a detector that does not detect.

## The honest limits of this number

**n = 2 is very small.** The confidence interval spans 0.000 to 0.793, so the
true recall could be anything from "finds nothing" to "finds most of them". What
the sample *does* establish:

- The detector missed a real acquisition and a real management change that were
  present in event-bearing passages.
- It fired on 2 passages that contained no event.

That is consistent with the precision measurement and with the specific failure
modes observed earlier. It is not a precise recall estimate.

**This is recall on event-bearing passages**, not corpus-wide recall. Corpus-wide
recall would be lower still, because the denominator would include every passage
in every filing.

## The two misses, and why

**`pe0001` — management change.** "Following Mr. Huang's resignation, the Board
appointed NVIDIA's Executive Vice President and Chief Financial Officer Colette
M. Kress..."

The pattern requires a role *before* the verb ("the CFO resigned"). This sentence
puts the verb first ("Following Mr. Huang's resignation"). It never fires.

**The acquisition miss** is the same class of problem: the event is reported in
a form the pattern does not anticipate.

Both are **semantic** failures, not lexical ones. The words are present; the
structure is different. This is exactly the case a verifier that reads the
sentence can handle and a regex cannot.

## What this settles

The question from `docs/precision-result.md` — "is classification quality the
bottleneck?" — is answered: **yes, decisively.**

The event study's negative result is not evidence about the market. It is
evidence that the extractor produces almost no real events and misses most of
the real ones. Whether real filing events predict returns remains **untested**.

## Consequence

Step 18 (score the zero-shot model) is now the cheapest informative experiment
available. If a model cannot beat 0.055 precision and 0.000 recall on the same
labels, then keyword and zero-shot approaches are both insufficient, and the
answer is the candidate-generator + verifier architecture in
`docs/next-experiment.md`.

## Method notes

- **Blind labelling**: the detector's answers were in a sidecar file, never
  shown during labelling.
- **Filtered sample**: 516 passages → 38 that mention event vocabulary. The
  filter is deliberately generous, so it over-includes rather than missing a
  real event.
- **One annotator**, no inter-annotator agreement.
- **The detector saw only the passage**, not the whole document, so
  section-based filters could not help. Recall here is a lower bound on
  document-level performance.
