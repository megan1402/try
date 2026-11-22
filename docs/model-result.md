# Model scoring result (2026-11-22)

The zero-shot NLI model (`cross-encoder/nli-distilroberta-base`) scored on the
same labelled data as the rule baseline.

## Result

| | rules | zero-shot | reading |
|---|---|---|---|
| **Precision** (149 events) | 0.055 [0.028, 0.104] | **0.055** [0.028, 0.105] | identical |
| **Recall** (38 passages) | 0.000 (0/2) | **0.500** (1/2) | model finds events the rules miss |
| Fires on event-free passages | 2/36 | 34/36 | model fires on almost everything |

The precision figures are not a coincidence: both predictors got 8 of 145
predicted events right, and both got the same 8.

## What this actually shows

**The model is not better. It is differently broken.**

- **Precision is identical (0.055).** The model fires on 145 of 149 snippets —
  it almost never says `none`. Its precision is therefore dominated by the base
  rate, exactly like the rules.
- **Recall is higher (0.500 vs 0.000).** On the 38 event-bearing passages it
  found 1 of 2 real events, where the rules found 0. But it also fired on 34 of
  the 36 event-free passages, so the recall comes from firing indiscriminately,
  not from understanding.
- **The one real win is `management_change`.** The model caught the NVIDIA CFO
  resignation that the rules missed — the verb-before-role case. That is a
  genuine capability difference, and it is the case a verifier could exploit.

## Why precision did not improve

The model's `none` hypothesis is:

> "the text describes ordinary business results or boilerplate risk factors"

Filing prose is *mostly* ordinary business results, so this hypothesis should
win often. It does not — the model assigns higher entailment to the event
hypotheses. That is a calibration failure, not a knowledge failure: the model
can tell an acquisition from a divestiture, but it cannot tell an event from
non-event prose.

This is fixable in principle by thresholding: require a minimum entailment score
before accepting an event. The current code takes the argmax with no threshold,
which guarantees it always fires.

## Against the pre-registered decision rule

`docs/model-scoring-plan.md` said:

| model precision | next step |
|---|---|
| ≥ 0.3 | build the verifier prototype |
| 0.1 – 0.3 | inspect errors first |
| < 0.1 | neither approach works off the shelf |

**Precision 0.055 falls in the third band.** By the rule stated in advance,
neither approach works off the shelf.

But the rule was written assuming precision and recall move together. They did
not: recall improved substantially while precision stayed flat. That is a
different situation from "the model is no better", and it deserves a different
response.

## The honest reading

The model demonstrates a capability the rules lack — it can recognise a
management change phrased in an unanticipated way. It fails at the thing the
rules also fail at: deciding whether a passage is about an event at all.

That is exactly the split the proposed architecture assumes:

```
candidate generator  -> high recall, low precision   (the model, unthresholded)
verifier             -> high precision               (needs to be built)
```

The model is a usable **candidate generator**. It is not a usable detector.

## What to do next

**Threshold the model before concluding anything.** The current run uses argmax
with no minimum score. A threshold sweep — accept an event only above score
*t* — would show whether precision and recall can be traded against each other
usefully. That is a small change and it directly tests whether the model's
scores carry signal.

If a threshold exists where precision exceeds 0.3 while recall stays above 0.3,
the model is a viable candidate generator and the verifier prototype is
justified. If no threshold works, the model's scores carry no usable signal and
the conclusion is firmer.

**This is not fine-tuning, and it is not a new architecture.** It is one
parameter, measured.

## What this does not say

- **Not a verdict on LLMs.** This is a 330 MB NLI model with hand-written
  hypotheses. A modern LLM with a verification prompt is a different experiment.
- **Not a recall estimate.** 2 real events in the sample; the interval is
  0.095–0.905.
- **Not evidence about returns.** Extraction still does not work well enough for
  the event study to mean anything.
