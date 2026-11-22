# Threshold sweep result (2026-11-22)

The zero-shot model's acceptance threshold swept from 0.0 to 0.9 on both
labelled sets.

## Result: shape C — the scores carry no usable signal

### Events (149 labelled)

| threshold | predicted | precision | recall |
|---|---|---|---|
| 0.0 | 146 | 0.055 | 0.800 |
| 0.2 | 129 | 0.062 | 0.800 |
| 0.3 | 95 | 0.084 | 0.800 |
| 0.4 | 80 | 0.100 | 0.800 |
| 0.5 | 45 | **0.156** | 0.700 |
| 0.6 | 25 | 0.120 | 0.300 |
| 0.7 | 8 | 0.125 | 0.100 |
| 0.8 | 0 | — | 0.000 |

### Passages (38 labelled)

| threshold | predicted | precision | recall |
|---|---|---|---|
| 0.0 | 38 | 0.026 | 0.500 |
| 0.3 | 15 | 0.067 | 0.500 |
| 0.5 | 6 | **0.167** | 0.500 |
| 0.6 | 1 | 0.000 | 0.000 |

**No threshold reaches precision ≥ 0.3 with recall ≥ 0.3 on either set.**

## What the curve shows

Precision rises from 0.055 to a peak of **0.156** at threshold 0.5, then falls
back. That peak is the whole story:

- **The best precision the model can achieve is 0.156** — roughly one in six
  predictions correct, at the cost of losing 10% of recall.
- **Above 0.5, precision does not improve.** It drops to 0.120 and 0.125. If the
  scores carried signal, higher confidence would mean higher precision. It does
  not.
- **The model's confidence is essentially unrelated to its correctness.** That is
  the definition of scores carrying no usable signal.

This is shape C from `docs/threshold-plan.md`, written before the run:

> **The scores carry no usable signal.** The model's confidence is unrelated to
> its correctness. This would be the firmest negative result: not "the model
> needs tuning" but "this model's outputs are not informative for this task".

## What this settles

| approach | precision | recall | verdict |
|---|---|---|---|
| Rule baseline | 0.055 | 0.000 | does not detect |
| Zero-shot, argmax | 0.055 | 0.500 | fires on everything |
| Zero-shot, best threshold | 0.156 | 0.700 | still unusable |

**Neither approach works off the shelf.** The zero-shot model's apparent recall
advantage was an artefact of never abstaining; once it is allowed to abstain, its
precision improves only to 0.156 and its confidence proves uninformative.

## Why this is a useful result, not a dead end

It rules out the cheapest hypothesis. Before this sweep, "the model might be
better with a threshold" was a live possibility that would have justified
building a verifier on top of it. It is now closed, and closed cheaply — one
parameter, measured on data that already existed.

It also narrows what remains:

1. **The hypotheses are untuned.** A better prompt could move these numbers. This
   sweep tested one configuration, not the model's ceiling.
2. **This is a 330 MB NLI model, not an LLM.** A verifier built on a modern LLM
   with a verification prompt is a different experiment.
3. **Fine-tuning remains untested** — and now has a clear baseline to beat
   (0.156 precision at 0.700 recall).

## What it does not say

- **Not a verdict on LLMs.** DistilRoBERTa-NLI is not GPT-class.
- **Not a verdict on the task.** It says these two off-the-shelf approaches fail,
  not that event extraction is impossible.
- **Not evidence about returns.** Extraction still does not work well enough for
  the event study to mean anything.

## Consequence for the roadmap

The candidate-generator + verifier architecture in `docs/next-experiment.md`
assumed the model could serve as a candidate generator. **It cannot** — a
generator with 0.156 precision at 0.700 recall produces mostly noise for the
verifier to sift.

Two options remain, and they should be chosen deliberately:

**Option 1 — tune the hypotheses.** The `HYPOTHESES` table was written once and
never revised. A prompt-engineering pass, measured on the same labels, is cheap
and directly tests whether the model's ceiling is higher than this configuration
suggests.

**Option 2 — accept that off-the-shelf fails and go to fine-tuning.** The
labelled sets (149 events, 38 passages) are too small to train on, but they are
enough to *evaluate* a fine-tuned model, and the error taxonomy would say what to
train on.

Option 1 is cheaper and should come first. If a revised hypothesis set cannot
beat 0.156, the case for fine-tuning is much stronger.
