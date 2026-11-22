# Model scoring: what the comparison means

Recorded 2026-11-22, before the zero-shot model has been scored. Written first so
the interpretation cannot be fitted to the result.

## The baseline to beat

| | rule baseline |
|---|---|
| Precision (149 detected events) | **0.055** [0.028, 0.104] |
| Recall (38 event-bearing passages) | **0.000** (0 of 2) |

Sources: `docs/precision-result.md`, `docs/recall-result.md`.

## What is being scored

`app/models/zero_shot.py` — `cross-encoder/nli-distilroberta-base`, used
zero-shot. Each event class is a hypothesis sentence; the highest entailment
score wins, with `none` competing on equal terms.

Scored on the **same labelled data** as the baseline, so the comparison is
like-for-like:

- 149 stratified detected events → precision
- 38 event-bearing passages → recall

## How to read the result

### If the model beats the baseline clearly

Say precision ≥ 0.3 or recall ≥ 0.3. Then a model is worth pursuing, and the
next step is the candidate-generator + verifier architecture in
`docs/next-experiment.md`, using this model as the verifier.

### If the model is roughly the same

Say precision 0.05–0.15 and recall near zero. Then zero-shot NLI is no better
than regex for this task. That is a real finding: it means the problem is not
"we used the wrong tool" but "this task is hard for off-the-shelf models on
filing prose".

The next step would be a fine-tuned classifier — but only after the error
taxonomy (step 19) says what to train on.

### If the model is worse

Possible, and informative. Zero-shot NLI models are trained on general-domain
premises; filing prose is dense, table-heavy and full of accounting vocabulary.
A model that fires on everything would show low precision and high recall — the
opposite failure mode to the rules.

## What this does *not* settle

- **Not a verdict on LLMs.** This is a 330 MB NLI model, not a large language
  model. A verifier built on a modern LLM with a good prompt is a different
  experiment, and the one `docs/next-experiment.md` proposes.
- **Not a recall estimate for the corpus.** 38 passages, 2 real events. The
  interval spans most of [0, 1].
- **Not a fair test of the model's ceiling.** The hypotheses were written by
  hand and never tuned. A better prompt could move the numbers substantially —
  which is itself worth knowing.

## The comparison table to fill in

| | rules | zero-shot |
|---|---|---|
| precision | 0.055 | ? |
| recall | 0.000 | ? |
| fires on event-free passages | 2/36 | ? |

The third row matters as much as the first two: a model that fires on everything
has high recall and useless precision, and the confusion matrix will show it.

## Decision rule, stated in advance

1. **Model precision ≥ 0.3** → build the verifier prototype.
2. **Model precision 0.1–0.3** → inspect errors first; the taxonomy may point to
   a prompt fix rather than an architecture change.
3. **Model precision < 0.1** → neither approach works off the shelf. Fine-tuning
   becomes the only remaining option, and the error taxonomy decides what to
   train on.

No outcome justifies returning to the event study. Extraction must work first.
