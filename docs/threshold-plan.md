# Threshold sweep: what to look for

Recorded 2026-11-22, before the sweep result. Written first so the reading cannot
be fitted to the outcome.

## The problem being tested

The zero-shot model's precision (0.055) exactly matched the rule baseline, and
the reason is mechanical: `predict()` takes the argmax with **no minimum score**.
Some class is always highest-scoring, so the model fires on 145 of 149 snippets.
Its precision is therefore the base rate, not a measure of understanding.

`predict_with_threshold()` makes abstention possible: if the best event score is
below `t`, the answer is `none`.

## What the sweep reports

For each threshold in 0.0, 0.1, … 0.9:

- **predicted_events** — how many snippets still get an event label
- **precision** — of those, how many are right
- **recall** — of the real events, how many are found

Both with Wilson intervals.

## The three possible shapes

### A. Precision rises, recall holds

```
threshold  predicted  precision  recall
0.0        145        0.055      0.800
0.5         40        0.300      0.600
0.7         15        0.500      0.400
```

**The scores carry signal.** The model knows which predictions it is confident
about, and confidence correlates with correctness. A threshold exists where both
precision and recall clear 0.3, and the model is a viable candidate generator.

### B. Precision rises only by collapsing recall

```
threshold  predicted  precision  recall
0.0        145        0.055      0.800
0.7          3        0.667      0.100
```

**The scores carry weak signal.** Precision improves only by predicting almost
nothing. Not useful as a candidate generator — a generator that finds 10% of
events is not a generator.

### C. Precision barely moves

```
threshold  predicted  precision  recall
0.0        145        0.055      0.800
0.5         90        0.070      0.700
0.9         20        0.100      0.200
```

**The scores carry no usable signal.** The model's confidence is unrelated to
its correctness. This would be the firmest negative result: not "the model needs
tuning" but "this model's outputs are not informative for this task".

## The decision rule, stated in advance

| shape | next step |
|---|---|
| **A** — precision ≥ 0.3 and recall ≥ 0.3 at some threshold | build the verifier prototype |
| **B** — precision improves but recall collapses | inspect errors; the taxonomy decides |
| **C** — precision stays near the base rate | neither rules nor this model work; fine-tuning or a different task definition |

## What this does not test

- **Not the hypotheses.** They are hand-written and untuned. A better prompt
  could move every number. This sweep tests *this* configuration.
- **Not an LLM.** A 330 MB NLI model, not a large language model with a
  verification prompt.
- **Not recall on the corpus.** 2 real events in the sample.

## Why this is worth doing before anything larger

It is one parameter, measured on data that already exists. If the answer is C,
building a verifier on top of this model would be building on sand — and that is
much cheaper to discover now than after the prototype.
