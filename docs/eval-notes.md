# Evaluation notes — golden set v0 (frozen)

Written 2026-11-16. This file exists so that the numbers produced by
`make score` are never quoted more strongly than they deserve. It records what
the 24 labelled snippets support, what they do not, and why the set was frozen
rather than extended.

## Decision (2026-11-16)

**Freeze this set and stop labelling. Improve evaluation later.** The pool is
too biased to serve as a benchmark, and extending it as constructed would deepen
the bias rather than fix it. Development moves on to return-based evidence
(`make study`), which needs no labels: it needs *filings*, not annotations.

Evaluation is revisited only when it can be built properly — rule-blind
sampling, per-class balance, 8-K earnings releases, several issuers, and blind
labelling with an inter-annotator check. The construction notes below are the
specification for that work.

## What was measured

24 of 60 snippets were labelled by hand: **13 `none`, 7
`legal_or_regulatory`, 4 `acquisition`**. No other class has support.

| label | support | predicted | tp | fp | fn | precision | recall |
|---|---|---|---|---|---|---|---|
| acquisition | 4 | 11 | 4 | 7 | 0 | 0.36 | 1.00 |
| divestiture | 0 | 1 | 0 | 1 | 0 | 0.00 | — |
| earnings_miss | 0 | 2 | 0 | 2 | 0 | 0.00 | — |
| legal_or_regulatory | 7 | 10 | 6 | 4 | 1 | 0.60 | 0.86 |
| none | 13 | 0 | 0 | 0 | 13 | — | 0.00 |

Macro precision 0.48, macro recall 0.62, macro F1 0.62, accuracy 0.42.

## What this *does* support

Two findings, both about the rule baseline's behaviour rather than its quality:

1. **The rules never abstain.** Of 13 snippets a human called `none`, the
   detector fired an event on all 13. Any detector that cannot say "nothing
   here" will have low precision on real text no matter how good its patterns
   are. This is a design gap, not a tuning problem.
2. **`acquisition` is the weak class and `legal_or_regulatory` the strong one.**
   Acquisition language ("acquired", "acquisition") is ordinary accounting
   vocabulary; litigation language ("subpoena", "investigation", "settlement")
   is not. Rules work in proportion to how literal their trigger words are.

A third, weaker observation: accuracy (0.42) is below the majority-class floor
(13/24 = 0.54 by always answering `none`), so the baseline is not merely
imprecise — it is worse than a constant predictor on this pool.

## What this does *not* support

- **Not an estimate of production accuracy.** The pool was built by mining
  paragraphs that tripped a rule or contained hand-picked trigger words, which
  measures the rules on their own vocabulary. Recall on language the rules do
  not already look for is invisible here.
- **Not a class-balanced benchmark.** 30 of 60 snippets are deliberate
  near-misses; in real filings, event-bearing paragraphs are a small minority.
  Accuracy on this mix is meaningless.
- **Not generalisable.** All four filings are NVIDIA, 2025–2026. Style,
  vocabulary and section layout are one company's.
- **Not quotable as macro-F1.** Five of nine classes have zero support, so the
  macro average runs over a skewed subset.
- **Not a validated label set.** One annotator, no written rubric applied before
  the fact, and the machine `proposed` label was visible while labelling. Seeing
  the rules' own guess anchors the annotator toward it, which can only inflate
  the baseline's score.

## Why it was frozen

Extending the pool as built would have made the bias worse, not better: more
rule-mined snippets means more of the same vocabulary. A set worth extending
needs a different construction:

1. **Rule-blind sampling** — random prose paragraphs, not rule hits, stratified
   by section and filing.
2. **Per-class balance** — a target count per class (~12–15), sampled on
   purpose, including classes that 10-K/10-Q prose does not contain.
3. **8-K earnings releases (Item 2.02 / Exhibit 99.1)** — guidance and
   management-change language lives there, not in periodic reports.
4. **Multiple issuers and years** — 5–8 companies, 2024–2026, with a temporal
   hold-out.
5. **Blind labelling** — `proposed` hidden, plus a written rubric and a second
   annotator (or a re-label after a delay) to estimate the noise floor.

Without (5) in particular, a difference between two predictors smaller than the
annotator's own inconsistency cannot be interpreted.

## Allowed claims

- "On a rule-mined, single-issuer pool of 60 snippets, the baseline never
  predicted `none`, and `acquisition` precision was 0.36."
- "`legal_or_regulatory` is the one class where the rules perform respectably
  (0.60 precision, 0.86 recall) on that pool."

## Disallowed claims

- "The baseline has F1 0.62."
- "Event detection accuracy is 42%."
- "The zero-shot model beats / loses to the rules." — the model has never been
  scored against these labels.

## Status

The 60 snippets stay in `tests/fixtures/events_gold.jsonl` with 24 labels and
are treated as a **debugging set**. `make score` still runs and is useful for
spot-checking rule changes; its numbers should be read as diagnostics. Building
a real benchmark is deferred until it can be built properly (see above).
