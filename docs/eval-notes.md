# Evaluation notes — golden set v0 (frozen)

Written 2026-11-16, **corrected** the same day. This file exists so that the
numbers produced by `make score` are never quoted more strongly than they
deserve. It records what the 60 labelled snippets support, what they do not, and
why the set was frozen rather than extended.

## Correction notice

The first version of this file described a **24-label** snapshot and claimed the
rules *never abstain*. Both were out of date within the same commit: the
remaining 36 snippets were labelled at the same time, taking the set to 60
labels, and the notes were not recomputed against them.

Every number below was re-derived from the committed fixture with `make score`.
One earlier finding did not survive — see *What changed* below. The freeze
decision is unchanged.

## Decision (2026-11-16)

**Freeze this set and stop labelling. Improve evaluation later.** The pool is
too biased to serve as a benchmark, and extending it as constructed would deepen
the bias rather than fix it. Development moves on to return-based evidence
(`make study`, `make study-all`), which needs no labels: it needs *filings*, not
annotations.

Evaluation is revisited only when it can be built properly — rule-blind
sampling, per-class balance, 8-K earnings releases, several issuers, and blind
labelling with an inter-annotator check. The construction notes below are the
specification for that work.

## What was measured

All 60 snippets are labelled: **47 `none`, 8 `legal_or_regulatory`, 5
`acquisition`**. Seven of nine classes still have no support at all
(`earnings_beat`, `earnings_miss`, `guidance_raise`, `guidance_cut`,
`divestiture`, `management_change`, `restructuring`).

| label | support | predicted | tp | fp | fn | precision | recall | f1 |
|---|---|---|---|---|---|---|---|---|
| acquisition | 5 | 17 | 5 | 12 | 0 | 0.29 | 1.00 | 0.45 |
| divestiture | 0 | 1 | 0 | 1 | 0 | 0.00 | — | — |
| earnings_miss | 0 | 2 | 0 | 2 | 0 | 0.00 | — | — |
| legal_or_regulatory | 8 | 10 | 6 | 4 | 2 | 0.60 | 0.75 | 0.67 |
| none | 47 | 30 | 29 | 1 | 18 | 0.97 | 0.62 | 0.75 |

Macro precision 0.62, macro recall 0.79, macro F1 0.62, **accuracy 0.67**. There
are 20 disagreements in total.

## What this *does* support

1. **The baseline is still worse than a constant predictor.** Always answering
   `none` scores 47/60 = **0.78**; the rules score **0.67**. So the detector is
   not merely imprecise — on this pool it is beaten by a rule that never reads
   the text. This is the most important number here, and the extra labels only
   made it clearer.
2. **`acquisition` is the weak class and `legal_or_regulatory` the strong one.**
   Acquisition precision is 0.29 — 12 of 17 predictions are false positives. The
   rules catch every true acquisition and bury them under accounting boilerplate
   ("property and equipment acquired"). Legal or regulatory language
   ("subpoena", "investigation", "settlement") is far more literal, and scores
   0.60 precision at 0.75 recall.
3. **The rules *do* abstain — they are not indiscriminate.** 29 of the 47
   negatives are correctly left alone, making `none` the *most* precise class
   (0.97). The 18 remaining false positives sit on genuinely ambiguous
   vocabulary, not everywhere.
4. **The failure is concentrated, not uniform.** 20 disagreements over 60
   snippets, and 12 of them are `acquisition`. One bad class accounts for most of
   the error, which makes the fix measurable: tighten that class and the macro
   numbers should move.

## What this does *not* support

- **Not an estimate of production accuracy.** The pool was built by mining
  paragraphs that tripped a rule or contained hand-picked trigger words, which
  measures the rules on their own vocabulary. Recall on language the rules do
  not already look for is invisible here.
- **Not a class-balanced benchmark.** 30 of 60 snippets are deliberate
  near-misses; in real filings, event-bearing paragraphs are a small minority.
  At a realistic 2% event rate the same precision gives a very different
  false-positive load.
- **Not generalisable.** All four filings are NVIDIA, 2025–2026. Style,
  vocabulary and section layout are one company's.
- **Not quotable as macro-F1.** Seven of nine classes have zero support, so the
  macro average runs over a skewed subset.
- **Not a validated label set.** One annotator, no written rubric applied before
  the fact, and the machine `proposed` label was visible while labelling. Seeing
  the rules' own guess anchors the annotator toward it, which can only inflate
  the baseline's score. This matters most for `none`: 47 of 60 labels are the
  abstention decision, and it is the decision most exposed to anchoring.

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

## What changed from the first version

| claim | 24-label snapshot | 60-label set |
|---|---|---|
| rules ever abstain? | no (0 of 24) | **yes** (30 of 60) |
| accuracy | 0.42 | 0.67 |
| majority-class floor | 0.54 | 0.78 |
| `acquisition` precision | 0.36 | 0.29 |
| `legal_or_regulatory` | P 0.60 / R 0.86 | P 0.60 / R 0.75 |

The "never abstains" claim was an artefact of the snapshot: the first 24 labels
happened to be snippets where a rule fired. It was reported as a *design gap* —
"any detector that cannot say nothing will have low precision" — and that
diagnosis is now **refuted**. The rules abstain on half the pool. The real gap is
that one event class (`acquisition`) is systematically over-fired, which is a
tuning problem, not an architectural one.

The accuracy conclusion survives in the same direction: the baseline sits below
the majority-class floor in both snapshots. That is the finding worth keeping.

## Allowed claims

- "On a rule-mined, single-issuer pool of 60 snippets, the baseline scores 0.67
  accuracy against a 0.78 all-`none` floor, and `acquisition` precision is 0.29."
- "`legal_or_regulatory` is the one event class where the rules perform
  respectably (0.60 precision, 0.75 recall) on that pool."
- "The rules abstain on roughly half the pool, so the failure is a
  false-positive problem in one class rather than indiscriminate firing."

## Disallowed claims

- "The baseline has F1 0.62." — macro F1 over classes that mostly have no
  support.
- "Event detection accuracy is 67%." — on a pool half composed of deliberate
  near-misses.
- "The zero-shot model beats / loses to the rules." — the model has never been
  scored against these labels.

## Status

The 60 snippets stay in `tests/fixtures/events_gold.jsonl`, fully labelled, and
are treated as a **debugging set**. `make score` still runs and is useful for
spot-checking rule changes; its numbers should be read as diagnostics. Building
a real benchmark is deferred until it can be built properly (see above).
