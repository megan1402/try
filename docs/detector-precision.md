# Detector precision

The first rule baseline matched bare vocabulary. Across 276 filings it produced
**2 439 events**, of which almost none were events: 751 "acquisitions" that were
mostly the word `acquisition` in an accounting note.

This document records what was wrong, what was changed, and what the change
costs.

## What was wrong

Two independent problems, needing separate fixes.

### Where the match was

| Section | Share of false positives | Why |
|---|---|---|
| Item 1 — Financial Statements | largest | "property and equipment acquired" is capex; "identified for disposition" is how you retire an asset |
| Item 1A — Risk Factors | second | hypothetical by construction: "an acquisition could harm us" |
| Item 2 — MD&A | mixed | reports real events *and* discusses accounting policy |
| Item 6 — Exhibits | small | a list of document titles |

### What the match looked like

A bare noun is not an event. `acquisition` is a topic; `We completed the
acquisition of Acme` is an event. The original rules could not tell them apart.

## What changed

`app/extraction/precision.py` applies four filters to every candidate match:

1. **Section suppression** — Item 1, Item 1A, Item 5 and Item 6 produce no
   events, except `legal_or_regulatory`, which is reported in the notes and in
   Part II Item 1. Risk Factors is stricter still: *nothing* survives there.
2. **Hypothetical language** — sentences containing `could`, `may`, `might`,
   `if we`, `risk that`, `potential`, `from time to time` are rejected. This is
   what removes risk-factor prose even when the section filter is bypassed.
3. **Accounting vocabulary** — phrases such as `property, plant and equipment`,
   `purchase obligations`, `net of acquired cash`, `repurchase program`,
   `discontinued operations` mark a sentence as accounting, not corporate
   action.
4. **Subject and object** — each structured class must match a phrase that names
   an actor and an object: `completed the acquisition of X`, `sold the EUC
   business to KKR`, `the CFO resigned`, `implemented a restructuring plan`,
   `entered into a settlement`. A bare `acquisition`, `retired` or `settlement`
   fails.

Every event now also carries the **sentence** it came from, so a decision can be
audited without re-parsing the filing. The matched span alone was often a single
word and told a reader nothing.

## Result

| event type | before | after | cut |
|---|---|---|---|
| acquisition | 751 | 168 | 78% |
| legal_or_regulatory | 729 | 42 | 94% |
| restructuring | 342 | 7 | 98% |
| divestiture | 240 | 3 | 99% |
| management_change | 222 | 0 | 100% |
| earnings_miss | 58 | 0 | 100% |
| earnings_beat | 52 | 0 | 100% |
| guidance_raise | 25 | 0 | 100% |
| guidance_cut | 20 | 0 | 100% |
| **total** | **2 439** | **220** | **91%** |

## What this does *not* prove

**The reduction is not a precision measurement.** There are no labels for these
276 filings, so "91% fewer events" says nothing about how many of the 220 are
correct. It says the obvious false positives are gone.

**Zero is not always right.** `management_change`, `earnings_*` and `guidance_*`
now produce nothing. For guidance that is expected — 10-K and 10-Q prose does
not contain forward guidance; it lives in 8-K earnings releases, which the
corpus does not download yet. For `management_change` it is a genuine loss: the
stricter pattern requires a named role, and 10-Q prose often says only "our
Chief Financial Officer resigned" in a section that is now suppressed.

**Recall was traded away.** Every filter is a precision/recall trade. A real
acquisition described as "the transaction closed" no longer matches. The
baseline is now conservative rather than noisy, which is the right direction for
an event study but is still a choice, not a free win.

**The remaining 168 acquisitions are unaudited.** Sampling them shows a mix:
`On July 1, 2024, we sold the EUC business to KKR & Co.` is real;
`Payments for acquisition of property, plant and equipment` should have been
caught by the accounting filter and was not, because the phrase is split across
a table row.

## Next

1. **8-K earnings releases** (Item 2.02 / Exhibit 99.1) — the only place
   guidance language appears, and the classes most likely to move prices.
2. **Re-run `make study-all`** with the cleaner events. If a signal appears now
   that did not before, that is evidence the noise was masking it. If nothing
   appears, the negative result is much stronger than it was.
3. **Label a sample of the 220** to turn "fewer events" into an actual precision
   number. This is the evaluation work deferred in `docs/eval-notes.md`, and it
   is now cheap: 220 events is a tractable set.
