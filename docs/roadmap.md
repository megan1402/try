# Roadmap

What is done, what is next, and what is deliberately not planned.

## Done

| # | Step | Output |
|---|---|---|
| 1 | Market data + forward returns | `app/ingestion/market.py`, `app/quant/returns.py` |
| 2 | SEC EDGAR retrieval | `app/ingestion/sec.py` |
| 3 | Filing index (SQLite) | `app/ingestion/store.py` |
| 4 | HTML → text, Part/Item sections | `app/extraction/html.py`, `sections.py` |
| 5 | Rule-based events | `app/extraction/events.py` |
| 6 | Precision filters | `app/extraction/precision.py` |
| 7 | Bulk collector, 8-K exhibits | `app/ingestion/collect.py` |
| 8 | Event study + 3 controls | `app/quant/event_study.py`, `controls.py` |
| 9 | Precision + recall measurement | `app/evaluation/` |
| 10 | Zero-shot model + threshold sweep | `app/models/` |
| 11 | LLM annotation pipeline | `app/annotation/` |

Results: [`results.md`](results.md). Method: [`methodology.md`](methodology.md).

## Next

### 1. Label the holdout batch
**Produces:** the only clean evaluation set.
**Done when:** `make review-holdout` completes and `events_holdout.jsonl` has 150
labels.
**Why:** every number so far is on a development set the rubric was tuned on.

### 2. Re-run the LLM against the holdout
**Produces:** a defensible precision and recall figure.
**Done when:** `make annotate` runs against the holdout and the result is
recorded in `results.md`.
**Decision point:** precision ≥ 0.4 justifies a verifier; below 0.2 means the
LLM is not the answer either.

### 3. Build the verifier
**Produces:** an extractor that can plausibly clear 0.4 precision.
**Done when:** precision and recall are both measured on the holdout.
**Shape:** high-recall candidate generator (the LLM) → verifier that rejects
boilerplate, historical references and unclosed deals → temporal validation.

### 4. Widen the corpus
**Produces:** 30–50 issuers, 3 years, so each class has 100+ independent dates.
**Done when:** `make study-all` reports ≥100 distinct dates per major class.

### 5. Re-run the event study
**Produces:** the first interpretable return result.
**Done when:** the study runs on events with precision high enough to mean
something.

## Deliberately not planned

- **Fine-tuning.** 187 labels is not enough to train on, and the failure modes
  are not yet characterised well enough to know what to train on.
- **Jenkins.** GitHub Actions runs the same four gates.
- **Docker before a database exists.** A venv is faster for pure-Python work.
- **A UI before the numbers exist.** Nothing worth showing yet.
- **More statistics on the event study.** The controls are sound; the input is
  the problem.
