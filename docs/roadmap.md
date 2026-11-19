# Roadmap

Every step from here to the finished system, in dependency order. Each entry
says what it produces and how you know it is done.

Status: Phases 1–3 are built. The event study runs with statistical controls and
finds no significant effect, which is a real result — the detector was too noisy
to test. Precision work has since cut events by 91%.

---

## Done

| # | Step | Output |
|---|---|---|
| 1 | Market data + forward returns | `app/ingestion/market.py`, `app/quant/returns.py` |
| 2 | SEC EDGAR retrieval | `app/ingestion/sec.py` |
| 3 | Filing index (SQLite) | `app/ingestion/store.py` |
| 4 | HTML → text, Part/Item sections | `app/extraction/html.py`, `sections.py` |
| 5 | Rule-based events | `app/extraction/events.py` |
| 6 | Golden-set harness + 24 labels | `app/evaluation/`, `docs/eval-notes.md` |
| 7 | Zero-shot model comparison | `app/models/` |
| 8 | Event study + 3 controls | `app/quant/event_study.py`, `controls.py` |
| 9 | Bulk collector | `app/ingestion/collect.py` |
| 10 | Detector precision filters | `app/extraction/precision.py` |
| 11 | 8-K exhibit collection | `sec.list_exhibits`, `collect --exhibits` |
| 12 | 8-K press releases collected | 60 filings; **no guidance found** |
| 13 | Study re-run on filtered events | no significant effect |

---

## Next: finish the corpus

### 12. Collect 8-K earnings releases — **done, with a negative finding**
**Produces:** press releases (EX-99.1) for 10 issuers.
**Done when:** `make collect TICKERS="..." FORMS="8-K" LIMIT=6` downloads
exhibits, and `make index` shows them. ✅ 60 collected, 0 failures.

**Finding (2026-11-19):** these 10 mega-caps do **not** give numeric guidance in
their press releases. Apple's release reports results, a dividend and a
forward-looking-statements disclaimer; Tesla's only "forecast" is a
supply-chain risk ("we will need to accurately forecast, purchase and warehouse
components"). After filtering that, `guidance_raise` and `guidance_cut` have
**zero** events across 336 documents.

**Decision:** drop `guidance_*` from the study rather than manufacture events.
Guidance lives in earnings-call transcripts and investor decks, not SEC filings;
that is a different pipeline with its own licensing questions, and it is not on
the critical path. The classes with data — acquisition, legal, restructuring,
divestiture — are enough to answer the project's question.

### 13. Re-run the event study on the cleaner corpus — **done**
**Produces:** per-class abnormal returns with the precision filters applied.
**Done when:** `make study-all` runs on the filtered events. ✅
**Result:** no class survives benchmark adjustment, clustering and the null
comparison. See `docs/event-study-methodology.md`.

### 14. Widen the corpus — **next**
**Produces:** 30–50 issuers, 3 years, so each class has 100+ independent dates.
**Done when:** `make study-all` reports ≥100 distinct dates per major class.
**Why:** 52 dates cannot support a per-class claim, and ten mega-caps file
within days of each other. This is the binding constraint, not the missing
guidance class.

---

## Then: measure the detector

### 15. Label a sample of the surviving events
**Produces:** the first real precision number for the rules.
**Done when:** 100–150 events labelled, `make score` reports precision/recall
per class.
**Why now:** the set is small (220 events), so this is tractable — unlike the
earlier 60-snippet pool, which was rule-mined and biased.

### 16. Build a proper benchmark set
**Produces:** a rule-blind, class-balanced, multi-issuer labelled set.
**Done when:** sampling is random over prose (not rule hits), each class has
12–15 examples, and labelling is blind to `proposed`.
**Spec:** `docs/eval-notes.md`, "Why it was frozen".

### 17. Beat the baseline with a model
**Produces:** a measured comparison, not an impression.
**Done when:** a fine-tuned classifier beats the rule baseline on the benchmark
set, with the test split untouched during training.

---

## Then: the actual product

### 18. Event embeddings
**Produces:** a vector per event, so "similar events" becomes a query.
**Done when:** `sentence-transformers` embeddings are stored and nearest
neighbours look sensible by eye.

### 19. PostgreSQL + pgvector
**Produces:** the store the README promised.
**Done when:** events and embeddings live in Postgres, and retrieval is a SQL
query. **This is the first point Docker earns its place** — and only for the
database.

### 20. Event-conditioned retrieval
**Produces:** given a new event, the K most similar historical events and what
followed each.
**Done when:** Recall@K and MRR are measured against the benchmark set.

### 21. LLM reasoning with citations
**Produces:** the written hypothesis — direction, confidence, supporting and
contradicting evidence, analogue counts, median returns, risks.
**Done when:** every claim in the output cites a filing passage, and citation
accuracy is measured.

### 22. Agent orchestration
**Produces:** Fundamental / Quant-analogue / Risk-critic agents with LangGraph.
**Done when:** the critic can veto a hypothesis, and the disagreement is logged.

### 23. Evaluation of the whole system
**Produces:** faithfulness, hallucination rate, latency, cost, directional
accuracy, information coefficient.
**Done when:** the numbers are reproducible from a single command.

### 24. FastAPI + minimal UI
**Produces:** a query endpoint and one page showing the hypothesis with its
evidence.
**Done when:** a ticker + event returns a cited answer in a browser.

### 25. Deployment
**Produces:** containerised, deployable to AWS.
**Done when:** it runs somewhere other than your laptop.

---

## Deliberately not planned

- **Jenkins.** GitHub Actions already runs the same four gates.
- **Docker before step 19.** A venv is faster for pure-Python work.
- **A UI before step 21.** There is nothing worth showing until the numbers
  exist.
- **More statistics.** The controls are sound; more tests will not create signal.

---

## The honest summary

Steps 12–14 decide whether this project has a finding. Everything after that is
engineering on top of whatever the answer turns out to be. If the answer is "no
measurable effect", that is still a defensible result — and steps 15–17 become
the interesting part, because then the question is whether a better detector
finds something the rules missed.
