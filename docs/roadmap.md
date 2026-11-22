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
| 14 | Corpus widened to 41 issuers | 516 documents, 118 event dates |
| 15 | Precision measured | **0.055 pooled** — detector is the bottleneck |

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

### 15. Label a sample of the surviving events — **done**
**Produces:** the first real precision number for the rules.
**Done when:** 100–150 events labelled, `make score` reports precision/recall
per class. ✅ 149 labelled.

**Result (2026-11-21): pooled precision 0.055 [0.028, 0.104].** Of 149 detected
events, 8 were real. Acquisition 0.076, legal 0.105, five classes at 0.000.
See `docs/precision-result.md`.

**Consequence:** the negative event-study result is not evidence about the
market — it is evidence that the detector finds almost no real events. Whether
real filing events predict returns remains **untested**.

### 16. Build a rule-blind benchmark — **done**
**Produces:** a labelled set sampled from filing passages rather than rule hits,
so **recall** becomes measurable for the first time.
**Done when:** passages are sampled independently of the detector, labelled
blind, and precision *and* recall are reported per class. ✅ 516 passages
sampled, 38 event-bearing ones filtered for labelling.

### 17. Establish recall — **done**
**Produces:** the missing half of the extraction measurement.
**Done when:** `make recall-events` reports recall on event-bearing passages. ✅

**Result (2026-11-22): recall 0.000 (0 of 2 real events found), and 2 false
positives on 36 event-free passages.** With n=2 the interval is 0.000–0.793, so
this is not a precise estimate — but combined with precision 0.055 the
conclusion is firm: the rule baseline does not detect events in either
direction. See `docs/recall-result.md`.

### 18. Score the zero-shot model against the same labels — **done**
**Produces:** a measured answer to "would a model do better than 5.5% precision
and 0% recall?" ✅

**Result (2026-11-22): precision 0.055 (identical), recall 0.500 (vs 0.000).**
The model is not better, it is differently broken: it fires on 145 of 149
snippets, so its precision is the base rate, but it caught the management change
the rules missed. See `docs/model-result.md`.

**Reading:** the model is a usable *candidate generator*, not a usable detector.
It recognises events phrased in unanticipated ways but cannot tell event prose
from non-event prose.

### 19. Threshold the model — **next**
**Produces:** whether the model's scores carry usable signal.
**Done when:** a threshold sweep shows precision and recall at several cut-offs.
**Why:** the current run takes the argmax with no minimum score, which guarantees
it always fires. If a threshold exists with precision > 0.3 and recall > 0.3, the
model is a viable candidate generator and the verifier prototype is justified.
If no threshold works, the conclusion is firmer. One parameter, measured.

### 20. Inspect the errors and build a taxonomy
**Produces:** the failure modes the next architecture must handle.
**Done when:** every false positive and false negative has a recorded cause.

### 21. Prototype: candidate generator + LLM verifier + temporal validation
**Produces:** an extractor that can plausibly clear the 5.5% floor.
**Done when:** precision and recall are both measured on the labelled sets.
**Spec:** `docs/next-experiment.md`.

**Not doing:** fine-tuning. 149 labels is not enough, and the failure modes are
not yet characterised well enough to know what to train on.

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
