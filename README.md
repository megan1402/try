# SignalForge

**SignalForge** is a financial *event intelligence* system that combines AI
engineering (NLP, embeddings, retrieval, LLM reasoning, evaluation) with
quantitative finance (market data, forward returns, historical analogues).

> Status: **Phases 1–2 complete; Phase 3 in progress — filing text and section
> extraction**.

---

## What SignalForge will eventually become

The finished system is intended to:

1. **Ingest** real-world financial data — historical prices, SEC filings
   (10-K / 10-Q), and later earnings releases, news and macro data.
2. **Extract structured events** from documents — earnings beat/miss, guidance
   raise/cut, acquisitions, regulatory events, management changes, product
   announcements.
3. **Use Hugging Face models** for text classification, zero-shot
   classification, embeddings, sentence similarity and summarisation.
4. **Store** events and document embeddings in PostgreSQL + `pgvector`.
5. **Run event-conditioned RAG**: embed the current event, retrieve historically
   similar events, look up what the market actually did afterwards, compute
   statistics over those outcomes, and hand the evidence to an LLM.
6. **Orchestrate a few meaningful agents** (Fundamental, Quant / Historical
   Analogue, Risk / Critic) with LangGraph.
7. **Produce an evidence-backed market hypothesis** — direction, confidence,
   supporting and contradicting evidence, historical analogue counts, median
   historical returns, risks and citations.
8. **Evaluate rigorously** — event-classification precision/recall/F1, Recall@K
   and MRR, citation accuracy, RAG faithfulness, hallucination rate, latency,
   token/API cost, directional accuracy and abnormal returns / information
   coefficient.
9. **Avoid look-ahead bias** — a query at time *T* must never see data from
   after *T*.
10. **Ship** behind FastAPI plus a small frontend, containerised, deployable to
    AWS.

None of that is built yet. We are deliberately going one phase at a time.

---

## What Phase 1 implements

Phase 1 is the market-data foundation only:

- Download historical daily OHLCV bars for a ticker (`yfinance` prototype).
- Validate input and handle invalid tickers, empty responses and network errors.
- Compute **forward** returns over 1, 5 and 20 trading days.
- A runnable smoke-test example, plus pytest tests for the return maths.
- Quality gates: lint + format (`ruff`), static types (`mypy`), `pytest`, and CI.

Phase 1 deliberately stops there — no filings, models, databases or services.

---

## What Phase 2 implements

Phase 2 adds SEC EDGAR ingestion:

- Resolve a **ticker to a CIK** using EDGAR's `company_tickers.json`.
- List a company's **10-K / 10-Q filings**, newest first, with form type,
  filing date, report date, acceptance timestamp, accession number and
  document URL.
- **Download and store** the raw filing document under `data/raw/sec/`.
- 39 pytest tests driven by saved fixtures — **no network access**.

Two real-world details worth knowing:

- EDGAR requires a `User-Agent` containing a contact address, or it returns
  `403`. See *Credentials for Phase 2* above.
- The **acceptance timestamp** is the precise moment EDGAR accepted a filing.
  It is usually *after the market close*, so it — not `filing_date` — is what
  later phases must use to decide whether information existed at a point in
  time.

Deliberately **not** included yet: Hugging Face models, LangChain / LangGraph,
LLM APIs, vector databases, PostgreSQL, Docker, AWS and frontend frameworks.

---

## What Phase 3 and the event study implement (in progress)

Phase 3 turns filing documents into structured events; the event study is the
first place those events meet prices.

- **Filing index** (`app/ingestion/store.py`) — every listed filing recorded in
  a local SQLite database under `data/index/`, keyed by accession number, with
  its `acceptance_datetime` and, once downloaded, the path of the local copy.
  Re-running the SEC example updates rows instead of duplicating them.
- **HTML to text** (`app/extraction/html.py`) — EDGAR's machine-generated XHTML
  reduced to plain text with one line per block, entities decoded and scripts,
  styles and non-breaking spaces removed.
- **Section splitting** (`app/extraction/sections.py`) — text split into Parts
  and Items (`part_ii_item_1a` is *Risk Factors*), with character offsets so a
  later citation can point at the exact passage.
- **Event detection** (`app/extraction/events.py`) — a **rule-based baseline**:
  regexes for guidance raise/cut, beat/miss, acquisitions, divestitures,
  management changes, restructuring and legal or regulatory language. Every
  event carries a fixed confidence, absolute character offsets and a
  timezone-aware `as_of` (EDGAR's `acceptance_datetime`) — an event without a
  precise instant cannot be used for point-in-time analysis.
- **Event study** (`app/quant/event_study.py`) — attaches forward returns to
  events and summarises them per event type: count, median, mean, hit rate and
  a t-statistic.

The baseline is deliberately dumb. It reads *"an acquisition could harm us"* in
Risk Factors as an acquisition, and says so in its confidence weights. That is
the point: a deterministic, fully explained floor for a model to beat — and the
thing a labelled golden set has to measure.

### Where the return clock starts

The file is accepted *after* the close — typically ~20:00–22:00 UTC — so the
last price that could **not** have reacted to it is that session's close. Each
event is therefore anchored on the **last session whose close precedes the
acceptance timestamp**, and the already-computed forward returns from that close
are reused. A 21:00 acceptance anchors on the same session; a 10:00 one anchors
on the previous session, so the whole of the following day counts as reaction.

Two mistakes this design rules out:

- anchoring on the *next* session, which silently discards the overnight gap
  where most of the reaction happens;
- dating the event by `filing_date` and counting a return that begins *before*
  the news, which is look-ahead bias: trading on information you do not have.

Anchoring at or before the close does not claim you could have traded at that
close; it only fixes the window in which the market digested the news. Events
whose nearest earlier session is more than a week away, or that fall outside the
price history, get no anchor and are reported as `NaN` rather than borrowing a
stale close.

Still to come: labels for the golden set, a measured comparison of the rules and
the zero-shot model below, and then embeddings + retrieval.

### Measuring the baseline

`tests/fixtures/events_gold.jsonl` holds 60 real snippets, one JSON object per
line, each with an empty `label` and a machine `proposed` guess. Label them with
`make review` — one keystroke per snippet, saved after every answer — then
`make score` prints precision / recall / F1 per class plus every disagreement.

`proposed` is never used for scoring. If it were, the rules would be graded
against themselves and the numbers would be meaningless. The value of the set is
precisely where a human disagrees with the rules: on the current pool the
baseline already proposes `acquisition` for *"property, equipment and intangible
assets acquired but not paid for"* and `divestiture` for *"identified for
retirement or disposition"*.

Known gap in the pool: 10-Ks and 10-Qs contain almost no forward-guidance
language, so `guidance_raise`, `guidance_cut`, `management_change` and
`restructuring` currently have no examples. Those live in 8-K earnings releases
(Exhibit 99.1), which the corpus does not download yet.

### Zero-shot model (optional, needs no training)

`app/models/zero_shot.py` turns a natural-language-inference model into an event
classifier: each class is written as a sentence, and the best entailment score
wins. It needs the optional extras and is skipped by CI:

```bash
pip install -r backend/requirements-ml.txt
make compare          # rules vs model, side by side, on the 60 snippets
```

The first run agrees with the rules on **30/60** snippets. That is a disagreement
map, not accuracy — the labels decide who is right. Two things it already shows:

- the model almost never predicts `none` (7 of the 30 no-event snippets), so it
  over-fires; the `none` hypothesis and its calibration need work before its
  scores mean anything;
- both predictors struggle with accounting boilerplate
  ("property and equipment acquired", "identified for retirement"), which is the
  kind of negative a golden set exists to catch.

Two details the extraction code has to fight, both of which the tests cover:

- A **table of contents** looks exactly like a list of headings. It is removed
  because its rows end in page numbers, sit before the first `Part` heading and
  are followed by almost no text, while real sections are followed by
  paragraphs.
- **Cross-references** in prose mention items mid-sentence and must not open a
  section. Requiring `Item N.` at the start of a line plus a title filters
  almost all of them.

---

## Project structure

```
.
├── .github/workflows/ci.yml     # CI: lint, typecheck and test on every push
├── backend/
│   ├── requirements.txt         # runtime dependencies
│   ├── requirements-dev.txt     # dev tooling (pytest, ruff, mypy, stubs)
│   └── app/
│       ├── __init__.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── gold.py          # labelled snippet set and its file format
│       │   ├── report.py        # precision / recall / F1 for a predictor
│       │   └── review.py        # terminal tool for labelling snippets
│       ├── extraction/
│       │   ├── __init__.py
│       │   ├── events.py        # Phase 3: text -> structured events (rules)
│       │   ├── html.py          # Phase 3: filing HTML -> plain text
│       │   └── sections.py      # Phase 3: text -> Part / Item sections
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── market.py        # Phase 1: historical OHLCV data
│       │   ├── sec.py           # Phase 2: SEC EDGAR filings
│       │   └── store.py         # Phase 3: local SQLite filing index
│       ├── models/
│       │   ├── __init__.py
│       │   ├── compare.py       # Phase 3c: rules vs zero-shot model
│       │   └── zero_shot.py     # Phase 3c: NLI model as a classifier
│       └── quant/
│           ├── __init__.py
│           ├── event_study.py   # events + forward returns -> statistics
│           └── returns.py       # forward-return calculations
├── data/                        # local data (contents are git-ignored)
├── tests/
│   ├── fixtures/                # saved EDGAR JSON + HTML responses (offline)
│   ├── test_event_study.py
│   ├── test_events.py
│   ├── test_html.py
│   ├── test_market.py
│   ├── test_returns.py
│   ├── test_sections.py
│   ├── test_sec.py
│   └── test_store.py
├── conftest.py                  # puts backend/ on sys.path for pytest
├── pyproject.toml               # ruff / mypy / pytest configuration
├── Makefile                     # developer entry points
├── .gitignore
└── README.md
```

> **Why `backend/`?** The importable package is `app` and its root is
> `backend/`, so commands are run from inside `backend/`
> (`python -m app.ingestion.market`). This keeps the package root separate from
> repository-level files such as `data/` and `tests/`.

---

## Setup

Requires **Python 3.11+**.

```bash
# Option A — one command does everything
make setup

# Option B — by hand
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows
pip install -r backend/requirements.txt -r backend/requirements-dev.txt
```

---

## Credentials for Phase 2 (SEC EDGAR)

EDGAR rejects requests whose `User-Agent` does not identify a real contact.
Export one before running the SEC example:

```bash
export SEC_USER_AGENT="Your Name your@email.com"
```

`company_tickers.json` also lives on `www.sec.gov`, so the same header is needed
for the ticker lookup. Nothing else in the project needs credentials.

---

## Run the market-data example

`app` must be importable, so run the command from inside `backend/`:

```bash
make run                 # or, by hand:
cd backend
python -m app.ingestion.market
```

This fetches roughly one year of NVDA daily bars, adds the forward returns and
prints the last 10 rows.

---

## Run the SEC EDGAR example

Set a contact address first (see *Credentials* above), then:

```bash
export SEC_USER_AGENT="Your Name your@email.com"

make run-sec             # or, by hand:
cd backend
python -m app.ingestion.sec
```

This resolves `NVDA` to its CIK, lists the recent 10-K / 10-Q filings, then
downloads and saves the most recent one into `data/raw/sec/`. Every listed
filing is also recorded in `data/index/filings.db`; `make index` prints it.

---

## Run the tests

From the repository root, with the virtual environment active:

```bash
pytest          # or: make test
```

The tests build small hand-written DataFrames (Phase 1) or replay saved EDGAR
JSON fixtures (Phase 2). **No test touches the network.**

---

## Development workflow

Every change should pass the same gates that CI runs. Use one command:

```bash
make check      # lint -> format check -> typecheck -> tests
```

| Command | What it does |
|---|---|
| `make setup` | Create `.venv` and install all dependencies |
| `make run` | Run the Phase 1 market-data example |
| `make run-sec` | Run the Phase 2 SEC EDGAR example (also fills the index) |
| `make index` | Show the local filing index |
| `make review` | Label golden-set snippets interactively |
| `make score` | Score the rule baseline against the labelled snippets |
| `make compare` | Rules vs zero-shot model (needs `requirements-ml.txt`) |
| `make test` | Run the test suite |
| `make lint` | `ruff check` + `ruff format --check` (no edits) |
| `make format` | Auto-fix lint issues and reformat in place |
| `make typecheck` | `mypy` over `backend/app` |
| `make check` | All gates, in CI order |
| `make clean` | Remove caches and `__pycache__` |

Rules of thumb while working:

1. **Write the test first** for anything numeric — return maths especially.
2. Run `make format` before committing; `make lint` catches whatever is left.
3. If `mypy` complains about pandas, prefer fixing the annotation over adding
   `# type: ignore`. `pandas-stubs` is installed so the checks are real.
4. Never let a test hit the network. Provider calls are isolated in
   `app/ingestion` behind a function that a test can monkeypatch or skip.

CI (`.github/workflows/ci.yml`) runs the same four steps on Python 3.11 for
every push and pull request, so a green `make check` locally means a green
build remotely.

---

## Understanding forward returns

For a horizon of `h` trading days:

$$\text{forward\_return}_h[t] = \frac{\text{Close}[t+h]}{\text{Close}[t]} - 1$$

In pandas that is `close.shift(-h) / close - 1`. `shift(-h)` moves the price
from `h` rows *later* up into the current row, so each row is paired with a
**future** price. The final `h` rows are therefore `NaN` — their future does not
exist yet — and leaving them as `NaN` is what keeps the calculation honest.

Do **not** confuse this with `pct_change()`, which looks *backwards*
(`Close[t] / Close[t-1] - 1`). Using it here would leak future information into
the past and silently invalidate every later evaluation.

---

## Next up (Phase 3b — label, then measure)

The harness is in place; the snippets are not labelled yet. Run `make review`,
work through the 60 snippets, then `make score` for the baseline's first
precision and recall numbers. After that, add 8-K earnings releases to the
corpus so the guidance classes have examples — and only then start on a Hugging
Face model, so that "the model is better" is a measurement rather than an
impression.

Using what exists today, from inside `backend/`:

```python
from datetime import UTC, datetime

from app.extraction.events import events_to_frame, extract_events
from app.extraction.html import html_to_text
from app.extraction.sections import find_item, parse_sections
from app.ingestion.market import get_market_data
from app.quant.event_study import attach_forward_returns, summarise_events
from app.quant.returns import add_forward_returns

text = html_to_text(open("path/to/filing.htm", encoding="utf-8").read())
sections = parse_sections(text)

mda = find_item(sections, "2", part="I")  # Item 2, Part I
events = extract_events(sections, as_of=datetime(2026, 7, 26, 20, 4, tzinfo=UTC))
frame = events_to_frame(events, ticker="NVDA")

prices = add_forward_returns(get_market_data("NVDA", period="1y"))
print(summarise_events(attach_forward_returns(frame, prices)))
```
