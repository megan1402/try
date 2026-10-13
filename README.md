# SignalForge

**SignalForge** is a financial *event intelligence* system that combines AI
engineering (NLP, embeddings, retrieval, LLM reasoning, evaluation) with
quantitative finance (market data, forward returns, historical analogues).

> Status: **Phases 1–2 — Market data foundation + SEC EDGAR ingestion**.

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

## Project structure

```
.
├── .github/workflows/ci.yml     # CI: lint, typecheck and test on every push
├── backend/
│   ├── requirements.txt         # runtime dependencies
│   ├── requirements-dev.txt     # dev tooling (pytest, ruff, mypy, stubs)
│   └── app/
│       ├── __init__.py
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── market.py        # Phase 1: historical OHLCV data
│       │   └── sec.py           # Phase 2: SEC EDGAR filings
│       └── quant/
│           ├── __init__.py
│           └── returns.py       # forward-return calculations
├── data/                        # local data (contents are git-ignored)
├── tests/
│   ├── fixtures/                # saved EDGAR JSON responses (offline tests)
│   ├── test_market.py
│   ├── test_returns.py
│   └── test_sec.py
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
downloads and saves the most recent one into `data/raw/sec/`.

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
| `make run-sec` | Run the Phase 2 SEC EDGAR example |
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

## Next up (Phase 3 — not started)

Event extraction: turn raw filing text into structured financial events
(earnings beat/miss, guidance changes, acquisitions, management changes, ...).
This is where Hugging Face models and evaluation first appear.
