# SignalForge

**SignalForge** is a financial *event intelligence* system that combines AI
engineering (NLP, embeddings, retrieval, LLM reasoning, evaluation) with
quantitative finance (market data, forward returns, historical analogues).

> Status: **Phase 1 — Market Data Foundation**.

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

Deliberately **not** included yet: SEC ingestion, Hugging Face models,
LangChain / LangGraph, LLM APIs, vector databases, PostgreSQL, Docker, AWS and
frontend frameworks.

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
│       │   └── market.py        # retrieve historical OHLCV data
│       └── quant/
│           ├── __init__.py
│           └── returns.py       # forward-return calculations
├── data/                        # local data (contents are git-ignored)
├── tests/
│   ├── test_market.py
│   └── test_returns.py
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
Option A — one command does everything
make setup

# Option B — by hand
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows
pip install -r backend/requirements.txt -r backend/requirements-dev# macOS / Linux
# .venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r backend/requirements.txt
```

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

## Run the tests

From the repository root, with the virtual environment active:

```bash
pytest          # or: make test
```

The tests build small hand-written DataFrames and never touch the network.

---

## Development workflow

Every change should pass the same gates that CI runs. Use one command:

```bash
make check      # lint -> format check -> typecheck -> tests
```

| Command | What it does |
|---|---|
| `make setup` | Create `.venv` and install all dependencies |
| `make run` | Run the NVDA market-data example |
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

## Next up (Phase 2 — not started)

SEC EDGAR ingestion: download 10-K / 10-Q filings and store the raw documents,
ready for event extraction in a later phase.
