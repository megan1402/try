# SignalForge

Financial **event intelligence**: extract corporate events from SEC filings, then
measure what the market did after similar events historically.

> **Status: extraction does not work yet.** Three approaches have been measured
> and none is precise enough to support a return analysis. See
> [`docs/results.md`](docs/results.md) for the numbers and
> [`docs/roadmap.md`](docs/roadmap.md) for what is next.

## What it does

1. **Ingest** — daily OHLCV bars, SEC filings (10-K, 10-Q, 8-K with exhibits).
2. **Extract** — filing HTML → text → sections → structured events.
3. **Measure** — forward returns after each event, benchmark-adjusted, clustered
   by date, compared against a null distribution.
4. **Evaluate** — precision and recall against hand-labelled passages.

Steps 1–2 work. Step 3 runs but has nothing meaningful to measure. Step 4 is
what showed why.

## Quick start

Requires **Python 3.11+**.

```bash
make setup          # create .venv and install dependencies
make check          # lint, format, typecheck, test
```

Fetch filings (needs a contact address for EDGAR):

```bash
export SEC_USER_AGENT="Your Name your@email.com"
make collect TICKERS="NVDA MSFT AAPL" LIMIT=8
make index
```

Run the event study:

```bash
make study-all      # benchmark-adjusted, clustered, null-compared
```

## Commands

| Command | What it does |
|---|---|
| `make setup` | Create `.venv` and install dependencies |
| `make check` | All gates: lint → format → typecheck → test |
| `make collect` | Download filings: `TICKERS="NVDA MSFT" [FORMS="8-K"] [LIMIT=5]` |
| `make index` | Show the local filing index |
| `make study-all` | Cross-issuer event study with controls |
| `make sample-events` | Build a stratified precision sample |
| `make review-precision` | Label it |
| `make precision` | Per-class precision with Wilson intervals |
| `make sample-passages` | Build a rule-blind passage sample |
| `make review-events` | Label event-bearing passages |
| `make recall-events` | Recall on event-bearing passages |
| `make annotate` | LLM annotation (`PROVIDER=deepseek`) |
| `make annotate-dry` | Exercise the pipeline without API calls |

## Layout

```
backend/app/
├── ingestion/     market data, EDGAR, filing index, bulk collector
├── extraction/    HTML → text → sections → events, precision filters
├── quant/         forward returns, event study, statistical controls
├── evaluation/    sampling, labelling, precision and recall
├── models/        zero-shot NLI classifier, threshold sweep
└── annotation/    LLM annotation with schema validation
```

The importable package is `app` and its root is `backend/`, so commands run from
inside `backend/` (`python -m app.ingestion.market`). The `Makefile` handles this.

## Documentation

| File | Contents |
|---|---|
| [`docs/methodology.md`](docs/methodology.md) | How every number is produced, and what each safeguard protects against |
| [`docs/results.md`](docs/results.md) | Every measurement, what it showed, what it does not show |
| [`docs/roadmap.md`](docs/roadmap.md) | What is done, what is next, what is deliberately not planned |

## Optional extras

Not installed by `make setup` or CI — the test suite runs offline without them.

```bash
pip install -r backend/requirements-ml.txt    # transformers, torch
pip install -r backend/requirements-llm.txt   # anthropic, pydantic
```

## Credentials

| Variable | Needed for |
|---|---|
| `SEC_USER_AGENT` | Any EDGAR request — must contain a contact address |
| `DEEPSEEK_API_KEY` | `make annotate` |
| `ANTHROPIC_API_KEY` | `make annotate PROVIDER=anthropic` |

Put them in `.env` (git-ignored) and load with `set -a; source .env; set +a`.

## Development

Every change must pass `make check`, which runs the same four gates as CI:

```bash
make format     # auto-fix lint and reformat
make check      # lint → format check → typecheck → test
```

Rules of thumb:

1. **Write the test first** for anything numeric — return maths, interval maths,
   sampling allocation.
2. **Never let a test hit the network.** Provider calls are injected.
3. **Prefer fixing an annotation over adding `# type: ignore`.**
4. **Documentation is a snapshot, not a log.** Update the existing file; git
   already stores history.
