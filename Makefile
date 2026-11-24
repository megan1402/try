# SignalForge — developer entry points.
# Run `make` (or `make help`) to list every target.

PYTHON := python3
VENV   := .venv
BIN    := $(VENV)/bin
PY     := $(BIN)/python

.DEFAULT_GOAL := help
.PHONY: help setup test lint format typecheck check run run-sec index review unlabel score sample-events sample-holdout review-holdout split-holdout review-development review-test review-precision precision relabel sample-passages review-passages recall filter-passages review-events recall-events compare score-model threshold annotate-dry annotate annotate-holdout study study-all collect clean

help: ## List available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup: ## Create the virtual environment and install all dependencies
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r backend/requirements.txt -r backend/requirements-dev.txt

test: ## Run the test suite
	$(PY) -m pytest

lint: ## Check style and formatting without modifying files
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format: ## Auto-fix lint issues and reformat the code
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

typecheck: ## Run static type checking
	$(BIN)/mypy

check: lint typecheck test ## Run every gate CI runs, in the same order

run: ## Run the Phase 1 NVDA market-data example
	cd backend && ../$(PY) -m app.ingestion.market

run-sec: ## Run the Phase 2 SEC EDGAR example
	cd backend && ../$(PY) -m app.ingestion.sec

index: ## Show the local filing index built by the SEC example
	cd backend && ../$(PY) -m app.ingestion.store

review: ## Label golden-set snippets interactively
	cd backend && ../$(PY) -m app.evaluation.review

unlabel: ## Clear one label: make unlabel ID=k007
	@test -n "$(ID)" || { echo "usage: make unlabel ID=k007"; exit 1; }
	cd backend && ../$(PY) -m app.evaluation.review --clear $(ID)

score: ## Score the rule baseline against the labelled snippets
	cd backend && ../$(PY) -m app.evaluation.report

sample-events: ## Build the stratified precision sample from detected events
	cd backend && ../$(PY) -m app.evaluation.sample_events

sample-holdout: ## Second batch, excluding already-labelled events
	cd backend && ../$(PY) -m app.evaluation.sample_events \
		--output ../tests/fixtures/events_holdout.jsonl \
		--seed 20260122 \
		--exclude ../tests/fixtures/events_precision.jsonl

review-holdout: ## Label the held-out batch
	cd backend && ../$(PY) -m app.evaluation.review ../tests/fixtures/events_holdout.jsonl

split-holdout: ## Split the holdout into development and test partitions, grouped by ticker
	cd backend && ../$(PY) -m app.evaluation.split_holdout

review-development: ## Label the development partition
	cd backend && ../$(PY) -m app.evaluation.review ../tests/fixtures/holdout_development.jsonl

review-test: ## Label the test partition (only after the rubric is frozen)
	cd backend && ../$(PY) -m app.evaluation.review ../tests/fixtures/holdout_test.jsonl

review-precision: ## Label the precision sample
	cd backend && ../$(PY) -m app.evaluation.review ../tests/fixtures/events_precision.jsonl

precision: ## Per-class precision with Wilson confidence intervals
	cd backend && ../$(PY) -m app.evaluation.precision_report

relabel: ## Clear labels so snippets return to the queue: make relabel IDS="ev0009 ev0011"
	@test -n "$(IDS)" || { echo 'usage: make relabel IDS="ev0009 ev0011"'; exit 1; }
	cd backend && ../$(PY) -m app.evaluation.relabel $(IDS)

sample-passages: ## Build a rule-blind passage sample for recall
	cd backend && ../$(PY) -m app.evaluation.sample_passages

review-passages: ## Label the passage sample
	cd backend && ../$(PY) -m app.evaluation.review ../tests/fixtures/passages_recall.jsonl

recall: ## Recall and confusion matrix from labelled passages
	cd backend && ../$(PY) -m app.evaluation.recall_report

filter-passages: ## Narrow the passage sample to event-bearing passages
	cd backend && ../$(PY) -m app.evaluation.filter_passages

review-events: ## Label the event-bearing passage sample
	cd backend && ../$(PY) -m app.evaluation.review ../tests/fixtures/passages_events.jsonl

recall-events: ## Recall on event-bearing passages
	cd backend && ../$(PY) -m app.evaluation.recall_report ../tests/fixtures/passages_events.jsonl

compare: ## Compare the rule baseline with the zero-shot model (needs requirements-ml)
	cd backend && ../$(PY) -m app.models.compare

score-model: ## Score the zero-shot model on the labelled sets (needs requirements-ml)
	cd backend && ../$(PY) -m app.models.score

threshold: ## Sweep the model's acceptance threshold (needs requirements-ml)
	cd backend && ../$(PY) -m app.models.threshold

annotate-dry: ## Exercise the LLM annotation pipeline without API calls
	cd backend && ../$(PY) -m app.annotation.run_phase1 --dry-run

annotate: ## Annotate the human-labelled passages with an LLM (PROVIDER=deepseek)
	cd backend && ../$(PY) -m app.annotation.run_phase1 --provider $(or $(PROVIDER),deepseek)

annotate-holdout: ## Annotate the labelled development partition and compare
	cd backend && ../$(PY) -m app.annotation.run_phase1 \
		--provider $(or $(PROVIDER),deepseek) \
		--input ../tests/fixtures/holdout_development.jsonl \
		--output ../tests/fixtures/llm_holdout.jsonl

study: ## Event study over the local filings (TICKER=NVDA CIK=1045810)
	cd backend && ../$(PY) -m app.quant.event_study --ticker $(or $(TICKER),NVDA) --cik $(or $(CIK),1045810)

study-all: ## Cross-issuer event study with controls (BENCHMARK=SPY SAMPLES=1000)
	cd backend && ../$(PY) -m app.quant.corpus_study --benchmark $(or $(BENCHMARK),SPY) --samples $(or $(SAMPLES),1000)

collect: ## Download filings: make collect TICKERS="NVDA MSFT" [FORMS="8-K"] [LIMIT=5]
	@test -n "$(TICKERS)" || { echo 'usage: make collect TICKERS="NVDA MSFT" [FORMS="8-K"] [LIMIT=5]'; exit 1; }
	cd backend && ../$(PY) -m app.ingestion.collect $(TICKERS) \
		$(if $(FORMS),--forms $(FORMS),) \
		$(if $(LIMIT),--limit $(LIMIT),)

clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
