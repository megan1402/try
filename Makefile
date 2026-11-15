# SignalForge — developer entry points.
# Run `make` (or `make help`) to list every target.

PYTHON := python3
VENV   := .venv
BIN    := $(VENV)/bin
PY     := $(BIN)/python

.DEFAULT_GOAL := help
.PHONY: help setup test lint format typecheck check run run-sec index review unlabel score compare study collect clean

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

compare: ## Compare the rule baseline with the zero-shot model (needs requirements-ml)
	cd backend && ../$(PY) -m app.models.compare

study: ## Event study over the local filings (TICKER=NVDA CIK=1045810)
	cd backend && ../$(PY) -m app.quant.event_study --ticker $(or $(TICKER),NVDA) --cik $(or $(CIK),1045810)

collect: ## Download filings: make collect TICKERS="NVDA MSFT" [LIMIT=5]
	@test -n "$(TICKERS)" || { echo 'usage: make collect TICKERS="NVDA MSFT" [LIMIT=5]'; exit 1; }
	cd backend && ../$(PY) -m app.ingestion.collect $(TICKERS) $(if $(LIMIT),--limit $(LIMIT),)

clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
