# SignalForge — developer entry points.
# Run `make` (or `make help`) to list every target.

PYTHON := python3
VENV   := .venv
BIN    := $(VENV)/bin
PY     := $(BIN)/python

.DEFAULT_GOAL := help
.PHONY: help setup test lint format typecheck check run run-sec index review score clean

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

score: ## Score the rule baseline against the labelled snippets
	cd backend && ../$(PY) -m app.evaluation.report

clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
