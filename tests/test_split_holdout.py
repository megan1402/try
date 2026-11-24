"""Tests for :mod:`app.evaluation.split_holdout`.

The property that matters is grouping: no ticker may appear in both partitions.
A random split would leak vocabulary and section structure across the boundary.
"""

from __future__ import annotations

import pytest

from app.evaluation.gold import GoldSnippet
from app.evaluation.split_holdout import split_by_ticker


def snippet(snippet_id: str, ticker: str | None) -> GoldSnippet:
    return GoldSnippet(id=snippet_id, text="text", ticker=ticker)


def make_set(tickers: list[str], per_ticker: int = 3) -> list[GoldSnippet]:
    return [
        snippet(f"{ticker}-{index}", ticker)
        for ticker in tickers
        for index in range(per_ticker)
    ]


# --- Grouping --------------------------------------------------------------


def test_no_ticker_appears_in_both_partitions():
    snippets = make_set([f"T{i}" for i in range(10)])

    split = split_by_ticker(snippets, test_fraction=0.4)

    development = {s.ticker for s in split.development}
    test = {s.ticker for s in split.test}
    assert not development & test


def test_every_snippet_lands_in_exactly_one_partition():
    snippets = make_set([f"T{i}" for i in range(10)])

    split = split_by_ticker(snippets, test_fraction=0.4)

    assert len(split.development) + len(split.test) == len(snippets)
    ids = {s.id for s in split.development} | {s.id for s in split.test}
    assert len(ids) == len(snippets)


def test_all_snippets_of_a_ticker_stay_together():
    snippets = make_set(["AAA", "BBB", "CCC", "DDD"], per_ticker=5)

    split = split_by_ticker(snippets, test_fraction=0.5)

    for ticker in split.test_tickers:
        assert all(s.ticker == ticker for s in split.test if s.ticker == ticker)
        assert sum(1 for s in split.test if s.ticker == ticker) == 5


# --- Proportions -----------------------------------------------------------


def test_test_fraction_controls_the_ticker_count():
    snippets = make_set([f"T{i}" for i in range(10)])

    split = split_by_ticker(snippets, test_fraction=0.3)

    assert len(split.test_tickers) == 3
    assert len(split.development_tickers) == 7


def test_at_least_one_ticker_is_held_out():
    snippets = make_set(["AAA", "BBB"])

    split = split_by_ticker(snippets, test_fraction=0.1)

    assert len(split.test_tickers) >= 1


def test_at_least_one_ticker_stays_in_development():
    snippets = make_set(["AAA", "BBB"])

    split = split_by_ticker(snippets, test_fraction=0.9)

    assert len(split.development_tickers) >= 1


# --- Reproducibility -------------------------------------------------------


def test_the_same_seed_gives_the_same_split():
    snippets = make_set([f"T{i}" for i in range(10)])

    first = split_by_ticker(snippets, seed=7)
    second = split_by_ticker(snippets, seed=7)

    assert first.test_tickers == second.test_tickers


def test_different_seeds_give_different_splits():
    snippets = make_set([f"T{i}" for i in range(20)])

    first = split_by_ticker(snippets, seed=1)
    second = split_by_ticker(snippets, seed=2)

    assert first.test_tickers != second.test_tickers


# --- Edge cases ------------------------------------------------------------


def test_snippets_without_a_ticker_go_to_development():
    snippets = [snippet("a", "AAA"), snippet("b", "BBB"), snippet("c", None)]

    split = split_by_ticker(snippets, test_fraction=0.5)

    assert any(s.id == "c" for s in split.development)


def test_a_single_ticker_cannot_be_split():
    with pytest.raises(ValueError, match="at least two tickers"):
        split_by_ticker(make_set(["AAA"]))


def test_an_invalid_fraction_is_rejected():
    with pytest.raises(ValueError, match="test_fraction"):
        split_by_ticker(make_set(["AAA", "BBB"]), test_fraction=1.5)


def test_summary_reports_both_partitions():
    split = split_by_ticker(make_set([f"T{i}" for i in range(10)]))

    summary = split.summary()

    assert "development:" in summary
    assert "test:" in summary
