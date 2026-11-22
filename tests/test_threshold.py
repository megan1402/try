"""Tests for :mod:`app.models.threshold`.

The sweep arithmetic is tested with hand-built scores, so no model is loaded.
"""

from __future__ import annotations

import pytest

from app.evaluation.gold import GoldSnippet
from app.models.threshold import (
    SWEEP_COLUMNS,
    ScoredSnippet,
    best_threshold,
    score_snippets,
    sweep,
)


def scored(label: str | None, **scores: float) -> ScoredSnippet:
    return ScoredSnippet(label=label, scores=dict(scores))


# --- Predicting at a threshold ---------------------------------------------


def test_below_threshold_abstains():
    snippet = scored("acquisition", acquisition=0.2, none=0.8)

    assert snippet.predict(0.5) == "none"


def test_above_threshold_returns_the_event():
    snippet = scored("acquisition", acquisition=0.9, none=0.1)

    assert snippet.predict(0.5) == "acquisition"


def test_zero_threshold_reproduces_the_argmax():
    snippet = scored("acquisition", acquisition=0.01, none=0.99)

    assert snippet.predict(0.0) == "acquisition"


def test_none_is_never_returned_by_the_argmax():
    # Even when none scores highest, the argmax picks an event; the threshold is
    # what makes abstention possible.
    snippet = scored("none", acquisition=0.1, none=0.9)

    assert snippet.predict(0.0) == "acquisition"
    assert snippet.predict(0.5) == "none"


def test_best_event_ignores_none():
    snippet = scored("none", acquisition=0.3, none=0.9)

    label, score = snippet.best_event()

    assert label == "acquisition"
    assert score == pytest.approx(0.3)


def test_best_event_breaks_ties_alphabetically():
    snippet = scored("none", divestiture=0.5, acquisition=0.5)

    assert snippet.best_event()[0] == "acquisition"


def test_no_event_scores_means_none():
    snippet = scored("none", none=0.9)

    assert snippet.best_event() == ("none", 0.0)
    assert snippet.predict(0.0) == "none"


# --- Scoring ---------------------------------------------------------------


def test_score_snippets_keeps_labels_and_scores():
    snippets = [GoldSnippet(id="a", text="hello", label="acquisition")]

    scored_snippets = score_snippets(snippets, lambda text: {"acquisition": 0.7})

    assert scored_snippets[0].label == "acquisition"
    assert scored_snippets[0].scores == {"acquisition": 0.7}


# --- Sweeping --------------------------------------------------------------


def test_sweep_has_the_documented_columns():
    table = sweep([scored("acquisition", acquisition=0.9)], thresholds=[0.5])

    assert list(table.columns) == list(SWEEP_COLUMNS)


def test_a_high_threshold_reduces_predictions():
    scored_snippets = [
        scored("acquisition", acquisition=0.9),
        scored("none", acquisition=0.2),
    ]

    table = sweep(scored_snippets, thresholds=[0.0, 0.5])

    assert table.iloc[0]["predicted_events"] == 2
    assert table.iloc[1]["predicted_events"] == 1


def test_precision_improves_as_the_threshold_rises():
    scored_snippets = [
        scored("acquisition", acquisition=0.9),  # correct, high score
        scored("none", acquisition=0.2),  # wrong, low score
    ]

    table = sweep(scored_snippets, thresholds=[0.0, 0.5])

    assert table.iloc[0]["precision"] == pytest.approx(0.5)
    assert table.iloc[1]["precision"] == pytest.approx(1.0)


def test_recall_falls_as_the_threshold_rises():
    scored_snippets = [
        scored("acquisition", acquisition=0.9),
        scored("acquisition", acquisition=0.2),
    ]

    table = sweep(scored_snippets, thresholds=[0.0, 0.5])

    assert table.iloc[0]["recall"] == pytest.approx(1.0)
    assert table.iloc[1]["recall"] == pytest.approx(0.5)


def test_sweep_skips_unlabelled_and_skipped():
    scored_snippets = [
        scored("acquisition", acquisition=0.9),
        scored(None, acquisition=0.9),
        scored("skip", acquisition=0.9),
    ]

    table = sweep(scored_snippets, thresholds=[0.0])

    assert table.iloc[0]["predicted_events"] == 1


def test_sweep_requires_labels():
    with pytest.raises(ValueError, match="No labelled"):
        sweep([scored(None, acquisition=0.9)])


def test_sweep_reports_intervals():
    table = sweep([scored("acquisition", acquisition=0.9)], thresholds=[0.0])

    row = table.iloc[0]
    assert 0.0 <= row["precision_low"] <= row["precision_high"] <= 1.0
    assert 0.0 <= row["recall_low"] <= row["recall_high"] <= 1.0


# --- Choosing a threshold --------------------------------------------------


def test_best_threshold_returns_the_lowest_qualifying():
    scored_snippets = [
        scored("acquisition", acquisition=0.9),
        scored("acquisition", acquisition=0.6),
        scored("none", acquisition=0.2),
    ]
    table = sweep(scored_snippets, thresholds=[0.0, 0.5, 0.8])

    # At 0.0 precision is already 2/3, so it qualifies and is the lowest.
    winner = best_threshold(table, minimum_precision=0.5, minimum_recall=0.5)

    assert winner is not None
    assert winner["threshold"] == pytest.approx(0.0)


def test_best_threshold_skips_thresholds_that_fail_the_precision_floor():
    scored_snippets = [
        scored("acquisition", acquisition=0.9),
        scored("none", acquisition=0.6),
        scored("none", acquisition=0.2),
    ]
    table = sweep(scored_snippets, thresholds=[0.0, 0.5, 0.8])

    # At 0.0 precision is 1/3; at 0.5 it is exactly 1/2, which meets the floor.
    winner = best_threshold(table, minimum_precision=0.5, minimum_recall=0.5)

    assert winner is not None
    assert winner["threshold"] == pytest.approx(0.5)


def test_best_threshold_returns_none_when_precision_never_clears():
    scored_snippets = [
        scored("acquisition", acquisition=0.9),
        scored("none", acquisition=0.6),
        scored("none", acquisition=0.2),
    ]
    table = sweep(scored_snippets, thresholds=[0.0, 0.5])

    # Precision tops out at 0.5, so a 0.6 floor is never met.
    assert best_threshold(table, minimum_precision=0.6, minimum_recall=0.5) is None


def test_best_threshold_is_none_when_nothing_qualifies():
    scored_snippets = [scored("none", acquisition=0.9)]
    table = sweep(scored_snippets, thresholds=[0.0, 0.5])

    assert best_threshold(table, minimum_precision=0.3, minimum_recall=0.3) is None
