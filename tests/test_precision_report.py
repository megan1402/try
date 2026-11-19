"""Tests for :mod:`app.evaluation.precision`.

The Wilson interval is checked against published values, because a subtly wrong
interval would quietly misstate how much the sample can support.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from app.evaluation.gold import GoldSnippet
from app.evaluation.precision import (
    PRECISION_COLUMNS,
    pooled_precision,
    precision_table,
    wilson_interval,
)


def snippet(detected: str, label: str | None) -> GoldSnippet:
    return GoldSnippet(
        id=f"{detected}-{label}",
        text="text",
        label=label,
        detected_type=detected,
    )


# --- Wilson interval -------------------------------------------------------


@pytest.mark.parametrize(
    ("successes", "trials", "expected"),
    [
        # Published Wilson 95% intervals.
        (5, 5, (0.566, 1.0)),
        (0, 5, (0.0, 0.434)),
        (50, 100, (0.404, 0.596)),
        (1, 10, (0.018, 0.404)),
    ],
)
def test_wilson_matches_published_values(
    successes: int, trials: int, expected: tuple[float, float]
):
    low, high = wilson_interval(successes, trials)

    assert low == pytest.approx(expected[0], abs=0.002)
    assert high == pytest.approx(expected[1], abs=0.002)


def test_wilson_stays_inside_the_unit_interval():
    for successes in range(0, 11):
        low, high = wilson_interval(successes, 10)
        assert 0.0 <= low <= high <= 1.0


def test_wilson_is_narrower_with_more_data():
    small = wilson_interval(8, 10)
    large = wilson_interval(80, 100)

    assert (large[1] - large[0]) < (small[1] - small[0])


def test_wilson_of_nothing_is_nan():
    low, high = wilson_interval(0, 0)

    assert math.isnan(low)
    assert math.isnan(high)


def test_wilson_rejects_impossible_counts():
    with pytest.raises(ValueError, match="Invalid counts"):
        wilson_interval(5, 3)


def test_wilson_rejects_an_invalid_confidence():
    with pytest.raises(ValueError, match="confidence"):
        wilson_interval(1, 2, confidence=1.5)


def test_a_ninety_percent_interval_is_narrower_than_a_ninety_five():
    wide = wilson_interval(8, 10, confidence=0.95)
    narrow = wilson_interval(8, 10, confidence=0.90)

    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


# --- Per-class precision ---------------------------------------------------


def test_precision_counts_a_matching_label_as_correct():
    snippets = [
        snippet("acquisition", "acquisition"),
        snippet("acquisition", "acquisition"),
        snippet("acquisition", "none"),
    ]

    table = precision_table(snippets)

    row = table.iloc[0]
    assert row["labelled"] == 3
    assert row["correct"] == 2
    assert row["precision"] == pytest.approx(2 / 3)


def test_none_is_an_error_not_a_missing_value():
    snippets = [snippet("acquisition", "none")]

    table = precision_table(snippets)

    assert table.iloc[0]["correct"] == 0
    assert table.iloc[0]["precision"] == pytest.approx(0.0)


def test_a_different_event_class_is_an_error():
    snippets = [snippet("acquisition", "divestiture")]

    assert precision_table(snippets).iloc[0]["correct"] == 0


def test_precision_is_reported_per_class():
    snippets = [
        snippet("acquisition", "acquisition"),
        snippet("divestiture", "none"),
    ]

    table = precision_table(snippets)

    assert set(table["event_type"]) == {"acquisition", "divestiture"}
    assert table[table["event_type"] == "acquisition"].iloc[0]["precision"] == 1.0
    assert table[table["event_type"] == "divestiture"].iloc[0]["precision"] == 0.0


def test_unlabelled_and_skipped_snippets_are_excluded():
    snippets = [
        snippet("acquisition", "acquisition"),
        snippet("acquisition", None),
        snippet("acquisition", "skip"),
    ]

    table = precision_table(snippets)

    assert table.iloc[0]["labelled"] == 1


def test_table_has_the_documented_columns():
    table = precision_table([snippet("acquisition", "acquisition")])

    assert list(table.columns) == list(PRECISION_COLUMNS)


def test_interval_width_is_reported():
    table = precision_table([snippet("acquisition", "acquisition")])

    row = table.iloc[0]
    assert row["ci_width"] == pytest.approx(row["ci_high"] - row["ci_low"])


def test_precision_requires_labelled_detected_events():
    with pytest.raises(ValueError, match="No labelled detected events"):
        precision_table([snippet("acquisition", None)])


def test_snippets_without_a_detected_type_are_ignored():
    plain = GoldSnippet(id="x", text="text", label="acquisition")

    with pytest.raises(ValueError, match="No labelled detected events"):
        precision_table([plain])


# --- Pooled precision ------------------------------------------------------


def test_pooled_precision_combines_classes():
    snippets = [
        snippet("acquisition", "acquisition"),
        snippet("divestiture", "none"),
    ]

    pooled = pooled_precision(snippets)

    assert pooled["labelled"] == 2
    assert pooled["correct"] == 1
    assert pooled["precision"] == pytest.approx(0.5)


def test_pooled_precision_requires_labels():
    with pytest.raises(ValueError, match="No labelled detected events"):
        pooled_precision([snippet("acquisition", None)])


def test_pooled_precision_returns_a_dataframe_compatible_dict():
    pooled = pooled_precision([snippet("acquisition", "acquisition")])

    assert set(pooled) == {"labelled", "correct", "precision", "ci_low", "ci_high"}
    assert isinstance(pd.Series(pooled), pd.Series)
