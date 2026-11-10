"""Tests for :mod:`app.evaluation.report`.

The scorer is tested with a hand-built prediction function so the arithmetic is
checked independently of the rule baseline — otherwise a bug in the rules could
hide behind an identical bug in the score.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from app.evaluation import gold
from app.evaluation.report import (
    ERROR_COLUMNS,
    REPORT_COLUMNS,
    accuracy,
    macro_scores,
    misclassified,
    predict_label,
    score,
)


def snippet(
    snippet_id: str, label: str | None, text: str | None = None
) -> gold.GoldSnippet:
    return gold.GoldSnippet(id=snippet_id, text=text or snippet_id, label=label)


def row(table: pd.DataFrame, label: str) -> pd.Series:
    match = table[table["label"] == label]
    assert len(match) == 1
    return match.iloc[0]


def fixed(predictions: dict[str, str]):
    return lambda text: predictions[text]


# --- Confusion counts ------------------------------------------------------


SAMPLE = [
    snippet("a", "guidance_raise"),
    snippet("b", "guidance_raise"),
    snippet("c", "none"),
]
SAMPLE_PREDICTIONS = {"a": "guidance_raise", "b": "none", "c": "guidance_raise"}


def test_counts_and_metrics_are_correct():
    table = score(SAMPLE, fixed(SAMPLE_PREDICTIONS))

    raise_row = row(table, "guidance_raise")
    assert raise_row["support"] == 2
    assert raise_row["predicted"] == 2
    assert raise_row["tp"] == 1
    assert raise_row["fp"] == 1
    assert raise_row["fn"] == 1
    assert raise_row["precision"] == pytest.approx(0.5)
    assert raise_row["recall"] == pytest.approx(0.5)
    assert raise_row["f1"] == pytest.approx(0.5)

    none_row = row(table, "none")
    assert none_row["support"] == 1
    assert none_row["predicted"] == 1
    assert none_row["tp"] == 0
    assert none_row["precision"] == pytest.approx(0.0)
    assert none_row["recall"] == pytest.approx(0.0)
    assert none_row["f1"] == pytest.approx(0.0)


def test_table_has_the_documented_columns():
    table = score(SAMPLE, fixed(SAMPLE_PREDICTIONS))

    assert list(table.columns) == list(REPORT_COLUMNS)


def test_skipped_and_unlabelled_snippets_are_excluded():
    snippets = [
        snippet("a", "guidance_raise"),
        snippet("b", gold.SKIP_LABEL),
        snippet("c", None),
    ]

    table = score(snippets, fixed({"a": "guidance_raise"}))

    assert table["support"].sum() == 1
    assert table["predicted"].sum() == 1


def test_a_predicted_only_class_appears_with_zero_support():
    snippets = [snippet("a", "none")]

    table = score(snippets, fixed({"a": "acquisition"}))

    acquisition = row(table, "acquisition")
    assert acquisition["support"] == 0
    assert acquisition["precision"] == pytest.approx(0.0)
    assert math.isnan(acquisition["recall"])
    assert math.isnan(acquisition["f1"])


def test_scoring_nothing_is_an_error():
    with pytest.raises(ValueError, match="No labelled snippets"):
        score([snippet("a", None), snippet("b", gold.SKIP_LABEL)], fixed({}))


def test_macro_scores_average_only_classes_with_support():
    snippets = [snippet("a", "guidance_raise"), snippet("b", "none")]
    table = score(snippets, fixed({"a": "guidance_raise", "b": "acquisition"}))

    macro = macro_scores(table)

    # guidance_raise is perfect; "none" has support but was never predicted, so
    # its precision is undefined and its recall is 0. "acquisition" was only
    # predicted, so it has no support and is left out of the average.
    assert macro["macro_precision"] == pytest.approx(1.0)
    assert macro["macro_recall"] == pytest.approx(0.5)
    assert macro["macro_f1"] == pytest.approx(1.0)


def test_accuracy_counts_exact_matches():
    assert accuracy(SAMPLE, fixed(SAMPLE_PREDICTIONS)) == pytest.approx(1 / 3)


def test_accuracy_requires_labels():
    with pytest.raises(ValueError, match="No labelled snippets"):
        accuracy([snippet("a", None)], fixed({}))


def test_misclassified_lists_the_disagreements():
    errors = misclassified(SAMPLE, fixed(SAMPLE_PREDICTIONS))

    assert list(errors.columns) == list(ERROR_COLUMNS)
    assert sorted(errors["id"]) == ["b", "c"]


def test_misclassified_is_empty_when_everything_matches():
    snippets = [snippet("a", "guidance_raise")]

    assert misclassified(snippets, fixed({"a": "guidance_raise"})).empty


# --- The real predictor ----------------------------------------------------


def test_predictor_maps_clear_language_to_its_class():
    assert predict_label("We raised our full-year guidance.") == "guidance_raise"
    assert predict_label("We completed the acquisition of Acme Corp.") == "acquisition"


def test_predictor_returns_none_for_neutral_prose():
    assert predict_label("Operating expenses increased 9% year over year.") == "none"


def test_predictor_picks_the_highest_confidence_match():
    # guidance_raise is weighted above acquisition, so it wins the tie.
    text = "We raised guidance and completed the acquisition of Acme Corp."

    assert predict_label(text) == "guidance_raise"
