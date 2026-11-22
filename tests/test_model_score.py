"""Tests for :mod:`app.models.score`.

The scoring arithmetic is tested with a fake predictor, so no model is loaded and
no network is touched.
"""

from __future__ import annotations

import pytest

from app.evaluation.gold import GoldSnippet
from app.models.score import (
    Prediction,
    pooled,
    precision_by_class,
    predict_all,
    recall_by_class,
)


def snippet(snippet_id: str, text: str, label: str | None) -> GoldSnippet:
    return GoldSnippet(id=snippet_id, text=text, label=label)


def prediction(predicted: str, label: str | None) -> Prediction:
    return Prediction(snippet_id="x", predicted=predicted, label=label)


# --- Running the predictor -------------------------------------------------


def test_predict_all_keeps_the_label_alongside():
    snippets = [snippet("a", "text a", "acquisition")]

    predictions = predict_all(snippets, lambda text: "acquisition")

    assert predictions[0].predicted == "acquisition"
    assert predictions[0].label == "acquisition"


def test_predict_all_passes_the_snippet_text():
    seen: list[str] = []

    def predict(text: str) -> str:
        seen.append(text)
        return "none"

    predict_all([snippet("a", "hello", None)], predict)

    assert seen == ["hello"]


# --- Precision -------------------------------------------------------------


def test_precision_counts_matching_predictions():
    predictions = [
        prediction("acquisition", "acquisition"),
        prediction("acquisition", "none"),
    ]

    table = precision_by_class(predictions)

    row = table.iloc[0]
    assert row["labelled"] == 2
    assert row["correct"] == 1
    assert row["precision"] == pytest.approx(0.5)


def test_precision_includes_none_as_a_predicted_class():
    predictions = [prediction("none", "none"), prediction("none", "acquisition")]

    table = precision_by_class(predictions)

    row = table[table["event_type"] == "none"].iloc[0]
    assert row["correct"] == 1
    assert row["precision"] == pytest.approx(0.5)


def test_precision_skips_unlabelled_and_skipped():
    predictions = [
        prediction("acquisition", "acquisition"),
        prediction("acquisition", None),
        prediction("acquisition", "skip"),
    ]

    table = precision_by_class(predictions)

    assert table.iloc[0]["labelled"] == 1


def test_precision_requires_labels():
    with pytest.raises(ValueError, match="No labelled"):
        precision_by_class([prediction("acquisition", None)])


# --- Recall ----------------------------------------------------------------


def test_recall_counts_found_events():
    predictions = [
        prediction("acquisition", "acquisition"),
        prediction("none", "acquisition"),
    ]

    table = recall_by_class(predictions)

    row = table.iloc[0]
    assert row["labelled"] == 2
    assert row["correct"] == 1
    assert row["recall"] == pytest.approx(0.5)


def test_recall_ignores_none_labels():
    table = recall_by_class([prediction("none", "none")])

    assert table.empty


def test_recall_requires_labels():
    with pytest.raises(ValueError, match="No labelled"):
        recall_by_class([prediction("none", None)])


# --- Pooled ----------------------------------------------------------------


def test_pooled_precision_and_recall():
    predictions = [
        prediction("acquisition", "acquisition"),  # correct prediction, found
        prediction("acquisition", "none"),  # false positive
        prediction("none", "acquisition"),  # false negative
        prediction("none", "none"),  # correct rejection
    ]

    summary = pooled(predictions)

    assert summary["predicted_events"] == 2
    assert summary["precision"] == pytest.approx(0.5)
    assert summary["real_events"] == 2
    assert summary["found"] == 1
    assert summary["recall"] == pytest.approx(0.5)


def test_pooled_handles_a_model_that_never_fires():
    predictions = [prediction("none", "acquisition"), prediction("none", "none")]

    summary = pooled(predictions)

    assert summary["predicted_events"] == 0
    assert summary["precision"] == 0.0
    assert summary["recall"] == 0.0


def test_pooled_handles_no_real_events():
    predictions = [prediction("none", "none")]

    summary = pooled(predictions)

    assert summary["real_events"] == 0
    assert summary["recall"] == 0.0


def test_pooled_reports_intervals():
    summary = pooled([prediction("acquisition", "acquisition")])

    assert 0.0 <= summary["precision_low"] <= summary["precision_high"] <= 1.0
    assert 0.0 <= summary["recall_low"] <= summary["recall_high"] <= 1.0


def test_pooled_requires_labels():
    with pytest.raises(ValueError, match="No labelled"):
        pooled([prediction("none", None)])
