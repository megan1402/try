"""Tests for :mod:`app.models.zero_shot` and :mod:`app.models.compare`.

No model is downloaded and no network is touched: the pipeline is injected, which
is exactly why :class:`ZeroShotEventClassifier` takes a callable.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
import pytest

from app.evaluation.gold import NONE_LABEL, GoldSnippet
from app.extraction.events import EVENT_TYPES
from app.models.compare import COMPARISON_COLUMNS, agreement_rate, compare
from app.models.zero_shot import (
    HYPOTHESES,
    ZeroShotEventClassifier,
    check_hypotheses,
    hypotheses_in_order,
)


def fake_pipeline(
    scores_by_label: Mapping[str, float], seen: list[list[str]] | None = None
):
    """Build a stand-in for ``transformers.pipeline`` that returns fixed scores."""

    def classify(text: str, hypotheses: Sequence[str]) -> Mapping[str, Any]:
        if seen is not None:
            seen.append(list(hypotheses))
        by_hypothesis = {
            HYPOTHESES[label]: score for label, score in scores_by_label.items()
        }
        # Return in reverse order on purpose: real pipelines sort by score, so
        # the code must not rely on positional alignment.
        hypotheses = list(hypotheses)[::-1]
        return {
            "labels": hypotheses,
            "scores": [by_hypothesis.get(hypothesis, 0.0) for hypothesis in hypotheses],
        }

    return classify


# --- Hypothesis table ------------------------------------------------------


def test_hypotheses_cover_every_event_class():
    assert set(HYPOTHESES) == {*EVENT_TYPES, NONE_LABEL}
    check_hypotheses()  # must not raise.


def test_every_hypothesis_is_a_usable_sentence():
    assert all(hypothesis == hypothesis.strip() for hypothesis in HYPOTHESES.values())
    assert all(len(hypothesis) > 15 for hypothesis in HYPOTHESES.values())


def test_hypotheses_are_unique():
    assert len(set(hypotheses_in_order())) == len(HYPOTHESES)


def test_drift_between_hypotheses_and_event_classes_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
):
    module = importlib.import_module("app.models.zero_shot")
    monkeypatch.setattr(module, "HYPOTHESES", {**HYPOTHESES, "made_up": "something"})

    with pytest.raises(ValueError, match="describe exactly"):
        check_hypotheses()


# --- Mapping scores to labels ---------------------------------------------


def test_scores_are_matched_by_hypothesis_not_by_position():
    classifier = ZeroShotEventClassifier(
        classifier=fake_pipeline({"guidance_cut": 0.91, NONE_LABEL: 0.05})
    )

    scores = classifier.scores("We lowered our outlook.")

    assert scores["guidance_cut"] == pytest.approx(0.91)
    assert scores[NONE_LABEL] == pytest.approx(0.05)


def test_predict_returns_the_highest_scoring_label():
    classifier = ZeroShotEventClassifier(
        classifier=fake_pipeline({"acquisition": 0.87, NONE_LABEL: 0.10})
    )

    assert classifier.predict("We acquired Acme Corp.") == "acquisition"


def test_predict_can_return_none():
    classifier = ZeroShotEventClassifier(
        classifier=fake_pipeline({NONE_LABEL: 0.72, "acquisition": 0.20})
    )

    assert classifier.predict("Depreciation is computed straight-line.") == NONE_LABEL


def test_predict_breaks_ties_deterministically():
    classifier = ZeroShotEventClassifier(
        classifier=fake_pipeline({"divestiture": 0.5, "acquisition": 0.5})
    )

    assert classifier.predict("anything") == "acquisition"


def test_the_classifier_receives_every_hypothesis():
    seen: list[list[str]] = []
    classifier = ZeroShotEventClassifier(classifier=fake_pipeline({}, seen=seen))

    classifier.scores("anything")

    assert seen == [list(hypotheses_in_order())]


# --- Rules versus model ----------------------------------------------------


def make_snippet(snippet_id: str, text: str) -> GoldSnippet:
    return GoldSnippet(id=snippet_id, text=text)


def test_compare_records_agreement_and_columns():
    snippets = [
        make_snippet("a", "We raised our full-year guidance."),
        make_snippet("b", "Depreciation is computed straight-line."),
    ]
    model = {
        "We raised our full-year guidance.": "guidance_raise",
        "Depreciation is computed straight-line.": "acquisition",
    }.__getitem__

    frame = compare(snippets, model)

    assert list(frame.columns) == ["id", "rules", "model", "agree", "text"]
    assert frame["agree"].tolist() == [True, False]
    assert agreement_rate(frame) == pytest.approx(0.5)


def test_compare_truncates_long_text():
    snippets = [make_snippet("a", "word " * 100)]

    frame = compare(snippets, lambda text: "none")

    assert len(frame.loc[0, "text"]) <= 160


def test_agreement_rate_of_nothing_is_nan():
    empty = pd.DataFrame(columns=list(COMPARISON_COLUMNS))

    assert math.isnan(agreement_rate(empty))
