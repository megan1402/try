"""Tests for :mod:`app.evaluation.filter_passages`."""

from __future__ import annotations

import pytest

from app.evaluation.filter_passages import (
    EVENT_VOCABULARY,
    filter_passages,
    filter_sidecar,
    mentions_event,
)
from app.evaluation.gold import GoldSnippet


def snippet(snippet_id: str, text: str) -> GoldSnippet:
    return GoldSnippet(id=snippet_id, text=text, ticker="NVDA")


# --- Vocabulary matching ---------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "We completed the acquisition of Acme Corp.",
        "The company acquired a wafer fabrication facility.",
        "Following Mr. Huang's resignation, the Board appointed a successor.",
        "We recorded a settlement of $50 million.",
        "We announced a restructuring plan.",
        "We raised our full-year guidance.",
        "Revenue exceeded consensus expectations.",
    ],
)
def test_event_vocabulary_is_detected(text: str):
    assert mentions_event(text)


@pytest.mark.parametrize(
    "text",
    [
        "Revenue grew 12% year over year.",
        "Deferred tax assets were $3,473 million.",
        "The Company has a 48% ownership interest in the resort.",
        "Balance at October 25, 2025 was $(16) million.",
    ],
)
def test_ordinary_prose_is_not_flagged(text: str):
    assert not mentions_event(text)


def test_matching_is_case_insensitive():
    assert mentions_event("ACQUISITION of Acme")


def test_vocabulary_is_documented():
    assert EVENT_VOCABULARY.pattern


# --- Filtering -------------------------------------------------------------


def test_filtering_keeps_only_event_passages():
    snippets = [
        snippet("ps0001", "Revenue grew 12% year over year."),
        snippet("ps0002", "We completed the acquisition of Acme."),
        snippet("ps0003", "Deferred tax assets were $3,473 million."),
    ]

    kept = filter_passages(snippets)

    assert len(kept) == 1
    assert "acquisition" in kept[0].text


def test_filtered_snippets_are_renumbered():
    snippets = [
        snippet("ps0001", "We completed the acquisition of Acme."),
        snippet("ps0002", "We recorded a settlement of $50 million."),
    ]

    kept = filter_passages(snippets)

    assert [s.id for s in kept] == ["pe0001", "pe0002"]


def test_filtered_snippets_keep_provenance_and_stay_unlabelled():
    snippets = [snippet("ps0001", "We completed the acquisition of Acme.")]

    kept = filter_passages(snippets)[0]

    assert kept.ticker == "NVDA"
    assert kept.label is None
    assert kept.proposed is None
    assert kept.detected_type is None


def test_filtering_nothing_returns_nothing():
    assert filter_passages([snippet("ps0001", "Revenue grew 12%.")]) == []


# --- Sidecar re-keying -----------------------------------------------------


def test_sidecar_answers_follow_the_filtered_ids():
    snippets = [
        snippet("ps0001", "Revenue grew 12%."),
        snippet("ps0002", "We completed the acquisition of Acme."),
    ]
    answers = {"ps0001": [], "ps0002": ["acquisition"]}

    kept = filter_passages(snippets)
    filtered = filter_sidecar(answers, snippets, kept)

    assert filtered == {"pe0001": ["acquisition"]}


def test_sidecar_keeps_empty_answers():
    snippets = [snippet("ps0001", "We recorded a settlement of $50 million.")]
    answers = {"ps0001": []}

    filtered = filter_sidecar(answers, snippets, filter_passages(snippets))

    assert filtered == {"pe0001": []}
