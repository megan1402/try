"""Tests for :mod:`app.evaluation.sampling`.

The allocation logic is the part that decides whether a rare class gets a usable
sample, so it is tested against hand-computed quotas.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.evaluation.sampling import (
    CONTEXT_CHARS,
    DetectedEvent,
    allocate,
    context_around,
    from_filing_event,
    sample_events,
    to_snippets,
)
from app.extraction.events import FilingEvent


def make_event(event_type: str, index: int = 0) -> DetectedEvent:
    return DetectedEvent(
        ticker="NVDA",
        accession_number=f"acc-{index}",
        section_key="part_i_item_2",
        event_type=event_type,
        sentence=f"sentence {index}",
        context=f"context {index}",
    )


# --- Allocation ------------------------------------------------------------


def test_allocation_gives_every_class_a_floor():
    quotas = allocate({"acquisition": 100, "divestiture": 20}, 10, minimum=3)

    assert quotas["divestiture"] >= 3
    assert quotas["acquisition"] > quotas["divestiture"]
    assert sum(quotas.values()) == 10


def test_allocation_caps_a_floor_at_what_is_available():
    # Only two divestitures exist, so the floor of three cannot be met.
    quotas = allocate({"acquisition": 100, "divestiture": 2}, 10, minimum=3)

    assert quotas["divestiture"] == 2
    assert sum(quotas.values()) == 10


def test_allocation_never_exceeds_what_is_available():
    quotas = allocate({"acquisition": 100, "divestiture": 2}, 10, minimum=5)

    assert quotas["divestiture"] == 2
    assert sum(quotas.values()) <= 10


def test_allocation_sums_to_the_total_when_there_is_capacity():
    quotas = allocate({"a": 100, "b": 100, "c": 100}, 30, minimum=5)

    assert sum(quotas.values()) == 30


def test_allocation_is_proportional_for_large_classes():
    quotas = allocate({"big": 900, "small": 100}, 100, minimum=5)

    assert quotas["big"] > quotas["small"]
    assert sum(quotas.values()) == 100


def test_allocation_ignores_empty_classes():
    quotas = allocate({"a": 10, "b": 0}, 5, minimum=1)

    assert "b" not in quotas


def test_allocation_of_nothing_is_empty():
    assert allocate({}, 10) == {}


def test_allocation_rejects_a_non_positive_total():
    with pytest.raises(ValueError, match="total"):
        allocate({"a": 1}, 0)


def test_allocation_rejects_a_negative_minimum():
    with pytest.raises(ValueError, match="minimum"):
        allocate({"a": 1}, 10, minimum=-1)


# --- Sampling --------------------------------------------------------------


def test_sampling_is_reproducible_with_the_same_seed():
    events = [make_event("acquisition", i) for i in range(50)]

    first = sample_events(events, total=10, seed=7)
    second = sample_events(events, total=10, seed=7)

    assert [e.accession_number for e in first] == [e.accession_number for e in second]


def test_different_seeds_give_different_samples():
    events = [make_event("acquisition", i) for i in range(50)]

    first = sample_events(events, total=10, seed=1)
    second = sample_events(events, total=10, seed=2)

    assert [e.accession_number for e in first] != [e.accession_number for e in second]


def test_sampling_does_not_repeat_an_event():
    events = [make_event("acquisition", i) for i in range(50)]

    sample = sample_events(events, total=20)

    assert len({e.accession_number for e in sample}) == 20


def test_sampling_keeps_rare_classes_represented():
    events = [make_event("acquisition", i) for i in range(200)]
    events += [make_event("divestiture", i) for i in range(3)]

    sample = sample_events(events, total=50, minimum=3)

    assert sum(1 for e in sample if e.event_type == "divestiture") == 3


def test_sampling_returns_everything_when_the_corpus_is_small():
    events = [make_event("acquisition", i) for i in range(4)]

    assert len(sample_events(events, total=150)) == 4


def test_sampling_of_nothing_is_empty():
    assert sample_events([], total=10) == []


# --- Snippets --------------------------------------------------------------


def test_snippets_carry_provenance_and_no_proposal():
    snippets = to_snippets([make_event("acquisition", 3)])

    snippet = snippets[0]
    assert snippet.detected_type == "acquisition"
    assert snippet.ticker == "NVDA"
    assert snippet.accession_number == "acc-3"
    assert snippet.section_key == "part_i_item_2"
    assert snippet.context == "context 3"
    assert snippet.label is None
    # No proposal: showing the detector's own answer would anchor the reviewer.
    assert snippet.proposed is None


def test_snippet_ids_are_unique_and_ordered():
    snippets = to_snippets([make_event("acquisition", i) for i in range(3)])

    assert [s.id for s in snippets] == ["ev0001", "ev0002", "ev0003"]


# --- Context ---------------------------------------------------------------


def test_context_includes_text_either_side():
    text = "A" * 300 + "MATCH" + "B" * 300
    start = text.index("MATCH")

    context = context_around(text, start, start + 5, width=50)

    assert "MATCH" in context
    assert len(context) <= 105


def test_context_is_clamped_at_the_document_edges():
    text = "MATCH at the start"

    context = context_around(text, 0, 5, width=100)

    assert context == text


def test_context_collapses_whitespace():
    text = "before\n\n  MATCH   after"

    context = context_around(text, text.index("MATCH"), text.index("MATCH") + 5)

    assert "\n" not in context
    assert "  " not in context


def test_context_width_is_documented():
    assert CONTEXT_CHARS > 0


# --- From a detector result ------------------------------------------------


def test_from_filing_event_uses_the_sentence_and_offsets():
    text = "Intro. We completed the acquisition of Acme Corp. Outro."
    start = text.index("completed")
    event = FilingEvent(
        event_type="acquisition",
        confidence=0.5,
        start=start,
        end=start + 9,
        section_key="part_i_item_2",
        evidence="completed",
        as_of=datetime(2026, 1, 1, tzinfo=UTC),
        sentence="We completed the acquisition of Acme Corp.",
    )

    detected = from_filing_event(
        event, ticker="NVDA", accession_number="acc-1", text=text
    )

    assert detected.sentence == "We completed the acquisition of Acme Corp."
    assert "Intro" in detected.context
    assert detected.event_type == "acquisition"
