"""Tests for :mod:`app.extraction.events`.

These pin the *baseline* rules. They are not claims of accuracy — a later
golden set measures precision and recall. What they guarantee is that each rule
still fires on its canonical phrasing, that offsets point at the evidence, and
that the point-in-time contract (a tz-aware ``as_of``) is enforced.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.extraction.events import (
    EVENT_COLUMNS,
    EVENT_TYPES,
    FilingEvent,
    events_to_frame,
    extract_events,
)
from app.extraction.sections import Section

AS_OF = datetime(2026, 7, 26, 20, 4, tzinfo=UTC)


def make_section(
    text: str,
    *,
    start: int = 0,
    part: str | None = "I",
    item: str = "2",
) -> Section:
    return Section(
        part=part,
        item=item,
        title="MD&A",
        start=start,
        end=start + len(text),
        text=text,
    )


def types_of(text: str) -> set[str]:
    return {
        event.event_type for event in extract_events([make_section(text)], as_of=AS_OF)
    }


# --- One rule at a time ----------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("We raised our full-year guidance for fiscal 2027.", "guidance_raise"),
        ("Management increased its outlook for the next quarter.", "guidance_raise"),
        ("We lowered our outlook for the second half of the year.", "guidance_cut"),
        ("Guidance was reduced to reflect weaker demand.", "guidance_cut"),
        ("Revenue exceeded consensus expectations.", "earnings_beat"),
        ("Revenue fell short of consensus expectations.", "earnings_miss"),
        ("We completed the acquisition of Acme Corp.", "acquisition"),
        ("We divested our networking business in the quarter.", "divestiture"),
        (
            "Our Chief Financial Officer resigned effective immediately.",
            "management_change",
        ),
        ("We announced a restructuring plan including layoffs.", "restructuring"),
        ("We received a subpoena from the regulator.", "legal_or_regulatory"),
    ],
)
def test_each_event_type_has_a_working_rule(text: str, expected: str):
    assert expected in types_of(text)


def test_every_rule_uses_a_known_event_type():
    # Guards against a typo'd string in the rule table, which would raise only
    # when that rule happened to fire.
    assert set(EVENT_TYPES) == {
        "guidance_raise",
        "guidance_cut",
        "earnings_beat",
        "earnings_miss",
        "acquisition",
        "divestiture",
        "management_change",
        "restructuring",
        "legal_or_regulatory",
    }


def test_neutral_prose_produces_no_events():
    text = (
        "Revenue was $1,000 million, an increase of 12% compared with the "
        "prior-year period, driven by higher unit volumes across our products."
    )

    assert extract_events([make_section(text)], as_of=AS_OF) == []


def test_rules_do_not_simply_match_the_word_increase():
    # "increase" on its own is not guidance; it needs a guidance noun nearby.
    text = "Operating expenses increased 9% year over year."

    assert "guidance_raise" not in types_of(text)


# --- Evidence and deduplication -------------------------------------------


def test_offsets_are_absolute_and_point_at_the_evidence():
    prefix = "Margin commentary. "
    text = prefix + "We raised our guidance for the year."
    section = make_section(text, start=5_000)

    event = extract_events([section], as_of=AS_OF)[0]

    absolute = event.evidence
    assert absolute.lower().startswith("raised our guidance")
    assert section.text[event.start - 5_000 : event.end - 5_000] == absolute


def test_repeated_mentions_collapse_to_one_event_per_section():
    text = "We raised our guidance. Later we raised our guidance again."

    events = extract_events([make_section(text)], as_of=AS_OF)

    assert [event.event_type for event in events] == ["guidance_raise"]


def test_different_types_in_the_same_section_are_reported_separately():
    text = "We raised our guidance and completed the acquisition of Acme Corp."

    events = extract_events([make_section(text)], as_of=AS_OF)

    assert {event.event_type for event in events} == {"guidance_raise", "acquisition"}


def test_the_same_type_in_two_sections_yields_two_events():
    sections = [
        make_section("We raised our guidance.", start=0, part="I", item="2"),
        make_section("We also raised our guidance.", start=100, part="II", item="1A"),
    ]

    events = extract_events(sections, as_of=AS_OF)

    assert [event.section_key for event in events] == [
        "part_i_item_2",
        "part_ii_item_1a",
    ]


def test_events_are_sorted_by_position():
    text = "We completed the acquisition of Acme. We also lowered our outlook."
    section = make_section(text)

    starts = [event.start for event in extract_events([section], as_of=AS_OF)]

    assert starts == sorted(starts)


def test_evidence_is_whitespace_collapsed():
    text = "We raised\n   our\tguidance for the year."
    section = make_section(text)

    event = extract_events([section], as_of=AS_OF)[0]

    assert "\n" not in event.evidence
    assert "  " not in event.evidence


# --- Point-in-time contract ------------------------------------------------


def test_naive_as_of_is_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        extract_events(
            [make_section("We raised our guidance.")],
            as_of=datetime(2026, 7, 26, 20, 4),
        )


def test_event_rejects_naive_as_of():
    with pytest.raises(ValueError, match="timezone-aware"):
        FilingEvent(
            event_type="guidance_raise",
            confidence=0.7,
            start=0,
            end=5,
            section_key="part_i_item_2",
            evidence="raised",
            as_of=datetime(2026, 7, 26, 20, 4),
        )


def test_event_rejects_confidence_outside_the_unit_interval():
    with pytest.raises(ValueError, match="confidence"):
        FilingEvent(
            event_type="guidance_raise",
            confidence=1.5,
            start=0,
            end=5,
            section_key="part_i_item_2",
            evidence="raised",
            as_of=AS_OF,
        )


def test_event_rejects_an_empty_span():
    with pytest.raises(ValueError, match="span"):
        FilingEvent(
            event_type="guidance_raise",
            confidence=0.7,
            start=5,
            end=5,
            section_key="part_i_item_2",
            evidence="raised",
            as_of=AS_OF,
        )


# --- Frame hand-off --------------------------------------------------------


def test_events_to_frame_has_the_expected_shape():
    events = extract_events([make_section("We raised our guidance.")], as_of=AS_OF)

    frame = events_to_frame(events, ticker="nvda", accession_number="0001-26-0001")

    assert list(frame.columns) == list(EVENT_COLUMNS)
    assert frame["ticker"].tolist() == ["NVDA"]
    assert frame["as_of"].dt.tz == UTC
    assert frame["evidence"].iloc[0].lower().startswith("raised our guidance")


def test_events_to_frame_of_nothing_is_empty_but_shaped():
    frame = events_to_frame([], ticker="NVDA")

    assert frame.empty
    assert list(frame.columns) == list(EVENT_COLUMNS)


def test_events_to_frame_requires_a_ticker():
    with pytest.raises(ValueError, match="ticker"):
        events_to_frame([], ticker="   ")
