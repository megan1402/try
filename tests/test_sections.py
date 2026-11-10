"""Tests for :mod:`app.extraction.sections`.

The 10-Q fixture exercises the whole pipeline (HTML -> text -> sections) on
realistic markup; inline text documents isolate the individual heuristics.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.extraction.html import html_to_text
from app.extraction.sections import Section, find_item, parse_sections

FIXTURES = Path(__file__).parent / "fixtures"

FIXTURE_TEXT = html_to_text((FIXTURES / "filing_10q.html").read_text(encoding="utf-8"))


def by_key(sections: list[Section]) -> dict[str, Section]:
    return {section.key: section for section in sections}


# --- The full 10-Q ---------------------------------------------------------


def test_finds_every_item_exactly_once():
    sections = parse_sections(FIXTURE_TEXT)

    assert [section.key for section in sections] == [
        "part_i_item_1",
        "part_i_item_2",
        "part_i_item_3",
        "part_i_item_4",
        "part_ii_item_1",
        "part_ii_item_1a",
        "part_ii_item_2",
        "part_ii_item_6",
    ]


def test_titles_are_cleaned_up():
    sections = by_key(parse_sections(FIXTURE_TEXT))

    assert sections["part_ii_item_1a"].title == "Risk Factors"
    assert sections["part_i_item_3"].title == (
        "Quantitative and Qualitative Disclosures About Market Risk"
    )


def test_table_of_contents_and_cover_page_are_not_sections():
    keys = {section.key for section in parse_sections(FIXTURE_TEXT)}

    # "Item 2." appears twice: once in the TOC (no page number) and once in the
    # body. Only the body survives.
    assert "part_i_item_2" in keys
    assert len(keys) == 8


def test_script_and_style_text_never_becomes_a_section():
    items = {section.item for section in parse_sections(FIXTURE_TEXT)}

    assert "98" not in items
    assert "99" not in items


def test_cross_reference_in_prose_does_not_start_a_section():
    section = find_item(parse_sections(FIXTURE_TEXT), "1", part="I")

    assert section is not None
    # The line "Item 1A of our Annual Report..." lives inside Item 1 and must
    # stay there instead of opening a new section.
    assert "Item 1A of our Annual Report" in section.text


def test_sections_are_contiguous_and_match_the_source():
    sections = parse_sections(FIXTURE_TEXT)

    for section in sections:
        assert section.text == FIXTURE_TEXT[section.start : section.end]

    for first, second in zip(sections, sections[1:], strict=False):
        assert first.end == second.start

    assert sections[-1].end == len(FIXTURE_TEXT)


def test_heading_is_the_first_line_and_body_follows():
    section = find_item(parse_sections(FIXTURE_TEXT), "1A", part="II")

    assert section is not None
    assert section.heading == "Item 1A. Risk Factors"
    assert "no material changes" in section.text


def test_section_stops_at_the_next_heading():
    section = find_item(parse_sections(FIXTURE_TEXT), "4", part="I")

    assert section is not None
    assert "disclosure controls and procedures" in section.text
    assert "Item 1. Legal Proceedings" not in section.text


def test_find_item_returns_none_for_unknown_item():
    sections = parse_sections(FIXTURE_TEXT)

    assert find_item(sections, "9") is None
    assert find_item(sections, "1", part="IV") is None


# --- Individual heuristics -------------------------------------------------


def test_bare_item_numbers_are_not_headings():
    text = "Item 1.\nItem 2.\n" + "x" * 200

    assert parse_sections(text) == []


def test_cross_reference_at_the_start_of_a_line_is_not_a_heading():
    text = (
        "Item 7. Management's Discussion and Analysis\n"
        + "x" * 200
        + "\nItem 1A of our Annual Report is incorporated by reference.\n"
    )

    assert [section.key for section in parse_sections(text)] == ["item_7"]


def test_prose_ending_in_a_colon_is_not_a_heading():
    text = "Item 8. Note the following:\n" + "x" * 200

    assert parse_sections(text) == []


def test_eight_k_decimal_items_are_recognised():
    text = (
        "Item 2.02 Results of Operations and Financial Condition\n" + "z" * 200 + "\n"
        "Item 9.01 Financial Statements and Exhibits\n" + "w" * 200
    )

    sections = parse_sections(text)

    assert [section.item for section in sections] == ["2.02", "9.01"]
    assert sections[0].title == "Results of Operations and Financial Condition"


def test_ten_k_items_without_a_part_heading_are_still_found():
    text = "Item 1. Business\n" + "b" * 200

    sections = parse_sections(text)

    assert [section.key for section in sections] == ["item_1"]
    assert sections[0].part is None


def test_min_body_chars_controls_short_sections():
    text = "Item 6. Exhibits\nsee below"

    assert parse_sections(text) == []
    assert parse_sections(text, min_body_chars=1)[0].key == "item_6"


def test_duplicate_item_keeps_the_occurrence_with_the_most_text():
    text = (
        "Item 1. Financial Statements\n"
        + "short"
        + "\nItem 1. Financial Statements\n"
        + "y" * 300
    )

    sections = parse_sections(text)

    assert len(sections) == 1
    assert len(sections[0].text) > 300


def test_non_string_input_is_rejected():
    with pytest.raises(TypeError):
        parse_sections(None)
