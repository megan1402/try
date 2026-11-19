"""Tests for :mod:`app.extraction.precision`.

Each filter is tested against the exact phrasing that motivated it, taken from
real filings in the corpus. The point of these tests is that the *false
positives* stay rejected — a regression here silently inflates every event
count downstream.
"""

from __future__ import annotations

import pytest

from app.extraction.precision import (
    ACCOUNTING_NOISE,
    HYPOTHETICAL_MARKERS,
    SUPPRESSED_SECTIONS,
    has_event_structure,
    is_accounting_noise,
    is_hypothetical,
    is_suppressed_section,
    sentence_around,
    should_keep,
)

# --- Section suppression ---------------------------------------------------


@pytest.mark.parametrize(
    "section_key",
    ["part_i_item_1", "item_1", "part_ii_item_1a", "item_1a", "part_ii_item_6"],
)
def test_financial_and_risk_sections_are_suppressed(section_key: str):
    assert is_suppressed_section(section_key, "acquisition")


def test_mda_is_not_suppressed():
    assert not is_suppressed_section("part_i_item_2", "acquisition")


def test_legal_events_survive_in_suppressed_sections():
    # A settlement reported in the notes is still a real event.
    assert not is_suppressed_section("part_i_item_1", "legal_or_regulatory")


def test_risk_factors_suppress_every_type_including_legal():
    # Item 1A is hypothetical by construction, so nothing there is an event.
    assert is_suppressed_section("part_ii_item_1a", "legal_or_regulatory")
    assert is_suppressed_section("item_1a", "legal_or_regulatory")


def test_suppression_is_case_insensitive():
    assert is_suppressed_section("PART_I_ITEM_1", "acquisition")


def test_suppressed_sections_are_documented():
    assert "part_i_item_1" in SUPPRESSED_SECTIONS
    assert "part_ii_item_1a" in SUPPRESSED_SECTIONS


# --- Hypothetical language -------------------------------------------------


@pytest.mark.parametrize(
    "sentence",
    [
        "An acquisition could harm our business.",
        "We may acquire complementary businesses in the future.",
        "If we fail to integrate an acquisition, our results might suffer.",
        "There is a risk that litigation could arise.",
        "We anticipate potential acquisitions.",
        "No assurance can be given that we will complete any acquisition.",
        "From time to time we are subject to litigation.",
    ],
)
def test_risk_factor_language_is_hypothetical(sentence: str):
    assert is_hypothetical(sentence)


@pytest.mark.parametrize(
    "sentence",
    [
        "We completed the acquisition of Acme Corp. in the quarter.",
        "The company acquired Acme Corp. for $2.0 billion.",
        "Our Chief Financial Officer resigned effective immediately.",
    ],
)
def test_reported_events_are_not_hypothetical(sentence: str):
    assert not is_hypothetical(sentence)


def test_hypothetical_markers_are_documented():
    assert "could" in HYPOTHETICAL_MARKERS


# --- Accounting vocabulary -------------------------------------------------


@pytest.mark.parametrize(
    "evidence",
    [
        "property and equipment acquired",
        "intangible assets acquired",
        "accumulated depreciation",
        "identified for disposition",
        "stock-based compensation",
        "right-of-use",
    ],
)
def test_accounting_phrases_are_noise(evidence: str):
    assert is_accounting_noise(evidence)


@pytest.mark.parametrize(
    "evidence",
    ["acquisition of Acme Corp", "divested our networking business"],
)
def test_real_event_phrases_are_not_noise(evidence: str):
    assert not is_accounting_noise(evidence)


def test_accounting_noise_list_is_documented():
    assert "property and equipment" in ACCOUNTING_NOISE


@pytest.mark.parametrize(
    "sentence",
    [
        "We will need to accurately forecast, purchase and warehouse components.",
        "We must forecast demand accurately to avoid excess inventory.",
    ],
)
def test_forecasting_as_an_activity_is_not_guidance(sentence: str):
    assert is_accounting_noise("forecast", sentence)


def test_a_real_guidance_sentence_is_not_noise():
    assert not is_accounting_noise(
        "guidance", "We raised our full-year guidance for fiscal 2027."
    )


# --- Subject and object ----------------------------------------------------


@pytest.mark.parametrize(
    "sentence",
    [
        "We completed the acquisition of Acme Corp.",
        "The company acquired Acme Corp. for $2.0 billion.",
        "We entered into a merger agreement with Acme.",
        "The acquisition of Acme closed in the quarter.",
    ],
)
def test_acquisition_with_an_object_passes(sentence: str):
    assert has_event_structure("acquisition", sentence)


@pytest.mark.parametrize(
    "sentence",
    [
        "Our acquisition strategy focuses on tuck-in deals.",
        "Acquisition-related expenses were $12 million.",
        "The acquisition was accounted for using the purchase method.",
    ],
)
def test_bare_acquisition_noun_fails(sentence: str):
    assert not has_event_structure("acquisition", sentence)


@pytest.mark.parametrize(
    "sentence",
    [
        "We divested our networking business in the quarter.",
        "We completed the sale of our data centre division.",
    ],
)
def test_divestiture_with_an_object_passes(sentence: str):
    assert has_event_structure("divestiture", sentence)


def test_bare_disposition_fails():
    assert not has_event_structure("divestiture", "Assets held for disposition.")


@pytest.mark.parametrize(
    "sentence",
    [
        "Our Chief Financial Officer resigned effective immediately.",
        "The Board appointed Jane Doe as Chief Executive Officer.",
    ],
)
def test_management_change_with_a_role_passes(sentence: str):
    assert has_event_structure("management_change", sentence)


def test_bare_retirement_fails():
    assert not has_event_structure(
        "management_change", "Once an asset is identified for retirement."
    )


@pytest.mark.parametrize(
    "sentence",
    [
        "We announced a restructuring plan in the quarter.",
        "The company eliminated approximately 1,200 positions.",
    ],
)
def test_restructuring_with_an_action_passes(sentence: str):
    assert has_event_structure("restructuring", sentence)


def test_bare_restructuring_noun_fails():
    assert not has_event_structure(
        "restructuring", "Restructuring charges are included in operating expenses."
    )


def test_unstructured_types_are_not_checked():
    # Guidance and earnings vocabulary is specific enough on its own.
    assert has_event_structure("guidance_raise", "anything at all")
    assert has_event_structure("earnings_beat", "anything at all")


@pytest.mark.parametrize(
    "sentence",
    [
        "We recorded a settlement of $50 million in the quarter.",
        "The company received a subpoena from the regulator.",
        "We are subject to an investigation by the SEC.",
        "A shareholder filed a lawsuit against the company.",
    ],
)
def test_reported_legal_events_pass(sentence: str):
    assert has_event_structure("legal_or_regulatory", sentence)


@pytest.mark.parametrize(
    "sentence",
    [
        "Litigation is described in Note 12.",
        "We may be subject to litigation in the ordinary course.",
        "Legal proceedings are discussed below.",
    ],
)
def test_bare_legal_nouns_fail(sentence: str):
    assert not has_event_structure("legal_or_regulatory", sentence)


# --- Sentence extraction ---------------------------------------------------


def test_sentence_around_returns_the_containing_sentence():
    text = "First sentence. We acquired Acme Corp. Third sentence."

    sentence = sentence_around(text, text.index("acquired"), text.index("Acme") + 4)

    assert "acquired Acme Corp" in sentence
    assert "First sentence" not in sentence


def test_sentence_around_handles_newlines():
    text = "Header line\nWe acquired Acme Corp.\nNext line"

    sentence = sentence_around(text, text.index("acquired"), text.index("Acme") + 4)

    assert "acquired Acme Corp" in sentence
    assert "Header" not in sentence


def test_sentence_around_at_the_start_of_the_text():
    text = "We acquired Acme Corp. Then more."

    assert sentence_around(text, 3, 11).startswith("We acquired")


def test_sentence_around_at_the_end_of_the_text():
    text = "Intro. We acquired Acme Corp"

    assert "acquired Acme Corp" in sentence_around(text, 10, 20)


# --- The combined decision -------------------------------------------------


def test_a_real_event_in_mda_is_kept():
    assert should_keep(
        event_type="acquisition",
        section_key="part_i_item_2",
        sentence="We completed the acquisition of Acme Corp. in the quarter.",
        evidence="acquisition of Acme",
    )


def test_the_same_event_in_financial_statements_is_dropped():
    assert not should_keep(
        event_type="acquisition",
        section_key="part_i_item_1",
        sentence="We completed the acquisition of Acme Corp. in the quarter.",
        evidence="acquisition of Acme",
    )


def test_a_hypothetical_in_mda_is_dropped():
    assert not should_keep(
        event_type="acquisition",
        section_key="part_i_item_2",
        sentence="An acquisition could harm our business.",
        evidence="acquisition",
    )


def test_accounting_noise_in_mda_is_dropped():
    assert not should_keep(
        event_type="acquisition",
        section_key="part_i_item_2",
        sentence="Property and equipment acquired was $1.2 billion.",
        evidence="acquired",
    )


def test_a_settlement_in_the_notes_is_kept():
    assert should_keep(
        event_type="legal_or_regulatory",
        section_key="part_i_item_1",
        sentence="We recorded a settlement of $50 million in the quarter.",
        evidence="settlement",
    )
