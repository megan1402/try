"""Tests for exhibit discovery in :mod:`app.ingestion.sec`.

The index page is the only way to find an 8-K's press release, so its parsing is
tested against a saved excerpt of a real index rather than a live request.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion import sec

FIXTURES = Path(__file__).parent / "fixtures"
INDEX_HTML = (FIXTURES / "filing_index_8k.html").read_text(encoding="utf-8")


# --- URLs ------------------------------------------------------------------


def test_filing_index_url_uses_the_padded_cik_and_plain_accession():
    url = sec.filing_index_url(1045810, "0001045810-26-000088")

    assert url == (
        "https://www.sec.gov/Archives/edgar/data/1045810/"
        "000104581026000088/0001045810-26-000088-index.htm"
    )


def test_filing_index_url_rejects_an_empty_accession():
    with pytest.raises(ValueError, match="accession_number"):
        sec.filing_index_url(1045810, "  ")


# --- Parsing the index -----------------------------------------------------


def test_parse_filing_index_finds_the_documents():
    documents = sec.parse_filing_index(INDEX_HTML)

    assert not documents.empty
    assert list(documents.columns) == [
        "sequence",
        "document_type",
        "filename",
        "description",
    ]


def test_parse_filing_index_finds_the_press_release():
    documents = sec.parse_filing_index(INDEX_HTML)

    press_release = documents[documents["document_type"] == "EX-99.1"]

    assert len(press_release) == 1
    assert press_release.iloc[0]["filename"] == "q3fy27pr.htm"


def test_parse_filing_index_keeps_the_primary_document():
    documents = sec.parse_filing_index(INDEX_HTML)

    assert "nvda-20261117.htm" in documents["filename"].tolist()


def test_parse_filing_index_drops_xbrl_and_graphics():
    documents = sec.parse_filing_index(INDEX_HTML)

    types = documents["document_type"].str.upper().tolist()

    assert not any(document_type.startswith("EX-101") for document_type in types)
    assert "GRAPHIC" not in types
    assert "XML" not in types


def test_parse_filing_index_drops_the_complete_submission_row():
    documents = sec.parse_filing_index(INDEX_HTML)

    assert not documents["filename"].str.endswith(".txt").any()


def test_parse_filing_index_ignores_rows_without_a_filename():
    html = "<table><tr><td>1</td><td>8-K</td><td></td><td>8-K</td></tr></table>"

    assert sec.parse_filing_index(html).empty


def test_parse_filing_index_of_empty_html_is_empty():
    assert sec.parse_filing_index("").empty


def test_parse_filing_index_rejects_non_strings():
    with pytest.raises(TypeError, match="html must be a string"):
        sec.parse_filing_index(None)


# --- Selecting exhibits ----------------------------------------------------


def test_list_exhibits_builds_absolute_urls(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sec, "download_document", lambda url: INDEX_HTML)

    exhibits = sec.list_exhibits(1045810, "0001045810-26-000088")

    assert exhibits["filename"].tolist() == ["q3fy27pr.htm"]
    assert exhibits["url"].tolist() == [
        "https://www.sec.gov/Archives/edgar/data/1045810/"
        "000104581026000088/q3fy27pr.htm"
    ]


def test_list_exhibits_can_select_several_types(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sec, "download_document", lambda url: INDEX_HTML)

    exhibits = sec.list_exhibits(
        1045810, "0001045810-26-000088", document_types=("EX-99.1", "EX-99.2")
    )

    assert set(exhibits["document_type"]) == {"EX-99.1", "EX-99.2"}


def test_list_exhibits_returns_nothing_when_the_type_is_absent(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(sec, "download_document", lambda url: INDEX_HTML)

    exhibits = sec.list_exhibits(
        1045810, "0001045810-26-000088", document_types=("EX-10.1",)
    )

    assert exhibits.empty
    assert "url" in exhibits.columns
