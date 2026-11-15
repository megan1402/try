"""Tests for the cross-issuer pieces: company mapping and corpus reading.

Everything runs offline — the "corpus" is a two-file temporary directory.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from app.ingestion import store
from app.quant.corpus_study import main as corpus_main
from app.quant.event_study import events_from_index, independence_counts

LISTING_COLUMNS = [
    "accession_number",
    "form",
    "filing_date",
    "report_date",
    "acceptance_datetime",
    "primary_document",
    "url",
]

FILING_HTML = """
<div style="font-weight:700">Part I. Financial Information</div>
<div style="font-weight:700">Item 2. Management's Discussion and Analysis</div>
<div>We raised our full-year guidance for the year and now expect revenue
growth across every segment in the second half of the fiscal year.</div>
"""


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    connection = store.connect(tmp_path / "index.db")
    yield connection
    connection.close()


def make_row(accession: str) -> dict[str, str]:
    return {
        "accession_number": accession,
        "form": "10-Q",
        "filing_date": "2026-04-26",
        "report_date": "2026-04-26",
        "acceptance_datetime": "2026-04-26T20:05:00.000Z",
        "primary_document": "f.htm",
        "url": "https://example.test/f.htm",
    }


def index_filing(
    conn: sqlite3.Connection,
    tmp_path: Path,
    accession: str,
    *,
    download: bool = True,
) -> Path:
    path = tmp_path / "raw" / "1045810" / accession.replace("-", "") / "f.htm"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(FILING_HTML, encoding="utf-8")

    store.upsert_filings(
        conn, "1045810", pd.DataFrame([make_row(accession)], columns=LISTING_COLUMNS)
    )
    if download:
        store.mark_saved(conn, accession, path)
    return path


# --- Company mapping -------------------------------------------------------


def test_upsert_company_normalises_and_round_trips(conn: sqlite3.Connection):
    store.upsert_company(conn, " nvda ", 1045810)

    companies = store.get_companies(conn)

    assert companies["ticker"].tolist() == ["NVDA"]
    assert companies["cik"].tolist() == ["0001045810"]
    assert list(companies.columns) == list(store.COMPANY_COLUMNS)


def test_upsert_company_is_idempotent(conn: sqlite3.Connection):
    store.upsert_company(conn, "NVDA", 1045810)
    store.upsert_company(conn, "NVDA", 1045810)

    assert len(store.get_companies(conn)) == 1


def test_companies_are_sorted_and_allow_share_classes(conn: sqlite3.Connection):
    store.upsert_company(conn, "GOOG", 1652044)
    store.upsert_company(conn, "GOOGL", 1652044)

    assert store.get_companies(conn)["ticker"].tolist() == ["GOOG", "GOOGL"]


def test_blank_ticker_is_rejected(conn: sqlite3.Connection):
    with pytest.raises(ValueError, match="ticker"):
        store.upsert_company(conn, "  ", 1045810)


def test_empty_index_has_no_companies(conn: sqlite3.Connection):
    assert store.get_companies(conn).empty


# --- Reading a corpus ------------------------------------------------------


def test_events_from_index_extracts_events_from_local_filings(
    conn: sqlite3.Connection, tmp_path: Path
):
    index_filing(conn, tmp_path, "acc-1")

    frame, used = events_from_index(conn, cik="1045810", ticker="NVDA")

    assert used == 1
    assert len(frame) == 1
    assert frame.loc[0, "event_type"] == "guidance_raise"
    assert frame.loc[0, "section_key"] == "part_i_item_2"
    assert frame.loc[0, "ticker"] == "NVDA"
    assert frame.loc[0, "accession_number"] == "acc-1"
    assert pd.notna(frame.loc[0, "as_of"])


def test_filings_without_a_download_are_skipped(
    conn: sqlite3.Connection, tmp_path: Path
):
    index_filing(conn, tmp_path, "acc-1")
    index_filing(conn, tmp_path, "acc-2", download=False)

    frame, used = events_from_index(conn, cik="1045810", ticker="NVDA")

    assert used == 1
    assert len(frame) == 1


def test_a_deleted_document_is_skipped(conn: sqlite3.Connection, tmp_path: Path):
    path = index_filing(conn, tmp_path, "acc-1")
    path.unlink()

    frame, used = events_from_index(conn, cik="1045810", ticker="NVDA")

    assert used == 0
    assert frame.empty


def test_unknown_cik_yields_nothing(conn: sqlite3.Connection):
    frame, used = events_from_index(conn, cik="9999999999", ticker="NONE")

    assert used == 0
    assert frame.empty


def test_events_from_several_filings_are_concatenated(
    conn: sqlite3.Connection, tmp_path: Path
):
    index_filing(conn, tmp_path, "acc-1")
    index_filing(conn, tmp_path, "acc-2")

    frame, used = events_from_index(conn, cik="1045810", ticker="NVDA")

    assert used == 2
    assert len(frame) == 2
    assert frame["accession_number"].nunique() == 2


# --- Independence ----------------------------------------------------------


def test_independence_counts_distinct_dates_not_events():
    joined = pd.DataFrame(
        {
            "event_type": ["acquisition"] * 4,
            "event_date": pd.to_datetime(
                ["2026-01-05", "2026-01-05", "2026-01-05", "2026-02-10"]
            ),
        }
    )

    dates, events = independence_counts(joined)

    assert dates == 2
    assert events == 4


def test_events_without_an_anchor_are_excluded():
    joined = pd.DataFrame(
        {
            "event_type": ["acquisition", "acquisition"],
            "event_date": pd.to_datetime(["2026-01-05", None]),
        }
    )

    assert independence_counts(joined) == (1, 1)


# --- Entry point -----------------------------------------------------------


def test_corpus_study_without_companies_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(store, "DEFAULT_INDEX_PATH", tmp_path / "empty.db")
    monkeypatch.setattr("sys.argv", ["corpus_study"])

    corpus_main()

    assert "No companies in the index" in capsys.readouterr().out
