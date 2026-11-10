"""Tests for :mod:`app.ingestion.store`.

Every test uses a temporary SQLite file, so the real ``data/index/filings.db``
is never touched.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from app.ingestion import store

FRAME_COLUMNS = [
    "accession_number",
    "form",
    "filing_date",
    "report_date",
    "acceptance_datetime",
    "primary_document",
    "url",
]


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    connection = store.connect(tmp_path / "filings.db")
    yield connection
    connection.close()


def make_row(
    accession: str = "0000000001-26-000001",
    *,
    form: str = "10-Q",
    filing_date: str = "2026-04-26",
    acceptance: str = "2026-04-26T20:05:00.000Z",
    report_date: str = "2026-04-26",
    document: str = "acme-20260426.htm",
) -> dict[str, str]:
    return {
        "accession_number": accession,
        "form": form,
        "filing_date": filing_date,
        "report_date": report_date,
        "acceptance_datetime": acceptance,
        "primary_document": document,
        "url": (
            "https://www.sec.gov/Archives/edgar/data/1/"
            f"{accession.replace('-', '')}/{document}"
        ),
    }


def make_filings(*rows: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=FRAME_COLUMNS)


# --- Writing ---------------------------------------------------------------


def test_upsert_inserts_rows(conn: sqlite3.Connection):
    written = store.upsert_filings(conn, "1045810", make_filings(make_row()))

    assert written == 1
    assert len(store.get_filings(conn)) == 1


def test_cik_is_stored_padded(conn: sqlite3.Connection):
    store.upsert_filings(conn, 1045810, make_filings(make_row()))

    assert store.get_filings(conn)["cik"].tolist() == ["0001045810"]


def test_upsert_is_idempotent(conn: sqlite3.Connection):
    filings = make_filings(make_row())
    store.upsert_filings(conn, "1045810", filings)
    store.upsert_filings(conn, "1045810", filings)

    assert len(store.get_filings(conn)) == 1


def test_upsert_updates_changed_fields(conn: sqlite3.Connection):
    store.upsert_filings(conn, "1045810", make_filings(make_row(form="10-Q")))
    store.upsert_filings(
        conn,
        "1045810",
        make_filings(make_row(form="10-K", filing_date="2026-01-31")),
    )

    row = store.get_filings(conn).iloc[0]
    assert row["form"] == "10-K"
    assert row["filing_date"] == "2026-01-31"


def test_missing_report_date_is_stored_as_null(conn: sqlite3.Connection):
    store.upsert_filings(conn, "1045810", make_filings(make_row(report_date="")))

    row = store.get_filings(conn).iloc[0]
    assert pd.isna(row["report_date"])


def test_upsert_rejects_a_frame_without_required_columns(conn: sqlite3.Connection):
    incomplete = pd.DataFrame({"form": ["10-Q"]})

    with pytest.raises(ValueError, match="missing required columns"):
        store.upsert_filings(conn, "1045810", incomplete)


# --- Local copies ----------------------------------------------------------


def test_mark_saved_records_the_path(conn: sqlite3.Connection, tmp_path: Path):
    store.upsert_filings(conn, "1045810", make_filings(make_row()))
    document = tmp_path / "acme-20260426.htm"
    document.write_text("filing body", encoding="utf-8")

    store.mark_saved(conn, "0000000001-26-000001", document)

    row = store.get_filings(conn).iloc[0]
    assert row["local_path"] == str(document.resolve())


def test_reupserting_keeps_the_saved_path(conn: sqlite3.Connection, tmp_path: Path):
    filings = make_filings(make_row())
    store.upsert_filings(conn, "1045810", filings)
    document = tmp_path / "acme-20260426.htm"
    store.mark_saved(conn, "0000000001-26-000001", document)

    store.upsert_filings(conn, "1045810", filings)

    row = store.get_filings(conn).iloc[0]
    assert row["local_path"] == str(document.resolve())


def test_mark_saved_rejects_an_unindexed_filing(
    conn: sqlite3.Connection, tmp_path: Path
):
    with pytest.raises(KeyError, match="not indexed"):
        store.mark_saved(conn, "0000000001-26-000099", tmp_path / "x.htm")


# --- Reading ---------------------------------------------------------------


def test_filings_are_ordered_by_acceptance_newest_first(conn: sqlite3.Connection):
    store.upsert_filings(
        conn,
        "1045810",
        make_filings(
            make_row("0000000001-26-000001", acceptance="2026-01-31T21:00:00.000Z"),
            make_row("0000000001-26-000002", acceptance="2026-10-25T21:00:00.000Z"),
            make_row("0000000001-26-000003", acceptance="2026-04-26T21:00:00.000Z"),
        ),
    )

    order = store.get_filings(conn)["accession_number"].tolist()

    assert order == [
        "0000000001-26-000002",
        "0000000001-26-000003",
        "0000000001-26-000001",
    ]


def test_filings_can_be_filtered_by_form(conn: sqlite3.Connection):
    store.upsert_filings(
        conn,
        "1045810",
        make_filings(
            make_row("0000000001-26-000001", form="10-Q"),
            make_row("0000000001-26-000002", form="10-K"),
        ),
    )

    forms = store.get_filings(conn, form="10-k")["form"].tolist()

    assert forms == ["10-K"]


def test_filings_can_be_filtered_by_cik(conn: sqlite3.Connection):
    store.upsert_filings(conn, "1045810", make_filings(make_row()))
    store.upsert_filings(
        conn,
        "789019",
        make_filings(make_row("0000000789-26-000001")),
    )

    frame = store.get_filings(conn, cik=789019)

    assert frame["cik"].tolist() == ["0000789019"]


def test_empty_index_returns_an_empty_frame_with_the_right_columns(
    conn: sqlite3.Connection,
):
    frame = store.get_filings(conn)

    assert frame.empty
    assert list(frame.columns) == list(store.COLUMNS)


def test_connect_is_safe_to_call_twice(tmp_path: Path):
    path = tmp_path / "filings.db"
    store.connect(path).close()
    store.connect(path).close()

    with sqlite3.connect(path) as raw:
        tables = raw.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()

    assert ("filings",) in tables


# --- Paths -----------------------------------------------------------------


def test_display_path_is_relative_inside_the_project():
    path = store.PROJECT_ROOT / "data" / "index"

    assert store.display_path(path) == "data/index"


def test_display_path_stays_absolute_outside_the_project(tmp_path: Path):
    assert store.display_path(tmp_path) == str(tmp_path.resolve())
