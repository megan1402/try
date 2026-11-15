"""Tests for :mod:`app.ingestion.collect`.

No network: the listing and the documents are injected, and files are written
into a temporary root.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from app.ingestion import collect as collector
from app.ingestion import store
from app.ingestion.sec import SecDataError

LISTING_COLUMNS = [
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
    connection = store.connect(tmp_path / "index.db")
    yield connection
    connection.close()


def make_listing(accessions: list[str]) -> pd.DataFrame:
    rows = [
        {
            "accession_number": accession,
            "form": "10-Q",
            "filing_date": "2026-04-26",
            "report_date": "2026-04-26",
            "acceptance_datetime": "2026-04-26T20:05:00.000Z",
            "primary_document": f"{accession}.htm",
            "url": f"https://example.test/{accession}.htm",
        }
        for accession in accessions
    ]
    return pd.DataFrame(rows, columns=LISTING_COLUMNS)


def fake_fetcher(calls: list[str], *, failing: set[str] | None = None):
    def fetch(url: str) -> str:
        calls.append(url)
        if failing and any(name in url for name in failing):
            raise SecDataError(f"boom: {url}")
        return f"document body for {url}"

    return fetch


def run(
    conn: sqlite3.Connection,
    tmp_path: Path,
    *,
    accessions: list[str] | None = None,
    limit: int | None = None,
    failing: set[str] | None = None,
    pause=None,
):
    accessions = accessions or ["acc-1", "acc-2"]
    calls: list[str] = []
    pauses: list[int] = []

    def counted_pause() -> None:
        pauses.append(1)

    summary = collector.collect_company(
        "NVDA",
        conn=conn,
        limit=limit,
        pause=pause if pause is not None else counted_pause,
        root=tmp_path / "raw",
        fetch_listing=lambda cik, forms: make_listing(accessions),
        fetch_document=fake_fetcher(calls, failing=failing),
        resolve_cik=lambda ticker: "0001045810",
    )
    return summary, calls, pauses


# --- Downloading -----------------------------------------------------------


def test_downloads_new_filings(conn: sqlite3.Connection, tmp_path: Path):
    summary, calls, _ = run(conn, tmp_path)

    assert summary.downloaded == 2
    assert summary.failed == 0
    assert len(calls) == 2
    assert (tmp_path / "raw" / "1045810" / "acc1" / "acc-1.htm").exists()


def test_listed_filings_are_indexed_even_before_download(
    conn: sqlite3.Connection, tmp_path: Path
):
    run(conn, tmp_path)

    indexed = store.get_filings(conn)

    assert len(indexed) == 2
    assert indexed["local_path"].notna().all()
    assert indexed["cik"].unique().tolist() == ["0001045810"]


def test_second_run_skips_documents_already_on_disk(
    conn: sqlite3.Connection, tmp_path: Path
):
    run(conn, tmp_path)
    summary, calls, _ = run(conn, tmp_path)

    assert summary.downloaded == 0
    assert summary.skipped == 2
    assert calls == []


def test_limit_caps_the_number_of_downloads(conn: sqlite3.Connection, tmp_path: Path):
    summary, calls, _ = run(conn, tmp_path, accessions=["a1", "a2", "a3"], limit=1)

    assert summary.downloaded == 1
    assert len(calls) == 1
    assert summary.listed == 3


def test_listing_is_recorded_even_when_the_limit_is_zero(
    conn: sqlite3.Connection, tmp_path: Path
):
    summary, calls, _ = run(conn, tmp_path, limit=0)

    assert summary.downloaded == 0
    assert calls == []
    assert len(store.get_filings(conn)) == 2


# --- Failures --------------------------------------------------------------


def test_a_failed_document_is_counted_and_does_not_stop_the_run(
    conn: sqlite3.Connection, tmp_path: Path
):
    summary, calls, _ = run(conn, tmp_path, failing={"acc-1"})

    assert summary.failed == 1
    assert summary.downloaded == 1
    assert len(calls) == 2  # the second document was still attempted.


def test_unresolvable_ticker_raises(conn: sqlite3.Connection, tmp_path: Path):
    def boom(ticker: str) -> str:
        raise SecDataError("ticker not found")

    with pytest.raises(SecDataError):
        collector.collect_company(
            "NOPE",
            conn=conn,
            root=tmp_path / "raw",
            fetch_listing=lambda cik, forms: make_listing([]),
            resolve_cik=boom,
        )


def test_negative_limit_is_rejected(conn: sqlite3.Connection, tmp_path: Path):
    with pytest.raises(ValueError, match="limit"):
        run(conn, tmp_path, limit=-1)


# --- Rate limiting ---------------------------------------------------------


def test_every_network_request_is_paced(conn: sqlite3.Connection, tmp_path: Path):
    _, _, pauses = run(conn, tmp_path)

    # One pause before the listing plus one per document.
    assert len(pauses) == 3


def test_throttle_returns_a_callable(monkeypatch: pytest.MonkeyPatch):
    slept: list[float] = []
    monkeypatch.setattr(collector.time, "sleep", slept.append)

    pause = collector.throttle(4.0)
    pause()

    assert slept == [pytest.approx(0.25)]


def test_throttle_rejects_a_non_positive_rate():
    with pytest.raises(ValueError, match="positive"):
        collector.throttle(0)


# --- Several companies -----------------------------------------------------


def test_collect_walks_every_ticker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []

    def fake_collect_company(ticker: str, **kwargs) -> collector.CollectionSummary:
        calls.append(ticker)
        return collector.CollectionSummary(
            ticker=ticker.upper(), cik="0", listed=0, downloaded=0, skipped=0, failed=0
        )

    monkeypatch.setattr(collector, "collect_company", fake_collect_company)

    summaries = collector.collect(
        ["nvda", "msft"],
        index_path=tmp_path / "index.db",
        requests_per_second=1000,
        root=tmp_path / "raw",
    )

    assert calls == ["nvda", "msft"]
    assert [summary.ticker for summary in summaries] == ["NVDA", "MSFT"]


def test_collect_survives_a_missing_ticker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def boom(ticker: str) -> str:
        raise SecDataError(f"no CIK for {ticker}")

    monkeypatch.setattr(collector, "get_cik", boom)

    summaries = collector.collect(
        ["NOPE"],
        index_path=tmp_path / "index.db",
        forms=("10-Q",),
        limit=0,
        requests_per_second=1000,
        root=tmp_path / "raw",
    )

    assert summaries == []
