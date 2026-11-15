"""Bulk, resumable collection of filings into the local corpus.

One filing is a demo; a corpus is a dataset. This module turns the single-filing
path from :mod:`app.ingestion.sec` into a loop that can be interrupted and
restarted without wasting requests:

* every listed filing is recorded in the index first, so a crash leaves a
  catalogue rather than a mystery;
* documents already on disk are skipped, so re-running resumes instead of
  re-downloading;
* requests are throttled to stay well inside SEC's 10-per-second limit — the
  limit applies to *every* caller of EDGAR, and tripping it earns a block, not
  just a 429;
* a single failed document is counted and reported, never fatal.

The result is a local mirror that later phases (event study, embeddings) can
read without touching the network.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.ingestion import store
from app.ingestion.sec import (
    DEFAULT_FORMS,
    DEFAULT_RAW_DIR,
    SecDataError,
    document_path,
    download_document,
    get_cik,
    list_filings,
    save_document,
)

logger = logging.getLogger(__name__)

#: Requests per second. SEC asks for at most 10; half of that is polite and
#: leaves room for clock drift in the sleep.
DEFAULT_REQUESTS_PER_SECOND = 5.0


@dataclass(frozen=True)
class CollectionSummary:
    """Per-company outcome of a collection run."""

    ticker: str
    cik: str
    listed: int
    downloaded: int
    skipped: int
    failed: int

    @property
    def total(self) -> int:
        return self.downloaded + self.skipped


def throttle(requests_per_second: float) -> Callable[[], None]:
    """Return a callable that sleeps long enough to respect the rate limit.

    Raises
    ------
    ValueError
        If ``requests_per_second`` is not positive.
    """
    if requests_per_second <= 0:
        raise ValueError("requests_per_second must be positive.")

    minimum_interval = 1.0 / requests_per_second

    def pause() -> None:
        time.sleep(minimum_interval)

    return pause


def collect_company(
    ticker: str,
    *,
    conn: sqlite3.Connection,
    forms: Sequence[str] = DEFAULT_FORMS,
    limit: int | None = None,
    pause: Callable[[], None] | None = None,
    root: Path = DEFAULT_RAW_DIR,
    fetch_listing: Callable[[str | int, Sequence[str]], pd.DataFrame] | None = None,
    fetch_document: Callable[[str], str] | None = None,
    resolve_cik: Callable[[str], str] | None = None,
) -> CollectionSummary:
    """Download every listed filing for ``ticker`` that is not already on disk.

    Parameters
    ----------
    ticker:
        Symbol, e.g. ``"NVDA"``.
    conn:
        Index connection from :func:`app.ingestion.store.connect`.
    forms:
        Form types to collect.
    limit:
        Maximum number of documents to download for this company. Listings are
        visited newest first, so a small limit gives the most recent filings.
    pause:
        Called before every network request; defaults to no throttling so tests
        do not sleep. :func:`throttle` builds the production one.
    root:
        Directory the documents are written under.
    fetch_listing, fetch_document, resolve_cik:
        Injected for tests; they default to the real EDGAR functions.

    Returns
    -------
    CollectionSummary
        Counts of what was listed, downloaded, skipped and failed.

    Raises
    ------
    SecDataError
        If the ticker cannot be resolved or its listing cannot be retrieved.
        Per-document failures are counted instead of raised.
    """
    if limit is not None and limit < 0:
        raise ValueError("limit must be zero or greater.")

    sleep = pause or _no_pause
    listing_fn = fetch_listing or list_filings
    document_fn = fetch_document or download_document
    cik_fn = resolve_cik or get_cik

    cik = cik_fn(ticker)

    sleep()
    listed = listing_fn(cik, forms)
    store.upsert_filings(conn, cik, listed)

    downloaded = 0
    skipped = 0
    failed = 0

    for row in listed.itertuples(index=False):
        if limit is not None and downloaded >= limit:
            break

        accession = str(row.accession_number)
        document = str(row.primary_document)
        path = document_path(cik, accession, document, root)

        if path.exists():
            store.mark_saved(conn, accession, path)
            skipped += 1
            continue

        try:
            sleep()
            text = document_fn(str(row.url))
        except SecDataError as exc:
            logger.warning("%s %s: %s", ticker, accession, exc)
            failed += 1
            continue

        saved = save_document(text, cik, accession, document, root)
        store.mark_saved(conn, accession, saved)
        downloaded += 1

    return CollectionSummary(
        ticker=ticker.strip().upper(),
        cik=cik,
        listed=len(listed),
        downloaded=downloaded,
        skipped=skipped,
        failed=failed,
    )


def collect(
    tickers: Sequence[str],
    *,
    index_path: Path | None = None,
    forms: Sequence[str] = DEFAULT_FORMS,
    limit: int | None = None,
    requests_per_second: float = DEFAULT_REQUESTS_PER_SECOND,
    root: Path = DEFAULT_RAW_DIR,
) -> list[CollectionSummary]:
    """Collect several companies into one index, returning their summaries."""
    conn = store.connect(index_path)
    sleep = throttle(requests_per_second)
    summaries: list[CollectionSummary] = []

    try:
        for ticker in tickers:
            logger.info("collecting %s", ticker)
            try:
                summary = collect_company(
                    ticker,
                    conn=conn,
                    forms=forms,
                    limit=limit,
                    pause=sleep,
                    root=root,
                )
            except SecDataError as exc:
                logger.error("%s: %s", ticker, exc)
                continue
            logger.info(
                "%s: %d listed, %d downloaded, %d skipped, %d failed",
                summary.ticker,
                summary.listed,
                summary.downloaded,
                summary.skipped,
                summary.failed,
            )
            summaries.append(summary)
    finally:
        conn.close()

    return summaries


def _no_pause() -> None:
    """Default no-op throttle; tests must not sleep."""


def main() -> None:
    """Manual entry point: ``python -m app.ingestion.collect NVDA MSFT AAPL``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="+", help="one or more symbols")
    parser.add_argument(
        "--forms", nargs="+", default=list(DEFAULT_FORMS), help="form types"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="max downloads per company"
    )
    parser.add_argument(
        "--rps",
        type=float,
        default=DEFAULT_REQUESTS_PER_SECOND,
        help="requests per second (SEC allows at most 10)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    summaries = collect(
        args.tickers,
        forms=tuple(args.forms),
        limit=args.limit,
        requests_per_second=args.rps,
    )

    print()
    print(f"{'ticker':8} {'cik':12} {'listed':>7} {'new':>5} {'skip':>5} {'fail':>5}")
    for summary in summaries:
        print(
            f"{summary.ticker:8} {summary.cik:12} {summary.listed:7} "
            f"{summary.downloaded:5} {summary.skipped:5} {summary.failed:5}"
        )

    total_new = sum(summary.downloaded for summary in summaries)
    total_failed = sum(summary.failed for summary in summaries)
    print(f"\n{total_new} document(s) downloaded, {total_failed} failed.")


if __name__ == "__main__":
    main()
