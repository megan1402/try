"""Local index of downloaded filings.

Retrieval and analysis stay separate everywhere in this project, and this module
is the seam between them. :mod:`app.ingestion.sec` *fetches* filings and knows
nothing about what they mean; extraction and (later) evaluation want to answer
questions such as *"which 10-Qs does NVDA have, and where is each one on disk?"*
without re-downloading anything.

The index is a single SQLite file under ``data/index/`` — one file, no server,
readable by any tool, and cheap enough to rebuild from scratch by deleting it.
It is deliberately **not** the final database: PostgreSQL + ``pgvector`` arrive
much later, once there are embeddings to store.

Point-in-time correctness starts here. Every row carries
``acceptance_datetime``, the moment EDGAR accepted the filing, which is what
later phases must compare against when deciding whether information existed at
some instant. ``filing_date`` is kept for display only.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.ingestion.sec import normalize_cik

logger = logging.getLogger(__name__)

#: Where the index lives. ``data/`` is git-ignored.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "index" / "filings.db"

#: Column order of the ``filings`` table, also used for query results.
COLUMNS: tuple[str, ...] = (
    "accession_number",
    "cik",
    "form",
    "filing_date",
    "report_date",
    "acceptance_datetime",
    "primary_document",
    "url",
    "local_path",
    "ingested_at",
)

#: Columns :func:`upsert_filings` needs from a :func:`list_filings` frame.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "accession_number",
    "form",
    "filing_date",
    "report_date",
    "acceptance_datetime",
    "primary_document",
    "url",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS filings (
    accession_number    TEXT PRIMARY KEY,
    cik                 TEXT NOT NULL,
    form                TEXT NOT NULL,
    filing_date         TEXT NOT NULL,
    report_date         TEXT,
    acceptance_datetime TEXT NOT NULL,
    primary_document    TEXT NOT NULL,
    url                 TEXT NOT NULL,
    local_path          TEXT,
    ingested_at         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_filings_cik_filing_date
    ON filings (cik, filing_date DESC);
"""

_UPSERT = """
INSERT INTO filings (
    accession_number, cik, form, filing_date, report_date,
    acceptance_datetime, primary_document, url, local_path, ingested_at
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (accession_number) DO UPDATE SET
    cik                 = excluded.cik,
    form                = excluded.form,
    filing_date         = excluded.filing_date,
    report_date         = excluded.report_date,
    acceptance_datetime = excluded.acceptance_datetime,
    primary_document    = excluded.primary_document,
    url                 = excluded.url,
    ingested_at         = excluded.ingested_at
"""


def display_path(path: Path) -> str:
    """Return ``path`` relative to the project root when it lives inside it.

    Paths are shown relative to the repository because that is how the README
    refers to them, but an index or document outside the tree keeps its
    absolute path rather than raising.
    """
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def connect(path: Path | None = None) -> sqlite3.Connection:
    """Open (creating if necessary) the index and ensure its schema.

    ``path`` defaults to :data:`DEFAULT_INDEX_PATH`, read at call time so the
    constant can be redirected in tests. The caller owns the connection and
    should close it; every write is committed inside the function that
    performs it.
    """
    index_path = DEFAULT_INDEX_PATH if path is None else path
    index_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(index_path)
    connection.row_factory = sqlite3.Row
    with connection:
        connection.executescript(_SCHEMA)
    return connection


def upsert_filings(
    conn: sqlite3.Connection,
    cik: str | int,
    filings: pd.DataFrame,
) -> int:
    """Insert or update the rows of ``filings``, returning how many were written.

    ``filings`` is normally the output of
    :func:`app.ingestion.sec.list_filings`. Re-running against the same filing
    is safe: ``accession_number`` is the primary key and the row is updated
    rather than duplicated. An existing ``local_path`` is preserved, because
    re-listing a company should not forget which documents are already on disk.

    Raises
    ------
    ValueError
        If ``filings`` is missing any of :data:`REQUIRED_COLUMNS`.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in filings.columns]
    if missing:
        raise ValueError(f"filings is missing required columns: {missing}.")

    rows = _rows(cik, filings)
    with conn:
        conn.executemany(_UPSERT, rows)
    return len(rows)


def mark_saved(
    conn: sqlite3.Connection,
    accession_number: str,
    path: Path,
) -> None:
    """Record that the document for ``accession_number`` is stored at ``path``.

    The path is stored relative to the project root when possible, so the index
    survives the repository being moved.

    A filing that is not in the index yet raises :class:`KeyError` — call
    :func:`upsert_filings` first.
    """
    stored = display_path(path)

    with conn:
        cursor = conn.execute(
            "UPDATE filings SET local_path = ? WHERE accession_number = ?",
            (stored, accession_number),
        )

    if cursor.rowcount == 0:
        raise KeyError(f"Filing {accession_number!r} is not indexed.")


def get_filings(
    conn: sqlite3.Connection,
    cik: str | int | None = None,
    form: str | None = None,
) -> pd.DataFrame:
    """Return indexed filings, newest acceptance first.

    Parameters
    ----------
    conn:
        Connection returned by :func:`connect`.
    cik:
        Restrict to one company. Accepts any CIK form; it is zero-padded.
    form:
        Restrict to one form type, e.g. ``"10-Q"``. Case-insensitive.
    """
    query = "SELECT * FROM filings"
    conditions: list[str] = []
    parameters: list[Any] = []

    if cik is not None:
        conditions.append("cik = ?")
        parameters.append(normalize_cik(cik))
    if form is not None:
        conditions.append("UPPER(form) = ?")
        parameters.append(form.strip().upper())

    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY acceptance_datetime DESC"

    rows = conn.execute(query, parameters).fetchall()
    return pd.DataFrame([tuple(row) for row in rows], columns=list(COLUMNS))


def _rows(cik: str | int, filings: pd.DataFrame) -> list[tuple[Any, ...]]:
    """Turn a filings frame into the positional tuples the UPSERT expects."""
    ingested_at = datetime.now(UTC).isoformat(timespec="seconds")
    padded_cik = normalize_cik(cik)

    return [
        (
            str(accession),
            padded_cik,
            str(form),
            str(filing_date),
            _optional_text(report_date),
            str(acceptance_datetime),
            str(primary_document),
            str(url),
            None,  # local_path: filled in later by mark_saved().
            ingested_at,
        )
        for (
            accession,
            form,
            filing_date,
            report_date,
            acceptance_datetime,
            primary_document,
            url,
        ) in zip(
            filings["accession_number"],
            filings["form"],
            filings["filing_date"],
            filings["report_date"],
            filings["acceptance_datetime"],
            filings["primary_document"],
            filings["url"],
            strict=True,
        )
    ]


def _optional_text(value: object) -> str | None:
    """Return ``value`` as trimmed text, or ``None`` when it is missing."""
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None

    text = str(value).strip()
    return text or None


def main() -> None:
    """Manual smoke test: ``python -m app.ingestion.store``."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    if not DEFAULT_INDEX_PATH.exists():
        print(
            f"No index at {display_path(DEFAULT_INDEX_PATH)} yet. "
            "Run `make run-sec` to fetch a filing into it."
        )
        return

    conn = connect()
    try:
        filings = get_filings(conn)
    finally:
        conn.close()

    print(f"\n{display_path(DEFAULT_INDEX_PATH)}: {len(filings)} filing(s)")

    if filings.empty:
        return

    print(
        f"{filings['cik'].nunique()} company/companies, "
        f"forms {sorted(filings['form'].unique())}\n"
    )

    columns = ["cik", "form", "filing_date", "acceptance_datetime", "local_path"]
    print(filings[columns].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
