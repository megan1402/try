"""SEC EDGAR ingestion.

EDGAR is the SEC's public filing system. Every US-registered company has a
**CIK** (Central Index Key), and every filing has an **accession number**.
Three public JSON endpoints are used — no API key, no authentication:

1. ``company_tickers.json`` — ticker -> CIK lookup.
2. ``submissions/CIK##########.json`` — a company's recent filings.
3. ``www.sec.gov/Archives/...`` — the filing documents themselves.

Two real-world rules apply:

* SEC requires a ``User-Agent`` header that identifies you with a contact
  address. Set it before running::

      export SEC_USER_AGENT="Your Name your@email.com"

* SEC asks for at most **10 requests per second**. This module makes one
  request per call and does not throttle; keep call volume low.

This module retrieves and stores *raw* filings. Extracting structured events
from them is a later phase — as always, retrieval and analysis stay separate.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# --- Endpoints -------------------------------------------------------------

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"

REQUEST_TIMEOUT_SECONDS = 30

#: Filing types retrieved by default.
DEFAULT_FORMS: tuple[str, ...] = ("10-K", "10-Q")

#: Where downloaded documents are written. ``data/`` is git-ignored.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "sec"

# EDGAR returns camelCase keys; we rename them to snake_case columns.
_EDGAR_TO_SNAKE: dict[str, str] = {
    "form": "form",
    "filingDate": "filing_date",
    "reportDate": "report_date",
    "acceptanceDateTime": "acceptance_datetime",
    "accessionNumber": "accession_number",
    "primaryDocument": "primary_document",
}

#: Keys that must be present for a filing list to be usable.
_REQUIRED_EDGAR_KEYS: tuple[str, ...] = (
    "form",
    "filingDate",
    "accessionNumber",
    "primaryDocument",
)


class SecDataError(RuntimeError):
    """Raised when EDGAR data cannot be retrieved or is not shaped as expected."""


# --- Small pure helpers ----------------------------------------------------


def normalize_cik(cik: str | int) -> str:
    """Return ``cik`` padded to the 10-digit form EDGAR uses in URLs.

    >>> normalize_cik(1045810)
    '0001045810'
    """
    text = str(cik).strip()
    if not text.isdigit():
        raise ValueError(f"CIK must be numeric, got {cik!r}.")
    return text.zfill(10)


def submissions_url(cik: str | int) -> str:
    """Return the submissions endpoint for ``cik``."""
    return f"{SUBMISSIONS_URL}/CIK{normalize_cik(cik)}.json"


def filing_url(cik: str | int, accession_number: str, document: str) -> str:
    """Build the download URL for a single filing document.

    Note the two different CIK formats: the submissions *API* uses the padded
    CIK, while the Archives *file path* uses the unpadded one.
    """
    if not accession_number.strip():
        raise ValueError("accession_number must be a non-empty string.")
    if not document.strip():
        raise ValueError("document must be a non-empty string.")

    cik_path = str(int(normalize_cik(cik)))  # Archives uses the plain number.
    accession_path = accession_number.replace("-", "")
    return f"{ARCHIVES_URL}/{cik_path}/{accession_path}/{document}"


# --- HTTP layer ------------------------------------------------------------


def _headers() -> dict[str, str]:
    """Build request headers, including the SEC-required User-Agent."""
    user_agent = os.environ.get(
        "SEC_USER_AGENT",
        "SignalForge/0.1 (educational project; set SEC_USER_AGENT to your email)",
    )
    return {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}


def _fetch_json(url: str) -> Any:
    """GET ``url`` and parse the body as JSON, wrapping every failure mode."""
    try:
        response = requests.get(
            url, headers=_headers(), timeout=REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SecDataError(f"Request to {url} failed: {exc}") from exc

    try:
        return response.json()
    except ValueError as exc:
        raise SecDataError(f"Response from {url} was not valid JSON.") from exc


# --- Public API ------------------------------------------------------------


def get_cik(ticker: str) -> str:
    """Resolve a ticker symbol to its 10-digit CIK.

    Parameters
    ----------
    ticker:
        Stock symbol, e.g. ``"NVDA"``. Case-insensitive; trimmed.

    Returns
    -------
    str
        The zero-padded 10-digit CIK, e.g. ``"0001045810"``.

    Raises
    ------
    ValueError
        If ``ticker`` is not a non-empty string.
    SecDataError
        If the ticker map cannot be retrieved, or the ticker is not listed.
    """
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("ticker must be a non-empty string, e.g. 'NVDA'.")

    symbol = ticker.strip().upper()
    payload = _fetch_json(TICKER_MAP_URL)

    if not isinstance(payload, dict):
        raise SecDataError("Unexpected ticker-map payload from EDGAR.")

    for entry in payload.values():
        if isinstance(entry, dict) and str(entry.get("ticker", "")).upper() == symbol:
            return normalize_cik(entry["cik_str"])

    raise SecDataError(
        f"Ticker {symbol!r} was not found in EDGAR's ticker map. "
        "EDGAR only lists US-registered companies."
    )


def list_filings(
    cik: str | int,
    forms: Sequence[str] = DEFAULT_FORMS,
) -> pd.DataFrame:
    """List a company's filings of the requested ``forms``, newest first.

    Returns one row per filing with the columns ``form``, ``filing_date``,
    ``report_date``, ``acceptance_datetime``, ``accession_number``,
    ``primary_document`` and ``url``.

    ``acceptance_datetime`` is the precise moment EDGAR accepted the filing.
    Use it — not ``filing_date`` — when you later need to decide whether
    information existed *at* a given moment, because filings published after
    the market closes take effect the next day.

    Limitations
    -----------
    Only ``filings.recent`` is read, which covers roughly the most recent 1000
    filings. For 10-K/10-Q that is many years of history, but a company with
    very heavy filing activity could have older records elsewhere.

    Raises
    ------
    ValueError
        If ``cik`` is not numeric.
    SecDataError
        If the submissions payload cannot be retrieved or lacks expected keys.
    """
    payload = _fetch_json(submissions_url(cik))

    try:
        recent = payload["filings"]["recent"]
    except (KeyError, TypeError) as exc:
        raise SecDataError(
            f"Unexpected submissions payload for CIK {normalize_cik(cik)}."
        ) from exc

    missing = [key for key in _REQUIRED_EDGAR_KEYS if key not in recent]
    if missing:
        raise SecDataError(
            f"Submissions payload for CIK {normalize_cik(cik)} is missing "
            f"expected keys: {missing}."
        )

    frame = pd.DataFrame(
        {
            snake: recent[edgar]
            for edgar, snake in _EDGAR_TO_SNAKE.items()
            if edgar in recent
        }
    )

    wanted = {form.strip().upper() for form in forms}
    frame = frame[frame["form"].str.upper().isin(wanted)]
    frame = frame.sort_values("filing_date", ascending=False).reset_index(drop=True)

    frame["url"] = [
        filing_url(cik, accession, document)
        for accession, document in zip(
            frame["accession_number"], frame["primary_document"], strict=True
        )
    ]
    return frame


def download_document(url: str) -> str:
    """Download a filing document and return it as text.

    Raises
    ------
    ValueError
        If ``url`` is not a non-empty string.
    SecDataError
        If the request fails.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("url must be a non-empty string.")

    try:
        response = requests.get(
            url, headers=_headers(), timeout=REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SecDataError(f"Request to {url} failed: {exc}") from exc

    return response.text


def save_document(
    text: str,
    cik: str | int,
    accession_number: str,
    document: str,
    root: Path = DEFAULT_RAW_DIR,
) -> Path:
    """Write ``text`` under ``root/<cik>/<accession>/<document>``.

    Returns the path that was written. The filename is reduced to its final
    component, so a hostile ``document`` value cannot escape ``root``.
    """
    safe_name = Path(document).name
    if not safe_name:
        raise ValueError("document must contain a filename.")

    directory = root / str(int(normalize_cik(cik))) / accession_number.replace("-", "")
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / safe_name
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    """Manual smoke test: ``python -m app.ingestion.sec``.

    Resolves NVDA to a CIK, lists recent 10-K/10-Q filings, records them in the
    local index, then downloads and saves the most recent one.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    # Imported here, not at module level, for the same reason the market example
    # imports :mod:`app.quant` inside its ``main``: retrieval stays independent
    # of everything built on top of it. It also breaks the import cycle, since
    # the index module reuses :func:`normalize_cik` from this one.
    from app.ingestion import store

    ticker = "NVDA"

    try:
        cik = get_cik(ticker)
        filings = list_filings(cik)
    except (ValueError, SecDataError) as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc

    print(f"\n{ticker} -> CIK {cik}")
    print(f"{len(filings)} filing(s) of type {DEFAULT_FORMS}. Five most recent:\n")

    columns = ["form", "filing_date", "report_date", "acceptance_datetime"]
    print(filings[columns].head(5).to_string(index=False))

    connection = store.connect()
    try:
        store.upsert_filings(connection, cik, filings)
        print(
            f"\nIndexed {len(filings)} filing(s) in "
            f"{store.display_path(store.DEFAULT_INDEX_PATH)}"
        )

        if filings.empty:
            print("\nNothing to download.")
            return

        latest = filings.iloc[0]
        print(f"\nDownloading the {latest['form']} filed {latest['filing_date']} ...")

        try:
            text = download_document(str(latest["url"]))
            path = save_document(
                text,
                cik,
                str(latest["accession_number"]),
                str(latest["primary_document"]),
            )
            store.mark_saved(connection, str(latest["accession_number"]), path)
        except (ValueError, SecDataError) as exc:
            logger.error("%s", exc)
            raise SystemExit(1) from exc

        print(f"Saved {len(text):,} characters to {path.relative_to(PROJECT_ROOT)}")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
