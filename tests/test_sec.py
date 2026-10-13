"""Tests for :mod:`app.ingestion.sec`.

Every EDGAR response comes from the saved fixtures in ``tests/fixtures`` (or
from a small inline payload), so **no test touches the network**. The fixtures
are hand-written to match the shape of EDGAR's real JSON responses; they are
not live captures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.ingestion import sec

FIXTURES = Path(__file__).parent / "fixtures"
TICKER_MAP: dict[str, Any] = json.loads(
    (FIXTURES / "company_tickers.json").read_text(encoding="utf-8")
)
SUBMISSIONS: dict[str, Any] = json.loads(
    (FIXTURES / "submissions_CIK0001045810.json").read_text(encoding="utf-8")
)

NVDA_CIK = "0001045810"


# --- Helpers ---------------------------------------------------------------


class FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(
        self, *, text: str = "", payload: Any = None, status: int = 200
    ) -> None:
        self.text = text
        self.status_code = status
        self._payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise sec.requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("body is not JSON")
        return self._payload


def make_filing(
    form: str,
    filing_date: str,
    *,
    accession: str = "0001045810-24-000001",
    document: str = "doc.htm",
) -> dict[str, str]:
    """Build one EDGAR 'filing' entry as parallel-array source data."""
    return {
        "accessionNumber": accession,
        "filingDate": filing_date,
        "reportDate": filing_date,
        "acceptanceDateTime": f"{filing_date}T21:00:00.000Z",
        "form": form,
        "primaryDocument": document,
    }


def make_submissions_payload(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Build a submissions payload from a list of filing entries."""
    keys = [
        "accessionNumber",
        "filingDate",
        "reportDate",
        "acceptanceDateTime",
        "form",
        "primaryDocument",
    ]
    return {
        "cik": "1045810",
        "filings": {"recent": {key: [row[key] for row in rows] for key in keys}},
    }


@pytest.fixture
def fake_edgar(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route every EDGAR JSON request to the saved fixtures."""

    def _fake_fetch_json(url: str) -> Any:
        if url == sec.TICKER_MAP_URL:
            return TICKER_MAP
        if url == sec.submissions_url(NVDA_CIK):
            return SUBMISSIONS
        raise AssertionError(f"unexpected URL requested: {url}")

    monkeypatch.setattr(sec, "_fetch_json", _fake_fetch_json)


# --- Pure helpers ----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (1045810, "0001045810"),
        (1045810.0, None),  # floats are not accepted
        ("1045810", "0001045810"),
        ("0001045810", "0001045810"),
        (" 320193 ", "0000320193"),
    ],
)
def test_normalize_cik(raw: object, expected: str | None) -> None:
    if expected is None:
        with pytest.raises(ValueError):
            sec.normalize_cik(raw)  # type: ignore[arg-type]
    else:
        assert sec.normalize_cik(raw) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_cik", ["", "   ", "abc", "12-34"])
def test_normalize_cik_rejects_non_numeric(bad_cik: str) -> None:
    with pytest.raises(ValueError):
        sec.normalize_cik(bad_cik)


def test_submissions_url_uses_padded_cik() -> None:
    assert sec.submissions_url(1045810) == (
        "https://data.sec.gov/submissions/CIK0001045810.json"
    )


def test_filing_url_uses_unpadded_cik_and_strips_dashes() -> None:
    url = sec.filing_url(NVDA_CIK, "0001045810-25-000023", "nvda-20250126.htm")

    assert url == (
        "https://www.sec.gov/Archives/edgar/data/"
        "1045810/000104581025000023/nvda-20250126.htm"
    )


@pytest.mark.parametrize(
    ("accession", "document"),
    [("", "doc.htm"), ("0001-25-000023", ""), ("   ", "doc.htm")],
)
def test_filing_url_rejects_empty_parts(accession: str, document: str) -> None:
    with pytest.raises(ValueError):
        sec.filing_url(NVDA_CIK, accession, document)


# --- get_cik ---------------------------------------------------------------


@pytest.mark.parametrize("ticker", ["NVDA", "nvda", "  nvda  "])
def test_get_cik_resolves_ticker(fake_edgar: None, ticker: str) -> None:
    assert sec.get_cik(ticker) == NVDA_CIK


def test_get_cik_returns_padded_cik(fake_edgar: None) -> None:
    assert sec.get_cik("AAPL") == "0000320193"


def test_get_cik_unknown_ticker_raises(fake_edgar: None) -> None:
    with pytest.raises(sec.SecDataError, match="not found"):
        sec.get_cik("NOTATICKER")


@pytest.mark.parametrize("bad_ticker", ["", "   ", None, 123])
def test_get_cik_invalid_input_raises_value_error(
    fake_edgar: None, bad_ticker: object
) -> None:
    with pytest.raises(ValueError):
        sec.get_cik(bad_ticker)  # type: ignore[arg-type]


# --- list_filings ----------------------------------------------------------


def test_list_filings_returns_expected_columns(fake_edgar: None) -> None:
    frame = sec.list_filings(NVDA_CIK)

    assert list(frame.columns) == [
        "form",
        "filing_date",
        "report_date",
        "acceptance_datetime",
        "accession_number",
        "primary_document",
        "url",
    ]


def test_list_filings_filters_out_other_forms(fake_edgar: None) -> None:
    frame = sec.list_filings(NVDA_CIK)

    # The fixture contains one 8-K, which must not appear.
    assert set(frame["form"]) == {"10-K", "10-Q"}
    assert len(frame) == 7


def test_list_filings_sorted_newest_first(fake_edgar: None) -> None:
    frame = sec.list_filings(NVDA_CIK)

    dates = list(frame["filing_date"])
    assert dates == sorted(dates, reverse=True)
    assert dates[0] == "2025-02-26"


def test_list_filings_sorts_when_source_is_oldest_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        make_filing("10-Q", "2024-01-05"),
        make_filing("10-Q", "2024-06-05"),
        make_filing("10-K", "2024-12-05"),
    ]
    monkeypatch.setattr(sec, "_fetch_json", lambda url: make_submissions_payload(rows))

    frame = sec.list_filings(NVDA_CIK)

    assert list(frame["filing_date"]) == ["2024-12-05", "2024-06-05", "2024-01-05"]


def test_list_filings_builds_filing_urls(fake_edgar: None) -> None:
    frame = sec.list_filings(NVDA_CIK)
    first = frame.iloc[0]

    expected = sec.filing_url(
        NVDA_CIK, str(first["accession_number"]), str(first["primary_document"])
    )
    assert first["url"] == expected
    assert first["url"].endswith("nvda-20250126.htm")


def test_list_filings_accepts_custom_forms(fake_edgar: None) -> None:
    frame = sec.list_filings(NVDA_CIK, forms=("8-K",))

    assert set(frame["form"]) == {"8-K"}
    assert len(frame) == 1


def test_list_filings_returns_empty_frame_when_nothing_matches(
    fake_edgar: None,
) -> None:
    frame = sec.list_filings(NVDA_CIK, forms=("DEF 14A",))

    assert frame.empty
    assert "url" in frame.columns


def test_list_filings_missing_keys_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sec, "_fetch_json", lambda url: {"filings": {"recent": {}}})

    with pytest.raises(sec.SecDataError, match="missing expected keys"):
        sec.list_filings(NVDA_CIK)


def test_list_filings_bad_payload_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sec, "_fetch_json", lambda url: {"unexpected": True})

    with pytest.raises(sec.SecDataError):
        sec.list_filings(NVDA_CIK)


# --- HTTP behaviour --------------------------------------------------------


def test_download_document_returns_text_and_sends_user_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_get(url: str, headers: dict[str, str], timeout: int) -> FakeResponse:
        captured["url"] = url
        captured["headers"] = headers
        captured["timeout"] = timeout
        return FakeResponse(text="<html>10-K</html>")

    monkeypatch.setattr(sec.requests, "get", _fake_get)

    text = sec.download_document("https://www.sec.gov/Archives/example.htm")

    assert text == "<html>10-K</html>"
    assert captured["url"] == "https://www.sec.gov/Archives/example.htm"
    assert "User-Agent" in captured["headers"]
    assert captured["timeout"] == sec.REQUEST_TIMEOUT_SECONDS


def test_download_document_wraps_network_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*args: Any, **kwargs: Any) -> FakeResponse:
        raise sec.requests.ConnectionError("network is down")

    monkeypatch.setattr(sec.requests, "get", _boom)

    with pytest.raises(sec.SecDataError, match="failed"):
        sec.download_document("https://www.sec.gov/Archives/example.htm")


def test_fetch_json_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sec.requests, "get", lambda *a, **k: FakeResponse(text="nope", status=403)
    )

    with pytest.raises(sec.SecDataError, match="failed"):
        sec._fetch_json(sec.TICKER_MAP_URL)


def test_fetch_json_raises_on_non_json_body(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sec.requests, "get", lambda *a, **k: FakeResponse(text="<html>oops</html>")
    )

    with pytest.raises(sec.SecDataError, match="not valid JSON"):
        sec._fetch_json(sec.TICKER_MAP_URL)


# --- save_document ---------------------------------------------------------


def test_save_document_writes_to_expected_path(tmp_path: Path) -> None:
    path = sec.save_document(
        "<html>filing</html>",
        NVDA_CIK,
        "0001045810-25-000023",
        "nvda-20250126.htm",
        root=tmp_path,
    )

    assert path == tmp_path / "1045810" / "000104581025000023" / "nvda-20250126.htm"
    assert path.read_text(encoding="utf-8") == "<html>filing</html>"


def test_save_document_creates_parent_directories(tmp_path: Path) -> None:
    path = sec.save_document("x", NVDA_CIK, "0001-25-000023", "a.htm", root=tmp_path)

    assert path.exists()
    assert path.parent.is_dir()


def test_save_document_cannot_escape_root(tmp_path: Path) -> None:
    path = sec.save_document(
        "x", NVDA_CIK, "0001-25-000023", "../../evil.htm", root=tmp_path
    )

    assert path.name == "evil.htm"
    assert tmp_path in path.parents
