"""Tests for :mod:`app.ingestion.market`.

These cover local input validation only — they never make a network request,
so they stay fast and reliable in CI.
"""

from __future__ import annotations

import pytest

from app.ingestion.market import get_market_data


@pytest.mark.parametrize("bad_ticker", ["", "   ", None, 123])
def test_invalid_ticker_raises_value_error(bad_ticker):
    with pytest.raises(ValueError):
        get_market_data(bad_ticker)  # type: ignore[arg-type]
