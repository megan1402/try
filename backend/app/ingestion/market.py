"""Historical market-data ingestion.

This module answers a single question: *"What did the market do for ticker X?"*
It retrieves raw OHLCV bars and does nothing else. All derived analytics
(returns, features, statistics) live in :mod:`app.quant`, so retrieval and
analysis can evolve and be tested independently.

The data source is ``yfinance`` for the Phase 1 prototype. Consumers only
depend on the shape of the returned :class:`pandas.DataFrame`, so the source
can later be swapped for a production market-data API without touching them.
"""

from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

#: Columns we require from a provider before handing data to callers.
REQUIRED_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")


class MarketDataError(RuntimeError):
    """Raised when market data cannot be retrieved for a ticker."""


def get_market_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Retrieve historical daily OHLCV bars for ``ticker``.

    Parameters
    ----------
    ticker:
        Stock symbol, e.g. ``"NVDA"``. Case-insensitive; surrounding whitespace
        is trimmed.
    period:
        Look-back window understood by the provider, e.g. ``"1y"``, ``"6mo"``,
        ``"5d"`` or ``"max"``.

    Returns
    -------
    pandas.DataFrame
        One row per trading day, indexed by a ``DatetimeIndex`` named ``Date``
        and ordered oldest-first, with at least the columns ``Open``, ``High``,
        ``Low``, ``Close`` and ``Volume``.

    Raises
    ------
    ValueError
        If ``ticker`` is not a non-empty string.
    MarketDataError
        If the provider returns no data (typically an unknown or delisted
        symbol) or the request fails because of a network/API error.
    """
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("ticker must be a non-empty string, e.g. 'NVDA'.")

    symbol = ticker.strip().upper()

    try:
        # auto_adjust=False keeps an unmodified "Close" column (and an
        # "Adj Close" column) so the price the caller asked for is what they get.
        data = yf.Ticker(symbol).history(
            period=period, interval="1d", auto_adjust=False
        )
    except Exception as exc:  # yfinance raises a variety of HTTP/parse errors.
        raise MarketDataError(
            f"Could not download market data for {symbol!r} over period "
            f"{period!r}: {exc}"
        ) from exc

    if data is None or data.empty:
        raise MarketDataError(
            f"No market data returned for {symbol!r} over period {period!r}. "
            "The ticker may be invalid or delisted, or the period unsupported."
        )

    # A MultiIndex appears only when several tickers are requested at once;
    # flatten defensively so callers always see simple column names.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise MarketDataError(
            f"Market data for {symbol!r} is missing expected columns: {missing}."
        )

    data.index.name = "Date"
    return data


def main() -> None:
    """Manual smoke test: ``python -m app.ingestion.market``.

    Deliberately keeps the ingestion/analytics separation intact: the analytics
    import happens here, in the example entry point, not at module level.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    from app.quant.returns import add_forward_returns

    ticker = "NVDA"
    try:
        prices = get_market_data(ticker, period="1y")
    except (ValueError, MarketDataError) as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc

    enriched = add_forward_returns(prices)
    columns = ["Close", "return_1d", "return_5d", "return_20d"]

    print(
        f"\n{ticker}: {len(enriched)} trading days "
        f"({enriched.index.min().date()} -> {enriched.index.max().date()})\n"
    )
    print(enriched[columns].tail(10).to_string())
    print(
        "\nThe last rows show NaN forward returns because the price they would "
        "need is in the future and does not exist yet. That is expected, and it "
        "is exactly what prevents look-ahead bias."
    )


if __name__ == "__main__":
    main()
