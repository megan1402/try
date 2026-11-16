"""Benchmark-relative (abnormal) forward returns.

A raw forward return mixes two things: what the *company* did and what its
*market* did. On a day the index falls 3%, a stock that falls 3% told you
nothing about the news in its filing — yet the raw number reports -3% and the
event study dutifully averages it in. Only the difference is attributable to
the event:

    abnormal_return_h[t] = forward_return_h[t] - benchmark_forward_return_h[t]

For ``h = 5`` that pairs the stock's five-session move with the benchmark's
move over the *same five sessions*.

Two decisions worth knowing:

* **Both legs share one calendar.** The stock's and the benchmark's histories
  are intersected before any return is computed, so a horizon always spans the
  same sessions on both sides. Computing each leg separately and subtracting
  afterwards would compare a five-day window with a four-day one whenever the
  calendars differ — and would silently produce an "abnormal" return out of a
  holiday.
* **The adjusted frame keeps the pipeline's column names.** ``return_{h}d`` in
  the result holds the *abnormal* return, so it drops straight into
  :func:`app.quant.event_study.attach_forward_returns` and
  :func:`app.quant.event_study.summarise_events` unchanged. Raw and
  market-adjusted studies are therefore two different price frames, not two
  code paths.

This is the *market-adjusted* model, the simplest one that controls for the
market at all. It does not adjust for beta (a high-beta stock moves more than
its benchmark by construction), for sector or factor exposure, or for
volatility clustering, and it does not make overlapping windows independent.
A CAPM/factor model estimated over a pre-event window is the natural next step
once there are enough observations to make the estimation window meaningful.

Look-ahead is not reintroduced here: the anchor close is still chosen by
:func:`app.quant.event_study.attach_forward_returns` from the event's own
``as_of``; this module only changes *what is measured* after that anchor.
"""

from __future__ import annotations

import pandas as pd

from app.quant.returns import FORWARD_RETURN_HORIZONS

#: Benchmark used when a caller does not name one. A broad US equity index
#: tracks the market exposure that most large-cap filings are judged against.
DEFAULT_BENCHMARK = "SPY"

#: Daily abnormal-return columns returned by :func:`market_adjusted_returns`.
ABNORMAL_COLUMNS: tuple[str, str, str] = (
    "asset_return",
    "benchmark_return",
    "abnormal_return",
)


def daily_returns(prices: pd.DataFrame) -> pd.Series:
    """Return close-to-close returns of ``prices``: ``Close[t] / Close[t-1] - 1``.

    The first row is ``NaN`` because it has no previous close. Missing prices
    are *not* forward-filled: inventing a flat day would fabricate a zero
    return, so a gap simply yields ``NaN`` for that step.

    Raises
    ------
    KeyError
        If ``prices`` has no ``Close`` column.
    """
    if "Close" not in prices.columns:
        raise KeyError(
            f"daily_returns expects a 'Close' column; got {list(prices.columns)}."
        )
    return prices["Close"].astype(float).pct_change(fill_method=None)


def market_adjusted_returns(
    prices: pd.DataFrame, benchmark_prices: pd.DataFrame
) -> pd.DataFrame:
    """Return daily stock, benchmark and abnormal returns on their shared dates.

    Parameters
    ----------
    prices:
        Price history with a ``DatetimeIndex`` and a ``Close`` column.
    benchmark_prices:
        The same for the benchmark, e.g. ``get_market_data("SPY", period="2y")``.

    Returns
    -------
    pandas.DataFrame
        Indexed by the trading dates present in *both* histories, with the
        columns ``asset_return``, ``benchmark_return`` and ``abnormal_return``
        (the difference of the first two).

    Raises
    ------
    ValueError
        If either history is not indexed by a ``DatetimeIndex`` or the two have
        no trading date in common.
    """
    closes = _common_closes(prices, benchmark_prices)

    asset_return = closes["asset_close"].pct_change(fill_method=None)
    benchmark_return = closes["benchmark_close"].pct_change(fill_method=None)

    return pd.DataFrame(
        {
            "asset_return": asset_return,
            "benchmark_return": benchmark_return,
            "abnormal_return": asset_return - benchmark_return,
        },
        index=closes.index,
    )


def add_abnormal_returns(
    prices: pd.DataFrame, benchmark_prices: pd.DataFrame
) -> pd.DataFrame:
    """Return a price frame whose forward returns are net of the benchmark.

    The result is a drop-in replacement for
    :func:`app.quant.returns.add_forward_returns`: it has a ``DatetimeIndex``
    and ``return_1d``, ``return_5d`` and ``return_20d`` columns, so the event
    study accepts it without modification. Those columns hold the
    **benchmark-relative** return — the stock's forward return minus the
    benchmark's over the identical, aligned horizon.

    Also included for inspection: ``Close`` (the stock's close on the shared
    calendar) and ``Benchmark`` (the benchmark's close on the same dates).

    Parameters
    ----------
    prices:
        Price history with a ``DatetimeIndex`` and a ``Close`` column.
    benchmark_prices:
        The same for the benchmark.

    Returns
    -------
    pandas.DataFrame
        Indexed by the trading dates in both histories. The final rows of each
        ``return_{h}d`` column are ``NaN``: their future does not exist yet.

    Raises
    ------
    ValueError
        If either history is not indexed by a ``DatetimeIndex`` or the two have
        no trading date in common.
    """
    closes = _common_closes(prices, benchmark_prices)

    result = pd.DataFrame(index=closes.index)
    result.index.name = "Date"
    result["Close"] = closes["asset_close"]
    result["Benchmark"] = closes["benchmark_close"]

    for horizon in FORWARD_RETURN_HORIZONS:
        asset = closes["asset_close"]
        benchmark = closes["benchmark_close"]
        # shift(-h) pairs each row with the price h rows later: the same
        # forward, never-backward convention as add_forward_returns.
        asset_forward = asset.shift(-horizon) / asset - 1.0
        benchmark_forward = benchmark.shift(-horizon) / benchmark - 1.0
        result[f"return_{horizon}d"] = asset_forward - benchmark_forward

    return result


def normalise_benchmark(value: str | None) -> str | None:
    """Interpret a CLI-style benchmark argument.

    ``None``, an empty string, ``"none"`` and ``"off"`` disable the adjustment
    and return ``None``; anything else is stripped and upper-cased, so ``" spy "``
    becomes ``"SPY"``.
    """
    if value is None:
        return None
    symbol = value.strip().upper()
    if symbol in {"", "NONE", "OFF"}:
        return None
    return symbol


def _common_closes(
    prices: pd.DataFrame, benchmark_prices: pd.DataFrame
) -> pd.DataFrame:
    """Return ``asset_close`` and ``benchmark_close`` on shared calendar dates."""
    asset = _close_by_date(prices, "prices")
    benchmark = _close_by_date(benchmark_prices, "benchmark_prices")

    closes = asset.join(benchmark, how="inner")
    if closes.empty:
        raise ValueError(
            "prices and benchmark_prices share no trading dates "
            f"({len(asset)} vs {len(benchmark)} rows); check the two symbols "
            "and the requested period."
        )
    return closes


def _close_by_date(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """One close per calendar date, timezone and time-of-day removed.

    Daily bars are often labelled at midnight in a provider-specific timezone
    and can repeat a date when a request is paginated. Taking the calendar date
    as written keeps two histories from different providers aligned, and
    ``keep="last"`` makes a duplicate deterministic rather than an error.
    """
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(
            f"{name} must be indexed by a DatetimeIndex, got "
            f"{type(frame.index).__name__}."
        )
    if "Close" not in frame.columns:
        raise KeyError(f"{name} must have a 'Close' column; got {list(frame.columns)}.")
    if frame.empty:
        raise ValueError(f"{name} is empty; there is no history to compare.")

    index = frame.index.normalize()
    if index.tz is not None:
        index = index.tz_localize(None)

    column = "asset_close" if name == "prices" else "benchmark_close"
    close = pd.Series(frame["Close"].to_numpy(dtype=float), index=index, name=column)
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close.to_frame()
