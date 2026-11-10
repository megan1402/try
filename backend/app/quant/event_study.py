"""Event study: what actually happened after each event.

An *event study* attaches forward returns to events and summarises them per
event class. It answers the only question that makes event extraction
worthwhile: **do events of this type have a distinguishable effect on the
price?**

The one rule that matters here is where the return clock starts.

A filing is accepted by EDGAR *after* the market closes (``acceptance_datetime``
is typically ~20:00–22:00 UTC). So the last price that could *not* have reacted
to the news is that day's close. We therefore anchor each event on **the last
trading day at or before its timestamp**, and reuse the forward returns already
computed there:

    close(t0)  ->  close(t0 + h)  /  close(t0) - 1        t0 = anchor close

This is the standard event-study convention, and it is honest as long as you
remember what it does *not* claim: you could not have traded at that close.
The first horizon includes the overnight gap in which the market digested the
news. Anchoring on the *next* trading day instead would quietly throw that gap
away and understate every reaction.

The reverse mistake — dating the event by ``filing_date`` and using a return
that starts before the news — is look-ahead bias: a backtest that buys on
information it does not yet have. ``as_of`` exists to make that impossible.

Anchoring at or before the timestamp also handles the pre-market case: an
acceptance at 10:00 UTC is before that session's open, so the whole of the
session is a reaction and the previous close is the right anchor.

Events with no forward window yet (the event is too recent, or the price
history ends) keep ``NaT``/`NaN`; :func:`summarise_events` leaves them out of
the statistics instead of pretending they are zero.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import pandas as pd

from app.quant.returns import FORWARD_RETURN_HORIZONS

#: Columns of :func:`summarise_events`.
SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "event_type",
    "horizon_days",
    "n",
    "median",
    "mean",
    "hit_rate",
    "t_stat",
)

#: Hour, in the price index's timezone, at which a daily session is assumed to
#: have closed. 16:00 is the standard US equity close.
SESSION_CLOSE_HOUR: Final[int] = 16


def attach_forward_returns(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    max_staleness_days: int = 7,
) -> pd.DataFrame:
    """Return ``events`` with ``event_date`` and ``return_{h}d`` columns added.

    Parameters
    ----------
    events:
        Table with an ``as_of`` column of timestamps (naive or tz-aware) and an
        ``event_type`` column — normally the output of
        :func:`app.extraction.events.events_to_frame`.
    prices:
        Price history with a ``DatetimeIndex`` and the forward-return columns
        produced by :func:`app.quant.returns.add_forward_returns`
        (``return_1d``, ``return_5d``, ``return_20d``).
    max_staleness_days:
        An event whose nearest earlier session is older than this is treated as
        outside the price history: it gets no anchor and no returns. The default
        of 7 days leaves room for weekends and holidays without letting a stale
        close stand in for a date the data does not cover.

    Returns
    -------
    pandas.DataFrame
        A copy of ``events`` with ``event_date`` (the anchor close, ``NaT`` when
        the event precedes or follows the price history) and one column per
        horizon.

    Raises
    ------
    ValueError
        If ``as_of`` is missing, the price index is not a ``DatetimeIndex``, or
        a forward-return column is missing.
    """
    if "as_of" not in events.columns:
        raise ValueError("events must have an 'as_of' column.")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError(
            "prices must be indexed by a DatetimeIndex, got "
            f"{type(prices.index).__name__}."
        )
    if prices.empty:
        raise ValueError("prices is empty; there is no history to anchor on.")

    returns = {horizon: f"return_{horizon}d" for horizon in FORWARD_RETURN_HORIZONS}
    missing = [column for column in returns.values() if column not in prices.columns]
    if missing:
        raise ValueError(
            f"prices is missing forward-return columns: {missing}. "
            "Run app.quant.returns.add_forward_returns first."
        )

    # Compare wall-clock times, so a tz-aware index and a tz-aware timestamp in
    # a different zone still line up with the session they belong to.
    trading_days = prices.index.normalize()
    as_of = pd.to_datetime(events["as_of"], utc=True)
    as_of = (
        as_of.dt.tz_convert(trading_days.tz)
        if trading_days.tz is not None
        else as_of.dt.tz_convert(None)
    )

    # yfinance labels daily bars at midnight, so the bar date alone cannot say
    # whether an intraday acceptance came before or after that session's close.
    # Assume a close at ``SESSION_CLOSE_HOUR`` local time and compare against
    # that instead: a 10:00 acceptance anchors on the previous close, a 21:00
    # one on the same session's close.
    session_close = trading_days + pd.Timedelta(hours=SESSION_CLOSE_HOUR)
    comparable_close = (
        session_close.tz_localize(None)
        if session_close.tz is not None
        else session_close
    )
    comparable_as_of = as_of.dt.tz_localize(None) if as_of.dt.tz is not None else as_of

    # side="right" then -1 gives the last close at or before the event: the
    # close the market could not yet have reacted to.
    positions = comparable_close.searchsorted(comparable_as_of.to_numpy(), "right")
    positions = positions - 1

    # ``take`` needs a valid position even for events with no anchor; those rows
    # are overwritten with NaT/NaN immediately afterwards.
    safe_positions = np.clip(positions, 0, len(trading_days) - 1)
    anchor_days = pd.Series(trading_days.take(safe_positions), index=events.index)

    # An event far beyond the end of the price history must not silently borrow
    # the last available close as its anchor.
    staleness = (as_of.dt.normalize() - anchor_days).dt.days
    has_anchor = (positions >= 0) & (staleness <= max_staleness_days)

    result = events.copy()

    event_dates = anchor_days.copy()
    event_dates[~has_anchor] = None
    result["event_date"] = event_dates

    for horizon, column in returns.items():
        values = np.asarray(prices[column], dtype=float)[safe_positions]
        result[f"return_{horizon}d"] = np.where(has_anchor, values, np.nan)

    return result


def summarise_events(events: pd.DataFrame) -> pd.DataFrame:
    """Summarise returns per event type and horizon.

    Returns one row per ``(event_type, horizon_days)`` with the number of usable
    observations, median, mean, the share of positive returns and a t-statistic
    for "mean is not zero".

    ``t_stat`` is ``mean / (std / sqrt(n))`` using the sample standard
    deviation; it is ``NaN`` when fewer than two observations exist or the
    returns do not vary. Treat it as a smell test, not a p-value: these returns
    are heavily overlapping and not independent.

    Raises
    ------
    ValueError
        If a forward-return column is missing.
    """
    missing = [
        f"return_{horizon}d"
        for horizon in FORWARD_RETURN_HORIZONS
        if f"return_{horizon}d" not in events.columns
    ]
    if missing:
        raise ValueError(
            f"events is missing forward-return columns: {missing}. "
            "Run attach_forward_returns first."
        )

    rows: list[dict[str, object]] = []

    for event_type, group in events.groupby("event_type", sort=True):
        for horizon in FORWARD_RETURN_HORIZONS:
            values = group[f"return_{horizon}d"].dropna().to_numpy(dtype=float)
            rows.append(
                _summarise_values(str(event_type), horizon, values),
            )

    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))


def _summarise_values(
    event_type: str,
    horizon: int,
    values: np.ndarray,
) -> dict[str, object]:
    """One summary row; ``NaN`` where there is nothing to report."""
    n = int(values.size)
    if n == 0:
        return {
            "event_type": event_type,
            "horizon_days": horizon,
            "n": 0,
            "median": float("nan"),
            "mean": float("nan"),
            "hit_rate": float("nan"),
            "t_stat": float("nan"),
        }

    mean = float(values.mean())
    spread = float(values.std(ddof=1)) if n > 1 else float("nan")
    t_stat = mean / (spread / math.sqrt(n)) if n > 1 and spread > 0.0 else float("nan")

    return {
        "event_type": event_type,
        "horizon_days": horizon,
        "n": n,
        "median": float(np.median(values)),
        "mean": mean,
        "hit_rate": float((values > 0).mean()),
        "t_stat": t_stat,
    }
