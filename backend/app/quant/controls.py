"""Statistical controls for the event study.

The raw event study reports a median return per event class. On its own that
number cannot distinguish an event effect from three much more mundane
explanations:

1. **The market moved.** A −1% median after every event class is a statement
   about the sample period, not about events. Subtracting a benchmark's return
   over the same window removes the common factor.
2. **Events cluster.** Ten mega-cap issuers file within days of each other, so
   827 events can sit on 52 distinct dates. Treating them as 827 independent
   trials inflates every t-statistic by roughly the square root of the cluster
   size.
3. **Any day looks like this.** Without a null distribution there is no way to
   know whether an event day differs from an ordinary trading day.

This module supplies all three, in that order:

* :func:`abnormal_returns` — event return minus benchmark return, same window.
* :func:`cluster_by_date` — one observation per anchor date, so the unit of
  analysis matches the unit of independence.
* :func:`null_distribution` — the same statistics on randomly chosen non-filing
  days, seeded for reproducibility.

Look-ahead safety is inherited, not re-implemented: every return here comes from
:func:`app.quant.event_study.attach_forward_returns`, which anchors on the last
close at or before the event's ``acceptance_datetime``. The benchmark is aligned
to the *same* anchor date, so the subtraction never mixes windows.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

import numpy as np
import pandas as pd

from app.quant.returns import FORWARD_RETURN_HORIZONS

#: Columns of :func:`cluster_by_date`.
CLUSTER_COLUMNS: Final[tuple[str, ...]] = (
    "event_type",
    "event_date",
    "n_events",
    "return_1d",
    "return_5d",
    "return_20d",
)

#: Columns of :func:`compare_to_null`.
NULL_COLUMNS: Final[tuple[str, ...]] = (
    "event_type",
    "horizon_days",
    "n_dates",
    "event_median",
    "null_median",
    "null_p05",
    "null_p95",
    "percentile",
    "p_value",
)

#: Default seed. Fixed so a reported number can be reproduced exactly.
DEFAULT_SEED: Final[int] = 20260101

#: Trading days either side of an event date that the null must avoid. A
#: 20-day forward window means a day 20 sessions away still overlaps the event.
EXCLUSION_WINDOW_DAYS: Final[int] = 30


def forward_return_columns() -> tuple[str, ...]:
    """The forward-return column names, in horizon order."""
    return tuple(f"return_{horizon}d" for horizon in FORWARD_RETURN_HORIZONS)


def abnormal_returns(
    events: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> pd.DataFrame:
    """Add ``abnormal_{h}d`` columns: event return minus benchmark return.

    Parameters
    ----------
    events:
        Output of :func:`app.quant.event_study.attach_forward_returns`, so it
        already has ``event_date`` and ``return_{h}d``.
    benchmark:
        Benchmark price history with a ``DatetimeIndex`` and the same forward
        return columns, e.g. ``add_forward_returns(get_market_data("SPY"))``.

    Returns
    -------
    pandas.DataFrame
        A copy of ``events`` with one ``abnormal_{h}d`` column per horizon.
        Rows whose anchor date is missing from the benchmark keep ``NaN``
        rather than silently falling back to the raw return.

    Raises
    ------
    ValueError
        If ``event_date`` is missing, the benchmark index is not a
        ``DatetimeIndex``, or a forward-return column is absent from either
        frame.
    """
    if "event_date" not in events.columns:
        raise ValueError(
            "events must have an 'event_date' column; run attach_forward_returns first."
        )
    if not isinstance(benchmark.index, pd.DatetimeIndex):
        raise ValueError(
            "benchmark must be indexed by a DatetimeIndex, got "
            f"{type(benchmark.index).__name__}."
        )

    columns = forward_return_columns()
    missing_events = [column for column in columns if column not in events.columns]
    if missing_events:
        raise ValueError(f"events is missing forward-return columns: {missing_events}.")
    missing_benchmark = [
        column for column in columns if column not in benchmark.columns
    ]
    if missing_benchmark:
        raise ValueError(
            f"benchmark is missing forward-return columns: {missing_benchmark}."
        )

    # Align on the calendar date, not the timestamp: the benchmark bar for the
    # same session is the one that shares the event's window.
    benchmark_days = benchmark.index.normalize()
    if benchmark_days.tz is not None:
        benchmark_days = benchmark_days.tz_localize(None)

    lookup = {
        column: pd.Series(
            np.asarray(benchmark[column], dtype=float), index=benchmark_days
        )
        for column in columns
    }

    event_days = pd.to_datetime(events["event_date"])
    if isinstance(event_days.dtype, pd.DatetimeTZDtype):
        event_days = event_days.dt.tz_localize(None)
    event_days = event_days.dt.normalize()

    result = events.copy()
    for column in columns:
        horizon = column.removeprefix("return_").removesuffix("d")
        benchmark_values = event_days.map(lookup[column])
        result[f"abnormal_{horizon}d"] = result[column].to_numpy(
            dtype=float
        ) - benchmark_values.to_numpy(dtype=float)

    return result


def cluster_by_date(
    events: pd.DataFrame,
    *,
    return_prefix: str = "abnormal_",
) -> pd.DataFrame:
    """Collapse events to one observation per ``(event_type, event_date)``.

    Events sharing an anchor date share a return window, so they are not
    independent trials. Averaging within the date first makes the sample size
    equal the number of distinct dates, which is what a significance test
    should use.

    Parameters
    ----------
    events:
        Frame with ``event_type``, ``event_date`` and the return columns.
    return_prefix:
        ``"abnormal_"`` (default) to cluster benchmark-adjusted returns, or
        ``"return_"`` to cluster raw ones.

    Returns
    -------
    pandas.DataFrame
        One row per ``(event_type, event_date)`` with ``n_events`` and the mean
        return per horizon. Rows without an anchor date are dropped.

    Raises
    ------
    ValueError
        If a required column is missing.
    """
    required = ["event_type", "event_date"]
    missing = [column for column in required if column not in events.columns]
    if missing:
        raise ValueError(f"events is missing required columns: {missing}.")

    columns = [f"{return_prefix}{horizon}d" for horizon in FORWARD_RETURN_HORIZONS]
    missing_returns = [column for column in columns if column not in events.columns]
    if missing_returns:
        raise ValueError(f"events is missing return columns: {missing_returns}.")

    usable = events.dropna(subset=["event_date"])
    if usable.empty:
        return pd.DataFrame(columns=list(CLUSTER_COLUMNS))

    grouped = (
        usable.groupby(["event_type", "event_date"], sort=True)
        .agg(
            n_events=("event_type", "size"),
            **{
                f"return_{horizon}d": (f"{return_prefix}{horizon}d", "mean")
                for horizon in FORWARD_RETURN_HORIZONS
            },
        )
        .reset_index()
    )

    return grouped.loc[:, list(CLUSTER_COLUMNS)]


def null_distribution(
    prices: pd.DataFrame,
    *,
    exclude_dates: Sequence[pd.Timestamp] = (),
    n_samples: int = 1_000,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Forward returns on randomly chosen non-filing trading days.

    This is the control the event study was missing: if event days look like
    ordinary days, the event carries no measurable drift.

    Parameters
    ----------
    prices:
        Price history with a ``DatetimeIndex`` and forward-return columns.
    exclude_dates:
        Anchor dates of real events. Any trading day within
        :data:`EXCLUSION_WINDOW_DAYS` of one is excluded, so the null cannot
        accidentally sample the very days being tested.
    n_samples:
        Number of days to draw. Capped at the number of eligible days.
    seed:
        Seed for the random generator; the same seed and inputs always give the
        same sample.

    Returns
    -------
    pandas.DataFrame
        One row per sampled day with ``event_date`` and the forward returns.

    Raises
    ------
    ValueError
        If the index is not a ``DatetimeIndex``, a return column is missing, or
        ``n_samples`` is not positive.
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError(
            "prices must be indexed by a DatetimeIndex, got "
            f"{type(prices.index).__name__}."
        )
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    columns = forward_return_columns()
    missing = [column for column in columns if column not in prices.columns]
    if missing:
        raise ValueError(f"prices is missing forward-return columns: {missing}.")

    eligible = _eligible_days(prices.index, exclude_dates)
    if eligible.empty:
        return pd.DataFrame(columns=["event_date", *columns])

    rng = np.random.default_rng(seed)
    count = min(n_samples, len(eligible))
    chosen = rng.choice(eligible.to_numpy(), size=count, replace=False)
    chosen.sort()

    # ``_eligible_days`` strips the timezone for date arithmetic, so the sample
    # is looked up by position rather than by label: the labels are tz-aware
    # while the chosen dates are not. Positions must be taken against the
    # *original* index, because ``eligible`` is a filtered subset of it.
    all_days = prices.index.normalize()
    if all_days.tz is not None:
        all_days = all_days.tz_localize(None)

    positions = np.flatnonzero(all_days.isin(pd.DatetimeIndex(chosen)))
    sample = prices.iloc[positions][list(columns)].copy()
    sample.insert(0, "event_date", prices.index[positions])
    return sample.reset_index(drop=True)


def compare_to_null(
    clustered: pd.DataFrame,
    null: pd.DataFrame,
) -> pd.DataFrame:
    """Compare clustered event returns with the null distribution.

    For each event class and horizon, reports the event median, the null median
    and its 5th/95th percentiles, where the event median falls in the null
    distribution (``percentile``), and a two-sided empirical ``p_value``.

    The p-value is the share of null samples at least as extreme as the event
    median. It is an empirical, distribution-free quantity: no normality
    assumption, which matters because these returns are fat-tailed.

    Raises
    ------
    ValueError
        If either frame lacks the columns it needs.
    """
    columns = forward_return_columns()
    missing = [column for column in columns if column not in clustered.columns]
    if missing:
        raise ValueError(f"clustered is missing return columns: {missing}.")
    missing_null = [column for column in columns if column not in null.columns]
    if missing_null:
        raise ValueError(f"null is missing return columns: {missing_null}.")

    rows: list[dict[str, object]] = []

    for event_type, group in clustered.groupby("event_type", sort=True):
        for horizon, column in zip(FORWARD_RETURN_HORIZONS, columns, strict=True):
            event_values = group[column].dropna().to_numpy(dtype=float)
            null_values = null[column].dropna().to_numpy(dtype=float)
            rows.append(
                _compare_row(str(event_type), horizon, event_values, null_values)
            )

    return pd.DataFrame(rows, columns=list(NULL_COLUMNS))


def _compare_row(
    event_type: str,
    horizon: int,
    event_values: np.ndarray,
    null_values: np.ndarray,
) -> dict[str, object]:
    """One row of :func:`compare_to_null`; ``NaN`` where undefined."""
    empty = {
        "event_type": event_type,
        "horizon_days": horizon,
        "n_dates": int(event_values.size),
        "event_median": float("nan"),
        "null_median": float("nan"),
        "null_p05": float("nan"),
        "null_p95": float("nan"),
        "percentile": float("nan"),
        "p_value": float("nan"),
    }
    if event_values.size == 0 or null_values.size == 0:
        return empty

    event_median = float(np.median(event_values))
    null_median = float(np.median(null_values))
    p05, p95 = (float(value) for value in np.percentile(null_values, [5, 95]))

    # Two-sided empirical p-value: how often the null is at least as far from
    # its own median as the event median is.
    distance = abs(event_median - null_median)
    at_least_as_extreme = np.abs(null_values - null_median) >= distance
    p_value = float(at_least_as_extreme.mean())

    percentile = float((null_values < event_median).mean())

    return {
        "event_type": event_type,
        "horizon_days": horizon,
        "n_dates": int(event_values.size),
        "event_median": event_median,
        "null_median": null_median,
        "null_p05": p05,
        "null_p95": p95,
        "percentile": percentile,
        "p_value": p_value,
    }


def _eligible_days(
    index: pd.DatetimeIndex,
    exclude_dates: Sequence[pd.Timestamp],
) -> pd.DatetimeIndex:
    """Trading days that are not within the exclusion window of an event."""
    days = index.normalize()
    if days.tz is not None:
        days = days.tz_localize(None)

    if not exclude_dates:
        return days

    excluded = pd.DatetimeIndex(pd.to_datetime(list(exclude_dates))).normalize()
    if excluded.tz is not None:
        excluded = excluded.tz_localize(None)

    mask = np.ones(len(days), dtype=bool)
    for event_day in excluded:
        mask &= np.abs((days - event_day).days) > EXCLUSION_WINDOW_DAYS

    return days[mask]


def cluster_t_stat(values: np.ndarray) -> float:
    """t-statistic of the mean, treating each value as one observation.

    ``NaN`` when fewer than two values exist or they do not vary. Applied to
    :func:`cluster_by_date` output, this is the cluster-aware test: the sample
    size is the number of dates, not the number of events.
    """
    if values.size < 2:
        return float("nan")
    spread = float(values.std(ddof=1))
    if spread == 0.0:
        return float("nan")
    return float(values.mean()) / (spread / math.sqrt(values.size))
