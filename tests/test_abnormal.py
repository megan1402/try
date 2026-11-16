"""Tests for :mod:`app.quant.abnormal` — benchmark-relative returns.

The cases that matter are the arithmetic (a benchmark move must come out of the
stock's), the calendar (both legs must span the *same* sessions) and the
interchangeability (the adjusted frame must feed the event study unchanged).
Everything is offline; the "market" is a hand-written DataFrame.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from app.quant.abnormal import (
    ABNORMAL_COLUMNS,
    add_abnormal_returns,
    daily_returns,
    market_adjusted_returns,
    normalise_benchmark,
)
from app.quant.event_study import attach_forward_returns, summarise_events


def make_prices(
    closes: list[float | None],
    *,
    start: str = "2024-01-01",
    tz: str | None = None,
) -> pd.DataFrame:
    index = pd.bdate_range(start, periods=len(closes), name="Date")
    if tz is not None:
        index = index.tz_localize(tz)
    return pd.DataFrame({"Close": closes}, index=index)


def make_prices_on(dates: list[str], closes: list[float]) -> pd.DataFrame:
    index = pd.DatetimeIndex(pd.to_datetime(dates), name="Date")
    return pd.DataFrame({"Close": closes}, index=index)


# --- Daily returns ---------------------------------------------------------


def test_daily_returns_use_the_previous_close():
    returns = daily_returns(make_prices([100.0, 110.0, 121.0]))

    assert pd.isna(returns.iloc[0])
    assert returns.iloc[1] == pytest.approx(0.10)
    assert returns.iloc[2] == pytest.approx(0.10)


def test_missing_prices_are_not_forward_filled():
    """A gap must stay a gap: padding it would fabricate a flat session."""
    returns = daily_returns(make_prices([100.0, None, 121.0]))

    assert returns.iloc[1:].isna().all()
    assert len(returns) == 3


def test_daily_returns_require_a_close_column():
    with pytest.raises(KeyError, match="Close"):
        daily_returns(pd.DataFrame({"Adj Close": [1.0]}))


# --- Market-adjusted daily returns ----------------------------------------


def test_abnormal_daily_return_subtracts_the_benchmark_move():
    result = market_adjusted_returns(
        make_prices([100.0, 110.0, 121.0]),
        make_prices([100.0, 104.0, 108.16]),
    )

    assert list(result.columns) == list(ABNORMAL_COLUMNS)
    assert result["abnormal_return"].iloc[1] == pytest.approx(0.10 - 0.04)
    assert result["abnormal_return"].iloc[2] == pytest.approx(0.10 - 0.04)


def test_identical_series_have_no_abnormal_return():
    prices = make_prices([100.0, 110.0, 121.0, 99.0])

    result = market_adjusted_returns(prices, prices.copy())

    assert result["abnormal_return"].iloc[1:].abs().max() == pytest.approx(0.0)


# --- Forward abnormal returns ---------------------------------------------


def test_forward_abnormal_returns_net_out_the_benchmark():
    asset = make_prices([100.0 * 1.01**i for i in range(21)])
    benchmark = make_prices([100.0 * 1.005**i for i in range(21)])

    result = add_abnormal_returns(asset, benchmark)

    assert result.loc[result.index[0], "return_1d"] == pytest.approx(1.01 - 1.005)
    assert result.loc[result.index[0], "return_5d"] == pytest.approx(1.01**5 - 1.005**5)
    assert result.loc[result.index[0], "return_20d"] == pytest.approx(
        1.01**20 - 1.005**20
    )


def test_forward_abnormal_returns_are_nan_where_the_future_is_missing():
    asset = make_prices([100.0 * 1.01**i for i in range(21)])
    benchmark = make_prices([100.0 * 1.005**i for i in range(21)])

    result = add_abnormal_returns(asset, benchmark)

    for horizon in (1, 5, 20):
        assert result[f"return_{horizon}d"].iloc[-horizon:].isna().all()


def test_adjusted_frame_exposes_both_closes():
    result = add_abnormal_returns(
        make_prices([100.0, 110.0]), make_prices([50.0, 51.0])
    )

    assert result["Close"].tolist() == [100.0, 110.0]
    assert result["Benchmark"].tolist() == [50.0, 51.0]
    assert result.index.name == "Date"


# --- Calendar alignment ----------------------------------------------------


def test_both_legs_span_the_same_sessions():
    """A benchmark session missing from the stock must not become a window.

    The benchmark leaps on Wednesday, a day the stock never traded. Measuring
    Tuesday-to-Thursday on both legs gives +4% for the benchmark; measuring
    Tuesday-to-Wednesday would give +92% and an absurd "abnormal" return.
    """
    asset = make_prices_on(
        ["2024-01-01", "2024-01-02", "2024-01-04"],
        [100.0, 110.0, 132.0],
    )
    benchmark = make_prices_on(
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        [100.0, 104.0, 200.0, 108.16],
    )

    result = add_abnormal_returns(asset, benchmark)
    tuesday = pd.Timestamp("2024-01-02")

    assert len(result) == 3
    assert pd.Timestamp("2024-01-03") not in result.index
    assert result.loc[tuesday, "return_1d"] == pytest.approx(0.20 - 0.04)


def test_missing_benchmark_dates_are_dropped_from_the_frame():
    asset = make_prices_on(
        ["2024-01-01", "2024-01-02", "2024-01-03"], [100.0, 110.0, 121.0]
    )
    benchmark = make_prices_on(["2024-01-01", "2024-01-03"], [100.0, 101.0])

    result = market_adjusted_returns(asset, benchmark)

    assert list(result.index) == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-03"),
    ]
    assert result["asset_return"].iloc[1] == pytest.approx(0.21)


def test_timezones_are_ignored_when_matching_days():
    """A UTC benchmark and a New York stock share calendar dates, not offsets."""
    closes = [100.0, 110.0, 121.0]

    result = market_adjusted_returns(
        make_prices(closes, tz="America/New_York"),
        make_prices(closes, tz="UTC"),
    )

    assert result["abnormal_return"].iloc[1:].abs().max() == pytest.approx(0.0)


def test_duplicate_dates_keep_the_last_bar():
    index = pd.DatetimeIndex(
        pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]), name="Date"
    )
    asset = pd.DataFrame({"Close": [100.0, 200.0, 210.0]}, index=index)
    benchmark = make_prices_on(["2024-01-02", "2024-01-03"], [100.0, 101.0])

    result = add_abnormal_returns(asset, benchmark)

    assert result.loc[pd.Timestamp("2024-01-02"), "Close"] == 200.0
    # 210 / 200 - 1 = +5% against the benchmark's +1%.
    assert result.loc[pd.Timestamp("2024-01-02"), "return_1d"] == pytest.approx(
        0.05 - 0.01
    )


# --- Input validation ------------------------------------------------------


def test_no_common_dates_raise():
    asset = make_prices_on(["2024-01-01"], [100.0])
    benchmark = make_prices_on(["2024-02-01"], [100.0])

    with pytest.raises(ValueError, match="share no trading dates"):
        add_abnormal_returns(asset, benchmark)


def test_a_range_index_is_rejected():
    asset = pd.DataFrame({"Close": [100.0, 110.0]})
    benchmark = make_prices([100.0, 101.0])

    with pytest.raises(ValueError, match="DatetimeIndex"):
        add_abnormal_returns(asset, benchmark)


def test_a_benchmark_without_a_close_is_rejected():
    benchmark = make_prices([100.0, 101.0]).rename(columns={"Close": "Last"})

    with pytest.raises(KeyError, match="benchmark_prices"):
        add_abnormal_returns(make_prices([100.0, 110.0]), benchmark)


def test_an_empty_history_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        add_abnormal_returns(make_prices([]), make_prices([100.0]))


def test_no_pandas_deprecation_warnings():
    prices = make_prices([100.0, 110.0, 121.0])

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        daily_returns(prices)
        add_abnormal_returns(prices, prices.copy())


# --- Benchmark argument ----------------------------------------------------


@pytest.mark.parametrize("value", [None, "", "   ", "none", "NONE", "off"])
def test_benchmark_can_be_disabled(value: str | None):
    assert normalise_benchmark(value) is None


@pytest.mark.parametrize(
    ("value", "expected"), [(" spy ", "SPY"), ("qqq", "QQQ"), ("IWM", "IWM")]
)
def test_benchmark_symbols_are_normalised(value: str, expected: str):
    assert normalise_benchmark(value) == expected


# --- Pipeline compatibility ------------------------------------------------


def test_adjusted_frame_feeds_the_event_study_unchanged():
    asset = make_prices([100.0 * 1.01**i for i in range(21)])
    benchmark = make_prices([100.0 * 1.005**i for i in range(21)])
    prices = add_abnormal_returns(asset, benchmark)
    events = pd.DataFrame(
        [{"event_type": "acquisition", "as_of": "2024-01-01T21:00:00Z"}]
    )

    summary = summarise_events(attach_forward_returns(events, prices))
    row = summary[
        (summary["event_type"] == "acquisition") & (summary["horizon_days"] == 1)
    ].iloc[0]

    assert row["n"] == 1
    assert row["mean"] == pytest.approx(0.01 - 0.005)
