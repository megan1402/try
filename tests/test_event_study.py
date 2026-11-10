"""Tests for :mod:`app.quant.event_study`.

The important cases are the anchoring ones: a filing accepted after the close
must anchor on that session's close, not the next one, or the overnight
reaction to the news is silently excluded.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.quant.event_study import (
    SUMMARY_COLUMNS,
    attach_forward_returns,
    summarise_events,
)
from app.quant.returns import add_forward_returns


def make_prices(closes: list[float]) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-01", periods=len(closes), name="Date")
    return add_forward_returns(pd.DataFrame({"Close": closes}, index=index))


def make_events(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["event_type", "as_of"])


# --- Anchoring -------------------------------------------------------------


def test_post_close_acceptance_anchors_on_that_sessions_close():
    prices = make_prices([100.0, 110.0, 121.0])
    events = make_events([("guidance_raise", "2024-01-01T21:00:00Z")])

    result = attach_forward_returns(events, prices)

    assert result.loc[0, "event_date"] == pd.Timestamp("2024-01-01")
    # The 100 -> 110 move is the reaction and must not be thrown away.
    assert result.loc[0, "return_1d"] == pytest.approx(0.10)


def test_pre_open_acceptance_anchors_on_the_previous_close():
    prices = make_prices([100.0, 110.0, 121.0])
    events = make_events([("guidance_raise", "2024-01-02T10:00:00Z")])

    result = attach_forward_returns(events, prices)

    assert result.loc[0, "event_date"] == pd.Timestamp("2024-01-01")


def test_weekend_event_anchors_on_friday():
    prices = make_prices([100.0] * 5)  # Mon 1 Jan .. Fri 5 Jan 2024.
    events = make_events([("guidance_raise", "2024-01-06T12:00:00Z")])

    result = attach_forward_returns(events, prices)

    assert result.loc[0, "event_date"] == pd.Timestamp("2024-01-05")


def test_event_after_the_history_has_no_anchor_or_returns():
    prices = make_prices([100.0, 110.0, 121.0])
    events = make_events([("guidance_raise", "2024-06-01T21:00:00Z")])

    result = attach_forward_returns(events, prices)

    assert pd.isna(result.loc[0, "event_date"])
    assert pd.isna(result.loc[0, "return_1d"])


def test_event_before_the_history_has_no_anchor_or_returns():
    prices = make_prices([100.0, 110.0, 121.0])
    events = make_events([("guidance_raise", "2023-12-01T21:00:00Z")])

    result = attach_forward_returns(events, prices)

    assert pd.isna(result.loc[0, "event_date"])


def test_returns_come_from_the_anchored_row():
    prices = make_prices([100.0, 110.0, 121.0, 121.0, 121.0, 121.0, 130.0])
    events = make_events([("guidance_raise", "2024-01-02T21:00:00Z")])

    result = attach_forward_returns(events, prices)

    # Anchored on 2 Jan: 5 trading days later is 9 Jan (index 6).
    expected = prices.loc[pd.Timestamp("2024-01-02"), "return_5d"]
    assert result.loc[0, "return_5d"] == pytest.approx(expected)
    assert result.loc[0, "return_5d"] == pytest.approx(130.0 / 110.0 - 1.0)


def test_forward_returns_stay_nan_at_the_end_of_the_history():
    prices = make_prices([100.0, 110.0, 121.0])
    events = make_events([("guidance_raise", "2024-01-03T21:00:00Z")])

    result = attach_forward_returns(events, prices)

    assert result.loc[0, "event_date"] == pd.Timestamp("2024-01-03")
    assert pd.isna(result.loc[0, "return_1d"])


# --- Frame behaviour -------------------------------------------------------


def test_original_events_are_not_modified():
    prices = make_prices([100.0, 110.0, 121.0])
    events = make_events([("guidance_raise", "2024-01-01T21:00:00Z")])

    attach_forward_returns(events, prices)

    assert list(events.columns) == ["event_type", "as_of"]


def test_extra_columns_and_order_are_preserved():
    prices = make_prices([100.0, 110.0, 121.0])
    events = make_events(
        [
            ("guidance_raise", "2024-01-01T21:00:00Z"),
            ("guidance_cut", "2024-01-02T21:00:00Z"),
        ]
    )

    result = attach_forward_returns(events, prices)

    assert result["event_type"].tolist() == ["guidance_raise", "guidance_cut"]
    assert list(result.index) == [0, 1]
    assert {"event_date", "return_1d", "return_5d", "return_20d"} <= set(result.columns)


def test_missing_as_of_is_rejected():
    with pytest.raises(ValueError, match="as_of"):
        attach_forward_returns(pd.DataFrame({"event_type": ["x"]}), make_prices([1.0]))


def test_non_datetime_index_is_rejected():
    prices = pd.DataFrame({"return_1d": [0.0]})
    with pytest.raises(ValueError, match="DatetimeIndex"):
        attach_forward_returns(
            make_events([("guidance_raise", "2024-01-01T21:00:00Z")]), prices
        )


def test_missing_return_columns_are_rejected():
    prices = make_prices([100.0, 110.0]).drop(columns=["return_20d"])
    with pytest.raises(ValueError, match="return_20d"):
        attach_forward_returns(
            make_events([("guidance_raise", "2024-01-01T21:00:00Z")]), prices
        )


# --- Summary ---------------------------------------------------------------


def summary_row(summary: pd.DataFrame, event_type: str, horizon: int) -> pd.Series:
    row = summary[
        (summary["event_type"] == event_type) & (summary["horizon_days"] == horizon)
    ]
    assert len(row) == 1
    return row.iloc[0]


def test_summary_reports_counts_and_statistics():
    events = pd.DataFrame(
        {
            "event_type": ["guidance_raise"] * 3,
            "return_1d": [0.10, -0.05, 0.02],
            "return_5d": [0.20, 0.10, 0.00],
            "return_20d": [float("nan")] * 3,
        }
    )

    row = summary_row(summarise_events(events), "guidance_raise", 1)

    assert row["n"] == 3
    assert row["median"] == pytest.approx(0.02)
    assert row["mean"] == pytest.approx((0.10 - 0.05 + 0.02) / 3)
    assert row["hit_rate"] == pytest.approx(2 / 3)
    assert row["t_stat"] > 0


def test_summary_leaves_out_missing_returns():
    events = pd.DataFrame(
        {
            "event_type": ["guidance_cut"] * 3,
            "return_1d": [0.05, float("nan"), float("nan")],
            "return_5d": [float("nan")] * 3,
            "return_20d": [float("nan")] * 3,
        }
    )

    summary = summarise_events(events)

    assert summary_row(summary, "guidance_cut", 1)["n"] == 1
    assert summary_row(summary, "guidance_cut", 5)["n"] == 0
    assert pd.isna(summary_row(summary, "guidance_cut", 5)["median"])


def test_summary_groups_by_event_type():
    events = pd.DataFrame(
        {
            "event_type": ["guidance_raise", "guidance_raise", "guidance_cut"],
            "return_1d": [0.02, 0.04, -0.06],
            "return_5d": [0.0, 0.0, 0.0],
            "return_20d": [0.0, 0.0, 0.0],
        }
    )

    summary = summarise_events(events)

    assert set(summary["event_type"]) == {"guidance_raise", "guidance_cut"}
    assert summary_row(summary, "guidance_raise", 1)["hit_rate"] == pytest.approx(1.0)
    assert summary_row(summary, "guidance_cut", 1)["hit_rate"] == pytest.approx(0.0)


def test_summary_t_stat_is_undefined_for_a_single_observation():
    events = pd.DataFrame(
        {
            "event_type": ["acquisition"],
            "return_1d": [0.03],
            "return_5d": [float("nan")],
            "return_20d": [float("nan")],
        }
    )

    assert pd.isna(summary_row(summarise_events(events), "acquisition", 1)["t_stat"])


def test_summary_t_stat_is_undefined_when_returns_do_not_vary():
    events = pd.DataFrame(
        {
            "event_type": ["acquisition"] * 2,
            "return_1d": [0.03, 0.03],
            "return_5d": [float("nan")] * 2,
            "return_20d": [float("nan")] * 2,
        }
    )

    assert pd.isna(summary_row(summarise_events(events), "acquisition", 1)["t_stat"])


def test_summary_has_a_stable_shape():
    events = pd.DataFrame(
        {
            "event_type": ["acquisition"],
            "return_1d": [0.03],
            "return_5d": [0.03],
            "return_20d": [0.03],
        }
    )

    summary = summarise_events(events)

    assert list(summary.columns) == list(SUMMARY_COLUMNS)
    assert summary["horizon_days"].tolist() == [1, 5, 20]


def test_summary_requires_return_columns():
    with pytest.raises(ValueError, match="return_1d"):
        summarise_events(pd.DataFrame({"event_type": ["acquisition"]}))
