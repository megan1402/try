"""Tests for :mod:`app.quant.controls`.

Each control is tested against a hand-built case where the right answer is known
by construction: a benchmark that moves exactly with the stock must leave zero
abnormal return, a null drawn from the same series must be centred on zero, and
clustering must reduce the sample size to the number of dates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.quant.controls import (
    CLUSTER_COLUMNS,
    DEFAULT_SEED,
    EXCLUSION_WINDOW_DAYS,
    NULL_COLUMNS,
    abnormal_returns,
    cluster_by_date,
    cluster_t_stat,
    compare_to_null,
    null_distribution,
)
from app.quant.returns import add_forward_returns


def make_prices(closes: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    index = pd.bdate_range(start, periods=len(closes), name="Date")
    return add_forward_returns(pd.DataFrame({"Close": closes}, index=index))


def make_events(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def event_row(
    event_type: str,
    event_date: str,
    *,
    return_1d: float = 0.0,
    return_5d: float = 0.0,
    return_20d: float = 0.0,
) -> dict[str, object]:
    return {
        "event_type": event_type,
        "event_date": pd.Timestamp(event_date),
        "return_1d": return_1d,
        "return_5d": return_5d,
        "return_20d": return_20d,
    }


# --- 1. Benchmark adjustment ----------------------------------------------


def test_abnormal_return_is_zero_when_the_stock_tracks_the_benchmark():
    # The first bar has no forward return, so the anchor is the second bar.
    events = make_events([event_row("acquisition", "2024-01-02", return_1d=0.05)])
    benchmark = make_prices([100.0, 100.0, 105.0])

    result = abnormal_returns(events, benchmark)

    assert result.loc[0, "abnormal_1d"] == pytest.approx(0.0)


def test_abnormal_return_subtracts_the_benchmark_move():
    events = make_events([event_row("acquisition", "2024-01-02", return_1d=0.10)])
    benchmark = make_prices([100.0, 100.0, 104.0])  # +4% over the same window.

    result = abnormal_returns(events, benchmark)

    assert result.loc[0, "abnormal_1d"] == pytest.approx(0.10 - 0.04)


def test_abnormal_returns_cover_every_horizon():
    events = make_events(
        [
            event_row(
                "acquisition",
                "2024-01-02",
                return_1d=0.01,
                return_5d=0.02,
                return_20d=0.03,
            )
        ]
    )
    benchmark = make_prices([100.0] * 30)

    result = abnormal_returns(events, benchmark)

    assert {"abnormal_1d", "abnormal_5d", "abnormal_20d"} <= set(result.columns)
    assert result.loc[0, "abnormal_20d"] == pytest.approx(0.03)


def test_benchmark_is_aligned_on_the_anchor_date_not_the_row_order():
    # The benchmark's 2 Jan return is +10%; 3 Jan is -50%. If alignment were
    # positional the answer would be wildly different.
    benchmark = make_prices([100.0, 100.0, 110.0, 55.0])
    events = make_events([event_row("acquisition", "2024-01-02", return_1d=0.10)])

    result = abnormal_returns(events, benchmark)

    assert result.loc[0, "abnormal_1d"] == pytest.approx(0.0)


def test_missing_benchmark_date_yields_nan_not_the_raw_return():
    benchmark = make_prices([100.0, 110.0])
    events = make_events([event_row("acquisition", "2024-06-03", return_1d=0.10)])

    result = abnormal_returns(events, benchmark)

    assert pd.isna(result.loc[0, "abnormal_1d"])


def test_events_without_an_anchor_keep_nan():
    benchmark = make_prices([100.0, 110.0])
    events = make_events(
        [
            {
                "event_type": "acquisition",
                "event_date": pd.NaT,
                "return_1d": float("nan"),
                "return_5d": float("nan"),
                "return_20d": float("nan"),
            }
        ]
    )

    result = abnormal_returns(events, benchmark)

    assert pd.isna(result.loc[0, "abnormal_1d"])


def test_abnormal_returns_does_not_modify_the_input():
    events = make_events([event_row("acquisition", "2024-01-02", return_1d=0.05)])
    original = list(events.columns)

    abnormal_returns(events, make_prices([100.0, 105.0]))

    assert list(events.columns) == original


def test_abnormal_returns_requires_event_date():
    with pytest.raises(ValueError, match="event_date"):
        abnormal_returns(pd.DataFrame({"return_1d": [0.1]}), make_prices([1.0]))


def test_abnormal_returns_requires_a_datetime_benchmark_index():
    events = make_events([event_row("acquisition", "2024-01-02")])
    with pytest.raises(ValueError, match="DatetimeIndex"):
        abnormal_returns(events, pd.DataFrame({"return_1d": [0.0]}))


def test_abnormal_returns_requires_matching_columns():
    events = make_events([event_row("acquisition", "2024-01-02")])
    benchmark = make_prices([100.0, 110.0]).drop(columns=["return_20d"])

    with pytest.raises(ValueError, match="return_20d"):
        abnormal_returns(events, benchmark)


# --- 2. Clustering ---------------------------------------------------------


def test_clustering_collapses_events_on_the_same_date():
    events = make_events(
        [
            event_row("acquisition", "2024-01-02", return_1d=0.02),
            event_row("acquisition", "2024-01-02", return_1d=0.04),
            event_row("acquisition", "2024-01-03", return_1d=0.10),
        ]
    )

    clustered = cluster_by_date(events, return_prefix="return_")

    assert len(clustered) == 2
    first = clustered[clustered["event_date"] == pd.Timestamp("2024-01-02")].iloc[0]
    assert first["n_events"] == 2
    assert first["return_1d"] == pytest.approx(0.03)


def test_clustering_keeps_event_types_separate():
    events = make_events(
        [
            event_row("acquisition", "2024-01-02", return_1d=0.02),
            event_row("restructuring", "2024-01-02", return_1d=0.06),
        ]
    )

    clustered = cluster_by_date(events, return_prefix="return_")

    assert len(clustered) == 2
    assert set(clustered["event_type"]) == {"acquisition", "restructuring"}


def test_clustering_drops_events_without_an_anchor():
    events = make_events(
        [
            event_row("acquisition", "2024-01-02", return_1d=0.02),
            {
                "event_type": "acquisition",
                "event_date": pd.NaT,
                "return_1d": 0.5,
                "return_5d": 0.5,
                "return_20d": 0.5,
            },
        ]
    )

    clustered = cluster_by_date(events, return_prefix="return_")

    assert len(clustered) == 1
    assert clustered.loc[0, "n_events"] == 1


def test_clustering_has_the_documented_columns():
    events = make_events([event_row("acquisition", "2024-01-02")])

    clustered = cluster_by_date(events, return_prefix="return_")

    assert list(clustered.columns) == list(CLUSTER_COLUMNS)


def test_clustering_nothing_returns_an_empty_shaped_frame():
    events = make_events(
        [
            {
                "event_type": "acquisition",
                "event_date": pd.NaT,
                "return_1d": float("nan"),
                "return_5d": float("nan"),
                "return_20d": float("nan"),
            }
        ]
    )

    clustered = cluster_by_date(events, return_prefix="return_")

    assert clustered.empty
    assert list(clustered.columns) == list(CLUSTER_COLUMNS)


def test_clustering_requires_return_columns():
    events = make_events(
        [{"event_type": "acquisition", "event_date": pd.Timestamp("2024-01-02")}]
    )

    with pytest.raises(ValueError, match="return columns"):
        cluster_by_date(events, return_prefix="return_")


def test_cluster_t_stat_uses_dates_not_events():
    # Three events on one date and one on another: the clustered sample is 2.
    events = make_events(
        [
            event_row("acquisition", "2024-01-02", return_1d=0.02),
            event_row("acquisition", "2024-01-02", return_1d=0.02),
            event_row("acquisition", "2024-01-02", return_1d=0.02),
            event_row("acquisition", "2024-01-03", return_1d=0.06),
        ]
    )
    clustered = cluster_by_date(events, return_prefix="return_")

    values = clustered["return_1d"].to_numpy(dtype=float)

    assert values.size == 2
    assert cluster_t_stat(values) == pytest.approx(0.04 / (0.0282842712 / np.sqrt(2)))


def test_cluster_t_stat_is_nan_for_a_single_date():
    assert np.isnan(cluster_t_stat(np.array([0.05])))


def test_cluster_t_stat_is_nan_when_values_do_not_vary():
    assert np.isnan(cluster_t_stat(np.array([0.05, 0.05])))


# --- 3. Null distribution --------------------------------------------------


def test_null_distribution_is_reproducible_with_the_same_seed():
    prices = make_prices([100.0 + i for i in range(200)])

    first = null_distribution(prices, n_samples=50, seed=7)
    second = null_distribution(prices, n_samples=50, seed=7)

    pd.testing.assert_frame_equal(first, second)


def test_different_seeds_give_different_samples():
    prices = make_prices([100.0 + i for i in range(200)])

    first = null_distribution(prices, n_samples=50, seed=1)
    second = null_distribution(prices, n_samples=50, seed=2)

    assert not first["event_date"].equals(second["event_date"])


def test_null_distribution_respects_the_sample_size():
    prices = make_prices([100.0 + i for i in range(200)])

    sample = null_distribution(prices, n_samples=25)

    assert len(sample) == 25


def test_null_distribution_caps_at_the_number_of_eligible_days():
    prices = make_prices([100.0 + i for i in range(10)])

    sample = null_distribution(prices, n_samples=1_000)

    assert len(sample) == 10


def test_null_distribution_never_repeats_a_day():
    prices = make_prices([100.0 + i for i in range(200)])

    sample = null_distribution(prices, n_samples=100)

    assert sample["event_date"].is_unique


def test_null_distribution_excludes_days_near_events():
    prices = make_prices([100.0 + i for i in range(200)])
    event_day = prices.index[100]

    sample = null_distribution(
        prices, exclude_dates=[event_day], n_samples=1_000, seed=3
    )

    gaps = (sample["event_date"] - event_day).dt.days.abs()
    assert (gaps > EXCLUSION_WINDOW_DAYS).all()


def test_null_distribution_returns_the_forward_returns_of_the_sampled_days():
    prices = make_prices([100.0 + i for i in range(50)])

    sample = null_distribution(prices, n_samples=10, seed=5)

    for row in sample.itertuples(index=False):
        expected = prices.loc[row.event_date, "return_1d"]
        assert row.return_1d == pytest.approx(expected)


def test_null_distribution_works_with_a_timezone_aware_index():
    index = pd.bdate_range("2024-01-01", periods=120, tz="UTC", name="Date")
    prices = add_forward_returns(
        pd.DataFrame({"Close": [100.0 + i for i in range(120)]}, index=index)
    )

    sample = null_distribution(prices, n_samples=20, seed=11)

    assert len(sample) == 20
    assert sample["return_1d"].notna().all()
    # The sampled dates must be real bars from the original index.
    assert set(sample["event_date"]).issubset(set(prices.index))


def test_null_distribution_has_the_documented_columns():
    prices = make_prices([100.0 + i for i in range(50)])

    sample = null_distribution(prices, n_samples=5)

    assert list(sample.columns) == [
        "event_date",
        "return_1d",
        "return_5d",
        "return_20d",
    ]


def test_null_distribution_rejects_a_non_datetime_index():
    with pytest.raises(ValueError, match="DatetimeIndex"):
        null_distribution(pd.DataFrame({"return_1d": [0.0]}), n_samples=5)


def test_null_distribution_rejects_a_non_positive_sample_size():
    with pytest.raises(ValueError, match="positive"):
        null_distribution(make_prices([1.0, 2.0]), n_samples=0)


def test_null_distribution_requires_return_columns():
    prices = make_prices([1.0, 2.0]).drop(columns=["return_20d"])

    with pytest.raises(ValueError, match="return_20d"):
        null_distribution(prices, n_samples=1)


def test_default_seed_is_fixed():
    assert DEFAULT_SEED == 20260101


# --- Comparing events with the null ---------------------------------------


def test_compare_reports_the_documented_columns():
    clustered = pd.DataFrame(
        [
            {
                "event_type": "acquisition",
                "event_date": pd.Timestamp("2024-01-02"),
                "n_events": 1,
                "return_1d": 0.05,
                "return_5d": 0.0,
                "return_20d": 0.0,
            }
        ]
    )
    null = pd.DataFrame(
        {
            "event_date": pd.bdate_range("2024-02-01", periods=100),
            "return_1d": np.linspace(-0.05, 0.05, 100),
            "return_5d": np.zeros(100),
            "return_20d": np.zeros(100),
        }
    )

    table = compare_to_null(clustered, null)

    assert list(table.columns) == list(NULL_COLUMNS)
    assert table["horizon_days"].tolist() == [1, 5, 20]


def test_an_extreme_event_median_gets_a_small_p_value():
    clustered = pd.DataFrame(
        [
            {
                "event_type": "acquisition",
                "event_date": pd.Timestamp("2024-01-02"),
                "n_events": 1,
                "return_1d": 0.50,
                "return_5d": 0.0,
                "return_20d": 0.0,
            }
        ]
    )
    null = pd.DataFrame(
        {
            "event_date": pd.bdate_range("2024-02-01", periods=100),
            "return_1d": np.linspace(-0.05, 0.05, 100),
            "return_5d": np.zeros(100),
            "return_20d": np.zeros(100),
        }
    )

    row = compare_to_null(clustered, null).iloc[0]

    assert row["p_value"] == pytest.approx(0.0)
    assert row["percentile"] == pytest.approx(1.0)


def test_a_typical_event_median_gets_a_large_p_value():
    clustered = pd.DataFrame(
        [
            {
                "event_type": "acquisition",
                "event_date": pd.Timestamp("2024-01-02"),
                "n_events": 1,
                "return_1d": 0.0,
                "return_5d": 0.0,
                "return_20d": 0.0,
            }
        ]
    )
    null = pd.DataFrame(
        {
            "event_date": pd.bdate_range("2024-02-01", periods=100),
            "return_1d": np.linspace(-0.05, 0.05, 100),
            "return_5d": np.zeros(100),
            "return_20d": np.zeros(100),
        }
    )

    row = compare_to_null(clustered, null).iloc[0]

    assert row["p_value"] > 0.5


def test_compare_reports_the_number_of_dates():
    clustered = pd.DataFrame(
        [
            {
                "event_type": "acquisition",
                "event_date": pd.Timestamp("2024-01-02"),
                "n_events": 3,
                "return_1d": 0.01,
                "return_5d": 0.0,
                "return_20d": 0.0,
            },
            {
                "event_type": "acquisition",
                "event_date": pd.Timestamp("2024-01-03"),
                "n_events": 2,
                "return_1d": 0.02,
                "return_5d": 0.0,
                "return_20d": 0.0,
            },
        ]
    )
    null = pd.DataFrame(
        {
            "event_date": pd.bdate_range("2024-02-01", periods=10),
            "return_1d": np.zeros(10),
            "return_5d": np.zeros(10),
            "return_20d": np.zeros(10),
        }
    )

    row = compare_to_null(clustered, null).iloc[0]

    assert row["n_dates"] == 2


def test_compare_requires_return_columns():
    with pytest.raises(ValueError, match="return columns"):
        compare_to_null(pd.DataFrame({"event_type": ["x"]}), pd.DataFrame())
