"""Tests for :mod:`app.quant.returns`.

Every test builds a tiny, hand-written DataFrame: no network access, so the
suite is fast and fully deterministic.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.quant.returns import add_forward_returns


def make_prices(closes: list[float]) -> pd.DataFrame:
    """Build a minimal price frame with a daily ``Date`` index."""
    index = pd.date_range("2024-01-01", periods=len(closes), freq="D", name="Date")
    return pd.DataFrame({"Close": closes}, index=index)


def test_adds_the_three_expected_columns():
    result = add_forward_returns(make_prices([100.0, 101.0, 102.0]))

    assert {"return_1d", "return_5d", "return_20d"} <= set(result.columns)


def test_return_1d_uses_the_next_day_price():
    # 100 -> 110 is +10%; 110 -> 99 is -10%.
    result = add_forward_returns(make_prices([100.0, 110.0, 99.0]))

    assert result["return_1d"].iloc[0] == pytest.approx(0.10)
    assert result["return_1d"].iloc[1] == pytest.approx(-0.10)


def test_returns_look_forward_not_backward():
    """Guards against accidentally using ``pct_change()`` (backward-looking)."""
    closes = [100.0, 110.0, 99.0]
    result = add_forward_returns(make_prices(closes))
    backward_looking = make_prices(closes)["Close"].pct_change()

    # Forward: 110 -> 99.
    assert result["return_1d"].iloc[1] == pytest.approx(-0.10)
    # Backward would have been 100 -> 110 (+10%). They must not agree.
    assert result["return_1d"].iloc[1] != pytest.approx(backward_looking.iloc[1])


def test_return_5d_uses_the_price_five_days_later():
    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 150.0]
    result = add_forward_returns(make_prices(closes))

    assert result["return_5d"].iloc[0] == pytest.approx(150.0 / 100.0 - 1.0)


def test_return_20d_uses_the_price_twenty_days_later():
    closes = [100.0] * 20 + [80.0]
    result = add_forward_returns(make_prices(closes))

    assert len(result) == 21
    assert result["return_20d"].iloc[0] == pytest.approx(-0.20)


def test_trailing_rows_are_nan_because_the_future_is_unknown():
    result = add_forward_returns(make_prices([100.0] * 21))

    assert pd.isna(result["return_1d"].iloc[-1])
    assert result["return_5d"].iloc[-5:].isna().all()
    assert result["return_20d"].iloc[-20:].isna().all()


def test_leading_rows_are_filled_for_valid_horizons():
    # With 21 rows, row 0 has a full 20-day future, so nothing should be NaN there.
    result = add_forward_returns(make_prices([100.0] * 21))

    assert result["return_1d"].iloc[0] == pytest.approx(0.0)
    assert result["return_5d"].iloc[0] == pytest.approx(0.0)
    assert result["return_20d"].iloc[0] == pytest.approx(0.0)


def test_does_not_mutate_the_input_frame():
    original = make_prices([100.0, 110.0, 121.0])
    snapshot = original.copy()

    add_forward_returns(original)

    pd.testing.assert_frame_equal(original, snapshot)


def test_index_and_row_order_are_preserved():
    original = make_prices([100.0, 110.0, 99.0])

    result = add_forward_returns(original)

    pd.testing.assert_index_equal(result.index, original.index)


def test_missing_close_column_raises_key_error():
    df = pd.DataFrame({"Open": [1.0, 2.0]})

    with pytest.raises(KeyError):
        add_forward_returns(df)
