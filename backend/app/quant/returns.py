"""Forward-return calculations.

A *forward* return measures what the price does **after** a point in time. For
a horizon of ``h`` trading days::

    forward_return_h[t] = Close[t + h] / Close[t] - 1

Forward returns are the quantity we will eventually try to explain with
financial events: given an event observed on day ``t``, what happened to the
stock over the following ``h`` days?

The last ``h`` rows of each column are ``NaN`` on purpose — their future does
not exist yet in the data. Leaving them ``NaN`` (rather than filling them) is
what prevents look-ahead bias.
"""

from __future__ import annotations

import pandas as pd

#: Forward horizons, in trading days, added by :func:`add_forward_returns`.
FORWARD_RETURN_HORIZONS: tuple[int, ...] = (1, 5, 20)


def add_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with forward-return columns added.

    Adds ``return_1d``, ``return_5d`` and ``return_20d``, each computed from the
    ``Close`` column as ``Close[t + h] / Close[t] - 1``.

    Parameters
    ----------
    df:
        Price history ordered oldest-row-first, containing a ``Close`` column.
        Typically the output of :func:`app.ingestion.market.get_market_data`.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame (``df`` itself is not modified) with the extra columns.
        Row order and index are preserved.

    Raises
    ------
    KeyError
        If ``df`` has no ``Close`` column.
    """
    if "Close" not in df.columns:
        raise KeyError(
            "add_forward_returns expects a 'Close' column; "
            f"got columns {list(df.columns)}."
        )

    # Work on a copy so the caller's DataFrame is never mutated.
    result = df.copy()
    close = result["Close"]

    for horizon in FORWARD_RETURN_HORIZONS:
        # shift(-h) pulls the price from h rows *later* up into the current row,
        # so each row is paired with a FUTURE price. The sign matters: a
        # positive shift (shift(+h)) would look backwards and leak information.
        future_close = close.shift(-horizon)
        result[f"return_{horizon}d"] = future_close / close - 1.0

    return result
