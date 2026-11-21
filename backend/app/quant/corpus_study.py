"""Cross-issuer event study over the whole local corpus, with controls.

The single-issuer study answers *"what happened to this company after its
events?"*. That is one observation per filing date and cannot support a claim.
This module answers the question the project actually asks — *"what usually
happens after events of this type?"* — by pooling every collected issuer, and
then applies the three controls from :mod:`app.quant.controls`:

1. **Benchmark adjustment** — every event return has the benchmark's return over
   the same window subtracted, so a market-wide move cannot masquerade as an
   event effect.
2. **Clustering** — events sharing an anchor date are averaged into one
   observation, so the sample size is the number of dates, not the number of
   events.
3. **Null distribution** — the same statistics on randomly chosen non-filing
   days, seeded, so "is this different from an ordinary day?" has an answer.

The number to watch is **distinct event dates**, not events:

    two dozen events inside one filing are one observation
    two dozen events across two dozen filings are twenty-four

Pooling is where look-ahead discipline pays off. Each event was anchored on the
last close before its own `acceptance_datetime`, and the benchmark is aligned to
that same anchor date, so no window mixes in information the market did not
have.

Limits worth stating in any write-up:

* large-cap US issuers only — no small caps, no ADRs, no delistings;
* one price series per ticker, so a filing that moved the whole sector is
  counted once per issuer rather than once;
* the benchmark is a single index, so a sector-wide move is only partly removed.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from app.quant.controls import (
    DEFAULT_SEED,
    abnormal_returns,
    cluster_by_date,
    cluster_t_stat,
    compare_to_null,
    null_distribution,
)
from app.quant.event_study import (
    attach_forward_returns,
    events_from_index,
    independence_counts,
    summarise_events,
)

#: Default benchmark for abnormal returns.
DEFAULT_BENCHMARK = "SPY"


def main() -> None:
    """Manual entry point: ``python -m app.quant.corpus_study``."""
    from app.ingestion import store
    from app.ingestion.market import MarketDataError, get_market_data
    from app.quant.returns import add_forward_returns

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", default="2y", help="price history window")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--samples", type=int, default=1_000, help="null draws")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--min-dates",
        type=int,
        default=10,
        help="skip classes with fewer independent dates than this",
    )
    args = parser.parse_args()

    # Check the index before touching the network: an empty corpus should say so
    # immediately, not after a failed price download.
    connection = store.connect()
    try:
        companies = store.get_companies(connection)
    finally:
        connection.close()

    if companies.empty:
        print("No companies in the index; run `make collect` first.")
        return

    try:
        benchmark_prices = add_forward_returns(
            get_market_data(args.benchmark, period=args.period)
        )
    except (MarketDataError, ValueError) as exc:
        print(f"Benchmark {args.benchmark} unavailable: {exc}")
        return

    connection = store.connect()
    try:
        frames: list[pd.DataFrame] = []
        nulls: list[pd.DataFrame] = []
        failures: list[str] = []

        for row in companies.itertuples(index=False):
            ticker = str(row.ticker)
            cik = str(row.cik)
            events, used = events_from_index(connection, cik=cik, ticker=ticker)
            if events.empty:
                print(f"  {ticker:6} {used:3} filing(s)  no events")
                continue

            try:
                prices = add_forward_returns(
                    get_market_data(ticker, period=args.period)
                )
            except (MarketDataError, ValueError) as exc:
                print(f"  {ticker:6} skipped: {exc}")
                failures.append(ticker)
                continue

            joined = attach_forward_returns(events, prices)
            dates, usable = independence_counts(joined)

            # The null must avoid the event windows, or it samples the very
            # days being tested.
            event_dates = joined["event_date"].dropna().unique()
            nulls.append(
                null_distribution(
                    prices,
                    exclude_dates=list(event_dates),
                    n_samples=args.samples,
                    seed=args.seed,
                )
            )

            print(
                f"  {ticker:6} {used:3} filing(s)  {len(events):3} event(s)  "
                f"{dates:3} independent date(s)"
            )
            frames.append(joined)
    finally:
        connection.close()

    if not frames:
        print("\nNothing to summarise.")
        return

    joined = pd.concat(frames, ignore_index=True)
    dates, usable = independence_counts(joined)

    print(f"\n{len(frames)} issuer(s), {len(joined)} event(s)")
    print("\n--- raw returns (no controls) ---")
    print(summarise_events(joined).to_string(index=False, float_format="%.3f"))

    adjusted = abnormal_returns(joined, benchmark_prices)
    clustered = cluster_by_date(adjusted)
    null = pd.concat(nulls, ignore_index=True)

    print(f"\n--- benchmark-adjusted, clustered by date (vs {args.benchmark}) ---")
    print(_cluster_table(clustered).to_string(index=False, float_format="%.3f"))

    print(f"\n--- event median vs null distribution ({len(null)} draws) ---")
    comparison = compare_to_null(clustered, null)
    print(comparison.to_string(index=False, float_format="%.3f"))

    print(
        f"\nIndependent observations: {dates} distinct event date(s) across "
        f"{usable} event(s)."
    )

    # A class with a handful of dates cannot support a claim, however small its
    # p-value looks. Say so explicitly rather than leaving it to the reader.
    underpowered = comparison[comparison["n_dates"] < args.min_dates]
    if not underpowered.empty:
        classes = sorted(underpowered["event_type"].unique())
        print(
            f"Underpowered (fewer than {args.min_dates} dates, not interpretable): "
            f"{', '.join(classes)}"
        )

    survivors = comparison[
        (comparison["p_value"] < 0.05) & (comparison["n_dates"] >= args.min_dates)
    ]
    if survivors.empty:
        print(
            "No class survives the controls at p < 0.05 with adequate sample "
            "size. That is the result, not a failure to find one."
        )
    else:
        print("Classes surviving the controls (p < 0.05, adequate sample):")
        print(
            survivors[
                ["event_type", "horizon_days", "n_dates", "event_median", "p_value"]
            ].to_string(index=False, float_format="%.3f")
        )

    if failures:
        print(f"Price data unavailable for: {', '.join(failures)}")


def _cluster_table(clustered: pd.DataFrame) -> pd.DataFrame:
    """Per-class summary of the clustered abnormal returns.

    ``n_dates`` is the sample size; ``t_stat`` uses those dates as independent
    observations, which is the whole point of clustering.
    """
    rows: list[dict[str, object]] = []

    for event_type, group in clustered.groupby("event_type", sort=True):
        for horizon in (1, 5, 20):
            values = group[f"return_{horizon}d"].dropna().to_numpy(dtype=float)
            rows.append(
                {
                    "event_type": event_type,
                    "horizon_days": horizon,
                    "n_dates": int(values.size),
                    "median": float(np.median(values)) if values.size else float("nan"),
                    "mean": float(values.mean()) if values.size else float("nan"),
                    "hit_rate": (
                        float((values > 0).mean()) if values.size else float("nan")
                    ),
                    "t_stat": cluster_t_stat(values),
                }
            )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
