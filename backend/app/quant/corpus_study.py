"""Cross-issuer event study over the whole local corpus.

The single-issuer study answers *"what happened to this company after its
events?"*. That is one observation per filing date and cannot support a claim.
This module answers the question the project actually asks — *"what usually
happens after events of this type?"* — by pooling every collected issuer.

The number to watch is **distinct event dates**, not events:

    two dozen events inside one filing are one observation
    two dozen events across two dozen filings are twenty-four

Pooling is where look-ahead discipline pays off. Each event was anchored on the
last close before its own `acceptance_datetime`, so pooling issuers does not
mix in any information the market did not have.

Limits worth stating in any write-up:

* large-cap US issuers only — no small caps, no ADRs, no delistings;
* one price series per ticker, so a filing that moved the whole sector is
  counted once per issuer rather than once;
* returns are raw, not market-adjusted. A benchmark-relative version is the
  next step, not this one.
"""

from __future__ import annotations

import argparse

import pandas as pd

from app.quant.event_study import (
    attach_forward_returns,
    events_from_index,
    independence_counts,
    summarise_events,
)


def main() -> None:
    """Manual entry point: ``python -m app.quant.corpus_study``."""
    from app.ingestion import store
    from app.ingestion.market import MarketDataError, get_market_data
    from app.quant.returns import add_forward_returns

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", default="2y", help="price history window")
    args = parser.parse_args()

    connection = store.connect()
    try:
        companies = store.get_companies(connection)
        if companies.empty:
            print("No companies in the index; run `make collect` first.")
            return

        frames: list[pd.DataFrame] = []
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
    print(summarise_events(joined).to_string(index=False, float_format="%.3f"))
    print(
        f"\nIndependent observations: {dates} distinct event date(s) across "
        f"{usable} event(s)."
    )
    print(
        "Still not a market-wide study: large caps only, raw returns, and "
        "sector-wide moves are counted once per issuer."
    )
    if failures:
        print(f"Price data unavailable for: {', '.join(failures)}")


if __name__ == "__main__":
    main()
