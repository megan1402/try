# Event study methodology

How `make study-all` turns filings into a claim about returns, and what each
control is protecting against. Read this before quoting any number it prints.

## The question

> After an event of type *X* is filed, does the stock move differently from how
> it would have moved anyway?

Every part of that sentence needs a control. "Moves" needs a benchmark. "After"
needs a window that starts at the right instant. "Differently" needs a null.
"Would have moved anyway" needs the market's own behaviour on ordinary days.

## 1. Where the clock starts

EDGAR accepts filings after the close — `acceptance_datetime` is typically
20:00–22:00 UTC. The last price that could **not** have reacted to the news is
that session's close, so each event is anchored on the **last session whose
close precedes the acceptance timestamp**:

```
close(t0)  ->  close(t0 + h) / close(t0) - 1        t0 = anchor close
```

A 21:00 acceptance anchors on the same session; a 10:00 one anchors on the
previous session, so the whole of the following day counts as reaction.

Two mistakes this rules out:

- anchoring on the *next* session, which discards the overnight gap where most
  of the reaction happens;
- dating the event by `filing_date` and counting a return that begins *before*
  the news — look-ahead bias, i.e. trading on information you do not have.

Anchoring at or before the close does not claim you could have traded at that
close. It fixes the window in which the market digested the news.

## 2. Benchmark adjustment

A −1% median after *every* event class is a statement about the sample period,
not about events. For each event and horizon:

```
abnormal_h = return_h(stock) - return_h(benchmark)
```

The benchmark is aligned to the **same anchor date**, so both legs cover the
same window. If the anchor date is missing from the benchmark, the abnormal
return is `NaN` — never a silent fallback to the raw return.

Default benchmark: `SPY`. A single index does not remove sector rotation; an
equal-weight basket of the other issuers would, and is a reasonable next step.

## 3. Clustering by anchor date

Ten mega-cap issuers file within days of each other, so 827 events can sit on 52
distinct dates. Events sharing an anchor date share a return window and are
**not** independent trials. Treating them as such inflates every t-statistic by
roughly the square root of the cluster size.

So events are averaged within `(event_type, event_date)` first:

```
one observation per (event class, anchor date)
```

The sample size is then the number of dates, and `t_stat` is computed over those
dates. `n_dates` is printed next to every statistic for exactly this reason.

## 4. Null distribution

The remaining question is whether an event day differs from an ordinary day. For
each ticker, 1 000 trading days are drawn at random from the price history and
given the same forward returns, using the same horizons.

- **Seeded** (`DEFAULT_SEED = 20260101`) so a reported number is reproducible.
- **Event windows excluded** — any day within 30 calendar days of a real anchor
  date is ineligible, so the null cannot sample the days being tested.
- **Capped** at the number of eligible days; days are drawn without replacement.

The comparison reports the event median, the null median, the null's 5th/95th
percentiles, where the event median falls in the null distribution, and a
two-sided empirical p-value:

```
p = share of null draws at least as far from the null median as the event median
```

This is distribution-free — no normality assumption, which matters because
these returns are fat-tailed.

## 5. What counts as a result

A class is only called significant if **all** of the following hold:

1. the empirical p-value is small **after** benchmark adjustment;
2. the sample size is the number of **dates**, not events;
3. the effect is not explained by the null distribution;
4. the direction is stable across horizons rather than a single spike.

Anything else is reported as "no measurable effect in this sample", which is a
legitimate and useful finding.

## Known limits

- **Large caps only.** Ten US mega-caps, 2025–2026. No small caps, ADRs or
  delistings, so there is no survivorship correction and no breadth.
- **Sector moves are counted once per issuer.** A filing that moved all ten
  names appears ten times in the event count and once per date in the clustered
  count — but the dates themselves are still correlated.
- **Raw keyword events.** The detector is a regex baseline that never abstains
  (see `docs/eval-notes.md`), so many "events" are boilerplate. A null
  comparison cannot fix a mislabelled event; it can only show that the label
  carries no return signal.
- **Overlapping windows.** 20-day forward returns on dates a few days apart
  overlap. Clustering reduces but does not eliminate this.
- **No transaction costs, no liquidity, no borrow.** Nothing here is a strategy.

## Reproducing

```bash
make collect TICKERS="NVDA MSFT AAPL AMZN GOOGL META TSLA AVGO AMD INTC" LIMIT=8
make study-all                       # SPY benchmark, 1000 draws, fixed seed
make study-all BENCHMARK=QQQ SAMPLES=5000
```
