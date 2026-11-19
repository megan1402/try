"""Build a stratified labelling set from the events the detector found.

Run ``python -m app.evaluation.sample_events`` to write
``tests/fixtures/events_precision.jsonl``: a seeded, class-balanced sample of
detected events, each carrying its filing, section, sentence and surrounding
context, and with **no** proposed label.

Why no proposed label: the reviewer is judging whether the detector was right.
Showing the detector's own answer would anchor that judgement and inflate the
measured precision. The class is still visible as ``detected_type`` because the
question being asked is "is this really a <class>?", which is unanswerable
without knowing what was claimed.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from app.evaluation.gold import save_gold
from app.evaluation.sampling import (
    DEFAULT_SEED,
    DetectedEvent,
    from_filing_event,
    sample_events,
    to_snippets,
)
from app.extraction.events import extract_events
from app.extraction.html import html_to_text
from app.extraction.sections import as_sections

#: Where the precision sample is written.
DEFAULT_OUTPUT: Path = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "events_precision.jsonl"
)


def collect_events() -> list[DetectedEvent]:
    """Extract every detected event from the local corpus."""
    from app.ingestion import store

    connection = store.connect()
    try:
        filings = store.get_filings(connection)
    finally:
        connection.close()

    events: list[DetectedEvent] = []
    for row in filings.itertuples(index=False):
        if not isinstance(row.local_path, str) or not row.local_path:
            continue
        path = Path(row.local_path)
        if not path.is_absolute():
            path = store.PROJECT_ROOT / path
        if not path.exists():
            continue

        text = html_to_text(path.read_text(encoding="utf-8"))
        sections = as_sections(text)
        as_of = _as_of(row.acceptance_datetime)

        for event in extract_events(sections, as_of=as_of):
            events.append(
                from_filing_event(
                    event,
                    ticker=_ticker_for(str(row.cik)),
                    accession_number=str(row.accession_number),
                    text=text,
                )
            )

    return events


def _as_of(value: object) -> datetime:
    """Parse an acceptance timestamp into a timezone-aware datetime."""
    return pd.Timestamp(str(value)).to_pydatetime()


def _ticker_for(cik: str) -> str:
    """Map a CIK back to a ticker using the index, falling back to the CIK."""
    from app.ingestion import store

    connection = store.connect()
    try:
        companies = store.get_companies(connection)
    finally:
        connection.close()

    match = companies[companies["cik"] == cik]
    return str(match.iloc[0]["ticker"]) if not match.empty else str(cik)


def main() -> None:
    """Manual entry point: ``python -m app.evaluation.sample_events``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total", type=int, default=150, help="sample size")
    parser.add_argument("--minimum", type=int, default=5, help="floor per class")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    events = collect_events()
    if not events:
        print("No events in the corpus; run `make collect` first.")
        return

    sample = sample_events(
        events, total=args.total, minimum=args.minimum, seed=args.seed
    )
    snippets = to_snippets(sample)
    save_gold(snippets, args.output)

    counts: dict[str, int] = {}
    for event in sample:
        counts[event.event_type] = counts.get(event.event_type, 0) + 1

    print(f"{len(events)} event(s) in the corpus -> {len(sample)} sampled")
    print(f"written to {args.output}")
    print("\nper class:")
    for event_type in sorted(counts):
        print(f"  {event_type:22} {counts[event_type]:4}")
    print(
        "\nNext: `make review-precision` to label them, then "
        "`make precision` for per-class precision with confidence intervals."
    )


if __name__ == "__main__":
    main()
