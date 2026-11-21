"""Build a rule-blind passage sample and attach the detector's answers.

Run ``python -m app.evaluation.sample_passages`` to write
``tests/fixtures/passages_recall.jsonl``.

The sample is drawn from filing prose with **no reference to the detector**, so
it can measure recall. The detector's answer for each passage is stored in a
sidecar file rather than in the snippet, so the reviewer never sees it while
labelling — otherwise the recall estimate would be circular.

    tests/fixtures/passages_recall.jsonl          # what the reviewer sees
    tests/fixtures/passages_recall.detected.json  # what the detector said
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from app.evaluation.gold import save_gold
from app.evaluation.passages import (
    DEFAULT_SEED,
    Passage,
    passages_per_document,
    sample_passages,
    to_snippets,
)
from app.extraction.events import extract_events
from app.extraction.html import html_to_text
from app.extraction.sections import as_sections

#: Where the passage sample is written.
DEFAULT_OUTPUT: Path = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "passages_recall.jsonl"
)


def collect_documents() -> list[tuple[str, str, str, str]]:
    """Return ``(ticker, accession, section_key, text)`` for every local filing."""
    from app.ingestion import store

    connection = store.connect()
    try:
        filings = store.get_filings(connection)
        companies = store.get_companies(connection)
    finally:
        connection.close()

    tickers = dict(zip(companies["cik"], companies["ticker"], strict=True))
    documents: list[tuple[str, str, str, str]] = []

    for row in filings.itertuples(index=False):
        if not isinstance(row.local_path, str) or not row.local_path:
            continue
        path = Path(row.local_path)
        if not path.is_absolute():
            path = store.PROJECT_ROOT / path
        if not path.exists():
            continue

        documents.append(
            (
                tickers.get(str(row.cik), str(row.cik)),
                str(row.accession_number),
                str(row.form),
                html_to_text(path.read_text(encoding="utf-8")),
            )
        )

    return documents


def detector_answers(passages: list[Passage]) -> dict[str, list[str]]:
    """Run the detector over each sampled passage and record what it found.

    The detector is applied to the passage text alone, which is the honest
    comparison: a reviewer judging a passage is also seeing only that passage.
    """
    answers: dict[str, list[str]] = {}

    for index, passage in enumerate(passages, start=1):
        sections = as_sections(passage.text)
        events = extract_events(sections, as_of=datetime(2026, 1, 1, tzinfo=UTC))
        answers[f"ps{index:04d}"] = sorted({event.event_type for event in events})

    return answers


def main() -> None:
    """Manual entry point: ``python -m app.evaluation.sample_passages``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total", type=int, default=300, help="target passages")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    documents = collect_documents()
    if not documents:
        print("No local filings; run `make collect` first.")
        return

    per_document = passages_per_document(args.total, len(documents))
    passages = sample_passages(documents, per_document=per_document, seed=args.seed)
    if not passages:
        print("No usable passages found.")
        return

    snippets = to_snippets(passages)
    save_gold(snippets, args.output)

    answers = detector_answers(passages)
    sidecar = args.output.with_suffix(".detected.json")
    sidecar.write_text(json.dumps(answers, indent=2) + "\n", encoding="utf-8")

    flagged = sum(1 for value in answers.values() if value)
    print(f"{len(documents)} document(s) -> {len(passages)} passage(s)")
    print(f"written to {args.output}")
    print(f"detector answers in {sidecar.name}")
    print(
        f"\ndetector flagged {flagged}/{len(passages)} passage(s) "
        f"({flagged / len(passages):.1%})"
    )
    print("\nNext: `make review-passages` to label them, then `make recall`.")


if __name__ == "__main__":
    main()
