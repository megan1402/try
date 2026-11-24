"""Run Phase 1: annotate the human-labelled passages with an LLM and compare.

    python -m app.annotation.run_phase1 --dry-run     # no API calls
    python -m app.annotation.run_phase1               # real run

``--dry-run`` uses a stub transport that always answers ``none``. It exists so
the pipeline can be exercised end to end without spending money, and so the
comparison arithmetic can be checked against a known-bad annotator.

``--input`` points the run at a different labelled set, which is how the
development partition is scored:

    python -m app.annotation.run_phase1 --input ../tests/fixtures/holdout_development.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.annotation.batch import (
    DEFAULT_OUTPUT,
    annotate_batch,
    load_annotations,
    predicted_labels,
)
from app.annotation.compare import (
    binary_table,
    confusion_table,
    disagreements,
    per_class_table,
)
from app.annotation.llm import (
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_MODEL,
    LlmAnnotator,
    build_transport,
)
from app.evaluation.gold import load_gold

#: The two human-labelled sets. Both are development sets, not test sets.
DEFAULT_INPUTS: tuple[Path, ...] = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "events_precision.jsonl",
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "passages_events.jsonl",
)


def stub_transport(prompt: str) -> str:
    """Always answer ``none``.

    A deliberately useless annotator: it establishes the floor. Any real model
    that cannot beat this is not worth its cost.
    """
    return json.dumps(
        {
            "label": 10,
            "event_type": "none",
            "is_genuine_event": False,
            "event_date": None,
            "evidence": "stub",
            "reason": "dry run",
            "requires_review": False,
        }
    )


def main() -> None:
    """Manual entry point: ``python -m app.annotation.run_phase1``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=("deepseek", "anthropic"),
        default="deepseek",
        help="which API to call (default: deepseek)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"model id (default: {DEFAULT_DEEPSEEK_MODEL} or {DEFAULT_MODEL})",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--input",
        type=Path,
        nargs="*",
        default=None,
        help="labelled set(s) to annotate (default: the two development sets)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="use a stub annotator that always answers none",
    )
    args = parser.parse_args()

    inputs = tuple(args.input) if args.input else DEFAULT_INPUTS
    snippets = []
    for path in inputs:
        snippets.extend(load_gold(path))

    print(f"{len(snippets)} labelled passage(s) across {len(inputs)} set(s)")

    if args.dry_run:
        annotator = LlmAnnotator(transport=stub_transport, attempts=1)
        model = "stub-none"
    else:
        transport = build_transport(args.provider, args.model)
        annotator = LlmAnnotator(transport=transport)
        model = args.model or (
            DEFAULT_DEEPSEEK_MODEL if args.provider == "deepseek" else DEFAULT_MODEL
        )
        print(f"provider: {args.provider}, model: {model}")

    annotated, failed = annotate_batch(
        snippets,
        annotator=annotator,
        model=model,
        output=args.output,
        sampling_method="human_labelled_development_set",
        limit=args.limit,
    )
    print(f"\nannotated {annotated}, failed {failed}")

    records = load_annotations(args.output)
    predictions = predicted_labels(records)

    print("\n--- event vs none ---")
    print(
        binary_table(snippets, predictions).to_string(
            index=False, float_format=lambda v: f"{v:.3f}"
        )
    )

    print("\n--- per class ---")
    print(
        per_class_table(snippets, predictions).to_string(
            index=False, float_format=lambda v: f"{v:.3f}"
        )
    )

    print("\n--- confusion (human x model) ---")
    print(confusion_table(snippets, predictions).to_string(index=False))

    print("\n--- disagreements ---")
    for row in disagreements(snippets, predictions, limit=10):
        print(f"  {row['id']}  human={row['human']}  model={row['model']}")
        print(f"    {row['text']}")

    print(
        "\nReminder: these are development-set numbers. The rubric was tuned on "
        "this data, so agreement here is not a test-set result."
    )


if __name__ == "__main__":
    main()
