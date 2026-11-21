"""Report recall and the confusion matrix from labelled passages.

Run ``python -m app.evaluation.recall_report`` after labelling the sample
produced by :mod:`app.evaluation.sample_passages`.

The detector's answers live in a sidecar file, so they are joined here rather
than being visible during labelling.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from app.evaluation.gold import GoldSnippet, load_gold
from app.evaluation.recall import confusion_table, recall_table

#: Default input: the passage sample.
DEFAULT_INPUT: Path = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "passages_recall.jsonl"
)


def attach_detections(path: Path) -> list[GoldSnippet]:
    """Load the sample and fill ``detected_type`` from the sidecar file.

    The sidecar is written by the sampler and never shown to the reviewer, so
    joining it here keeps the labelling blind.
    """
    snippets = load_gold(path)
    sidecar = path.with_suffix(".detected.json")
    if not sidecar.exists():
        raise SystemExit(
            f"Missing {sidecar.name}. Re-run `make sample-passages` to regenerate it."
        )

    answers: dict[str, list[str]] = json.loads(sidecar.read_text(encoding="utf-8"))
    return [
        replace(
            snippet,
            detected_type=",".join(answers[snippet.id])
            if answers.get(snippet.id)
            else None,
        )
        for snippet in snippets
    ]


def main() -> None:
    """Manual entry point: ``python -m app.evaluation.recall_report``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    snippets = attach_detections(args.path)
    labelled = [s for s in snippets if s.label is not None]
    if not labelled:
        print(
            f"{args.path.name} has {len(snippets)} passage(s) but none labelled. "
            "Run `make review-passages` first."
        )
        return

    table = recall_table(snippets, confidence=args.confidence)

    print(f"\n{args.path.name}: {len(labelled)}/{len(snippets)} labelled")
    print(f"recall per class ({args.confidence:.0%} Wilson intervals)\n")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    print("\nconfusion (human label x detector answer):")
    print(confusion_table(snippets).to_string(index=False))

    missed = table[table["false_negatives"] > 0]
    if not missed.empty:
        total_missed = int(missed["false_negatives"].sum())
        print(
            f"\n{total_missed} real event(s) the detector did not report. "
            "These are invisible to a sample drawn from detections."
        )

    print(
        "\nRecall is estimated from sampled passages, so it carries sampling "
        "error. With a few hundred passages the interval is wide."
    )


if __name__ == "__main__":
    main()
