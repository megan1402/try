"""Report per-class precision with confidence intervals.

Run ``python -m app.evaluation.precision_report`` after labelling the sample
produced by :mod:`app.evaluation.sample_events`.

The output is deliberately blunt about sample size: a class with fewer than
:data:`app.evaluation.precision.MIN_INTERPRETABLE` labelled items is flagged,
because a precision of 1.0 from five items is not evidence of a good detector.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.evaluation.gold import load_gold
from app.evaluation.precision import (
    MIN_INTERPRETABLE,
    pooled_precision,
    precision_table,
)

#: Default input: the precision sample.
DEFAULT_INPUT: Path = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "events_precision.jsonl"
)


def main() -> None:
    """Manual entry point: ``python -m app.evaluation.precision_report``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    snippets = load_gold(args.path)
    labelled = [s for s in snippets if s.label is not None]
    if not labelled:
        print(
            f"{args.path.name} has {len(snippets)} snippet(s) but none labelled. "
            "Run `make review-precision` first."
        )
        return

    table = precision_table(snippets, confidence=args.confidence)
    pooled = pooled_precision(snippets, confidence=args.confidence)

    print(f"\n{args.path.name}: {len(labelled)}/{len(snippets)} labelled")
    print(f"precision per class ({args.confidence:.0%} Wilson intervals)\n")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    print(
        f"\npooled precision {pooled['precision']:.3f} "
        f"[{pooled['ci_low']:.3f}, {pooled['ci_high']:.3f}] "
        f"over {int(pooled['labelled'])} item(s)"
    )

    small = table[table["labelled"] < MIN_INTERPRETABLE]
    if not small.empty:
        classes = ", ".join(small["event_type"])
        print(
            f"\nToo few labels to interpret (< {MIN_INTERPRETABLE}): {classes}."
            "\nWiden the interval, not the claim."
        )

    print(
        "\nThis measures precision only. Recall needs a sample of filing passages "
        "the detector did not flag, which this set cannot provide."
    )


if __name__ == "__main__":
    main()
