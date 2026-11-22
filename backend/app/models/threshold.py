"""Sweep the zero-shot model's acceptance threshold.

The model's argmax always fires: some class is always highest-scoring, so a
snippet about ordinary results still gets an event label. That is why its
precision matched the base rate (0.055) while its recall looked better.

A threshold makes abstention possible. This module scores the model once, keeps
the raw scores, and then reports precision and recall at several cut-offs — so
the trade-off is visible rather than assumed.

The question it answers: **is there a threshold where the model is useful?**
If precision and recall both clear ~0.3 somewhere, the model is a viable
candidate generator. If no threshold works, its scores carry no usable signal.

Scores are computed once and reused across thresholds, because running the model
is the expensive part.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from app.evaluation.gold import NONE_LABEL, SKIP_LABEL, GoldSnippet, load_gold
from app.evaluation.precision import wilson_interval
from app.models.zero_shot import DEFAULT_MODEL, ZeroShotEventClassifier

#: Thresholds to sweep. Coarse on purpose: the point is the shape of the curve,
#: not a precise optimum that would not generalise from 149 items.
DEFAULT_THRESHOLDS: Final[tuple[float, ...]] = (
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
)

#: Columns of :func:`sweep`.
SWEEP_COLUMNS: Final[tuple[str, ...]] = (
    "threshold",
    "predicted_events",
    "precision",
    "precision_low",
    "precision_high",
    "recall",
    "recall_low",
    "recall_high",
)

#: Default inputs.
DEFAULT_EVENTS: Path = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "events_precision.jsonl"
)
DEFAULT_PASSAGES: Path = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "passages_events.jsonl"
)


@dataclass(frozen=True)
class ScoredSnippet:
    """A snippet with the model's per-class scores already computed."""

    label: str | None
    scores: dict[str, float]

    def best_event(self) -> tuple[str, float]:
        """The highest-scoring event class and its score."""
        events = {
            name: score for name, score in self.scores.items() if name != NONE_LABEL
        }
        if not events:
            return (NONE_LABEL, 0.0)
        return min(events.items(), key=lambda item: (-item[1], item[0]))

    def predict(self, threshold: float) -> str:
        """The label at ``threshold``: the best event, or ``none``."""
        label, score = self.best_event()
        return label if score >= threshold else NONE_LABEL


def score_snippets(
    snippets: Sequence[GoldSnippet],
    score_fn: Callable[[str], dict[str, float]],
) -> list[ScoredSnippet]:
    """Run the model once per snippet, keeping the raw scores."""
    return [
        ScoredSnippet(label=snippet.label, scores=score_fn(snippet.text))
        for snippet in snippets
    ]


def sweep(
    scored: Sequence[ScoredSnippet],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    """Precision and recall at each threshold.

    Precision is over predicted events; recall is over labelled real events.
    Both carry Wilson intervals, because 149 items cannot pin either down
    tightly.

    Raises
    ------
    ValueError
        If no snippet is labelled.
    """
    judged = [
        snippet
        for snippet in scored
        if snippet.label is not None and snippet.label != SKIP_LABEL
    ]
    if not judged:
        raise ValueError("No labelled snippets to sweep.")

    real_events = [snippet for snippet in judged if snippet.label != NONE_LABEL]

    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        predicted = [
            snippet for snippet in judged if snippet.predict(threshold) != NONE_LABEL
        ]
        correct = sum(
            1 for snippet in predicted if snippet.predict(threshold) == snippet.label
        )
        found = sum(
            1 for snippet in real_events if snippet.predict(threshold) == snippet.label
        )

        precision_low, precision_high = wilson_interval(correct, len(predicted))
        recall_low, recall_high = wilson_interval(found, len(real_events))

        rows.append(
            {
                "threshold": threshold,
                "predicted_events": len(predicted),
                "precision": correct / len(predicted) if predicted else 0.0,
                "precision_low": precision_low,
                "precision_high": precision_high,
                "recall": found / len(real_events) if real_events else 0.0,
                "recall_low": recall_low,
                "recall_high": recall_high,
            }
        )

    return pd.DataFrame(rows, columns=list(SWEEP_COLUMNS))


def best_threshold(
    table: pd.DataFrame,
    *,
    minimum_precision: float = 0.3,
    minimum_recall: float = 0.3,
) -> pd.Series | None:
    """The lowest threshold meeting both floors, or ``None``.

    Lowest rather than highest: a lower threshold keeps more recall for the same
    precision floor, which is what a candidate generator wants.
    """
    qualifying = table[
        (table["precision"] >= minimum_precision) & (table["recall"] >= minimum_recall)
    ]
    if qualifying.empty:
        return None
    return qualifying.sort_values("threshold").iloc[0]


def main() -> None:
    """Manual entry point: ``python -m app.models.threshold``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--passages", type=Path, default=DEFAULT_PASSAGES)
    args = parser.parse_args()

    classifier = ZeroShotEventClassifier.load(args.model)
    print(f"model: {args.model}")
    print("scoring snippets once, then sweeping thresholds ...")

    for name, path in (("events", args.events), ("passages", args.passages)):
        snippets = [s for s in load_gold(path) if s.label is not None]
        if not snippets:
            print(f"\n{name}: no labels; skipping")
            continue

        scored = score_snippets(snippets, classifier.scores)
        table = sweep(scored)

        print(f"\n=== {name} ({len(snippets)} labelled) ===")
        print(table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

        winner = best_threshold(table)
        if winner is None:
            print(
                "\nNo threshold reaches precision >= 0.3 and recall >= 0.3. "
                "The model's scores carry no usable signal on this data."
            )
        else:
            print(
                f"\nBest threshold: {winner['threshold']:.2f} "
                f"(precision {winner['precision']:.3f}, recall {winner['recall']:.3f})"
            )

    print("\nBaseline for comparison: rules precision 0.055, recall 0.000.")


if __name__ == "__main__":
    main()
