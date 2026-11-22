"""Score the zero-shot model against the same labels as the rule baseline.

The rule baseline measured **0.055 precision** and **0.000 recall**. This module
answers the only question that matters next: does a model do better on the same
data?

Two things are scored, because they answer different questions:

* **Precision** on the 149 stratified detected events — of the events the model
  calls events, how many are real? Comparable to the rule baseline's 0.055.
* **Recall** on the 38 event-bearing passages — of the real events present, how
  many does the model find? Comparable to the rule baseline's 0.000.

The model is applied to the same text the reviewer saw, so the comparison is
like-for-like. Nothing is trained; the model is used zero-shot.

Results are written to ``docs/model-result.md`` by hand after a run, because a
number in a document is easier to reason about than one in a terminal.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

import pandas as pd

from app.evaluation.gold import NONE_LABEL, SKIP_LABEL, GoldSnippet, load_gold
from app.evaluation.precision import wilson_interval
from app.models.zero_shot import DEFAULT_MODEL, ZeroShotEventClassifier

#: Columns of :func:`score_predictions`.
SCORE_COLUMNS: Final = (
    "event_type",
    "labelled",
    "correct",
    "precision",
    "ci_low",
    "ci_high",
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
class Prediction:
    """One model prediction for one snippet."""

    snippet_id: str
    predicted: str
    label: str | None


def predict_all(
    snippets: Sequence[GoldSnippet],
    predict: Callable[[str], str],
) -> list[Prediction]:
    """Run ``predict`` over every snippet, keeping the human label alongside."""
    return [
        Prediction(
            snippet_id=snippet.id,
            predicted=predict(snippet.text),
            label=snippet.label,
        )
        for snippet in snippets
    ]


def precision_by_class(predictions: Sequence[Prediction]) -> pd.DataFrame:
    """Precision per predicted class, with Wilson intervals.

    A prediction is correct when it equals the human label. ``none`` counts as a
    real answer: predicting an event where there is none is an error, and
    predicting ``none`` where there is none is correct.
    """
    judged = [p for p in predictions if p.label is not None and p.label != SKIP_LABEL]
    if not judged:
        raise ValueError("No labelled snippets to score.")

    rows: list[dict[str, object]] = []
    for event_type in sorted({p.predicted for p in judged}):
        group = [p for p in judged if p.predicted == event_type]
        correct = sum(1 for p in group if p.label == event_type)
        low, high = wilson_interval(correct, len(group))
        rows.append(
            {
                "event_type": event_type,
                "labelled": len(group),
                "correct": correct,
                "precision": correct / len(group),
                "ci_low": low,
                "ci_high": high,
            }
        )

    return pd.DataFrame(rows, columns=list(SCORE_COLUMNS))


def recall_by_class(predictions: Sequence[Prediction]) -> pd.DataFrame:
    """Recall per labelled class, with Wilson intervals.

    A true positive is a snippet labelled ``X`` that the model also predicted as
    ``X``. A false negative is a snippet labelled ``X`` predicted as anything
    else — including ``none``.
    """
    judged = [p for p in predictions if p.label is not None and p.label != SKIP_LABEL]
    if not judged:
        raise ValueError("No labelled snippets to score.")

    rows: list[dict[str, object]] = []
    labelled_classes = {
        p.label for p in judged if p.label is not None and p.label != NONE_LABEL
    }
    for event_type in sorted(labelled_classes):
        group = [p for p in judged if p.label == event_type]
        hits = sum(1 for p in group if p.predicted == event_type)
        low, high = wilson_interval(hits, len(group))
        rows.append(
            {
                "event_type": event_type,
                "labelled": len(group),
                "correct": hits,
                "recall": hits / len(group),
                "ci_low": low,
                "ci_high": high,
            }
        )

    return pd.DataFrame(rows)


def pooled(predictions: Sequence[Prediction]) -> dict[str, float]:
    """Pooled precision and recall over all labelled snippets."""
    judged = [p for p in predictions if p.label is not None and p.label != SKIP_LABEL]
    if not judged:
        raise ValueError("No labelled snippets to score.")

    predicted_events = [p for p in judged if p.predicted != NONE_LABEL]
    correct_predictions = sum(1 for p in predicted_events if p.predicted == p.label)

    real_events = [p for p in judged if p.label != NONE_LABEL]
    found = sum(1 for p in real_events if p.predicted == p.label)

    precision_low, precision_high = wilson_interval(
        correct_predictions, len(predicted_events)
    )
    recall_low, recall_high = wilson_interval(found, len(real_events))

    return {
        "labelled": float(len(judged)),
        "predicted_events": float(len(predicted_events)),
        "precision": (
            correct_predictions / len(predicted_events) if predicted_events else 0.0
        ),
        "precision_low": precision_low,
        "precision_high": precision_high,
        "real_events": float(len(real_events)),
        "found": float(found),
        "recall": found / len(real_events) if real_events else 0.0,
        "recall_low": recall_low,
        "recall_high": recall_high,
    }


def attach_sidecar(snippets: Sequence[GoldSnippet], path: Path) -> list[GoldSnippet]:
    """Fill ``detected_type`` from a sidecar file, for the passage sample."""
    sidecar = path.with_suffix(".detected.json")
    if not sidecar.exists():
        return list(snippets)

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
    """Manual entry point: ``python -m app.models.score``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--passages", type=Path, default=DEFAULT_PASSAGES)
    args = parser.parse_args()

    classifier = ZeroShotEventClassifier.load(args.model)

    print(f"model: {args.model}")
    print("loading snippets ...")

    events = load_gold(args.events)
    passages = load_gold(args.passages)

    for name, snippets in (("events", events), ("passages", passages)):
        labelled = [s for s in snippets if s.label is not None]
        if not labelled:
            print(f"\n{name}: no labels in {args.events.name}; skipping")
            continue

        print(f"\n=== {name} ({len(labelled)} labelled) ===")
        predictions = predict_all(labelled, classifier.predict)
        summary = pooled(predictions)

        print(
            f"precision {summary['precision']:.3f} "
            f"[{summary['precision_low']:.3f}, {summary['precision_high']:.3f}] "
            f"over {int(summary['predicted_events'])} predicted event(s)"
        )
        print(
            f"recall    {summary['recall']:.3f} "
            f"[{summary['recall_low']:.3f}, {summary['recall_high']:.3f}] "
            f"over {int(summary['real_events'])} real event(s)"
        )

        print("\nprecision by predicted class:")
        print(
            precision_by_class(predictions).to_string(
                index=False, float_format=lambda value: f"{value:.3f}"
            )
        )

        if summary["real_events"]:
            print("\nrecall by labelled class:")
            print(
                recall_by_class(predictions).to_string(
                    index=False, float_format=lambda value: f"{value:.3f}"
                )
            )

    print(
        "\nCompare with the rule baseline: precision 0.055, recall 0.000 "
        "(docs/precision-result.md, docs/recall-result.md)."
    )


if __name__ == "__main__":
    main()
