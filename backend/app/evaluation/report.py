"""Measure the rule baseline against the labelled snippets.

Precision answers *"when the detector fires, how often is it right?"*; recall
answers *"of the real events, how many did it find?"*. Neither can be computed
from the code alone, which is the whole point of :mod:`app.evaluation.gold`.

The snippet size matters when reading these numbers: one paragraph per snippet
makes recall look worse than a whole-section boundary would, because an event is
counted once per section there. Keep that in mind before comparing against a
paper.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import pandas as pd

from app.evaluation.gold import (
    DEFAULT_GOLD_PATH,
    NONE_LABEL,
    SKIP_LABEL,
    GoldSnippet,
    load_gold,
)
from app.extraction.events import extract_events
from app.extraction.sections import Section

#: Columns of the per-class table.
REPORT_COLUMNS: Final[tuple[str, ...]] = (
    "label",
    "support",
    "predicted",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
)

#: Column order of :func:`misclassified`.
ERROR_COLUMNS: Final[tuple[str, ...]] = (
    "id",
    "label",
    "predicted",
    "text",
)

# Predictions do not depend on the timestamp; the events only need *some*
# timezone-aware instant to satisfy the point-in-time contract.
_AS_OF: Final[datetime] = datetime(2024, 1, 1, tzinfo=UTC)

Predictor = Callable[[str], str]


def predict_label(text: str) -> str:
    """Predict one label for a snippet with the rule baseline.

    The rubric is one label per snippet, so several matches collapse to the
    highest-confidence type; ties break alphabetically to stay deterministic.
    """
    section = Section(
        part=None,
        item="1",
        title="snippet",
        start=0,
        end=len(text),
        text=text,
    )
    events = extract_events([section], as_of=_AS_OF)
    if not events:
        return NONE_LABEL

    best = sorted(events, key=lambda event: (-event.confidence, event.event_type))
    return best[0].event_type


def score(
    snippets: Sequence[GoldSnippet],
    predict: Predictor = predict_label,
) -> pd.DataFrame:
    """Per-class precision, recall and F1 over the scored snippets.

    Skipped and unlabelled snippets are excluded, so the table only reflects
    judgements a human actually made.

    Raises
    ------
    ValueError
        If nothing has been labelled yet.
    """
    judged: list[tuple[GoldSnippet, str]] = []
    for snippet in snippets:
        if snippet.label is None or snippet.label == SKIP_LABEL:
            continue
        judged.append((snippet, snippet.label))

    if not judged:
        raise ValueError(
            "No labelled snippets yet. Run `make review` to label some, then "
            f"score again ({DEFAULT_GOLD_PATH.name})."
        )

    predictions = {snippet.id: predict(snippet.text) for snippet, _ in judged}
    classes = sorted(
        {label for _, label in judged} | set(predictions.values()),
    )

    rows: list[dict[str, object]] = []
    for name in classes:
        support = sum(1 for _, label in judged if label == name)
        predicted = sum(1 for value in predictions.values() if value == name)
        true_positives = sum(
            1
            for snippet, label in judged
            if label == name and predictions[snippet.id] == name
        )

        precision = _ratio(true_positives, predicted)
        recall = _ratio(true_positives, support)
        rows.append(
            {
                "label": name,
                "support": support,
                "predicted": predicted,
                "tp": true_positives,
                "fp": predicted - true_positives,
                "fn": support - true_positives,
                "precision": precision,
                "recall": recall,
                "f1": _f1(precision, recall),
            }
        )

    return pd.DataFrame(rows, columns=list(REPORT_COLUMNS))


def macro_scores(table: pd.DataFrame) -> dict[str, float]:
    """Unweighted mean of precision/recall/F1 over classes with support."""
    with_support = table[table["support"] > 0]
    return {
        "macro_precision": float(with_support["precision"].mean()),
        "macro_recall": float(with_support["recall"].mean()),
        "macro_f1": float(with_support["f1"].mean()),
    }


def accuracy(
    snippets: Sequence[GoldSnippet],
    predict: Predictor = predict_label,
) -> float:
    """Share of scored snippets whose predicted label matches the human label."""
    judged = [
        snippet
        for snippet in snippets
        if snippet.label is not None and snippet.label != SKIP_LABEL
    ]
    if not judged:
        raise ValueError("No labelled snippets yet.")
    correct = sum(1 for snippet in judged if predict(snippet.text) == snippet.label)
    return correct / len(judged)


def misclassified(
    snippets: Sequence[GoldSnippet],
    predict: Predictor = predict_label,
) -> pd.DataFrame:
    """Every disagreement, so a bad number comes with its evidence."""
    rows = [
        {
            "id": snippet.id,
            "label": snippet.label,
            "predicted": predict(snippet.text),
            "text": " ".join(snippet.text.split())[:200],
        }
        for snippet in snippets
        if snippet.label is not None
        and snippet.label != SKIP_LABEL
        and predict(snippet.text) != snippet.label
    ]
    return pd.DataFrame(rows, columns=list(ERROR_COLUMNS))


def main() -> None:
    """Manual entry point: ``python -m app.evaluation.report [path]``."""
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_GOLD_PATH
    snippets = load_gold(path)

    if not any(snippet.label is not None for snippet in snippets):
        print(
            f"{path.name} has {len(snippets)} snippet(s) but none labelled yet. "
            "Run `make review` first."
        )
        return

    table = score(snippets)
    macro = macro_scores(table)

    print(f"\n{path.name}: {len(snippets)} snippet(s)")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(
        "\nmacro precision {macro_precision:.2f} | macro recall {macro_recall:.2f} "
        "| macro F1 {macro_f1:.2f} | accuracy {accuracy:.2f}".format(
            accuracy=accuracy(snippets), **macro
        )
    )

    errors = misclassified(snippets)
    if errors.empty:
        print("\nNo disagreements.")
        return

    print(f"\n{len(errors)} disagreement(s):")
    print(errors[["id", "label", "predicted"]].to_string(index=False))


def _ratio(numerator: int, denominator: int) -> float:
    """``numerator / denominator``, or NaN when the denominator is zero."""
    return numerator / denominator if denominator else float("nan")


def _f1(precision: float, recall: float) -> float:
    """Harmonic mean; NaN unless both components are defined."""
    if math.isnan(precision) or math.isnan(recall):
        return float("nan")
    total = precision + recall
    return 2 * precision * recall / total if total else 0.0


if __name__ == "__main__":
    main()
