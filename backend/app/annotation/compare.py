"""Compare LLM annotations against human labels.

This is the Phase 1 experiment: can an LLM reproduce the human annotations?

What the numbers mean, and what they do not
-------------------------------------------
The 187 human labels are a **development set**, not a test set. The rubric was
tuned against them once (the descriptive-mention rule), and the rule baseline was
measured on them. So agreement here answers:

    "does the LLM agree with one human annotator, on a set that annotator's
     rubric was tuned on?"

That is a useful agreement check. It is **not** a validation of annotation
quality, and it is not a test-set result.

The dataset is also heavily imbalanced — 172 of 187 labels are ``none`` — so
accuracy is meaningless. A model that always answers ``none`` scores 0.92. Every
metric here is reported per class, with intervals, and the positive counts are
stated explicitly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

import pandas as pd

from app.evaluation.gold import NONE_LABEL, SKIP_LABEL, GoldSnippet
from app.evaluation.precision import wilson_interval

#: Columns of :func:`per_class_table`.
PER_CLASS_COLUMNS: Final[tuple[str, ...]] = (
    "event_type",
    "support",
    "predicted",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
)

#: Columns of :func:`binary_table`.
BINARY_COLUMNS: Final[tuple[str, ...]] = (
    "metric",
    "value",
    "ci_low",
    "ci_high",
    "n",
)


def _scored(
    snippets: Sequence[GoldSnippet],
    predictions: Mapping[str, str],
) -> list[tuple[str, str]]:
    """``(human label, model label)`` pairs for every scored snippet."""
    pairs: list[tuple[str, str]] = []
    for snippet in snippets:
        if snippet.label is None or snippet.label == SKIP_LABEL:
            continue
        predicted = predictions.get(snippet.id)
        if predicted is None:
            continue
        pairs.append((snippet.label, predicted))
    return pairs


def per_class_table(
    snippets: Sequence[GoldSnippet],
    predictions: Mapping[str, str],
) -> pd.DataFrame:
    """Precision, recall and F1 per class.

    Raises
    ------
    ValueError
        If nothing is scored.
    """
    pairs = _scored(snippets, predictions)
    if not pairs:
        raise ValueError("No scored snippets: check labels and predictions.")

    classes = sorted({label for label, _ in pairs} | {pred for _, pred in pairs})
    rows: list[dict[str, object]] = []

    for name in classes:
        support = sum(1 for label, _ in pairs if label == name)
        predicted = sum(1 for _, pred in pairs if pred == name)
        true_positives = sum(
            1 for label, pred in pairs if label == name and pred == name
        )
        precision = true_positives / predicted if predicted else float("nan")
        recall = true_positives / support if support else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall and not _isnan(precision) and not _isnan(recall)
            else float("nan")
        )
        rows.append(
            {
                "event_type": name,
                "support": support,
                "predicted": predicted,
                "tp": true_positives,
                "fp": predicted - true_positives,
                "fn": support - true_positives,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    return pd.DataFrame(rows, columns=list(PER_CLASS_COLUMNS))


def binary_table(
    snippets: Sequence[GoldSnippet],
    predictions: Mapping[str, str],
) -> pd.DataFrame:
    """Event-versus-none metrics, with Wilson intervals.

    This is the metric that matters most: the dataset is 92% ``none``, so
    per-class numbers for rare classes rest on a handful of items, while the
    binary question — "is this an event at all?" — has enough support to be
    interpretable.
    """
    pairs = _scored(snippets, predictions)
    if not pairs:
        raise ValueError("No scored snippets: check labels and predictions.")

    def is_event(label: str) -> bool:
        return label not in (NONE_LABEL, SKIP_LABEL)

    true_positives = sum(
        1 for label, pred in pairs if is_event(label) and is_event(pred)
    )
    false_positives = sum(
        1 for label, pred in pairs if not is_event(label) and is_event(pred)
    )
    false_negatives = sum(
        1 for label, pred in pairs if is_event(label) and not is_event(pred)
    )
    true_negatives = sum(
        1 for label, pred in pairs if not is_event(label) and not is_event(pred)
    )

    rows: list[dict[str, object]] = []

    def add(metric: str, numerator: int, denominator: int) -> None:
        low, high = wilson_interval(numerator, denominator)
        rows.append(
            {
                "metric": metric,
                "value": numerator / denominator if denominator else float("nan"),
                "ci_low": low,
                "ci_high": high,
                "n": denominator,
            }
        )

    add("precision", true_positives, true_positives + false_positives)
    add("recall", true_positives, true_positives + false_negatives)
    add("specificity", true_negatives, true_negatives + false_positives)
    add("accuracy", true_positives + true_negatives, len(pairs))

    return pd.DataFrame(rows, columns=list(BINARY_COLUMNS))


def confusion_table(
    snippets: Sequence[GoldSnippet],
    predictions: Mapping[str, str],
) -> pd.DataFrame:
    """Counts of ``(human label, model label)`` pairs."""
    pairs = _scored(snippets, predictions)
    counts: dict[tuple[str, str], int] = {}
    for label, pred in pairs:
        counts[(label, pred)] = counts.get((label, pred), 0) + 1

    rows = [
        {"human": label, "model": pred, "count": count}
        for (label, pred), count in sorted(counts.items())
    ]
    return pd.DataFrame(rows, columns=["human", "model", "count"])


def disagreements(
    snippets: Sequence[GoldSnippet],
    predictions: Mapping[str, str],
    *,
    limit: int = 20,
) -> list[dict[str, str]]:
    """Human/model disagreements, with the passage text for inspection."""
    by_id = {snippet.id: snippet for snippet in snippets}
    out: list[dict[str, str]] = []

    for snippet_id, predicted in predictions.items():
        snippet = by_id.get(snippet_id)
        if snippet is None or snippet.label is None or snippet.label == SKIP_LABEL:
            continue
        if snippet.label == predicted:
            continue
        out.append(
            {
                "id": snippet_id,
                "human": snippet.label,
                "model": predicted,
                "text": " ".join(snippet.text.split())[:200],
            }
        )
        if len(out) >= limit:
            break

    return out


def _isnan(value: float) -> bool:
    return value != value
