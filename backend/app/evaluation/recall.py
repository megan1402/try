"""Recall, estimated from labelled passages.

Precision alone cannot tell you whether a detector is useful. A detector that
reports one event per filing and is right every time has precision 1.0 and
recall near zero. Recall is the other half:

    recall = of the real events in the corpus, how many did the detector report

It is estimated by sampling **passages** (see :mod:`app.evaluation.passages`),
labelling them, and comparing each label against what the detector found in that
same passage:

* label is an event, detector flagged it      -> true positive
* label is an event, detector flagged nothing -> **false negative**
* label is `none`, detector flagged something -> false positive
* label is `none`, detector flagged nothing   -> true negative

The false negatives are the whole point: they are invisible to any sample drawn
from detections, which is why every earlier number in this project was
precision-only.

This is an estimate from a sample, so it carries the same Wilson interval as
precision. With a few hundred passages the interval is wide; that is honest, not
a defect.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import pandas as pd

from app.evaluation.gold import NONE_LABEL, SKIP_LABEL, GoldSnippet
from app.evaluation.precision import DEFAULT_CONFIDENCE, wilson_interval

#: Columns of :func:`recall_table`.
RECALL_COLUMNS: Final[tuple[str, ...]] = (
    "event_type",
    "true_positives",
    "false_negatives",
    "recall",
    "ci_low",
    "ci_high",
)

#: Columns of :func:`confusion_table`.
CONFUSION_COLUMNS: Final[tuple[str, ...]] = (
    "labelled",
    "detected",
    "count",
)


def recall_table(
    snippets: Sequence[GoldSnippet],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
) -> pd.DataFrame:
    """Recall per event class, estimated from labelled passages.

    A passage counts as a true positive for class ``X`` when the human labelled
    it ``X`` **and** the detector reported ``X`` in it. A false negative is a
    passage labelled ``X`` that the detector did not report as ``X``.

    Parameters
    ----------
    snippets:
        Labelled passages. ``detected_type`` must hold the detector's answer for
        the passage — a comma-separated list when it reported several classes,
        or ``None`` when it reported nothing.
    confidence:
        Two-sided confidence level for the Wilson intervals.

    Returns
    -------
    pandas.DataFrame
        One row per class that appears in either the labels or the detections.

    Raises
    ------
    ValueError
        If no passage is labelled.
    """
    judged = [
        snippet
        for snippet in snippets
        if snippet.label is not None and snippet.label != SKIP_LABEL
    ]
    if not judged:
        raise ValueError(
            "No labelled passages. Label some with `make review-passages`."
        )

    labelled_classes = {
        snippet.label
        for snippet in judged
        if snippet.label is not None and snippet.label != NONE_LABEL
    }
    detected_classes = {
        detected for snippet in judged for detected in _detected_set(snippet)
    }
    classes = sorted(labelled_classes | detected_classes)

    rows: list[dict[str, object]] = []
    for event_type in classes:
        true_positives = sum(
            1
            for snippet in judged
            if snippet.label == event_type and event_type in _detected_set(snippet)
        )
        false_negatives = sum(
            1
            for snippet in judged
            if snippet.label == event_type and event_type not in _detected_set(snippet)
        )
        positives = true_positives + false_negatives
        low, high = wilson_interval(true_positives, positives, confidence=confidence)
        rows.append(
            {
                "event_type": event_type,
                "true_positives": true_positives,
                "false_negatives": false_negatives,
                "recall": true_positives / positives if positives else float("nan"),
                "ci_low": low,
                "ci_high": high,
            }
        )

    return pd.DataFrame(rows, columns=list(RECALL_COLUMNS))


def confusion_table(snippets: Sequence[GoldSnippet]) -> pd.DataFrame:
    """Counts of ``(human label, detector answer)`` pairs over labelled passages.

    The detector answer is ``"none"`` when it reported nothing, so the table
    reads directly: the ``none`` row shows what the detector missed.
    """
    judged = [
        snippet
        for snippet in snippets
        if snippet.label is not None and snippet.label != SKIP_LABEL
    ]

    counts: dict[tuple[str, str], int] = {}
    for snippet in judged:
        detected = _detected_set(snippet)
        answer = ",".join(sorted(detected)) if detected else NONE_LABEL
        key = (str(snippet.label), answer)
        counts[key] = counts.get(key, 0) + 1

    rows = [
        {"labelled": labelled, "detected": detected, "count": count}
        for (labelled, detected), count in sorted(counts.items())
    ]
    return pd.DataFrame(rows, columns=list(CONFUSION_COLUMNS))


def _detected_set(snippet: GoldSnippet) -> set[str]:
    """The detector's classes for a passage, parsed from ``detected_type``."""
    if not snippet.detected_type:
        return set()
    return {
        part.strip()
        for part in snippet.detected_type.split(",")
        if part.strip() and part.strip() != NONE_LABEL
    }
