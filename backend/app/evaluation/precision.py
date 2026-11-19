"""Precision per event class, with confidence intervals.

Precision answers one question: **of the events the detector reported, what
share are real?** It is estimated from a sample of detected events, so every
number here is an estimate with uncertainty attached — and with 5–20 items per
class, that uncertainty is large. Reporting a bare "0.80" would imply a
precision the sample cannot support.

The interval is **Wilson's score interval**, not the textbook normal
approximation. With small samples and proportions near 0 or 1 the normal
approximation produces intervals that extend below 0 or above 1 and badly
understate the uncertainty; Wilson stays inside [0, 1] and behaves sensibly at
the extremes. That matters here because a class with 5 items and 5 correct
answers has a true precision that could plausibly be anywhere from ~0.57 to 1.0.

What this module does **not** measure
-------------------------------------
Recall. A sample of detected events cannot contain the events the detector
missed, so recall is not estimable from it. See
:mod:`app.evaluation.sampling` for what measuring recall would require.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

import pandas as pd

from app.evaluation.gold import SKIP_LABEL, GoldSnippet

#: Columns of :func:`precision_table`.
PRECISION_COLUMNS: Final[tuple[str, ...]] = (
    "event_type",
    "labelled",
    "correct",
    "precision",
    "ci_low",
    "ci_high",
    "ci_width",
)

#: Two-sided confidence level for the intervals.
DEFAULT_CONFIDENCE: Final[float] = 0.95

#: Below this many labelled items a class is flagged as too small to interpret.
MIN_INTERPRETABLE: Final[int] = 10


def wilson_interval(
    successes: int,
    trials: int,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Parameters
    ----------
    successes:
        Number of correct labels.
    trials:
        Number of labelled items.
    confidence:
        Two-sided confidence level, e.g. ``0.95``.

    Returns
    -------
    tuple[float, float]
        ``(low, high)``, both inside ``[0, 1]``. ``(nan, nan)`` when
        ``trials`` is zero.

    Raises
    ------
    ValueError
        If the counts are inconsistent or ``confidence`` is not in ``(0, 1)``.

    Examples
    --------
    >>> low, high = wilson_interval(5, 5)
    >>> round(low, 2), round(high, 2)
    (0.57, 1.0)
    """
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError(
            f"Invalid counts: {successes} successes out of {trials} trials."
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}.")
    if trials == 0:
        return (float("nan"), float("nan"))

    z = _z_score(confidence)
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (proportion + z * z / (2 * trials)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials**2))
        / denominator
    )

    return (max(0.0, centre - margin), min(1.0, centre + margin))


def precision_table(
    snippets: Sequence[GoldSnippet],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
) -> pd.DataFrame:
    """Precision per detected class, with Wilson intervals.

    A snippet counts as **correct** when the human label equals the class the
    detector assigned. ``none`` is a legitimate label and counts as an error,
    because the detector claimed an event that was not there.

    Snippets that are unlabelled or skipped are excluded, so the table only
    reflects judgements a human actually made.

    Raises
    ------
    ValueError
        If no snippet carries a ``detected_type`` or none is labelled.
    """
    judged = [
        snippet
        for snippet in snippets
        if snippet.detected_type is not None
        and snippet.label is not None
        and snippet.label != SKIP_LABEL
    ]
    if not judged:
        raise ValueError(
            "No labelled detected events. Label some with `make review` first."
        )

    rows: list[dict[str, object]] = []
    detected_types = sorted(
        {snippet.detected_type for snippet in judged if snippet.detected_type}
    )
    for event_type in detected_types:
        group = [s for s in judged if s.detected_type == event_type]
        correct = sum(1 for snippet in group if snippet.label == event_type)
        low, high = wilson_interval(correct, len(group), confidence=confidence)
        rows.append(
            {
                "event_type": event_type,
                "labelled": len(group),
                "correct": correct,
                "precision": correct / len(group),
                "ci_low": low,
                "ci_high": high,
                "ci_width": high - low,
            }
        )

    return pd.DataFrame(rows, columns=list(PRECISION_COLUMNS))


def pooled_precision(
    snippets: Sequence[GoldSnippet],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, float]:
    """Precision over all labelled detected events, ignoring class.

    Useful as a headline, but it is dominated by whichever class was sampled
    most, so it should never replace the per-class table.
    """
    judged = [
        snippet
        for snippet in snippets
        if snippet.detected_type is not None
        and snippet.label is not None
        and snippet.label != SKIP_LABEL
    ]
    if not judged:
        raise ValueError("No labelled detected events.")

    correct = sum(1 for snippet in judged if snippet.label == snippet.detected_type)
    low, high = wilson_interval(correct, len(judged), confidence=confidence)
    return {
        "labelled": float(len(judged)),
        "correct": float(correct),
        "precision": correct / len(judged),
        "ci_low": low,
        "ci_high": high,
    }


def _z_score(confidence: float) -> float:
    """Two-sided normal quantile for ``confidence``.

    Uses the inverse error function so no SciPy dependency is needed.
    """
    return math.sqrt(2.0) * _erfinv(confidence)


def _erfinv(value: float) -> float:
    """Inverse error function, via a rational approximation.

    Accurate to about 1e-9 over the range used here, which is far beyond what a
    confidence interval needs.
    """
    if value <= -1.0 or value >= 1.0:
        raise ValueError(f"erfinv domain is (-1, 1), got {value!r}.")

    # Winitzki's approximation, refined by one Newton step.
    a = 0.147
    ln_term = math.log(1.0 - value * value)
    first = 2.0 / (math.pi * a) + ln_term / 2.0
    guess = math.copysign(
        math.sqrt(math.sqrt(first * first - ln_term / a) - first), value
    )

    # One Newton iteration on erf(x) - value.
    error = _erf(guess) - value
    derivative = 2.0 / math.sqrt(math.pi) * math.exp(-guess * guess)
    return guess - error / derivative


def _erf(value: float) -> float:
    """Error function, via the complementary-error continued fraction."""
    return math.erf(value)
