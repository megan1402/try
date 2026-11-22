"""Stratified sampling of detected events for precision measurement.

A random sample of all detected events would be ~60% acquisitions, because that
is what the detector produces most of. Precision for the small classes would
then rest on a handful of items and be useless. Sampling **within each class**
fixes that: every class gets a quota, so each precision estimate has a known,
stated sample size.

What this sample can and cannot measure
---------------------------------------
A sample drawn from *detected* events estimates **precision**: of the events the
detector reported, what share are real? It cannot estimate **recall**, because
recall needs the events the detector *missed*, and those are not in the sample
by construction. Measuring recall requires independently sampling filing
passages — including documents where the detector found nothing — and labelling
those. That is a separate exercise, and this module deliberately does not
pretend to do it.

Sampling is seeded, so the same corpus and seed always produce the same set and
a reported precision can be reproduced exactly.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from app.evaluation.gold import GoldSnippet
from app.extraction.events import EVENT_TYPES, FilingEvent

#: Default seed. Fixed so a reported precision can be reproduced.
DEFAULT_SEED: Final[int] = 20260119

#: Characters of surrounding text kept for context. Enough to see the sentence
#: before and after, which is what makes a borderline case judgeable.
CONTEXT_CHARS: Final[int] = 240


@dataclass(frozen=True)
class DetectedEvent:
    """One event as it came out of the detector, with its provenance."""

    ticker: str
    accession_number: str
    section_key: str
    event_type: str
    sentence: str
    context: str


def allocate(
    counts: dict[str, int],
    total: int,
    *,
    minimum: int = 5,
) -> dict[str, int]:
    """Split ``total`` across classes in proportion to ``counts``.

    Every class with at least one event gets at least ``minimum`` slots (or all
    of its events, if it has fewer), so a rare class is never squeezed out by a
    common one. The remainder is distributed by largest-remainder, which keeps
    the total exact.

    Parameters
    ----------
    counts:
        Events available per class.
    total:
        How many to sample overall.
    minimum:
        Floor per non-empty class.

    Returns
    -------
    dict[str, int]
        Quota per class, never exceeding the number available.

    Raises
    ------
    ValueError
        If ``total`` is not positive or ``minimum`` is negative.

    Examples
    --------
    >>> quotas = allocate({"a": 100, "b": 20}, 10, minimum=3)
    >>> quotas["b"] >= 3 and sum(quotas.values()) == 10
    True
    """
    if total <= 0:
        raise ValueError("total must be positive.")
    if minimum < 0:
        raise ValueError("minimum must not be negative.")

    available = {name: count for name, count in counts.items() if count > 0}
    if not available:
        return {}

    # Floors first, capped by what each class actually has.
    quotas = {name: min(minimum, count) for name, count in available.items()}
    remaining = total - sum(quotas.values())

    if remaining > 0:
        # Distribute the remainder in proportion to the *original* counts, not
        # to the leftover capacity. Proportional-to-capacity would give a small
        # class a second share and let it overtake a large one.
        total_available = sum(available.values())
        share = {
            name: remaining * available[name] / total_available for name in available
        }
        for name, value in share.items():
            quotas[name] += min(int(math.floor(value)), available[name] - quotas[name])

        # Largest remainder, so the quotas add up to ``total`` exactly.
        leftover = total - sum(quotas.values())
        order = sorted(
            available,
            key=lambda name: (-(share[name] - math.floor(share[name])), name),
        )
        for name in order:
            if leftover <= 0:
                break
            if quotas[name] < available[name]:
                quotas[name] += 1
                leftover -= 1

    return quotas


def sample_events(
    events: Sequence[DetectedEvent],
    *,
    total: int = 150,
    minimum: int = 5,
    seed: int = DEFAULT_SEED,
    exclude: Sequence[DetectedEvent] = (),
) -> list[DetectedEvent]:
    """Draw a stratified sample of ``events``, seeded and reproducible.

    Parameters
    ----------
    events:
        Every detected event in the corpus.
    total:
        Target sample size. The result may be smaller if the corpus is.
    minimum:
        Floor per non-empty class; see :func:`allocate`.
    seed:
        Seed for the random generator.
    exclude:
        Events already labelled elsewhere. They are removed before sampling, so
        a second batch never re-asks about something already judged — which is
        what makes a held-out split possible.

    Returns
    -------
    list[DetectedEvent]
        The sample, ordered by class then by position in ``events``.
    """
    excluded = {
        (event.accession_number, event.section_key, event.event_type, event.sentence)
        for event in exclude
    }
    pool = [
        event
        for event in events
        if (event.accession_number, event.section_key, event.event_type, event.sentence)
        not in excluded
    ]

    by_type: dict[str, list[DetectedEvent]] = {}
    for event in pool:
        by_type.setdefault(event.event_type, []).append(event)

    quotas = allocate(
        {name: len(items) for name, items in by_type.items()},
        total,
        minimum=minimum,
    )

    rng = np.random.default_rng(seed)
    chosen: list[DetectedEvent] = []

    for event_type in sorted(quotas):
        items = by_type[event_type]
        quota = quotas[event_type]
        if quota >= len(items):
            chosen.extend(items)
            continue
        indices = rng.choice(len(items), size=quota, replace=False)
        chosen.extend(items[index] for index in sorted(indices))

    return chosen


def to_snippets(
    events: Sequence[DetectedEvent],
    *,
    prefix: str = "ev",
) -> list[GoldSnippet]:
    """Turn sampled events into unlabelled golden-set snippets.

    ``proposed`` is left empty on purpose. The reviewer is judging whether the
    *detector* was right, so showing the detector's own answer would anchor the
    judgement and inflate the measured precision.
    """
    return [
        GoldSnippet(
            id=f"{prefix}{index:04d}",
            text=event.sentence,
            label=None,
            proposed=None,
            ticker=event.ticker,
            accession_number=event.accession_number,
            section_key=event.section_key,
            detected_type=event.event_type,
            context=event.context,
        )
        for index, event in enumerate(events, start=1)
    ]


def context_around(text: str, start: int, end: int, width: int = CONTEXT_CHARS) -> str:
    """Return ``width`` characters either side of ``[start, end)``, collapsed."""
    left = max(0, start - width)
    right = min(len(text), end + width)
    return " ".join(text[left:right].split())


def event_types_in_order() -> tuple[str, ...]:
    """The event classes, in the canonical order."""
    return EVENT_TYPES


def from_filing_event(
    event: FilingEvent,
    *,
    ticker: str,
    accession_number: str,
    text: str,
) -> DetectedEvent:
    """Build a :class:`DetectedEvent` from a detector result and its source."""
    return DetectedEvent(
        ticker=ticker,
        accession_number=accession_number,
        section_key=event.section_key,
        event_type=event.event_type,
        sentence=event.sentence or event.evidence,
        context=context_around(text, event.start, event.end),
    )
