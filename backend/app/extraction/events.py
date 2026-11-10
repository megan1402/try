"""Rule-based event detection in filing text.

This is the **baseline**, not the finished system. Regular expressions over the
sections produced by :mod:`app.extraction.sections` are crude: the same sentence
in a Risk Factors section that says *"an acquisition could harm us"* looks like
an acquisition event. That is fine, and deliberate — a dumb, deterministic,
fully explained baseline is what a later Hugging Face model has to beat, and it
is what makes "the model is better" a measurable statement rather than a vibe.

Every event carries:

* ``event_type`` — one of :data:`EVENT_TYPES`;
* ``confidence`` — a fixed weight per rule, not a learned score;
* ``start``/``end`` — **absolute** character offsets into the filing text, so
  the evidence can be quoted later instead of paraphrased;
* ``as_of`` — the moment the filing became public (EDGAR's
  ``acceptance_datetime``). It is required and must be timezone-aware, because
  an event without an instant is not usable in a backtest.

At most one event of each type is reported per section: a section that says
"we raised guidance" five times is one event, and flooding the statistics with
duplicates would flatter the results.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final

import pandas as pd

from app.extraction.sections import Section

#: Event classes the baseline can recognise.
EVENT_TYPES: Final[tuple[str, ...]] = (
    "guidance_raise",
    "guidance_cut",
    "earnings_beat",
    "earnings_miss",
    "acquisition",
    "divestiture",
    "management_change",
    "restructuring",
    "legal_or_regulatory",
)

#: Column order of :func:`events_to_frame`.
EVENT_COLUMNS: Final[tuple[str, ...]] = (
    "ticker",
    "accession_number",
    "event_type",
    "confidence",
    "as_of",
    "section_key",
    "start",
    "end",
    "evidence",
)


@dataclass(frozen=True)
class FilingEvent:
    """One detected event, located in the filing text.

    Attributes
    ----------
    event_type:
        One of :data:`EVENT_TYPES`.
    confidence:
        Weight of the rule that matched, in ``[0, 1]``. Not calibrated — treat
        it as a ranking hint, not a probability.
    start:
        Absolute offset of the match in the filing text.
    end:
        Absolute offset just past the match.
    section_key:
        Section the match was found in, e.g. ``"part_i_item_2"``.
    evidence:
        The matched text itself, whitespace-collapsed.
    as_of:
        When the filing became public. Must be timezone-aware.
    """

    event_type: str
    confidence: float
    start: int
    end: int
    section_key: str
    evidence: str
    as_of: datetime

    def __post_init__(self) -> None:
        if self.event_type not in EVENT_TYPES:
            raise ValueError(
                f"Unknown event_type {self.event_type!r}; expected one of {EVENT_TYPES}."
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be within [0, 1], got {self.confidence!r}."
            )
        if self.end <= self.start or self.start < 0:
            raise ValueError(
                f"Invalid span ({self.start}, {self.end}) for the matched evidence."
            )
        if not self.evidence.strip():
            raise ValueError("evidence must not be empty.")
        if self.as_of.tzinfo is None:
            raise ValueError(
                "as_of must be timezone-aware: an event without a precise "
                "instant cannot be used for point-in-time analysis."
            )


@dataclass(frozen=True)
class _Rule:
    """One regular expression that implies one event type."""

    event_type: str
    pattern: re.Pattern[str]
    confidence: float


def _rule(event_type: str, pattern: str, confidence: float) -> _Rule:
    return _Rule(event_type, re.compile(pattern, re.IGNORECASE), confidence)


# Order matters only for readability; conflicts are resolved by confidence.
_RULES: Final[tuple[_Rule, ...]] = (
    # Guidance: both word orders occur ("raised its outlook", "outlook raised").
    _rule(
        "guidance_raise",
        r"\b(?:rais|increas|lift|boost)\w*\b[^.]{0,40}"
        r"\b(?:guidance|outlook|forecast)\b",
        0.70,
    ),
    _rule(
        "guidance_raise",
        r"\b(?:guidance|outlook|forecast)\b[^.]{0,40}"
        r"\b(?:rais|increas|lift|boost)\w*\b",
        0.70,
    ),
    _rule(
        "guidance_cut",
        r"\b(?:lower|reduc|cut|trim|slash)\w*\b[^.]{0,40}"
        r"\b(?:guidance|outlook|forecast)\b",
        0.70,
    ),
    _rule(
        "guidance_cut",
        r"\b(?:guidance|outlook|forecast)\b[^.]{0,40}"
        r"\b(?:lower|reduc|cut|trim|slash|below)\w*\b",
        0.70,
    ),
    # Results versus expectations.
    _rule(
        "earnings_beat",
        r"\b(?:exceeded|beat|surpassed|topped|ahead of|above)\b[^.]{0,30}"
        r"\b(?:expectations?|consensus|estimates?)\b",
        0.60,
    ),
    _rule(
        "earnings_miss",
        r"\b(?:missed|below|short of|fell short of)\b[^.]{0,30}"
        r"\b(?:expectations?|consensus|estimates?)\b",
        0.60,
    ),
    # Corporate actions. Broader patterns, so lower confidence: the words also
    # appear in boilerplate ("our acquisition strategy", "we may acquire").
    _rule(
        "acquisition",
        r"\b(?:acquired|acquires|acquiring|acquisition|merger|"
        r"business combination)\b",
        0.50,
    ),
    _rule(
        "divestiture",
        r"\b(?:divested|divests|divesting|divestiture|disposition|"
        r"sold [^.]{0,30}business)\b",
        0.60,
    ),
    _rule(
        "management_change",
        r"\b(?:resigned|resignation|stepped down|stepping down|appointed|"
        r"retired|retirement)\b",
        0.60,
    ),
    _rule(
        "restructuring",
        r"\b(?:restructuring|workforce reduction|layoffs?|severance)\b",
        0.60,
    ),
    _rule(
        "legal_or_regulatory",
        r"\b(?:subpoena|investigation|inquiry|settlement|litigation|"
        r"enforcement action|consent order)\b",
        0.55,
    ),
)


def extract_events(
    sections: Sequence[Section],
    *,
    as_of: datetime,
) -> list[FilingEvent]:
    """Detect events in ``sections``, attributing them to ``as_of``.

    Parameters
    ----------
    sections:
        Sections from :func:`app.extraction.sections.parse_sections`.
    as_of:
        When the filing became public — EDGAR's ``acceptance_datetime``, not
        ``filing_date``. Must be timezone-aware.

    Returns
    -------
    list[FilingEvent]
        At most one event per ``(section, event_type)``; the highest-confidence
        match wins, and the earliest one breaks ties. Sorted by position.
    """
    if as_of.tzinfo is None:
        raise ValueError(
            "as_of must be timezone-aware (e.g. EDGAR's acceptance_datetime)."
        )

    best: dict[tuple[str, str], FilingEvent] = {}

    for section in sections:
        for rule in _RULES:
            for match in rule.pattern.finditer(section.text):
                event = FilingEvent(
                    event_type=rule.event_type,
                    confidence=rule.confidence,
                    start=section.start + match.start(),
                    end=section.start + match.end(),
                    section_key=section.key,
                    evidence=_collapse(match.group(0)),
                    as_of=as_of,
                )
                key = (section.key, rule.event_type)
                current = best.get(key)
                if current is None or (event.confidence, -event.start) > (
                    current.confidence,
                    -current.start,
                ):
                    best[key] = event

    return sorted(best.values(), key=lambda event: (event.start, event.event_type))


def events_to_frame(
    events: Sequence[FilingEvent],
    *,
    ticker: str,
    accession_number: str | None = None,
) -> pd.DataFrame:
    """Turn events into a table ready for the event study.

    The result has :data:`EVENT_COLUMNS`, an empty (but correctly shaped) frame
    when ``events`` is empty, and ``as_of`` converted to a UTC timestamp.
    """
    if not ticker.strip():
        raise ValueError("ticker must be a non-empty string.")

    rows = [
        {
            "ticker": ticker.strip().upper(),
            "accession_number": accession_number,
            "event_type": event.event_type,
            "confidence": event.confidence,
            "as_of": event.as_of,
            "section_key": event.section_key,
            "start": event.start,
            "end": event.end,
            "evidence": event.evidence,
        }
        for event in events
    ]

    frame = pd.DataFrame(rows, columns=list(EVENT_COLUMNS))
    frame["as_of"] = pd.to_datetime(frame["as_of"], utc=True)
    return frame


def _collapse(text: str) -> str:
    """Collapse whitespace in a matched snippet, including newlines."""
    return " ".join(text.split())
