"""Precision filters for rule-based event detection.

The first baseline matched bare vocabulary: ``acquisition``, ``acquired``,
``litigation``. Across 84 filings that produced 751 "acquisitions" — one per
paragraph that happened to contain the word. Almost none were events.

Two independent things were wrong, and they need separate fixes:

**Where the match is.** Item 1 of a 10-Q is the financial statements, where
"assets acquired" is capital expenditure and "disposition" is how you retire a
fixed asset. Item 1A is Risk Factors, which is *hypothetical by construction*:
"an acquisition could harm us" is not an acquisition. Item 2 (MD&A) is mixed —
it reports real events and also discusses accounting policy.

**What the match looks like.** A bare noun is not an event. "We completed the
acquisition of Acme" is; "acquisition" alone is a topic. Requiring a subject and
an object, or an explicit completion verb, removes most of the noise without
needing a model.

This module holds both filters as pure functions so they can be tested against
text alone, and so the rule table in :mod:`app.extraction.events` stays readable.
"""

from __future__ import annotations

import re
from typing import Final

#: Sections where a keyword is almost never a reportable event.
#:
#: * ``part_i_item_1`` / ``item_1`` — financial statements and the notes to
#:   them: "property and equipment acquired", "identified for disposition".
#: * ``part_ii_item_1a`` / ``item_1a`` — risk factors: hypothetical by design.
#: * ``part_ii_item_6`` / ``item_6`` — exhibits: a list of document titles.
#: * ``part_ii_item_5`` / ``item_5`` — other information, usually boilerplate.
SUPPRESSED_SECTIONS: Final[frozenset[str]] = frozenset(
    {
        "part_i_item_1",
        "item_1",
        "part_ii_item_1a",
        "item_1a",
        "part_ii_item_5",
        "item_5",
        "part_ii_item_6",
        "item_6",
    }
)

#: Event types that survive in a suppressed section. Legal proceedings are
#: *reported* in Item 1 of Part II, and a settlement is a real event even when
#: it appears in the notes — but only when the sentence reports one, which
#: :func:`has_event_structure` enforces.
SECTION_EXEMPT_TYPES: Final[frozenset[str]] = frozenset({"legal_or_regulatory"})

#: Sections where even an exempt type must pass the structure test. Risk
#: Factors is hypothetical by construction, so "we could face litigation" is
#: never an event no matter how the sentence is phrased.
STRICT_SECTIONS: Final[frozenset[str]] = frozenset({"part_ii_item_1a", "item_1a"})

#: Phrases that make a sentence hypothetical or forward-looking rather than a
#: report of something that happened. Matched case-insensitively against the
#: sentence containing the match.
HYPOTHETICAL_MARKERS: Final[tuple[str, ...]] = (
    "could",
    "may ",
    "might",
    "would",
    "if we",
    "if our",
    "in the event",
    "risk that",
    "risks related",
    "potential",
    "possible",
    "anticipate",
    "no assurance",
    "from time to time",
)

#: Accounting vocabulary that looks like an event but is not one. These are
#: matched against the evidence span *and* the sentence, so they cannot
#: suppress a real event mentioned elsewhere in the same sentence.
ACCOUNTING_NOISE: Final[tuple[str, ...]] = (
    "property and equipment",
    "property, plant and equipment",
    "intangible assets acquired",
    "assets acquired",
    "accumulated depreciation",
    "useful lives",
    "straight-line",
    "amortization",
    "impairment",
    "capitalized",
    "capital expenditure",
    "stock-based compensation",
    "unearned",
    "deferred",
    "fair value",
    "carrying value",
    "identified for retirement",
    "identified for disposition",
    "held for sale",
    "leasehold",
    "right-of-use",
    "purchase obligations",
    "purchase price adjustments",
    "net of acquired cash",
    "acquisition of property",
    "acquisition and other investment",
    "discontinued operations",
    "expected long-term return",
    "return on assets",
    "repurchase program",
    "repurchased",
    "repurchases",
    "shares of common stock",
)

#: A real corporate action names what was acquired, sold or merged. This is the
#: "subject and object" test: ``acquisition of Acme`` passes, ``acquisition``
#: alone does not.
_ACQUISITION_EVENT_RE = re.compile(
    r"\b(?:"
    r"(?:completed|closed|consummated|announced|entered into)\s+(?:the\s+|a\s+)?"
    r"(?:acquisition|merger|purchase)\s+(?:of|agreement|with)\b"
    r"|(?:acquisition|merger)\s+of\s+(?:[A-Z][\w.&'-]*|our\b|the\b)"
    r"|acquired\s+(?:[A-Z][\w.&'-]*|all of the|the outstanding|a \d)"
    r"|business combination\s+with\b"
    r")",
    re.IGNORECASE,
)

_DIVESTITURE_EVENT_RE = re.compile(
    r"\b(?:"
    r"(?:completed|closed|consummated|announced)\s+(?:the\s+)?"
    r"(?:divestiture|sale|disposition)\s+of\b"
    r"|divested\s+(?:our|its|the|[A-Z][\w.&'-]*)"
    r"|sold\s+(?:our|its|the)\s+\w+\s+(?:business|segment|subsidiary|division)"
    r")",
    re.IGNORECASE,
)

_MANAGEMENT_EVENT_RE = re.compile(
    r"\b(?:"
    r"(?:chief|senior vice president|vice president|president|director|officer)"
    r"[^.]{0,60}\b(?:resigned|retired|was appointed|has been appointed)\b"
    r"|(?:resigned|retired)\s+(?:as|from|effective)\b"
    r"|appointed\s+(?:[A-Z][\w.&'-]*\s+){1,3}as\b"
    r")",
    re.IGNORECASE,
)

#: A management change names a person or a role. "we retired shares" is a
#: buyback, not a resignation, so the bare verb is not enough.
_MANAGEMENT_PERSON_RE = re.compile(
    r"\b(?:"
    r"(?:chief executive officer|chief financial officer|chief operating officer"
    r"|chief accounting officer|general counsel|president|vice president"
    r"|director|officer|CEO|CFO|COO)\b[^.]{0,80}"
    r"\b(?:resigned|retired|stepped down|was appointed|has been appointed)\b"
    r"|(?:resigned|retired|stepped down)\s+(?:as|from)\s+(?:the\s+)?"
    r"(?:chief|president|vice president|director|officer|CEO|CFO|COO)\b"
    r"|appointed\s+(?:[A-Z][\w.&'-]*\s+){1,3}as\s+(?:chief|president|vice president"
    r"|director|officer|CEO|CFO|COO)\b"
    r")",
    re.IGNORECASE,
)

#: A management change names a person or a role. "we retired shares" is a
#: buyback, not a resignation, so the bare verb is not enough.
_MANAGEMENT_PERSON_RE = re.compile(
    r"\b(?:"
    r"(?:chief executive officer|chief financial officer|chief operating officer"
    r"|chief accounting officer|general counsel|president|vice president"
    r"|director|officer|CEO|CFO|COO)\b[^.]{0,80}"
    r"\b(?:resigned|retired|stepped down|was appointed|has been appointed)\b"
    r"|(?:resigned|retired|stepped down)\s+(?:as|from)\s+(?:the\s+)?"
    r"(?:chief|president|vice president|director|officer|CEO|CFO|COO)\b"
    r"|appointed\s+(?:[A-Z][\w.&'-]*\s+){1,3}as\s+(?:chief|president|vice president"
    r"|director|officer|CEO|CFO|COO)\b"
    r")",
    re.IGNORECASE,
)

_RESTRUCTURING_EVENT_RE = re.compile(
    r"\b(?:"
    r"(?:announced|approved|implemented|initiated|commenced)\s+(?:a\s+)?"
    r"(?:restructuring|workforce reduction|reduction in force)"
    r"|restructuring\s+(?:plan|program|charge|initiative)\b"
    r"|(?:eliminat|reduc)\w*\s+(?:approximately\s+)?[\d,]+\s+(?:positions|employees|jobs)"
    r")",
    re.IGNORECASE,
)

#: Event types whose rules must pass the subject-and-object test.
STRUCTURED_EVENT_TYPES: Final[frozenset[str]] = frozenset(
    {"acquisition", "divestiture", "management_change", "restructuring"}
)

#: A reported legal event names an action and a counterparty or outcome.
#: "litigation" alone is a topic; "we recorded a settlement of $50 million" is
#: an event.
_LEGAL_EVENT_RE = re.compile(
    r"\b(?:"
    r"(?:recorded|recognized|paid|agreed to pay|entered into)\s+(?:a\s+)?"
    r"(?:settlement|consent order|plea agreement)"
    r"|(?:received|was served with|were served with)\s+(?:a\s+)?"
    r"(?:subpoena|civil investigative demand)"
    r"|(?:is|are|was|were)\s+(?:subject to|under)\s+(?:an?\s+)?"
    r"(?:investigation|inquiry|enforcement action)"
    r"|(?:filed|commenced)\s+(?:a\s+)?(?:lawsuit|complaint|action)\s+against"
    r")",
    re.IGNORECASE,
)

_STRUCTURED_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "acquisition": _ACQUISITION_EVENT_RE,
    "divestiture": _DIVESTITURE_EVENT_RE,
    "management_change": _MANAGEMENT_PERSON_RE,
    "restructuring": _RESTRUCTURING_EVENT_RE,
    "legal_or_regulatory": _LEGAL_EVENT_RE,
}


def is_suppressed_section(section_key: str, event_type: str) -> bool:
    """True when ``event_type`` should not be reported from ``section_key``.

    Examples
    --------
    >>> is_suppressed_section("part_i_item_1", "acquisition")
    True
    >>> is_suppressed_section("part_i_item_1", "legal_or_regulatory")
    False
    """
    key = section_key.lower()
    if key in STRICT_SECTIONS:
        return True
    if event_type in SECTION_EXEMPT_TYPES:
        return False
    return key in SUPPRESSED_SECTIONS


def is_hypothetical(sentence: str) -> bool:
    """True when ``sentence`` describes a possibility rather than a fact.

    Risk-factor prose is written in the conditional, so this removes most of
    Item 1A even when the section filter is bypassed.
    """
    lowered = sentence.lower()
    return any(marker in lowered for marker in HYPOTHETICAL_MARKERS)


def is_accounting_noise(evidence: str, sentence: str = "") -> bool:
    """True when the match is accounting vocabulary, not an event.

    The evidence span alone is often just the word ``acquired``, so the
    surrounding sentence is checked too: "property and equipment acquired" is
    noise even though the matched span is only ``acquired``.
    """
    lowered = f"{evidence} {sentence}".lower()
    return any(phrase in lowered for phrase in ACCOUNTING_NOISE)


def has_event_structure(event_type: str, sentence: str) -> bool:
    """True when ``sentence`` contains a subject-and-object event phrase.

    Applies to :data:`STRUCTURED_EVENT_TYPES`; guidance and earnings classes
    keep their original rules, because their vocabulary is already specific
    ("guidance", "consensus").
    """
    pattern = _STRUCTURED_PATTERNS.get(event_type)
    if pattern is None:
        return True
    return pattern.search(sentence) is not None


def sentence_around(text: str, start: int, end: int) -> str:
    """Return the sentence of ``text`` containing ``[start, end)``.

    Sentence boundaries are approximated by ``.``, ``;`` and newlines, which is
    enough for filing prose and avoids a dependency on a tokeniser.
    """
    left = max(text.rfind(".", 0, start), text.rfind("\n", 0, start))
    right_candidates = [
        position
        for position in (text.find(".", end), text.find("\n", end))
        if position != -1
    ]
    right = min(right_candidates) if right_candidates else len(text)
    return text[left + 1 : right + 1].strip()


def should_keep(
    *,
    event_type: str,
    section_key: str,
    sentence: str,
    evidence: str,
) -> bool:
    """Apply every precision filter to one candidate match.

    Order matters only for speed; each check is independent.

    Examples
    --------
    >>> should_keep(
    ...     event_type="acquisition",
    ...     section_key="part_i_item_2",
    ...     sentence="We completed the acquisition of Acme Corp.",
    ...     evidence="acquisition of Acme",
    ... )
    True
    """
    if is_suppressed_section(section_key, event_type):
        return False
    if is_hypothetical(sentence):
        return False
    if is_accounting_noise(evidence, sentence):
        return False
    return has_event_structure(event_type, sentence)
