"""Sample filing passages independently of the detector.

Every evaluation so far has been **precision-only**, because the sample was
drawn from events the detector reported. That cannot measure recall: the events
it missed are absent by construction.

This module samples **passages**, not detections. A passage is a paragraph of
filing prose, chosen at random from the corpus with no reference to what the
detector thinks. Labelling those gives both halves of the picture:

    precision = of the events the detector reported, how many are real
    recall    = of the real events in the corpus, how many did it report

Recall is estimated by comparing the labels on sampled passages against what the
detector found in the same passages. A passage labelled `acquisition` that the
detector did not flag is a **false negative**, and those are invisible to any
sample drawn from detections.

Sampling is seeded and stratified by document, so a reported recall can be
reproduced exactly.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from app.evaluation.gold import GoldSnippet

#: Default seed. Fixed so a reported recall can be reproduced.
DEFAULT_SEED: Final[int] = 20260121

#: Passages shorter than this are headings, table rows or fragments, not prose
#: a human can judge.
MIN_PASSAGE_CHARS: Final[int] = 120

#: Passages longer than this are truncated at a sentence boundary, so a reviewer
#: is not asked to read a whole page.
MAX_PASSAGE_CHARS: Final[int] = 600


@dataclass(frozen=True)
class Passage:
    """One paragraph of filing prose, with its provenance."""

    ticker: str
    accession_number: str
    section_key: str
    text: str
    #: What the detector reported in this passage, if anything. Recorded so the
    #: scorer can compare without re-running extraction, and so a reviewer can
    #: see the detector's answer *after* labelling.
    detected_types: tuple[str, ...] = ()


def split_passages(text: str) -> list[str]:
    """Split filing text into paragraph-sized passages.

    Lines are already one block per element (see
    :func:`app.extraction.html.html_to_text`), so a passage is a run of
    consecutive lines joined until it reaches a usable length. Short lines are
    merged rather than dropped, because a heading followed by a sentence is a
    perfectly good passage.
    """
    passages: list[str] = []
    buffer: list[str] = []
    length = 0

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        buffer.append(stripped)
        length += len(stripped) + 1

        if length >= MIN_PASSAGE_CHARS:
            passages.append(_truncate(" ".join(buffer)))
            buffer = []
            length = 0

    if buffer:
        tail = " ".join(buffer)
        if len(tail) >= MIN_PASSAGE_CHARS:
            passages.append(_truncate(tail))

    return passages


def sample_passages(
    documents: Sequence[tuple[str, str, str, str]],
    *,
    per_document: int = 3,
    seed: int = DEFAULT_SEED,
) -> list[Passage]:
    """Draw ``per_document`` passages from each document, seeded.

    Parameters
    ----------
    documents:
        ``(ticker, accession_number, section_key, text)`` tuples. One entry per
        document, not per section: sampling per document keeps a long 10-K from
        dominating a short 8-K.
    per_document:
        Passages to draw from each document. Documents with fewer usable
        passages contribute all of them.
    seed:
        Seed for the random generator.

    Returns
    -------
    list[Passage]
        The sample, ordered by document then by position in the document.

    Raises
    ------
    ValueError
        If ``per_document`` is not positive.
    """
    if per_document <= 0:
        raise ValueError("per_document must be positive.")

    rng = np.random.default_rng(seed)
    chosen: list[Passage] = []

    for ticker, accession, section_key, text in documents:
        passages = _unique(split_passages(text))
        if not passages:
            continue

        count = min(per_document, len(passages))
        indices = (
            range(len(passages))
            if count == len(passages)
            else sorted(rng.choice(len(passages), size=count, replace=False))
        )

        for index in indices:
            chosen.append(
                Passage(
                    ticker=ticker,
                    accession_number=accession,
                    section_key=section_key,
                    text=passages[index],
                )
            )

    return chosen


def to_snippets(
    passages: Sequence[Passage],
    *,
    prefix: str = "ps",
) -> list[GoldSnippet]:
    """Turn sampled passages into unlabelled golden-set snippets.

    ``detected_type`` is left empty on purpose: the reviewer must not know what
    the detector thought, or the recall estimate becomes circular. The detector's
    answer is recorded separately and joined at scoring time.
    """
    return [
        GoldSnippet(
            id=f"{prefix}{index:04d}",
            text=passage.text,
            label=None,
            proposed=None,
            ticker=passage.ticker,
            accession_number=passage.accession_number,
            section_key=passage.section_key,
            detected_type=None,
            context=None,
        )
        for index, passage in enumerate(passages, start=1)
    ]


def _unique(passages: Sequence[str]) -> list[str]:
    """Drop repeated passages, keeping the first occurrence.

    A long filing restates the same paragraph in several places — the MD&A
    summary and the note it summarises, for instance. Sampling both would ask
    the reviewer the same question twice and count one judgement as two.
    """
    seen: set[str] = set()
    unique: list[str] = []
    for passage in passages:
        key = " ".join(passage.split())
        if key in seen:
            continue
        seen.add(key)
        unique.append(passage)
    return unique


def _truncate(text: str) -> str:
    """Cut ``text`` at a sentence boundary near :data:`MAX_PASSAGE_CHARS`."""
    if len(text) <= MAX_PASSAGE_CHARS:
        return text

    window = text[:MAX_PASSAGE_CHARS]
    cut = window.rfind(". ")
    if cut >= MIN_PASSAGE_CHARS:
        return window[: cut + 1]
    return window.rstrip() + " ..."


def passages_per_document(total: int, documents: int) -> int:
    """How many passages to draw per document to reach ``total`` overall.

    Rounded up, so the sample is at least ``total`` when the corpus allows it.
    """
    if documents <= 0:
        return 0
    return max(1, math.ceil(total / documents))
