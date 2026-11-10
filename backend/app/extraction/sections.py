"""Split filing text into Part / Item sections.

A 10-K and a 10-Q are organised as numbered *items* (``Item 7. Management's
Discussion and Analysis...``) grouped under *parts* (``Part II. Other
Information``). Everything downstream — event extraction, retrieval, citations
— wants to talk about a specific item, so the first job is to find where each
one starts.

There is no markup to help: after :func:`app.extraction.html.html_to_text` a
heading is just a line of text. The rules used here are therefore heuristic,
but each one exists for a concrete reason:

* a heading **starts its line** and carries a title — ``Item 1.`` on its own is
  a table-of-contents cell, not a heading;
* a heading is **short** (at most :data:`MAX_HEADING_CHARS` characters);
* a heading does **not end in a digit** — a table-of-contents row usually ends
  with a page number;
* a heading does **not end with ``,``, ``;`` or ``:``** — prose does.

Cross-references are the other trap: ``refer to Item 1A of our Annual
Report`` mentions an item mid-sentence and must not start a section. Requiring
a period after the number and a title after the period filters almost all of
them.

When the same item appears more than once — once in the table of contents and
once in the body — the occurrence followed by the most text wins. That single
rule removes the entire table of contents in practice, because every entry in
it is followed by only a few characters.

Known limitation
----------------
A table-of-contents row whose page number was dropped by the HTML conversion
could still be mistaken for a heading. Such a row has a tiny body and is
discarded by the ``min_body_chars`` filter, but a *last* TOC row followed by
cover-page text can occasionally survive. The labelled golden set added in the
next phase is what will measure this properly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Longest line that can still be considered a heading.
MAX_HEADING_CHARS = 140

#: Shortest title worth accepting; guards against bare ``Item 1.`` cells.
MIN_TITLE_CHARS = 2

#: Shortest body a section must have to be kept. Table-of-contents entries are
#: followed by a handful of characters; real items are followed by paragraphs.
DEFAULT_MIN_BODY_CHARS = 80

# 8-K items carry a decimal number and no period after it, e.g.
# "Item 2.02 Results of Operations and Financial Condition".
_ITEM_DECIMAL_RE = re.compile(
    r"^item\s+(?P<item>\d{1,2}\.\d{1,2})\s+(?P<title>\S.*)$",
    re.IGNORECASE,
)

# 10-K / 10-Q items, e.g. "Item 7A. Quantitative and Qualitative Disclosures".
# The title may not start with a digit: that is how "Item 2.02 ..." is kept
# away from this pattern if the decimal form above did not match.
_ITEM_RE = re.compile(
    r"^item\s+(?P<number>\d{1,2})(?P<letter>[a-z])?\s*\.\s*(?P<title>[^\d\s].*)$",
    re.IGNORECASE,
)

# "Part II. Other Information" / "PART II — OTHER INFORMATION".
_PART_RE = re.compile(
    r"^part\s+(?P<part>iv|iii|ii|i)\b[\s.:\u2014-]*(?P<title>.*)$",
    re.IGNORECASE,
)

_TRAILING_PUNCTUATION = (",", ";", ":")


@dataclass(frozen=True)
class Section:
    """One Item of a filing, located in the source text.

    Attributes
    ----------
    part:
        Roman numeral of the enclosing part (``"I"``, ``"II"``, ...), or
        ``None`` when no part heading was found before this item. 10-Ks often
        place items before the first part heading, so ``None`` is normal.
    item:
        Normalised item number: ``"1"``, ``"1A"``, ``"7"``, ``"2.02"``.
    title:
        Heading text after the item number, whitespace-collapsed, e.g.
        ``"Risk Factors"``.
    start:
        Character offset of the heading's first character in the source text.
    end:
        Character offset just past the last character of the section.
    text:
        Exactly ``source[start:end]``, heading line included.
    """

    part: str | None
    item: str
    title: str
    start: int
    end: int
    text: str

    @property
    def key(self) -> str:
        """Stable identifier, e.g. ``"part_ii_item_1a"``.

        The part is included because item numbers repeat across parts: in a
        10-Q, ``Item 1`` is *Financial Statements* in Part I and *Legal
        Proceedings* in Part II.
        """
        item = f"item_{self.item.lower()}"
        return f"part_{self.part.lower()}_{item}" if self.part else item

    @property
    def heading(self) -> str:
        """The heading line, e.g. ``"Item 1A. Risk Factors"``."""
        return self.text.split("\n", 1)[0].strip()


@dataclass(frozen=True)
class _Candidate:
    """A heading found in the text, before duplicates are resolved."""

    part: str | None
    item: str
    title: str
    start: int


def parse_sections(
    text: str,
    *,
    min_body_chars: int = DEFAULT_MIN_BODY_CHARS,
) -> list[Section]:
    """Find the item sections in ``text``, in document order.

    Parameters
    ----------
    text:
        Filing text with one block per line, as produced by
        :func:`app.extraction.html.html_to_text`.
    min_body_chars:
        Sections with a shorter body are discarded. Raise it to be stricter
        about table-of-contents leftovers, lower it to keep very short items.

    Returns
    -------
    list[Section]
        One entry per item, ordered by position. Sections are contiguous: each
        one ends where the next begins, and the last one ends at ``len(text)``.
        Text before the first heading (cover page, table of contents) is not
        returned.

    Examples
    --------
    >>> text = "Item 7. MD&A\\n" + "x" * 100
    >>> [section.key for section in parse_sections(text)]
    ['item_7']
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}.")

    spans = _spans(_collect_candidates(text), len(text))
    best = _drop_shadowed_toc_entries(_best_per_item(spans))
    kept = [
        candidate
        for candidate, body_end in best
        if body_end - candidate.start >= min_body_chars
    ]

    sections: list[Section] = []
    for index, candidate in enumerate(kept):
        end = kept[index + 1].start if index + 1 < len(kept) else len(text)
        sections.append(
            Section(
                part=candidate.part,
                item=candidate.item,
                title=candidate.title,
                start=candidate.start,
                end=end,
                text=text[candidate.start : end],
            )
        )
    return sections


def find_item(
    sections: list[Section],
    item: str,
    part: str | None = None,
) -> Section | None:
    """Return the section for ``item`` (and optionally ``part``), if present.

    Examples
    --------
    >>> find_item([], "1A") is None
    True
    """
    wanted_item = item.strip().upper()
    wanted_part = part.strip().upper() if part else None

    for section in sections:
        if section.item.upper() != wanted_item:
            continue
        if wanted_part is None or section.part == wanted_part:
            return section
    return None


def _collect_candidates(text: str) -> list[_Candidate]:
    """Scan ``text`` line by line, tracking the enclosing part."""
    candidates: list[_Candidate] = []
    current_part: str | None = None
    offset = 0

    for raw_line in text.splitlines():
        line = raw_line.strip()
        start = offset + (len(raw_line) - len(raw_line.lstrip()))
        offset += len(raw_line) + 1  # splitlines() dropped one "\n".

        if not _looks_like_heading(line):
            continue

        part_match = _PART_RE.match(line)
        if part_match is not None:
            title = _clean_title(part_match.group("title"))
            if len(title) >= MIN_TITLE_CHARS:
                current_part = part_match.group("part").upper()
            continue

        item_match = _ITEM_DECIMAL_RE.match(line) or _ITEM_RE.match(line)
        if item_match is None:
            continue

        title = _clean_title(item_match.group("title"))
        if len(title) < MIN_TITLE_CHARS:
            continue

        candidates.append(
            _Candidate(
                part=current_part,
                item=_item_number(item_match),
                title=title,
                start=start,
            )
        )

    return candidates


def _looks_like_heading(line: str) -> bool:
    """Cheap structural checks applied to every line."""
    if not line or len(line) > MAX_HEADING_CHARS:
        return False
    if line.endswith(_TRAILING_PUNCTUATION):
        return False
    # A table-of-contents row ends with its page number.
    return not line[-1].isdigit()


def _item_number(match: re.Match[str]) -> str:
    """Return the normalised item number from either item pattern."""
    groups = match.groupdict()
    decimal = groups.get("item")
    if decimal:
        return decimal

    number = int(groups["number"])
    letter = (groups.get("letter") or "").upper()
    return f"{number}{letter}"


def _clean_title(title: str) -> str:
    """Collapse whitespace and drop trailing punctuation from a heading."""
    return " ".join(title.split()).rstrip(" .:")


def _spans(
    candidates: list[_Candidate],
    text_length: int,
) -> list[tuple[_Candidate, int]]:
    """Pair each candidate with the end offset implied by the *next* one."""
    ordered = sorted(candidates, key=lambda candidate: candidate.start)
    spans: list[tuple[_Candidate, int]] = []

    for index, candidate in enumerate(ordered):
        end = ordered[index + 1].start if index + 1 < len(ordered) else text_length
        spans.append((candidate, end))

    return spans


def _best_per_item(
    spans: list[tuple[_Candidate, int]],
) -> list[tuple[_Candidate, int]]:
    """Keep the longest occurrence of each ``(part, item)`` pair.

    This is what removes a table of contents: its ``Item 1.`` entry is followed
    by almost nothing, while the real ``Item 1.`` in the body is followed by
    paragraphs.
    """
    best: dict[tuple[str | None, str], tuple[_Candidate, int]] = {}

    for candidate, end in spans:
        key = (candidate.part, candidate.item)
        current = best.get(key)
        if current is None or (end - candidate.start, candidate.start) > (
            current[1] - current[0].start,
            current[0].start,
        ):
            best[key] = (candidate, end)

    return sorted(best.values(), key=lambda span: span[0].start)


def _drop_shadowed_toc_entries(
    spans: list[tuple[_Candidate, int]],
) -> list[tuple[_Candidate, int]]:
    """Drop copies of an item that appear before the first part heading.

    A table of contents sits at the very front of the document, before the
    first ``Part I`` heading, so its entries have ``part=None`` while the real
    sections that follow have a part. When the same item number appears both
    ways, the copy without a part is the TOC row.
    """
    items_with_part = {
        candidate.item for candidate, _ in spans if candidate.part is not None
    }
    return [
        (candidate, end)
        for candidate, end in spans
        if candidate.part is not None or candidate.item not in items_with_part
    ]
