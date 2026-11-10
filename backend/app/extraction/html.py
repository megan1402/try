"""HTML/XHTML to plain text for SEC filings.

An SEC filing is machine-generated XHTML: inline-styled ``<div>`` elements each
holding a single paragraph, tables used for page layout, and punctuation stored
as entities (``&#8217;`` for a curly apostrophe, ``&#160;`` for a non-breaking
space). There are no semantic ``<h1>``/``<h2>`` headings — a section title is
just a bold paragraph — so tag names alone cannot tell you anything about
document structure.

This module therefore does one deliberately simple thing: **preserve block
structure**. Every block-level element (``div``, ``p``, ``tr``, ``li``, ...)
becomes its own line, table cells stay on one line, scripts and styles are
dropped, entities are decoded and runs of whitespace collapse to a single
space. The output is plain text whose *line structure* is meaningful, which is
what :mod:`app.extraction.sections` relies on to find headings.

Keeping the two concerns apart means the heading logic can be unit-tested
against hand-written text, without needing HTML at all.
"""

from __future__ import annotations

import re
import warnings

from bs4 import BeautifulSoup, NavigableString, XMLParsedAsHTMLWarning

#: Tags that start and end a line. ``br`` is handled separately: it is empty,
#: so it is replaced by a newline rather than being wrapped in one.
_BLOCK_TAGS: tuple[str, ...] = (
    "address",
    "article",
    "blockquote",
    "caption",
    "dd",
    "div",
    "dl",
    "dt",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "tbody",
    "tfoot",
    "thead",
    "tr",
    "ul",
)

#: Tags that keep their text on one line. A table row full of cells stays a
#: single line, which is far more readable than one word per line.
_CELL_TAGS: tuple[str, ...] = ("td", "th")

#: Tags whose contents are never part of the document's prose.
_DROP_TAGS: tuple[str, ...] = ("script", "style", "noscript", "template")

#: Whitespace worth collapsing. ``\u00a0`` is the non-breaking space EDGAR uses
#: for column padding; treating it as ordinary space is what makes ``&#160;``
#: disappear from the output instead of surviving as look-alike garbage.
_INLINE_WHITESPACE_RE = re.compile(r"[ \t\r\f\v\u00a0]+")


def html_to_text(html: str) -> str:
    """Convert ``html`` to plain text, one line per block-level element.

    Parameters
    ----------
    html:
        A complete HTML/XHTML document, as returned by
        :func:`app.ingestion.sec.download_document`.

    Returns
    -------
    str
        Text with tags removed, entities decoded, one line per block, blank
        lines dropped and runs of spaces collapsed. Non-breaking spaces become
        ordinary spaces.

    Notes
    -----
    Whitespace inside ``<pre>`` is normalised too. EDGAR documents do not use
    ``<pre>`` for anything whose alignment matters, and collapsing it keeps the
    output predictable.
    """
    if not isinstance(html, str):
        raise TypeError(f"html must be a string, got {type(html).__name__}.")

    # EDGAR documents open with an XML declaration because their contents are
    # inline XBRL, so BeautifulSoup warns when we parse them as HTML. Treating
    # them as HTML is exactly what we want: an XML parser would reject the
    # first unclosed tag in a real filing.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "lxml")

    for tag in soup.find_all(list(_DROP_TAGS)):
        tag.decompose()

    _separate_blocks(soup)
    return _normalise(soup.get_text())


def _separate_blocks(soup: BeautifulSoup) -> None:
    """Wrap every block-level element in newlines, in place.

    ``insert``/``append`` take strings wrapped in
    :class:`~bs4.NavigableString` so the markers become ordinary text nodes
    rather than new tags.
    """
    newline = "\n"

    for tag in soup.find_all(list(_BLOCK_TAGS)):
        tag.insert(0, NavigableString(newline))
        tag.append(NavigableString(newline))

    # ``br`` carries no content, so it is a line break in its own right.
    for br in soup.find_all("br"):
        br.replace_with(NavigableString(newline))

    # Cells are separated by a space so a row reads as one line.
    for cell in soup.find_all(list(_CELL_TAGS)):
        cell.append(NavigableString(" "))


def _normalise(text: str) -> str:
    """Collapse whitespace and drop blank lines."""
    lines = (_INLINE_WHITESPACE_RE.sub(" ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)
