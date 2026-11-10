"""Labelled snippets used to measure event detection.

Regression tests pin behaviour: *"does the code still do what we decided?"*
They cannot answer *"was the decision any good?"* For that you need labels
written by a human who is not looking at the rule table, and this module is the
only agreed format for them.

One snippet per line, JSON Lines, so the file is diff-able, append-able and
safe to edit by hand or through :mod:`app.evaluation.review`::

    {"id": "nvda-10q-2026-07#p014", "text": "...", "label": "none", "proposed": "none", "notes": ""}

``label`` is the human verdict and the only field that counts. ``proposed`` is a
machine guess shown as a default, never used for scoring — if it were, the
evaluation would be circular.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Final

from app.extraction.events import EVENT_TYPES

#: No event in this snippet. Counted as a real answer, not as a missing one.
NONE_LABEL: Final[str] = "none"

#: Too ambiguous to judge. Excluded from every metric instead of being scored as
#: a mistake — an unanswerable snippet should never move the numbers.
SKIP_LABEL: Final[str] = "skip"

#: Every value ``label`` may take.
LABELS: Final[tuple[str, ...]] = (*EVENT_TYPES, NONE_LABEL, SKIP_LABEL)

#: The hand-off file, inside ``tests/fixtures`` so the tests can read it offline.
DEFAULT_GOLD_PATH: Final[Path] = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "events_gold.jsonl"
)


@dataclass(frozen=True)
class GoldSnippet:
    """One snippet and, once a human has decided, its label."""

    id: str
    text: str
    label: str | None = None
    proposed: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("id must not be empty.")
        if not self.text.strip():
            raise ValueError(f"{self.id}: text must not be empty.")
        for name, value in (("label", self.label), ("proposed", self.proposed)):
            if value is not None and value not in LABELS:
                raise ValueError(
                    f"{self.id}: unknown {name} {value!r}; expected one of {LABELS}."
                )

    @property
    def is_labelled(self) -> bool:
        return self.label is not None


def load_gold(path: Path = DEFAULT_GOLD_PATH) -> list[GoldSnippet]:
    """Read the JSON Lines file at ``path``.

    Blank lines are ignored, an empty ``label`` counts as unlabelled, and a
    malformed line reports the file and line number rather than a bare
    ``KeyError``.

    Raises
    ------
    ValueError
        If a line is not JSON, is not an object, or carries an unknown label.
    """
    snippets: list[GoldSnippet] = []
    seen: set[str] = set()

    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path.name}:{number}: invalid JSON ({exc.msg})."
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path.name}:{number}: expected a JSON object.")

        snippet = GoldSnippet(
            id=str(payload.get("id") or ""),
            text=str(payload.get("text") or ""),
            label=_optional(payload.get("label")),
            proposed=_optional(payload.get("proposed")),
            notes=str(payload.get("notes") or ""),
        )
        if snippet.id in seen:
            raise ValueError(f"{path.name}:{number}: duplicate id {snippet.id!r}.")
        seen.add(snippet.id)
        snippets.append(snippet)

    return snippets


def save_gold(snippets: Sequence[GoldSnippet], path: Path = DEFAULT_GOLD_PATH) -> None:
    """Write ``snippets`` back as JSON Lines, one object per line."""
    lines = [json.dumps(asdict(snippet), ensure_ascii=False) for snippet in snippets]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def labelled(snippets: Sequence[GoldSnippet]) -> list[GoldSnippet]:
    """Snippets a human has judged and that are not skipped."""
    return [snippet for snippet in snippets if is_scored(snippet)]


def unlabelled(snippets: Sequence[GoldSnippet]) -> list[GoldSnippet]:
    """Snippets still waiting for a verdict."""
    return [snippet for snippet in snippets if snippet.label is None]


def is_scored(snippet: GoldSnippet) -> bool:
    """True when the snippet counts towards precision and recall."""
    return snippet.label is not None and snippet.label != SKIP_LABEL


def with_label(
    snippets: Sequence[GoldSnippet],
    snippet_id: str,
    label: str,
) -> list[GoldSnippet]:
    """Return ``snippets`` with ``snippet_id`` relabelled.

    Raises
    ------
    ValueError
        If ``label`` is unknown or ``snippet_id`` is not in ``snippets``.
    """
    if label not in LABELS:
        raise ValueError(f"Unknown label {label!r}; expected one of {LABELS}.")
    if not any(snippet.id == snippet_id for snippet in snippets):
        raise ValueError(f"No snippet with id {snippet_id!r}.")

    return [
        replace(snippet, label=label) if snippet.id == snippet_id else snippet
        for snippet in snippets
    ]


def _optional(value: object) -> str | None:
    """Treat a missing or blank field as ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None
