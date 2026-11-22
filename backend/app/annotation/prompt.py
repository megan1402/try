"""Prompt construction for LLM-assisted annotation.

The prompt is built from three parts, each versioned so a stored annotation can
be traced back to the exact instructions that produced it:

* the **rubric**, read from ``docs/methodology.md`` so the LLM and the human
  annotator work from the same document;
* the **class list**, generated from :data:`app.annotation.schema.EVENT_CLASSES`
  so it cannot drift from the schema;
* the **passage**, with filing metadata but *never* the human label or the
  detector's prediction.

That last point is the whole experiment. If the model saw the human label, the
agreement number would be meaningless.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from app.annotation.schema import EVENT_CLASSES, LABEL_INDEX
from app.evaluation.gold import GoldSnippet

#: Bumped whenever the prompt text changes, so stored annotations stay traceable.
#: 2026-11-22.2 adds the undated-event, unclosed-transaction and heading rules
#: to the rubric, after Phase 1 showed the model applying the rubric where it
#: was silent.
PROMPT_VERSION: Final[str] = "2026-11-22.2"

#: The rubric the prompt embeds. Read at build time so the LLM and the human
#: annotator cannot drift apart.
RUBRIC_PATH: Final[Path] = (
    Path(__file__).resolve().parents[3] / "docs" / "methodology.md"
)

#: Instructions that are not part of the human rubric but are needed for
#: structured output.
_OUTPUT_CONTRACT: Final[str] = """
## Output format

Return a single JSON object and nothing else:

{
  "label": <integer 1-11>,
  "event_type": "<class name>",
  "is_genuine_event": <true|false>,
  "event_date": "<YYYY-MM-DD>" or null,
  "evidence": "<exact excerpt from the passage>",
  "reason": "<one or two sentences>",
  "requires_review": <true|false>
}

Rules for the output:

- `label` and `event_type` must agree; the integer indexes the class list above.
- `is_genuine_event` is true only for classes 1-9, false for `none` and `skip`.
- `event_date` is the date the event **occurred**, not the filing or announcement
  date. Use null when the passage does not state it. Never infer or guess a date.
- `evidence` must be copied verbatim from the passage, not paraphrased.
- `requires_review` is true when you are genuinely uncertain. It is not a
  probability and will not be treated as one.
"""


@dataclass(frozen=True)
class Prompt:
    """A rendered prompt plus the metadata needed to reproduce it."""

    text: str
    prompt_version: str
    rubric_version: str


def rubric_version() -> str:
    """Short hash of the rubric file, so a change is detectable."""
    content = RUBRIC_PATH.read_text(encoding="utf-8")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]


def class_list() -> str:
    """The numbered class list, generated from the schema."""
    return "\n".join(f"{LABEL_INDEX[name]}. {name}" for name in EVENT_CLASSES)


def build_prompt(snippet: GoldSnippet) -> Prompt:
    """Render the annotation prompt for one passage.

    The snippet's ``label``, ``proposed`` and ``detected_type`` are deliberately
    not read. Only the text and filing metadata reach the model.
    """
    rubric = RUBRIC_PATH.read_text(encoding="utf-8")

    metadata = [
        f"- Ticker: {snippet.ticker or 'unknown'}",
        f"- Filing: {snippet.accession_number or 'unknown'}",
        f"- Section: {snippet.section_key or 'unknown'}",
    ]

    context_block = ""
    if snippet.context and snippet.context != snippet.text:
        context_block = (
            "\n## Surrounding context\n\n"
            "The passage is taken from the following text. Use it only to "
            "disambiguate; label the passage, not the context.\n\n"
            f"{snippet.context}\n"
        )

    text = (
        "You are annotating a passage from a US SEC filing for a financial "
        "event-detection dataset.\n\n"
        "## Annotation rubric\n\n"
        f"{rubric}\n\n"
        "## Event classes\n\n"
        f"{class_list()}\n\n"
        "## Filing metadata\n\n"
        + "\n".join(metadata)
        + "\n\n## Passage to annotate\n\n"
        f"{snippet.text}\n"
        f"{context_block}"
        f"{_OUTPUT_CONTRACT}"
    )

    return Prompt(
        text=text,
        prompt_version=PROMPT_VERSION,
        rubric_version=rubric_version(),
    )
