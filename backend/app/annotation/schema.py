"""Structured output schema for LLM-assisted annotation.

The LLM returns JSON; this module is the contract it must satisfy. Validation is
strict on purpose: a malformed annotation is worse than a missing one, because it
silently enters the dataset.

Two design decisions worth stating:

* **The label is an integer index into a frozen list**, not a free string. The
  model cannot invent a class, and the mapping is stable across prompt versions.
* **``event_date`` is optional and never inferred.** A model that guesses a date
  from context would corrupt the event study, which anchors returns on the
  filing date. ``None`` is the honest answer when the passage does not state one.

Nothing here knows about the rule detector. The schema describes what a human
annotator would record, so the same structure can hold a human label later.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.evaluation.gold import NONE_LABEL, SKIP_LABEL
from app.extraction.events import EVENT_TYPES

#: The frozen class list. The integer in ``label`` indexes this tuple, so the
#: order must never change — a reordering would silently relabel every stored
#: annotation.
EVENT_CLASSES: Final[tuple[str, ...]] = (*EVENT_TYPES, NONE_LABEL, SKIP_LABEL)

#: ``event_type`` -> integer label, as specified in the annotation contract.
LABEL_INDEX: Final[dict[str, int]] = {
    name: index for index, name in enumerate(EVENT_CLASSES, start=1)
}

#: Integer label -> ``event_type``.
INDEX_LABEL: Final[dict[int, str]] = {
    index: name for name, index in LABEL_INDEX.items()
}

#: Classes that describe a genuine corporate event.
EVENT_CLASS_SET: Final[frozenset[str]] = frozenset(EVENT_TYPES)

#: ISO date, optionally with a time component the model may include.
_ISO_DATE_RE: Final[re.Pattern[str]] = re.compile(r"^\d{4}-\d{2}-\d{2}")


class Annotation(BaseModel):
    """One LLM annotation of one passage.

    Attributes
    ----------
    passage_id:
        Identifier of the passage that was annotated.
    label:
        Integer index into :data:`EVENT_CLASSES`, 1-based.
    event_type:
        The class name. Must agree with ``label``.
    is_genuine_event:
        Whether the passage reports a real corporate event. Must be ``False``
        for ``none`` and ``skip``.
    event_date:
        ISO date of the event, or ``None`` when the passage does not state one.
        Never inferred.
    evidence:
        Exact excerpt from the passage supporting the label.
    reason:
        One or two sentences explaining the decision.
    requires_review:
        Set by the model when it is uncertain. Not a calibrated probability.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    passage_id: str = Field(min_length=1)
    label: int = Field(ge=1, le=len(EVENT_CLASSES))
    event_type: str
    is_genuine_event: bool
    event_date: date | None = None
    evidence: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    requires_review: bool = False

    @field_validator("event_type")
    @classmethod
    def _known_class(cls, value: str) -> str:
        if value not in LABEL_INDEX:
            raise ValueError(
                f"Unknown event_type {value!r}; expected one of {EVENT_CLASSES}."
            )
        return value

    @field_validator("event_date", mode="before")
    @classmethod
    def _parse_date(cls, value: object) -> object:
        """Accept an ISO date or datetime string; reject anything else.

        A model that returns ``"April 2026"`` or ``"Q3"`` is guessing. Rejecting
        it forces ``null``, which is the honest answer.
        """
        if value is None or isinstance(value, date):
            return value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            if _ISO_DATE_RE.match(text):
                return date.fromisoformat(text[:10])
            raise ValueError(
                f"event_date must be an ISO date (YYYY-MM-DD) or null, got {value!r}."
            )
        raise ValueError(
            f"event_date must be a string or null, got {type(value).__name__}."
        )

    @model_validator(mode="after")
    def _consistent(self) -> Annotation:
        """Cross-field checks the model cannot satisfy by accident."""
        expected = INDEX_LABEL[self.label]
        if self.event_type != expected:
            raise ValueError(
                f"label {self.label} is {expected!r} but event_type is "
                f"{self.event_type!r}."
            )

        if self.event_type in EVENT_CLASS_SET and not self.is_genuine_event:
            raise ValueError(
                f"{self.event_type!r} is an event class but is_genuine_event is false."
            )
        if self.event_type not in EVENT_CLASS_SET and self.is_genuine_event:
            raise ValueError(
                f"{self.event_type!r} is not an event class but is_genuine_event is true."
            )
        if self.event_type in (NONE_LABEL, SKIP_LABEL) and self.event_date is not None:
            raise ValueError(f"{self.event_type!r} must not carry an event_date.")
        return self


def parse_annotation(payload: object, *, passage_id: str) -> Annotation:
    """Validate a decoded JSON payload into an :class:`Annotation`.

    ``passage_id`` is supplied by the caller rather than trusted from the model,
    so a confused response cannot be attributed to the wrong passage.

    Raises
    ------
    ValueError
        If the payload is not an object or fails validation.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object, got {type(payload).__name__}.")

    data = {**payload, "passage_id": passage_id}
    try:
        return Annotation.model_validate(data)
    except Exception as exc:  # pydantic.ValidationError and friends
        raise ValueError(f"Invalid annotation for {passage_id}: {exc}") from exc
