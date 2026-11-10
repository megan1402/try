"""Zero-shot event classification with a Hugging Face NLI model.

Nothing is trained here. A *natural language inference* (NLI) model already
knows how to answer "does this premise support this hypothesis?", so labelling
each event class as a sentence and taking the best score is enough to get a
classifier out of the box::

    premise:    "We raised our full-year guidance for fiscal 2027."
    hypothesis: "the company raised its financial guidance or outlook"
    -> entailment score 0.94

That is the point of starting here: it needs zero labels, so it can be measured
against the rule baseline immediately.

Two implementation notes:

* ``transformers`` is an **optional** dependency. It is imported lazily inside
  :func:`load_classifier` so the test suite, CI and ``make check`` keep working
  without a multi-hundred-megabyte download.
* Prefer BPE/WordPiece models (RoBERTa, BART, DistilBERT). DeBERTa-v3 is stronger
  on NLI but its tokeniser needs ``sentencepiece``, which has no wheel on every
  platform — and a classifier you cannot install is not a baseline.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from app.evaluation.gold import NONE_LABEL
from app.extraction.events import EVENT_TYPES

#: Default NLI model. ~330 MB, RoBERTa tokeniser, no ``sentencepiece`` needed.
DEFAULT_MODEL: Final[str] = "cross-encoder/nli-distilroberta-base"

#: Natural-language description of each class. The model compares the snippet
#: against these sentences, so the wording *is* the classifier's prior: keep
#: them concrete and mutually distinguishable.
HYPOTHESES: Final[dict[str, str]] = {
    "guidance_raise": "the company raised its financial guidance or outlook",
    "guidance_cut": "the company lowered its financial guidance or outlook",
    "earnings_beat": "the company reported results that beat expectations",
    "earnings_miss": "the company reported results that missed expectations",
    "acquisition": "the company acquired another company or business",
    "divestiture": "the company sold or divested a business",
    "management_change": "an executive resigned, retired or was appointed",
    "restructuring": "the company announced restructuring or layoffs",
    "legal_or_regulatory": "the company faces litigation, an investigation or a "
    "regulatory action",
    NONE_LABEL: "the text describes ordinary business results or boilerplate "
    "risk factors",
}

#: A callable with the shape of ``transformers.pipeline(...)``: text plus
#: candidate hypotheses in, ``{"labels": [...], "scores": [...]}`` out.
Classifier = Callable[[str, Sequence[str]], Mapping[str, Any]]


def load_classifier(model_name: str = DEFAULT_MODEL) -> Classifier:
    """Load a zero-shot pipeline, or explain how to install it.

    Raises
    ------
    RuntimeError
        If ``transformers`` is not installed. It is described as optional in
        :mod:`app.models`; failing with a clear message beats an ImportError
        traceback in the middle of a sweep.
    """
    try:
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "transformers is not installed. Install the optional ML extras with "
            "`pip install -r backend/requirements-ml.txt`."
        ) from exc

    pipe = pipeline("zero-shot-classification", model=model_name)

    def classify(text: str, hypotheses: Sequence[str]) -> Mapping[str, Any]:
        return pipe(
            text,
            candidate_labels=list(hypotheses),
            hypothesis_template="This text is about {}.",
        )

    return classify


def hypotheses_in_order() -> tuple[str, ...]:
    """The hypothesis sentences, in :data:`HYPOTHESES` order."""
    return tuple(HYPOTHESES.values())


@dataclass
class ZeroShotEventClassifier:
    """Maps a snippet to exactly one event label, or ``none``.

    ``classifier`` is injected rather than created here so tests can run the
    whole mapping logic against a fake pipeline — no model download, no network.
    """

    classifier: Classifier = field(default_factory=load_classifier)
    hypotheses: Mapping[str, str] = field(default_factory=lambda: dict(HYPOTHESES))

    @classmethod
    def load(cls, model_name: str = DEFAULT_MODEL) -> ZeroShotEventClassifier:
        """Build a classifier backed by a real Hugging Face model."""
        return cls(classifier=load_classifier(model_name))

    def scores(self, text: str) -> dict[str, float]:
        """Return one score per event label.

        The pipeline sorts its output by score, so labels are matched back by
        their hypothesis text rather than by position.
        """
        response = self.classifier(text, tuple(self.hypotheses.values()))
        by_hypothesis = {
            hypothesis: float(score)
            for hypothesis, score in zip(
                response["labels"], response["scores"], strict=True
            )
        }
        return {
            label: by_hypothesis[hypothesis]
            for label, hypothesis in self.hypotheses.items()
        }

    def predict(self, text: str) -> str:
        """Return the highest-scoring label, ``none`` included.

        ``none`` competes on equal terms: a snippet about ordinary results should
        win *no-event* rather than be forced into the nearest corporate action.
        Ties break alphabetically so the output is deterministic.
        """
        scores = self.scores(text)
        # Highest score wins; ties go to the alphabetically first label so the
        # output never depends on dictionary order.
        return min(scores.items(), key=lambda item: (-item[1], item[0]))[0]


def check_hypotheses() -> None:
    """Fail loudly if the hypothesis table drifts from the event classes."""
    expected = {*EVENT_TYPES, NONE_LABEL}
    if set(HYPOTHESES) != expected:
        raise ValueError(
            "HYPOTHESES must describe exactly the event classes: "
            f"missing {sorted(expected - set(HYPOTHESES))}, "
            f"unexpected {sorted(set(HYPOTHESES) - expected)}."
        )
