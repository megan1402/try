"""LLM client for annotation, with an injectable transport.

No provider is hard-coded. The annotator takes a ``Callable[[str], str]`` that
maps a prompt to a JSON string, so:

* the experiment can be re-run with any model by swapping one function;
* the test suite runs offline with a fake;
* a provider outage is a retry, not a rewrite.

The default transport uses the Anthropic SDK if it is installed and
``ANTHROPIC_API_KEY`` is set. It is imported lazily, so nothing here breaks when
the optional dependency is absent.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final

from app.annotation.schema import Annotation, parse_annotation

#: Default model. Chosen for rubric-following and structured output, not for
#: benchmark scores.
DEFAULT_MODEL: Final[str] = "claude-sonnet-4-5"

#: Attempts per passage before giving up. A malformed response is retried,
#: because the usual cause is a truncated JSON body.
DEFAULT_ATTEMPTS: Final[int] = 3

#: Seconds to wait between retries, multiplied by the attempt number.
RETRY_BACKOFF_SECONDS: Final[float] = 1.0

#: A transport maps a prompt to the model's raw text response.
Transport = Callable[[str], str]


class AnnotationError(RuntimeError):
    """Raised when a passage cannot be annotated after every attempt."""


@dataclass
class LlmAnnotator:
    """Annotates passages with an LLM, validating every response.

    Attributes
    ----------
    transport:
        Maps a prompt to raw model output. Injected so tests need no network.
    attempts:
        How many times to try a passage before raising.
    sleep:
        Called between retries; injected so tests do not wait.
    """

    transport: Transport
    attempts: int = DEFAULT_ATTEMPTS
    sleep: Callable[[float], None] = field(default=time.sleep)

    def annotate(self, prompt: str, *, passage_id: str) -> Annotation:
        """Annotate one passage, retrying malformed responses.

        Raises
        ------
        AnnotationError
            If every attempt fails. The message includes the last error, so a
            systematic problem (bad prompt, wrong model) is visible rather than
            buried in a retry loop.
        """
        last_error: Exception | None = None

        for attempt in range(1, self.attempts + 1):
            try:
                raw = self.transport(prompt)
                payload = _decode(raw)
                return parse_annotation(payload, passage_id=passage_id)
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < self.attempts:
                    self.sleep(RETRY_BACKOFF_SECONDS * attempt)

        raise AnnotationError(
            f"Could not annotate {passage_id} after {self.attempts} attempt(s): "
            f"{last_error}"
        )


def _decode(raw: str) -> object:
    """Extract a JSON object from a model response.

    Models sometimes wrap JSON in a fenced code block or add a sentence before
    it. Rather than fail, take the outermost ``{...}`` span — but only if it
    parses, so a truncated body still raises.
    """
    text = raw.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def anthropic_transport(model: str = DEFAULT_MODEL) -> Transport:
    """Build a transport backed by the Anthropic API.

    Raises
    ------
    RuntimeError
        If the SDK is not installed or ``ANTHROPIC_API_KEY`` is unset. Failing
        with a clear message beats an ImportError mid-run.
    """
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "The anthropic SDK is not installed. Install the optional extras "
            "with `pip install -r backend/requirements-llm.txt`."
        ) from exc

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it before running annotation."
        )

    client = anthropic.Anthropic()

    def transport(prompt: str) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")

    return transport
