"""LLM client for annotation, with an injectable transport.

No provider is hard-coded. The annotator takes a ``Callable[[str], str]`` that
maps a prompt to a JSON string, so:

* the experiment can be re-run with any model by swapping one function;
* the test suite runs offline with a fake;
* a provider outage is a retry, not a rewrite.

Two transports are provided: DeepSeek (OpenAI-compatible, via ``requests``) and
Anthropic (via its SDK, imported lazily). Both are optional — nothing here breaks
when the SDK or key is absent, because the transport is injected.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final

import requests

from app.annotation.schema import Annotation, parse_annotation

#: Default model. Chosen for rubric-following and structured output, not for
#: benchmark scores.
DEFAULT_MODEL: Final[str] = "claude-sonnet-4-5"

#: DeepSeek's OpenAI-compatible endpoint. Fixed, not configurable: the only
#: thing a user supplies is the key.
DEEPSEEK_BASE_URL: Final[str] = "https://api.deepseek.com/chat/completions"

#: DeepSeek's general model. ``deepseek-reasoner`` is slower and better at
#: multi-step judgement, at roughly 3x the output cost.
DEFAULT_DEEPSEEK_MODEL: Final[str] = "deepseek-chat"

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


def deepseek_transport(
    model: str = DEFAULT_DEEPSEEK_MODEL,
    *,
    timeout: float = 120.0,
) -> Transport:
    """Build a transport backed by DeepSeek's OpenAI-compatible API.

    Uses ``requests`` rather than an SDK: the request shape is a handful of
    fields, and ``requests`` is already a runtime dependency, so this adds
    nothing to install.

    ``temperature=0`` makes a re-run reproducible, and
    ``response_format={"type": "json_object"}`` asks the API for JSON, which
    removes most of the fenced-code-block noise.

    Raises
    ------
    RuntimeError
        If ``DEEPSEEK_API_KEY`` is unset.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. Export it before running annotation."
        )

    def transport(prompt: str) -> str:
        response = requests.post(
            DEEPSEEK_BASE_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 1024,
                "response_format": {"type": "json_object"},
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload["choices"][0]["message"]["content"])

    return transport


def build_transport(provider: str, model: str | None = None) -> Transport:
    """Return a transport for ``provider``.

    Parameters
    ----------
    provider:
        ``"deepseek"`` or ``"anthropic"``.
    model:
        Overrides the provider's default model.

    Raises
    ------
    ValueError
        If ``provider`` is unknown.
    """
    if provider == "deepseek":
        return deepseek_transport(model or DEFAULT_DEEPSEEK_MODEL)
    if provider == "anthropic":
        return anthropic_transport(model or DEFAULT_MODEL)
    raise ValueError(
        f"Unknown provider {provider!r}; expected 'deepseek' or 'anthropic'."
    )
