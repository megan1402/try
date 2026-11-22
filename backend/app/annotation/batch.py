"""Batch annotation with checkpointing.

Annotating 187 passages takes minutes and costs money; annotating 5,000 takes
hours. Both need to survive an interruption, so results are appended to a JSONL
file as they arrive and a re-run skips anything already present.

Each stored record carries the metadata needed to reproduce it: model, prompt
version, rubric version, timestamp, source filing and sampling method. Without
those, an annotation is an assertion with no provenance.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from app.annotation.llm import AnnotationError, LlmAnnotator
from app.annotation.prompt import build_prompt
from app.annotation.schema import Annotation
from app.evaluation.gold import GoldSnippet

#: Where annotations are written by default.
DEFAULT_OUTPUT: Final[Path] = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "llm_annotations.jsonl"
)


@dataclass(frozen=True)
class AnnotationRecord:
    """One stored annotation, with the provenance needed to reproduce it."""

    passage_id: str
    model: str
    prompt_version: str
    rubric_version: str
    annotated_at: str
    source_filing: str | None
    sampling_method: str
    human_verified: bool
    annotation: dict[str, object]

    @classmethod
    def build(
        cls,
        *,
        snippet: GoldSnippet,
        annotation: Annotation,
        model: str,
        prompt_version: str,
        rubric_version: str,
        sampling_method: str,
    ) -> AnnotationRecord:
        return cls(
            passage_id=snippet.id,
            model=model,
            prompt_version=prompt_version,
            rubric_version=rubric_version,
            annotated_at=datetime.now(UTC).isoformat(timespec="seconds"),
            source_filing=snippet.accession_number,
            sampling_method=sampling_method,
            # AI labels are never ground truth until a human says so.
            human_verified=False,
            annotation=annotation.model_dump(mode="json"),
        )


def load_done(path: Path) -> set[str]:
    """Passage ids already annotated in ``path``."""
    if not path.exists():
        return set()

    done: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            done.add(json.loads(line)["passage_id"])
        except (json.JSONDecodeError, KeyError):
            # A truncated final line from an interrupted run; ignore it and let
            # the passage be re-annotated.
            continue
    return done


def annotate_batch(
    snippets: Sequence[GoldSnippet],
    *,
    annotator: LlmAnnotator,
    model: str,
    output: Path = DEFAULT_OUTPUT,
    sampling_method: str = "unknown",
    limit: int | None = None,
    progress: bool = True,
) -> tuple[int, int]:
    """Annotate ``snippets``, appending to ``output`` and skipping done ones.

    Returns
    -------
    tuple[int, int]
        ``(annotated, failed)`` for this run.

    Notes
    -----
    A failure is recorded and the run continues: one unparseable response should
    not discard an hour of work. Failed passages are simply absent from the
    output, so a re-run retries them.
    """
    done = load_done(output)
    pending = [snippet for snippet in snippets if snippet.id not in done]
    if limit is not None:
        pending = pending[:limit]

    annotated = 0
    failed = 0

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for index, snippet in enumerate(pending, start=1):
            prompt = build_prompt(snippet)
            try:
                annotation = annotator.annotate(prompt.text, passage_id=snippet.id)
            except AnnotationError as exc:
                failed += 1
                if progress:
                    print(f"  [{index}/{len(pending)}] {snippet.id} FAILED: {exc}")
                continue

            record = AnnotationRecord.build(
                snippet=snippet,
                annotation=annotation,
                model=model,
                prompt_version=prompt.prompt_version,
                rubric_version=prompt.rubric_version,
                sampling_method=sampling_method,
            )
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            handle.flush()
            annotated += 1

            if progress:
                print(
                    f"  [{index}/{len(pending)}] {snippet.id} -> "
                    f"{annotation.event_type}"
                )

    return annotated, failed


def load_annotations(path: Path) -> list[AnnotationRecord]:
    """Read stored annotation records."""
    records: list[AnnotationRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        records.append(
            AnnotationRecord(
                passage_id=payload["passage_id"],
                model=payload["model"],
                prompt_version=payload["prompt_version"],
                rubric_version=payload["rubric_version"],
                annotated_at=payload["annotated_at"],
                source_filing=payload.get("source_filing"),
                sampling_method=payload.get("sampling_method", "unknown"),
                human_verified=bool(payload.get("human_verified", False)),
                annotation=payload["annotation"],
            )
        )
    return records


def predicted_labels(records: Iterable[AnnotationRecord]) -> dict[str, str]:
    """Map passage id to the model's ``event_type``."""
    return {
        record.passage_id: str(record.annotation["event_type"]) for record in records
    }
