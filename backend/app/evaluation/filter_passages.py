"""Filter a passage sample down to passages that mention event vocabulary.

Labelling all 516 sampled passages is mostly confirming `none`: only ~5% of
filing prose mentions an event at all. This narrows the sample to passages that
*do*, which answers a narrower but more useful question:

    of passages that talk about events, how many does the detector catch?

That is **recall on event-bearing passages**, not corpus-wide recall. The
distinction matters and is stated in the output, because a corpus-wide recall
would need the full sample.

The filter is deliberately generous — it matches vocabulary, not events — so it
over-includes rather than missing a real event. Over-inclusion costs a few extra
`none` labels; under-inclusion would bias recall upward.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from app.evaluation.gold import GoldSnippet, load_gold, save_gold

#: Default input: the full passage sample.
DEFAULT_INPUT: Path = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "passages_recall.jsonl"
)

#: Default output: the filtered sample.
DEFAULT_OUTPUT: Path = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "passages_events.jsonl"
)

#: Vocabulary that suggests a passage may describe an event. Deliberately broad:
#: the point is to avoid missing a real event, not to be precise.
EVENT_VOCABULARY: re.Pattern[str] = re.compile(
    r"\b(?:"
    r"acquisition|acquired|acquire|merger|merged|divest\w*|disposition"
    r"|settlement|settled|subpoena|investigation|inquiry|litigation|lawsuit"
    r"|restructuring|layoffs?|severance|workforce reduction"
    r"|resign\w*|appointed|stepped down|retired"
    r"|guidance|outlook|forecast"
    r"|exceeded|missed|consensus|estimates"
    r")\b",
    re.IGNORECASE,
)


def mentions_event(text: str) -> bool:
    """True when ``text`` contains vocabulary that may indicate an event."""
    return EVENT_VOCABULARY.search(text) is not None


def filter_passages(snippets: list[GoldSnippet]) -> list[GoldSnippet]:
    """Keep only passages that mention event vocabulary, renumbered."""
    kept = [snippet for snippet in snippets if mentions_event(snippet.text)]
    return [
        GoldSnippet(
            id=f"pe{index:04d}",
            text=snippet.text,
            label=None,
            proposed=None,
            ticker=snippet.ticker,
            accession_number=snippet.accession_number,
            section_key=snippet.section_key,
            detected_type=None,
            context=None,
        )
        for index, snippet in enumerate(kept, start=1)
    ]


def filter_sidecar(
    answers: dict[str, list[str]],
    snippets: list[GoldSnippet],
    kept: list[GoldSnippet],
) -> dict[str, list[str]]:
    """Re-key the detector answers to the filtered snippet ids.

    The original ids are matched by position, because the filter preserves
    order: the n-th kept passage is the n-th snippet that mentioned an event.
    """
    kept_texts = [snippet.text for snippet in kept]
    by_text = {snippet.text: snippet.id for snippet in snippets}

    filtered: dict[str, list[str]] = {}
    for index, text in enumerate(kept_texts, start=1):
        original_id = by_text.get(text)
        filtered[f"pe{index:04d}"] = answers.get(original_id, []) if original_id else []
    return filtered


def main() -> None:
    """Manual entry point: ``python -m app.evaluation.filter_passages``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    snippets = load_gold(args.path)
    kept = filter_passages(snippets)
    if not kept:
        print("No passage mentions event vocabulary.")
        return

    save_gold(kept, args.output)

    sidecar_in = args.path.with_suffix(".detected.json")
    sidecar_out = args.output.with_suffix(".detected.json")
    if sidecar_in.exists():
        answers = json.loads(sidecar_in.read_text(encoding="utf-8"))
        filtered = filter_sidecar(answers, snippets, kept)
        sidecar_out.write_text(json.dumps(filtered, indent=2) + "\n", encoding="utf-8")

    flagged = (
        sum(1 for value in filtered.values() if value) if sidecar_in.exists() else 0
    )
    print(f"{len(snippets)} passage(s) -> {len(kept)} mention event vocabulary")
    print(f"written to {args.output}")
    print(f"detector flagged {flagged}/{len(kept)} of them")
    print(
        "\nThis measures recall on event-bearing passages, not corpus-wide "
        "recall: the full sample would be needed for that."
    )
    print("\nNext: `make review-events`, then `make recall-events`.")


if __name__ == "__main__":
    main()
