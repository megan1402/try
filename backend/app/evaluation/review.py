"""Label golden-set snippets from the terminal.

No spreadsheet, no external tool: run ``make review``, and for each snippet press
a number. The file is rewritten after every answer, so Ctrl-C loses at most one
label and everything is a plain-text diff you can read later.

Showing a proposal is a convenience, not an answer key — ``proposed`` came from
the very rules being measured, so accepting it without thinking turns the
evaluation into a mirror. When in doubt, press ``s`` (skip); a skipped snippet
is excluded from the numbers instead of being scored as a mistake.
"""

from __future__ import annotations

import sys
from pathlib import Path

from app.evaluation.gold import (
    DEFAULT_GOLD_PATH,
    LABELS,
    NONE_LABEL,
    SKIP_LABEL,
    GoldSnippet,
    load_gold,
    save_gold,
    unlabelled,
    with_label,
)

#: What a keystroke means. ``""`` accepts the proposal, ``q`` saves and leaves.
_ALIASES: dict[str, str] = {
    "": "",
    "n": NONE_LABEL,
    "s": SKIP_LABEL,
    "q": "quit",
}


def resolve_answer(answer: str, proposed: str | None) -> str | None:
    """Turn one keystroke into a label, or ``None`` when it is not valid.

    ``""`` means *accept the proposal*; ``"q"`` returns ``"quit"``; a digit
    ``1``–``9`` selects :data:`app.evaluation.gold.LABELS`; ``n``/``s`` select
    ``none``/``skip``; any full label name is accepted too.
    """
    text = answer.strip().lower()

    alias = _ALIASES.get(text)
    if alias is not None:
        return proposed if alias == "" else alias

    if text.isdigit():
        index = int(text) - 1
        return LABELS[index] if 0 <= index < len(LABELS) else None

    return text if text in LABELS else None


def main() -> None:
    """Manual entry point: ``python -m app.evaluation.review [path]``."""
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_GOLD_PATH
    snippets = load_gold(path)
    pending = unlabelled(snippets)

    if not pending:
        print(
            f"Nothing to label: all {len(snippets)} snippet(s) in {path.name} are done."
        )
        return

    _print_keys()
    print(f"{len(pending)} snippet(s) to label in {path.name}\n")

    for position, snippet in enumerate(pending, start=1):
        _print_snippet(snippet, position, len(pending))

        while True:
            try:
                answer = input("label> ")
            except (EOFError, KeyboardInterrupt):
                save_gold(snippets, path)
                print(f"\nSaved {path.name}.")
                return

            label = resolve_answer(answer, snippet.proposed)
            if label is None:
                print("? press 1-9, n, s, enter (accept proposal) or q to quit")
                continue
            if label == "quit":
                save_gold(snippets, path)
                print(f"Saved {path.name}.")
                return
            break

        snippets = with_label(snippets, snippet.id, label)
        save_gold(snippets, path)

    save_gold(snippets, path)
    done = sum(1 for snippet in snippets if snippet.label is not None)
    print(f"\nDone: {done}/{len(snippets)} snippet(s) labelled. Next: `make score`.")


def _print_keys() -> None:
    print("Keys:")
    for index, label in enumerate(LABELS, start=1):
        print(f"  {index}. {label}")
    print("  n = none   s = skip   enter = accept proposal   q = save and quit\n")


def _print_snippet(snippet: GoldSnippet, position: int, total: int) -> None:
    text = " ".join(snippet.text.split())
    print(f"[{position}/{total}] {snippet.id}")
    print(f"  {text}")
    print(f"  proposed: {snippet.proposed or '-'}")
    if snippet.notes:
        print(f"  notes:    {snippet.notes}")


if __name__ == "__main__":
    main()
