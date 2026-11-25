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

import argparse
from collections.abc import Sequence
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
from app.extraction.events import EVENT_TYPES as EVENT_CLASSES

#: What a keystroke means. ``""`` accepts the proposal, ``b`` undoes the last
#: label, ``q`` saves and leaves.
_ALIASES: dict[str, str] = {
    "": "",
    "b": "back",
    "back": "back",
    "n": NONE_LABEL,
    "s": SKIP_LABEL,
    "q": "quit",
}


def resolve_answer(answer: str, proposed: str | None) -> str | None:
    """Turn one keystroke into a label, or ``None`` when it is not valid.

    ``""`` means *accept the proposal*; ``"b"`` goes back one snippet; ``"q"``
    returns ``"quit"``; a digit ``1``–``11`` selects
    :data:`app.evaluation.gold.LABELS`; ``n``/``s`` select ``none``/``skip``;
    any full label name is accepted too.
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
    """Manual entry point: ``python -m app.evaluation.review [path] [--clear ID]``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_GOLD_PATH,
        help="golden-set file (default: tests/fixtures/events_gold.jsonl)",
    )
    parser.add_argument(
        "--clear",
        metavar="ID",
        help="clear one snippet's label and exit, e.g. --clear k007",
    )
    args = parser.parse_args()

    snippets = load_gold(args.path)

    if args.clear:
        snippets = with_label(snippets, args.clear, None)
        save_gold(snippets, args.path)
        print(f"Cleared the label for {args.clear}; `make review` will show it again.")
        return

    pending = unlabelled(snippets)

    if not pending:
        print(
            f"Nothing to label: all {len(snippets)} snippet(s) in "
            f"{args.path.name} are done."
        )
        return

    _print_keys()
    print(f"{len(pending)} snippet(s) to label in {args.path.name}\n")

    queue = [snippet.id for snippet in pending]
    labelled_this_session: list[str] = []
    position = 0

    while position < len(queue):
        snippet = _find(snippets, queue[position])
        _print_snippet(snippet, position + 1, len(queue))

        answer = _ask(snippet)
        if answer is None or answer == "quit":
            save_gold(snippets, args.path)
            print(f"\nSaved {args.path.name}.")
            return

        if answer == "back":
            if not labelled_this_session:
                print("Already at the first snippet of this session — nothing to undo.")
                continue
            previous = labelled_this_session.pop()
            snippets = with_label(snippets, previous, None)
            save_gold(snippets, args.path)
            position -= 1
            print(f"Cleared the label for {previous}; showing it again.")
            continue

        snippets = with_label(snippets, snippet.id, answer)
        labelled_this_session.append(snippet.id)
        save_gold(snippets, args.path)
        position += 1

    save_gold(snippets, args.path)
    done = sum(1 for snippet in snippets if snippet.label is not None)
    print(f"\nDone: {done}/{len(snippets)} snippet(s) labelled. Next: `make score`.")


def _ask(snippet: GoldSnippet) -> str | None:
    """Prompt until the keystroke is understood.

    Returns a label, ``"back"`` or ``"quit"``; ``None`` means end of input or
    Ctrl-C, which the caller treats as *save and leave*.
    """
    while True:
        try:
            answer = input("label> ")
        except (EOFError, KeyboardInterrupt):
            return None

        label = resolve_answer(answer, snippet.proposed)
        if label is not None:
            return label

        # Only mention "accept proposal" when there is one; otherwise the hint
        # invites a keystroke that cannot work.
        hint = "1-11, n, s, b (back) or q to quit"
        if snippet.proposed:
            hint = "1-11, n, s, b (back), enter (accept proposal) or q to quit"
        print(f"? press {hint}")


def _find(snippets: Sequence[GoldSnippet], snippet_id: str) -> GoldSnippet:
    """Return the snippet with ``snippet_id``, with its current label state."""
    for snippet in snippets:
        if snippet.id == snippet_id:
            return snippet
    raise KeyError(f"No snippet with id {snippet_id!r}.")


def _print_keys() -> None:
    """Print the key list.

    Only the event classes are numbered. ``none`` and ``skip`` are shown as
    letters because they are not event classes — they are the absence of one and
    an abstention — and numbering them invited the question of why ``10`` and
    ``n`` both exist. The numbers still work for all eleven; they are just not
    advertised for the last two.
    """
    print("Keys:")
    for index, label in enumerate(EVENT_CLASSES, start=1):
        print(f"  {index}. {label}")
    print(
        "  n = none (no event here)   s = skip (too ambiguous to judge)\n"
        "  b = back (undo the last label)   q = save and quit\n"
    )


def _print_snippet(snippet: GoldSnippet, position: int, total: int) -> None:
    text = " ".join(snippet.text.split())
    print(f"[{position}/{total}] {snippet.id}")
    if snippet.detected_type:
        print(f"  detected: {snippet.detected_type}")
    if snippet.ticker or snippet.section_key:
        print(
            f"  source:   {snippet.ticker or '?'} {snippet.section_key or ''}".rstrip()
        )
    print(f"  {text}")
    if snippet.context and snippet.context != text:
        print(f"  context:  {snippet.context}")
    print(f"  proposed: {snippet.proposed or '-'}")
    if snippet.notes:
        print(f"  notes:    {snippet.notes}")


if __name__ == "__main__":
    main()
