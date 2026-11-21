"""Clear labels so specific snippets return to the review queue.

Used when a rubric change means some earlier labels were made under a different
rule. Clearing is safer than editing by hand: the snippet keeps its provenance
and simply becomes unlabelled again.

    python -m app.evaluation.relabel ev0009 ev0011 ev0005

The file is rewritten once, so an interrupted run cannot leave it half-updated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.evaluation.gold import load_gold, save_gold, with_label

#: Default input: the precision sample.
DEFAULT_INPUT: Path = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "events_precision.jsonl"
)


def main() -> None:
    """Manual entry point: ``python -m app.evaluation.relabel ID [ID ...]``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ids", nargs="+", help="snippet ids to clear")
    parser.add_argument("--path", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()

    snippets = load_gold(args.path)
    known = {snippet.id for snippet in snippets}

    unknown = [snippet_id for snippet_id in args.ids if snippet_id not in known]
    if unknown:
        print(f"Unknown id(s): {', '.join(unknown)}", file=sys.stderr)
        raise SystemExit(1)

    cleared = 0
    for snippet_id in args.ids:
        snippets = with_label(snippets, snippet_id, None)
        cleared += 1

    save_gold(snippets, args.path)
    print(f"Cleared {cleared} label(s) in {args.path.name}.")
    print("Run `make review-precision` to label them again.")


if __name__ == "__main__":
    main()
