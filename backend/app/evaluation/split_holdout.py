"""Split a labelled set into development and test partitions.

The rule this enforces: **a set used to refine the rubric cannot also be used to
evaluate it.** Once you look at a passage and change the rubric because of it,
that passage is development data forever.

The split is **grouped by ticker**, not random. Two passages from the same
company's filings share vocabulary, section structure and often the same
underlying event, so a random split would leak information across the boundary
and inflate the test score.

The test partition is written to a separate file and should not be opened until
the rubric is frozen.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from app.evaluation.gold import GoldSnippet, load_gold, save_gold

#: Default seed. Fixed so a reported split is reproducible.
DEFAULT_SEED: Final[int] = 20260124

#: Fraction of *tickers* held out for the test partition.
DEFAULT_TEST_FRACTION: Final[float] = 0.4

#: Default input: the unlabelled holdout batch.
DEFAULT_INPUT: Final[Path] = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "events_holdout.jsonl"
)


@dataclass(frozen=True)
class Split:
    """A grouped train/test partition of a snippet set."""

    development: list[GoldSnippet]
    test: list[GoldSnippet]
    development_tickers: tuple[str, ...]
    test_tickers: tuple[str, ...]

    def summary(self) -> str:
        """One-line description of the partition."""
        return (
            f"development: {len(self.development)} snippet(s) from "
            f"{len(self.development_tickers)} ticker(s); "
            f"test: {len(self.test)} snippet(s) from "
            f"{len(self.test_tickers)} ticker(s)"
        )


def split_by_ticker(
    snippets: Sequence[GoldSnippet],
    *,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    seed: int = DEFAULT_SEED,
) -> Split:
    """Partition ``snippets`` by ticker, so no company straddles the boundary.

    Parameters
    ----------
    snippets:
        The set to split. Snippets without a ticker are placed in development,
        because they cannot be grouped reliably.
    test_fraction:
        Share of *tickers* to hold out, in ``(0, 1)``.
    seed:
        Seed for the ticker shuffle.

    Returns
    -------
    Split
        The two partitions and the tickers in each.

    Raises
    ------
    ValueError
        If ``test_fraction`` is outside ``(0, 1)`` or there are fewer than two
        tickers to split.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction!r}.")

    tickers = sorted({snippet.ticker for snippet in snippets if snippet.ticker})
    if len(tickers) < 2:
        raise ValueError(f"Need at least two tickers to split, found {len(tickers)}.")

    rng = np.random.default_rng(seed)
    shuffled = list(tickers)
    rng.shuffle(shuffled)

    # Clamp to [1, len - 1]: a split that empties either side is not a split,
    # and rounding can reach the extremes for small ticker counts.
    count = min(max(1, round(len(shuffled) * test_fraction)), len(shuffled) - 1)
    test_tickers = tuple(sorted(shuffled[:count]))
    development_tickers = tuple(sorted(shuffled[count:]))

    test_set = set(test_tickers)
    development = [
        snippet
        for snippet in snippets
        if not snippet.ticker or snippet.ticker not in test_set
    ]
    test = [snippet for snippet in snippets if snippet.ticker in test_set]

    return Split(
        development=development,
        test=test,
        development_tickers=development_tickers,
        test_tickers=test_tickers,
    )


def main() -> None:
    """Manual entry point: ``python -m app.evaluation.split_holdout``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing partitions",
    )
    args = parser.parse_args()

    snippets = load_gold(args.path)
    if any(snippet.label is not None for snippet in snippets):
        print(
            f"{args.path.name} already has labels. Split before labelling, or the "
            "test partition may contain passages you have already seen."
        )
        return

    split = split_by_ticker(snippets, test_fraction=args.test_fraction, seed=args.seed)

    development_path = args.path.with_name("holdout_development.jsonl")
    test_path = args.path.with_name("holdout_test.jsonl")

    for path in (development_path, test_path):
        if path.exists() and not args.force:
            print(f"{path.name} exists; pass --force to overwrite.")
            return

    save_gold(split.development, development_path)
    save_gold(split.test, test_path)

    print(split.summary())
    print(f"\ndevelopment -> {development_path.name}")
    print(f"test        -> {test_path.name}")
    print(
        "\nLabel the development partition first. If the rubric changes because "
        "of it, the test partition stays untouched and remains a valid holdout."
    )
    print("Do not open the test partition until the rubric is frozen.")


if __name__ == "__main__":
    main()
