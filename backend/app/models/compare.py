"""Compare the rule baseline with the zero-shot model, snippet by snippet.

This is a *diagnostic*, not a scoreboard: with no labels yet, all it can say is
where the two disagree. That is still useful — the disagreements are exactly the
snippets worth spending human attention on, and the labels decide who is right.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

import pandas as pd

from app.evaluation.gold import DEFAULT_GOLD_PATH, GoldSnippet, load_gold
from app.evaluation.report import predict_label
from app.models.zero_shot import DEFAULT_MODEL, ZeroShotEventClassifier

#: Columns of the comparison table.
COMPARISON_COLUMNS: Final[tuple[str, ...]] = (
    "id",
    "rules",
    "model",
    "agree",
    "text",
)


def compare(
    snippets: Sequence[GoldSnippet],
    model_predict: Callable[[str], str],
) -> pd.DataFrame:
    """Run both predictors over ``snippets`` and record where they differ."""
    rows = []
    for snippet in snippets:
        rules = predict_label(snippet.text)
        model = model_predict(snippet.text)
        rows.append(
            {
                "id": snippet.id,
                "rules": rules,
                "model": model,
                "agree": rules == model,
                "text": " ".join(snippet.text.split())[:160],
            }
        )
    return pd.DataFrame(rows, columns=list(COMPARISON_COLUMNS))


def agreement_rate(frame: pd.DataFrame) -> float:
    """Share of snippets on which both predictors agree."""
    return float(frame["agree"].mean()) if len(frame) else float("nan")


def main() -> None:
    """Manual entry point: ``python -m app.models.compare [path] [--model id]``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_GOLD_PATH,
        help="golden-set file (default: tests/fixtures/events_gold.jsonl)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="NLI model id")
    parser.add_argument(
        "--max-shown", type=int, default=25, help="disagreements to print"
    )
    args = parser.parse_args()

    snippets = load_gold(args.path)
    print(
        f"{len(snippets)} snippet(s) from {args.path.name}. "
        f"Loading {args.model} (the first run downloads the weights) ..."
    )

    classifier = ZeroShotEventClassifier.load(args.model)
    frame = compare(snippets, classifier.predict)

    agreed = int(frame["agree"].sum())
    print(
        f"\nagreement: {agreed}/{len(frame)} ({agreement_rate(frame):.0%})"
        "  — note that agreeing is not the same as being right"
    )

    print("\nrules x model:")
    print(pd.crosstab(frame["rules"], frame["model"]).to_string())

    disagreements = frame[~frame["agree"]].head(args.max_shown)
    print(f"\n{len(frame) - agreed} disagreement(s), first {len(disagreements)} shown:")
    for row in disagreements.itertuples(index=False):
        print(f"\n  {row.id}   rules={row.rules}   model={row.model}\n    {row.text}")


if __name__ == "__main__":
    main()
