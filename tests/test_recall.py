"""Tests for :mod:`app.evaluation.passages` and :mod:`app.evaluation.recall`.

The recall arithmetic is the part that matters: a false negative is the whole
reason this module exists, so it is tested explicitly.
"""

from __future__ import annotations

import pytest

from app.evaluation.gold import GoldSnippet
from app.evaluation.passages import (
    MAX_PASSAGE_CHARS,
    MIN_PASSAGE_CHARS,
    Passage,
    passages_per_document,
    sample_passages,
    split_passages,
    to_snippets,
)
from app.evaluation.recall import confusion_table, recall_table

# --- Splitting -------------------------------------------------------------


def test_short_lines_are_merged_into_a_passage():
    text = "\n".join(["A short line."] * 10)

    passages = split_passages(text)

    assert len(passages) == 1
    assert len(passages[0]) >= MIN_PASSAGE_CHARS


def test_a_long_document_yields_several_passages():
    text = "\n".join(
        [f"Sentence number {i} with enough words to matter." for i in range(40)]
    )

    assert len(split_passages(text)) > 1


def test_passages_are_truncated_at_a_sentence_boundary():
    text = "First sentence here. " * 60

    passages = split_passages(text)

    assert all(len(passage) <= MAX_PASSAGE_CHARS + 5 for passage in passages)


def test_a_trailing_fragment_is_dropped():
    text = "A" * 200 + "\nshort"

    passages = split_passages(text)

    assert all(len(passage) >= MIN_PASSAGE_CHARS for passage in passages)


def test_empty_text_yields_no_passages():
    assert split_passages("") == []


# --- Sampling --------------------------------------------------------------


def make_document(ticker: str, accession: str, lines: int = 20):
    text = "\n".join(
        [
            f"{ticker} sentence {i} with enough words to be a passage."
            for i in range(lines)
        ]
    )
    return (ticker, accession, "10-Q", text)


def test_sampling_is_reproducible_with_the_same_seed():
    documents = [make_document("NVDA", "a1"), make_document("MSFT", "a2")]

    first = sample_passages(documents, per_document=2, seed=5)
    second = sample_passages(documents, per_document=2, seed=5)

    assert [p.text for p in first] == [p.text for p in second]


def test_sampling_draws_from_every_document():
    documents = [make_document("NVDA", "a1"), make_document("MSFT", "a2")]

    passages = sample_passages(documents, per_document=2)

    assert {p.ticker for p in passages} == {"NVDA", "MSFT"}


def test_sampling_respects_per_document():
    documents = [make_document("NVDA", "a1", lines=40)]

    assert len(sample_passages(documents, per_document=3)) == 3


def test_sampling_takes_everything_when_the_document_is_short():
    documents = [make_document("NVDA", "a1", lines=3)]

    passages = sample_passages(documents, per_document=10)

    assert len(passages) == len(split_passages(documents[0][3]))


def test_sampling_rejects_a_non_positive_count():
    with pytest.raises(ValueError, match="per_document"):
        sample_passages([make_document("NVDA", "a1")], per_document=0)


def test_passages_per_document_rounds_up():
    assert passages_per_document(300, 100) == 3
    assert passages_per_document(301, 100) == 4
    assert passages_per_document(10, 0) == 0


# --- Snippets --------------------------------------------------------------


def test_snippets_hide_the_detector_answer():
    passage = Passage(
        ticker="NVDA",
        accession_number="a1",
        section_key="10-Q",
        text="Some passage text.",
        detected_types=("acquisition",),
    )

    snippet = to_snippets([passage])[0]

    # The reviewer must not see what the detector thought, or recall is circular.
    assert snippet.detected_type is None
    assert snippet.proposed is None
    assert snippet.label is None
    assert snippet.ticker == "NVDA"


# --- Recall ----------------------------------------------------------------


def passage(label: str | None, detected: str | None) -> GoldSnippet:
    return GoldSnippet(
        id=f"{label}-{detected}",
        text="text",
        label=label,
        detected_type=detected,
    )


def test_a_detected_real_event_is_a_true_positive():
    table = recall_table([passage("acquisition", "acquisition")])

    row = table.iloc[0]
    assert row["true_positives"] == 1
    assert row["false_negatives"] == 0
    assert row["recall"] == pytest.approx(1.0)


def test_a_missed_real_event_is_a_false_negative():
    table = recall_table([passage("acquisition", None)])

    row = table.iloc[0]
    assert row["true_positives"] == 0
    assert row["false_negatives"] == 1
    assert row["recall"] == pytest.approx(0.0)


def test_recall_combines_hits_and_misses():
    snippets = [
        passage("acquisition", "acquisition"),
        passage("acquisition", None),
        passage("acquisition", None),
        passage("acquisition", "acquisition"),
    ]

    row = recall_table(snippets).iloc[0]

    assert row["true_positives"] == 2
    assert row["false_negatives"] == 2
    assert row["recall"] == pytest.approx(0.5)


def test_a_wrong_class_counts_as_a_miss():
    # Detected as divestiture, labelled acquisition: the acquisition was missed.
    row = recall_table([passage("acquisition", "divestiture")]).iloc[0]

    assert row["false_negatives"] == 1


def test_multiple_detections_are_parsed():
    row = recall_table([passage("acquisition", "acquisition,divestiture")]).iloc[0]

    assert row["true_positives"] == 1


def test_none_labels_do_not_create_a_class():
    table = recall_table([passage("none", None)])

    assert "none" not in table["event_type"].tolist()


def test_skipped_and_unlabelled_passages_are_excluded():
    snippets = [
        passage("acquisition", "acquisition"),
        passage(None, None),
        passage("skip", None),
    ]

    row = recall_table(snippets).iloc[0]

    assert row["true_positives"] + row["false_negatives"] == 1


def test_recall_requires_labels():
    with pytest.raises(ValueError, match="No labelled passages"):
        recall_table([passage(None, None)])


def test_recall_interval_is_reported():
    row = recall_table([passage("acquisition", "acquisition")]).iloc[0]

    assert 0.0 <= row["ci_low"] <= row["recall"] <= row["ci_high"] <= 1.0


# --- Confusion -------------------------------------------------------------


def test_confusion_counts_label_detector_pairs():
    snippets = [
        passage("acquisition", "acquisition"),
        passage("acquisition", None),
        passage("none", "acquisition"),
    ]

    table = confusion_table(snippets)

    counts = {(row.labelled, row.detected): row.count for row in table.itertuples()}
    assert counts[("acquisition", "acquisition")] == 1
    assert counts[("acquisition", "none")] == 1
    assert counts[("none", "acquisition")] == 1


def test_confusion_uses_none_for_silent_detector():
    table = confusion_table([passage("acquisition", None)])

    assert table.iloc[0]["detected"] == "none"
