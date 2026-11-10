"""Tests for :mod:`app.evaluation.gold`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.evaluation import gold


def write_lines(path: Path, *lines: str) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def make_snippet(snippet_id: str = "a#1", label: str | None = None) -> gold.GoldSnippet:
    return gold.GoldSnippet(id=snippet_id, text="We raised our guidance.", label=label)


# --- Reading and writing ---------------------------------------------------


def test_roundtrip_preserves_every_field(tmp_path: Path):
    path = tmp_path / "gold.jsonl"
    snippets = [
        gold.GoldSnippet(
            id="a#1",
            text="We raised our guidance.",
            label="guidance_raise",
            proposed="guidance_raise",
            notes="clear",
        ),
        gold.GoldSnippet(id="a#2", text="Revenue grew.", label="none"),
    ]

    gold.save_gold(snippets, path)

    assert gold.load_gold(path) == snippets


def test_file_is_one_json_object_per_line(tmp_path: Path):
    path = tmp_path / "gold.jsonl"
    gold.save_gold([make_snippet()], path)

    lines = path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 1
    assert json.loads(lines[0])["id"] == "a#1"


def test_blank_lines_are_ignored(tmp_path: Path):
    path = write_lines(
        tmp_path / "gold.jsonl",
        '{"id": "a#1", "text": "x"}',
        "",
        '{"id": "a#2", "text": "y"}',
    )

    assert [snippet.id for snippet in gold.load_gold(path)] == ["a#1", "a#2"]


def test_empty_label_counts_as_unlabelled(tmp_path: Path):
    path = write_lines(
        tmp_path / "gold.jsonl", '{"id": "a#1", "text": "x", "label": ""}'
    )

    assert gold.load_gold(path)[0].label is None


def test_unknown_label_is_rejected(tmp_path: Path):
    path = write_lines(
        tmp_path / "gold.jsonl", '{"id": "a#1", "text": "x", "label": "vibes"}'
    )

    with pytest.raises(ValueError, match="unknown label"):
        gold.load_gold(path)


def test_duplicate_ids_are_rejected(tmp_path: Path):
    path = write_lines(
        tmp_path / "gold.jsonl",
        '{"id": "a#1", "text": "x"}',
        '{"id": "a#1", "text": "y"}',
    )

    with pytest.raises(ValueError, match="duplicate id"):
        gold.load_gold(path)


def test_invalid_json_reports_the_line(tmp_path: Path):
    path = write_lines(
        tmp_path / "gold.jsonl", '{"id": "a#1", "text": "x"}', "not json"
    )

    with pytest.raises(ValueError, match=":2: invalid JSON"):
        gold.load_gold(path)


def test_empty_text_is_rejected():
    with pytest.raises(ValueError, match="text must not be empty"):
        gold.GoldSnippet(id="a#1", text="   ")


# --- Selection helpers -----------------------------------------------------


def test_labelled_excludes_unlabelled_and_skipped():
    snippets = [
        make_snippet("a#1", "guidance_raise"),
        make_snippet("a#2", None),
        make_snippet("a#3", gold.SKIP_LABEL),
        make_snippet("a#4", "none"),
    ]

    assert [s.id for s in gold.labelled(snippets)] == ["a#1", "a#4"]
    assert [s.id for s in gold.unlabelled(snippets)] == ["a#2"]


def test_with_label_replaces_only_the_named_snippet():
    snippets = [make_snippet("a#1"), make_snippet("a#2")]

    updated = gold.with_label(snippets, "a#2", "acquisition")

    assert [s.label for s in updated] == [None, "acquisition"]
    assert snippets[1].label is None  # the input is not mutated.


def test_with_label_rejects_unknown_label_or_id():
    with pytest.raises(ValueError, match="Unknown label"):
        gold.with_label([make_snippet()], "a#1", "vibes")
    with pytest.raises(ValueError, match="No snippet"):
        gold.with_label([make_snippet()], "zzz", "none")


# --- The file we actually ship ---------------------------------------------


def test_committed_gold_file_is_valid():
    snippets = gold.load_gold()

    assert snippets, "the golden set should not be empty"
    assert len({snippet.id for snippet in snippets}) == len(snippets)
    assert all(snippet.text.strip() for snippet in snippets)
