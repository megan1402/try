"""Tests for :mod:`app.evaluation.review`.

Only the pure answer-parsing is tested; the input loop itself is covered by
running it.
"""

from __future__ import annotations

import pytest

from app.evaluation.gold import LABELS, NONE_LABEL, SKIP_LABEL
from app.evaluation.review import resolve_answer


@pytest.mark.parametrize(("answer", "expected"), [("1", LABELS[0]), ("5", LABELS[4])])
def test_digits_select_a_label(answer: str, expected: str):
    assert resolve_answer(answer, None) == expected


@pytest.mark.parametrize("answer", ["0", "12", "99"])
def test_out_of_range_digits_are_invalid(answer: str):
    assert resolve_answer(answer, "acquisition") is None


def test_digits_also_reach_none_and_skip():
    assert resolve_answer("10", None) == NONE_LABEL
    assert resolve_answer("11", None) == SKIP_LABEL


def test_enter_accepts_the_proposal():
    assert resolve_answer("", "acquisition") == "acquisition"


def test_enter_without_a_proposal_is_invalid():
    assert resolve_answer("", None) is None


def test_shortcuts_map_to_none_and_skip():
    assert resolve_answer("n", None) == NONE_LABEL
    assert resolve_answer("s", None) == SKIP_LABEL


@pytest.mark.parametrize("answer", ["q", "Q", " q "])
def test_q_quits(answer: str):
    assert resolve_answer(answer, None) == "quit"


def test_full_label_names_are_accepted_case_insensitively():
    assert resolve_answer("  Guidance_Cut ", None) == "guidance_cut"


def test_unknown_input_is_invalid():
    assert resolve_answer("vibes", None) is None
