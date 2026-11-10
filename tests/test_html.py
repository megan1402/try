"""Tests for :mod:`app.extraction.html`.

Small inline documents cover each rule in isolation; the saved 10-Q fixture
covers the combination of rules on realistic, machine-generated markup. No test
touches the network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.extraction.html import html_to_text

FIXTURES = Path(__file__).parent / "fixtures"


def test_script_and_style_content_is_removed():
    html = "<style>.a{color:red}</style><script>var x = 1;</script><div>Body</div>"

    assert html_to_text(html) == "Body"


def test_each_block_becomes_its_own_line():
    html = "<div>First</div><div>Second</div><p>Third</p>"

    assert html_to_text(html).splitlines() == ["First", "Second", "Third"]


def test_nested_blocks_do_not_run_together():
    html = "<div>Outer<div>Inner</div>Tail</div>"

    assert html_to_text(html).splitlines() == ["Outer", "Inner", "Tail"]


def test_table_cells_stay_on_one_line():
    html = (
        "<table>"
        "<tr><td>Cash</td><td>12</td></tr>"
        "<tr><td>Debt</td><td>7</td></tr>"
        "</table>"
    )

    assert html_to_text(html).splitlines() == ["Cash 12", "Debt 7"]


def test_br_starts_a_new_line():
    html = "<div>Line one<br/>Line two</div>"

    assert html_to_text(html).splitlines() == ["Line one", "Line two"]


def test_entities_are_decoded():
    html = "<div>Caf&#233; &#8220;quoted&#8221; &#8217;tis</div>"

    assert html_to_text(html) == "Café “quoted” ’tis"


def test_non_breaking_spaces_become_ordinary_spaces():
    html = "<div>revenue&#160;&#160;$1,000</div>"

    assert html_to_text(html) == "revenue $1,000"


def test_whitespace_runs_collapse_and_blank_lines_disappear():
    html = "<div>  spaced   out  </div><div>   </div><div>next</div>"

    assert html_to_text(html) == "spaced out\nnext"


def test_empty_document_produces_no_text():
    assert html_to_text("") == ""


@pytest.mark.parametrize("value", [None, b"<div>x</div>", 42])
def test_non_string_input_is_rejected(value):
    with pytest.raises(TypeError):
        html_to_text(value)


def test_realistic_filing_keeps_one_block_per_line():
    html = (FIXTURES / "filing_10q.html").read_text(encoding="utf-8")

    lines = html_to_text(html).splitlines()

    assert "Item 1. Financial Statements (Unaudited)" in lines
    assert (
        "Item 2. Management’s Discussion and Analysis of Financial Condition "
        "and Results of Operations"
    ) in lines
    assert "Cash and cash equivalents $ 12,345" in lines
    assert all(line == line.strip() for line in lines)


def test_realistic_filing_contains_no_script_or_style_noise():
    html = (FIXTURES / "filing_10q.html").read_text(encoding="utf-8")

    text = html_to_text(html)

    assert "Item 98" not in text
    assert "Item 99" not in text
    assert "\u00a0" not in text
