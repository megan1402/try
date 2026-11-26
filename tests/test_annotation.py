"""Tests for the annotation schema, prompt, client and comparison.

No network and no API key: the transport is injected everywhere.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from app.annotation.batch import (
    AnnotationRecord,
    annotate_batch,
    load_annotations,
    load_done,
    predicted_labels,
)
from app.annotation.compare import (
    binary_table,
    confusion_table,
    disagreements,
    per_class_table,
)
from app.annotation.llm import (
    AnnotationError,
    LlmAnnotator,
    _decode,
    build_transport,
    deepseek_transport,
)
from app.annotation.prompt import build_prompt, class_list, rubric_version
from app.annotation.schema import (
    EVENT_CLASSES,
    INDEX_LABEL,
    LABEL_INDEX,
    parse_annotation,
)
from app.evaluation.gold import GoldSnippet


def valid_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "label": 5,
        "event_type": "acquisition",
        "is_genuine_event": True,
        "event_date": "2026-04-15",
        "evidence": "We completed the acquisition of Acme Corp.",
        "reason": "A completed acquisition within the period.",
        "requires_review": False,
    }
    payload.update(overrides)
    return payload


# --- Schema ----------------------------------------------------------------


def test_class_list_is_frozen_and_ordered():
    assert EVENT_CLASSES[0] == "guidance_raise"
    assert EVENT_CLASSES[-2:] == ("none", "skip")
    assert LABEL_INDEX["acquisition"] == 5
    assert INDEX_LABEL[5] == "acquisition"


def test_a_valid_payload_parses():
    annotation = parse_annotation(valid_payload(), passage_id="p1")

    assert annotation.event_type == "acquisition"
    assert annotation.event_date == date(2026, 4, 15)
    assert annotation.passage_id == "p1"


def test_passage_id_comes_from_the_caller_not_the_model():
    annotation = parse_annotation(
        valid_payload(passage_id="model-invented"), passage_id="real-id"
    )

    assert annotation.passage_id == "real-id"


def test_label_and_event_type_must_agree():
    with pytest.raises(ValueError, match="label 5"):
        parse_annotation(valid_payload(event_type="divestiture"), passage_id="p1")


def test_unknown_event_type_is_rejected():
    with pytest.raises(ValueError, match="Unknown event_type"):
        parse_annotation(valid_payload(event_type="vibes"), passage_id="p1")


def test_none_must_not_be_a_genuine_event():
    with pytest.raises(ValueError, match="is_genuine_event"):
        parse_annotation(
            valid_payload(label=10, event_type="none", is_genuine_event=True),
            passage_id="p1",
        )


def test_an_event_class_must_be_genuine():
    with pytest.raises(ValueError, match="is_genuine_event"):
        parse_annotation(valid_payload(is_genuine_event=False), passage_id="p1")


def test_none_must_not_carry_a_date():
    with pytest.raises(ValueError, match="must not carry an event_date"):
        parse_annotation(
            valid_payload(label=10, event_type="none", is_genuine_event=False),
            passage_id="p1",
        )


def test_a_missing_date_is_allowed():
    annotation = parse_annotation(valid_payload(event_date=None), passage_id="p1")

    assert annotation.event_date is None


def test_a_datetime_string_is_truncated_to_a_date():
    annotation = parse_annotation(
        valid_payload(event_date="2026-04-15T00:00:00Z"), passage_id="p1"
    )

    assert annotation.event_date == date(2026, 4, 15)


@pytest.mark.parametrize("value", ["April 2026", "Q3 2026", "last quarter", "2026"])
def test_a_non_iso_date_is_rejected(value: str):
    with pytest.raises(ValueError, match="ISO date"):
        parse_annotation(valid_payload(event_date=value), passage_id="p1")


def test_extra_fields_are_rejected():
    with pytest.raises(ValueError, match="Invalid annotation"):
        parse_annotation(valid_payload(confidence=0.9), passage_id="p1")


def test_empty_evidence_is_rejected():
    with pytest.raises(ValueError, match="Invalid annotation"):
        parse_annotation(valid_payload(evidence=""), passage_id="p1")


def test_a_non_object_payload_is_rejected():
    with pytest.raises(ValueError, match="Expected a JSON object"):
        parse_annotation(["not", "an", "object"], passage_id="p1")


# --- Prompt ----------------------------------------------------------------


def snippet(**overrides: object) -> GoldSnippet:
    data: dict[str, object] = {
        "id": "p1",
        "text": "We completed the acquisition of Acme Corp.",
        "label": "acquisition",
        "proposed": "acquisition",
        "detected_type": "acquisition",
        "ticker": "NVDA",
        "accession_number": "acc-1",
        "section_key": "part_i_item_2",
    }
    data.update(overrides)
    return GoldSnippet(**data)  # type: ignore[arg-type]


def test_prompt_contains_the_rubric_and_the_passage():
    prompt = build_prompt(snippet())

    assert "Annotation rubric" in prompt.text
    assert "We completed the acquisition of Acme Corp." in prompt.text
    assert "acquisition" in prompt.text


def test_prompt_never_reveals_the_human_label_or_detector():
    """The whole experiment depends on this.

    The rubric text discusses labels and detector output (it explains that they
    are hidden), so grepping for those words proves nothing. The real property
    is that the *snippet's* label values never reach the prompt as an answer.
    """
    prompt = build_prompt(snippet())

    assert '"label": "acquisition"' not in prompt.text
    assert "detected_type: acquisition" not in prompt.text
    assert "proposed: acquisition" not in prompt.text
    # The passage is present as content, never as an answer.
    assert "We completed the acquisition of Acme Corp." in prompt.text


def test_prompt_is_identical_for_different_hidden_labels():
    """Two snippets with the same text but different labels must prompt alike.

    This is the strongest form of the check: if the label leaked anywhere, the
    two prompts would differ.
    """
    first = build_prompt(snippet(label="acquisition", detected_type="acquisition"))
    second = build_prompt(snippet(label="none", detected_type="divestiture"))

    assert first.text == second.text


def test_prompt_includes_filing_metadata():
    prompt = build_prompt(snippet())

    assert "NVDA" in prompt.text
    assert "acc-1" in prompt.text


def test_prompt_includes_context_when_it_differs():
    prompt = build_prompt(snippet(context="Surrounding sentence here."))

    assert "Surrounding context" in prompt.text
    assert "Surrounding sentence here." in prompt.text


def test_prompt_omits_context_when_it_matches_the_text():
    text = "We completed the acquisition of Acme Corp."
    prompt = build_prompt(snippet(context=text))

    assert "Surrounding context" not in prompt.text


def test_class_list_is_numbered():
    listing = class_list()

    assert listing.startswith("1. guidance_raise")
    assert "10. none" in listing
    assert "11. skip" in listing


def test_rubric_version_is_a_short_hash():
    version = rubric_version()

    assert len(version) == 12
    assert all(character in "0123456789abcdef" for character in version)


# --- Decoding model output -------------------------------------------------


def test_plain_json_decodes():
    assert _decode('{"a": 1}') == {"a": 1}


def test_fenced_json_decodes():
    raw = '```json\n{"a": 1}\n```'

    assert _decode(raw) == {"a": 1}


def test_json_with_a_leading_sentence_decodes():
    raw = 'Here is the annotation:\n{"a": 1}'

    assert _decode(raw) == {"a": 1}


def test_truncated_json_raises():
    with pytest.raises(json.JSONDecodeError):
        _decode('{"a": 1')


# --- Annotator -------------------------------------------------------------


def test_annotator_returns_a_validated_annotation():
    annotator = LlmAnnotator(transport=lambda prompt: json.dumps(valid_payload()))

    annotation = annotator.annotate("prompt", passage_id="p1")

    assert annotation.event_type == "acquisition"


def test_annotator_retries_a_malformed_response():
    responses = iter(["not json", json.dumps(valid_payload())])
    annotator = LlmAnnotator(
        transport=lambda prompt: next(responses), attempts=2, sleep=lambda _: None
    )

    annotation = annotator.annotate("prompt", passage_id="p1")

    assert annotation.event_type == "acquisition"


def test_annotator_raises_after_every_attempt_fails():
    annotator = LlmAnnotator(
        transport=lambda prompt: "not json", attempts=2, sleep=lambda _: None
    )

    with pytest.raises(AnnotationError, match="after 2 attempt"):
        annotator.annotate("prompt", passage_id="p1")


def test_annotator_retries_an_invalid_schema():
    bad = json.dumps(valid_payload(event_type="divestiture"))
    responses = iter([bad, json.dumps(valid_payload())])
    annotator = LlmAnnotator(
        transport=lambda prompt: next(responses), attempts=2, sleep=lambda _: None
    )

    assert annotator.annotate("prompt", passage_id="p1").event_type == "acquisition"


# --- Batch -----------------------------------------------------------------


def test_batch_writes_records_with_provenance(tmp_path):
    output = tmp_path / "annotations.jsonl"
    annotator = LlmAnnotator(transport=lambda prompt: json.dumps(valid_payload()))

    annotated, failed = annotate_batch(
        [snippet()],
        annotator=annotator,
        model="test-model",
        output=output,
        sampling_method="unit-test",
        progress=False,
    )

    assert (annotated, failed) == (1, 0)
    record = load_annotations(output)[0]
    assert record.model == "test-model"
    assert record.sampling_method == "unit-test"
    assert record.human_verified is False
    assert record.prompt_version
    assert record.rubric_version


def test_batch_skips_passages_already_done(tmp_path):
    output = tmp_path / "annotations.jsonl"
    annotator = LlmAnnotator(transport=lambda prompt: json.dumps(valid_payload()))
    annotate_batch(
        [snippet()], annotator=annotator, model="m", output=output, progress=False
    )

    annotated, _ = annotate_batch(
        [snippet()], annotator=annotator, model="m", output=output, progress=False
    )

    assert annotated == 0
    assert len(load_annotations(output)) == 1


def test_batch_continues_after_a_failure(tmp_path):
    output = tmp_path / "annotations.jsonl"
    calls = {"n": 0}

    def transport(prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return "not json"
        return json.dumps(valid_payload())

    annotator = LlmAnnotator(transport=transport, attempts=1, sleep=lambda _: None)
    snippets = [snippet(id="a"), snippet(id="b")]

    annotated, failed = annotate_batch(
        snippets, annotator=annotator, model="m", output=output, progress=False
    )

    assert (annotated, failed) == (1, 1)


def test_load_done_ignores_a_truncated_line(tmp_path):
    output = tmp_path / "annotations.jsonl"
    output.write_text('{"passage_id": "a"}\n{"passage_id": "b"', encoding="utf-8")

    assert load_done(output) == {"a"}


def test_load_done_of_a_missing_file_is_empty(tmp_path):
    assert load_done(tmp_path / "nope.jsonl") == set()


def test_predicted_labels_maps_ids_to_classes(tmp_path):
    output = tmp_path / "annotations.jsonl"
    annotator = LlmAnnotator(transport=lambda prompt: json.dumps(valid_payload()))
    annotate_batch(
        [snippet()], annotator=annotator, model="m", output=output, progress=False
    )

    assert predicted_labels(load_annotations(output)) == {"p1": "acquisition"}


def test_record_marks_ai_labels_as_unverified():
    record = AnnotationRecord.build(
        snippet=snippet(),
        annotation=parse_annotation(valid_payload(), passage_id="p1"),
        model="m",
        prompt_version="v1",
        rubric_version="r1",
        sampling_method="s",
    )

    assert record.human_verified is False


# --- Transports ------------------------------------------------------------


def test_deepseek_transport_requires_a_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        deepseek_transport()


def test_deepseek_transport_posts_to_the_fixed_endpoint(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"choices": [{"message": {"content": '{"a": 1}'}}]}

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("app.annotation.llm.requests.post", fake_post)

    transport = deepseek_transport()
    result = transport("prompt")

    assert result == '{"a": 1}'
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["Authorization"] == "Bearer test-key"
    body = captured["json"]
    assert isinstance(body, dict)
    assert body["temperature"] == 0.0
    assert body["response_format"] == {"type": "json_object"}


def test_build_transport_rejects_an_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        build_transport("openai")


def test_build_transport_selects_deepseek(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    transport = build_transport("deepseek")

    assert callable(transport)


# --- Comparison ------------------------------------------------------------


def labelled(snippet_id: str, label: str) -> GoldSnippet:
    return GoldSnippet(id=snippet_id, text="text", label=label)


def test_per_class_table_counts_correctly():
    snippets = [labelled("a", "acquisition"), labelled("b", "none")]
    predictions = {"a": "acquisition", "b": "acquisition"}

    table = per_class_table(snippets, predictions)

    acquisition = table[table["event_type"] == "acquisition"].iloc[0]
    assert acquisition["support"] == 1
    assert acquisition["predicted"] == 2
    assert acquisition["tp"] == 1
    assert acquisition["fp"] == 1
    assert acquisition["precision"] == pytest.approx(0.5)


def test_per_class_table_reports_zero_not_nan_for_an_unpredicted_class():
    snippets = [labelled("a", "acquisition"), labelled("b", "none")]
    predictions = {"a": "none", "b": "none"}

    table = per_class_table(snippets, predictions)

    acquisition = table[table["event_type"] == "acquisition"].iloc[0]
    assert acquisition["predicted"] == 0
    assert acquisition["precision"] == pytest.approx(0.0)
    assert acquisition["recall"] == pytest.approx(0.0)
    assert acquisition["f1"] == pytest.approx(0.0)


def test_per_class_table_has_intervals():
    snippets = [labelled("a", "acquisition"), labelled("b", "none")]
    predictions = {"a": "acquisition", "b": "acquisition"}

    table = per_class_table(snippets, predictions)

    # A class with no support or no predictions has no interval to report, so
    # only the defined rows are checked.
    defined = table.dropna(subset=["precision_low", "recall_low"])
    assert not defined.empty
    for _, row in defined.iterrows():
        assert row["precision_low"] <= row["precision"] <= row["precision_high"]
        assert row["recall_low"] <= row["recall"] <= row["recall_high"]


def test_binary_table_reports_event_versus_none():
    snippets = [
        labelled("a", "acquisition"),
        labelled("b", "none"),
        labelled("c", "none"),
    ]
    predictions = {"a": "acquisition", "b": "none", "c": "acquisition"}

    table = binary_table(snippets, predictions)
    values = dict(zip(table["metric"], table["value"], strict=True))

    assert values["precision"] == pytest.approx(0.5)
    assert values["recall"] == pytest.approx(1.0)
    assert values["specificity"] == pytest.approx(0.5)


def test_binary_table_has_intervals():
    table = binary_table([labelled("a", "acquisition")], {"a": "acquisition"})

    defined = table.dropna(subset=["value"])
    assert (defined["ci_low"] <= defined["value"]).all()
    assert (defined["value"] <= defined["ci_high"]).all()


def test_confusion_table_counts_pairs():
    snippets = [labelled("a", "acquisition"), labelled("b", "none")]
    predictions = {"a": "none", "b": "none"}

    table = confusion_table(snippets, predictions)

    counts = {(row.human, row.model): row.count for row in table.itertuples()}
    assert counts[("acquisition", "none")] == 1
    assert counts[("none", "none")] == 1


def test_disagreements_lists_only_mismatches():
    snippets = [labelled("a", "acquisition"), labelled("b", "none")]
    predictions = {"a": "none", "b": "none"}

    rows = disagreements(snippets, predictions)

    assert [row["id"] for row in rows] == ["a"]


def test_comparison_requires_scored_snippets():
    with pytest.raises(ValueError, match="No scored snippets"):
        per_class_table([labelled("a", "skip")], {"a": "none"})
