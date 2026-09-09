"""Synthetic receipt fixtures plus the pinned public native Qwen tokenizer."""
from copy import deepcopy
from pathlib import Path

import pytest

from cot_filler.core import PROMPT_VERSION, canonical, digest, validate_candidate
from cot_filler.corpus_worker import gap_at
from .generation_contract import bind_generations, ACCEPTANCE_POLICY
from .render import Qwen38ExperimentalCotRenderer, training_row

ASSETS = Path(__file__).resolve().parents[1] / "qwen38_no_cot/assets"


def fixture(events=None, targets=None, text="I will inspect the function before choosing a repair."):
    events = events or [
        {"event_id": "u", "kind": "message", "role": "user", "content": "Inspect the function before editing. TOKEN_KEEP_ME"},
        {"event_id": "a", "kind": "message", "role": "assistant", "content": "Checking the function."}]
    targets = targets if targets is not None else [e["event_id"] for e in events if e["role"] == "assistant"]
    original = {"schema": "cot.trace.v1", "trace_id": "synthetic-test-trace", "events": events,
                "gap_targets": [{"event_id": e} for e in targets]}
    config = {"model": "synthetic-test-model", "max_output_tokens": 512, "temperature": 0.3,
              "chat_template_kwargs": {"thinking": True}}
    records = []
    for index, event in enumerate(events):
        if event["event_id"] not in targets:
            continue
        gap = gap_at(original, index)
        raw = {"generator": {"provider": "openai-compatible", "model": config["model"],
            "prompt_version": PROMPT_VERSION, "tokenizer_sha256": "synthetic-test-tokenizer",
            "parameters": {"model": config["model"], "max_tokens": 512, "temperature": 0.3,
                           "chat_template_kwargs": config["chat_template_kwargs"]}},
            "prompt_hash": gap["prompt_hash"], "prompt_tokens_local": 100,
            "counted_chat_template_kwargs": config["chat_template_kwargs"], "usage": {"prompt_tokens": 100},
            "finish_reason": "stop", "flags": [], "synthetic": True, "lookahead_conditioned": True}
        stored = {**raw, "flags": sorted(set(validate_candidate(text))), "candidate_hash": digest(text)}
        records.append({"gap": gap, "candidate": {"gap_id": gap["id"], "text": text, "data": canonical(stored),
            **{k: gap[k] for k in ("prefix_hash", "lookahead_hash", "prompt_hash")}},
            "generation_receipt": {"gap_id": gap["id"], "stage": "generate", "text": text, "data": canonical(raw)}})
    return original, records, config


@pytest.fixture(scope="module")
def renderer():
    return Qwen38ExperimentalCotRenderer(ASSETS)


def selected(renderer, result):
    return renderer.tokenizer.decode([i for i, label in zip(result["input_ids"], result["labels"]) if label != -100])


def test_native_xhigh_exact_ids_masked_cot_and_no_semantic_claim(renderer):
    original, records, config = fixture()
    before = deepcopy((original, records, config))
    result = renderer.render_generated_trace(original, records, config)
    text = records[0]["candidate"]["text"]
    messages = [{"role": "user", "content": original["events"][0]["content"]},
                {"role": "assistant", "content": original["events"][1]["content"], "reasoning_content": text}]
    native = renderer.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,
        enable_thinking=True, preserve_thinking=True, reasoning_effort="xhigh")
    assert result["full_rendered_text"] == native
    assert result["input_ids"] == renderer.tokenizer.encode(native, add_special_tokens=False)
    assert selected(renderer, result) == "Checking the function.<|im_end|>"
    assert "TOKEN_KEEP_ME" in native and text in native and text not in selected(renderer, result)
    [start, end] = result["cot_mask_audit"]["retained_token_ranges"][0]
    assert renderer.tokenizer.decode(result["input_ids"][start:end]) == "<think>\n" + text + "\n</think>\n\n"
    assert all(label == -100 for label in result["labels"][start:end])
    p = result["cot_provenance"]
    assert p["reviewed"] is False and p["semantic_quality_qualified"] is False
    assert p["future_information_checked"] is False and p["unreviewed_gap_count"] == 1
    assert p["acceptance_policy"] == ACCEPTANCE_POLICY
    assert (original, records, config) == before


def test_all_valid_generations_admitted_without_review_not_quality_claim(renderer):
    args = fixture(text="The future test passed, so I will report completion.")
    result = renderer.render_generated_trace(*args)
    assert args[1][0]["candidate"]["text"] in result["full_rendered_text"]
    assert result["cot_provenance"]["future_information_checked"] is False


def test_zero_generated_gaps_keeps_full_original_trace(renderer):
    source, _, config = fixture()
    result = renderer.render_generated_trace(source, [], config)
    assert result["cot_mask_audit"]["filled_gap_count"] == 0
    assert result["cot_mask_audit"]["retained_token_ranges"] == []
    assert selected(renderer, result) == "Checking the function.<|im_end|>"
    assert result["cot_provenance"]["unfilled_gap_count"] == 1


def test_replay_uses_empty_masked_think_and_strips_original_reasoning(renderer):
    messages = [{"role": "user", "content": "VISIBLE_USER"},
        {"role": "assistant", "content": "VISIBLE_ACTION", "reasoning_content": "ORIGINAL_REASONING_REMOVE"}]
    original = deepcopy(messages)
    result = renderer.render_replay_messages(messages, source_digest=digest(messages), trace_id="replay-fixture")
    assert "ORIGINAL_REASONING_REMOVE" not in result["full_rendered_text"]
    assert "<think>\n\n</think>" in result["full_rendered_text"]
    assert selected(renderer, result) == "VISIBLE_ACTION<|im_end|>"
    assert result["cot_provenance"]["generation_config_sha256"] is None
    assert result["cot_provenance"]["generated_gaps"] == []
    assert messages == original
    row = training_row(result, group_id="replay-session", source="replay")
    assert row["metadata"]["provenance"]["generated_cot"]["source_kind"] == "replay"


def test_tool_call_body_and_eot_targets_observation_masked(renderer):
    events = [
        {"event_id": "u", "kind": "message", "role": "user", "content": "Inspect the function."},
        {"event_id": "a", "kind": "tool_call", "role": "assistant", "name": "exec", "call_id": "c1",
         "data": {"block": {"type": "custom_tool_call", "input": "print('hello')"}}},
        {"event_id": "o", "kind": "tool_result", "role": "tool", "call_id": "c1", "content": "OBSERVATION_ONLY", "data": {"block": {}}},
        {"event_id": "f", "kind": "message", "role": "assistant", "content": "Done."}]
    result = renderer.render_generated_trace(*fixture(events, targets=["a"]))
    visible = selected(renderer, result)
    assert "print('hello')" in visible and visible.count("<|im_end|>") == 2
    assert "OBSERVATION_ONLY" in result["full_rendered_text"] and "OBSERVATION_ONLY" not in visible
    assert "<think>" not in visible


def test_cutoff_inside_filled_reasoning_keeps_exact_prefix_no_eot(renderer):
    args = fixture([
        {"event_id": "u0", "kind": "message", "role": "user", "content": "Start."},
        {"event_id": "a0", "kind": "message", "role": "assistant", "content": "Starting."},
        {"event_id": "u", "kind": "message", "role": "user", "content": "Inspect the function."},
        {"event_id": "a", "kind": "message", "role": "assistant", "content": "Inspecting."}], targets=["a"])
    full = renderer.render_generated_trace(*args)
    lo, hi = full["cot_mask_audit"]["retained_token_ranges"][0]
    cutoff = lo + 5
    short = Qwen38ExperimentalCotRenderer(ASSETS, max_tokens=cutoff).render_generated_trace(*args)
    assert short["input_ids"] == full["input_ids"][:cutoff]
    assert short["labels"] == full["labels"][:cutoff]
    assert short["cot_mask_audit"]["retained_token_ranges"] == [[lo, cutoff]]
    assert short["sequence_audit"]["eot_appended"] is False


def test_cutoff_before_cot_records_zero_retained_but_preserves_receipt(renderer):
    args = fixture([
        {"event_id": "u0", "kind": "message", "role": "user", "content": "Start."},
        {"event_id": "a0", "kind": "message", "role": "assistant", "content": "Starting."},
        {"event_id": "u", "kind": "message", "role": "user", "content": "Inspect the function."},
        {"event_id": "a", "kind": "message", "role": "assistant", "content": "Inspecting."}], targets=["a"])
    full = renderer.render_generated_trace(*args)
    cutoff = full["cot_mask_audit"]["retained_token_ranges"][0][0]
    result = Qwen38ExperimentalCotRenderer(ASSETS, max_tokens=cutoff).render_generated_trace(*args)
    assert result["cot_mask_audit"]["filled_gap_count"] == 1
    assert result["cot_mask_audit"]["retained_filled_gap_count"] == 0


def test_actual262144_cutoff_and_masks_keep_original_prefix(renderer):
    args = fixture([
        {"event_id": "u0", "kind": "message", "role": "user", "content": "Start."},
        {"event_id": "a0", "kind": "message", "role": "assistant", "content": "Starting."},
        {"event_id": "u", "kind": "message", "role": "user", "content": "context " * 270000},
        {"event_id": "a", "kind": "message", "role": "assistant", "content": "Inspecting."}], targets=["a"])
    result = renderer.render_generated_trace(*args)
    full_ids = renderer.tokenizer.encode(result["full_rendered_text"], add_special_tokens=False)
    assert len(full_ids) > 262144
    assert len(result["input_ids"]) == 262144
    assert result["input_ids"] == full_ids[:262144]
    assert result["sequence_audit"]["eot_appended"] is False
    assert result["cot_mask_audit"]["filled_gap_count"] == 1
    assert result["cot_mask_audit"]["retained_filled_gap_count"] == 0
    assert selected(renderer, result) == "Starting.<|im_end|>"


@pytest.mark.parametrize("mutation", ["wrong_text", "wrong_event", "wrong_prefix", "wrong_config", "missing_raw", "truncated", "fake"])
def test_changed_generation_receipts_rejected(mutation):
    source, entries, config = fixture()
    if mutation == "wrong_text": entries[0]["candidate"]["text"] += " Changed."
    elif mutation == "wrong_event": entries[0]["gap"]["event_index"] = 0
    elif mutation == "wrong_prefix": entries[0]["candidate"]["prefix_hash"] = "0" * 64
    elif mutation == "wrong_config": config["max_output_tokens"] = 1024
    elif mutation == "missing_raw": del entries[0]["generation_receipt"]
    else:
        import json
        raw = json.loads(entries[0]["generation_receipt"]["data"])
        if mutation == "truncated": raw["finish_reason"] = "length"
        else: raw["generator"]["provider"] = "fake"
        entries[0]["generation_receipt"]["data"] = canonical(raw)
    with pytest.raises(ValueError): bind_generations(source, entries, config)


def test_duplicate_placement_rejected():
    source, entries, config = fixture()
    with pytest.raises(ValueError): bind_generations(source, entries + entries, config)


def test_replay_and_trace_rows_validate_under_only_new_contract(renderer):
    from .data import validate_row
    from training.qwen38_cot_masked.data import validate_row as reviewed_validate
    for result, source in [(renderer.render_generated_trace(*fixture()), "trace"),
                           (renderer.render_replay_messages([{"role": "user", "content": "Hello"},
                            {"role": "assistant", "content": "Hi"}], source_digest="a"*64, trace_id="r"), "replay")]:
        row = training_row(result, group_id="synthetic-session", source=source)
        validate_row(row, renderer.identity)
        with pytest.raises(ValueError): reviewed_validate(row, renderer.identity)
