"""Independent CPU regressions for source-event to native-turn loss boundaries.

Uses the real pinned public tokenizer and the existing pure-Python NeMo shift.
No model weights, backend requests, GPU work, or production artifacts are used.
"""
from copy import deepcopy
from pathlib import Path

import pytest

from training.qwen38_no_cot.nemo_data import shift_next_token_row
from training.qwen38_no_cot.upstream_manager.render import EOT, UnsupportedMessage
from .source_adapters import canonical_messages, rollout_messages, atif_messages
from .render import Qwen38Renderer, training_row

ASSETS = Path(__file__).resolve().parents[1] / "qwen38_no_cot/assets"
ANNOUNCE = "I will inspect the configuration."
OBSERVATION = "ORIGINAL_TOOL_OBSERVATION"
FINAL = "The configuration is present."


def message(event_id, role, content, **metadata):
    return {"event_id": event_id, "kind": "message", "role": role, "content": content,
            "source": {"pointer": "/input/" + event_id},
            "data": {"block": {"type": "message", "role": role}}, **metadata}


def call(event_id="call", call_id="c1", cmd="cat config.txt", **metadata):
    return {"event_id": event_id, "kind": "tool_call", "role": "assistant", "name": "exec_command",
            "call_id": call_id, "source": {"pointer": "/input/" + event_id},
            "data": {"block": {"type": "function_call", "name": "exec_command", "call_id": call_id,
                               "arguments": {"cmd": cmd}}}, **metadata}


def observation(event_id="observation", call_id="c1", content=OBSERVATION):
    return {"event_id": event_id, "kind": "tool_result", "role": "tool", "call_id": call_id,
            "content": content, "source": {"pointer": "/input/" + event_id},
            "data": {"block": {"type": "function_call_output", "call_id": call_id}}}


def events():
    return [message("u", "user", "Inspect the configuration."),
            message("announce", "assistant", ANNOUNCE, channel="commentary"), call(), observation(),
            message("final", "assistant", FINAL, channel="final")]


def selected(renderer, result):
    return renderer.tokenizer.decode([token for token, label in zip(result["input_ids"], result["labels"])
                                     if label != -100])


def call_text(cmd="cat config.txt"):
    return "<tool_call>\n<function=exec_command>\n<parameter=cmd>\n" + cmd + "\n</parameter>\n</function>\n</tool_call>"


@pytest.fixture(scope="module")
def renderer():
    return Qwen38Renderer(ASSETS)


def test_canonical_announcement_call_single_native_assistant_and_exact_mask(renderer):
    original = events()
    before = deepcopy(original)
    messages, audit = canonical_messages(original)
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "assistant"]
    assert messages[1]["content"] == ANNOUNCE
    assert messages[1]["tool_calls"][0]["id"] == "c1"
    result = renderer.render(messages, turn_boundary_audit=audit["turn_boundary_audit"])
    native = renderer.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,
        enable_thinking=False, preserve_thinking=True)
    assert result["full_rendered_text"] == native
    assert result["input_ids"] == renderer.tokenizer.encode(native, add_special_tokens=False)
    assert selected(renderer, result) == ANNOUNCE + "\n\n" + call_text() + EOT + FINAL + EOT
    assert ANNOUNCE + EOT not in native
    assert native.count("<|im_start|>assistant\n") == 2
    assert OBSERVATION in native and OBSERVATION not in selected(renderer, result)
    assert "<think>" not in selected(renderer, result) and "</think>" not in selected(renderer, result)
    assert "<|im_start|>" not in selected(renderer, result)
    assert original == before
    assert audit["turn_boundary_audit"]
    # Independently reconstruct the supervised ranges from native content order.
    spans = []
    for body in [ANNOUNCE + "\n\n" + call_text() + EOT, FINAL + EOT]:
        start = native.index(body)
        spans.append((start, start + len(body)))
    for (start, end), token, label in zip(result["token_offsets"], result["input_ids"], result["labels"]):
        inside = any(left <= start and end <= right and start < end for left, right in spans)
        assert label == (token if inside else -100)


def test_parallel_calls_and_results_keep_order_and_ids(renderer):
    source = [message("u", "user", "Inspect two files."), message("a", "assistant", ANNOUNCE),
              call("c-a", "a", "cat a"), call("c-b", "b", "cat b"),
              observation("o-b", "b", "B_RESULT"), observation("o-a", "a", "A_RESULT"),
              message("f", "assistant", FINAL, channel="final")]
    messages, audit = canonical_messages(source)
    assert [c["id"] for c in messages[1]["tool_calls"]] == ["a", "b"]
    assert [m["tool_call_id"] for m in messages if m["role"] == "tool"] == ["b", "a"]
    result = renderer.render(messages, turn_boundary_audit=audit["turn_boundary_audit"])
    target = selected(renderer, result)
    assert target.count(EOT) == 2
    assert target.index("cat a") < target.index("cat b") < target.index(EOT)
    assert "B_RESULT" not in target and "A_RESULT" not in target


def test_raw_rollout_response_items_group_without_ui_duplicates(renderer):
    raw = [{"type": "session_meta", "payload": {"session_id": "cpu-fixture"}},
           {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Inspect."}]}},
           {"type": "response_item", "payload": {"type": "message", "role": "assistant", "channel": "commentary", "content": [{"type": "output_text", "text": ANNOUNCE}]}},
           {"type": "event_msg", "payload": {"type": "agent_message", "message": ANNOUNCE}},
           {"type": "response_item", "payload": {"type": "function_call", "name": "exec_command", "call_id": "c1", "arguments": '{"cmd":"cat config.txt"}'}},
           {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "c1", "output": OBSERVATION}},
           {"type": "response_item", "payload": {"type": "message", "role": "assistant", "channel": "final", "content": [{"type": "output_text", "text": FINAL}]}}]
    before = deepcopy(raw)
    messages, audit, group = rollout_messages(raw)
    assert group == "session:cpu-fixture"
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "assistant"]
    target = selected(renderer, renderer.render(messages, turn_boundary_audit=audit["turn_boundary_audit"]))
    assert target == ANNOUNCE + "\n\n" + call_text() + EOT + FINAL + EOT
    assert target.count(ANNOUNCE) == 1 and raw == before


def test_atif_separate_announcement_and_action_group_before_observation(renderer):
    doc = {"session_id": "atif-fixture", "steps": [
        {"source": "user", "message": "Inspect."}, {"source": "agent", "message": ANNOUNCE},
        {"source": "agent", "message": "", "tool_calls": [{"tool_call_id": "c1", "function_name": "exec_command", "arguments": {"cmd": "cat config.txt"}}],
         "observation": {"results": [{"source_call_id": "c1", "content": OBSERVATION}]}},
        {"source": "agent", "message": FINAL}]}
    before = deepcopy(doc)
    messages, audit, _ = atif_messages(doc)
    target = selected(renderer, renderer.render(messages, turn_boundary_audit=audit["turn_boundary_audit"]))
    assert target == ANNOUNCE + "\n\n" + call_text() + EOT + FINAL + EOT
    assert doc == before


@pytest.mark.parametrize("role", ["user", "system", "developer"])
def test_real_role_barriers_preserve_prior_assistant_boundary(renderer, role):
    source = [message("u", "user", "Begin."), message("a", "assistant", "Before the barrier."),
              message("barrier", role, "NEW_CONTEXT"), call(), observation(),
              message("f", "assistant", FINAL, channel="final")]
    messages, audit = canonical_messages(source)
    assistants = [m for m in messages if m["role"] == "assistant"]
    assert len(assistants) == 3 and "tool_calls" not in assistants[0]
    result = renderer.render(messages, turn_boundary_audit=audit["turn_boundary_audit"])
    assert selected(renderer, result).count(EOT) == 3
    assert "NEW_CONTEXT" not in selected(renderer, result)


def test_explicit_final_remains_separate_from_later_call(renderer):
    source = [message("u", "user", "Begin."), message("f0", "assistant", "First task finished.", channel="final"),
              message("a", "assistant", ANNOUNCE, channel="commentary"), call(), observation()]
    messages, audit = canonical_messages(source)
    assistants = [m for m in messages if m["role"] == "assistant"]
    assert len(assistants) == 2
    assert assistants[0]["content"] == "First task finished." and "tool_calls" not in assistants[0]
    target = selected(renderer, renderer.render(messages, turn_boundary_audit=audit["turn_boundary_audit"]))
    assert target == "First task finished." + EOT + ANNOUNCE + "\n\n" + call_text() + EOT


def test_explicit_turn_identity_change_is_not_coalesced(renderer):
    source = [message("u", "user", "Begin."), message("a", "assistant", ANNOUNCE, turn_id="turn-1"),
              call(turn_id="turn-2"), observation()]
    messages, audit = canonical_messages(source)
    assert len([m for m in messages if m["role"] == "assistant"]) == 2
    assert selected(renderer, renderer.render(messages, turn_boundary_audit=audit["turn_boundary_audit"])).count(EOT) == 2


def test_reasoning_only_replay_message_is_removed_before_grouping(renderer):
    raw = [{"type": "session_meta", "payload": {"session_id": "reasoning-only-fixture"}},
           {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Inspect."}]}},
           {"type": "response_item", "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "<think>PRIVATE_REASONING_ONLY</think>"}]}},
           {"type": "response_item", "payload": {"type": "function_call", "name": "exec_command", "call_id": "c1", "arguments": '{"cmd":"cat config.txt"}'}},
           {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "c1", "output": OBSERVATION}}]
    messages, audit, _ = rollout_messages(raw)
    assert len([m for m in messages if m["role"] == "assistant"]) == 1
    result = renderer.render(messages, turn_boundary_audit=audit["turn_boundary_audit"])
    assert selected(renderer, result) == call_text() + EOT
    assert "PRIVATE_REASONING_ONLY" not in result["full_rendered_text"]


def test_reasoning_only_replay_without_visible_action_cannot_train_empty_eot(renderer):
    doc = {"session_id": "empty", "steps": [{"source": "user", "message": "Inspect."},
        {"source": "agent", "message": "<think>PRIVATE_ONLY</think>", "extra": {"thinking": "ALSO_PRIVATE"}}]}
    from training.qwen38_no_cot.prepare_data import UnsupportedTrace
    with pytest.raises((UnsupportedTrace, UnsupportedMessage)):
        messages, audit, _ = atif_messages(doc)
        renderer.render(messages, turn_boundary_audit=audit["turn_boundary_audit"])


def test_shift_once_preserves_exact_action_targets(renderer):
    messages, audit = canonical_messages(events())
    result = renderer.render(messages, turn_boundary_audit=audit["turn_boundary_audit"])
    row = training_row(result, group_id="synthetic-group", source="trace")
    shifted = shift_next_token_row(row)
    assert row["labels"] == result["labels"]
    assert shifted["input_ids"] == row["input_ids"]
    assert shifted["labels"] == row["labels"][1:] + [-100]
    assert renderer.tokenizer.decode([v for v in shifted["labels"] if v != -100]) == selected(renderer, result)
    assert result["sequence_audit"]["causal_shift_applied"] is False
    with pytest.raises(ValueError):
        shift_next_token_row({**shifted, "causal_shift_applied": True})


def test_cutoff_inside_merged_tool_call_keeps_exact_prefix_and_no_eot(renderer):
    messages, audit = canonical_messages(events())
    full = renderer.render(messages, turn_boundary_audit=audit["turn_boundary_audit"])
    tool_start = next(i for i,(a,b) in enumerate(full["token_offsets"]) if a >= full["full_rendered_text"].index("<tool_call>"))
    cutoff = tool_start + 4
    cut = Qwen38Renderer(ASSETS, max_tokens=cutoff).render(messages, turn_boundary_audit=audit["turn_boundary_audit"])
    for key in ("input_ids", "labels", "attention_mask", "loss_mask", "token_offsets"):
        assert cut[key] == full[key][:cutoff]
    assert ANNOUNCE in selected(renderer, cut)
    assert EOT not in selected(renderer, cut)
    assert cut["sequence_audit"]["eot_appended"] is False


def test_full_262144_prefix_is_not_retokenized_or_completed(renderer):
    source = [message("u", "user", "Inspect."), message("a", "assistant", ANNOUNCE),
              call(cmd="word " * 270000), observation()]
    messages, audit = canonical_messages(source)
    result = renderer.render(messages, turn_boundary_audit=audit["turn_boundary_audit"])
    full_ids = renderer.tokenizer.encode(result["full_rendered_text"], add_special_tokens=False)
    assert len(full_ids) > 262144 and len(result["input_ids"]) == 262144
    assert result["input_ids"] == full_ids[:262144]
    assert EOT not in selected(renderer, result)
    assert result["sequence_audit"]["dropped_input_tokens"] > 0
    assert result["sequence_audit"]["eot_appended"] is False
    shifted = shift_next_token_row(result)
    assert len(shifted["input_ids"]) == 262144 and shifted["labels"][-1] == -100
    from .data import validate_row
    row = training_row(result, group_id="long-synthetic-group", source="trace")
    assert validate_row(row, renderer.identity) == shifted


def test_same_turn_text_after_tool_is_rejected_not_moved_before_it():
    from .normalize import ContinuationMappingError
    source = [message("u", "user", "Inspect."), call(),
              message("after", "assistant", "I have now issued the call."), observation()]
    before = deepcopy(source)
    with pytest.raises(ContinuationMappingError) as error:
        canonical_messages(source)
    assert error.value.code == "unrepresentable_text_after_tool"
    assert source == before


def test_explicit_final_after_tool_is_preserved_in_original_order(renderer):
    source = [message("u", "user", "Inspect."), call(),
              message("after", "assistant", "The request is submitted.", channel="final"), observation()]
    messages, audit = canonical_messages(source)
    target = selected(renderer, renderer.render(messages, turn_boundary_audit=audit["turn_boundary_audit"]))
    assert target == call_text() + EOT + "The request is submitted." + EOT


def test_removed_reasoning_preserves_known_turn_identity_boundary():
    source = [message("u", "user", "Inspect."), message("a", "assistant", ANNOUNCE, turn_id="T1"),
              {"event_id": "hidden", "kind": "reasoning", "role": "assistant", "content": "REMOVED", "turn_id": "T2"},
              call(), observation()]
    messages, audit = canonical_messages(source)
    assert len([m for m in messages if m["role"] == "assistant"]) == 2


def test_different_message_item_ids_are_not_invented_turn_barriers(renderer):
    source = events()
    source[1]["data"]["block"]["id"] = "item-111"
    source[2]["data"]["block"]["id"] = "item-222"
    source[1]["data"]["block"]["response_id"] = "response-111"
    source[2]["data"]["block"]["response_id"] = "response-222"
    messages, audit = canonical_messages(source)
    assert len([m for m in messages if m["role"] == "assistant"]) == 2
    assert ANNOUNCE + EOT not in selected(renderer, renderer.render(messages, turn_boundary_audit=audit["turn_boundary_audit"]))


def test_same_source_text_blocks_keep_exact_content_without_added_whitespace(renderer):
    first = message("a0", "assistant", "First fragment")
    second = message("a1", "assistant", " continues.")
    first["source"]["pointer"] = "/input/1/content/0"
    second["source"]["pointer"] = "/input/1/content/1"
    source = [message("u", "user", "Inspect."), first, second, call(), observation()]
    messages, audit = canonical_messages(source)
    assert messages[1]["content"] == "First fragment continues."
    assert selected(renderer, renderer.render(messages, turn_boundary_audit=audit["turn_boundary_audit"])) == "First fragment continues.\n\n" + call_text() + EOT


def test_masked_first_generation_retains_receipt_and_exact_position():
    from training.qwen38_cot_experimental.test_render import fixture
    from .masked import Qwen38TurnBoundaryMaskedRenderer
    original, entries, config = fixture(events(), targets=["announce"])
    before = deepcopy((original, entries, config))
    renderer = Qwen38TurnBoundaryMaskedRenderer(ASSETS)
    result = renderer.render_generated_trace(original, entries, config)
    cot = entries[0]["candidate"]["text"]
    target = selected(renderer, result)
    assert target == ANNOUNCE + "\n\n" + call_text() + EOT + FINAL + EOT
    text = result["full_rendered_text"]
    assert text.index(cot) < text.index(ANNOUNCE) < text.index("<tool_call>")
    assert cot not in target and text.count(cot) == 1
    assert result["cot_provenance"]["reviewed"] is False
    assert result["cot_provenance"]["future_information_checked"] is False
    for start, end in result["cot_mask_audit"]["retained_token_ranges"]:
        assert all(v == -100 for v in result["labels"][start:end])
    assert (original, entries, config) == before
    clean, _ = canonical_messages(original["events"])
    clean[1]["reasoning_content"] = cot
    native = renderer.tokenizer.apply_chat_template(clean, tokenize=False, add_generation_prompt=False,
        enable_thinking=True, preserve_thinking=True, reasoning_effort="xhigh")
    assert text == native
    assert result["input_ids"] == renderer.tokenizer.encode(native, add_special_tokens=False)


def test_masked_later_generation_cannot_move_before_earlier_announcement():
    from training.qwen38_cot_experimental.test_render import fixture
    from .masked import Qwen38TurnBoundaryMaskedRenderer, MaskedBoundaryExclusion
    original, entries, config = fixture(events(), targets=["call"])
    before = deepcopy((original, entries, config))
    with pytest.raises(MaskedBoundaryExclusion):
        Qwen38TurnBoundaryMaskedRenderer(ASSETS).render_generated_trace(original, entries, config)
    assert (original, entries, config) == before


def test_masked_replay_reasoning_only_message_never_adds_eot_target():
    from cot_filler.core import digest
    from .masked import Qwen38TurnBoundaryMaskedRenderer
    messages = [{"role": "user", "content": "Inspect."},
                {"role": "assistant", "content": "<think>PRIVATE_ONLY</think>"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "exec_command", "arguments": {"cmd": "cat config.txt"}}}]}]
    renderer = Qwen38TurnBoundaryMaskedRenderer(ASSETS)
    result = renderer.render_replay_messages(messages, source_digest=digest(messages), trace_id="synthetic-replay")
    assert selected(renderer, result) == call_text() + EOT
    assert "PRIVATE_ONLY" not in result["full_rendered_text"]


def test_masked_replay_with_only_removed_reasoning_cannot_train_eot():
    from cot_filler.core import digest
    from .masked import Qwen38TurnBoundaryMaskedRenderer
    messages = [{"role": "user", "content": "Inspect."},
                {"role": "assistant", "content": "<think>PRIVATE_ONLY</think>"}]
    with pytest.raises(UnsupportedMessage):
        Qwen38TurnBoundaryMaskedRenderer(ASSETS).render_replay_messages(
            messages, source_digest=digest(messages), trace_id="empty-reasoning-only")


def test_renderer_requires_hash_bound_normalization_audit(renderer):
    source = events()
    messages, audit = canonical_messages(source)
    with pytest.raises(TypeError):
        renderer.render(messages)
    mutated = deepcopy(messages)
    mutated[1]["content"] += " Changed."
    with pytest.raises(ValueError, match="audit"):
        renderer.render(mutated, turn_boundary_audit=audit["turn_boundary_audit"])


def test_masked_explicit_end_turn_is_not_lost_during_event_mapping():
    from training.qwen38_cot_experimental.test_render import fixture
    from .masked import Qwen38TurnBoundaryMaskedRenderer
    source = events()
    source[1].pop("channel")
    source[1]["data"]["block"]["end_turn"] = True
    renderer = Qwen38TurnBoundaryMaskedRenderer(ASSETS)
    result = renderer.render_generated_trace(*fixture(source, targets=["call"]))
    assert selected(renderer, result).count(EOT) == 3
    assert result["cot_mask_audit"]["filled_gap_count"] == 1
    assert result["cot_provenance"]["reasoning_relocated"] is False


def test_new_loader_validates_actual_native_row_and_shifts_once(renderer):
    from .data import validate_identity, validate_row
    messages, audit = canonical_messages(events())
    result = renderer.render(messages, turn_boundary_audit=audit["turn_boundary_audit"])
    row = training_row(result, group_id="synthetic-group", source="trace")
    validate_identity(renderer.identity)
    loaded = validate_row(row, renderer.identity)
    assert loaded["input_ids"] == result["input_ids"]
    assert loaded["labels"] == result["labels"][1:] + [-100]


@pytest.mark.parametrize("region", ["assistant_header", "think_wrapper", "observation", "eot"])
def test_new_loader_rejects_mismasked_native_regions(renderer, region):
    from .data import validate_row
    messages, audit = canonical_messages(events())
    result = renderer.render(messages, turn_boundary_audit=audit["turn_boundary_audit"])
    row = training_row(result, group_id="synthetic-group", source="trace")
    row = deepcopy(row)
    if region == "eot":
        index = next(i for i,(token,label) in enumerate(zip(row["input_ids"],row["labels"])) if token == renderer.tokenizer.convert_tokens_to_ids(EOT) and label != -100)
        row["labels"][index] = -100
    else:
        needle = {"assistant_header": "<|im_start|>assistant", "think_wrapper": "<think>", "observation": OBSERVATION}[region]
        char_start = result["full_rendered_text"].index(needle)
        index = next(i for i,(start,end) in enumerate(result["token_offsets"]) if start <= char_start < end)
        assert row["labels"][index] == -100
        row["labels"][index] = row["input_ids"][index]
    with pytest.raises(ValueError, match="labels"):
        validate_row(row, renderer.identity)


def test_analysis_channel_inside_canonical_block_never_becomes_visible_target(renderer):
    source = events()
    private = message("hidden", "assistant", "PRIVATE_BLOCK_ANALYSIS_NOT_A_TARGET")
    private["data"]["block"]["channel"] = "analysis"
    source.insert(1, private)
    messages, audit = canonical_messages(source)
    result = renderer.render(messages, turn_boundary_audit=audit["turn_boundary_audit"])
    assert "PRIVATE_BLOCK_ANALYSIS_NOT_A_TARGET" not in result["full_rendered_text"]
    assert selected(renderer, result) == ANNOUNCE + "\n\n" + call_text() + EOT + FINAL + EOT


def test_omitted_tool_definition_retains_explicit_turn_transition(renderer):
    source = events()
    source[1]["turn_id"] = "T1"
    source.insert(2, {"event_id": "capability", "kind": "tool_definition", "role": "system",
                      "turn_id": "T2", "content": "OMITTED_CAPABILITY_SCHEMA"})
    messages, audit = canonical_messages(source)
    assert len([m for m in messages if m["role"] == "assistant"]) == 3
    result = renderer.render(messages, turn_boundary_audit=audit["turn_boundary_audit"])
    assert ANNOUNCE + EOT in selected(renderer, result)
    assert "OMITTED_CAPABILITY_SCHEMA" not in result["full_rendered_text"]


def test_typed_analysis_channel_tool_call_remains_an_action(renderer):
    source = events()
    source[2]["channel"] = "analysis"
    source[2]["data"]["block"]["channel"] = "analysis"
    messages, audit = canonical_messages(source)
    result = renderer.render(messages, turn_boundary_audit=audit["turn_boundary_audit"])
    assert selected(renderer, result) == ANNOUNCE + "\n\n" + call_text() + EOT + FINAL + EOT


def test_raw_rollout_typed_analysis_call_is_not_removed_as_reasoning(renderer):
    raw = [{"type": "session_meta", "payload": {"session_id": "typed-call-channel"}},
           {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Inspect."}]}},
           {"type": "response_item", "payload": {"type": "message", "role": "assistant", "channel": "commentary", "content": [{"type": "output_text", "text": ANNOUNCE}]}},
           {"type": "response_item", "payload": {"type": "function_call", "channel": "analysis", "name": "exec_command", "call_id": "c1", "arguments": '{"cmd":"cat config.txt"}'}},
           {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "c1", "output": OBSERVATION}}]
    messages, audit, _ = rollout_messages(raw)
    result = renderer.render(messages, turn_boundary_audit=audit["turn_boundary_audit"])
    assert selected(renderer, result) == ANNOUNCE + "\n\n" + call_text() + EOT
