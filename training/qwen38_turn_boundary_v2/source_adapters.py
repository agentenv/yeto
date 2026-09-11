"""Versioned source adapters preserving continuation and explicit turn barriers.

The original adapters and every saved dataset remain immutable. Source item IDs
and response IDs are deliberately NOT inferred to be conversation turn IDs.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import re

from training.qwen38_no_cot import prepare_data as original
from . import normalize
from .normalize import coalesce_assistant_continuations

VERSION = "codex-source-turn-normalization/v2"
_LEADING_REASONING = re.compile(r"\A\s*<(think|thinking|analysis|reasoning)\b[^>]*>.*?(?:</\1\s*>|\Z)", re.I | re.S)


def boundary_metadata(item, block=None):
    """Extract only explicit source boundary facts; do not guess from item IDs."""
    block = block if isinstance(block, dict) else {}
    source = item.get("source", {})
    source = source if isinstance(source, dict) else {}
    values = (item, block, source)
    result = {}
    for value in values:
        channel = value.get("channel")
        if channel in ("final", "commentary", "analysis"):
            if "channel" in result and result["channel"] != channel:
                raise original.UnsupportedTrace("conflicting_source_channel")
            result["channel"] = channel
        turn = value.get("source_turn_id", value.get("turn_id"))
        if turn is not None:
            if not isinstance(turn, str) or not turn:
                raise original.UnsupportedTrace("invalid_explicit_source_turn_id")
            if "source_turn_id" in result and result["source_turn_id"] != turn:
                raise original.UnsupportedTrace("conflicting_source_turn_id")
            result["source_turn_id"] = turn
        for key in ("barrier_before", "barrier_after"):
            if value.get(key) is True:
                result[key] = True
        if value.get("end_turn") is True:
            result["barrier_after"] = True
    if result.get("channel") == "final":
        result["barrier_after"] = True
    return result


def _text(value, role, counts):
    text = original.visible_text(value)
    if role == "assistant":
        while (cleaned := _LEADING_REASONING.sub("", text, count=1)) != text:
            text = cleaned
            counts["leading_reasoning_blocks_removed_before_turn_mapping"] += 1
    return text


def _finish(messages, boundaries, counts, *, raw_tool_call_events=None,
            raw_tool_call_content_bytes=0):
    original.validate_messages(messages)
    output, audit = coalesce_assistant_continuations(messages, boundaries=boundaries, reasoning_policy="drop")
    original.validate_messages(output)
    input_call_payloads = [call for message in messages for call in message.get('tool_calls', [])]
    output_call_payloads = [call for message in output for call in message.get('tool_calls', [])]
    input_result_payloads = [{key: message[key] for key in ('tool_call_id', 'content')}
                             for message in messages if message['role'] == 'tool']
    output_result_payloads = [{key: message[key] for key in ('tool_call_id', 'content')}
                              for message in output if message['role'] == 'tool']
    input_calls = [call['id'] for call in input_call_payloads]
    output_calls = [call['id'] for call in output_call_payloads]
    input_results = [result['tool_call_id'] for result in input_result_payloads]
    output_results = [result['tool_call_id'] for result in output_result_payloads]
    omitted_terminal = counts.get('incomplete_terminal_call_omitted', 0)
    if raw_tool_call_events is None:
        raw_tool_call_events = len(input_calls) + omitted_terminal
    tool_audit = {
        'schema': 'qwen38-raw-tool-cardinality-audit/v1',
        'verified': (input_call_payloads == output_call_payloads
                     and input_result_payloads == output_result_payloads),
        'source_tool_calls': len(input_calls), 'native_structured_tool_calls': len(output_calls),
        'raw_tool_call_events': raw_tool_call_events,
        'intentionally_omitted_incomplete_terminal_tool_calls': omitted_terminal,
        'source_tool_results': len(input_results), 'native_tool_results': len(output_results),
        'source_tool_call_ids_sha256': normalize.message_digest(input_calls),
        'native_tool_call_ids_sha256': normalize.message_digest(output_calls),
        'source_tool_result_ids_sha256': normalize.message_digest(input_results),
        'native_tool_result_ids_sha256': normalize.message_digest(output_results),
        'source_tool_call_payloads_sha256': normalize.message_digest(input_call_payloads),
        'native_tool_call_payloads_sha256': normalize.message_digest(output_call_payloads),
        'source_tool_result_payloads_sha256': normalize.message_digest(input_result_payloads),
        'native_tool_result_payloads_sha256': normalize.message_digest(output_result_payloads),
        'raw_tool_call_event_content_bytes': raw_tool_call_content_bytes,
        'raw_tool_call_event_content_projected_bytes': 0,
    }
    tool_audit['verified'] = (tool_audit['verified']
        and raw_tool_call_events == len(input_calls) + omitted_terminal)
    if not tool_audit['verified']:
        raise original.UnsupportedTrace('tool_cardinality_or_result_identity_changed')
    audit['raw_tool_cardinality'] = tool_audit
    return output, {"version": VERSION, "source_normalization_counts": dict(counts), "turn_boundary_audit": audit}


def canonical_messages(events):
    messages, boundaries, counts = [], [], Counter()
    previous_key = None
    raw_tool_call_content_bytes = 0
    raw_tool_call_events = 0
    for event_index, event in enumerate(events):
        kind, role = event.get("kind"), event.get("role")
        data = event.get("data", {})
        block = data.get("block", data) if isinstance(data, dict) else data
        boundary = boundary_metadata(event, block)
        private_channel = event.get("channel") in original.PRIVATE or (isinstance(block, dict) and block.get("channel") in original.PRIVATE)
        if kind in {"reasoning", "empty_message", "capture_metadata"} or (kind == "message" and private_channel):
            counts["reasoning_or_nonvisible_events_removed"] += 1
            if boundary.get("barrier_before") or boundary.get("barrier_after") or boundary.get("source_turn_id"):
                messages.append({"role": "assistant", "content": ""}); boundaries.append(boundary)
                previous_key = None
            continue
        if kind == "tool_definition":
            counts["capability_schemas_omitted"] += 1
            if boundary.get("barrier_before") or boundary.get("barrier_after") or boundary.get("source_turn_id"):
                messages.append({"role": "assistant", "content": ""}); boundaries.append(boundary)
                previous_key = None
            continue
        source = event.get("source", {})
        pointer = source.get("pointer", event.get("event_id")) if isinstance(source, dict) else event.get("event_id")
        if kind == "unknown" and isinstance(block, dict) and block.get("type") == "tool_search_output":
            kind, role = "tool_result", "tool"
            counts["tool_search_output_as_observation"] += 1
        elif (kind == "unknown" and block is None and not event.get("content") and event_index == len(events)-1
              and isinstance(pointer, str) and pointer.startswith("/response/")):
            counts["empty_terminal_capture_event_omitted"] += 1
            continue
        elif kind in {"media", "unknown", "compaction"}:
            raise original.UnsupportedTrace("unsupported_canonical_" + kind)
        if role == "agent":
            role = "assistant"
        if role == "developer":
            role = "system"
            counts["developer_as_chronological_system"] += 1
        key = re.sub(r"/content/\d+$", "", pointer) if isinstance(pointer, str) else None
        if kind == "message":
            if role not in {"system", "user", "assistant"}:
                raise original.UnsupportedTrace("unsupported_message_role")
            if isinstance(block, dict) and block.get("type") not in {None, "message", "agent_message", "text", "input_text", "output_text", "refusal"}:
                raise original.UnsupportedTrace("unsupported_message_block")
            text = _text(event.get("content"), role, counts)
            if (key is not None and previous_key == (key, role) and messages and messages[-1]["role"] == role
                    and "tool_calls" not in messages[-1] and boundaries[-1] == boundary):
                messages[-1]["content"] += text
                counts["same_source_message_blocks_joined"] += 1
            else:
                messages.append({"role": role, "content": text}); boundaries.append(boundary)
            previous_key = (key, role)
        elif kind == "tool_call":
            raw_tool_call_events += 1
            if role != "assistant" or not isinstance(block, dict):
                raise original.UnsupportedTrace("invalid_canonical_tool_call")
            if (event_index == len(events)-1 and isinstance(pointer, str) and pointer.startswith("/response/output/")
                    and block.get("type") == "function_call" and not any(k in block for k in ("arguments", "input", "function"))):
                counts["incomplete_terminal_call_omitted"] += 1
                continue
            call = original.tool_call(block, name=event.get("name"), call_id=event.get("call_id"), counts=counts)
            raw_content = event.get('content')
            if isinstance(raw_content, str):
                raw_tool_call_content_bytes += len(raw_content.encode())
            messages.append({"role": "assistant", "content": "", "tool_calls": [call]}); boundaries.append(boundary)
            previous_key = None
        elif kind == "tool_result":
            if not isinstance(block, dict):
                raise original.UnsupportedTrace("invalid_canonical_tool_result")
            identity = event.get("call_id") or block.get("call_id") or block.get("tool_call_id") or block.get("tool_use_id")
            if not isinstance(identity, str) or not identity:
                raise original.UnsupportedTrace("tool_result_missing_identity")
            messages.append({"role": "tool", "tool_call_id": identity, "content": original.observation(event.get("content"))})
            boundaries.append(boundary); previous_key = None
        else:
            raise original.UnsupportedTrace("unknown_canonical_kind")
    return _finish(messages, boundaries, counts,
                   raw_tool_call_events=raw_tool_call_events,
                   raw_tool_call_content_bytes=raw_tool_call_content_bytes)


def rollout_messages(events):
    messages, boundaries, counts = [], [], Counter()
    session_id = None
    if not any(e.get("type") == "response_item" for e in events):
        raise original.UnsupportedTrace("rollout_without_response_items")
    pending_turn = None
    raw_tool_call_content_bytes = 0
    raw_tool_call_events = 0
    for event in events:
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        if event.get("type") == "turn_context" and payload.get("turn_id"):
            pending_turn = payload["turn_id"]
        if event.get("type") == "session_meta":
            session_id = payload.get("session_id") or payload.get("id") or session_id
            base = payload.get("base_instructions")
            if isinstance(base, dict): base = base.get("text")
            if base:
                messages.append({"role": "system", "content": original.visible_text(base)}); boundaries.append({})
        if event.get("type") != "response_item":
            continue
        boundary = boundary_metadata(event, payload)
        if pending_turn is not None and "source_turn_id" not in boundary:
            boundary["source_turn_id"] = pending_turn
        kind = payload.get("type")
        if kind in original.PRIVATE or (kind == "message" and (payload.get("channel") in original.PRIVATE or event.get("channel") in original.PRIVATE)):
            counts["reasoning_items_removed"] += 1
            if boundary.get("barrier_before") or boundary.get("barrier_after") or boundary.get("source_turn_id"):
                messages.append({"role": "assistant", "content": ""}); boundaries.append(boundary)
            continue
        if kind == "message":
            role = payload.get("role")
            if role == "developer":
                role = "system"; counts["developer_as_chronological_system"] += 1
            if role not in {"system", "user", "assistant"}:
                raise original.UnsupportedTrace("unsupported_rollout_role")
            messages.append({"role": role, "content": _text(payload.get("content"), role, counts)})
        elif kind in {"function_call", "custom_tool_call"}:
            raw_tool_call_events += 1
            raw_content = event.get('content')
            if isinstance(raw_content, str):
                raw_tool_call_content_bytes += len(raw_content.encode())
            messages.append({"role": "assistant", "content": "", "tool_calls": [original.tool_call(payload, counts=counts)]})
        elif kind in {"function_call_output", "custom_tool_call_output"}:
            identity = payload.get("call_id")
            if not isinstance(identity, str) or not identity:
                raise original.UnsupportedTrace("tool_result_missing_identity")
            messages.append({"role": "tool", "tool_call_id": identity, "content": original.observation(payload.get("output"))})
        else:
            raise original.UnsupportedTrace("unsupported_rollout_response_item")
        boundaries.append(boundary)
    if not isinstance(session_id, str) or not session_id:
        raise original.UnsupportedTrace("rollout_missing_session_id")
    output, audit = _finish(messages, boundaries, counts,
                            raw_tool_call_events=raw_tool_call_events,
                            raw_tool_call_content_bytes=raw_tool_call_content_bytes)
    return output, audit, "session:" + session_id


def atif_messages(doc):
    # ATIF already binds each step's text and calls, but reasoning-only text can
    # become empty after stripping. Preserve explicit source boundaries here too.
    clean = deepcopy(doc)
    counts = Counter()
    for step in clean.get("steps", []):
        role = {"agent": "assistant", "developer": "system"}.get(step.get("source"), step.get("source"))
        step["message"] = _text(step.get("message"), role, counts)
    messages, original_counts, group = original.atif_messages(clean)
    raw_tool_call_events = sum(len(step.get("tool_calls")
        or step.get("extra", {}).get("requested_tool_calls") or [])
        for step in clean["steps"])
    boundaries = []
    # Reproduce only the original adapter's inclusion decisions, preserving its
    # authoritative call/result conversion rather than inventing new identities.
    for step in clean["steps"]:
        raw = step.get("tool_calls") or step.get("extra", {}).get("requested_tool_calls") or []
        if step.get("message") or raw:
            boundaries.append(boundary_metadata(step, step.get("extra")))
        for result in (step.get("observation") or {}).get("results", []):
            boundaries.append({})
    if len(boundaries) != len(messages):
        raise original.UnsupportedTrace("atif_boundary_alignment_failed")
    counts.update(original_counts)
    output, audit = _finish(messages, boundaries, counts,
                            raw_tool_call_events=raw_tool_call_events)
    return output, audit, group
