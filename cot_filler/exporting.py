"""Bounded old-converter-compatible messages export; original events retained."""
from __future__ import annotations

import copy
import json


def tool_call(call):
    if not isinstance(call, dict):
        raise ValueError("Tool call must be an object")
    if call.get("type") == "function" and isinstance(call.get("function"), dict):
        if not call.get("id") or not call["function"].get("name") or "arguments" not in call["function"]:
            raise ValueError("Standard function call requires id, name, and arguments")
        return copy.deepcopy(call)
    if call.get("type") == "custom_tool_call":
        raise ValueError("Custom tool calls require an explicit target chat-template adapter")
    identifier = call.get("call_id", call.get("tool_call_id", call.get("id")))
    name = call.get("name", call.get("function_name", call.get("tool_id")))
    arguments = call.get("arguments", call.get("input"))
    if not identifier or not name or arguments is None:
        raise ValueError("Cannot map tool call without an explicit call ID, name, and arguments")
    return {"id": identifier, "type": "function", "function": {"name": name, "arguments": copy.deepcopy(arguments)}}


def as_messages(row):
    messages, ranges, original_events, pending = [], [], [], None
    for event in row["events"]:
        if event["kind"] == "synthetic_rationale":
            if pending is not None:
                raise ValueError("Multiple rationale events before one assistant action")
            pending = event
            continue
        original_events.append(copy.deepcopy(event))
        kind, role = event.get("kind", "message"), event["role"]
        if kind not in {"message", "tool_call", "tool_result"}:
            raise ValueError(f"Messages export needs a template adapter for event kind {kind!r}; canonical events export remains lossless")
        content = event.get("content", "")
        if not isinstance(content, str):
            raise ValueError("Messages export currently supports string content only; use canonical export for structured/multimodal content")
        message = {"role": role, "content": content}
        # Preserve ordinary message fields understood by existing chat loaders.
        for key in ("name", "tool_call_id", "tool_calls"):
            if key in event:
                message[key] = copy.deepcopy(event[key])
        if kind == "tool_call" and not message.get("tool_calls"):
            payload = event.get("data", event.get("payload", {}))
            payload = payload.get("block", payload) if isinstance(payload, dict) else payload
            message["tool_calls"] = [payload]
        if message.get("tool_calls"):
            if role != "assistant":
                raise ValueError("Tool calls must belong to assistant messages")
            message["tool_calls"] = [tool_call(call) for call in message["tool_calls"]]
        if role == "tool" and not message.get("tool_call_id"):
            payload = event.get("data", event.get("payload", {}))
            payload = payload.get("block", payload) if isinstance(payload, dict) else payload
            if isinstance(payload, dict):
                message["tool_call_id"] = payload.get("call_id", payload.get("source_call_id"))
            if not message.get("tool_call_id"):
                raise ValueError("Tool result needs its original tool call ID for messages export")
        index = len(messages)
        if pending is not None:
            if role != "assistant" or pending["before_event_id"] != event["event_id"]:
                raise ValueError("Rationale is not immediately before its associated assistant action")
            prefix = "<cot>\n" + pending["content"] + "\n</cot>\n\n"
            message["content"] = prefix + content
            ranges.append({"message_index": index, "event_id": pending["event_id"], "start_char": 0, "end_char": len(prefix),
                           "loss_intent": "mask" if row["metadata"]["arm"] == "masked" else "train", "segment": "synthetic_rationale_with_delimiters"})
            ranges.append({"message_index": index, "event_id": event["event_id"], "start_char": len(prefix), "end_char": len(prefix) + len(content), "loss_intent": "train", "segment": "original_content"})
            pending = None
        else:
            ranges.append({"message_index": index, "event_id": event["event_id"], "start_char": 0, "end_char": len(content), "loss_intent": "train" if role == "assistant" else "mask", "segment": "original_content"})
        messages.append(message)
    if pending is not None:
        raise ValueError("Rationale has no associated action")
    return {"schema": "cot.reviewed-messages.v1", "messages": messages,
            "metadata": {**copy.deepcopy(row["metadata"]), "trace_id": row["trace_id"], "reasoning_status": "teacher_backfilled" if row["metadata"]["arm"] != "none" else "omitted",
                         "original_events": original_events, "message_content_segments": ranges,
                         "training_ready": False, "required_next_step": "Verify target chat template, <cot> delimiters, tokenized tool fields, and token-aligned masks; character offsets are not token masks"}}
