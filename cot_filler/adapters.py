"""Versioned, lossless visible-event adapters; original input files stay untouched."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from .core import REASONING_FIELDS, SCHEMA, validate_trace

REASONING_KEYS = REASONING_FIELDS


def reject_generated(value):
    """Check source schema provenance before wrapping; do not inspect tool data."""
    if (value.get("synthetic") or value.get("synthetic_reasoning") or value.get("lookahead_conditioned")
            or value.get("kind") == "synthetic_rationale" or value.get("reasoning_status") == "teacher_backfilled"):
        raise ValueError("Generated source rationale cannot be used as original input")


def encrypted(value):
    if isinstance(value, dict):
        return value.get("type") in {"encrypted", "encrypted_reasoning", "redacted_thinking"} or bool(value.get("encrypted_content")) or any(encrypted(v) for v in value.values())
    if isinstance(value, list):
        return any(encrypted(v) for v in value)
    return isinstance(value, str) and any(x in value.lower() for x in ("[encrypted", "<encrypted", "encrypted_reasoning", "encrypted reasoning", "redacted_thinking"))


def clean(value):
    return copy.deepcopy(value)


def visible_message(value, assistant):
    result = clean(value)
    if assistant:
        for key in REASONING_KEYS:
            result.pop(key, None)
        if isinstance(result.get("content"), list):
            result["content"] = [block for block in result["content"] if not (isinstance(block, dict) and block.get("type") in {"reasoning", "thinking", "redacted_thinking"})]
    return result


def reasoning_values(message):
    """Only known assistant-message schema slots, never nested tool arguments."""
    values = [message[k] for k in REASONING_KEYS if message.get(k)]
    if isinstance(message.get("content"), list):
        values += [b for b in message["content"] if isinstance(b, dict) and b.get("type") in {"reasoning", "thinking", "redacted_thinking"}]
    return values


def from_atif(doc, trace_id):
    reject_generated(doc)
    events, targets = [], []
    for i, step in enumerate(doc["steps"]):
        if not isinstance(step, dict):
            raise ValueError("ATIF step must be an object")
        role = {"agent": "assistant"}.get(step.get("source"), step.get("source"))
        if role == "assistant":
            reject_generated(step)
            reject_generated(step.get("extra", {}))
        if role not in {"assistant", "user", "system", "developer", "tool"}:
            raise ValueError(f"Unsupported ATIF step source at {i}")
        sid = str(step.get("step_id", i))
        event_id = f"atif:step:{sid}" if "step_id" in step else f"atif:index:{i}"
        payload = clean({k: v for k, v in step.items() if k != "observation"})
        if isinstance(payload.get("extra"), dict):
            payload["extra"].pop("thinking", None)
        event = {"event_id": event_id, "role": role, "kind": "message", "content": copy.deepcopy(step.get("message", "")), "source_step": payload}
        thinking = step.get("extra", {}).get("thinking")
        if role == "assistant" and thinking and not encrypted(thinking):
            event["original_reasoning_present"] = True
        calls = step.get("tool_calls") or step.get("extra", {}).get("requested_tool_calls")
        if calls:
            event["tool_calls"] = clean(calls)
        events.append(event)
        if role == "assistant" and encrypted(step.get("extra", {}).get("thinking")):
            targets.append({"event_id": event_id, "reason": "encrypted_marker", "source_step_id": sid})
        observation = step.get("observation") or {}
        for j, result in enumerate(observation.get("results", [])):
            events.append({"event_id": f"{event_id}:result:{j}", "role": "tool", "kind": "tool_result", "content": clean(result.get("content", "")), "tool_call_id": result.get("source_call_id"), "result": clean(result), "observation_metadata": clean({k: v for k, v in observation.items() if k != "results"})})
        if observation and not observation.get("results"):
            raise ValueError("ATIF observation without results needs an explicit canonical event adapter")
    return {"schema": SCHEMA, "trace_id": trace_id, "source_format": "atif/v1", "events": events,
            "gap_targets": targets, "metadata": clean({k: v for k, v in doc.items() if k != "steps"})}


def from_session(raw, trace_id):
    events, targets, pending = [], [], []
    pending_plaintext = False
    has_responses = any(e.get("type") == "response_item" for e in raw)

    def append(event):
        nonlocal pending, pending_plaintext
        if pending:
            if event["role"] != "assistant":
                raise ValueError("Encrypted marker is followed by a non-assistant event; resolve its insertion boundary explicitly")
            targets.append({"event_id": event["event_id"], "reason": "encrypted_marker", "source_marker_ids": pending})
            pending = []
        if pending_plaintext:
            if event["role"] != "assistant":
                raise ValueError("Original reasoning boundary is ambiguous; use canonical adapter")
            event["original_reasoning_present"] = True
            pending_plaintext = False
        events.append(event)

    for i, original in enumerate(raw):
        typ = original.get("type")
        p = original.get("payload", {}) if typ in {"response_item", "event_msg"} else original
        if not isinstance(p, dict):
            raise ValueError("Session payload must be an object")
        kind = p.get("type", typ)
        if kind in {"message", "assistant_message", "agent_message", "function_call", "custom_tool_call", "tool_invocation"}:
            reject_generated(original)
            reject_generated(p)
        sid = str(original.get("id", p.get("id", f"session:{i}")))
        if kind in {"reasoning", "thinking", "redacted_thinking"} or p.get("channel") in {"analysis", "reasoning"}:
            if encrypted(p):
                pending.append(sid)
            else:
                pending_plaintext = True
            continue
        if typ == "event_msg" and has_responses:
            continue  # Codex mirrors response messages as UI events.
        if kind in {"message", "user_message", "assistant_message", "agent_message", "system_message"}:
            role = p.get("role") or {"user_message": "user", "assistant_message": "assistant", "agent_message": "assistant", "system_message": "system"}.get(kind)
            if role is None:
                raise ValueError("Message missing role")
            visible = visible_message(p, role == "assistant")
            event = {"event_id": sid, "role": role, "kind": "message", "content": clean(visible.get("content", visible.get("text", visible.get("message", "")))), "payload": visible}
            values = reasoning_values(p) if role == "assistant" else []
            if any(not encrypted(v) for v in values):
                event["original_reasoning_present"] = True
            if any(encrypted(v) for v in values):
                pending.append(sid + ":embedded-marker")
            append(event)
        elif kind in {"function_call", "custom_tool_call"}:
            append({"event_id": sid, "role": "assistant", "kind": "tool_call", "content": "", "tool_calls": [clean(p)]})
        elif kind in {"function_call_output", "custom_tool_call_output"}:
            append({"event_id": sid, "role": "tool", "kind": "tool_result", "content": clean(p.get("output", "")), "tool_call_id": p.get("call_id"), "payload": clean(p)})
        elif kind == "tool_invocation":
            call = {k: clean(v) for k, v in p.items() if k not in {"output", "success"}}
            append({"event_id": sid, "role": "assistant", "kind": "tool_call", "content": "", "tool_calls": [call]})
            append({"event_id": sid + ":result", "role": "tool", "kind": "tool_result", "content": clean(p.get("output", "")), "tool_call_id": p.get("call_id"), "success": p.get("success")})
        elif typ in {"session_meta", "turn_context", "event_msg"}:
            if pending and typ != "turn_context":
                raise ValueError("Marker boundary contains an unknown event; use canonical adapter")
        else:
            raise ValueError(f"Unsupported session event {typ!r}; supply an explicit canonical visible projection")
    if pending:
        raise ValueError("Trailing encrypted marker has no assistant action")
    return {"schema": SCHEMA, "trace_id": trace_id, "source_format": "codex-puffer/v1", "events": events, "gap_targets": targets, "metadata": {}}


def from_messages(doc, trace_id):
    reject_generated(doc)
    events, targets = [], copy.deepcopy(doc.get("gap_targets", []))
    for i, message in enumerate(doc["messages"]):
        if message.get("role") == "assistant":
            reject_generated(message)
        event = visible_message(message, message.get("role") == "assistant")
        event.update(event_id=str(message.get("event_id", f"message:{i}")), kind="message")
        values = reasoning_values(message) if message.get("role") == "assistant" else []
        if any(not encrypted(v) for v in values):
            event["original_reasoning_present"] = True
        if any(encrypted(v) for v in values):
            targets.append({"event_id": event["event_id"], "reason": "encrypted_marker", "source_marker_ids": [event["event_id"] + ":embedded-marker"]})
        events.append(event)
    return {"schema": SCHEMA, "trace_id": trace_id, "source_format": "yeto-messages/v1", "events": events,
            "gap_targets": targets, "metadata": clean(doc.get("metadata", {}))}


def from_resolved_archive(doc, trace_id):
    """Bridge already-resolved archive events; no DB access, no compact projection."""
    if doc.get("resolved") is not True:
        raise ValueError("Archive bridge requires resolved:true after upstream text resolution")
    reject_generated(doc)
    events, targets, markers = [], copy.deepcopy(doc.get("gap_targets", [])), []
    original_reasoning_present = False
    for archived in doc["archive_events"]:
        if not isinstance(archived, dict) or not {"event_id", "role", "kind", "content", "data"} <= archived.keys():
            raise ValueError("Archive event requires event_id, role, kind, content, and data")
        kind, channel = archived["kind"], archived.get("channel")
        if kind == "reasoning" or channel in {"analysis", "reasoning"}:
            if encrypted(archived["data"]):
                markers.append(str(archived["event_id"]))
            else:
                original_reasoning_present = True
            continue
        if kind in {"empty_message", "capture_metadata"}:
            continue
        event = clean(archived)
        event["role"] = {"agent": "assistant"}.get(event["role"], event["role"])
        event["event_id"] = str(event["event_id"])
        reject_generated(event)
        if event["role"] == "assistant" and kind == "message":
            payload = event["data"]
            if not isinstance(payload, dict):
                raise ValueError("Archive assistant message data must be a known object payload")
            wrapped = "block" in payload
            block = payload["block"] if wrapped else payload
            if not isinstance(block, (dict, str)):
                raise ValueError("Archive message block must be a string or known object")
            if isinstance(block, dict):
                reject_generated(block)
            fields = payload.get("message_fields", {}) if wrapped else {}
            if not isinstance(fields, dict):
                raise ValueError("Archive message_fields must be an object")
            reject_generated(fields)
            values = (reasoning_values(block) if isinstance(block, dict) else []) + (reasoning_values(payload) if wrapped else []) + reasoning_values(fields)
            if any(encrypted(v) for v in values):
                markers.append(event["event_id"] + ":embedded-marker")
            if any(not encrypted(v) for v in values):
                original_reasoning_present = True
            visible = visible_message(block, True) if isinstance(block, dict) else block
            if wrapped:
                event["data"] = visible_message(payload, True)
                event["data"]["block"] = visible
                if "message_fields" in payload:
                    event["data"]["message_fields"] = visible_message(fields, True)
            else:
                event["data"] = visible
            # Re-render known text blocks after removing assistant reasoning blocks.
            if isinstance(visible, str):
                event["content"] = visible
            elif isinstance(visible.get("content"), list):
                texts = []
                for item in visible["content"]:
                    if not isinstance(item, dict) or item.get("type") not in {"text", "input_text", "output_text"} or not isinstance(item.get("text"), str):
                        raise ValueError("Archive multimodal/unknown message block needs an explicit visible adapter")
                    texts.append(item["text"])
                event["content"] = "\n".join(texts)
        if markers or original_reasoning_present:
            if event["role"] != "assistant":
                raise ValueError("Archive reasoning marker must immediately precede an assistant action")
            if markers:
                targets.append({"event_id": event["event_id"], "reason": "encrypted_marker", "source_marker_ids": markers})
            if original_reasoning_present:
                event["original_reasoning_present"] = True
            markers, original_reasoning_present = [], False
        events.append(event)
    if markers or original_reasoning_present:
        raise ValueError("Trailing archive reasoning slot has no visible assistant action")
    return {"schema": SCHEMA, "trace_id": trace_id, "source_format": "resolved-trace-archive/v1", "events": events,
            "gap_targets": targets, "metadata": clean(doc.get("metadata", {}))}


def load_traces(path, source_format="auto"):
    path = Path(path)
    raw = path.read_bytes()
    file_sha = hashlib.sha256(raw).hexdigest()
    if path.suffix == ".jsonl":
        documents = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        if source_format == "session" or (source_format == "auto" and documents and "type" in documents[0]):
            documents = [{"events": documents}]
    else:
        doc = json.loads(raw)
        documents = doc if isinstance(doc, list) else [doc]
    for index, doc in enumerate(documents):
        trace_id = str(doc.get("trace_id", doc.get("session_id", doc.get("capture_id", f"{file_sha}:{index}"))))
        if doc.get("schema") == SCHEMA:
            trace = copy.deepcopy(doc)
        elif "archive_events" in doc:
            trace = from_resolved_archive(doc, trace_id)
        elif "steps" in doc:
            trace = from_atif(doc, trace_id)
        elif "messages" in doc:
            trace = from_messages(doc, trace_id)
        elif "events" in doc or "session" in doc:
            trace = from_session(doc.get("events", doc.get("session", {}).get("events", [])), trace_id)
        else:
            raise ValueError("Unsupported input. Use cot.trace.v1, ATIF, clean messages, or supported session events")
        validate_trace(trace)
        yield trace, {"path": str(path.resolve()), "file_sha256": file_sha, "row": index, "adapter_version": "1"}
