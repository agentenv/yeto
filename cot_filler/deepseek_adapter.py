"""Strict canonical-event adapter for the DeepSeek V4 native renderer."""
from __future__ import annotations

import copy
import json


def _arguments(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _call(value):
    block = value.get("function", value) if isinstance(value, dict) else None
    if not isinstance(block, dict):
        raise ValueError("Tool call must be an object")
    name = block.get("name")
    args = block.get("arguments", block.get("input"))
    identifier = value.get("id", value.get("call_id")) if isinstance(value, dict) else None
    if not identifier or not name or args is None:
        raise ValueError("DeepSeek tool calls require id, name, and arguments")
    return {"id": str(identifier), "type": "function", "function": {"name": str(name), "arguments": _arguments(args)}}


def _definitions(event):
    payload = event.get("data", event.get("payload", event.get("tool")))
    if isinstance(payload, dict) and "block" in payload:
        payload = payload["block"]
    if isinstance(payload, dict) and payload.get("type") == "additional_tools":
        tools = payload.get("tools") or []
    elif isinstance(payload, dict):
        tools = [payload]
    else:
        raise ValueError("tool_definition payload must be an object")
    result = []
    for tool in tools:
        if isinstance(tool, dict) and tool.get("type") not in {None, "function"}:
            raise ValueError(f"DeepSeek cannot map tool-definition type {tool.get('type')!r} without an explicit adapter")
        fn = tool.get("function", tool) if isinstance(tool, dict) else None
        if not isinstance(fn, dict) or not fn.get("name") or not isinstance(fn.get("parameters"), dict):
            raise ValueError("tool_definition needs function name and parameters schema")
        result.append({"type": "function", "function": fn})
    if not result:
        raise ValueError("tool_definition contains no function schemas")
    return result


def events_to_messages(events):
    """Map lossless canonical events to the OpenAI shape accepted by V4.

    Tool definitions must already be complete OpenAI function schemas. Ambiguous
    Codex metadata is rejected rather than silently omitted.
    """
    messages, pending_tools = [], []
    for event in events:
        kind, role = event.get("kind", "message"), event.get("role")
        if kind == "synthetic_rationale":
            raise ValueError("Render synthetic rationale after mapping, immediately before its target assistant event")
        if kind == "tool_definition":
            schema = event.get("tool", event.get("data", event.get("payload")))
            if not isinstance(schema, dict) or not schema.get("name") or not isinstance(schema.get("parameters"), dict):
                raise ValueError("tool_definition needs an explicit function name and parameters schema")
            pending_tools.append({"type": "function", "function": schema})
            continue
        if kind == "message":
            if role not in {"system", "developer", "user", "assistant"}:
                raise ValueError(f"Unsupported message role {role!r}")
            message = {"role": role, "content": event.get("content", "")}
            if pending_tools:
                if role not in {"user", "developer"}:
                    raise ValueError("Tool definitions must precede a user/developer message")
                message["tools"] = pending_tools
                pending_tools = []
            if "tool_calls" in event:
                message["tool_calls"] = [_call(c) for c in event["tool_calls"]]
            messages.append(message)
        elif kind == "tool_call":
            calls = event.get("tool_calls") or [event.get("data", event.get("payload", {}))]
            messages.append({"role": "assistant", "content": event.get("content", ""), "tool_calls": [_call(c.get("block", c) if isinstance(c, dict) else c) for c in calls]})
        elif kind == "tool_result":
            call_id = event.get("tool_call_id") or event.get("data", {}).get("call_id") if isinstance(event.get("data"), dict) else event.get("tool_call_id")
            if not call_id:
                raise ValueError("Tool result requires tool_call_id")
            messages.append({"role": "tool", "tool_call_id": str(call_id), "content": event.get("content", "")})
        else:
            raise ValueError(f"DeepSeek adapter cannot map event kind {kind!r}")
    if pending_tools:
        raise ValueError("Trailing tool definitions have no following user/developer message")
    return messages


def render_events(events, encoding_module, *, thinking_mode="thinking", drop_thinking=True):
    messages = events_to_messages(copy.deepcopy(events))
    return encoding_module.encode_messages(messages, thinking_mode=thinking_mode, drop_thinking=drop_thinking)


def render_events_with_segments(events, encoding_module, *, thinking_mode="thinking", drop_thinking=True):
    """Render canonical events and return conservative character loss spans.

    Text and DeepSeek DSML tool spans are located in order in the exact native-rendered string.
    If the encoder transforms a field unexpectedly, the example is rejected.
    """
    source = copy.deepcopy(events)
    messages, intents, pending, pending_tools = [], [], None, []
    for event in source:
        kind, role = event.get("kind", "message"), event.get("role")
        if kind == "synthetic_rationale":
            pending = event
            continue
        if kind == "tool_definition":
            pending_tools.extend(_definitions(event))
            continue
        if kind == "message":
            if role not in {"system", "developer", "user", "assistant"}:
                raise ValueError(f"Unsupported message role {role!r}")
            content = event.get("content", "")
            rationale = None
            if pending:
                if role != "assistant" or pending.get("before_event_id") != event.get("event_id"):
                    raise ValueError("Rationale is not immediately before its assistant action")
                rationale = "<cot>\n" + pending.get("content", "") + "\n</cot>\n\n"
                content = rationale + content
                pending = None
            message = {"role": role, "content": content}
            if pending_tools:
                if role not in {"user", "developer"}:
                    raise ValueError("Tool definitions must precede a user/developer message")
                message["tools"] = pending_tools
                pending_tools = []
            messages.append(message)
            intents.append((event.get("event_id"), role, rationale, event.get("content", ""), event))
        elif kind in {"tool_call", "tool_result"}:
            messages.extend(events_to_messages([event]))
            intents.append((event.get("event_id"), role, None, event.get("content", ""), event))
        else:
            raise ValueError(f"DeepSeek adapter cannot map event kind {kind!r}")
    if pending:
        raise ValueError("Trailing rationale has no associated action")
    if pending_tools:
        raise ValueError("Trailing tool definitions have no following user/developer message")
    rendered = encoding_module.encode_messages(messages, thinking_mode=thinking_mode, drop_thinking=drop_thinking)
    spans, cursor = [], 0
    for event_id, role, rationale, content, event in intents:
        if rationale:
            start = rendered.find(rationale, cursor)
            if start < 0:
                raise ValueError("Rationale delimiter/content was transformed by the target renderer")
            spans.append({"event_id": "synthetic:" + str(event_id), "start_char": start,
                          "end_char": start + len(rationale), "loss_intent": "mask", "segment": "synthetic_rationale"})
            cursor = start + len(rationale)
        if content and event.get("kind") != "tool_result":
            start = rendered.find(content, cursor)
            if start < 0:
                raise ValueError("Message content was transformed by the target renderer")
            spans.append({"event_id": event_id, "start_char": start, "end_char": start + len(content),
                          "loss_intent": "train" if role == "assistant" else "mask", "segment": "original_content"})
            cursor = start + len(content)
        if event.get("kind") == "tool_call":
            calls = event.get("tool_calls") or [event.get("data", event.get("payload", {}))]
            blocks = []
            for raw in calls:
                call = _call(raw.get("block", raw) if isinstance(raw, dict) else raw)
                fn = call["function"]
                blocks.append(encoding_module.tool_call_template.format(dsml_token=encoding_module.dsml_token,
                    name=fn["name"], arguments=encoding_module.encode_arguments_to_dsml({"name": fn["name"], "arguments": fn["arguments"]})))
            expected = "\n\n" + encoding_module.tool_calls_template.format(dsml_token=encoding_module.dsml_token,
                tool_calls="\n".join(blocks), tc_block_name=encoding_module.tool_calls_block_name)
            start = rendered.find(expected, cursor)
            if start < 0:
                raise ValueError("DeepSeek DSML tool-call block was transformed by the target renderer")
            spans.append({"event_id": event_id, "start_char": start, "end_char": start + len(expected), "loss_intent": "train", "segment": "tool_call_dsml"})
            cursor = start + len(expected)
        elif event.get("kind") == "tool_result":
            expected = encoding_module.tool_output_template.format(content=content or "")
            start = rendered.find(expected, cursor)
            if start < 0:
                raise ValueError("DeepSeek tool-result block was transformed by the target renderer")
            spans.append({"event_id": event_id, "start_char": start, "end_char": start + len(expected), "loss_intent": "mask", "segment": "tool_result_dsml"})
            cursor = start + len(expected)
    return rendered, spans
