"""Request/response translation for the three gateway entries (task 4.1).

Every entry is reduced to one canonical chat form::

    {"role": "system"|"user"|"assistant"|"tool", "content": str,
     "reasoning_content"?: str, "tool_calls"?: [{"id","type":"function","function":{"name","arguments"}}],
     "tool_call_id"?: str}

and the backend chat completion (one choice, ``message`` with optional
``reasoning_content``/``tool_calls``) is rendered back into the entry's wire
shape.  Unsupported shapes raise ``UnsupportedShape`` (fail closed, never
silently ignored).  Tokenization never happens here (TITO owns tokens).
"""
from __future__ import annotations

import json
from typing import Any


class UnsupportedShape(ValueError):
    pass


def _text(content: Any, kinds: tuple[str, ...]) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") not in kinds:
                raise UnsupportedShape(f"unsupported content block {block!r}")
            parts.append(str(block.get("text", "")))
        return "".join(parts)
    raise UnsupportedShape(f"unsupported content {type(content).__name__}")


def _assistant(content: str = "", reasoning: str | None = None, tool_calls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _tool_call(call_id: str, name: str, arguments: Any) -> dict[str, Any]:
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


# ------------------------------------------------------------------ chat/completions

def chat_to_messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise UnsupportedShape("chat request needs messages")
    out = []
    for m in messages:
        if not isinstance(m, dict) or m.get("role") not in {"system", "user", "assistant", "tool"}:
            raise UnsupportedShape(f"unsupported chat message {m!r}")
        canon = {"role": m["role"], "content": _text(m.get("content") or "", ("text",))}
        for key in ("reasoning_content", "tool_calls", "tool_call_id"):
            if key in m:
                canon[key] = m[key]
        out.append(canon)
    return out


def completion_to_chat(completion: dict[str, Any]) -> dict[str, Any]:
    return completion


# ------------------------------------------------------------------ /v1/responses (Codex)

def responses_to_messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    if body.get("store") is not False:
        raise UnsupportedShape("Responses requests must use store:false (full replay)")
    if body.get("parallel_tool_calls") not in (False, None):
        raise UnsupportedShape("parallel_tool_calls must be false")
    summary = (body.get("reasoning") or {}).get("summary", "none")
    if summary not in ("none", None):
        raise UnsupportedShape("reasoning.summary must be none")
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise UnsupportedShape(f"unsupported tool type {tool.get('type') if isinstance(tool, dict) else tool!r}")
    messages: list[dict[str, Any]] = []
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions:
        messages.append({"role": "system", "content": instructions})
    pending_reasoning: str | None = None
    pending_assistant: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal pending_assistant, pending_reasoning
        if pending_assistant is not None:
            messages.append(pending_assistant)
        pending_assistant, pending_reasoning = None, None

    items = body.get("input")
    if isinstance(items, str):
        items = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": items}]}]
    if not isinstance(items, list) or not items:
        raise UnsupportedShape("Responses input must be a non-empty list")
    for item in items:
        kind = item.get("type", "message") if isinstance(item, dict) else None
        if kind == "message":
            role = item.get("role")
            if role in ("user", "system", "developer"):
                flush()
                messages.append({"role": "system" if role != "user" else "user", "content": _text(item.get("content"), ("input_text", "text"))})
            elif role == "assistant":
                if pending_assistant is None:
                    pending_assistant = _assistant(reasoning=pending_reasoning)
                pending_assistant["content"] += _text(item.get("content"), ("output_text", "text"))
            else:
                raise UnsupportedShape(f"unsupported message role {role!r}")
        elif kind == "reasoning":
            flush()
            pending_reasoning = "".join(str(c.get("text", "")) for c in item.get("content") or []) or item.get("encrypted_content") or ""
        elif kind == "function_call":
            if pending_assistant is None:
                pending_assistant = _assistant(reasoning=pending_reasoning)
            pending_assistant.setdefault("tool_calls", []).append(_tool_call(str(item["call_id"]), str(item["name"]), item.get("arguments", "")))
        elif kind == "function_call_output":
            flush()
            output = item.get("output")
            messages.append({"role": "tool", "tool_call_id": str(item["call_id"]), "content": output if isinstance(output, str) else json.dumps(output)})
        elif kind in {"compaction", "compaction_trigger", "context_compaction"}:
            raise UnsupportedShape("harness-side compaction is forbidden")
        else:
            raise UnsupportedShape(f"unsupported Responses item {kind!r}")
    flush()
    return messages


def completion_to_responses(completion: dict[str, Any], *, response_id: str) -> dict[str, Any]:
    message = completion["choices"][0]["message"]
    output: list[dict[str, Any]] = []
    if message.get("reasoning_content"):
        output.append({"type": "reasoning", "id": f"rs_{response_id}", "summary": [], "content": [{"type": "reasoning_text", "text": message["reasoning_content"]}]})
    if message.get("content"):
        output.append({"type": "message", "id": f"msg_{response_id}", "role": "assistant", "status": "completed",
                       "content": [{"type": "output_text", "text": message["content"], "annotations": []}]})
    for call in message.get("tool_calls") or []:
        output.append({"type": "function_call", "id": f"fc_{call['id']}", "call_id": call["id"], "status": "completed",
                       "name": call["function"]["name"], "arguments": call["function"]["arguments"]})
    usage = completion.get("usage") or {}
    return {"id": f"resp_{response_id}", "object": "response", "status": "completed", "output": output,
            "usage": {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0),
                      "total_tokens": usage.get("total_tokens", 0)}}


# ------------------------------------------------------------------ /v1/messages (Anthropic)

def messages_to_messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    system = body.get("system")
    if system:
        out.append({"role": "system", "content": _text(system, ("text",))})
    items = body.get("messages")
    if not isinstance(items, list) or not items:
        raise UnsupportedShape("Messages request needs messages")
    for m in items:
        role, content = m.get("role"), m.get("content")
        if role == "user":
            if isinstance(content, str):
                out.append({"role": "user", "content": content})
                continue
            text_parts: list[str] = []
            for block in content or []:
                if block.get("type") == "text":
                    text_parts.append(block["text"])
                elif block.get("type") == "tool_result":
                    if text_parts:
                        out.append({"role": "user", "content": "".join(text_parts)}); text_parts = []
                    out.append({"role": "tool", "tool_call_id": str(block["tool_use_id"]), "content": _text(block.get("content") or "", ("text",))})
                else:
                    raise UnsupportedShape(f"unsupported user block {block.get('type')!r}")
            if text_parts:
                out.append({"role": "user", "content": "".join(text_parts)})
        elif role == "assistant":
            text, reasoning, calls = "", None, []
            for block in content if isinstance(content, list) else [{"type": "text", "text": content}]:
                kind = block.get("type")
                if kind == "text":
                    text += block["text"]
                elif kind == "thinking":
                    reasoning = (reasoning or "") + block.get("thinking", "")
                elif kind == "tool_use":
                    calls.append(_tool_call(str(block["id"]), str(block["name"]), block.get("input", {})))
                else:
                    raise UnsupportedShape(f"unsupported assistant block {kind!r}")
            out.append(_assistant(text, reasoning, calls))
        else:
            raise UnsupportedShape(f"unsupported Messages role {role!r}")
    return out


def completion_to_messages(completion: dict[str, Any], *, response_id: str, model: str) -> dict[str, Any]:
    message = completion["choices"][0]["message"]
    content: list[dict[str, Any]] = []
    if message.get("reasoning_content"):
        content.append({"type": "thinking", "thinking": message["reasoning_content"]})
    if message.get("content"):
        content.append({"type": "text", "text": message["content"]})
    for call in message.get("tool_calls") or []:
        try:
            arguments = json.loads(call["function"]["arguments"] or "{}")
        except ValueError as exc:
            raise UnsupportedShape("tool call arguments are not JSON") from exc
        content.append({"type": "tool_use", "id": call["id"], "name": call["function"]["name"], "input": arguments})
    finish = completion["choices"][0].get("finish_reason")
    stop = "tool_use" if message.get("tool_calls") else ("max_tokens" if finish == "length" else "end_turn")
    usage = completion.get("usage") or {}
    return {"id": f"msg_{response_id}", "type": "message", "role": "assistant", "model": model, "content": content,
            "stop_reason": stop, "usage": {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)}}
