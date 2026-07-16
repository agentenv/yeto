"""Convert Codex/Puffer trace artifacts into Yeto chat JSONL rows.

The preferred source is an ATIF ``trajectory.json`` artifact. It already
normalizes streamed deltas, tool calls, observations, retries, and final
assistant messages into ordered ``steps``. Standalone session transcript JSON
and JSONL files are also accepted as a fallback.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable


CHAT_ROLES = {"system", "user", "assistant", "tool"}
REASONING_POLICIES = ("skip", "teacher-backfill")
ENCRYPTED_REASONING_KEYS = {
    "ciphertext",
    "cipher_text",
    "encrypted_content",
    "encrypted_reasoning",
    "encrypted_thinking",
    "jwe",
}
ENCRYPTED_REASONING_MARKERS = (
    "[encrypted]",
    "<encrypted",
    "encrypted:",
    "enc:",
    "-----begin pgp message-----",
)
JWE_RE = re.compile(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$")
OPAQUE_BASE64_RE = re.compile(r"^[A-Za-z0-9+/_=-]+$")

TeacherBackfillFn = Callable[[list[dict[str, Any]], str], str | None]


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _truncate(text: str, limit: int | None) -> str:
    if limit is None or limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def _is_opaque_base64(text: str) -> bool:
    compact = text.strip()
    if len(compact) < 120 or any(ch.isspace() for ch in compact):
        return False
    if not OPAQUE_BASE64_RE.fullmatch(compact):
        return False
    padded = compact + ("=" * (-len(compact) % 4))
    try:
        base64.b64decode(padded, validate=True)
    except Exception:
        try:
            base64.urlsafe_b64decode(padded)
        except Exception:
            return False
    return True


def _looks_encrypted_reasoning(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        lowered = {str(k).lower() for k in value}
        if lowered & ENCRYPTED_REASONING_KEYS:
            return True
        for key in ("encrypted", "is_encrypted"):
            if value.get(key) is True:
                return True
        marker = value.get("type") or value.get("status") or value.get("format")
        if isinstance(marker, str) and marker.lower() in {"encrypted", "ciphertext", "jwe"}:
            return True
        return any(_looks_encrypted_reasoning(v) for v in value.values())
    if isinstance(value, list):
        return any(_looks_encrypted_reasoning(item) for item in value)
    if not isinstance(value, str):
        return False

    text = value.strip()
    lowered = text.lower()
    if lowered.startswith(ENCRYPTED_REASONING_MARKERS):
        return True
    if JWE_RE.fullmatch(text):
        return True
    if _is_opaque_base64(text):
        return True
    if text.startswith("{"):
        try:
            return _looks_encrypted_reasoning(json.loads(text))
        except json.JSONDecodeError:
            return False
    return False


def _tool_call(call: dict[str, Any], index: int) -> dict[str, Any]:
    name = call.get("function_name") or call.get("name") or call.get("tool_id") or "tool"
    args = call.get("arguments", call.get("input", {}))
    if not isinstance(args, str):
        args = json.dumps(args, ensure_ascii=False, sort_keys=True)
    return {
        "id": call.get("tool_call_id") or call.get("id") or f"call_{index}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": args,
        },
    }


def _codex_output_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return _text(content)
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            parts.append(str(item))
            continue
        if item.get("type") in {"output_text", "input_text", "text"}:
            parts.append(_text(item.get("text")))
    return "\n".join(part for part in parts if part)


def _visible_context(
    messages: list[dict[str, Any]],
    assistant_content: str,
    *,
    max_chars: int = 12000,
) -> str:
    parts: list[str] = []
    for msg in messages[-20:]:
        role = msg.get("role", "unknown")
        content = _text(msg.get("content"))
        if content:
            parts.append(f"{role}: {content}")
    if assistant_content.strip():
        parts.append(f"assistant: {assistant_content.strip()}")
    context = "\n\n".join(parts)
    if len(context) <= max_chars:
        return context
    return context[-max_chars:]


def _extract_response_text(payload: dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    output = payload.get("output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text")
                        if isinstance(text, str):
                            parts.append(text)
        if parts:
            return "\n".join(parts).strip()
    return ""


def _openai_teacher_backfill(
    model: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    max_output_tokens: int = 160,
) -> TeacherBackfillFn:
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "teacher-backfill requires OPENAI_API_KEY or --teacher-api-key"
        )
    root = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    url = f"{root}/responses"

    def backfill(messages: list[dict[str, Any]], assistant_content: str) -> str | None:
        context = _visible_context(messages, assistant_content)
        body = {
            "model": model,
            "instructions": (
                "You are backfilling missing encrypted reasoning for an SFT dataset. "
                "Use only the visible transcript below. Do not invent hidden facts, "
                "do not mention encryption, and write a concise rationale in 1-3 sentences."
            ),
            "input": (
                "Visible transcript:\n"
                f"{context}\n\n"
                "Backfill the likely high-level reasoning that connects the context "
                "to the assistant response."
            ),
            "max_output_tokens": max_output_tokens,
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"teacher backfill request failed: {detail}") from error
        return _extract_response_text(payload) or None

    return backfill


def _maybe_backfill_reasoning(
    *,
    messages: list[dict[str, Any]],
    assistant_content: str,
    reasoning_policy: str,
    teacher_backfill_fn: TeacherBackfillFn | None,
) -> str:
    if reasoning_policy == "skip":
        return ""
    if reasoning_policy != "teacher-backfill":
        raise ValueError(f"reasoning_policy must be one of {REASONING_POLICIES}")
    if teacher_backfill_fn is None:
        raise RuntimeError("teacher-backfill requires a teacher_backfill_fn")
    visible_messages = [dict(msg) for msg in messages]
    return (teacher_backfill_fn(visible_messages, assistant_content) or "").strip()


def _append_message(messages: list[dict[str, Any]], role: str, content: str, **extra: Any) -> None:
    content = content.strip()
    if not content and not extra:
        return
    msg = {"role": role if role in CHAT_ROLES else "system", "content": content}
    msg.update({k: v for k, v in extra.items() if v is not None})
    messages.append(msg)


def row_from_atif(
    doc: dict[str, Any],
    *,
    include_failures: bool = False,
    include_thinking: bool = False,
    max_tool_output_chars: int | None = 4000,
    reasoning_policy: str = "skip",
    teacher_backfill_fn: TeacherBackfillFn | None = None,
) -> dict[str, Any] | None:
    """Build one Yeto row from an ATIF trajectory document."""
    success = doc.get("extra", {}).get("success")
    if success is False and not include_failures:
        return None

    steps = doc.get("steps")
    if not isinstance(steps, list):
        return None

    messages: list[dict[str, Any]] = []
    skipped_encrypted_reasoning = 0
    teacher_backfilled_reasoning = 0
    for step in steps:
        if not isinstance(step, dict):
            continue
        source = step.get("source")
        content = _text(step.get("message"))
        if include_thinking:
            thinking_value = step.get("extra", {}).get("thinking")
            if _looks_encrypted_reasoning(thinking_value):
                skipped_encrypted_reasoning += 1
                thinking = _maybe_backfill_reasoning(
                    messages=messages,
                    assistant_content=content,
                    reasoning_policy=reasoning_policy,
                    teacher_backfill_fn=teacher_backfill_fn,
                )
                if thinking:
                    teacher_backfilled_reasoning += 1
            else:
                thinking = _text(thinking_value)
            if thinking:
                content = f"{content}\n\n[thinking]\n{thinking}".strip()

        if source == "user":
            _append_message(messages, "user", content)
            continue
        if source not in {"agent", "assistant"}:
            continue

        raw_calls = step.get("tool_calls") or step.get("extra", {}).get("requested_tool_calls") or []
        tool_calls = [
            _tool_call(call, i)
            for i, call in enumerate(raw_calls)
            if isinstance(call, dict)
        ]
        _append_message(
            messages,
            "assistant",
            content,
            tool_calls=tool_calls or None,
        )

        observation = step.get("observation")
        results = observation.get("results", []) if isinstance(observation, dict) else []
        for i, result in enumerate(results):
            if not isinstance(result, dict):
                continue
            output = _truncate(_text(result.get("content")), max_tool_output_chars)
            call_id = result.get("source_call_id") or (tool_calls[i]["id"] if i < len(tool_calls) else None)
            _append_message(messages, "tool", output, tool_call_id=call_id)

    if not _has_trainable_pair(messages):
        return None
    row: dict[str, Any] = {"messages": messages}
    agent = doc.get("agent", {})
    if isinstance(agent, dict):
        metadata = {
            "source": "atif",
            "session_id": doc.get("session_id"),
            "model": agent.get("model_name"),
            "success": success,
        }
        if skipped_encrypted_reasoning:
            metadata["reasoning_status"] = "encrypted_skipped"
            metadata["encrypted_reasoning_skipped"] = skipped_encrypted_reasoning
        if teacher_backfilled_reasoning:
            metadata["reasoning_status"] = "teacher_backfilled"
            metadata["teacher_backfilled_reasoning"] = teacher_backfilled_reasoning
            metadata["synthetic_reasoning"] = True
        row["metadata"] = metadata
    return row


def row_from_session_events(
    events: Iterable[dict[str, Any]],
    *,
    max_tool_output_chars: int | None = 4000,
    reasoning_policy: str = "skip",
    teacher_backfill_fn: TeacherBackfillFn | None = None,
) -> dict[str, Any] | None:
    """Build one Yeto row from Puffer session transcript events."""
    events = list(events)
    has_codex_event_messages = any(
        isinstance(event, dict)
        and event.get("type") == "event_msg"
        and isinstance(event.get("payload"), dict)
        and event["payload"].get("type") in {"user_message", "agent_message"}
        for event in events
    )
    messages: list[dict[str, Any]] = []
    skipped_encrypted_reasoning = 0
    pending_encrypted_reasoning = 0
    teacher_backfilled_reasoning = 0
    for event in events:
        if not isinstance(event, dict):
            continue
        typ = event.get("type")
        if typ == "user_message":
            _append_message(messages, "user", _text(event.get("text")))
        elif typ == "assistant_message":
            content = _text(event.get("text"))
            if pending_encrypted_reasoning:
                thinking = _maybe_backfill_reasoning(
                    messages=messages,
                    assistant_content=content,
                    reasoning_policy=reasoning_policy,
                    teacher_backfill_fn=teacher_backfill_fn,
                )
                if thinking:
                    content = f"{content}\n\n[thinking]\n{thinking}".strip()
                    teacher_backfilled_reasoning += 1
                pending_encrypted_reasoning = 0
            _append_message(messages, "assistant", content)
        elif typ == "system_message":
            # System messages include UI/status noise. Keep them out of the
            # first-pass SFT data unless they are promoted upstream.
            continue
        elif typ == "tool_invocation":
            tool_call = _tool_call(
                {
                    "tool_call_id": event.get("call_id"),
                    "function_name": event.get("tool_id"),
                    "input": event.get("input"),
                },
                0,
            )
            _append_message(messages, "assistant", "", tool_calls=[tool_call])
            status = "ok" if event.get("success") else "error"
            content = (
                f"Tool {event.get('tool_id', 'tool')} [{status}]\n"
                f"input: {_text(event.get('input'))}\n"
                f"{_truncate(_text(event.get('output')), max_tool_output_chars)}"
            ).strip()
            _append_message(messages, "tool", content, tool_call_id=event.get("call_id"))
        elif typ == "event_msg" and isinstance(event.get("payload"), dict):
            payload = event["payload"]
            payload_type = payload.get("type")
            if payload_type == "user_message":
                _append_message(messages, "user", _text(payload.get("message")))
            elif payload_type == "agent_message":
                content = _text(payload.get("message"))
                if pending_encrypted_reasoning:
                    thinking = _maybe_backfill_reasoning(
                        messages=messages,
                        assistant_content=content,
                        reasoning_policy=reasoning_policy,
                        teacher_backfill_fn=teacher_backfill_fn,
                    )
                    if thinking:
                        content = f"{content}\n\n[thinking]\n{thinking}".strip()
                        teacher_backfilled_reasoning += 1
                    pending_encrypted_reasoning = 0
                _append_message(messages, "assistant", content)
        elif typ == "response_item" and isinstance(event.get("payload"), dict):
            payload = event["payload"]
            payload_type = payload.get("type")
            if payload_type == "reasoning" and _looks_encrypted_reasoning(payload):
                skipped_encrypted_reasoning += 1
                pending_encrypted_reasoning += 1
            elif payload_type == "message" and not has_codex_event_messages:
                role = payload.get("role")
                content = _codex_output_text(payload.get("content"))
                if role == "assistant" and pending_encrypted_reasoning:
                    thinking = _maybe_backfill_reasoning(
                        messages=messages,
                        assistant_content=content,
                        reasoning_policy=reasoning_policy,
                        teacher_backfill_fn=teacher_backfill_fn,
                    )
                    if thinking:
                        content = f"{content}\n\n[thinking]\n{thinking}".strip()
                        teacher_backfilled_reasoning += 1
                    pending_encrypted_reasoning = 0
                if role in {"user", "assistant"} and not any(
                    msg.get("role") == role and msg.get("content") == content
                    for msg in messages[-3:]
                ):
                    _append_message(messages, role, content)

    if not _has_trainable_pair(messages):
        return None
    metadata: dict[str, Any] = {"source": "session"}
    if skipped_encrypted_reasoning:
        metadata["reasoning_status"] = "encrypted_skipped"
        metadata["encrypted_reasoning_skipped"] = skipped_encrypted_reasoning
    if teacher_backfilled_reasoning:
        metadata["reasoning_status"] = "teacher_backfilled"
        metadata["teacher_backfilled_reasoning"] = teacher_backfilled_reasoning
        metadata["synthetic_reasoning"] = True
    return {"messages": messages, "metadata": metadata}


def rows_from_json_doc(
    doc: Any,
    *,
    include_failures: bool = False,
    include_thinking: bool = False,
    max_tool_output_chars: int | None = 4000,
    reasoning_policy: str = "skip",
    teacher_backfill_fn: TeacherBackfillFn | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(doc, dict):
        return []
    if str(doc.get("schema_version", "")).startswith("ATIF") or "steps" in doc:
        row = row_from_atif(
            doc,
            include_failures=include_failures,
            include_thinking=include_thinking,
            max_tool_output_chars=max_tool_output_chars,
            reasoning_policy=reasoning_policy,
            teacher_backfill_fn=teacher_backfill_fn,
        )
        return [row] if row else []
    if isinstance(doc.get("events"), list):
        row = row_from_session_events(
            doc["events"],
            max_tool_output_chars=max_tool_output_chars,
            reasoning_policy=reasoning_policy,
            teacher_backfill_fn=teacher_backfill_fn,
        )
        return [row] if row else []
    if isinstance(doc.get("session"), dict) and isinstance(doc["session"].get("events"), list):
        row = row_from_session_events(
            doc["session"]["events"],
            max_tool_output_chars=max_tool_output_chars,
            reasoning_policy=reasoning_policy,
            teacher_backfill_fn=teacher_backfill_fn,
        )
        return [row] if row else []
    return []


def rows_from_jsonl(
    path: Path,
    *,
    max_tool_output_chars: int | None = 4000,
    reasoning_policy: str = "skip",
    teacher_backfill_fn: TeacherBackfillFn | None = None,
) -> list[dict[str, Any]]:
    events = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    row = row_from_session_events(
        events,
        max_tool_output_chars=max_tool_output_chars,
        reasoning_policy=reasoning_policy,
        teacher_backfill_fn=teacher_backfill_fn,
    )
    return [row] if row else []


def convert_paths(
    paths: Iterable[Path],
    *,
    include_failures: bool = False,
    include_thinking: bool = False,
    max_tool_output_chars: int | None = 4000,
    reasoning_policy: str = "skip",
    teacher_backfill_fn: TeacherBackfillFn | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _iter_files(paths):
        if path.suffix == ".jsonl":
            rows.extend(
                rows_from_jsonl(
                    path,
                    max_tool_output_chars=max_tool_output_chars,
                    reasoning_policy=reasoning_policy,
                    teacher_backfill_fn=teacher_backfill_fn,
                )
            )
            continue
        if path.suffix != ".json":
            continue
        try:
            doc = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        rows.extend(
            rows_from_json_doc(
                doc,
                include_failures=include_failures,
                include_thinking=include_thinking,
                max_tool_output_chars=max_tool_output_chars,
                reasoning_policy=reasoning_policy,
                teacher_backfill_fn=teacher_backfill_fn,
            )
        )
    return rows


def write_jsonl(rows: Iterable[dict[str, Any]], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _iter_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(
                p
                for p in path.rglob("*")
                if p.is_file() and p.suffix in {".json", ".jsonl"}
            )
        elif path.is_file():
            yield path


def _has_trainable_pair(messages: list[dict[str, Any]]) -> bool:
    return any(m.get("role") == "user" for m in messages) and any(
        m.get("role") == "assistant" and m.get("content") for m in messages
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Convert Codex/Puffer ATIF trajectories or session logs to Yeto chat JSONL."
    )
    p.add_argument("--input", nargs="+", required=True, help="Input trace files or directories")
    p.add_argument("--output", required=True, help="Output JSONL path")
    p.add_argument("--include-failures", action="store_true", help="Keep failed ATIF runs")
    p.add_argument(
        "--include-thinking",
        action="store_true",
        help="Include stored thinking traces in assistant content",
    )
    p.add_argument(
        "--reasoning-policy",
        choices=REASONING_POLICIES,
        default="skip",
        help=(
            "How to handle encrypted reasoning when --include-thinking is set. "
            "skip omits it; teacher-backfill synthesizes a visible-context rationale."
        ),
    )
    p.add_argument(
        "--teacher-model",
        default=os.environ.get("YETO_TEACHER_MODEL", "gpt-5.4"),
        help="Teacher model for --reasoning-policy teacher-backfill",
    )
    p.add_argument(
        "--teacher-api-key",
        default=None,
        help="OpenAI API key for teacher backfill; defaults to OPENAI_API_KEY",
    )
    p.add_argument(
        "--teacher-base-url",
        default=None,
        help="OpenAI-compatible base URL for teacher backfill; defaults to OPENAI_BASE_URL or https://api.openai.com/v1",
    )
    p.add_argument(
        "--teacher-max-output-tokens",
        type=int,
        default=160,
        help="Max output tokens for each teacher backfill rationale",
    )
    p.add_argument(
        "--max-tool-output-chars",
        type=int,
        default=4000,
        help="Truncate each tool observation to this many characters; <=0 disables truncation",
    )
    args = p.parse_args(argv)

    limit = args.max_tool_output_chars if args.max_tool_output_chars > 0 else None
    teacher_backfill_fn = None
    if args.reasoning_policy == "teacher-backfill":
        if not args.include_thinking:
            p.error("--reasoning-policy teacher-backfill requires --include-thinking")
        teacher_backfill_fn = _openai_teacher_backfill(
            args.teacher_model,
            api_key=args.teacher_api_key,
            base_url=args.teacher_base_url,
            max_output_tokens=args.teacher_max_output_tokens,
        )
    rows = convert_paths(
        [Path(p) for p in args.input],
        include_failures=args.include_failures,
        include_thinking=args.include_thinking,
        max_tool_output_chars=limit,
        reasoning_policy=args.reasoning_policy,
        teacher_backfill_fn=teacher_backfill_fn,
    )
    count = write_jsonl(rows, Path(args.output))
    print(f"wrote {count} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
