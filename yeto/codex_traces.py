"""Convert Codex/Puffer trace artifacts into Yeto chat JSONL rows.

The preferred source is an ATIF ``trajectory.json`` artifact. It already
normalizes streamed deltas, tool calls, observations, retries, and final
assistant messages into ordered ``steps``. Standalone session transcript JSON
and JSONL files are also accepted as a fallback.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


CHAT_ROLES = {"system", "user", "assistant", "tool"}


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
) -> dict[str, Any] | None:
    """Build one Yeto row from an ATIF trajectory document."""
    success = doc.get("extra", {}).get("success")
    if success is False and not include_failures:
        return None

    steps = doc.get("steps")
    if not isinstance(steps, list):
        return None

    messages: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        source = step.get("source")
        content = _text(step.get("message"))
        if include_thinking:
            thinking = _text(step.get("extra", {}).get("thinking"))
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
        row["metadata"] = {
            "source": "atif",
            "session_id": doc.get("session_id"),
            "model": agent.get("model_name"),
            "success": success,
        }
    return row


def row_from_session_events(
    events: Iterable[dict[str, Any]],
    *,
    max_tool_output_chars: int | None = 4000,
) -> dict[str, Any] | None:
    """Build one Yeto row from Puffer session transcript events."""
    messages: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        typ = event.get("type")
        if typ == "user_message":
            _append_message(messages, "user", _text(event.get("text")))
        elif typ == "assistant_message":
            _append_message(messages, "assistant", _text(event.get("text")))
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

    if not _has_trainable_pair(messages):
        return None
    return {"messages": messages, "metadata": {"source": "session"}}


def rows_from_json_doc(
    doc: Any,
    *,
    include_failures: bool = False,
    include_thinking: bool = False,
    max_tool_output_chars: int | None = 4000,
) -> list[dict[str, Any]]:
    if not isinstance(doc, dict):
        return []
    if str(doc.get("schema_version", "")).startswith("ATIF") or "steps" in doc:
        row = row_from_atif(
            doc,
            include_failures=include_failures,
            include_thinking=include_thinking,
            max_tool_output_chars=max_tool_output_chars,
        )
        return [row] if row else []
    if isinstance(doc.get("events"), list):
        row = row_from_session_events(
            doc["events"], max_tool_output_chars=max_tool_output_chars
        )
        return [row] if row else []
    if isinstance(doc.get("session"), dict) and isinstance(doc["session"].get("events"), list):
        row = row_from_session_events(
            doc["session"]["events"], max_tool_output_chars=max_tool_output_chars
        )
        return [row] if row else []
    return []


def rows_from_jsonl(
    path: Path,
    *,
    max_tool_output_chars: int | None = 4000,
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
    row = row_from_session_events(events, max_tool_output_chars=max_tool_output_chars)
    return [row] if row else []


def convert_paths(
    paths: Iterable[Path],
    *,
    include_failures: bool = False,
    include_thinking: bool = False,
    max_tool_output_chars: int | None = 4000,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _iter_files(paths):
        if path.suffix == ".jsonl":
            rows.extend(rows_from_jsonl(path, max_tool_output_chars=max_tool_output_chars))
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
        "--max-tool-output-chars",
        type=int,
        default=4000,
        help="Truncate each tool observation to this many characters; <=0 disables truncation",
    )
    args = p.parse_args(argv)

    limit = args.max_tool_output_chars if args.max_tool_output_chars > 0 else None
    rows = convert_paths(
        [Path(p) for p in args.input],
        include_failures=args.include_failures,
        include_thinking=args.include_thinking,
        max_tool_output_chars=limit,
    )
    count = write_jsonl(rows, Path(args.output))
    print(f"wrote {count} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
