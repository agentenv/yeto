#!/usr/bin/env python3
"""Align selected Codex reasoning items with authentic visible chat turns.

The output always pairs the real user request and final visible assistant
answer from the source session. In mixed mode, strictly screened recording
replays become Qwen-compatible ``reasoning_content`` and public summaries fill
the gaps. Replays are never substituted for the visible answer and remain
marked as unverified in row metadata.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts.format_cot_extracted_jsonl import keep_text
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from format_cot_extracted_jsonl import keep_text


FERNET_BLOB_RE = re.compile(r"\bgAAAAA[A-Za-z0-9_-]*")


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True).strip()


def _summary_texts(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return [value.strip()] if value.strip() else []
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = _text(item.get("text")) if isinstance(item, dict) else _text(item)
        if text and text not in out:
            out.append(text)
    return out


def load_manifest(path: Path) -> tuple[set[str], set[str], dict[str, str]]:
    reasoning_ids: set[str] = set()
    source_prefixes: set[str] = set()
    replay_text_by_id: dict[str, str] = {}
    with path.open(encoding="utf-8") as src:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            reasoning_id = _text(row.get("id"))
            source = _text(row.get("src"))
            if reasoning_id:
                reasoning_ids.add(reasoning_id)
            if source:
                source_prefixes.add(source)
            replay = _text(row.get("text"))
            if (
                reasoning_id
                and reasoning_id not in replay_text_by_id
                and row.get("method") == "recording"
                and keep_text(replay, min_chars=120, max_chars=12000)
            ):
                replay_text_by_id[reasoning_id] = replay
    return reasoning_ids, source_prefixes, replay_text_by_id


def _quality_ok(
    user: str,
    assistant: str,
    *,
    min_user_chars: int,
    max_user_chars: int,
    min_assistant_chars: int,
    max_assistant_chars: int,
) -> bool:
    if not (min_user_chars <= len(user) <= max_user_chars):
        return False
    if not (min_assistant_chars <= len(assistant) <= max_assistant_chars):
        return False
    return not FERNET_BLOB_RE.search(user) and not FERNET_BLOB_RE.search(assistant)


def aligned_rows_from_events(
    events: Iterable[dict[str, Any]],
    selected_ids: set[str],
    *,
    session_id: str | None = None,
    reasoning_source: str = "none",
    replay_text_by_id: dict[str, str] | None = None,
    include_public_summaries: bool = False,
    min_user_chars: int = 12,
    max_user_chars: int = 6000,
    min_assistant_chars: int = 80,
    max_assistant_chars: int = 12000,
) -> list[dict[str, Any]]:
    if reasoning_source not in {"none", "public-summary", "mixed"}:
        raise ValueError("reasoning_source must be none, public-summary, or mixed")
    if include_public_summaries and reasoning_source == "none":
        reasoning_source = "public-summary"
    replay_text_by_id = replay_text_by_id or {}
    rows: list[dict[str, Any]] = []
    current_user = ""
    reasoning_ids: list[str] = []
    summaries_by_id: dict[str, list[str]] = {}

    def reset_reasoning() -> None:
        reasoning_ids.clear()
        summaries_by_id.clear()

    for event in events:
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        payload = event.get("payload")
        if event_type == "session_meta" and isinstance(payload, dict):
            session_id = session_id or _text(payload.get("id")) or None
            continue
        if not isinstance(payload, dict):
            continue

        payload_type = payload.get("type")
        if event_type == "event_msg" and payload_type == "user_message":
            current_user = _text(payload.get("message"))
            reset_reasoning()
            continue

        if event_type == "response_item" and payload_type == "reasoning":
            reasoning_id = _text(payload.get("id"))
            if reasoning_id in selected_ids:
                reasoning_ids.append(reasoning_id)
                summaries_by_id[reasoning_id] = _summary_texts(payload.get("summary"))
            continue

        if event_type != "event_msg" or payload_type != "agent_message":
            continue
        if payload.get("phase") != "final_answer":
            continue

        assistant = _text(payload.get("message"))
        if current_user and reasoning_ids and _quality_ok(
            current_user,
            assistant,
            min_user_chars=min_user_chars,
            max_user_chars=max_user_chars,
            min_assistant_chars=min_assistant_chars,
            max_assistant_chars=max_assistant_chars,
        ):
            assistant_message: dict[str, Any] = {"role": "assistant", "content": assistant}
            reasoning_parts: list[str] = []
            reasoning_segments: list[dict[str, str]] = []
            for reasoning_id in reasoning_ids:
                replay = replay_text_by_id.get(reasoning_id) if reasoning_source == "mixed" else None
                if replay:
                    reasoning_parts.append(replay)
                    reasoning_segments.append(
                        {
                            "reasoning_id": reasoning_id,
                            "source": "replay_text",
                            "validation": "strict_text_filter",
                            "authenticity": "unverified",
                        }
                    )
                    continue
                if reasoning_source in {"public-summary", "mixed"}:
                    public_summaries = summaries_by_id.get(reasoning_id, [])
                    if public_summaries:
                        reasoning_parts.extend(public_summaries)
                        reasoning_segments.append(
                            {
                                "reasoning_id": reasoning_id,
                                "source": "public_summary",
                                "validation": "source_session",
                                "authenticity": "verified_public",
                            }
                        )
            if reasoning_parts:
                assistant_message["reasoning_content"] = "\n\n".join(reasoning_parts)
            rows.append(
                {
                    "messages": [
                        {"role": "user", "content": current_user},
                        assistant_message,
                    ],
                    "metadata": {
                        "source": "codex_session_aligned",
                        "session_id": session_id,
                        "reasoning_ids": list(reasoning_ids),
                        "reasoning_source": reasoning_source,
                        "reasoning_segments": reasoning_segments,
                        "extracted_replay_used": False,
                    },
                }
            )
            if any(segment["source"] == "replay_text" for segment in reasoning_segments):
                rows[-1]["metadata"]["extracted_replay_used"] = True
        current_user = ""
        reset_reasoning()

    return rows


def _matching_session_files(session_dir: Path, source_prefixes: set[str]) -> list[Path]:
    return [
        path
        for path in sorted(session_dir.rglob("*.jsonl"))
        if not source_prefixes or any(path.name.startswith(prefix) for prefix in source_prefixes)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="cot_extracted.jsonl ID manifest")
    parser.add_argument("--sessions", required=True, help="Codex sessions directory")
    parser.add_argument("--output", required=True, help="Aligned Yeto JSONL output")
    parser.add_argument(
        "--include-public-summaries",
        action="store_true",
        help="Deprecated alias for --reasoning-source public-summary",
    )
    parser.add_argument(
        "--reasoning-source",
        choices=["none", "public-summary", "mixed"],
        default="none",
        help="Reasoning content: none, public summaries, or validated replay with summary fallback",
    )
    parser.add_argument("--min-user-chars", type=int, default=12)
    parser.add_argument("--max-user-chars", type=int, default=6000)
    parser.add_argument("--min-assistant-chars", type=int, default=80)
    parser.add_argument("--max-assistant-chars", type=int, default=12000)
    args = parser.parse_args(argv)

    manifest = Path(args.manifest).expanduser()
    session_dir = Path(args.sessions).expanduser()
    output = Path(args.output).expanduser()
    selected_ids, source_prefixes, replay_text_by_id = load_manifest(manifest)
    session_files = _matching_session_files(session_dir, source_prefixes)

    output.parent.mkdir(parents=True, exist_ok=True)
    wrote = matched_ids = 0
    with output.open("w", encoding="utf-8") as dst:
        for path in session_files:
            events = []
            with path.open(encoding="utf-8") as src:
                for line in src:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            rows = aligned_rows_from_events(
                events,
                selected_ids,
                session_id=path.stem,
                reasoning_source=args.reasoning_source,
                replay_text_by_id=replay_text_by_id,
                include_public_summaries=args.include_public_summaries,
                min_user_chars=args.min_user_chars,
                max_user_chars=args.max_user_chars,
                min_assistant_chars=args.min_assistant_chars,
                max_assistant_chars=args.max_assistant_chars,
            )
            for row in rows:
                matched_ids += len(row["metadata"]["reasoning_ids"])
                dst.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                wrote += 1

    print(
        f"selected {len(selected_ids)} reasoning ids across {len(source_prefixes)} sources; "
        f"validated {len(replay_text_by_id)} replay candidates; "
        f"scanned {len(session_files)} sessions; wrote {wrote} aligned turns "
        f"covering {matched_ids} selected reasoning items to {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
