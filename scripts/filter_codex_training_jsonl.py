#!/usr/bin/env python3
"""Filter Codex/Yeto chat JSONL into cleaner SFT examples.

The default mode emits simple user -> assistant pairs. That is intentionally
conservative for tiny models: it drops progress chatter, tool/log-heavy
assistant messages, and synthetic thinking blocks so the model mostly sees
usable final-answer style supervision.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PROGRESS_PATTERNS = [
    re.compile(r"^\s*i('|')?ll\b", re.I),
    re.compile(r"^\s*i am going to\b", re.I),
    re.compile(r"^\s*i('|')?m going to\b", re.I),
    re.compile(r"^\s*i('|')?ll (read|check|inspect|look|run|open|trace|verify)\b", re.I),
    re.compile(r"\bi('|')?ll (read|check|inspect|look|run|open|trace|verify|quickly confirm)\b", re.I),
    re.compile(r"\bi('|')?m checking\b", re.I),
    re.compile(r"\bi am checking\b", re.I),
    re.compile(r"^found\b.+\bchecking\b", re.I | re.S),
]

REPEATED_LINE_RE = re.compile(r"^(?P<line>.+)(?:\n(?P=line)){3,}$", re.S)
THINKING_RE = re.compile(r"\n?\[thinking\]\n.*", re.S)
FERNET_BLOB_RE = re.compile(r"\bgAAAAA[A-Za-z0-9_-]{80,}={0,2}\b")


def strip_thinking(text: str) -> str:
    return THINKING_RE.sub("", text).strip()


def repeated_token_ratio(text: str) -> float:
    words = re.findall(r"[A-Za-z0-9_./-]+", text.lower())
    if not words:
        return 1.0
    return 1.0 - (len(set(words)) / len(words))


def code_or_log_line_ratio(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 1.0
    noisy = 0
    for line in lines:
        if line.startswith(("```", "$ ", "./", ".venv/", "Traceback", "File ", "2026-")):
            noisy += 1
        elif len(re.findall(r"[/{}()[\]=<>|\\]", line)) >= 6:
            noisy += 1
        elif re.search(r"\b(INFO|DEBUG|WARNING|ERROR)\b", line):
            noisy += 1
    return noisy / len(lines)


def looks_like_progress(text: str) -> bool:
    short = len(text) < 320
    return short and any(pattern.search(text) for pattern in PROGRESS_PATTERNS)


def is_good_assistant(text: str, *, min_chars: int, max_chars: int) -> bool:
    stripped = text.strip()
    if len(stripped) < min_chars or len(stripped) > max_chars:
        return False
    if FERNET_BLOB_RE.search(stripped):
        return False
    if looks_like_progress(stripped):
        return False
    if REPEATED_LINE_RE.match(stripped):
        return False
    if repeated_token_ratio(stripped) > 0.72:
        return False
    if code_or_log_line_ratio(stripped) > 0.55:
        return False
    return True


def is_good_user(text: str, *, min_chars: int, max_chars: int) -> bool:
    stripped = text.strip()
    if len(stripped) < min_chars or len(stripped) > max_chars:
        return False
    if FERNET_BLOB_RE.search(stripped):
        return False
    lowered = stripped.lower()
    if any(
        marker in lowered
        for marker in (
            "traceback (most recent call last)",
            "npm notice",
            "packages are looking for funding",
            "run `npm audit`",
            "loading weights:",
            "generating train split:",
        )
    ):
        return False
    if code_or_log_line_ratio(stripped) > 0.75:
        return False
    return True


def pair_rows(
    row: dict[str, Any],
    *,
    drop_thinking: bool,
    min_user_chars: int,
    min_assistant_chars: int,
    max_user_chars: int,
    max_assistant_chars: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    last_user: str | None = None
    source_meta = row.get("metadata", {})
    for msg in row.get("messages", []):
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        if role == "user":
            text = content.strip()
            last_user = (
                text
                if is_good_user(text, min_chars=min_user_chars, max_chars=max_user_chars)
                else None
            )
            continue
        if role != "assistant" or last_user is None:
            continue
        assistant = strip_thinking(content) if drop_thinking else content.strip()
        if not is_good_assistant(
            assistant,
            min_chars=min_assistant_chars,
            max_chars=max_assistant_chars,
        ):
            continue
        out.append(
            {
                "messages": [
                    {"role": "user", "content": last_user},
                    {"role": "assistant", "content": assistant},
                ],
                "metadata": {
                    "source": source_meta.get("source", "codex"),
                    "filtered_from": "codex_traces",
                    "filter_mode": "pairs",
                },
            }
        )
        last_user = None
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input Yeto chat JSONL")
    parser.add_argument("--output", required=True, help="Filtered output JSONL")
    parser.add_argument("--keep-thinking", action="store_true", help="Keep [thinking] blocks")
    parser.add_argument("--min-user-chars", type=int, default=12)
    parser.add_argument("--min-assistant-chars", type=int, default=160)
    parser.add_argument("--max-user-chars", type=int, default=1800)
    parser.add_argument("--max-assistant-chars", type=int, default=6000)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    read_rows = 0
    wrote_rows = 0
    with input_path.open() as src, output_path.open("w") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            read_rows += 1
            row = json.loads(line)
            for filtered in pair_rows(
                row,
                drop_thinking=not args.keep_thinking,
                min_user_chars=args.min_user_chars,
                min_assistant_chars=args.min_assistant_chars,
                max_user_chars=args.max_user_chars,
                max_assistant_chars=args.max_assistant_chars,
            ):
                dst.write(json.dumps(filtered, ensure_ascii=False, sort_keys=True) + "\n")
                wrote_rows += 1

    print(f"read {read_rows} rows; wrote {wrote_rows} filtered pair rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
