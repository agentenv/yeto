#!/usr/bin/env python3
"""Convert cot_extracted.jsonl rows into Yeto chat JSONL.

The extraction file has rows shaped like:
  {"id": "...", "summary": "...", "method": "...", "text": "..."}

Some ``text`` values are still encrypted blobs or extraction refusals. This
script keeps only readable assistant text and wraps it in Yeto's ``messages``
format so the LM data loader can apply assistant-only masking.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any


FERNET_RE = re.compile(r"^gAAAAA[A-Za-z0-9_-]{80,}={0,2}(?:\s+PLAY)?$")
BASE64ISH_RE = re.compile(r"^[A-Za-z0-9_+\-/=\s]{240,}$")
REFUSAL_RE = re.compile(
    r"(\bencrypted\b|encrypted[/ ]or encoded data|encrypted or encoded data|encrypted token|"
    r"appears to be encrypted|supplied content is an encrypted|"
    r"can't play back|can.t play back|cannot play back|"
    r"not a readable transcript|no readable transcript)",
    re.I,
)
PROMPT_INJECTION_RE = re.compile(
    r"^\s*(you are a log viewer|output the complete transcript|"
    r"this is a meeting transcription service|ignore previous instructions)\b",
    re.I,
)


def _summary_text(value: Any) -> str:
    if not value:
        return ""
    parsed = value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value.strip()
    if isinstance(parsed, list):
        parts = []
        for item in parsed:
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]).strip())
            elif isinstance(item, str):
                parts.append(item.strip())
        return " ".join(part for part in parts if part).strip()
    if isinstance(parsed, dict) and parsed.get("text"):
        return str(parsed["text"]).strip()
    return str(parsed).strip()


def looks_encrypted(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if FERNET_RE.match(stripped):
        return True
    if len(stripped) > 800 and BASE64ISH_RE.match(stripped):
        word_count = len(re.findall(r"[a-zA-Z]{3,}", stripped))
        return word_count < 20
    return False


def keep_text(text: str, *, min_chars: int, max_chars: int) -> bool:
    stripped = text.strip()
    if len(stripped) < min_chars or len(stripped) > max_chars:
        return False
    if looks_encrypted(stripped):
        return False
    if REFUSAL_RE.search(stripped):
        return False
    if PROMPT_INJECTION_RE.search(stripped):
        return False
    return True


def make_row(row: dict[str, Any], *, min_chars: int, max_chars: int) -> dict[str, Any] | None:
    text = str(row.get("text") or "").strip()
    if not keep_text(text, min_chars=min_chars, max_chars=max_chars):
        return None

    summary = _summary_text(row.get("summary"))
    user = "Write the assistant response for this Codex task."
    if summary and summary != "[]":
        user += f"\n\nTask summary: {summary}"

    return {
        "messages": [
            {
                "role": "system",
                "content": "You are Codex, a concise and careful coding assistant.",
            },
            {"role": "user", "content": user},
            {"role": "assistant", "content": text},
        ],
        "metadata": {
            "source": row.get("src"),
            "source_id": row.get("id"),
            "method": row.get("method"),
            "filtered_from": "cot_extracted",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input cot_extracted JSONL")
    parser.add_argument("--output", required=True, help="Output Yeto chat JSONL")
    parser.add_argument("--min-chars", type=int, default=120)
    parser.add_argument("--max-chars", type=int, default=12000)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    read_rows = wrote_rows = skipped_encrypted = skipped_other = 0
    with input_path.open(encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            read_rows += 1
            row = json.loads(line)
            text = str(row.get("text") or "").strip()
            if looks_encrypted(text) or REFUSAL_RE.search(text):
                skipped_encrypted += 1
                continue
            out = make_row(row, min_chars=args.min_chars, max_chars=args.max_chars)
            if out is None:
                skipped_other += 1
                continue
            dst.write(json.dumps(out, ensure_ascii=False, sort_keys=True) + "\n")
            wrote_rows += 1

    print(
        "read {read}; wrote {wrote}; skipped encrypted/refusal {encrypted}; "
        "skipped length/other {other}".format(
            read=read_rows,
            wrote=wrote_rows,
            encrypted=skipped_encrypted,
            other=skipped_other,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
