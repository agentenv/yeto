#!/usr/bin/env python3
"""Probe whether assistant-only SFT uses native chat formatting for models."""

from __future__ import annotations

import argparse

from transformers import AutoTokenizer

from yeto.data import _row_tokens
from yeto.models import resolve


ROW = {
    "messages": [
        {"role": "system", "content": "Follow the user's request."},
        {"role": "user", "content": "Say exactly: ok"},
        {"role": "assistant", "content": "ok"},
    ]
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["glm45-air", "deepseek31-bf16", "kimi-k2-thinking-bf16", "qwen3-8b", "gemma4"],
        help="model aliases or HF ids to probe",
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    for model in args.models:
        model_id = resolve(model)
        print(f"=== {model} -> {model_id} ===")
        try:
            tok = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                local_files_only=args.local_files_only,
            )
            ids, weights = _row_tokens(tok, ROW, "assistant")
            decoded = tok.decode(ids)
            synthetic = "<|assistant|>" in decoded or "<|user|>" in decoded
            print(f"chat_template={bool(getattr(tok, 'chat_template', None))}")
            print(f"tokens={len(ids)} weighted={sum(weights):.1f} synthetic_markers={synthetic}")
            print(decoded[:500].replace("\n", "\\n"))
        except Exception as exc:
            print(f"ERROR {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
