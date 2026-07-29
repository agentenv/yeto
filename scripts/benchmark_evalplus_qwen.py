#!/usr/bin/env python3
"""Generate EvalPlus samples with Yeto's Qwen3.6-compatible HF loader."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from eval_codex_prompts import load_full_model


INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script that solves the following "
    "problem in a markdown code block:"
)
RESPONSE_PREFIX = (
    "Below is a Python script with a self-contained function that solves the "
    "problem and passes the corresponding tests:\n```python\n"
)


def render_code_prompt(tokenizer, task_prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": f"{INSTRUCTION_PREFIX}\n```python\n{task_prompt.strip()}\n```",
            },
            {"role": "assistant", "content": RESPONSE_PREFIX},
        ],
        tokenize=False,
        continue_final_message=True,
        enable_thinking=False,
    )


def _load_problems(dataset: str):
    try:
        from evalplus.data import get_human_eval_plus, get_mbpp_plus
    except ImportError as exc:
        raise SystemExit(
            "EvalPlus is required: python -m pip install 'evalplus==0.3.1'"
        ) from exc
    return get_human_eval_plus() if dataset == "humaneval" else get_mbpp_plus()


def _completed_task_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    completed = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                completed.add(json.loads(line)["task_id"])
    return completed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model ID or exported checkpoint")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--limit", type=int, help="Generation smoke-test limit")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    try:
        from evalplus.sanitize import sanitize
    except ImportError as exc:
        raise SystemExit(
            "EvalPlus is required: python -m pip install 'evalplus==0.3.1'"
        ) from exc

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_task_ids(output_path)
    problems = _load_problems(args.dataset)
    tasks = list(problems.items())
    if args.limit is not None:
        tasks = tasks[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    model = load_full_model(args.model, args.device)

    pending = [(task_id, task) for task_id, task in tasks if task_id not in completed]
    for index, (task_id, task) in enumerate(pending, 1):
        rendered = render_code_prompt(tokenizer, task["prompt"])
        inputs = tokenizer(rendered, return_tensors="pt").to(args.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(
            generated[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        )
        solution = sanitize(completion, entrypoint=task["entry_point"])
        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"task_id": task_id, "solution": solution}) + "\n")
            f.flush()
        print(f"[{index}/{len(pending)}] {task_id}", flush=True)

    print(f"wrote {len(tasks)} {args.dataset} samples to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
