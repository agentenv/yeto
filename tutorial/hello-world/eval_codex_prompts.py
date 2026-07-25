"""Run a small qualitative Codex-style eval against a trained model.

The script writes JSONL rows with prompt and candidate response. It intentionally
does not score automatically; read the outputs or feed them to a separate judge.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPTS = [
    "A Python test fails only when run with the whole suite, but passes alone. What should I check first?",
    "A CLI conversion script silently drops some records from a JSONL trace. How would you debug it?",
    "Explain why an encrypted reasoning field should not be included directly in SFT training data.",
    "A training run saves an adapter, but base and adapter generations are identical. What are the likely causes?",
    "A long API backfill job times out after an hour. How would you make it resilient without losing the whole run?",
    "A JSONL dataset contains tool logs, progress updates, final answers, and synthetic thinking. What would you filter before SFT?",
    "A model fine-tuned for 100 steps starts repeating fake command flags. What training settings would you change first?",
    "You need to compare a base model and a LoRA model fairly. What mistakes should you avoid in the evaluation script?",
    "A local file path works in one terminal but the training process says FileNotFoundError. What should you verify?",
    "A pytest suite has order-dependent failures after one test mutates global state. How would you isolate the culprit?",
    "A repo has merge conflicts in a Python converter and its tests. What is a safe process for resolving them?",
    "A model training dataset includes assistant messages that say 'I'll inspect the file now.' Should those be training targets?",
    "A teacher model backfills reasoning from visible context. What should the prompt forbid to avoid bad synthetic data?",
    "A small model gets worse after full SFT but LoRA is stable. What does that suggest about the recipe?",
    "A command-line tool writes output only at the end of a long run. How can you monitor whether it is still alive?",
    "A JSON parser skips malformed lines silently. How would you add observability without breaking valid conversions?",
    "A training eval has only three prompts and no scoring rubric. What minimum eval would you build next?",
    "A trace contains user-visible conversations about encrypted_content and ciphertext. Why can grep produce false leak alarms?",
    "A Qwen chat model is evaluated with raw 'User:' and 'Assistant:' strings. Why might that be misleading?",
    "A manager asks whether the new tuned model is better. What evidence would you need before saying yes?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="google/gemma-4-12B-it")
    parser.add_argument("--candidate", required=True, help="Full fine-tuned model directory")
    parser.add_argument("--candidate-kind", choices=["full"], default="full")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--limit", type=int, default=len(PROMPTS))
    return parser.parse_args()


def render_prompt(tokenizer, prompt: str):
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    return tokenizer(f"User: {prompt}\nAssistant:", return_tensors="pt").input_ids


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    input_ids = render_prompt(tokenizer, prompt).to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0, input_ids.shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.candidate,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prompts = PROMPTS[: args.limit]
    with output.open("w") as f:
        for index, prompt in enumerate(prompts, 1):
            print(f"[{index}/{len(prompts)}] {prompt}", flush=True)
            candidate = generate(model, tokenizer, prompt, args.max_new_tokens)
            f.write(json.dumps({"prompt": prompt, "candidate": candidate}) + "\n")
            f.flush()
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

