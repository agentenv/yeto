#!/usr/bin/env python3
"""Run held-out loss and qualitative Codex-style evaluation.

The base and candidate are loaded sequentially so 27B checkpoints do not need
to coexist in GPU or host memory. The output JSONL contains paired generations;
a companion metrics JSON records assistant-only held-out loss.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)

from yeto.data import build_packed_dataset


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


SYSTEM_PROMPT = "You are Codex, a concise and careful coding assistant."


def _model_class(source: str):
    config = AutoConfig.from_pretrained(source, trust_remote_code=True)
    architectures = set(getattr(config, "architectures", None) or [])
    if config.model_type == "qwen3_5" or any(
        name.endswith("ForConditionalGeneration") for name in architectures
    ):
        return AutoModelForImageTextToText
    return AutoModelForCausalLM


def _patch_qwen35_causal_conv_update(model) -> int:
    """Adapt the Qwen3.5 cached decode input to causal-conv1d's 2D API."""
    if getattr(getattr(model, "config", None), "model_type", None) != "qwen3_5":
        return 0

    patched = 0
    for module in model.modules():
        update = getattr(module, "causal_conv1d_update", None)
        if (
            update is None
            or getattr(module, "_yeto_causal_conv_update_patched", False)
            or not getattr(update, "__module__", "").startswith("causal_conv1d")
        ):
            continue

        def compatible_update(hidden_states, *args, _update=update, **kwargs):
            if hidden_states.ndim == 3 and hidden_states.shape[-1] == 1:
                return _update(hidden_states.squeeze(-1), *args, **kwargs).unsqueeze(-1)
            return _update(hidden_states, *args, **kwargs)

        module.causal_conv1d_update = compatible_update
        module._yeto_causal_conv_update_patched = True
        patched += 1
    return patched


def load_full_model(source: str, device: str):
    model = _model_class(source).from_pretrained(
        source,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": device},
        low_cpu_mem_usage=True,
    ).eval()
    patched = _patch_qwen35_causal_conv_update(model)
    if patched:
        print(f"patched cached Qwen3.5 generation in {patched} linear-attention layers")
    return model


def load_candidate(base_model: str, candidate: str, kind: str, device: str):
    if kind == "full":
        return load_full_model(candidate, device)
    if kind == "lora":
        from peft import PeftModel

        return PeftModel.from_pretrained(
            load_full_model(base_model, device), candidate
        ).eval()
    raise ValueError(f"unknown candidate kind: {kind}")


def render_prompt(tokenizer, text: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def generate(tokenizer, model, prompt: str, *, max_new_tokens: int, device: str) -> str:
    rendered = render_prompt(tokenizer, prompt)
    inputs = tokenizer(rendered, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    ).strip()


def heldout_loss(
    tokenizer,
    model,
    data_path: str,
    *,
    seq_len: int,
    max_blocks: int,
    device: str,
) -> dict[str, float | int]:
    dataset = build_packed_dataset(
        data_path,
        tokenizer,
        learner_id=0,
        num_learners=1,
        seq_len=seq_len,
        train_on="assistant",
    )
    blocks = min(len(dataset), max_blocks)
    loss_sum = 0.0
    target_tokens = 0
    with torch.inference_mode():
        for idx in range(blocks):
            input_ids, weights = dataset[idx]
            input_ids = input_ids.unsqueeze(0).to(device)
            weights = weights.unsqueeze(0).to(device)
            labels = input_ids.masked_fill(weights == 0, -100)
            output = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                labels=labels,
                use_cache=False,
            )
            count = int((labels[:, 1:] != -100).sum().item())
            loss_sum += float(output.loss.item()) * count
            target_tokens += count
    if target_tokens == 0:
        raise ValueError("held-out data produced no assistant target tokens")
    mean_loss = loss_sum / target_tokens
    return {
        "loss": mean_loss,
        "perplexity": float(torch.exp(torch.tensor(min(mean_loss, 20.0))).item()),
        "target_tokens": target_tokens,
        "blocks": blocks,
    }


def release_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_model(
    tokenizer,
    model,
    prompts: list[str],
    *,
    eval_data: str | None,
    seq_len: int,
    max_eval_blocks: int,
    max_new_tokens: int,
    device: str,
) -> tuple[list[str], dict[str, float | int] | None]:
    metrics = None
    if eval_data:
        metrics = heldout_loss(
            tokenizer,
            model,
            eval_data,
            seq_len=seq_len,
            max_blocks=max_eval_blocks,
            device=device,
        )
    generations = []
    for idx, prompt in enumerate(prompts, 1):
        print(f"[{idx}/{len(prompts)}] {prompt}")
        generations.append(
            generate(
                tokenizer,
                model,
                prompt,
                max_new_tokens=max_new_tokens,
                device=device,
            )
        )
    return generations, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--candidate", required=True, help="Full model dir or LoRA adapter dir")
    parser.add_argument("--candidate-kind", choices=["full", "lora"], required=True)
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--metrics-output", help="Metrics JSON path; defaults beside --output")
    parser.add_argument("--eval-data", help="Held-out Yeto chat JSONL")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-eval-blocks", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--limit", type=int, default=len(PROMPTS))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    prompts = PROMPTS[: args.limit]

    print(f"loading base model: {args.base_model}")
    base = load_full_model(args.base_model, args.device)
    base_generations, base_metrics = evaluate_model(
        tokenizer,
        base,
        prompts,
        eval_data=args.eval_data,
        seq_len=args.seq_len,
        max_eval_blocks=args.max_eval_blocks,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    del base
    release_cuda_cache()

    print(f"loading candidate: {args.candidate}")
    candidate = load_candidate(
        args.base_model, args.candidate, args.candidate_kind, args.device
    )
    candidate_generations, candidate_metrics = evaluate_model(
        tokenizer,
        candidate,
        prompts,
        eval_data=args.eval_data,
        seq_len=args.seq_len,
        max_eval_blocks=args.max_eval_blocks,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    del candidate
    release_cuda_cache()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for idx, (prompt, base_text, candidate_text) in enumerate(
            zip(prompts, base_generations, candidate_generations), 1
        ):
            f.write(
                json.dumps(
                    {
                        "id": idx,
                        "prompt": prompt,
                        "base": base_text,
                        "candidate": candidate_text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    metrics_path = (
        Path(args.metrics_output)
        if args.metrics_output
        else output_path.with_suffix(output_path.suffix + ".metrics.json")
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "base_model": args.base_model,
                "candidate": args.candidate,
                "candidate_kind": args.candidate_kind,
                "eval_data": args.eval_data,
                "base": base_metrics,
                "candidate_metrics": candidate_metrics,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"wrote eval results to {output_path}")
    print(f"wrote eval metrics to {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
