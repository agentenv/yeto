#!/usr/bin/env python3
"""Run one exact-shape Qwen2.5-7B full-tune learner step for v9 admission."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from argparse import Namespace
from pathlib import Path


MODEL_ID = "Qwen/Qwen2.5-7B"
MODEL_REVISION = "d149729398750b98c0af14eb82c78cfe92750796"
EXACT_PARAMETERS = 7_615_616_512


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--packed-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-repo", type=Path, default=Path("/root/yeto"))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing smoke proof: {args.output}")
    if git(args.source_repo, "status", "--porcelain=v1", "--untracked-files=all"):
        raise SystemExit("Qwen smoke requires a clean exact-commit checkout")

    import torch
    from safetensors import safe_open

    from yeto.learner import load_model_and_tokenizer, trainable_params
    from yeto.losses import sft_loss

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("Qwen smoke requires CUDA")
    torch.cuda.set_device(device)
    torch.manual_seed(9_700_001)
    torch.cuda.manual_seed_all(9_700_001)
    properties = torch.cuda.get_device_properties(device)
    if "H200" not in properties.name.upper():
        raise SystemExit(f"Qwen v9 smoke requires H200, got {properties.name}")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    load_args = Namespace(
        model=str(args.model.resolve()),
        shard="ddp",
        tuning="full",
        lora_r=16,
        lora_alpha=32,
        lora_targets="auto",
    )
    load_start = time.perf_counter()
    model, _tokenizer = load_model_and_tokenizer(load_args, device)
    model.config.use_cache = False
    model.train()
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_start
    params = trainable_params(model)
    parameter_count = sum(parameter.numel() for parameter in params.values())
    if parameter_count != EXACT_PARAMETERS:
        raise SystemExit(
            f"Qwen parameter count {parameter_count} != {EXACT_PARAMETERS}"
        )
    optimizer = torch.optim.AdamW(params.values(), lr=0.001, weight_decay=0.01)
    with safe_open(args.packed_input.resolve(), framework="pt", device="cpu") as handle:
        input_ids = handle.get_tensor("input_ids")[:1].to(device)
        weights = handle.get_tensor("weights")[:1].to(device)
        metadata = handle.metadata()
    if tuple(input_ids.shape) != (1, 128) or tuple(weights.shape) != (1, 128):
        raise SystemExit("packed input does not have campaign shape [1,128]")

    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    step_start = time.perf_counter()
    output = model(input_ids=input_ids, use_cache=False)
    loss_sum, trained_tokens = sft_loss(
        output.logits, input_ids, "cross_entropy", weights
    )
    loss = loss_sum / trained_tokens.clamp(min=1.0)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params.values(), 1.0)
    optimizer.step()
    torch.cuda.synchronize(device)
    step_seconds = time.perf_counter() - step_start
    loss_value = float(loss.item())
    if not math.isfinite(loss_value):
        raise SystemExit("Qwen one-step loss is nonfinite")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    result = {
        "schema": "yeto_outer_mup_v9_qwen_smoke_v1",
        "status": "PASS",
        "scientific_verification_cell": False,
        "endpoint_verification_loss_seen": False,
        "started_unix_s": started,
        "completed_unix_s": time.time(),
        "source_git_commit": git(args.source_repo, "rev-parse", "HEAD"),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_path": str(args.model.resolve()),
        "exact_parameters": parameter_count,
        "dtype": str(next(iter(params.values())).dtype),
        "packed_input": {
            "path": str(args.packed_input.resolve()),
            "bytes": args.packed_input.stat().st_size,
            "sha256": sha256_file(args.packed_input),
            "metadata": metadata,
        },
        "pipeline": {
            "tuning": "full",
            "learner_gpu_count": 1,
            "campaign_learner_count": 4,
            "inner_optimizer": "AdamW",
            "inner_lr": 0.001,
            "weight_decay": 0.01,
            "micro_batch_size": 1,
            "sequence_length": 128,
            "gradient_accumulation": 1,
            "train_on": "assistant",
            "trained_token_weight_sum": float(trained_tokens.item()),
        },
        "timing_seconds": {
            "model_load": load_seconds,
            "one_optimizer_step": step_seconds,
        },
        "one_step_loss_per_trained_token": loss_value,
        "gpu": {
            "name": properties.name,
            "total_bytes": properties.total_memory,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "free_after_step_bytes": free_bytes,
            "total_after_step_bytes": total_bytes,
        },
    }
    write_json_atomic(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
