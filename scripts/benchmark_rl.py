#!/usr/bin/env python3
"""Benchmark native Miles, one Yeto island, and federated Yeto RL.

For a requested M and G, every arm owns M*G GPUs and processes the same
round-major prompt groups.  The harness runs on one host and does not create
cloud resources.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from yeto.benchmark_resume import write_json_atomic
from yeto.provenance import file_sha256, source_tree_sha256

SYNCER_BIN = REPO_ROOT / "syncer/target/release/yeto-syncer"
_MILES_PORT_STRIDE = 1000
_ARM_KINDS = ("native", "single", "federated", "decoupled")
_RESUME_EXCLUDES = {
    "_active_seed",
    "_pass_ks",
    "dry_run",
    "overwrite",
    "report_dir",
    "resume",
    "work_dir",
}
_IMPLEMENTATION_PATHS = (
    Path(__file__),
    REPO_ROOT / "yeto",
    REPO_ROOT / "syncer/src",
    REPO_ROOT / "syncer/Cargo.toml",
    REPO_ROOT / "syncer/Cargo.lock",
    SYNCER_BIN,
)


@dataclass(frozen=True)
class Arm:
    name: str
    kind: str
    benchmark_islands: int
    islands: int
    gpus_per_island: int
    groups_per_round: int
    fragments: int = 1
    pipeline: int = 1
    local_horizon: int = 1


@dataclass(frozen=True)
class PromptStreams:
    combined_rows: tuple[dict[str, Any], ...]
    island_rows: tuple[tuple[dict[str, Any], ...], ...]
    combined_ids: tuple[int, ...]
    island_ids: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class WorkerSpec:
    learner_id: int
    gpus: int
    groups_per_round: int
    prompt_path: Path
    policy_sync: bool


def _positive_csv(spec: str, flag: str) -> list[int]:
    try:
        values = [int(value.strip()) for value in spec.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError(f"{flag} must be a comma-separated list of integers") from exc
    if not values:
        raise ValueError(f"{flag} must contain at least one value")
    if any(value <= 0 for value in values):
        raise ValueError(f"{flag} values must be positive")
    if len(values) != len(set(values)):
        raise ValueError(f"{flag} contains duplicates")
    return values


def parse_arm_kinds(spec: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in spec.split(",") if value.strip())
    if not values:
        raise ValueError("--arms must contain at least one arm")
    if len(values) != len(set(values)):
        raise ValueError("--arms contains duplicates")
    unknown = sorted(set(values) - set(_ARM_KINDS))
    if unknown:
        raise ValueError(f"--arms contains unknown arm(s): {', '.join(unknown)}")
    return values


def select_arms(
    spec: str,
    gpus_per_island: int,
    groups_per_island: int,
    kinds: tuple[str, ...] = _ARM_KINDS,
    *,
    fragments: int = 8,
    pipeline: int = 2,
    local_horizon: int = 4,
) -> list[Arm]:
    selected = set(kinds)
    if not selected or selected - set(_ARM_KINDS):
        raise ValueError(
            "arm kinds must be selected from native,single,federated,decoupled"
        )
    if "decoupled" in selected:
        if fragments < 2:
            raise ValueError("decoupled benchmark requires at least 2 fragments")
        if not 1 <= pipeline <= fragments:
            raise ValueError("decoupled benchmark pipeline must be between 1 and fragments")
        if local_horizon < 2:
            raise ValueError("decoupled benchmark local horizon must be at least 2")
    arms = []
    for islands in _positive_csv(spec, "--islands"):
        total_gpus = islands * gpus_per_island
        total_groups = islands * groups_per_island
        arms.extend(
            arm
            for arm in (
                Arm(
                    f"native-miles-m{islands}",
                    "native",
                    islands,
                    1,
                    total_gpus,
                    total_groups,
                ),
                Arm(
                    f"yeto-single-m{islands}",
                    "single",
                    islands,
                    1,
                    total_gpus,
                    total_groups,
                ),
                Arm(
                    f"yeto-federated-m{islands}",
                    "federated",
                    islands,
                    islands,
                    gpus_per_island,
                    groups_per_island,
                ),
                Arm(
                    f"yeto-decoupled-m{islands}",
                    "decoupled",
                    islands,
                    islands,
                    gpus_per_island,
                    groups_per_island,
                    fragments,
                    pipeline,
                    local_horizon,
                ),
            )
            if arm.kind in selected
        )
    return arms


def workload(
    arm: Arm, *, rounds: int, samples_per_group: int
) -> dict[str, int | float]:
    prompt_groups = arm.islands * arm.groups_per_round * rounds
    total_gpus = arm.islands * arm.gpus_per_island
    return {
        "total_gpus": total_gpus,
        "prompt_groups": prompt_groups,
        "trajectories": prompt_groups * samples_per_group,
        "groups_per_gpu_per_round": (arm.islands * arm.groups_per_round / total_gpus),
    }


def validate_workload(args) -> None:
    for name in (
        "global_rounds",
        "groups_per_island",
        "samples_per_group",
        "optimizer_steps",
        "gpus_per_island",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.optimizer_steps != 1:
        raise ValueError("Miles RL benchmark requires one optimizer step per rollout")
    samples = args.groups_per_island * args.samples_per_group
    divisor = args.optimizer_steps * args.gpus_per_island
    if samples % divisor:
        raise ValueError(
            "groups-per-island*samples-per-group must be divisible by "
            "optimizer-steps*gpus-per-island"
        )


def paired_prompt_streams(
    rows: list[dict[str, Any]],
    *,
    islands: int,
    groups: int,
    rounds: int,
) -> PromptStreams:
    if not rows:
        raise ValueError("RL benchmark training split is empty")
    count = islands * groups * rounds
    ids = tuple(index % len(rows) for index in range(count))
    combined = tuple(dict(rows[index]) for index in ids)
    island_ids = tuple(
        tuple(
            ids[round_id * islands * groups + island_id * groups + offset]
            for round_id in range(rounds)
            for offset in range(groups)
        )
        for island_id in range(islands)
    )
    island_rows = tuple(
        tuple(dict(rows[index]) for index in indices) for indices in island_ids
    )
    return PromptStreams(combined, island_rows, ids, island_ids)


def _normalized_prompt(row: dict[str, Any], prompt_id: int | str) -> dict[str, Any]:
    value = row.get("messages", row.get("prompt", row.get("input")))
    if isinstance(value, str):
        value = [{"role": "user", "content": value}]
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(message, dict) for message in value)
    ):
        raise ValueError("RL rows must contain messages or a string prompt/input")
    metadata = dict(row.get("metadata") or {})
    for key, item in row.items():
        if key not in {"messages", "prompt", "input", "label", "metadata", "tools"}:
            metadata.setdefault(key, item)
    metadata["benchmark_prompt_id"] = prompt_id
    output = {
        "messages": value,
        "label": row.get("label"),
        "metadata": metadata,
    }
    if "tools" in row:
        output["tools"] = row["tools"]
    return output


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    temporary.replace(path)


def write_prompt_files(
    streams: PromptStreams,
    evaluation_rows: list[dict[str, Any]],
    directory: Path,
) -> tuple[Path, tuple[Path, ...], Path]:
    combined = directory / "combined.jsonl"
    _write_jsonl(
        combined,
        (
            _normalized_prompt(row, prompt_id)
            for row, prompt_id in zip(
                streams.combined_rows, streams.combined_ids, strict=True
            )
        ),
    )
    island_paths = []
    for island_id, (rows, prompt_ids) in enumerate(
        zip(streams.island_rows, streams.island_ids, strict=True)
    ):
        path = directory / f"island-{island_id}.jsonl"
        _write_jsonl(
            path,
            (
                _normalized_prompt(row, prompt_id)
                for row, prompt_id in zip(rows, prompt_ids, strict=True)
            ),
        )
        island_paths.append(path)
    evaluation = directory / "eval.jsonl"
    _write_jsonl(
        evaluation,
        (
            _normalized_prompt(row, f"eval:{index}")
            for index, row in enumerate(evaluation_rows)
        ),
    )
    return combined, tuple(island_paths), evaluation


def verify_worker_inputs(
    *,
    prompt_path: Path,
    prompt_sha256: str,
    reward_function: str,
    reward_sha256: str,
) -> None:
    from yeto.provenance import python_spec_sha256

    actual_prompt = file_sha256(prompt_path)
    if actual_prompt != prompt_sha256.lower():
        raise RuntimeError(
            "prompt source SHA256 mismatch: "
            f"expected {prompt_sha256.lower()}, got {actual_prompt}"
        )
    actual_reward = python_spec_sha256(reward_function, base_dir=REPO_ROOT)
    if actual_reward != reward_sha256.lower():
        raise RuntimeError(
            "reward source SHA256 mismatch: "
            f"expected {reward_sha256.lower()}, got {actual_reward}"
        )


def worker_specs(
    arm: Arm,
    combined_prompt_path: Path,
    island_prompt_paths: tuple[Path, ...],
) -> list[WorkerSpec]:
    if arm.kind not in {"federated", "decoupled"}:
        return [
            WorkerSpec(
                0,
                arm.gpus_per_island,
                arm.groups_per_round,
                combined_prompt_path,
                arm.kind != "native",
            )
        ]
    if len(island_prompt_paths) != arm.islands:
        raise ValueError("federated prompt file count does not match its roster")
    return [
        WorkerSpec(
            learner_id,
            arm.gpus_per_island,
            arm.groups_per_round,
            island_prompt_paths[learner_id],
            True,
        )
        for learner_id in range(arm.islands)
    ]


def syncer_command(
    arm: Arm,
    port: int,
    run_dir: Path,
    *,
    rounds: int,
    resume_from_step: int | None = None,
) -> list[str]:
    if resume_from_step is not None and arm.kind != "decoupled":
        raise ValueError("only decoupled benchmark consolidation can resume")
    consolidation = resume_from_step is not None
    total_steps = (
        resume_from_step + arm.fragments
        if consolidation
        else rounds * arm.fragments
    )
    command = [
        str(SYNCER_BIN),
        "--port",
        str(port),
        "--learners",
        str(arm.islands),
        "--quorum",
        str(arm.islands),
        "--grace-ms",
        "0",
        "--pipeline",
        str(1 if consolidation else arm.pipeline),
        "--sync-interval-steps",
        str(
            0
            if consolidation or arm.kind != "decoupled"
            else arm.local_horizon
        ),
        "--delta-correction",
        "none",
        "--total-steps",
        str(total_steps),
        "--outer-lr",
        "0.7" if arm.kind == "decoupled" else "1",
        "--outer-momentum",
        "0.9" if arm.kind == "decoupled" else "0",
        "--max-base-lag",
        "0",
        "--learner-weight",
        "equal",
        "--checkpoint-path",
        str(run_dir / "state.ckpt"),
        "--checkpoint-every",
        "1",
        "--event-tape",
        str(run_dir / "syncer.jsonl"),
    ]
    if arm.kind == "decoupled":
        if consolidation:
            command.extend(("--resume", "--mark-final-checkpoint"))
        else:
            command.extend(("--learner-budget-steps", str(rounds)))
    else:
        command.append("--resume")
    return command


def miles_extra_argv(worker: WorkerSpec, run_dir: Path, rounds: int) -> list[str]:
    values = [
        "--save-debug-rollout-data",
        str(run_dir / "rollouts" / f"island-{worker.learner_id}" / "{rollout_id}.pt"),
    ]
    if not worker.policy_sync:
        values.extend(
            (
                "--save",
                str(run_dir / "native-checkpoint"),
                "--save-interval",
                str(rounds),
            )
        )
    return values


def canonical_native_adapter_tensors(tensors, specs) -> dict[str, Any]:
    expected = {spec.name: tuple(spec.shape) for spec in specs}
    mapped = {}
    for name, tensor in tensors.items():
        canonical_name = name if name in expected else f"base_model.model.{name}"
        if (
            canonical_name not in expected
            or canonical_name in mapped
            or tuple(tensor.shape) != expected[canonical_name]
        ):
            raise RuntimeError("native Miles adapter does not match the PEFT contract")
        mapped[canonical_name] = tensor
    if set(mapped) != set(expected):
        raise RuntimeError("native Miles adapter does not match the PEFT contract")
    return {name: mapped[name] for name in expected}


def standardize_native_adapter(args, source: Path, output: Path) -> Path:
    import torch

    from yeto.rl.core import (
        canonical_layout_hash,
        canonical_lora_config_hash,
        canonical_state,
    )
    from yeto.rl.export import (
        adapter_targets,
        derive_peft_lora_specs,
        write_peft_adapter,
    )

    specs = derive_peft_lora_specs(
        args.model,
        args.model_revision,
        rank=args.lora_r,
        targets=args.lora_targets,
        trust_remote_code=args.trust_remote_code,
    )
    raw = torch.load(
        source / "adapter_model.bin",
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(raw, dict):
        raise TypeError("native Miles adapter does not contain a tensor mapping")
    targets = adapter_targets(specs)
    state = canonical_state(
        args.global_rounds,
        canonical_native_adapter_tensors(raw, specs),
        base_model_revision=args.model_revision,
        lora_config_hash=canonical_lora_config_hash(
            rank=args.lora_r,
            target_modules=targets,
        ),
        layout_hash=canonical_layout_hash(specs),
        expected_specs=specs,
    )
    write_peft_adapter(
        state,
        output,
        base_model=args.model,
        model_revision=args.model_revision,
        rank=args.lora_r,
    )
    return output


def worker_payload(
    args,
    worker: WorkerSpec,
    *,
    arm: Arm,
    run_dir: Path,
    model_path: Path,
    syncer: str | None,
    reward_sha256: str,
) -> dict[str, Any]:
    worker_dir = run_dir / f"island-{worker.learner_id}"
    miles_port_base = args.miles_port_base + worker.learner_id * _MILES_PORT_STRIDE
    values = {
        "model": args.model,
        "model_revision": args.model_revision,
        "data": args.data,
        "data_revision": file_sha256(worker.prompt_path),
        "syncer": syncer,
        "learner_id": worker.learner_id,
        "reward_function": args.reward_function,
        "reward_sha256": reward_sha256,
        "source_sha256": getattr(args, "source_sha256", None)
        or source_tree_sha256(),
        "global_rounds": args.global_rounds,
        "groups_per_round": worker.groups_per_round,
        "samples_per_group": args.samples_per_group,
        "over_sampling_batch_size": worker.groups_per_round,
        "optimizer_steps": args.optimizer_steps,
        "rollout_max_response_len": args.rollout_max_response_len,
        "apply_chat_template_kwargs": args.apply_chat_template_kwargs,
        "custom_generate_function_path": None,
        "use_session_server": False,
        "session_server_ip": None,
        "session_server_port": None,
        "tito_model": None,
        "completed_groups_path": str(worker_dir / "completed-groups.pt"),
        "event_tape": str(worker_dir / "events.jsonl"),
        "audit_dir": str(worker_dir / "audit"),
        "actor_num_nodes": 1,
        "actor_num_gpus_per_node": worker.gpus,
        "expert_parallel": args.expert_parallel,
        "lora_r": args.lora_r,
        "lora_targets": args.lora_targets,
        "inner_lr": args.inner_lr,
        "seq_len": args.seq_len,
        "seed": args._active_seed,
        "rollout_seed": args._active_seed,
        "rollout_engine_base_port": miles_port_base + 100,
        "sglang_router_port": miles_port_base,
        "sglang_router_prometheus_port": miles_port_base + 1,
        "train_master_base_port": miles_port_base + 2,
        "wan_streams": args.wan_streams,
        "miles_root": str(args.miles_root.expanduser().resolve()),
        "trust_remote_code": args.trust_remote_code,
    }
    if arm.kind == "decoupled":
        values.update(
            sync_preset="decoupled",
            fragments=arm.fragments,
            pipeline=arm.pipeline,
            local_horizon=arm.local_horizon,
            total_fragment_steps=args.global_rounds * arm.fragments,
            learner_budget_steps=args.global_rounds,
        )
    return {
        "arguments": values,
        "model_path": str(model_path),
        "prompt_path": str(worker.prompt_path),
        "policy_sync": worker.policy_sync,
        "extra_argv": miles_extra_argv(worker, run_dir, args.global_rounds),
    }


def run_training_worker(config_path: Path) -> int:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    args = SimpleNamespace(**payload["arguments"])
    miles_root = str(Path(args.miles_root).expanduser().resolve())
    for path in (miles_root, str(REPO_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)

    verify_worker_inputs(
        prompt_path=Path(payload["prompt_path"]),
        prompt_sha256=args.data_revision,
        reward_function=args.reward_function,
        reward_sha256=args.reward_sha256,
    )

    from miles.utils.misc import load_function

    from yeto.rl.learner import _miles_callable, run_miles

    load_function(_miles_callable(args.reward_function))
    run_miles(
        args,
        model_path=payload["model_path"],
        prompt_path=payload["prompt_path"],
        yeto_policy_sync=bool(payload["policy_sync"]),
        extra_argv=tuple(payload["extra_argv"]),
    )
    return 0


def summarize_rollouts(
    rollout_paths: tuple[tuple[Path, ...], ...],
    *,
    expected_prompt_ids: tuple[tuple[int, ...], ...],
    samples_per_group: int,
) -> dict[str, Any]:
    import torch

    if len(rollout_paths) != len(expected_prompt_ids):
        raise RuntimeError("rollout island count does not match prompt manifest")
    rewards = []
    action_tokens = 0
    trajectories = 0
    truncated = 0
    for island_paths, expected in zip(rollout_paths, expected_prompt_ids, strict=True):
        observed = []
        for path in island_paths:
            # Miles' own capture contains NumPy routing arrays and is generated
            # inside this run, so use its trusted debug-data load semantics.
            payload = torch.load(path, map_location="cpu", weights_only=False)
            samples = payload.get("samples") if isinstance(payload, dict) else None
            if not isinstance(samples, list) or len(samples) % samples_per_group:
                raise RuntimeError(f"invalid Miles rollout capture: {path}")
            for start in range(0, len(samples), samples_per_group):
                group = samples[start : start + samples_per_group]
                prompt_ids = {
                    (sample.get("metadata") or {}).get("benchmark_prompt_id")
                    for sample in group
                    if isinstance(sample, dict)
                }
                if len(prompt_ids) != 1:
                    raise RuntimeError("Miles rollout group lost prompt identity")
                observed.append(prompt_ids.pop())
                for sample in group:
                    try:
                        reward = float(sample["reward"])
                        length = int(sample["response_length"])
                    except (KeyError, TypeError, ValueError) as exc:
                        raise RuntimeError(
                            "Miles rollout capture lacks scalar work metrics"
                        ) from exc
                    if length < 0 or not math.isfinite(reward):
                        raise RuntimeError(
                            "Miles rollout capture has invalid work metrics"
                        )
                    status = sample.get("status")
                    if status not in {"completed", "truncated"}:
                        raise RuntimeError(
                            f"Miles rollout capture has invalid status {status!r}"
                        )
                    rewards.append(reward)
                    action_tokens += length
                    trajectories += 1
                    truncated += status == "truncated"
        if tuple(observed) != tuple(expected):
            raise RuntimeError(
                f"prompt stream mismatch: expected {tuple(expected)}, got {tuple(observed)}"
            )
    return {
        "prompt_groups": trajectories // samples_per_group,
        "trajectories": trajectories,
        "action_tokens": action_tokens,
        "reward_mean": statistics.fmean(rewards),
        "reward_std": statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
        "truncated_trajectories": truncated,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON objects in {path}")
            rows.append(value)
    return rows


def prepare_evaluation_prompt(
    tokenizer,
    row: dict[str, Any],
    *,
    max_prompt_tokens: int,
    device: str,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    import torch
    from miles.utils.chat_template_utils import apply_chat_template

    prompt = apply_chat_template(
        row["messages"],
        tokenizer=tokenizer,
        tools=row.get("tools"),
        tokenize=False,
        add_generation_prompt=True,
        **(chat_template_kwargs or {}),
    )
    encoded = dict(
        tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )
    )
    input_ids = encoded.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise RuntimeError("tokenizer did not return batched input_ids")
    prompt_tokens = input_ids.shape[-1]
    for name, value in encoded.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"tokenizer returned non-tensor field {name!r}")
        if value.ndim >= 2 and value.shape[-1] == prompt_tokens:
            value = value[..., -max_prompt_tokens:]
        encoded[name] = value.to(device)
    if "attention_mask" not in encoded:
        encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])
    return prompt, encoded


def generation_pad_token_id(tokenizer) -> int | None:
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    eos = tokenizer.eos_token_id
    if isinstance(eos, (list, tuple)):
        return int(eos[0]) if eos else None
    return int(eos) if eos is not None else None


async def evaluate_rewards(args, samples: list[Any]) -> list[Any]:
    from miles.rollout.rm_hub import async_rm

    return list(await asyncio.gather(*(async_rm(args, sample) for sample in samples)))


def load_peft_adapter(peft_model, model, adapter_path):
    message = "Found missing adapter keys while loading the checkpoint:"
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=message)
        try:
            return peft_model.from_pretrained(model, adapter_path)
        except UserWarning as exc:
            if not str(exc).startswith(message):
                raise
            raise RuntimeError(str(exc)) from exc


def run_evaluation_worker(config_path: Path) -> int:
    started = time.monotonic()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    miles_root = str(Path(payload["miles_root"]).expanduser().resolve())
    for path in (miles_root, str(REPO_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)

    import torch
    from miles.utils.types import Sample
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from yeto.rl.export import _rl_model_factory
    from yeto.rl.learner import _miles_callable

    verify_worker_inputs(
        prompt_path=Path(payload["eval_path"]),
        prompt_sha256=payload["eval_sha256"],
        reward_function=payload["reward_function"],
        reward_sha256=payload["reward_sha256"],
    )

    device = payload["device"]
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(
        payload["model_path"],
        trust_remote_code=payload["trust_remote_code"],
    )
    config = AutoConfig.from_pretrained(
        payload["model_path"],
        trust_remote_code=payload["trust_remote_code"],
    )
    model_factory = _rl_model_factory(config)
    model_kwargs = {
        "config": config,
        "dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if model_factory is AutoModelForCausalLM:
        model_kwargs["trust_remote_code"] = payload["trust_remote_code"]
    model = model_factory.from_pretrained(
        payload["model_path"],
        **model_kwargs,
    )
    model = load_peft_adapter(PeftModel, model, payload["adapter_path"])
    model.to(device)
    model.eval()

    rows = _read_jsonl(Path(payload["eval_path"]))
    max_response = int(payload["max_response_len"])
    max_prompt = int(payload["seq_len"]) - max_response
    reward_args = SimpleNamespace(
        **payload["reward_arguments"],
        custom_rm_path=_miles_callable(payload["reward_function"]),
        multi_lora=False,
    )
    samples = []
    sample_records = []
    with torch.no_grad():
        for prompt_index, row in enumerate(rows):
            prompt, encoded = prepare_evaluation_prompt(
                tokenizer,
                row,
                max_prompt_tokens=max_prompt,
                device=device,
                chat_template_kwargs=payload["apply_chat_template_kwargs"],
            )
            prompt_tokens = encoded["input_ids"][0].tolist()
            for sample_index in range(payload["samples_per_prompt"]):
                generation_seed = (
                    int(payload["seed"])
                    + prompt_index * payload["samples_per_prompt"]
                    + sample_index
                )
                torch.manual_seed(generation_seed)
                if device != "cpu":
                    torch.cuda.manual_seed_all(generation_seed)
                generation = {
                    "max_new_tokens": max_response,
                    "do_sample": payload["temperature"] > 0,
                    "pad_token_id": generation_pad_token_id(tokenizer),
                }
                if generation["do_sample"]:
                    generation.update(
                        temperature=payload["temperature"],
                        top_p=payload["top_p"],
                    )
                output = model.generate(**encoded, **generation)
                response_tokens = output[0, encoded["input_ids"].shape[1] :].tolist()
                eos_ids = tokenizer.eos_token_id
                eos_ids = set(
                    eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids]
                )
                eos_ids.discard(None)
                truncated = len(response_tokens) == max_response and (
                    not response_tokens or response_tokens[-1] not in eos_ids
                )
                metadata = dict(row.get("metadata") or {})
                if row.get("tools") is not None:
                    metadata["tools"] = row["tools"]
                sample = Sample(
                    group_index=prompt_index,
                    index=len(samples),
                    prompt=prompt,
                    tokens=prompt_tokens + response_tokens,
                    response=tokenizer.decode(
                        response_tokens, skip_special_tokens=True
                    ),
                    response_length=len(response_tokens),
                    label=row.get("label"),
                    metadata=metadata,
                    status=(
                        Sample.Status.TRUNCATED
                        if truncated
                        else Sample.Status.COMPLETED
                    ),
                )
                samples.append(sample)
                sample_records.append(
                    {
                        "prompt_index": prompt_index,
                        "sample_index": sample_index,
                        "generation_seed": generation_seed,
                        "response": sample.response,
                        "response_tokens": sample.response_length,
                        "truncated": truncated,
                    }
                )

    rewards = asyncio.run(evaluate_rewards(reward_args, samples))
    if not isinstance(rewards, list) or len(rewards) != len(samples):
        raise RuntimeError("Miles reward callable did not return one reward per sample")
    scalar_rewards = []
    for record, sample, value in zip(sample_records, samples, rewards, strict=True):
        try:
            reward = float(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "RL benchmark requires scalar evaluation rewards"
            ) from exc
        if not math.isfinite(reward):
            raise RuntimeError("evaluation reward contains NaN or Inf")
        sample.reward = reward
        record["reward"] = reward
        scalar_rewards.append(reward)

    grouped = [
        scalar_rewards[index : index + payload["samples_per_prompt"]]
        for index in range(0, len(scalar_rewards), payload["samples_per_prompt"])
    ]
    summary = summarize_rewards(
        grouped,
        pass_ks=tuple(payload["pass_ks"]),
        threshold=float(payload["pass_threshold"]),
    )
    summary.update(
        {
            "prompts": len(rows),
            "samples": len(samples),
            "response_tokens": sum(sample.response_length for sample in samples),
            "truncated_samples": sum(
                sample.status == Sample.Status.TRUNCATED for sample in samples
            ),
            "wall_s": time.monotonic() - started,
        }
    )
    result_path = Path(payload["result_path"])
    write_json_atomic(result_path, summary)
    _write_jsonl(Path(payload["samples_path"]), sample_records)
    return 0


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _tail(path: Path, lines: int = 40) -> str:
    if not path.exists():
        return ""
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    )


def _stop_process(process: subprocess.Popen, timeout: int = 20) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            process.kill()
        process.wait(timeout=10)


def _visible_cuda_devices() -> list[str] | None:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or not raw.strip():
        return None
    return [value.strip() for value in raw.split(",") if value.strip()]


def _visible_gpu_uuids() -> set[str] | None:
    visible = _visible_cuda_devices()
    if visible is None:
        return None
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    index_to_uuid = {}
    for line in result.stdout.splitlines():
        parts = [value.strip() for value in line.split(",")]
        if len(parts) >= 2:
            index_to_uuid[parts[0]] = parts[1]
    return {index_to_uuid.get(value, value) for value in visible}


def summarize_gpu_samples(
    samples: list[tuple[float, dict[str, float]]],
    *,
    started: float,
    ended: float,
    expected_gpus: int,
) -> dict[str, float]:
    if ended <= started or expected_gpus < 1 or not samples:
        raise RuntimeError("GPU activity sampling produced no usable interval")
    gpu_ids = tuple(samples[0][1])
    if len(gpu_ids) != expected_gpus or any(
        tuple(values) != gpu_ids for _, values in samples
    ):
        raise RuntimeError("GPU activity samples do not match the benchmark allocation")
    active = {gpu_id: 0.0 for gpu_id in gpu_ids}
    utilization = {gpu_id: 0.0 for gpu_id in gpu_ids}
    for index, (timestamp, values) in enumerate(samples):
        interval_start = started if index == 0 else timestamp
        interval_end = samples[index + 1][0] if index + 1 < len(samples) else ended
        seconds = max(0.0, min(ended, interval_end) - max(started, interval_start))
        for gpu_id, value in values.items():
            if not math.isfinite(value) or not 0 <= value <= 100:
                raise RuntimeError("nvidia-smi returned invalid GPU utilization")
            utilization[gpu_id] += seconds * value
            if value > 0:
                active[gpu_id] += seconds
    wall = ended - started
    means = [utilization[gpu_id] / wall for gpu_id in gpu_ids]
    active_seconds = sum(active.values())
    return {
        "gpu_active_seconds": active_seconds,
        "gpu_active_fraction": active_seconds / (wall * expected_gpus),
        "mean_gpu_utilization": statistics.fmean(means),
        "min_gpu_utilization": min(means),
    }


class _GpuSampler:
    def __init__(self, expected_gpus: int, interval_s: float = 1.0) -> None:
        self.expected_gpus = expected_gpus
        self.interval_s = interval_s
        self.visible = _visible_gpu_uuids()
        self.samples: list[tuple[float, dict[str, float]]] = []
        self.error: BaseException | None = None
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def _sample(self) -> None:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
        values = []
        for line in result.stdout.splitlines():
            parts = [value.strip() for value in line.split(",")]
            if len(parts) != 2 or (self.visible is not None and parts[0] not in self.visible):
                continue
            values.append((parts[0], float(parts[1])))
        if len(values) < self.expected_gpus:
            raise RuntimeError("nvidia-smi exposed fewer GPUs than the benchmark arm")
        self.samples.append((time.monotonic(), dict(values[: self.expected_gpus])))

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval_s):
            try:
                self._sample()
            except (OSError, RuntimeError, ValueError) as error:
                self.error = error
                return

    def start(self) -> None:
        self._sample()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()


def wait_for_free_gpus(limit_mb: int = 2000, timeout_s: int = 300) -> None:
    visible_uuids = _visible_gpu_uuids()
    deadline = time.monotonic() + timeout_s
    last = ""
    while True:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
        holders = []
        for line in result.stdout.splitlines():
            parts = [value.strip() for value in line.split(",")]
            if len(parts) < 4:
                continue
            gpu_uuid, pid, name, memory = parts[0], parts[1], parts[2], parts[-1]
            if visible_uuids is not None and gpu_uuid not in visible_uuids:
                continue
            if not memory.isdigit() or int(memory) > limit_mb:
                holders.append(f"pid {pid} ({name}): {memory} MiB")
        if not holders:
            return
        current = "; ".join(holders)
        if current != last:
            print(f"[rl-benchmark] waiting for GPUs: {current}", flush=True)
            last = current
        if time.monotonic() >= deadline:
            raise RuntimeError(f"GPUs still occupied after {timeout_s}s: {current}")
        time.sleep(3)


@contextmanager
def local_ray_cluster(total_gpus: int):
    if os.environ.get("RAY_ADDRESS"):
        raise RuntimeError("unset RAY_ADDRESS before running the local RL benchmark")
    import ray

    if ray.is_initialized():
        raise RuntimeError("RL benchmark requires no pre-existing Ray connection")
    context = ray.init(
        num_gpus=total_gpus,
        include_dashboard=True,
        logging_level="ERROR",
    )
    try:
        yield context.address_info["address"]
    finally:
        ray.shutdown()


def _wait_for_port(
    port: int, process: subprocess.Popen, log: Path, timeout_s: int = 30
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"syncer exited before listening with code {process.returncode}:\n{_tail(log)}"
            )
        with socket.socket() as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise RuntimeError(f"syncer did not listen on port {port}:\n{_tail(log)}")


def _wait_for_training(
    processes: list[subprocess.Popen],
    logs: list[Path],
    *,
    syncer: subprocess.Popen | None,
    syncer_log: Path | None,
    timeout_s: int,
) -> None:
    deadline = time.monotonic() + timeout_s
    pending = set(range(len(processes)))
    while pending:
        for index in list(pending):
            returncode = processes[index].poll()
            if returncode is None:
                continue
            pending.remove(index)
            if returncode != 0:
                raise RuntimeError(
                    f"Miles island {index} failed with code {returncode}:\n{_tail(logs[index])}"
                )
        if syncer is not None and syncer.poll() not in (None, 0):
            raise RuntimeError(
                f"syncer failed with code {syncer.returncode}:\n"
                f"{_tail(syncer_log) if syncer_log else ''}"
            )
        if time.monotonic() >= deadline:
            raise RuntimeError(f"RL benchmark arm timed out after {timeout_s}s")
        if pending:
            time.sleep(1)
    if syncer is not None:
        try:
            returncode = syncer.wait(timeout=60)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("syncer did not finish after all Miles islands") from exc
        if returncode != 0:
            raise RuntimeError(
                f"syncer failed with code {returncode}:\n"
                f"{_tail(syncer_log) if syncer_log else ''}"
            )


def _wait_for_budget_cutoff(
    syncer: subprocess.Popen,
    syncer_log: Path,
    learners: list[subprocess.Popen],
    learner_logs: list[Path],
    *,
    timeout_s: int,
) -> None:
    deadline = time.monotonic() + timeout_s
    while syncer.poll() is None:
        for index, learner in enumerate(learners):
            returncode = learner.poll()
            if returncode is not None:
                raise RuntimeError(
                    f"Miles island {index} exited before the budget cutoff with code "
                    f"{returncode}:\n{_tail(learner_logs[index])}"
                )
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"RL benchmark budget cutoff timed out after {timeout_s}s:\n"
                f"{_tail(syncer_log)}"
            )
        time.sleep(1)
    if syncer.returncode != 0:
        raise RuntimeError(
            f"syncer failed during budget cutoff with code {syncer.returncode}:\n"
            f"{_tail(syncer_log)}"
        )


def _training_environment(ray_address: str, miles_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["RAY_ADDRESS"] = ray_address
    existing = env.get("PYTHONPATH")
    paths = [str(miles_root), str(REPO_ROOT)]
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["NVTE_FLASH_ATTN"] = "0"
    env["NVTE_FUSED_ATTN"] = "0"
    env["NVTE_UNFUSED_ATTN"] = "1"
    return env


def _run_training_processes(
    args,
    arm: Arm,
    workers: list[WorkerSpec],
    *,
    run_dir: Path,
    model_path: Path,
    reward_sha256: str,
) -> tuple[float, dict[str, float]]:
    port = _free_port() if arm.kind != "native" else None
    syncer_address = f"127.0.0.1:{port}" if port is not None else None
    payload_paths = []
    for worker in workers:
        path = run_dir / f"island-{worker.learner_id}" / "worker.json"
        write_json_atomic(
            path,
            worker_payload(
                args,
                worker,
                arm=arm,
                run_dir=run_dir,
                model_path=model_path,
                syncer=syncer_address,
                reward_sha256=reward_sha256,
            ),
        )
        payload_paths.append(path)

    syncer = None
    syncer_handle = None
    syncer_log = run_dir / "syncer.log"
    processes = []
    handles = []
    logs = []
    started = time.monotonic()
    sampler = _GpuSampler(arm.islands * arm.gpus_per_island)
    sampler.start()
    try:
        with local_ray_cluster(
            arm.islands * arm.gpus_per_island,
        ) as ray_address:
            if port is not None:
                syncer_handle = syncer_log.open("w", encoding="utf-8")
                syncer = subprocess.Popen(
                    syncer_command(
                        arm,
                        port,
                        run_dir,
                        rounds=args.global_rounds,
                    ),
                    cwd=REPO_ROOT,
                    stdout=syncer_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                _wait_for_port(port, syncer, syncer_log)
            env = _training_environment(ray_address, args.miles_root)
            for worker, payload_path in zip(workers, payload_paths, strict=True):
                log = run_dir / f"island-{worker.learner_id}" / "miles.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                handle = log.open("w", encoding="utf-8")
                handles.append(handle)
                logs.append(log)
                processes.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "_train-worker",
                            str(payload_path),
                        ],
                        cwd=REPO_ROOT,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        env=env,
                        start_new_session=True,
                    )
                )
            timeout_s = args.arm_timeout_min * 60
            if arm.kind == "decoupled":
                _wait_for_budget_cutoff(
                    syncer,
                    syncer_log,
                    processes,
                    logs,
                    timeout_s=timeout_s,
                )
                from yeto.final_marker import read_checkpoint_global_step

                cutoff_step = read_checkpoint_global_step(run_dir / "state.ckpt")
                syncer = subprocess.Popen(
                    syncer_command(
                        arm,
                        port,
                        run_dir,
                        rounds=args.global_rounds,
                        resume_from_step=cutoff_step,
                    ),
                    cwd=REPO_ROOT,
                    stdout=syncer_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                _wait_for_port(port, syncer, syncer_log)
                _wait_for_training(
                    processes,
                    logs,
                    syncer=syncer,
                    syncer_log=syncer_log,
                    timeout_s=timeout_s,
                )
                from yeto.budget_finalization import validate_consolidation_tape

                validate_consolidation_tape(
                    run_dir / "syncer.jsonl",
                    cutoff_step=cutoff_step,
                    fragments=arm.fragments,
                    learners=arm.islands,
                    budget_steps=args.global_rounds,
                )
            else:
                _wait_for_training(
                    processes,
                    logs,
                    syncer=syncer,
                    syncer_log=syncer_log,
                    timeout_s=timeout_s,
                )
    finally:
        for process in processes:
            _stop_process(process)
        if syncer is not None:
            _stop_process(syncer)
        for handle in handles:
            handle.close()
        if syncer_handle is not None:
            syncer_handle.close()
        sampler.stop()
    ended = time.monotonic()
    if sampler.error is not None:
        raise RuntimeError("GPU activity sampling failed") from sampler.error
    return ended - started, summarize_gpu_samples(
        sampler.samples,
        started=started,
        ended=ended,
        expected_gpus=arm.islands * arm.gpus_per_island,
    )


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summarize_yeto_events(run_dir: Path, islands: int) -> dict[str, Any]:
    local_rounds = []
    apply_events = []
    fragment_pushes = []
    fragment_broadcasts = []
    final_cuts = []
    hook_events = []
    policy_snapshots = []
    for island_id in range(islands):
        path = run_dir / f"island-{island_id}" / "events.jsonl"
        if not path.exists():
            continue
        for event in _read_jsonl(path):
            if event.get("event") == "rl_local_round":
                local_rounds.append(event)
            elif event.get("event") == "rl_policy_apply":
                apply_events.append(event)
            elif event.get("event") == "rl_fragment_push":
                fragment_pushes.append(event)
            elif event.get("event") == "rl_fragment_bcast":
                fragment_broadcasts.append(event)
            elif event.get("event") == "rl_final_cut":
                final_cuts.append(event)
            elif event.get("event") == "rl_sync_hook":
                hook_events.append(event)
            elif event.get("event") == "rl_policy_snapshot":
                policy_snapshots.append(event)
    sync_records = (
        _read_jsonl(run_dir / "syncer.jsonl")
        if (run_dir / "syncer.jsonl").exists()
        else []
    )

    def values(name: str) -> list[float]:
        return [
            float(event[name]) for event in local_rounds if event.get(name) is not None
        ]

    def event_values(events: list[dict[str, Any]], name: str) -> list[float]:
        return [float(event[name]) for event in events if event.get(name) is not None]

    sent_values = values("sync/fragment_payload_bytes_sent")
    received_values = values("sync/fragment_payload_bytes_received") + event_values(
        final_cuts, "sync/fragment_payload_bytes_received"
    )
    sent = int(sum(sent_values)) if sent_values else None
    received = int(sum(received_values)) if received_values else None
    legacy_sent = int(sum(values("sync/bytes_sent")))
    legacy_received = int(sum(values("sync/bytes_received")))
    hook_seconds = event_values(hook_events, "sync/hook_seconds")
    finalization_seconds = event_values(
        [event for event in hook_events if event.get("sync/finalization")],
        "sync/hook_seconds",
    )
    responders = [len(record.get("responders", [])) for record in sync_records]

    return {
        "local_rounds": len(local_rounds),
        "policy_applies": len(apply_events),
        "policy_snapshots": len(policy_snapshots),
        "in_process_applies": sum(
            bool(event.get("partial_fragment_apply")) for event in apply_events
        ),
        "rollout_s": sum(values("rollout_seconds")),
        "optimizer_train_s": sum(values("train_seconds")),
        "hook_s": sum(hook_seconds),
        "finalization_s": sum(finalization_seconds),
        "remote_quorum_wait_s": sum(
            event_values(hook_events, "sync/remote_quorum_wait_seconds")
        ),
        "mean_kl": _mean(values("mean_kl")),
        "mean_ess_ratio": _mean(values("ess_ratio")),
        "mean_clip_fraction": _mean(values("clip_fraction")),
        "sync_bytes_sent": legacy_sent,
        "sync_bytes_received": legacy_received,
        "fragment_payload_bytes_sent": sent,
        "fragment_payload_bytes_received": received,
        "fragment_payload_traffic_bytes": (
            None if sent is None and received is None else (sent or 0) + (received or 0)
        ),
        "mean_realized_h": _mean(event_values(fragment_pushes, "realized_h")),
        "mean_pull_to_push_s": _mean(
            event_values(fragment_pushes, "pull_to_push_seconds")
        ),
        "mean_bcast_queue_s": _mean(
            event_values(fragment_broadcasts, "queue_seconds")
        ),
        "mean_bcast_apply_s": _mean(
            event_values(
                [
                    event
                    for event in apply_events
                    if event.get("partial_fragment_apply")
                ],
                "sync/apply_seconds",
            )
        ),
        "mean_responders": _mean([float(value) for value in responders]),
        "mean_sync_ms": _mean(
            [
                float(event["ms"])
                for event in sync_records
                if event.get("ms") is not None
            ]
        ),
    }


def ensure_syncer() -> None:
    subprocess.run(
        ["cargo", "build", "--release", "--locked", "--quiet"],
        cwd=REPO_ROOT / "syncer",
        check=True,
    )


def resolve_model_path(args) -> Path:
    from huggingface_hub import snapshot_download

    from yeto.models import resolve

    model = resolve(args.model)
    args.model = model
    return Path(snapshot_download(repo_id=model, revision=args.model_revision))


def _run_checked(
    command: list[str],
    log: Path,
    *,
    env: dict[str, str] | None = None,
    timeout_s: int,
) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            _stop_process(process)
            raise RuntimeError(
                f"command timed out: {' '.join(command)}\n{_tail(log)}"
            ) from exc
        except BaseException:
            _stop_process(process)
            raise
    if returncode != 0:
        raise RuntimeError(
            f"command failed with code {returncode}: {' '.join(command)}\n{_tail(log)}"
        )


def evaluate_artifact(
    args,
    *,
    adapter_path: Path,
    model_path: Path,
    eval_path: Path,
    reward_sha256: str,
    run_dir: Path,
    seed: int,
) -> dict[str, Any]:
    result_path = run_dir / "eval.json"
    payload = {
        "model_path": str(model_path),
        "adapter_path": str(adapter_path),
        "eval_path": str(eval_path),
        "eval_sha256": file_sha256(eval_path),
        "reward_function": args.reward_function,
        "reward_sha256": reward_sha256,
        "reward_arguments": {
            "reward_function": args.reward_function,
            "n_samples_per_prompt": args.samples_per_group,
            "rollout_batch_size": args.groups_per_island,
            "rollout_max_response_len": args.rollout_max_response_len,
            "seed": seed,
            "rm_type": None,
            "group_rm": False,
            "reward_key": None,
            "eval_reward_key": None,
        },
        "seq_len": args.seq_len,
        "max_response_len": args.rollout_max_response_len,
        "apply_chat_template_kwargs": args.apply_chat_template_kwargs,
        "samples_per_prompt": args.eval_samples_per_prompt,
        "pass_ks": list(args._pass_ks),
        "pass_threshold": args.pass_threshold,
        "temperature": args.eval_temperature,
        "top_p": args.eval_top_p,
        "seed": args.eval_seed + seed,
        "device": args.eval_device,
        "trust_remote_code": args.trust_remote_code,
        "miles_root": str(args.miles_root),
        "result_path": str(result_path),
        "samples_path": str(run_dir / "eval-samples.jsonl"),
    }
    config_path = run_dir / "eval-worker.json"
    write_json_atomic(config_path, payload)
    env = dict(os.environ)
    paths = [str(args.miles_root), str(REPO_ROOT)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    if args.eval_device == "cuda":
        visible = _visible_cuda_devices()
        env["CUDA_VISIBLE_DEVICES"] = (visible or ["0"])[0]
    _run_checked(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "_eval-worker",
            str(config_path),
        ],
        run_dir / "eval.log",
        env=env,
        timeout_s=args.arm_timeout_min * 60,
    )
    if not result_path.is_file():
        raise RuntimeError("evaluation worker produced no result")
    return json.loads(result_path.read_text(encoding="utf-8"))


def run_arm(
    args,
    arm: Arm,
    *,
    seed: int,
    model_path: Path,
    prompt_paths: tuple[Path, tuple[Path, ...], Path],
    streams: PromptStreams,
    reward_sha256: str,
) -> dict[str, Any]:
    run_dir = args.work_dir / f"seed-{seed}" / arm.name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    combined_path, island_paths, eval_path = prompt_paths
    workers = worker_specs(arm, combined_path, island_paths)
    args._active_seed = seed
    wait_for_free_gpus()
    train_wall_s, gpu_activity = _run_training_processes(
        args,
        arm,
        workers,
        run_dir=run_dir,
        model_path=model_path,
        reward_sha256=reward_sha256,
    )

    artifact_s = None
    if arm.kind == "native":
        native_adapter = (
            run_dir
            / "native-checkpoint"
            / f"iter_{args.global_rounds - 1:07d}"
            / "adapter"
        )
        export_started = time.monotonic()
        adapter_path = standardize_native_adapter(
            args,
            native_adapter,
            run_dir / "adapter",
        )
        artifact_s = time.monotonic() - export_started
    else:
        checkpoint = run_dir / "state.ckpt"
        adapter_path = run_dir / "adapter"
        export_started = time.monotonic()
        from yeto.rl.export import export_rl_checkpoint

        expected_version = args.global_rounds
        if arm.kind == "decoupled":
            from yeto.final_marker import validate_final_checkpoint

            expected_version = validate_final_checkpoint(checkpoint)
        state = export_rl_checkpoint(
            checkpoint,
            adapter_path,
            model=args.model,
            model_revision=args.model_revision,
            rank=args.lora_r,
            lora_targets=args.lora_targets,
            trust_remote_code=args.trust_remote_code,
            sync_preset=(
                "decoupled" if arm.kind == "decoupled" else "strict-avg"
            ),
            fragments=arm.fragments,
            pipeline=arm.pipeline,
            local_horizon=arm.local_horizon,
            benchmark_learner_budget_steps=(
                args.global_rounds if arm.kind == "decoupled" else None
            ),
        )
        artifact_s = time.monotonic() - export_started
        if state.policy_version != expected_version:
            raise RuntimeError(
                f"authoritative RL checkpoint ended at v{state.policy_version}, "
                f"expected v{expected_version}"
            )
    if not (adapter_path / "adapter_config.json").is_file() or not any(
        (adapter_path / name).is_file()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise RuntimeError(f"arm produced no standard PEFT adapter: {adapter_path}")

    rollout_paths = tuple(
        tuple(
            run_dir / "rollouts" / f"island-{worker.learner_id}" / f"{round_id}.pt"
            for round_id in range(args.global_rounds)
        )
        for worker in workers
    )
    expected_ids = (
        streams.island_ids
        if arm.kind in {"federated", "decoupled"}
        else (streams.combined_ids,)
    )
    training = summarize_rollouts(
        rollout_paths,
        expected_prompt_ids=expected_ids,
        samples_per_group=args.samples_per_group,
    )
    expected_work = workload(
        arm,
        rounds=args.global_rounds,
        samples_per_group=args.samples_per_group,
    )
    if (
        training["prompt_groups"] != expected_work["prompt_groups"]
        or training["trajectories"] != expected_work["trajectories"]
    ):
        raise RuntimeError("Miles completed work does not match the paired budget")
    sync = None if arm.kind == "native" else summarize_yeto_events(run_dir, arm.islands)
    if sync is not None and sync["local_rounds"] != arm.islands * args.global_rounds:
        raise RuntimeError("Yeto event tape is missing a local RL round")

    wait_for_free_gpus()
    evaluation = evaluate_artifact(
        args,
        adapter_path=adapter_path,
        model_path=model_path,
        eval_path=eval_path,
        reward_sha256=reward_sha256,
        run_dir=run_dir,
        seed=seed,
    )
    total_gpus = arm.islands * arm.gpus_per_island
    gpu_hours = (
        total_gpus * train_wall_s
        + (evaluation["wall_s"] if args.eval_device == "cuda" else 0.0)
    ) / 3600.0
    return {
        "arm": arm.name,
        "kind": arm.kind,
        "m": arm.benchmark_islands,
        "seed": seed,
        "islands": arm.islands,
        "gpus_per_island": arm.gpus_per_island,
        "total_gpus": total_gpus,
        "train_wall_s": train_wall_s,
        "artifact_s": artifact_s,
        "artifact_ready_s": train_wall_s + (artifact_s or 0.0),
        "gpu_hours": gpu_hours,
        "gpu_activity": gpu_activity,
        "estimated_cost": (
            gpu_hours * args.gpu_hour_cost if args.gpu_hour_cost is not None else None
        ),
        "training": training,
        "sync": sync,
        "eval": evaluation,
    }


def summarize_rewards(
    grouped_rewards: list[list[float]],
    *,
    pass_ks: tuple[int, ...],
    threshold: float,
) -> dict[str, Any]:
    if not grouped_rewards or any(not group for group in grouped_rewards):
        raise ValueError("reward groups must be non-empty")
    group_size = len(grouped_rewards[0])
    if any(len(group) != group_size for group in grouped_rewards):
        raise ValueError("reward groups must have one fixed sample count")
    if any(k <= 0 or k > group_size for k in pass_ks):
        raise ValueError("pass@k must be positive and cannot exceed samples per prompt")
    values = [float(value) for group in grouped_rewards for value in group]
    pass_at_k = {}
    for k in pass_ks:
        estimates = []
        for group in grouped_rewards:
            successes = sum(value > threshold for value in group)
            failures = group_size - successes
            miss = (
                0.0
                if failures < k
                else math.comb(failures, k) / math.comb(group_size, k)
            )
            estimates.append(1.0 - miss)
        pass_at_k[str(k)] = statistics.fmean(estimates)
    return {
        "reward_mean": statistics.fmean(values),
        "reward_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "pass_at_k": pass_at_k,
    }


def expected_record_keys(
    seeds: tuple[int, ...], arms: list[Arm]
) -> set[tuple[str, int, int]]:
    return {(arm.name, arm.benchmark_islands, seed) for seed in seeds for arm in arms}


def validate_result_records(records: list[dict], expected: set[tuple]) -> None:
    seen = set()
    for record in records:
        key = (record.get("arm"), record.get("m"), record.get("seed"))
        if key not in expected:
            raise ValueError(f"results contain an unexpected record: {key}")
        if key in seen:
            raise ValueError(f"results contain a duplicate record: {key}")
        seen.add(key)


def annotate_deltas(records: list[dict]) -> list[dict]:
    rewards = {
        (record["m"], record["seed"], record["arm"]): record["eval"]["reward_mean"]
        for record in records
    }
    output = []
    for record in records:
        m, seed = record["m"], record["seed"]
        reward = record["eval"]["reward_mean"]
        native = rewards.get((m, seed, f"native-miles-m{m}"))
        single = rewards.get((m, seed, f"yeto-single-m{m}"))
        strict = rewards.get((m, seed, f"yeto-federated-m{m}"))
        row = dict(record)
        row["delta_vs_native"] = (
            None
            if record["arm"].startswith("native-") or native is None
            else reward - native
        )
        row["delta_vs_single"] = (
            reward - single
            if record["arm"].startswith("yeto-federated-") and single is not None
            else None
        )
        row["delta_vs_strict"] = (
            reward - strict
            if record["arm"].startswith("yeto-decoupled-") and strict is not None
            else None
        )
        output.append(row)
    return output


def parse_seeds(spec: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(value.strip()) for value in spec.split(",") if value.strip())
    except ValueError as exc:
        raise ValueError("--seeds must be a comma-separated list of integers") from exc
    if not seeds:
        raise ValueError("--seeds must contain at least one value")
    if len(seeds) != len(set(seeds)):
        raise ValueError("--seeds contains duplicates")
    return seeds


def validate_args(args, arms: list[Arm], *, check_runtime: bool) -> None:
    from yeto.models import resolve
    from yeto.provenance import is_immutable_commit, is_local_reference

    validate_workload(args)
    parse_seeds(args.seeds)
    if is_local_reference(resolve(args.model)):
        raise ValueError("RL benchmark does not accept a mutable local model")
    if not is_immutable_commit(args.model_revision):
        raise ValueError("--model-revision must be an immutable commit")
    data_path = Path(args.data).expanduser()
    if data_path.exists():
        if args.data_revision is not None:
            raise ValueError("--data-revision cannot be used with a local dataset")
    elif not is_immutable_commit(args.data_revision or ""):
        raise ValueError("remote --data requires an immutable --data-revision")
    module, separator, function = args.reward_function.partition(":")
    if not separator or not module or not function.isidentifier():
        raise ValueError("--reward-function must be package.module:function")
    if args.eval_prompts <= 0 or args.eval_samples_per_prompt <= 0:
        raise ValueError("evaluation prompt and sample counts must be positive")
    if args.seq_len <= args.rollout_max_response_len:
        raise ValueError("--seq-len must exceed --rollout-max-response-len")
    if not 0 <= args.eval_temperature or not 0 < args.eval_top_p <= 1:
        raise ValueError(
            "evaluation temperature must be non-negative and top-p in (0, 1]"
        )
    if args.inner_lr <= 0 or args.lora_r <= 0 or args.wan_streams <= 0:
        raise ValueError("LoRA rank, learning rate, and WAN streams must be positive")
    if args.arm_timeout_min <= 0:
        raise ValueError("--arm-timeout-min must be positive")
    if args.gpu_hour_cost is not None and args.gpu_hour_cost < 0:
        raise ValueError("--gpu-hour-cost must be non-negative")
    args._pass_ks = tuple(_positive_csv(args.pass_k, "--pass-k"))
    if max(args._pass_ks) > args.eval_samples_per_prompt:
        raise ValueError("pass@k cannot exceed --eval-samples-per-prompt")
    if args.expert_parallel is not None and (
        args.expert_parallel <= 0 or args.gpus_per_island % args.expert_parallel
    ):
        raise ValueError("--expert-parallel must divide --gpus-per-island")
    port_range_end = (
        args.miles_port_base
        + max(arm.benchmark_islands for arm in arms) * _MILES_PORT_STRIDE
        - 1
    )
    if args.miles_port_base <= 0 or port_range_end > 65535:
        raise ValueError("Miles host port ranges must fit in host port space")
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if args.work_dir.expanduser().resolve() == args.report_dir.expanduser().resolve():
        raise ValueError("--work-dir and --report-dir must differ")
    if check_runtime:
        if not args.trust_remote_code:
            raise ValueError("pinned Miles requires explicit --trust-remote-code")
        if not args.miles_root.expanduser().is_dir():
            raise ValueError(f"Miles checkout does not exist: {args.miles_root}")
        import torch

        required = max(arm.islands * arm.gpus_per_island for arm in arms)
        if torch.cuda.device_count() < required:
            raise ValueError(
                f"largest arm needs {required} GPUs, but torch sees "
                f"{torch.cuda.device_count()}"
            )


def materialize_prompt_matrix(
    args, islands: tuple[int, ...]
) -> tuple[dict[int, tuple[tuple[Path, tuple[Path, ...], Path], PromptStreams]], int]:
    from yeto.data import load_rows

    dataset = load_rows(args.data, revision=args.data_revision)
    total_rows = len(dataset)
    if total_rows <= args.eval_prompts:
        raise ValueError(
            f"prompt dataset has {total_rows} rows; need more than {args.eval_prompts}"
        )
    train_rows = total_rows - args.eval_prompts
    required = max(islands) * args.groups_per_island * args.global_rounds
    source_train = [dict(dataset[index]) for index in range(min(train_rows, required))]
    evaluation = [dict(dataset[index]) for index in range(train_rows, total_rows)]
    output = {}
    for m in islands:
        streams = paired_prompt_streams(
            source_train,
            islands=m,
            groups=args.groups_per_island,
            rounds=args.global_rounds,
        )
        paths = write_prompt_files(
            streams,
            evaluation,
            args.work_dir / "data" / f"m{m}",
        )
        output[m] = (paths, streams)
    return output, train_rows


def load_prompt_matrix(
    data_root: Path, islands: tuple[int, ...]
) -> dict[int, tuple[tuple[Path, tuple[Path, ...], Path], PromptStreams]]:
    output = {}
    for m in islands:
        directory = data_root / f"m{m}"
        combined_path = directory / "combined.jsonl"
        island_paths = tuple(directory / f"island-{index}.jsonl" for index in range(m))
        eval_path = directory / "eval.jsonl"
        combined_rows = tuple(_read_jsonl(combined_path))
        island_rows = tuple(tuple(_read_jsonl(path)) for path in island_paths)

        def ids(rows) -> tuple[int, ...]:
            return tuple(int(row["metadata"]["benchmark_prompt_id"]) for row in rows)

        streams = PromptStreams(
            combined_rows,
            island_rows,
            ids(combined_rows),
            tuple(ids(rows) for rows in island_rows),
        )
        output[m] = ((combined_path, island_paths, eval_path), streams)
    return output


def _resume_identity(args, arms: list[Arm]) -> dict[str, Any]:
    from yeto.benchmark_resume import implementation_fingerprint, jsonable_arguments
    from yeto.rl import MILES_COMMIT

    return {
        "format_version": 1,
        "benchmark": "miles-rl-lm",
        "arguments": jsonable_arguments(args, exclude=_RESUME_EXCLUDES),
        "arms": [asdict(arm) for arm in arms],
        "miles_commit": MILES_COMMIT,
        "implementation_sha256": implementation_fingerprint(
            REPO_ROOT,
            _IMPLEMENTATION_PATHS,
        ),
    }


def write_run_config(
    args,
    arms: list[Arm],
    *,
    data_manifest: dict[str, Any],
) -> None:
    from yeto.benchmark_resume import jsonable_arguments
    from yeto.rl import MILES_COMMIT

    config = {
        "format_version": 1,
        "arguments": jsonable_arguments(
            args,
            exclude={"_active_seed", "_pass_ks"},
        ),
        "arms": [asdict(arm) for arm in arms],
        "miles_commit": MILES_COMMIT,
        "resume_identity": _resume_identity(args, arms),
        "data_manifest": data_manifest,
        "fairness_contract": {
            "same_model_reward_recipe_and_training_seed": True,
            "same_total_gpus": True,
            "same_global_rounds": True,
            "same_prompt_groups_and_trajectory_budget": True,
            "same_per_rank_batch": True,
            "same_expert_parallel": True,
            "paired_round_major_prompt_streams": True,
            "paired_held_out_generation_seeds": True,
            "yeto_artifact": "authoritative syncer checkpoint export",
            "native_artifact": "Miles final LoRA save, PEFT-normalized by the harness",
        },
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.report_dir / "config.json", config)


def write_results(report_dir: Path, records: list[dict[str, Any]]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "results.jsonl"
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(path)


def load_results(report_dir: Path) -> list[dict[str, Any]]:
    path = report_dir / "results.jsonl"
    return _read_jsonl(path) if path.exists() else []


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return statistics.fmean(values), statistics.stdev(values) if len(
        values
    ) > 1 else 0.0


def aggregate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated = annotate_deltas(records)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in annotated:
        grouped.setdefault((record["arm"], record["m"]), []).append(record)
    output = []
    for (arm, m), group in grouped.items():
        rewards = [record["eval"]["reward_mean"] for record in group]
        reward_mean, reward_std = _mean_std(rewards)
        pass_keys = sorted(group[0]["eval"]["pass_at_k"], key=int)
        sync_records = [record["sync"] for record in group if record["sync"]]
        output.append(
            {
                "arm": arm,
                "m": m,
                "runs": len(group),
                "total_gpus": group[0]["total_gpus"],
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "pass_at_k": {
                    key: statistics.fmean(
                        record["eval"]["pass_at_k"][key] for record in group
                    )
                    for key in pass_keys
                },
                "delta_vs_native": _mean(
                    [
                        record["delta_vs_native"]
                        for record in group
                        if record["delta_vs_native"] is not None
                    ]
                ),
                "delta_vs_single": _mean(
                    [
                        record["delta_vs_single"]
                        for record in group
                        if record["delta_vs_single"] is not None
                    ]
                ),
                "delta_vs_strict": _mean(
                    [
                        record["delta_vs_strict"]
                        for record in group
                        if record["delta_vs_strict"] is not None
                    ]
                ),
                "train_wall_s": statistics.fmean(
                    record["train_wall_s"] for record in group
                ),
                "artifact_s": _mean(
                    [
                        record["artifact_s"]
                        for record in group
                        if record["artifact_s"] is not None
                    ]
                ),
                "artifact_ready_s": statistics.fmean(
                    record["artifact_ready_s"] for record in group
                ),
                "eval_wall_s": statistics.fmean(
                    record["eval"]["wall_s"] for record in group
                ),
                "trajectories_per_s": statistics.fmean(
                    record["training"]["trajectories"] / record["train_wall_s"]
                    for record in group
                ),
                "action_tokens_per_s": statistics.fmean(
                    record["training"]["action_tokens"] / record["train_wall_s"]
                    for record in group
                ),
                "gpu_hours": statistics.fmean(record["gpu_hours"] for record in group),
                "gpu_active_seconds": _mean(
                    [
                        record["gpu_activity"]["gpu_active_seconds"]
                        for record in group
                        if record.get("gpu_activity")
                    ]
                ),
                "gpu_active_fraction": _mean(
                    [
                        record["gpu_activity"]["gpu_active_fraction"]
                        for record in group
                        if record.get("gpu_activity")
                    ]
                ),
                "mean_gpu_utilization": _mean(
                    [
                        record["gpu_activity"]["mean_gpu_utilization"]
                        for record in group
                        if record.get("gpu_activity")
                    ]
                ),
                "min_gpu_utilization": _mean(
                    [
                        record["gpu_activity"]["min_gpu_utilization"]
                        for record in group
                        if record.get("gpu_activity")
                    ]
                ),
                "estimated_cost": _mean(
                    [
                        record["estimated_cost"]
                        for record in group
                        if record["estimated_cost"] is not None
                    ]
                ),
                "mean_kl": _mean(
                    [
                        record["mean_kl"]
                        for record in sync_records
                        if record["mean_kl"] is not None
                    ]
                ),
                "mean_sync_ms": _mean(
                    [
                        record["mean_sync_ms"]
                        for record in sync_records
                        if record["mean_sync_ms"] is not None
                    ]
                ),
                "sync_bytes_sent": _mean(
                    [float(record["sync_bytes_sent"]) for record in sync_records]
                ),
                "fragment_payload_bytes_sent": _mean(
                    [
                        float(record["fragment_payload_bytes_sent"])
                        for record in sync_records
                        if record.get("fragment_payload_bytes_sent") is not None
                    ]
                ),
                "fragment_payload_bytes_received": _mean(
                    [
                        float(record["fragment_payload_bytes_received"])
                        for record in sync_records
                        if record.get("fragment_payload_bytes_received") is not None
                    ]
                ),
                "fragment_payload_traffic_bytes": _mean(
                    [
                        float(record["fragment_payload_traffic_bytes"])
                        for record in sync_records
                        if record.get("fragment_payload_traffic_bytes") is not None
                    ]
                ),
                "hook_s": _mean(
                    [
                        float(record["hook_s"])
                        for record in sync_records
                        if record.get("hook_s") is not None
                    ]
                ),
                "finalization_s": _mean(
                    [
                        float(record["finalization_s"])
                        for record in sync_records
                        if record.get("finalization_s") is not None
                    ]
                ),
                "mean_realized_h": _mean(
                    [
                        float(record["mean_realized_h"])
                        for record in sync_records
                        if record.get("mean_realized_h") is not None
                    ]
                ),
                "mean_pull_to_push_s": _mean(
                    [
                        float(record["mean_pull_to_push_s"])
                        for record in sync_records
                        if record.get("mean_pull_to_push_s") is not None
                    ]
                ),
                "mean_bcast_queue_s": _mean(
                    [
                        float(record["mean_bcast_queue_s"])
                        for record in sync_records
                        if record.get("mean_bcast_queue_s") is not None
                    ]
                ),
            }
        )
    order = {
        "native-miles": 0,
        "yeto-single": 1,
        "yeto-federated": 2,
        "yeto-decoupled": 3,
    }
    return sorted(
        output,
        key=lambda row: (
            row["m"],
            next(
                rank for prefix, rank in order.items() if row["arm"].startswith(prefix)
            ),
        ),
    )


def write_report(args, records: list[dict[str, Any]]) -> None:
    aggregates = aggregate_records(records)
    (args.report_dir / "summary.json").write_text(
        json.dumps(aggregates, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    def fmt(value, digits=3):
        return "-" if value is None else f"{value:.{digits}f}"

    pass_keys = sorted({key for row in aggregates for key in row["pass_at_k"]}, key=int)
    lines = [
        f"# Miles RL LM benchmark: {args.model}",
        "",
        (
            f"Rounds: {args.global_rounds}; seeds: {args.seeds}; held-out prompts: "
            f"{args.eval_prompts}; samples/prompt: {args.eval_samples_per_prompt}"
        ),
        "",
        "## Quality",
        "",
        "| arm | M | runs | reward | "
        + " | ".join(f"pass@{key}" for key in pass_keys)
        + " | delta vs native | delta vs single | delta vs strict |",
        "|---|---:|---:|---:|" + "---:|" * len(pass_keys) + "---:|---:|---:|",
    ]
    for row in aggregates:
        reward = f"{row['reward_mean']:.4f} +/- {row['reward_std']:.4f}"
        passes = " | ".join(fmt(row["pass_at_k"].get(key), 4) for key in pass_keys)
        lines.append(
            f"| {row['arm']} | {row['m']} | {row['runs']} | {reward} | {passes} | "
            f"{fmt(row['delta_vs_native'], 4)} | {fmt(row['delta_vs_single'], 4)} | "
            f"{fmt(row['delta_vs_strict'], 4)} |"
        )
    lines.extend(
        [
            "",
            "## Systems",
            "",
            "| arm | GPUs | train s | artifact-ready s | eval s | traj/s | action tok/s | active GPU-s | active % | util avg/min % | GPU-h | cost | hook s | final s | H | PULL-to-PUSH s | BCAST queue s | sync ms | fragment payload MB | KL |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregates:
        traffic_mb = (
            None
            if row["fragment_payload_traffic_bytes"] is None
            else row["fragment_payload_traffic_bytes"] / 1e6
        )
        lines.append(
            f"| {row['arm']} | {row['total_gpus']} | {row['train_wall_s']:.1f} | "
            f"{row['artifact_ready_s']:.1f} | {row['eval_wall_s']:.1f} | "
            f"{row['trajectories_per_s']:.2f} | "
            f"{row['action_tokens_per_s']:.1f} | {fmt(row['gpu_active_seconds'], 1)} | "
            f"{fmt(row['gpu_active_fraction'], 3)} | "
            f"{fmt(row['mean_gpu_utilization'], 1)}/{fmt(row['min_gpu_utilization'], 1)} | "
            f"{row['gpu_hours']:.3f} | "
            f"{fmt(row['estimated_cost'], 2)} | {fmt(row['hook_s'], 2)} | "
            f"{fmt(row['finalization_s'], 2)} | "
            f"{fmt(row['mean_realized_h'], 2)} | "
            f"{fmt(row.get('mean_pull_to_push_s'), 3)} | "
            f"{fmt(row['mean_bcast_queue_s'], 3)} | {fmt(row['mean_sync_ms'], 2)} | "
            f"{fmt(traffic_mb, 3)} | {fmt(row['mean_kl'], 5)} |"
        )
    report = "\n".join(lines) + "\n"
    (args.report_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--data-revision", default=None)
    parser.add_argument("--reward-function", required=True)
    parser.add_argument(
        "--arms",
        default=",".join(_ARM_KINDS),
        help="comma-separated benchmark arms: native,single,federated,decoupled",
    )
    parser.add_argument("--islands", default="2")
    parser.add_argument("--seeds", default="17,29,43")
    parser.add_argument("--global-rounds", type=int, default=8)
    parser.add_argument("--groups-per-island", type=int, default=4)
    parser.add_argument("--samples-per-group", type=int, default=4)
    parser.add_argument("--optimizer-steps", type=int, default=1)
    parser.add_argument("--gpus-per-island", type=int, default=1)
    parser.add_argument("--fragments", type=int, default=8)
    parser.add_argument("--pipeline", type=int, default=2)
    parser.add_argument("--local-horizon", type=int, default=4)
    parser.add_argument("--expert-parallel", type=int, default=1)
    parser.add_argument("--miles-port-base", type=int, default=21000)
    parser.add_argument("--rollout-max-response-len", type=int, default=256)
    parser.add_argument(
        "--apply-chat-template-kwargs",
        type=json.loads,
        default={},
        help="JSON object forwarded to Miles and held-out chat-template rendering",
    )
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--inner-lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument(
        "--lora-targets",
        choices=["auto", "attention", "all-linear"],
        default="auto",
    )
    parser.add_argument("--wan-streams", type=int, default=4)
    parser.add_argument("--eval-prompts", type=int, default=64)
    parser.add_argument("--eval-samples-per-prompt", type=int, default=None)
    parser.add_argument("--pass-k", default="1,4")
    parser.add_argument("--pass-threshold", type=float, default=0.0)
    parser.add_argument("--eval-temperature", type=float, default=1.0)
    parser.add_argument("--eval-top-p", type=float, default=1.0)
    parser.add_argument("--eval-seed", type=int, default=100000)
    parser.add_argument("--eval-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--gpu-hour-cost", type=float, default=None)
    parser.add_argument("--arm-timeout-min", type=int, default=240)
    parser.add_argument(
        "--miles-root",
        type=Path,
        default=Path.home() / "miles",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=REPO_ROOT / "rl-benchmark-work",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=REPO_ROOT / "rl-benchmark-report",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def print_plan(args, arms: list[Arm]) -> None:
    plan = {
        "model": args.model,
        "expert_parallel": args.expert_parallel,
        "seeds": parse_seeds(args.seeds),
        "arms": [
            {
                **asdict(arm),
                **workload(
                    arm,
                    rounds=args.global_rounds,
                    samples_per_group=args.samples_per_group,
                ),
            }
            for arm in arms
        ],
        "fairness": {
            "same_total_gpus": True,
            "same_prompt_groups": True,
            "same_trajectories": True,
            "same_per_rank_batch": True,
            "same_expert_parallel": True,
            "paired_prompt_streams": True,
        },
    }
    for arm in arms:
        print(
            f"{arm.name}: {arm.islands} island(s) x {arm.gpus_per_island} GPU, "
            f"{arm.groups_per_round} groups/round"
        )
    print("PLAN_JSON " + json.dumps(plan, sort_keys=True))


def main(argv=None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] in {"_train-worker", "_eval-worker"}:
        if len(raw_argv) != 2:
            raise SystemExit(f"{raw_argv[0]} requires one config path")
        worker = (
            run_training_worker
            if raw_argv[0] == "_train-worker"
            else run_evaluation_worker
        )
        return worker(Path(raw_argv[1]))

    args = build_parser().parse_args(raw_argv)
    args.miles_root = args.miles_root.expanduser().resolve()
    args.work_dir = args.work_dir.expanduser().resolve()
    args.report_dir = args.report_dir.expanduser().resolve()
    if args.eval_samples_per_prompt is None:
        args.eval_samples_per_prompt = args.samples_per_group
    try:
        arms = select_arms(
            args.islands,
            args.gpus_per_island,
            args.groups_per_island,
            parse_arm_kinds(args.arms),
            fragments=args.fragments,
            pipeline=args.pipeline,
            local_horizon=args.local_horizon,
        )
        validate_args(args, arms, check_runtime=not args.dry_run)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print_plan(args, arms)
    if args.dry_run:
        return 0

    from yeto.benchmark_resume import (
        build_data_manifest,
        load_resume_config,
        validate_data_manifest,
    )
    from yeto.provenance import python_spec_sha256
    from yeto.rl.miles import verify_miles_revision

    miles_root = str(args.miles_root)
    if miles_root not in sys.path:
        sys.path.insert(0, miles_root)
    reward_sha256 = python_spec_sha256(args.reward_function, base_dir=REPO_ROOT)
    args.reward_sha256 = reward_sha256
    args.source_sha256 = source_tree_sha256()
    verify_miles_revision(args.miles_root)
    if any(arm.kind != "native" for arm in arms):
        ensure_syncer()
    model_path = resolve_model_path(args)
    distinct_m = tuple(sorted({arm.benchmark_islands for arm in arms}))

    if args.resume:
        if not args.work_dir.is_dir() or not args.report_dir.is_dir():
            raise SystemExit("--resume requires existing work and report directories")
        try:
            manifest = load_resume_config(
                args.report_dir / "config.json",
                _resume_identity(args, arms),
            )
            data_root, _, _ = validate_data_manifest(args.work_dir, manifest)
            prompt_matrix = load_prompt_matrix(data_root, distinct_m)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    else:
        for path in (args.work_dir, args.report_dir):
            if path.exists():
                if not args.overwrite:
                    raise SystemExit(
                        f"{path} already exists; pass --overwrite to replace it"
                    )
                shutil.rmtree(path)
        args.work_dir.mkdir(parents=True)
        args.report_dir.mkdir(parents=True)
        try:
            prompt_matrix, train_rows = materialize_prompt_matrix(args, distinct_m)
        except (RuntimeError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        first_eval = prompt_matrix[distinct_m[0]][0][2]
        source = args.data if Path(args.data).expanduser().exists() else None
        data_manifest = build_data_manifest(
            args.work_dir,
            args.work_dir / "data",
            first_eval,
            train_rows=train_rows,
            eval_rows=args.eval_prompts,
            source=source,
        )
        write_run_config(args, arms, data_manifest=data_manifest)

    records = load_results(args.report_dir) if args.resume else []
    expected = expected_record_keys(parse_seeds(args.seeds), arms)
    try:
        validate_result_records(records, expected)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    completed = {(record["arm"], record["m"], record["seed"]) for record in records}
    for seed in parse_seeds(args.seeds):
        for arm in arms:
            key = (arm.name, arm.benchmark_islands, seed)
            if key in completed:
                print(f"[rl-benchmark] resume: skipping {arm.name} seed={seed}")
                continue
            print(f"[rl-benchmark] running {arm.name} seed={seed}", flush=True)
            paths, streams = prompt_matrix[arm.benchmark_islands]
            record = run_arm(
                args,
                arm,
                seed=seed,
                model_path=model_path,
                prompt_paths=paths,
                streams=streams,
                reward_sha256=reward_sha256,
            )
            records.append(record)
            completed.add(key)
            write_results(args.report_dir, records)

    write_report(args, records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
