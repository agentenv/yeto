#!/usr/bin/env python3
"""Benchmark diffusion DiLoCo against an equal-hardware synchronous baseline.

The experiment keeps the model, LoRA recipe, per-GPU batch, total device
count, training-sample budget, and held-out evaluation draws fixed.  The only
intended difference is synchronization:

* ``baseline-mN`` uses one synchronous process group across all N islands'
  devices with ``--syncer none``;
* a DiLoCo arm uses N independent process groups, one per island, connected
  to the real Rust syncer.

Each DiLoCo result is evaluated from the syncer's checkpoint exported through
``yeto.diffusion.export``.  A learner's locally blended adapter is never used
as the result.  Matching logical ranks use the same training rows and RNG
streams, and held-out loss pairs rows, timesteps, and noise draws across arms.

This is a local execution harness.  It partitions the visible CUDA devices
between learner processes but does not provision cloud machines.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from yeto.benchmark_resume import (  # noqa: E402
    build_data_manifest,
    implementation_fingerprint,
    jsonable_arguments,
    load_resume_config,
    validate_data_manifest,
    validate_record_keys,
    write_json_atomic,
)

SYNCER_BIN = REPO_ROOT / "syncer/target/release/yeto-syncer"
_USED_PORTS: set[int] = set()
_RESUME_IDENTITY_EXCLUDES = {
    "dry_run",
    "eval_payload",
    "materialized_data",
    "overwrite",
    "report_dir",
    "resume",
    "seeds",
    "settings",
    "work_dir",
}
_IMPLEMENTATION_PATHS = (
    Path(__file__),
    REPO_ROOT / "yeto/budget_finalization.py",
    REPO_ROOT / "yeto/benchmark_resume.py",
    REPO_ROOT / "yeto/final_marker.py",
    REPO_ROOT / "yeto/diffusion",
    REPO_ROOT / "yeto/data.py",
    REPO_ROOT / "yeto/losses.py",
    REPO_ROOT / "yeto/models.py",
    REPO_ROOT / "yeto/fragments.py",
    REPO_ROOT / "yeto/protocol.py",
    REPO_ROOT / "yeto/tensor_io.py",
    REPO_ROOT / "syncer/src",
    REPO_ROOT / "syncer/Cargo.toml",
    REPO_ROOT / "syncer/Cargo.lock",
)


@dataclass(frozen=True)
class Arm:
    """One diffusion synchronization configuration."""

    name: str
    learners: int = 2
    fragments: int = 8
    fragment_pattern: str = "binpack"
    merge_alpha: float = 0.5
    wire_dtype: str = "bf16"
    pipeline: int = 2
    delta_correction: str = "heloco"
    quorum: int | None = None
    outer_lr: float = 0.7
    outer_momentum: float = 0.9
    sync_interval_steps: float = 24.0


# The default arms are production-shaped.  ``unthrottled`` is the explicit
# low-latency diagnostic; ordinary m2/m4 arms retain the production H target.
PRESETS: dict[str, Arm] = {
    "m2": Arm("m2"),
    "m4": Arm("m4", learners=4),
    "alpha0": Arm("alpha0", merge_alpha=0.0),
    "q4": Arm("q4", wire_dtype="q4"),
    "serial": Arm("serial", pipeline=1),
    "noheloco": Arm("noheloco", delta_correction="none"),
    "strided": Arm("strided", fragment_pattern="strided"),
    # This is intentionally named direct-rda, not avg: non-embedding
    # fragments still use RDA.  The arm removes Nesterov gain and local
    # blending so the syncer applies each merged RDA delta directly.
    "direct-rda": Arm(
        "direct-rda",
        merge_alpha=0.0,
        outer_lr=1.0,
        outer_momentum=0.0,
    ),
    "unthrottled": Arm("unthrottled", sync_interval_steps=0.0),
}


def select_arms(spec: str, fragments: int = 8) -> list[Arm]:
    names = list(PRESETS) if spec == "all" else [v.strip() for v in spec.split(",") if v.strip()]
    unknown = [name for name in names if name not in PRESETS]
    if unknown:
        raise ValueError(f"unknown settings {unknown}; choose from {list(PRESETS)}")
    return [replace(PRESETS[name], fragments=fragments) for name in names]


def parse_seeds(spec: str) -> list[int]:
    try:
        seeds = [int(value.strip()) for value in spec.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError("--seeds must be a comma-separated list of integers") from exc
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds contains duplicates")
    return seeds


def steps_for_samples(
    sample_budget: int,
    micro_batch_size: int,
    grad_accum: int,
    total_ranks: int,
) -> int:
    """Optimizer steps needed to process at least ``sample_budget`` rows."""
    per_step = micro_batch_size * grad_accum * total_ranks
    if sample_budget < 1 or per_step < 1:
        raise ValueError("sample budget, batch size, accumulation, and ranks must be positive")
    return math.ceil(sample_budget / per_step)


def processed_samples(
    steps: int,
    micro_batch_size: int,
    grad_accum: int,
    total_ranks: int,
) -> int:
    return steps * micro_batch_size * grad_accum * total_ranks


def effective_grad_accum(micro_batch_size: int, requested_grad_accum: int) -> int:
    """Match diffusion learner's explicit-batch accumulation rebalance."""
    return max(1, math.ceil(requested_grad_accum / micro_batch_size))


def free_port() -> int:
    while True:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
        if port not in _USED_PORTS:
            _USED_PORTS.add(port)
            return port


def _distributed_prefix(nproc: int) -> list[str]:
    if nproc < 1:
        raise ValueError("nproc must be positive")
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        "--master_addr=127.0.0.1",
        f"--master_port={free_port()}",
        "-m",
        "yeto.diffusion.learner",
    ]


def learner_command(
    args,
    train_data: Path,
    output_dir: Path,
    *,
    nproc: int,
    learner_id: int,
    num_learners: int,
    syncer: str,
    max_steps: int,
    seed: int,
    arm: Arm | None,
) -> list[str]:
    cmd = _distributed_prefix(nproc)
    cmd += [
        "--model",
        args.model,
        "--data",
        str(train_data),
        "--syncer",
        syncer,
        "--learner-id",
        str(learner_id),
        "--num-learners",
        str(num_learners),
        "--loss-function",
        "flow_matching",
        "--tuning",
        "lora",
        "--shard",
        args.shard,
        "--lora-r",
        str(args.lora_r),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-targets",
        args.lora_targets,
        "--micro-batch-size",
        str(args.micro_batch_size),
        "--grad-accum",
        str(args.grad_accum),
        "--inner-lr",
        str(args.inner_lr),
        "--weight-decay",
        str(args.weight_decay),
        "--warmup-steps",
        str(args.warmup_steps),
        "--stream-workers",
        str(args.stream_workers),
        "--seed",
        str(seed),
        "--max-local-steps",
        str(max_steps),
        "--wan-streams",
        str(args.wan_streams),
        "--output-dir",
        str(output_dir),
        "--image-column",
        args.image_column,
        "--video-column",
        args.video_column,
        "--prompt-column",
        args.prompt_column,
        "--latent-column",
        args.latent_column,
        "--text-embeds-column",
        args.text_embeds_column,
        "--text-attention-mask-column",
        args.text_attention_mask_column,
        "--pooled-text-embeds-column",
        args.pooled_text_embeds_column,
        "--diffusion-loss-weighting",
        args.diffusion_loss_weighting,
        "--diffusion-min-snr-gamma",
        str(args.diffusion_min_snr_gamma),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--resize-mode",
        args.resize_mode,
    ]
    if nproc == 1 or not args.device.startswith("cuda"):
        cmd += ["--device", args.device]
    if args.diffusion_adapter:
        cmd += ["--diffusion-adapter", args.diffusion_adapter]
    if args.cache_latents:
        cmd += ["--cache-latents"]
    if args.cache_text_embeds:
        cmd += ["--cache-text-embeds"]
    if args.bucket_by_shape:
        cmd += ["--bucket-by-shape"]
    if args.num_frames is not None:
        cmd += ["--num-frames", str(args.num_frames)]
    if args.fps is not None:
        cmd += ["--fps", str(args.fps)]
    if arm is not None:
        cmd += [
            "--learner-budget-steps",
            str(max_steps),
            "--fragments",
            str(arm.fragments),
            "--fragment-pattern",
            arm.fragment_pattern,
            "--merge-alpha",
            str(arm.merge_alpha),
            "--wire-dtype",
            arm.wire_dtype,
        ]
    return cmd


def syncer_command(
    args,
    arm: Arm,
    port: int,
    arm_dir: Path,
    total_steps: int,
    learner_budget_steps: int | None = None,
    resume_consolidation: bool = False,
) -> list[str]:
    if learner_budget_steps is not None and resume_consolidation:
        raise ValueError("budget cutoff and resumed consolidation are separate stages")
    quorum = arm.learners if resume_consolidation else arm.quorum or arm.learners
    pipeline = 1 if resume_consolidation else arm.pipeline
    sync_interval = 0.0 if resume_consolidation else arm.sync_interval_steps
    command = [
        str(SYNCER_BIN),
        "--port",
        str(port),
        "--learners",
        str(arm.learners),
        "--quorum",
        str(quorum),
        "--grace-ms",
        str(args.grace_ms),
        "--pipeline",
        str(pipeline),
        "--sync-interval-steps",
        str(sync_interval),
        "--delta-correction",
        arm.delta_correction,
        "--outer-lr",
        str(arm.outer_lr),
        "--outer-momentum",
        str(arm.outer_momentum),
        "--total-steps",
        str(total_steps),
        "--checkpoint-path",
        str(arm_dir / "state.ckpt"),
        "--checkpoint-every",
        "1",
        "--event-tape",
        str(arm_dir / "tape.jsonl"),
    ]
    if learner_budget_steps is not None:
        command += ["--learner-budget-steps", str(learner_budget_steps)]
    if resume_consolidation:
        command += [
            "--resume",
            "--mark-final-checkpoint",
        ]
    return command


def _visible_cuda_devices() -> list[str] | None:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or not raw.strip():
        return None
    return [value.strip() for value in raw.split(",") if value.strip()]


def cuda_env(start: int, count: int, device: str) -> dict[str, str] | None:
    if not device.startswith("cuda"):
        return None
    visible = _visible_cuda_devices()
    if visible is None:
        chosen = [str(index) for index in range(start, start + count)]
    else:
        chosen = visible[start : start + count]
        if len(chosen) != count:
            raise ValueError(
                f"need CUDA device slice [{start}:{start + count}], but "
                f"CUDA_VISIBLE_DEVICES exposes only {visible}"
            )
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(chosen)
    return env


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
    )
    index_to_uuid = {}
    for line in result.stdout.splitlines():
        parts = [value.strip() for value in line.split(",")]
        if len(parts) >= 2:
            index_to_uuid[parts[0]] = parts[1]
    return {index_to_uuid.get(value, value) for value in visible}


def wait_for_free_gpus(device: str, limit_mb: int = 2000, timeout_s: int = 300) -> None:
    if not device.startswith("cuda"):
        return
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
        )
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
            print(f"[diffusion-benchmark] waiting for GPUs: {current}", flush=True)
            last = current
        if time.monotonic() >= deadline:
            raise RuntimeError(f"GPUs still occupied after {timeout_s}s: {current}")
        time.sleep(3)


def materialize_data_source(data: str, destination: Path) -> str:
    """Stage an S3 dataset prefix locally; pass other sources through."""
    if not data.startswith("s3://"):
        return data
    if shutil.which("aws") is None:
        raise RuntimeError(
            "S3 benchmark data requires the AWS CLI and ambient read credentials"
        )

    destination.mkdir(parents=True, exist_ok=True)
    print(f"[diffusion-benchmark] syncing {data} to {destination}", flush=True)
    try:
        subprocess.run(
            [
                "aws",
                "s3",
                "sync",
                data,
                str(destination),
                "--only-show-errors",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"failed to sync benchmark data from {data} (aws exit {exc.returncode})"
        ) from exc
    if not any(destination.iterdir()):
        raise RuntimeError(f"S3 dataset prefix {data} contains no objects")
    return str(destination.resolve())


def _source_root(data: str) -> Path | None:
    path = Path(os.path.expanduser(data))
    if not path.exists():
        return None
    return path.resolve() if path.is_dir() else path.resolve().parent


def _resolve_row_paths(row: dict, columns: tuple[str, ...], root: Path | None) -> dict:
    if root is None:
        return row
    out = dict(row)
    for column in columns:
        value = out.get(column)
        if not isinstance(value, str):
            continue
        path = Path(os.path.expanduser(value))
        if path.is_absolute():
            continue
        candidate = root / path
        if candidate.exists():
            out[column] = str(candidate.resolve())
    return out


def _copy_cache_metadata(source_root: Path | None, destination: Path) -> None:
    if source_root is None:
        return
    source = source_root / "yeto_diffusion_cache.json"
    if source.exists() and destination.is_dir():
        shutil.copy2(source, destination / source.name)


def _save_subset(
    dataset,
    indices: range,
    destination: Path,
    *,
    path_columns: tuple[str, ...],
    source_root: Path | None,
) -> Path:
    if hasattr(dataset, "select") and hasattr(dataset, "save_to_disk"):
        subset = dataset.select(list(indices))
        if source_root is not None:
            subset = subset.map(
                lambda row: _resolve_row_paths(row, path_columns, source_root),
                desc=f"resolve paths for {destination.name}",
            )
        subset.save_to_disk(str(destination))
        _copy_cache_metadata(source_root, destination)
        return destination

    output = destination.with_suffix(".jsonl")
    with output.open("w", encoding="utf-8") as handle:
        for index in indices:
            row = _resolve_row_paths(dict(dataset[index]), path_columns, source_root)
            try:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            except TypeError as exc:
                raise TypeError(
                    "non-Hugging-Face datasets must contain JSON-serializable rows; "
                    "use a datasets.Dataset for decoded Image/Video values"
                ) from exc
    _copy_cache_metadata(source_root, output.parent)
    return output


def split_data(args, work_dir: Path, data: str | None = None) -> tuple[Path, Path, int]:
    """Create identical train/eval subsets for every benchmark arm."""
    from yeto.data import load_rows

    data = data or args.data
    dataset = load_rows(data)
    usable = len(dataset)
    if args.max_train_rows is not None:
        usable = min(usable, args.max_train_rows + args.eval_rows)
    if usable <= args.eval_rows:
        raise ValueError(
            f"dataset has {usable} usable rows; need more than --eval-rows {args.eval_rows}"
        )
    if args.num_frames is None:
        for index in range(min(usable, 8)):
            row = dataset[index]
            video = row.get(args.video_column)
            latent_frames = row.get("latent_num_frames", row.get("num_frames"))
            has_video = video is not None
            if isinstance(video, str):
                has_video = bool(video)
            elif isinstance(video, (list, tuple)):
                has_video = bool(video)
            try:
                has_multiple_frames = latent_frames is not None and int(latent_frames) > 1
            except (TypeError, ValueError):
                has_multiple_frames = False
            if has_video or has_multiple_frames:
                raise ValueError(
                    "video benchmarks require explicit --num-frames so every arm "
                    "uses the same temporal shape"
                )
    train_rows = usable - args.eval_rows
    source_root = _source_root(data)
    path_columns = (
        args.image_column,
        args.video_column,
        args.latent_column,
        args.text_embeds_column,
        args.text_attention_mask_column,
        args.pooled_text_embeds_column,
    )
    train_data = _save_subset(
        dataset,
        range(train_rows),
        work_dir / "train",
        path_columns=path_columns,
        source_root=source_root,
    )
    eval_data = _save_subset(
        dataset,
        range(train_rows, usable),
        work_dir / "eval",
        path_columns=path_columns,
        source_root=source_root,
    )
    return train_data, eval_data, train_rows


def ensure_syncer() -> None:
    if SYNCER_BIN.exists():
        return
    print("[diffusion-benchmark] building Rust syncer", flush=True)
    subprocess.run(
        ["cargo", "build", "--release", "--quiet"],
        cwd=REPO_ROOT / "syncer",
        check=True,
    )


def _tail(path: Path, lines: int = 16) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def _stop_process(process: subprocess.Popen, timeout: int = 20) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            process.kill()
        process.wait(timeout=10)


def _wait_for_syncer(
    process: subprocess.Popen,
    log: Path,
    timeout_s: int,
    learners: list[subprocess.Popen] | None = None,
    learner_logs: list[Path] | None = None,
) -> None:
    deadline = time.monotonic() + timeout_s
    while process.poll() is None:
        if learners is not None:
            for index, learner in enumerate(learners):
                returncode = learner.poll()
                if returncode is not None:
                    detail = _tail(learner_logs[index]) if learner_logs else ""
                    raise RuntimeError(
                        f"learner {index} exited before the budget cutoff "
                        f"with code {returncode}:\n{detail}"
                    )
        if time.monotonic() >= deadline:
            raise RuntimeError(f"syncer timed out after {timeout_s}s:\n{_tail(log)}")
        time.sleep(1)
    if process.returncode != 0:
        raise RuntimeError(
            f"syncer failed with exit code {process.returncode}:\n{_tail(log)}"
        )


def run_logged(
    command: list[str],
    log_path: Path,
    *,
    env: dict[str, str] | None = None,
    timeout_s: int | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
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
        except subprocess.TimeoutExpired:
            _stop_process(process)
            raise RuntimeError(f"command timed out: {' '.join(command)}\n{_tail(log_path)}")
    if returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {returncode}: {' '.join(command)}\n{_tail(log_path)}"
        )


def _set_eval_mode(pipe, adapter) -> None:
    import torch

    from yeto.diffusion.learner import _trainable_module_items

    seen: set[int] = set()
    for value in getattr(pipe, "components", {}).values():
        if isinstance(value, torch.nn.Module) and id(value) not in seen:
            value.eval()
            seen.add(id(value))
    for _, module in _trainable_module_items(pipe, adapter):
        if id(module) not in seen:
            module.eval()
            seen.add(id(module))


def evaluate_loss(args, adapter_dir: Path | None, eval_data: Path) -> dict:
    """Paired held-out flow-matching loss per predicted element."""
    import torch

    from yeto.data import load_rows
    from yeto.diffusion import learner, sample

    device = torch.device(args.eval_device)
    eval_args = SimpleNamespace(**vars(args))
    eval_args.device = args.eval_device
    eval_args.seed = parse_seeds(args.seeds)[0]
    eval_args.tuning = "lora"
    eval_args.loss_function = "flow_matching"
    if adapter_dir is None:
        adapter = learner.load_diffusion_adapter(args.diffusion_adapter)
        pipe = learner.load_pipeline(eval_args, device, adapter)
    else:
        eval_args.dtype = args.eval_dtype
        pipe, _metadata, adapter = sample.load_artifact_pipeline(adapter_dir, eval_args)
    _set_eval_mode(pipe, adapter)

    dataset = load_rows(str(eval_data))
    root = eval_data if eval_data.is_dir() else eval_data.parent
    total_loss = 0.0
    total_elements = 0.0
    cases: list[float] = []
    with torch.no_grad():
        for row_index in range(len(dataset)):
            row = dict(dataset[row_index])
            row["__yeto_data_root__"] = str(root)
            for repetition in range(args.eval_repeats):
                rng_seed = learner.diffusion_eval_seed(
                    args.eval_seed,
                    row_index,
                    repetition,
                )
                loss, denominator = learner.compute_diffusion_loss(
                    pipe,
                    [row],
                    eval_args,
                    device,
                    adapter=adapter,
                    rng_seed=rng_seed,
                )
                loss_value = float(loss.item())
                denominator_value = float(denominator.item())
                total_loss += loss_value
                total_elements += denominator_value
                cases.append(loss_value / max(denominator_value, 1.0))
    return {
        "loss_per_element": total_loss / max(total_elements, 1.0),
        "total_loss": total_loss,
        "elements": total_elements,
        "cases": len(cases),
        "case_mean": statistics.fmean(cases) if cases else None,
        "case_std": statistics.stdev(cases) if len(cases) > 1 else 0.0,
    }


def _jsonable_args(args) -> dict:
    return jsonable_arguments(args, exclude={"eval_payload"})


def evaluate_in_subprocess(
    args,
    adapter_dir: Path | None,
    eval_data: Path,
    log_path: Path,
) -> dict:
    payload = {
        "args": _jsonable_args(args),
        "adapter_dir": str(adapter_dir) if adapter_dir is not None else None,
        "eval_data": str(eval_data),
    }
    command = [
        sys.executable,
        __file__,
        "--eval-payload",
        json.dumps(payload, separators=(",", ":")),
    ]
    wait_for_free_gpus(args.eval_device)
    env = cuda_env(0, 1, args.eval_device)
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    wait_for_free_gpus(args.eval_device)
    for line in reversed(result.stdout.splitlines()):
        if line.startswith("EVAL_JSON "):
            return json.loads(line.removeprefix("EVAL_JSON "))
    raise RuntimeError(
        f"evaluation failed with exit code {result.returncode}:\n{_tail(log_path)}"
    )


def summarize_tape(path: Path, learners: int) -> dict:
    records = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    c_steps = [
        int(responder.get("c_steps", 0))
        for record in records
        for responder in record.get("responders", [])
    ]
    responders = [len(record.get("responders", [])) for record in records]
    sync_ms = [float(record.get("ms", 0.0)) for record in records]
    staleness = []
    version_history: dict[int, list[int]] = {}
    for record in records:
        fragment = int(record.get("fragment", -1))
        history = version_history.setdefault(fragment, [0])
        previous = history[-1]
        for responder in record.get("responders", []):
            base = int(responder.get("base_version", previous))
            staleness.append(sum(1 for version in history[1:] if version > base))
        if fragment >= 0:
            history.append(int(record.get("step", previous)))
    return {
        "merges": len(records),
        "responses": len(c_steps),
        "mean_h": statistics.fmean(c_steps) if c_steps else None,
        "median_h": statistics.median(c_steps) if c_steps else None,
        "mean_responders": statistics.fmean(responders) if responders else None,
        "participation_rate": (
            sum(responders) / (len(records) * learners)
            if records and learners > 0
            else None
        ),
        "mean_sync_ms": statistics.fmean(sync_ms) if sync_ms else None,
        "mean_staleness": statistics.fmean(staleness) if staleness else None,
        "max_staleness": max(staleness) if staleness else None,
    }


def _tensor_bytes(numel: int, wire_dtype: str, *, broadcast: bool) -> int:
    if wire_dtype == "f32":
        return numel * 4
    if wire_dtype == "q4" and not broadcast:
        return math.ceil(numel / 256) * (4 + 128)
    return numel * 2


def estimate_tensor_bytes(
    fragment_numels: list[int],
    tape_path: Path,
    wire_dtype: str,
    learners: int,
) -> int:
    records = []
    if tape_path.exists():
        for line in tape_path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    init = sum(_tensor_bytes(numel, wire_dtype, broadcast=True) for numel in fragment_numels)
    total = init + learners * init
    for record in records:
        fragment = int(record.get("fragment", -1))
        if not 0 <= fragment < len(fragment_numels):
            continue
        numel = fragment_numels[fragment]
        total += len(record.get("responders", [])) * _tensor_bytes(
            numel,
            wire_dtype,
            broadcast=False,
        )
        total += learners * _tensor_bytes(numel, wire_dtype, broadcast=True)
    return total


def run_baseline(args, m: int, seed: int, train_data: Path, seed_dir: Path) -> dict:
    run_dir = seed_dir / f"baseline-m{m}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    output_dir = run_dir / "adapter"
    total_ranks = m * args.learner_gpus
    grad_accum = effective_grad_accum(args.micro_batch_size, args.grad_accum)
    steps = steps_for_samples(
        args.sample_budget,
        args.micro_batch_size,
        grad_accum,
        total_ranks,
    )
    command = learner_command(
        args,
        train_data,
        output_dir,
        nproc=total_ranks,
        learner_id=0,
        num_learners=1,
        syncer="none",
        max_steps=steps,
        seed=seed,
        arm=None,
    )
    wait_for_free_gpus(args.device)
    started = time.monotonic()
    run_logged(
        command,
        run_dir / "learner.log",
        env=cuda_env(0, total_ranks, args.device),
        timeout_s=args.arm_timeout_min * 60,
    )
    wall = time.monotonic() - started
    return {
        "artifact": output_dir,
        "wall_s": wall,
        "steps_per_learner": steps,
        "processed_samples": processed_samples(
            steps,
            args.micro_batch_size,
            grad_accum,
            total_ranks,
        ),
        "total_ranks": total_ranks,
        "total_gpus": total_ranks if args.device.startswith("cuda") else 0,
    }


def _wait_for_learners(
    processes: list[subprocess.Popen],
    logs: list[Path],
    timeout_s: int,
    syncer: subprocess.Popen | None = None,
    syncer_log: Path | None = None,
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
                    f"learner {index} failed with exit code {returncode}:\n{_tail(logs[index])}"
                )
        if not pending:
            return
        if syncer is not None and syncer.poll() not in (None, 0):
            detail = _tail(syncer_log) if syncer_log is not None else ""
            raise RuntimeError(
                f"syncer failed during final consolidation with code "
                f"{syncer.returncode}:\n{detail}"
            )
        if time.monotonic() >= deadline:
            raise RuntimeError(f"learners timed out after {timeout_s}s")
        time.sleep(1)


def _export_command(args, arm: Arm, checkpoint: Path, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "yeto.diffusion.export",
        "--checkpoint",
        str(checkpoint),
        "--model",
        args.model,
        "--tuning",
        "lora",
        "--lora-r",
        str(args.lora_r),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-targets",
        args.lora_targets,
        "--fragments",
        str(arm.fragments),
        "--fragment-pattern",
        arm.fragment_pattern,
        "--output-dir",
        str(output_dir),
        "--device",
        args.export_device,
    ]
    if args.diffusion_adapter:
        command += ["--diffusion-adapter", args.diffusion_adapter]
    return command


def run_diloco(args, arm: Arm, seed: int, train_data: Path, seed_dir: Path) -> dict:
    run_dir = seed_dir / arm.name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    total_ranks = arm.learners * args.learner_gpus
    grad_accum = effective_grad_accum(args.micro_batch_size, args.grad_accum)
    steps = steps_for_samples(
        args.sample_budget,
        args.micro_batch_size,
        grad_accum,
        total_ranks,
    )
    port = free_port()
    # This is only a ceiling.  Learners own the sample budget and stop first.
    syncer_steps = max(arm.fragments, steps * arm.learners * arm.fragments * 2)
    syncer_log_path = run_dir / "syncer.log"
    checkpoint = run_dir / "state.ckpt"
    syncer_log = syncer_log_path.open("w", encoding="utf-8")
    syncer = subprocess.Popen(
        syncer_command(
            args,
            arm,
            port,
            run_dir,
            syncer_steps,
            learner_budget_steps=steps,
        ),
        cwd=REPO_ROOT,
        stdout=syncer_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    processes: list[subprocess.Popen] = []
    handles = []
    learner_logs: list[Path] = []
    wait_for_free_gpus(args.device)
    started = time.monotonic()
    try:
        for learner_id in range(arm.learners):
            output_dir = run_dir / f"learner-{learner_id}"
            command = learner_command(
                args,
                train_data,
                output_dir,
                nproc=args.learner_gpus,
                learner_id=learner_id,
                num_learners=arm.learners,
                syncer=f"127.0.0.1:{port}",
                max_steps=steps,
                seed=seed,
                arm=arm,
            )
            log_path = run_dir / f"learner-{learner_id}.log"
            handle = log_path.open("w", encoding="utf-8")
            handles.append(handle)
            learner_logs.append(log_path)
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env=cuda_env(
                        learner_id * args.learner_gpus,
                        args.learner_gpus,
                        args.device,
                    ),
                    start_new_session=True,
                )
            )
        timeout_s = args.arm_timeout_min * 60
        _wait_for_syncer(
            syncer,
            syncer_log_path,
            timeout_s,
            learners=processes,
            learner_logs=learner_logs,
        )
        from yeto.final_marker import read_checkpoint_global_step

        cutoff_step = read_checkpoint_global_step(checkpoint)
        time.sleep(args.drain_seconds)
        syncer = subprocess.Popen(
            syncer_command(
                args,
                arm,
                port,
                run_dir,
                cutoff_step + arm.fragments,
                resume_consolidation=True,
            ),
            cwd=REPO_ROOT,
            stdout=syncer_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _wait_for_learners(
            processes,
            learner_logs,
            timeout_s,
            syncer=syncer,
            syncer_log=syncer_log_path,
        )
        _wait_for_syncer(syncer, syncer_log_path, timeout_s)
        from yeto.budget_finalization import validate_consolidation_tape

        validate_consolidation_tape(
            run_dir / "tape.jsonl",
            cutoff_step=cutoff_step,
            fragments=arm.fragments,
            learners=arm.learners,
            budget_steps=steps,
        )
    finally:
        for process in processes:
            _stop_process(process)
        _stop_process(syncer)
        for handle in handles:
            handle.close()
        syncer_log.close()
    wall = time.monotonic() - started

    if not checkpoint.exists():
        raise RuntimeError(
            f"{arm.name}: no syncer checkpoint was produced\n{_tail(syncer_log_path)}"
        )
    from yeto.final_marker import validate_final_checkpoint

    validate_final_checkpoint(checkpoint)
    export_dir = run_dir / "export"
    wait_for_free_gpus(args.export_device)
    export_started = time.monotonic()
    run_logged(
        _export_command(args, arm, checkpoint, export_dir),
        run_dir / "export.log",
        timeout_s=args.arm_timeout_min * 60,
    )
    export_s = time.monotonic() - export_started

    from yeto.export import parse_checkpoint

    parsed = parse_checkpoint(checkpoint)
    fragment_numels = [int(flat.numel()) for _, flat, _ in parsed.fragments]
    tape_path = run_dir / "tape.jsonl"
    tape = summarize_tape(tape_path, arm.learners)
    tape["estimated_tensor_bytes"] = estimate_tensor_bytes(
        fragment_numels,
        tape_path,
        arm.wire_dtype,
        arm.learners,
    )
    return {
        "artifact": export_dir,
        "wall_s": wall,
        "export_s": export_s,
        "steps_per_learner": steps,
        "processed_samples": processed_samples(
            steps,
            args.micro_batch_size,
            grad_accum,
            total_ranks,
        ),
        "total_ranks": total_ranks,
        "total_gpus": total_ranks if args.device.startswith("cuda") else 0,
        "global_step": parsed.global_step,
        "fragment_versions": [version for version, _, _ in parsed.fragments],
        "tape": tape,
    }


def make_record(
    *,
    kind: str,
    arm: str,
    seed: int | None,
    learners: int,
    run: dict,
    evaluation: dict,
    gpu_hour_cost: float | None,
) -> dict:
    wall_s = float(run.get("wall_s", 0.0))
    total_gpus = int(run.get("total_gpus", 0))
    gpu_hours = total_gpus * wall_s / 3600.0
    samples = int(run.get("processed_samples", 0))
    return {
        "kind": kind,
        "arm": arm,
        "seed": seed,
        "learners": learners,
        "total_ranks": int(run.get("total_ranks", 0)),
        "total_gpus": total_gpus,
        "steps_per_learner": int(run.get("steps_per_learner", 0)),
        "processed_samples": samples,
        "wall_s": wall_s,
        "export_s": float(run.get("export_s", 0.0)),
        "samples_per_s": samples / wall_s if wall_s > 0 else None,
        "gpu_hours": gpu_hours,
        "estimated_cost": gpu_hours * gpu_hour_cost if gpu_hour_cost is not None else None,
        "eval": evaluation,
        "global_step": run.get("global_step"),
        "fragment_versions": run.get("fragment_versions"),
        "tape": run.get("tape"),
    }


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate_records(records: list[dict]) -> list[dict]:
    baselines = {
        (record["seed"], record["learners"]): record["eval"]["loss_per_element"]
        for record in records
        if record["kind"] == "baseline"
    }
    order = []
    grouped: dict[tuple[str, str, int], list[dict]] = {}
    for record in records:
        key = (record["kind"], record["arm"], record["learners"])
        if key not in grouped:
            order.append(key)
            grouped[key] = []
        grouped[key].append(record)

    output = []
    for kind, arm, learners in order:
        group = grouped[(kind, arm, learners)]
        losses = [record["eval"]["loss_per_element"] for record in group]
        loss_mean, loss_std = _mean_std(losses)
        deltas = []
        if kind == "diloco":
            for record in group:
                baseline = baselines[(record["seed"], learners)]
                deltas.append(100.0 * (record["eval"]["loss_per_element"] - baseline) / baseline)
        delta_mean, delta_std = _mean_std(deltas)
        tape_records = [record.get("tape") or {} for record in group]

        def tape_mean(name: str):
            values = [tape[name] for tape in tape_records if tape.get(name) is not None]
            return statistics.fmean(values) if values else None

        output.append(
            {
                "kind": kind,
                "arm": arm,
                "learners": learners,
                "runs": len(group),
                "total_gpus": group[0]["total_gpus"],
                "processed_samples": group[0]["processed_samples"],
                "loss_mean": loss_mean,
                "loss_std": loss_std,
                "delta_mean_pct": delta_mean,
                "delta_std_pct": delta_std,
                "wall_mean_s": statistics.fmean(record["wall_s"] for record in group),
                "samples_per_s_mean": statistics.fmean(
                    record["samples_per_s"] for record in group if record["samples_per_s"] is not None
                )
                if any(record["samples_per_s"] is not None for record in group)
                else None,
                "gpu_hours_mean": statistics.fmean(record["gpu_hours"] for record in group),
                "estimated_cost_mean": statistics.fmean(
                    record.get("estimated_cost")
                    for record in group
                    if record.get("estimated_cost") is not None
                )
                if any(record.get("estimated_cost") is not None for record in group)
                else None,
                "mean_h": tape_mean("mean_h"),
                "participation_rate": tape_mean("participation_rate"),
                "mean_staleness": tape_mean("mean_staleness"),
                "sync_bytes_mean": tape_mean("estimated_tensor_bytes"),
            }
        )
    return output


def write_partial_results(report_dir: Path, records: list[dict]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "results.jsonl"
    temporary = report_dir / "results.jsonl.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(path)


def load_partial_results(report_dir: Path) -> list[dict]:
    path = report_dir / "results.jsonl"
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _record_key(record: dict) -> tuple[str | None, str | None, int | None]:
    return record.get("kind"), record.get("arm"), record.get("seed")


def _resume_identity(args, arms: list[Arm]) -> dict:
    return {
        "format_version": 1,
        "benchmark": "diffusion-diloco",
        "arguments": jsonable_arguments(
            args,
            exclude=_RESUME_IDENTITY_EXCLUDES,
        ),
        "seeds": parse_seeds(args.seeds),
        "effective_grad_accum": effective_grad_accum(
            args.micro_batch_size,
            args.grad_accum,
        ),
        "arms": [asdict(arm) for arm in arms],
        "implementation_sha256": implementation_fingerprint(
            REPO_ROOT,
            _IMPLEMENTATION_PATHS,
        ),
    }


def _expected_record_keys(args, arms: list[Arm]) -> set[tuple]:
    seeds = parse_seeds(args.seeds)
    keys = {("base", "base", None)}
    for seed in seeds:
        keys.update(
            ("baseline", f"baseline-m{learners}", seed)
            for learners in {arm.learners for arm in arms}
        )
        keys.update(("diloco", arm.name, seed) for arm in arms)
    return keys


def write_config(args, arms: list[Arm], data_manifest: dict) -> None:
    args.report_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "format_version": 2,
        "arguments": _jsonable_args(args),
        "seeds": parse_seeds(args.seeds),
        "effective_grad_accum": effective_grad_accum(
            args.micro_batch_size,
            args.grad_accum,
        ),
        "arms": [asdict(arm) for arm in arms],
        "resume_identity": _resume_identity(args, arms),
        "data_manifest": data_manifest,
        "fairness_contract": {
            "same_total_ranks": True,
            "same_per_gpu_batch": True,
            "same_sample_budget": True,
            "paired_rank_data_order": True,
            "paired_training_rng": True,
            "paired_eval_rng": True,
            "diloco_artifact": "syncer checkpoint export",
        },
    }
    write_json_atomic(args.report_dir / "config.json", config)


def load_resume_data(args, arms: list[Arm]) -> tuple[Path, Path, int]:
    manifest = load_resume_config(
        args.report_dir / "config.json",
        _resume_identity(args, arms),
    )
    return validate_data_manifest(args.work_dir, manifest)


def write_report(args, arms: list[Arm], records: list[dict]) -> list[dict]:
    write_partial_results(args.report_dir, records)

    aggregates = aggregate_records(records)
    with (args.report_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregates, handle, indent=2, sort_keys=True)
        handle.write("\n")

    lines = [
        f"# Diffusion DiLoCo benchmark: {args.model}",
        "",
        f"Sample budget: {args.sample_budget}; seeds: {args.seeds}; "
        f"eval rows/repeats: {args.eval_rows}/{args.eval_repeats}",
        "",
        "| arm | M | runs | GPUs | samples | train s | samples/s | GPU-h | cost | loss/elem | delta vs sync | mean H | participation | stale | sync GB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(value, digits=3):
        return "-" if value is None else f"{value:.{digits}f}"

    for item in aggregates:
        loss = (
            "-"
            if item["loss_mean"] is None
            else f"{item['loss_mean']:.6f} +/- {item['loss_std']:.6f}"
        )
        delta = (
            "-"
            if item["delta_mean_pct"] is None
            else f"{item['delta_mean_pct']:+.2f}% +/- {item['delta_std_pct']:.2f}"
        )
        participation = (
            None
            if item["participation_rate"] is None
            else 100.0 * item["participation_rate"]
        )
        sync_gb = (
            None if item["sync_bytes_mean"] is None else item["sync_bytes_mean"] / 1e9
        )
        lines.append(
            f"| {item['arm']} | {item['learners'] or '-'} | {item['runs']} "
            f"| {item['total_gpus'] or '-'} | {item['processed_samples'] or '-'} "
            f"| {item['wall_mean_s']:.1f} | {fmt(item['samples_per_s_mean'])} "
            f"| {item['gpu_hours_mean']:.3f} | {fmt(item['estimated_cost_mean'], 2)} "
            f"| {loss} | {delta} | {fmt(item['mean_h'], 2)} "
            f"| {fmt(participation, 1)} | {fmt(item['mean_staleness'], 2)} "
            f"| {fmt(sync_gb, 3)} |"
        )
    report = "\n".join(lines) + "\n"
    (args.report_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    return aggregates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model")
    parser.add_argument("--data", help="Hugging Face id, local path, or S3 prefix")
    parser.add_argument("--settings", default="m2")
    parser.add_argument("--seeds", default="17,29,43")
    parser.add_argument("--sample-budget", type=int, default=4096)
    parser.add_argument("--eval-rows", type=int, default=16)
    parser.add_argument("--eval-repeats", type=int, default=4)
    parser.add_argument("--eval-seed", type=int, default=12345)
    parser.add_argument("--max-train-rows", type=int, default=None)

    parser.add_argument("--shard", choices=["ddp", "fsdp"], default="fsdp")
    parser.add_argument("--learner-gpus", type=int, default=1)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--inner-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument(
        "--lora-targets",
        choices=["auto", "attention", "all-linear"],
        default="auto",
    )
    parser.add_argument("--fragments", type=int, default=8)
    parser.add_argument("--wan-streams", type=int, default=4)
    parser.add_argument("--grace-ms", type=int, default=1000)
    parser.add_argument(
        "--stream-workers",
        type=int,
        default=0,
        help="fixed at 0 so matching baseline and DiLoCo rank streams stay paired",
    )

    parser.add_argument("--diffusion-adapter", default=None)
    parser.add_argument("--cache-latents", action="store_true")
    parser.add_argument("--cache-text-embeds", action="store_true")
    parser.add_argument("--bucket-by-shape", action="store_true")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--video-column", default="video")
    parser.add_argument("--prompt-column", default="prompt")
    parser.add_argument("--latent-column", default="latents")
    parser.add_argument("--text-embeds-column", default="prompt_embeds")
    parser.add_argument("--text-attention-mask-column", default="prompt_attention_mask")
    parser.add_argument("--pooled-text-embeds-column", default="pooled_prompt_embeds")
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument(
        "--resize-mode",
        choices=["stretch", "center-crop"],
        default="stretch",
    )
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument(
        "--diffusion-loss-weighting",
        choices=["none", "linear", "sigma", "snr", "min-snr"],
        default="none",
    )
    parser.add_argument("--diffusion-min-snr-gamma", type=float, default=5.0)

    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--eval-device", choices=["cpu", "cuda"], default=None)
    parser.add_argument(
        "--eval-dtype",
        choices=["auto", "bf16", "fp16", "f32"],
        default="auto",
    )
    parser.add_argument("--export-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--gpu-hour-cost", type=float, default=None)
    parser.add_argument("--arm-timeout-min", type=int, default=240)
    parser.add_argument("--drain-seconds", type=float, default=3.0)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=REPO_ROOT / "diffusion-benchmark-work",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=REPO_ROOT / "diffusion-benchmark-report",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eval-payload", help=argparse.SUPPRESS)
    return parser


def validate_args(args, arms: list[Arm], *, check_devices: bool = True) -> None:
    if not args.model or not args.data:
        raise ValueError("--model and --data are required")
    parse_seeds(args.seeds)
    if args.height is None or args.width is None:
        raise ValueError(
            "pass explicit --height and --width so every arm trains the same shape"
        )
    if args.height < 1 or args.width < 1:
        raise ValueError("--height and --width must be positive")
    if args.num_frames is not None and args.num_frames < 1:
        raise ValueError("--num-frames must be positive")
    if args.fps is not None and args.fps <= 0:
        raise ValueError("--fps must be positive")
    for name in (
        "sample_budget",
        "eval_rows",
        "eval_repeats",
        "learner_gpus",
        "micro_batch_size",
        "grad_accum",
        "fragments",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.stream_workers != 0:
        raise ValueError(
            "paired diffusion benchmarks require --stream-workers 0"
        )
    if args.grace_ms < 0:
        raise ValueError("--grace-ms must be non-negative")
    if args.device == "cpu" and args.shard == "fsdp":
        raise ValueError("CPU benchmarks require --shard ddp")
    if args.eval_device is None:
        args.eval_device = args.device
    if check_devices and args.device.startswith("cuda"):
        import torch

        required = max(arm.learners for arm in arms) * args.learner_gpus
        available = torch.cuda.device_count()
        if available < required:
            raise ValueError(
                f"largest arm needs {required} GPUs, but torch sees {available}"
            )


def print_plan(args, arms: list[Arm]) -> None:
    seeds = parse_seeds(args.seeds)
    grad_accum = effective_grad_accum(args.micro_batch_size, args.grad_accum)
    print(
        f"[diffusion-benchmark] model={args.model} samples={args.sample_budget} "
        f"shape={args.height}x{args.width} seeds={seeds}"
    )
    for m in sorted({arm.learners for arm in arms}):
        ranks = m * args.learner_gpus
        steps = steps_for_samples(
            args.sample_budget,
            args.micro_batch_size,
            grad_accum,
            ranks,
        )
        actual = processed_samples(
            steps,
            args.micro_batch_size,
            grad_accum,
            ranks,
        )
        print(
            f"  baseline-m{m}: one process group, {ranks} ranks, "
            f"{steps} steps, {actual} samples"
        )
    for arm in arms:
        ranks = arm.learners * args.learner_gpus
        steps = steps_for_samples(
            args.sample_budget,
            args.micro_batch_size,
            grad_accum,
            ranks,
        )
        print(
            f"  {arm.name}: {arm.learners} islands x {args.learner_gpus} ranks, "
            f"{steps} steps/learner, P={arm.fragments}, H={arm.sync_interval_steps}, "
            f"alpha={arm.merge_alpha}, wire={arm.wire_dtype}"
        )
    print(f"  repetitions: {len(seeds)} training seed(s)")
    if grad_accum != args.grad_accum:
        print(
            f"  learner rebalance: requested grad_accum={args.grad_accum}, "
            f"effective grad_accum={grad_accum} at micro_batch={args.micro_batch_size}"
        )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.eval_payload:
        payload = json.loads(args.eval_payload)
        eval_args = SimpleNamespace(**payload["args"])
        result = evaluate_loss(
            eval_args,
            Path(payload["adapter_dir"]) if payload["adapter_dir"] else None,
            Path(payload["eval_data"]),
        )
        print("EVAL_JSON " + json.dumps(result, sort_keys=True))
        return 0

    try:
        arms = select_arms(args.settings, args.fragments)
        if args.overwrite and args.resume:
            raise ValueError("--overwrite and --resume are mutually exclusive")
        if args.eval_device is None:
            args.eval_device = args.device
        validate_args(args, arms, check_devices=not args.dry_run)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print_plan(args, arms)
    if args.dry_run:
        return 0

    if args.resume:
        if not args.work_dir.is_dir() or not args.report_dir.is_dir():
            raise SystemExit(
                "--resume requires the existing --work-dir and --report-dir"
            )
        try:
            train_data, eval_data, train_rows = load_resume_data(args, arms)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(
            "[diffusion-benchmark] resume manifest verified; "
            "reusing materialized splits"
        )
    else:
        if args.work_dir.exists():
            if not args.overwrite:
                raise SystemExit(
                    f"{args.work_dir} already exists; pass --overwrite to replace benchmark output"
                )
            shutil.rmtree(args.work_dir)
        if args.report_dir.exists():
            if not args.overwrite:
                raise SystemExit(
                    f"{args.report_dir} already exists; pass --overwrite to replace benchmark output"
                )
            shutil.rmtree(args.report_dir)
        args.work_dir.mkdir(parents=True, exist_ok=True)
        args.report_dir.mkdir(parents=True, exist_ok=True)
        try:
            materialized_data = materialize_data_source(
                args.data,
                args.work_dir / "source-data",
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        if materialized_data != args.data:
            args.materialized_data = materialized_data
        train_data, eval_data, train_rows = split_data(
            args,
            args.work_dir,
            materialized_data,
        )
        data_manifest = build_data_manifest(
            args.work_dir,
            train_data,
            eval_data,
            train_rows=train_rows,
            eval_rows=args.eval_rows,
            source=_source_root(materialized_data),
        )
        write_config(args, arms, data_manifest)

    consumers = max(arm.learners for arm in arms) * args.learner_gpus * max(
        1, args.stream_workers
    )
    if train_rows < consumers:
        raise SystemExit(
            f"only {train_rows} train rows for up to {consumers} learner/rank/worker "
            "consumers; use more rows or fewer --stream-workers"
        )
    print(
        f"[diffusion-benchmark] materialized {train_rows} train rows and "
        f"{args.eval_rows} held-out rows"
    )
    ensure_syncer()

    records = load_partial_results(args.report_dir) if args.resume else []
    if args.resume:
        try:
            validate_record_keys(records, _expected_record_keys(args, arms))
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    completed = {_record_key(record) for record in records}
    if ("base", "base", None) in completed:
        print("[diffusion-benchmark] resume: skipping completed base evaluation")
    else:
        base_eval = evaluate_in_subprocess(
            args,
            None,
            eval_data,
            args.work_dir / "base-eval.log",
        )
        records.append(
            make_record(
                kind="base",
                arm="base",
                seed=None,
                learners=0,
                run={},
                evaluation=base_eval,
                gpu_hour_cost=args.gpu_hour_cost,
            )
        )
        completed.add(("base", "base", None))
        write_partial_results(args.report_dir, records)
        print(
            f"[diffusion-benchmark] base loss/element={base_eval['loss_per_element']:.6f}"
        )

    distinct_m = sorted({arm.learners for arm in arms})
    for seed in parse_seeds(args.seeds):
        seed_dir = args.work_dir / f"seed-{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for m in distinct_m:
            key = ("baseline", f"baseline-m{m}", seed)
            if key in completed:
                print(
                    f"[diffusion-benchmark] resume: skipping seed={seed} baseline-m{m}",
                    flush=True,
                )
                continue
            print(f"[diffusion-benchmark] seed={seed} baseline-m{m}", flush=True)
            run = run_baseline(args, m, seed, train_data, seed_dir)
            evaluation = evaluate_in_subprocess(
                args,
                run["artifact"],
                eval_data,
                seed_dir / f"baseline-m{m}" / "eval.log",
            )
            records.append(
                make_record(
                    kind="baseline",
                    arm=f"baseline-m{m}",
                    seed=seed,
                    learners=m,
                    run=run,
                    evaluation=evaluation,
                    gpu_hour_cost=args.gpu_hour_cost,
                )
            )
            completed.add(key)
            write_partial_results(args.report_dir, records)
        for arm in arms:
            key = ("diloco", arm.name, seed)
            if key in completed:
                print(
                    f"[diffusion-benchmark] resume: skipping seed={seed} arm={arm.name}",
                    flush=True,
                )
                continue
            print(f"[diffusion-benchmark] seed={seed} arm={arm.name}", flush=True)
            run = run_diloco(args, arm, seed, train_data, seed_dir)
            evaluation = evaluate_in_subprocess(
                args,
                run["artifact"],
                eval_data,
                seed_dir / arm.name / "eval.log",
            )
            records.append(
                make_record(
                    kind="diloco",
                    arm=arm.name,
                    seed=seed,
                    learners=arm.learners,
                    run=run,
                    evaluation=evaluation,
                    gpu_hour_cost=args.gpu_hour_cost,
                )
            )
            completed.add(key)
            write_partial_results(args.report_dir, records)

    write_report(args, arms, records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
