#!/usr/bin/env python3
"""Benchmark LM DiLoCo against an equal-hardware synchronous baseline.

For an arm with M islands and G ranks per island, ``baseline-mM`` uses one
synchronous process group with M*G ranks while the DiLoCo arm uses M process
groups with G ranks each.  Both sides run the same optimizer-step count, use
the same per-rank batch, and therefore process the same raw-token budget.
For assistant-masked SFT the harness also pairs rank-local conversation streams
and rejects a run unless the positive-weight target-token counts match.

DiLoCo is always evaluated from the real Rust syncer's exported checkpoint,
never from a learner's locally blended adapter.  The final rows are held out
before training, and every artifact is scored on the same packed evaluation
tokens.  Repeated training seeds quantify optimization variance.

The harness partitions devices on one host; it does not provision cloud
instances or emulate WAN latency.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

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
    "adapter_dir",
    "dry_run",
    "eval_only",
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
    REPO_ROOT / "yeto/benchmark_resume.py",
    REPO_ROOT / "yeto/learner.py",
    REPO_ROOT / "yeto/data.py",
    REPO_ROOT / "yeto/losses.py",
    REPO_ROOT / "yeto/export.py",
    REPO_ROOT / "yeto/fragments.py",
    REPO_ROOT / "yeto/protocol.py",
    REPO_ROOT / "yeto/tensor_io.py",
    REPO_ROOT / "syncer/src",
    REPO_ROOT / "syncer/Cargo.toml",
    REPO_ROOT / "syncer/Cargo.lock",
)


@dataclass(frozen=True)
class Arm:
    """One DiLoCo configuration under test."""

    name: str
    learners: int = 2
    fragments: int = 8
    fragment_pattern: str = "binpack"
    matrix_merge: str = "rda"
    merge_alpha: float = 0.5
    wire_dtype: str = "bf16"
    pipeline: int = 2
    delta_correction: str = "heloco"
    quorum: int | None = None  # None -> all M learners each round
    outer_lr: float = 0.7
    outer_momentum: float = 0.9
    sync_interval_steps: float = 24.0


PRESETS: dict[str, Arm] = {
    "m2": Arm("m2"),
    "m4": Arm("m4", learners=4),
    "alpha0": Arm("alpha0", merge_alpha=0.0),
    "q4": Arm("q4", wire_dtype="q4"),
    "serial": Arm("serial", pipeline=1),
    "noheloco": Arm("noheloco", delta_correction="none"),
    "strided": Arm("strided", fragment_pattern="strided"),
    "iso": Arm("iso", matrix_merge="iso"),
    # Non-embedding fragments still use RDA; this removes Nesterov gain and
    # local blending so each merged RDA delta is applied directly.
    "direct-rda": Arm(
        "direct-rda", outer_lr=1.0, outer_momentum=0.0, merge_alpha=0.0
    ),
    "unthrottled": Arm("unthrottled", sync_interval_steps=0.0),
}


def select_arms(spec: str, fragments: int = 8) -> list[Arm]:
    names = (
        list(PRESETS)
        if spec == "all"
        else [value.strip() for value in spec.split(",") if value.strip()]
    )
    if not names:
        raise ValueError("--settings must select at least one benchmark arm")
    unknown = [n for n in names if n not in PRESETS]
    if unknown:
        raise ValueError(f"unknown settings {unknown}; choose from {list(PRESETS)}")
    return [replace(PRESETS[n], fragments=fragments) for n in names]


def parse_seeds(spec: str) -> list[int]:
    try:
        seeds = [int(value.strip()) for value in spec.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError("--seeds must be a comma-separated list of integers") from exc
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    if len(seeds) != len(set(seeds)):
        raise ValueError("--seeds contains duplicates")
    return seeds


def steps_for(
    token_budget: int,
    mbs: int,
    seq_len: int,
    learners: int,
    world: int = 1,
    grad_accum: int = 1,
) -> int:
    """Inner steps per learner so the arm consumes ~token_budget in total.
    `world` is the DDP/FSDP ranks per learner: every rank processes its own
    micro-batch per step, so tokens/step scale by world."""
    per_step = mbs * seq_len * learners * world * grad_accum
    if token_budget < 1 or per_step < 1:
        raise ValueError(
            "token budget, batch size, sequence length, and ranks must be positive"
        )
    return math.ceil(token_budget / per_step)


def processed_tokens(
    steps: int,
    mbs: int,
    seq_len: int,
    total_ranks: int,
    grad_accum: int = 1,
) -> int:
    return steps * mbs * seq_len * total_ranks * grad_accum


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


def gpu_env(
    learner_id: int, gpus_per_learner: int, device: str = "cuda"
) -> dict[str, str] | None:
    """Backward-compatible learner-indexed CUDA device partition."""
    if gpus_per_learner <= 0:
        return None
    return cuda_env(learner_id * gpus_per_learner, gpus_per_learner, device)


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
        "yeto.learner",
    ]


def learner_command(
    args,
    arm_dir: Path,
    *,
    learner_id: int,
    num_learners: int,
    syncer: str,
    max_steps: int,
    arm: Arm | None,
    nproc: int | None = None,
    seed: int = 0,
    train_data: Path | None = None,
) -> list[str]:
    if nproc is None:
        nproc = max(1, args.learner_gpus)
    cmd = _distributed_prefix(nproc)
    cmd += [
        "--model", args.model,
        "--data", str(train_data or arm_dir.parent / "train.jsonl"),
        "--syncer", syncer,
        "--learner-id", str(learner_id),
        "--num-learners", str(num_learners),
        "--tuning", "lora",
        "--base-quantization", getattr(args, "base_quantization", "none"),
        "--lora-r", str(args.lora_r),
        "--lora-alpha", str(args.lora_alpha),
        "--lora-targets", getattr(args, "lora_targets", "auto"),
        "--seq-len", str(args.seq_len),
        "--micro-batch-size", str(args.micro_batch_size),
        "--grad-accum", str(getattr(args, "grad_accum", 1)),
        "--inner-lr", str(args.inner_lr),
        "--weight-decay", str(getattr(args, "weight_decay", 0.01)),
        "--warmup-steps", str(getattr(args, "warmup_steps", 10)),
        "--seed", str(seed),
        "--max-local-steps", str(max_steps),
        "--tokenize", "stream",
        "--stream-workers", "0",
        "--train-on", getattr(args, "train_on", "assistant"),
        "--assistant-mask-mode", getattr(args, "assistant_mask_mode", "native"),
        "--gradient-checkpointing", getattr(args, "gradient_checkpointing", "auto"),
        "--wan-streams", str(getattr(args, "wan_streams", 4)),
        "--shard", args.shard,
        "--output-dir", str(arm_dir / f"learner-{learner_id}"),
    ]
    if nproc == 1 or not args.device.startswith("cuda"):
        cmd += ["--device", args.device]
    if arm is not None:
        cmd += [
            "--fragments", str(arm.fragments),
            "--fragment-pattern", arm.fragment_pattern,
            "--matrix-merge", arm.matrix_merge,
            "--merge-alpha", str(arm.merge_alpha),
            "--wire-dtype", arm.wire_dtype,
        ]
    return cmd


def syncer_command(
    arm: Arm,
    port: int,
    arm_dir: Path,
    total_steps: int,
    *,
    grace_ms: int = 1000,
) -> list[str]:
    # The syncer takes no fragment count: the layout arrives in HELLO.
    return [
        str(SYNCER_BIN),
        "--port", str(port),
        "--learners", str(arm.learners),
        "--quorum", str(arm.quorum or arm.learners),
        "--grace-ms", str(grace_ms),
        "--total-steps", str(total_steps),
        "--pipeline", str(arm.pipeline),
        "--delta-correction", arm.delta_correction,
        "--outer-lr", str(arm.outer_lr),
        "--outer-momentum", str(arm.outer_momentum),
        "--checkpoint-path", str(arm_dir / "state.ckpt"),
        "--checkpoint-every", "1",
        "--event-tape", str(arm_dir / "tape.jsonl"),
        "--sync-interval-steps", str(arm.sync_interval_steps),
    ]


def free_port() -> int:
    while True:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = int(s.getsockname()[1])
        if port not in _USED_PORTS:
            _USED_PORTS.add(port)
            return port


def materialize_data_source(data: str, destination: Path) -> str:
    """Stage an S3 dataset prefix locally; pass other sources through."""
    if not data.startswith("s3://"):
        return data
    if shutil.which("aws") is None:
        raise RuntimeError(
            "S3 benchmark data requires the AWS CLI and ambient read credentials"
        )
    destination.mkdir(parents=True, exist_ok=True)
    print(f"[lm-benchmark] syncing {data} to {destination}", flush=True)
    try:
        subprocess.run(
            ["aws", "s3", "sync", data, str(destination), "--only-show-errors"],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"failed to sync benchmark data from {data} (aws exit {exc.returncode})"
        ) from exc
    if not any(destination.iterdir()):
        raise RuntimeError(f"S3 dataset prefix {data} contains no objects")
    return str(destination.resolve())


def split_data(
    data: str, work: Path, eval_rows: int, max_rows: int | None
) -> tuple[Path, Path, int]:
    """Materialize --data as train.jsonl / eval.jsonl under `work`.

    The eval split comes off the END of the row stream so every arm trains
    on an identical prefix and none has seen the eval rows.
    """
    from yeto.data import load_rows

    ds = load_rows(data)
    n = len(ds)
    if max_rows is not None:
        n = min(n, max_rows + eval_rows)
    if n <= eval_rows:
        raise SystemExit(f"--data has {n} usable rows; need > --eval-rows {eval_rows}")
    for index in range(n):
        if not ds[index].get("messages"):
            raise SystemExit(
                f"--data row {index} has no messages; the LM benchmark requires "
                "messages-format conversation rows"
            )
    work.mkdir(parents=True, exist_ok=True)

    def dump(path: Path, idxs) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for i in idxs:
                row = ds[i]
                materialized = {
                    key: row[key] for key in ("messages", "tools") if key in row
                }
                handle.write(json.dumps(materialized) + "\n")

    train, evalf = work / "train.jsonl", work / "eval.jsonl"
    dump(train, range(n - eval_rows))
    dump(evalf, range(n - eval_rows, n))
    return train, evalf, n - eval_rows


def evaluate_loss(model_id: str, adapter_dir: Path | None, eval_file: Path,
                  seq_len: int, device: str, train_on: str = "assistant",
                  base_quantization: str = "none",
                  assistant_mask_mode: str = "native") -> dict:
    """Held-out masked CE per trained token — the comparison metric."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from yeto.data import build_packed_dataset
    from yeto.learner import _from_pretrained_offline_first
    from yeto.losses import sft_loss
    from yeto.models import resolve

    resolved = resolve(model_id)
    tok = _from_pretrained_offline_first(
        AutoTokenizer, resolved, trust_remote_code=True
    )
    if base_quantization == "nf4":
        if device == "cpu":
            raise ValueError("NF4 benchmark evaluation requires CUDA")
        from transformers import BitsAndBytesConfig

        model = _from_pretrained_offline_first(
            AutoModelForCausalLM,
            resolved,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            ),
            device_map={"": 0},
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    else:
        # bf16 on accelerators (a 10B+ base in fp32 would not fit one GPU);
        # fp32 on cpu, where bf16 matmuls are slow and memory is plentiful.
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        model = _from_pretrained_offline_first(
            AutoModelForCausalLM,
            resolved,
            dtype=dtype,
            trust_remote_code=True,
        )
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir))
    if base_quantization == "none":
        model.to(device)
    model.eval()
    ds = build_packed_dataset(
        str(eval_file),
        tok,
        0,
        1,
        seq_len,
        train_on=train_on,
        assistant_mask_mode=assistant_mask_mode,
    )
    total_loss, total_tokens = 0.0, 0.0
    block_losses = []
    with torch.no_grad():
        for i in range(len(ds)):
            ids, weights = ds[i]
            ids = ids.unsqueeze(0).to(device)
            weights = weights.unsqueeze(0).to(device)
            out = model(input_ids=ids)
            loss, n = sft_loss(out.logits, ids, "cross_entropy", weights)
            total_loss += loss.item()
            total_tokens += n.item()
            if n.item() > 0:
                block_losses.append(loss.item() / n.item())
    if total_tokens <= 0:
        raise ValueError(
            "held-out rows contain no positive-weight target tokens; use rows "
            "with assistant responses or pass --train-on all"
        )
    loss_per_token = total_loss / total_tokens
    return {
        "loss_per_token": loss_per_token,
        "perplexity": math.exp(loss_per_token) if loss_per_token < 80 else None,
        "total_loss": total_loss,
        "trained_tokens": int(total_tokens),
        "blocks": len(ds),
        "block_mean": statistics.fmean(block_losses) if block_losses else None,
        "block_std": statistics.stdev(block_losses) if len(block_losses) > 1 else 0.0,
    }


def eval_loss_per_token(model_id: str, adapter_dir: Path | None, eval_file: Path,
                        seq_len: int, device: str, train_on: str = "assistant",
                        assistant_mask_mode: str = "native") -> float:
    """Compatibility wrapper for callers that only need the scalar metric."""
    return evaluate_loss(
        model_id,
        adapter_dir,
        eval_file,
        seq_len,
        device,
        train_on,
        assistant_mask_mode=assistant_mask_mode,
    )["loss_per_token"]


def _visible_gpu_uuids() -> set[str] | None:
    visible = _visible_cuda_devices()
    if visible is None:
        return None
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
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
    """Block until no compute process holds more than `limit_mb` on any GPU.

    Arms and evals run strictly one after another, but a just-exited CUDA
    process's memory is not always released by the driver the instant
    subprocess.run returns — spawning the next arm into that window OOMs
    (observed on 4xL40S: the eval child's 25 GB were still resident when
    the baseline learner loaded). Fails loudly with the offending pids.
    """
    if not device.startswith("cuda"):
        return
    visible_uuids = _visible_gpu_uuids()
    deadline = time.monotonic() + timeout_s
    last = ""
    while True:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        ).stdout.strip()
        holders = []
        for line in out.splitlines():
            parts = [v.strip() for v in line.split(",")]
            if len(parts) < 4:
                continue
            gpu_uuid, pid, name, mem = parts[0], parts[1], parts[2], parts[-1]
            if visible_uuids is not None and gpu_uuid not in visible_uuids:
                continue
            # Drivers that report [N/A] per-process memory would otherwise
            # slip a fully-loaded process past the numeric check — ANY
            # listed compute app counts as occupying the GPU.
            if not mem.isdigit() or int(mem) > limit_mb:
                holders.append(f"pid {pid} ({name}): {mem} MiB")
        if not holders:
            return
        if last != "; ".join(holders):
            last = "; ".join(holders)
            print(f"[compare] waiting for GPUs to drain: {last}", flush=True)
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"GPUs still occupied after {timeout_s}s: {'; '.join(holders)}"
            )
        time.sleep(3)


def eval_in_subprocess(args, adapter_dir: Path | None, eval_file: Path,
                       log_path: Path | None = None) -> dict:
    """Score in a child process so the model's VRAM is released on exit —
    an in-process eval would keep the base resident on GPU 0 and starve the
    next arm's learners (found the hard way on a 4xL40S box)."""
    cmd = [
        sys.executable, __file__, "--eval-only",
        "--model", args.model,
        "--data", str(eval_file),
        "--seq-len", str(args.seq_len),
        "--device", args.eval_device,
        "--train-on", args.train_on,
        "--assistant-mask-mode", args.assistant_mask_mode,
        "--base-quantization", args.base_quantization,
    ]
    if adapter_dir is not None:
        cmd += ["--adapter-dir", str(adapter_dir)]
    wait_for_free_gpus(args.eval_device)
    env = (
        cuda_env(0, 1, args.eval_device)
        if args.eval_device.startswith("cuda")
        else None
    )
    out = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, env=env
    )
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(out.stdout + out.stderr, encoding="utf-8")
    # The child has exited, but the driver may release its VRAM lazily;
    # don't hand the GPUs to the next arm until it is actually gone.
    wait_for_free_gpus(args.eval_device)
    for line in reversed(out.stdout.splitlines()):
        if line.startswith("EVAL_JSON "):
            return json.loads(line.removeprefix("EVAL_JSON "))
    raise RuntimeError(
        f"eval subprocess failed ({out.returncode}):\n{out.stdout[-800:]}\n{out.stderr[-800:]}"
    )


def _tail(path: Path, lines: int = 16) -> str:
    try:
        return "\n".join(
            path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
        )
    except OSError:
        return ""


_TRAINING_METRICS_RE = re.compile(
    r"inner loop done [^\r\n]*?raw_tokens=(\d+) target_tokens=(\d+)"
)
_METRICS_VERSION_RE = re.compile(r"\bmetrics_version=(\d+)\b")
_METRICS_SCOPE_RE = re.compile(r"\bmetrics_scope=([a-z][a-z0-9_-]*)\b")
_SUPPORTED_TRAINING_TELEMETRY = {(1, "rank"), (2, "island")}


def _training_telemetry_schema(record: str) -> tuple[int, str]:
    versions = _METRICS_VERSION_RE.findall(record)
    scopes = _METRICS_SCOPE_RE.findall(record)
    if not versions and not scopes:
        return 1, "rank"
    if len(versions) != 1 or len(scopes) != 1:
        raise RuntimeError(
            "learner log contains malformed final token telemetry; versioned "
            "records require one metrics_version and one metrics_scope"
        )
    schema = int(versions[0]), scopes[0]
    if schema not in _SUPPORTED_TRAINING_TELEMETRY:
        raise RuntimeError(
            "learner log uses unsupported final token telemetry schema "
            f"version={schema[0]} scope={schema[1]!r}"
        )
    return schema


def summarize_training_logs(paths: list[Path]) -> dict:
    """Sum compatible rank- or island-scoped final token telemetry."""
    raw_tokens = 0
    target_tokens = 0
    reported_units = 0
    schema: tuple[int, str] | None = None
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in _TRAINING_METRICS_RE.finditer(text):
            record_schema = _training_telemetry_schema(match.group(0))
            if schema is None:
                schema = record_schema
            elif record_schema != schema:
                raise RuntimeError(
                    "learner logs mix final token telemetry schemas; refusing "
                    "to combine rank- and island-scoped counters"
                )
            reported_units += 1
            raw_tokens += int(match.group(1))
            target_tokens += int(match.group(2))
    if reported_units == 0:
        raise RuntimeError(
            "learner logs contain no final token telemetry; the run may have "
            "used an incompatible learner version"
        )
    if target_tokens == 0:
        raise RuntimeError(
            "training processed no positive-weight target tokens; use rows "
            "with assistant responses or pass --train-on all"
        )
    assert schema is not None
    version, scope = schema
    return {
        "telemetry_version": version,
        "telemetry_scope": scope,
        "reported_units": reported_units,
        "processed_tokens": raw_tokens,
        "processed_target_tokens": target_tokens,
        "target_density": target_tokens / raw_tokens if raw_tokens else None,
    }


def validate_training_telemetry_units(
    telemetry: dict,
    *,
    label: str,
    total_ranks: int,
    islands: int,
) -> None:
    """Require complete final telemetry for the schema's accounting scope."""
    scope = telemetry.get("telemetry_scope")
    expected = {"rank": total_ranks, "island": islands}.get(scope)
    if expected is None:
        raise RuntimeError(f"{label}: unknown telemetry scope {scope!r}")
    actual = telemetry.get("reported_units")
    if actual != expected:
        raise RuntimeError(
            f"{label}: expected {expected} {scope}-scoped telemetry records, "
            f"found {actual}"
        )


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


def run_checked(cmd: list[str], log: Path, env: dict | None = None,
                timeout_s: int | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
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
            raise RuntimeError(f"command timed out: {' '.join(cmd)}\n{_tail(log)}")
    if returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {returncode}: {' '.join(cmd)}\n{_tail(log)}"
        )


def _wait_for_learners(processes: list[subprocess.Popen], logs: list[Path],
                       timeout_s: int) -> None:
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
        if time.monotonic() >= deadline:
            raise RuntimeError(f"learners timed out after {timeout_s}s")
        time.sleep(1)


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
    c_tokens = [
        int(responder.get("c_tokens", 0))
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
        "mean_tokens_per_response": (
            statistics.fmean(c_tokens) if c_tokens else None
        ),
        "median_tokens_per_response": (
            statistics.median(c_tokens) if c_tokens else None
        ),
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


def estimate_tensor_bytes(fragment_numels: list[int], tape_path: Path,
                          wire_dtype: str, learners: int) -> int:
    records = []
    if tape_path.exists():
        for line in tape_path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    init = sum(_tensor_bytes(n, wire_dtype, broadcast=True) for n in fragment_numels)
    total = init + learners * init
    for record in records:
        fragment = int(record.get("fragment", -1))
        if not 0 <= fragment < len(fragment_numels):
            continue
        numel = fragment_numels[fragment]
        total += len(record.get("responders", [])) * _tensor_bytes(
            numel, wire_dtype, broadcast=False
        )
        total += learners * _tensor_bytes(numel, wire_dtype, broadcast=True)
    return total


def run_baseline(args, m: int, seed: int, train_data: Path, seed_dir: Path) -> dict:
    run_dir = seed_dir / f"baseline-m{m}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    total_ranks = m * args.learner_gpus
    steps = steps_for(
        args.token_budget,
        args.micro_batch_size,
        args.seq_len,
        1,
        world=total_ranks,
        grad_accum=args.grad_accum,
    )
    command = learner_command(
        args,
        run_dir,
        learner_id=0,
        num_learners=1,
        syncer="none",
        max_steps=steps,
        arm=None,
        nproc=total_ranks,
        seed=seed,
        train_data=train_data,
    )
    wait_for_free_gpus(args.device)
    started = time.monotonic()
    learner_log = run_dir / "learner.log"
    run_checked(
        command,
        learner_log,
        env=cuda_env(0, total_ranks, args.device),
        timeout_s=args.arm_timeout_min * 60,
    )
    wall = time.monotonic() - started
    telemetry = summarize_training_logs([learner_log])
    validate_training_telemetry_units(
        telemetry,
        label=f"baseline-m{m}",
        total_ranks=total_ranks,
        islands=1,
    )
    expected_tokens = processed_tokens(
        steps,
        args.micro_batch_size,
        args.seq_len,
        total_ranks,
        args.grad_accum,
    )
    if telemetry["processed_tokens"] != expected_tokens:
        raise RuntimeError(
            f"baseline-m{m}: expected {expected_tokens} raw tokens, learner "
            f"reported {telemetry['processed_tokens']}"
        )
    return {
        "artifact": run_dir / "learner-0",
        "wall_s": wall,
        "steps_per_learner": steps,
        "total_ranks": total_ranks,
        "total_gpus": total_ranks if args.device.startswith("cuda") else 0,
        **telemetry,
    }


def run_diloco(args, arm: Arm, seed: int, train_data: Path, seed_dir: Path) -> dict:
    run_dir = seed_dir / arm.name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    total_ranks = arm.learners * args.learner_gpus
    steps = steps_for(
        args.token_budget,
        args.micro_batch_size,
        args.seq_len,
        arm.learners,
        world=args.learner_gpus,
        grad_accum=args.grad_accum,
    )
    port = free_port()
    syncer_steps = max(arm.fragments, steps * arm.learners * arm.fragments * 2)
    syncer_log_path = run_dir / "syncer.log"
    syncer_log = syncer_log_path.open("w", encoding="utf-8")
    syncer = subprocess.Popen(
        syncer_command(
            arm,
            port,
            run_dir,
            total_steps=syncer_steps,
            grace_ms=args.grace_ms,
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
            command = learner_command(
                args,
                run_dir,
                learner_id=learner_id,
                num_learners=arm.learners,
                syncer=f"127.0.0.1:{port}",
                max_steps=steps,
                arm=arm,
                nproc=args.learner_gpus,
                seed=seed,
                train_data=train_data,
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
        _wait_for_learners(processes, learner_logs, args.arm_timeout_min * 60)
        time.sleep(args.drain_seconds)
        if syncer.poll() is not None:
            raise RuntimeError(
                f"{arm.name}: syncer exited before the token budget ended\n"
                f"{_tail(syncer_log_path)}"
            )
    finally:
        for process in processes:
            _stop_process(process)
        _stop_process(syncer)
        for handle in handles:
            handle.close()
        syncer_log.close()
    wall = time.monotonic() - started
    telemetry = summarize_training_logs(learner_logs)
    validate_training_telemetry_units(
        telemetry,
        label=arm.name,
        total_ranks=total_ranks,
        islands=arm.learners,
    )
    expected_tokens = processed_tokens(
        steps,
        args.micro_batch_size,
        args.seq_len,
        total_ranks,
        args.grad_accum,
    )
    if telemetry["processed_tokens"] != expected_tokens:
        raise RuntimeError(
            f"{arm.name}: expected {expected_tokens} raw tokens, learner "
            f"reported {telemetry['processed_tokens']}"
        )

    checkpoint = run_dir / "state.ckpt"
    if not checkpoint.exists():
        raise RuntimeError(
            f"{arm.name}: no syncer checkpoint was produced\n{_tail(syncer_log_path)}"
        )
    export_dir = run_dir / "export"
    wait_for_free_gpus(args.export_device)
    export_started = time.monotonic()
    run_checked(
        [
            sys.executable,
            "-m",
            "yeto.export",
            "--checkpoint",
            str(checkpoint),
            "--model",
            args.model,
            "--tuning",
            "lora",
            "--base-quantization",
            args.base_quantization,
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
            "--matrix-merge",
            arm.matrix_merge,
            "--output-dir",
            str(export_dir),
            "--device",
            args.export_device,
        ],
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
        fragment_numels, tape_path, arm.wire_dtype, arm.learners
    )
    return {
        "artifact": export_dir,
        "wall_s": wall,
        "export_s": export_s,
        "steps_per_learner": steps,
        "total_ranks": total_ranks,
        "total_gpus": total_ranks if args.device.startswith("cuda") else 0,
        "global_step": parsed.global_step,
        "fragment_versions": [version for version, _, _ in parsed.fragments],
        "tape": tape,
        **telemetry,
    }


def ensure_syncer() -> None:
    if SYNCER_BIN.exists():
        return
    print("[lm-benchmark] building Rust syncer", flush=True)
    subprocess.run(
        ["cargo", "build", "--release", "--quiet"],
        cwd=REPO_ROOT / "syncer",
        check=True,
    )


def make_record(*, kind: str, arm: str, seed: int | None, learners: int,
                run: dict, evaluation: dict,
                gpu_hour_cost: float | None) -> dict:
    wall_s = float(run.get("wall_s", 0.0))
    total_gpus = int(run.get("total_gpus", 0))
    gpu_hours = total_gpus * wall_s / 3600.0
    tokens = int(run.get("processed_tokens", 0))
    target_tokens = int(run.get("processed_target_tokens", 0))
    return {
        "kind": kind,
        "arm": arm,
        "seed": seed,
        "learners": learners,
        "total_ranks": int(run.get("total_ranks", 0)),
        "total_gpus": total_gpus,
        "steps_per_learner": int(run.get("steps_per_learner", 0)),
        "processed_tokens": tokens,
        "processed_target_tokens": target_tokens,
        "target_density": run.get("target_density"),
        "wall_s": wall_s,
        "export_s": float(run.get("export_s", 0.0)),
        "tokens_per_s": tokens / wall_s if wall_s > 0 else None,
        "target_tokens_per_s": target_tokens / wall_s if wall_s > 0 else None,
        "gpu_hours": gpu_hours,
        "estimated_cost": (
            gpu_hours * gpu_hour_cost if gpu_hour_cost is not None else None
        ),
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
        (record["seed"], record["learners"]): record["eval"]["loss_per_token"]
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
        losses = [record["eval"]["loss_per_token"] for record in group]
        loss_mean, loss_std = _mean_std(losses)
        perplexities = [
            record["eval"]["perplexity"]
            for record in group
            if record["eval"].get("perplexity") is not None
        ]
        perplexity_mean, perplexity_std = _mean_std(perplexities)
        deltas = []
        if kind == "diloco":
            for record in group:
                baseline = baselines[(record["seed"], learners)]
                deltas.append(
                    100.0
                    * (record["eval"]["loss_per_token"] - baseline)
                    / baseline
                )
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
                "processed_tokens": group[0]["processed_tokens"],
                "target_tokens_mean": statistics.fmean(
                    record.get("processed_target_tokens", 0) for record in group
                ),
                "target_density_mean": statistics.fmean(
                    record["target_density"]
                    for record in group
                    if record.get("target_density") is not None
                )
                if any(record.get("target_density") is not None for record in group)
                else None,
                "loss_mean": loss_mean,
                "loss_std": loss_std,
                "perplexity_mean": perplexity_mean,
                "perplexity_std": perplexity_std,
                "delta_mean_pct": delta_mean,
                "delta_std_pct": delta_std,
                "wall_mean_s": statistics.fmean(record["wall_s"] for record in group),
                "tokens_per_s_mean": (
                    statistics.fmean(
                        record["tokens_per_s"]
                        for record in group
                        if record["tokens_per_s"] is not None
                    )
                    if any(record["tokens_per_s"] is not None for record in group)
                    else None
                ),
                "target_tokens_per_s_mean": (
                    statistics.fmean(
                        record["target_tokens_per_s"]
                        for record in group
                        if record.get("target_tokens_per_s") is not None
                    )
                    if any(
                        record.get("target_tokens_per_s") is not None
                        for record in group
                    )
                    else None
                ),
                "gpu_hours_mean": statistics.fmean(record["gpu_hours"] for record in group),
                "estimated_cost_mean": (
                    statistics.fmean(
                        record["estimated_cost"]
                        for record in group
                        if record["estimated_cost"] is not None
                    )
                    if any(record["estimated_cost"] is not None for record in group)
                    else None
                ),
                "mean_h": tape_mean("mean_h"),
                "mean_tokens_per_response": tape_mean("mean_tokens_per_response"),
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


def validate_target_token_match(
    records: list[dict], arm: Arm, seed: int, run: dict
) -> None:
    baseline = next(
        (
            record
            for record in records
            if record.get("kind") == "baseline"
            and record.get("seed") == seed
            and record.get("learners") == arm.learners
        ),
        None,
    )
    if baseline is None:
        raise RuntimeError(
            f"{arm.name}: missing seed={seed} baseline-m{arm.learners} record"
        )
    expected = baseline.get("processed_target_tokens")
    actual = run.get("processed_target_tokens")
    if expected != actual:
        raise RuntimeError(
            f"{arm.name}: target-token mismatch against baseline-m{arm.learners} "
            f"for seed={seed}: baseline={expected}, arm={actual}; data order is "
            "not paired, so this run is invalid"
        )


def validate_eval_token_match(records: list[dict], evaluation: dict) -> None:
    base = next(
        (record for record in records if record.get("kind") == "base"),
        None,
    )
    if base is None:
        raise RuntimeError("missing base evaluation record")
    expected = base["eval"]["trained_tokens"]
    actual = evaluation["trained_tokens"]
    if actual != expected:
        raise RuntimeError(
            f"held-out target-token mismatch: base={expected}, artifact={actual}"
        )


def _jsonable_args(args) -> dict:
    return jsonable_arguments(args, exclude={"eval_only", "adapter_dir"})


def _resume_identity(args, arms: list[Arm]) -> dict:
    return {
        "format_version": 1,
        "benchmark": "lm-diloco",
        "arguments": jsonable_arguments(
            args,
            exclude=_RESUME_IDENTITY_EXCLUDES,
        ),
        "seeds": parse_seeds(args.seeds),
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
        "arms": [asdict(arm) for arm in arms],
        "resume_identity": _resume_identity(args, arms),
        "data_manifest": data_manifest,
        "fairness_contract": {
            "same_total_ranks": True,
            "same_per_rank_batch": True,
            "same_optimizer_steps": True,
            "same_raw_token_budget": True,
            "same_training_target_tokens": True,
            "same_held_out_eval_tokens": True,
            "paired_rank_data_order": True,
            "repeated_training_seeds": True,
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
    (args.report_dir / "summary.json").write_text(
        json.dumps(aggregates, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        f"# LM DiLoCo benchmark: {args.model}",
        "",
        f"Raw-token budget: {args.token_budget}; seeds: {args.seeds}; "
        f"held-out rows: {args.eval_rows}",
        "",
        "## Quality And Token Accounting",
        "",
        "| arm | M | runs | GPUs | raw tokens | target tokens | target % "
        "| CE/token | perplexity | delta vs sync |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
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
        perplexity = (
            "-"
            if item["perplexity_mean"] is None
            else f"{item['perplexity_mean']:.3f} +/- {item['perplexity_std']:.3f}"
        )
        target_density = (
            None
            if item["target_density_mean"] is None
            else 100.0 * item["target_density_mean"]
        )
        target_tokens = (
            "-" if item["kind"] == "base" else f"{item['target_tokens_mean']:.0f}"
        )
        lines.append(
            f"| {item['arm']} | {item['learners'] or '-'} | {item['runs']} "
            f"| {item['total_gpus'] or '-'} | {item['processed_tokens'] or '-'} "
            f"| {target_tokens} | {fmt(target_density, 1)} "
            f"| {loss} | {perplexity} | {delta} |"
        )

    lines.extend(
        [
            "",
            "## Systems Diagnostics",
            "",
            "| arm | M | train s | raw tok/s | target tok/s | GPU-h | cost "
            "| mean H | tokens/response | participation | stale | sync GB |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in aggregates:
        participation = (
            None
            if item["participation_rate"] is None
            else 100.0 * item["participation_rate"]
        )
        sync_gb = (
            None if item["sync_bytes_mean"] is None else item["sync_bytes_mean"] / 1e9
        )
        lines.append(
            f"| {item['arm']} | {item['learners'] or '-'} "
            f"| {item['wall_mean_s']:.1f} | {fmt(item['tokens_per_s_mean'], 1)} "
            f"| {fmt(item['target_tokens_per_s_mean'], 1)} "
            f"| {item['gpu_hours_mean']:.3f} | {fmt(item['estimated_cost_mean'], 2)} "
            f"| {fmt(item['mean_h'], 2)} "
            f"| {fmt(item['mean_tokens_per_response'], 0)} "
            f"| {fmt(participation, 1)} | {fmt(item['mean_staleness'], 2)} "
            f"| {fmt(sync_gb, 3)} |"
        )
    report = "\n".join(lines) + "\n"
    (args.report_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    return aggregates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="lfm25-230m")
    parser.add_argument(
        "--data", required=True, help="messages-format HF id, local path, or S3 prefix"
    )
    parser.add_argument("--settings", default="m2")
    parser.add_argument("--seeds", default="17,29,43")
    parser.add_argument("--token-budget", type=int, default=500_000)
    parser.add_argument("--eval-rows", type=int, default=64)
    parser.add_argument("--max-rows", type=int, default=None, help="cap training rows")

    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--inner-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--train-on", choices=["assistant", "all"], default="assistant")
    parser.add_argument(
        "--assistant-mask-mode",
        choices=["native", "legacy"],
        default="native",
        help="assistant-only masking mode; keep fixed between training and evaluation",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument(
        "--base-quantization", choices=["none", "nf4"], default="none"
    )
    parser.add_argument(
        "--lora-targets",
        choices=["auto", "attention", "all-linear"],
        default="auto",
    )
    parser.add_argument("--fragments", type=int, default=8)
    parser.add_argument("--wan-streams", type=int, default=4)
    parser.add_argument("--grace-ms", type=int, default=1000)
    parser.add_argument(
        "--gradient-checkpointing", choices=["auto", "on", "off"], default="auto"
    )

    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--eval-device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--export-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--shard", choices=["ddp", "fsdp"], default="ddp")
    parser.add_argument(
        "--learner-gpus",
        type=int,
        default=1,
        help="ranks per island; on CUDA, each rank owns one GPU",
    )
    parser.add_argument("--gpu-hour-cost", type=float, default=None)
    parser.add_argument("--arm-timeout-min", type=int, default=240)
    parser.add_argument("--drain-seconds", type=float, default=3.0)
    parser.add_argument(
        "--work-dir", type=Path, default=REPO_ROOT / "lm-benchmark-work"
    )
    parser.add_argument(
        "--report-dir", type=Path, default=REPO_ROOT / "lm-benchmark-report"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eval-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--adapter-dir", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def validate_args(args, arms: list[Arm], *, check_devices: bool = True) -> None:
    parse_seeds(args.seeds)
    for name in (
        "token_budget",
        "eval_rows",
        "seq_len",
        "micro_batch_size",
        "grad_accum",
        "learner_gpus",
        "fragments",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_rows is not None and args.max_rows < 1:
        raise ValueError("--max-rows must be positive")
    if args.inner_lr <= 0 or args.weight_decay < 0:
        raise ValueError("--inner-lr must be positive and --weight-decay non-negative")
    if args.warmup_steps < 0 or args.arm_timeout_min <= 0:
        raise ValueError("--warmup-steps must be non-negative and timeout positive")
    if args.lora_r < 1 or args.lora_alpha < 1 or args.wan_streams < 1:
        raise ValueError("LoRA dimensions and --wan-streams must be positive")
    if args.gpu_hour_cost is not None and args.gpu_hour_cost < 0:
        raise ValueError("--gpu-hour-cost must be non-negative")
    if args.grace_ms < 0 or args.drain_seconds < 0:
        raise ValueError("--grace-ms and --drain-seconds must be non-negative")
    if args.device == "cpu" and args.shard == "fsdp":
        raise ValueError("CPU benchmarks require --shard ddp")
    if args.eval_device is None:
        args.eval_device = args.device
    if check_devices and any(
        value.startswith("cuda")
        for value in (args.device, args.eval_device, args.export_device)
    ):
        import torch

        required = (
            max(arm.learners for arm in arms) * args.learner_gpus
            if args.device.startswith("cuda")
            else 1
        )
        available = torch.cuda.device_count()
        if available < required:
            raise ValueError(
                f"largest arm needs {required} GPUs, but torch sees {available}"
            )


def print_plan(args, arms: list[Arm]) -> None:
    seeds = parse_seeds(args.seeds)
    print(
        f"[lm-benchmark] model={args.model} tokens={args.token_budget} "
        f"seq_len={args.seq_len} train_on={args.train_on} "
        f"assistant_mask_mode={args.assistant_mask_mode} "
        f"stream_workers=0 seeds={seeds}"
    )
    for m in sorted({arm.learners for arm in arms}):
        ranks = m * args.learner_gpus
        steps = steps_for(
            args.token_budget,
            args.micro_batch_size,
            args.seq_len,
            1,
            world=ranks,
            grad_accum=args.grad_accum,
        )
        actual = processed_tokens(
            steps,
            args.micro_batch_size,
            args.seq_len,
            ranks,
            args.grad_accum,
        )
        print(
            f"  baseline-m{m}: one process group, {ranks} ranks, "
            f"{steps} steps, {actual} raw tokens"
        )
    for arm in arms:
        ranks = arm.learners * args.learner_gpus
        steps = steps_for(
            args.token_budget,
            args.micro_batch_size,
            args.seq_len,
            arm.learners,
            world=args.learner_gpus,
            grad_accum=args.grad_accum,
        )
        print(
            f"  {arm.name}: {arm.learners} islands x {args.learner_gpus} ranks, "
            f"{steps} steps/island, P={arm.fragments}, H={arm.sync_interval_steps}, "
            f"alpha={arm.merge_alpha}, wire={arm.wire_dtype}, "
            f"merge={arm.matrix_merge}"
        )
    print(f"  repetitions: {len(seeds)} training seed(s)")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.eval_only:
        result = evaluate_loss(
            args.model,
            args.adapter_dir,
            Path(args.data),
            args.seq_len,
            args.device,
            args.train_on,
            args.base_quantization,
            args.assistant_mask_mode,
        )
        print("EVAL_JSON " + json.dumps(result, sort_keys=True))
        return 0

    try:
        arms = select_arms(args.settings, args.fragments)
        if args.overwrite and args.resume:
            raise ValueError("--overwrite and --resume are mutually exclusive")
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
        print("[lm-benchmark] resume manifest verified; reusing materialized splits")
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
                args.data, args.work_dir / "source-data"
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        if materialized_data != args.data:
            args.materialized_data = materialized_data
        train_data, eval_data, train_rows = split_data(
            materialized_data,
            args.work_dir,
            args.eval_rows,
            args.max_rows,
        )
        data_manifest = build_data_manifest(
            args.work_dir,
            train_data,
            eval_data,
            train_rows=train_rows,
            eval_rows=args.eval_rows,
            source=materialized_data,
        )
        write_config(args, arms, data_manifest)

    consumers = max(arm.learners for arm in arms) * args.learner_gpus
    if train_rows < consumers:
        raise SystemExit(
            f"only {train_rows} train rows for up to {consumers} learner/rank consumers"
        )
    print(
        f"[lm-benchmark] materialized {train_rows} train rows and "
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
    if ("base", "base", None) not in completed:
        base_eval = eval_in_subprocess(
            args, None, eval_data, args.work_dir / "base-eval.log"
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
            f"[lm-benchmark] base CE/token={base_eval['loss_per_token']:.6f}",
            flush=True,
        )
    else:
        print("[lm-benchmark] resume: skipping completed base evaluation")

    distinct_m = sorted({arm.learners for arm in arms})
    for seed in parse_seeds(args.seeds):
        seed_dir = args.work_dir / f"seed-{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for m in distinct_m:
            key = ("baseline", f"baseline-m{m}", seed)
            if key in completed:
                print(
                    f"[lm-benchmark] resume: skipping seed={seed} baseline-m{m}",
                    flush=True,
                )
                continue
            print(f"[lm-benchmark] seed={seed} baseline-m{m}", flush=True)
            run = run_baseline(args, m, seed, train_data, seed_dir)
            evaluation = eval_in_subprocess(
                args,
                run["artifact"],
                eval_data,
                seed_dir / f"baseline-m{m}" / "eval.log",
            )
            validate_eval_token_match(records, evaluation)
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
                    f"[lm-benchmark] resume: skipping seed={seed} arm={arm.name}",
                    flush=True,
                )
                continue
            print(f"[lm-benchmark] seed={seed} arm={arm.name}", flush=True)
            run = run_diloco(args, arm, seed, train_data, seed_dir)
            validate_target_token_match(records, arm, seed, run)
            evaluation = eval_in_subprocess(
                args,
                run["artifact"],
                eval_data,
                seed_dir / arm.name / "eval.log",
            )
            validate_eval_token_match(records, evaluation)
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
