#!/usr/bin/env python3
"""Decoupled-DiLoCo quality comparison: async fragment merging vs a
synchronous baseline, at a fixed training-token budget, scored by held-out
eval loss.

The claim under test: yeto's async sync "does not hurt much" — M learners
merging through the syncer should land within a few percent of the eval
loss of one synchronous learner that saw the same total tokens.

Arms (all sharing model, LoRA config, seq len, lr, and token budget):

  base        the base model, untrained (reference floor)
  baseline    ONE learner, --syncer none: the synchronous reference. Per-
              step math is identical to DDP-mean / FSDP2 gradient sync, so
              locally this stands in for the multi-GPU synchronous run; on
              a GPU cluster the same arm with --shard fsdp IS the FSDP2
              baseline (see scripts/baseline_ddp.py for a cloud recipe).
  <preset>    M learners + a real syncer under a settings preset; each
              learner trains budget/M tokens on its disjoint shard, so the
              arm consumes the same data and token budget as the baseline.

The DiLoCo arms are scored on the SYNCER's merged global parameters
(yeto-export from its checkpoint) — the artifact a real run ships — not on
any single learner's local weights. Held-out rows are split off --data
before training so no arm ever sees them.

Presets (--settings, comma-separated or 'all'):

  m2        M=2, everything default (bf16 wire, alpha 0.5, pipelined)
  m4        M=4
  alpha0    broadcasts overwrite (large-M recommendation)
  q4        4-bit E3M0 delta pushes on the wire
  serial    --pipeline 1 (pre-pipelining scheduler behavior)
  noheloco  delta correction off (pure Alg. 2)
  strided   depth-interleaved fragments
  avg       merge = plain weighted averaging (outer lr 1.0, mu 0, alpha 0)
  m2h24     stock DiLoCo throttled to its design-point sync interval (H~24)

Runs locally on one box (CPU by default; --device mps/cuda where torch
supports it): the syncer is the real Rust binary, the learners are real
yeto.learner processes over localhost TCP.

    python scripts/compare_diloco.py --dry-run
    python scripts/compare_diloco.py --model lfm25-230m --data chat.jsonl \
        --token-budget 500000 --settings m2,q4,alpha0 --device cpu

Report: eval loss/token per arm + delta vs baseline, written to
--report-dir (report.md + results.jsonl) and printed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SYNCER_BIN = REPO_ROOT / "syncer/target/release/yeto-syncer"


@dataclass(frozen=True)
class Arm:
    """One DiLoCo configuration under test."""

    name: str
    m: int = 2  # learner islands
    fragments: int = 4
    fragment_pattern: str = "binpack"
    merge_alpha: float = 0.5
    wire_dtype: str = "bf16"
    pipeline: int = 2
    delta_correction: str = "heloco"
    quorum: int | None = None  # None -> all M learners each round
    outer_lr: float = 0.7
    outer_lr_by_fragment: str | None = None
    outer_momentum: float = 0.9
    # Floor on time between round launches (--min-round-interval-ms). On
    # localhost rounds otherwise complete every couple of learner steps
    # (H~2), far off the outer optimizer's H~24 design point; a WAN spaces
    # them naturally. 0 = unthrottled.
    round_interval_ms: int = 0
    # Adaptive H target (--sync-interval-steps). The SYNCER defaults this to
    # 24; comparison arms default it OFF so each arm's sync frequency is an
    # explicit experimental variable, not an ambient default.
    sync_interval_steps: float = 0.0


PRESETS: dict[str, Arm] = {
    "m2": Arm("m2"),
    "m4": Arm("m4", m=4),
    "m12": Arm("m12", m=12, fragments=12, quorum=6),
    "alpha0": Arm("alpha0", merge_alpha=0.0),
    "q4": Arm("q4", wire_dtype="q4"),
    "serial": Arm("serial", pipeline=1),
    "noheloco": Arm("noheloco", delta_correction="none"),
    "strided": Arm("strided", fragment_pattern="strided"),
    # Merge reduced to plain weighted parameter averaging: no outer
    # momentum, full step, overwrite broadcasts. At high merge frequency
    # this approximates synchronous training — if THIS arm matches the
    # baseline where stock m2 lagged, the gap was outer-optimizer gain at
    # off-design sync intervals, not asynchrony itself.
    "avg": Arm("avg", outer_lr=1.0, outer_momentum=0.0, merge_alpha=0.0),
    # Stock DiLoCo at its design-point sync interval, via the syncer's
    # adaptive throttle (H = 24 inner steps per fragment, sized from the
    # measured step time — hardware-independent).
    "m2h24": Arm("m2h24", sync_interval_steps=24.0),
}


def select_arms(spec: str) -> list[Arm]:
    names = list(PRESETS) if spec == "all" else [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [n for n in names if n not in PRESETS]
    if unknown:
        raise SystemExit(f"unknown presets: {unknown} (have {list(PRESETS)})")
    return [PRESETS[n] for n in names]


def steps_for(token_budget: int, mbs: int, seq_len: int, learners: int,
              world: int = 1) -> int:
    """Inner steps per learner so the arm consumes ~token_budget in total.
    `world` is the DDP/FSDP ranks per learner: every rank processes its own
    micro-batch per step, so tokens/step scale by world."""
    return max(1, math.ceil(token_budget / (mbs * seq_len * learners * world)))


def _float_list_value(spec: str, idx: int) -> float:
    vals = [float(v.strip()) for v in spec.split(",") if v.strip()]
    if not vals:
        return 0.0
    return vals[idx % len(vals)]


def learner_env(args, learner_id: int) -> dict[str, str] | None:
    """CUDA_VISIBLE_DEVICES block for one learner: learner i owns GPUs
    [i*g, (i+1)*g). None when GPU partitioning is off."""
    import os

    env = dict(os.environ)
    learner_gpus = getattr(args, "learner_gpus", 0)
    gpu_slots = getattr(args, "gpu_slots", 0)
    gpu_offset = getattr(args, "gpu_offset", 0)
    if learner_gpus <= 0:
        if gpu_slots > 0 and args.device.startswith("cuda"):
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_offset + (learner_id % gpu_slots))
            return env
        return None
    lo = learner_id * learner_gpus
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_offset + g) for g in range(lo, lo + learner_gpus))
    return env


def assigned_gpu_ids(args, arms: list[Arm] | None = None) -> list[int] | None:
    """Physical GPU ids owned by this compare invocation, or None for all."""
    if not getattr(args, "device", "").startswith("cuda"):
        return None
    learner_gpus = getattr(args, "learner_gpus", 0)
    gpu_slots = getattr(args, "gpu_slots", 0)
    gpu_offset = getattr(args, "gpu_offset", 0)
    if learner_gpus > 0:
        if arms is None:
            arms = select_arms(args.settings)
        need = max([1] + [arm.m for arm in arms]) * learner_gpus
    elif gpu_slots > 0:
        need = gpu_slots
    else:
        return None
    return list(range(gpu_offset, gpu_offset + need))


def eval_env(args) -> dict[str, str] | None:
    ids = assigned_gpu_ids(args)
    if not ids:
        return None
    import os

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in ids)
    return env


def gpu_env(learner_id: int, gpus_per_learner: int) -> dict[str, str] | None:
    """Compatibility wrapper used by pure-logic tests."""
    if gpus_per_learner <= 0:
        return None
    return learner_env(
        argparse.Namespace(device="cuda", learner_gpus=gpus_per_learner, gpu_slots=0),
        learner_id,
    )


def learner_command(args, arm_dir: Path, *, learner_id: int, num_learners: int,
                    syncer: str, max_steps: int, arm: Arm | None) -> list[str]:
    if args.learner_gpus > 1:
        # Multi-GPU learner: torchrun ranks over this learner's GPU block
        # (models whose frozen base exceeds one GPU need --shard fsdp).
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={args.learner_gpus}",
            f"--master_port={free_port()}",
            "-m", "yeto.learner",
        ]
    else:
        cmd = [sys.executable, "-m", "yeto.learner"]
    cmd += [
        "--model", args.model,
        "--data", str(arm_dir.parent / "train.jsonl"),
        "--syncer", syncer,
        "--learner-id", str(learner_id),
        "--num-learners", str(num_learners),
        "--tuning", "lora",
        "--lora-r", str(args.lora_r),
        "--lora-alpha", str(args.lora_alpha),
        "--seq-len", str(args.seq_len),
        "--micro-batch-size", str(args.micro_batch_size),
        "--grad-accum", "1",
        "--inner-lr", str(args.inner_lr),
        "--max-local-steps", str(max_steps),
        "--tokenize", "preload",
        "--shard", args.shard,
        "--output-dir", str(arm_dir / f"learner-{learner_id}"),
    ]
    if args.learner_gpus <= 1:
        # torchrun ranks pick their own cuda device from LOCAL_RANK;
        # single-process learners take the explicit one.
        cmd += ["--device", args.device]
    if arm is not None:
        step_sleep = _float_list_value(getattr(args, "learner_step_sleep_ms", "0"), learner_id)
        push_delay = _float_list_value(getattr(args, "learner_push_delay_ms", "0"), learner_id)
        delay_jitter_ms = getattr(args, "learner_delay_jitter_ms", 0.0)
        if step_sleep or push_delay or delay_jitter_ms:
            cmd += [
                "--debug-step-sleep-ms", str(step_sleep),
                "--debug-push-delay-ms", str(push_delay),
                "--debug-delay-jitter-ms", str(delay_jitter_ms),
            ]
        fixed_window_tokens = getattr(args, "fixed_window_tokens", 0)
        fixed_window_microsteps = getattr(args, "fixed_window_microsteps", 0)
        if fixed_window_tokens:
            cmd += ["--fixed-window-tokens", str(fixed_window_tokens)]
        if fixed_window_microsteps:
            cmd += ["--fixed-window-microsteps", str(fixed_window_microsteps)]
        if getattr(args, "pad_to_fixed_window_tokens", False):
            cmd += ["--pad-to-fixed-window-tokens"]
        if getattr(args, "freeze_delta_before_delay", False):
            cmd += ["--freeze-delta-before-delay"]
        cmd += [
            "--fragments", str(arm.fragments),
            "--fragment-pattern", arm.fragment_pattern,
            "--merge-alpha", str(arm.merge_alpha),
            "--wire-dtype", arm.wire_dtype,
        ]
        if getattr(args, "probe_data", None):
            probe_data = (
                str(arm_dir.parent / "eval.jsonl")
                if args.probe_data == "eval"
                else str(args.probe_data)
            )
            cmd += [
                "--probe-data", probe_data,
                "--probe-log", str(arm_dir / f"fragment_probe_learner_{learner_id}.jsonl"),
                "--probe-every", str(args.probe_every),
                "--probe-batches", str(args.probe_batches),
                "--probe-batch-size", str(args.probe_batch_size),
                "--probe-max-rows", str(args.probe_max_rows),
                "--probe-outer-lr", str(args.probe_outer_lr),
                "--probe-freshness-scale", str(args.probe_freshness_scale),
            ]
    return cmd


def syncer_command(
    arm: Arm,
    port: int,
    arm_dir: Path,
    total_steps: int,
    *,
    probe_capture: bool = False,
    probe_capture_every: int = 1,
) -> list[str]:
    # The syncer takes no fragment count: the layout arrives in HELLO.
    cmd = [
        str(SYNCER_BIN),
        "--port", str(port),
        "--learners", str(arm.m),
        "--quorum", str(arm.quorum or arm.m),
        "--grace-ms", "200",
        "--total-steps", str(total_steps),
        "--pipeline", str(arm.pipeline),
        "--delta-correction", arm.delta_correction,
        "--outer-lr", str(arm.outer_lr),
        "--outer-momentum", str(arm.outer_momentum),
        "--checkpoint-path", str(arm_dir / "state.ckpt"),
        "--checkpoint-every", "1",
        "--event-tape", str(arm_dir / "tape.jsonl"),
        "--min-round-interval-ms", str(arm.round_interval_ms),
        "--sync-interval-steps", str(arm.sync_interval_steps),
    ]
    if arm.outer_lr_by_fragment:
        cmd += ["--outer-lr-by-fragment", arm.outer_lr_by_fragment]
    if probe_capture:
        cmd += [
            "--probe-capture-dir", str(arm_dir / "syncer_probe"),
            "--probe-capture-every", str(probe_capture_every),
        ]
    return cmd


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def split_data(
    data: str,
    work: Path,
    eval_rows: int,
    max_rows: int | None,
    shuffle_seed: int | None = None,
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
    work.mkdir(parents=True, exist_ok=True)
    idxs = list(range(n))
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(idxs)

    def dump(path: Path, idxs) -> None:
        with open(path, "w") as f:
            for i in idxs:
                row = ds[i]
                f.write(json.dumps({k: row[k] for k in ("messages", "tools") if k in row}) + "\n")

    train, evalf = work / "train.jsonl", work / "eval.jsonl"
    dump(train, idxs[: n - eval_rows])
    dump(evalf, idxs[n - eval_rows :])
    return train, evalf, n - eval_rows


def eval_loss_per_token(model_id: str, adapter_dir: Path | None, eval_file: Path,
                        seq_len: int, device: str, train_on: str = "assistant") -> float:
    """Held-out masked CE per trained token — the comparison metric."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from yeto.data import build_packed_dataset
    from yeto.losses import sft_loss
    from yeto.models import resolve
    from yeto.learner import accelerator_model_dtype

    resolved = resolve(model_id)
    tok = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    # Keep eval dtype aligned with learner loading. T4-class CUDA devices do
    # not have native bf16, so using bf16 there creates a very slow path.
    dtype = accelerator_model_dtype(torch.device(device))
    model = AutoModelForCausalLM.from_pretrained(
        resolved, dtype=dtype, trust_remote_code=True
    )
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.to(device).eval()
    ds = build_packed_dataset(str(eval_file), tok, 0, 1, seq_len, train_on=train_on)
    total_loss, total_tokens = 0.0, 0.0
    with torch.no_grad():
        for i in range(len(ds)):
            ids, weights = ds[i]
            ids = ids.unsqueeze(0).to(device)
            weights = weights.unsqueeze(0).to(device)
            out = model(input_ids=ids)
            loss, n = sft_loss(out.logits, ids, "cross_entropy", weights)
            total_loss += loss.item()
            total_tokens += n.item()
    return total_loss / max(total_tokens, 1.0)


def wait_for_free_gpus(
    device: str,
    limit_mb: int = 2000,
    timeout_s: int = 180,
    gpu_ids: list[int] | None = None,
) -> None:
    """Block until no compute process holds more than `limit_mb` on any GPU.

    Arms and evals run strictly one after another, but a just-exited CUDA
    process's memory is not always released by the driver the instant
    subprocess.run returns — spawning the next arm into that window OOMs
    (observed on 4xL40S: the eval child's 25 GB were still resident when
    the baseline learner loaded). Fails loudly with the offending pids.
    """
    if not device.startswith("cuda"):
        return
    target_uuids: set[str] | None = None
    if gpu_ids is not None:
        gpu_out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True,
        ).stdout.strip()
        index_to_uuid = {}
        for line in gpu_out.splitlines():
            parts = [v.strip() for v in line.split(",")]
            if len(parts) >= 2 and parts[0].isdigit():
                index_to_uuid[int(parts[0])] = parts[1]
        missing = [i for i in gpu_ids if i not in index_to_uuid]
        if missing:
            raise RuntimeError(
                f"cannot map GPU id(s) {missing} from nvidia-smi output"
            )
        target_uuids = {index_to_uuid[i] for i in gpu_ids}
    deadline = time.monotonic() + timeout_s
    last = ""
    while True:
        query = (
            "gpu_uuid,pid,process_name,used_gpu_memory"
            if target_uuids is not None
            else "pid,process_name,used_gpu_memory"
        )
        out = subprocess.run(
            ["nvidia-smi", f"--query-compute-apps={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        ).stdout.strip()
        holders = []
        for line in out.splitlines():
            parts = [v.strip() for v in line.split(",")]
            if target_uuids is not None:
                if len(parts) < 4:
                    continue
                uuid, pid, name, mem = parts[0], parts[1], parts[2], parts[-1]
                if uuid not in target_uuids:
                    continue
            else:
                if len(parts) < 3:
                    continue
                pid, name, mem = parts[0], parts[1], parts[-1]
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


def eval_in_subprocess(args, adapter_dir: Path | None, eval_file: Path) -> float:
    """Score in a child process so the model's VRAM is released on exit —
    an in-process eval would keep the base resident on GPU 0 and starve the
    next arm's learners (found the hard way on a 4xL40S box)."""
    cmd = [
        sys.executable, __file__, "--eval-only",
        "--model", args.model,
        "--data", str(eval_file),
        "--seq-len", str(args.seq_len),
        "--device", args.device,
    ]
    if adapter_dir is not None:
        cmd += ["--adapter-dir", str(adapter_dir)]
    wait_for_free_gpus(args.device, gpu_ids=assigned_gpu_ids(args))
    out = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, env=eval_env(args)
    )
    # The child has exited, but the driver may release its VRAM lazily;
    # don't hand the GPUs to the next arm until it is actually gone.
    wait_for_free_gpus(args.device, gpu_ids=assigned_gpu_ids(args))
    for line in reversed(out.stdout.splitlines()):
        if line.startswith("EVAL_LOSS "):
            return float(line.split()[1])
    raise RuntimeError(
        f"eval subprocess failed ({out.returncode}):\n{out.stdout[-800:]}\n{out.stderr[-800:]}"
    )


def run_baseline(args, work: Path) -> tuple[Path, float]:
    arm_dir = work / "baseline"
    steps = steps_for(args.token_budget, args.micro_batch_size, args.seq_len, 1,
                      world=max(1, args.learner_gpus))
    cmd = learner_command(args, arm_dir, learner_id=0, num_learners=1,
                          syncer="none", max_steps=steps, arm=None)
    wait_for_free_gpus(args.device, gpu_ids=assigned_gpu_ids(args))
    t0 = time.monotonic()
    run_checked(cmd, arm_dir / "learner.log", env=learner_env(args, 0))
    return arm_dir / "learner-0", time.monotonic() - t0


def run_diloco(args, arm: Arm, work: Path) -> tuple[Path, float]:
    arm_dir = work / arm.name
    arm_dir.mkdir(parents=True, exist_ok=True)
    steps = steps_for(args.token_budget, args.micro_batch_size, args.seq_len, arm.m,
                      world=max(1, args.learner_gpus))
    port = free_port()
    # Generous round ceiling: learners stop at their token budget, and the
    # syncer is terminated once they exit; the checkpoint (written every
    # round) carries the merged params up to the last completed round.
    syncer = subprocess.Popen(
        syncer_command(
            arm,
            port,
            arm_dir,
            total_steps=steps * arm.m * 4,
            probe_capture=getattr(args, "syncer_probe_capture", False),
            probe_capture_every=getattr(args, "syncer_probe_capture_every", 1),
        ),
        stdout=open(arm_dir / "syncer.log", "w"), stderr=subprocess.STDOUT,
    )
    wait_for_free_gpus(args.device, gpu_ids=assigned_gpu_ids(args))
    t0 = time.monotonic()
    learners = []
    try:
        for i in range(arm.m):
            cmd = learner_command(args, arm_dir, learner_id=i, num_learners=arm.m,
                                  syncer=f"127.0.0.1:{port}", max_steps=steps, arm=arm)
            log = open(arm_dir / f"learner-{i}.log", "w")
            learners.append(subprocess.Popen(
                cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
                env=learner_env(args, i),
            ))
        for proc in learners:
            rc = proc.wait(timeout=args.arm_timeout_min * 60)
            if rc != 0:
                raise RuntimeError(f"{arm.name}: a learner exited {rc}; see {arm_dir}")
    finally:
        for proc in learners:
            if proc.poll() is None:
                proc.terminate()
        syncer.terminate()
        syncer.wait(timeout=30)
    wall = time.monotonic() - t0
    ckpt = arm_dir / "state.ckpt"
    if not ckpt.exists():
        raise RuntimeError(f"{arm.name}: no syncer checkpoint (no round completed); see {arm_dir}")
    # Export the merged global parameters to a peft adapter dir.
    export_dir = arm_dir / "export"
    run_checked(
        [
            sys.executable, "-m", "yeto.export",
            "--checkpoint", str(ckpt),
            "--model", args.model,
            "--tuning", "lora",
            "--lora-r", str(args.lora_r),
            "--lora-alpha", str(args.lora_alpha),
            "--fragments", str(arm.fragments),
            "--fragment-pattern", arm.fragment_pattern,
            "--output-dir", str(export_dir),
            "--device", "cpu",
        ],
        arm_dir / "export.log",
    )
    return export_dir, wall


def run_checked(cmd: list[str], log: Path, env: dict | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        rc = subprocess.run(
            cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT, env=env
        ).returncode
    if rc != 0:
        tail = "\n".join(log.read_text().splitlines()[-6:])
        raise RuntimeError(f"command failed ({rc}): {' '.join(cmd)}\n{tail}")


def ensure_syncer() -> None:
    if not SYNCER_BIN.exists():
        print("[compare] building syncer (cargo build --release)")
        subprocess.run(["cargo", "build", "--release", "-q"], cwd=REPO_ROOT / "syncer", check=True)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="lfm25-230m")
    p.add_argument("--data", required=True, help="messages-format chat rows (HF id or local path)")
    p.add_argument("--token-budget", type=int, default=500_000,
                   help="total training tokens per arm (split across an arm's learners)")
    p.add_argument("--settings", default="m2", help=f"comma list of {list(PRESETS)} or 'all'")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--micro-batch-size", type=int, default=2)
    p.add_argument("--inner-lr", type=float, default=3e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--eval-rows", type=int, default=64, help="held-out rows for scoring")
    p.add_argument("--round-interval-ms", type=int, default=None,
                   help="override the round-launch floor of throttled presets "
                   "(m2h24): the right value is H * step_time / fragments, and "
                   "step time depends on hardware")
    p.add_argument("--baseline-loss", type=float, default=None,
                   help="skip the synchronous baseline arm and compare against "
                   "this eval loss/token (from a previous run with the same "
                   "model, data, seed and budget)")
    p.add_argument("--max-rows", type=int, default=None, help="cap training rows")
    p.add_argument(
        "--shuffle-rows-seed",
        type=int,
        default=None,
        help="deterministically shuffle rows before train/eval split; useful "
        "when row-index learner sharding would otherwise preserve dataset order",
    )
    p.add_argument("--device", default="cpu", help="learner/eval device (cpu, mps, cuda)")
    p.add_argument("--shard", choices=["ddp", "fsdp"], default="ddp",
                   help="multi-GPU strategy inside a learner (fsdp shards the "
                   "frozen base when it exceeds one GPU)")
    p.add_argument("--learner-gpus", type=int, default=0,
                   help="GPUs per learner; learner i owns the GPU block "
                   "[i*g, (i+1)*g) and runs under torchrun when g > 1. "
                   "0 = single process on --device")
    p.add_argument("--gpu-slots", type=int, default=0,
                   help="when --learner-gpus 0 and --device cuda, assign "
                   "single-process learners round-robin over this many GPUs")
    p.add_argument("--gpu-offset", type=int, default=0,
                   help="first physical GPU id to assign when partitioning learners")
    p.add_argument("--learner-step-sleep-ms", default="0",
                   help="comma-separated per-learner sleep after each optimizer step")
    p.add_argument("--learner-push-delay-ms", default="0",
                   help="comma-separated per-learner sleep before each fragment push")
    p.add_argument("--learner-delay-jitter-ms", type=float, default=0.0,
                   help="uniform [0, jitter] ms added to each debug sleep")
    p.add_argument("--fixed-window-tokens", type=int, default=0,
                   help="answer async pulls from snapshots taken after this "
                   "many post-reset learner tokens")
    p.add_argument("--fixed-window-microsteps", type=int, default=0,
                   help="answer async pulls from snapshots taken after this "
                   "many post-reset optimizer steps")
    p.add_argument("--pad-to-fixed-window-tokens", action="store_true",
                   help="accepted for fixed-token experiment configs; windows "
                   "round to whole optimizer steps")
    p.add_argument("--freeze-delta-before-delay", action="store_true",
                   help="materialize fragment payloads before push delay stress")
    p.add_argument("--arm-timeout-min", type=int, default=120)
    p.add_argument(
        "--probe-data",
        default=None,
        help="optional held-out data for fragment utility probe in async arms; "
        "use 'eval' to reuse this script's held-out eval split",
    )
    p.add_argument("--probe-every", type=int, default=1)
    p.add_argument("--probe-batches", type=int, default=2)
    p.add_argument("--probe-batch-size", type=int, default=1)
    p.add_argument("--probe-max-rows", type=int, default=64)
    p.add_argument("--probe-outer-lr", type=float, default=1.0)
    p.add_argument(
        "--outer-lr",
        type=float,
        default=None,
        help="override the outer learning rate for every selected async arm",
    )
    p.add_argument(
        "--outer-momentum",
        type=float,
        default=None,
        help="override outer Nesterov momentum for every selected async arm",
    )
    p.add_argument(
        "--outer-lr-by-fragment",
        default=None,
        help="comma-separated per-fragment outer learning rates for every selected async arm",
    )
    p.add_argument("--probe-freshness-scale", type=float, default=24.0)
    p.add_argument(
        "--syncer-probe-capture",
        action="store_true",
        help="capture pre-merge syncer checkpoints and candidate fragments "
        "for offline syncer-current utility probes",
    )
    p.add_argument(
        "--syncer-probe-capture-every",
        type=int,
        default=1,
        help="capture every Nth outer step when --syncer-probe-capture is set",
    )
    p.add_argument("--work-dir", type=Path, default=REPO_ROOT / "compare-work")
    p.add_argument("--report-dir", type=Path, default=REPO_ROOT / "compare-report")
    p.add_argument("--dry-run", action="store_true", help="print the plan; run nothing")
    # Internal: scoring runs as a child process so VRAM is freed on exit.
    p.add_argument("--eval-only", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--adapter-dir", type=Path, default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.eval_only:
        loss = eval_loss_per_token(
            args.model, args.adapter_dir, Path(args.data), args.seq_len, args.device
        )
        print(f"EVAL_LOSS {loss:.6f}")
        return 0

    arms = select_arms(args.settings)
    if (
        args.outer_lr is not None
        or args.outer_lr_by_fragment is not None
        or args.outer_momentum is not None
    ):
        from dataclasses import replace as _replace

        arms = [
            _replace(
                arm,
                outer_lr=arm.outer_lr if args.outer_lr is None else args.outer_lr,
                outer_lr_by_fragment=(
                    arm.outer_lr_by_fragment
                    if args.outer_lr_by_fragment is None
                    else args.outer_lr_by_fragment
                ),
                outer_momentum=(
                    arm.outer_momentum
                    if args.outer_momentum is None
                    else args.outer_momentum
                ),
            )
            for arm in arms
        ]
    if args.round_interval_ms is not None:
        from dataclasses import replace as _replace

        arms = [_replace(a, round_interval_ms=args.round_interval_ms)
                if a.round_interval_ms else a for a in arms]
    world = max(1, args.learner_gpus)
    base_steps = steps_for(args.token_budget, args.micro_batch_size, args.seq_len, 1, world)
    print(f"[compare] model={args.model} budget={args.token_budget} tokens "
          f"(baseline: {base_steps} steps of {args.micro_batch_size}x{args.seq_len}"
          f"{f' x{world} ranks' if world > 1 else ''})")
    for arm in arms:
        s = steps_for(args.token_budget, args.micro_batch_size, args.seq_len, arm.m, world)
        print(f"  {arm.name:<10} M={arm.m} {s} steps/learner "
              f"P={arm.fragments} alpha={arm.merge_alpha} wire={arm.wire_dtype} "
              f"pipeline={arm.pipeline} correction={arm.delta_correction}")
    if args.dry_run:
        return 0

    if args.learner_gpus > 0 and args.gpu_slots > 0:
        raise SystemExit("--gpu-slots is only valid when --learner-gpus is 0")
    if args.learner_gpus > 0 or args.gpu_slots > 0:
        import torch

        need = (
            max(arm.m for arm in arms) * args.learner_gpus
            if args.learner_gpus > 0
            else args.gpu_slots
        )
        have = torch.cuda.device_count()
        if args.gpu_offset + need > have:
            raise SystemExit(
                f"largest arm needs physical GPU ids "
                f"{args.gpu_offset}..{args.gpu_offset + need - 1} "
                f"but only {have} visible GPU(s) exist"
            )
    ensure_syncer()
    if args.work_dir.exists():
        shutil.rmtree(args.work_dir)
    train, evalf, n_train = split_data(
        args.data, args.work_dir, args.eval_rows, args.max_rows, args.shuffle_rows_seed
    )
    print(f"[compare] {n_train} train rows, {args.eval_rows} eval rows")

    records = []
    base = eval_in_subprocess(args, None, evalf)
    records.append({"arm": "base (untrained)", "m": 0, "wall_s": 0.0, "eval_loss": base})
    print(f"[compare] base eval loss/token: {base:.4f}", flush=True)

    if args.baseline_loss is not None:
        bl = args.baseline_loss
        records.append({"arm": "baseline (sync, injected)", "m": 1, "wall_s": 0.0,
                        "eval_loss": bl})
        print(f"[compare] baseline eval loss/token: {bl:.4f} (injected)", flush=True)
    else:
        adapters, wall = run_baseline(args, args.work_dir)
        bl = eval_in_subprocess(args, adapters, evalf)
        records.append({"arm": "baseline (sync)", "m": 1, "wall_s": round(wall, 1),
                        "eval_loss": bl})
        print(f"[compare] baseline eval loss/token: {bl:.4f} ({wall:.0f}s)", flush=True)

    for arm in arms:
        adapters, wall = run_diloco(args, arm, args.work_dir)
        loss = eval_in_subprocess(args, adapters, evalf)
        records.append({"arm": arm.name, "m": arm.m, "wall_s": round(wall, 1),
                        "eval_loss": loss})
        print(f"[compare] {arm.name} eval loss/token: {loss:.4f} ({wall:.0f}s)", flush=True)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    with open(args.report_dir / "results.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    md = [
        f"# DiLoCo vs synchronous baseline — {args.model}, "
        f"{args.token_budget} tokens/arm",
        "",
        "| arm | M | wall (s) | eval loss/token | Δ vs baseline |",
        "|---|---|---|---|---|",
    ]
    for r in records:
        delta = "—" if r["arm"].startswith(("base", "baseline")) or bl == 0 else (
            f"{100 * (r['eval_loss'] - bl) / bl:+.2f}%"
        )
        md.append(f"| {r['arm']} | {r['m'] or '—'} | {r['wall_s']:.0f} "
                  f"| {r['eval_loss']:.4f} | {delta} |")
    (args.report_dir / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
