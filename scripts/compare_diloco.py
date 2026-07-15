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
  iso       IsoLoCo: Iso-C isotropic aggregation on matrix fragments
  probe_shadow  four-responder exact LOO probe; always commit A0
  probe_loo_v1  four-responder exact LOO probe; commit the selected preview
  probe_lr_shadow  predeclared scalar probe; always commit x1 (A0)
  probe_lr_v1  predeclared scalar probe; commit the selected scaled preview

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
import hashlib
import json
import math
import random
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SYNCER_BIN = REPO_ROOT / "syncer/target/release/yeto-syncer"
OUTER_OPTIMIZERS = (
    "nesterov",
    "normalized-ema",
    "restarted-ema",
    "rho-adaptive",
    "capped-nesterov",
    "capped-nesterov-gc",
    "capped-nesterov-r",
    "capped-nesterov-curv",
    "capped-nesterov-wsub",
    "block-rms",
    "block-yogi",
    "cheb-sgd",
    "cttn",
    "cttn-scalar",
    "cttn-shadow",
)
# Matrix-fragment aggregation modes the harness understands (client-side
# --matrix-merge). "worker-snr" is the memoryless cross-worker consensus merge.
MATRIX_MERGES = ("rda", "iso", "worker-snr")
COMMIT_POLICIES = (
    "token_weighted",
    "probe_shadow",
    "probe_loo_v1",
    "probe_lr_shadow",
    "probe_lr_v1",
    "cttn_v1",
    "cttn_scalar_v1",
    "cttn_shadow_v1",
)
LOO_COMMIT_POLICIES = frozenset(("probe_shadow", "probe_loo_v1"))
ACTION_PROBE_MIN_GAIN = 0.00025
ACTION_PROBE_LCB_Z = 2.365
ACTION_PROBE_MIN_WIN_RATE = 0.75
ACTION_PROBE_ACTIONS = ("A0", "A1", "A2", "A3", "A4")
ACTION_PROBE_SHADOW_POLICIES = frozenset(("probe_shadow", "probe_lr_shadow"))
ACTION_PROBE_ACTIVE_POLICIES = frozenset(
    ("probe_loo_v1", "probe_lr_v1", "cttn_v1", "cttn_scalar_v1")
)
ACTION_PROBE_SCALAR_MULTIPLIERS = {
    action: multiplier
    for action, multiplier in zip(
        ACTION_PROBE_ACTIONS, (1.0, 0.75, 1.125, 1.25, 1.5)
    )
}
ACTION_PROBE_DIGEST_RE = re.compile(r"[0-9a-fA-F]{64}\Z")
ACTION_PROBE_SUMMARY_FILENAME = "action_probe_run_summary.json"
ACTION_PROBE_TRANSPORT_CONFIG_FALLBACK_REASONS = frozenset(
    ("preview_construction_error", "protocol_error", "unsafe_probe_response")
)
ACTION_PROBE_TRANSPORT_CONFIG_FALLBACK_PREFIXES = (
    "action_probe_",
    "config_",
    "probe_",
    "transport_",
)


@dataclass(frozen=True)
class Arm:
    """One DiLoCo configuration under test."""

    name: str
    m: int = 2  # learner islands
    fragments: int = 4
    fragment_pattern: str = "binpack"
    # Aggregation for non-embedding (matrix) fragments: "rda" (default) or
    # "iso" (Iso-C-style isotropic aggregation, IsoLoCo, arXiv 2607.03011).
    matrix_merge: str = "rda"
    merge_alpha: float = 0.5
    wire_dtype: str = "bf16"
    pipeline: int = 2
    delta_correction: str = "heloco"
    quorum: int | None = None  # None -> all M learners each round
    strict_quorum: bool = False
    outer_lr: float = 0.7
    outer_lr_by_fragment: str | None = None
    outer_momentum: float = 0.9
    outer_optimizer: str = "nesterov"
    outer_restart_cos_threshold: float = 0.0
    commit_policy: str = "token_weighted"
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
    # E1 noise-floor control: one learner behind the identical syncer stack
    # (fragments, fixed windows, outer step, probe capture) with quorum
    # defaulting to 1. Every "merge" is a single learner's window delta.
    "m1": Arm("m1", m=1),
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
    # IsoLoCo (arXiv 2607.03011): Iso-C isotropic aggregation on the matrix
    # fragments, composed with the default Nesterov outer optimizer.
    "iso": Arm("iso", matrix_merge="iso"),
    # Exact production leave-one-out actions. Both arms pay the same sidecar
    # latency; shadow records the recommendation but commits A0.
    "probe_shadow": Arm(
        "probe_shadow",
        m=4,
        quorum=4,
        strict_quorum=True,
        commit_policy="probe_shadow",
    ),
    "probe_loo_v1": Arm(
        "probe_loo_v1",
        m=4,
        quorum=4,
        strict_quorum=True,
        commit_policy="probe_loo_v1",
    ),
    # Scalar full-group actions use the predeclared selector grid from the
    # completed offline replay: A0=x1 fallback, then x0.75/x1.125/x1.25/x1.5.
    "probe_lr_shadow": Arm(
        "probe_lr_shadow",
        m=4,
        quorum=4,
        strict_quorum=True,
        commit_policy="probe_lr_shadow",
    ),
    "probe_lr_v1": Arm(
        "probe_lr_v1",
        m=4,
        quorum=4,
        strict_quorum=True,
        commit_policy="probe_lr_v1",
    ),
}


def select_arms(spec: str) -> list[Arm]:
    names = (
        list(PRESETS)
        if spec == "all"
        else [s.strip() for s in spec.split(",") if s.strip()]
    )
    unknown = [n for n in names if n not in PRESETS]
    if unknown:
        raise SystemExit(f"unknown presets: {unknown} (have {list(PRESETS)})")
    return [PRESETS[n] for n in names]


def apply_arm_overrides(
    arms: list[Arm],
    *,
    outer_lr: float | None = None,
    outer_lr_by_fragment: str | None = None,
    outer_momentum: float | None = None,
    outer_optimizer: str | None = None,
    outer_restart_cos_threshold: float | None = None,
    delta_correction: str | None = None,
    commit_policy: str | None = None,
    matrix_merge: str | None = None,
) -> list[Arm]:
    """Apply CLI-wide async-arm overrides without mutating presets."""
    if all(
        value is None
        for value in (
            outer_lr,
            outer_lr_by_fragment,
            outer_momentum,
            outer_optimizer,
            outer_restart_cos_threshold,
            delta_correction,
            commit_policy,
            matrix_merge,
        )
    ):
        return arms

    from dataclasses import replace

    normalized = []
    for arm in arms:
        selected_optimizer = (
            arm.outer_optimizer if outer_optimizer is None else outer_optimizer
        )
        selected_momentum = (
            arm.outer_momentum if outer_momentum is None else outer_momentum
        )
        selected_policy = arm.commit_policy if commit_policy is None else commit_policy
        if selected_optimizer in ("cttn", "cttn-scalar", "cttn-shadow"):
            selected_policy = (
                "cttn_shadow_v1"
                if selected_optimizer == "cttn-shadow"
                else "cttn_v1"
                if selected_optimizer == "cttn"
                else "cttn_scalar_v1"
            )
        if selected_policy in ("cttn_v1", "cttn_scalar_v1", "cttn_shadow_v1"):
            if outer_momentum is None:
                selected_momentum = 0.0
        normalized.append(
            replace(
                arm,
                matrix_merge=(
                    arm.matrix_merge if matrix_merge is None else matrix_merge
                ),
                outer_lr=arm.outer_lr if outer_lr is None else outer_lr,
                outer_lr_by_fragment=(
                    arm.outer_lr_by_fragment
                    if outer_lr_by_fragment is None
                    else outer_lr_by_fragment
                ),
                outer_momentum=selected_momentum,
                outer_optimizer=selected_optimizer,
                outer_restart_cos_threshold=(
                    arm.outer_restart_cos_threshold
                    if outer_restart_cos_threshold is None
                    else outer_restart_cos_threshold
                ),
                delta_correction=(
                    arm.delta_correction
                    if delta_correction is None
                    else delta_correction
                ),
                commit_policy=selected_policy,
            )
        )
    return normalized


def steps_for(
    token_budget: int, mbs: int, seq_len: int, learners: int, world: int = 1
) -> int:
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
            physical_id = gpu_offset + (learner_id % gpu_slots)
            inventory = getattr(args, "_frozen_gpu_inventory", {})
            assigned = inventory.get(physical_id)
            env["CUDA_VISIBLE_DEVICES"] = (
                assigned["uuid"] if assigned is not None else str(physical_id)
            )
            env["YETO_ASSIGNED_PHYSICAL_GPU_INDEX"] = str(physical_id)
            if assigned is not None:
                env["YETO_ASSIGNED_GPU_UUID"] = assigned["uuid"]
                env["YETO_ASSIGNED_GPU_NAME"] = assigned["name"]
            return env
        return None
    lo = learner_id * learner_gpus
    env["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(gpu_offset + g) for g in range(lo, lo + learner_gpus)
    )
    return env


def freeze_gpu_uuid_inventory(args, arms: list[Arm]) -> dict:
    """Freeze physical CUDA index/UUIDs and the four-learner bijection."""
    if (
        not args.device.startswith("cuda")
        or args.learner_gpus != 0
        or args.gpu_slots != 4
        or len(arms) != 1
        or arms[0].m != 4
    ):
        raise SystemExit(
            "distinct learner GPU UUID proof requires one M=4 CUDA arm, "
            "--learner-gpus 0, and --gpu-slots 4"
        )
    requested = assigned_gpu_ids(args, arms)
    if requested is None or len(requested) != 4:
        raise SystemExit("GPU UUID proof requires exactly four assigned CUDA indices")
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        raise SystemExit(f"nvidia-smi GPU UUID inventory failed: {result.stderr[-1000:]}")
    inventory: dict[int, dict[str, str | int]] = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) != 3:
            raise SystemExit(f"malformed nvidia-smi inventory row: {line!r}")
        try:
            index = int(parts[0])
        except ValueError as exc:
            raise SystemExit(f"invalid nvidia-smi CUDA index: {parts[0]!r}") from exc
        inventory[index] = {"cuda_index": index, "uuid": parts[1], "name": parts[2]}
    if set(requested) - set(inventory):
        raise SystemExit("nvidia-smi inventory omits an assigned CUDA index")
    selected = {index: inventory[index] for index in requested}
    uuids = [str(selected[index]["uuid"]) for index in requested]
    if len(set(uuids)) != 4 or any(not uuid.startswith("GPU-") for uuid in uuids):
        raise SystemExit("packing canary requires four distinct full-GPU UUIDs")
    if any("A100" not in str(selected[index]["name"]).upper() for index in requested):
        raise SystemExit("packing canary requires four NVIDIA A100 devices")
    args._frozen_gpu_inventory = selected
    proof = {
        "schema": "yeto_learner_gpu_uuid_bijection_v1",
        "nvidia_smi_command": command,
        "gpu_inventory": [selected[index] for index in requested],
        "learner_assignments": [
            {
                "learner_id": learner,
                "physical_cuda_index": requested[learner],
                "gpu_uuid": selected[requested[learner]]["uuid"],
                "gpu_name": selected[requested[learner]]["name"],
            }
            for learner in range(4)
        ],
        "distinct_gpu_uuid_count": 4,
        "one_learner_per_distinct_gpu_uuid": True,
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "gpu-allocation.json").write_text(
        json.dumps(proof, indent=2, sort_keys=True) + "\n"
    )
    return proof


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


def parse_gpu_ids(spec: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in spec.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "GPU ids must be a comma-separated integer list"
        ) from exc
    if (
        not values
        or any(value < 0 for value in values)
        or len(set(values)) != len(values)
    ):
        raise argparse.ArgumentTypeError(
            "GPU ids must be a non-empty unique nonnegative list"
        )
    return values


def action_probe_env(args) -> dict[str, str]:
    import os

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in args.action_probe_gpus)
    return env


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


def learner_command(
    args,
    arm_dir: Path,
    *,
    learner_id: int,
    num_learners: int,
    syncer: str,
    max_steps: int,
    arm: Arm | None,
) -> list[str]:
    if args.learner_gpus > 1:
        # Multi-GPU learner: torchrun ranks over this learner's GPU block
        # (models whose frozen base exceeds one GPU need --shard fsdp).
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={args.learner_gpus}",
            f"--master_port={free_port()}",
            "-m",
            "yeto.learner",
        ]
    else:
        cmd = [sys.executable, "-m", "yeto.learner"]
    cmd += [
        "--model",
        args.model,
        "--data",
        str(arm_dir.parent / "train.jsonl"),
        "--syncer",
        syncer,
        "--learner-id",
        str(learner_id),
        "--num-learners",
        str(num_learners),
        "--seed",
        str(getattr(args, "training_seed", 0)),
        "--tuning",
        getattr(args, "tuning", "lora"),
        "--lora-r",
        str(args.lora_r),
        "--lora-alpha",
        str(args.lora_alpha),
        "--seq-len",
        str(args.seq_len),
        "--micro-batch-size",
        str(args.micro_batch_size),
        "--grad-accum",
        "1",
        "--inner-lr",
        str(args.inner_lr),
        "--max-local-steps",
        str(max_steps),
        "--tokenize",
        "preload",
        "--shard",
        args.shard,
        "--output-dir",
        str(arm_dir / f"learner-{learner_id}"),
    ]
    if args.learner_gpus <= 1:
        # torchrun ranks pick their own cuda device from LOCAL_RANK;
        # single-process learners take the explicit one.
        cmd += ["--device", args.device]
    if arm is not None:
        step_sleep = _float_list_value(
            getattr(args, "learner_step_sleep_ms", "0"), learner_id
        )
        push_delay = _float_list_value(
            getattr(args, "learner_push_delay_ms", "0"), learner_id
        )
        delay_jitter_ms = getattr(args, "learner_delay_jitter_ms", 0.0)
        if step_sleep or push_delay or delay_jitter_ms:
            cmd += [
                "--debug-step-sleep-ms",
                str(step_sleep),
                "--debug-push-delay-ms",
                str(push_delay),
                "--debug-delay-jitter-ms",
                str(delay_jitter_ms),
            ]
        fixed_window_tokens = getattr(args, "fixed_window_tokens", 0)
        fixed_window_microsteps = getattr(args, "fixed_window_microsteps", 0)
        if fixed_window_tokens:
            cmd += ["--fixed-window-tokens", str(fixed_window_tokens)]
        if fixed_window_microsteps:
            cmd += ["--fixed-window-microsteps", str(fixed_window_microsteps)]
        fixed_window_schedule = getattr(args, "fixed_window_schedule", None)
        if fixed_window_schedule:
            cmd += ["--fixed-window-schedule", str(fixed_window_schedule)]
        if getattr(args, "pad_to_fixed_window_tokens", False):
            cmd += ["--pad-to-fixed-window-tokens"]
        if getattr(args, "freeze_delta_before_delay", False):
            cmd += ["--freeze-delta-before-delay"]
        if getattr(args, "barrier_sync", False):
            cmd += ["--barrier-sync"]
        lag_commits = int(getattr(args, "learner_broadcast_lag_commits", 0))
        if lag_commits > 0:
            cmd += ["--debug-broadcast-lag-commits", str(lag_commits)]
        cmd += [
            "--fragments",
            str(arm.fragments),
            "--fragment-pattern",
            arm.fragment_pattern,
            "--merge-alpha",
            str(arm.merge_alpha),
            "--wire-dtype",
            arm.wire_dtype,
        ]
        if arm.matrix_merge != "rda":
            # Only non-default values are passed, keeping baseline learner
            # command lines byte-identical to the pre-iso harness.
            cmd += ["--matrix-merge", arm.matrix_merge]
        if getattr(args, "probe_data", None):
            probe_data = (
                str(arm_dir.parent / "eval.jsonl")
                if args.probe_data == "eval"
                else str(args.probe_data)
            )
            cmd += [
                "--probe-data",
                probe_data,
                "--probe-log",
                str(arm_dir / f"fragment_probe_learner_{learner_id}.jsonl"),
                "--probe-every",
                str(args.probe_every),
                "--probe-batches",
                str(args.probe_batches),
                "--probe-batch-size",
                str(args.probe_batch_size),
                "--probe-max-rows",
                str(args.probe_max_rows),
                "--probe-outer-lr",
                str(args.probe_outer_lr),
                "--probe-freshness-scale",
                str(args.probe_freshness_scale),
            ]
    return cmd


def syncer_command(
    arm: Arm,
    port: int,
    arm_dir: Path,
    total_steps: int,
    *,
    checkpoint_every: int = 1,
    probe_capture: bool = False,
    probe_capture_every: int = 1,
    delta_norm_ref: float = 0.0,
    version_matched_anchor: bool = False,
    anchor_drift_log: bool = False,
    action_probe_endpoint: str | None = None,
    action_probe_timeout_ms: int = 300_000,
    action_probe_run_uuid: str | None = None,
    action_probe_expected_config: Path | None = None,
    cttn_rho: float = 0.10,
    cttn_shadow_samples: int = 32,
) -> list[str]:
    # The syncer takes no fragment count: the layout arrives in HELLO.
    cttn_pseudo_optimizer = arm.outer_optimizer in ("cttn", "cttn-scalar", "cttn-shadow")
    syncer_outer_optimizer = (
        "nesterov" if cttn_pseudo_optimizer else arm.outer_optimizer
    )
    if cttn_pseudo_optimizer:
        commit_policy = (
            "cttn_shadow_v1"
            if arm.outer_optimizer == "cttn-shadow"
            else "cttn_scalar_v1"
            if arm.outer_optimizer == "cttn-scalar"
            else "cttn_v1"
        )
    else:
        commit_policy = arm.commit_policy
    cttn_enabled = commit_policy in ("cttn_v1", "cttn_scalar_v1", "cttn_shadow_v1")
    cmd = [
        str(SYNCER_BIN),
        "--port",
        str(port),
        "--learners",
        str(arm.m),
        "--quorum",
        str(arm.quorum or arm.m),
        "--grace-ms",
        "200",
        "--total-steps",
        str(total_steps),
        "--pipeline",
        str(arm.pipeline),
        "--delta-correction",
        arm.delta_correction,
        "--outer-lr",
        str(arm.outer_lr),
        "--outer-momentum",
        str(arm.outer_momentum),
        "--outer-optimizer",
        syncer_outer_optimizer,
        "--outer-restart-cos-threshold",
        str(arm.outer_restart_cos_threshold),
        "--checkpoint-path",
        str(arm_dir / "state.ckpt"),
        "--checkpoint-every",
        str(checkpoint_every),
        "--event-tape",
        str(arm_dir / "tape.jsonl"),
        "--min-round-interval-ms",
        str(arm.round_interval_ms),
        "--sync-interval-steps",
        str(arm.sync_interval_steps),
        "--commit-policy",
        commit_policy,
    ]
    if cttn_enabled:
        cmd += ["--cttn-mu", "0.9", "--cttn-rho", str(cttn_rho)]
    if commit_policy == "cttn_shadow_v1":
        cmd += ["--cttn-shadow-samples", str(cttn_shadow_samples)]
    if arm.outer_lr_by_fragment:
        cmd += ["--outer-lr-by-fragment", arm.outer_lr_by_fragment]
    if delta_norm_ref > 0.0:
        # Post-merge renormalization for mediation-control experiments; the
        # flag is only emitted when active so default command lines (and the
        # syncer behavior behind them) stay byte-identical.
        cmd += ["--delta-norm-ref", str(delta_norm_ref)]
    # EXP2.46 3-arm current-anchor control. --version-matched-anchor changes the
    # merge (arms A/B); --anchor-drift-log only instruments (arm C). Both are
    # emitted only when active so default command lines stay byte-identical.
    if version_matched_anchor:
        cmd += ["--version-matched-anchor"]
    if anchor_drift_log and not version_matched_anchor:
        cmd += ["--anchor-drift-log"]
    if arm.strict_quorum:
        cmd += ["--strict-quorum"]
    if probe_capture:
        cmd += [
            "--probe-capture-dir",
            str(arm_dir / "syncer_probe"),
            "--probe-capture-every",
            str(probe_capture_every),
        ]
    if commit_policy != "token_weighted":
        missing = [
            name
            for name, value in (
                ("action_probe_endpoint", action_probe_endpoint),
                ("action_probe_run_uuid", action_probe_run_uuid),
                ("action_probe_expected_config", action_probe_expected_config),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                f"{commit_policy} requires sidecar settings: {', '.join(missing)}"
            )
        cmd += [
            "--action-probe-endpoint",
            str(action_probe_endpoint),
            "--action-probe-timeout-ms",
            str(action_probe_timeout_ms),
            "--action-probe-run-uuid",
            str(action_probe_run_uuid),
            "--action-probe-expected-config",
            str(action_probe_expected_config),
        ]
    return cmd


@dataclass
class ActionProbeProcess:
    process: subprocess.Popen
    log_handle: object
    endpoint: str
    expected_config: Path
    run_uuid: str


def action_probe_command(args, arm: Arm, endpoint: str) -> list[str]:
    local_gpus = ",".join(str(index) for index in range(len(args.action_probe_gpus)))
    min_gain = getattr(args, "action_probe_min_gain", None)
    lcb_z = getattr(args, "action_probe_lcb_z", None)
    min_win_rate = getattr(args, "action_probe_min_win_rate", None)
    return [
        sys.executable,
        "-m",
        "yeto.action_probe_server",
        "--listen",
        endpoint,
        "--model",
        args.model,
        "--anchor-manifest",
        str(args.action_probe_anchor_manifest),
        "--gpus",
        local_gpus,
        "--seq-len",
        str(args.action_probe_seq_len),
        "--panels",
        str(args.action_probe_panels),
        "--blocks-per-panel",
        str(args.action_probe_blocks_per_panel),
        "--lora-r",
        str(args.lora_r),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-targets",
        args.action_probe_lora_targets,
        "--fragments",
        str(arm.fragments),
        "--fragment-pattern",
        arm.fragment_pattern,
        "--startup-timeout-s",
        str(args.action_probe_startup_timeout_s),
        "--request-timeout-s",
        str(args.action_probe_timeout_s),
        "--client-timeout-s",
        str(max(args.action_probe_timeout_s + 5.0, args.action_probe_timeout_s * 2.0)),
        "--min-gain",
        str(ACTION_PROBE_MIN_GAIN if min_gain is None else min_gain),
        "--lcb-z",
        str(ACTION_PROBE_LCB_Z if lcb_z is None else lcb_z),
        "--min-win-rate",
        str(ACTION_PROBE_MIN_WIN_RATE if min_win_rate is None else min_win_rate),
    ]


def ping_action_probe(endpoint: str, timeout_s: float) -> dict:
    from yeto.action_probe import PROTOCOL, recv_frame, send_frame

    host, port_text = endpoint.rsplit(":", 1)
    with socket.create_connection((host, int(port_text)), timeout=timeout_s) as client:
        client.settimeout(timeout_s)
        send_frame(client, {"protocol": PROTOCOL, "type": "ping"})
        response = recv_frame(client, timeout_s=timeout_s).header
    if response.get("protocol") != PROTOCOL or response.get("type") != "pong":
        raise RuntimeError(f"action-probe readiness ping returned {response!r}")
    if response.get("ok") is not True:
        raise RuntimeError("action-probe readiness ping was not healthy")
    return response


def _read_action_probe_ready(log_path: Path, process, timeout_s: float) -> dict:
    deadline = time.monotonic() + timeout_s
    marker = "ACTION_PROBE_READY "
    while time.monotonic() < deadline:
        if log_path.exists():
            for line in log_path.read_text(errors="replace").splitlines():
                if line.startswith(marker):
                    try:
                        return json.loads(line[len(marker) :])
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(
                            f"malformed action-probe readiness record in {log_path}"
                        ) from exc
        rc = process.poll()
        if rc is not None:
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-12:])
            raise RuntimeError(
                f"action-probe sidecar exited {rc} before readiness:\n{tail}"
            )
        time.sleep(0.1)
    raise RuntimeError(
        f"action-probe sidecar did not become ready within {timeout_s}s; see {log_path}"
    )


def expected_probe_config(ready: dict, arm: Arm, args) -> dict:
    backend = ready.get("backend")
    if not isinstance(backend, dict) or backend.get("healthy") is not True:
        raise RuntimeError("action-probe readiness record has no healthy backend")
    workers = backend.get("workers")
    if not isinstance(workers, list) or not workers:
        raise RuntimeError("action-probe readiness record has no workers")
    if any(worker.get("lora_r") != args.lora_r for worker in workers):
        raise RuntimeError("action-probe workers do not match the learner LoRA rank")
    fragment_layout = backend.get("fragment_layout")
    if not isinstance(fragment_layout, dict) or len(fragment_layout) != arm.fragments:
        raise RuntimeError("action-probe fragment layout does not match the arm")
    return {
        "protocol": ready.get("protocol"),
        "anchor_manifest_sha256": backend.get("anchor_manifest_sha256"),
        "anchor_tensors_sha256": backend.get("anchor_tensors_sha256"),
        "probe_config_sha256": backend.get("probe_config_sha256"),
        "layout_hash": backend.get("layout_hash"),
        "fragment_pattern": arm.fragment_pattern,
        "lora_r": args.lora_r,
        "fragment_layout": fragment_layout,
    }


def launch_action_probe(args, arm: Arm, arm_dir: Path) -> ActionProbeProcess:
    if arm.commit_policy == "token_weighted":
        raise ValueError("token_weighted does not launch an action-probe sidecar")
    port = free_port()
    endpoint = f"127.0.0.1:{port}"
    log_path = arm_dir / "action_probe.log"
    log_handle = open(log_path, "w")
    process = subprocess.Popen(
        action_probe_command(args, arm, endpoint),
        cwd=REPO_ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=action_probe_env(args),
    )
    try:
        ready = _read_action_probe_ready(
            log_path, process, args.action_probe_startup_timeout_s
        )
        if ready.get("protocol") != "yeto-action-probe-v1":
            raise RuntimeError("action-probe sidecar reported an unexpected protocol")
        if ready.get("listen") != endpoint:
            raise RuntimeError(
                f"action-probe sidecar bound {ready.get('listen')!r}, expected {endpoint!r}"
            )
        ping_action_probe(endpoint, min(10.0, args.action_probe_timeout_s))
        config_path = arm_dir / "action_probe_expected.json"
        config_path.write_text(
            json.dumps(
                expected_probe_config(ready, arm, args), indent=2, sort_keys=True
            )
            + "\n"
        )
        run_uuid = args.action_probe_run_uuid or str(uuid.uuid4())
        return ActionProbeProcess(
            process=process,
            log_handle=log_handle,
            endpoint=endpoint,
            expected_config=config_path,
            run_uuid=run_uuid,
        )
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        log_handle.close()
        raise


def stop_action_probe(sidecar: ActionProbeProcess) -> None:
    try:
        if sidecar.process.poll() is None:
            sidecar.process.terminate()
        try:
            sidecar.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            sidecar.process.kill()
            sidecar.process.wait(timeout=5)
    finally:
        sidecar.log_handle.close()


def _load_event_tape(arm: Arm, tape_path: Path) -> list[dict]:
    try:
        lines = tape_path.read_text().splitlines()
    except OSError as exc:
        raise RuntimeError(f"{arm.name}: cannot read event tape {tape_path}: {exc}") from exc

    def reject_nonfinite(value: str):
        raise ValueError(f"non-finite JSON constant {value}")

    records = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line, parse_constant=reject_nonfinite)
        except ValueError as exc:
            raise RuntimeError(
                f"{arm.name}: malformed event tape JSON at line {line_number}: {exc}"
            ) from exc
        if not isinstance(record, dict):
            raise RuntimeError(
                f"{arm.name}: event tape line {line_number} is not a JSON object"
            )
        records.append(record)
    return records


def validate_event_tape_records(
    arm: Arm, records: list[dict], *, expected_steps: int = 0
) -> None:
    if expected_steps and len(records) != expected_steps:
        raise RuntimeError(
            f"{arm.name}: expected {expected_steps} outer steps, "
            f"event tape has {len(records)}"
        )

    probe_policy = arm.commit_policy != "token_weighted"
    if not (arm.strict_quorum or probe_policy):
        return
    expected_responders = list(range(arm.m))
    mode = "probe" if probe_policy else "strict-quorum"
    for index, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise RuntimeError(
                f"{arm.name}: {mode} event tape record {index} is not an object"
            )
        responders = record.get("responders")
        step = record.get("step", "?")
        if not isinstance(responders, list):
            raise RuntimeError(
                f"{arm.name}: {mode} run has malformed responders at step {step}: "
                "responders must be a list"
            )
        responder_ids = []
        for position, responder in enumerate(responders):
            if not isinstance(responder, dict):
                raise RuntimeError(
                    f"{arm.name}: {mode} run has malformed responder {position} "
                    f"at step {step}: expected an object, got {responder!r}"
                )
            missing = [
                field
                for field in ("id", "base_version", "c_steps", "c_tokens", "weight")
                if field not in responder
            ]
            if missing:
                raise RuntimeError(
                    f"{arm.name}: {mode} run has malformed responder {position} "
                    f"at step {step}: missing {missing}"
                )
            responder_id = responder["id"]
            base_version = responder["base_version"]
            c_steps = responder["c_steps"]
            c_tokens = responder["c_tokens"]
            if (
                isinstance(responder_id, bool)
                or not isinstance(responder_id, int)
                or responder_id < 0
            ):
                raise RuntimeError(
                    f"{arm.name}: {mode} run has malformed responder {position} "
                    f"at step {step}: id must be a non-negative integer"
                )
            if (
                isinstance(base_version, bool)
                or not isinstance(base_version, int)
                or base_version < 0
            ):
                raise RuntimeError(
                    f"{arm.name}: {mode} run has malformed responder {position} "
                    f"at step {step}: base_version must be a non-negative integer"
                )
            for field, value in (("c_steps", c_steps), ("c_tokens", c_tokens)):
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise RuntimeError(
                        f"{arm.name}: {mode} run has malformed responder {position} "
                        f"at step {step}: {field} must be a positive integer"
                    )
            weight = responder["weight"]
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise RuntimeError(
                    f"{arm.name}: {mode} run has malformed responder {position} "
                    f"at step {step}: weight must be a positive finite number"
                )
            try:
                weight_number = float(weight)
                expected_weight = float(c_tokens) ** 2 / c_steps
            except OverflowError:
                raise RuntimeError(
                    f"{arm.name}: {mode} run has malformed responder {position} "
                    f"at step {step}: weight fields overflow"
                ) from None
            if not math.isfinite(weight_number) or weight_number <= 0 or not math.isclose(
                weight_number, expected_weight, rel_tol=1e-12, abs_tol=0.0
            ):
                raise RuntimeError(
                    f"{arm.name}: {mode} run has malformed responder {position} "
                    f"at step {step}: weight {weight_number!r} does not match "
                    f"c_tokens^2/c_steps ({expected_weight!r})"
                )
            responder_ids.append(responder_id)
        normalized_ids = sorted(responder_ids)
        if normalized_ids != expected_responders:
            raise RuntimeError(
                f"{arm.name}: {mode} run has invalid responders at step {step}: "
                f"expected IDs {expected_responders}, got {normalized_ids}"
            )


def _probe_record_error(
    arm: Arm, index: int, record: dict, message: str
) -> NoReturn:
    step = record.get("step", "?")
    fragment = record.get("fragment", "?")
    raise RuntimeError(
        f"{arm.name}: malformed action-probe record {index} "
        f"(step={step}, fragment={fragment}): {message}"
    )


def _positive_finite_probe_number(
    arm: Arm, index: int, record: dict, field: str
) -> float:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _probe_record_error(
            arm, index, record, f"{field} must be a positive finite number"
        )
    try:
        number = float(value)
    except (OverflowError, ValueError):
        _probe_record_error(
            arm, index, record, f"{field} must be a positive finite number"
        )
    if not math.isfinite(number) or number <= 0:
        _probe_record_error(
            arm, index, record, f"{field} must be a positive finite number"
        )
    return number


def _probe_percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _is_transport_or_config_fallback(reason: str) -> bool:
    return reason in ACTION_PROBE_TRANSPORT_CONFIG_FALLBACK_REASONS or reason.startswith(
        ACTION_PROBE_TRANSPORT_CONFIG_FALLBACK_PREFIXES
    )


def validate_action_probe_run(
    arm: Arm,
    records: list[dict],
    arm_dir: Path,
    *,
    expected_steps: int = 0,
) -> dict:
    policy = arm.commit_policy
    if policy == "cttn_shadow_v1":
        validate_event_tape_records(arm, records, expected_steps=expected_steps)
        for index, record in enumerate(records, 1):
            if record.get("policy") != policy:
                _probe_record_error(arm, index, record, "wrong shadow policy")
            if record.get("committed_action") != "SGD":
                _probe_record_error(
                    arm, index, record, "CTTN shadow must commit ordinary SGD"
                )
            if record.get("committed_multiplier") != 1.0:
                _probe_record_error(
                    arm, index, record, "CTTN shadow SGD multiplier must be 1"
                )
        from scripts.analyze_cttn_shadow import analyze_records

        analysis = analyze_records(records, expected_samples=32)
        summary = {
            "arm": arm.name,
            "policy": policy,
            "record_count": len(records),
            **analysis,
        }
        (arm_dir / ACTION_PROBE_SUMMARY_FILENAME).write_text(
            json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n"
        )
        (arm_dir / "cttn_shadow_analysis.json").write_text(
            json.dumps(analysis, indent=2, sort_keys=True) + "\n"
        )
        return summary
    probe_policies = ACTION_PROBE_SHADOW_POLICIES | ACTION_PROBE_ACTIVE_POLICIES
    if policy not in probe_policies:
        raise ValueError(f"{policy} is not an action-probe commit policy")

    validate_event_tape_records(arm, records, expected_steps=expected_steps)
    if not records:
        raise RuntimeError(f"{arm.name}: action-probe event tape has no records")
    action_names = (
        ("A0", "CTTN")
        if policy == "cttn_v1"
        else ("A0", "CTTN-SCALAR")
        if policy == "cttn_scalar_v1"
        else ACTION_PROBE_ACTIONS
    )

    selected_counts: Counter[str] = Counter()
    committed_counts: Counter[str] = Counter()
    fallback_reasons: Counter[str] = Counter()
    latencies = []
    fallback_count = 0
    transport_config_fallback_count = 0

    for index, record in enumerate(records, 1):
        if record.get("policy") != policy:
            _probe_record_error(
                arm,
                index,
                record,
                f"policy must be {policy!r}, got {record.get('policy')!r}",
            )

        selected_action = record.get("selected_action")
        committed_action = record.get("committed_action")
        if selected_action not in action_names:
            _probe_record_error(
                arm, index, record, f"unknown selected_action {selected_action!r}"
            )
        if committed_action not in action_names:
            _probe_record_error(
                arm, index, record, f"unknown committed_action {committed_action!r}"
            )

        selected_multiplier = _positive_finite_probe_number(
            arm, index, record, "selected_multiplier"
        )
        committed_multiplier = _positive_finite_probe_number(
            arm, index, record, "committed_multiplier"
        )
        latency = _positive_finite_probe_number(
            arm, index, record, "probe_latency_ms"
        )
        request_digest = record.get("request_digest")
        if not isinstance(request_digest, str) or not ACTION_PROBE_DIGEST_RE.fullmatch(
            request_digest
        ):
            _probe_record_error(
                arm, index, record, "request_digest must be exactly 64 hexadecimal characters"
            )

        fallback = record.get("fallback")
        fallback_reason = record.get("fallback_reason")
        if not isinstance(fallback, bool):
            _probe_record_error(arm, index, record, "fallback must be boolean")
        if fallback_reason is not None and (
            not isinstance(fallback_reason, str)
            or not fallback_reason
            or len(fallback_reason) > 256
        ):
            _probe_record_error(
                arm,
                index,
                record,
                "fallback_reason must be null or a non-empty string of at most 256 characters",
            )
        if fallback != (selected_action == "A0"):
            _probe_record_error(
                arm,
                index,
                record,
                "fallback must be true exactly when A0 is selected",
            )
        if fallback and fallback_reason is None:
            _probe_record_error(
                arm, index, record, "an A0 fallback must include fallback_reason"
            )
        if not fallback and fallback_reason is not None:
            _probe_record_error(
                arm, index, record, "a selected alternative cannot carry fallback_reason"
            )

        if selected_action == "A0" and selected_multiplier != 1.0:
            _probe_record_error(
                arm, index, record, "A0 selected_multiplier must be exactly 1.0"
            )
        if policy in ("probe_lr_shadow", "probe_lr_v1"):
            expected_multiplier = ACTION_PROBE_SCALAR_MULTIPLIERS[selected_action]
            if selected_multiplier != expected_multiplier:
                _probe_record_error(
                    arm,
                    index,
                    record,
                    f"{selected_action} selected_multiplier must be exactly "
                    f"{expected_multiplier} for {policy}",
                )
        if policy in ("cttn_v1", "cttn_scalar_v1") and not fallback:
            if record.get("cttn_bind") not in (True, False):
                _probe_record_error(
                    arm, index, record, "successful CTTN record must include cttn_bind"
                )
            tau = record.get("cttn_tau")
            zero_budget_limit = (
                tau is None
                and record.get("cttn_bind") is True
                and record.get("cttn_budget") == 0.0
                and record.get("cttn_e_after") == 0.0
            )
            if not zero_budget_limit and (
                not isinstance(tau, (int, float)) or not math.isfinite(tau)
            ):
                _probe_record_error(
                    arm,
                    index,
                    record,
                    "cttn_tau must be finite or null for the binding zero-budget limit",
                )
            for field in (
                "cttn_retention",
                "cttn_e_before",
                "cttn_e_after",
                "cttn_budget",
                "cttn_ritz_max",
                "cttn_loss",
            ):
                value = record.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    _probe_record_error(
                        arm, index, record, f"{field} must be a finite number"
                    )
            n_modes_90 = record.get("cttn_n_modes_90")
            if not isinstance(n_modes_90, int) or isinstance(n_modes_90, bool) or n_modes_90 < 0:
                _probe_record_error(
                    arm, index, record, "cttn_n_modes_90 must be a non-negative integer"
                )

        if policy in ACTION_PROBE_SHADOW_POLICIES:
            if committed_action != "A0" or committed_multiplier != 1.0:
                _probe_record_error(
                    arm,
                    index,
                    record,
                    "shadow policy must commit exactly A0 with multiplier 1.0",
                )
        elif (
            committed_action != selected_action
            or committed_multiplier != selected_multiplier
        ):
            _probe_record_error(
                arm,
                index,
                record,
                "active policy must commit exactly the selected action and multiplier",
            )

        selected_counts[selected_action] += 1
        committed_counts[committed_action] += 1
        latencies.append(latency)
        if fallback:
            fallback_count += 1
            fallback_reasons[fallback_reason] += 1
            if _is_transport_or_config_fallback(fallback_reason):
                transport_config_fallback_count += 1

    def ordered_counts(counts: Counter[str]) -> dict[str, int]:
        return {action: counts[action] for action in action_names if counts[action]}

    summary = {
        "arm": arm.name,
        "policy": policy,
        "record_count": len(records),
        "fallback_count": fallback_count,
        "fallback_reason_counts": dict(sorted(fallback_reasons.items())),
        "transport_config_fallback_count": transport_config_fallback_count,
        "selected_action_counts": ordered_counts(selected_counts),
        "committed_action_counts": ordered_counts(committed_counts),
        "probe_latency_ms": {
            "mean": sum(latencies) / len(latencies),
            "p95": _probe_percentile(latencies, 0.95),
        },
    }
    summary_path = arm_dir / ACTION_PROBE_SUMMARY_FILENAME
    try:
        summary_path.write_text(
            json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n"
        )
    except OSError as exc:
        raise RuntimeError(
            f"{arm.name}: cannot write action-probe run summary {summary_path}: {exc}"
        ) from exc
    if transport_config_fallback_count == len(records):
        reasons = ", ".join(
            f"{reason}={count}" for reason, count in sorted(fallback_reasons.items())
        )
        raise RuntimeError(
            f"{arm.name}: all {len(records)} action-probe decisions were "
            f"transport/config fallbacks ({reasons})"
        )
    return summary


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _git_metadata(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return f"unavailable: {shlex.join(command)}: {exc}\n"
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
        return (
            f"unavailable: {shlex.join(command)} exited {result.returncode}: "
            f"{detail}\n"
        )
    return result.stdout


def persist_reproducibility_metadata(
    report_dir: Path, argv: list[str] | None = None
) -> Path:
    """Write replay command and current git state beside the report directory."""
    run_root = report_dir.parent
    command_text = shlex.join(list(sys.argv if argv is None else argv)) + "\n"
    commit_text = _git_metadata(["git", "rev-parse", "HEAD"])
    diff_text = _git_metadata(["git", "diff"])
    try:
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "command.sh").write_text(command_text)
        (run_root / "git_commit.txt").write_text(commit_text)
        (run_root / "git_diff.patch").write_text(diff_text)
    except OSError as exc:
        raise RuntimeError(
            f"cannot write reproducibility metadata under {run_root}: {exc}"
        ) from exc
    return run_root


def split_data(
    data: str,
    work: Path,
    eval_rows: int,
    max_rows: int | None,
    shuffle_seed: int | None = None,
    eval_split_seed: int | None = None,
    confirmation_audit_rows: int = 0,
) -> tuple[Path, Path, int]:
    """Materialize --data as train.jsonl / eval.jsonl under `work`.

    The eval split comes off the END of the row stream so every arm trains
    on an identical prefix and none has seen the eval rows.
    """
    from yeto.data import load_rows

    ds = load_rows(data)
    source_rows = len(ds)
    if confirmation_audit_rows < 0:
        raise SystemExit("--confirmation-audit-rows must be non-negative")
    if source_rows <= eval_rows + confirmation_audit_rows:
        raise SystemExit(
            f"--data has {source_rows} usable rows; need > evaluation rows "
            f"{eval_rows + confirmation_audit_rows}"
        )
    work.mkdir(parents=True, exist_ok=True)
    idxs = list(range(source_rows))
    if eval_split_seed is None:
        # Historical behavior for non-phase-map callers: shuffle the selected
        # prefix before splitting. New evidence commands must pass the separate
        # frozen eval seed below.
        n = source_rows
        if max_rows is not None:
            n = min(n, max_rows + eval_rows)
        idxs = idxs[:n]
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(idxs)
        train_idxs = idxs[: n - eval_rows]
        eval_idxs = idxs[n - eval_rows :]
    else:
        # Select the locked evaluation set once, then shuffle only the disjoint
        # training remainder. This keeps eval identity invariant across every
        # development/confirmation training seed.
        population_count = source_rows
        if max_rows is not None:
            population_count = min(
                source_rows,
                max_rows + eval_rows + confirmation_audit_rows,
            )
        idxs = idxs[:population_count]
        random.Random(eval_split_seed).shuffle(idxs)
        train_count = population_count - eval_rows - confirmation_audit_rows
        train_pool_idxs = idxs[:train_count]
        eval_idxs = idxs[train_count : train_count + eval_rows]
        audit_eval_idxs = idxs[train_count + eval_rows :]
        train_idxs = list(train_pool_idxs)
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(train_idxs)
        if max_rows is not None:
            train_idxs = train_idxs[:max_rows]

    def dump(path: Path, idxs) -> None:
        with open(path, "w") as f:
            for i in idxs:
                row = ds[i]
                f.write(
                    json.dumps(
                        {k: row[k] for k in ("messages", "tools") if k in row},
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    train, evalf = work / "train.jsonl", work / "eval.jsonl"
    dump(train, train_idxs)
    dump(evalf, eval_idxs)
    audit_evalf = work / "confirmation-audit.jsonl"
    if confirmation_audit_rows:
        dump(audit_evalf, audit_eval_idxs)
    index_payload = {
        "schema": "yeto_split_provenance_v1",
        "source_row_count": source_rows,
        "source_population_count": (
            len(train_pool_idxs) + len(eval_idxs) + confirmation_audit_rows
            if eval_split_seed is not None
            else len(train_idxs) + len(eval_idxs)
        ),
        "train_row_count": len(train_idxs),
        "eval_row_count": len(eval_idxs),
        "eval_split_seed": eval_split_seed,
        "train_shuffle_seed": shuffle_seed,
        "train_source_indices": train_idxs,
        "eval_source_indices": eval_idxs,
    }
    if eval_split_seed is not None:
        index_payload["train_pool_source_indices"] = train_pool_idxs
        index_payload["confirmation_audit_row_count"] = confirmation_audit_rows
        index_payload["audit_eval_source_indices"] = audit_eval_idxs
    (work / "split_provenance.json").write_text(
        json.dumps(index_payload, indent=2, sort_keys=True) + "\n"
    )
    return train, evalf, len(train_idxs)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tensor(tensor, dtype: str) -> str:
    """Hash one tensor in an explicit little-endian representation."""
    array = tensor.detach().cpu().contiguous().numpy().astype(dtype, copy=False)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def materialize_eval_provenance(
    model_id: str,
    eval_file: Path,
    seq_len: int,
    output_dir: Path,
    train_on: str = "assistant",
    split_provenance: Path | None = None,
) -> dict:
    """Freeze exact source-row and packed-token identities before training."""
    import torch
    from transformers import AutoTokenizer

    from yeto.data import build_packed_dataset
    from yeto.models import resolve

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "eval_rows.jsonl"
    sequences_path = output_dir / "eval_sequences.jsonl"
    packed_path = output_dir / "eval_packed.jsonl"
    summary_path = output_dir / "eval_provenance.json"

    row_records = []
    with eval_file.open("rb") as handle:
        for row_index, raw in enumerate(handle):
            if not raw.strip():
                continue
            row_records.append(
                {
                    "row_index": row_index,
                    "row_id": hashlib.sha256(raw.rstrip(b"\r\n")).hexdigest(),
                    "raw_line_sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
    rows_path.write_text(
        "".join(
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
            for record in row_records
        )
    )
    example_ids = [record["row_id"] for record in row_records]
    example_ids_hash = hashlib.sha256(
        json.dumps(
            example_ids,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    resolved = resolve(model_id)
    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    dataset = build_packed_dataset(
        str(eval_file), tokenizer, 0, 1, seq_len, train_on=train_on
    )
    sequence_records = []
    packed_records = []
    token_id_arrays = []
    for sequence_index in range(len(dataset)):
        input_ids, weights = dataset[sequence_index]
        input_ids_list = [int(value) for value in input_ids.tolist()]
        weights_list = [float(value) for value in weights.tolist()]
        attention_mask = [1] * len(input_ids_list)
        target_token_mask = [bool(value > 0) for value in weights_list[1:]]
        token_id_arrays.append(input_ids_list)
        input_ids_sha256 = _sha256_tensor(input_ids, "<i8")
        weights_sha256 = _sha256_tensor(weights, "<f4")
        attention_mask_sha256 = _sha256_tensor(
            input_ids.new_ones(input_ids.shape, dtype=torch.uint8),
            "|u1",
        )
        sequence_records.append(
            {
                "sequence_index": sequence_index,
                "sequence_id": hashlib.sha256(
                    (input_ids_sha256 + weights_sha256).encode("ascii")
                ).hexdigest(),
                "input_ids_sha256": input_ids_sha256,
                "labels_sha256": input_ids_sha256,
                "attention_mask_sha256": attention_mask_sha256,
                "supervision_weights_sha256": weights_sha256,
                "target_token_mask_sha256": hashlib.sha256(
                    bytes(target_token_mask)
                ).hexdigest(),
                "sequence_length": int(input_ids.numel()),
                "supervised_token_count": int((weights[1:] > 0).sum().item()),
            }
        )
        packed_records.append(
            {
                "sequence_index": sequence_index,
                "input_ids": input_ids_list,
                "labels": input_ids_list,
                "attention_mask": attention_mask,
                "supervision_weights": weights_list,
                "sequence_length": len(input_ids_list),
                "target_token_mask": target_token_mask,
                "target_token_count": sum(target_token_mask),
            }
        )
    sequences_path.write_text(
        "".join(
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
            for record in sequence_records
        )
    )
    packed_path.write_text(
        "".join(
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
            for record in packed_records
        )
    )
    token_ids_registry_hash = hashlib.sha256(
        json.dumps(
            token_id_arrays,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    summary = {
        "schema": "yeto_eval_provenance_v1",
        "created_before_training": True,
        "model": model_id,
        "resolved_model": str(resolved),
        "seq_len": seq_len,
        "train_on": train_on,
        "eval_file_sha256": _sha256_file(eval_file),
        "eval_row_count": len(row_records),
        "eval_sequence_count": len(sequence_records),
        "eval_supervised_token_count": sum(
            record["supervised_token_count"] for record in sequence_records
        ),
        "eval_rows_hash": _sha256_file(eval_file),
        "eval_example_ids_hash": example_ids_hash,
        "eval_packed_hash": _sha256_file(packed_path),
        "eval_token_ids_hash": token_ids_registry_hash,
        "rows_file": rows_path.name,
        "sequences_file": sequences_path.name,
        "packed_file": packed_path.name,
    }
    if split_provenance is not None:
        split = json.loads(split_provenance.read_text())
        summary.update(
            {
                "eval_split_seed": split.get("eval_split_seed"),
                "train_shuffle_seed": split.get("train_shuffle_seed"),
                "eval_source_indices_hash": hashlib.sha256(
                    json.dumps(
                        split.get("eval_source_indices"),
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
                "train_source_indices_hash": hashlib.sha256(
                    json.dumps(
                        split.get("train_source_indices"),
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
                "train_pool_source_indices_hash": hashlib.sha256(
                    json.dumps(
                        split.get(
                            "train_pool_source_indices",
                            split.get("train_source_indices"),
                        ),
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
                "split_provenance_sha256": _sha256_file(split_provenance),
            }
        )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def validate_materialized_anchor_disjointness(
    *,
    anchor_hashes: set[str],
    data_files: dict[str, Path],
    summary_path: Path,
    manifest_sha256: str,
    anchor_data_sha256: str,
) -> dict:
    """Fail if the exact materialized train/eval rows overlap the anchor."""

    from yeto.action_probe import canonical_anchor_hash

    files = {}
    all_overlaps = set()
    for label, path in data_files.items():
        row_hashes = []
        try:
            lines = path.read_text().splitlines()
        except OSError as exc:
            raise RuntimeError(f"cannot read materialized {label} data {path}: {exc}") from exc
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"{path}:{line_number}: invalid materialized data JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_number}: expected a JSON object")
            row_hashes.append(canonical_anchor_hash(row))
        overlap = sorted(anchor_hashes.intersection(row_hashes))
        all_overlaps.update(overlap)
        files[label] = {
            "path": str(path),
            "row_count": len(row_hashes),
            "unique_canonical_count": len(set(row_hashes)),
            "overlap_count": len(overlap),
            "overlap_hashes": overlap,
        }

    summary = {
        "schema": "materialized_anchor_overlap_v1",
        "manifest_sha256": manifest_sha256,
        "anchor_data_sha256": anchor_data_sha256,
        "anchor_unique_canonical_count": len(anchor_hashes),
        "files": files,
        "overlap_count": len(all_overlaps),
        "verified_zero_overlap": not all_overlaps,
    }
    try:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    except OSError as exc:
        raise RuntimeError(
            f"cannot write materialized anchor-overlap summary {summary_path}: {exc}"
        ) from exc
    if all_overlaps:
        raise RuntimeError(
            "materialized probe run data overlaps the action-probe anchor: "
            f"{len(all_overlaps)} canonical row(s); see {summary_path}"
        )
    return summary


def eval_loss_per_token(
    model_id: str,
    adapter_dir: Path | None,
    eval_file: Path,
    seq_len: int,
    device: str,
    train_on: str = "assistant",
    tuning: str = "lora",
    output_path: Path | None = None,
) -> float:
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
    if adapter_dir is not None and tuning == "full":
        # Full-parameter export is a complete model directory, not an adapter.
        model = AutoModelForCausalLM.from_pretrained(
            str(adapter_dir), dtype=dtype, trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            resolved, dtype=dtype, trust_remote_code=True
        )
        if adapter_dir is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.to(device).eval()
    ds = build_packed_dataset(str(eval_file), tok, 0, 1, seq_len, train_on=train_on)
    total_loss, total_tokens = 0.0, 0
    records = []
    with torch.no_grad():
        for i in range(len(ds)):
            ids, weights = ds[i]
            ids = ids.unsqueeze(0).to(device)
            weights = weights.unsqueeze(0).to(device)
            out = model(input_ids=ids)
            loss, n = sft_loss(out.logits, ids, "cross_entropy", weights)
            loss_sum = float(loss.item())
            token_count = int(n.item())
            total_loss += loss_sum
            total_tokens += token_count
            input_ids_sha256 = _sha256_tensor(ids.squeeze(0), "<i8")
            weights_sha256 = _sha256_tensor(weights.squeeze(0), "<f4")
            attention_mask_sha256 = _sha256_tensor(
                torch.ones_like(ids.squeeze(0), dtype=torch.uint8), "|u1"
            )
            target_token_mask = [
                bool(value > 0)
                for value in weights.squeeze(0).detach().cpu().tolist()[1:]
            ]
            records.append(
                {
                    "sequence_index": i,
                    "sequence_id": hashlib.sha256(
                        (input_ids_sha256 + weights_sha256).encode("ascii")
                    ).hexdigest(),
                    "input_ids_sha256": input_ids_sha256,
                    "labels_sha256": input_ids_sha256,
                    "attention_mask_sha256": attention_mask_sha256,
                    "supervision_weights_sha256": weights_sha256,
                    "target_token_mask_sha256": hashlib.sha256(
                        bytes(target_token_mask)
                    ).hexdigest(),
                    "sequence_length": int(ids.numel()),
                    "loss_sum": loss_sum,
                    "token_count": token_count,
                    "loss_per_token": loss_sum / max(token_count, 1),
                }
            )
    aggregate = total_loss / max(total_tokens, 1)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
        )
    return aggregate


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
            capture_output=True,
            text=True,
        ).stdout.strip()
        index_to_uuid = {}
        for line in gpu_out.splitlines():
            parts = [v.strip() for v in line.split(",")]
            if len(parts) >= 2 and parts[0].isdigit():
                index_to_uuid[int(parts[0])] = parts[1]
        missing = [i for i in gpu_ids if i not in index_to_uuid]
        if missing:
            raise RuntimeError(f"cannot map GPU id(s) {missing} from nvidia-smi output")
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
            [
                "nvidia-smi",
                f"--query-compute-apps={query}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
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


def eval_in_subprocess(
    args,
    adapter_dir: Path | None,
    eval_file: Path,
    output_path: Path | None = None,
) -> float:
    """Score in a child process so the model's VRAM is released on exit —
    an in-process eval would keep the base resident on GPU 0 and starve the
    next arm's learners (found the hard way on a 4xL40S box)."""
    cmd = [
        sys.executable,
        __file__,
        "--eval-only",
        "--model",
        args.model,
        "--data",
        str(eval_file),
        "--seq-len",
        str(args.seq_len),
        "--device",
        args.device,
        "--tuning",
        getattr(args, "tuning", "lora"),
    ]
    if adapter_dir is not None:
        cmd += ["--adapter-dir", str(adapter_dir)]
    if output_path is not None:
        cmd += ["--eval-output", str(output_path)]
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


def write_acquisition_state(report_dir: Path, phase: str, **extra) -> None:
    """Atomically expose lifecycle phase without inferring scientific cause."""
    path = report_dir / "acquisition-state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "yeto_acquisition_state_v1", "phase": phase, **extra}
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_baseline(args, work: Path) -> tuple[Path, float]:
    arm_dir = work / "baseline"
    steps = steps_for(
        args.token_budget,
        args.micro_batch_size,
        args.seq_len,
        1,
        world=max(1, args.learner_gpus),
    )
    cmd = learner_command(
        args,
        arm_dir,
        learner_id=0,
        num_learners=1,
        syncer="none",
        max_steps=steps,
        arm=None,
    )
    wait_for_free_gpus(args.device, gpu_ids=assigned_gpu_ids(args))
    t0 = time.monotonic()
    run_checked(cmd, arm_dir / "learner.log", env=learner_env(args, 0))
    return arm_dir / "learner-0", time.monotonic() - t0


def run_diloco(args, arm: Arm, work: Path) -> tuple[Path, float]:
    arm_dir = work / arm.name
    arm_dir.mkdir(parents=True, exist_ok=True)
    budget_steps = steps_for(
        args.token_budget,
        args.micro_batch_size,
        args.seq_len,
        arm.m,
        world=max(1, args.learner_gpus),
    )
    steps = args.learner_max_steps or budget_steps
    total_outer_steps = args.syncer_total_steps or (budget_steps * arm.m * 4)
    port = free_port()
    # Generous round ceiling: learners stop at their token budget, and the
    # syncer is terminated once they exit; the checkpoint (written every
    # round) carries the merged params up to the last completed round.
    sidecar = None
    syncer = None
    syncer_log = None
    t0 = None
    learners = []
    learner_logs = []
    try:
        if arm.commit_policy != "token_weighted":
            wait_for_free_gpus(
                "cuda",
                gpu_ids=list(args.action_probe_gpus),
                timeout_s=int(args.action_probe_startup_timeout_s),
            )
            sidecar = launch_action_probe(args, arm, arm_dir)
        syncer_args = {}
        if sidecar is not None:
            syncer_args = {
                "action_probe_endpoint": sidecar.endpoint,
                "action_probe_timeout_ms": max(
                    1, int(args.action_probe_timeout_s * 1000)
                ),
                "action_probe_run_uuid": sidecar.run_uuid,
                "action_probe_expected_config": sidecar.expected_config,
            }
        syncer_log = open(arm_dir / "syncer.log", "w")
        syncer = subprocess.Popen(
            syncer_command(
                arm,
                port,
                arm_dir,
                total_steps=total_outer_steps,
                checkpoint_every=getattr(args, "syncer_checkpoint_every", 1),
                probe_capture=getattr(args, "syncer_probe_capture", False),
                probe_capture_every=getattr(args, "syncer_probe_capture_every", 1),
                delta_norm_ref=getattr(args, "delta_norm_ref", 0.0),
                version_matched_anchor=getattr(args, "version_matched_anchor", False),
                anchor_drift_log=getattr(args, "anchor_drift_log", False),
                cttn_rho=getattr(args, "cttn_rho", 0.10),
                cttn_shadow_samples=getattr(args, "cttn_shadow_samples", 32),
                **syncer_args,
            ),
            stdout=syncer_log,
            stderr=subprocess.STDOUT,
        )
        wait_for_free_gpus(args.device, gpu_ids=assigned_gpu_ids(args))
        t0 = time.monotonic()
        for i in range(arm.m):
            cmd = learner_command(
                args,
                arm_dir,
                learner_id=i,
                num_learners=arm.m,
                syncer=f"127.0.0.1:{port}",
                max_steps=steps,
                arm=arm,
            )
            log = open(arm_dir / f"learner-{i}.log", "w")
            learner_logs.append(log)
            learners.append(
                subprocess.Popen(
                    cmd,
                    cwd=REPO_ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=learner_env(args, i),
                )
            )
        for proc in learners:
            rc = proc.wait(timeout=args.arm_timeout_min * 60)
            if rc != 0:
                raise RuntimeError(f"{arm.name}: a learner exited {rc}; see {arm_dir}")
        if args.syncer_total_steps:
            # Full-parameter models leave the syncer a sizable merge +
            # checkpoint backlog after learners disconnect; a 30s wait
            # killed a healthy SmolLM2-135M full-tune run.
            #
            # CTTN policies (esp. the shadow diagnostic) run real-Hessian
            # block-Lanczos on held-out data in the sidecar. Much of that
            # work is still draining when the learners disconnect, and each
            # HVP sample is ~8 min, so the tail can take *hours*, not the
            # flat 900s. A 900s cap force-killed a healthy shadow run at
            # its m3 (M=3, 7813 steps/learner) segment. Scale the drain
            # budget with the sample count for any CTTN HVP policy.
            syncer_drain_timeout = 900
            if arm.outer_optimizer in ("cttn", "cttn-scalar", "cttn-shadow"):
                samples = getattr(args, "cttn_shadow_samples", 32) or 32
                # ~15 min/sample ceiling (matches the per-probe
                # action-probe-timeout) + generous margin; the syncer
                # exits as soon as it is actually done, so this is only
                # an upper bound on patience.
                syncer_drain_timeout = max(900, samples * 900)
            rc = syncer.wait(timeout=syncer_drain_timeout)
            if rc != 0:
                raise RuntimeError(f"{arm.name}: syncer exited {rc}; see {arm_dir}")
    finally:
        for proc in learners:
            if proc.poll() is None:
                proc.terminate()
        for proc in learners:
            if proc.poll() is None:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
        for log in learner_logs:
            log.close()
        if syncer is not None and syncer.poll() is None:
            syncer.terminate()
        if syncer is not None:
            try:
                syncer.wait(timeout=30)
            except subprocess.TimeoutExpired:
                syncer.kill()
                syncer.wait(timeout=5)
        if syncer_log is not None:
            syncer_log.close()
        if sidecar is not None:
            stop_action_probe(sidecar)
    assert t0 is not None
    wall = time.monotonic() - t0
    ckpt = arm_dir / "state.ckpt"
    if not ckpt.exists():
        raise RuntimeError(
            f"{arm.name}: no syncer checkpoint (no round completed); see {arm_dir}"
        )
    probe_policy = arm.commit_policy != "token_weighted"
    if args.syncer_total_steps or arm.strict_quorum or probe_policy:
        tape_path = arm_dir / "tape.jsonl"
        records = _load_event_tape(arm, tape_path)
        if probe_policy:
            validate_action_probe_run(
                arm,
                records,
                arm_dir,
                expected_steps=args.syncer_total_steps,
            )
        else:
            validate_event_tape_records(
                arm, records, expected_steps=args.syncer_total_steps
            )
    # Export the merged global parameters to a peft adapter dir.
    export_dir = arm_dir / "export"
    run_checked(
        [
            sys.executable,
            "-m",
            "yeto.export",
            "--checkpoint",
            str(ckpt),
            "--model",
            args.model,
            "--tuning",
            args.tuning,
            "--lora-r",
            str(args.lora_r),
            "--lora-alpha",
            str(args.lora_alpha),
            "--fragments",
            str(arm.fragments),
            "--fragment-pattern",
            arm.fragment_pattern,
            "--output-dir",
            str(export_dir),
            "--device",
            "cpu",
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
    syncer_dir = REPO_ROOT / "syncer"
    build_inputs = [syncer_dir / "Cargo.toml", syncer_dir / "Cargo.lock"]
    build_inputs.extend(sorted((syncer_dir / "src").rglob("*.rs")))
    binary_mtime = SYNCER_BIN.stat().st_mtime_ns if SYNCER_BIN.exists() else None
    stale = binary_mtime is None or any(
        path.is_file() and path.stat().st_mtime_ns > binary_mtime
        for path in build_inputs
    )
    if stale:
        print("[compare] building syncer (cargo build --release)")
        subprocess.run(
            ["cargo", "build", "--release", "-q"], cwd=syncer_dir, check=True
        )


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="lfm25-230m")
    p.add_argument(
        "--data", required=True, help="messages-format chat rows (HF id or local path)"
    )
    p.add_argument(
        "--token-budget",
        type=int,
        default=500_000,
        help="total training tokens per arm (split across an arm's learners)",
    )
    p.add_argument(
        "--settings", default="m2", help=f"comma list of {list(PRESETS)} or 'all'"
    )
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--micro-batch-size", type=int, default=2)
    p.add_argument("--inner-lr", type=float, default=3e-4)
    p.add_argument(
        "--tuning",
        choices=["lora", "full"],
        default="lora",
        help="learner tuning mode; 'full' trains and exports every parameter",
    )
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument(
        "--eval-rows", type=int, default=64, help="held-out rows for scoring"
    )
    p.add_argument(
        "--prebound-development-eval",
        type=Path,
        default=None,
        help="use this already-frozen development file with --data as the "
        "already-frozen training file; no source split is materialized",
    )
    p.add_argument(
        "--round-interval-ms",
        type=int,
        default=None,
        help="override the round-launch floor of throttled presets "
        "(m2h24): the right value is H * step_time / fragments, and "
        "step time depends on hardware",
    )
    p.add_argument(
        "--baseline-loss",
        type=float,
        default=None,
        help="skip the synchronous baseline arm and compare against "
        "this eval loss/token (from a previous run with the same "
        "model, data, seed and budget)",
    )
    p.add_argument(
        "--skip-baseline",
        action="store_true",
        help="do not run or inject a synchronous baseline; use live async controls",
    )
    p.add_argument(
        "--skip-untrained-eval",
        action="store_true",
        help="do not repeatedly score the untrained model in phase-map cells",
    )
    p.add_argument("--max-rows", type=int, default=None, help="cap training rows")
    p.add_argument(
        "--shuffle-rows-seed",
        type=int,
        default=None,
        help="deterministically shuffle only the training remainder when "
        "--eval-split-seed is set",
    )
    p.add_argument(
        "--eval-split-seed",
        type=int,
        default=None,
        help="freeze evaluation rows before applying --shuffle-rows-seed to "
        "the disjoint training remainder",
    )
    p.add_argument(
        "--confirmation-audit-rows",
        type=int,
        default=0,
        help="reserve a second locked loss-free audit split; ordinary compare "
        "runs never evaluate or emit outcomes for it",
    )
    p.add_argument(
        "--training-seed",
        type=int,
        default=0,
        help="base learner RNG seed; keep equal across compared arms",
    )
    p.add_argument(
        "--device", default="cpu", help="learner/eval device (cpu, mps, cuda)"
    )
    p.add_argument(
        "--shard",
        choices=["ddp", "fsdp"],
        default="ddp",
        help="multi-GPU strategy inside a learner (fsdp shards the "
        "frozen base when it exceeds one GPU)",
    )
    p.add_argument(
        "--learner-gpus",
        type=int,
        default=0,
        help="GPUs per learner; learner i owns the GPU block "
        "[i*g, (i+1)*g) and runs under torchrun when g > 1. "
        "0 = single process on --device",
    )
    p.add_argument(
        "--gpu-slots",
        type=int,
        default=0,
        help="when --learner-gpus 0 and --device cuda, assign "
        "single-process learners round-robin over this many GPUs",
    )
    p.add_argument(
        "--gpu-offset",
        type=int,
        default=0,
        help="first physical GPU id to assign when partitioning learners",
    )
    p.add_argument(
        "--require-distinct-learner-gpu-uuids",
        action="store_true",
        help="freeze and enforce a four-A100 learner-to-GPU-UUID bijection",
    )
    p.add_argument(
        "--action-probe-gpus",
        type=parse_gpu_ids,
        default=None,
        help="comma-separated physical GPUs reserved exclusively for the persistent action-probe sidecar",
    )
    p.add_argument(
        "--action-probe-anchor-manifest",
        type=Path,
        default=None,
        help="verified disjoint anchor manifest consumed by action_probe_server",
    )
    p.add_argument("--action-probe-timeout-s", type=float, default=300.0)
    p.add_argument("--action-probe-startup-timeout-s", type=float, default=1800.0)
    p.add_argument("--action-probe-run-uuid", default=None)
    p.add_argument(
        "--action-probe-min-gain",
        type=float,
        default=None,
        help=f"override the sidecar selector minimum gain (default {ACTION_PROBE_MIN_GAIN})",
    )
    p.add_argument(
        "--action-probe-lcb-z",
        type=float,
        default=None,
        help=f"override the sidecar selector LCB z score (default {ACTION_PROBE_LCB_Z})",
    )
    p.add_argument(
        "--action-probe-min-win-rate",
        type=float,
        default=None,
        help=f"override the sidecar selector panel win rate (default {ACTION_PROBE_MIN_WIN_RATE})",
    )
    p.add_argument("--action-probe-seq-len", type=int, default=128)
    p.add_argument("--action-probe-panels", type=int, default=8)
    p.add_argument("--action-probe-blocks-per-panel", type=int, default=2)
    p.add_argument(
        "--action-probe-lora-targets",
        choices=["auto", "attention", "all-linear"],
        default="auto",
    )
    p.add_argument(
        "--learner-step-sleep-ms",
        default="0",
        help="comma-separated per-learner sleep after each optimizer step",
    )
    p.add_argument(
        "--learner-push-delay-ms",
        default="0",
        help="comma-separated per-learner sleep before each fragment push",
    )
    p.add_argument(
        "--learner-delay-jitter-ms",
        type=float,
        default=0.0,
        help="uniform [0, jitter] ms added to each debug sleep",
    )
    p.add_argument(
        "--learner-broadcast-lag-commits",
        type=int,
        default=0,
        help="EXP: every learner applies each fragment broadcast only after "
        "this many newer commits for that fragment (K-commits-old base)",
    )
    p.add_argument(
        "--fixed-window-tokens",
        type=int,
        default=0,
        help="answer async pulls from snapshots taken after this "
        "many post-reset learner tokens",
    )
    p.add_argument(
        "--fixed-window-microsteps",
        type=int,
        default=0,
        help="answer async pulls from snapshots taken after this "
        "many post-reset optimizer steps",
    )
    p.add_argument(
        "--pad-to-fixed-window-tokens",
        action="store_true",
        help="accepted for fixed-token experiment configs; windows "
        "round to whole optimizer steps",
    )
    p.add_argument(
        "--fixed-window-schedule",
        default=None,
        help="EXP: per-learner online sync-horizon schedule "
        "'commit1:h1,commit2:h2,...' forwarded to every async learner "
        "(see yeto.learner --fixed-window-schedule)",
    )
    p.add_argument(
        "--freeze-delta-before-delay",
        action="store_true",
        help="materialize fragment payloads before push delay stress",
    )
    p.add_argument(
        "--barrier-sync",
        action="store_true",
        help="EXP: true lockstep (barrier-synchronized) DiLoCo — forwarded to "
        "every learner (see yeto.learner --barrier-sync). Each learner blocks "
        "after pushing a fragment delta until the syncer's merged broadcast "
        "for that fragment returns, taking no inner steps while a merge is in "
        "flight. Off by default (non-barrier strict-quorum schedule).",
    )
    p.add_argument(
        "--version-matched-anchor",
        action="store_true",
        help="EXP2.46: the syncer differences each learner's delta against the "
        "retained global at the learner's pushed base_version (version-matched "
        "anchoring, arms A/B) instead of the current global (current-anchor, arm "
        "C). Implies --anchor-drift-log. Off by default = byte-identical "
        "current-anchor.",
    )
    p.add_argument(
        "--anchor-drift-log",
        action="store_true",
        help="EXP2.46: log per-push anchor-drift diagnostics (||anchor_drift||, "
        "||true_local_delta||, ratio, cos(drift, outer momentum)) into the event "
        "tape WITHOUT changing the merge (arm C). Implied by "
        "--version-matched-anchor.",
    )
    p.add_argument("--arm-timeout-min", type=int, default=120)
    p.add_argument(
        "--syncer-total-steps",
        type=int,
        default=0,
        help="finish async arms after exactly this many outer steps; 0 keeps "
        "the learner-budget-driven ceiling",
    )
    p.add_argument(
        "--learner-max-steps",
        type=int,
        default=0,
        help="maximum local steps per async learner; 0 derives it from the token budget",
    )
    p.add_argument(
        "--strict-quorum",
        action="store_true",
        help="require the configured quorum for every merge and reject partial timeout/tail commits",
    )
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
        help="override outer Nesterov momentum or EMA beta for every selected async arm",
    )
    p.add_argument(
        "--outer-optimizer",
        choices=OUTER_OPTIMIZERS,
        default=None,
        help="override the outer optimizer for every selected async arm",
    )
    p.add_argument(
        "--cttn-rho",
        type=float,
        default=0.10,
        help="CTTN dimensionless transverse curvature budget",
    )
    p.add_argument(
        "--cttn-shadow-samples",
        type=int,
        default=32,
        help="stratified HVP samples for cttn-shadow (32 = 8 per fragment)",
    )
    p.add_argument(
        "--outer-restart-cos-threshold",
        type=float,
        default=None,
        help="override the restarted-EMA cosine threshold for every selected async arm",
    )
    p.add_argument(
        "--delta-correction",
        choices=["heloco", "none"],
        default=None,
        help="override delta correction for every selected async arm",
    )
    p.add_argument(
        "--matrix-merge",
        choices=MATRIX_MERGES,
        default=None,
        help="override matrix-fragment aggregation (rda, Iso-C 'iso', or "
        "'worker-snr' cross-worker consensus) for every selected async arm",
    )
    p.add_argument(
        "--commit-policy",
        choices=COMMIT_POLICIES,
        default=None,
        help="override the commit policy for every selected async arm",
    )
    p.add_argument(
        "--outer-lr-by-fragment",
        default=None,
        help="comma-separated per-fragment outer learning rates for every selected async arm",
    )
    p.add_argument("--probe-freshness-scale", type=float, default=24.0)
    p.add_argument(
        "--syncer-checkpoint-every",
        type=int,
        default=1,
        help="syncer checkpoint cadence in outer steps; raise for large "
        "(full-parameter) states where a per-step ~1GB write throttles the "
        "syncer (the final step is checkpointed when total steps divide it)",
    )
    p.add_argument(
        "--delta-norm-ref",
        type=float,
        default=0.0,
        help="rescale every merged delta to this L2 norm after the "
        "production merge and before the outer step (per fragment); "
        "post-merge renormalization for mediation-control experiments. "
        "0 = off (byte-identical production path)",
    )
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
    p.add_argument("--eval-output", type=Path, default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.syncer_total_steps < 0 or args.learner_max_steps < 0:
        p.error("--syncer-total-steps and --learner-max-steps must be non-negative")
    if not math.isfinite(args.cttn_rho) or args.cttn_rho < 0.0:
        p.error("--cttn-rho must be finite and non-negative")
    if args.cttn_shadow_samples <= 0:
        p.error("--cttn-shadow-samples must be positive")
    if args.fixed_window_schedule is not None:
        from yeto.learner import parse_fixed_window_schedule

        try:  # fail before any arm spends GPU time on a malformed schedule
            parse_fixed_window_schedule(args.fixed_window_schedule)
        except ValueError as exc:
            p.error(f"--fixed-window-schedule: {exc}")
    if args.strict_quorum and args.syncer_total_steps == 0:
        p.error(
            "--strict-quorum requires --syncer-total-steps so learners do not disconnect first"
        )
    if args.skip_baseline and args.baseline_loss is not None:
        p.error("--skip-baseline and --baseline-loss are mutually exclusive")

    if args.eval_only:
        loss = eval_loss_per_token(
            args.model,
            args.adapter_dir,
            Path(args.data),
            args.seq_len,
            args.device,
            tuning=args.tuning,
            output_path=args.eval_output,
        )
        # Full double precision (17 sig digits, round-trippable): 6dp rounding
        # once masked a spurious iso bit-identity (EXP2.40). eval_in_subprocess
        # and the summarizers parse this line, so the extra digits flow into
        # results.jsonl and every downstream comparison.
        print(f"EVAL_LOSS {loss:.17g}")
        return 0

    arms = apply_arm_overrides(
        select_arms(args.settings),
        outer_lr=args.outer_lr,
        outer_lr_by_fragment=args.outer_lr_by_fragment,
        outer_momentum=args.outer_momentum,
        outer_optimizer=args.outer_optimizer,
        outer_restart_cos_threshold=args.outer_restart_cos_threshold,
        delta_correction=args.delta_correction,
        commit_policy=args.commit_policy,
        matrix_merge=args.matrix_merge,
    )
    if args.round_interval_ms is not None:
        from dataclasses import replace as _replace

        arms = [
            _replace(a, round_interval_ms=args.round_interval_ms)
            if a.round_interval_ms
            else a
            for a in arms
        ]
    if args.strict_quorum:
        from dataclasses import replace as _replace

        arms = [_replace(a, strict_quorum=True) for a in arms]
    world = max(1, args.learner_gpus)
    base_steps = steps_for(
        args.token_budget, args.micro_batch_size, args.seq_len, 1, world
    )
    print(
        f"[compare] model={args.model} budget={args.token_budget} tokens "
        f"(baseline: {base_steps} steps of {args.micro_batch_size}x{args.seq_len}"
        f"{f' x{world} ranks' if world > 1 else ''})"
    )
    for arm in arms:
        s = steps_for(
            args.token_budget, args.micro_batch_size, args.seq_len, arm.m, world
        )
        print(
            f"  {arm.name:<10} M={arm.m} {s} steps/learner "
            f"P={arm.fragments} alpha={arm.merge_alpha} wire={arm.wire_dtype} "
            f"pipeline={arm.pipeline} optimizer={arm.outer_optimizer} "
            f"restart_cos={arm.outer_restart_cos_threshold} "
            f"correction={arm.delta_correction} policy={arm.commit_policy}"
        )
    if args.dry_run:
        return 0

    if args.learner_gpus > 0 and args.gpu_slots > 0:
        raise SystemExit("--gpu-slots is only valid when --learner-gpus is 0")
    probe_arms = [arm for arm in arms if arm.commit_policy != "token_weighted"]
    if probe_arms:
        if args.action_probe_gpus is None:
            raise SystemExit("probe shadow/active arms require --action-probe-gpus")
        if len(args.action_probe_gpus) > 4:
            raise SystemExit("action_probe_server supports at most four probe GPUs")
        if args.action_probe_anchor_manifest is None:
            raise SystemExit(
                "probe shadow/active arms require --action-probe-anchor-manifest"
            )
        if args.action_probe_timeout_s <= 0 or args.action_probe_startup_timeout_s <= 0:
            raise SystemExit(
                "action-probe request and startup timeouts must be positive"
            )
        if (
            args.action_probe_seq_len <= 1
            or args.action_probe_panels < 2
            or args.action_probe_blocks_per_panel <= 0
        ):
            raise SystemExit(
                "action-probe seq-len > 1, panels >= 2, and blocks-per-panel > 0 are required"
            )
        non_m4 = [
            arm.name
            for arm in probe_arms
            if arm.commit_policy in LOO_COMMIT_POLICIES and arm.m != 4
        ]
        if non_m4:
            raise SystemExit(
                f"probe_loo_v1 requires exactly four learners; invalid arms: {non_m4}"
            )
        learner_ids = assigned_gpu_ids(args, arms)
        if args.device.startswith("cuda") and learner_ids is None:
            raise SystemExit(
                "probe arms require --learner-gpus or --gpu-slots so learner and probe GPU sets can be proven disjoint"
            )
        overlap = sorted(set(learner_ids or ()) & set(args.action_probe_gpus))
        if overlap:
            raise SystemExit(
                f"learner and action-probe GPU sets must be disjoint; overlap={overlap}"
            )

    if args.learner_gpus > 0 or args.gpu_slots > 0 or probe_arms:
        import torch

        have = torch.cuda.device_count()
        requested = list(assigned_gpu_ids(args, arms) or ())
        if probe_arms:
            requested.extend(args.action_probe_gpus)
        if requested and max(requested) >= have:
            raise SystemExit(
                f"comparison needs physical GPU ids through {max(requested)} "
                f"but only {have} visible GPU(s) exist"
            )
    if args.require_distinct_learner_gpu_uuids:
        freeze_gpu_uuid_inventory(args, arms)
    persist_reproducibility_metadata(args.report_dir)
    ensure_syncer()
    if args.work_dir.exists():
        shutil.rmtree(args.work_dir)
    if args.prebound_development_eval is not None:
        if args.confirmation_audit_rows:
            raise SystemExit(
                "prebound development execution forbids a second evaluation split"
            )
        train = Path(args.data)
        evalf = args.prebound_development_eval
        if not train.is_file() or not evalf.is_file():
            raise SystemExit("prebound train/development files must exist")
        n_train = sum(1 for line in train.open() if line.strip())
        n_eval = sum(1 for line in evalf.open() if line.strip())
        if args.max_rows is not None and n_train != args.max_rows:
            raise SystemExit("prebound training row count differs from --max-rows")
        if n_eval != args.eval_rows:
            raise SystemExit("prebound development row count differs from --eval-rows")
    else:
        train, evalf, n_train = split_data(
            args.data,
            args.work_dir,
            args.eval_rows,
            args.max_rows,
            args.shuffle_rows_seed,
            args.eval_split_seed,
            args.confirmation_audit_rows,
        )
    eval_provenance = materialize_eval_provenance(
        args.model,
        evalf,
        args.seq_len,
        args.report_dir / "eval-provenance",
        split_provenance=(
            None
            if args.prebound_development_eval is not None
            else args.work_dir / "split_provenance.json"
        ),
    )
    write_acquisition_state(
        args.report_dir,
        "frozen_inputs_ready",
        eval_hash=eval_provenance["eval_file_sha256"],
    )
    if probe_arms:
        from yeto.action_probe import load_anchor_manifest

        anchor_manifest = load_anchor_manifest(args.action_probe_anchor_manifest)
        validate_materialized_anchor_disjointness(
            anchor_hashes=set(anchor_manifest.canonical_hashes),
            data_files={"train": train, "eval": evalf},
            summary_path=args.report_dir.parent / "anchor_overlap_check.json",
            manifest_sha256=anchor_manifest.manifest_sha256,
            anchor_data_sha256=anchor_manifest.data_sha256,
        )
    print(f"[compare] {n_train} train rows, {args.eval_rows} eval rows")

    records = []
    if not args.skip_untrained_eval:
        base = eval_in_subprocess(
            args,
            None,
            evalf,
            args.report_dir / "per-example-loss" / "base.jsonl",
        )
        records.append(
            {"arm": "base (untrained)", "m": 0, "wall_s": 0.0, "eval_loss": base}
        )
        print(f"[compare] base eval loss/token: {base:.4f}", flush=True)

    bl = None
    if args.baseline_loss is not None:
        bl = args.baseline_loss
        records.append(
            {"arm": "baseline (sync, injected)", "m": 1, "wall_s": 0.0, "eval_loss": bl}
        )
        print(f"[compare] baseline eval loss/token: {bl:.4f} (injected)", flush=True)
    elif not args.skip_baseline:
        adapters, wall = run_baseline(args, args.work_dir)
        bl = eval_in_subprocess(
            args,
            adapters,
            evalf,
            args.report_dir / "per-example-loss" / "baseline.jsonl",
        )
        records.append(
            {
                "arm": "baseline (sync)",
                "m": 1,
                "wall_s": round(wall, 1),
                "eval_loss": bl,
            }
        )
        print(f"[compare] baseline eval loss/token: {bl:.4f} ({wall:.0f}s)", flush=True)

    for arm in arms:
        write_acquisition_state(args.report_dir, "training_started", arm=arm.name)
        adapters, wall = run_diloco(args, arm, args.work_dir)
        write_acquisition_state(args.report_dir, "endpoint_started", arm=arm.name)
        loss = eval_in_subprocess(
            args,
            adapters,
            evalf,
            args.report_dir / "per-example-loss" / f"{arm.name}.jsonl",
        )
        endpoint_status = "DIVERGED" if not math.isfinite(loss) else "COMPLETED"
        if endpoint_status == "DIVERGED":
            (args.report_dir / "scientific-divergence.json").write_text(
                json.dumps(
                    {
                        "schema": "yeto_scientific_divergence_v1",
                        "arm": arm.name,
                        "reason": "nonfinite_endpoint_nll",
                        "loss_repr": repr(loss),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
        write_acquisition_state(
            args.report_dir,
            "endpoint_recorded",
            arm=arm.name,
            scientific_status=endpoint_status,
        )
        records.append(
            {
                "arm": arm.name,
                "m": arm.m,
                "wall_s": round(wall, 1),
                "eval_loss": loss,
                "eval_rows": eval_provenance["eval_row_count"],
                "eval_tokens": eval_provenance["eval_supervised_token_count"],
                "eval_example_ids_hash": eval_provenance["eval_example_ids_hash"],
                "eval_token_ids_hash": eval_provenance["eval_token_ids_hash"],
            }
        )
        print(
            f"[compare] {arm.name} eval loss/token: {loss:.4f} ({wall:.0f}s)",
            flush=True,
        )

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
        delta = (
            "—"
            if r["arm"].startswith(("base", "baseline")) or bl in (None, 0)
            else (f"{100 * (r['eval_loss'] - bl) / bl:+.2f}%")
        )
        md.append(
            f"| {r['arm']} | {r['m'] or '—'} | {r['wall_s']:.0f} "
            f"| {r['eval_loss']:.4f} | {delta} |"
        )
    (args.report_dir / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
