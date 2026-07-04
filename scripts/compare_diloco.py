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
  alpha0    broadcasts overwrite (paper's recommendation at large M)
  q4        4-bit E3M0 delta pushes on the wire
  serial    --pipeline 1 (pre-pipelining scheduler behavior)
  noheloco  delta correction off (pure paper Alg. 2)
  strided   depth-interleaved fragments

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
    outer_momentum: float = 0.9


PRESETS: dict[str, Arm] = {
    "m2": Arm("m2"),
    "m4": Arm("m4", m=4),
    "alpha0": Arm("alpha0", merge_alpha=0.0),
    "q4": Arm("q4", wire_dtype="q4"),
    "serial": Arm("serial", pipeline=1),
    "noheloco": Arm("noheloco", delta_correction="none"),
    "strided": Arm("strided", fragment_pattern="strided"),
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


def gpu_env(learner_id: int, gpus_per_learner: int) -> dict[str, str] | None:
    """CUDA_VISIBLE_DEVICES block for one learner: learner i owns GPUs
    [i*g, (i+1)*g). None when GPU partitioning is off."""
    if gpus_per_learner <= 0:
        return None
    lo = learner_id * gpus_per_learner
    import os

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in range(lo, lo + gpus_per_learner))
    return env


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
        cmd += [
            "--fragments", str(arm.fragments),
            "--fragment-pattern", arm.fragment_pattern,
            "--merge-alpha", str(arm.merge_alpha),
            "--wire-dtype", arm.wire_dtype,
        ]
    return cmd


def syncer_command(arm: Arm, port: int, arm_dir: Path, total_steps: int) -> list[str]:
    # The syncer takes no fragment count: the layout arrives in HELLO.
    return [
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
    ]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def split_data(data: str, work: Path, eval_rows: int, max_rows: int | None) -> tuple[Path, Path, int]:
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

    def dump(path: Path, idxs) -> None:
        with open(path, "w") as f:
            for i in idxs:
                row = ds[i]
                f.write(json.dumps({k: row[k] for k in ("messages", "tools") if k in row}) + "\n")

    train, evalf = work / "train.jsonl", work / "eval.jsonl"
    dump(train, range(n - eval_rows))
    dump(evalf, range(n - eval_rows, n))
    return train, evalf, n - eval_rows


def eval_loss_per_token(model_id: str, adapter_dir: Path | None, eval_file: Path,
                        seq_len: int, device: str, train_on: str = "assistant") -> float:
    """Held-out masked CE per trained token — the comparison metric."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from yeto.data import build_packed_dataset
    from yeto.losses import sft_loss
    from yeto.models import resolve

    resolved = resolve(model_id)
    tok = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    # bf16 on accelerators (a 10B+ base in fp32 would not fit one GPU);
    # fp32 on cpu, where bf16 matmuls are slow and memory is plentiful.
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
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


def wait_for_free_gpus(device: str, limit_mb: int = 2000, timeout_s: int = 180) -> None:
    """Block until no compute process holds more than `limit_mb` on any GPU.

    Arms and evals run strictly one after another, but a just-exited CUDA
    process's memory is not always released by the driver the instant
    subprocess.run returns — spawning the next arm into that window OOMs
    (observed on 4xL40S: the eval child's 25 GB were still resident when
    the baseline learner loaded). Fails loudly with the offending pids.
    """
    if not device.startswith("cuda"):
        return
    deadline = time.monotonic() + timeout_s
    last = ""
    while True:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        ).stdout.strip()
        holders = []
        for line in out.splitlines():
            parts = [v.strip() for v in line.split(",")]
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
    wait_for_free_gpus(args.device)
    out = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    # The child has exited, but the driver may release its VRAM lazily;
    # don't hand the GPUs to the next arm until it is actually gone.
    wait_for_free_gpus(args.device)
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
    wait_for_free_gpus(args.device)
    t0 = time.monotonic()
    run_checked(cmd, arm_dir / "learner.log", env=gpu_env(0, args.learner_gpus))
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
        syncer_command(arm, port, arm_dir, total_steps=steps * arm.m * 4),
        stdout=open(arm_dir / "syncer.log", "w"), stderr=subprocess.STDOUT,
    )
    wait_for_free_gpus(args.device)
    t0 = time.monotonic()
    learners = []
    try:
        for i in range(arm.m):
            cmd = learner_command(args, arm_dir, learner_id=i, num_learners=arm.m,
                                  syncer=f"127.0.0.1:{port}", max_steps=steps, arm=arm)
            log = open(arm_dir / f"learner-{i}.log", "w")
            learners.append(subprocess.Popen(
                cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
                env=gpu_env(i, args.learner_gpus),
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
    p.add_argument("--max-rows", type=int, default=None, help="cap training rows")
    p.add_argument("--device", default="cpu", help="learner/eval device (cpu, mps, cuda)")
    p.add_argument("--shard", choices=["ddp", "fsdp"], default="ddp",
                   help="multi-GPU strategy inside a learner (fsdp shards the "
                   "frozen base when it exceeds one GPU)")
    p.add_argument("--learner-gpus", type=int, default=0,
                   help="GPUs per learner; learner i owns the GPU block "
                   "[i*g, (i+1)*g) and runs under torchrun when g > 1. "
                   "0 = single process on --device")
    p.add_argument("--arm-timeout-min", type=int, default=120)
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

    if args.learner_gpus > 0:
        import torch

        need = max(arm.m for arm in arms) * args.learner_gpus
        have = torch.cuda.device_count()
        if have < need:
            raise SystemExit(
                f"largest arm needs {need} GPUs ({args.learner_gpus} per learner) "
                f"but only {have} are visible"
            )
    ensure_syncer()
    if args.work_dir.exists():
        shutil.rmtree(args.work_dir)
    train, evalf, n_train = split_data(args.data, args.work_dir, args.eval_rows, args.max_rows)
    print(f"[compare] {n_train} train rows, {args.eval_rows} eval rows")

    records = []
    base = eval_in_subprocess(args, None, evalf)
    records.append({"arm": "base (untrained)", "m": 0, "wall_s": 0.0, "eval_loss": base})
    print(f"[compare] base eval loss/token: {base:.4f}", flush=True)

    adapters, wall = run_baseline(args, args.work_dir)
    bl = eval_in_subprocess(args, adapters, evalf)
    records.append({"arm": "baseline (sync)", "m": 1, "wall_s": round(wall, 1), "eval_loss": bl})
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
        delta = "—" if r["arm"].startswith(("base", "baseline")) else (
            f"{100 * (r['eval_loss'] - bl) / bl:+.2f}%"
        )
        md.append(f"| {r['arm']} | {r['m'] or '—'} | {r['wall_s']:.0f} "
                  f"| {r['eval_loss']:.4f} | {delta} |")
    (args.report_dir / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
