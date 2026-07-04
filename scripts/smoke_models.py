#!/usr/bin/env python3
"""Comprehensive model smoke: launch a short auto-planned run for every
supported model alias and report pass/fail.

For each alias in yeto/models.py the script runs

    yeto launch --model <alias> --data <data> --budget <B> --confirm ...

with NO --gpu (the shape planner picks the fleet under the budget) and the
auto knobs left at their defaults (--micro-batch-size auto probes VRAM,
--lora-targets auto picks attention-only for MoE, gradient checkpointing
auto). Each run does a handful of outer steps on a few hundred rows, then
the cluster is torn down and the result recorded.

Models are tiered by their bf16 footprint so a smoke sweep can be sized to
a budget: --tier small is cents, --tier xl is money. Runs are sequential —
smokes share spot quota and Hub bandwidth, and a failure should be
attributable to one model, not to fleet contention.

    # see what would run, and the launch line per model, without launching
    python scripts/smoke_models.py --tier small --dry-run

    # smoke every small model for real (sequential, self-cleaning)
    python scripts/smoke_models.py --tier small --budget 15

    # specific models only
    python scripts/smoke_models.py --only lfm25-230m,qwen35-4b,llama32-1b

Results land in --report-dir as results.jsonl (one record per model) and
report.md (a table). Every cluster is torn down when its run finishes,
fails, or times out (--keep disables).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from yeto import runs  # noqa: E402
from yeto.models import MODEL_ALIASES, MODEL_WEIGHT_GB  # noqa: E402

# Tier cutoffs on the bf16 weight footprint (GB). Aliases whose size is
# resolved from the Hub at plan time (no static entry) land in xl: if we
# cannot bound the cost statically, it does not belong in a cheap sweep.
TIERS = {"small": 10, "medium": 80, "large": 300, "xl": float("inf")}

TERMINAL_STATES = (runs.SUCCEEDED, runs.FAILED, runs.DOWN)


@dataclass
class Result:
    alias: str
    hf_id: str
    weight_gb: float | None
    tier: str
    state: str  # SUCCEEDED | FAILED | TIMEOUT | DRY_RUN
    seconds: float
    note: str = ""


def tier_of(alias: str) -> str:
    gb = MODEL_WEIGHT_GB.get(alias)
    if gb is None:
        return "xl"
    for name, cutoff in TIERS.items():
        if gb <= cutoff:
            return name
    return "xl"


def select_models(args) -> list[str]:
    """Aliases to smoke, deterministically ordered smallest-first so cheap
    failures surface before expensive launches."""
    if args.only:
        picked = [a.strip() for a in args.only.split(",") if a.strip()]
        unknown = [a for a in picked if a not in MODEL_ALIASES]
        if unknown:
            raise SystemExit(f"unknown aliases: {unknown} (see yeto/models.py)")
    else:
        allowed = list(TIERS)[: list(TIERS).index(args.tier) + 1]
        picked = [a for a in MODEL_ALIASES if tier_of(a) in allowed]
    skip = {a.strip() for a in (args.skip or "").split(",") if a.strip()}
    picked = [a for a in picked if a not in skip]
    return sorted(picked, key=lambda a: (MODEL_WEIGHT_GB.get(a) or float("inf"), a))


def run_name(alias: str) -> str:
    """Cluster-prefix-safe run name (lowercase alnum + dashes)."""
    return "smk-" + "".join(c if c.isalnum() else "-" for c in alias.lower())


def launch_command(alias: str, args) -> list[str]:
    """The yeto launch line for one model's smoke. No --gpu: the shape
    planner sizes the fleet under --budget; --confirm skips the prompt."""
    return [
        sys.executable, "-m", "yeto.cli", "launch",
        "--model", alias,
        "--data", args.data,
        "--budget", str(args.budget),
        "--confirm",
        "--controller", "local",
        "--cluster-prefix", run_name(alias),
        "--total-steps", str(args.total_steps),
        "--fragments", str(args.fragments),
        "--quorum", "1",
        "--seq-len", str(args.seq_len),
        "--max-rows", str(args.max_rows),
        "--grad-accum", "1",
        "--tokenize", "preload",
    ]


def smoke_one(alias: str, args) -> Result:
    name = run_name(alias)
    cmd = launch_command(alias, args)
    t0 = time.monotonic()
    proc = subprocess.Popen(
        cmd, cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    note = ""
    deadline = t0 + args.timeout_min * 60
    state = "TIMEOUT"
    try:
        # The launch CLI streams until the run reaches a terminal state; the
        # registry is the authority either way.
        while time.monotonic() < deadline:
            meta = runs.load_run(name)
            if meta and meta.get("state") in TERMINAL_STATES:
                state = meta["state"]
                break
            if proc.poll() is not None and proc.returncode != 0:
                # Fast failures (planner rejection, bad flags) exit before
                # the registry reaches a terminal state.
                meta = runs.load_run(name)
                state = (meta or {}).get("state") or runs.FAILED
                if state not in TERMINAL_STATES:
                    state = runs.FAILED
                break
            time.sleep(10)
        else:
            note = f"no terminal state within {args.timeout_min} min"
    finally:
        if proc.poll() is None:
            proc.terminate()
        if not args.keep:
            subprocess.run(
                [sys.executable, "-m", "yeto.cli", "down", name],
                cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
    if state != runs.SUCCEEDED and not note:
        note = tail_of_log(name)
    return Result(
        alias=alias,
        hf_id=MODEL_ALIASES[alias],
        weight_gb=MODEL_WEIGHT_GB.get(alias),
        tier=tier_of(alias),
        state=state,
        seconds=round(time.monotonic() - t0, 1),
        note=note,
    )


def tail_of_log(name: str, lines: int = 3) -> str:
    try:
        text = runs.log_path(name).read_text().strip().splitlines()
        return " | ".join(text[-lines:])[:400]
    except OSError:
        return ""


def write_report(results: list[Result], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")
    ok = sum(r.state == runs.SUCCEEDED for r in results)
    md = [
        f"# Model smoke report — {ok}/{len(results)} succeeded",
        "",
        "| alias | HF id | GB | tier | state | min | note |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        gb = "(Hub)" if r.weight_gb is None else f"{r.weight_gb:g}"
        md.append(
            f"| `{r.alias}` | `{r.hf_id}` | {gb} | {r.tier} | {r.state} "
            f"| {r.seconds / 60:.1f} | {r.note} |"
        )
    (report_dir / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--tier", choices=list(TIERS), default="small",
                   help="include models up to this tier (cumulative; default small)")
    p.add_argument("--only", default=None, help="comma-separated aliases (overrides --tier)")
    p.add_argument("--skip", default=None, help="comma-separated aliases to exclude")
    p.add_argument("--data", default="armand0e/claude-fable-5-claude-code",
                   help="chat dataset for the smokes (HF id or local path)")
    p.add_argument("--budget", type=float, default=15.0,
                   help="$/hr ceiling handed to the shape planner, per model")
    p.add_argument("--total-steps", type=int, default=8, help="outer steps per smoke")
    p.add_argument("--fragments", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--max-rows", type=int, default=512)
    p.add_argument("--timeout-min", type=int, default=90, help="per-model wall clock cap")
    p.add_argument("--keep", action="store_true", help="skip teardown after each model")
    p.add_argument("--report-dir", type=Path, default=REPO_ROOT / "smoke-report")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and per-model launch lines; launch nothing")
    args = p.parse_args()

    models = select_models(args)
    if not models:
        raise SystemExit("no models selected")
    print(f"[smoke] {len(models)} model(s), sequential, budget ${args.budget}/hr each, "
          f"timeout {args.timeout_min} min each")
    results: list[Result] = []
    for alias in models:
        cmd = launch_command(alias, args)
        if args.dry_run:
            print(f"  {alias:<22} ({tier_of(alias)}, {MODEL_WEIGHT_GB.get(alias) or '(Hub)'} GB)")
            print(f"    {' '.join(cmd)}")
            results.append(Result(alias, MODEL_ALIASES[alias], MODEL_WEIGHT_GB.get(alias),
                                  tier_of(alias), "DRY_RUN", 0.0))
            continue
        print(f"[smoke] {alias} ...", flush=True)
        r = smoke_one(alias, args)
        print(f"[smoke] {alias}: {r.state} in {r.seconds / 60:.1f} min {r.note}")
        results.append(r)
    if not args.dry_run:
        write_report(results, args.report_dir)
    failed = [r for r in results if r.state not in (runs.SUCCEEDED, "DRY_RUN")]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
