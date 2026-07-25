#!/usr/bin/env python3
"""Run and summarize a short pseudo-gradient autocorrelation probe.

The default protocol launches one exact, loss-blind ``compare_diloco.py``
arm with rho telemetry enabled.  It is intentionally short: 20--40 global
outer rounds, divisible across four fragments.  The resulting report contains
fragment-balanced lag-1..4 rho estimates with deterministic bootstrap
confidence intervals, merged and worker pseudo-gradient norm summaries, and
cross-worker cosine summaries.

Example:

    python3 scripts/rho_probe.py \
      --scale 135m --h 256 --m 4 --eta 0.175 --mu 0.9 \
      --data /data/train.jsonl --eval-data /data/development.jsonl \
      --output /results/rho-probe-h256.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
COMPARE_SCRIPT = REPO_ROOT / "scripts" / "compare_diloco.py"
TELEMETRY_SCHEMA = "yeto_rho_telemetry_v1"
REPORT_SCHEMA = "yeto_rho_probe_report_v1"
FRAGMENT_COUNT = 4
PROJECTION_DIMENSION = 4096
PROJECTION_SEED = "0x5945544f52484f31"
DEFAULT_BOOTSTRAP_SEED = 20260724
SCALE_MODELS = {
    "135m": "HuggingFaceTB/SmolLM2-135M",
    "360m": "HuggingFaceTB/SmolLM2-360M",
    "1.7b": "HuggingFaceTB/SmolLM2-1.7B",
}
SUPPORTED_LEARNER_COUNTS = (1, 2, 4, 8, 16)


class ProbeError(RuntimeError):
    """The short-run or its telemetry violated the probe contract."""


Runner = Callable[[Sequence[str], Path], None]


def _default_runner(command: Sequence[str], cwd: Path) -> None:
    subprocess.run(list(command), cwd=cwd, check=True)


def _parse_scale(value: str) -> str:
    normalized = value.strip().lower().replace(" ", "")
    aliases = {
        "135m": "135m",
        "360m": "360m",
        "1.7b": "1.7b",
        "1p7b": "1.7b",
        "1700m": "1.7b",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            "scale must be one of 135m, 360m, or 1.7b"
        ) from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", required=True, type=_parse_scale)
    parser.add_argument(
        "--model",
        default=None,
        help="optional model path/ID override; scale remains report metadata",
    )
    parser.add_argument("--h", required=True, type=int, help="fixed local horizon H")
    parser.add_argument(
        "--m",
        required=True,
        type=int,
        choices=SUPPORTED_LEARNER_COUNTS,
        help="number of learner workers",
    )
    parser.add_argument("--eta", required=True, type=float, help="outer learning rate")
    parser.add_argument(
        "--mu", required=True, type=float, help="outer Nesterov momentum"
    )
    parser.add_argument(
        "--data", required=True, help="training messages JSONL or dataset ID"
    )
    parser.add_argument(
        "--eval-data",
        default=None,
        help="optional already-frozen development JSONL; no endpoint loss is evaluated",
    )
    parser.add_argument("--output", required=True, type=Path, help="probe report JSON")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="run artifact root (default: <output-stem>-run beside --output)",
    )
    parser.add_argument(
        "--outer-rounds",
        type=int,
        default=32,
        help="global outer rounds; must be 20--40 and divisible by four",
    )
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--inner-lr", type=float, default=0.001)
    parser.add_argument("--eval-rows", type=int, default=64)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard", choices=("ddp", "fsdp"), default="ddp")
    parser.add_argument("--learner-gpus", type=int, default=0)
    parser.add_argument(
        "--gpu-slots",
        type=int,
        default=None,
        help="single-process CUDA slots (default M on cuda, otherwise zero)",
    )
    parser.add_argument("--arm-timeout-min", type=int, default=120)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing report; existing telemetry is never appended",
    )
    args = parser.parse_args(argv)

    if args.h <= 0:
        parser.error("--h must be positive")
    if not math.isfinite(args.eta) or args.eta <= 0.0:
        parser.error("--eta must be finite and positive")
    if not math.isfinite(args.mu) or not 0.0 <= args.mu < 1.0:
        parser.error("--mu must be finite and in [0, 1)")
    if not 20 <= args.outer_rounds <= 40:
        parser.error("--outer-rounds must be between 20 and 40 inclusive")
    if args.outer_rounds % FRAGMENT_COUNT:
        parser.error("--outer-rounds must be divisible by four fragments")
    if args.seed < 0 or args.bootstrap_seed < 0:
        parser.error("seeds must be non-negative")
    if args.bootstrap_replicates <= 0:
        parser.error("--bootstrap-replicates must be positive")
    if args.seq_len <= 0 or args.micro_batch_size <= 0:
        parser.error("--seq-len and --micro-batch-size must be positive")
    if not math.isfinite(args.inner_lr) or args.inner_lr <= 0.0:
        parser.error("--inner-lr must be finite and positive")
    if args.eval_rows <= 0:
        parser.error("--eval-rows must be positive")
    if args.max_rows is not None and args.max_rows <= 0:
        parser.error("--max-rows must be positive")
    if args.learner_gpus < 0:
        parser.error("--learner-gpus must be non-negative")
    if args.gpu_slots is not None and args.gpu_slots < 0:
        parser.error("--gpu-slots must be non-negative")
    if args.learner_gpus > 0 and args.gpu_slots not in (None, 0):
        parser.error("--gpu-slots cannot be used with --learner-gpus")
    if args.arm_timeout_min <= 0:
        parser.error("--arm-timeout-min must be positive")
    return args


def _run_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    output = args.output.expanduser().resolve()
    run_dir = (
        args.run_dir.expanduser().resolve()
        if args.run_dir is not None
        else output.parent / f"{output.stem}-run"
    )
    work_dir = run_dir / "work"
    telemetry = work_dir / f"m{args.m}" / "rho-telemetry.jsonl"
    return output, run_dir, telemetry


def build_compare_command(args: argparse.Namespace) -> tuple[list[str], dict[str, int]]:
    fragment_rounds = args.outer_rounds // FRAGMENT_COUNT
    learner_max_steps = fragment_rounds * args.h
    learner_world = max(1, args.learner_gpus)
    token_budget = (
        args.m
        * learner_max_steps
        * args.micro_batch_size
        * args.seq_len
        * learner_world
    )
    output, run_dir, _telemetry = _run_paths(args)
    del output
    model = args.model or SCALE_MODELS[args.scale]
    gpu_slots = args.gpu_slots
    if gpu_slots is None:
        gpu_slots = args.m if args.device == "cuda" and args.learner_gpus == 0 else 0
    command = [
        sys.executable,
        str(COMPARE_SCRIPT),
        "--model",
        model,
        "--data",
        args.data,
        "--settings",
        f"m{args.m}",
        "--tuning",
        "full",
        "--skip-baseline",
        "--skip-untrained-eval",
        "--train-only-sealed-checkpoint",
        "--token-budget",
        str(token_budget),
        "--seq-len",
        str(args.seq_len),
        "--micro-batch-size",
        str(args.micro_batch_size),
        "--inner-lr",
        str(args.inner_lr),
        "--eval-rows",
        str(args.eval_rows),
        "--training-seed",
        str(args.seed),
        "--device",
        args.device,
        "--shard",
        args.shard,
        "--learner-gpus",
        str(args.learner_gpus),
        "--delta-correction",
        "none",
        "--matrix-merge",
        "rda",
        "--outer-optimizer",
        "nesterov",
        "--outer-momentum",
        format(args.mu, ".17g"),
        "--outer-lr",
        format(args.eta, ".17g"),
        "--fixed-window-microsteps",
        str(args.h),
        "--fixed-window-tokens",
        str(args.h * args.micro_batch_size * args.seq_len * learner_world),
        "--pad-to-fixed-window-tokens",
        "--freeze-delta-before-delay",
        "--learner-push-delay-ms",
        ",".join("0" for _ in range(args.m)),
        "--learner-delay-jitter-ms",
        "0",
        "--syncer-total-steps",
        str(args.outer_rounds),
        "--learner-max-steps",
        str(learner_max_steps),
        "--strict-quorum",
        "--pipeline-depth",
        str(FRAGMENT_COUNT),
        "--wan-streams",
        "0",
        "--barrier-sync",
        "--version-matched-anchor",
        "--syncer-checkpoint-every",
        str(args.outer_rounds),
        "--rho-telemetry",
        "--arm-timeout-min",
        str(args.arm_timeout_min),
        "--work-dir",
        str(run_dir / "work"),
        "--report-dir",
        str(run_dir / "compare-report"),
    ]
    if args.eval_data is not None:
        command.extend(["--prebound-development-eval", args.eval_data])
    else:
        command.extend(
            [
                "--shuffle-rows-seed",
                str(args.seed),
                "--eval-split-seed",
                str(DEFAULT_BOOTSTRAP_SEED),
            ]
        )
    if args.max_rows is not None:
        command.extend(["--max-rows", str(args.max_rows)])
    if gpu_slots:
        command.extend(["--gpu-slots", str(gpu_slots)])
    return command, {
        "fragment_rounds": fragment_rounds,
        "learner_max_steps": learner_max_steps,
        "token_budget": token_budget,
        "learner_world_size": learner_world,
        "gpu_slots": gpu_slots,
    }


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProbeError(f"duplicate JSON key {key!r} in rho telemetry")
        result[key] = value
    return result


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProbeError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ProbeError(f"{label} must be a finite number")
    return number


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProbeError(f"{label} must be an integer >= {minimum}")
    return value


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProbeError(f"{label} must be an object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ProbeError(f"{label} must be an array")
    return value


def load_telemetry(
    path: Path,
    *,
    expected_rounds: int,
    expected_workers: int,
) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise ProbeError(f"rho telemetry is missing or unsafe: {path}")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    expected_pairs = expected_workers * (expected_workers - 1) // 2
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw, object_pairs_hook=_strict_object)
            except (json.JSONDecodeError, ProbeError) as exc:
                raise ProbeError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            row = _object(row, f"{path}:{line_number}")
            if row.get("schema") != TELEMETRY_SCHEMA:
                raise ProbeError(f"{path}:{line_number}: unexpected telemetry schema")
            if row.get("event") != "outer_round_fragment":
                raise ProbeError(f"{path}:{line_number}: unexpected telemetry event")
            step = _integer(
                row.get("outer_step"), f"line {line_number} outer_step", minimum=1
            )
            fragment = _integer(row.get("fragment"), f"line {line_number} fragment")
            fragment_round = _integer(
                row.get("fragment_round"),
                f"line {line_number} fragment_round",
                minimum=1,
            )
            if fragment >= FRAGMENT_COUNT:
                raise ProbeError(
                    f"line {line_number}: fragment {fragment} is out of range"
                )
            if (step, fragment) in seen:
                raise ProbeError(f"line {line_number}: duplicate step/fragment record")
            seen.add((step, fragment))

            pseudo = _object(
                row.get("pseudo_gradient"), f"line {line_number} pseudo_gradient"
            )
            norm = _finite(pseudo.get("l2_norm"), f"line {line_number} l2_norm")
            projected_norm = _finite(
                pseudo.get("projected_l2_norm"),
                f"line {line_number} projected_l2_norm",
            )
            if norm < 0.0 or projected_norm < 0.0:
                raise ProbeError(
                    f"line {line_number}: pseudo-gradient norms must be non-negative"
                )

            autocorrelation = _object(
                row.get("autocorrelation"), f"line {line_number} autocorrelation"
            )
            for lag in range(1, 5):
                value = autocorrelation.get(f"lag_{lag}")
                if value is None:
                    continue
                rho = _finite(value, f"line {line_number} lag_{lag}")
                if not -1.0 <= rho <= 1.0:
                    raise ProbeError(
                        f"line {line_number}: lag_{lag} is outside [-1, 1]"
                    )

            sketch = _object(row.get("sketch"), f"line {line_number} sketch")
            if sketch.get("method") != "count_sketch_v1":
                raise ProbeError(f"line {line_number}: unexpected sketch method")
            if sketch.get("seed") != PROJECTION_SEED:
                raise ProbeError(f"line {line_number}: unexpected sketch seed")
            if (
                _integer(
                    sketch.get("dimension_per_tensor_group"),
                    f"line {line_number} sketch dimension",
                    minimum=1,
                )
                != PROJECTION_DIMENSION
            ):
                raise ProbeError(f"line {line_number}: unexpected sketch dimension")
            _integer(
                sketch.get("tensor_group_count"),
                f"line {line_number} tensor_group_count",
                minimum=1,
            )
            if sketch.get("retained_lags") != 4:
                raise ProbeError(f"line {line_number}: retained_lags must be 4")

            cross = _object(row.get("cross_worker"), f"line {line_number} cross_worker")
            if cross.get("estimator") != "exact_cosine":
                raise ProbeError(
                    f"line {line_number}: unexpected cross-worker estimator"
                )
            if cross.get("worker_count") != expected_workers:
                raise ProbeError(
                    f"line {line_number}: cross-worker worker_count mismatch"
                )
            if cross.get("pair_count") != expected_pairs:
                raise ProbeError(
                    f"line {line_number}: cross-worker pair_count mismatch"
                )
            workers = _list(cross.get("workers"), f"line {line_number} workers")
            pairs = _list(cross.get("pairs"), f"line {line_number} pairs")
            if len(workers) != expected_workers or len(pairs) != expected_pairs:
                raise ProbeError(
                    f"line {line_number}: cross-worker array lengths mismatch"
                )
            for worker in workers:
                worker = _object(worker, f"line {line_number} worker")
                _integer(worker.get("learner_id"), f"line {line_number} learner_id")
                worker_norm = _finite(
                    worker.get("l2_norm"), f"line {line_number} worker norm"
                )
                if worker_norm < 0.0:
                    raise ProbeError(
                        f"line {line_number}: worker norm must be non-negative"
                    )
            defined_pairs = 0
            for pair in pairs:
                pair = _object(pair, f"line {line_number} pair")
                _integer(pair.get("learner_a"), f"line {line_number} learner_a")
                _integer(pair.get("learner_b"), f"line {line_number} learner_b")
                cosine = pair.get("cosine")
                if cosine is None:
                    continue
                cosine = _finite(cosine, f"line {line_number} pair cosine")
                if not -1.0 <= cosine <= 1.0:
                    raise ProbeError(f"line {line_number}: pair cosine outside [-1, 1]")
                defined_pairs += 1
            if cross.get("defined_pair_count") != defined_pairs:
                raise ProbeError(f"line {line_number}: defined_pair_count mismatch")
            mean_cosine = cross.get("mean_cosine")
            if mean_cosine is not None:
                mean_cosine = _finite(
                    mean_cosine, f"line {line_number} mean cross-worker cosine"
                )
                if not -1.0 <= mean_cosine <= 1.0:
                    raise ProbeError(f"line {line_number}: mean cosine outside [-1, 1]")
            row["outer_step"] = step
            row["fragment"] = fragment
            row["fragment_round"] = fragment_round
            rows.append(row)

    if len(rows) != expected_rounds:
        raise ProbeError(
            f"rho telemetry has {len(rows)} records; expected exactly {expected_rounds}"
        )
    if {row["outer_step"] for row in rows} != set(range(1, expected_rounds + 1)):
        raise ProbeError("rho telemetry outer steps are not exactly 1..outer_rounds")
    by_fragment: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_fragment[row["fragment"]].append(row)
    expected_fragment_rounds = expected_rounds // FRAGMENT_COUNT
    if set(by_fragment) != set(range(FRAGMENT_COUNT)):
        raise ProbeError("rho telemetry does not cover all four fragments")
    for fragment, fragment_rows in by_fragment.items():
        observed = sorted(row["fragment_round"] for row in fragment_rows)
        if observed != list(range(1, expected_fragment_rounds + 1)):
            raise ProbeError(
                f"fragment {fragment} rounds are not exactly 1..{expected_fragment_rounds}"
            )
    return rows


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _describe(values: Sequence[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "sample_sd": None,
            "min": None,
            "p10": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    mean = sum(finite) / len(finite)
    sample_sd = (
        math.sqrt(sum((value - mean) ** 2 for value in finite) / (len(finite) - 1))
        if len(finite) > 1
        else None
    )
    return {
        "count": len(finite),
        "mean": mean,
        "sample_sd": sample_sd,
        "min": min(finite),
        "p10": _quantile(finite, 0.10),
        "p50": _quantile(finite, 0.50),
        "p90": _quantile(finite, 0.90),
        "p95": _quantile(finite, 0.95),
        "max": max(finite),
    }


def _fragment_balanced_bootstrap(
    values_by_fragment: dict[int, list[float]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    nonempty = {
        fragment: values for fragment, values in values_by_fragment.items() if values
    }
    if not nonempty:
        return {"low": None, "high": None}
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(replicates):
        fragment_means = []
        for values in nonempty.values():
            resampled = [values[rng.randrange(len(values))] for _ in values]
            fragment_means.append(sum(resampled) / len(resampled))
        samples.append(sum(fragment_means) / len(fragment_means))
    return {
        "low": _quantile(samples, 0.025),
        "high": _quantile(samples, 0.975),
    }


def summarize_telemetry(
    rows: Sequence[dict[str, Any]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    rho_lags: dict[str, Any] = {}
    for lag in range(1, 5):
        values_by_fragment: dict[int, list[float]] = defaultdict(list)
        for row in rows:
            value = row["autocorrelation"][f"lag_{lag}"]
            if value is not None:
                values_by_fragment[row["fragment"]].append(float(value))
        per_fragment = {
            str(fragment): {
                "count": len(values),
                "estimate": _mean(values),
            }
            for fragment, values in sorted(values_by_fragment.items())
        }
        fragment_estimates = [
            value["estimate"]
            for value in per_fragment.values()
            if value["estimate"] is not None
        ]
        rho_lags[str(lag)] = {
            "estimate": _mean(fragment_estimates),
            "ci_95": _fragment_balanced_bootstrap(
                values_by_fragment,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + lag,
            ),
            "defined_observations": sum(
                len(values) for values in values_by_fragment.values()
            ),
            "fragments_with_estimate": len(fragment_estimates),
            "per_fragment": per_fragment,
        }

    merged_norms = [float(row["pseudo_gradient"]["l2_norm"]) for row in rows]
    worker_norms = [
        float(worker["l2_norm"])
        for row in rows
        for worker in row["cross_worker"]["workers"]
    ]
    sync_mean_cosines = [
        float(row["cross_worker"]["mean_cosine"])
        for row in rows
        if row["cross_worker"]["mean_cosine"] is not None
    ]
    pair_cosines = [
        float(pair["cosine"])
        for row in rows
        for pair in row["cross_worker"]["pairs"]
        if pair["cosine"] is not None
    ]
    undefined_pairs = sum(
        pair["cosine"] is None for row in rows for pair in row["cross_worker"]["pairs"]
    )
    merged_by_fragment: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        merged_by_fragment[row["fragment"]].append(
            float(row["pseudo_gradient"]["l2_norm"])
        )
    return {
        "rho": {
            "estimator": "fragment_balanced_mean_of_projected_cosines",
            "bootstrap": {
                "method": "fragment_stratified_iid_percentile",
                "confidence_level": 0.95,
                "replicates": bootstrap_replicates,
                "seed": bootstrap_seed,
            },
            "lags": rho_lags,
        },
        "norms": {
            "merged_pseudo_gradient_l2": {
                **_describe(merged_norms),
                "per_fragment_mean": {
                    str(fragment): _mean(values)
                    for fragment, values in sorted(merged_by_fragment.items())
                },
            },
            "worker_pseudo_gradient_l2": _describe(worker_norms),
        },
        "cross_worker": {
            "sync_mean_cosine": _describe(sync_mean_cosines),
            "pair_cosine": _describe(pair_cosines),
            "undefined_pair_count": undefined_pairs,
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(path: Path, report: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise ProbeError(f"report already exists (use --overwrite): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_probe(
    args: argparse.Namespace, *, runner: Runner = _default_runner
) -> dict[str, Any]:
    output, run_dir, telemetry_path = _run_paths(args)
    if output.exists() and not args.overwrite:
        raise ProbeError(f"report already exists (use --overwrite): {output}")
    if telemetry_path.exists():
        raise ProbeError(
            f"refusing to append or reuse existing telemetry; choose a fresh --run-dir: {telemetry_path}"
        )
    command, derived = build_compare_command(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    runner(command, REPO_ROOT)
    wall_seconds = time.monotonic() - started
    rows = load_telemetry(
        telemetry_path,
        expected_rounds=args.outer_rounds,
        expected_workers=args.m,
    )
    summary = summarize_telemetry(
        rows,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    if any(summary["rho"]["lags"][str(lag)]["estimate"] is None for lag in range(1, 5)):
        raise ProbeError("at least one lag-1..4 rho estimate is undefined")
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "config": {
            "scale": args.scale,
            "model": args.model or SCALE_MODELS[args.scale],
            "h": args.h,
            "m": args.m,
            "eta": args.eta,
            "mu": args.mu,
            "outer_rounds": args.outer_rounds,
            "fragment_count": FRAGMENT_COUNT,
            "seed": args.seed,
            "seq_len": args.seq_len,
            "micro_batch_size": args.micro_batch_size,
            "inner_lr": args.inner_lr,
            "tuning": "full",
            "matrix_merge": "rda",
            "delta_correction": "none",
            "outer_optimizer": "nesterov",
            "barrier_sync": True,
            "version_matched_anchor": True,
            **derived,
        },
        "runner": {
            "executed": True,
            "cwd": str(REPO_ROOT),
            "command": command,
            "wall_seconds": wall_seconds,
        },
        "telemetry": {
            "schema": TELEMETRY_SCHEMA,
            "path": str(telemetry_path),
            "sha256": _sha256_file(telemetry_path),
            "record_count": len(rows),
            "outer_steps": [1, args.outer_rounds],
            "fragments": list(range(FRAGMENT_COUNT)),
            "projection": {
                "method": "count_sketch_v1",
                "dimension_per_tensor_group": PROJECTION_DIMENSION,
                "seed": PROJECTION_SEED,
                "retained_lags": 4,
            },
        },
        "work_evidence": {
            "full_registered_outer_rounds": len(rows) == args.outer_rounds,
            "telemetry_present": telemetry_path.is_file(),
            "all_lags_defined": all(
                summary["rho"]["lags"][str(lag)]["estimate"] is not None
                for lag in range(1, 5)
            ),
        },
        **summary,
    }
    _write_report(output, report, overwrite=args.overwrite)
    return report


def main(argv: Sequence[str] | None = None, *, runner: Runner = _default_runner) -> int:
    args = parse_args(argv)
    run_probe(args, runner=runner)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProbeError as exc:
        print(f"rho probe error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
