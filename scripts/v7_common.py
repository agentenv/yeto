"""Shared constants and deterministic builders for the v7 27B LoRA lane."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

try:
    import analyze_v7
except ModuleNotFoundError:  # package import in tests
    from scripts import analyze_v7


REPO = Path(__file__).resolve().parent.parent
RESULT_LINK = Path("/root/yeto-results-v7")
RESULT_TARGET = Path("/data/yeto-results-v7")
MODEL = Path(
    "/data/yeto-hf-cache/hub/models--Qwen--Qwen3.6-27B/"
    "snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)
TRAIN = Path("/root/yeto-data/outer-mup-v3/scale-s2560/raw/train.jsonl")
EVAL = Path("/root/yeto-data/outer-mup-v3/scale-s2560/raw/eval.jsonl")
CONTRACT_JSON = REPO / "experiment-specs/outer-mup-v7-27b-lora-prereg.json"
CONTRACT_MD = REPO / "experiment-specs/outer-mup-v7-27b-lora-prereg.md"
ANALYZER = REPO / "scripts/analyze_v7.py"
CONTRACT_JSON_SHA256 = (
    "fe9dab44fde1aabdb5733377af9b423d7819be51032d5af3c57ddc07ced921fd"
)
CONTRACT_MD_SHA256 = "9f1d628e81e3ec521224dbdd632e62c031496db144afabb460b4d40cc5cb0f9f"
ANALYZER_SHA256 = "c835189056d407535cb866c4095b49a35361391a43b8f23a3669b40914d18f75"
REGISTRATION_COMMIT = "6dcf53a821761959ec960f10ccc36189b6a1c6d9"
NODES = ("h200-n1", "h200-n2")
MAIN_SEEDS = (701, 709, 719)
PILOT_SEED = 691
SMOKE_SEED = 683
HORIZON_CENTER_RATIO = 0.35036736670682456
FULL_OFFSETS = (-1.5, -0.5, 0.5, 1.5)
REDUCED_T20_MU0_OFFSETS = (-1.5, 0.0, 1.5)
FLEET_HOUR_THRESHOLD = 20.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def verify_frozen_contract() -> None:
    expected = {
        CONTRACT_JSON: CONTRACT_JSON_SHA256,
        CONTRACT_MD: CONTRACT_MD_SHA256,
        ANALYZER: ANALYZER_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"registered v7 artifact hash mismatch: {path}")


def training_seed(seed: int) -> int:
    return int(f"{seed}{seed}")


def command_for(cell: dict, attempt_number: int) -> list[str]:
    attempt = RESULT_LINK / cell["cell_id"] / f"attempt-{attempt_number}"
    outer_steps = 4 * int(cell["t"])
    token_budget = int(cell["s"]) * 128 * 2 * 4
    return [
        "/root/yeto-venv/bin/python",
        "/root/yeto/scripts/compare_diloco.py",
        "--model",
        str(MODEL),
        "--data",
        str(TRAIN),
        "--prebound-development-eval",
        str(EVAL),
        "--settings",
        "m2",
        "--tuning",
        "lora",
        "--shard",
        "fsdp",
        "--learner-gpus",
        "4",
        "--gpu-offset",
        "0",
        "--skip-baseline",
        "--skip-untrained-eval",
        "--token-budget",
        str(token_budget),
        "--seq-len",
        "128",
        "--micro-batch-size",
        "1",
        "--inner-lr",
        "0.0003",
        "--lora-r",
        "16",
        "--lora-alpha",
        "32",
        "--eval-rows",
        "1024",
        "--max-rows",
        "13758",
        "--training-seed",
        str(cell["training_seed"]),
        "--device",
        "cuda",
        "--delta-correction",
        "none",
        "--matrix-merge",
        "rda",
        "--outer-optimizer",
        "nesterov",
        "--outer-momentum",
        str(cell["mu"]),
        "--outer-lr",
        repr(float(cell["eta"])),
        "--fixed-window-microsteps",
        str(cell["h"]),
        "--fixed-window-tokens",
        str(cell["fixed_window_tokens"]),
        "--pad-to-fixed-window-tokens",
        "--freeze-delta-before-delay",
        "--learner-push-delay-ms",
        "0,0",
        "--learner-delay-jitter-ms",
        "0",
        "--syncer-total-steps",
        str(outer_steps),
        "--learner-max-steps",
        str(cell["s"]),
        "--strict-quorum",
        "--pipeline-depth",
        "4",
        "--wan-streams",
        "0",
        "--version-matched-anchor",
        "--syncer-checkpoint-every",
        str(outer_steps),
        "--rho-telemetry",
        "--arm-timeout-min",
        str(cell["timeout_minutes"]),
        "--work-dir",
        str(attempt / "work"),
        "--report-dir",
        str(attempt / "report"),
    ]


def bind_commands(cell: dict, source_commit: str) -> dict:
    cell["source_git_commit"] = source_commit
    initial = command_for(cell, 1)
    retry = command_for(cell, 2)
    cell["command"] = initial
    cell["command_hash"] = canonical_sha256(initial)
    cell["registered_retry_commands"] = [
        {
            "attempt_number": 2,
            "command": retry,
            "command_hash": canonical_sha256(retry),
            "allowed_only_under": (
                "loss-blind whole-group retry authority for an enumerated "
                "infrastructure reason"
            ),
        }
    ]
    return cell


def expected_for(cell: dict) -> dict:
    outer_steps = 4 * int(cell["t"])
    return {
        "learner_count": 2,
        "ranks_per_learner": 4,
        "learner_steps_per_learner": int(cell["s"]),
        "outer_steps": outer_steps,
        "telemetry_rows": outer_steps,
        "eval_rows": 1024,
        "c_steps": int(cell["h"]),
        "c_tokens": int(cell["fixed_window_tokens"]),
        "lora_trainable_tensors": 992,
        "lora_trainable_parameters": 116727808,
    }


def select_pilot_center(etas: list[float], losses: list[float]) -> dict:
    if len(etas) != 3 or len(losses) != 3:
        raise ValueError("v7 pilot requires exactly three eta/loss pairs")
    finite_indices = [index for index, loss in enumerate(losses) if math.isfinite(loss)]
    if not finite_indices:
        raise ValueError("v7 pilot has no finite endpoint")
    fit = (
        analyze_v7.fit_quadratic(etas, losses)
        if len(finite_indices) == 3
        else {
            "status": "NOT_FIT_PARTIAL_FINITE_PILOT",
            "a": None,
            "b": None,
            "c": None,
            "vertex_log2_eta": None,
            "eta_star": None,
            "accepted": False,
        }
    )
    lower = math.log2(min(etas)) - 0.5
    upper = math.log2(max(etas)) + 0.5
    vertex = fit.get("vertex_log2_eta")
    vertex_accepted = bool(
        isinstance(fit.get("a"), (int, float))
        and fit["a"] > 0.0
        and isinstance(vertex, (int, float))
        and math.isfinite(vertex)
        and lower < vertex < upper
    )
    if vertex_accepted:
        selected = 2.0**vertex
        method = "accepted_quadratic_vertex"
    else:
        selected_index = min(
            finite_indices, key=lambda index: (losses[index], etas[index])
        )
        selected = etas[selected_index]
        method = "minimum_finite_pilot_eta_fallback"
    return {
        "fit": fit,
        "vertex_acceptance_interval_log2_eta": [lower, upper],
        "vertex_accepted": vertex_accepted,
        "selected_eta_star": selected,
        "selection_method": method,
    }


def two_slot_lpt_makespan_seconds(durations: list[float]) -> tuple[float, list[float]]:
    loads = [0.0, 0.0]
    for duration in sorted(durations, reverse=True):
        slot = min(range(2), key=lambda index: (loads[index], index))
        loads[slot] += duration
    return max(loads), loads


def select_grid_variant(max_pilot_wall_seconds: float) -> dict:
    if not math.isfinite(max_pilot_wall_seconds) or max_pilot_wall_seconds <= 0.0:
        raise ValueError("pilot wall time must be finite and positive")
    short = max_pilot_wall_seconds
    long = 4.0 * short
    makespan, loads = two_slot_lpt_makespan_seconds([long] * 24 + [short] * 24)
    fleet_hours = makespan / 3600.0
    variant = "FULL_48" if fleet_hours <= FLEET_HOUR_THRESHOLD else "REDUCED_T20_MU0_45"
    return {
        "short_duration_seconds": short,
        "long_duration_seconds": long,
        "slot_load_seconds": loads,
        "projected_fleet_hours": fleet_hours,
        "threshold_fleet_hours": FLEET_HOUR_THRESHOLD,
        "variant": variant,
    }


def derive_eta_grids(pilot_center: float, variant: str) -> dict[str, list[float]]:
    if not math.isfinite(pilot_center) or pilot_center <= 0.0:
        raise ValueError("pilot center must be finite and positive")
    if variant not in ("FULL_48", "REDUCED_T20_MU0_45"):
        raise ValueError(f"unknown v7 grid variant {variant!r}")
    centers = {
        "T5_mu0": pilot_center,
        "T20_mu0": pilot_center * HORIZON_CENTER_RATIO,
    }
    centers["T5_mu0.9"] = centers["T5_mu0"] * 0.1 * analyze_v7.G4C_OBSERVED_D[5]
    centers["T20_mu0.9"] = centers["T20_mu0"] * 0.1 * analyze_v7.G4C_OBSERVED_D[20]
    offsets = {key: FULL_OFFSETS for key in centers}
    if variant == "REDUCED_T20_MU0_45":
        offsets["T20_mu0"] = REDUCED_T20_MU0_OFFSETS
    return {
        key: [center * 2.0**offset for offset in offsets[key]]
        for key, center in centers.items()
    }


def contract_record() -> dict:
    return {
        "json_path": str(CONTRACT_JSON.relative_to(REPO)),
        "json_sha256": CONTRACT_JSON_SHA256,
        "md_path": str(CONTRACT_MD.relative_to(REPO)),
        "md_sha256": CONTRACT_MD_SHA256,
        "analyzer_path": str(ANALYZER.relative_to(REPO)),
        "analyzer_sha256": ANALYZER_SHA256,
        "scientific_registration_git_commit": REGISTRATION_COMMIT,
    }
