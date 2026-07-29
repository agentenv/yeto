"""Hash-bound helpers shared by the day-3 fleet execution pack."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
REMOTE_REPO = Path("/root/yeto-day3")
RESULT_ROOT = Path("/root/yeto-results-day3")
RESULT_TARGET = Path("/data/yeto-results-day3")
NODES = ("h200-n1", "h200-n2")
GPUS = tuple(range(8))
V19_SEEDS = (1201, 1213, 1217)
# Populated only by live protected foreign GPU processes at manifest-freeze time.
# The previously protected n1:0 process drained before this execution was frozen.
PROTECTED_SLOTS: tuple[tuple[str, int], ...] = ()
CAPACITY_HEADROOM_MIB = 10 * 1024
CAPACITY_EXCLUSIVE_MAX_USED_MIB = 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"{path}: expected a JSON object")
    return value


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def expected_standard_cell(cell: dict) -> dict:
    return {
        "learner_count": 4,
        "learner_steps_per_learner": int(cell["s"]),
        "outer_steps": 4 * int(cell["t"]),
        "telemetry_rows": 4 * int(cell["t"]),
        "eval_rows": 1024,
        "fixed_window_microsteps": int(cell["h"]),
        "fixed_window_tokens": int(cell["h"]) * 128,
    }


def command_for(
    cell: dict, attempt_number: int, *, gpu_override: int | None = None
) -> list[str]:
    attempt = RESULT_ROOT / cell["cell_id"] / f"attempt-{attempt_number}"
    gpu = int(cell["assignment"]["gpus"][0]) if gpu_override is None else gpu_override
    return [
        "/root/yeto-venv/bin/python",
        str(REMOTE_REPO / "scripts/compare_diloco.py"),
        "--model",
        cell["model_path"],
        "--data",
        cell["train_path"],
        "--prebound-development-eval",
        cell["eval_path"],
        "--settings",
        "m4",
        "--tuning",
        "full",
        "--skip-baseline",
        "--skip-untrained-eval",
        "--token-budget",
        str(int(cell["s"]) * 128 * 4),
        "--seq-len",
        "128",
        "--micro-batch-size",
        "1",
        "--inner-lr",
        "0.001",
        "--eval-rows",
        "1024",
        "--max-rows",
        str(cell.get("max_rows", 13758)),
        "--shuffle-rows-seed",
        str(cell["seed"]),
        "--eval-split-seed",
        "331",
        "--training-seed",
        str(cell["training_seed"]),
        "--device",
        "cuda",
        "--gpu-slots",
        "1",
        "--gpu-offset",
        str(gpu),
        "--delta-correction",
        "none",
        "--matrix-merge",
        "rda",
        "--outer-optimizer",
        cell["outer_optimizer"],
        "--outer-momentum",
        repr(float(cell["mu"])),
        "--outer-lr",
        repr(float(cell["eta"])),
        "--fixed-window-microsteps",
        str(cell["h"]),
        "--fixed-window-tokens",
        str(int(cell["h"]) * 128),
        "--pad-to-fixed-window-tokens",
        "--freeze-delta-before-delay",
        "--learner-push-delay-ms",
        "0,0,0,0",
        "--learner-delay-jitter-ms",
        "0",
        "--syncer-total-steps",
        str(4 * int(cell["t"])),
        "--learner-max-steps",
        str(cell["s"]),
        "--strict-quorum",
        "--pipeline-depth",
        "4",
        "--wan-streams",
        "0",
        "--barrier-sync",
        "--version-matched-anchor",
        "--syncer-checkpoint-every",
        str(4 * int(cell["t"])),
        "--rho-telemetry",
        "--arm-timeout-min",
        str(cell["timeout_minutes"]),
        "--work-dir",
        str(attempt / "work"),
        "--report-dir",
        str(attempt / "report"),
    ]


def bind_cell(
    cell: dict,
    source_commit: str,
    *,
    retry_gpus: tuple[int, ...] | list[int] | None = None,
) -> dict:
    cell["source_git_commit"] = source_commit
    cell["expected"] = expected_standard_cell(cell)
    cell["arm_name"] = "m4"
    initial = command_for(cell, 1)
    cell["command"] = initial
    cell["command_hash"] = canonical_sha256(initial)
    cell["attempts"] = [1, 2]
    cell["attempt2_supersedes_attempt1"] = True
    node = str(cell["assignment"]["node"])
    primary_gpu = int(cell["assignment"]["gpus"][0])
    alternatives = tuple(
        gpu for gpu in (retry_gpus or (primary_gpu,)) if int(gpu) != primary_gpu
    )
    if not alternatives:
        alternatives = (primary_gpu,)
    cell["registered_retry_commands"] = []
    for gpu_value in alternatives:
        gpu = int(gpu_value)
        retry = command_for(cell, 2, gpu_override=gpu)
        cell["registered_retry_commands"].append(
            {
                "attempt_number": 2,
                "node": node,
                "gpu": gpu,
                "command": retry,
                "command_hash": canonical_sha256(retry),
                "allowed_only_under": (
                    "registered loss-blind whole-group infrastructure retry; "
                    "relocated away from the attempt-1 GPU"
                ),
            }
        )
    return cell


def command_hash_allowed(cell: dict, attempt: int, command_hash: object) -> bool:
    if attempt == 1:
        return command_hash == cell["command_hash"]
    return command_hash in {
        record["command_hash"]
        for record in cell["registered_retry_commands"]
        if int(record["attempt_number"]) == attempt
    }


def capacity_contract(*, estimated_peak_mib: int) -> dict[str, object]:
    return {
        "class": "135M-class",
        "estimated_cell_peak_mib": int(estimated_peak_mib),
        "required_headroom_mib": CAPACITY_HEADROOM_MIB,
        "minimum_free_before_launch_mib": int(estimated_peak_mib)
        + CAPACITY_HEADROOM_MIB,
        "exclusive_max_prelaunch_used_mib": CAPACITY_EXCLUSIVE_MAX_USED_MIB,
        "consecutive_clear_samples": 2,
        "clear_sample_interval_seconds": 5,
        "recheck_before_every_cell": True,
        "protected_slots": [
            {"node": node, "gpu": gpu} for node, gpu in PROTECTED_SLOTS
        ],
    }
