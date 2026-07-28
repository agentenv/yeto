"""Shared deterministic manifest/command helpers for tonight-8.5."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
RESULT_ROOT = Path("/root/yeto-results-tonight85")
RESULT_TARGET = Path("/data/yeto-results-tonight85")
NODES = ("h200-n1", "h200-n2")
GPUS = tuple(range(8))
SLOTS = tuple((node, gpu) for node in NODES for gpu in GPUS)
SCAN_SEEDS = (981, 983, 991)
V11_TRUTH_SEEDS = (971, 977)


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


def training_seed(seed: int) -> int:
    return int(f"{seed}{seed}")


def expected(cell: dict) -> dict:
    if cell.get("island"):
        return {
            "learner_count": 2,
            "ranks_per_learner": 4,
            "learner_steps_per_learner": int(cell["s"]),
            "outer_steps": 4 * int(cell["t"]),
            "telemetry_rows": 4 * int(cell["t"]),
            "eval_rows": 1024,
            "c_steps": int(cell["h"]),
            "c_tokens": int(cell["h"]) * 4 * 128,
            "lora_trainable_tensors": 992,
            "lora_trainable_parameters": 116727808,
        }
    return {
        "learner_count": int(cell.get("m", 4)),
        "learner_steps_per_learner": int(cell["s"]),
        "outer_steps": 4 * int(cell["t"]),
        "telemetry_rows": 4 * int(cell["t"]),
        "eval_rows": 1024,
        "fixed_window_microsteps": int(cell["h"]),
        "fixed_window_tokens": int(cell["h"]) * 128,
    }


def command_for(cell: dict, attempt_number: int) -> list[str]:
    attempt = RESULT_ROOT / cell["cell_id"] / f"attempt-{attempt_number}"
    island = bool(cell.get("island"))
    command = [
        "/root/yeto-venv/bin/python",
        "/root/yeto/scripts/compare_diloco.py",
        "--model",
        cell["model_path"],
        "--data",
        cell["train_path"],
        "--prebound-development-eval",
        cell["eval_path"],
        "--settings",
        "m2" if island else "m4",
        "--tuning",
        "lora" if island else "full",
    ]
    if island:
        command.extend(
            [
                "--shard",
                "fsdp",
                "--learner-gpus",
                "4",
                "--gpu-offset",
                "0",
                "--lora-r",
                "16",
                "--lora-alpha",
                "32",
            ]
        )
    command.extend(
        [
            "--skip-baseline",
            "--skip-untrained-eval",
            "--token-budget",
            str(int(cell["s"]) * 128 * int(cell.get("m", 4)) * (4 if island else 1)),
            "--seq-len",
            "128",
            "--micro-batch-size",
            "1",
            "--inner-lr",
            "0.0003" if island else "0.001",
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
        ]
    )
    if not island:
        command.extend(
            [
                "--gpu-slots",
                "1",
                "--gpu-offset",
                str(cell["assignment"]["gpus"][0]),
            ]
        )
    command.extend(
        [
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
            str(cell["h"] * (4 if island else 1) * 128),
            "--pad-to-fixed-window-tokens",
            "--freeze-delta-before-delay",
            "--learner-push-delay-ms",
            "0,0" if island else "0,0,0,0",
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
    )
    return command


def bind(cell: dict, source_commit: str) -> dict:
    cell["source_git_commit"] = source_commit
    if cell.get("island"):
        cell["fixed_window_tokens"] = int(cell["h"]) * 4 * 128
    cell["expected"] = expected(cell)
    cell["arm_name"] = "m2" if cell.get("island") else "m4"
    command = command_for(cell, 1)
    retry = command_for(cell, 2)
    cell["command"] = command
    cell["command_hash"] = canonical_sha256(command)
    cell["attempts"] = [1, 2]
    cell["attempt2_supersedes_attempt1"] = True
    cell["registered_retry_commands"] = [
        {
            "attempt_number": 2,
            "command": retry,
            "command_hash": canonical_sha256(retry),
            "allowed_only_under": "registered loss-blind whole-group infrastructure retry",
        }
    ]
    return cell
