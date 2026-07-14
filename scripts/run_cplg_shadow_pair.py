#!/usr/bin/env python3
"""Run the frozen stock-only CPLG shadow pair and publish its verdict.

The compare process executes capture-OFF then capture-ON from the same frozen
seed/configuration. This wrapper independently verifies the exact initial
state, stock trajectory, final state/export, unrounded producer interval, and
writer closure before invoking the hash-pinned full-vector shadow evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
COMPARE = REPO_ROOT / "scripts" / "compare_diloco.py"
ANALYZER = REPO_ROOT / "scripts" / "replay_cplg_shadow.py"
SYNCER_DIR = REPO_ROOT / "syncer"
PINNED_HELPER = SYNCER_DIR / "target" / "release" / "cplg_libm_oracle"
OFF_ARM = "cplg_shadow_off"
ON_ARM = "cplg_shadow_on"
BASE_ARM = "base (untrained)"
FRAGMENT_ORDER = [0, 1, 2, 3] * 8
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class RunnerError(RuntimeError):
    pass


class EvidenceError(RunnerError):
    pass


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _regular_bytes(path: Path, label: str) -> bytes:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise RunnerError(f"{label} is missing: {path}") from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise RunnerError(f"{label} must be a regular non-symlink file: {path}")
    return path.read_bytes()


def _strict_object(raw: bytes, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise RunnerError(f"{label}: duplicate JSON field {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RunnerError(f"{label}: invalid UTF-8 JSON") from error
    if type(value) is not dict:
        raise RunnerError(f"{label}: expected a JSON object")
    return value


def _closed(value: dict[str, Any], fields: set[str], label: str) -> None:
    if set(value) != fields:
        raise RunnerError(
            f"{label}: closed fields differ; missing={sorted(fields - value.keys())}, "
            f"unexpected={sorted(value.keys() - fields)}"
        )


def _checksummed_object(path: Path, label: str) -> tuple[dict[str, Any], str]:
    raw = _regular_bytes(path, label)
    digest = _sha256_bytes(raw)
    sidecar = Path(f"{path}.sha256")
    expected = f"{digest}  {path.name}\n".encode("ascii")
    if _regular_bytes(sidecar, f"{label} checksum") != expected:
        raise RunnerError(f"{label}: checksum sidecar mismatch")
    return _strict_object(raw, label), digest


def _declared_option(argv: list[str], name: str) -> str:
    values: list[str] = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == name:
            if index + 1 >= len(argv):
                raise RunnerError(f"compare argv has no value after {name}")
            values.append(argv[index + 1])
            index += 2
            continue
        prefix = name + "="
        if argument.startswith(prefix):
            values.append(argument[len(prefix) :])
        index += 1
    if len(values) != 1 or not values[0]:
        raise RunnerError(
            f"compare argv must declare {name} exactly once; found {len(values)}"
        )
    return values[0]


def _require_exact_fields(
    value: dict[str, Any], expected: dict[str, Any], label: str
) -> None:
    for field, expected_value in expected.items():
        if field not in value:
            raise RunnerError(f"{label} is missing {field!r}")
        observed = value[field]
        if type(observed) is not type(expected_value) or observed != expected_value:
            raise RunnerError(
                f"{label} {field} differs from the frozen mechanism: "
                f"expected {expected_value!r}, observed {observed!r}"
            )


def _cli_number(value: Any, label: str) -> str:
    if type(value) is int:
        return str(value)
    if type(value) is float and math.isfinite(value):
        return format(value, ".15g")
    raise RunnerError(f"{label} must be a finite exact JSON number")


def _validate_frozen_config(
    config: dict[str, Any],
    compare_argv: list[str],
) -> None:
    try:
        workload = config["workload"]
        runtime = config["runtime"]
        capture = config["capture"]
        analysis = config["analysis"]
        gates = config["gates"]
        lifecycle = config["lifecycle"]
    except KeyError as error:
        raise RunnerError(
            f"frozen configuration is missing {error.args[0]!r}"
        ) from error
    if any(
        type(value) is not dict
        for value in (workload, runtime, capture, analysis, gates, lifecycle)
    ):
        raise RunnerError("frozen scientific configuration sections must be objects")
    fixed_identities = {
        "status": "frozen_before_direction_outcome",
        "run_id": "exp2-cplg-shadow-direction-r2",
    }
    for field, expected in fixed_identities.items():
        if config.get(field) != expected:
            raise RunnerError(f"frozen configuration {field} differs from {expected!r}")
    _require_exact_fields(
        runtime,
        {
            "provider": "gcp",
            "project": "model-training-497007",
            "zone": "us-central1-c",
            "machine_type": "a2-highgpu-1g",
            "accelerator_count": 1,
            "maximum_total_accelerators": 1,
            "provisioning_model": "SPOT",
            "expected_source_image_id": "7290368630472593484",
        },
        "frozen runtime",
    )
    _require_exact_fields(
        workload,
        {
            "arms_in_order": [OFF_ARM, ON_ARM],
            "result_rows_in_order": [BASE_ARM, OFF_ARM, ON_ARM],
            "capture_enabled_by_arm": [False, True],
            "sequence_length": 128,
            "micro_batch_size": 1,
            "raw_local_training_tokens": 4352,
            "compare_token_budget": 4352,
            "expected_terminal_local_steps": 34,
            "learner_max_steps_liveness_cap": 96,
            "learners": 1,
            "fragments": 4,
            "quorum": 1,
            "fixed_window_microsteps": 4,
            "outer_commits": 32,
            "fragment_order": FRAGMENT_ORDER,
            "wire_dtype": "f32",
            "merge_alpha": 0.0,
            "matrix_merge": "rda",
            "delta_correction": "none",
            "outer_optimizer": "nesterov",
            "outer_learning_rate": 0.28,
            "outer_momentum": 0.0,
            "inner_optimizer": "adamw",
            "inner_learning_rate": 0.001,
            "adamw_betas": [0.9, 0.999],
            "adamw_epsilon": 1e-08,
            "weight_decay": 0.01,
            "warmup_steps": 10,
            "scheduler": "linear warmup for 10 optimizer steps, then constant",
            "gradient_clip_norm": 1.0,
            "gradient_checkpointing": "off",
            "loss_function": "cross_entropy",
            "train_on": "assistant",
            "tuning": "lora",
            "lora_rank": 2,
            "lora_alpha": 4,
            "lora_targets": "all-linear",
            "shard": "ddp",
            "grad_accumulation": 1,
            "tokenization": "preload",
            "shuffle_rows_seed": 271,
            "training_seed": 271271,
            "max_rows": 5000,
            "evaluation_rows": 8,
            "device": "cuda",
            "gpu_slots": 1,
            "arm_timeout_minutes": 20,
            "strict_quorum": True,
            "barrier_sync": True,
            "deterministic_commit_order": True,
            "skip_baseline": True,
        },
        "frozen workload",
    )
    _require_exact_fields(
        capture,
        {
            "capture_session_uuid": "667f5de8-6d6d-4ce0-9344-efc239583abf",
            "expected_records": 32,
            "vector_format": "canonical little-endian f32 full stock pseudo-gradient",
            "initial_identity": (
                "producer-derived canonical live layout plus exact f32 initial state"
            ),
            "interval_scope": (
                "post_global_initialization_to_durable_vector_writer_close"
            ),
            "matched_behavior_requirement": (
                "normalized event tapes, final syncer checkpoints, exported adapter "
                "trees, and evaluation losses are exact"
            ),
        },
        "frozen capture",
    )
    _require_exact_fields(
        analysis,
        {
            "authoritative_transcendentals": (
                "Rust libm 0.2.15 helper whose executable SHA-256 is reported"
            ),
            "resolved_scores": 20,
            "scores_per_fragment": 5,
            "unresolved_tail_shadows": 4,
            "bootstrap_draws": 20000,
            "bootstrap_seed": 0x43504C47,
            "bootstrap_block_length": 2,
            "bootstrap_lower_index": 1000,
            "multiplicity_adjustment": (
                "none; exactly one frozen primary candidate and statistic"
            ),
        },
        "frozen analysis",
    )
    _require_exact_fields(
        gates,
        {
            "minimum_simulated_nonstock_actions": 8,
            "minimum_mean_direction_gain": 0.001,
            "minimum_positive_fragment_means": 3,
            "minimum_one_sided_95_percent_bootstrap_lower_endpoint": 0.0,
            "maximum_matched_capture_overhead_fraction": 0.02,
            "maximum_nonfinite_values": 0,
            "maximum_missing_or_invalid_records": 0,
        },
        "frozen gates",
    )
    _require_exact_fields(
        lifecycle,
        {
            "terminal_verdicts": [
                "PASS",
                "FAIL",
                "INCONCLUSIVE",
                "UNIDENTIFIABLE",
                "INFRA_FAILURE",
            ]
        },
        "frozen lifecycle",
    )
    expected_options = {
        "--model": runtime.get("model"),
        "--data": runtime.get("data"),
        "--settings": ",".join(workload["arms_in_order"]),
        "--seq-len": str(workload.get("sequence_length")),
        "--micro-batch-size": str(workload.get("micro_batch_size")),
        "--inner-lr": _cli_number(
            workload.get("inner_learning_rate"), "inner_learning_rate"
        ),
        "--lora-r": str(workload.get("lora_rank")),
        "--lora-alpha": str(workload.get("lora_alpha")),
        "--eval-rows": str(workload.get("evaluation_rows")),
        "--max-rows": str(workload.get("max_rows")),
        "--shuffle-rows-seed": str(workload.get("shuffle_rows_seed")),
        "--training-seed": str(workload.get("training_seed")),
        "--device": workload.get("device"),
        "--gpu-slots": str(workload.get("gpu_slots")),
        "--delta-correction": workload.get("delta_correction"),
        "--matrix-merge": workload.get("matrix_merge"),
        "--outer-momentum": _cli_number(
            workload.get("outer_momentum"), "outer_momentum"
        ),
        "--outer-lr": _cli_number(
            workload.get("outer_learning_rate"), "outer_learning_rate"
        ),
        "--token-budget": str(workload.get("compare_token_budget")),
        "--syncer-total-steps": str(workload.get("outer_commits")),
        "--learner-max-steps": str(workload.get("learner_max_steps_liveness_cap")),
        "--fixed-window-microsteps": str(workload.get("fixed_window_microsteps")),
        "--stock-shadow-capture-session": capture.get("capture_session_uuid"),
        "--arm-timeout-min": str(workload.get("arm_timeout_minutes")),
        "--shard": workload.get("shard"),
        "--tuning": workload.get("tuning"),
    }
    for flag, expected in expected_options.items():
        if type(expected) is not str or not expected or expected == "None":
            raise RunnerError(f"frozen configuration has no canonical value for {flag}")
        observed = _declared_option(compare_argv, flag)
        if observed != expected:
            raise RunnerError(
                f"compare {flag} differs from frozen configuration: "
                f"expected {expected!r}, observed {observed!r}"
            )
    for flag in (
        "--strict-quorum",
        "--barrier-sync",
        "--deterministic-commit-order",
        "--skip-baseline",
    ):
        if compare_argv.count(flag) != 1:
            raise RunnerError(f"compare argv must contain exactly one frozen {flag}")
    for forbidden in ("--baseline-loss", "--outer-optimizer"):
        if any(
            token == forbidden or token.startswith(f"{forbidden}=")
            for token in compare_argv
        ):
            raise RunnerError(f"compare argv must not override frozen {forbidden}")


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    raw = _regular_bytes(path, label)
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise RunnerError(f"{label}: not UTF-8") from error
    rows = [
        _strict_object(line.encode(), f"{label}:{index}")
        for index, line in enumerate(lines, 1)
        if line.strip()
    ]
    if not rows:
        raise RunnerError(f"{label}: empty JSONL")
    return rows


def _initial_receipt(path: Path, capture_enabled: bool) -> dict[str, Any]:
    value, _digest = _checksummed_object(path, "initial-state receipt")
    _closed(
        value,
        {
            "schema",
            "capture_enabled",
            "layout_sha256",
            "initial_state_sha256",
        },
        str(path),
    )
    if value["schema"] != "cplg_stock_shadow_initial_state_v1":
        raise RunnerError(f"{path}: wrong initial-state schema")
    if value["capture_enabled"] is not capture_enabled:
        raise RunnerError(f"{path}: capture identity mismatch")
    for field in ("layout_sha256", "initial_state_sha256"):
        if type(value[field]) is not str or SHA256_RE.fullmatch(value[field]) is None:
            raise RunnerError(f"{path}: noncanonical {field}")
    return value


def _completion_receipt(path: Path, capture_enabled: bool) -> dict[str, Any]:
    value, _digest = _checksummed_object(path, "completion receipt")
    _closed(
        value,
        {
            "schema",
            "capture_enabled",
            "layout_sha256",
            "initial_state_sha256",
            "interval_scope",
            "interval_start_monotonic_ns",
            "interval_end_monotonic_ns",
            "commits",
        },
        str(path),
    )
    if value["schema"] != "cplg_stock_shadow_completion_v1":
        raise RunnerError(f"{path}: wrong completion schema")
    if value["capture_enabled"] is not capture_enabled:
        raise RunnerError(f"{path}: capture identity mismatch")
    if value["interval_scope"] != (
        "post_global_initialization_to_durable_vector_writer_close"
    ):
        raise RunnerError(f"{path}: interval scope drifted")
    for field in (
        "interval_start_monotonic_ns",
        "interval_end_monotonic_ns",
        "commits",
    ):
        if type(value[field]) is not int:
            raise RunnerError(f"{path}: {field} must be an exact integer")
    if (
        value["interval_start_monotonic_ns"] != 0
        or value["interval_end_monotonic_ns"] <= 0
        or value["commits"] != 32
    ):
        raise RunnerError(f"{path}: invalid frozen interval or commit count")
    return value


def _terminal_local_steps(log_path: Path) -> int:
    text = _regular_bytes(log_path, "learner log").decode("utf-8")
    matches = re.findall(r"inner loop done at local_step=(\d+) global_step=\d+", text)
    if len(matches) != 1:
        raise RunnerError(f"{log_path}: expected one terminal local-step record")
    return int(matches[0])


def _tree_sha256(root: Path) -> str:
    if not root.is_dir() or root.is_symlink():
        raise RunnerError(f"export tree is missing or unsafe: {root}")
    digest = hashlib.sha256()
    entries = sorted(root.rglob("*"))
    for path in entries:
        if path.is_symlink():
            raise RunnerError(f"export tree contains a symlink: {path}")
    files = [path for path in entries if path.is_file()]
    if not files:
        raise RunnerError(f"export tree is empty: {root}")
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        raw = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def _schedule(arm_dir: Path) -> tuple[list[int], int, str, list[dict[str, Any]]]:
    rows = _read_jsonl(arm_dir / "tape.jsonl", "stock event tape")
    if len(rows) != 32:
        raise RunnerError(f"{arm_dir}: expected 32 event rows, observed {len(rows)}")
    fragments = []
    for commit_seq, row in enumerate(rows, 1):
        if type(row.get("commit_seq")) is not int or row["commit_seq"] != commit_seq:
            raise RunnerError(f"{arm_dir}: noncontiguous commit sequence")
        fragment = row.get("fragment")
        if type(fragment) is not int:
            raise RunnerError(f"{arm_dir}: noninteger fragment")
        fragments.append(fragment)
        responders = row.get("responders")
        if (
            type(responders) is not list
            or len(responders) != 1
            or type(responders[0]) is not dict
            or responders[0].get("id") != 0
        ):
            raise RunnerError(f"{arm_dir}: expected exactly responder zero")
    if fragments != FRAGMENT_ORDER:
        raise RunnerError(f"{arm_dir}: fragment order differs from frozen schedule")
    local_steps = _terminal_local_steps(arm_dir / "learner-0.log")
    if local_steps != 34:
        raise RunnerError(f"{arm_dir}: expected 34 local steps, observed {local_steps}")
    contract = {
        "schema": "cplg_shadow_schedule_v1",
        "commits": 32,
        "local_steps": local_steps,
        "fixed_h": 4,
        "fragment_order": fragments,
    }
    raw = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return fragments, local_steps, _sha256_bytes(raw), rows


def _normalize_event_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for source in rows:
        row = dict(source)
        for field in ("commit_elapsed_ns", "ms"):
            row.pop(field, None)
        normalized.append(row)
    return normalized


def _evaluation_losses(results: Path) -> tuple[float, float]:
    rows = _read_jsonl(results, "comparison results")
    observed_arms = [row.get("arm") for row in rows]
    if observed_arms != [BASE_ARM, OFF_ARM, ON_ARM]:
        raise RunnerError(
            "comparison results must contain exactly the untrained base followed by "
            "the frozen OFF/ON arms"
        )
    base_loss = rows[0].get("eval_loss")
    if type(base_loss) not in (int, float) or not math.isfinite(base_loss):
        raise RunnerError(f"{BASE_ARM}: evaluation loss is not finite")
    by_arm = {row.get("arm"): row for row in rows}
    losses = []
    for arm in (OFF_ARM, ON_ARM):
        value = by_arm[arm].get("eval_loss")
        if type(value) not in (int, float) or not math.isfinite(value):
            raise RunnerError(f"{arm}: evaluation loss is not finite")
        losses.append(float(value))
    if losses[0].hex() != losses[1].hex():
        raise RunnerError(
            f"capture changed final evaluation: OFF={losses[0]!r}, ON={losses[1]!r}"
        )
    return losses[0], losses[1]


def _writer_from_manifest(
    path: Path,
    *,
    expected_initial_state_sha256: str,
    expected_layout_sha256: str,
    expected_run_config_sha256: str,
) -> tuple[dict[str, Any], str]:
    value, _digest = _checksummed_object(path, "stock-vector manifest")
    for field, expected in (
        ("initial_state_sha256", expected_initial_state_sha256),
        ("layout_sha256", expected_layout_sha256),
        ("run_config_sha256", expected_run_config_sha256),
        ("status", "COMPLETE"),
        ("records", 32),
    ):
        if value.get(field) != expected:
            raise RunnerError(f"{path}: {field} mismatch")
    tape_path = path.with_name("stock_tape.jsonl")
    tape_sha256 = _sha256_bytes(_regular_bytes(tape_path, "stock-vector tape"))
    if value.get("stock_tape_sha256") != tape_sha256:
        raise RunnerError(f"{path}: stock tape SHA-256 mismatch")
    writer = value.get("writer")
    if type(writer) is not dict:
        raise RunnerError(f"{path}: missing writer accounting")
    return writer, tape_sha256


def _analysis_outcome(
    path: Path, *, expected_helper_sha256: str | None = None
) -> tuple[str, str]:
    value, digest = _checksummed_object(path, "CPLG shadow analysis")
    if value.get("schema") != "cplg_full_vector_stock_shadow_v1":
        raise EvidenceError(
            "CPLG shadow analysis schema differs from the frozen schema"
        )
    decision = value.get("decision")
    if decision not in ("PASS", "FAIL"):
        raise EvidenceError("CPLG shadow analysis has no closed PASS/FAIL decision")
    if expected_helper_sha256 is not None:
        contract = value.get("reference_contract")
        if (
            type(contract) is not dict
            or contract.get("rust_libm_helper_sha256") != expected_helper_sha256
        ):
            raise EvidenceError(
                "CPLG shadow analysis helper identity differs from preflight"
            )
    return decision, digest


def _path_entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _install_fresh_atomic(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
    except FileExistsError as error:
        raise RunnerError(
            f"refusing to replace stale evidence: {destination}"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_checksummed_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if _path_entry_exists(path) or _path_entry_exists(Path(f"{path}.sha256")):
        raise RunnerError(f"output path is not fresh: {path}")
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    _install_fresh_atomic(temporary, path)
    digest = _sha256_bytes(raw)
    sidecar = Path(f"{path}.sha256")
    sidecar_tmp = sidecar.with_name(f".{sidecar.name}.tmp.{os.getpid()}")
    with sidecar_tmp.open("xb") as handle:
        handle.write(f"{digest}  {path.name}\n".encode("ascii"))
        handle.flush()
        os.fsync(handle.fileno())
    _install_fresh_atomic(sidecar_tmp, sidecar)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _require_fresh_checksummed_output(path: Path, label: str) -> None:
    if _path_entry_exists(path) or _path_entry_exists(Path(f"{path}.sha256")):
        raise RunnerError(f"{label} output path is not fresh: {path}")


def _build_and_pin_helper(
    helper: Path,
    *,
    expected_source_commit: str,
    run_config_sha256: str,
    preflight_output: Path,
) -> str:
    if helper.absolute() != PINNED_HELPER.absolute():
        raise RunnerError(
            f"Rust-libm helper must be the frozen release target {PINNED_HELPER}"
        )
    subprocess.run(
        [
            "cargo",
            "build",
            "--locked",
            "--release",
            "--bin",
            "yeto-syncer",
            "--bin",
            "cplg_libm_oracle",
        ],
        cwd=SYNCER_DIR,
        check=True,
    )
    helper_sha256 = _sha256_bytes(_regular_bytes(helper, "Rust-libm helper"))
    cargo_lock_sha256 = _sha256_bytes(
        _regular_bytes(SYNCER_DIR / "Cargo.lock", "syncer Cargo.lock")
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if COMMIT_RE.fullmatch(commit) is None:
        raise RunnerError("source checkout does not have a full clean commit identity")
    if commit != expected_source_commit:
        raise RunnerError("source checkout differs from the immutable expected commit")
    tracked_status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if tracked_status:
        raise RunnerError(
            "source checkout has tracked modifications after helper build"
        )
    _atomic_checksummed_json(
        preflight_output,
        {
            "schema": "cplg_shadow_preflight_v1",
            "source_commit": commit,
            "run_config_sha256": run_config_sha256,
            "cargo_lock_sha256": cargo_lock_sha256,
            "rust_libm_helper": helper.relative_to(REPO_ROOT).as_posix(),
            "rust_libm_helper_sha256": helper_sha256,
        },
    )
    return helper_sha256


def _verify_preflight_identity(
    preflight_output: Path,
    *,
    helper: Path,
    expected_helper_sha256: str,
    expected_source_commit: str,
    expected_run_config_sha256: str,
) -> None:
    receipt, _digest = _checksummed_object(preflight_output, "CPLG preflight")
    _closed(
        receipt,
        {
            "schema",
            "source_commit",
            "run_config_sha256",
            "cargo_lock_sha256",
            "rust_libm_helper",
            "rust_libm_helper_sha256",
        },
        "CPLG preflight",
    )
    if receipt["schema"] != "cplg_shadow_preflight_v1":
        raise RunnerError("CPLG preflight schema differs")
    expected_values = {
        "source_commit": expected_source_commit,
        "run_config_sha256": expected_run_config_sha256,
        "rust_libm_helper": helper.relative_to(REPO_ROOT).as_posix(),
        "rust_libm_helper_sha256": expected_helper_sha256,
        "cargo_lock_sha256": _sha256_bytes(
            _regular_bytes(SYNCER_DIR / "Cargo.lock", "syncer Cargo.lock")
        ),
    }
    for field, expected in expected_values.items():
        if receipt.get(field) != expected:
            raise RunnerError(f"CPLG preflight identity differs at {field}")
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current_commit != expected_source_commit:
        raise RunnerError("source checkout changed after frozen preflight")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RunnerError("source checkout became dirty after frozen preflight")
    if _sha256_bytes(_regular_bytes(helper, "Rust-libm helper")) != (
        expected_helper_sha256
    ):
        raise RunnerError("Rust-libm helper changed after frozen preflight")


def build_overhead_evidence(
    *,
    work_dir: Path,
    report_dir: Path,
    input_manifest: Path,
    run_config_sha256: str,
    output: Path,
) -> Path:
    off_dir = work_dir / OFF_ARM
    on_dir = work_dir / ON_ARM
    off_initial = _initial_receipt(off_dir / "stock_shadow_initial_state.json", False)
    on_initial = _initial_receipt(on_dir / "stock_shadow_initial_state.json", True)
    for field in ("layout_sha256", "initial_state_sha256"):
        if off_initial[field] != on_initial[field]:
            raise RunnerError(f"matched initial receipt differs at {field}")
    off_completion = _completion_receipt(
        off_dir / "stock_shadow_completion.json", False
    )
    on_completion = _completion_receipt(on_dir / "stock_shadow_completion.json", True)
    for receipt, initial in (
        (off_completion, off_initial),
        (on_completion, on_initial),
    ):
        for field in ("layout_sha256", "initial_state_sha256"):
            if receipt[field] != initial[field]:
                raise RunnerError(f"completion receipt differs at {field}")

    off_order, off_steps, off_schedule, off_rows = _schedule(off_dir)
    on_order, on_steps, on_schedule, on_rows = _schedule(on_dir)
    if (off_order, off_steps, off_schedule) != (on_order, on_steps, on_schedule):
        raise RunnerError("matched schedule evidence differs")
    if _normalize_event_rows(off_rows) != _normalize_event_rows(on_rows):
        raise RunnerError("capture changed the normalized production event tape")
    if _sha256_bytes(_regular_bytes(off_dir / "state.ckpt", "OFF checkpoint")) != (
        _sha256_bytes(_regular_bytes(on_dir / "state.ckpt", "ON checkpoint"))
    ):
        raise RunnerError("capture changed the exact final syncer checkpoint")
    if _tree_sha256(off_dir / "export") != _tree_sha256(on_dir / "export"):
        raise RunnerError("capture changed the exact exported adapter tree")
    _evaluation_losses(report_dir / "results.jsonl")

    writer, tape_sha256 = _writer_from_manifest(
        on_dir / "stock_vectors" / "stock_tape.manifest.json",
        expected_initial_state_sha256=on_initial["initial_state_sha256"],
        expected_layout_sha256=on_initial["layout_sha256"],
        expected_run_config_sha256=run_config_sha256,
    )
    input_manifest_sha256 = _sha256_bytes(
        _regular_bytes(input_manifest, "input provenance manifest")
    )
    zero_writer = {
        "state": "disabled",
        "accepted_items": 0,
        "completed_items": 0,
        "accepted_bytes": 0,
        "completed_bytes": 0,
        "dropped_items": 0,
        "dropped_bytes": 0,
        "abandoned_items": 0,
        "abandoned_bytes": 0,
        "pending_items": 0,
        "pending_bytes": 0,
        "error": None,
    }

    def arm(
        completion: dict[str, Any],
        *,
        capture_enabled: bool,
        stock_tape_sha256: str | None,
        writer: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "capture_enabled": capture_enabled,
            "interval_start_monotonic_ns": completion["interval_start_monotonic_ns"],
            "interval_end_monotonic_ns": completion["interval_end_monotonic_ns"],
            "commits": 32,
            "local_steps": off_steps,
            "fragment_order": off_order,
            "initial_state_sha256": off_initial["initial_state_sha256"],
            "input_manifest_sha256": input_manifest_sha256,
            "schedule_sha256": off_schedule,
            "runner_exit_code": 0,
            "evaluation_finite": True,
            "stock_tape_sha256": stock_tape_sha256,
            "writer": writer,
        }

    document = {
        "schema": "cplg_shadow_overhead_v1",
        "off": arm(
            off_completion,
            capture_enabled=False,
            stock_tape_sha256=None,
            writer=zero_writer,
        ),
        "on": arm(
            on_completion,
            capture_enabled=True,
            stock_tape_sha256=tape_sha256,
            writer=writer,
        ),
    }
    _atomic_checksummed_json(output, document)
    return on_dir / "stock_vectors" / "stock_tape.jsonl"


def run_compare_then_analyze(
    *,
    compare_argv: list[str],
    output: Path,
    overhead_output: Path,
    rust_libm_helper: Path,
    input_manifest: Path,
    run_config: Path,
    preflight_output: Path,
    expected_source_commit: str,
    python: str,
) -> tuple[str, str]:
    if not compare_argv:
        raise RunnerError("missing compare argv after --")
    for path, label in (
        (output.resolve(), "analysis"),
        (overhead_output.resolve(), "overhead"),
        (preflight_output.resolve(), "preflight"),
    ):
        _require_fresh_checksummed_output(path, label)
    work_dir = Path(_declared_option(compare_argv, "--work-dir")).resolve()
    report_dir = Path(_declared_option(compare_argv, "--report-dir")).resolve()
    if _declared_option(compare_argv, "--settings") != f"{OFF_ARM},{ON_ARM}":
        raise RunnerError("compare --settings differs from the frozen shadow pair")
    run_config_sha256 = _declared_option(
        compare_argv, "--stock-shadow-run-config-sha256"
    )
    if SHA256_RE.fullmatch(run_config_sha256) is None:
        raise RunnerError("run-config identity is not canonical SHA-256")
    if COMMIT_RE.fullmatch(expected_source_commit) is None:
        raise RunnerError("expected source commit is not a full lowercase Git SHA")
    config, actual_run_config_sha256 = _checksummed_object(
        run_config, "frozen run configuration"
    )
    if run_config_sha256 != actual_run_config_sha256:
        raise RunnerError(
            "compare run-config identity differs from the frozen configuration file"
        )
    _validate_frozen_config(config, compare_argv)

    helper_sha256 = _build_and_pin_helper(
        rust_libm_helper,
        expected_source_commit=expected_source_commit,
        run_config_sha256=run_config_sha256,
        preflight_output=preflight_output.resolve(),
    )

    try:
        subprocess.run([python, str(COMPARE), *compare_argv], cwd=REPO_ROOT, check=True)
    except subprocess.CalledProcessError as error:
        raise RunnerError(
            f"comparison process failed with exit status {error.returncode}"
        ) from error
    _verify_preflight_identity(
        preflight_output.resolve(),
        helper=rust_libm_helper,
        expected_helper_sha256=helper_sha256,
        expected_source_commit=expected_source_commit,
        expected_run_config_sha256=run_config_sha256,
    )
    try:
        tape = build_overhead_evidence(
            work_dir=work_dir,
            report_dir=report_dir,
            input_manifest=input_manifest.resolve(),
            run_config_sha256=run_config_sha256,
            output=overhead_output.resolve(),
        )
    except Exception as error:
        raise EvidenceError(f"matched evidence validation failed: {error}") from error
    try:
        subprocess.run(
            [
                python,
                str(ANALYZER),
                "--stock-tape",
                str(tape),
                "--rust-libm-helper",
                str(rust_libm_helper.resolve()),
                "--enforce-shadow-gate",
                "--overhead-evidence",
                str(overhead_output.resolve()),
                "--out",
                str(output.resolve()),
            ],
            cwd=REPO_ROOT,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        raise EvidenceError(
            f"frozen analyzer failed with exit status {error.returncode}"
        ) from error
    _verify_preflight_identity(
        preflight_output.resolve(),
        helper=rust_libm_helper,
        expected_helper_sha256=helper_sha256,
        expected_source_commit=expected_source_commit,
        expected_run_config_sha256=run_config_sha256,
    )
    try:
        return _analysis_outcome(output.resolve(), expected_helper_sha256=helper_sha256)
    except RunnerError as error:
        raise EvidenceError(f"frozen analysis validation failed: {error}") from error


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overhead-output", type=Path, required=True)
    parser.add_argument("--rust-libm-helper", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--preflight-output", type=Path, required=True)
    parser.add_argument("--terminal-output", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("compare_argv", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.compare_argv[:1] == ["--"]:
        args.compare_argv = args.compare_argv[1:]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if _path_entry_exists(args.terminal_output) or _path_entry_exists(
        Path(f"{args.terminal_output}.sha256")
    ):
        print("CPLG shadow terminal output path is not fresh", file=sys.stderr)
        return 2
    decision = "INFRA_FAILURE"
    stage = "configuration_or_runtime"
    error_class: str | None = None
    error_message: str | None = None
    analysis_sha256: str | None = None
    return_code = 2
    try:
        decision, analysis_sha256 = run_compare_then_analyze(
            compare_argv=args.compare_argv,
            output=args.output,
            overhead_output=args.overhead_output,
            rust_libm_helper=args.rust_libm_helper,
            input_manifest=args.input_manifest,
            run_config=args.run_config,
            preflight_output=args.preflight_output,
            expected_source_commit=args.expected_source_commit,
            python=args.python,
        )
        stage = "completed_analysis"
        return_code = 0
    except EvidenceError as error:
        decision = "INCONCLUSIVE"
        stage = "post_acquisition_evidence_validation"
        error_class = type(error).__name__
        error_message = str(error)
        return_code = 3
        print(f"CPLG shadow evidence error: {error}", file=sys.stderr)
    except (RunnerError, subprocess.CalledProcessError, OSError) as error:
        error_class = type(error).__name__
        error_message = str(error)
        print(f"CPLG shadow infrastructure error: {error}", file=sys.stderr)
    except Exception as error:
        error_class = type(error).__name__
        error_message = str(error)
        print(f"CPLG shadow unexpected infrastructure error: {error}", file=sys.stderr)
    terminal = {
        "schema": "cplg_shadow_terminal_verdict_v1",
        "decision": decision,
        "stage": stage,
        "analysis": (str(args.output.resolve()) if analysis_sha256 else None),
        "analysis_sha256": analysis_sha256,
        "error_class": error_class,
        "error": error_message,
    }
    try:
        _atomic_checksummed_json(args.terminal_output.resolve(), terminal)
    except Exception as error:
        print(
            f"CPLG shadow terminal verdict publication failed: {error}",
            file=sys.stderr,
        )
        return 4
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
