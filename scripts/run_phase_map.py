#!/usr/bin/env python3
"""Loss-blind acquisition runner for the preregistered best-paper phase map.

The runner materializes a deterministic block randomization, freezes evaluation
row/token identity before training, launches one exact compare command per
cell, validates full-quorum work from the event tape, and writes an immutable
attempt-level acquisition manifest.  It never selects learning rates or opens
later seeds; adaptive rounds require a new invocation and study id.
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
import subprocess
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPARE = REPO_ROOT / "scripts" / "compare_diloco.py"
AUTHORITATIVE_PREREG_PATH = Path(
    "experiment-specs/best-paper-phase-map-p0-p1-prereg.json"
)
AUTHORITATIVE_PREREG_COMMIT = "16d27bc60deb6d8910bf0111c7fb57c9d0eb5b80"
AUTHORITATIVE_PREREG_SHA256 = (
    "7cba3c62328b4bfe15fffbc523979274e834e8e720e16f70d79621eaf6ebdb7b"
)
sys.path.insert(0, str(REPO_ROOT))
DIRECT_INFRASTRUCTURE_FAILURE_REASONS = frozenset(
    {
        "provider_spot_preemption",
        "vm_host_gpu_failure",
        "process_exit_before_scientific_divergence",
        "missing_or_checksum_invalid_required_artifact",
        "pre_unblinding_validator_provenance_failure",
    }
)
PEER_BLOCK_RETRY_REASON = "peer_block_invalidated_by_infra_failure"


class PhaseMapError(RuntimeError):
    pass


def verify_authoritative_prereg(args: argparse.Namespace) -> dict[str, Any]:
    expected_path = (REPO_ROOT / AUTHORITATIVE_PREREG_PATH).resolve()
    if args.prereg_template.resolve() != expected_path:
        raise PhaseMapError("--prereg-template must be the exact authoritative repo path")
    blob = run_checked(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "show",
            f"{AUTHORITATIVE_PREREG_COMMIT}:{AUTHORITATIVE_PREREG_PATH.as_posix()}",
        ]
    ).stdout.encode("utf-8")
    if sha256_bytes(blob) != AUTHORITATIVE_PREREG_SHA256:
        raise PhaseMapError("authoritative prereg Git blob hash differs from pinned hash")
    if args.prereg_template.read_bytes() != blob:
        raise PhaseMapError("working prereg template differs from authoritative Git blob")
    ancestry = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "merge-base",
            "--is-ancestor",
            AUTHORITATIVE_PREREG_COMMIT,
            "HEAD",
        ],
        text=True,
        capture_output=True,
    )
    if ancestry.returncode:
        raise PhaseMapError("runtime source is not a descendant of the prereg commit")
    value = json.loads(blob)
    if not isinstance(value, dict):
        raise PhaseMapError("authoritative prereg root is not an object")
    return value


def enforce_stage_design(args: argparse.Namespace, template: dict[str, Any]) -> str:
    protocol = template["protocol"]
    exact = {
        "model_id": template["frozen"]["model_id"],
        "model_revision": template["frozen"]["model_revision"],
        "image_numeric_id": template["frozen"]["image_id"],
        "eval_split_seed": protocol["eval_split_seed"],
        "train_rows": protocol["train_rows"],
        "eval_rows": protocol["development_eval_rows"],
        "confirmation_audit_rows": protocol["audit_eval_rows"],
        "seq_len": protocol["seq_len"],
        "micro_batch_size": protocol["micro_batch_size"],
        "inner_lr": protocol["inner_lr"],
        "minimum_confirmatory_seeds": template["min_confirmatory_seeds"],
    }
    for field, expected in exact.items():
        if getattr(args, field) != expected:
            raise PhaseMapError(
                f"{field}={getattr(args, field)!r} differs from authority {expected!r}"
            )
    if sorted(args.mu) != [0.0, 0.5, 0.9]:
        raise PhaseMapError("every authorized stage requires mu={0,.5,.9}")
    if args.study_phase in ("p0a_canary", "p0b_canary"):
        stage_name = "p0a" if args.study_phase == "p0a_canary" else "p0b"
        stage = template["canary_stages"][stage_name]
        if args.seed != stage["shuffle_seed"] or args.training_seed != stage["training_seed"]:
            raise PhaseMapError(f"{stage_name} seed pair differs from authority")
        if (
            args.h != [stage["h"]]
            or sorted(args.eta) != [stage["eta"]]
            or sorted(args.mu) != stage["mu"]
            or args.token_budget != stage["token_budget"]
        ):
            raise PhaseMapError(f"{stage_name} block/work differs from authority")
        if (
            args.gpu_slots != stage["gpu_slots"]
            or args.resource_class != stage["machine_type"]
        ):
            raise PhaseMapError(f"{stage_name} machine/gpu slots differ from authority")
        if not args.capture_every_step:
            raise PhaseMapError(f"{stage_name} requires every-step raw capture")
        has_parent = all(
            (
                args.parent_manifest,
                args.expected_parent_manifest_hash,
                args.parent_replay_report,
                args.expected_parent_replay_report_hash,
            )
        )
        any_parent = any(
            (
                args.parent_manifest,
                args.expected_parent_manifest_hash,
                args.parent_replay_report,
                args.expected_parent_replay_report_hash,
            )
        )
        if stage_name == "p0a":
            if any_parent:
                raise PhaseMapError("p0a is the only parentless stage")
            if args.require_distinct_learner_gpu_uuids:
                raise PhaseMapError("p0a may not claim the four-GPU UUID proof")
            return "p0a_single_gpu_bound"
        if not has_parent:
            raise PhaseMapError("p0b requires sealed p0a parent and replay PASS")
        if not args.require_distinct_learner_gpu_uuids:
            raise PhaseMapError("p0b requires the learner/GPU UUID bijection proof")
        return "p0b_four_gpu_bound"
    if args.study_phase == "p1_development":
        grid = template["expected_grid"]
        if args.study_id != template["study_id"]:
            raise PhaseMapError("P1-R0 study id differs from authoritative template")
        if (
            sorted(args.h) != grid["h"]
            or sorted(args.mu) != grid["mu"]
            or sorted(args.eta) != grid["eta"]
            or args.seed != grid["seeds"][0]
            or args.training_seed != template["seed_pairs"][str(args.seed)]
            or args.token_budget != protocol["token_budget"]
        ):
            raise PhaseMapError("P1-R0 grid or seed pair differs from authority")
        if args.gpu_slots != 4 or args.resource_class != "a2-highgpu-4g":
            raise PhaseMapError("P1-R0 requires Spot a2-highgpu-4g and gpu_slots=4")
        if not all(
            (
                args.parent_manifest,
                args.expected_parent_manifest_hash,
                args.parent_replay_report,
                args.expected_parent_replay_report_hash,
            )
        ):
            raise PhaseMapError("P1-R0 requires sealed p0b parent and replay PASS")
        if args.require_distinct_learner_gpu_uuids:
            raise PhaseMapError("P1 may not inherit the non-evidence UUID canary flag")
        return "initial_bound_p1_r0"
    raise PhaseMapError(
        "this runner supports only authority-bound P0 and initial P1-R0; "
        "later stages require a registered cumulative parent-lineage builder"
    )


def json_pointer(value: Any, pointer: str) -> Any:
    """Resolve one RFC 6901-style pointer from an authority document."""
    if not pointer.startswith("/"):
        raise PhaseMapError(f"invalid authority JSON pointer: {pointer!r}")
    current = value
    for raw_part in pointer[1:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            raise PhaseMapError(f"authority JSON pointer is absent: {pointer}")
    return current


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PhaseMapError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise PhaseMapError(f"{label} must contain one JSON object")
    return value


def validate_parent_and_replay(
    args: argparse.Namespace,
    template: dict[str, Any],
    descendant_kind: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Bind P0b/P1 to the exact sealed parent and post-deletion replay.

    The report hash is deliberately over the raw report bytes while the
    parent lineage hash uses the preregistered canonical-JSON definition.
    This makes whitespace changes to the replay evidence detectable without
    making the manifest's identity depend on its pretty-printing.
    """
    policy = template["lineage_policy"]["registered_descendant_kinds"][
        descendant_kind
    ]
    if not policy.get("parent_required"):
        return None, None
    if args.parent_manifest is None or args.parent_replay_report is None:
        raise PhaseMapError(
            f"{descendant_kind} requires parent manifest and replay report paths"
        )

    parent = load_json_object(args.parent_manifest, "parent phase-map manifest")
    parent_hash = sha256_bytes(canonical_json(parent))
    expected_parent_hash = require_sha256(
        args.expected_parent_manifest_hash,
        "--expected-parent-manifest-hash",
    )
    if parent_hash != expected_parent_hash:
        raise PhaseMapError("parent canonical hash differs from frozen authority")
    if parent.get("status") != "sealed_results":
        raise PhaseMapError("parent is not a sealed-results manifest")
    lineage = parent.get("lineage")
    required_parent_kind = policy.get("parent_kind_required")
    if (
        not isinstance(lineage, dict)
        or lineage.get("descendant_kind") != required_parent_kind
    ):
        raise PhaseMapError(
            f"parent must have descendant_kind={required_parent_kind!r}"
        )
    if required_parent_kind == "p0a_single_gpu_bound" and (
        lineage.get("parent_manifest_sha256") is not None
        or lineage.get("parent_replay_report_sha256") is not None
    ):
        raise PhaseMapError("P0a parent unexpectedly has upstream lineage")
    if required_parent_kind in ("p0a_single_gpu_bound", "p0b_four_gpu_bound"):
        if parent.get("mode") != "development":
            raise PhaseMapError("canary parent must remain development-only")
        if template.get("canary_stages", {}).get("evidence") is not False:
            raise PhaseMapError("canary authority unexpectedly permits evidence")
    parent_cells = parent.get("expected_cells")
    parent_rows = parent.get("results")
    final_parent_rows: dict[str, dict[str, Any]] = {}
    if isinstance(parent_rows, list):
        for row in parent_rows:
            cell_id = str(row.get("cell_id", "")) if isinstance(row, dict) else ""
            if cell_id and (
                cell_id not in final_parent_rows
                or int(row.get("attempt", 0))
                > int(final_parent_rows[cell_id].get("attempt", 0))
            ):
                final_parent_rows[cell_id] = row
    if (
        not isinstance(parent_cells, list)
        or len(parent_cells) != 3
        or {float(cell.get("mu")) for cell in parent_cells} != {0.0, 0.5, 0.9}
        or not isinstance(parent_rows, list)
        or len(parent_rows) < 3
        or {str(row.get("cell_id")) for row in parent_rows}
        != {str(cell.get("cell_id")) for cell in parent_cells}
        or any(
            row.get("status") not in ("COMPLETED", "DIVERGED", "INFRA_FAILURE")
            for row in parent_rows
        )
        or set(final_parent_rows)
        != {str(cell.get("cell_id")) for cell in parent_cells}
        or any(
            row.get("status") not in ("COMPLETED", "DIVERGED")
            for row in final_parent_rows.values()
        )
    ):
        raise PhaseMapError("parent does not contain one resolved full-mu block")

    report_raw_hash = sha256_file(args.parent_replay_report)
    expected_report_hash = require_sha256(
        args.expected_parent_replay_report_hash,
        "--expected-parent-replay-report-hash",
    )
    if report_raw_hash != expected_report_hash:
        raise PhaseMapError("parent replay-report raw hash differs from authority")
    replay = load_json_object(args.parent_replay_report, "parent replay report")
    if (
        replay.get("schema") != "yeto_p0_cpu_replay_v1"
        or replay.get("status") != "PASS"
        or replay.get("gpu_deleted_before_replay") is not True
        or replay.get("all_steps_replayed") is not True
        or replay.get("phase_map_integrity_status") != "VALIDATED"
    ):
        raise PhaseMapError("parent replay report is not a complete PASS")
    if replay.get("phase_map_manifest_canonical_sha256") != parent_hash:
        raise PhaseMapError("replay report does not bind the canonical parent")
    if replay.get("phase_map_manifest_sha256") != sha256_file(args.parent_manifest):
        raise PhaseMapError("replay report does not bind the raw parent file")
    if replay.get("replay_validator_git_commit") != parent["frozen"]["git_commit"]:
        raise PhaseMapError("parent replay did not run from the source commit")
    for field in (
        "replay_validator_script_sha256",
        "replay_validator_git_blob_sha256",
        "phase_map_validator_report_sha256",
        "acquisition_manifest_sha256",
        "deletion_evidence_sha256",
    ):
        require_sha256(replay.get(field), f"parent replay {field}")
    if replay.get("cell_count") != len(parent_cells):
        raise PhaseMapError("replay report cell count differs from its parent")
    if replay.get("frozen_tolerance") != {
        "param_atol": 2e-6,
        "param_rtol": 2e-6,
        "tape_norm_rtol": 2e-4,
        "replay_dtype": "numpy_little_endian_f32_with_f64_norm_accumulation",
    }:
        raise PhaseMapError("replay report does not use frozen replay tolerances")
    replay_cells = replay.get("cells")
    if (
        not isinstance(replay_cells, list)
        or {str(cell.get("cell_id")) for cell in replay_cells}
        != {str(cell.get("cell_id")) for cell in parent_cells}
        or any(cell.get("all_steps_replayed") is not True for cell in replay_cells)
    ):
        raise PhaseMapError("replay report does not replay every parent cell")
    return parent, replay


def validate_parent_equality(
    template: dict[str, Any],
    candidate: dict[str, Any],
    parent: dict[str, Any] | None,
    descendant_kind: str,
) -> None:
    if parent is None:
        return
    policy = template["lineage_policy"]["registered_descendant_kinds"][
        descendant_kind
    ]
    pointer_field = {
        "p0b_four_gpu_bound": "must_equal_p0a_parent_paths",
        "initial_bound_p1_r0": "must_equal_p0b_parent_paths",
    }.get(descendant_kind)
    if pointer_field is None:
        return
    for pointer in policy[pointer_field]:
        if json_pointer(candidate, pointer) != json_pointer(parent, pointer):
            raise PhaseMapError(
                f"{descendant_kind} differs from parent at immutable path {pointer}"
            )

    if descendant_kind == "p0b_four_gpu_bound":
        parent_cells = {
            (cell["h"], float(cell["mu"]), float(cell["eta"]), cell["seed"]): cell
            for cell in parent["expected_cells"]
        }
        child_cells = {
            (cell["h"], float(cell["mu"]), float(cell["eta"]), cell["seed"]): cell
            for cell in candidate["expected_cells"]
        }
        if set(parent_cells) != set(child_cells):
            raise PhaseMapError("P0b coordinates differ from P0a")
        for coordinate in sorted(parent_cells):
            before = parent_cells[coordinate].get("normalized_workload_command_hash")
            after = child_cells[coordinate].get("normalized_workload_command_hash")
            if before is None or before != after:
                raise PhaseMapError(
                    f"P0b normalized workload argv differs from P0a at {coordinate}"
                )


def validate_authorized_template_diff(
    template: dict[str, Any],
    candidate: dict[str, Any],
    descendant_kind: str,
    *,
    baseline: dict[str, Any] | None = None,
) -> None:
    registered = template["lineage_policy"]["registered_descendant_kinds"][
        descendant_kind
    ]
    allowed = set(registered["allowed_exact_paths"])
    violations: list[str] = []

    def walk(before: Any, after: Any, pointer: str) -> None:
        if before == after or pointer in allowed:
            return
        if isinstance(before, dict) and isinstance(after, dict):
            for key in sorted(set(before) | set(after)):
                escaped = key.replace("~", "~0").replace("/", "~1")
                child = f"{pointer}/{escaped}"
                if key not in before or key not in after:
                    if child not in allowed:
                        violations.append(child)
                else:
                    walk(before[key], after[key], child)
            return
        violations.append(pointer or "/")

    walk(template if baseline is None else baseline, candidate, "")
    if violations:
        raise PhaseMapError(
            "bound manifest changed unregistered authority paths: "
            + ", ".join(violations[:8])
        )


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def parse_ints(value: str) -> list[int]:
    try:
        result = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("values must be positive")
    return result


def parse_floats(value: str) -> list[float]:
    try:
        result = [float(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated floats") from exc
    if not result or any(not math.isfinite(item) or item < 0 for item in result):
        raise argparse.ArgumentTypeError("values must be finite and non-negative")
    return result


def slug_float(value: float) -> str:
    return format(value, ".12g").replace(".", "p")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def validate_frozen_retry_policy(policy: Any) -> dict[str, Any]:
    if not isinstance(policy, dict):
        raise PhaseMapError("retry_policy must be an object")
    direct = policy.get("direct_infrastructure_failure_reasons")
    allowed = policy.get("allowed_reasons")
    if set(direct or []) != set(DIRECT_INFRASTRUCTURE_FAILURE_REASONS):
        raise PhaseMapError("frozen direct infrastructure-failure reasons drifted")
    if set(allowed or []) != set(DIRECT_INFRASTRUCTURE_FAILURE_REASONS) | {
        PEER_BLOCK_RETRY_REASON
    }:
        raise PhaseMapError("frozen allowed retry reasons drifted")
    if policy.get("peer_retry_reason") != PEER_BLOCK_RETRY_REASON:
        raise PhaseMapError("frozen peer retry reason drifted")
    required_true = (
        "loss_blind_only",
        "rerun_entire_incomplete_block",
        "retain_all_attempts",
        "retry_lineage_required",
        "peer_retry_reason_is_never_failure_reason",
        "infra_failure_reason_must_be_direct_infrastructure_failure_reason",
        "preserve_completed_peer_status_and_artifacts",
        "retry_block_rows_must_be_contiguous",
        "result_acquisition_is_append_only",
        "mechanical_sealing_before_human_or_analysis_unblinding_is_loss_blind",
        "trigger_must_be_genuine_infra_failure_in_immediately_prior_same_block",
        "shared_block_retry_authorization_required",
    )
    if any(policy.get(field) is not True for field in required_true):
        raise PhaseMapError("frozen retry policy lost a required fail-closed rule")
    return policy


def semantics(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "tuning": "full",
        "learners": 4,
        "fragments": 4,
        "inner_optimizer": "adamw",
        "inner_lr": args.inner_lr,
        "outer_optimizer": "nesterov",
        "matrix_merge": "rda",
        "strict_quorum": True,
        "barrier_sync": True,
        "version_matched_anchor": True,
        "delta_correction": "none",
        "injected_baseline": False,
        "wire_dtype": "bf16",
        "syncer_dtype": "f32",
        "fixed_window": True,
        "pad_to_fixed_window_tokens": True,
        "learner_push_delay_ms": [0, 0, 0, 0],
        "learner_delay_jitter_ms": 0,
        "seq_len": args.seq_len,
        "micro_batch_size": args.micro_batch_size,
    }


def cell_id(study_id: str, h: int, mu: float, eta: float, seed: int) -> str:
    return (
        f"{study_id}-h{h}-mu{slug_float(mu)}-eta{slug_float(eta)}-s{seed}"
    )


def compare_command(
    args: argparse.Namespace,
    *,
    h: int,
    mu: float,
    eta: float,
) -> list[str]:
    outer_steps = args.token_budget // (h * args.seq_len)
    frozen_split = args.run_dir / "frozen-eval" / f"seed-{args.seed}" / "materialized"
    command = [
        args.python_executable,
        str(args.command_repo_root / "scripts" / "compare_diloco.py"),
        "--model",
        str(args.model_path),
        "--data",
        str(frozen_split / "train.jsonl"),
        "--prebound-development-eval",
        str(frozen_split / "eval.jsonl"),
        "--settings",
        "m4",
        "--tuning",
        "full",
        "--skip-baseline",
        "--skip-untrained-eval",
        "--token-budget",
        str(args.token_budget),
        "--seq-len",
        str(args.seq_len),
        "--micro-batch-size",
        str(args.micro_batch_size),
        "--inner-lr",
        str(args.inner_lr),
        "--eval-rows",
        str(args.eval_rows),
        "--max-rows",
        str(args.train_rows),
        "--shuffle-rows-seed",
        str(args.seed),
        "--eval-split-seed",
        str(args.eval_split_seed),
        "--training-seed",
        str(args.training_seed),
        "--device",
        args.device,
        "--gpu-slots",
        str(args.gpu_slots),
        "--delta-correction",
        "none",
        "--matrix-merge",
        "rda",
        "--outer-optimizer",
        "nesterov",
        "--outer-momentum",
        format(mu, ".12g"),
        "--outer-lr",
        format(eta, ".12g"),
        "--fixed-window-microsteps",
        str(h),
        "--fixed-window-tokens",
        str(h * args.seq_len),
        "--pad-to-fixed-window-tokens",
        "--freeze-delta-before-delay",
        "--learner-push-delay-ms",
        "0,0,0,0",
        "--learner-delay-jitter-ms",
        "0",
        "--syncer-total-steps",
        str(outer_steps),
        "--learner-max-steps",
        str(args.learner_max_steps),
        "--strict-quorum",
        "--barrier-sync",
        "--version-matched-anchor",
        "--syncer-checkpoint-every",
        str(args.syncer_checkpoint_every),
        "--arm-timeout-min",
        str(args.arm_timeout_min),
        "--work-dir",
        "work",
        "--report-dir",
        "report",
    ]
    if args.capture_every_step:
        command.extend(
            ["--syncer-probe-capture", "--syncer-probe-capture-every", "1"]
        )
    if args.require_distinct_learner_gpu_uuids:
        command.append("--require-distinct-learner-gpu-uuids")
    return command


def normalized_workload_command(command: Sequence[str]) -> list[str]:
    """Remove only the registered P0a/P0b hardware-launch differences."""
    normalized: list[str] = []
    skip_value = False
    role_path_flags = {
        "--model": "<FROZEN_MODEL>",
        "--data": "<PREBOUND_TRAIN>",
        "--prebound-development-eval": "<PREBOUND_DEVELOPMENT_EVAL>",
    }
    for index, token in enumerate(command):
        if skip_value:
            skip_value = False
            continue
        if token == "--gpu-slots":
            skip_value = True
            continue
        if token == "--require-distinct-learner-gpu-uuids":
            continue
        if token in role_path_flags:
            if index + 1 >= len(command):
                raise PhaseMapError(f"{token} lacks a value in normalized command")
            normalized.extend((token, role_path_flags[token]))
            skip_value = True
            continue
        normalized.append(token)
    if skip_value:
        raise PhaseMapError("--gpu-slots lacks a value in normalized command")
    return normalized


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    if 0.0 not in args.mu:
        raise PhaseMapError("every randomized block requires a live mu=0 control")
    if args.token_budget % args.seq_len:
        raise PhaseMapError("token budget must be divisible by seq_len")
    blocks = [(h, eta) for h in sorted(args.h) for eta in sorted(args.eta)]
    rng = random.Random(args.order_seed)
    rng.shuffle(blocks)
    cells = []
    order_index = 0
    for block_index, (h, eta) in enumerate(blocks):
        outer_steps = args.token_budget // (h * args.seq_len)
        if args.token_budget % (h * args.seq_len):
            raise PhaseMapError(f"token budget is not exact for H={h}")
        if outer_steps % 4:
            raise PhaseMapError(f"outer step count must be divisible by fragments: H={h}")
        block_id = f"{args.study_id}-block-h{h}-eta{slug_float(eta)}-s{args.seed}"
        block_mu = list(args.mu)
        rng.shuffle(block_mu)
        control_id = cell_id(args.study_id, h, 0.0, eta, args.seed)
        for within_block_index, mu in enumerate(block_mu):
            command = compare_command(args, h=h, mu=mu, eta=eta)
            cells.append(
                {
                    "cell_id": cell_id(args.study_id, h, mu, eta, args.seed),
                    "H": h,
                    "mu": mu,
                    "eta": eta,
                    "seed": args.seed,
                    "training_seed": args.training_seed,
                    "command_hash": sha256_bytes(canonical_json(command)),
                    "paired_control_id": control_id,
                    "resource_class": args.resource_class,
                    "target_work": {
                        "tokens": args.token_budget,
                        "microsteps": args.token_budget // args.seq_len,
                        "outer_steps": outer_steps,
                    },
                    "randomization": {
                        "block_id": block_id,
                        "block_order_index": block_index,
                        "within_block_index": within_block_index,
                        "order_index": order_index,
                    },
                    "command": command,
                }
            )
            order_index += 1
    plan = {
        "schema": "yeto_phase_map_randomization_v1",
        "study_id": args.study_id,
        "seed": args.seed,
        "training_seed": args.training_seed,
        "order_seed": args.order_seed,
        "block_fields": ["H", "eta", "seed"],
        "within_block_field": "mu",
        "cells": cells,
    }
    plan["randomization_plan_hash"] = sha256_bytes(canonical_json(plan))
    return plan


def campaign_command_hash(plan: dict[str, Any]) -> str:
    registry = [
        {"cell_id": cell["cell_id"], "command_hash": cell["command_hash"]}
        for cell in plan["cells"]
    ]
    return sha256_bytes(canonical_json(registry))


def build_bound_manifest(
    args: argparse.Namespace,
    plan: dict[str, Any],
    *,
    model_hash: str,
    data_hash: str,
    train_rows_hash: str,
    development_eval_rows_hash: str,
    development_eval_packed_hash: str,
    development_eval_example_ids_hash: str,
    development_eval_token_ids_hash: str,
    development_eval_source_indices_hash: str,
    audit_eval_rows_hash: str,
    audit_eval_packed_hash: str,
    audit_eval_example_ids_hash: str,
    audit_eval_token_ids_hash: str,
    audit_eval_source_indices_hash: str,
    audit_access_policy_hash: str,
    train_pool_source_indices_hash: str,
    train_source_indices_hash: str,
) -> dict[str, Any]:
    template = verify_authoritative_prereg(args)
    descendant_kind = enforce_stage_design(args, template)
    if str(template.get("schema_version")) != "0.2":
        raise PhaseMapError("preregistration template must use schema_version 0.2")
    policy_hash = sha256_bytes(canonical_json(template["confirmation_policy"]))
    if audit_access_policy_hash != policy_hash:
        raise PhaseMapError("audit-access policy hash differs from frozen authority")
    parent, _parent_replay = validate_parent_and_replay(
        args, template, descendant_kind
    )
    # P0b is a registered hardware-only descendant of P0a.  Building it from
    # that exact sealed parent makes every inherited hash/work field immutable
    # by construction.  P0a and the initial P1 binding are compared directly
    # with the authoritative preregistration template.
    manifest = deepcopy(
        parent if descendant_kind == "p0b_four_gpu_bound" else template
    )
    manifest["status"] = "bound_launch_authority"
    manifest["study_id"] = args.study_id
    manifest["mode"] = "development"
    manifest["min_confirmatory_seeds"] = args.minimum_confirmatory_seeds
    manifest["expected_grid"] = {
        "h": sorted(args.h),
        "mu": sorted(args.mu),
        "eta": sorted(args.eta),
        "seeds": [args.seed],
    }
    manifest["seed_pairs"] = {str(args.seed): args.training_seed}
    manifest["expected_cells"] = [
        {
            "cell_id": cell["cell_id"],
            "h": cell["H"],
            "mu": cell["mu"],
            "eta": cell["eta"],
            "seed": cell["seed"],
            "training_seed": cell["training_seed"],
            "block_id": cell["randomization"]["block_id"],
            "paired_control_id": cell["paired_control_id"],
            "command_hash": cell["command_hash"],
            "normalized_workload_command_hash": sha256_bytes(
                canonical_json(normalized_workload_command(cell["command"]))
            ),
        }
        for cell in plan["cells"]
    ]
    command_hash = campaign_command_hash(plan)
    manifest["frozen"].update(
        {
            "git_commit": args.git_commit,
            "image_id": args.image_numeric_id,
            "image_digest": args.image_digest,
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "model_hash": model_hash,
            "data_hash": data_hash,
            "development_eval_rows_hash": development_eval_rows_hash,
            "development_eval_packed_hash": development_eval_packed_hash,
            "development_eval_example_ids_hash": development_eval_example_ids_hash,
            "development_eval_token_ids_hash": development_eval_token_ids_hash,
            "development_eval_source_indices_hash": (
                development_eval_source_indices_hash
            ),
            "audit_eval_rows_hash": audit_eval_rows_hash,
            "audit_eval_packed_hash": audit_eval_packed_hash,
            "audit_eval_example_ids_hash": audit_eval_example_ids_hash,
            "audit_eval_token_ids_hash": audit_eval_token_ids_hash,
            "audit_eval_source_indices_hash": audit_eval_source_indices_hash,
            "audit_access_policy_hash": audit_access_policy_hash,
            "train_pool_source_indices_hash": train_pool_source_indices_hash,
            "train_source_indices_hashes": {
                str(args.seed): train_source_indices_hash
            },
            "train_rows_hashes": {str(args.seed): train_rows_hash},
            "command_hash": command_hash,
            "cell_command_hashes": {
                cell["cell_id"]: cell["command_hash"] for cell in plan["cells"]
            },
            "randomization_plan_hash": plan["randomization_plan_hash"],
        }
    )
    manifest["protocol"].update(
        {
            "tuning": "full",
            "train_rows": args.train_rows,
            "development_eval_rows": args.eval_rows,
            "audit_eval_rows": args.confirmation_audit_rows,
            "split_population_rows": (
                args.train_rows + args.eval_rows + args.confirmation_audit_rows
            ),
            "seq_len": args.seq_len,
            "micro_batch_size": args.micro_batch_size,
            "learners": 4,
            "fragments": 4,
            "inner_optimizer": "adamw",
            "inner_lr": args.inner_lr,
            "outer_optimizer": "nesterov",
            "matrix_merge": "rda",
            "delta_correction": "none",
            "wire_dtype": "bf16",
            "syncer_dtype": "f32",
            "strict_quorum": True,
            "barrier": True,
            "version_matched": True,
            "fixed_window": True,
            "pad_to_fixed_window_tokens": True,
            "learner_push_delay_ms": [0, 0, 0, 0],
            "learner_delay_jitter_ms": 0,
            "eval_split_seed": args.eval_split_seed,
            "token_budget": args.token_budget,
            "gpu_slots": args.gpu_slots,
            "machine_type": args.resource_class,
            "spot_only": True,
            "on_demand_fallback": False,
            "injected_baseline": False,
            "per_example_loss_required": True,
        }
    )
    manifest["horizon_work"] = {
        str(h): {
            "fixed_window_microsteps": h,
            "fixed_window_tokens": h * args.seq_len,
            "outer_steps": args.token_budget // (h * args.seq_len),
        }
        for h in sorted(args.h)
    }
    manifest["randomization"].update(
        {
            "unit": "arm",
            "block_fields": ["h", "eta", "seed"],
            "required_mu_per_block": sorted(args.mu),
            "block_order": "materialized_pseudorandom_permutation",
            "within_block_order": "materialized_pseudorandom_permutation",
            "loss_blind": True,
            "plan_hash": plan["randomization_plan_hash"],
        }
    )
    policy = validate_frozen_retry_policy(manifest.get("retry_policy"))
    manifest["frozen"]["retry_policy_hash"] = sha256_bytes(
        canonical_json(policy)
    )
    manifest["lineage"].update(
        {
            "authoritative_prereg_path": AUTHORITATIVE_PREREG_PATH.as_posix(),
            "authoritative_prereg_source_commit": AUTHORITATIVE_PREREG_COMMIT,
            "authoritative_prereg_template_sha256": AUTHORITATIVE_PREREG_SHA256,
            "parent_manifest_sha256": (
                None
                if descendant_kind == "p0a_single_gpu_bound"
                else require_sha256(
                    args.expected_parent_manifest_hash,
                    "--expected-parent-manifest-hash",
                )
            ),
            "parent_replay_report_sha256": (
                None
                if descendant_kind == "p0a_single_gpu_bound"
                else require_sha256(
                    args.expected_parent_replay_report_hash,
                    "--expected-parent-replay-report-hash",
                )
            ),
            "descendant_kind": descendant_kind,
        }
    )
    manifest["results"] = []
    validate_authorized_template_diff(
        template,
        manifest,
        descendant_kind,
        baseline=parent if descendant_kind == "p0b_four_gpu_bound" else None,
    )
    validate_parent_equality(template, manifest, parent, descendant_kind)
    return manifest


def build_schema_fixture(
    manifest: dict[str, Any], plan: dict[str, Any]
) -> dict[str, Any]:
    """Populate deterministic fake completed rows for schema integration tests."""
    fixture = deepcopy(manifest)
    frozen = fixture["frozen"]
    protocol = fixture["protocol"]
    rows = []
    for cell in plan["cells"]:
        artifact_sha = sha256_bytes(cell["cell_id"].encode())
        h = cell["H"]
        eta = cell["eta"]
        # An interior synthetic optimum keeps validator summaries deterministic.
        loss = 2.0 + 0.01 * math.log2(eta / 0.04375) ** 2 + 0.02 * cell["mu"]
        rows.append(
            {
                "attempt_id": f"{cell['cell_id']}-attempt-1",
                "cell_id": cell["cell_id"],
                "h": h,
                "mu": cell["mu"],
                "eta": eta,
                "seed": cell["seed"],
                "training_seed": cell["training_seed"],
                "status": "COMPLETED",
                "evaluation_role": "development",
                "failure_reason": None,
                "loss": loss,
                "work": {
                    "fixed_window_microsteps": h,
                    "fixed_window_tokens": h * protocol["seq_len"],
                    "outer_steps": cell["target_work"]["outer_steps"],
                    "token_budget": protocol["token_budget"],
                    "eval_rows": protocol["development_eval_rows"],
                },
                "observed_work": {
                    "tokens": protocol["token_budget"],
                    "microsteps": protocol["token_budget"] // protocol["seq_len"],
                    "outer_steps": cell["target_work"]["outer_steps"],
                    "full_quorum": True,
                    "fixed_window_exact": True,
                    "version_matched_anchor_resolved": True,
                },
                "git_commit": frozen["git_commit"],
                "image_digest": frozen["image_digest"],
                "model_hash": frozen["model_hash"],
                "data_hash": frozen["data_hash"],
                "eval_source_indices_hash": frozen[
                    "development_eval_source_indices_hash"
                ],
                "train_pool_source_indices_hash": frozen[
                    "train_pool_source_indices_hash"
                ],
                "train_source_indices_hash": frozen[
                    "train_source_indices_hashes"
                ][str(cell["seed"])],
                "train_rows_hash": frozen["train_rows_hashes"][str(cell["seed"])],
                "eval_rows_hash": frozen["development_eval_rows_hash"],
                "eval_hash": frozen["development_eval_packed_hash"],
                "eval_example_ids_hash": frozen[
                    "development_eval_example_ids_hash"
                ],
                "eval_token_ids_hash": frozen[
                    "development_eval_token_ids_hash"
                ],
                "command_hash": frozen["cell_command_hashes"][cell["cell_id"]],
                "normalized_workload_command_hash": sha256_bytes(
                    canonical_json(normalized_workload_command(cell["command"]))
                ),
                "capture_uri": f"gs://schema-fixture/{cell['cell_id']}/capture",
                "capture_sha256": artifact_sha,
                "result_uri": f"gs://schema-fixture/{cell['cell_id']}/result.json",
                "result_sha256": artifact_sha,
                "per_example_loss_uri": f"gs://schema-fixture/{cell['cell_id']}/eval.jsonl",
                "per_example_loss_sha256": artifact_sha,
                "paired_control_id": cell["paired_control_id"],
                "barrier": True,
                "version_matched": True,
                "matrix_merge": "rda",
                "strict_quorum": True,
                "delta_correction": "none",
                "injected_baseline": False,
                "spot": True,
                "block_id": cell["randomization"]["block_id"],
                "order_index": cell["randomization"]["within_block_index"],
                "attempt": 1,
                "retry_of": None,
                "retry_reason": None,
                "retry_authorization": None,
                "hardware": {
                    "market": "spot",
                    "provider": "gcp",
                    "instance_type": protocol["machine_type"],
                    "region": "us-central1",
                    "project": "schema-fixture",
                    "zone": "us-central1-b",
                    "instance_name": "schema-fixture-vm",
                    "instance_id": "123456789",
                    "instance_numeric_id": "123456789",
                    "boot_disk_name": "schema-fixture-disk",
                    "boot_disk_id": "987654321",
                    "boot_disk_numeric_id": "987654321",
                    "source_image_id": frozen["image_id"],
                    "source_image_numeric_id": frozen["image_id"],
                    "image_id": frozen["image_id"],
                    "provisioning_evidence_uri": "gs://schema-fixture/spot.json",
                    "provisioning_evidence_sha256": artifact_sha,
                },
                "started_at": "2026-07-14T12:00:00Z",
                "ended_at": "2026-07-14T13:00:00Z",
            }
        )
    if fixture["lineage"]["descendant_kind"] in (
        "p0a_single_gpu_bound",
        "p0b_four_gpu_bound",
    ):
        for row in rows:
            evidence_sha = row["capture_sha256"]
            row["hardware"].update(
                {
                    "acquisition_status": "sealed_acquisition_pending_teardown",
                    "acquisition_manifest_sha256": evidence_sha,
                    "acquisition_manifest_canonical_sha256": evidence_sha,
                    "acquisition_checksum_sha256": evidence_sha,
                    "acquisition_seal_sha256": evidence_sha,
                    "final_manifest_status": "sealed_results",
                    "deletion_evidence_sha256": evidence_sha,
                    "artifact_sealed_at": "2026-07-14T13:05:00Z",
                    "deletion_requested_at": "2026-07-14T13:06:00Z",
                    "deletion_completed_at": "2026-07-14T13:07:00Z",
                    "finalized_at": "2026-07-14T13:08:00Z",
                }
            )
    if fixture["lineage"]["descendant_kind"] == "p0b_four_gpu_bound":
        for row in rows:
            evidence_sha = row["capture_sha256"]
            row["hardware"].update(
                {
                    "provisioning_started_at": "2026-07-14T11:00:00Z",
                    "provisioning_completed_at": "2026-07-14T11:05:00Z",
                    "nvidia_smi_inventory_uri": row["capture_uri"]
                    + "/gpu-allocation.json",
                    "nvidia_smi_inventory_sha256": evidence_sha,
                    "learner_gpu_map_uri": row["capture_uri"]
                    + "/gpu-allocation.json",
                    "learner_gpu_map_sha256": evidence_sha,
                    "barrier_version_trace_uri": row["capture_uri"]
                    + "/tape.jsonl",
                    "barrier_version_trace_sha256": evidence_sha,
                    "distinct_a100_gpu_uuid_count": 4,
                    "learner_gpu_uuid_bijection": {
                        str(learner): f"GPU-schema-fixture-{learner}"
                        for learner in range(4)
                    },
                    "instance_not_found_evidence_uri": row["capture_uri"]
                    + "/instance-not-found.json",
                    "instance_not_found_evidence_sha256": evidence_sha,
                    "disk_not_found_evidence_uri": row["capture_uri"]
                    + "/disk-not-found.json",
                    "disk_not_found_evidence_sha256": evidence_sha,
                    "zero_accelerator_evidence_uri": row["capture_uri"]
                    + "/zero-accelerator.json",
                    "zero_accelerator_evidence_sha256": evidence_sha,
                }
            )
    fixture["results"] = rows
    fixture["status"] = "sealed_results"
    return fixture


def build_retry_schema_fixture(
    manifest: dict[str, Any], plan: dict[str, Any]
) -> dict[str, Any]:
    """Build a mixed-status whole-block retry fixture for contract tests."""
    prior = build_schema_fixture(manifest, plan)
    first_block_id = plan["cells"][0]["randomization"]["block_id"]
    block_rows = [
        row for row in prior["results"] if row["block_id"] == first_block_id
    ]
    if len(block_rows) != 3:
        raise PhaseMapError("schema fixture expected one three-arm randomized block")
    trigger = block_rows[1]
    trigger["status"] = "INFRA_FAILURE"
    trigger["failure_reason"] = "provider_spot_preemption"
    trigger["loss"] = None
    trigger["observed_work"] = {
        "tokens": 0,
        "microsteps": 0,
        "outer_steps": 0,
        "full_quorum": False,
        "fixed_window_exact": False,
        "version_matched_anchor_resolved": False,
    }
    trigger["per_example_loss_uri"] = None
    trigger["per_example_loss_sha256"] = None
    trigger["ended_at"] = "2026-07-14T12:30:00Z"

    prior_hash = sha256_bytes(canonical_json(prior))
    authorization = {
        "loss_blind": True,
        "policy_hash": prior["frozen"]["retry_policy_hash"],
        "trigger_attempt_id": trigger["attempt_id"],
        "trigger_reason": trigger["failure_reason"],
        "trigger_block_id": first_block_id,
        "prior_manifest_sha256": prior_hash,
        "authorized_at_utc": "2026-07-14T14:00:00Z",
    }
    fixture = deepcopy(prior)
    for previous in block_rows:
        row = deepcopy(previous)
        row["attempt"] = 2
        row["attempt_id"] = f"{row['cell_id']}-attempt-2"
        row["retry_of"] = previous["attempt_id"]
        row["retry_reason"] = (
            previous["failure_reason"]
            if previous["status"] == "INFRA_FAILURE"
            else PEER_BLOCK_RETRY_REASON
        )
        row["retry_authorization"] = deepcopy(authorization)
        row["status"] = "COMPLETED"
        row["failure_reason"] = None
        row["loss"] = 2.05 + 0.02 * float(row["mu"])
        row["observed_work"] = {
            "tokens": fixture["protocol"]["token_budget"],
            "microsteps": (
                fixture["protocol"]["token_budget"]
                // fixture["protocol"]["seq_len"]
            ),
            "outer_steps": fixture["horizon_work"][str(row["h"])][
                "outer_steps"
            ],
            "full_quorum": True,
            "fixed_window_exact": True,
            "version_matched_anchor_resolved": True,
        }
        artifact_sha = sha256_bytes(row["attempt_id"].encode())
        row["capture_uri"] += "/attempt-2"
        row["capture_sha256"] = artifact_sha
        row["result_uri"] += "/attempt-2"
        row["result_sha256"] = artifact_sha
        row["per_example_loss_uri"] = (
            f"gs://schema-fixture/{row['cell_id']}/attempt-2/eval.jsonl"
        )
        row["per_example_loss_sha256"] = artifact_sha
        row["started_at"] = "2026-07-14T14:05:00Z"
        row["ended_at"] = "2026-07-14T15:05:00Z"
        fixture["results"].append(row)
    return fixture


def run_checked(command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(list(command), text=True, capture_output=True, **kwargs)
    if result.returncode:
        detail = (result.stderr or result.stdout)[-4000:]
        raise PhaseMapError(f"command failed ({result.returncode}): {shlex.join(command)}\n{detail}")
    return result


def verify_sha256_manifest(root: Path, manifest: Path) -> None:
    lines = manifest.read_text().splitlines()
    if not lines:
        raise PhaseMapError(f"empty checksum manifest: {manifest}")
    seen = set()
    for line in lines:
        digest, separator, name = line.partition("  ")
        if not separator or len(digest) != 64 or name.startswith(("/", "../")):
            raise PhaseMapError(f"invalid checksum manifest line: {line!r}")
        path = root / name
        if name in seen or not path.is_file() or path.is_symlink():
            raise PhaseMapError(f"invalid or duplicate model artifact: {name}")
        seen.add(name)
        if sha256_file(path) != digest:
            raise PhaseMapError(f"model checksum mismatch: {name}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest
    }
    if actual != seen:
        missing = sorted(seen - actual)
        extra = sorted(actual - seen)
        raise PhaseMapError(
            f"model tree differs from manifest: missing={missing[:5]} extra={extra[:5]}"
        )


def stage_and_verify_inputs(args: argparse.Namespace) -> tuple[str, str]:
    model_path = args.model_path
    if args.model_source_uri:
        if model_path.exists() and any(model_path.iterdir()):
            raise PhaseMapError(f"model staging directory is not empty: {model_path}")
        model_path.mkdir(parents=True, exist_ok=True)
        run_checked(
            [
                "gcloud",
                "storage",
                "rsync",
                "--recursive",
                args.model_source_uri,
                str(model_path),
            ]
        )
    manifest = model_path / "model-files.sha256"
    if not manifest.is_file():
        raise PhaseMapError(f"missing model checksum manifest: {manifest}")
    verify_sha256_manifest(model_path, manifest)
    revision_path = model_path / "model-revision.txt"
    model_id_path = model_path / "model-id.txt"
    if revision_path.read_text().strip() != args.model_revision:
        raise PhaseMapError("staged model revision marker differs from frozen revision")
    if model_id_path.read_text().strip() != args.model_id:
        raise PhaseMapError("staged model id marker differs from frozen model id")
    model_hash = sha256_file(manifest)
    data_hash = sha256_file(args.data)
    if args.expected_model_hash and model_hash != args.expected_model_hash:
        raise PhaseMapError("model manifest hash differs from frozen value")
    if args.expected_data_hash and data_hash != args.expected_data_hash:
        raise PhaseMapError("data hash differs from frozen value")
    return model_hash, data_hash


def verify_source_checkout(args: argparse.Namespace) -> str:
    command_root = args.command_repo_root.resolve()
    runtime_root = REPO_ROOT.resolve()
    if command_root != runtime_root:
        raise PhaseMapError(
            f"command repo root {command_root} is not runtime source root {runtime_root}"
        )
    actual_commit = run_checked(
        ["git", "-C", str(command_root), "rev-parse", "HEAD"]
    ).stdout.strip()
    if actual_commit != args.git_commit:
        raise PhaseMapError("runtime Git commit differs from frozen commit")
    dirty = run_checked(
        [
            "git",
            "-C",
            str(command_root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ]
    ).stdout
    if dirty.strip():
        raise PhaseMapError("runtime command checkout has tracked or untracked changes")
    compare = command_root / "scripts" / "compare_diloco.py"
    if not compare.is_file() or compare.resolve() != COMPARE.resolve():
        raise PhaseMapError("exact compare entrypoint is not in the verified checkout")
    return actual_commit


def require_sha256(value: str | None, label: str) -> str:
    if value is None or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise PhaseMapError(f"{label} must be a lowercase 64-hex SHA-256")
    return value


def prepare_eval_bundle(
    args: argparse.Namespace,
    *,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    from scripts.compare_diloco import materialize_eval_provenance, split_data

    train, eval_file, train_rows = split_data(
        str(args.data),
        output_dir / "materialized",
        args.eval_rows,
        args.train_rows,
        seed,
        args.eval_split_seed,
        args.confirmation_audit_rows,
    )
    if train_rows != args.train_rows:
        raise PhaseMapError(
            f"materialized train rows {train_rows} != frozen {args.train_rows}"
        )
    summary = materialize_eval_provenance(
        str(args.model_path),
        eval_file,
        args.seq_len,
        output_dir / "provenance",
        split_provenance=output_dir / "materialized" / "split_provenance.json",
    )
    summary["train_file_sha256"] = sha256_file(train)
    summary.update(
        {
            "development_eval_rows_hash": summary["eval_rows_hash"],
            "development_eval_packed_hash": summary["eval_packed_hash"],
            "development_eval_example_ids_hash": summary[
                "eval_example_ids_hash"
            ],
            "development_eval_token_ids_hash": summary["eval_token_ids_hash"],
            "development_eval_source_indices_hash": summary[
                "eval_source_indices_hash"
            ],
        }
    )
    audit_file = output_dir / "materialized" / "confirmation-audit.jsonl"
    audit_summary = materialize_eval_provenance(
        str(args.model_path),
        audit_file,
        args.seq_len,
        output_dir / "audit-provenance",
    )
    split = json.loads(
        (output_dir / "materialized" / "split_provenance.json").read_text()
    )
    audit_indices = split.get("audit_eval_source_indices")
    if not isinstance(audit_indices, list) or len(audit_indices) != args.confirmation_audit_rows:
        raise PhaseMapError("materialized confirmation-audit split has wrong size")
    summary.update(
        {
            "audit_eval_rows_hash": audit_summary["eval_rows_hash"],
            "audit_eval_example_ids_hash": audit_summary[
                "eval_example_ids_hash"
            ],
            "audit_eval_packed_hash": audit_summary["eval_packed_hash"],
            "audit_eval_token_ids_hash": audit_summary["eval_token_ids_hash"],
            "audit_eval_source_indices_hash": sha256_bytes(
                canonical_json(audit_indices)
            ),
            "audit_eval_row_count": audit_summary["eval_row_count"],
            "audit_eval_sequence_count": audit_summary["eval_sequence_count"],
            "audit_eval_supervised_token_count": audit_summary[
                "eval_supervised_token_count"
            ],
            "audit_model_evaluation_accesses": [],
            "audit_outcome_fields_emitted": False,
        }
    )
    authority = verify_authoritative_prereg(args)
    summary["audit_access_policy_hash"] = sha256_bytes(
        canonical_json(authority["confirmation_policy"])
    )
    # Pre-P3 scientific processes receive only the frozen train/development
    # paths.  Retain the prebound confirmation hashes, but remove raw rows,
    # token packs, and source indices from the execution filesystem.
    audit_file.unlink()
    shutil.rmtree(output_dir / "audit-provenance")
    sanitized_split = {
        key: value
        for key, value in split.items()
        if "audit" not in key and "confirmation" not in key
    }
    write_json(
        output_dir / "materialized" / "split_provenance.json", sanitized_split
    )
    summary["seed"] = seed
    write_json(output_dir / "eval-freeze.json", summary)
    summary["_eval_sequences_path"] = str(
        output_dir / "provenance" / "eval_sequences.jsonl"
    )
    return summary


def load_provider_evidence(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    try:
        evidence = json.loads(args.provider_evidence.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PhaseMapError(f"cannot read provider evidence: {exc}") from exc
    if evidence.get("provider") != "gcp" or evidence.get("market") != "spot":
        raise PhaseMapError("provider evidence must identify GCP Spot")
    if evidence.get("provisioning_model") != "SPOT":
        raise PhaseMapError("provider evidence lacks provisioningModel=SPOT")
    if evidence.get("instance_termination_action") != "DELETE":
        raise PhaseMapError("provider evidence lacks termination action DELETE")
    for field in ("instance_id", "boot_disk_id", "source_image_id"):
        if not re.fullmatch(r"[0-9]+", str(evidence.get(field, ""))):
            raise PhaseMapError(f"provider evidence lacks numeric {field}")
    for field in ("project", "zone", "region", "instance_name", "instance_type"):
        if not isinstance(evidence.get(field), str) or not evidence[field]:
            raise PhaseMapError(f"provider evidence lacks {field}")
    if str(evidence["source_image_id"]) != args.image_numeric_id:
        raise PhaseMapError("provider evidence source image differs from frozen image")
    if evidence["instance_type"] != args.resource_class:
        raise PhaseMapError("provider evidence machine type differs from frozen stage")
    if args.require_distinct_learner_gpu_uuids:
        for field in (
            "boot_disk_name",
            "provisioning_started_at",
            "provisioning_completed_at",
        ):
            if not isinstance(evidence.get(field), str) or not evidence[field]:
                raise PhaseMapError(f"P0b provider evidence lacks {field}")
    return evidence, sha256_file(args.provider_evidence)


def snapshot_provider_evidence(
    args: argparse.Namespace,
    evidence: dict[str, Any],
    digest: str,
) -> tuple[Path, str]:
    instance_id = str(evidence.get("instance_id", ""))
    if not re.fullmatch(r"[0-9]+", instance_id):
        raise PhaseMapError("provider evidence instance_id must be numeric")
    destination = (
        args.run_dir / "provider-evidence" / f"instance-{instance_id}-{digest}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = args.provider_evidence.read_bytes()
    if destination.exists() and destination.read_bytes() != raw:
        raise PhaseMapError("immutable provider evidence snapshot already differs")
    if not destination.exists():
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_bytes(raw)
        temporary.replace(destination)
    if sha256_file(destination) != digest:
        raise PhaseMapError("provider evidence snapshot hash mismatch")
    return destination, uri_for(args, destination)


def retain_prior_provider_evidence(
    args: argparse.Namespace, prior_manifest: dict[str, Any]
) -> None:
    expected: dict[str, set[str]] = defaultdict(set)
    for row in prior_manifest.get("results", []):
        hardware = row.get("hardware") or {}
        digest = hardware.get("provisioning_evidence_sha256")
        uri = hardware.get("provisioning_evidence_uri")
        if isinstance(digest, str) and isinstance(uri, str):
            expected[digest].add(uri)
    supplied = {sha256_file(path): path for path in args.prior_provider_evidence}
    if set(supplied) != set(expected):
        raise PhaseMapError(
            "retry must retain exactly every prior provider-evidence digest"
        )
    records = []
    for digest, uris in sorted(expected.items()):
        source = supplied[digest]
        destination = args.run_dir / "provider-evidence" / "prior" / f"{digest}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        raw = source.read_bytes()
        if destination.exists() and destination.read_bytes() != raw:
            raise PhaseMapError("retained prior provider evidence differs")
        if not destination.exists():
            temporary = destination.with_suffix(".json.tmp")
            temporary.write_bytes(raw)
            temporary.replace(destination)
        records.append(
            {
                "sha256": digest,
                "original_uris": sorted(uris),
                "retained_uri": uri_for(args, destination),
                "retained_path": destination.relative_to(args.run_dir).as_posix(),
            }
        )
    write_json(
        args.run_dir / "provider-evidence" / "lineage.json",
        {
            "schema": "yeto_provider_evidence_lineage_v1",
            "append_only": True,
            "prior_evidence": records,
        },
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PhaseMapError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise PhaseMapError(f"{path}:{line_number}: expected object")
        rows.append(value)
    return rows


def validate_tape(path: Path, cell: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(path)
    expected_steps = cell["target_work"]["outer_steps"]
    if len(rows) != expected_steps:
        raise PhaseMapError(f"event tape has {len(rows)} rows, expected {expected_steps}")
    fragments = Counter()
    responder_microsteps = 0
    responder_tokens = 0
    base_versions: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows, 1):
        if row.get("step") != index:
            raise PhaseMapError(f"event tape step {index} is missing or reordered")
        fragment = row.get("fragment")
        if fragment not in (0, 1, 2, 3):
            raise PhaseMapError(f"invalid fragment at tape step {index}")
        fragments[fragment] += 1
        responders = row.get("responders")
        if not isinstance(responders, list) or sorted(r.get("id") for r in responders) != [0, 1, 2, 3]:
            raise PhaseMapError(f"step {index} lacks exact full quorum")
        for responder in responders:
            if responder.get("c_steps") != cell["H"]:
                raise PhaseMapError(f"step {index} has non-H microstep work")
            if responder.get("c_tokens") != cell["H"] * args.seq_len:
                raise PhaseMapError(f"step {index} has non-H token work")
            if responder.get("anchor_base_resolved") is not True:
                raise PhaseMapError(f"step {index} lacks version-matched anchor proof")
            responder_microsteps += responder["c_steps"]
            responder_tokens += responder["c_tokens"]
            base_versions[(responder["id"], fragment)].append(
                int(responder["base_version"])
            )
    expected_per_fragment = expected_steps // 4
    if fragments != Counter({i: expected_per_fragment for i in range(4)}):
        raise PhaseMapError("outer commits are not balanced across four fragments")
    for key, versions in base_versions.items():
        if versions != sorted(versions) or len(versions) != len(set(versions)):
            raise PhaseMapError(f"non-monotone base versions for learner/fragment {key}")
    observed = {
        "tokens": responder_tokens // 4,
        "microsteps": responder_microsteps // 4,
        "outer_steps": len(rows),
        "per_fragment_outer_steps": dict(sorted(fragments.items())),
        "full_quorum": True,
        "fixed_window_exact": True,
        "version_matched_anchor_resolved": True,
    }
    for key in ("tokens", "microsteps", "outer_steps"):
        if observed[key] != cell["target_work"][key]:
            raise PhaseMapError(
                f"observed {key}={observed[key]} != target {cell['target_work'][key]}"
            )
    return observed


def validate_layout(attempt_dir: Path) -> tuple[str, list[str]]:
    paths = [
        attempt_dir / "work" / "m4" / f"learner-{learner}" / "resolved-layout.json"
        for learner in range(4)
    ]
    layouts = [json.loads(path.read_text()) for path in paths]
    hashes = [sha256_file(path) for path in paths]
    if len(set(hashes)) != 1:
        raise PhaseMapError("learner resolved-layout artifacts are not identical")
    first = layouts[0]
    fragments = first.get("fragments", [])
    if first.get("matrix_merge") != "rda" or len(fragments) != 4:
        raise PhaseMapError("resolved layout does not identify four-fragment RDA")
    modes = [fragment.get("merge_mode") for fragment in fragments]
    if len(modes) != 4 or "rda" not in modes or any(mode not in ("avg", "rda") for mode in modes):
        raise PhaseMapError(f"unexpected resolved fragment merge modes: {modes}")
    return hashes[0], modes


def validate_gpu_uuid_bijection(attempt_dir: Path) -> dict[str, Any]:
    """Validate the attempt-local four-A100 inventory and learner bindings."""
    allocation_path = attempt_dir / "report" / "gpu-allocation.json"
    allocation = load_json_object(allocation_path, "GPU allocation proof")
    inventory = allocation.get("gpu_inventory")
    assignments = allocation.get("learner_assignments")
    if (
        allocation.get("schema") != "yeto_learner_gpu_uuid_bijection_v1"
        or allocation.get("distinct_gpu_uuid_count") != 4
        or allocation.get("one_learner_per_distinct_gpu_uuid") is not True
        or not isinstance(inventory, list)
        or len(inventory) != 4
        or not isinstance(assignments, list)
        or len(assignments) != 4
    ):
        raise PhaseMapError("P0b GPU allocation proof is incomplete")
    inventory_by_index: dict[int, dict[str, Any]] = {}
    for row in inventory:
        if not isinstance(row, dict) or isinstance(row.get("cuda_index"), bool):
            raise PhaseMapError("P0b nvidia-smi inventory row is malformed")
        index = int(row.get("cuda_index"))
        uuid = row.get("uuid")
        name = row.get("name")
        if (
            index in inventory_by_index
            or not isinstance(uuid, str)
            or not uuid.startswith("GPU-")
            or not isinstance(name, str)
            or "A100" not in name.upper()
        ):
            raise PhaseMapError("P0b inventory is not four distinct full A100s")
        inventory_by_index[index] = row
    if len({row["uuid"] for row in inventory_by_index.values()}) != 4:
        raise PhaseMapError("P0b inventory repeats a GPU UUID")

    assignment_by_learner: dict[int, dict[str, Any]] = {}
    for row in assignments:
        if not isinstance(row, dict) or isinstance(row.get("learner_id"), bool):
            raise PhaseMapError("P0b learner/GPU assignment row is malformed")
        learner = int(row.get("learner_id"))
        physical = int(row.get("physical_cuda_index"))
        if learner in assignment_by_learner or physical not in inventory_by_index:
            raise PhaseMapError("P0b learner/GPU assignment is not a bijection")
        inventory_row = inventory_by_index[physical]
        if (
            row.get("gpu_uuid") != inventory_row["uuid"]
            or row.get("gpu_name") != inventory_row["name"]
        ):
            raise PhaseMapError("P0b learner assignment differs from inventory")
        assignment_by_learner[learner] = row
    if set(assignment_by_learner) != {0, 1, 2, 3} or len(
        {row["gpu_uuid"] for row in assignment_by_learner.values()}
    ) != 4:
        raise PhaseMapError("P0b does not bind learners 0..3 to four UUIDs")

    device_paths: list[Path] = []
    for learner, assigned in sorted(assignment_by_learner.items()):
        path = (
            attempt_dir
            / "work"
            / "m4"
            / f"learner-{learner}"
            / "resolved-device.json"
        )
        device = load_json_object(path, f"learner {learner} resolved device")
        if (
            device.get("schema") != "yeto_resolved_device_v1"
            or device.get("learner_id") != learner
            or device.get("rank") != 0
            or device.get("physical_cuda_index")
            != assigned["physical_cuda_index"]
            or device.get("assigned_gpu_uuid") != assigned["gpu_uuid"]
            or device.get("assigned_gpu_name") != assigned["gpu_name"]
            or device.get("cuda_visible_devices") != assigned["gpu_uuid"]
            or device.get("torch_cuda_device_count") != 1
            or device.get("logical_cuda_index") != 0
            or device.get("resolved_gpu_name") != assigned["gpu_name"]
            or device.get("resolved_gpu_uuid") not in (None, assigned["gpu_uuid"])
        ):
            raise PhaseMapError(
                f"learner {learner} resolved-device proof differs from allocation"
            )
        device_paths.append(path)

    return {
        "allocation_path": allocation_path,
        "allocation_sha256": sha256_file(allocation_path),
        "learner_gpu_uuid_bijection": {
            str(learner): row["gpu_uuid"]
            for learner, row in sorted(assignment_by_learner.items())
        },
        "device_paths": device_paths,
        "device_sha256": {
            str(learner): sha256_file(path)
            for learner, path in enumerate(device_paths)
        },
    }


def validate_eval(
    report_dir: Path,
    result_loss: float,
    expected_eval: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    summary = json.loads(
        (report_dir / "eval-provenance" / "eval_provenance.json").read_text()
    )
    for key in (
        "eval_file_sha256",
        "eval_rows_hash",
        "eval_packed_hash",
        "eval_example_ids_hash",
        "eval_token_ids_hash",
        "eval_row_count",
        "eval_supervised_token_count",
    ):
        if summary.get(key) != expected_eval.get(key):
            raise PhaseMapError(f"cell evaluation provenance mismatch: {key}")
    losses_path = report_dir / "per-example-loss" / "m4.jsonl"
    losses = read_jsonl(losses_path)
    frozen_sequences = read_jsonl(Path(expected_eval["_eval_sequences_path"]))
    if len(losses) != len(frozen_sequences):
        raise PhaseMapError("per-sequence loss count differs from frozen evaluation")
    identity_fields = (
        "sequence_index",
        "sequence_id",
        "input_ids_sha256",
        "labels_sha256",
        "attention_mask_sha256",
        "supervision_weights_sha256",
        "target_token_mask_sha256",
        "sequence_length",
    )
    total_loss = 0.0
    total_tokens = 0
    saw_nonfinite = False
    for index, (loss_row, frozen_row) in enumerate(
        zip(losses, frozen_sequences, strict=True)
    ):
        for field in identity_fields:
            if loss_row.get(field) != frozen_row.get(field):
                raise PhaseMapError(
                    f"per-sequence identity mismatch at {index}: {field}"
                )
        token_count = loss_row.get("token_count")
        if (
            not isinstance(token_count, int)
            or isinstance(token_count, bool)
            or token_count <= 0
            or token_count != frozen_row.get("supervised_token_count")
        ):
            raise PhaseMapError(f"per-sequence token count mismatch at {index}")
        loss_sum = float(loss_row["loss_sum"])
        loss_per_token = float(loss_row["loss_per_token"])
        if math.isfinite(loss_sum) and math.isfinite(loss_per_token):
            if not math.isclose(
                loss_per_token,
                loss_sum / token_count,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise PhaseMapError(
                    f"per-sequence loss arithmetic mismatch at {index}"
                )
        else:
            saw_nonfinite = True
        total_loss += loss_sum
        total_tokens += token_count
    if total_tokens != expected_eval["eval_supervised_token_count"]:
        raise PhaseMapError("per-sequence target-token total does not match freeze")
    aggregate = total_loss / max(total_tokens, 1)
    if math.isfinite(result_loss):
        if saw_nonfinite or not math.isclose(
            aggregate, result_loss, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise PhaseMapError("per-sequence losses do not reproduce aggregate endpoint")
    elif not saw_nonfinite and math.isfinite(aggregate):
        raise PhaseMapError("nonfinite endpoint lacks per-sequence divergence evidence")
    return summary, losses_path


def validate_preconfirmation_surface(
    attempt_dir: Path, command: Sequence[str]
) -> None:
    """Prove a pre-P3 cell cannot name or emit confirmation artifacts."""
    if any("audit" in token.casefold() for token in command):
        raise PhaseMapError("pre-P3 scientific argv names an audit surface")
    forbidden_paths = [
        path
        for path in attempt_dir.rglob("*")
        if "audit" in path.relative_to(attempt_dir).as_posix().casefold()
    ]
    if forbidden_paths:
        raise PhaseMapError("pre-P3 attempt emitted an audit-named artifact")

    def has_forbidden_key(value: Any) -> bool:
        if isinstance(value, dict):
            return any(
                "audit" in str(key).casefold() or has_forbidden_key(item)
                for key, item in value.items()
            )
        if isinstance(value, list):
            return any(has_forbidden_key(item) for item in value)
        if isinstance(value, str):
            return "audit" in value.casefold()
        return False

    for path in attempt_dir.rglob("*.json"):
        try:
            value = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise PhaseMapError(f"invalid JSON while checking quarantine: {path}") from exc
        if has_forbidden_key(value):
            raise PhaseMapError("pre-P3 attempt JSON emitted an audit field")


def uri_for(args: argparse.Namespace, path: Path) -> str:
    relative = path.relative_to(args.run_dir).as_posix()
    return args.artifact_uri.rstrip("/") + "/" + relative


def result_attempt(
    args: argparse.Namespace,
    cell: dict[str, Any],
    attempt_dir: Path,
    provider: dict[str, Any],
    provider_sha: str,
    common: dict[str, Any],
    expected_eval: dict[str, Any],
    started_at: str,
    retry_context: dict[str, Any] | None,
) -> dict[str, Any]:
    results_path = attempt_dir / "report" / "results.jsonl"
    rows = read_jsonl(results_path)
    arm = [row for row in rows if row.get("arm") == "m4"]
    if len(rows) != 1 or len(arm) != 1:
        raise PhaseMapError("phase-map compare output must contain exactly one live m4 arm")
    raw_loss = float(arm[0]["eval_loss"])
    status = "COMPLETED" if math.isfinite(raw_loss) else "DIVERGED"
    loss = raw_loss if status == "COMPLETED" else None
    loss_kind = (
        "endpoint_nll_per_target_token"
        if status == "COMPLETED"
        else "scientific_nonfinite_divergence"
    )
    tape = attempt_dir / "work" / "m4" / "tape.jsonl"
    observed_work = validate_tape(tape, cell, args)
    layout_sha, merge_modes = validate_layout(attempt_dir)
    gpu_evidence = (
        validate_gpu_uuid_bijection(attempt_dir)
        if args.require_distinct_learner_gpu_uuids
        else None
    )
    _summary, losses_path = validate_eval(
        attempt_dir / "report", raw_loss, expected_eval
    )
    command_path = attempt_dir / "command.json"
    command_hash = sha256_bytes(canonical_json(cell["command"]))
    if command_hash != cell["command_hash"]:
        raise PhaseMapError("executed command hash differs from frozen expected cell")
    retry_authorization = (
        None if retry_context is None else retry_context["retry_authorization"]
    )
    hardware = {
        "market": "spot",
        "provider": "gcp",
        "instance_type": provider["instance_type"],
        "region": provider["region"],
        "project": provider["project"],
        "zone": provider["zone"],
        "instance_name": provider["instance_name"],
        "instance_id": provider["instance_id"],
        "instance_numeric_id": provider["instance_id"],
        "boot_disk_name": provider.get("boot_disk_name"),
        "boot_disk_id": provider["boot_disk_id"],
        "boot_disk_numeric_id": provider["boot_disk_id"],
        "source_image_id": provider["source_image_id"],
        "source_image_numeric_id": provider["source_image_id"],
        "image_id": args.image_numeric_id,
        "provisioning_evidence_uri": common["provider_evidence_uri"],
        "provisioning_evidence_sha256": provider_sha,
    }
    if gpu_evidence is not None:
        hardware.update(
            {
                "provisioning_started_at": provider.get(
                    "provisioning_started_at"
                ),
                "provisioning_completed_at": provider.get(
                    "provisioning_completed_at"
                ),
                "nvidia_smi_inventory_uri": uri_for(
                    args, gpu_evidence["allocation_path"]
                ),
                "nvidia_smi_inventory_sha256": gpu_evidence[
                    "allocation_sha256"
                ],
                "learner_gpu_map_uri": uri_for(
                    args, gpu_evidence["allocation_path"]
                ),
                "learner_gpu_map_sha256": gpu_evidence["allocation_sha256"],
                "barrier_version_trace_uri": uri_for(args, tape),
                "barrier_version_trace_sha256": sha256_file(tape),
                "distinct_a100_gpu_uuid_count": 4,
                "learner_gpu_uuid_bijection": gpu_evidence[
                    "learner_gpu_uuid_bijection"
                ],
                "resolved_device_sha256": gpu_evidence["device_sha256"],
            }
        )
    return {
        "attempt_id": f"{cell['cell_id']}-attempt-{args.attempt}",
        "cell_id": cell["cell_id"],
        "attempt": args.attempt,
        "status": status,
        "evaluation_role": "development",
        "reason_code": (
            "completed_exact_work"
            if status == "COMPLETED"
            else "scientific_nonfinite_divergence"
        ),
        "failure_reason": (
            None
            if status == "COMPLETED"
            else "non-finite endpoint under frozen scientific command"
        ),
        "loss": loss,
        "raw_loss": raw_loss if math.isfinite(raw_loss) else None,
        "loss_kind": loss_kind,
        "h": cell["H"],
        "mu": cell["mu"],
        "eta": cell["eta"],
        "seed": cell["seed"],
        "training_seed": cell["training_seed"],
        "paired_control_id": cell["paired_control_id"],
        "resource_class": cell["resource_class"],
        "work": {
            "fixed_window_microsteps": cell["H"],
            "fixed_window_tokens": cell["H"] * args.seq_len,
            "outer_steps": cell["target_work"]["outer_steps"],
            "token_budget": cell["target_work"]["tokens"],
            "eval_rows": args.eval_rows,
        },
        "observed_work": observed_work,
        "started_at": started_at,
        "ended_at": utc_now(),
        "retry_of": None if retry_context is None else retry_context["retry_of"],
        "retry_reason": (
            None if retry_context is None else retry_context["retry_reason"]
        ),
        "retry_authorization": retry_authorization,
        "git_commit": common["git_commit"],
        "image_digest": common["image_digest"],
        "model_hash": common["model_hash"],
        "data_hash": common["data_hash"],
        "eval_source_indices_hash": common["eval_source_indices_hash"],
        "train_pool_source_indices_hash": common[
            "train_pool_source_indices_hash"
        ],
        "train_source_indices_hash": common["train_source_indices_hash"],
        "train_rows_hash": common["train_rows_hash"],
        "eval_rows_hash": common["eval_rows_hash"],
        "eval_hash": common["eval_hash"],
        "eval_example_ids_hash": common["eval_example_ids_hash"],
        "eval_token_ids_hash": common["eval_token_ids_hash"],
        "command_hash": command_hash,
        "normalized_workload_command_hash": sha256_bytes(
            canonical_json(normalized_workload_command(cell["command"]))
        ),
        "command_uri": uri_for(args, command_path),
        "command_sha256": sha256_file(command_path),
        "capture_uri": uri_for(args, tape),
        "capture_sha256": sha256_file(tape),
        "result_uri": uri_for(args, results_path),
        "result_sha256": sha256_file(results_path),
        "per_example_loss_uri": uri_for(args, losses_path),
        "per_example_loss_sha256": sha256_file(losses_path),
        "layout_uri": uri_for(
            args,
            attempt_dir / "work" / "m4" / "learner-0" / "resolved-layout.json",
        ),
        "layout_sha256": layout_sha,
        "resolved_merge_modes": merge_modes,
        "barrier": True,
        "version_matched": True,
        "matrix_merge": "rda",
        "strict_quorum": True,
        "delta_correction": "none",
        "injected_baseline": False,
        "spot": True,
        "block_id": cell["randomization"]["block_id"],
        "order_index": cell["randomization"]["within_block_index"],
        "global_order_index": cell["randomization"]["order_index"],
        "hardware": hardware,
    }


def infra_failure_attempt(
    args: argparse.Namespace,
    cell: dict[str, Any],
    attempt_dir: Path,
    provider: dict[str, Any],
    provider_sha: str,
    common: dict[str, Any],
    *,
    started_at: str,
    reason: str,
    retry_context: dict[str, Any] | None,
) -> dict[str, Any]:
    if reason not in DIRECT_INFRASTRUCTURE_FAILURE_REASONS:
        raise PhaseMapError("INFRA_FAILURE requires a frozen direct reason")
    sentinel = attempt_dir / "infra-failure.json"
    write_json(
        sentinel,
        {
            "schema": "yeto_phase_map_infra_failure_v1",
            "cell_id": cell["cell_id"],
            "attempt": args.attempt,
            "reason": reason,
            "recorded_at": utc_now(),
            "loss_inspected": False,
        },
    )
    sentinel_sha = sha256_file(sentinel)
    evidence_uri = common["provider_evidence_uri"]
    retry_authorization = (
        None if retry_context is None else retry_context["retry_authorization"]
    )
    return {
        "attempt_id": f"{cell['cell_id']}-attempt-{args.attempt}",
        "cell_id": cell["cell_id"],
        "h": cell["H"],
        "mu": cell["mu"],
        "eta": cell["eta"],
        "seed": cell["seed"],
        "training_seed": cell["training_seed"],
        "status": "INFRA_FAILURE",
        "evaluation_role": "development",
        "failure_reason": reason,
        "loss": None,
        "loss_kind": None,
        "git_commit": common["git_commit"],
        "image_digest": common["image_digest"],
        "model_hash": common["model_hash"],
        "data_hash": common["data_hash"],
        "eval_source_indices_hash": common["eval_source_indices_hash"],
        "train_pool_source_indices_hash": common[
            "train_pool_source_indices_hash"
        ],
        "train_source_indices_hash": common["train_source_indices_hash"],
        "train_rows_hash": common["train_rows_hash"],
        "eval_rows_hash": common["eval_rows_hash"],
        "eval_hash": common["eval_hash"],
        "eval_example_ids_hash": common["eval_example_ids_hash"],
        "eval_token_ids_hash": common["eval_token_ids_hash"],
        "command_hash": cell["command_hash"],
        "normalized_workload_command_hash": sha256_bytes(
            canonical_json(normalized_workload_command(cell["command"]))
        ),
        "capture_uri": uri_for(args, sentinel),
        "capture_sha256": sentinel_sha,
        "result_uri": uri_for(args, sentinel),
        "result_sha256": sentinel_sha,
        "per_example_loss_uri": uri_for(args, sentinel),
        "per_example_loss_sha256": sentinel_sha,
        "paired_control_id": cell["paired_control_id"],
        "barrier": True,
        "version_matched": True,
        "matrix_merge": "rda",
        "strict_quorum": True,
        "delta_correction": "none",
        "injected_baseline": False,
        "spot": True,
        "block_id": cell["randomization"]["block_id"],
        "order_index": cell["randomization"]["within_block_index"],
        "global_order_index": cell["randomization"]["order_index"],
        "attempt": args.attempt,
        "retry_of": None if retry_context is None else retry_context["retry_of"],
        "retry_reason": (
            None if retry_context is None else retry_context["retry_reason"]
        ),
        "retry_authorization": retry_authorization,
        "hardware": {
            "market": "spot",
            "provider": "gcp",
            "instance_type": provider["instance_type"],
            "region": provider["region"],
            "project": provider["project"],
            "zone": provider["zone"],
            "instance_name": provider["instance_name"],
            "instance_id": provider["instance_id"],
            "boot_disk_id": provider["boot_disk_id"],
            "source_image_id": provider["source_image_id"],
            "image_id": args.image_numeric_id,
            "provisioning_evidence_uri": evidence_uri,
            "provisioning_evidence_sha256": provider_sha,
        },
        "work": {
            "fixed_window_microsteps": cell["H"],
            "fixed_window_tokens": cell["H"] * args.seq_len,
            "outer_steps": cell["target_work"]["outer_steps"],
            "token_budget": cell["target_work"]["tokens"],
            "eval_rows": args.eval_rows,
        },
        "observed_work": {
            "tokens": 0,
            "microsteps": 0,
            "outer_steps": 0,
            "full_quorum": False,
            "fixed_window_exact": False,
            "version_matched_anchor_resolved": False,
        },
        "started_at": started_at,
        "ended_at": utc_now(),
    }


def scientific_failure_attempt(
    args: argparse.Namespace,
    cell: dict[str, Any],
    attempt_dir: Path,
    provider: dict[str, Any],
    provider_sha: str,
    common: dict[str, Any],
    *,
    started_at: str,
    reason: str,
    retry_context: dict[str, Any] | None,
) -> dict[str, Any]:
    """Record an unresolved scientific/process terminal without retry license."""
    if reason in DIRECT_INFRASTRUCTURE_FAILURE_REASONS:
        raise PhaseMapError("scientific FAILED reason may not grant infra retry")
    sentinel = attempt_dir / "scientific-failure.json"
    write_json(
        sentinel,
        {
            "schema": "yeto_phase_map_scientific_failure_v1",
            "cell_id": cell["cell_id"],
            "attempt": args.attempt,
            "reason": reason,
            "recorded_at": utc_now(),
            "retryable": False,
        },
    )
    digest = sha256_file(sentinel)
    retry_authorization = (
        None if retry_context is None else retry_context["retry_authorization"]
    )
    return {
        "attempt_id": f"{cell['cell_id']}-attempt-{args.attempt}",
        "cell_id": cell["cell_id"],
        "h": cell["H"],
        "mu": cell["mu"],
        "eta": cell["eta"],
        "seed": cell["seed"],
        "training_seed": cell["training_seed"],
        "status": "FAILED",
        "evaluation_role": "development",
        "failure_reason": reason,
        "loss": None,
        "loss_kind": None,
        "git_commit": common["git_commit"],
        "image_digest": common["image_digest"],
        "model_hash": common["model_hash"],
        "data_hash": common["data_hash"],
        "eval_source_indices_hash": common["eval_source_indices_hash"],
        "train_pool_source_indices_hash": common[
            "train_pool_source_indices_hash"
        ],
        "train_source_indices_hash": common["train_source_indices_hash"],
        "train_rows_hash": common["train_rows_hash"],
        "eval_rows_hash": common["eval_rows_hash"],
        "eval_hash": common["eval_hash"],
        "eval_example_ids_hash": common["eval_example_ids_hash"],
        "eval_token_ids_hash": common["eval_token_ids_hash"],
        "command_hash": cell["command_hash"],
        "normalized_workload_command_hash": sha256_bytes(
            canonical_json(normalized_workload_command(cell["command"]))
        ),
        "capture_uri": uri_for(args, sentinel),
        "capture_sha256": digest,
        "result_uri": uri_for(args, sentinel),
        "result_sha256": digest,
        "per_example_loss_uri": None,
        "per_example_loss_sha256": None,
        "paired_control_id": cell["paired_control_id"],
        "barrier": True,
        "version_matched": True,
        "matrix_merge": "rda",
        "strict_quorum": True,
        "delta_correction": "none",
        "injected_baseline": False,
        "spot": True,
        "block_id": cell["randomization"]["block_id"],
        "order_index": cell["randomization"]["within_block_index"],
        "global_order_index": cell["randomization"]["order_index"],
        "attempt": args.attempt,
        "retry_of": None if retry_context is None else retry_context["retry_of"],
        "retry_reason": (
            None if retry_context is None else retry_context["retry_reason"]
        ),
        "retry_authorization": retry_authorization,
        "hardware": {
            "market": "spot",
            "provider": "gcp",
            "instance_type": provider["instance_type"],
            "region": provider["region"],
            "project": provider["project"],
            "zone": provider["zone"],
            "instance_name": provider["instance_name"],
            "instance_id": provider["instance_id"],
            "boot_disk_id": provider["boot_disk_id"],
            "source_image_id": provider["source_image_id"],
            "image_id": args.image_numeric_id,
            "provisioning_evidence_uri": common["provider_evidence_uri"],
            "provisioning_evidence_sha256": provider_sha,
        },
        "work": {
            "fixed_window_microsteps": cell["H"],
            "fixed_window_tokens": cell["H"] * args.seq_len,
            "outer_steps": cell["target_work"]["outer_steps"],
            "token_budget": cell["target_work"]["tokens"],
            "eval_rows": args.eval_rows,
        },
        "observed_work": {
            "tokens": 0,
            "microsteps": 0,
            "outer_steps": 0,
            "full_quorum": False,
            "fixed_window_exact": False,
            "version_matched_anchor_resolved": False,
        },
        "started_at": started_at,
        "ended_at": utc_now(),
    }


def classify_unmarked_process_exit(attempt_dir: Path) -> str:
    """Fail closed: a child exit is never itself mechanical infra evidence."""
    lifecycle = attempt_dir / "report" / "acquisition-state.json"
    phase = None
    if lifecycle.is_file():
        try:
            phase = json.loads(lifecycle.read_text()).get("phase")
        except json.JSONDecodeError:
            phase = "invalid_lifecycle_marker"
    if phase in ("endpoint_started", "endpoint_recorded"):
        return "process_exit_after_scientific_endpoint_started"
    return "process_exit_without_mechanical_preoutcome_infra_evidence"


def result_validation_failure_is_retryable(exc: BaseException) -> bool:
    """Only a missing/unreadable acquisition artifact is mechanical infra."""
    return isinstance(exc, OSError)


def build_retry_contexts(
    args: argparse.Namespace,
    selected: list[dict[str, Any]],
    prior_manifest: dict[str, Any],
    prior_manifest_hash: str,
) -> dict[str, dict[str, Any]]:
    if len({cell["randomization"]["block_id"] for cell in selected}) != 1:
        raise PhaseMapError("one retry invocation must contain exactly one whole block")
    if {float(cell["mu"]) for cell in selected} != set(args.mu):
        raise PhaseMapError("retry selection does not contain every frozen mu arm")
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prior_manifest.get("results", []):
        by_cell[str(row.get("cell_id"))].append(row)
    previous: dict[str, dict[str, Any]] = {}
    for cell in selected:
        attempts = sorted(
            by_cell.get(cell["cell_id"], []), key=lambda row: row.get("attempt", 0)
        )
        if not attempts or attempts[-1].get("attempt") != args.attempt - 1:
            raise PhaseMapError(
                "retry selection is not the immediate successor for "
                f"{cell['cell_id']}"
            )
        previous[cell["cell_id"]] = attempts[-1]
    trigger_matches = [
        row
        for row in previous.values()
        if row.get("attempt_id") == args.retry_trigger_attempt_id
    ]
    if len(trigger_matches) != 1:
        raise PhaseMapError("retry trigger is not in the immediate prior block")
    trigger = trigger_matches[0]
    if (
        trigger.get("status") != "INFRA_FAILURE"
        or trigger.get("failure_reason")
        not in DIRECT_INFRASTRUCTURE_FAILURE_REASONS
    ):
        raise PhaseMapError("retry trigger is not a genuine direct INFRA_FAILURE")
    block_id = selected[0]["randomization"]["block_id"]
    if trigger.get("block_id") != block_id:
        raise PhaseMapError("retry trigger block does not match selected block")
    if any(
        row.get("status") not in ("COMPLETED", "INFRA_FAILURE")
        for row in previous.values()
    ):
        raise PhaseMapError(
            "whole-block retry supports only completed peers and direct infra failures"
        )
    if any(
        row.get("status") == "INFRA_FAILURE"
        and row.get("failure_reason")
        not in DIRECT_INFRASTRUCTURE_FAILURE_REASONS
        for row in previous.values()
    ):
        raise PhaseMapError("prior block contains a non-direct INFRA_FAILURE reason")
    authorization = {
        "loss_blind": True,
        "policy_hash": prior_manifest["frozen"]["retry_policy_hash"],
        "trigger_attempt_id": trigger["attempt_id"],
        "trigger_reason": trigger["failure_reason"],
        "trigger_block_id": block_id,
        "prior_manifest_sha256": prior_manifest_hash,
        "authorized_at_utc": args.retry_authorized_at,
    }
    contexts = {}
    for cell in selected:
        row = previous[cell["cell_id"]]
        contexts[cell["cell_id"]] = {
            "retry_of": row["attempt_id"],
            "retry_reason": (
                PEER_BLOCK_RETRY_REASON
                if row["status"] == "COMPLETED"
                else row["failure_reason"]
            ),
            "retry_authorization": deepcopy(authorization),
        }
    return contexts


def acquisition_paths(run_dir: Path, manifest: dict[str, Any]) -> list[Path]:
    is_canary = manifest.get("lineage", {}).get("descendant_kind") in (
        "p0a_single_gpu_bound",
        "p0b_four_gpu_bound",
    )
    paths = [
        run_dir / "randomization-plan.json",
        run_dir / "expected-manifest.json",
        run_dir
        / (
            "phase-map-acquisition-manifest.json"
            if is_canary
            else "phase-map-manifest.json"
        ),
        run_dir / "acquisition-seal.json",
        run_dir / "provider-evidence.json",
    ]
    frozen_eval = run_dir / "frozen-eval"
    if frozen_eval.is_dir():
        paths.extend(
            path
            for path in frozen_eval.rglob("*")
            if path.is_file()
            and "audit" not in path.relative_to(frozen_eval).as_posix().casefold()
        )
    provider_dir = run_dir / "provider-evidence"
    if provider_dir.is_dir():
        paths.extend(path for path in provider_dir.rglob("*") if path.is_file())
    for result in manifest["results"]:
        attempt = run_dir / "cells" / result["cell_id"] / f"attempt-{result['attempt']}"
        if not attempt.exists():
            continue
        paths.extend(
            [
                attempt / "command.json",
                attempt / "command.sh",
                attempt / "attempt-start.json",
                attempt / "compare.log",
            ]
        )
        if result["status"] == "COMPLETED" or result["status"] == "DIVERGED":
            paths.extend(
                [
                    attempt / "report" / "results.jsonl",
                    attempt / "report" / "acquisition-state.json",
                    attempt / "report" / "per-example-loss" / "m4.jsonl",
                    attempt / "report" / "eval-provenance" / "eval_rows.jsonl",
                    attempt
                    / "report"
                    / "eval-provenance"
                    / "eval_sequences.jsonl",
                    attempt
                    / "report"
                    / "eval-provenance"
                    / "eval_provenance.json",
                    attempt / "work" / "m4" / "tape.jsonl",
                    attempt / "work" / "m4" / "state.ckpt",
                    *[
                        attempt
                        / "work"
                        / "m4"
                        / f"learner-{learner}"
                        / "resolved-layout.json"
                        for learner in range(4)
                    ],
                ]
            )
            if manifest["lineage"]["descendant_kind"] == "p0b_four_gpu_bound":
                paths.append(attempt / "report" / "gpu-allocation.json")
                paths.extend(
                    attempt
                    / "work"
                    / "m4"
                    / f"learner-{learner}"
                    / "resolved-device.json"
                    for learner in range(4)
                )
            divergence = attempt / "report" / "scientific-divergence.json"
            if divergence.is_file():
                paths.append(divergence)
            capture = attempt / "work" / "m4" / "syncer_probe"
            if capture.is_dir():
                paths.extend(path for path in capture.rglob("*") if path.is_file())
        elif result["status"] == "INFRA_FAILURE":
            paths.append(attempt / "infra-failure.json")
        else:
            paths.append(attempt / "scientific-failure.json")
    return paths


def write_seal(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    is_canary = manifest.get("lineage", {}).get("descendant_kind") in (
        "p0a_single_gpu_bound",
        "p0b_four_gpu_bound",
    )
    manifest["status"] = (
        "sealed_acquisition_pending_teardown" if is_canary else "sealed_results"
    )
    final_path = args.run_dir / "phase-map-manifest.json"
    write_json(final_path, manifest)
    if is_canary:
        write_json(args.run_dir / "phase-map-acquisition-manifest.json", manifest)
    canonical_manifest_hash = sha256_bytes(canonical_json(manifest))
    write_text(
        args.run_dir / "phase-map.sha256",
        f"{sha256_file(final_path)}  phase-map-manifest.json\n",
    )
    write_json(
        args.run_dir / "acquisition-seal.json",
        {
            "schema": "yeto_phase_map_acquisition_seal_v1",
            "sealed_at_utc": utc_now(),
            "phase_map_manifest_sha256": sha256_file(final_path),
            "phase_map_manifest_canonical_sha256": canonical_manifest_hash,
            "loss_blind_mechanical_seal": True,
        },
    )
    lines = []
    for path in sorted(set(acquisition_paths(args.run_dir, manifest))):
        if not path.is_file():
            raise PhaseMapError(f"cannot seal missing acquisition artifact: {path}")
        lines.append(f"{sha256_file(path)}  {path.relative_to(args.run_dir).as_posix()}")
    write_text(args.run_dir / "acquisition.sha256", "\n".join(lines) + "\n")


def execute(args: argparse.Namespace) -> int:
    plan = build_plan(args)
    if args.print_plan:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if not args.run_dir.is_absolute():
        raise PhaseMapError("--run-dir must be absolute")
    if args.phase == "materialize":
        verify_source_checkout(args)
        require_sha256(args.image_digest, "--image-digest")
        model_hash = require_sha256(args.expected_model_hash, "--expected-model-hash")
        data_hash = require_sha256(args.expected_data_hash, "--expected-data-hash")
        train_hash = require_sha256(
            args.expected_train_rows_hash, "--expected-train-rows-hash"
        )
        development_rows_hash = require_sha256(
            args.expected_development_eval_rows_hash,
            "--expected-development-eval-rows-hash",
        )
        development_packed_hash = require_sha256(
            args.expected_development_eval_packed_hash,
            "--expected-development-eval-packed-hash",
        )
        example_hash = require_sha256(
            args.expected_development_eval_example_ids_hash,
            "--expected-development-eval-example-ids-hash",
        )
        token_hash = require_sha256(
            args.expected_development_eval_token_ids_hash,
            "--expected-development-eval-token-ids-hash",
        )
        eval_indices_hash = require_sha256(
            args.expected_development_eval_source_indices_hash,
            "--expected-development-eval-source-indices-hash",
        )
        audit_rows_hash = require_sha256(
            args.expected_audit_eval_rows_hash,
            "--expected-audit-eval-rows-hash",
        )
        audit_packed_hash = require_sha256(
            args.expected_audit_eval_packed_hash,
            "--expected-audit-eval-packed-hash",
        )
        audit_example_hash = require_sha256(
            args.expected_audit_eval_example_ids_hash,
            "--expected-audit-eval-example-ids-hash",
        )
        audit_token_hash = require_sha256(
            args.expected_audit_eval_token_ids_hash,
            "--expected-audit-eval-token-ids-hash",
        )
        audit_indices_hash = require_sha256(
            args.expected_audit_eval_source_indices_hash,
            "--expected-audit-eval-source-indices-hash",
        )
        train_indices_hash = require_sha256(
            args.expected_train_source_indices_hash,
            "--expected-train-source-indices-hash",
        )
        train_pool_indices_hash = require_sha256(
            args.expected_train_pool_source_indices_hash,
            "--expected-train-pool-source-indices-hash",
        )
        bound = build_bound_manifest(
            args,
            plan,
            model_hash=model_hash,
            data_hash=data_hash,
            train_rows_hash=train_hash,
            development_eval_rows_hash=development_rows_hash,
            development_eval_packed_hash=development_packed_hash,
            development_eval_example_ids_hash=example_hash,
            development_eval_token_ids_hash=token_hash,
            development_eval_source_indices_hash=eval_indices_hash,
            audit_eval_rows_hash=audit_rows_hash,
            audit_eval_packed_hash=audit_packed_hash,
            audit_eval_example_ids_hash=audit_example_hash,
            audit_eval_token_ids_hash=audit_token_hash,
            audit_eval_source_indices_hash=audit_indices_hash,
            audit_access_policy_hash=sha256_bytes(
                canonical_json(verify_authoritative_prereg(args)["confirmation_policy"])
            ),
            train_pool_source_indices_hash=train_pool_indices_hash,
            train_source_indices_hash=train_indices_hash,
        )
        if args.run_dir.exists() and any(args.run_dir.iterdir()):
            raise PhaseMapError(
                f"materialization output directory is not empty: {args.run_dir}"
            )
        args.run_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.run_dir / "randomization-plan.json", plan)
        write_json(args.run_dir / "bound-manifest.json", bound)
        result = {
            "randomization_plan_hash": plan["randomization_plan_hash"],
            "bound_manifest_hash": sha256_bytes(canonical_json(bound)),
            "campaign_command_hash": bound["frozen"]["command_hash"],
            "cell_count": len(plan["cells"]),
        }
        write_json(args.run_dir / "materialization.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.attempt > 1:
        if not args.retry_authorized_at:
            raise PhaseMapError("retry attempt requires authorization time")
        if not args.retry_trigger_attempt_id:
            raise PhaseMapError("retry attempt requires a genuine trigger attempt id")
        if not args.prior_manifest or not args.expected_prior_manifest_hash:
            raise PhaseMapError("retry attempt requires the exact prior manifest and hash")
        if not args.prior_provider_evidence:
            raise PhaseMapError("retry attempt must retain prior provider evidence")
        if len(args.only_block) != 1:
            raise PhaseMapError("retry attempt must name exactly one complete block")
    elif (
        args.retry_authorized_at
        or args.retry_trigger_attempt_id
        or args.prior_manifest
        or args.expected_prior_manifest_hash
        or args.prior_provider_evidence
        or args.only_block
    ):
        raise PhaseMapError("initial attempt may not declare retry metadata")
    expected_plan_hash = require_sha256(
        args.expected_randomization_plan_hash,
        "--expected-randomization-plan-hash",
    )
    expected_bound_hash = require_sha256(
        args.expected_bound_manifest_hash,
        "--expected-bound-manifest-hash",
    )
    if plan["randomization_plan_hash"] != expected_plan_hash:
        raise PhaseMapError("runtime randomization plan differs from sealed plan hash")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    model_hash, data_hash = stage_and_verify_inputs(args)
    provider, provider_sha = load_provider_evidence(args)
    _provider_snapshot, provider_uri = snapshot_provider_evidence(
        args, provider, provider_sha
    )
    actual_commit = verify_source_checkout(args)

    expected_eval = prepare_eval_bundle(
        args, seed=args.seed, output_dir=args.run_dir / "frozen-eval" / f"seed-{args.seed}"
    )
    expected_pairs = {
        "development_eval_rows_hash": args.expected_development_eval_rows_hash,
        "development_eval_packed_hash": (
            args.expected_development_eval_packed_hash
        ),
        "development_eval_example_ids_hash": (
            args.expected_development_eval_example_ids_hash
        ),
        "development_eval_token_ids_hash": (
            args.expected_development_eval_token_ids_hash
        ),
        "development_eval_source_indices_hash": (
            args.expected_development_eval_source_indices_hash
        ),
        "audit_eval_rows_hash": args.expected_audit_eval_rows_hash,
        "audit_eval_packed_hash": args.expected_audit_eval_packed_hash,
        "audit_eval_example_ids_hash": args.expected_audit_eval_example_ids_hash,
        "audit_eval_token_ids_hash": args.expected_audit_eval_token_ids_hash,
        "audit_eval_source_indices_hash": (
            args.expected_audit_eval_source_indices_hash
        ),
        "train_pool_source_indices_hash": (
            args.expected_train_pool_source_indices_hash
        ),
        "train_source_indices_hash": args.expected_train_source_indices_hash,
        "train_file_sha256": args.expected_train_rows_hash,
    }
    if not args.require_frozen_eval:
        raise PhaseMapError("scientific execution requires --require-frozen-eval")
    if args.require_frozen_eval:
        for key, expected in expected_pairs.items():
            require_sha256(expected, f"frozen {key}")
            if expected_eval[key] != expected:
                raise PhaseMapError(f"runtime {key} differs from frozen value")
    for extra_seed in args.freeze_additional_eval_seed:
        prepare_eval_bundle(
            args,
            seed=extra_seed,
            output_dir=args.run_dir / "frozen-eval" / f"seed-{extra_seed}",
        )

    bound = build_bound_manifest(
        args,
        plan,
        model_hash=model_hash,
        data_hash=data_hash,
        train_rows_hash=expected_eval["train_file_sha256"],
        development_eval_rows_hash=expected_eval["development_eval_rows_hash"],
        development_eval_packed_hash=expected_eval[
            "development_eval_packed_hash"
        ],
        development_eval_example_ids_hash=expected_eval[
            "development_eval_example_ids_hash"
        ],
        development_eval_token_ids_hash=expected_eval[
            "development_eval_token_ids_hash"
        ],
        development_eval_source_indices_hash=expected_eval[
            "development_eval_source_indices_hash"
        ],
        audit_eval_rows_hash=expected_eval["audit_eval_rows_hash"],
        audit_eval_packed_hash=expected_eval["audit_eval_packed_hash"],
        audit_eval_example_ids_hash=expected_eval[
            "audit_eval_example_ids_hash"
        ],
        audit_eval_token_ids_hash=expected_eval["audit_eval_token_ids_hash"],
        audit_eval_source_indices_hash=expected_eval[
            "audit_eval_source_indices_hash"
        ],
        audit_access_policy_hash=expected_eval["audit_access_policy_hash"],
        train_pool_source_indices_hash=expected_eval[
            "train_pool_source_indices_hash"
        ],
        train_source_indices_hash=expected_eval["train_source_indices_hash"],
    )
    if sha256_bytes(canonical_json(bound)) != expected_bound_hash:
        raise PhaseMapError("runtime bound manifest differs from sealed manifest hash")
    common = {
        "git_commit": actual_commit,
        "image_digest": args.image_digest,
        "model_hash": model_hash,
        "data_hash": data_hash,
        "eval_source_indices_hash": expected_eval[
            "development_eval_source_indices_hash"
        ],
        "train_pool_source_indices_hash": expected_eval[
            "train_pool_source_indices_hash"
        ],
        "train_source_indices_hash": expected_eval["train_source_indices_hash"],
        "train_rows_hash": expected_eval["train_file_sha256"],
        "eval_rows_hash": expected_eval["development_eval_rows_hash"],
        "eval_hash": expected_eval["development_eval_packed_hash"],
        "eval_example_ids_hash": expected_eval[
            "development_eval_example_ids_hash"
        ],
        "eval_token_ids_hash": expected_eval[
            "development_eval_token_ids_hash"
        ],
        "randomization_plan_hash": plan["randomization_plan_hash"],
        "retry_policy_hash": bound["frozen"]["retry_policy_hash"],
        "provider_evidence_uri": provider_uri,
    }
    prior_hash = None
    if args.attempt == 1:
        manifest = bound
    else:
        prior = json.loads(args.prior_manifest.read_text())
        prior_hash = sha256_bytes(canonical_json(prior))
        if prior_hash != require_sha256(
            args.expected_prior_manifest_hash,
            "--expected-prior-manifest-hash",
        ):
            raise PhaseMapError("prior manifest differs from authorized canonical hash")
        prior_without_results = deepcopy(prior)
        prior_without_results["results"] = []
        prior_without_results["status"] = "bound_launch_authority"
        if canonical_json(prior_without_results) != canonical_json(bound):
            raise PhaseMapError("prior manifest frozen design differs from bound manifest")
        manifest = prior
        retain_prior_provider_evidence(args, prior)
    write_json(args.run_dir / "randomization-plan.json", plan)
    write_json(args.run_dir / "expected-manifest.json", bound)
    write_json(args.run_dir / "phase-map-manifest.partial.json", manifest)

    only_blocks = set(args.only_block)
    selected = [
        cell
        for cell in plan["cells"]
        if not only_blocks or cell["randomization"]["block_id"] in only_blocks
    ]
    if only_blocks and {c["randomization"]["block_id"] for c in selected} != only_blocks:
        raise PhaseMapError("--only-block names an unknown randomized block")
    by_block: dict[str, list[dict[str, Any]]] = {}
    for cell in selected:
        by_block.setdefault(cell["randomization"]["block_id"], []).append(cell)
    retry_contexts: dict[str, dict[str, Any]] = {}
    if args.attempt > 1:
        assert prior_hash is not None
        retry_contexts = build_retry_contexts(
            args, selected, manifest, prior_hash
        )

    had_infra_failure = False
    had_scientific_failure = False
    for block_cells in by_block.values():
        block_rows: list[dict[str, Any]] = []
        for cell in block_cells:
            attempt_dir = (
                args.run_dir / "cells" / cell["cell_id"] / f"attempt-{args.attempt}"
            )
            if attempt_dir.exists():
                raise PhaseMapError(f"attempt directory already exists: {attempt_dir}")
            attempt_dir.mkdir(parents=True)
            write_json(attempt_dir / "command.json", cell["command"])
            write_text(attempt_dir / "command.sh", shlex.join(cell["command"]) + "\n")
            started = utc_now()
            write_json(
                attempt_dir / "attempt-start.json",
                {
                    "attempt_id": f"{cell['cell_id']}-attempt-{args.attempt}",
                    "cell_id": cell["cell_id"],
                    "attempt": args.attempt,
                    "started_at": started,
                    "command_hash": cell["command_hash"],
                    "provider_evidence_sha256": provider_sha,
                },
            )
            log_path = attempt_dir / "compare.log"
            with log_path.open("w") as log:
                process = subprocess.run(
                    cell["command"],
                    cwd=attempt_dir,
                    text=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            validate_preconfirmation_surface(attempt_dir, cell["command"])
            if process.returncode:
                row = scientific_failure_attempt(
                    args,
                    cell,
                    attempt_dir,
                    provider,
                    provider_sha,
                    common,
                    started_at=started,
                    reason=classify_unmarked_process_exit(attempt_dir),
                    retry_context=retry_contexts.get(cell["cell_id"]),
                )
            else:
                try:
                    row = result_attempt(
                        args,
                        cell,
                        attempt_dir,
                        provider,
                        provider_sha,
                        common,
                        expected_eval,
                        started,
                        retry_contexts.get(cell["cell_id"]),
                    )
                except (OSError, ValueError, PhaseMapError) as exc:
                    if result_validation_failure_is_retryable(exc):
                        row = infra_failure_attempt(
                            args,
                            cell,
                            attempt_dir,
                            provider,
                            provider_sha,
                            common,
                            started_at=started,
                            reason="missing_or_checksum_invalid_required_artifact",
                            retry_context=retry_contexts.get(cell["cell_id"]),
                        )
                    else:
                        row = scientific_failure_attempt(
                            args,
                            cell,
                            attempt_dir,
                            provider,
                            provider_sha,
                            common,
                            started_at=started,
                            reason=(
                                "nonretryable_protocol_or_scientific_"
                                "validation_failure"
                            ),
                            retry_context=retry_contexts.get(cell["cell_id"]),
                        )
            if row["status"] == "INFRA_FAILURE":
                had_infra_failure = True
            if row["status"] == "FAILED":
                had_scientific_failure = True
            block_rows.append(row)
        # A randomized retry block is one append-only contiguous three-row suffix.
        manifest["results"].extend(block_rows)
        write_json(args.run_dir / "phase-map-manifest.partial.json", manifest)

    write_seal(args, manifest)
    if had_scientific_failure:
        raise PhaseMapError(
            "one or more cells ended FAILED without mechanical retry authority"
        )
    if had_infra_failure:
        raise PhaseMapError(
            "one or more complete randomized blocks ended in INFRA_FAILURE; "
            "retry requires a new append-only attempt"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--study-phase", required=True)
    parser.add_argument(
        "--phase", choices=("materialize", "execute"), default="execute"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--artifact-uri", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--command-repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--image-numeric-id", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-source-uri")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--expected-model-hash")
    parser.add_argument("--expected-data-hash")
    parser.add_argument("--expected-train-rows-hash")
    parser.add_argument(
        "--expected-development-eval-rows-hash",
        "--expected-eval-hash",
        dest="expected_development_eval_rows_hash",
    )
    parser.add_argument("--expected-development-eval-packed-hash")
    parser.add_argument(
        "--expected-development-eval-example-ids-hash",
        "--expected-eval-example-ids-hash",
        dest="expected_development_eval_example_ids_hash",
    )
    parser.add_argument(
        "--expected-development-eval-token-ids-hash",
        "--expected-eval-token-ids-hash",
        dest="expected_development_eval_token_ids_hash",
    )
    parser.add_argument(
        "--expected-development-eval-source-indices-hash",
        "--expected-eval-source-indices-hash",
        dest="expected_development_eval_source_indices_hash",
    )
    parser.add_argument(
        "--expected-audit-eval-rows-hash",
        "--expected-audit-eval-hash",
        dest="expected_audit_eval_rows_hash",
    )
    parser.add_argument("--expected-audit-eval-packed-hash")
    parser.add_argument("--expected-audit-eval-example-ids-hash")
    parser.add_argument("--expected-audit-eval-token-ids-hash")
    parser.add_argument("--expected-audit-eval-source-indices-hash")
    parser.add_argument("--expected-train-pool-source-indices-hash")
    parser.add_argument("--expected-train-source-indices-hash")
    parser.add_argument("--require-frozen-eval", action="store_true")
    parser.add_argument("--expected-randomization-plan-hash")
    parser.add_argument("--expected-bound-manifest-hash")
    parser.add_argument("--parent-manifest", type=Path)
    parser.add_argument("--expected-parent-manifest-hash")
    parser.add_argument(
        "--parent-replay-report",
        "--p0-replay-report",
        dest="parent_replay_report",
        type=Path,
    )
    parser.add_argument(
        "--expected-parent-replay-report-hash",
        "--expected-p0-replay-report-hash",
        dest="expected_parent_replay_report_hash",
    )
    parser.add_argument(
        "--prereg-template",
        type=Path,
        default=REPO_ROOT
        / "experiment-specs"
        / "best-paper-phase-map-p0-p1-prereg.json",
    )
    parser.add_argument("--provider-evidence", type=Path, required=True)
    parser.add_argument("--h", type=parse_ints, required=True)
    parser.add_argument("--mu", type=parse_floats, required=True)
    parser.add_argument("--eta", type=parse_floats, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--order-seed", type=int, required=True)
    parser.add_argument("--eval-split-seed", type=int, default=331)
    parser.add_argument("--freeze-additional-eval-seed", type=int, action="append", default=[])
    parser.add_argument("--token-budget", type=int, default=655_360)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--inner-lr", type=float, default=0.001)
    parser.add_argument("--train-rows", type=int, default=5000)
    parser.add_argument("--eval-rows", type=int, default=1024)
    parser.add_argument("--confirmation-audit-rows", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-slots", type=int, default=4)
    parser.add_argument("--learner-max-steps", type=int, default=1500)
    parser.add_argument("--syncer-checkpoint-every", type=int, default=4)
    parser.add_argument("--arm-timeout-min", type=int, default=240)
    parser.add_argument("--resource-class", required=True)
    parser.add_argument("--minimum-confirmatory-seeds", type=int, default=8)
    parser.add_argument("--divergence-loss-cap", type=float, default=10.0)
    parser.add_argument("--bracketing-tolerance", type=float, default=0.0)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--retry-authorized-at")
    parser.add_argument("--retry-trigger-attempt-id")
    parser.add_argument("--prior-manifest", type=Path)
    parser.add_argument("--expected-prior-manifest-hash")
    parser.add_argument(
        "--prior-provider-evidence", type=Path, action="append", default=[]
    )
    parser.add_argument("--only-block", action="append", default=[])
    parser.add_argument("--capture-every-step", action="store_true")
    parser.add_argument(
        "--require-distinct-learner-gpu-uuids", action="store_true"
    )
    parser.add_argument("--print-plan", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return execute(build_parser().parse_args(argv))
    except (OSError, ValueError, PhaseMapError) as exc:
        print(f"phase-map error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
