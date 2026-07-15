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
ADOPTED_PARALLEL_AMENDMENT_PATH = Path("docs/AMENDMENT-parallel-cells.md")
ADOPTED_PARALLEL_AMENDMENT_SHA256 = (
    "e2c87fd6c2ec0e4b91f488b5771334e0befd175560a3e2ccfcf349be1ee8b3dd"
)
P0A_SOURCE_REBIND_FROM_COMMIT = "0af7f4a80426babc14896c7c1f7885abcb331d46"
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


class WorkEvidenceError(PhaseMapError):
    """A cell lacks positive evidence for a completed training workload."""

    def __init__(self, message: str, *, cell_id: str | None = None):
        super().__init__(message)
        self.cell_id = cell_id


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
    seed_pairs = stage_seed_pairs(args)
    if args.study_phase in ("p0a_canary", "p0b_canary"):
        stage_name = "p0a" if args.study_phase == "p0a_canary" else "p0b"
        stage = template["canary_stages"][stage_name]
        if seed_pairs != [(stage["shuffle_seed"], stage["training_seed"])]:
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
            or seed_pairs
            != [
                (
                    grid["seeds"][0],
                    template["seed_pairs"][str(grid["seeds"][0])],
                )
            ]
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
    if args.study_phase in {"p1_adaptive", "p1_adaptive_bracket"}:
        if (
            seed_pairs != [(347, 347347)]
            or args.token_budget != protocol["token_budget"]
        ):
            raise PhaseMapError("adaptive P1 must remain on seed 347 and frozen work")
        if args.gpu_slots != 4 or args.resource_class != "a2-highgpu-4g":
            raise PhaseMapError(
                "adaptive P1 requires Spot a2-highgpu-4g and gpu_slots=4"
            )
        if not args.parent_manifest or not args.expected_parent_manifest_hash:
            raise PhaseMapError("adaptive P1 requires the exact sealed P1 parent")
        if args.parent_replay_report or args.expected_parent_replay_report_hash:
            raise PhaseMapError("adaptive P1 cites its sealed P1 parent, not a replay")
        if args.require_distinct_learner_gpu_uuids:
            raise PhaseMapError(
                "adaptive P1 may not inherit the non-evidence UUID canary flag"
            )
        return "adaptive_bracket_round"
    if getattr(args, "study_phase", None) == "p2_additional_development":
        required_seeds = template["stage_seeds"]["p2_additional_development"]
        required_pairs = [(seed, int(f"{seed}{seed}")) for seed in required_seeds]
        if (
            sorted(args.h) != template["expected_grid"]["h"]
            or seed_pairs != required_pairs
            or args.token_budget != protocol["token_budget"]
        ):
            raise PhaseMapError("P2 grid, work, or two registered seed pairs differ")
        if args.gpu_slots != 4 or args.resource_class != "a2-highgpu-4g":
            raise PhaseMapError("P2 requires Spot a2-highgpu-4g and gpu_slots=4")
        if not args.parent_manifest or not args.expected_parent_manifest_hash:
            raise PhaseMapError("P2 requires the exact sealed passing P1 parent")
        if args.parent_replay_report or args.expected_parent_replay_report_hash:
            raise PhaseMapError("P2 cites its sealed P1 parent, not a canary replay")
        if args.require_distinct_learner_gpu_uuids:
            raise PhaseMapError("P2 may not inherit the non-evidence UUID canary flag")
        return "additional_development_stage"
    raise PhaseMapError(
        "this runner supports authority-bound P0, initial P1-R0, and registered "
        "P1 adaptive/P2 cumulative development; P3 requires separate "
        "train-then-audit tooling"
    )


def stage_seed_pairs(args: argparse.Namespace) -> list[tuple[int, int]]:
    """Return the ordered seed pairs declared by one stage invocation."""
    extra_seeds = list(getattr(args, "additional_seed", []))
    extra_training = list(getattr(args, "additional_training_seed", []))
    if len(extra_seeds) != len(extra_training):
        raise PhaseMapError(
            "--additional-seed and --additional-training-seed counts must match"
        )
    pairs = [(args.seed, args.training_seed), *zip(extra_seeds, extra_training)]
    if len({seed for seed, _training in pairs}) != len(pairs):
        raise PhaseMapError("stage seed pairs contain a duplicate shuffle seed")
    return pairs


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


def authorize_p0b_source_rebind(
    parent: dict[str, Any], candidate_commit: str
) -> bool:
    """Allow the adopted one-time P0a -> fixed-production source transition."""
    parent_commit = str((parent.get("frozen") or {}).get("git_commit", ""))
    if parent_commit == candidate_commit:
        return False
    if (
        parent.get("lineage", {}).get("descendant_kind") != "p0a_single_gpu_bound"
        or parent_commit != P0A_SOURCE_REBIND_FROM_COMMIT
        or not re.fullmatch(r"[0-9a-f]{40}", candidate_commit)
    ):
        raise PhaseMapError("P0b source rebind is not the adopted P0a transition")
    for commit in (parent_commit, candidate_commit):
        available = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "cat-file", "-e", f"{commit}^{{commit}}"],
            check=False,
            capture_output=True,
        )
        if available.returncode != 0:
            raise PhaseMapError("P0b source rebind lacks an exact commit object")
    ancestry = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "merge-base",
            "--is-ancestor",
            parent_commit,
            candidate_commit,
        ],
        check=False,
        capture_output=True,
    )
    if ancestry.returncode != 0:
        raise PhaseMapError("P0b fixed production commit is not a P0a descendant")
    amendment = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "show",
            f"{candidate_commit}:{ADOPTED_PARALLEL_AMENDMENT_PATH.as_posix()}",
        ],
        check=False,
        capture_output=True,
    )
    if (
        amendment.returncode != 0
        or hashlib.sha256(amendment.stdout).hexdigest()
        != ADOPTED_PARALLEL_AMENDMENT_SHA256
    ):
        raise PhaseMapError("P0b source rebind lacks the exact adopted amendment")
    return True


def final_result_rows(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return one highest-attempt result per expected cell, fail closed."""
    expected = manifest.get("expected_cells")
    results = manifest.get("results")
    if not isinstance(expected, list) or not expected or not isinstance(results, list):
        raise PhaseMapError("parent lacks expected_cells or retained result rows")
    expected_ids = []
    for cell in expected:
        if not isinstance(cell, dict) or not isinstance(cell.get("cell_id"), str):
            raise PhaseMapError("parent expected_cells contains a malformed cell")
        expected_ids.append(cell["cell_id"])
    if len(set(expected_ids)) != len(expected_ids):
        raise PhaseMapError("parent expected_cells contains duplicate cell IDs")
    final: dict[str, dict[str, Any]] = {}
    for row in results:
        if not isinstance(row, dict) or not isinstance(row.get("cell_id"), str):
            raise PhaseMapError("parent results contains a malformed row")
        attempt = row.get("attempt")
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt <= 0:
            raise PhaseMapError("parent result lacks a positive integer attempt")
        cell = row["cell_id"]
        if cell not in expected_ids:
            raise PhaseMapError("parent contains an unexpected result cell")
        if cell not in final or attempt > int(final[cell]["attempt"]):
            final[cell] = row
    if set(final) != set(expected_ids):
        raise PhaseMapError("parent does not resolve every expected cell")
    return final


def verify_parent_hash_chain(manifests: Sequence[dict[str, Any]]) -> None:
    """Verify an oldest-to-newest canonical parent-hash chain.

    Cumulative P1/P2 hops additionally preserve parent expected cells and result
    rows as exact ordered prefixes.  That makes row mutation and reordering
    detectable even when every manifest is otherwise valid JSON.
    """
    if len(manifests) < 2:
        raise PhaseMapError("parent hash chain requires at least two manifests")
    cumulative_kinds = {"adaptive_bracket_round", "additional_development_stage"}
    for index, (parent, child) in enumerate(zip(manifests, manifests[1:]), 1):
        if parent.get("status") != "sealed_results":
            raise PhaseMapError(f"lineage hop {index} parent is not sealed_results")
        lineage = child.get("lineage")
        if not isinstance(lineage, dict):
            raise PhaseMapError(f"lineage hop {index} lacks lineage metadata")
        expected_hash = sha256_bytes(canonical_json(parent))
        if lineage.get("parent_manifest_sha256") != expected_hash:
            raise PhaseMapError(f"lineage hop {index} parent canonical hash mismatch")
        if lineage.get("descendant_kind") in cumulative_kinds:
            parent_cells = parent.get("expected_cells")
            child_cells = child.get("expected_cells")
            parent_rows = parent.get("results")
            child_rows = child.get("results")
            if (
                not isinstance(parent_cells, list)
                or not isinstance(child_cells, list)
                or child_cells[: len(parent_cells)] != parent_cells
            ):
                raise PhaseMapError(
                    f"lineage hop {index} mutates or reorders parent expected cells"
                )
            if (
                not isinstance(parent_rows, list)
                or not isinstance(child_rows, list)
                or child_rows[: len(parent_rows)] != parent_rows
            ):
                raise PhaseMapError(
                    f"lineage hop {index} mutates or reorders parent result rows"
                )


def p1_selected_eta_neighborhoods(
    parent: dict[str, Any], template: dict[str, Any]
) -> dict[tuple[int, float], tuple[float, float, float]]:
    """Derive the registered P1 selected eta and immediate bracket neighbors."""
    validate_campaign_work_evidence(parent)
    try:
        from scripts.validate_phase_map import ManifestError, validate_and_summarize

        validate_and_summarize(
            parent,
            claim_level="development",
            require_bracketed=True,
            _skip_authority_for_tests=True,
        )
    except (ImportError, ManifestError) as exc:
        raise PhaseMapError(f"P1 parent fails complete development validation: {exc}") from exc

    cells = {
        str(cell["cell_id"]): cell
        for cell in parent["expected_cells"]
        if isinstance(cell, dict) and isinstance(cell.get("cell_id"), str)
    }
    final = final_result_rows(parent)
    curves: dict[tuple[int, float], list[tuple[float, float]]] = defaultdict(list)
    for cell_id, cell in cells.items():
        seed = int(cell.get("seed", -1))
        if seed != 347:
            raise PhaseMapError("P1 parent contains a non-development seed before P2")
        row = final[cell_id]
        loss = row.get("loss")
        if (
            row.get("status") != "COMPLETED"
            or isinstance(loss, bool)
            or not isinstance(loss, (int, float))
            or not math.isfinite(float(loss))
        ):
            raise PhaseMapError("P1 parent contains a non-completed or nonfinite cell")
        curves[(int(cell["h"]), float(cell["mu"]))].append(
            (float(cell["eta"]), float(loss))
        )

    required_curves = {
        (int(h), float(mu))
        for h in template["expected_grid"]["h"]
        for mu in template["expected_grid"]["mu"]
    }
    if set(curves) != required_curves:
        raise PhaseMapError("P1 parent does not contain exactly nine LR curves")
    neighborhoods: dict[tuple[int, float], tuple[float, float, float]] = {}
    tuned: dict[tuple[int, float], float] = {}
    for key, points in curves.items():
        points.sort(key=lambda pair: pair[0])
        etas = [eta for eta, _loss in points]
        if len(etas) != len(set(etas)):
            raise PhaseMapError(f"P1 curve {key} repeats an eta")
        selected_eta, selected_loss = min(points, key=lambda pair: (pair[1], pair[0]))
        selected_index = etas.index(selected_eta)
        if selected_index == 0 or selected_index == len(etas) - 1:
            raise PhaseMapError(f"P1 curve {key} is not bracketed")
        lower_loss = points[selected_index - 1][1]
        upper_loss = points[selected_index + 1][1]
        if not (lower_loss > selected_loss and upper_loss > selected_loss):
            raise PhaseMapError(f"P1 curve {key} lacks two worse immediate neighbors")
        neighborhoods[key] = (
            etas[selected_index - 1],
            selected_eta,
            etas[selected_index + 1],
        )
        tuned[key] = selected_loss

    d_short = tuned[(16, 0.9)] - tuned[(16, 0.0)]
    d_long = min(tuned[(256, 0.5)], tuned[(256, 0.9)]) - tuned[(256, 0.0)]
    gate = template["go_kill"]
    if d_short < float(gate["d_short_min"]) or d_long > float(gate["d_long_max"]):
        raise PhaseMapError(
            f"P1 parent fails registered P2 gate: D_short={d_short}, D_long={d_long}"
        )
    return neighborhoods


def adaptive_eta_blocks(
    parent: dict[str, Any], template: dict[str, Any]
) -> list[tuple[int, float]]:
    """Derive exactly the next registered Section 6 P1 block suffix."""
    validate_campaign_work_evidence(parent)
    try:
        from scripts.validate_phase_map import ManifestError, validate_and_summarize

        validate_and_summarize(
            parent,
            claim_level="development",
            _skip_authority_for_tests=True,
        )
    except (ImportError, ManifestError) as exc:
        raise PhaseMapError(f"adaptive P1 parent fails validation: {exc}") from exc
    if parent.get("lineage", {}).get("descendant_kind") not in {
        "initial_bound_p1_r0",
        "adaptive_bracket_round",
    }:
        raise PhaseMapError("adaptive P1 parent must be an initial/adaptive P1 manifest")

    final = final_result_rows(parent)
    cells = {
        str(cell["cell_id"]): cell
        for cell in parent["expected_cells"]
        if isinstance(cell, dict) and isinstance(cell.get("cell_id"), str)
    }
    curves: dict[tuple[int, float], list[tuple[float, float]]] = defaultdict(list)
    for cell_id, cell in cells.items():
        if int(cell.get("seed", -1)) != 347:
            raise PhaseMapError("adaptive P1 parent contains a later seed")
        row = final[cell_id]
        loss = row.get("loss")
        if (
            row.get("status") != "COMPLETED"
            or isinstance(loss, bool)
            or not isinstance(loss, (int, float))
            or not math.isfinite(float(loss))
        ):
            raise PhaseMapError("adaptive P1 parent has unresolved/nonfinite work")
        curves[(int(cell["h"]), float(cell["mu"]))].append(
            (float(cell["eta"]), float(loss))
        )

    adaptive = template["adaptive_bracket"]
    initial = [float(value) for value in adaptive["initial_eta"]]
    downward = [float(value) for value in adaptive["downward_extension"]]
    upward = [float(value) for value in adaptive["upward_extension"]]
    boundary_lattice = {*initial, *downward, *upward}
    by_h: dict[int, set[float]] = defaultdict(set)
    for (h, _mu), points in curves.items():
        points.sort(key=lambda pair: pair[0])
        etas = [eta for eta, _loss in points]
        best_eta, _best_loss = min(points, key=lambda pair: (pair[1], pair[0]))
        best_index = etas.index(best_eta)
        candidates: list[float] = []
        if best_index == 0:
            chain = [min(initial), *downward]
            for index, eta in enumerate(chain[:-1]):
                if math.isclose(best_eta, eta, rel_tol=0.0, abs_tol=1e-15):
                    candidates = [chain[index + 1]]
                    break
        elif best_index == len(etas) - 1:
            chain = [max(initial), *upward]
            for index, eta in enumerate(chain[:-1]):
                if math.isclose(best_eta, eta, rel_tol=0.0, abs_tol=1e-15):
                    candidates = [chain[index + 1]]
                    break
        elif not any(
            not any(
                math.isclose(eta, base, rel_tol=0.0, abs_tol=1e-15)
                for base in boundary_lattice
            )
            for eta in etas
        ):
            candidates = [
                math.sqrt(etas[best_index - 1] * best_eta),
                math.sqrt(best_eta * etas[best_index + 1]),
            ]
        for eta in candidates:
            if not any(
                math.isclose(eta, sampled, rel_tol=0.0, abs_tol=1e-15)
                for sampled in etas
            ):
                by_h[h].add(eta)
    blocks = [(h, eta) for h in sorted(by_h) for eta in sorted(by_h[h])]
    if not blocks:
        raise PhaseMapError("sealed P1 parent authorizes no further adaptive eta block")
    return blocks


def p2_eta_blocks(
    parent: dict[str, Any], template: dict[str, Any]
) -> list[tuple[int, float]]:
    neighborhoods = p1_selected_eta_neighborhoods(parent, template)
    by_h: dict[int, set[float]] = defaultdict(set)
    for (h, _mu), etas in neighborhoods.items():
        by_h[h].update(etas)
    blocks = [(h, eta) for h in sorted(by_h) for eta in sorted(by_h[h])]
    if not 9 <= len(blocks) <= 27:
        raise PhaseMapError("registered P2 eta-block union must contain 9..27 blocks")
    return blocks


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
    if args.parent_manifest is None:
        raise PhaseMapError(f"{descendant_kind} requires a parent manifest path")

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
    if not isinstance(lineage, dict):
        raise PhaseMapError("parent lacks registered lineage metadata")
    if required_parent_kind and lineage.get("descendant_kind") != required_parent_kind:
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
    final_parent_rows = final_result_rows(parent)
    if any(row.get("status") != "COMPLETED" for row in final_parent_rows.values()):
        raise PhaseMapError("parent does not contain completed finite work for every cell")
    validate_campaign_work_evidence(parent)
    if required_parent_kind in ("p0a_single_gpu_bound", "p0b_four_gpu_bound") and (
        not isinstance(parent_cells, list)
        or len(parent_cells) != 3
        or {float(cell.get("mu")) for cell in parent_cells} != {0.0, 0.5, 0.9}
    ):
        raise PhaseMapError("canary parent does not contain one resolved full-mu block")
    if descendant_kind == "additional_development_stage":
        if lineage.get("descendant_kind") not in {
            "initial_bound_p1_r0",
            "adaptive_bracket_round",
        }:
            raise PhaseMapError("P2 parent must be a sealed P1 cumulative manifest")
        p2_eta_blocks(parent, template)
    if descendant_kind == "adaptive_bracket_round":
        if lineage.get("descendant_kind") not in {
            "initial_bound_p1_r0",
            "adaptive_bracket_round",
        }:
            raise PhaseMapError("adaptive parent must be a sealed P1 manifest")
        adaptive_eta_blocks(parent, template)

    if not policy.get("parent_replay_pass_report_required"):
        return parent, None
    if args.parent_replay_report is None:
        raise PhaseMapError(f"{descendant_kind} requires a parent replay report")

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
    replay_commit = replay.get("replay_validator_git_commit")
    parent_commit = parent["frozen"]["git_commit"]
    if replay_commit != parent_commit:
        source_rebind = bool(
            descendant_kind == "p0b_four_gpu_bound"
            and replay_commit == args.git_commit
            and authorize_p0b_source_rebind(parent, args.git_commit)
        )
        if not source_rebind:
            raise PhaseMapError("parent replay did not run from an authorized source")
        if (
            replay.get("replay_source_rebind_from_git_commit") != parent_commit
            or replay.get("replay_source_rebind_amendment_path")
            != ADOPTED_PARALLEL_AMENDMENT_PATH.as_posix()
            or replay.get("replay_source_rebind_amendment_sha256")
            != ADOPTED_PARALLEL_AMENDMENT_SHA256
        ):
            raise PhaseMapError("parent replay source-rebind attestation is incomplete")
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
    *,
    p0b_source_rebind: bool = False,
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
        if p0b_source_rebind and pointer == "/frozen/git_commit":
            continue
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


def parse_seed_sha256(value: str) -> tuple[int, str]:
    raw_seed, separator, digest = value.partition("=")
    try:
        seed = int(raw_seed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected SEED=SHA256") from exc
    if not separator or seed <= 0 or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise argparse.ArgumentTypeError("expected positive SEED=lowercase-64-hex-SHA256")
    return seed, digest


def seed_hash_map(
    values: Sequence[tuple[int, str]], label: str
) -> dict[int, str]:
    output: dict[int, str] = {}
    for seed, digest in values:
        if seed in output:
            raise PhaseMapError(f"{label} repeats seed {seed}")
        output[seed] = require_sha256(digest, label)
    return output


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


def exact_learner_max_steps(args: argparse.Namespace) -> int:
    """Derive the immutable per-learner physical-step ceiling.

    Four learners jointly consume the registered token budget.  The ceiling
    is derived here rather than accepted from a caller so receiver scheduling
    after the final broadcast cannot permit an extra local optimizer step.
    """
    denominator = 4 * args.micro_batch_size * args.seq_len
    if denominator <= 0:
        raise PhaseMapError("micro batch size and sequence length must be positive")
    steps, remainder = divmod(args.token_budget, denominator)
    if remainder or steps <= 0:
        raise PhaseMapError(
            "token budget must define an exact positive per-learner step cap"
        )
    return steps


def compare_command(
    args: argparse.Namespace,
    *,
    h: int,
    mu: float,
    eta: float,
    seed: int | None = None,
    training_seed: int | None = None,
) -> list[str]:
    seed = args.seed if seed is None else seed
    training_seed = args.training_seed if training_seed is None else training_seed
    outer_steps = args.token_budget // (h * args.seq_len)
    learner_max_steps = exact_learner_max_steps(args)
    frozen_split = args.run_dir / "frozen-eval" / f"seed-{seed}" / "materialized"
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
        str(seed),
        "--eval-split-seed",
        str(args.eval_split_seed),
        "--training-seed",
        str(training_seed),
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
        str(learner_max_steps),
        "--strict-quorum",
        "--pipeline-depth",
        "4",
        "--wan-streams",
        "0",
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
    learner_steps = exact_learner_max_steps(args)
    seed_pairs = stage_seed_pairs(args)
    study_phase = getattr(args, "study_phase", None)
    if study_phase in {"p1_adaptive", "p1_adaptive_bracket"}:
        template = verify_authoritative_prereg(args)
        parent = load_json_object(args.parent_manifest, "parent phase-map manifest")
        expected_parent_hash = require_sha256(
            args.expected_parent_manifest_hash,
            "--expected-parent-manifest-hash",
        )
        if sha256_bytes(canonical_json(parent)) != expected_parent_hash:
            raise PhaseMapError("parent canonical hash differs from frozen authority")
        eta_blocks = adaptive_eta_blocks(parent, template)
        if sorted(set(args.h)) != sorted({h for h, _eta in eta_blocks}):
            raise PhaseMapError("adaptive P1 --h must exactly declare the next blocks")
        if sorted(set(args.eta)) != sorted({eta for _h, eta in eta_blocks}):
            raise PhaseMapError("adaptive P1 --eta must exactly declare the next blocks")
        if seed_pairs != [(347, 347347)]:
            raise PhaseMapError("adaptive P1 may use only seed pair 347/347347")
        blocks = [(h, eta, 347, 347347) for h, eta in eta_blocks]
    elif study_phase == "p2_additional_development":
        template = verify_authoritative_prereg(args)
        parent = load_json_object(args.parent_manifest, "parent phase-map manifest")
        expected_parent_hash = require_sha256(
            args.expected_parent_manifest_hash,
            "--expected-parent-manifest-hash",
        )
        if sha256_bytes(canonical_json(parent)) != expected_parent_hash:
            raise PhaseMapError("parent canonical hash differs from frozen authority")
        eta_blocks = p2_eta_blocks(parent, template)
        declared_etas = sorted(set(args.eta))
        derived_etas = sorted({eta for _h, eta in eta_blocks})
        if declared_etas != derived_etas:
            raise PhaseMapError(
                "P2 --eta must exactly declare the sealed P1 selected-neighbor union"
            )
        blocks = [
            (h, eta, seed, training_seed)
            for seed, training_seed in seed_pairs
            for h, eta in eta_blocks
        ]
    else:
        if len(seed_pairs) != 1:
            raise PhaseMapError("only registered P2 may bind multiple stage seeds")
        seed, training_seed = seed_pairs[0]
        blocks = [
            (h, eta, seed, training_seed)
            for h in sorted(args.h)
            for eta in sorted(args.eta)
        ]
    rng = random.Random(args.order_seed)
    rng.shuffle(blocks)
    cells = []
    order_index = 0
    for block_index, (h, eta, seed, training_seed) in enumerate(blocks):
        outer_steps = args.token_budget // (h * args.seq_len)
        if args.token_budget % (h * args.seq_len):
            raise PhaseMapError(f"token budget is not exact for H={h}")
        if outer_steps % 4:
            raise PhaseMapError(f"outer step count must be divisible by fragments: H={h}")
        block_id = f"{args.study_id}-block-h{h}-eta{slug_float(eta)}-s{seed}"
        block_mu = list(args.mu)
        rng.shuffle(block_mu)
        control_id = cell_id(args.study_id, h, 0.0, eta, seed)
        for within_block_index, mu in enumerate(block_mu):
            command = compare_command(
                args,
                h=h,
                mu=mu,
                eta=eta,
                seed=seed,
                training_seed=training_seed,
            )
            cells.append(
                {
                    "cell_id": cell_id(args.study_id, h, mu, eta, seed),
                    "H": h,
                    "mu": mu,
                    "eta": eta,
                    "seed": seed,
                    "training_seed": training_seed,
                    "command_hash": sha256_bytes(canonical_json(command)),
                    "paired_control_id": control_id,
                    "resource_class": args.resource_class,
                    "target_work": {
                        "tokens": args.token_budget,
                        "microsteps": args.token_budget // args.seq_len,
                        "outer_steps": outer_steps,
                        "learner_count": 4,
                        "learner_steps_per_learner": learner_steps,
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
        "seed_pairs": {str(seed): training for seed, training in seed_pairs},
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
    additional_train_rows_hashes: dict[int, str] | None = None,
    additional_train_source_indices_hashes: dict[int, str] | None = None,
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
    p0b_source_rebind = bool(
        descendant_kind == "p0b_four_gpu_bound"
        and parent is not None
        and authorize_p0b_source_rebind(parent, args.git_commit)
    )
    cumulative = descendant_kind in {
        "adaptive_bracket_round",
        "additional_development_stage",
    }
    # P0b and cumulative descendants are constructed from the exact parent so
    # inherited fields cannot be accidentally regenerated or normalized.
    manifest = deepcopy(
        parent if descendant_kind == "p0b_four_gpu_bound" or cumulative else template
    )
    manifest["status"] = "bound_launch_authority"
    manifest["study_id"] = args.study_id
    manifest["mode"] = "development"
    manifest["min_confirmatory_seeds"] = args.minimum_confirmatory_seeds
    new_expected_cells = [
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
            "expected_learner_count": cell["target_work"]["learner_count"],
            "expected_learner_steps": cell["target_work"][
                "learner_steps_per_learner"
            ],
            "normalized_workload_command_hash": sha256_bytes(
                canonical_json(normalized_workload_command(cell["command"]))
            ),
        }
        for cell in plan["cells"]
    ]
    if cumulative:
        assert parent is not None
        parent_cells = parent.get("expected_cells")
        if not isinstance(parent_cells, list):
            raise PhaseMapError("cumulative parent expected_cells must be an array")
        old_ids = {
            str(cell.get("cell_id")) for cell in parent_cells if isinstance(cell, dict)
        }
        new_ids = {cell["cell_id"] for cell in new_expected_cells}
        if old_ids & new_ids:
            raise PhaseMapError("cumulative descendant repeats a parent cell ID")
        manifest["expected_cells"] = deepcopy(parent_cells) + new_expected_cells
        manifest["seed_pairs"].update(
            {str(seed): training for seed, training in stage_seed_pairs(args)}
        )
        all_coordinates = manifest["expected_cells"]
        manifest["expected_grid"] = {
            "h": sorted({int(cell["h"]) for cell in all_coordinates}),
            "mu": sorted({float(cell["mu"]) for cell in all_coordinates}),
            "eta": sorted({float(cell["eta"]) for cell in all_coordinates}),
            "seeds": sorted({int(cell["seed"]) for cell in all_coordinates}),
        }
    else:
        manifest["expected_grid"] = {
            "h": sorted(args.h),
            "mu": sorted(args.mu),
            "eta": sorted(args.eta),
            "seeds": [args.seed],
        }
        manifest["seed_pairs"] = {str(args.seed): args.training_seed}
        manifest["expected_cells"] = new_expected_cells

    supplied_train_rows = {args.seed: train_rows_hash}
    supplied_train_rows.update(additional_train_rows_hashes or {})
    supplied_train_indices = {args.seed: train_source_indices_hash}
    supplied_train_indices.update(additional_train_source_indices_hashes or {})
    required_new_seeds = {cell["seed"] for cell in plan["cells"]}
    if set(supplied_train_rows) != required_new_seeds:
        raise PhaseMapError("train-row hashes must cover exactly the new stage seeds")
    if set(supplied_train_indices) != required_new_seeds:
        raise PhaseMapError("train-index hashes must cover exactly the new stage seeds")
    train_rows_by_seed = (
        deepcopy(parent["frozen"]["train_rows_hashes"]) if cumulative else {}
    )
    train_indices_by_seed = (
        deepcopy(parent["frozen"]["train_source_indices_hashes"])
        if cumulative
        else {}
    )
    train_rows_by_seed.update(
        {str(seed): digest for seed, digest in supplied_train_rows.items()}
    )
    train_indices_by_seed.update(
        {str(seed): digest for seed, digest in supplied_train_indices.items()}
    )
    cell_command_hashes = (
        deepcopy(parent["frozen"]["cell_command_hashes"]) if cumulative else {}
    )
    cell_command_hashes.update(
        {cell["cell_id"]: cell["command_hash"] for cell in plan["cells"]}
    )
    command_hash = sha256_bytes(
        canonical_json(
            [
                {"cell_id": cell["cell_id"], "command_hash": cell_command_hashes[cell["cell_id"]]}
                for cell in manifest["expected_cells"]
            ]
        )
    )
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
            "train_source_indices_hashes": train_indices_by_seed,
            "train_rows_hashes": train_rows_by_seed,
            "command_hash": command_hash,
            "cell_command_hashes": cell_command_hashes,
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
            "descendant_kind": descendant_kind,
        }
    )
    if descendant_kind == "p0a_single_gpu_bound":
        manifest["lineage"]["parent_replay_report_sha256"] = None
    elif descendant_kind in {"p0b_four_gpu_bound", "initial_bound_p1_r0"}:
        manifest["lineage"]["parent_replay_report_sha256"] = require_sha256(
            args.expected_parent_replay_report_hash,
            "--expected-parent-replay-report-hash",
        )
    manifest["results"] = deepcopy(parent["results"]) if cumulative else []
    comparison_baseline = (
        parent
        if descendant_kind == "p0b_four_gpu_bound" or cumulative
        else None
    )
    if p0b_source_rebind:
        assert parent is not None
        comparison_baseline = deepcopy(parent)
        comparison_baseline["frozen"]["git_commit"] = args.git_commit
    validate_authorized_template_diff(
        template,
        manifest,
        descendant_kind,
        baseline=comparison_baseline,
    )
    validate_parent_equality(
        template,
        manifest,
        parent,
        descendant_kind,
        p0b_source_rebind=p0b_source_rebind,
    )
    if cumulative:
        assert parent is not None
        verify_parent_hash_chain([parent, manifest])
        parent_registry = parent["frozen"]["cell_command_hashes"]
        if any(
            cell_command_hashes.get(cell_id) != digest
            for cell_id, digest in parent_registry.items()
        ):
            raise PhaseMapError("cumulative descendant mutates a parent command hash")
        new_blocks: dict[tuple[int, float, int], set[float]] = defaultdict(set)
        for cell in new_expected_cells:
            new_blocks[(cell["h"], float(cell["eta"]), cell["seed"])].add(
                float(cell["mu"])
            )
        required_mu = set(template["expected_grid"]["mu"])
        if not new_blocks or any(mu != required_mu for mu in new_blocks.values()):
            raise PhaseMapError(
                "every new cumulative eta point must be one complete three-mu block"
            )
        if descendant_kind == "additional_development_stage":
            expected_coordinates = {
                (h, mu, eta, seed)
                for seed, _training in stage_seed_pairs(args)
                for h, eta in p2_eta_blocks(parent, template)
                for mu in required_mu
            }
            actual_coordinates = {
                (
                    int(cell["h"]),
                    float(cell["mu"]),
                    float(cell["eta"]),
                    int(cell["seed"]),
                )
                for cell in new_expected_cells
            }
            if actual_coordinates != expected_coordinates:
                raise PhaseMapError(
                    "P2 cells must exactly equal the sealed P1 selected-neighbor "
                    "blocks on both registered seeds"
                )
        if descendant_kind == "adaptive_bracket_round":
            expected_coordinates = {
                (h, mu, eta, 347)
                for h, eta in adaptive_eta_blocks(parent, template)
                for mu in required_mu
            }
            actual_coordinates = {
                (
                    int(cell["h"]),
                    float(cell["mu"]),
                    float(cell["eta"]),
                    int(cell["seed"]),
                )
                for cell in new_expected_cells
            }
            if actual_coordinates != expected_coordinates:
                raise PhaseMapError(
                    "adaptive P1 cells must exactly equal the next registered "
                    "boundary/midpoint block suffix"
                )
    return manifest


def build_schema_fixture(
    manifest: dict[str, Any], plan: dict[str, Any]
) -> dict[str, Any]:
    """Populate deterministic fake completed rows for schema integration tests."""
    fixture = deepcopy(manifest)
    frozen = fixture["frozen"]
    protocol = fixture["protocol"]
    rows = deepcopy(fixture.get("results", []))
    new_rows: list[dict[str, Any]] = []
    for cell in plan["cells"]:
        artifact_sha = sha256_bytes(cell["cell_id"].encode())
        h = cell["H"]
        eta = cell["eta"]
        # An interior synthetic optimum keeps validator summaries deterministic.
        loss = 2.0 + 0.01 * math.log2(eta / 0.04375) ** 2 + 0.02 * cell["mu"]
        new_rows.append(
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
                    "learner_count": cell["target_work"]["learner_count"],
                    "learner_steps_per_learner": cell["target_work"][
                        "learner_steps_per_learner"
                    ],
                },
                "observed_work": {
                    "tokens": protocol["token_budget"],
                    "microsteps": protocol["token_budget"] // protocol["seq_len"],
                    "outer_steps": cell["target_work"]["outer_steps"],
                    "per_fragment_outer_steps": {
                        fragment: cell["target_work"]["outer_steps"] // 4
                        for fragment in range(4)
                    },
                    "full_quorum": True,
                    "fixed_window_exact": True,
                    "version_matched_anchor_resolved": True,
                    "learner_step_counts": {
                        str(learner): cell["target_work"][
                            "learner_steps_per_learner"
                        ]
                        for learner in range(cell["target_work"]["learner_count"])
                    },
                },
                "exit_statuses": {
                    "runner": 0,
                    "syncer": 0,
                    "learners": [0] * cell["target_work"]["learner_count"],
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
    rows.extend(new_rows)
    if fixture["lineage"]["descendant_kind"] in (
        "p0a_single_gpu_bound",
        "p0b_four_gpu_bound",
    ):
        for row in new_rows:
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
        for row in new_rows:
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
                    + "/barrier-version-trace.json",
                    "barrier_version_trace_sha256": evidence_sha,
                    "barrier_version_trace_canonical_sha256": evidence_sha,
                    "barrier_trace_validated": True,
                    "base_versions_match": True,
                    "no_inner_step_while_blocked": True,
                    "barrier_trace_learner_count": 4,
                    "barrier_trace_commit_count": row["work"]["outer_steps"],
                    "barrier_trace_inner_steps_per_learner": (
                        row["work"]["outer_steps"] // 4
                        * row["work"]["fixed_window_microsteps"]
                    ),
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
            "per_fragment_outer_steps": {
                fragment: fixture["horizon_work"][str(row["h"])][
                    "outer_steps"
                ]
                // 4
                for fragment in range(4)
            },
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
    raw = args.provider_evidence.read_bytes()
    root_destination = args.run_dir / "provider-evidence.json"
    if root_destination.exists() and root_destination.read_bytes() != raw:
        raise PhaseMapError("phase-map root provider evidence already differs")
    if not root_destination.exists():
        temporary = root_destination.with_suffix(".json.tmp")
        temporary.write_bytes(raw)
        temporary.replace(root_destination)
    if root_destination.is_symlink() or sha256_file(root_destination) != digest:
        raise PhaseMapError("phase-map root provider evidence hash mismatch")
    destination = (
        args.run_dir / "provider-evidence" / f"instance-{instance_id}-{digest}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
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
    prior_fragment_commit = {fragment: 0 for fragment in range(4)}
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
        row_base_versions = {responder.get("base_version") for responder in responders}
        if row_base_versions != {prior_fragment_commit[fragment]}:
            raise PhaseMapError(
                f"step {index} responder bases do not equal fragment {fragment}'s "
                "immediately prior committed global step"
            )
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
        prior_fragment_commit[fragment] = index
    expected_per_fragment = expected_steps // 4
    if fragments != Counter({i: expected_per_fragment for i in range(4)}):
        raise PhaseMapError("outer commits are not balanced across four fragments")
    for key, versions in base_versions.items():
        expected_versions = [
            0,
            *[
                int(row["step"])
                for row in rows
                if row.get("fragment") == key[1]
            ][:-1],
        ]
        if versions != expected_versions:
            raise PhaseMapError(
                f"non-exact base-version progression for learner/fragment {key}"
            )
    observed = {
        "tokens": responder_tokens // 4,
        "microsteps": responder_microsteps // 4,
        "outer_steps": len(rows),
        "per_fragment_outer_steps": dict(sorted(fragments.items())),
        "full_quorum": True,
        "fixed_window_exact": True,
        "version_matched_anchor_resolved": True,
        "base_versions_match": True,
        "fragment_base_progression_exact": True,
    }
    for key in ("tokens", "microsteps", "outer_steps"):
        if observed[key] != cell["target_work"][key]:
            raise PhaseMapError(
                f"observed {key}={observed[key]} != target {cell['target_work'][key]}"
            )
    return observed


def validate_barrier_version_trace(
    attempt_dir: Path,
    tape_rows: list[dict[str, Any]],
    cell: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Validate the sealed learner-local barrier state machines against tape.

    Cross-machine clocks and event sequence numbers are deliberately never
    compared.  Each learner trace is checked in its own causal order, then its
    pushes and applied broadcasts are joined exactly to the syncer tape.
    """
    registry_path = attempt_dir / "report" / "barrier-version-trace.json"
    registry = load_json_object(registry_path, "barrier version trace registry")
    if (
        registry.get("schema") != "yeto_barrier_version_trace_v1"
        or registry.get("learner_count") != 4
    ):
        raise PhaseMapError("barrier trace registry has the wrong schema/count")

    def verify_entry(entry: Any, expected_relative: str, label: str) -> Path:
        if not isinstance(entry, dict) or entry.get("path") != expected_relative:
            raise PhaseMapError(f"{label} registry path differs from the exact artifact")
        path = attempt_dir / expected_relative
        if not path.is_file() or path.is_symlink():
            raise PhaseMapError(f"{label} registry artifact is missing or unsafe")
        if entry.get("sha256") != sha256_file(path):
            raise PhaseMapError(f"{label} registry hash mismatch")
        size = entry.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size != path.stat().st_size:
            raise PhaseMapError(f"{label} registry size mismatch")
        return path

    tape_path = verify_entry(
        registry.get("syncer_tape"), "work/m4/tape.jsonl", "syncer tape"
    )
    if read_jsonl(tape_path) != tape_rows:
        raise PhaseMapError("barrier registry tape differs from validated event tape")

    entries = registry.get("learner_traces")
    if not isinstance(entries, list) or len(entries) != 4:
        raise PhaseMapError("barrier registry must contain exactly four learner traces")
    by_learner = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise PhaseMapError("barrier learner registry entry is malformed")
        learner_id = entry.get("learner_id")
        if (
            isinstance(learner_id, bool)
            or not isinstance(learner_id, int)
            or learner_id not in range(4)
            or learner_id in by_learner
        ):
            raise PhaseMapError("barrier registry learner IDs must be exactly 0..3")
        by_learner[learner_id] = verify_entry(
            entry,
            f"work/m4/learner-{learner_id}/barrier-version-trace.jsonl",
            f"learner {learner_id} trace",
        )
    if set(by_learner) != {0, 1, 2, 3}:
        raise PhaseMapError("barrier registry learner IDs must be exactly 0..3")

    expected_pushes: dict[tuple[int, int], dict[str, int]] = {}
    for row in tape_rows:
        step = int(row["step"])
        fragment = int(row["fragment"])
        for responder in row["responders"]:
            learner_id = int(responder["id"])
            expected_pushes[(learner_id, step)] = {
                "fragment": fragment,
                "base_version": int(responder["base_version"]),
                "c_steps": int(responder["c_steps"]),
                "c_tokens": int(responder["c_tokens"]),
            }

    # Each H-step optimizer window advances all four fragments concurrently.
    # Thus the per-learner optimizer-step count is aggregate fragment work / 4.
    target_microsteps = int(cell["target_work"]["microsteps"])
    if target_microsteps % 4:
        raise PhaseMapError("P0 target microsteps are not divisible by four fragments")
    expected_inner_steps = target_microsteps // 4
    push_counts: dict[int, int] = {}
    broadcast_counts: dict[int, int] = {}
    inner_counts: dict[int, int] = {}
    for learner_id in range(4):
        rows = read_jsonl(by_learner[learner_id])
        awaiting: dict[int, tuple[int, int, int]] = {}
        reset_local_step = {fragment: 0 for fragment in range(4)}
        initial_fragments: set[int] = set()
        seen_pushes: set[int] = set()
        seen_broadcasts: set[int] = set()
        next_inner_step = 1
        previous_local_step = 0
        for sequence, row in enumerate(rows, 1):
            if (
                row.get("schema") != "yeto_barrier_trace_v1"
                or row.get("event_seq") != sequence
                or row.get("learner_id") != learner_id
            ):
                raise PhaseMapError(
                    f"learner {learner_id} barrier trace schema/sequence/identity mismatch"
                )
            local_step = row.get("local_step")
            if (
                isinstance(local_step, bool)
                or not isinstance(local_step, int)
                or local_step < previous_local_step
            ):
                raise PhaseMapError(
                    f"learner {learner_id} barrier trace local_step is not monotone"
                )
            previous_local_step = local_step
            declared_awaiting = row.get("awaiting_fragments")
            event = row.get("event")
            if event == "initial_broadcast_applied":
                fragment = row.get("fragment")
                initial_version = row.get("broadcast_version")
                if (
                    sequence not in range(1, 5)
                    or isinstance(fragment, bool)
                    or not isinstance(fragment, int)
                    or fragment != sequence - 1
                    or local_step != 0
                    or isinstance(initial_version, bool)
                    or not isinstance(initial_version, int)
                    or initial_version != 0
                    or declared_awaiting != []
                    or awaiting
                    or fragment in initial_fragments
                ):
                    raise PhaseMapError(
                        f"learner {learner_id} initial broadcast prefix is not exact"
                    )
                initial_fragments.add(fragment)
                continue
            if event == "inner_step_started":
                if (
                    initial_fragments != {0, 1, 2, 3}
                    or awaiting
                    or declared_awaiting != []
                    or local_step != next_inner_step
                ):
                    raise PhaseMapError(
                        f"learner {learner_id} starts an inner step while blocked or out of order"
                    )
                next_inner_step += 1
                continue
            if event == "push_sent":
                fragment = row.get("fragment")
                pull_step = row.get("pull_step")
                base_version = row.get("base_version")
                expected = expected_pushes.get((learner_id, pull_step))
                if (
                    isinstance(fragment, bool)
                    or not isinstance(fragment, int)
                    or fragment not in range(4)
                    or isinstance(pull_step, bool)
                    or not isinstance(pull_step, int)
                    or pull_step in seen_pushes
                    or fragment in awaiting
                    or initial_fragments != {0, 1, 2, 3}
                    or expected is None
                    or expected
                    != {
                        "fragment": fragment,
                        "base_version": base_version,
                        "c_steps": row.get("c_steps"),
                        "c_tokens": row.get("c_tokens"),
                    }
                    or local_step
                    != ((pull_step - 1) // 4 + 1) * int(cell["H"])
                    or local_step
                    != reset_local_step[fragment] + int(cell["H"])
                ):
                    raise PhaseMapError(
                        f"learner {learner_id} push does not biject to the event tape"
                    )
                awaiting[fragment] = (pull_step, int(base_version), local_step)
                seen_pushes.add(pull_step)
            elif event == "broadcast_applied":
                fragment = row.get("fragment")
                pending = awaiting.get(fragment)
                if pending is None:
                    raise PhaseMapError(
                        f"learner {learner_id} applies a broadcast without an outstanding push"
                    )
                pull_step, base_version, push_local_step = pending
                if (
                    row.get("pushed_base_version") != base_version
                    or row.get("broadcast_version") != pull_step
                    or pull_step <= base_version
                    or local_step != push_local_step
                    or pull_step in seen_broadcasts
                ):
                    raise PhaseMapError(
                        f"learner {learner_id} broadcast does not release the exact pushed round"
                    )
                del awaiting[fragment]
                reset_local_step[fragment] = local_step
                seen_broadcasts.add(pull_step)
            else:
                raise PhaseMapError(
                    f"learner {learner_id} barrier trace has unknown event {event!r}"
                )
            if declared_awaiting != sorted(awaiting):
                raise PhaseMapError(
                    f"learner {learner_id} declared awaiting set differs from causal state"
                )
        expected_steps = {
            step for (expected_learner, step) in expected_pushes if expected_learner == learner_id
        }
        if awaiting or seen_pushes != expected_steps or seen_broadcasts != expected_steps:
            raise PhaseMapError(
                f"learner {learner_id} barrier trace lacks exact push/broadcast coverage"
            )
        if initial_fragments != {0, 1, 2, 3}:
            raise PhaseMapError(
                f"learner {learner_id} lacks exact initial broadcast coverage"
            )
        if next_inner_step - 1 != expected_inner_steps:
            raise PhaseMapError(
                f"learner {learner_id} has {next_inner_step - 1} inner steps, "
                f"expected {expected_inner_steps}"
            )
        if set(reset_local_step.values()) != {expected_inner_steps}:
            raise PhaseMapError(
                f"learner {learner_id} fragment windows do not terminate at "
                f"local step {expected_inner_steps}"
            )
        push_counts[learner_id] = len(seen_pushes)
        broadcast_counts[learner_id] = len(seen_broadcasts)
        inner_counts[learner_id] = next_inner_step - 1

    return {
        "registry_path": registry_path,
        "registry_sha256": sha256_file(registry_path),
        "registry_canonical_sha256": sha256_bytes(canonical_json(registry)),
        "barrier_trace_validated": True,
        "base_versions_match": True,
        "no_inner_step_while_blocked": True,
        "learner_count": 4,
        "commit_count": len(tape_rows),
        "inner_steps_per_learner": expected_inner_steps,
        "push_counts": push_counts,
        "broadcast_counts": broadcast_counts,
        "inner_step_counts": inner_counts,
    }


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
            or token_count < 0
            or token_count != frozen_row.get("supervised_token_count")
        ):
            raise PhaseMapError(f"per-sequence token count mismatch at {index}")
        loss_sum = float(loss_row["loss_sum"])
        loss_per_token = float(loss_row["loss_per_token"])
        if token_count == 0:
            if (
                not math.isfinite(loss_sum)
                or not math.isfinite(loss_per_token)
                or loss_sum != 0.0
                or loss_per_token != 0.0
            ):
                raise PhaseMapError(
                    f"zero-target sequence has nonzero/nonfinite loss at {index}"
                )
        elif math.isfinite(loss_sum) and math.isfinite(loss_per_token):
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

    def is_saved_model_tokenizer(path: Path) -> bool:
        parts = path.relative_to(attempt_dir).parts
        return (
            len(parts) == 4
            and parts[:2] == ("work", "m4")
            and parts[-1] == "tokenizer.json"
            and (parts[2] == "export" or parts[2].startswith("learner-"))
        )

    for path in attempt_dir.rglob("*.json"):
        if is_saved_model_tokenizer(path):
            continue
        try:
            value = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise PhaseMapError(f"invalid JSON while checking quarantine: {path}") from exc
        if has_forbidden_key(value):
            raise PhaseMapError("pre-P3 attempt JSON emitted an audit field")


def uri_for(args: argparse.Namespace, path: Path) -> str:
    relative = path.relative_to(args.run_dir).as_posix()
    return args.artifact_uri.rstrip("/") + "/" + relative


def exit_statuses_are_zero(
    runner: Any, syncer: Any, learners: Any, expected_learners: int
) -> bool:
    def is_zero(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value == 0

    return (
        is_zero(runner)
        and is_zero(syncer)
        and isinstance(learners, list)
        and len(learners) == expected_learners
        and all(is_zero(value) for value in learners)
    )


def validate_cell_work_evidence(
    args: argparse.Namespace,
    cell: dict[str, Any],
    attempt_dir: Path,
    *,
    runner_exit_code: int,
) -> tuple[Path, float, dict[str, Any], dict[str, Any]]:
    """Require positive learner work, a finite endpoint, and clean exits."""
    results_path = attempt_dir / "report" / "results.jsonl"
    try:
        rows = read_jsonl(results_path)
        arm = [row for row in rows if row.get("arm") == "m4"]
        if len(rows) != 1 or len(arm) != 1:
            raise WorkEvidenceError(
                "phase-map compare output must contain exactly one live m4 arm"
            )
        result = arm[0]
        raw_loss = float(result["eval_loss"])
        expected_learners = int(cell["target_work"]["learner_count"])
        expected_steps = int(
            cell["target_work"]["learner_steps_per_learner"]
        )
        learner_exit_codes = result.get("learner_exit_codes")
        syncer_exit_code = result.get("syncer_exit_code")
        if not exit_statuses_are_zero(
            runner_exit_code,
            syncer_exit_code,
            learner_exit_codes,
            expected_learners,
        ):
            raise WorkEvidenceError(
                "runner, syncer, and every expected learner must exit zero"
            )
        if not math.isfinite(raw_loss):
            raise WorkEvidenceError("cell terminal loss is missing or non-finite")

        tape = attempt_dir / "work" / "m4" / "tape.jsonl"
        observed_work = validate_tape(tape, cell, args)
        barrier_evidence = validate_barrier_version_trace(
            attempt_dir, read_jsonl(tape), cell, args
        )
        expected_counts = {
            learner: expected_steps for learner in range(expected_learners)
        }
        if barrier_evidence.get("inner_step_counts") != expected_counts:
            raise WorkEvidenceError(
                "every learner must reach the frozen expected optimizer-step count"
            )
    except WorkEvidenceError:
        raise
    except (KeyError, OSError, TypeError, ValueError, PhaseMapError) as exc:
        raise WorkEvidenceError(f"incomplete cell work evidence: {exc}") from exc

    observed_work["learner_step_counts"] = {
        str(learner): count
        for learner, count in barrier_evidence["inner_step_counts"].items()
    }
    exit_statuses = {
        "runner": runner_exit_code,
        "syncer": syncer_exit_code,
        "learners": learner_exit_codes,
    }
    return results_path, raw_loss, observed_work, {
        "barrier": barrier_evidence,
        "exit_statuses": exit_statuses,
    }


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
    runner_exit_code: int,
) -> dict[str, Any]:
    results_path, raw_loss, observed_work, work_evidence = (
        validate_cell_work_evidence(
            args,
            cell,
            attempt_dir,
            runner_exit_code=runner_exit_code,
        )
    )
    tape = attempt_dir / "work" / "m4" / "tape.jsonl"
    barrier_evidence = work_evidence["barrier"]
    layout_sha, merge_modes = validate_layout(attempt_dir)
    gpu_evidence = (
        validate_gpu_uuid_bijection(attempt_dir)
        if args.require_distinct_learner_gpu_uuids
        else None
    )
    try:
        _summary, losses_path = validate_eval(
            attempt_dir / "report", raw_loss, expected_eval
        )
    except (KeyError, OSError, TypeError, ValueError, PhaseMapError) as exc:
        raise WorkEvidenceError(f"incomplete terminal-loss evidence: {exc}") from exc
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
        "barrier_version_trace_uri": uri_for(
            args, barrier_evidence["registry_path"]
        ),
        "barrier_version_trace_sha256": barrier_evidence["registry_sha256"],
        "barrier_version_trace_canonical_sha256": barrier_evidence[
            "registry_canonical_sha256"
        ],
        "barrier_trace_validated": True,
        "base_versions_match": True,
        "no_inner_step_while_blocked": True,
        "barrier_trace_learner_count": barrier_evidence["learner_count"],
        "barrier_trace_commit_count": barrier_evidence["commit_count"],
        "barrier_trace_inner_steps_per_learner": barrier_evidence[
            "inner_steps_per_learner"
        ],
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
        "status": "COMPLETED",
        "evaluation_role": "development",
        "reason_code": "completed_exact_work",
        "failure_reason": None,
        "loss": raw_loss,
        "raw_loss": raw_loss,
        "loss_kind": "endpoint_nll_per_target_token",
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
            "learner_count": cell["target_work"]["learner_count"],
            "learner_steps_per_learner": cell["target_work"][
                "learner_steps_per_learner"
            ],
        },
        "observed_work": observed_work,
        "exit_statuses": work_evidence["exit_statuses"],
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


def validate_campaign_work_evidence(manifest: dict[str, Any]) -> None:
    """Refuse a clean seal unless every final cell proves completed work."""
    expected_rows = manifest.get("expected_cells")
    result_rows = manifest.get("results")
    if not isinstance(expected_rows, list) or not isinstance(result_rows, list):
        raise WorkEvidenceError("campaign lacks expected cells or cell results")
    expected = {
        str(row.get("cell_id")): row
        for row in expected_rows
        if isinstance(row, dict) and row.get("cell_id")
    }
    final: dict[str, dict[str, Any]] = {}
    for row in result_rows:
        if not isinstance(row, dict) or not row.get("cell_id"):
            raise WorkEvidenceError("campaign contains a malformed cell result")
        cell_id = str(row["cell_id"])
        attempt = row.get("attempt")
        if isinstance(attempt, bool) or not isinstance(attempt, int):
            raise WorkEvidenceError(f"cell {cell_id} lacks a numeric attempt")
        if cell_id not in final or attempt > int(final[cell_id]["attempt"]):
            final[cell_id] = row
    if not expected or set(final) != set(expected):
        raise WorkEvidenceError("campaign lacks one final result per expected cell")

    for cell_id, coordinates in expected.items():
        row = final[cell_id]
        if row.get("status") != "COMPLETED":
            raise WorkEvidenceError(
                f"cell {cell_id} did not complete", cell_id=cell_id
            )
        loss = row.get("loss")
        if (
            isinstance(loss, bool)
            or not isinstance(loss, (int, float))
            or not math.isfinite(float(loss))
        ):
            raise WorkEvidenceError(
                f"cell {cell_id} lacks a finite terminal loss", cell_id=cell_id
            )

        expected_learners = coordinates.get("expected_learner_count")
        expected_steps = coordinates.get("expected_learner_steps")
        if (
            isinstance(expected_learners, bool)
            or not isinstance(expected_learners, int)
            or expected_learners <= 0
            or isinstance(expected_steps, bool)
            or not isinstance(expected_steps, int)
            or expected_steps <= 0
        ):
            raise WorkEvidenceError(
                f"cell {cell_id} lacks frozen learner work coordinates",
                cell_id=cell_id,
            )
        expected_counts = {
            str(learner): expected_steps for learner in range(expected_learners)
        }
        observed = row.get("observed_work")
        if not isinstance(observed, dict) or observed.get(
            "learner_step_counts"
        ) != expected_counts:
            raise WorkEvidenceError(
                f"cell {cell_id} learners did not reach the frozen step count",
                cell_id=cell_id,
            )
        exits = row.get("exit_statuses")
        if not isinstance(exits, dict) or not exit_statuses_are_zero(
            exits.get("runner"),
            exits.get("syncer"),
            exits.get("learners"),
            expected_learners,
        ):
            raise WorkEvidenceError(
                f"cell {cell_id} lacks zero runner/syncer/learner exits",
                cell_id=cell_id,
            )


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
                    attempt / "report" / "barrier-version-trace.json",
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
                    *[
                        attempt
                        / "work"
                        / "m4"
                        / f"learner-{learner}"
                        / "barrier-version-trace.jsonl"
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
    validate_campaign_work_evidence(manifest)
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


def finalize_campaign(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    try:
        validate_campaign_work_evidence(manifest)
    except WorkEvidenceError as exc:
        if exc.cell_id is not None:
            cell_rows = [
                row
                for row in manifest.get("results", [])
                if isinstance(row, dict) and str(row.get("cell_id")) == exc.cell_id
            ]
            if cell_rows:
                row = max(cell_rows, key=lambda item: int(item.get("attempt", 0)))
                if row.get("status") == "COMPLETED":
                    row["status"] = "FAILED"
                    row["reason_code"] = "work_evidence_gate_failed"
                    row["failure_reason"] = str(exc)
                    row["loss"] = None
                    row["loss_kind"] = None
        manifest["status"] = "FAILED"
        write_json(args.run_dir / "phase-map-manifest.partial.json", manifest)
        raise
    write_seal(args, manifest)


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
        additional_train_hashes = seed_hash_map(
            args.expected_additional_train_rows_hash,
            "--expected-additional-train-rows-hash",
        )
        additional_train_indices_hashes = seed_hash_map(
            args.expected_additional_train_source_indices_hash,
            "--expected-additional-train-source-indices-hash",
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
            additional_train_rows_hashes=additional_train_hashes,
            additional_train_source_indices_hashes=(
                additional_train_indices_hashes
            ),
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

    stage_pairs = stage_seed_pairs(args)
    declared_train_rows = {args.seed: args.expected_train_rows_hash}
    declared_train_rows.update(
        seed_hash_map(
            args.expected_additional_train_rows_hash,
            "--expected-additional-train-rows-hash",
        )
    )
    declared_train_indices = {args.seed: args.expected_train_source_indices_hash}
    declared_train_indices.update(
        seed_hash_map(
            args.expected_additional_train_source_indices_hash,
            "--expected-additional-train-source-indices-hash",
        )
    )
    if set(declared_train_rows) != {seed for seed, _training in stage_pairs}:
        raise PhaseMapError("frozen train-row hashes do not cover the stage seeds")
    if set(declared_train_indices) != {seed for seed, _training in stage_pairs}:
        raise PhaseMapError("frozen train-index hashes do not cover the stage seeds")
    common_expected_pairs = {
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
    }
    if not args.require_frozen_eval:
        raise PhaseMapError("scientific execution requires --require-frozen-eval")
    expected_evals: dict[int, dict[str, Any]] = {}
    for seed, _training_seed in stage_pairs:
        expected_eval = prepare_eval_bundle(
            args,
            seed=seed,
            output_dir=args.run_dir / "frozen-eval" / f"seed-{seed}",
        )
        expected_evals[seed] = expected_eval
        expected_pairs = {
            **common_expected_pairs,
            "train_source_indices_hash": declared_train_indices[seed],
            "train_file_sha256": declared_train_rows[seed],
        }
        for key, expected in expected_pairs.items():
            require_sha256(expected, f"frozen {key}")
            if expected_eval[key] != expected:
                raise PhaseMapError(
                    f"runtime seed {seed} {key} differs from frozen value"
                )
    for extra_seed in args.freeze_additional_eval_seed:
        if extra_seed in expected_evals:
            continue
        prepare_eval_bundle(
            args,
            seed=extra_seed,
            output_dir=args.run_dir / "frozen-eval" / f"seed-{extra_seed}",
        )

    primary_eval = expected_evals[args.seed]
    additional_rows = {
        seed: expected_evals[seed]["train_file_sha256"]
        for seed, _training in stage_pairs
        if seed != args.seed
    }
    additional_indices = {
        seed: expected_evals[seed]["train_source_indices_hash"]
        for seed, _training in stage_pairs
        if seed != args.seed
    }
    bound = build_bound_manifest(
        args,
        plan,
        model_hash=model_hash,
        data_hash=data_hash,
        train_rows_hash=primary_eval["train_file_sha256"],
        development_eval_rows_hash=primary_eval["development_eval_rows_hash"],
        development_eval_packed_hash=primary_eval[
            "development_eval_packed_hash"
        ],
        development_eval_example_ids_hash=primary_eval[
            "development_eval_example_ids_hash"
        ],
        development_eval_token_ids_hash=primary_eval[
            "development_eval_token_ids_hash"
        ],
        development_eval_source_indices_hash=primary_eval[
            "development_eval_source_indices_hash"
        ],
        audit_eval_rows_hash=primary_eval["audit_eval_rows_hash"],
        audit_eval_packed_hash=primary_eval["audit_eval_packed_hash"],
        audit_eval_example_ids_hash=primary_eval[
            "audit_eval_example_ids_hash"
        ],
        audit_eval_token_ids_hash=primary_eval["audit_eval_token_ids_hash"],
        audit_eval_source_indices_hash=primary_eval[
            "audit_eval_source_indices_hash"
        ],
        audit_access_policy_hash=primary_eval["audit_access_policy_hash"],
        train_pool_source_indices_hash=primary_eval[
            "train_pool_source_indices_hash"
        ],
        train_source_indices_hash=primary_eval["train_source_indices_hash"],
        additional_train_rows_hashes=additional_rows,
        additional_train_source_indices_hashes=additional_indices,
    )
    if sha256_bytes(canonical_json(bound)) != expected_bound_hash:
        raise PhaseMapError("runtime bound manifest differs from sealed manifest hash")
    common_by_seed = {
        seed: {
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
            "train_source_indices_hash": expected_eval[
                "train_source_indices_hash"
            ],
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
        for seed, expected_eval in expected_evals.items()
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
        prior_without_results["results"] = deepcopy(bound["results"])
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

    for block_cells in by_block.values():
        block_rows: list[dict[str, Any]] = []
        for cell in block_cells:
            expected_eval = expected_evals[cell["seed"]]
            common = common_by_seed[cell["seed"]]
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
                        process.returncode,
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
            block_rows.append(row)
        # A randomized retry block is one append-only contiguous three-row suffix.
        manifest["results"].extend(block_rows)
        write_json(args.run_dir / "phase-map-manifest.partial.json", manifest)

    finalize_campaign(args, manifest)
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
    parser.add_argument(
        "--expected-additional-train-rows-hash",
        type=parse_seed_sha256,
        action="append",
        default=[],
        metavar="SEED=SHA256",
    )
    parser.add_argument(
        "--expected-additional-train-source-indices-hash",
        type=parse_seed_sha256,
        action="append",
        default=[],
        metavar="SEED=SHA256",
    )
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
    parser.add_argument("--additional-seed", type=int, action="append", default=[])
    parser.add_argument(
        "--additional-training-seed", type=int, action="append", default=[]
    )
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
