#!/usr/bin/env python3
"""Validate and summarize immutable phase-map result manifests.

The validator is intentionally stricter than a plotting script.  It treats a
complete paired training run as the experimental unit, retains every terminal
attempt, audits Spot/retry/provenance evidence, and will not turn a one-seed
development sweep into a confirmatory result.

The expected input is a bound descendant of
``experiment-specs/best-paper-phase-map-p0-p1-prereg.json`` with a ``results``
array added.  Adaptive LR extensions are separate manifests; mutating the
initial expected grid in place is not supported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "0.2"
HARD_MIN_CONFIRMATORY_SEEDS = 8
FINAL_STATUSES = {"COMPLETED", "DIVERGED", "FAILED", "INFRA_FAILURE"}
SCIENTIFIC_STATUSES = {"COMPLETED", "DIVERGED"}
SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")
GIT_RE = re.compile(r"^[0-9a-fA-F]{40,64}$")
REPO_ROOT = Path(__file__).resolve().parents[1]
AUTHORITATIVE_PREREG_PATH = (
    "experiment-specs/best-paper-phase-map-p0-p1-prereg.json"
)
AUTHORITATIVE_PREREG_SOURCE_COMMIT = "16d27bc60deb6d8910bf0111c7fb57c9d0eb5b80"
AUTHORITATIVE_PREREG_TEMPLATE_SHA256 = (
    "7cba3c62328b4bfe15fffbc523979274e834e8e720e16f70d79621eaf6ebdb7b"
)
ADOPTED_PARALLEL_AMENDMENT_PATH = "docs/AMENDMENT-parallel-cells.md"
ADOPTED_PARALLEL_AMENDMENT_SHA256 = (
    "e2c87fd6c2ec0e4b91f488b5771334e0befd175560a3e2ccfcf349be1ee8b3dd"
)
P0A_SOURCE_REBIND_FROM_COMMIT = "0af7f4a80426babc14896c7c1f7885abcb331d46"


class ManifestError(RuntimeError):
    """Raised when a result manifest is incomplete or scientifically invalid."""


@dataclass(frozen=True, order=True)
class Coordinate:
    h: int
    mu: float
    eta: float
    seed: int


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _canonical_status(value: Any) -> str:
    return str(value).strip().upper()


def _require_mapping(value: Any, label: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{label} must be an object")
        return {}
    return value


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256_canonical(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _run_git(*args: str) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        raise ManifestError(f"cannot execute git for preregistration authority: {exc}") from exc
    if result.returncode:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise ManifestError(
            f"cannot resolve authoritative preregistration with git {' '.join(args)}: "
            f"{detail}"
        )
    return result


def _authoritative_template() -> Mapping[str, Any]:
    raw = _run_git(
        "show",
        f"{AUTHORITATIVE_PREREG_SOURCE_COMMIT}:{AUTHORITATIVE_PREREG_PATH}",
    ).stdout
    digest = hashlib.sha256(raw).hexdigest()
    if digest != AUTHORITATIVE_PREREG_TEMPLATE_SHA256:
        raise ManifestError(
            "hard-pinned authoritative preregistration blob has an unexpected SHA-256"
        )
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"authoritative preregistration blob is invalid JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ManifestError("authoritative preregistration template must be an object")
    return value


def _json_pointer_escape(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _json_pointer_get(value: Any, pointer: str) -> Any:
    if pointer == "":
        return value
    if not pointer.startswith("/"):
        raise KeyError(pointer)
    current = value
    for raw_part in pointer[1:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping):
            current = current[part]
        elif isinstance(current, list):
            current = current[int(part)]
        else:
            raise KeyError(pointer)
    return current


def _json_differences(reference: Any, candidate: Any, pointer: str = "") -> set[str]:
    """Return the smallest object-field/list-root pointers that differ."""

    if isinstance(reference, Mapping) and isinstance(candidate, Mapping):
        differences: set[str] = set()
        for key in set(reference) | set(candidate):
            child = pointer + "/" + _json_pointer_escape(str(key))
            if key not in reference or key not in candidate:
                differences.add(child)
            else:
                differences.update(
                    _json_differences(reference[key], candidate[key], child)
                )
        return differences
    # Lists are intentionally atomic. An allowlist entry for a list field
    # licenses replacement of that exact list, not arbitrary neighboring data.
    if reference != candidate:
        return {pointer or "/"}
    return set()


def _path_is_allowed(path: str, allowed: set[str]) -> bool:
    return any(path == root or path.startswith(root + "/") for root in allowed)


def _manifest_kind(manifest: Mapping[str, Any]) -> str:
    lineage = manifest.get("lineage")
    if not isinstance(lineage, Mapping):
        return ""
    return str(lineage.get("descendant_kind", ""))


def _validate_p0b_source_rebind(
    manifest: Mapping[str, Any],
    parent: Mapping[str, Any],
    errors: list[str],
) -> bool:
    child_commit = str(
        _mapping_or_empty(manifest.get("frozen")).get("git_commit", "")
    )
    parent_commit = str(
        _mapping_or_empty(parent.get("frozen")).get("git_commit", "")
    )
    if child_commit == parent_commit:
        return False
    if (
        _manifest_kind(parent) != "p0a_single_gpu_bound"
        or parent_commit != P0A_SOURCE_REBIND_FROM_COMMIT
        or not re.fullmatch(r"[0-9a-f]{40}", child_commit)
    ):
        errors.append("P0b source rebind is not the adopted P0a transition")
        return False
    for commit in (parent_commit, child_commit):
        available = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "cat-file", "-e", f"{commit}^{{commit}}"],
            check=False,
            capture_output=True,
        )
        if available.returncode != 0:
            errors.append("P0b source rebind lacks an exact commit object")
            return False
    ancestry = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "merge-base",
            "--is-ancestor",
            parent_commit,
            child_commit,
        ],
        check=False,
        capture_output=True,
    )
    amendment = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "show",
            f"{child_commit}:{ADOPTED_PARALLEL_AMENDMENT_PATH}",
        ],
        check=False,
        capture_output=True,
    )
    if ancestry.returncode != 0:
        errors.append("P0b fixed production commit is not a P0a descendant")
        return False
    if (
        amendment.returncode != 0
        or hashlib.sha256(amendment.stdout).hexdigest()
        != ADOPTED_PARALLEL_AMENDMENT_SHA256
    ):
        errors.append("P0b source rebind lacks the exact adopted amendment")
        return False
    return True


def _expected_cell_records(
    manifest: Mapping[str, Any], errors: list[str]
) -> tuple[set[Coordinate], set[str]]:
    raw = manifest.get("expected_cells")
    if not isinstance(raw, list) or not raw:
        errors.append("expected_cells must be a non-empty authoritative coordinate list")
        return set(), set()
    coordinates: set[Coordinate] = set()
    cell_ids: set[str] = set()
    for index, item in enumerate(raw):
        row = _require_mapping(item, f"expected_cells[{index}]", errors)
        coordinate = _coordinate(row, f"expected_cells[{index}]", errors)
        if coordinate is not None:
            if coordinate in coordinates:
                errors.append(f"expected_cells contains duplicate coordinate {coordinate}")
            coordinates.add(coordinate)
        cell_id = row.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id.strip():
            errors.append(f"expected_cells[{index}].cell_id must be non-empty")
        elif cell_id in cell_ids:
            errors.append(f"expected_cells contains duplicate cell_id {cell_id!r}")
        else:
            cell_ids.add(cell_id)
    return coordinates, cell_ids


def _expected_cell_id_coordinates(
    manifest: Mapping[str, Any]
) -> dict[str, Coordinate]:
    output: dict[str, Coordinate] = {}
    raw = manifest.get("expected_cells")
    if not isinstance(raw, list):
        return output
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        cell_id = item.get("cell_id")
        errors: list[str] = []
        coordinate = _coordinate(item, "expected cell", errors)
        if isinstance(cell_id, str) and coordinate is not None and not errors:
            output[cell_id] = coordinate
    return output


def _validate_expected_cell_blocks(
    manifest: Mapping[str, Any], coordinates: set[Coordinate], errors: list[str]
) -> None:
    required_mu = {
        float(value)
        for value in _mapping_or_empty(manifest.get("randomization")).get(
            "required_mu_per_block", []
        )
    }
    blocks: dict[tuple[int, float, int], set[float]] = defaultdict(set)
    for coordinate in coordinates:
        blocks[(coordinate.h, coordinate.eta, coordinate.seed)].add(coordinate.mu)
    for key, observed in blocks.items():
        if observed != required_mu:
            errors.append(
                f"expected_cells block {key} has mu={sorted(observed)}, expected "
                f"the complete live-control block {sorted(required_mu)}"
            )


def _latest_parent_rows(
    parent: Mapping[str, Any], errors: list[str]
) -> dict[Coordinate, Mapping[str, Any]]:
    latest: dict[Coordinate, Mapping[str, Any]] = {}
    latest_attempt: dict[Coordinate, int] = {}
    rows = parent.get("results")
    if not isinstance(rows, list):
        errors.append("adaptive parent results must be an array")
        return latest
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            errors.append(f"adaptive parent results[{index}] must be an object")
            continue
        coordinate_errors: list[str] = []
        coordinate = _coordinate(raw, f"adaptive parent results[{index}]", coordinate_errors)
        errors.extend(coordinate_errors)
        attempt = raw.get("attempt")
        if coordinate is None or not _is_int(attempt):
            continue
        if attempt > latest_attempt.get(coordinate, -1):
            latest_attempt[coordinate] = attempt
            latest[coordinate] = raw
    return latest


def _same_float(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-15)


def _allowed_adaptive_etas(
    parent: Mapping[str, Any], manifest: Mapping[str, Any], errors: list[str]
) -> dict[int, set[float]]:
    """Return next registered eta values, keyed by H, from sealed parent losses."""

    latest = _latest_parent_rows(parent, errors)
    grouped: dict[tuple[int, float, int], list[tuple[float, float]]] = defaultdict(list)
    for coordinate, row in latest.items():
        status = _canonical_status(row.get("status"))
        if status == "COMPLETED" and _finite_number(row.get("loss")):
            loss = float(row["loss"])
        elif status == "DIVERGED":
            loss = math.inf
        else:
            errors.append(
                f"adaptive parent curve contains unresolved {status} at {coordinate}"
            )
            continue
        grouped[(coordinate.h, coordinate.mu, coordinate.seed)].append(
            (coordinate.eta, loss)
        )

    adaptive = _mapping_or_empty(manifest.get("adaptive_bracket"))
    initial = [float(value) for value in adaptive.get("initial_eta", [])]
    down = [float(value) for value in adaptive.get("downward_extension", [])]
    up = [float(value) for value in adaptive.get("upward_extension", [])]
    base_values = {*initial, *down, *up}
    allowed: dict[int, set[float]] = defaultdict(set)
    for (h, _mu, seed), values in grouped.items():
        if seed != adaptive.get("development_seed_only") or not values:
            continue
        values.sort()
        etas = [eta for eta, _loss in values]
        best_eta, _best_loss = min(values, key=lambda pair: (pair[1], pair[0]))
        best_index = etas.index(best_eta)
        candidate: list[float] = []
        if best_index == 0:
            boundary_chain = ([min(initial)] if initial else []) + down
            for index, eta in enumerate(boundary_chain[:-1]):
                if _same_float(best_eta, eta):
                    candidate = [boundary_chain[index + 1]]
                    break
        elif best_index == len(etas) - 1:
            boundary_chain = ([max(initial)] if initial else []) + up
            for index, eta in enumerate(boundary_chain[:-1]):
                if _same_float(best_eta, eta):
                    candidate = [boundary_chain[index + 1]]
                    break
        else:
            # Interior refinement is allowed once. Any already-sampled eta
            # outside the frozen boundary lattice is evidence that this curve
            # has received its one registered midpoint refinement round.
            already_refined = any(
                not any(_same_float(eta, base) for base in base_values)
                for eta in etas
            )
            if not already_refined:
                candidate = [
                    math.sqrt(etas[best_index - 1] * best_eta),
                    math.sqrt(best_eta * etas[best_index + 1]),
                ]
        for eta in candidate:
            if not any(_same_float(eta, sampled) for sampled in etas):
                allowed[h].add(eta)
    return allowed


def _validate_adaptive_extension(
    parent: Mapping[str, Any], manifest: Mapping[str, Any], errors: list[str]
) -> None:
    parent_cells, _ = _expected_cell_records(parent, errors)
    child_cells, _ = _expected_cell_records(manifest, errors)
    new_cells = child_cells - parent_cells
    allowed = _allowed_adaptive_etas(parent, manifest, errors)
    for coordinate in sorted(new_cells):
        if coordinate.seed != _mapping_or_empty(manifest.get("adaptive_bracket")).get(
            "development_seed_only"
        ):
            errors.append(f"adaptive cell uses a later/blinded seed: {coordinate}")
            continue
        if not any(
            _same_float(coordinate.eta, eta) for eta in allowed.get(coordinate.h, set())
        ):
            errors.append(
                f"adaptive cell eta is not the next registered boundary/midpoint: {coordinate}"
            )


def _registered_p2_coordinates(
    parent: Mapping[str, Any], errors: list[str]
) -> set[Coordinate]:
    """Derive P2's two-seed cells from sealed, passing P1 tuned choices."""
    latest = _latest_parent_rows(parent, errors)
    curves: dict[tuple[int, float], list[tuple[float, float]]] = defaultdict(list)
    for coordinate, row in latest.items():
        status = _canonical_status(row.get("status"))
        if coordinate.seed != 347:
            errors.append("P2 parent must contain only P1 development seed 347")
            continue
        if status != "COMPLETED" or not _finite_number(row.get("loss")):
            errors.append(f"P2 parent has unresolved/nonfinite P1 cell {coordinate}")
            continue
        curves[(coordinate.h, coordinate.mu)].append(
            (coordinate.eta, float(row["loss"]))
        )
    template = _authoritative_template()
    required_curves = {
        (int(h), float(mu))
        for h in _mapping_or_empty(template.get("expected_grid")).get("h", [])
        for mu in _mapping_or_empty(template.get("expected_grid")).get("mu", [])
    }
    if set(curves) != required_curves:
        errors.append("P2 parent must contain exactly nine complete P1 LR curves")
        return set()
    neighborhoods: dict[int, set[float]] = defaultdict(set)
    tuned: dict[tuple[int, float], float] = {}
    for key, points in curves.items():
        points.sort(key=lambda pair: pair[0])
        etas = [eta for eta, _loss in points]
        if len(etas) != len(set(etas)):
            errors.append(f"P2 parent curve {key} repeats an eta")
            continue
        selected_eta, selected_loss = min(points, key=lambda pair: (pair[1], pair[0]))
        selected_index = etas.index(selected_eta)
        if selected_index == 0 or selected_index == len(etas) - 1:
            errors.append(f"P2 parent curve {key} is not bracketed")
            continue
        if not (
            points[selected_index - 1][1] > selected_loss
            and points[selected_index + 1][1] > selected_loss
        ):
            errors.append(f"P2 parent curve {key} lacks worse immediate neighbors")
            continue
        neighborhoods[key[0]].update(
            (etas[selected_index - 1], selected_eta, etas[selected_index + 1])
        )
        tuned[key] = selected_loss
    if set(tuned) != required_curves:
        return set()
    gate = _mapping_or_empty(template.get("go_kill"))
    d_short = tuned[(16, 0.9)] - tuned[(16, 0.0)]
    d_long = min(tuned[(256, 0.5)], tuned[(256, 0.9)]) - tuned[(256, 0.0)]
    if d_short < float(gate.get("d_short_min", math.inf)):
        errors.append(f"P1 D_short={d_short} fails the registered P2 entry gate")
    if d_long > float(gate.get("d_long_max", -math.inf)):
        errors.append(f"P1 D_long={d_long} fails the registered P2 entry gate")
    seeds = [
        int(seed)
        for seed in _mapping_or_empty(
            _mapping_or_empty(template.get("lineage_policy")).get(
                "registered_descendant_kinds"
            )
        )
        .get("additional_development_stage", {})
        .get("required_new_seeds", [])
    ]
    required_mu = [
        float(mu)
        for mu in _mapping_or_empty(template.get("expected_grid")).get("mu", [])
    ]
    return {
        Coordinate(h, mu, eta, seed)
        for seed in seeds
        for h, etas in neighborhoods.items()
        for eta in etas
        for mu in required_mu
    }


def _validate_parent_replay_report(
    *,
    lineage: Mapping[str, Any],
    parent_manifest: Mapping[str, Any] | None,
    report: Mapping[str, Any] | None,
    report_sha256: str | None,
    errors: list[str],
) -> None:
    """Validate the sealed post-deletion replay cited by P0b or P1."""

    if report is None or report_sha256 is None:
        errors.append("stage requires the exact sealed parent CPU replay report")
        return
    if lineage.get("parent_replay_report_sha256") != report_sha256:
        errors.append(
            "lineage.parent_replay_report_sha256 does not match the raw report bytes"
        )
    if report.get("schema") != "yeto_p0_cpu_replay_v1":
        errors.append("parent replay report schema is not recognized")
    if report.get("status") != "PASS":
        errors.append("parent replay report did not PASS")
    if report.get("gpu_deleted_before_replay") is not True:
        errors.append("parent replay did not occur after verified GPU deletion")
    if report.get("all_steps_replayed") is not True:
        errors.append("parent replay report does not cover every captured step")
    if parent_manifest is not None:
        parent_hash = _sha256_canonical(parent_manifest)
        if report.get("phase_map_manifest_canonical_sha256") != parent_hash:
            errors.append("parent replay report is not bound to the cited parent manifest")


def _validate_canary_coordinates(
    manifest: Mapping[str, Any],
    kind_policy: Mapping[str, Any],
    errors: list[str],
) -> None:
    coordinates, _ = _expected_cell_records(manifest, errors)
    required = {
        Coordinate(
            int(kind_policy["required_h"]),
            float(mu),
            float(kind_policy["required_eta"]),
            int(kind_policy["required_shuffle_seed"]),
        )
        for mu in kind_policy.get("required_mu", [])
    }
    if coordinates != required:
        errors.append(
            "canary expected_cells must be exactly the registered H16/eta=.0875/"
            "mu={0,.5,.9}/seed337 block"
        )
    expected_grid = _mapping_or_empty(manifest.get("expected_grid"))
    expected_axes = {
        "h": [kind_policy.get("required_h")],
        "mu": kind_policy.get("required_mu"),
        "eta": [kind_policy.get("required_eta")],
        "seeds": [kind_policy.get("required_shuffle_seed")],
    }
    for field, expected in expected_axes.items():
        if expected_grid.get(field) != expected:
            errors.append(f"canary expected_grid.{field} must be {expected!r}")
    if _mapping_or_empty(manifest.get("seed_pairs")) != {
        str(kind_policy.get("required_shuffle_seed")): kind_policy.get(
            "required_training_seed"
        )
    }:
        errors.append("canary seed_pairs must be exactly {'337': 337337}")
    protocol = _mapping_or_empty(manifest.get("protocol"))
    for field, expected in (
        ("token_budget", kind_policy.get("required_token_budget")),
        ("machine_type", kind_policy.get("machine_type_required")),
        ("gpu_slots", kind_policy.get("gpu_slots_required")),
    ):
        if protocol.get(field) != expected:
            errors.append(f"canary protocol.{field} must be {expected!r}")
    horizon = _mapping_or_empty(
        _mapping_or_empty(manifest.get("horizon_work")).get(
            str(kind_policy.get("required_h"))
        )
    )
    if horizon.get("outer_steps") != kind_policy.get("required_global_commits"):
        errors.append("canary horizon_work must schedule exactly 32 global commits")
    fragments = protocol.get("fragments")
    if (
        not _is_int(fragments)
        or horizon.get("outer_steps") != fragments * kind_policy.get(
            "required_updates_per_fragment", -1
        )
    ):
        errors.append("canary must schedule exactly eight applied updates per fragment")


def _gpu_uuid_map(value: Any) -> dict[str, str] | None:
    if isinstance(value, Mapping):
        if any(not isinstance(item, str) for item in value.values()):
            return None
        result = {str(key): item for key, item in value.items()}
    elif isinstance(value, list):
        result = {}
        for item in value:
            if not isinstance(item, Mapping):
                return None
            learner = item.get("learner_id")
            uuid = item.get("gpu_uuid")
            if learner is None or not isinstance(uuid, str):
                return None
            result[str(learner)] = uuid
    else:
        return None
    return result


def _validate_p0b_hardware(
    manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], errors: list[str]
) -> None:
    policy = _mapping_or_empty(manifest.get("canary_hardware_evidence_policy"))
    required = policy.get("p0b_result_hardware_required_fields", [])
    for index, row in enumerate(rows):
        label = f"results[{index}].hardware"
        hardware = _require_mapping(row.get("hardware"), label, errors)
        missing = [field for field in required if field not in hardware]
        if missing:
            errors.append(f"{label} is missing P0b evidence fields: {missing}")
        for field in (
            "nvidia_smi_inventory_uri",
            "learner_gpu_map_uri",
            "barrier_version_trace_uri",
            "instance_not_found_evidence_uri",
            "disk_not_found_evidence_uri",
            "zero_accelerator_evidence_uri",
        ):
            _require_bound(hardware.get(field), f"{label}.{field}", errors)
        for field in (
            "nvidia_smi_inventory_sha256",
            "learner_gpu_map_sha256",
            "barrier_version_trace_sha256",
            "instance_not_found_evidence_sha256",
            "disk_not_found_evidence_sha256",
            "zero_accelerator_evidence_sha256",
        ):
            _require_sha256(hardware.get(field), f"{label}.{field}", errors)
        for field in (
            "zone",
            "instance_name",
            "instance_numeric_id",
            "boot_disk_name",
            "boot_disk_numeric_id",
            "source_image_numeric_id",
        ):
            _require_bound(hardware.get(field), f"{label}.{field}", errors)
        frozen_image = _mapping_or_empty(manifest.get("frozen")).get("image_id")
        if str(hardware.get("source_image_numeric_id")) != str(frozen_image):
            errors.append(f"{label}.source_image_numeric_id does not match frozen.image_id")
        if hardware.get("distinct_a100_gpu_uuid_count") != 4:
            errors.append(f"{label}.distinct_a100_gpu_uuid_count must be exactly 4")
        uuid_map = _gpu_uuid_map(hardware.get("learner_gpu_uuid_bijection"))
        if uuid_map is None or set(uuid_map) != {"0", "1", "2", "3"}:
            errors.append(f"{label}.learner_gpu_uuid_bijection must map learners 0..3")
        elif len(set(uuid_map.values())) != 4 or any(
            not value.startswith("GPU-") for value in uuid_map.values()
        ):
            errors.append(
                f"{label}.learner_gpu_uuid_bijection must contain four distinct GPU UUIDs"
            )

        # ``barrier_version_trace_uri`` binds one sealed registry whose raw
        # hashes cover the syncer tape and four per-learner causal JSONLs.  The
        # post-deletion CPU replay independently re-reads those five artifacts
        # and derives these attestations.  Requiring the replay-derived summary
        # here prevents a merge-only tape from masquerading as proof that every
        # learner applied its registered broadcast before its next inner step.
        for field in (
            "barrier_trace_validated",
            "base_versions_match",
            "no_inner_step_while_blocked",
        ):
            if hardware.get(field) is not True:
                errors.append(f"{label}.{field} must be true")
        if hardware.get("barrier_trace_learner_count") != 4:
            errors.append(f"{label}.barrier_trace_learner_count must be exactly 4")
        expected_commits = _mapping_or_empty(row.get("work")).get("outer_steps")
        if (
            not _is_int(expected_commits)
            or hardware.get("barrier_trace_commit_count") != expected_commits
        ):
            errors.append(
                f"{label}.barrier_trace_commit_count must equal the exact "
                "scientific outer-step count"
            )
        fixed_window_steps = _mapping_or_empty(row.get("work")).get(
            "fixed_window_microsteps"
        )
        fragment_count = _mapping_or_empty(manifest.get("protocol")).get(
            "fragments"
        )
        expected_inner_steps = (
            expected_commits // fragment_count * fixed_window_steps
            if _is_int(expected_commits)
            and _is_int(fragment_count)
            and fragment_count > 0
            and expected_commits % fragment_count == 0
            and _is_int(fixed_window_steps)
            and fixed_window_steps > 0
            else None
        )
        if (
            expected_inner_steps is None
            or hardware.get("barrier_trace_inner_steps_per_learner")
            != expected_inner_steps
        ):
            errors.append(
                f"{label}.barrier_trace_inner_steps_per_learner must equal the "
                f"exact physical learner-step count {expected_inner_steps}"
            )
        raw_per_fragment = _mapping_or_empty(
            _mapping_or_empty(row.get("observed_work")).get(
                "per_fragment_outer_steps"
            )
        )
        observed_per_fragment = {
            str(fragment): count for fragment, count in raw_per_fragment.items()
        }
        expected_per_fragment = (
            expected_commits // 4
            if _is_int(expected_commits) and expected_commits % 4 == 0
            else None
        )
        if observed_per_fragment != {
            str(fragment): expected_per_fragment for fragment in range(4)
        }:
            errors.append(
                f"{label}: sealed barrier evidence must report exactly "
                f"{expected_per_fragment} applied updates per fragment"
            )
        times = {
            field: _parse_time(hardware.get(field), f"{label}.{field}", errors)
            for field in (
                "provisioning_started_at",
                "provisioning_completed_at",
                "artifact_sealed_at",
                "deletion_requested_at",
                "deletion_completed_at",
            )
        }
        ordered = [times[field] for field in times]
        if all(value is not None for value in ordered) and any(
            right <= left for left, right in zip(ordered, ordered[1:])
        ):
            errors.append(
                f"{label} timestamps must order provisioning, seal, request, and deletion"
            )


def _validate_canary_lifecycle(
    manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], errors: list[str]
) -> None:
    """Bind immutable acquisition, deletion, and the separate final envelope."""

    for index, row in enumerate(rows):
        label = f"results[{index}].hardware"
        hardware = _require_mapping(row.get("hardware"), label, errors)
        if hardware.get("acquisition_status") != "sealed_acquisition_pending_teardown":
            errors.append(
                f"{label}.acquisition_status must identify the immutable intermediate seal"
            )
        if hardware.get("final_manifest_status") != "sealed_results":
            errors.append(f"{label}.final_manifest_status must be 'sealed_results'")
        for field in (
            "acquisition_manifest_sha256",
            "acquisition_manifest_canonical_sha256",
            "acquisition_checksum_sha256",
            "acquisition_seal_sha256",
            "deletion_evidence_sha256",
        ):
            _require_sha256(hardware.get(field), f"{label}.{field}", errors)
        acquisition_sealed = _parse_time(
            hardware.get("artifact_sealed_at"), f"{label}.artifact_sealed_at", errors
        )
        deletion_requested = _parse_time(
            hardware.get("deletion_requested_at"),
            f"{label}.deletion_requested_at",
            errors,
        )
        deletion_completed = _parse_time(
            hardware.get("deletion_completed_at"),
            f"{label}.deletion_completed_at",
            errors,
        )
        finalized = _parse_time(
            hardware.get("finalized_at"), f"{label}.finalized_at", errors
        )
        ordered = (
            acquisition_sealed,
            deletion_requested,
            deletion_completed,
            finalized,
        )
        if all(value is not None for value in ordered) and not (
            acquisition_sealed <= deletion_requested < deletion_completed < finalized
        ):
            errors.append(
                f"{label} must order immutable acquisition < deletion < final manifest"
            )


def _validate_p0b_matches_parent(
    manifest: Mapping[str, Any],
    parent: Mapping[str, Any],
    kind_policy: Mapping[str, Any],
    errors: list[str],
    *,
    source_rebind: bool,
) -> None:
    for path in kind_policy.get("must_equal_p0a_parent_paths", []):
        if source_rebind and path == "/frozen/git_commit":
            continue
        try:
            child_value = _json_pointer_get(manifest, str(path))
            parent_value = _json_pointer_get(parent, str(path))
        except (KeyError, ValueError):
            errors.append(f"P0a/P0b equality path is missing: {path}")
            continue
        if child_value != parent_value:
            errors.append(f"P0b differs from P0a at frozen path {path}")
    if manifest.get("status") != "sealed_results":
        return
    parent_by_coordinate = _latest_parent_rows(parent, errors)
    child_by_coordinate = _latest_parent_rows(manifest, errors)
    if set(parent_by_coordinate) != set(child_by_coordinate):
        errors.append("P0b result coordinates must exactly match P0a")
        return
    fields = (
        "training_seed",
        "work",
        "eval_source_indices_hash",
        "train_pool_source_indices_hash",
        "train_source_indices_hash",
        "train_rows_hash",
        "eval_rows_hash",
        "eval_hash",
        "eval_example_ids_hash",
        "eval_token_ids_hash",
        "barrier",
        "version_matched",
        "matrix_merge",
        "strict_quorum",
        "delta_correction",
        "injected_baseline",
        "normalized_workload_command_hash",
    )
    for coordinate in sorted(parent_by_coordinate):
        parent_row, child_row = parent_by_coordinate[coordinate], child_by_coordinate[coordinate]
        for field in fields:
            if parent_row.get(field) != child_row.get(field):
                errors.append(f"P0b {coordinate} differs from P0a workload field {field}")


def _validate_authority_and_lineage(
    manifest: Mapping[str, Any],
    errors: list[str],
    *,
    parent_manifest: Mapping[str, Any] | None,
    parent_replay_report: Mapping[str, Any] | None,
    parent_replay_report_sha256: str | None,
) -> None:
    template = _authoritative_template()
    lineage = _require_mapping(manifest.get("lineage"), "lineage", errors)
    if lineage.get("authoritative_prereg_path") != AUTHORITATIVE_PREREG_PATH:
        errors.append("lineage.authoritative_prereg_path is not the hard-pinned path")
    if lineage.get("authoritative_prereg_source_commit") != AUTHORITATIVE_PREREG_SOURCE_COMMIT:
        errors.append("lineage.authoritative_prereg_source_commit is not hard-pinned")
    if lineage.get("authoritative_prereg_template_sha256") != AUTHORITATIVE_PREREG_TEMPLATE_SHA256:
        errors.append("lineage.authoritative_prereg_template_sha256 is not hard-pinned")

    frozen_commit = _mapping_or_empty(manifest.get("frozen")).get("git_commit")
    if isinstance(frozen_commit, str) and re.fullmatch(r"[0-9a-f]{40}", frozen_commit):
        ancestry = subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "merge-base",
                "--is-ancestor",
                AUTHORITATIVE_PREREG_SOURCE_COMMIT,
                frozen_commit,
            ],
            check=False,
            capture_output=True,
        )
        if ancestry.returncode != 0:
            errors.append(
                "frozen.git_commit must be a repository descendant of the "
                "authoritative preregistration commit"
            )

    policy = _mapping_or_empty(template.get("lineage_policy"))
    kinds = _mapping_or_empty(policy.get("registered_descendant_kinds"))
    kind = _manifest_kind(manifest)
    kind_policy = kinds.get(kind)
    if not isinstance(kind_policy, Mapping):
        errors.append(f"lineage.descendant_kind {kind!r} is not registered")
        return

    compare_to = template
    p0b_source_rebind = False
    parent_required = kind_policy.get("parent_required") is True
    if parent_required:
        if parent_manifest is None:
            errors.append(f"lineage kind {kind!r} requires the exact sealed parent manifest")
        else:
            if parent_manifest.get("status") != "sealed_results":
                errors.append("lineage parent must have status sealed_results")
            expected_parent_hash = _sha256_canonical(parent_manifest)
            if lineage.get("parent_manifest_sha256") != expected_parent_hash:
                errors.append("lineage.parent_manifest_sha256 does not match the canonical parent")
            required_parent_kind = kind_policy.get("parent_kind_required")
            if required_parent_kind and _manifest_kind(parent_manifest) != required_parent_kind:
                errors.append(
                    f"lineage parent kind must be {required_parent_kind!r} for {kind!r}"
                )
            if kind == "additional_development_stage" and _manifest_kind(
                parent_manifest
            ) not in {"initial_bound_p1_r0", "adaptive_bracket_round"}:
                errors.append(
                    "additional_development_stage parent must be a sealed P1 manifest"
                )
            if kind == "adaptive_bracket_round" and _manifest_kind(
                parent_manifest
            ) not in {"initial_bound_p1_r0", "adaptive_bracket_round"}:
                errors.append("adaptive_bracket_round parent must be a sealed P1 manifest")
            if kind == "p0b_four_gpu_bound" and _manifest_kind(parent_manifest) == "p0a_single_gpu_bound":
                p0b_source_rebind = _validate_p0b_source_rebind(
                    manifest, parent_manifest, errors
                )
                try:
                    validate_and_summarize(parent_manifest, claim_level="integrity")
                except ManifestError as exc:
                    errors.append(f"cited P0a parent fails authoritative validation: {exc}")
            elif kind == "initial_bound_p1_r0" and _manifest_kind(parent_manifest) == "p0b_four_gpu_bound":
                # P0b's own authority validation consumed P0a and its replay.  At
                # P1 we can still independently validate its complete payload,
                # exact canary coordinates, and four-GPU evidence without
                # silently accepting a malformed sealed parent.
                try:
                    validate_and_summarize(
                        parent_manifest,
                        claim_level="integrity",
                        _skip_authority_for_tests=True,
                    )
                except ManifestError as exc:
                    errors.append(f"cited P0b parent fails payload validation: {exc}")
            if not kind_policy.get(
                "compare_allowed_paths_to_authoritative_template_not_p0b_parent"
            ):
                compare_to = parent_manifest
                if p0b_source_rebind:
                    compare_to = deepcopy(parent_manifest)
                    compare_to["frozen"]["git_commit"] = frozen_commit
                cumulative = kind in {
                    "adaptive_bracket_round",
                    "additional_development_stage",
                }
                if cumulative or kind_policy.get("parent_results_must_be_exact_prefix") is True:
                    parent_results = parent_manifest.get("results")
                    child_results = manifest.get("results")
                    if not isinstance(parent_results, list) or not isinstance(child_results, list):
                        errors.append("cumulative descendants require parent/child results arrays")
                    elif child_results[: len(parent_results)] != parent_results:
                        errors.append("existing parent results must remain an exact immutable prefix")
                    parent_expected = parent_manifest.get("expected_cells")
                    child_expected = manifest.get("expected_cells")
                    if (
                        not isinstance(parent_expected, list)
                        or not isinstance(child_expected, list)
                    ):
                        errors.append(
                            "cumulative descendants require parent/child expected_cells arrays"
                        )
                    elif child_expected[: len(parent_expected)] != parent_expected:
                        errors.append(
                            "existing parent expected_cells must remain an exact immutable prefix"
                        )
                if cumulative or kind_policy.get(
                    "expected_eta_and_cell_command_registry_must_be_strict_supersets"
                ) is True:
                    parent_cells, _ = _expected_cell_records(parent_manifest, errors)
                    child_cells, _ = _expected_cell_records(manifest, errors)
                    if not parent_cells < child_cells:
                        errors.append(
                            "cumulative expected_cells must be a strict superset of "
                            "the parent coordinates"
                        )
                    for field in (
                        "cell_command_hashes",
                        "train_source_indices_hashes",
                        "train_rows_hashes",
                    ):
                        parent_map = _mapping_or_empty(
                            _mapping_or_empty(parent_manifest.get("frozen")).get(
                                field
                            )
                        )
                        child_map = _mapping_or_empty(
                            _mapping_or_empty(manifest.get("frozen")).get(field)
                        )
                        if any(
                            key not in child_map or child_map[key] != value
                            for key, value in parent_map.items()
                        ):
                            errors.append(
                                f"cumulative frozen.{field} must preserve every "
                                "parent entry exactly"
                            )
                    parent_pairs = _mapping_or_empty(parent_manifest.get("seed_pairs"))
                    child_pairs = _mapping_or_empty(manifest.get("seed_pairs"))
                    if any(
                        key not in child_pairs or child_pairs[key] != value
                        for key, value in parent_pairs.items()
                    ):
                        errors.append(
                            "cumulative seed_pairs must preserve every parent entry exactly"
                        )
    else:
        if parent_manifest is not None:
            errors.append(f"lineage kind {kind!r} must be parentless")
        if lineage.get("parent_manifest_sha256") not in (None, ""):
            errors.append("parentless P0a lineage.parent_manifest_sha256 must be null")

    allowed = {
        str(path) for path in kind_policy.get("allowed_exact_paths", [])
        if isinstance(path, str)
    }
    for path in sorted(_json_differences(compare_to, manifest)):
        if not _path_is_allowed(path, allowed):
            errors.append(
                f"lineage kind {kind!r} illegally changes unregistered path {path}"
            )

    if kind in {"p0a_single_gpu_bound", "p0b_four_gpu_bound"}:
        _validate_canary_coordinates(manifest, kind_policy, errors)
    if kind == "p0a_single_gpu_bound":
        if lineage.get("parent_replay_report_sha256") not in (None, ""):
            errors.append("P0a cannot pre-bind a parent replay report")
    if kind == "p0b_four_gpu_bound":
        _validate_parent_replay_report(
            lineage=lineage,
            parent_manifest=parent_manifest,
            report=parent_replay_report,
            report_sha256=parent_replay_report_sha256,
            errors=errors,
        )
        if parent_manifest is not None:
            _validate_p0b_matches_parent(
                manifest,
                parent_manifest,
                kind_policy,
                errors,
                source_rebind=p0b_source_rebind,
            )

    if kind == "initial_bound_p1_r0":
        if parent_manifest is not None:
            for path in kind_policy.get("must_equal_p0b_parent_paths", []):
                try:
                    child_value = _json_pointer_get(manifest, str(path))
                    parent_value = _json_pointer_get(parent_manifest, str(path))
                except (KeyError, ValueError):
                    errors.append(f"P0b/P1 equality path is missing: {path}")
                    continue
                if child_value != parent_value:
                    errors.append(f"P1 differs from its passing P0b at frozen path {path}")
        _validate_parent_replay_report(
            lineage=lineage,
            parent_manifest=parent_manifest,
            report=parent_replay_report,
            report_sha256=parent_replay_report_sha256,
            errors=errors,
        )

    # R0 expected_cells is an explicit materialization of the frozen Cartesian
    # grid. Later kinds are ragged and are checked against their parent above.
    expected_cells, expected_ids = _expected_cell_records(manifest, errors)
    _validate_expected_cell_blocks(manifest, expected_cells, errors)
    required_new_seeds = kind_policy.get("required_new_seeds")
    if isinstance(required_new_seeds, list):
        parent_cells = (
            _expected_cell_records(parent_manifest, errors)[0]
            if parent_manifest is not None
            else set()
        )
        new_cells = expected_cells - parent_cells
        observed_stage_seeds = {coordinate.seed for coordinate in new_cells}
        required_stage_seeds = {int(seed) for seed in required_new_seeds}
        if observed_stage_seeds != required_stage_seeds:
            errors.append(
                f"{kind} new expected_cells seeds must equal the registered fresh stage "
                f"seeds {sorted(required_stage_seeds)}"
            )
    if kind == "initial_bound_p1_r0":
        grid_cells = _expected_coordinates_from_grid(manifest, errors)
        if expected_cells != grid_cells or len(expected_cells) != 36:
            errors.append("initial P1 expected_cells must equal the exact frozen 36-cell grid")
    if (
        kind == "adaptive_bracket_round"
        and parent_manifest is not None
        and kind_policy.get("only_next_registered_boundary_or_geometric_midpoint_eta_may_be_added")
        is True
    ):
        _validate_adaptive_extension(parent_manifest, manifest, errors)
    if kind == "additional_development_stage" and parent_manifest is not None:
        parent_cells, _ = _expected_cell_records(parent_manifest, errors)
        expected_new_cells = _registered_p2_coordinates(parent_manifest, errors)
        if expected_cells - parent_cells != expected_new_cells:
            errors.append(
                "P2 new expected_cells do not equal the sealed P1 selected-eta/"
                "immediate-neighbor blocks on both registered seeds"
            )
    registry_ids = set(
        _mapping_or_empty(_mapping_or_empty(manifest.get("frozen")).get("cell_command_hashes"))
    )
    if registry_ids != expected_ids:
        errors.append(
            "frozen.cell_command_hashes keys must equal authoritative expected_cells IDs"
        )


def _require_sha256(value: Any, label: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        errors.append(f"{label} must be a 64-hex SHA-256 (optional sha256: prefix)")


def _require_bound(value: Any, label: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not value.strip() or "BIND_" in value:
        errors.append(f"{label} is not immutably bound")


def _parse_time(value: Any, label: str, errors: list[str]) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{label} must be a timezone-aware ISO-8601 timestamp")
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        errors.append(f"{label} is not a valid ISO-8601 timestamp: {value!r}")
        return None
    if parsed.tzinfo is None:
        errors.append(f"{label} must include a timezone")
        return None
    return parsed


def _coordinate(row: Mapping[str, Any], label: str, errors: list[str]) -> Coordinate | None:
    error_count = len(errors)
    h, mu, eta, seed = (row.get("h"), row.get("mu"), row.get("eta"), row.get("seed"))
    if not _is_int(h) or h <= 0:
        errors.append(f"{label}.h must be a positive integer")
    if not _finite_number(mu) or not 0.0 <= float(mu) < 1.0:
        errors.append(f"{label}.mu must be finite in [0, 1)")
    if not _finite_number(eta) or float(eta) <= 0.0:
        errors.append(f"{label}.eta must be positive and finite")
    if not _is_int(seed) or seed < 0:
        errors.append(f"{label}.seed must be a non-negative integer")
    if len(errors) != error_count:
        return None
    return Coordinate(int(h), float(mu), float(eta), int(seed))


def _expected_coordinates_from_grid(
    manifest: Mapping[str, Any], errors: list[str]
) -> set[Coordinate]:
    grid = _require_mapping(manifest.get("expected_grid"), "expected_grid", errors)
    axes: dict[str, list[Any]] = {}
    for name in ("h", "mu", "eta", "seeds"):
        values = grid.get(name)
        if not isinstance(values, list) or not values:
            errors.append(f"expected_grid.{name} must be a non-empty list")
            axes[name] = []
        elif len({str(value) for value in values}) != len(values):
            errors.append(f"expected_grid.{name} contains duplicates")
            axes[name] = values
        else:
            axes[name] = values
    expected: set[Coordinate] = set()
    for h in axes["h"]:
        for mu in axes["mu"]:
            for eta in axes["eta"]:
                for seed in axes["seeds"]:
                    row = {"h": h, "mu": mu, "eta": eta, "seed": seed}
                    coordinate = _coordinate(row, "expected_grid cell", errors)
                    if coordinate is not None:
                        expected.add(coordinate)
    return expected


def _expected_coordinates(
    manifest: Mapping[str, Any], errors: list[str]
) -> set[Coordinate]:
    coordinates, _cell_ids = _expected_cell_records(manifest, errors)
    return coordinates


def _validate_frozen(manifest: Mapping[str, Any], errors: list[str]) -> None:
    frozen = _require_mapping(manifest.get("frozen"), "frozen", errors)
    git_commit = frozen.get("git_commit")
    if not isinstance(git_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", git_commit):
        errors.append("frozen.git_commit must be a bound lowercase 40-hex commit")
    for field in (
        "image_digest",
        "model_hash",
        "data_hash",
        "development_eval_source_indices_hash",
        "audit_eval_source_indices_hash",
        "train_pool_source_indices_hash",
        "development_eval_rows_hash",
        "development_eval_packed_hash",
        "development_eval_example_ids_hash",
        "development_eval_token_ids_hash",
        "audit_eval_rows_hash",
        "audit_eval_packed_hash",
        "audit_eval_example_ids_hash",
        "audit_eval_token_ids_hash",
        "audit_access_policy_hash",
        "command_hash",
        "randomization_plan_hash",
        "retry_policy_hash",
    ):
        _require_sha256(frozen.get(field), f"frozen.{field}", errors)
    expected_seed_keys = {
        str(coordinate.seed)
        for coordinate in _expected_coordinates(manifest, errors)
    }
    for map_name in ("train_source_indices_hashes", "train_rows_hashes"):
        values = _require_mapping(frozen.get(map_name), f"frozen.{map_name}", errors)
        if set(values) != expected_seed_keys:
            errors.append(
                f"frozen.{map_name} keys must exactly equal expected-cell seeds "
                f"{sorted(expected_seed_keys)}"
            )
        for seed, digest in values.items():
            _require_sha256(digest, f"frozen.{map_name}[{seed!r}]", errors)
    _require_bound(frozen.get("image_id"), "frozen.image_id", errors)
    _require_bound(frozen.get("model_id"), "frozen.model_id", errors)
    _require_bound(frozen.get("model_revision"), "frozen.model_revision", errors)

    registry = _require_mapping(
        frozen.get("cell_command_hashes"), "frozen.cell_command_hashes", errors
    )
    for cell_id, digest in registry.items():
        if not isinstance(cell_id, str) or not cell_id.strip():
            errors.append("frozen.cell_command_hashes keys must be non-empty cell IDs")
        _require_sha256(digest, f"frozen.cell_command_hashes[{cell_id!r}]", errors)

    expected_audit_policy_hash = _sha256_canonical(
        _require_mapping(
            manifest.get("confirmation_policy"), "confirmation_policy", errors
        )
    )
    if frozen.get("audit_access_policy_hash") != expected_audit_policy_hash:
        errors.append(
            "frozen.audit_access_policy_hash must equal canonical confirmation_policy"
        )

    fresh_confirmation = _manifest_kind(manifest) == "fresh_confirmation_stage"
    for field in (
        "audit_command_hash",
        "audit_randomization_plan_hash",
    ):
        value = frozen.get(field)
        if fresh_confirmation:
            _require_sha256(value, f"frozen.{field}", errors)
        elif value is not None:
            errors.append(f"frozen.{field} must remain null before P3")
    audit_registry = frozen.get("audit_cell_command_hashes")
    if fresh_confirmation:
        audit_registry = _require_mapping(
            audit_registry, "frozen.audit_cell_command_hashes", errors
        )
        for cell_id, digest in audit_registry.items():
            if not isinstance(cell_id, str) or not cell_id:
                errors.append("frozen.audit_cell_command_hashes keys must be cell IDs")
            _require_sha256(
                digest, f"frozen.audit_cell_command_hashes[{cell_id!r}]", errors
            )
    elif audit_registry is not None:
        errors.append("frozen.audit_cell_command_hashes must remain null before P3")

    retry_policy = _require_mapping(
        manifest.get("retry_policy"), "retry_policy", errors
    )
    expected_retry_hash = hashlib.sha256(
        _canonical_json_bytes(retry_policy)
    ).hexdigest()
    if frozen.get("retry_policy_hash") != expected_retry_hash:
        errors.append(
            "frozen.retry_policy_hash must equal SHA-256 of the exact canonical "
            "top-level retry_policy object"
        )


def _validate_protocol(manifest: Mapping[str, Any], errors: list[str]) -> None:
    protocol = _require_mapping(manifest.get("protocol"), "protocol", errors)
    expected = {
        "tuning": "full",
        "eval_split_seed": 331,
        "train_rows": 5000,
        "development_eval_rows": 1024,
        "audit_eval_rows": 1024,
        "split_population_rows": 7048,
        "split_population_rule": "canonical_source_indices_0_through_7047",
        "split_assignment_rule": (
            "python_random.Random(331).shuffle_once_then_positions_0_5000_train_"
            "5000_6024_development_6024_7048_audit"
        ),
        "train_pool_slice_rule": "shuffled_indices_half_open_0_5000",
        "development_eval_slice_rule": "shuffled_indices_half_open_5000_6024",
        "audit_eval_slice_rule": "shuffled_indices_half_open_6024_7048",
        "train_shuffle_rule": (
            "per_study_shuffle_seed_applies_only_to_disjoint_pre_shuffle_train_pool"
        ),
        "matrix_merge": "rda",
        "strict_quorum": True,
        "barrier": True,
        "version_matched": True,
        "delta_correction": "none",
        "spot_only": True,
        "on_demand_fallback": False,
        "injected_baseline": False,
        "per_example_loss_required": True,
    }
    for field, value in expected.items():
        if protocol.get(field) != value:
            errors.append(f"protocol.{field} must be {value!r}")
    adaptive = _require_mapping(
        manifest.get("adaptive_bracket"), "adaptive_bracket", errors
    )
    if adaptive.get("new_immutable_manifest_per_round") is not True:
        errors.append("adaptive_bracket.new_immutable_manifest_per_round must be true")

    randomization = _require_mapping(
        manifest.get("randomization"), "randomization", errors
    )
    if randomization.get("loss_blind") is not True:
        errors.append("randomization.loss_blind must be true")
    if randomization.get("block_fields") != ["h", "eta", "seed"]:
        errors.append("randomization.block_fields must be ['h', 'eta', 'seed']")

    retry = _require_mapping(manifest.get("retry_policy"), "retry_policy", errors)
    for field in ("loss_blind_only", "rerun_entire_incomplete_block", "retain_all_attempts", "retry_lineage_required"):
        if retry.get(field) is not True:
            errors.append(f"retry_policy.{field} must be true")
    direct_reasons = set(retry.get("direct_infrastructure_failure_reasons", []))
    allowed_reasons = set(retry.get("allowed_reasons", []))
    peer_reason = retry.get("peer_retry_reason")
    if not direct_reasons or not direct_reasons < allowed_reasons:
        errors.append(
            "retry_policy direct infrastructure reasons must be a non-empty proper "
            "subset of allowed_reasons"
        )
    if peer_reason not in allowed_reasons or peer_reason in direct_reasons:
        errors.append(
            "retry_policy.peer_retry_reason must be allowed but never a direct "
            "infrastructure-failure reason"
        )
    for field in (
        "peer_retry_reason_is_never_failure_reason",
        "infra_failure_reason_must_be_direct_infrastructure_failure_reason",
        "preserve_completed_peer_status_and_artifacts",
        "trigger_must_be_genuine_infra_failure_in_immediately_prior_same_block",
        "shared_block_retry_authorization_required",
        "retry_block_rows_must_be_contiguous",
        "result_acquisition_is_append_only",
    ):
        if retry.get(field) is not True:
            errors.append(f"retry_policy.{field} must be true")


def _validate_result_common(
    row: Mapping[str, Any],
    *,
    index: int,
    manifest: Mapping[str, Any],
    coordinate: Coordinate,
    errors: list[str],
) -> None:
    label = f"results[{index}]"
    frozen = _mapping_or_empty(manifest.get("frozen"))
    protocol = _mapping_or_empty(manifest.get("protocol"))
    seed_pairs = _mapping_or_empty(manifest.get("seed_pairs"))

    raw_required_fields = manifest.get("required_result_fields", [])
    required_fields = set(raw_required_fields if isinstance(raw_required_fields, list) else []) | {
        "result_uri",
        "result_sha256",
        "observed_work",
    }
    missing_fields = sorted(field for field in required_fields if field not in row)
    if missing_fields:
        errors.append(f"{label} is missing required fields: {missing_fields}")

    cell_id = row.get("cell_id")
    if not isinstance(cell_id, str) or not cell_id.strip():
        errors.append(f"{label}.cell_id must be non-empty")
    training_seed = row.get("training_seed")
    expected_training = seed_pairs.get(str(coordinate.seed))
    if not _is_int(training_seed) or training_seed != expected_training:
        errors.append(
            f"{label}.training_seed must equal seed_pairs[{coordinate.seed!r}] "
            f"({expected_training!r}), got {training_seed!r}"
        )

    for field in ("git_commit", "image_digest", "model_hash", "data_hash"):
        if row.get(field) != frozen.get(field):
            errors.append(f"{label}.{field} does not match the frozen manifest")
    if row.get("train_pool_source_indices_hash") != frozen.get(
        "train_pool_source_indices_hash"
    ):
        errors.append(
            f"{label}.train_pool_source_indices_hash does not match the frozen manifest"
        )
    fresh_confirmation = _manifest_kind(manifest) == "fresh_confirmation_stage"
    evaluation_role = row.get("evaluation_role")
    expected_role = "none" if fresh_confirmation else "development"
    if evaluation_role != expected_role:
        errors.append(f"{label}.evaluation_role must be {expected_role!r}")
    for field in (
        "audit_status",
        "audit_loss",
        "audit_command_hash",
        "audit_order_index",
        "audit_per_example_loss_uri",
        "audit_per_example_loss_sha256",
        "audit_started_at",
        "audit_ended_at",
        "audit_unblind_authorization_sha256",
    ):
        if row.get(field) is not None:
            errors.append(f"{label}.{field} must be null outside audit_results")
    development_hash_fields = {
        "eval_source_indices_hash": "development_eval_source_indices_hash",
        "eval_rows_hash": "development_eval_rows_hash",
        "eval_hash": "development_eval_packed_hash",
        "eval_example_ids_hash": "development_eval_example_ids_hash",
        "eval_token_ids_hash": "development_eval_token_ids_hash",
    }
    for row_field, frozen_field in development_hash_fields.items():
        if fresh_confirmation:
            if row.get(row_field) is not None:
                errors.append(f"{label}.{row_field} must be null for P3 training")
        elif row.get(row_field) != frozen.get(frozen_field):
            errors.append(
                f"{label}.{row_field} does not match frozen.{frozen_field}"
            )
    seed_key = str(coordinate.seed)
    split_maps = {
        "train_source_indices_hash": "train_source_indices_hashes",
        "train_rows_hash": "train_rows_hashes",
    }
    for row_field, frozen_map in split_maps.items():
        expected_value = _mapping_or_empty(frozen.get(frozen_map)).get(seed_key)
        if row.get(row_field) != expected_value:
            errors.append(
                f"{label}.{row_field} does not match frozen.{frozen_map}[{seed_key!r}]"
            )
    expected_command_hash = _mapping_or_empty(frozen.get("cell_command_hashes")).get(cell_id)
    if row.get("command_hash") != expected_command_hash:
        errors.append(
            f"{label}.command_hash does not match frozen.cell_command_hashes[{cell_id!r}]"
        )
    if _manifest_kind(manifest) in {"p0a_single_gpu_bound", "p0b_four_gpu_bound"}:
        _require_sha256(
            row.get("normalized_workload_command_hash"),
            f"{label}.normalized_workload_command_hash",
            errors,
        )

    semantic_expectations = {
        "barrier": protocol.get("barrier"),
        "version_matched": protocol.get("version_matched"),
        "matrix_merge": protocol.get("matrix_merge"),
        "strict_quorum": protocol.get("strict_quorum"),
        "delta_correction": protocol.get("delta_correction"),
        "injected_baseline": protocol.get("injected_baseline"),
    }
    for field, expected in semantic_expectations.items():
        if row.get(field) != expected:
            errors.append(f"{label}.{field} must equal frozen protocol value {expected!r}")

    if row.get("spot") is not True:
        errors.append(f"{label}.spot must be true for every GPU arm")
    hardware = _require_mapping(row.get("hardware"), f"{label}.hardware", errors)
    if hardware.get("market") != "spot":
        errors.append(f"{label}.hardware.market must be 'spot'")
    for field in ("provider", "instance_type", "region", "instance_id", "image_id", "provisioning_evidence_uri"):
        _require_bound(hardware.get(field), f"{label}.hardware.{field}", errors)
    if hardware.get("instance_type") != protocol.get("machine_type"):
        errors.append(
            f"{label}.hardware.instance_type does not match protocol.machine_type"
        )
    if str(hardware.get("image_id")) != str(frozen.get("image_id")):
        errors.append(f"{label}.hardware.image_id does not match frozen.image_id")
    _require_sha256(
        hardware.get("provisioning_evidence_sha256"),
        f"{label}.hardware.provisioning_evidence_sha256",
        errors,
    )

    start = _parse_time(row.get("started_at"), f"{label}.started_at", errors)
    end = _parse_time(row.get("ended_at"), f"{label}.ended_at", errors)
    if start is not None and end is not None and end <= start:
        errors.append(f"{label}.ended_at must be after started_at")

    status = _canonical_status(row.get("status"))
    _require_bound(row.get("result_uri"), f"{label}.result_uri", errors)
    _require_sha256(row.get("result_sha256"), f"{label}.result_sha256", errors)
    if status in SCIENTIFIC_STATUSES:
        _require_bound(row.get("capture_uri"), f"{label}.capture_uri", errors)
        _require_sha256(row.get("capture_sha256"), f"{label}.capture_sha256", errors)
    else:
        capture_uri, capture_sha = row.get("capture_uri"), row.get("capture_sha256")
        if (capture_uri is None) != (capture_sha is None):
            errors.append(
                f"{label}.capture_uri and capture_sha256 must both be null or both be present"
            )
        elif capture_uri is not None:
            _require_bound(capture_uri, f"{label}.capture_uri", errors)
            _require_sha256(capture_sha, f"{label}.capture_sha256", errors)

    if status == "COMPLETED" and not fresh_confirmation:
        for field in ("per_example_loss_uri",):
            _require_bound(row.get(field), f"{label}.{field}", errors)
        _require_sha256(
            row.get("per_example_loss_sha256"),
            f"{label}.per_example_loss_sha256",
            errors,
        )
    elif fresh_confirmation:
        for field in ("per_example_loss_uri", "per_example_loss_sha256"):
            if row.get(field) is not None:
                errors.append(f"{label}.{field} must be null for P3 training")
        if status == "COMPLETED":
            _require_bound(row.get("checkpoint_uri"), f"{label}.checkpoint_uri", errors)
            _require_sha256(
                row.get("checkpoint_sha256"), f"{label}.checkpoint_sha256", errors
            )
            _parse_time(
                row.get("checkpoint_sealed_at"),
                f"{label}.checkpoint_sealed_at",
                errors,
            )
        elif status == "DIVERGED":
            for field in ("checkpoint_uri", "checkpoint_sha256"):
                if row.get(field) is not None:
                    errors.append(f"{label}.{field} must be null for DIVERGED P3 training")

    block_id = row.get("block_id")
    if not isinstance(block_id, str) or not block_id.strip():
        errors.append(f"{label}.block_id must be non-empty")
    order_index = row.get("order_index")
    if not _is_int(order_index) or order_index < 0:
        errors.append(f"{label}.order_index must be a non-negative integer")


def _validate_status_and_work(
    row: Mapping[str, Any],
    *,
    index: int,
    manifest: Mapping[str, Any],
    coordinate: Coordinate,
    errors: list[str],
) -> None:
    label = f"results[{index}]"
    status = _canonical_status(row.get("status"))
    if status not in FINAL_STATUSES:
        errors.append(f"{label}.status must be one of {sorted(FINAL_STATUSES)}")
        return
    loss = row.get("loss")
    failure_reason = row.get("failure_reason")
    fresh_confirmation = _manifest_kind(manifest) == "fresh_confirmation_stage"
    if status == "COMPLETED":
        if fresh_confirmation:
            if loss is not None:
                errors.append(f"{label}.loss must be null for completed P3 training")
        elif not _finite_number(loss) or float(loss) <= 0.0:
            errors.append(f"{label}.loss must be positive and finite when COMPLETED")
        if failure_reason not in (None, ""):
            errors.append(f"{label}.failure_reason must be null/empty when COMPLETED")
    else:
        if loss is not None:
            errors.append(f"{label}.loss must be null when status is {status}")
        if not isinstance(failure_reason, str) or not failure_reason.strip():
            errors.append(f"{label}.failure_reason is required when status is {status}")

    reason_text = str(failure_reason or "").lower()
    if "preempt" in reason_text and status != "INFRA_FAILURE":
        errors.append(f"{label}: preemption must be classified as INFRA_FAILURE")
    direct_infra_reasons = set(
        _mapping_or_empty(manifest.get("retry_policy")).get(
            "direct_infrastructure_failure_reasons", []
        )
    )
    peer_reason = _mapping_or_empty(manifest.get("retry_policy")).get(
        "peer_retry_reason"
    )
    if status == "INFRA_FAILURE" and failure_reason not in direct_infra_reasons:
        errors.append(
            f"{label}.failure_reason {failure_reason!r} is not a frozen direct "
            "infrastructure-failure reason"
        )
    if status != "INFRA_FAILURE" and failure_reason in direct_infra_reasons:
        errors.append(
            f"{label}: frozen infrastructure reason must be classified as INFRA_FAILURE"
        )
    if failure_reason == peer_reason:
        errors.append(
            f"{label}: peer-only retry reason may never be used as failure_reason"
        )

    work = _require_mapping(row.get("work"), f"{label}.work", errors)
    horizon_work = _mapping_or_empty(
        _mapping_or_empty(manifest.get("horizon_work")).get(str(coordinate.h))
    )
    protocol = _mapping_or_empty(manifest.get("protocol"))
    expected_work = {
        "fixed_window_microsteps": horizon_work.get("fixed_window_microsteps"),
        "fixed_window_tokens": horizon_work.get("fixed_window_tokens"),
        "outer_steps": horizon_work.get("outer_steps"),
        "token_budget": protocol.get("token_budget"),
        "eval_rows": 0 if fresh_confirmation else protocol.get("development_eval_rows"),
    }
    for field, expected in expected_work.items():
        observed = work.get(field)
        if not _is_int(observed) or observed < 0:
            errors.append(f"{label}.work.{field} must be a non-negative integer")
        elif observed != expected:
            errors.append(
                f"{label}.work.{field}={observed} does not match frozen target {expected}"
            )

    observed_work = _require_mapping(
        row.get("observed_work"), f"{label}.observed_work", errors
    )
    expected_observed = {
        "tokens": protocol.get("token_budget"),
        "microsteps": (
            protocol.get("token_budget") // protocol.get("seq_len")
            if _is_int(protocol.get("token_budget"))
            and _is_int(protocol.get("seq_len"))
            and protocol.get("seq_len") > 0
            else None
        ),
        "outer_steps": horizon_work.get("outer_steps"),
    }
    for field, expected in expected_observed.items():
        observed = observed_work.get(field)
        if not _is_int(observed) or observed < 0:
            errors.append(f"{label}.observed_work.{field} must be a non-negative integer")
        elif status in SCIENTIFIC_STATUSES and observed != expected:
            errors.append(
                f"{label}.observed_work.{field}={observed} does not match scientific target {expected}"
            )
        elif status not in SCIENTIFIC_STATUSES and _is_int(expected) and observed > expected:
            errors.append(
                f"{label}.observed_work.{field} exceeds frozen target {expected}"
            )
    if status in SCIENTIFIC_STATUSES:
        for field in ("full_quorum", "fixed_window_exact", "version_matched_anchor_resolved"):
            if observed_work.get(field) is not True:
                errors.append(f"{label}.observed_work.{field} must be true")


def _final_attempts(
    rows: Sequence[Mapping[str, Any]],
    errors: list[str],
    manifest: Mapping[str, Any],
) -> dict[Coordinate, Mapping[str, Any]]:
    by_coordinate: dict[Coordinate, list[Mapping[str, Any]]] = defaultdict(list)
    seen_attempt_ids: set[str] = set()
    retry_policy = _mapping_or_empty(manifest.get("retry_policy"))
    allowed_reasons = set(retry_policy.get("allowed_reasons", []))
    frozen_retry_hash = _mapping_or_empty(manifest.get("frozen")).get("retry_policy_hash")

    for index, row in enumerate(rows):
        coordinate = _coordinate(row, f"results[{index}]", errors)
        if coordinate is None:
            continue
        by_coordinate[coordinate].append(row)
        attempt = row.get("attempt")
        if not _is_int(attempt) or attempt <= 0:
            errors.append(f"results[{index}].attempt must be a positive integer")
        attempt_id = row.get("attempt_id", f"{row.get('cell_id')}#{attempt}")
        if not isinstance(attempt_id, str) or not attempt_id.strip():
            errors.append(f"results[{index}].attempt_id must be non-empty when provided")
        elif attempt_id in seen_attempt_ids:
            errors.append(f"duplicate attempt identity {attempt_id!r}")
        else:
            seen_attempt_ids.add(attempt_id)

    finals: dict[Coordinate, Mapping[str, Any]] = {}
    for coordinate, attempts in by_coordinate.items():
        attempts.sort(key=lambda row: row.get("attempt", -1))
        numbers = [row.get("attempt") for row in attempts]
        if numbers != list(range(1, len(attempts) + 1)):
            errors.append(f"{coordinate}: attempts must be contiguous from 1, got {numbers}")
        cell_ids = {row.get("cell_id") for row in attempts}
        if len(cell_ids) != 1:
            errors.append(f"{coordinate}: retries must preserve the exact cell_id")
        for position, row in enumerate(attempts):
            retry_of = row.get("retry_of")
            if position == 0:
                if retry_of not in (None, ""):
                    errors.append(f"{coordinate}: first attempt cannot have retry_of")
                if row.get("retry_reason") not in (None, ""):
                    errors.append(f"{coordinate}: first attempt cannot have retry_reason")
                if row.get("retry_authorization") not in (None, {}):
                    errors.append(
                        f"{coordinate}: first attempt cannot have retry_authorization"
                    )
                continue
            previous = attempts[position - 1]
            previous_id = previous.get("attempt_id", f"{previous.get('cell_id')}#{previous.get('attempt')}")
            if retry_of != previous_id:
                errors.append(f"{coordinate}: retry_of must point to immediately prior attempt {previous_id!r}")
            reason = row.get("retry_reason")
            if reason not in allowed_reasons:
                errors.append(f"{coordinate}: retry_reason {reason!r} is not frozen/allowed")
            authorization = row.get("retry_authorization")
            authorization = authorization if isinstance(authorization, Mapping) else {}
            if authorization.get("loss_blind") is not True:
                errors.append(f"{coordinate}: retries require loss_blind=true authorization")
            if authorization.get("policy_hash") != frozen_retry_hash:
                errors.append(f"{coordinate}: retry authorization policy hash is not frozen")
            required_authorization = set(
                retry_policy.get("retry_authorization_required_fields", [])
            )
            missing_authorization = sorted(
                field for field in required_authorization if field not in authorization
            )
            if missing_authorization:
                errors.append(
                    f"{coordinate}: retry authorization missing fields "
                    f"{missing_authorization}"
                )
        finals[coordinate] = attempts[-1]
    return finals


def _validate_blocks(finals: Mapping[Coordinate, Mapping[str, Any]], manifest: Mapping[str, Any], errors: list[str]) -> None:
    required_mu = {
        float(value)
        for value in _mapping_or_empty(manifest.get("randomization")).get(
            "required_mu_per_block", []
        )
    }
    blocks: dict[tuple[int, float, int], list[tuple[Coordinate, Mapping[str, Any]]]] = defaultdict(list)
    for coordinate, row in finals.items():
        blocks[(coordinate.h, coordinate.eta, coordinate.seed)].append((coordinate, row))
    for key, entries in blocks.items():
        observed_mu = {coordinate.mu for coordinate, _ in entries}
        if observed_mu != required_mu:
            errors.append(f"randomization block {key} has mu={sorted(observed_mu)}, expected {sorted(required_mu)}")
        block_ids = {row.get("block_id") for _, row in entries}
        if len(block_ids) != 1:
            errors.append(f"randomization block {key} must share exactly one block_id")
        order = [row.get("order_index") for _, row in entries]
        if sorted(order) != list(range(len(entries))):
            errors.append(f"randomization block {key} order_index must be a permutation of 0..{len(entries)-1}")


def _validate_retry_blocks(rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any], errors: list[str]) -> None:
    """Enforce the frozen policy that an incomplete randomized block reruns whole."""

    if _mapping_or_empty(manifest.get("retry_policy")).get("rerun_entire_incomplete_block") is not True:
        errors.append("retry_policy.rerun_entire_incomplete_block must be true")
        return
    required_mu = {
        float(value)
        for value in _mapping_or_empty(manifest.get("randomization")).get("required_mu_per_block", [])
    }
    grouped: dict[
        tuple[int, float, int, int], list[tuple[int, Coordinate, Mapping[str, Any]]]
    ] = defaultdict(list)
    for index, row in enumerate(rows):
        coordinate_errors: list[str] = []
        coordinate = _coordinate(row, "retry block row", coordinate_errors)
        attempt = row.get("attempt")
        if coordinate is None or not _is_int(attempt):
            continue
        grouped[(coordinate.h, coordinate.eta, coordinate.seed, attempt)].append(
            (index, coordinate, row)
        )

    policy = _mapping_or_empty(manifest.get("retry_policy"))
    direct_reasons = set(policy.get("direct_infrastructure_failure_reasons", []))
    peer_reason = policy.get("peer_retry_reason")
    frozen_policy_hash = _mapping_or_empty(manifest.get("frozen")).get(
        "retry_policy_hash"
    )

    for key, entries in grouped.items():
        observed_mu = {coordinate.mu for _, coordinate, _ in entries}
        if observed_mu != required_mu:
            errors.append(
                f"retry/randomization block {key} is partial: mu={sorted(observed_mu)}, "
                f"expected {sorted(required_mu)}"
            )
        attempt = key[3]
        if attempt == 1:
            continue

        indices = sorted(index for index, _, _ in entries)
        if indices != list(range(indices[0], indices[0] + len(indices))):
            errors.append(f"retry block {key} rows must be contiguous and append-only")

        previous_key = (key[0], key[1], key[2], attempt - 1)
        previous_entries = grouped.get(previous_key, [])
        previous_by_mu = {
            coordinate.mu: row for _, coordinate, row in previous_entries
        }
        if set(previous_by_mu) != required_mu:
            errors.append(
                f"retry block {key} lacks a complete immediately prior block round"
            )
            continue

        current_rows = [row for _, _, row in entries]
        authorizations = [
            row.get("retry_authorization")
            if isinstance(row.get("retry_authorization"), Mapping)
            else {}
            for row in current_rows
        ]
        if not authorizations or any(
            authorization != authorizations[0]
            for authorization in authorizations[1:]
        ):
            errors.append(f"retry block {key} must share one exact authorization")
            authorization: Mapping[str, Any] = {}
        else:
            authorization = authorizations[0]

        prior_statuses = {
            mu: _canonical_status(row.get("status"))
            for mu, row in previous_by_mu.items()
        }
        invalid_prior = {
            mu: status
            for mu, status in prior_statuses.items()
            if status not in {"COMPLETED", "INFRA_FAILURE"}
        }
        if invalid_prior:
            errors.append(
                f"retry block {key} may not retry FAILED/DIVERGED prior outcomes: "
                f"{invalid_prior}"
            )

        infra_attempts = {
            row.get("attempt_id"): row
            for row in previous_by_mu.values()
            if _canonical_status(row.get("status")) == "INFRA_FAILURE"
            and row.get("failure_reason") in direct_reasons
        }
        trigger_id = authorization.get("trigger_attempt_id")
        trigger = infra_attempts.get(trigger_id)
        if trigger is None:
            errors.append(
                f"retry block {key} must cite a genuine direct INFRA_FAILURE in "
                "the immediately prior block"
            )
        else:
            if authorization.get("trigger_reason") != trigger.get("failure_reason"):
                errors.append(f"retry block {key} trigger_reason does not match trigger")
            if authorization.get("trigger_block_id") != trigger.get("block_id"):
                errors.append(f"retry block {key} trigger_block_id does not match trigger")

        block_ids = {row.get("block_id") for row in current_rows}
        prior_block_ids = {row.get("block_id") for row in previous_by_mu.values()}
        if len(block_ids) != 1 or block_ids != prior_block_ids:
            errors.append(
                f"retry block {key} must preserve the exact prior block_id"
            )
        elif authorization.get("trigger_block_id") not in block_ids:
            errors.append(
                f"retry block {key} authorization cites a different block_id"
            )

        if authorization.get("loss_blind") is not True:
            errors.append(f"retry block {key} authorization must be loss-blind")
        if authorization.get("policy_hash") != frozen_policy_hash:
            errors.append(f"retry block {key} authorization policy hash is not frozen")

        prior_manifest = dict(manifest)
        prior_manifest["results"] = list(rows[: indices[0]])
        expected_prior_hash = hashlib.sha256(
            _canonical_json_bytes(prior_manifest)
        ).hexdigest()
        if authorization.get("prior_manifest_sha256") != expected_prior_hash:
            errors.append(
                f"retry block {key} prior_manifest_sha256 does not match the "
                "canonical append-only manifest prefix"
            )

        current_by_mu = {coordinate.mu: row for _, coordinate, row in entries}
        for mu in required_mu & set(current_by_mu) & set(previous_by_mu):
            prior = previous_by_mu[mu]
            current = current_by_mu[mu]
            prior_status = _canonical_status(prior.get("status"))
            if prior_status == "COMPLETED":
                if current.get("retry_reason") != peer_reason:
                    errors.append(
                        f"retry block {key} completed peer mu={mu} must use "
                        f"retry_reason {peer_reason!r}"
                    )
            elif prior_status == "INFRA_FAILURE":
                if current.get("retry_reason") != prior.get("failure_reason"):
                    errors.append(
                        f"retry block {key} failed arm mu={mu} retry_reason must "
                        "match its direct failure_reason"
                    )


def _validate_pairing(finals: Mapping[Coordinate, Mapping[str, Any]], errors: list[str]) -> None:
    by_id: dict[str, tuple[Coordinate, Mapping[str, Any]]] = {}
    for coordinate, row in finals.items():
        cell_id = row.get("cell_id")
        if cell_id in by_id:
            errors.append(f"duplicate final cell_id {cell_id!r}")
        else:
            by_id[cell_id] = (coordinate, row)
    for coordinate, row in finals.items():
        control_id = row.get("paired_control_id")
        if coordinate.mu == 0.0:
            if control_id not in (None, "", row.get("cell_id")):
                errors.append(f"{coordinate}: mu=0 control must not point to a different control")
            continue
        if not isinstance(control_id, str) or control_id not in by_id:
            errors.append(f"{coordinate}: candidate requires an existing paired_control_id")
            continue
        control_coordinate, control = by_id[control_id]
        expected = (coordinate.h, coordinate.eta, coordinate.seed)
        actual = (control_coordinate.h, control_coordinate.eta, control_coordinate.seed)
        if control_coordinate.mu != 0.0 or actual != expected:
            errors.append(f"{coordinate}: paired control must be mu=0 at identical h/eta/seed")
        if row.get("training_seed") != control.get("training_seed"):
            errors.append(f"{coordinate}: paired control training_seed mismatch")
        if row.get("work") != control.get("work"):
            errors.append(f"{coordinate}: paired control work must match exactly")


def _validate_audit_top_level_quarantine(
    manifest: Mapping[str, Any], errors: list[str]
) -> None:
    """Forbid audit outcomes before a sealed P3 train-then-audit bundle."""

    fresh_confirmation = _manifest_kind(manifest) == "fresh_confirmation_stage"
    if not fresh_confirmation or manifest.get("status") == "bound_launch_authority":
        for field in (
            "audit_checkpoint_registry",
            "audit_unblind_authorization",
            "audit_randomization",
            "audit_results_seal",
        ):
            if manifest.get(field) is not None:
                errors.append(f"{field} must remain null before sealed P3 confirmation")
        for field in ("audit_access_log", "audit_results"):
            if manifest.get(field) != []:
                errors.append(f"{field} must remain empty before sealed P3 confirmation")


def _checkpoint_registry_cells(
    registry: Mapping[str, Any], errors: list[str]
) -> dict[str, Mapping[str, Any]]:
    raw = registry.get("cells")
    if not isinstance(raw, list):
        errors.append("audit_checkpoint_registry.cells must be an array")
        return {}
    output: dict[str, Mapping[str, Any]] = {}
    required = {
        "cell_id",
        "final_attempt_id",
        "status",
        "checkpoint_uri",
        "checkpoint_sha256",
        "command_hash",
        "training_completed_at",
    }
    for index, value in enumerate(raw):
        row = _require_mapping(
            value, f"audit_checkpoint_registry.cells[{index}]", errors
        )
        missing = sorted(required - set(row))
        if missing:
            errors.append(
                f"audit_checkpoint_registry.cells[{index}] missing fields {missing}"
            )
        cell_id = row.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id:
            errors.append(
                f"audit_checkpoint_registry.cells[{index}].cell_id must be non-empty"
            )
        elif cell_id in output:
            errors.append(f"audit checkpoint registry duplicates cell {cell_id!r}")
        else:
            output[cell_id] = row
    return output


def _validate_p3_audit(
    manifest: Mapping[str, Any],
    finals: Mapping[Coordinate, Mapping[str, Any]],
    errors: list[str],
) -> dict[str, Mapping[str, Any]]:
    """Validate the P3 train/seal/authorize/audit/seal/unblind sequence."""

    frozen = _mapping_or_empty(manifest.get("frozen"))
    policy = _mapping_or_empty(manifest.get("confirmation_policy"))
    expected_ids = {
        str(row.get("cell_id")) for row in finals.values() if row.get("cell_id")
    }
    by_id = {str(row.get("cell_id")): row for row in finals.values()}
    completed_ids = {
        cell_id
        for cell_id, row in by_id.items()
        if _canonical_status(row.get("status")) == "COMPLETED"
    }

    audit_commands = _require_mapping(
        frozen.get("audit_cell_command_hashes"),
        "frozen.audit_cell_command_hashes",
        errors,
    )
    if set(audit_commands) != expected_ids:
        errors.append(
            "frozen.audit_cell_command_hashes keys must equal expected P3 cell IDs"
        )

    registry = _require_mapping(
        manifest.get("audit_checkpoint_registry"),
        "audit_checkpoint_registry",
        errors,
    )
    registry_cells = _checkpoint_registry_cells(registry, errors)
    if set(registry_cells) != expected_ids:
        errors.append("audit checkpoint registry must cover expected P3 cells exactly")
    registry_sealed = _parse_time(
        registry.get("sealed_at_utc"),
        "audit_checkpoint_registry.sealed_at_utc",
        errors,
    )
    training_times: list[tuple[datetime, str]] = []
    for cell_id in sorted(expected_ids & set(registry_cells)):
        entry, result = registry_cells[cell_id], by_id[cell_id]
        status = _canonical_status(result.get("status"))
        if _canonical_status(entry.get("status")) != status:
            errors.append(f"checkpoint registry status mismatch for {cell_id}")
        if entry.get("final_attempt_id") != result.get("attempt_id"):
            errors.append(f"checkpoint registry final_attempt_id mismatch for {cell_id}")
        if entry.get("command_hash") != result.get("command_hash"):
            errors.append(f"checkpoint registry command_hash mismatch for {cell_id}")
        if entry.get("training_completed_at") != result.get("ended_at"):
            errors.append(f"checkpoint registry training_completed_at mismatch for {cell_id}")
        completed = _parse_time(
            entry.get("training_completed_at"),
            f"checkpoint registry {cell_id} training_completed_at",
            errors,
        )
        if completed is not None:
            training_times.append((completed, str(entry.get("training_completed_at"))))
        if status == "COMPLETED":
            if entry.get("checkpoint_uri") != result.get("checkpoint_uri"):
                errors.append(f"checkpoint registry checkpoint_uri mismatch for {cell_id}")
            if entry.get("checkpoint_sha256") != result.get("checkpoint_sha256"):
                errors.append(f"checkpoint registry checkpoint_sha256 mismatch for {cell_id}")
        elif status == "DIVERGED":
            if entry.get("checkpoint_uri") is not None or entry.get("checkpoint_sha256") is not None:
                errors.append(f"DIVERGED registry cell {cell_id} must have null checkpoint fields")
        else:
            errors.append(f"P3 training cell {cell_id} is unresolved with status {status}")
    training_max_record = max(training_times, default=None, key=lambda item: item[0])
    training_max = training_max_record[0] if training_max_record is not None else None
    if registry_sealed is not None and training_max is not None and registry_sealed < training_max:
        errors.append("checkpoint registry cannot seal before all P3 training completes")

    randomization = _require_mapping(
        manifest.get("audit_randomization"), "audit_randomization", errors
    )
    order = randomization.get("ordered_cell_ids")
    if not isinstance(order, list) or len(order) != len(expected_ids) or set(order) != expected_ids:
        errors.append("audit_randomization.ordered_cell_ids must be an exact P3 permutation")
        order = []
    elif len(order) != len(set(order)):
        errors.append("audit_randomization.ordered_cell_ids contains duplicates")
    if randomization.get("plan_hash") != frozen.get("audit_randomization_plan_hash"):
        errors.append("audit randomization plan_hash does not match the frozen plan")
    randomization_created = _parse_time(
        randomization.get("created_at_utc"),
        "audit_randomization.created_at_utc",
        errors,
    )

    authorization = _require_mapping(
        manifest.get("audit_unblind_authorization"),
        "audit_unblind_authorization",
        errors,
    )
    required_authorization = set(
        policy.get("audit_unblind_authorization_required_fields", [])
    )
    missing_authorization = sorted(required_authorization - set(authorization))
    if missing_authorization:
        errors.append(
            f"audit_unblind_authorization missing fields {missing_authorization}"
        )
    _require_bound(authorization.get("schema"), "audit_unblind_authorization.schema", errors)
    if authorization.get("loss_blind") is not True:
        errors.append("audit authorization must be loss_blind")
    if authorization.get("all_training_cells_resolved_and_checkpoint_registry_sealed") is not True:
        errors.append("audit authorization must attest all training cells resolved and sealed")
    if authorization.get("partial_results_withheld") is not True:
        errors.append("audit authorization must withhold partial results")
    _require_sha256(
        authorization.get("p3_manifest_canonical_sha256"),
        "audit_unblind_authorization.p3_manifest_canonical_sha256",
        errors,
    )
    if authorization.get("checkpoint_registry_sha256") != _sha256_canonical(registry):
        errors.append("audit authorization checkpoint_registry_sha256 mismatch")
    if authorization.get("audit_command_registry_sha256") != _sha256_canonical(
        audit_commands
    ):
        errors.append("audit authorization audit_command_registry_sha256 mismatch")
    if authorization.get("audit_randomization_plan_sha256") != frozen.get(
        "audit_randomization_plan_hash"
    ):
        errors.append("audit authorization randomization hash mismatch")
    training_max_text = (
        training_max_record[1] if training_max_record is not None else None
    )
    if authorization.get("training_completed_max_utc") != training_max_text:
        errors.append("audit authorization training_completed_max_utc mismatch")
    authorized = _parse_time(
        authorization.get("authorized_at_utc"),
        "audit_unblind_authorization.authorized_at_utc",
        errors,
    )
    if authorized is not None:
        if training_max is not None and authorized <= training_max:
            errors.append("audit authorization must occur after every training completion")
        if registry_sealed is not None and authorized <= registry_sealed:
            errors.append("audit authorization must occur after checkpoint registry sealing")
        if randomization_created is not None and authorized <= randomization_created:
            errors.append("audit randomization must be created before authorization")
    authorization_sha = _sha256_canonical(authorization)

    raw_audit_results = manifest.get("audit_results")
    if not isinstance(raw_audit_results, list):
        errors.append("audit_results must be an array")
        raw_audit_results = []
    audit_by_id: dict[str, Mapping[str, Any]] = {}
    required_audit = set(policy.get("audit_result_required_fields", []))
    audit_end_times: list[datetime] = []
    for index, value in enumerate(raw_audit_results):
        row = _require_mapping(value, f"audit_results[{index}]", errors)
        missing = sorted(required_audit - set(row))
        if missing:
            errors.append(f"audit_results[{index}] missing fields {missing}")
        cell_id = row.get("cell_id")
        if not isinstance(cell_id, str) or cell_id not in expected_ids:
            errors.append(f"audit_results[{index}].cell_id is not an expected P3 cell")
            continue
        if cell_id in audit_by_id:
            errors.append(f"audit_results duplicates cell {cell_id!r}")
            continue
        audit_by_id[cell_id] = row
        if row.get("evaluation_role") != "confirmation_audit":
            errors.append(
                f"audit_results[{index}].evaluation_role must be 'confirmation_audit'"
            )
        training = by_id[cell_id]
        registry_entry = registry_cells.get(cell_id, {})
        status = _canonical_status(training.get("status"))
        audit_status = _canonical_status(row.get("audit_status"))
        expected_order = order.index(cell_id) if cell_id in order else None
        if row.get("audit_order_index") != expected_order:
            errors.append(f"audit order index mismatch for {cell_id}")
        hash_fields = {
            "audit_eval_source_indices_hash": "audit_eval_source_indices_hash",
            "audit_eval_rows_hash": "audit_eval_rows_hash",
            "audit_eval_packed_hash": "audit_eval_packed_hash",
            "audit_eval_example_ids_hash": "audit_eval_example_ids_hash",
            "audit_eval_token_ids_hash": "audit_eval_token_ids_hash",
        }
        for row_field, frozen_field in hash_fields.items():
            if row.get(row_field) != frozen.get(frozen_field):
                errors.append(f"audit {cell_id} {row_field} does not match frozen audit data")
        if row.get("audit_command_hash") != audit_commands.get(cell_id):
            errors.append(f"audit command hash mismatch for {cell_id}")
        for field, expected in (
            ("checkpoint_uri", registry_entry.get("checkpoint_uri")),
            ("checkpoint_sha256", registry_entry.get("checkpoint_sha256")),
            ("training_attempt_id", training.get("attempt_id")),
            ("training_completed_at", registry_entry.get("training_completed_at")),
            ("audit_unblind_authorization_sha256", authorization_sha),
        ):
            if row.get(field) != expected:
                errors.append(f"audit {cell_id} {field} mismatch")
        if status == "COMPLETED":
            if audit_status != "COMPLETED" or not _finite_number(row.get("audit_loss")) or float(row["audit_loss"]) <= 0:
                errors.append(f"audit {cell_id} must have positive finite COMPLETED audit_loss")
            _require_bound(
                row.get("audit_per_example_loss_uri"),
                f"audit {cell_id} audit_per_example_loss_uri",
                errors,
            )
            _require_sha256(
                row.get("audit_per_example_loss_sha256"),
                f"audit {cell_id} audit_per_example_loss_sha256",
                errors,
            )
            started = _parse_time(row.get("audit_started_at"), f"audit {cell_id} started", errors)
            ended = _parse_time(row.get("audit_ended_at"), f"audit {cell_id} ended", errors)
            if started is not None and authorized is not None and started <= authorized:
                errors.append(f"audit {cell_id} started before shared authorization")
            if started is not None and training_max is not None and started <= training_max:
                errors.append(f"audit {cell_id} started before all P3 training completed")
            if started is not None and ended is not None and ended <= started:
                errors.append(f"audit {cell_id} ended before it started")
            if ended is not None:
                audit_end_times.append(ended)
        elif status == "DIVERGED":
            if audit_status != "DIVERGED" or row.get("audit_loss") is not None:
                errors.append(f"DIVERGED audit row {cell_id} must retain null audit_loss")
            for field in (
                "audit_per_example_loss_uri",
                "audit_per_example_loss_sha256",
                "audit_started_at",
                "audit_ended_at",
            ):
                if row.get(field) is not None:
                    errors.append(f"DIVERGED audit row {cell_id} must have null {field}")
    if set(audit_by_id) != expected_ids:
        errors.append("audit result IDs must cover expected P3 cells exactly")
    if order and [row.get("cell_id") for row in raw_audit_results] != order:
        errors.append("audit_results must be sealed in the frozen randomized order")

    access_log = manifest.get("audit_access_log")
    if not isinstance(access_log, list):
        errors.append("audit_access_log must be an array")
        access_log = []
    access_by_id: dict[str, Mapping[str, Any]] = {}
    required_access = set(policy.get("audit_access_log_entry_required_fields", []))
    for index, value in enumerate(access_log):
        entry = _require_mapping(value, f"audit_access_log[{index}]", errors)
        missing = sorted(required_access - set(entry))
        if missing:
            errors.append(f"audit_access_log[{index}] missing fields {missing}")
        cell_id = entry.get("cell_id")
        if not isinstance(cell_id, str) or cell_id not in completed_ids or cell_id in access_by_id:
            errors.append(f"audit_access_log[{index}] has unexpected/duplicate cell_id")
            continue
        access_by_id[cell_id] = entry
        audit_row = audit_by_id.get(cell_id, {})
        for field, expected in (
            ("checkpoint_sha256", registry_cells.get(cell_id, {}).get("checkpoint_sha256")),
            ("audit_eval_packed_hash", frozen.get("audit_eval_packed_hash")),
            ("audit_command_hash", audit_commands.get(cell_id)),
            ("access_started_at", audit_row.get("audit_started_at")),
            ("access_ended_at", audit_row.get("audit_ended_at")),
        ):
            if entry.get(field) != expected:
                errors.append(f"audit access log {cell_id} {field} mismatch")
    if set(access_by_id) != completed_ids:
        errors.append("audit access log must cover every and only evaluated checkpoints")

    seal = _require_mapping(
        manifest.get("audit_results_seal"), "audit_results_seal", errors
    )
    missing_seal = sorted(
        set(policy.get("audit_results_seal_required_fields", [])) - set(seal)
    )
    if missing_seal:
        errors.append(f"audit_results_seal missing fields {missing_seal}")
    _require_bound(seal.get("schema"), "audit_results_seal.schema", errors)
    if _canonical_status(seal.get("status")) not in {"PASS", "SEALED"}:
        errors.append("audit_results_seal.status must be PASS or SEALED")
    if seal.get("audit_result_registry_sha256") != _sha256_canonical(raw_audit_results):
        errors.append("audit_results_seal registry hash mismatch")
    if seal.get("audit_cell_count") != len(expected_ids):
        errors.append("audit_results_seal.audit_cell_count mismatch")
    if seal.get("expected_cell_ids_covered_exactly") is not True:
        errors.append("audit_results_seal must attest exact expected-cell coverage")
    if seal.get("partial_results_exposed") is not False:
        errors.append("partial audit results exposure is forbidden")
    sealed_at = _parse_time(seal.get("sealed_at_utc"), "audit_results_seal.sealed_at_utc", errors)
    unblinded_at = _parse_time(
        seal.get("unblinded_at_utc"), "audit_results_seal.unblinded_at_utc", errors
    )
    if sealed_at is not None and audit_end_times and sealed_at <= max(audit_end_times):
        errors.append("audit bundle must seal after every audit evaluation ends")
    if sealed_at is not None and unblinded_at is not None and unblinded_at < sealed_at:
        errors.append("audit unblinding cannot precede complete bundle sealing")
    return audit_by_id


def _mean_or_none(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return statistics.fmean(materialized) if materialized else None


def _phase_map_summary(finals: Mapping[Coordinate, Mapping[str, Any]], tolerance: float) -> list[dict[str, Any]]:
    groups: dict[tuple[int, float], dict[float, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for coordinate, row in finals.items():
        groups[(coordinate.h, coordinate.mu)][coordinate.eta].append(row)

    summaries: list[dict[str, Any]] = []
    for (h, mu), by_eta in sorted(groups.items()):
        points: list[dict[str, Any]] = []
        for eta, rows in sorted(by_eta.items()):
            statuses = Counter(_canonical_status(row.get("status")) for row in rows)
            completed_losses = [float(row["loss"]) for row in rows if _canonical_status(row.get("status")) == "COMPLETED"]
            resolvable = all(status in SCIENTIFIC_STATUSES for status in statuses)
            points.append(
                {
                    "eta": eta,
                    "n_planned": len(rows),
                    "status_counts": dict(sorted(statuses.items())),
                    "mean_loss": _mean_or_none(completed_losses),
                    "all_scientifically_resolved": resolvable,
                    "has_divergence": statuses.get("DIVERGED", 0) > 0,
                }
            )

        usable = [point for point in points if point["all_scientifically_resolved"] and point["mean_loss"] is not None and not point["has_divergence"]]
        if not usable:
            decision = "UNRESOLVED"
            optimum_eta = None
            reason = "no eta has completed, scientifically resolved losses"
        else:
            optimum = min(usable, key=lambda point: (point["mean_loss"], point["eta"]))
            optimum_eta = optimum["eta"]
            eta_values = [point["eta"] for point in points]
            index = eta_values.index(optimum_eta)
            if any(not point["all_scientifically_resolved"] for point in points):
                decision = "UNRESOLVED"
                reason = "at least one planned eta ends in FAILED/INFRA_FAILURE"
            elif index == 0:
                decision = "EXTEND_DOWNWARD"
                reason = "lowest tested eta has the best completed mean loss"
            elif index == len(points) - 1:
                decision = "EXTEND_UPWARD"
                reason = "highest tested eta has the best completed mean loss"
            else:
                lower, upper = points[index - 1], points[index + 1]
                def worse(point: Mapping[str, Any]) -> bool:
                    if point["has_divergence"]:
                        return True
                    value = point["mean_loss"]
                    return value is not None and value > optimum["mean_loss"] + tolerance
                if worse(lower) and worse(upper):
                    decision = "BRACKETED"
                    reason = "adjacent lower and upper eta are both worse"
                else:
                    decision = "REFINE_OR_EXTEND"
                    reason = "interior minimizer lacks two strictly worse adjacent points"
        summaries.append(
            {
                "h": h,
                "mu": mu,
                "points": points,
                "optimum_eta": optimum_eta,
                "bracket_decision": decision,
                "reason": reason,
            }
        )
    return summaries


def _paired_summary(
    finals: Mapping[Coordinate, Mapping[str, Any]],
    *,
    claim_scope: str = "paired_development_summary_only",
) -> list[dict[str, Any]]:
    by_id = {row.get("cell_id"): (coordinate, row) for coordinate, row in finals.items()}
    grouped: dict[tuple[int, float, float], list[float]] = defaultdict(list)
    unresolved: Counter[tuple[int, float, float]] = Counter()
    for coordinate, row in finals.items():
        if coordinate.mu == 0.0:
            continue
        key = (coordinate.h, coordinate.mu, coordinate.eta)
        control_entry = by_id.get(row.get("paired_control_id"))
        if control_entry is None:
            unresolved[key] += 1
            continue
        _, control = control_entry
        if _canonical_status(row.get("status")) == "COMPLETED" and _canonical_status(control.get("status")) == "COMPLETED":
            grouped[key].append(float(row["loss"]) - float(control["loss"]))
        else:
            unresolved[key] += 1

    output: list[dict[str, Any]] = []
    for key in sorted(set(grouped) | set(unresolved)):
        values = grouped.get(key, [])
        output.append(
            {
                "h": key[0],
                "mu": key[1],
                "eta": key[2],
                "n_paired_completed": len(values),
                "n_unresolved": unresolved[key],
                "mean_candidate_minus_control": _mean_or_none(values),
                "sample_sd": statistics.stdev(values) if len(values) >= 2 else None,
                "seed_level_differences": values,
                "claim_scope": claim_scope,
            }
        )
    return output


def validate_and_summarize(
    manifest: Mapping[str, Any],
    *,
    claim_level: str = "integrity",
    require_bracketed: bool = False,
    parent_manifest: Mapping[str, Any] | None = None,
    parent_replay_report: Mapping[str, Any] | None = None,
    parent_replay_report_sha256: str | None = None,
    _skip_authority_for_tests: bool = False,
) -> dict[str, Any]:
    """Validate a manifest and return a machine-readable audit/summary.

    ``claim_level='confirmatory'`` is deliberately rejected for development
    manifests or for fewer than eight independent seed blocks, irrespective of
    how many eval rows, commits, or bootstrap samples are present.
    """

    errors: list[str] = []
    if not isinstance(manifest, Mapping):
        raise ManifestError("manifest root must be a JSON object")
    if str(manifest.get("schema_version")) != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}")
    if manifest.get("status") not in {"bound_launch_authority", "sealed_results"}:
        errors.append(
            "status must be bound_launch_authority or sealed_results; preregistration "
            "templates/placeholders are not result manifests"
        )
    _require_bound(manifest.get("study_id"), "study_id", errors)
    mode = str(manifest.get("mode", "")).strip().lower()
    if mode not in {"development", "confirmation"}:
        errors.append("mode must be 'development' or 'confirmation'")
    _validate_frozen(manifest, errors)
    _validate_protocol(manifest, errors)
    _validate_audit_top_level_quarantine(manifest, errors)
    kind = _manifest_kind(manifest)
    if _skip_authority_for_tests and kind in {
        "p0a_single_gpu_bound",
        "p0b_four_gpu_bound",
    }:
        kind_policy = _mapping_or_empty(
            _mapping_or_empty(
                _mapping_or_empty(_authoritative_template().get("lineage_policy")).get(
                    "registered_descendant_kinds"
                )
            ).get(kind)
        )
        _validate_canary_coordinates(manifest, kind_policy, errors)
    if not _skip_authority_for_tests:
        _validate_authority_and_lineage(
            manifest,
            errors,
            parent_manifest=parent_manifest,
            parent_replay_report=parent_replay_report,
            parent_replay_report_sha256=parent_replay_report_sha256,
        )
    required_result_fields = manifest.get("required_result_fields")
    if (
        not isinstance(required_result_fields, list)
        or not required_result_fields
        or any(not isinstance(field, str) or not field for field in required_result_fields)
        or len(set(required_result_fields)) != len(required_result_fields)
    ):
        errors.append("required_result_fields must be a non-empty unique string list")
    expected = _expected_coordinates(manifest, errors)
    results = manifest.get("results")
    if not isinstance(results, list):
        errors.append("results must be an array retaining every attempt")
        results = []
    status = manifest.get("status")
    if status == "bound_launch_authority":
        cumulative_bound = kind in {
            "adaptive_bracket_round",
            "additional_development_stage",
        }
        if results and not cumulative_bound:
            errors.append("bound_launch_authority must have an empty results array")
        if cumulative_bound and parent_manifest is None:
            errors.append("cumulative bound authority requires its exact sealed parent")
        if claim_level != "integrity":
            errors.append("unexecuted launch authority supports integrity validation only")
        if errors:
            raise ManifestError("\n".join(f"- {message}" for message in errors))
        return {
            "schema_version": "1.0",
            "valid": True,
            "integrity_status": "BOUND_LAUNCH_AUTHORITY_VALIDATED",
            "study_id": manifest["study_id"],
            "manifest_mode": mode,
            "claim_level_requested": claim_level,
            "claim_scope": "unexecuted_launch_authority",
            "confirmatory_eligible": False,
            "expected_cell_count": len(expected),
            "final_cell_count": 0,
            "attempt_count": len(results),
            "warnings": [
                "This validates launch authority only; it contains no scientific outcomes."
            ],
        }
    if status == "sealed_results" and not results:
        errors.append("sealed_results must retain at least one terminal attempt")
    expected_by_id = _expected_cell_id_coordinates(manifest)
    rows: list[Mapping[str, Any]] = []
    for index, raw in enumerate(results):
        row = _require_mapping(raw, f"results[{index}]", errors)
        coordinate = _coordinate(row, f"results[{index}]", errors)
        if coordinate is not None:
            expected_coordinate = expected_by_id.get(str(row.get("cell_id")))
            if expected_coordinate != coordinate:
                errors.append(
                    f"results[{index}].cell_id is not bound to its declared "
                    f"expected_cells coordinate"
                )
            _validate_result_common(row, index=index, manifest=manifest, coordinate=coordinate, errors=errors)
            _validate_status_and_work(row, index=index, manifest=manifest, coordinate=coordinate, errors=errors)
        rows.append(row)
    finals = _final_attempts(rows, errors, manifest)

    observed = set(finals)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing:
        errors.append(f"missing expected cells: {missing}")
    if unexpected:
        errors.append(f"unexpected cells not in immutable grid: {unexpected}")
    _validate_blocks(finals, manifest, errors)
    _validate_retry_blocks(rows, manifest, errors)
    _validate_pairing(finals, errors)
    if kind in {"p0a_single_gpu_bound", "p0b_four_gpu_bound"}:
        _validate_canary_lifecycle(manifest, rows, errors)
    if kind == "p0b_four_gpu_bound":
        _validate_p0b_hardware(manifest, rows, errors)
    final_cell_ids = {row.get("cell_id") for row in finals.values()}
    registry_cell_ids = set(
        _mapping_or_empty(
            _mapping_or_empty(manifest.get("frozen")).get("cell_command_hashes")
        )
    )
    if registry_cell_ids != final_cell_ids:
        errors.append(
            "cell_command_hashes must cover exactly the immutable result cells; "
            f"missing={sorted(final_cell_ids - registry_cell_ids)}, "
            f"unexpected={sorted(registry_cell_ids - final_cell_ids)}"
        )

    fresh_confirmation = kind == "fresh_confirmation_stage"
    audit_by_id: dict[str, Mapping[str, Any]] = {}
    if fresh_confirmation:
        audit_by_id = _validate_p3_audit(manifest, finals, errors)
    registered_confirmation_seeds = (
        {
            coordinate.seed for coordinate in finals
        }
        if fresh_confirmation and _skip_authority_for_tests
        else {
            int(seed)
            for seed in _mapping_or_empty(
                _mapping_or_empty(
                    _mapping_or_empty(
                        _authoritative_template().get("lineage_policy")
                    ).get("registered_descendant_kinds")
                ).get("fresh_confirmation_stage")
            ).get("required_new_seeds", [])
        }
        if fresh_confirmation
        else set()
    )
    inference_finals = (
        {
            coordinate: row
            for coordinate, row in finals.items()
            if coordinate.seed in registered_confirmation_seeds
        }
        if fresh_confirmation
        else finals
    )
    if fresh_confirmation:
        audit_inference_finals: dict[Coordinate, Mapping[str, Any]] = {}
        for coordinate, training_row in inference_finals.items():
            cell_id = str(training_row.get("cell_id"))
            audit_row = audit_by_id.get(cell_id)
            if audit_row is None:
                continue
            outcome = dict(training_row)
            outcome["status"] = audit_row.get("audit_status")
            outcome["loss"] = audit_row.get("audit_loss")
            audit_inference_finals[coordinate] = outcome
        inference_outcomes: Mapping[Coordinate, Mapping[str, Any]] = audit_inference_finals
    else:
        inference_outcomes = inference_finals
    unique_seeds = sorted({coordinate.seed for coordinate in inference_finals})
    configured_min = manifest.get("min_confirmatory_seeds")
    if not _is_int(configured_min) or configured_min < HARD_MIN_CONFIRMATORY_SEEDS:
        errors.append(
            f"min_confirmatory_seeds must be at least {HARD_MIN_CONFIRMATORY_SEEDS}"
        )

    raw_tolerance = _mapping_or_empty(manifest.get("analysis_policy")).get(
        "bracketing_tolerance", 0.0
    )
    tolerance = float(raw_tolerance) if _finite_number(raw_tolerance) else -1.0
    if tolerance < 0.0 or not math.isfinite(tolerance):
        errors.append("analysis_policy.bracketing_tolerance must be finite and non-negative")
        tolerance = 0.0
    phase_map = [] if fresh_confirmation else _phase_map_summary(finals, tolerance)
    unbracketed = [
        {"h": row["h"], "mu": row["mu"], "decision": row["bracket_decision"]}
        for row in phase_map
        if row["bracket_decision"] != "BRACKETED"
    ]
    if require_bracketed and unbracketed and not fresh_confirmation:
        errors.append(f"not all LR curves are bracketed: {unbracketed}")

    claim_level = claim_level.strip().lower()
    if claim_level not in {"integrity", "development", "confirmatory"}:
        errors.append("claim_level must be integrity, development, or confirmatory")
    confirmatory_eligible = (
        fresh_confirmation
        and
        mode == "confirmation"
        and _is_int(configured_min)
        and len(unique_seeds) >= max(HARD_MIN_CONFIRMATORY_SEEDS, configured_min)
        and (fresh_confirmation or not unbracketed)
        and all(
            _canonical_status(row.get("status")) in SCIENTIFIC_STATUSES
            for row in inference_outcomes.values()
        )
        and (not fresh_confirmation or len(inference_outcomes) == len(inference_finals))
    )
    if claim_level == "confirmatory" and mode == "development":
        errors.append(
            "REFUSED_CONFIRMATORY_CLAIM: development manifests remain development-only, "
            "regardless of eval rows or bootstrap samples"
        )
    if claim_level == "confirmatory" and not fresh_confirmation:
        errors.append(
            "REFUSED_CONFIRMATORY_CLAIM: only a registered fresh_confirmation_stage "
            "with a sealed audit bundle can support confirmation"
        )
    if claim_level == "confirmatory" and len(unique_seeds) < max(HARD_MIN_CONFIRMATORY_SEEDS, int(configured_min or 0)):
        errors.append(
            "REFUSED_CONFIRMATORY_CLAIM: insufficient independent paired training seeds "
            f"({len(unique_seeds)} present)"
        )
    if claim_level == "confirmatory" and not confirmatory_eligible and not any(
        message.startswith("REFUSED_CONFIRMATORY_CLAIM") for message in errors
    ):
        errors.append("REFUSED_CONFIRMATORY_CLAIM: confirmatory integrity/bracketing requirements are unmet")

    if errors:
        raise ManifestError("\n".join(f"- {message}" for message in errors))

    status_counts = Counter(_canonical_status(row.get("status")) for row in rows)
    final_status_counts = Counter(_canonical_status(row.get("status")) for row in finals.values())
    overall_bracket = (
        "NOT_APPLICABLE_FROZEN_TUNED_CONFIRMATION"
        if fresh_confirmation
        else ("BRACKETED_DEVELOPMENT_ONLY" if not unbracketed else "EXTENSION_REQUIRED")
    )
    return {
        "schema_version": "1.0",
        "valid": True,
        "integrity_status": "VALIDATED",
        "study_id": manifest["study_id"],
        "manifest_mode": mode,
        "claim_level_requested": claim_level,
        "claim_scope": "confirmatory" if confirmatory_eligible else "development_only",
        "confirmatory_eligible": confirmatory_eligible,
        "independent_unit": "complete_paired_training_seed_block",
        "independent_seed_count": len(unique_seeds),
        "independent_seeds": unique_seeds,
        "expected_cell_count": len(expected),
        "final_cell_count": len(finals),
        "attempt_count": len(rows),
        "all_attempt_status_counts": dict(sorted(status_counts.items())),
        "final_status_counts": dict(sorted(final_status_counts.items())),
        "retained_noncompleted_attempt_count": sum(
            count for status, count in status_counts.items() if status != "COMPLETED"
        ),
        "retry_count": len(rows) - len(finals),
        "overall_bracket_decision": overall_bracket,
        "extension_requirements": unbracketed,
        "phase_map": phase_map,
        "paired_development_summaries": (
            [] if fresh_confirmation else _paired_summary(inference_finals)
        ),
        "paired_audit_summaries": (
            _paired_summary(
                inference_outcomes,
                claim_scope="paired_confirmation_audit_primary_endpoint",
            )
            if fresh_confirmation
            else []
        ),
        "warnings": []
        if confirmatory_eligible
        else [
            "This report is development-only. It is not evidence of training-seed replication or confirmation."
        ],
    }


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ManifestError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ManifestError("manifest root must be a JSON object")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="bound result manifest JSON")
    parser.add_argument(
        "--claim-level",
        choices=("integrity", "development", "confirmatory"),
        default="integrity",
    )
    parser.add_argument("--require-bracketed", action="store_true")
    parser.add_argument(
        "--parent-manifest",
        type=Path,
        help="exact sealed parent required by P1 and every later descendant",
    )
    parser.add_argument(
        "--parent-replay-report",
        "--p0-replay-report",
        dest="parent_replay_report",
        type=Path,
        help="sealed post-deletion CPU replay report required by P0b and initial P1",
    )
    parser.add_argument("--output", type=Path, help="write JSON report here; default stdout")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        replay_report = (
            _load_json(args.parent_replay_report)
            if args.parent_replay_report
            else None
        )
        replay_sha = None
        if args.parent_replay_report:
            replay_sha = hashlib.sha256(
                args.parent_replay_report.read_bytes()
            ).hexdigest()
        report = validate_and_summarize(
            _load_json(args.manifest),
            claim_level=args.claim_level,
            require_bracketed=args.require_bracketed,
            parent_manifest=(
                _load_json(args.parent_manifest) if args.parent_manifest else None
            ),
            parent_replay_report=replay_report,
            parent_replay_report_sha256=replay_sha,
        )
    except (ManifestError, OSError) as exc:
        print(f"PHASE_MAP_VALIDATION_ERROR\n{exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
