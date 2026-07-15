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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "0.1"
HARD_MIN_CONFIRMATORY_SEEDS = 8
FINAL_STATUSES = {"COMPLETED", "DIVERGED", "FAILED", "INFRA_FAILURE"}
SCIENTIFIC_STATUSES = {"COMPLETED", "DIVERGED"}
SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")
GIT_RE = re.compile(r"^[0-9a-fA-F]{40,64}$")
REPO_ROOT = Path(__file__).resolve().parents[1]
AUTHORITATIVE_PREREG_PATH = (
    "experiment-specs/best-paper-phase-map-p0-p1-prereg.json"
)
AUTHORITATIVE_PREREG_SOURCE_COMMIT = "af3605fc5c98bb5ed61b7225cd2bd45ca2b5a1cb"
AUTHORITATIVE_PREREG_TEMPLATE_SHA256 = (
    "0b1963771e3e28e89f74dbe4f11d5b2ef913618d5adde0839260aa25513ab775"
)


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


def _validate_authority_and_lineage(
    manifest: Mapping[str, Any],
    errors: list[str],
    *,
    parent_manifest: Mapping[str, Any] | None,
    p0_replay_report: Mapping[str, Any] | None,
    p0_replay_report_sha256: str | None,
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
            if kind == "initial_bound_p1_r0" and _manifest_kind(parent_manifest) == "p0_canary_bound":
                try:
                    validate_and_summarize(parent_manifest, claim_level="integrity")
                except ManifestError as exc:
                    errors.append(f"cited P0 parent fails authoritative validation: {exc}")
            if kind != "initial_bound_p1_r0":
                compare_to = parent_manifest
                if kind_policy.get("parent_results_must_be_exact_prefix") is True:
                    parent_results = parent_manifest.get("results")
                    child_results = manifest.get("results")
                    if not isinstance(parent_results, list) or not isinstance(child_results, list):
                        errors.append("cumulative descendants require parent/child results arrays")
                    elif child_results[: len(parent_results)] != parent_results:
                        errors.append("existing parent results must remain an exact immutable prefix")
                if kind_policy.get(
                    "expected_eta_and_cell_command_registry_must_be_strict_supersets"
                ) is True:
                    parent_cells, _ = _expected_cell_records(parent_manifest, errors)
                    child_cells, _ = _expected_cell_records(manifest, errors)
                    if not parent_cells < child_cells:
                        errors.append(
                            "adaptive expected_cells must be a strict superset of "
                            "the parent coordinates"
                        )
    else:
        if parent_manifest is not None:
            errors.append(f"lineage kind {kind!r} must be parentless")
        if lineage.get("parent_manifest_sha256") not in (None, ""):
            errors.append("parentless P0 lineage.parent_manifest_sha256 must be null")

    allowed = {
        str(path) for path in kind_policy.get("allowed_exact_paths", [])
        if isinstance(path, str)
    }
    for path in sorted(_json_differences(compare_to, manifest)):
        if not _path_is_allowed(path, allowed):
            errors.append(
                f"lineage kind {kind!r} illegally changes unregistered path {path}"
            )

    if kind == "p0_canary_bound":
        protocol = _mapping_or_empty(manifest.get("protocol"))
        if protocol.get("machine_type") != kind_policy.get("machine_type_required"):
            errors.append("P0 must use the registered one-A100 machine type")
        if protocol.get("gpu_slots") != kind_policy.get("gpu_slots_required"):
            errors.append("P0 must use gpu_slots=1")
        if lineage.get("p0_replay_report_sha256") not in (None, ""):
            errors.append("P0 cannot pre-bind a replay report that does not yet exist")

    if kind == "initial_bound_p1_r0":
        if parent_manifest is not None:
            for path in kind_policy.get("must_equal_p0_parent_paths", []):
                try:
                    child_value = _json_pointer_get(manifest, str(path))
                    parent_value = _json_pointer_get(parent_manifest, str(path))
                except (KeyError, ValueError):
                    errors.append(f"P0/P1 equality path is missing: {path}")
                    continue
                if child_value != parent_value:
                    errors.append(f"P1 differs from its passing P0 at frozen path {path}")
        if p0_replay_report is None or p0_replay_report_sha256 is None:
            errors.append("P1 requires the exact sealed P0 CPU replay report")
        else:
            if lineage.get("p0_replay_report_sha256") != p0_replay_report_sha256:
                errors.append("lineage.p0_replay_report_sha256 does not match the report bytes")
            if p0_replay_report.get("schema") != "yeto_p0_cpu_replay_v1":
                errors.append("P0 replay report schema is not recognized")
            if p0_replay_report.get("status") != "PASS":
                errors.append("P0 replay report did not PASS")
            if p0_replay_report.get("gpu_deleted_before_replay") is not True:
                errors.append("P0 replay did not occur after verified GPU deletion")
            if p0_replay_report.get("all_steps_replayed") is not True:
                errors.append("P0 replay report does not cover every captured step")
            if parent_manifest is not None:
                parent_hash = _sha256_canonical(parent_manifest)
                if p0_replay_report.get("phase_map_manifest_canonical_sha256") != parent_hash:
                    errors.append("P0 replay report is not bound to the cited parent manifest")

    # R0 expected_cells is an explicit materialization of the frozen Cartesian
    # grid. Later kinds are ragged and are checked against their parent above.
    expected_cells, expected_ids = _expected_cell_records(manifest, errors)
    _validate_expected_cell_blocks(manifest, expected_cells, errors)
    required_new_seeds = kind_policy.get("required_new_seeds")
    if isinstance(required_new_seeds, list):
        observed_stage_seeds = {coordinate.seed for coordinate in expected_cells}
        required_stage_seeds = {int(seed) for seed in required_new_seeds}
        if observed_stage_seeds != required_stage_seeds:
            errors.append(
                f"{kind} expected_cells seeds must equal the registered fresh stage "
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
        "eval_source_indices_hash",
        "train_pool_source_indices_hash",
        "eval_hash",
        "eval_example_ids_hash",
        "eval_token_ids_hash",
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
        "split_population_rule": (
            "canonical_source_indices_0_through_train_rows_plus_eval_rows_minus_1"
        ),
        "eval_selection_rule": (
            "python_random_seed_331_shuffle_once_then_final_eval_rows"
        ),
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
    eval_rows = protocol.get("eval_rows")
    if not _is_int(eval_rows) or eval_rows <= 0:
        errors.append("protocol.eval_rows must be a positive integer")

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

    for field in ("git_commit", "image_digest", "model_hash", "data_hash", "eval_hash"):
        if row.get(field) != frozen.get(field):
            errors.append(f"{label}.{field} does not match the frozen manifest")
    for field in ("eval_source_indices_hash", "train_pool_source_indices_hash"):
        if row.get(field) != frozen.get(field):
            errors.append(f"{label}.{field} does not match the frozen manifest")
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

    if status == "COMPLETED":
        for field in ("per_example_loss_uri",):
            _require_bound(row.get(field), f"{label}.{field}", errors)
        _require_sha256(
            row.get("per_example_loss_sha256"),
            f"{label}.per_example_loss_sha256",
            errors,
        )

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
    if status == "COMPLETED":
        if not _finite_number(loss) or float(loss) <= 0.0:
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
        "eval_rows": protocol.get("eval_rows"),
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


def _paired_summary(finals: Mapping[Coordinate, Mapping[str, Any]]) -> list[dict[str, Any]]:
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
                "claim_scope": "paired_development_summary_only",
            }
        )
    return output


def validate_and_summarize(
    manifest: Mapping[str, Any],
    *,
    claim_level: str = "integrity",
    require_bracketed: bool = False,
    parent_manifest: Mapping[str, Any] | None = None,
    p0_replay_report: Mapping[str, Any] | None = None,
    p0_replay_report_sha256: str | None = None,
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
    if not _skip_authority_for_tests:
        _validate_authority_and_lineage(
            manifest,
            errors,
            parent_manifest=parent_manifest,
            p0_replay_report=p0_replay_report,
            p0_replay_report_sha256=p0_replay_report_sha256,
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
        if results:
            errors.append("bound_launch_authority must have an empty results array")
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
            "attempt_count": 0,
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

    kind = _manifest_kind(manifest)
    fresh_confirmation = kind == "fresh_confirmation_stage"
    registered_confirmation_seeds = {
        int(seed)
        for seed in _mapping_or_empty(
            _mapping_or_empty(
                _mapping_or_empty(_authoritative_template().get("lineage_policy")).get(
                    "registered_descendant_kinds"
                )
            ).get("fresh_confirmation_stage")
        ).get("required_new_seeds", [])
    } if fresh_confirmation and not _skip_authority_for_tests else set()
    inference_finals = (
        {
            coordinate: row
            for coordinate, row in finals.items()
            if coordinate.seed in registered_confirmation_seeds
        }
        if fresh_confirmation
        else finals
    )
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
    phase_map = _phase_map_summary(finals, tolerance)
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
        mode == "confirmation"
        and _is_int(configured_min)
        and len(unique_seeds) >= max(HARD_MIN_CONFIRMATORY_SEEDS, configured_min)
        and (fresh_confirmation or not unbracketed)
        and all(
            _canonical_status(row.get("status")) in SCIENTIFIC_STATUSES
            for row in inference_finals.values()
        )
    )
    if claim_level == "confirmatory" and mode == "development":
        errors.append(
            "REFUSED_CONFIRMATORY_CLAIM: development manifests remain development-only, "
            "regardless of eval rows or bootstrap samples"
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
        "paired_development_summaries": _paired_summary(inference_finals),
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
        "--p0-replay-report",
        type=Path,
        help="sealed post-deletion CPU replay report required by initial P1",
    )
    parser.add_argument("--output", type=Path, help="write JSON report here; default stdout")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        replay_report = (
            _load_json(args.p0_replay_report) if args.p0_replay_report else None
        )
        replay_sha = None
        if args.p0_replay_report:
            replay_sha = hashlib.sha256(args.p0_replay_report.read_bytes()).hexdigest()
        report = validate_and_summarize(
            _load_json(args.manifest),
            claim_level=args.claim_level,
            require_bracketed=args.require_bracketed,
            parent_manifest=(
                _load_json(args.parent_manifest) if args.parent_manifest else None
            ),
            p0_replay_report=replay_report,
            p0_replay_report_sha256=replay_sha,
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
