"""Versioned study manifest: field groups, canonical hash and mode validation.

A manifest freezes one study. ``calibration`` manifests may leave resources and
quality rules unresolved; ``formal`` manifests must resolve everything before a
single GPU is touched. Validation never rewrites a manifest: an illegal or
contradictory manifest is rejected with the reason, not silently repaired.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from yeto.provenance import is_immutable_commit

SCHEMA_VERSION = 1
MODES = ("calibration", "formal")
FIELD_GROUPS = (
    "identity",
    "profile",
    "resources",
    "matrix",
    "work",
    "evaluation",
    "timing",
)
PARAMETER_MODES = ("lora", "full")
EXECUTION_MODES = ("colocated-serial", "partitioned-serial", "partitioned-overlap")
OUTER_PROTOCOLS = ("none", "strict-avg", "decoupled")
ARM_KINDS = (
    "legacy-fixed",
    "target-fixed-default",
    "target-fixed-sweep",
    "scheduled-rebuild",
    "scheduled-optimized",
    "auto",
)
FIXED_ARM_KINDS = ARM_KINDS[:3]
SCENARIOS = ("stable", "phased", "tail", "tool-wait", "oscillating")
MEASUREMENT_LEVELS = ("component", "single-turn", "agent")
EDGE_KINDS = ("rollout-only", "same-shape-restore", "trainer-dp", "role-transfer", "standby-scale")
DEFAULT_SEEDS = (17, 29, 43)
_UNRESOLVED = "unresolved"


class ManifestError(ValueError):
    """A manifest is malformed, contradictory or not frozen enough for its mode."""


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def manifest_hash(manifest: dict[str, Any]) -> str:
    """Content hash over every field group; ``study_hash`` itself is excluded."""
    body = {key: value for key, value in manifest.items() if key != "study_hash"}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read study manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ManifestError("study manifest must be a JSON object")
    validate_manifest(manifest)
    recorded = manifest.get("study_hash")
    if recorded is not None and recorded != manifest_hash(manifest):
        raise ManifestError("study manifest content does not match its recorded study_hash")
    return manifest


def dump_manifest(manifest: dict[str, Any], path: Path) -> str:
    """Validate, stamp ``study_hash`` and write atomically. Returns the hash."""
    validate_manifest(manifest)
    stamped = {key: value for key, value in manifest.items() if key != "study_hash"}
    stamped["study_hash"] = manifest_hash(stamped)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(stamped, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return stamped["study_hash"]


def validate_manifest(manifest: dict[str, Any]) -> None:
    _require_shape(manifest)
    mode = manifest["mode"]
    _validate_identity(manifest["identity"], mode)
    _validate_profile(manifest["profile"])
    _validate_matrix(manifest["matrix"])
    _validate_work(manifest["work"], manifest["profile"])
    _validate_evaluation(manifest["evaluation"], mode)
    _validate_timing(manifest["timing"], manifest["work"])
    if mode == "formal":
        _reject_unresolved(manifest)


def _require_shape(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(f"study manifest schema_version must be {SCHEMA_VERSION}")
    if manifest.get("mode") not in MODES:
        raise ManifestError(f"study manifest mode must be one of {MODES}")
    for group in FIELD_GROUPS:
        if not isinstance(manifest.get(group), dict):
            raise ManifestError(f"study manifest is missing the {group!r} field group")


def _validate_identity(identity: dict[str, Any], mode: str) -> None:
    if not isinstance(identity.get("study_id"), str) or not identity["study_id"]:
        raise ManifestError("identity.study_id must be a non-empty string")
    model = identity.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("id"), str):
        raise ManifestError("identity.model.id is required")
    fingerprints = identity.get("fingerprints")
    if not isinstance(fingerprints, dict):
        raise ManifestError("identity.fingerprints must be an object")
    if mode != "formal":
        return
    for label, block in (("model", model), ("data", identity.get("data") or {})):
        revision = block.get("revision")
        if not isinstance(revision, str) or not is_immutable_commit(revision):
            raise ManifestError(f"formal study requires an immutable identity.{label}.revision")
    for key in ("reward", "source", "runtime"):
        if not isinstance(fingerprints.get(key), str) or not fingerprints[key]:
            raise ManifestError(f"formal study requires identity.fingerprints.{key}")


def _validate_profile(profile: dict[str, Any]) -> None:
    if profile.get("parameter_mode") not in PARAMETER_MODES:
        raise ManifestError(f"profile.parameter_mode must be one of {PARAMETER_MODES}")
    if profile.get("execution_mode") not in EXECUTION_MODES:
        raise ManifestError(f"profile.execution_mode must be one of {EXECUTION_MODES}")
    if profile.get("outer_protocol") not in OUTER_PROTOCOLS:
        raise ManifestError(f"profile.outer_protocol must be one of {OUTER_PROTOCOLS}")
    for key in ("global_batch", "micro_batch", "optimizer_steps_per_round"):
        if not _positive_int(profile.get(key)):
            raise ManifestError(f"profile.{key} must be a positive integer")
    if profile["global_batch"] % profile["micro_batch"]:
        raise ManifestError("profile.micro_batch must divide profile.global_batch")
    age = profile.get("max_policy_age", 0)
    if not isinstance(age, int) or age < 0:
        raise ManifestError("profile.max_policy_age must be a non-negative integer")
    if profile["execution_mode"] != "partitioned-overlap" and age != 0:
        raise ManifestError("only partitioned-overlap profiles may allow a non-zero policy age")
    for key in ("publish_rule", "reset_rule", "loss_normalization"):
        if not isinstance(profile.get(key), str) or not profile[key]:
            raise ManifestError(f"profile.{key} must be a non-empty string")


def _validate_matrix(matrix: dict[str, Any]) -> None:
    arms = matrix.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ManifestError("matrix.arms must be a non-empty list")
    names = [arm.get("name") for arm in arms if isinstance(arm, dict)]
    if len(names) != len(arms) or len(set(names)) != len(names) or not all(names):
        raise ManifestError("matrix.arms entries need unique non-empty names")
    for arm in arms:
        _validate_arm(arm)
    scenarios = matrix.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ManifestError("matrix.scenarios must be a non-empty list")
    unknown = sorted(set(scenarios) - set(SCENARIOS))
    if unknown or len(set(scenarios)) != len(scenarios):
        raise ManifestError(f"matrix.scenarios must be distinct entries of {SCENARIOS}")
    seeds = matrix.get("seeds")
    if not isinstance(seeds, list) or not seeds or not all(_positive_int(s) for s in seeds):
        raise ManifestError("matrix.seeds must be a non-empty list of positive integers")
    if len(set(seeds)) != len(seeds):
        raise ManifestError("matrix.seeds contains duplicates")
    if not _positive_int(matrix.get("repeats", 1)):
        raise ManifestError("matrix.repeats must be a positive integer")
    if matrix.get("measurement_level") not in MEASUREMENT_LEVELS:
        raise ManifestError(f"matrix.measurement_level must be one of {MEASUREMENT_LEVELS}")


def _validate_arm(arm: dict[str, Any]) -> None:
    kind = arm.get("kind")
    if kind not in ARM_KINDS:
        raise ManifestError(f"arm {arm.get('name')!r} has unknown kind {kind!r}")
    if kind == "target-fixed-sweep":
        configs = arm.get("configs")
        if not isinstance(configs, list) or not configs:
            raise ManifestError(f"sweep arm {arm['name']!r} needs a non-empty configs list")
    elif kind in ("scheduled-rebuild", "scheduled-optimized"):
        plan = arm.get("switch_plan")
        if not isinstance(plan, list) or not plan:
            raise ManifestError(f"scheduled arm {arm['name']!r} needs a switch_plan")
        for step in plan:
            if not isinstance(step, dict) or not _positive_int(step.get("at_update")):
                raise ManifestError(f"arm {arm['name']!r} switch_plan entries need at_update")
            if not step.get("target"):
                raise ManifestError(f"arm {arm['name']!r} switch_plan entries need a target")
        if not arm.get("config"):
            raise ManifestError(f"scheduled arm {arm['name']!r} needs an initial config")
    elif kind == "auto":
        candidates = arm.get("candidates")
        if not isinstance(candidates, list) or len(candidates) < 2:
            raise ManifestError(f"auto arm {arm['name']!r} needs at least two candidates")
        if not arm.get("config"):
            raise ManifestError(f"auto arm {arm['name']!r} needs an initial config")
    elif not arm.get("config"):
        raise ManifestError(f"fixed arm {arm['name']!r} needs a config")


def _validate_work(work: dict[str, Any], profile: dict[str, Any]) -> None:
    for key in ("groups_per_update", "samples_per_group", "update_budget", "max_response_len"):
        if not _positive_int(work.get(key)):
            raise ManifestError(f"work.{key} must be a positive integer")
    if work["groups_per_update"] * work["samples_per_group"] != profile["global_batch"]:
        raise ManifestError(
            "work.groups_per_update * work.samples_per_group must equal profile.global_batch"
        )
    phases = work.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ManifestError("work.phases must be a non-empty list")
    total = 0
    for phase in phases:
        if not isinstance(phase, dict) or not phase.get("name"):
            raise ManifestError("work.phases entries need a name")
        if not _positive_int(phase.get("updates")):
            raise ManifestError(f"phase {phase.get('name')!r} needs a positive updates count")
        total += phase["updates"]
    if total != work["update_budget"]:
        raise ManifestError(
            f"work.update_budget ({work['update_budget']}) contradicts the phase sum ({total})"
        )
    for key in ("truncation", "filter", "retry"):
        if not isinstance(work.get(key), str) or not work[key]:
            raise ManifestError(f"work.{key} must name a rule")


def _validate_evaluation(evaluation: dict[str, Any], mode: str) -> None:
    splits = evaluation.get("splits")
    if not isinstance(splits, dict):
        raise ManifestError("evaluation.splits must be an object")
    for key in ("train", "calibration", "test", "held_out"):
        if not isinstance(splits.get(key), int) or splits[key] < 0:
            raise ManifestError(f"evaluation.splits.{key} must be a non-negative integer")
    if splits["train"] < 1 or splits["held_out"] < 1:
        raise ManifestError("evaluation.splits needs at least one train and one held_out row")
    if not _positive_int(evaluation.get("eval_seed")):
        raise ManifestError("evaluation.eval_seed must be a positive integer")
    points = evaluation.get("eval_points")
    if not isinstance(points, list) or not all(_positive_int(p) for p in points):
        raise ManifestError("evaluation.eval_points must be a list of positive update indexes")
    if points != sorted(set(points)):
        raise ManifestError("evaluation.eval_points must be strictly increasing")
    if mode == "formal":
        _validate_formal_quality_rules(evaluation)


def _validate_formal_quality_rules(evaluation: dict[str, Any]) -> None:
    metrics = evaluation.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ManifestError("formal study requires evaluation.metrics with predeclared bounds")
    for metric in metrics:
        if not isinstance(metric, dict) or metric.get("direction") not in ("higher", "lower"):
            raise ManifestError("each evaluation metric needs name and direction higher|lower")
        for key in ("delta_quality", "catastrophic"):
            if not _non_negative_number(metric.get(key)):
                raise ManifestError(f"metric {metric.get('name')!r} needs numeric {key}")
    for key in ("delta_stable", "min_speedup"):
        if not _non_negative_number(evaluation.get(key)):
            raise ManifestError(f"formal study requires numeric evaluation.{key}")
    statistics = evaluation.get("statistics")
    if not isinstance(statistics, dict) or not statistics.get("method"):
        raise ManifestError("formal study requires evaluation.statistics.method")
    if not (0 < float(statistics.get("confidence", 0)) < 1):
        raise ManifestError("evaluation.statistics.confidence must be in (0, 1)")


def _validate_timing(timing: dict[str, Any], work: dict[str, Any]) -> None:
    warmup = timing.get("warmup_updates", 0)
    if not isinstance(warmup, int) or warmup < 0:
        raise ManifestError("timing.warmup_updates must be a non-negative integer")
    measured = timing.get("measured_updates")
    if not _positive_int(measured):
        raise ManifestError("timing.measured_updates must be a positive integer")
    if warmup + measured > work["update_budget"]:
        raise ManifestError("timing.warmup_updates + measured_updates exceed work.update_budget")
    if not _positive_int(timing.get("timeout_s")):
        raise ManifestError("timing.timeout_s must be a positive integer")
    order = timing.get("arm_order")
    if order not in ("declared", "rotated", "seeded-random"):
        raise ManifestError("timing.arm_order must be declared|rotated|seeded-random")
    cost = timing.get("cost", {})
    if not isinstance(cost, dict):
        raise ManifestError("timing.cost must be an object")
    price = cost.get("gpu_hour_cost")
    if price is not None and not _non_negative_number(price):
        raise ManifestError("timing.cost.gpu_hour_cost must be a number or null (unknown)")


def _reject_unresolved(manifest: dict[str, Any]) -> None:
    unresolved = sorted(_unresolved_paths(manifest))
    if unresolved:
        detail = ", ".join(unresolved[:8])
        raise ManifestError(f"formal study still has unresolved fields: {detail}")
    resources = manifest["resources"]
    gpus = resources.get("gpus")
    if not isinstance(gpus, list) or not gpus:
        raise ManifestError("formal study requires resources.gpus with physical UUIDs")
    if not resources.get("pool_id") or not _positive_int(resources.get("pool_epoch")):
        raise ManifestError("formal study requires resources.pool_id and pool_epoch")


def _unresolved_paths(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        found = []
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            found.extend(_unresolved_paths(item, name))
        return found
    if isinstance(value, list):
        return [p for i, item in enumerate(value) for p in _unresolved_paths(item, f"{prefix}[{i}]")]
    if value == _UNRESOLVED:
        return [prefix or "root"]
    return []


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _non_negative_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0


def example_manifest(mode: str = "calibration") -> dict[str, Any]:
    """The documented starting point: Qwen3-4B LoRA, GRPO, strict-avg, P62/P44/P26."""
    formal = mode == "formal"
    revision = "0" * 40 if formal else _UNRESOLVED
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "identity": {
            "study_id": "elastic-calibration-example",
            "created_at": "1970-01-01T00:00:00Z",
            "model": {"id": "Qwen/Qwen3-4B", "revision": revision},
            "tokenizer": {"id": "Qwen/Qwen3-4B", "revision": revision},
            "data": {"id": "org/prompt-dataset", "revision": revision},
            "initial_policy": {"kind": "base"},
            "fingerprints": {
                "reward": "sha256:" + "0" * 64 if formal else _UNRESOLVED,
                "source": "sha256:" + "0" * 64 if formal else _UNRESOLVED,
                "runtime": "sha256:" + "0" * 64 if formal else _UNRESOLVED,
                "image": None,
            },
        },
        "profile": {
            "name": "lora-grpo-strict-avg-partitioned-serial",
            "parameter_mode": "lora",
            "algorithm": "grpo",
            "outer_protocol": "strict-avg",
            "execution_mode": "partitioned-serial",
            "max_policy_age": 0,
            "publish_rule": "after-every-update",
            "reset_rule": "strict-avg-moment-reset",
            "global_batch": 48,
            "micro_batch": 1,
            "optimizer_steps_per_round": 1,
            "packing": False,
            "loss_normalization": "token-mean",
            "max_in_flight_groups": 12,
        },
        "resources": {
            "pool_id": "example-pool" if formal else _UNRESOLVED,
            "pool_epoch": 1,
            "gpus": (
                [{"uuid": f"GPU-{i:08d}", "model": "H100", "node": "n0", "index": i} for i in range(8)]
                if formal
                else []
            ),
            "interconnect": "nvlink",
            "configs": {
                "P62": {"trainer": 6, "rollout": 2, "standby": 0},
                "P44": {"trainer": 4, "rollout": 4, "standby": 0},
                "P26": {"trainer": 2, "rollout": 6, "standby": 0},
                "P422": {"trainer": 4, "rollout": 2, "standby": 2},
            },
            "edges": [
                {"source": "P422", "target": "P44", "kind": "rollout-only"},
                {"source": "P44", "target": "P422", "kind": "rollout-only"},
                {"source": "P62", "target": "P44", "kind": "role-transfer"},
                {"source": "P44", "target": "P62", "kind": "role-transfer"},
            ],
            "lease": None,
        },
        "matrix": {
            "arms": [
                {"name": "legacy", "kind": "legacy-fixed", "config": "colocated"},
                {"name": "default", "kind": "target-fixed-default", "config": "P44"},
                {"name": "sweep", "kind": "target-fixed-sweep", "configs": ["P62", "P44", "P26"]},
                {
                    "name": "rebuild",
                    "kind": "scheduled-rebuild",
                    "config": "P62",
                    "switch_plan": [
                        {"at_update": 5, "target": "P44"},
                        {"at_update": 9, "target": "P62"},
                    ],
                },
                {
                    "name": "optimized",
                    "kind": "scheduled-optimized",
                    "config": "P62",
                    "switch_plan": [
                        {"at_update": 5, "target": "P44"},
                        {"at_update": 9, "target": "P62"},
                    ],
                },
                {"name": "auto", "kind": "auto", "config": "P44", "candidates": ["P62", "P44"]},
            ],
            "scenarios": ["stable", "phased"],
            "seeds": list(DEFAULT_SEEDS),
            "repeats": 1,
            "measurement_level": "single-turn",
        },
        "work": {
            "groups_per_update": 12,
            "samples_per_group": 4,
            "update_budget": 12,
            "max_response_len": 1024,
            "phases": [
                {"name": "A", "updates": 4, "mix": {"short": 0.8, "long": 0.2}},
                {"name": "B", "updates": 4, "mix": {"short": 0.2, "long": 0.8}},
                {"name": "A2", "updates": 4, "mix": {"short": 0.8, "long": 0.2}},
            ],
            "truncation": "hard-cap",
            "filter": "none",
            "retry": "none",
        },
        "evaluation": {
            "splits": {"train": 96, "calibration": 16, "test": 16, "held_out": 32},
            "eval_seed": 7,
            "eval_points": [4, 8, 12],
            "metrics": [
                {"name": "held_out_reward", "direction": "higher", "delta_quality": 0.02, "catastrophic": 0.1}
            ]
            if formal
            else [],
            "delta_stable": 0.05 if formal else _UNRESOLVED,
            "min_speedup": 1.1 if formal else _UNRESOLVED,
            "statistics": {"method": "paired-bootstrap", "confidence": 0.95},
        },
        "timing": {
            "warmup_updates": 2,
            "measured_updates": 10,
            "timeout_s": 7200,
            "arm_order": "rotated",
            "cost": {"gpu_hour_cost": None},
        },
    }
