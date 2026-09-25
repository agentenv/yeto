"""Resource configs, directed edges, physical pools and runtime attestation.

Everything here is pure validation. A config that fails is reported with its
reason and never replaced by a neighbouring legal config; an edge that the
runtime has not attested is ``blocked_dependency``, never silently downgraded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yeto.rl.elastic_benchmark.manifest import (
    ARM_KINDS,
    EDGE_KINDS,
    EXECUTION_MODES,
    FIXED_ARM_KINDS,
    ManifestError,
)

LEGACY_CONFIG = "colocated"
STATUS_SUPPORTED = "supported"
STATUS_UNSUPPORTED = "unsupported"
STATUS_BLOCKED = "blocked_dependency"
_ARM_EDGE_KINDS = {
    "scheduled-rebuild": ("rollout-only", "trainer-dp", "role-transfer", "standby-scale"),
    "scheduled-optimized": ("rollout-only", "trainer-dp", "role-transfer", "standby-scale"),
    "auto": ("rollout-only", "trainer-dp", "role-transfer", "standby-scale"),
}


@dataclass(frozen=True)
class ResourceConfig:
    name: str
    trainer: int
    rollout: int
    standby: int = 0

    @property
    def total(self) -> int:
        return self.trainer + self.rollout + self.standby

    @property
    def data_parallel(self) -> int:
        # First round certifies TP=PP=CP=EP=1, so trainer GPUs are the DP size.
        return self.trainer

    def gradient_accumulation(self, global_batch: int, micro_batch: int) -> int:
        return global_batch // (self.data_parallel * micro_batch)


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    kind: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.source, self.target, self.kind)


@dataclass(frozen=True)
class Attestation:
    """What the runtime says it can do. Absent attestation means nothing is certified."""

    runtime_fingerprint: str | None
    execution_modes: frozenset[str]
    certified_edges: frozenset[tuple[str, str, str]]
    optimized_paths: frozenset[str]
    auto_controller: bool
    partitioned_driver: bool

    @staticmethod
    def none() -> "Attestation":
        return Attestation(None, frozenset(), frozenset(), frozenset(), False, False)


def load_attestation(path: Path | None) -> Attestation:
    if path is None:
        return Attestation.none()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read capability attestation {path}: {exc}") from exc
    return attestation_from_dict(payload)


def attestation_from_dict(payload: dict[str, Any]) -> Attestation:
    if not isinstance(payload, dict):
        raise ManifestError("capability attestation must be a JSON object")
    modes = payload.get("execution_modes", [])
    unknown = sorted(set(modes) - set(EXECUTION_MODES))
    if unknown:
        raise ManifestError(f"attestation lists unknown execution modes: {unknown}")
    edges = []
    for edge in payload.get("certified_edges", []):
        parsed = parse_edge(edge)
        edges.append(parsed.key)
    fingerprint = payload.get("runtime_fingerprint")
    if fingerprint is not None and not isinstance(fingerprint, str):
        raise ManifestError("attestation runtime_fingerprint must be a string")
    return Attestation(
        runtime_fingerprint=fingerprint,
        execution_modes=frozenset(modes),
        certified_edges=frozenset(edges),
        optimized_paths=frozenset(payload.get("optimized_paths", [])),
        auto_controller=bool(payload.get("auto_controller", False)),
        partitioned_driver=bool(payload.get("partitioned_driver", False)),
    )


def parse_edge(edge: dict[str, Any]) -> Edge:
    if not isinstance(edge, dict):
        raise ManifestError("edges must be objects with source, target and kind")
    source, target, kind = edge.get("source"), edge.get("target"), edge.get("kind")
    if not source or not target or source == target:
        raise ManifestError(f"edge needs distinct source and target: {edge}")
    if kind not in EDGE_KINDS:
        raise ManifestError(f"edge {source}->{target} has unknown kind {kind!r}")
    return Edge(str(source), str(target), str(kind))


def parse_configs(resources: dict[str, Any]) -> dict[str, ResourceConfig]:
    raw = resources.get("configs")
    if not isinstance(raw, dict) or not raw:
        raise ManifestError("resources.configs must be a non-empty object")
    configs = {}
    for name, block in raw.items():
        if name == LEGACY_CONFIG:
            raise ManifestError(f"{LEGACY_CONFIG!r} is reserved for the legacy colocated arm")
        if not isinstance(block, dict):
            raise ManifestError(f"config {name!r} must be an object")
        values = {}
        for role in ("trainer", "rollout", "standby"):
            count = block.get(role, 0)
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ManifestError(f"config {name!r}.{role} must be a non-negative integer")
            values[role] = count
        configs[name] = ResourceConfig(name, **values)
    return configs


def validate_pool(resources: dict[str, Any]) -> int | None:
    """Return the physical pool size, or None when the pool is still unresolved."""
    gpus = resources.get("gpus")
    if not isinstance(gpus, list):
        raise ManifestError("resources.gpus must be a list")
    if not gpus:
        return None
    seen = set()
    for gpu in gpus:
        if not isinstance(gpu, dict) or not isinstance(gpu.get("uuid"), str):
            raise ManifestError("each resources.gpus entry needs a string uuid")
        if gpu["uuid"] in seen:
            raise ManifestError(f"duplicate GPU uuid in resources.gpus: {gpu['uuid']}")
        seen.add(gpu["uuid"])
    return len(gpus)


def config_rejection(
    config: ResourceConfig,
    *,
    profile: dict[str, Any],
    pool_size: int | None,
) -> str | None:
    """Why this config is illegal for the profile and pool, or None when legal."""
    if config.trainer < 1:
        return "needs at least one trainer GPU"
    if config.rollout < 1:
        return "needs at least one rollout GPU"
    if pool_size is not None and config.total != pool_size:
        return f"uses {config.total} GPUs but the pool has {pool_size}"
    if profile["parameter_mode"] == "full" and config.data_parallel > 1:
        return "dense full-parameter profile requires trainer DP=1"
    divisor = config.data_parallel * profile["micro_batch"]
    if profile["global_batch"] % divisor:
        return (
            f"global_batch {profile['global_batch']} is not divisible by "
            f"DP {config.data_parallel} x micro_batch {profile['micro_batch']}"
        )
    return None


def validate_edges(resources: dict[str, Any], configs: dict[str, ResourceConfig]) -> list[Edge]:
    edges = [parse_edge(edge) for edge in resources.get("edges", [])]
    keys = [edge.key for edge in edges]
    if len(set(keys)) != len(keys):
        raise ManifestError("resources.edges contains duplicate edges")
    for edge in edges:
        for endpoint in (edge.source, edge.target):
            if endpoint not in configs:
                raise ManifestError(f"edge references unknown config {endpoint!r}")
        _check_edge_shape(edge, configs[edge.source], configs[edge.target])
    return edges


def _check_edge_shape(edge: Edge, source: ResourceConfig, target: ResourceConfig) -> None:
    trainer_changes = source.trainer != target.trainer
    if edge.kind in ("rollout-only", "standby-scale") and trainer_changes:
        raise ManifestError(f"{edge.kind} edge {edge.source}->{edge.target} changes trainer count")
    if edge.kind == "same-shape-restore" and trainer_changes:
        raise ManifestError(f"same-shape-restore edge {edge.source}->{edge.target} changes DP")
    if edge.kind in ("trainer-dp", "role-transfer") and not trainer_changes:
        raise ManifestError(f"{edge.kind} edge {edge.source}->{edge.target} keeps trainer count")


def arm_status(
    arm: dict[str, Any],
    *,
    profile: dict[str, Any],
    configs: dict[str, ResourceConfig],
    edges: list[Edge],
    attestation: Attestation,
    pool_size: int | None,
) -> tuple[str, str | None]:
    """Classify an arm as supported / unsupported / blocked_dependency with a reason."""
    kind = arm["kind"]
    if kind not in ARM_KINDS:
        return STATUS_UNSUPPORTED, f"unknown arm kind {kind!r}"
    if kind == "legacy-fixed":
        return _legacy_status(arm)
    for name in _arm_config_names(arm):
        if name not in configs:
            return STATUS_UNSUPPORTED, f"unknown config {name!r}"
        reason = config_rejection(configs[name], profile=profile, pool_size=pool_size)
        if reason:
            return STATUS_UNSUPPORTED, f"config {name}: {reason}"
    if profile["execution_mode"] not in attestation.execution_modes:
        return STATUS_BLOCKED, f"runtime has not attested {profile['execution_mode']}"
    if not attestation.partitioned_driver:
        return STATUS_BLOCKED, "runtime has not attested the partitioned driver"
    if kind in FIXED_ARM_KINDS:
        return STATUS_SUPPORTED, None
    return _dynamic_status(arm, edges=edges, attestation=attestation)


def _legacy_status(arm: dict[str, Any]) -> tuple[str, str | None]:
    if arm.get("config") != LEGACY_CONFIG:
        return STATUS_UNSUPPORTED, f"legacy-fixed arm must use config {LEGACY_CONFIG!r}"
    return STATUS_SUPPORTED, None


def _dynamic_status(
    arm: dict[str, Any], *, edges: list[Edge], attestation: Attestation
) -> tuple[str, str | None]:
    declared = {(e.source, e.target): e for e in edges}
    for source, target in _arm_transitions(arm):
        edge = declared.get((source, target))
        if edge is None:
            return STATUS_UNSUPPORTED, f"no declared edge {source}->{target}"
        if edge.key not in attestation.certified_edges:
            return STATUS_BLOCKED, f"edge {source}->{target} ({edge.kind}) is not certified"
    if arm["kind"] == "scheduled-optimized" and not attestation.optimized_paths:
        return STATUS_BLOCKED, "no optimized migration path is certified"
    if arm["kind"] == "auto" and not attestation.auto_controller:
        return STATUS_BLOCKED, "auto controller is not certified"
    return STATUS_SUPPORTED, None


def _arm_config_names(arm: dict[str, Any]) -> list[str]:
    names = []
    if arm.get("config"):
        names.append(arm["config"])
    names.extend(arm.get("configs", []))
    names.extend(arm.get("candidates", []))
    names.extend(step["target"] for step in arm.get("switch_plan", []))
    return list(dict.fromkeys(names))


def _arm_transitions(arm: dict[str, Any]) -> list[tuple[str, str]]:
    if arm["kind"] == "auto":
        candidates = list(arm["candidates"])
        return [(a, b) for a in candidates for b in candidates if a != b]
    current = arm["config"]
    transitions = []
    for step in arm.get("switch_plan", []):
        transitions.append((current, step["target"]))
        current = step["target"]
    return transitions


def config_table(
    configs: dict[str, ResourceConfig], *, profile: dict[str, Any], pool_size: int | None
) -> list[dict[str, Any]]:
    """Legal table rows record accumulation; illegal rows keep their rejection reason."""
    rows = []
    for config in configs.values():
        reason = config_rejection(config, profile=profile, pool_size=pool_size)
        row = {
            "config": config.name,
            "trainer": config.trainer,
            "rollout": config.rollout,
            "standby": config.standby,
            "data_parallel": config.data_parallel,
            "legal": reason is None,
            "reason": reason,
        }
        if reason is None:
            row["gradient_accumulation"] = config.gradient_accumulation(
                profile["global_batch"], profile["micro_batch"]
            )
        rows.append(row)
    return rows
