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
    canonical_execution_mode,
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


PARALLEL_DIMS = ("tp", "pp", "cp", "ep")
EDGE_RECOVERY = ("reinit-rollout", "cut-restore", "rebuild-old")
CAPACITY_KEYS = ("gpu_mem_peak_gib", "cpu_rss_peak_gib", "pinned_gib", "object_store_gib", "disk_gib")


@dataclass(frozen=True)
class ResourceConfig:
    """One candidate config. Optional fields (task 1.6) default to the first-round shape."""

    name: str
    trainer: int
    rollout: int
    standby: int = 0
    parallel: tuple[tuple[str, int], ...] = ()  # fixed TP/PP/CP/EP, default all 1
    rollout_engine_gpus: int = 1
    placement: dict[str, Any] | None = None  # explicit GPU uuids per role
    gradient_accumulation_declared: int | None = None
    capacity: dict[str, float] | None = None

    @property
    def total(self) -> int:
        return self.trainer + self.rollout + self.standby

    @property
    def dims(self) -> dict[str, int]:
        return {dim: 1 for dim in PARALLEL_DIMS} | dict(self.parallel)

    @property
    def model_parallel(self) -> int:
        # Dense world = TP*PP*CP*DP; EP is laid out inside that world, not multiplied.
        d = self.dims
        return d["tp"] * d["pp"] * d["cp"]

    @property
    def data_parallel(self) -> int:
        return self.trainer // self.model_parallel

    def gradient_accumulation(self, global_batch: int, micro_batch: int, steps: int = 1) -> int:
        return (global_batch // steps) // (self.data_parallel * micro_batch)


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
    modes = [canonical_execution_mode(m) for m in payload.get("execution_modes", [])]
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
    recovery = edge.get("recovery")
    if recovery is not None and recovery not in EDGE_RECOVERY:
        raise ManifestError(f"edge recovery must be one of {EDGE_RECOVERY}")
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
        configs[name] = ResourceConfig(name, **values, **_parse_config_extras(name, block))
    return configs


def _parse_config_extras(name: str, block: dict[str, Any]) -> dict[str, Any]:
    extras: dict[str, Any] = {}
    parallel = block.get("parallel", {})
    if not isinstance(parallel, dict) or set(parallel) - set(PARALLEL_DIMS):
        raise ManifestError(f"config {name!r}.parallel keys must be among {PARALLEL_DIMS}")
    for dim, size in parallel.items():
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ManifestError(f"config {name!r}.parallel.{dim} must be a positive integer")
    extras["parallel"] = tuple(sorted(parallel.items()))
    engine = block.get("rollout_engine_gpus", 1)
    if not isinstance(engine, int) or isinstance(engine, bool) or engine < 1:
        raise ManifestError(f"config {name!r}.rollout_engine_gpus must be a positive integer")
    extras["rollout_engine_gpus"] = engine
    if "placement" in block:
        placement = block["placement"]
        if not isinstance(placement, dict) or set(placement) - {"trainer", "rollout", "standby"}:
            raise ManifestError(f"config {name!r}.placement needs trainer/rollout/standby lists")
        extras["placement"] = placement
    accum = block.get("gradient_accumulation")
    if accum is not None and (not isinstance(accum, int) or isinstance(accum, bool) or accum < 1):
        raise ManifestError(f"config {name!r}.gradient_accumulation must be a positive integer")
    extras["gradient_accumulation_declared"] = accum
    if "capacity" in block:
        capacity = block["capacity"]
        if not isinstance(capacity, dict) or set(capacity) - set(CAPACITY_KEYS):
            raise ManifestError(f"config {name!r}.capacity keys must be among {CAPACITY_KEYS}")
        for key, value in capacity.items():
            if value is not None and (
                not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0
            ):
                raise ManifestError(f"config {name!r}.capacity.{key} must be >= 0 or null")
        extras["capacity"] = capacity
    return extras


def pool_gpus(resources: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """uuid -> GPU entry for a resolved pool (empty when unresolved)."""
    validate_pool(resources)
    return {gpu["uuid"]: gpu for gpu in resources.get("gpus") or []}


def placement_rejection(config: ResourceConfig, pool: dict[str, dict[str, Any]]) -> str | None:
    """Illegal explicit GPU mapping, or None. Pure; the pool is the manifest's GPU list."""
    placement = config.placement
    if placement is None:
        return None
    if not pool:
        return "explicit placement needs a resolved resources.gpus pool"
    trainer = placement.get("trainer", [])
    engines = placement.get("rollout", [])
    standby = placement.get("standby", [])
    if not all(isinstance(x, list) for x in (trainer, engines, standby)):
        return "placement roles must be lists"
    if not all(isinstance(e, list) and e for e in engines):
        return "placement.rollout must be a list of per-engine GPU lists"
    flat = trainer + [u for e in engines for u in e] + standby
    if not all(isinstance(u, str) for u in flat):
        return "placement entries must be GPU uuid strings"
    dup = sorted({u for u in flat if flat.count(u) > 1})
    if dup:
        return f"GPU mapped to more than one slot: {dup}"
    unknown = sorted(set(flat) - set(pool))
    if unknown:
        return f"GPU not in pool: {unknown}"
    counts = (len(trainer), sum(len(e) for e in engines), len(standby))
    if counts != (config.trainer, config.rollout, config.standby):
        return (
            f"placement maps T{counts[0]} R{counts[1]} S{counts[2]} but config declares "
            f"T{config.trainer} R{config.rollout} S{config.standby}"
        )
    for engine in engines:
        if len(engine) != config.rollout_engine_gpus:
            return f"rollout engine {engine} does not have {config.rollout_engine_gpus} GPUs"
        if len({pool[u].get("node") for u in engine}) > 1:
            return f"rollout engine {engine} spans nodes"
    mp = config.model_parallel
    for start in range(0, len(trainer), mp):
        group = trainer[start : start + mp]
        if len({pool[u].get("node") for u in group}) > 1:
            return f"trainer model-parallel group {group} spans nodes"
    return None


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
    pool: dict[str, dict[str, Any]] | None = None,
) -> str | None:
    """Why this config is illegal for the profile and pool, or None when legal."""
    if config.trainer < 1:
        return "needs at least one trainer GPU"
    if config.rollout < 1:
        return "needs at least one rollout GPU"
    if pool_size is not None and config.total != pool_size:
        return f"uses {config.total} GPUs but the pool has {pool_size}"
    if config.trainer % config.model_parallel:
        return f"trainer GPUs {config.trainer} not divisible by TP*PP*CP {config.model_parallel}"
    if config.rollout % config.rollout_engine_gpus:
        return (
            f"rollout GPUs {config.rollout} not divisible by "
            f"{config.rollout_engine_gpus} GPUs per engine"
        )
    if profile["parameter_mode"] == "full" and config.data_parallel > 1:
        return "dense full-parameter profile requires trainer DP=1"
    steps = profile.get("optimizer_steps_per_round", 1)
    if profile["global_batch"] % steps:
        return f"global_batch {profile['global_batch']} not divisible by {steps} optimizer steps"
    step_batch = profile["global_batch"] // steps
    divisor = config.data_parallel * profile["micro_batch"]
    if step_batch % divisor:
        return (
            f"per-step batch {step_batch} is not divisible by "
            f"DP {config.data_parallel} x micro_batch {profile['micro_batch']}"
        )
    declared = config.gradient_accumulation_declared
    if declared is not None and declared * divisor != step_batch:
        return (
            f"declared gradient_accumulation {declared} x DP {config.data_parallel} x "
            f"micro_batch {profile['micro_batch']} != per-step batch {step_batch}"
        )
    if pool:
        reason = placement_rejection(config, pool)
        if reason:
            return reason
        mem = (config.capacity or {}).get("gpu_mem_peak_gib")
        limits = [g.get("memory_gib") for g in pool.values() if g.get("memory_gib")]
        if mem is not None and limits and mem > min(limits):
            return f"GPU memory peak {mem} GiB exceeds pool GPU memory {min(limits)} GiB"
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
    if source.dims != target.dims:
        raise ManifestError(
            f"edge {edge.source}->{edge.target} changes TP/PP/CP/EP; only DP may change"
        )
    if source.rollout_engine_gpus != target.rollout_engine_gpus:
        raise ManifestError(f"edge {edge.source}->{edge.target} changes the rollout engine shape")
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
    pool: dict[str, dict[str, Any]] | None = None,
    runtime_fingerprint: str | None = None,
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
        reason = config_rejection(configs[name], profile=profile, pool_size=pool_size, pool=pool)
        if reason:
            return STATUS_UNSUPPORTED, f"config {name}: {reason}"
    reason = fingerprint_rejection(runtime_fingerprint, attestation)
    if reason:
        return STATUS_BLOCKED, reason
    if canonical_execution_mode(profile["execution_mode"]) not in attestation.execution_modes:
        return STATUS_BLOCKED, f"runtime has not attested {profile['execution_mode']}"
    if not attestation.partitioned_driver:
        return STATUS_BLOCKED, "runtime has not attested the partitioned driver"
    if kind in FIXED_ARM_KINDS:
        return STATUS_SUPPORTED, None
    return _dynamic_status(arm, edges=edges, attestation=attestation)


def fingerprint_rejection(expected: str | None, attestation: Attestation) -> str | None:
    """The study must pin a runtime fingerprint equal to the attested one (fail closed)."""
    if expected is None or expected == "unresolved":
        return "study runtime fingerprint is unresolved; nothing can be certified against it"
    if attestation.runtime_fingerprint is None:
        return "runtime fingerprint not attested"
    if attestation.runtime_fingerprint != expected:
        return (
            f"unknown runtime fingerprint {attestation.runtime_fingerprint!r} "
            f"(study pins {expected!r})"
        )
    return None


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
    configs: dict[str, ResourceConfig],
    *,
    profile: dict[str, Any],
    pool_size: int | None,
    pool: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Legal table rows record accumulation; illegal rows keep their rejection reason."""
    rows = []
    for config in configs.values():
        reason = config_rejection(config, profile=profile, pool_size=pool_size, pool=pool)
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
                profile["global_batch"],
                profile["micro_batch"],
                profile.get("optimizer_steps_per_round", 1),
            )
        rows.append(row)
    return rows
