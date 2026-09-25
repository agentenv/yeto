"""Matrix expansion and work budget for a study. Pure; never touches a GPU."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from yeto.rl.elastic_benchmark import capabilities as caps
from yeto.rl.elastic_benchmark.evidence import MatrixKey
from yeto.rl.elastic_benchmark.manifest import FIXED_ARM_KINDS


@dataclass(frozen=True)
class MatrixItem:
    key: MatrixKey
    kind: str
    status: str
    reason: str | None
    budget: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class StudyPlan:
    study_hash: str
    items: tuple[MatrixItem, ...]
    config_table: list[dict[str, Any]]
    pool_size: int | None
    attested: bool

    @property
    def runnable(self) -> tuple[MatrixItem, ...]:
        return tuple(item for item in self.items if item.status == caps.STATUS_SUPPORTED)

    def counts(self) -> dict[str, int]:
        counts = {caps.STATUS_SUPPORTED: 0, caps.STATUS_UNSUPPORTED: 0, caps.STATUS_BLOCKED: 0}
        for item in self.items:
            counts[item.status] += 1
        return counts

    def total_budget(self) -> dict[str, int]:
        total = {"updates": 0, "groups": 0, "trajectories": 0, "gpu_updates": 0}
        for item in self.runnable:
            for name in total:
                total[name] += item.budget.get(name, 0)
        return total


def build_plan(manifest: dict[str, Any], attestation: caps.Attestation, *, study_hash: str) -> StudyPlan:
    resources, profile = manifest["resources"], manifest["profile"]
    configs = caps.parse_configs(resources)
    pool_size = caps.validate_pool(resources)
    edges = caps.validate_edges(resources, configs)
    items = []
    for arm in manifest["matrix"]["arms"]:
        status, reason = caps.arm_status(
            arm, profile=profile, configs=configs, edges=edges, attestation=attestation, pool_size=pool_size
        )
        for key in _arm_keys(arm, manifest["matrix"]):
            budget = _item_budget(manifest, arm, key, configs)
            items.append(MatrixItem(key, arm["kind"], status, reason, budget))
    return StudyPlan(
        study_hash=study_hash,
        items=tuple(sorted(items, key=lambda item: item.key)),
        config_table=caps.config_table(configs, profile=profile, pool_size=pool_size),
        pool_size=pool_size,
        attested=attestation.runtime_fingerprint is not None,
    )


def _arm_keys(arm: dict[str, Any], matrix: dict[str, Any]) -> list[MatrixKey]:
    config_axis = arm["configs"] if arm["kind"] == "target-fixed-sweep" else [None]
    keys = []
    for config in config_axis:
        for scenario in matrix["scenarios"]:
            for seed in matrix["seeds"]:
                keys.append(MatrixKey(arm["name"], scenario, int(seed), config))
    return keys


def _item_budget(
    manifest: dict[str, Any], arm: dict[str, Any], key: MatrixKey, configs: dict[str, caps.ResourceConfig]
) -> dict[str, int]:
    work = manifest["work"]
    updates = int(work["update_budget"]) * int(manifest["matrix"].get("repeats", 1))
    groups = updates * int(work["groups_per_update"])
    config_name = key.config or arm.get("config")
    gpus = configs[config_name].total if config_name in configs else 0
    return {
        "updates": updates,
        "groups": groups,
        "trajectories": groups * int(work["samples_per_group"]),
        "gpu_updates": updates * gpus,
    }


def plan_as_dict(plan: StudyPlan) -> dict[str, Any]:
    return {
        "study_hash": plan.study_hash,
        "attested": plan.attested,
        "pool_size": plan.pool_size,
        "counts": plan.counts(),
        "budget": plan.total_budget(),
        "configs": plan.config_table,
        "items": [
            {**item.key.as_dict(), "kind": item.kind, "status": item.status, "reason": item.reason, "budget": item.budget}
            for item in plan.items
        ],
    }


def render_plan(plan: StudyPlan) -> str:
    lines = [f"study {plan.study_hash[:12]}  pool={plan.pool_size or 'unresolved'}  attested={plan.attested}"]
    lines.append("configs:")
    for row in plan.config_table:
        verdict = f"accum={row['gradient_accumulation']}" if row["legal"] else f"REJECTED: {row['reason']}"
        lines.append(f"  {row['config']:<6} T{row['trainer']} R{row['rollout']} S{row['standby']}  {verdict}")
    lines.append("matrix:")
    for item in plan.items:
        label = item.key.arm if item.key.config is None else f"{item.key.arm}@{item.key.config}"
        suffix = "" if item.reason is None else f"  ({item.reason})"
        lines.append(f"  {item.status:<18} {label:<18} {item.key.scenario:<12} seed={item.key.seed}{suffix}")
    counts, budget = plan.counts(), plan.total_budget()
    lines.append(
        f"runnable={counts['supported']} unsupported={counts['unsupported']} "
        f"blocked_dependency={counts['blocked_dependency']}"
    )
    lines.append(
        f"runnable budget: {budget['updates']} updates, {budget['groups']} groups, "
        f"{budget['trajectories']} trajectories, {budget['gpu_updates']} gpu-updates"
    )
    fixed = sum(1 for item in plan.runnable if item.kind in FIXED_ARM_KINDS)
    lines.append(f"fixed baseline items runnable now: {fixed}")
    return "\n".join(lines)
