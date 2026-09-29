"""``EngineCapabilities``: startup capability declaration and handshake.

Serialization is the runtime capability attestation read by #66
(``yeto/rl/elastic_benchmark/capabilities.py::attestation_from_dict``):
top-level keys ``runtime_fingerprint``, ``execution_modes``,
``certified_edges``, ``optimized_paths``, ``auto_controller`` and
``partitioned_driver``. Engine-port fields (``schema``, ``engine``,
``parameter_layouts`` ...) are extra top-level keys, which that reader
ignores. Execution-mode and edge-kind vocabularies are imported from #66's
``elastic_benchmark.manifest`` so there is one capability schema (design D8b);
the driver uses the same names (``colocated-serial`` ...).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..elastic_benchmark.manifest import EDGE_KINDS, EXECUTION_MODES, canonical_execution_mode
from .trainable_state import TRAINABLE_LAYOUTS

CAPABILITY_SCHEMA = "yeto-rl-engine-capabilities-v1"

# Mechanism dimensions beyond R0's advantage_estimators/dynamic_sampling_filters
# (change rl-algorithm-capabilities, design D4). Absent keys in an older
# declaration read as the R0 mechanism set, so R0 declarations keep their
# meaning. They are extra top-level attestation keys, ignored by #66's reader.
R0_MECHANISMS: dict[str, frozenset[str]] = {
    "losses": frozenset({"policy_loss"}),
    "loss_aggregations": frozenset({"default"}),
    "kl_placements": frozenset({"none", "reward"}),
    "corrections": frozenset({"none"}),
    "reward_postprocessors": frozenset(),
    "features": frozenset(),
}

PLACEMENTS = frozenset({"colocated", "fixed-partition"})
# E1-E3 verbs reserved on the ports (design D9); declared only when implemented.
RESERVED_PORT_VERBS = frozenset(
    {
        "RolloutPool.add_engines",
        "RolloutPool.remove_engines",
        "TrainerGroup.save_cut",
        "TrainerGroup.restore_cut",
        "Placement.reconfigure",
    }
)
_FINGERPRINT = re.compile(r"sha256:[0-9a-f]{64}")


class CapabilityMismatch(ValueError):
    """Requested configuration is outside the declared engine capabilities."""


@dataclass(frozen=True)
class ExecutionCapabilities:
    """What the execution mode of this engine can provide (design D4, A1).

    ``max_policy_staleness`` is the largest policy age the declared execution
    mode may produce (rl-infra-spec ``ExecutionProfile.max_policy_age``);
    serial-colocated and partitioned-serial produce 0.
    """

    critic: bool = False
    max_policy_staleness: int = 0
    rollout_logprobs: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.critic, bool) or not isinstance(self.rollout_logprobs, bool):
            raise ValueError("execution.critic/rollout_logprobs must be booleans")
        age = self.max_policy_staleness
        if isinstance(age, bool) or not isinstance(age, int) or age < 0:
            raise ValueError("execution.max_policy_staleness must be a non-negative int")

    def to_dict(self) -> dict[str, Any]:
        return {
            "critic": self.critic,
            "max_policy_staleness": self.max_policy_staleness,
            "rollout_logprobs": self.rollout_logprobs,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "ExecutionCapabilities":
        if isinstance(payload, ExecutionCapabilities):
            return payload
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise ValueError("execution capabilities must be an object")
        return cls(
            critic=payload.get("critic", False),
            max_policy_staleness=payload.get("max_policy_staleness", 0),
            rollout_logprobs=payload.get("rollout_logprobs", False),
        )


def _names(values: Iterable[str], allowed: Iterable[str] | None, what: str) -> frozenset[str]:
    if isinstance(values, str):
        raise TypeError(f"{what} must be a collection of strings")
    out = frozenset(values)
    if allowed is not None:
        unknown = sorted(out - frozenset(allowed))
        if unknown:
            raise ValueError(f"unknown {what}: {unknown}")
    return out


@dataclass(frozen=True)
class EngineCapabilities:
    engine: str
    runtime_fingerprint: str
    parameter_layouts: frozenset[str]
    placements: frozenset[str]
    advantage_estimators: frozenset[str]
    dynamic_sampling_filters: frozenset[str]
    execution_modes: frozenset[str]
    port_verbs: frozenset[str] = frozenset()
    certified_edges: frozenset[tuple[str, str, str]] = frozenset()
    optimized_paths: frozenset[str] = frozenset()
    auto_controller: bool = False
    partitioned_driver: bool = False
    losses: frozenset[str] = R0_MECHANISMS["losses"]
    loss_aggregations: frozenset[str] = R0_MECHANISMS["loss_aggregations"]
    kl_placements: frozenset[str] = R0_MECHANISMS["kl_placements"]
    corrections: frozenset[str] = R0_MECHANISMS["corrections"]
    reward_postprocessors: frozenset[str] = R0_MECHANISMS["reward_postprocessors"]
    features: frozenset[str] = R0_MECHANISMS["features"]
    execution: ExecutionCapabilities = ExecutionCapabilities()
    # Mechanisms admitted for a single-island smoke without being declared
    # (--rl-allow-unverified-mechanism, design D11). Recorded in the
    # attestation so every artifact shows it contains unverified mechanisms.
    unverified_mechanisms: frozenset[str] = frozenset()
    extra: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not self.engine:
            raise ValueError("engine name is required")
        if not _FINGERPRINT.fullmatch(self.runtime_fingerprint):
            raise ValueError("runtime_fingerprint must be 'sha256:<64 hex>'")
        s = object.__setattr__
        s(self, "parameter_layouts", _names(self.parameter_layouts, TRAINABLE_LAYOUTS, "layouts"))
        s(self, "placements", _names(self.placements, PLACEMENTS, "placements"))
        s(self, "advantage_estimators", _names(self.advantage_estimators, None, "estimators"))
        s(self, "dynamic_sampling_filters", _names(self.dynamic_sampling_filters, None, "filters"))
        s(
            self,
            "execution_modes",
            _names(
                [canonical_execution_mode(m) for m in _names(self.execution_modes, None, "modes")],
                EXECUTION_MODES,
                "execution modes",
            ),
        )
        s(self, "port_verbs", _names(self.port_verbs, RESERVED_PORT_VERBS, "port verbs"))
        edges = frozenset(tuple(e) for e in self.certified_edges)
        for src, dst, kind in edges:
            if not src or not dst or src == dst or kind not in EDGE_KINDS:
                raise ValueError(f"invalid certified edge {(src, dst, kind)}")
        s(self, "certified_edges", edges)
        s(self, "optimized_paths", _names(self.optimized_paths, None, "optimized paths"))
        for dimension in R0_MECHANISMS:
            s(self, dimension, _names(getattr(self, dimension), None, dimension))
        s(self, "execution", ExecutionCapabilities.from_dict(self.execution))
        s(
            self,
            "unverified_mechanisms",
            _names(self.unverified_mechanisms, None, "unverified mechanisms"),
        )

    def declared_mechanisms(self) -> frozenset[str]:
        """Every declared mechanism as ``dimension:name`` (all dimensions)."""

        from .algorithm import MECHANISM_DIMENSIONS

        return frozenset(
            f"{dimension}:{name}"
            for dimension in MECHANISM_DIMENSIONS
            for name in self.mechanisms(dimension)
        )

    def mechanisms(self, dimension: str) -> frozenset[str]:
        """Declared mechanism names of one ``MECHANISM_DIMENSIONS`` entry."""

        return getattr(self, dimension)

    def with_unverified(self, names: Iterable[str]) -> "EngineCapabilities":
        """Same declaration plus a D11 allowance (``dimension:name`` entries)."""

        from dataclasses import replace

        from .algorithm import mechanism_names

        names = frozenset(names)
        unknown = sorted(names - mechanism_names())
        if unknown:
            raise CapabilityMismatch(
                f"unknown mechanism(s) {unknown} for --rl-allow-unverified-mechanism "
                f"(known: {sorted(mechanism_names())})"
            )
        return replace(self, unverified_mechanisms=self.unverified_mechanisms | names)

    # -- serialization (PR #66 attestation compatible) ----------------------
    def to_attestation_dict(self) -> dict[str, Any]:
        return {
            "schema": CAPABILITY_SCHEMA,
            "engine": self.engine,
            "runtime_fingerprint": self.runtime_fingerprint,
            "execution_modes": sorted(self.execution_modes),
            "certified_edges": [
                {"source": s, "target": t, "kind": k}
                for s, t, k in sorted(self.certified_edges)
            ],
            "optimized_paths": sorted(self.optimized_paths),
            "auto_controller": self.auto_controller,
            "partitioned_driver": self.partitioned_driver,
            "parameter_layouts": sorted(self.parameter_layouts),
            "placements": sorted(self.placements),
            "advantage_estimators": sorted(self.advantage_estimators),
            "dynamic_sampling_filters": sorted(self.dynamic_sampling_filters),
            "port_verbs": sorted(self.port_verbs),
            **{dimension: sorted(getattr(self, dimension)) for dimension in R0_MECHANISMS},
            "execution": self.execution.to_dict(),
            "unverified_mechanisms": sorted(self.unverified_mechanisms),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_attestation_dict(), sort_keys=True, indent=2) + "\n"

    @classmethod
    def from_attestation_dict(cls, payload: Mapping[str, Any]) -> "EngineCapabilities":
        if not isinstance(payload, Mapping):
            raise ValueError("capability declaration must be an object")
        if payload.get("schema") != CAPABILITY_SCHEMA:
            raise ValueError(f"unknown capability schema {payload.get('schema')!r}")
        modes = payload.get("execution_modes", [])
        return cls(
            engine=payload["engine"],
            runtime_fingerprint=payload["runtime_fingerprint"],
            parameter_layouts=payload.get("parameter_layouts", []),
            placements=payload.get("placements", []),
            advantage_estimators=payload.get("advantage_estimators", []),
            dynamic_sampling_filters=payload.get("dynamic_sampling_filters", []),
            execution_modes=modes,
            port_verbs=payload.get("port_verbs", []),
            certified_edges=[
                (e["source"], e["target"], e["kind"])
                for e in payload.get("certified_edges", [])
            ],
            optimized_paths=payload.get("optimized_paths", []),
            auto_controller=bool(payload.get("auto_controller", False)),
            partitioned_driver=bool(payload.get("partitioned_driver", False)),
            **{
                dimension: payload.get(dimension, default)
                for dimension, default in R0_MECHANISMS.items()
            },
            execution=ExecutionCapabilities.from_dict(payload.get("execution")),
            unverified_mechanisms=payload.get("unverified_mechanisms", []),
        )

    @classmethod
    def from_json(cls, text: str) -> "EngineCapabilities":
        return cls.from_attestation_dict(json.loads(text))

    # -- handshake -----------------------------------------------------------
    def check(
        self,
        *,
        layout: str,
        placement: str,
        execution_mode: str,
        algorithm: Any,
        max_policy_age: int | None = None,
    ) -> None:
        """Reject before any GPU process exists; lists every problem at once.

        Checks the run shape, then every mechanism the algorithm requires
        (``required_mechanisms``), its execution requirements and the
        rejection matrix (``rejections``). ``max_policy_age`` is the age the
        selected execution profile may produce (rl-infra-spec
        ``ExecutionProfile.max_policy_age``); None means the declared
        ``execution.max_policy_staleness``. Unverified-mechanism allowances
        exempt only the "not declared" problem of the named mechanisms.
        """

        problems = []
        for what, value, supported in (
            ("parameter layout", layout, self.parameter_layouts),
            ("placement", placement, self.placements),
            ("execution mode", canonical_execution_mode(execution_mode), self.execution_modes),
        ):
            if value not in supported:
                problems.append(f"{what} {value!r} not supported (supported: {sorted(supported)})")
        required = getattr(algorithm, "required_mechanisms", None)
        if not callable(required):
            # R0 duck-typed description: estimator + filter only.
            pairs = {("advantage_estimators", algorithm.advantage_estimator)}
            if algorithm.dynamic_sampling_filter is not None:
                pairs.add(("dynamic_sampling_filters", algorithm.dynamic_sampling_filter))
        else:
            pairs = set(required())
        labels = {
            "advantage_estimators": "advantage estimator",
            "dynamic_sampling_filters": "dynamic sampling filter",
        }
        for dimension, name in sorted(pairs):
            supported = self.mechanisms(dimension)
            if name in supported or f"{dimension}:{name}" in self.unverified_mechanisms:
                continue
            label = labels.get(dimension, f"{dimension} mechanism")
            hint = ""
            if (dimension, name) == ("reward_postprocessors", "custom_reward_postprocess"):
                hint = (
                    "; the yeto reward dispatcher needed by advantage transforms such as "
                    "maxrl/mapo is not declared yet (pending the rl-algo-grpo-knobs G1)"
                )
            problems.append(
                f"{label} {name!r} not supported (supported: {sorted(supported)}; "
                f"expressible but not enabled on this engine{hint})"
            )
        execution = getattr(algorithm, "execution", None)
        if execution is not None and callable(required):
            if execution.needs_critic and not self.execution.critic:
                problems.append(
                    "algorithm needs a critic, which this engine does not drive "
                    "(critic algorithms are supported only by --rl-engine legacy)"
                )
            if execution.needs_rollout_logprobs and not self.execution.rollout_logprobs:
                problems.append("algorithm needs rollout logprobs, which this engine does not return")
            age = self.execution.max_policy_staleness if max_policy_age is None else max_policy_age
            if age > execution.max_policy_staleness:
                problems.append(
                    f"execution mode may produce policy age {age} but the algorithm tolerates "
                    f"max_policy_staleness={execution.max_policy_staleness}"
                )
        rejections = getattr(algorithm, "rejections", None)
        if callable(rejections):
            problems.extend(rejections())
        if problems:
            raise CapabilityMismatch(f"engine {self.engine!r}: " + "; ".join(problems))
