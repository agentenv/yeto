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
    ) -> None:
        """Reject before any GPU process exists; lists supported options."""

        problems = []
        for what, value, supported in (
            ("parameter layout", layout, self.parameter_layouts),
            ("placement", placement, self.placements),
            ("execution mode", canonical_execution_mode(execution_mode), self.execution_modes),
            ("advantage estimator", algorithm.advantage_estimator, self.advantage_estimators),
        ):
            if value not in supported:
                problems.append(f"{what} {value!r} not supported (supported: {sorted(supported)})")
        flt = algorithm.dynamic_sampling_filter
        if flt is not None and flt not in self.dynamic_sampling_filters:
            problems.append(
                f"dynamic sampling filter {flt!r} not supported "
                f"(supported: {sorted(self.dynamic_sampling_filters)})"
            )
        if problems:
            raise CapabilityMismatch(f"engine {self.engine!r}: " + "; ".join(problems))
