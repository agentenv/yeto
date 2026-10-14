"""The hash-bound runtime contract of a full-parameter SAO streaming island.

A streaming SAO island opts in with an immutable JSON contract that names
both role-specific syncer sessions, the fresh trajectory-evidence path and
the layout attestation of both roles.  This module loads and checks that
contract; it is engine-neutral.  The Miles adapter
(``yeto.rl.adapters.miles.full_parameter.sao_binding``) turns the checked
role contracts into the two-role full-parameter sync (P0-D8).
"""

from __future__ import annotations

import hashlib
import json
import math
import stat
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..syncer_profile import SyncerSemanticProfile
from .local_learner import ComponentIdentity
from .trajectory_evidence import EVIDENCE_KINDS

_SCHEMA = "yeto.sao-streaming-runtime.v2"
_LAYOUT_ATTESTATION_SCHEMA = "miles.sao-streaming-layouts.v2"
_TRAJECTORY_EVIDENCE_SCHEMA_VERSION = 2
_HEX = frozenset("0123456789abcdef")
_TOP_LEVEL_FIELDS = {
    "schema",
    "sao_context_sha256",
    "trajectory_evidence",
    "layout_attestation",
    "syncer_profile",
    "actor",
    "critic",
}
_TRAJECTORY_EVIDENCE_FIELDS = {"directory", "kind", "schema_version"}
_ROLE_FIELDS = {
    "role",
    "component",
    "syncer",
    "learner_id",
    "learner_generation",
    "learner_generations",
    "total_fragment_steps",
    "expected_fragments",
    "parameter_layout_sha256",
    "local_horizon",
    "optimizer_steps_per_round",
    "training_contract_sha256",
    "wan_streams",
    "wait_timeout_seconds",
    "poll_seconds",
    "max_fragment_bytes",
    "max_chunk_bytes",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(name: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _mapping(name: str, value: object, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} fields do not match the schema")
    return value


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


@dataclass(frozen=True)
class SaoRoleContract:
    """One role's checked stream contract (engine-neutral)."""

    role: str
    component: ComponentIdentity
    syncer_addr: tuple[str, int]
    learner_id: int
    learner_generation: int
    learner_generations: tuple[int, ...]
    total_fragment_steps: int
    expected_fragments: int
    expected_layout_hash: str
    local_horizon: int
    optimizer_steps_per_round: int
    training_contract_hash: str
    wan_streams: int
    wait_timeout: float
    poll_seconds: float
    max_fragment_bytes: int
    max_chunk_bytes: int


def _parse_role(expected_role: str, raw: object) -> SaoRoleContract:
    role = _mapping(f"SAO {expected_role} stream", raw, _ROLE_FIELDS)
    if role["role"] != expected_role:
        raise ValueError(f"SAO {expected_role} stream role changed")
    component = _mapping(
        f"SAO {expected_role} component",
        role["component"],
        {"model_revision", "config_sha256"},
    )
    syncer = _mapping(
        f"SAO {expected_role} syncer",
        role["syncer"],
        {"host", "port"},
    )
    host = syncer["host"]
    if (
        type(host) is not str
        or not host
        or len(host) > 253
        or any(character.isspace() for character in host)
    ):
        raise ValueError(f"SAO {expected_role} syncer host is invalid")
    generations = role["learner_generations"]
    if not isinstance(generations, list):
        raise TypeError(f"SAO {expected_role} learner_generations must be a list")

    model_revision = component["model_revision"]
    if type(model_revision) is not str:
        raise ValueError(f"SAO {expected_role} model revision must be a string")
    return SaoRoleContract(
        role=expected_role,
        component=ComponentIdentity(
            role=expected_role,
            model_revision=model_revision,
            config_hash=_sha256(
                f"SAO {expected_role} component config",
                component["config_sha256"],
            ),
        ),
        syncer_addr=(
            host,
            _integer(f"SAO {expected_role} syncer port", syncer["port"], minimum=1),
        ),
        learner_id=_integer(f"SAO {expected_role} learner_id", role["learner_id"]),
        learner_generation=_integer(
            f"SAO {expected_role} learner_generation",
            role["learner_generation"],
        ),
        learner_generations=tuple(
            _integer(
                f"SAO {expected_role} learner_generations[{index}]",
                generation,
            )
            for index, generation in enumerate(generations)
        ),
        total_fragment_steps=_integer(
            f"SAO {expected_role} total_fragment_steps",
            role["total_fragment_steps"],
            minimum=1,
        ),
        expected_fragments=_integer(
            f"SAO {expected_role} expected_fragments",
            role["expected_fragments"],
            minimum=1,
        ),
        expected_layout_hash=_sha256(
            f"SAO {expected_role} parameter layout",
            role["parameter_layout_sha256"],
        ),
        local_horizon=_integer(
            f"SAO {expected_role} local_horizon",
            role["local_horizon"],
            minimum=1,
        ),
        optimizer_steps_per_round=_integer(
            f"SAO {expected_role} optimizer_steps_per_round",
            role["optimizer_steps_per_round"],
            minimum=1,
        ),
        training_contract_hash=_sha256(
            f"SAO {expected_role} training contract",
            role["training_contract_sha256"],
        ),
        wan_streams=_integer(
            f"SAO {expected_role} wan_streams",
            role["wan_streams"],
            minimum=1,
        ),
        wait_timeout=_finite_number(
            f"SAO {expected_role} wait_timeout_seconds",
            role["wait_timeout_seconds"],
        ),
        poll_seconds=_finite_number(
            f"SAO {expected_role} poll_seconds", role["poll_seconds"]
        ),
        max_fragment_bytes=_integer(
            f"SAO {expected_role} max_fragment_bytes",
            role["max_fragment_bytes"],
            minimum=4,
        ),
        max_chunk_bytes=_integer(
            f"SAO {expected_role} max_chunk_bytes",
            role["max_chunk_bytes"],
            minimum=4,
        ),
    )


@dataclass(frozen=True)
class SaoStreamingRuntime:
    """A checked runtime contract: both role streams and their shared context."""

    sao_context_sha256: str
    trajectory_evidence_dir: Path
    trajectory_evidence_kind: str
    trajectory_evidence_schema_version: int
    layout_attestation_path: Path
    layout_attestation_sha256: str
    syncer_profile: SyncerSemanticProfile
    roles: tuple[SaoRoleContract, ...]

    def role(self, name: str) -> SaoRoleContract:
        """The stream contract of role ``name`` (``actor`` or ``critic``)."""

        for contract in self.roles:
            if contract.role == name:
                return contract
        raise KeyError(f"SAO runtime has no {name} role")

    @property
    def actor(self) -> SaoRoleContract:
        return self.role("actor")

    @property
    def critic(self) -> SaoRoleContract:
        return self.role("critic")

    def with_syncer_addresses(
        self, *, actor: tuple[str, int], critic: tuple[str, int]
    ) -> SaoStreamingRuntime:
        """The same contract with the launch's syncer hosts.

        The contract is frozen before the syncer host is known.  The host is
        not part of any session hash, but each role's port is, so the ports
        must stay those the contract names.
        """

        addresses = {"actor": actor, "critic": critic}
        for contract in self.roles:
            if addresses[contract.role][1] != contract.syncer_addr[1]:
                raise ValueError("SAO syncer ports differ from the runtime contract")
        return replace(
            self,
            roles=tuple(
                replace(contract, syncer_addr=addresses[contract.role])
                for contract in self.roles
            ),
        )


def _load_layout_attestation(
    raw: object,
    *,
    expected_sao_context_sha256: str,
    actor: SaoRoleContract,
    critic: SaoRoleContract,
    syncer_profile_sha256: str,
) -> tuple[Path, str]:
    reference = _mapping(
        "SAO streaming layout attestation",
        raw,
        {"path", "sha256"},
    )
    path_value = reference["path"]
    if type(path_value) is not str:
        raise TypeError("SAO streaming layout attestation path must be a string")
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute() or ".." in candidate.parts or candidate.is_symlink():
        raise ValueError("SAO streaming layout attestation path is unsafe")
    path = candidate.resolve()
    expected_sha256 = _sha256(
        "SAO streaming layout attestation SHA256",
        reference["sha256"],
    )
    if (
        not path.is_file()
        or stat.S_IMODE(path.stat().st_mode) & 0o077
        or _sha256_file(path) != expected_sha256
    ):
        raise ValueError("SAO streaming layout attestation is not private and exact")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("SAO streaming layout attestation is malformed") from error
    context_field = "sao_context_sha256"
    payload = _mapping(
        "SAO streaming layout attestation",
        payload,
        {"schema", context_field, "settings", "actor", "critic"},
    )
    if payload["schema"] != _LAYOUT_ATTESTATION_SCHEMA:
        raise ValueError("unsupported SAO streaming layout attestation schema")
    if (
        _sha256(
            "layout-attested SAO context",
            payload[context_field],
        )
        != expected_sao_context_sha256
    ):
        raise ValueError("SAO layout attestation binds a different context")
    settings = _mapping(
        "SAO streaming layout settings",
        payload["settings"],
        {
            "algorithm",
            "fragment_strategy",
            "minimum_fragments",
            "max_fragment_bytes",
            "max_chunk_bytes",
            "wire_dtype",
            "syncer_profile_sha256",
        },
    )
    if settings != {
        "algorithm": "sao",
        "fragment_strategy": "owner_affine",
        "minimum_fragments": actor.expected_fragments,
        "max_fragment_bytes": actor.max_fragment_bytes,
        "max_chunk_bytes": actor.max_chunk_bytes,
        "wire_dtype": "fp32",
        "syncer_profile_sha256": syncer_profile_sha256,
    } or (
        actor.max_fragment_bytes != critic.max_fragment_bytes
        or actor.max_chunk_bytes != critic.max_chunk_bytes
    ):
        raise ValueError("SAO runtime transport differs from its layout probe")
    for role_config in (actor, critic):
        role = _mapping(
            f"SAO {role_config.role} layout attestation",
            payload[role_config.role],
            {
                "role",
                "component",
                "parameter_layout_sha256",
                "expected_fragments",
                "parameter_tensor_count",
                "parameter_scalar_count",
                "fragments",
                "manifests",
            },
        )
        component = _mapping(
            f"SAO {role_config.role} attested component",
            role["component"],
            {"model_revision", "config_sha256"},
        )
        if (
            role["role"] != role_config.role
            or component
            != {
                "model_revision": role_config.component.model_revision,
                "config_sha256": role_config.component.config_hash,
            }
            or role["parameter_layout_sha256"] != role_config.expected_layout_hash
            or role["expected_fragments"] != role_config.expected_fragments
        ):
            raise ValueError(
                f"SAO {role_config.role} runtime differs from its layout probe"
            )
        fragments = role["fragments"]
        manifests = role["manifests"]
        if (
            type(role["parameter_tensor_count"]) is not int
            or role["parameter_tensor_count"] < role_config.expected_fragments
            or type(role["parameter_scalar_count"]) is not int
            or role["parameter_scalar_count"] < role["parameter_tensor_count"]
            or not isinstance(fragments, list)
            or len(fragments) != role_config.expected_fragments
            or not isinstance(manifests, list)
            or not manifests
        ):
            raise ValueError(f"SAO {role_config.role} layout evidence is incomplete")
        for fragment_id, raw_fragment in enumerate(fragments):
            fragment = _mapping(
                f"SAO {role_config.role} fragment {fragment_id}",
                raw_fragment,
                {
                    "fragment_id",
                    "role",
                    "shard_id",
                    "parameter_count",
                    "scalar_count",
                    "payload_bytes",
                },
            )
            if (
                fragment["fragment_id"] != fragment_id
                or fragment["role"] != role_config.role
                or type(fragment["shard_id"]) is not str
                or not fragment["shard_id"]
                or type(fragment["parameter_count"]) is not int
                or fragment["parameter_count"] < 1
                or type(fragment["scalar_count"]) is not int
                or fragment["scalar_count"] < fragment["parameter_count"]
                or fragment["payload_bytes"] != fragment["scalar_count"] * 4
                or fragment["payload_bytes"] > role_config.max_fragment_bytes
            ):
                raise ValueError(
                    f"SAO {role_config.role} attested fragment is malformed"
                )
        if (
            sum(fragment["parameter_count"] for fragment in fragments)
            != role["parameter_tensor_count"]
            or sum(fragment["scalar_count"] for fragment in fragments)
            != role["parameter_scalar_count"]
        ):
            raise ValueError(
                f"SAO {role_config.role} fragment totals differ from the layout"
            )
    return path, expected_sha256


def load_sao_streaming_runtime(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_sao_context_sha256: str,
) -> SaoStreamingRuntime:
    """Load an exact streaming contract without changing runtime state."""

    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ValueError("SAO streaming context must not be a symlink")
    source = candidate.resolve()
    _sha256("SAO streaming context SHA256", expected_sha256)
    _sha256("SAO context SHA256", expected_sao_context_sha256)
    if not source.is_file() or _sha256_file(source) != expected_sha256:
        raise ValueError("SAO streaming context SHA256 mismatch")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("SAO streaming context is unreadable or malformed") from error
    if not isinstance(payload, dict):
        raise TypeError("SAO streaming context must be an object")
    schema = payload.get("schema")
    if schema != _SCHEMA:
        raise ValueError(f"unsupported SAO streaming schema: {schema!r}")
    payload = _mapping("SAO streaming context", payload, _TOP_LEVEL_FIELDS)
    evidence_contract = _mapping(
        "SAO trajectory evidence",
        payload["trajectory_evidence"],
        _TRAJECTORY_EVIDENCE_FIELDS,
    )
    evidence = evidence_contract["directory"]
    evidence_kind = evidence_contract["kind"]
    evidence_schema_version = evidence_contract["schema_version"]
    if (
        evidence_kind not in EVIDENCE_KINDS
        or evidence_schema_version != _TRAJECTORY_EVIDENCE_SCHEMA_VERSION
    ):
        raise ValueError("SAO trajectory evidence kind/schema is unsupported")
    bound_context = _sha256(
        "bound SAO context",
        payload["sao_context_sha256"],
    )
    if bound_context != expected_sao_context_sha256:
        raise ValueError("SAO streaming context binds a different SAO context")
    if type(evidence) is not str:
        raise ValueError("SAO trajectory evidence directory must be a string")
    evidence_path = Path(evidence)
    if (
        not evidence_path.is_absolute()
        or evidence_path.name in {"", ".", ".."}
        or ".." in evidence_path.parts
    ):
        raise ValueError(
            "SAO trajectory evidence directory must be an absolute child path"
        )

    syncer_profile = SyncerSemanticProfile.from_mapping(payload["syncer_profile"])
    syncer_profile.validate_sao()
    actor = _parse_role("actor", payload["actor"])
    critic = _parse_role("critic", payload["critic"])
    if (
        syncer_profile.learners != len(actor.learner_generations)
        or syncer_profile.total_steps != actor.total_fragment_steps
    ):
        raise ValueError("SAO syncer profile differs from the frozen learner schedule")
    if actor.expected_layout_hash == critic.expected_layout_hash:
        raise ValueError("SAO actor and critic require role-distinct parameter layouts")
    attestation_path, attestation_sha256 = _load_layout_attestation(
        payload["layout_attestation"],
        expected_sao_context_sha256=bound_context,
        actor=actor,
        critic=critic,
        syncer_profile_sha256=syncer_profile.sha256,
    )
    return SaoStreamingRuntime(
        bound_context,
        evidence_path,
        evidence_kind,
        evidence_schema_version,
        attestation_path,
        attestation_sha256,
        syncer_profile,
        (actor, critic),
    )


def streaming_runtime_schema() -> str:
    """Return the immutable schema identifier for external config builders."""

    return _SCHEMA


def streaming_layout_attestation_schema() -> str:
    """Return the schema emitted by the two-role Miles metadata probe."""

    return _LAYOUT_ATTESTATION_SCHEMA
