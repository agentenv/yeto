"""Hash-bound runtime wiring for full-parameter SAO streaming.

The proven centralized SAO entrypoint remains independent of this module.
Streaming launches opt in with a second, immutable JSON contract that names
both role-specific syncer sessions and the fresh trajectory-evidence path.
"""

from __future__ import annotations

import hashlib
import json
import math
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..syncer_profile import SyncerSemanticProfile
from .local_learner import ComponentIdentity
from .miles_sao_streaming import (
    MilesSaoRoleStreamConfig,
    MilesSaoStreamingConfig,
)

_LEGACY_SCHEMA = "yeto.sao-streaming-runtime.v1"
_SCHEMA = "yeto.sao-streaming-runtime.v2"
_LEGACY_LAYOUT_ATTESTATION_SCHEMA = "miles.sao-streaming-layouts.v1"
_LAYOUT_ATTESTATION_SCHEMA = "miles.sao-streaming-layouts.v2"
_SYNC_FACTORY = "yeto.rl.miles_sao_streaming.create_miles_sao_streaming_sync"
_IDENTITY_SETTER = "yeto.rl.miles.set_current_published_policy_identity"
_ROLLOUT_FUNCTION = "yeto.rl.miles.generate_rollout"
_HEX = frozenset("0123456789abcdef")
_LEGACY_TOP_LEVEL_FIELDS = {
    "schema",
    "sao_secrlenv_context_sha256",
    "trajectory_evidence_dir",
    "layout_attestation",
    "syncer_profile",
    "actor",
    "critic",
}
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
_TRAJECTORY_EVIDENCE_KINDS = frozenset({"secrlenv", "terminal-bench-2.1"})
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


def sao_streaming_sync_factory_path() -> str:
    """Return the one callback path owned by the streaming SAO entrypoint."""

    return _SYNC_FACTORY


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
        raise ValueError(f"{name} fields do not match the v1 schema")
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


def _parse_role(
    expected_role: str,
    raw: object,
    *,
    syncer_profile: SyncerSemanticProfile,
) -> MilesSaoRoleStreamConfig:
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
    return MilesSaoRoleStreamConfig(
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
        syncer_profile_hash=syncer_profile.sha256,
        pipeline_depth=syncer_profile.pipeline,
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
    """Validated runtime contract ready to bind into a Miles namespace."""

    sao_context_sha256: str
    trajectory_evidence_dir: Path
    trajectory_evidence_kind: str
    trajectory_evidence_schema_version: int
    layout_attestation_path: Path
    layout_attestation_sha256: str
    syncer_profile: SyncerSemanticProfile
    streams: MilesSaoStreamingConfig


def _load_layout_attestation(
    raw: object,
    *,
    expected_sao_context_sha256: str,
    benchmark_neutral: bool,
    actor: MilesSaoRoleStreamConfig,
    critic: MilesSaoRoleStreamConfig,
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
    context_field = (
        "sao_context_sha256"
        if benchmark_neutral
        else "sao_secrlenv_context_sha256"
    )
    expected_schema = (
        _LAYOUT_ATTESTATION_SCHEMA
        if benchmark_neutral
        else _LEGACY_LAYOUT_ATTESTATION_SCHEMA
    )
    payload = _mapping(
        "SAO streaming layout attestation",
        payload,
        {"schema", context_field, "settings", "actor", "critic"},
    )
    if payload["schema"] != expected_schema:
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
    if schema == _LEGACY_SCHEMA:
        payload = _mapping(
            "SAO streaming context", payload, _LEGACY_TOP_LEVEL_FIELDS
        )
        context_field = "sao_secrlenv_context_sha256"
        evidence = payload["trajectory_evidence_dir"]
        evidence_kind = "secrlenv"
        evidence_schema_version = 1
        benchmark_neutral = False
    elif schema == _SCHEMA:
        payload = _mapping("SAO streaming context", payload, _TOP_LEVEL_FIELDS)
        context_field = "sao_context_sha256"
        evidence_contract = _mapping(
            "SAO trajectory evidence",
            payload["trajectory_evidence"],
            _TRAJECTORY_EVIDENCE_FIELDS,
        )
        evidence = evidence_contract["directory"]
        evidence_kind = evidence_contract["kind"]
        evidence_schema_version = evidence_contract["schema_version"]
        if (
            evidence_kind not in _TRAJECTORY_EVIDENCE_KINDS
            or evidence_schema_version != 2
        ):
            raise ValueError(
                "SAO trajectory evidence kind/schema is unsupported"
            )
        benchmark_neutral = True
    else:
        raise ValueError(f"unsupported SAO streaming schema: {schema!r}")
    bound_context = _sha256(
        "bound SAO context",
        payload[context_field],
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
    actor = _parse_role(
        "actor",
        payload["actor"],
        syncer_profile=syncer_profile,
    )
    critic = _parse_role(
        "critic",
        payload["critic"],
        syncer_profile=syncer_profile,
    )
    if (
        syncer_profile.learners != len(actor.learner_generations)
        or syncer_profile.total_steps != actor.total_fragment_steps
    ):
        raise ValueError("SAO syncer profile differs from the frozen learner schedule")
    if actor.expected_layout_hash == critic.expected_layout_hash:
        raise ValueError("SAO actor and critic require role-distinct parameter layouts")
    streams = MilesSaoStreamingConfig(actor=actor, critic=critic)
    attestation_path, attestation_sha256 = _load_layout_attestation(
        payload["layout_attestation"],
        expected_sao_context_sha256=bound_context,
        benchmark_neutral=benchmark_neutral,
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
        streams,
    )


def _validate_miles_runtime(args: Any, runtime: SaoStreamingRuntime) -> None:
    actor = runtime.streams.actor
    critic = runtime.streams.critic
    if runtime.trajectory_evidence_kind == "terminal-bench-2.1":
        from .tbench_outcome import validate_hmac_key_source

        validate_hmac_key_source()
    unsafe_debug_flags = (
        "debug_skip_weight_update",
        "debug_disable_optimizer",
        "debug_train_only",
        "debug_rollout_only",
    )
    enabled_debug_flags = [
        name for name in unsafe_debug_flags if bool(getattr(args, name, False))
    ]
    unsafe_debug_values = (
        "load_debug_rollout_data",
        "ci_inject_rollout_data_path",
        "ci_ft_test_actions",
        "debug_exit_after_rollout",
    )
    enabled_debug_values = [
        name for name in unsafe_debug_values if getattr(args, name, None) is not None
    ]
    if enabled_debug_flags or enabled_debug_values:
        options = ", ".join(enabled_debug_flags + enabled_debug_values)
        raise ValueError(
            f"SAO streaming rejects state-bypassing debug options: {options}"
        )
    if getattr(args, "sao_online_recipe", None) is None:
        raise ValueError("SAO streaming requires --sao-online-recipe")
    if not getattr(args, "use_critic", False):
        raise ValueError("SAO streaming requires the Miles critic")
    if int(getattr(args, "num_critic_only_steps", 0)) != 0:
        raise ValueError(
            "SAO streaming requires actor and critic lockstep from rollout zero"
        )
    if int(getattr(args, "start_rollout_id", 0)) != 0:
        raise ValueError("SAO streaming currently requires a version-zero start")
    if int(getattr(args, "lora_rank", 0)) > 0:
        raise ValueError("SAO streaming requires full-parameter training")
    if getattr(args, "external_policy_sync_path", None) != _SYNC_FACTORY:
        raise ValueError(
            "SAO streaming requires its trusted external sync callback before "
            "Miles argument validation"
        )
    if getattr(args, "rollout_function_path", None) != _ROLLOUT_FUNCTION:
        raise ValueError(
            "SAO streaming requires Yeto's evidence-producing rollout function"
        )
    if getattr(args, "yeto_rl_learner_id", None) != actor.learner_id:
        raise ValueError("SAO streaming learner ID differs from the SecRLEnv context")
    if (
        getattr(args, "yeto_rl_base_model_revision", None)
        != actor.component.model_revision
    ):
        raise ValueError("SAO actor revision differs from the SecRLEnv base model")

    actor_steps = getattr(args, "num_steps_per_rollout", None)
    critic_epochs = getattr(args, "num_critic_epochs", None)
    if (
        type(actor_steps) is not int
        or actor_steps < 1
        or type(critic_epochs) is not int
        or critic_epochs < 1
        or actor.optimizer_steps_per_round != actor_steps
        or critic.optimizer_steps_per_round != actor_steps * critic_epochs
    ):
        raise ValueError("SAO role optimizer accounting differs from the Miles recipe")

    evidence = runtime.trajectory_evidence_dir
    parent = evidence.parent
    if (
        evidence.exists()
        or evidence.is_symlink()
        or not parent.is_dir()
        or parent.is_symlink()
    ):
        raise ValueError(
            "SAO trajectory evidence path must be fresh with a real parent"
        )


def bind_sao_streaming_runtime(args: Any, runtime: SaoStreamingRuntime) -> Path:
    """Install the validated streaming callback and create its private evidence dir."""

    _validate_miles_runtime(args, runtime)
    evidence = runtime.trajectory_evidence_dir
    try:
        evidence.mkdir(mode=0o700)
        evidence.chmod(0o700)
    except OSError as error:
        raise ValueError(
            "failed to create private SAO trajectory evidence directory"
        ) from error
    if stat.S_IMODE(evidence.stat().st_mode) != 0o700:
        raise RuntimeError("SAO trajectory evidence directory is not private")

    config = runtime.streams
    args.yeto_rl_sao_streaming_config = config
    args.yeto_rl_trajectory_evidence_dir = str(evidence)
    args.yeto_rl_trajectory_evidence_kind = runtime.trajectory_evidence_kind
    args.yeto_rl_trajectory_evidence_schema_version = (
        runtime.trajectory_evidence_schema_version
    )
    args.yeto_rl_sync_preset = "sao-streaming-full"
    args.yeto_rl_learner_generation = config.actor.learner_generation
    args.yeto_rl_learner_generations = config.actor.learner_generations
    args.yeto_rl_num_fragments = config.actor.expected_fragments
    args.yeto_rl_total_fragment_steps = config.actor.total_fragment_steps
    args.yeto_rl_sao_training_contract_sha256 = config.actor.training_contract_hash
    args.yeto_rl_sao_syncer_profile_sha256 = config.actor.syncer_profile_hash
    args.yeto_rl_sao_layout_attestation = str(runtime.layout_attestation_path)
    args.yeto_rl_sao_layout_attestation_sha256 = runtime.layout_attestation_sha256
    args.external_policy_sync_path = _SYNC_FACTORY
    args.external_policy_sync_run_until_stop = True
    args.external_policy_identity_setter_path = _IDENTITY_SETTER
    return evidence


def streaming_runtime_schema() -> str:
    """Return the immutable schema identifier for external config builders."""

    return _SCHEMA


def streaming_layout_attestation_schema() -> str:
    """Return the schema emitted by the two-role Miles metadata probe."""

    return _LAYOUT_ATTESTATION_SCHEMA
