from __future__ import annotations

import hashlib
import json
import stat
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from yeto.rl.miles_sao_streaming import MilesSaoStreamingConfig
from yeto.rl.sao_streaming_runtime import (
    bind_sao_streaming_runtime,
    load_sao_streaming_runtime,
    streaming_layout_attestation_schema,
    streaming_runtime_schema,
)
from yeto.syncer_profile import (
    SYNCER_SEMANTIC_PROFILE_SCHEMA,
    SyncerSemanticProfile,
)

SECRLENV_CONTEXT_SHA256 = "9" * 64


def _syncer_profile() -> dict[str, object]:
    return {
        "schema": SYNCER_SEMANTIC_PROFILE_SCHEMA,
        "learners": 2,
        "quorum": 2,
        "grace_ms": 0,
        "grace_gamma": 0.8,
        "grace_tau": 2.0,
        "pipeline": 2,
        "min_round_interval_ms": 0,
        "sync_interval_steps": 24.0,
        "delta_correction": "none",
        "quorum_timeout_s": 900,
        "final_ack_timeout_s": 900,
        "total_steps": 8,
        "policy_sweep_fragments": None,
        "outer_lr": 0.7,
        "outer_momentum": 0.9,
        "checkpoint_enabled": True,
        "checkpoint_every": 1,
        "resume": True,
        "mark_final_checkpoint": True,
        "learner_budget_steps": None,
        "max_base_lag": 0,
        "learner_weight": "equal",
        "require_profile_binding": True,
    }


def test_syncer_profile_matches_rust_canonical_vector():
    assert SyncerSemanticProfile.from_mapping(_syncer_profile()).sha256 == (
        "b904a25c417a24deef77b8526c4e982a13a35c65d331a4cafb74e579b584d4b7"
    )


def _role(role: str, *, port: int) -> dict[str, object]:
    actor = role == "actor"
    return {
        "role": role,
        "component": {
            "model_revision": ("a" if actor else "b") * 40,
            "config_sha256": ("1" if actor else "2") * 64,
        },
        "syncer": {"host": "127.0.0.1", "port": port},
        "learner_id": 0,
        "learner_generation": 0,
        "learner_generations": [0, 0],
        "total_fragment_steps": 8,
        "expected_fragments": 4,
        "parameter_layout_sha256": ("3" if actor else "4") * 64,
        "local_horizon": 1,
        "optimizer_steps_per_round": 1 if actor else 2,
        "training_contract_sha256": "5" * 64,
        "wan_streams": 4,
        "wait_timeout_seconds": 900.0,
        "poll_seconds": 0.01,
        "max_fragment_bytes": 2 << 30,
        "max_chunk_bytes": 256 << 20,
    }


def _payload(evidence: Path) -> dict[str, object]:
    return {
        "schema": streaming_runtime_schema(),
        "sao_secrlenv_context_sha256": SECRLENV_CONTEXT_SHA256,
        "trajectory_evidence_dir": str(evidence),
        "syncer_profile": _syncer_profile(),
        "actor": _role("actor", port=29400),
        "critic": _role("critic", port=29401),
    }


def _attested_role(role: dict[str, object]) -> dict[str, object]:
    name = role["role"]
    return {
        "role": name,
        "component": deepcopy(role["component"]),
        "parameter_layout_sha256": role["parameter_layout_sha256"],
        "expected_fragments": role["expected_fragments"],
        "parameter_tensor_count": 4,
        "parameter_scalar_count": 40,
        "fragments": [
            {
                "fragment_id": fragment_id,
                "role": name,
                "shard_id": "tp0-of-1.pp0-of-1.ep0-of-1.cp0-of-1.dp0-of-1",
                "parameter_count": 1,
                "scalar_count": 10,
                "payload_bytes": 40,
            }
            for fragment_id in range(role["expected_fragments"])
        ],
        "manifests": [{"topology": {}, "source_layout_sha256": "7" * 64}],
    }


def _bind_attestation(tmp_path: Path, payload: dict[str, object]) -> None:
    attestation = {
        "schema": streaming_layout_attestation_schema(),
        "sao_secrlenv_context_sha256": SECRLENV_CONTEXT_SHA256,
        "settings": {
            "algorithm": "sao",
            "fragment_strategy": "owner_affine",
            "minimum_fragments": payload["actor"]["expected_fragments"],
            "max_fragment_bytes": payload["actor"]["max_fragment_bytes"],
            "max_chunk_bytes": payload["actor"]["max_chunk_bytes"],
            "wire_dtype": "fp32",
            "syncer_profile_sha256": SyncerSemanticProfile.from_mapping(
                payload["syncer_profile"]
            ).sha256,
        },
        "actor": _attested_role(payload["actor"]),
        "critic": _attested_role(payload["critic"]),
    }
    path = tmp_path / "streaming-layouts.json"
    digest = _write_runtime(path, attestation)
    path.chmod(0o600)
    payload["layout_attestation"] = {"path": str(path), "sha256": digest}


def _write_runtime(path: Path, payload: dict[str, object]) -> str:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(tmp_path: Path, payload: dict[str, object] | None = None):
    evidence = tmp_path / "trajectory-evidence-0"
    source = tmp_path / "streaming.json"
    source_payload = _payload(evidence) if payload is None else payload
    _bind_attestation(tmp_path, source_payload)
    digest = _write_runtime(source, source_payload)
    runtime = load_sao_streaming_runtime(
        source,
        expected_sha256=digest,
        expected_sao_context_sha256=SECRLENV_CONTEXT_SHA256,
    )
    return runtime, evidence


def _args(**overrides):
    values = {
        "sao_online_recipe": "coding",
        "use_critic": True,
        "num_critic_only_steps": 0,
        "start_rollout_id": 0,
        "lora_rank": 0,
        "external_policy_sync_path": None,
        "rollout_function_path": "yeto.rl.miles.generate_rollout",
        "yeto_rl_learner_id": 0,
        "yeto_rl_base_model_revision": "a" * 40,
        "num_steps_per_rollout": 1,
        "num_critic_epochs": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_load_constructs_two_role_scoped_streams(tmp_path):
    runtime, evidence = _load(tmp_path)

    assert isinstance(runtime.streams, MilesSaoStreamingConfig)
    assert runtime.trajectory_evidence_dir == evidence
    assert runtime.streams.actor.role == "actor"
    assert runtime.streams.critic.role == "critic"
    assert runtime.streams.actor.syncer_addr == ("127.0.0.1", 29400)
    assert runtime.streams.critic.syncer_addr == ("127.0.0.1", 29401)
    assert runtime.streams.actor.expected_layout_hash == "3" * 64
    assert runtime.streams.critic.expected_layout_hash == "4" * 64
    expected_profile = SyncerSemanticProfile.from_mapping(_syncer_profile()).sha256
    assert runtime.streams.actor.syncer_profile_hash == expected_profile
    assert runtime.streams.critic.syncer_profile_hash == expected_profile
    assert runtime.streams.actor.pipeline_depth == 2
    assert runtime.syncer_profile.sha256 == expected_profile
    assert runtime.layout_attestation_path == tmp_path / "streaming-layouts.json"


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda value: value["critic"]["syncer"].update(port=29400), "distinct syncer"),
        (
            lambda value: value["syncer_profile"].update(total_steps=12),
            "frozen learner schedule",
        ),
        (
            lambda value: value["critic"].update(expected_fragments=2),
            "lockstep schedule",
        ),
        (
            lambda value: value["critic"].update(parameter_layout_sha256="3" * 64),
            "role-distinct parameter layouts",
        ),
    ),
)
def test_load_rejects_role_or_lockstep_drift(tmp_path, mutation, message):
    payload = deepcopy(_payload(tmp_path / "evidence"))
    mutation(payload)

    with pytest.raises(ValueError, match=message):
        _load(tmp_path, payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("require_profile_binding", False, "require HELLO profile binding"),
        ("quorum", 1, "quorum equal to learners"),
        ("grace_ms", 1, "grace_ms=0"),
        ("checkpoint_enabled", False, "requires checkpointing"),
        ("checkpoint_every", 2, "checkpointing every round"),
        ("resume", False, "checkpointing every round"),
        ("max_base_lag", 1, "max_base_lag=0"),
        ("policy_sweep_fragments", 4, "policy sweep"),
        ("learner_budget_steps", 2, "learner-budget cutoff mode"),
    ),
)
def test_load_rejects_non_fail_closed_sao_server_profile(
    tmp_path,
    field,
    value,
    message,
):
    payload = _payload(tmp_path / "evidence")
    payload["syncer_profile"][field] = value

    with pytest.raises(ValueError, match=message):
        _load(tmp_path, payload)


def test_load_requires_both_file_and_secrlenv_hashes(tmp_path):
    evidence = tmp_path / "evidence"
    source = tmp_path / "streaming.json"
    payload = _payload(evidence)
    _bind_attestation(tmp_path, payload)
    digest = _write_runtime(source, payload)

    with pytest.raises(ValueError, match="context SHA256 mismatch"):
        load_sao_streaming_runtime(
            source,
            expected_sha256="0" * 64,
            expected_sao_context_sha256=SECRLENV_CONTEXT_SHA256,
        )
    with pytest.raises(ValueError, match="different SecRLEnv context"):
        load_sao_streaming_runtime(
            source,
            expected_sha256=digest,
            expected_sao_context_sha256="8" * 64,
        )


def test_load_rejects_runtime_layout_drift_from_private_attestation(tmp_path):
    payload = _payload(tmp_path / "evidence")
    _bind_attestation(tmp_path, payload)
    payload["actor"]["parameter_layout_sha256"] = "8" * 64
    source = tmp_path / "streaming.json"
    digest = _write_runtime(source, payload)

    with pytest.raises(ValueError, match="actor runtime differs"):
        load_sao_streaming_runtime(
            source,
            expected_sha256=digest,
            expected_sao_context_sha256=SECRLENV_CONTEXT_SHA256,
        )


def test_bind_installs_streaming_callback_and_private_evidence(tmp_path):
    runtime, evidence = _load(tmp_path)
    args = _args()

    assert bind_sao_streaming_runtime(args, runtime) == evidence

    assert evidence.is_dir()
    assert stat.S_IMODE(evidence.stat().st_mode) == 0o700
    assert args.yeto_rl_sao_streaming_config is runtime.streams
    assert args.yeto_rl_trajectory_evidence_dir == str(evidence)
    assert args.yeto_rl_sync_preset == "sao-streaming-full"
    assert args.external_policy_sync_path == (
        "yeto.rl.miles_sao_streaming.create_miles_sao_streaming_sync"
    )
    assert args.external_policy_sync_run_until_stop is True
    assert args.external_policy_identity_setter_path == (
        "yeto.rl.miles.set_current_published_policy_identity"
    )
    assert args.yeto_rl_num_fragments == 4
    assert args.yeto_rl_total_fragment_steps == 8
    assert args.yeto_rl_sao_layout_attestation == str(
        tmp_path / "streaming-layouts.json"
    )


def test_bind_rejects_optimizer_drift_before_creating_evidence(tmp_path):
    runtime, evidence = _load(tmp_path)

    with pytest.raises(ValueError, match="optimizer accounting"):
        bind_sao_streaming_runtime(_args(num_critic_epochs=3), runtime)

    assert not evidence.exists()


@pytest.mark.parametrize(
    "overrides",
    (
        {"debug_skip_weight_update": True},
        {"debug_disable_optimizer": True},
        {"debug_train_only": True},
        {"debug_rollout_only": True},
        {"load_debug_rollout_data": "/tmp/replay.pt"},
        {"ci_inject_rollout_data_path": "/tmp/injected.pt"},
        {"ci_ft_test_actions": "[]"},
        {"debug_exit_after_rollout": 1},
    ),
)
def test_bind_rejects_state_bypassing_debug_shortcuts(tmp_path, overrides):
    runtime, evidence = _load(tmp_path)

    with pytest.raises(ValueError, match="state-bypassing debug options"):
        bind_sao_streaming_runtime(_args(**overrides), runtime)

    assert not evidence.exists()
