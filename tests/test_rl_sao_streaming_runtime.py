from __future__ import annotations

import hashlib
import json
import stat
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from yeto.rl.adapters.miles.full_parameter.sao_binding import (
    bind_sao_streaming_runtime,
    sao_sync_config,
)
from yeto.rl.adapters.miles.full_parameter.session import (
    CONFIG_ATTRIBUTE,
    FullParameterSyncConfig,
)
from yeto.rl.sao_streaming_runtime import (
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
        "85a50d0e4fd5103b81a2295536fca0e5ad96cd50531fb1396208af8459be571e"
    )


def test_syncer_profile_hash_ignores_resume():
    # P0-1.7b.1-Q7: a restarted syncer (--resume) keeps the islands' HELLO hash.
    fresh = _syncer_profile()
    fresh["resume"] = False
    assert SyncerSemanticProfile.from_mapping(fresh).sha256 == (
        SyncerSemanticProfile.from_mapping(_syncer_profile()).sha256
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
        "sao_context_sha256": SECRLENV_CONTEXT_SHA256,
        "trajectory_evidence": {
            "directory": str(evidence),
            "kind": "secrlenv",
            "schema_version": 2,
        },
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
        "sao_context_sha256": SECRLENV_CONTEXT_SHA256,
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
        "policy_objective": "sao_dis",
        "use_critic": True,
        "num_critic_only_steps": 0,
        "start_rollout_id": 0,
        "lora_rank": 0,
        "yeto_rl_learner_id": 0,
        "yeto_rl_base_model_revision": "a" * 40,
        "num_steps_per_rollout": 1,
        "num_critic_epochs": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_load_constructs_two_role_scoped_streams(tmp_path):
    runtime, evidence = _load(tmp_path)

    sync = sao_sync_config(runtime)
    assert isinstance(sync, FullParameterSyncConfig)
    assert not sync.sweep
    assert tuple(sync.by_role()) == ("actor", "critic")
    assert runtime.trajectory_evidence_dir == evidence
    assert runtime.actor.role == "actor"
    assert runtime.critic.role == "critic"
    assert runtime.actor.syncer_addr == ("127.0.0.1", 29400)
    assert runtime.critic.syncer_addr == ("127.0.0.1", 29401)
    assert runtime.actor.expected_layout_hash == "3" * 64
    assert runtime.critic.expected_layout_hash == "4" * 64
    expected_profile = SyncerSemanticProfile.from_mapping(_syncer_profile()).sha256
    for stream in sync.roles:
        assert stream.algorithm == "sao"
        assert stream.syncer_profile_hash == expected_profile
        assert stream.pipeline_depth == 2
    assert runtime.syncer_profile.sha256 == expected_profile
    assert runtime.layout_attestation_path == tmp_path / "streaming-layouts.json"


def test_load_rejects_the_retired_v1_runtime_schema(tmp_path):
    payload = _payload(tmp_path / "evidence")
    _bind_attestation(tmp_path, payload)
    payload["schema"] = "yeto.sao-streaming-runtime.v1"
    source = tmp_path / "streaming.json"
    digest = _write_runtime(source, payload)

    with pytest.raises(ValueError, match="unsupported SAO streaming schema"):
        load_sao_streaming_runtime(
            source,
            expected_sha256=digest,
            expected_sao_context_sha256=SECRLENV_CONTEXT_SHA256,
        )


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
            "fragment totals differ from the layout",
        ),
        (
            lambda value: value["critic"].update(parameter_layout_sha256="3" * 64),
            "role-distinct parameter layouts",
        ),
    ),
)
def test_load_or_role_pairing_rejects_role_or_lockstep_drift(tmp_path, mutation, message):
    """The neutral load checks the contract; the Miles sync config checks the
    role pairing.  The island runs both before any GPU work."""
    payload = deepcopy(_payload(tmp_path / "evidence"))
    mutation(payload)

    with pytest.raises(ValueError, match=message):
        sao_sync_config(_load(tmp_path, payload)[0])


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
    with pytest.raises(ValueError, match="different SAO context"):
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


def test_bind_installs_two_role_full_parameter_sync_and_private_evidence(tmp_path):
    runtime, evidence = _load(tmp_path)
    args = _args()

    assert bind_sao_streaming_runtime(args, runtime) == evidence

    assert evidence.is_dir()
    assert stat.S_IMODE(evidence.stat().st_mode) == 0o700
    assert getattr(args, CONFIG_ATTRIBUTE) == sao_sync_config(runtime)
    assert args.yeto_rl_trajectory_evidence_dir == str(evidence)
    assert args.yeto_rl_trajectory_evidence_kind == "secrlenv"
    assert args.yeto_rl_trajectory_evidence_schema_version == 2
    # The island sets the preset; the bind does not write it a second time.
    assert not hasattr(args, "yeto_rl_sync_preset")
    # The ports IslandDriver owns the loop: no Miles training-loop callback.
    assert not hasattr(args, "external_policy_sync_path")
    assert not hasattr(args, "rollout_function_path")
    assert args.yeto_rl_num_fragments == 4
    assert args.yeto_rl_total_fragment_steps == 8


def test_terminal_bench_runtime_binds_v2_without_any_reward_key(
    tmp_path, monkeypatch
):
    payload = _payload(tmp_path / "trajectory-evidence-0")
    payload["trajectory_evidence"]["kind"] = "terminal-bench-2.1"
    runtime, evidence = _load(tmp_path, payload)
    monkeypatch.delenv("TBENCH_REWARD_HMAC_KEY", raising=False)
    monkeypatch.delenv("TBENCH_REWARD_HMAC_KEY_FILE", raising=False)

    args = _args()
    bind_sao_streaming_runtime(args, runtime)
    assert args.yeto_rl_trajectory_evidence_kind == "terminal-bench-2.1"
    assert args.yeto_rl_trajectory_evidence_schema_version == 2


def test_bind_rejects_optimizer_drift_before_creating_evidence(tmp_path):
    runtime, evidence = _load(tmp_path)

    with pytest.raises(ValueError, match="optimizer accounting"):
        bind_sao_streaming_runtime(_args(num_critic_epochs=3), runtime)

    assert not evidence.exists()


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"policy_objective": "ppo"}, "SAO algorithm spec"),
        ({"use_critic": False}, "Miles critic"),
        ({"lora_rank": 16}, "full-parameter training"),
    ),
)
def test_bind_requires_the_sao_spec_with_a_full_parameter_critic(
    tmp_path, overrides, message
):
    runtime, evidence = _load(tmp_path)

    with pytest.raises(ValueError, match=message):
        bind_sao_streaming_runtime(_args(**overrides), runtime)

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


# --- the ports island entry (P0-1.7b.3) ------------------------------------


def _island_argv(*extra):
    from test_rl_engine_selection import _learner_argv

    argv = _learner_argv()
    lora = argv.index("--lora-r")
    return [*argv[:lora], *argv[lora + 4:], *extra]


def _sao_island_argv(runtime_path, runtime_sha256):
    return _island_argv(
        "--parameter-mode", "full",
        "--sync-preset", "sao-streaming-full",
        "--sao-streaming-runtime", str(runtime_path),
        "--sao-streaming-runtime-sha256", runtime_sha256,
        "--sao-context-sha256", SECRLENV_CONTEXT_SHA256,
    )


def test_island_parses_the_sao_preset_only_with_its_runtime_contract(tmp_path):
    from yeto.rl.adapters.miles import island_entry

    args = island_entry.parse_args(_sao_island_argv(tmp_path / "rt.json", "1" * 64))
    assert (args.parameter_mode, args.sync_preset) == ("full", "sao-streaming-full")

    argv = _sao_island_argv(tmp_path / "rt.json", "1" * 64)
    index = argv.index("--sao-context-sha256")
    del argv[index:index + 2]
    with pytest.raises(SystemExit):
        island_entry.parse_args(argv)
    with pytest.raises(SystemExit):  # the runtime belongs to the SAO preset only
        island_entry.parse_args(
            _island_argv(
                "--lora-r", "8", "--lora-targets", "attention",
                "--sao-streaming-runtime", str(tmp_path / "rt.json"),
                "--sao-streaming-runtime-sha256", "1" * 64,
                "--sao-context-sha256", SECRLENV_CONTEXT_SHA256,
            )
        )
    with pytest.raises(SystemExit):  # full parameters need a full-parameter preset
        island_entry.parse_args(_island_argv("--parameter-mode", "full"))


def test_island_loads_the_runtime_for_its_own_learner(tmp_path):
    from yeto.rl.adapters.miles import island_entry

    payload = _payload(tmp_path / "evidence")
    _bind_attestation(tmp_path, payload)
    source = tmp_path / "streaming.json"
    digest = _write_runtime(source, payload)
    args = island_entry.parse_args(_sao_island_argv(source, digest))

    args.syncer = "10.0.0.7:29400"
    args.critic_syncer = "10.0.0.7:29401"
    runtime = island_entry._load_sao_streaming_runtime(args)
    assert tuple(sao_sync_config(runtime).by_role()) == ("actor", "critic")
    # The contract fixes ports; the host follows the launch's syncer address.
    assert runtime.actor.syncer_addr == ("10.0.0.7", 29400)
    assert runtime.critic.syncer_addr == ("10.0.0.7", 29401)

    args.critic_syncer = "10.0.0.7:29402"
    with pytest.raises(ValueError, match="ports differ"):
        island_entry._load_sao_streaming_runtime(args)
    args.critic_syncer = None

    args.learner_id = 1
    with pytest.raises(ValueError, match="different learner"):
        island_entry._load_sao_streaming_runtime(args)


@pytest.mark.parametrize(
    ("engine", "sync", "overrides", "message"),
    (
        ("legacy", True, {}, "only on the ports engine"),
        ("ports", False, {}, "Yeto policy sync"),
        ("ports", True, {"eval_only": True}, "eval-only"),
    ),
)
def test_full_parameter_island_rejections(engine, sync, overrides, message):
    from yeto.rl.adapters.miles import island_entry

    args = SimpleNamespace(**{"eval_only": False, "rl_max_policy_age": 0, **overrides})
    with pytest.raises(ValueError, match=message):
        island_entry._check_full_parameter_sync(args, rl_engine=engine, yeto_policy_sync=sync)


def test_island_entry_binds_the_sao_streaming_runtime_end_to_end(monkeypatch, tmp_path):
    """run_miles with Megatron, Miles parsing and the ports loop stubbed: the
    SAO preset reaches the ports loop with the two-role sync bound."""
    from test_rl_launcher import _stub_full_parameter_island

    evidence = tmp_path / "audit" / "sao-evidence"
    payload = _payload(evidence)
    _bind_attestation(tmp_path, payload)
    source = tmp_path / "streaming.json"
    digest = _write_runtime(source, payload)
    captured, args, model, _config, prompts, _audit, _ref = _stub_full_parameter_island(
        monkeypatch,
        tmp_path,
        miles_fields=_args(yeto_rl_learner_id=None).__dict__,
        sync_preset="sao-streaming-full",
        sao_streaming_runtime=str(source),
        sao_streaming_runtime_sha256=digest,
        sao_context_sha256=SECRLENV_CONTEXT_SHA256,
        critic_syncer="127.0.0.1:29401",
        learner_id=0,
        model_revision="a" * 40,
        parameter_layout_sha256=None,
        syncer_profile_sha256=None,
    )
    from yeto.rl.adapters.miles import island_entry

    island_entry.run_miles(args, model_path=model, prompt_path=prompts)

    miles_args = captured["miles_args"]
    assert miles_args.yeto_rl_sync_preset == "sao-streaming-full"
    assert getattr(miles_args, CONFIG_ATTRIBUTE) == sao_sync_config(
        load_sao_streaming_runtime(
            source,
            expected_sha256=digest,
            expected_sao_context_sha256=SECRLENV_CONTEXT_SHA256,
        )
    )
    assert miles_args.yeto_rl_trajectory_evidence_dir == str(evidence)
    assert stat.S_IMODE(evidence.stat().st_mode) == 0o700
    assert "--lora-rank" not in miles_args.argv


def test_island_entry_rejects_an_evidence_root_others_can_write(monkeypatch, tmp_path):
    from test_rl_launcher import _stub_full_parameter_island

    evidence = tmp_path / "shared-root" / "sao-evidence"
    evidence.parent.mkdir()
    evidence.parent.chmod(0o777)
    payload = _payload(evidence)
    _bind_attestation(tmp_path, payload)
    source = tmp_path / "streaming.json"
    digest = _write_runtime(source, payload)
    _captured, args, model, _config, prompts, _audit, _ref = _stub_full_parameter_island(
        monkeypatch,
        tmp_path,
        miles_fields=_args(yeto_rl_learner_id=None).__dict__,
        sync_preset="sao-streaming-full",
        sao_streaming_runtime=str(source),
        sao_streaming_runtime_sha256=digest,
        sao_context_sha256=SECRLENV_CONTEXT_SHA256,
        critic_syncer="127.0.0.1:29401",
        learner_id=0,
        model_revision="a" * 40,
        parameter_layout_sha256=None,
        syncer_profile_sha256=None,
    )
    from yeto.rl.adapters.miles import island_entry

    with pytest.raises(ValueError, match="writable by others"):
        island_entry.run_miles(args, model_path=model, prompt_path=prompts)
    assert not evidence.exists()


def test_full_parameter_island_accepts_a_policy_age_limit():
    """P0-S: every trajectory records its own behavior policy (evidence schema 4)."""
    from yeto.rl.adapters.miles import island_entry

    args = SimpleNamespace(eval_only=False, rl_max_policy_age=1)
    island_entry._check_full_parameter_sync(args, rl_engine="ports", yeto_policy_sync=True)
