from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
import torch

from yeto.protocol import (
    DTYPE_F32,
    BcastFragment,
    FinalFragment,
    FinalManifest,
    PullRequest,
)
from yeto.rl.engine import progress
from yeto.rl.core import LocalRoundStats, PolicySnapshot, canonical_state
from yeto.rl.decoupled import DecoupledBridgeConfig, DecoupledRlBridge
from yeto.tensor_io import pack_fragment, unpack_fragment

MODEL_REVISION = "a" * 40
LORA_CONFIG_HASH = "b" * 64


def _tensors(offset: float = 0.0):
    return {
        "base_model.model.a.lora_A.weight": torch.tensor([[1.0, 2.0]]) + offset,
        "base_model.model.b.lora_A.weight": torch.tensor([[3.0, 4.0]]) + offset,
        "base_model.model.c.lora_A.weight": torch.tensor([[5.0]]) + offset,
        "base_model.model.d.lora_A.weight": torch.tensor([[6.0]]) + offset,
    }


def _state(version: int, offset: float = 0.0):
    return canonical_state(
        version,
        _tensors(offset),
        base_model_revision=MODEL_REVISION,
        lora_config_hash=LORA_CONFIG_HASH,
    )


def _config(tmp_path, initial, *, learner_id=0, horizon=2, learner_budget_steps=None):
    return DecoupledBridgeConfig(
        syncer_addr=("127.0.0.1", 1),
        learner_id=learner_id,
        total_fragment_steps=4,
        num_fragments=2,
        pipeline=2,
        local_horizon=horizon,
        expected_specs=initial.specs,
        base_model_revision=MODEL_REVISION,
        lora_config_hash=LORA_CONFIG_HASH,
        canonical_layout_hash=initial.layout_hash,
        wan_streams=0,
        learner_budget_steps=learner_budget_steps,
    )


@pytest.mark.parametrize("pipeline", [0, 3])
def test_decoupled_bridge_config_rejects_pipeline_outside_fragments(tmp_path, pipeline):
    initial = _state(0)
    with pytest.raises(ValueError, match="pipeline"):
        DecoupledBridgeConfig(
            syncer_addr=("127.0.0.1", 1),
            learner_id=0,
            total_fragment_steps=4,
            num_fragments=2,
            pipeline=pipeline,
            local_horizon=2,
            expected_specs=initial.specs,
            base_model_revision=MODEL_REVISION,
            lora_config_hash=LORA_CONFIG_HASH,
            canonical_layout_hash=initial.layout_hash,
        )


class _Client:
    def __init__(self):
        self.updates = []
        self.pulls = []
        self.inits = []
        self.pushes = []
        self.finalizing = SimpleNamespace(is_set=lambda: False)
        self.manifest = None
        self.finals = None
        self.finalization_timeout = 1.0

    def start(self):
        pass

    def check_health(self):
        pass

    def send_init(self, fragment_id, payload):
        self.inits.append((fragment_id, payload))

    def drain_updates(self):
        values, self.updates = self.updates, []
        return values

    def drain_pulls(self):
        values, self.pulls = self.pulls, []
        return values

    def push_fragment(self, *values):
        self.pushes.append(values)

    def wait_for_final_fragments(self):
        return self.manifest, self.finals

    def acknowledge_finalization(self, manifest):
        assert manifest == self.manifest

    def close(self):
        pass


def _bridge(tmp_path, *, learner_id=0, horizon=2, learner_budget_steps=None):
    initial = _state(0)
    bridge = DecoupledRlBridge(
        initial,
        _config(
            tmp_path,
            initial,
            learner_id=learner_id,
            horizon=horizon,
            learner_budget_steps=learner_budget_steps,
        ),
    )
    bridge.client = _Client()
    return bridge


def _bcast(bridge, fragment_id, version, state, *, received_at=None):
    fragment = bridge.layout.fragments[fragment_id]
    values = (
        fragment_id,
        version,
        pack_fragment(fragment, dict(state.tensors), DTYPE_F32),
    )
    return (
        BcastFragment(*values)
        if received_at is None
        else BcastFragment(*values, received_at=received_at)
    )


def test_initial_cut_sends_and_reassembles_every_fragment(tmp_path):
    bridge = _bridge(tmp_path)
    authoritative = _state(0, 10.0)
    bridge.client.updates = [
        _bcast(bridge, 1, 0, authoritative),
        _bcast(bridge, 0, 0, authoritative),
    ]

    bridge.start()
    cut = bridge.wait_for_initial_cut(optimizer_steps=7, action_tokens=11)

    assert [fragment_id for fragment_id, _ in bridge.client.inits] == [0, 1]
    with pytest.raises(RuntimeError, match="not initialized"):
        _ = bridge.fragment_versions
    bridge.commit_initial_cut(cut, optimizer_steps=7, action_tokens=11)
    initial = cut.state
    assert bridge.fragment_versions == (0, 0)
    assert bridge.steps_at_anchor == (7, 7)
    assert all(
        torch.equal(initial.tensors[name], authoritative.tensors[name])
        for name in initial.tensors
    )


def test_initial_cut_recovers_terminal_manifest_without_waiting_for_broadcast(tmp_path):
    bridge = _bridge(tmp_path)
    final = _state(2, 10.0)
    bridge.client.finalizing = SimpleNamespace(is_set=lambda: True)
    bridge.client.manifest = FinalManifest(4, (3, 4))
    bridge.client.finals = [
        FinalFragment(
            fragment_id,
            bridge.client.manifest.versions[fragment_id],
            pack_fragment(fragment, dict(final.tensors), DTYPE_F32),
        )
        for fragment_id, fragment in enumerate(bridge.layout.fragments)
    ]
    health_checks = 0

    def check_health():
        nonlocal health_checks
        health_checks += 1
        if health_checks > 1:
            raise RuntimeError("initialization waited for a missing broadcast")

    bridge.client.check_health = check_health

    cut = bridge.wait_for_initial_cut(
        optimizer_steps=2,
        action_tokens=10,
    )
    bridge.commit_initial_cut(cut, optimizer_steps=2, action_tokens=10)
    recovered = cut.state

    assert bridge.startup_final_manifest == bridge.client.manifest
    assert bridge.fragment_versions == (3, 4)
    assert bridge.steps_at_anchor == (2, 2)
    assert all(
        torch.equal(recovered.tensors[name], final.tensors[name])
        for name in final.tensors
    )


def test_nonleader_does_not_initialize_syncer(tmp_path):
    bridge = _bridge(tmp_path, learner_id=1)
    bridge.start()
    assert bridge.client.inits == []


def test_pull_waits_for_horizon_and_exact_same_fragment_broadcast(tmp_path):
    bridge = _bridge(tmp_path, horizon=2)
    initial = _state(0)
    bridge.client.updates = [
        _bcast(bridge, 0, 0, initial),
        _bcast(bridge, 1, 0, initial),
    ]
    cut = bridge.wait_for_initial_cut(optimizer_steps=0, action_tokens=0)
    bridge.commit_initial_cut(cut, optimizer_steps=0, action_tokens=0)
    received_at = time.monotonic() - 2.0
    bridge.client.pulls = [
        PullRequest(0, 1, 1, received_at=received_at),
        PullRequest(1, 2, 1, received_at=received_at),
    ]

    bridge.submit_ready(_state(1, 1.0), optimizer_steps=1, action_tokens=10)
    assert bridge.client.pushes == []

    submissions = bridge.submit_ready(
        _state(2, 2.0), optimizer_steps=2, action_tokens=20
    )
    assert [(push[0], push[1], push[3], push[5]) for push in bridge.client.pushes] == [
        (0, 1, 0, 2),
        (1, 2, 0, 2),
    ]
    for push in bridge.client.pushes:
        fragment = bridge.layout.fragments[push[0]]
        delta = unpack_fragment(fragment, push[-1], DTYPE_F32)
        assert torch.equal(delta, torch.full_like(delta, 2.0))
    assert all(submission.pull_to_push_seconds >= 1.9 for submission in submissions)

    bridge.client.pulls = [PullRequest(0, 3, 1)]
    bridge.submit_ready(_state(4, 4.0), optimizer_steps=4, action_tokens=40)
    assert len(bridge.client.pushes) == 2

    committed = _state(4, 10.0)
    bridge.client.updates = [_bcast(bridge, 0, 1, committed)]
    batch = bridge.drain_broadcasts(_state(4, 4.0), optimizer_steps=4, action_tokens=40)
    assert batch.fragment_ids == (0,)
    bridge.commit_broadcasts(batch, optimizer_steps=4, action_tokens=40)
    bridge.submit_ready(batch.state, optimizer_steps=5, action_tokens=50)
    assert len(bridge.client.pushes) == 2
    bridge.submit_ready(batch.state, optimizer_steps=6, action_tokens=60)
    assert len(bridge.client.pushes) == 3
    assert bridge.client.pushes[-1][3] == 1
    assert bridge.client.pushes[-1][5:7] == (2, 20)


def test_broadcast_state_commits_only_after_the_caller_accepts_the_batch(tmp_path):
    bridge = _bridge(tmp_path)
    initial = _state(0)
    bridge.client.updates = [
        _bcast(bridge, 0, 0, initial),
        _bcast(bridge, 1, 0, initial),
    ]
    cut = bridge.wait_for_initial_cut(optimizer_steps=0, action_tokens=0)
    bridge.commit_initial_cut(cut, optimizer_steps=0, action_tokens=0)
    authoritative = _state(1, 3.0)
    bridge.client.updates = [
        _bcast(
            bridge,
            0,
            1,
            authoritative,
            received_at=time.monotonic() - 2.0,
        )
    ]

    batch = bridge.drain_broadcasts(
        _state(1, 1.0),
        optimizer_steps=4,
        action_tokens=40,
    )

    assert bridge.fragment_versions == (0, 0)
    assert batch.broadcasts[0].queue_seconds >= 1.9
    bridge.commit_broadcasts(
        batch,
        optimizer_steps=4,
        action_tokens=40,
    )
    assert bridge.fragment_versions == (1, 0)
    assert bridge.steps_at_anchor == (4, 0)
    assert bridge.tokens_at_anchor == (40, 0)


def test_live_bridge_rejects_invalid_broadcast_and_pull_identities(tmp_path):
    bridge = _bridge(tmp_path)
    initial = _state(0)
    fragment = bridge.layout.fragments[0]
    bridge.client.updates = [
        BcastFragment(
            0,
            2,
            pack_fragment(fragment, dict(initial.tensors), DTYPE_F32),
        )
    ]
    with pytest.raises(RuntimeError, match="invalid fragment version"):
        bridge.drain_broadcasts(
            initial,
            optimizer_steps=0,
            action_tokens=0,
        )

    bridge.client.pulls = [PullRequest(0, 2, 1)]
    with pytest.raises(RuntimeError, match="invalid PULL permit"):
        bridge.submit_ready(initial, optimizer_steps=2, action_tokens=0)


def test_live_bridge_ignores_duplicate_broadcast_and_pull(tmp_path):
    bridge = _bridge(tmp_path)
    initial = _state(0)
    bridge.client.updates = [
        _bcast(bridge, 0, 0, initial),
        _bcast(bridge, 1, 0, initial),
    ]
    cut = bridge.wait_for_initial_cut(optimizer_steps=0, action_tokens=0)
    bridge.commit_initial_cut(cut, optimizer_steps=0, action_tokens=0)
    bridge.client.updates = [_bcast(bridge, 0, 0, initial)]
    assert (
        bridge.drain_broadcasts(
            initial,
            optimizer_steps=1,
            action_tokens=0,
        ).fragment_ids
        == ()
    )

    bridge.client.pulls = [
        PullRequest(0, 1, 1, received_at=time.monotonic() - 2.0),
        PullRequest(0, 1, 1, received_at=time.monotonic() - 1.0),
    ]
    submissions = bridge.submit_ready(
        _state(2, 1.0),
        optimizer_steps=2,
        action_tokens=0,
    )
    assert len(submissions) == 1
    assert submissions[0].pull_to_push_seconds >= 1.9
    assert len(bridge.client.pushes) == 1


def test_budget_consolidation_rejects_invalid_broadcast_identity(tmp_path):
    bridge = _bridge(tmp_path, learner_budget_steps=2)
    bridge.client.send_budget_done = lambda _target: 1
    bridge.client.wait_for_budget_restart = lambda _generation: None
    bridge.client.updates = [BcastFragment(2, 0, b"")]

    with pytest.raises(RuntimeError, match="invalid fragment version"):
        bridge.consolidate_budget(_state(2, 1.0), optimizer_steps=2, action_tokens=20)


def test_budget_consolidation_allows_a_zero_action_token_audit_counter(tmp_path):
    bridge = _bridge(tmp_path, learner_budget_steps=2)

    class ReachedProtocol(RuntimeError):
        pass

    def send_budget_done(_target):
        raise ReachedProtocol

    bridge.client.send_budget_done = send_budget_done
    with pytest.raises(ReachedProtocol):
        bridge.consolidate_budget(_state(2, 1.0), optimizer_steps=2, action_tokens=0)


def test_budget_consolidation_rejects_invalid_pull_identity(tmp_path):
    bridge = _bridge(tmp_path, learner_budget_steps=2)
    authoritative = _state(0)
    bridge.client.send_budget_done = lambda _target: 1
    bridge.client.wait_for_budget_restart = lambda _generation: None
    bridge.client.updates = [
        _bcast(bridge, 0, 0, authoritative),
        _bcast(bridge, 1, 0, authoritative),
    ]
    bridge.client.pulls = [PullRequest(0, 2, 1)]
    original_push = bridge.client.push_fragment

    def push(*values):
        original_push(*values)
        if len(bridge.client.pushes) == 1:
            bridge.client.pulls = [PullRequest(1, 2, 1)]

    bridge.client.push_fragment = push
    bridge.client.manifest = FinalManifest(3, (3, 2))
    final = _state(2, 1.0)
    bridge.client.finals = [
        FinalFragment(
            fragment_id,
            bridge.client.manifest.versions[fragment_id],
            pack_fragment(fragment, dict(final.tensors), DTYPE_F32),
        )
        for fragment_id, fragment in enumerate(bridge.layout.fragments)
    ]

    with pytest.raises(RuntimeError, match="invalid PULL permit"):
        bridge.consolidate_budget(final, optimizer_steps=2, action_tokens=20)


def test_final_cut_requires_complete_sweep_and_reassembles_f32(tmp_path):
    bridge = _bridge(tmp_path)
    final = _state(9, 20.0)
    bridge.client.manifest = FinalManifest(4, (3, 4))
    bridge.client.finals = [
        FinalFragment(
            fragment_id,
            bridge.client.manifest.versions[fragment_id],
            pack_fragment(fragment, dict(final.tensors), DTYPE_F32),
        )
        for fragment_id, fragment in enumerate(bridge.layout.fragments)
    ]

    manifest, state = bridge.wait_for_final_cut(policy_version=9)

    assert manifest == bridge.client.manifest
    assert all(
        torch.equal(state.tensors[name], final.tensors[name]) for name in state.tensors
    )
    bridge.acknowledge_finalization(manifest)


def test_final_cut_rejects_incomplete_fragment_sweep(tmp_path):
    bridge = _bridge(tmp_path)
    bridge.client.manifest = FinalManifest(3, (3, 2))
    bridge.client.finals = []
    with pytest.raises(RuntimeError, match="complete fragment sweep"):
        bridge.wait_for_final_cut(policy_version=2)


def _checkpoint_args(tmp_path):
    initial = _state(0)
    return SimpleNamespace(
        actor_num_gpus_per_node=1,
        actor_num_nodes=1,
        advantage_estimator="grpo",
        yeto_rl_model="org/model",
        yeto_rl_data="org/data",
        yeto_rl_base_model_revision=MODEL_REVISION,
        yeto_rl_data_revision="c" * 40,
        expert_model_parallel_size=1,
        yeto_rl_layout_hash=initial.layout_hash,
        lr=1e-5,
        yeto_rl_lora_config_hash=LORA_CONFIG_HASH,
        n_samples_per_prompt=2,
        num_steps_per_rollout=1,
        pipeline_model_parallel_size=1,
        over_sampling_batch_size=2,
        yeto_rl_reward_sha256="d" * 64,
        yeto_rl_source_sha256="f" * 64,
        rollout_batch_size=1,
        seq_length=128,
        seed=7,
        rollout_max_response_len=32,
        custom_generate_function_path=None,
        use_session_server=False,
        tito_model=None,
        yeto_rl_learner_id=0,
        yeto_rl_sync_preset="decoupled",
        yeto_rl_num_fragments=2,
        yeto_rl_pipeline=2,
        yeto_rl_local_horizon=2,
        yeto_rl_total_sweeps=2,
        yeto_rl_total_fragment_steps=4,
        yeto_rl_sync_layout_fingerprint="e" * 64,
        yeto_rl_bridge_config=object(),
        yeto_rl_completed_groups_path=str(tmp_path / "island.pt"),
    )


def test_decoupled_checkpoint_round_trips_progress_without_local_lora(tmp_path):
    args = _checkpoint_args(tmp_path)
    snapshot = PolicySnapshot.create(3, _state(3), (1, 2))

    progress._save_decoupled_checkpoint(
        args,
        snapshot=snapshot,
        optimizer_steps=3,
        action_tokens=41,
        rollout_metrics={"reward": 0.5},
        local_round_stats={"completed_groups": 1},
        completed_groups=[],
    )
    payload = progress._load_decoupled_checkpoint(args)

    assert payload["schema_version"] == 4
    assert payload["next_rollout_id"] == 3
    assert payload["optimizer_steps"] == 3
    assert payload["action_tokens"] == 41
    assert payload["policy_token"] == snapshot.token
    assert payload["fragment_versions"] == [1, 2]
    assert "initial_adapter_sha256" not in payload["config"]
    assert "tensors" not in payload and "local_lora" not in payload


@pytest.mark.parametrize(
    "name,value",
    [
        ("yeto_rl_learner_id", 1),
        ("yeto_rl_pipeline", 1),
        ("yeto_rl_local_horizon", 3),
        ("yeto_rl_source_sha256", "0" * 64),
        ("yeto_rl_initial_adapter_sha256", "9" * 64),
    ],
)
def test_decoupled_checkpoint_rejects_immutable_config_drift(tmp_path, name, value):
    args = _checkpoint_args(tmp_path)
    snapshot = PolicySnapshot.create(0, _state(0), (0, 0))
    progress._save_decoupled_checkpoint(
        args,
        snapshot=snapshot,
        optimizer_steps=0,
        action_tokens=0,
        rollout_metrics={},
        local_round_stats=None,
        completed_groups=[],
    )
    setattr(args, name, value)
    with pytest.raises(RuntimeError, match="configuration"):
        progress._load_decoupled_checkpoint(args)


def test_decoupled_checkpoint_rejects_invalid_fragment_version_vector(tmp_path):
    args = _checkpoint_args(tmp_path)
    snapshot = PolicySnapshot.create(0, _state(0), (0, 0))
    progress._save_decoupled_checkpoint(
        args,
        snapshot=snapshot,
        optimizer_steps=0,
        action_tokens=0,
        rollout_metrics={},
        local_round_stats=None,
        completed_groups=[],
    )
    path = tmp_path / "island.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["fragment_versions"] = [2, 2]
    torch.save(payload, path)

    with pytest.raises(RuntimeError, match="progress"):
        progress._load_decoupled_checkpoint(args)


def _stats(rollout_id=0):
    return LocalRoundStats(
        island_id=0,
        local_round_id=rollout_id + 1,
        base_policy_version=rollout_id,
        active_groups=1,
        completed_groups=1,
        cancelled_groups=0,
        completed_trajectories=2,
        action_tokens=5,
        tool_wait_seconds=0.0,
        group_p50_seconds=0.1,
        group_p95_seconds=0.1,
        group_p99_seconds=0.1,
        reward_mean=1.0,
        reward_std=0.0,
        zero_variance_group_ratio=1.0,
        mean_kl=0.0,
        ess_ratio=1.0,
        clip_fraction=0.0,
        delta_l2_norm=0.0,
        rollout_seconds=0.1,
        train_seconds=0.1,
        train_step=rollout_id,
        loss=-0.4,
        pg_loss=-0.5,
        grad_norm=1.25,
        lr=1e-6,
        pass_rate=0.5,
    )


