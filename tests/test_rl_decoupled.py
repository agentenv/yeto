from __future__ import annotations

import asyncio
import json
import sys
import time
import types
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
from yeto.rl import miles
from yeto.rl.core import (
    LocalRoundStats,
    PolicySnapshot,
    canonical_state,
    policy_tensor_hash,
)
from yeto.rl.decoupled import (
    BroadcastBatch,
    BudgetConsolidation,
    DecoupledBridgeConfig,
    DecoupledRlBridge,
    FragmentSubmission,
    InitialCut,
)
from yeto.rl.miles import DecoupledMilesPolicySync
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

    miles._save_decoupled_checkpoint(
        args,
        snapshot=snapshot,
        optimizer_steps=3,
        action_tokens=41,
        rollout_metrics={"reward": 0.5},
        local_round_stats={"completed_groups": 1},
        completed_groups=[],
    )
    payload = miles._load_decoupled_checkpoint(args)

    assert payload["schema_version"] == 3
    assert payload["next_rollout_id"] == 3
    assert payload["optimizer_steps"] == 3
    assert payload["action_tokens"] == 41
    assert payload["policy_token"] == snapshot.token
    assert payload["fragment_versions"] == [1, 2]
    assert "initial_adapter_sha256" not in payload["config"]
    assert "tensors" not in payload and "local_lora" not in payload


def test_decoupled_rollout_reads_snapshot_token_from_island_checkpoint(tmp_path):
    hook_args = _checkpoint_args(tmp_path)
    rollout_args = SimpleNamespace(**vars(hook_args))
    snapshot = PolicySnapshot.create(3, _state(3), (1, 2))
    miles._save_decoupled_checkpoint(
        hook_args,
        snapshot=snapshot,
        optimizer_steps=3,
        action_tokens=41,
        rollout_metrics={},
        local_round_stats=None,
        completed_groups=[],
    )

    assert miles._policy_token_for_rollout(rollout_args, 3) == snapshot.token


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
    miles._save_decoupled_checkpoint(
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
        miles._load_decoupled_checkpoint(args)


def test_decoupled_checkpoint_rejects_invalid_fragment_version_vector(tmp_path):
    args = _checkpoint_args(tmp_path)
    snapshot = PolicySnapshot.create(0, _state(0), (0, 0))
    miles._save_decoupled_checkpoint(
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
        miles._load_decoupled_checkpoint(args)


def test_decoupled_initialize_rejects_nonzero_cut_without_island_progress(
    tmp_path, monkeypatch
):
    args = _checkpoint_args(tmp_path)
    args.yeto_rl_event_tape = str(tmp_path / "events.jsonl")
    args.yeto_rl_learner_id = 0
    initial = _state(0)

    class Actor:
        async def export_trainable_state(self):
            return SimpleNamespace(
                policy_version=0,
                layout_hash=initial.layout_hash,
                tensors=initial.tensors,
            )

    class Bridge:
        fragment_versions = (1, 0)

        def __init__(self, *_args):
            pass

        def start(self):
            pass

        def wait_for_initial_cut(self, **_progress):
            return InitialCut(initial, self.fragment_versions)

    monkeypatch.setattr("yeto.rl.decoupled.DecoupledRlBridge", Bridge)
    hook = DecoupledMilesPolicySync(args)

    with pytest.raises(RuntimeError, match="no valid island checkpoint"):
        asyncio.run(hook._initialize(actor_model=Actor(), rollout_manager=object()))


def test_decoupled_initialize_seeds_a_fresh_phase_from_final_adapter(
    tmp_path, monkeypatch
):
    args = _checkpoint_args(tmp_path)
    args.yeto_rl_event_tape = str(tmp_path / "events.jsonl")
    args.yeto_rl_initial_adapter = "/parent-adapter"
    args.yeto_rl_initial_adapter_sha256 = "9" * 64
    fresh = _state(0)
    parent = _state(0, 4.0)
    calls = []

    def load(path, digest, *, model, expected):
        calls.append((path, digest, model, policy_tensor_hash(expected)))
        assert policy_tensor_hash(expected) == policy_tensor_hash(fresh)
        return parent

    class Actor:
        async def export_trainable_state(self):
            return SimpleNamespace(
                policy_version=0,
                layout_hash=fresh.layout_hash,
                tensors=fresh.tensors,
            )

    class Bridge:
        fragment_versions = (0, 0)
        startup_final_manifest = None

        def __init__(self, initial, _config):
            assert policy_tensor_hash(initial) == policy_tensor_hash(parent)

        def start(self):
            pass

        def wait_for_initial_cut(self, **progress):
            assert progress == {"optimizer_steps": 0, "action_tokens": 0}
            return InitialCut(parent, self.fragment_versions)

        def commit_initial_cut(self, cut, **progress):
            assert policy_tensor_hash(cut.state) == policy_tensor_hash(parent)
            assert progress == {"optimizer_steps": 0, "action_tokens": 0}

    monkeypatch.setattr("yeto.rl.initial_adapter.load_initial_adapter", load)
    monkeypatch.setattr("yeto.rl.decoupled.DecoupledRlBridge", Bridge)
    hook = DecoupledMilesPolicySync(args)

    async def apply(state, *, reset_optimizer):
        assert reset_optimizer is True
        return state

    async def publish(snapshot):
        args.yeto_rl_policy_token = snapshot.token

    hook._apply_decoupled_policy = apply
    hook._publish_snapshot = publish
    asyncio.run(hook._initialize(actor_model=Actor(), rollout_manager=object()))

    assert calls == [
        (
            "/parent-adapter",
            "9" * 64,
            "org/model",
            policy_tensor_hash(fresh),
        )
    ]
    recovered = DecoupledMilesPolicySync(args)
    recovered._apply_decoupled_policy = apply
    recovered._publish_snapshot = publish
    asyncio.run(recovered._initialize(actor_model=Actor(), rollout_manager=object()))

    payload = miles._load_decoupled_checkpoint(args)
    assert payload["config"]["initial_adapter_sha256"] == "9" * 64
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    initial_events = [event for event in events if event["event"] == "rl_initial_adapter"]
    assert len(initial_events) == 1
    initial_event = initial_events[0]
    assert initial_event["parent_adapter_sha256"] == "9" * 64
    assert initial_event["parent_policy_hash"] == policy_tensor_hash(parent)


@pytest.mark.parametrize("with_checkpoint", [False, True])
def test_decoupled_initialize_rejects_a_different_parent_version_zero_cut(
    tmp_path, monkeypatch, with_checkpoint
):
    args = _checkpoint_args(tmp_path)
    args.yeto_rl_event_tape = str(tmp_path / "events.jsonl")
    args.yeto_rl_initial_adapter = "/parent-adapter"
    args.yeto_rl_initial_adapter_sha256 = "9" * 64
    fresh = _state(0)
    parent = _state(0, 4.0)
    if with_checkpoint:
        miles._save_decoupled_checkpoint(
            args,
            snapshot=PolicySnapshot.create(0, parent, (0, 0)),
            optimizer_steps=0,
            action_tokens=0,
            rollout_metrics={},
            local_round_stats=None,
            completed_groups=[],
        )

    class Actor:
        async def export_trainable_state(self):
            return SimpleNamespace(
                policy_version=0,
                layout_hash=fresh.layout_hash,
                tensors=fresh.tensors,
            )

    class Bridge:
        fragment_versions = (0, 0)

        def __init__(self, *_args):
            pass

        def start(self):
            pass

        def wait_for_initial_cut(self, **_progress):
            return InitialCut(_state(0, 5.0), self.fragment_versions)

    monkeypatch.setattr(
        "yeto.rl.initial_adapter.load_initial_adapter",
        lambda *_args, **_kwargs: parent,
    )
    monkeypatch.setattr("yeto.rl.decoupled.DecoupledRlBridge", Bridge)
    hook = DecoupledMilesPolicySync(args)

    with pytest.raises(RuntimeError, match="initial adapter policy"):
        asyncio.run(hook._initialize(actor_model=Actor(), rollout_manager=object()))


def test_decoupled_recovery_applies_newer_uneven_cut_at_saved_local_progress(
    tmp_path, monkeypatch
):
    args = _checkpoint_args(tmp_path)
    args.yeto_rl_event_tape = str(tmp_path / "events.jsonl")
    args.yeto_rl_learner_id = 0
    initial = _state(0)
    saved = PolicySnapshot.create(2, _state(2), (1, 0))
    miles._save_decoupled_checkpoint(
        args,
        snapshot=saved,
        optimizer_steps=2,
        action_tokens=10,
        rollout_metrics={},
        local_round_stats=None,
        completed_groups=[],
    )
    checkpoint_path = tmp_path / "island.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    payload["completed_groups"] = [[{"weight_versions": [saved.token]}]]
    torch.save(payload, checkpoint_path)
    calls = []

    class Actor:
        async def export_trainable_state(self):
            return SimpleNamespace(
                policy_version=0,
                layout_hash=initial.layout_hash,
                tensors=initial.tensors,
            )

    class Bridge:
        fragment_versions = (3, 2)

        def __init__(self, *_args):
            pass

        def start(self):
            pass

        def wait_for_initial_cut(self, **progress):
            assert progress == {"optimizer_steps": 2, "action_tokens": 10}
            return InitialCut(_state(0, 4.0), self.fragment_versions)

        def commit_initial_cut(self, cut, **progress):
            calls.append("commit")
            assert cut.fragment_versions == self.fragment_versions
            assert progress == {"optimizer_steps": 2, "action_tokens": 10}

    monkeypatch.setattr("yeto.rl.decoupled.DecoupledRlBridge", Bridge)
    hook = DecoupledMilesPolicySync(args)
    applied = []

    async def apply(state, *, reset_optimizer):
        calls.append("apply")
        applied.append((state, reset_optimizer))
        return state

    async def publish(snapshot):
        args.yeto_rl_policy_token = snapshot.token

    hook._apply_decoupled_policy = apply
    hook._publish_snapshot = publish
    asyncio.run(hook._initialize(actor_model=Actor(), rollout_manager=object()))

    assert applied[0][0].policy_version == 2
    assert applied[0][1] is True
    assert calls == ["apply", "commit"]
    assert hook.snapshot.fragment_versions == (3, 2)
    assert hook.optimizer_steps == 2
    assert args.start_rollout_id == 2
    assert miles._load_decoupled_checkpoint(args)["completed_groups"] == []


def test_decoupled_recovery_preserves_groups_for_the_exact_rebuilt_snapshot(
    tmp_path, monkeypatch
):
    args = _checkpoint_args(tmp_path)
    args.yeto_rl_event_tape = str(tmp_path / "events.jsonl")
    recovered = _state(2, 4.0)
    snapshot = PolicySnapshot.create(2, recovered, (1, 0))
    miles._save_decoupled_checkpoint(
        args,
        snapshot=snapshot,
        optimizer_steps=2,
        action_tokens=10,
        rollout_metrics={},
        local_round_stats=None,
        completed_groups=[],
    )
    path = tmp_path / "island.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["completed_groups"] = [
        [
            {
                "status": "completed",
                "weight_versions": [snapshot.token],
                "index": index,
            }
            for index in range(2)
        ]
    ]
    torch.save(payload, path)

    initial = _state(0)

    class Actor:
        async def export_trainable_state(self):
            return SimpleNamespace(
                policy_version=0,
                layout_hash=initial.layout_hash,
                tensors=initial.tensors,
            )

    class Bridge:
        fragment_versions = (1, 0)

        def __init__(self, *_args):
            pass

        def start(self):
            pass

        def wait_for_initial_cut(self, **progress):
            assert progress == {"optimizer_steps": 2, "action_tokens": 10}
            return InitialCut(recovered, self.fragment_versions)

        def commit_initial_cut(self, cut, **progress):
            assert cut.fragment_versions == self.fragment_versions
            assert progress == {"optimizer_steps": 2, "action_tokens": 10}

    monkeypatch.setattr("yeto.rl.decoupled.DecoupledRlBridge", Bridge)
    hook = DecoupledMilesPolicySync(args)

    async def apply(state, *, reset_optimizer):
        assert reset_optimizer is True
        return state

    async def publish(value):
        args.yeto_rl_policy_token = value.token

    hook._apply_decoupled_policy = apply
    hook._publish_snapshot = publish
    asyncio.run(hook._initialize(actor_model=Actor(), rollout_manager=object()))

    restored = miles._load_decoupled_checkpoint(args)
    assert restored["policy_token"] == snapshot.token
    assert len(restored["completed_groups"]) == 1


def test_decoupled_recovery_acknowledges_terminal_cut_without_an_extra_rollout(
    tmp_path, monkeypatch
):
    args = _checkpoint_args(tmp_path)
    args.yeto_rl_event_tape = str(tmp_path / "events.jsonl")
    args.external_policy_sync_run_until_stop = True
    args.num_rollout = 2
    initial = _state(0)
    saved = PolicySnapshot.create(2, _state(2), (1, 0))
    miles._save_decoupled_checkpoint(
        args,
        snapshot=saved,
        optimizer_steps=2,
        action_tokens=10,
        rollout_metrics={},
        local_round_stats=None,
        completed_groups=[],
    )
    manifest = FinalManifest(4, (3, 4))

    class Actor:
        async def export_trainable_state(self):
            return SimpleNamespace(
                policy_version=0,
                layout_hash=initial.layout_hash,
                tensors=initial.tensors,
            )

    class Bridge:
        fragment_versions = manifest.versions
        startup_final_manifest = manifest
        final_payload_bytes_received = 16

        def __init__(self, *_args):
            self.acknowledged = None

        def start(self):
            pass

        def wait_for_initial_cut(self, **progress):
            assert progress == {"optimizer_steps": 2, "action_tokens": 10}
            return InitialCut(_state(2, 4.0), self.fragment_versions)

        def commit_initial_cut(self, cut, **progress):
            assert cut.fragment_versions == self.fragment_versions
            assert progress == {"optimizer_steps": 2, "action_tokens": 10}

        def acknowledge_finalization(self, value):
            self.acknowledged = value

    monkeypatch.setattr("yeto.rl.decoupled.DecoupledRlBridge", Bridge)
    hook = DecoupledMilesPolicySync(args)
    applied = []

    async def apply(state, *, reset_optimizer):
        applied.append((state, reset_optimizer))
        return state

    async def publish(snapshot):
        args.yeto_rl_policy_token = snapshot.token

    hook._apply_decoupled_policy = apply
    hook._publish_snapshot = publish
    asyncio.run(hook._initialize(actor_model=Actor(), rollout_manager=object()))

    assert applied[0][1] is True
    assert hook.bridge.acknowledged == manifest
    assert hook.finished
    assert args.external_policy_sync_run_until_stop is False
    assert args.num_rollout == args.start_rollout_id == 2
    assert miles._load_decoupled_checkpoint(args)["fragment_versions"] == [3, 4]


def test_decoupled_completed_groups_restore_only_the_exact_snapshot(
    tmp_path, monkeypatch
):
    args = _checkpoint_args(tmp_path)
    snapshot = PolicySnapshot.create(3, _state(3), (1, 2))
    args.yeto_rl_policy_token = snapshot.token
    miles._save_decoupled_checkpoint(
        args,
        snapshot=snapshot,
        optimizer_steps=3,
        action_tokens=10,
        rollout_metrics={},
        local_round_stats=None,
        completed_groups=[],
    )

    def sample(token, index):
        value = SimpleNamespace(
            status=SimpleNamespace(value="completed"),
            weight_versions=[token],
            index=index,
        )
        value.to_dict = lambda: {
            "status": "completed",
            "weight_versions": [token],
            "index": index,
        }
        return value

    class Sample:
        @staticmethod
        def from_dict(value):
            return sample(value["weight_versions"][0], value["index"])

    package = types.ModuleType("miles")
    utils = types.ModuleType("miles.utils")
    sample_types = types.ModuleType("miles.utils.types")
    sample_types.Sample = Sample
    package.utils = utils
    utils.types = sample_types
    monkeypatch.setitem(sys.modules, "miles", package)
    monkeypatch.setitem(sys.modules, "miles.utils", utils)
    monkeypatch.setitem(sys.modules, "miles.utils.types", sample_types)

    valid = [sample(snapshot.token, 0), sample(snapshot.token, 1)]
    stale = [sample("yeto:3", 2), sample("yeto:3", 3)]
    source = SimpleNamespace(buffer=[valid, stale])

    miles._save_completed_groups(args, 3, 4, source, {"reward": 1.0})

    payload = miles._load_decoupled_checkpoint(args)
    assert payload["optimizer_steps"] == 3
    assert len(payload["completed_groups"]) == 1
    restored = SimpleNamespace(
        buffer=[], add_samples=lambda groups: restored.buffer.extend(groups)
    )
    miles._restore_completed_groups(args, 3, restored)
    assert len(restored.buffer) == 1


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
    )


def test_decoupled_hook_preserves_optimizer_and_publishes_one_full_snapshot(
    tmp_path, monkeypatch
):
    args = _checkpoint_args(tmp_path)
    args.yeto_rl_event_tape = str(tmp_path / "events.jsonl")
    args.yeto_rl_learner_id = 0
    current = _state(0)
    initial_snapshot = PolicySnapshot.create(0, current, (0, 0))
    args.yeto_rl_policy_token = initial_snapshot.token
    miles._save_decoupled_checkpoint(
        args,
        snapshot=initial_snapshot,
        optimizer_steps=0,
        action_tokens=0,
        rollout_metrics={},
        local_round_stats=None,
        completed_groups=[],
    )

    trainable_state = types.ModuleType("miles.backends.megatron_utils.trainable_state")
    trainable_state.make_trainable_state = lambda version, values: SimpleNamespace(
        policy_version=version,
        layout_hash=current.layout_hash,
        tensors=values,
    )
    for name in ("miles", "miles.backends", "miles.backends.megatron_utils"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(
        sys.modules,
        "miles.backends.megatron_utils.trainable_state",
        trainable_state,
    )

    class Remote:
        def __init__(self, function):
            self.function = function

        async def remote(self, *args, **kwargs):
            return self.function(*args, **kwargs)

    published = []
    engine = SimpleNamespace(
        update_weight_version=Remote(lambda token: published.append(token))
    )
    rollout_manager = SimpleNamespace(
        get_updatable_engines_and_lock=Remote(
            lambda: SimpleNamespace(rollout_engines=[engine])
        )
    )

    class Actor:
        def __init__(self):
            self.state = SimpleNamespace(
                policy_version=0,
                layout_hash=current.layout_hash,
                tensors=_state(0, 1.0).tensors,
            )
            self.resets = []

        async def export_trainable_state(self):
            return self.state

        async def apply_trainable_state(self, state, *, reset_optimizer):
            bridge_calls.append("apply")
            self.resets.append(reset_optimizer)
            self.state = state
            return 0

    actor = Actor()
    bridge_calls = []

    class Bridge:
        fragment_versions = (1, 0)
        finalizing = False

        def drain_broadcasts(self, local, **progress):
            bridge_calls.append("broadcast")
            assert local.policy_version == 1
            assert progress == {"optimizer_steps": 1, "action_tokens": 5}
            return BroadcastBatch(
                _state(1, 2.0),
                (
                    SimpleNamespace(
                        fragment_id=0,
                        version=1,
                        anchor=torch.tensor([5.0, 6.0, 7.0]),
                        payload_bytes=8,
                        queue_seconds=0.25,
                    ),
                ),
            )

        def commit_broadcasts(self, batch, **progress):
            bridge_calls.append("commit")
            assert batch.fragment_ids == (0,)
            assert progress == {"optimizer_steps": 1, "action_tokens": 5}

        def submit_ready(self, local, **progress):
            bridge_calls.append("pull")
            assert local.policy_version == 1
            assert progress == {"optimizer_steps": 1, "action_tokens": 5}
            return (FragmentSubmission(1, 2, 1, 0, 1, 5, 0.5, 8, 0.1),)

    hook = DecoupledMilesPolicySync(args)
    hook.actor_model = actor
    hook.rollout_manager = rollout_manager
    hook.bridge = Bridge()
    hook.current = current
    hook.snapshot = initial_snapshot
    hook.optimizer_steps = 0
    hook.action_tokens = 0
    hook._round_stats = lambda *_args, **_kwargs: _stats(0)

    should_stop = asyncio.run(
        hook.after_local_train(
            rollout_id=0,
            actor_model=actor,
            rollout_data=object(),
        )
    )

    assert not should_stop
    assert actor.resets == [False]
    assert bridge_calls == ["broadcast", "apply", "commit", "pull"]
    assert len(published) == 1
    assert published[0].startswith("yeto:1:")
    payload = miles._load_decoupled_checkpoint(args)
    assert payload["next_rollout_id"] == 1
    assert payload["fragment_versions"] == [1, 0]
    assert payload["completed_groups"] == []
    snapshots = [
        event
        for event in (
            json.loads(line)
            for line in (tmp_path / "events.jsonl").read_text().splitlines()
        )
        if event["event"] == "rl_policy_snapshot"
    ]
    assert snapshots[-1]["rl/policy_token"] == published[0]
    assert snapshots[-1]["rl/canonical_layout_hash"] == current.layout_hash
    assert snapshots[-1]["rl/sync_layout_fingerprint"] == "e" * 64
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    hooks = [event for event in events if event["event"] == "rl_sync_hook"]
    assert len(hooks) == 1
    assert hooks[0]["sync/hook_seconds"] >= 0
    assert hooks[0]["sync/remote_quorum_wait_seconds"] == 0
    assert hooks[0]["sync/finalization"] is False
    applies = [event for event in events if event["event"] == "rl_policy_apply"]
    assert applies[-1]["sync/apply_seconds"] >= 0
    broadcasts = [event for event in events if event["event"] == "rl_fragment_bcast"]
    assert len(broadcasts) == 1
    assert {
        key: broadcasts[0][key]
        for key in ("fragment_id", "version", "payload_bytes", "queue_seconds")
    } == {
        "fragment_id": 0,
        "version": 1,
        "payload_bytes": 8,
        "queue_seconds": 0.25,
    }
    local_round = next(event for event in events if event["event"] == "rl_local_round")
    assert local_round["rl/completed_groups"] == 1
    assert local_round["rl/action_tokens"] == 5
    assert local_round["rl/current_vs_rollout_kl"] == 0.0
    assert local_round["sync/fragment_payload_bytes_received"] == 8
    assert local_round["sync/fragment_payload_bytes_sent"] == 8


def test_decoupled_hook_records_local_work_when_finalization_is_already_pending(
    tmp_path,
):
    args = _checkpoint_args(tmp_path)
    args.yeto_rl_event_tape = str(tmp_path / "events.jsonl")
    current = _state(0)
    snapshot = PolicySnapshot.create(0, current, (0, 0))
    args.yeto_rl_policy_token = snapshot.token
    miles._save_decoupled_checkpoint(
        args,
        snapshot=snapshot,
        optimizer_steps=0,
        action_tokens=0,
        rollout_metrics={},
        local_round_stats=None,
        completed_groups=[],
    )

    class Actor:
        async def export_trainable_state(self):
            return SimpleNamespace(
                policy_version=0,
                layout_hash=current.layout_hash,
                tensors=_state(0, 1.0).tensors,
            )

    class Bridge:
        finalizing = True
        final_payload_bytes_received = 16

        def __init__(self):
            self.acknowledged = None

        def wait_for_final_cut(self, *, policy_version):
            assert policy_version == 1
            return FinalManifest(4, (3, 4)), _state(1, 3.0)

        def acknowledge_finalization(self, manifest):
            self.acknowledged = manifest

    actor = Actor()
    hook = DecoupledMilesPolicySync(args)
    hook.actor_model = actor
    hook.rollout_manager = object()
    hook.bridge = Bridge()
    hook.current = current
    hook.snapshot = snapshot
    hook._round_stats = lambda *_args, **_kwargs: _stats(0)

    async def apply(state, *, reset_optimizer):
        assert reset_optimizer is False
        return state

    async def publish(value):
        args.yeto_rl_policy_token = value.token

    hook._apply_decoupled_policy = apply
    hook._publish_snapshot = publish

    assert asyncio.run(
        hook._after_local_train(
            rollout_id=0,
            actor_model=actor,
            rollout_data=object(),
        )
    )
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert [event["event"] for event in events].count("rl_local_round") == 1
    final = next(event for event in events if event["event"] == "rl_final_cut")
    assert final["sync/fragment_payload_bytes_received"] == 16


def test_decoupled_hook_waits_after_submitting_the_last_fragment_step(tmp_path):
    args = _checkpoint_args(tmp_path)
    args.yeto_rl_total_fragment_steps = 2
    current = _state(0)
    snapshot = PolicySnapshot.create(0, current, (0, 0))
    args.yeto_rl_policy_token = snapshot.token

    class Actor:
        async def export_trainable_state(self):
            return object()

    class Bridge:
        finalizing = False
        fragment_versions = (0, 0)

        def drain_broadcasts(self, local, **_progress):
            return BroadcastBatch(local, ())

        def submit_ready(self, _local, **_progress):
            return (FragmentSubmission(1, 2, 1, 0, 2, 5, 0.5, 8, 0.1),)

    actor = Actor()
    hook = DecoupledMilesPolicySync(args)
    hook.actor_model = actor
    hook.bridge = Bridge()
    hook.current = current
    hook.snapshot = snapshot
    hook._canonical_at_progress = lambda _exported, version: _state(version)
    hook._round_stats = lambda *_args, **_kwargs: _stats(0)
    hook._record_local_round = lambda *_args, **_kwargs: None
    hook._save_progress = lambda *_args, **_kwargs: None

    async def publish(_snapshot):
        pass

    hook._publish_snapshot = publish
    finalization = []

    async def finish(*, policy_version, stats):
        finalization.append((policy_version, stats))
        return True

    hook._finish = finish

    should_stop = asyncio.run(
        hook._after_local_train(
            rollout_id=0,
            actor_model=actor,
            rollout_data=object(),
        )
    )

    assert should_stop
    assert len(finalization) == 1
    assert finalization[0][0] == 1
    assert finalization[0][1].delta_l2_norm == 0.5


def test_decoupled_hook_freezes_at_benchmark_budget_and_stops_after_final_cut(tmp_path):
    args = _checkpoint_args(tmp_path)
    args.yeto_rl_event_tape = str(tmp_path / "events.jsonl")
    args.yeto_rl_learner_id = 0
    args.yeto_rl_learner_budget_steps = 1
    current = _state(0)
    snapshot = PolicySnapshot.create(0, current, (0, 0))
    args.yeto_rl_policy_token = snapshot.token
    miles._save_decoupled_checkpoint(
        args,
        snapshot=snapshot,
        optimizer_steps=0,
        action_tokens=0,
        rollout_metrics={},
        local_round_stats=None,
        completed_groups=[],
    )

    class Actor:
        async def export_trainable_state(self):
            return SimpleNamespace(
                policy_version=0,
                layout_hash=current.layout_hash,
                tensors=_state(0, 1.0).tensors,
            )

    actor = Actor()

    class Bridge:
        finalizing = False

        def __init__(self):
            self.acknowledged = None

        def consolidate_budget(self, frozen, **progress):
            assert frozen.policy_version == 1
            assert progress == {"optimizer_steps": 1, "action_tokens": 5}
            return BudgetConsolidation(
                FinalManifest(2, (1, 2)),
                _state(1, 3.0),
                (FragmentSubmission(0, 1, 1, 0, 1, 5, 0.5, 8, 0.1),),
                16,
            )

        def acknowledge_finalization(self, manifest):
            self.acknowledged = manifest

    bridge = Bridge()
    hook = DecoupledMilesPolicySync(args)
    hook.actor_model = actor
    hook.rollout_manager = object()
    hook.bridge = bridge
    hook.current = current
    hook.snapshot = snapshot
    hook._round_stats = lambda *_args, **_kwargs: _stats(0)
    resets = []

    async def apply(state, *, reset_optimizer):
        resets.append(reset_optimizer)
        return state

    async def publish(value):
        args.yeto_rl_policy_token = value.token

    hook._apply_decoupled_policy = apply
    hook._publish_snapshot = publish

    should_stop = asyncio.run(
        hook._after_local_train(
            rollout_id=0,
            actor_model=actor,
            rollout_data=object(),
        )
    )

    assert should_stop
    assert resets == [False]
    assert bridge.acknowledged == FinalManifest(2, (1, 2))
    assert hook.finished
    payload = miles._load_decoupled_checkpoint(args)
    assert payload["next_rollout_id"] == 1
    assert payload["fragment_versions"] == [1, 2]
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert [event["event"] for event in events].count("rl_local_round") == 1
    local_round = next(event for event in events if event["event"] == "rl_local_round")
    assert local_round["sync/fragment_payload_bytes_sent"] == 8
    assert local_round["sync/fragment_payload_bytes_received"] == 16
    assert [event["event"] for event in events].count("rl_fragment_push") == 1
