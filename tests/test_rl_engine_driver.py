"""CPU tests for the ports-path IslandDriver (tasks 4.1-4.4) on the fake engine."""

from __future__ import annotations

import json
import sys
import threading
from types import SimpleNamespace

import pytest
import torch

from yeto.rl import miles
from yeto.rl.bridge import BridgeConfig
from yeto.rl.core import (
    LocalRoundStats,
    PolicySnapshot,
    StrictRlInvariantError,
    build_avg_layout,
    build_rl_fragment_layout,
    canonical_state,
)
from yeto.protocol import FinalManifest
from yeto.rl.decoupled import (
    BroadcastBatch,
    BudgetConsolidation,
    DecoupledBridgeConfig,
    FragmentSubmission,
    InitialCut,
)
from yeto.rl.engine.algorithm import AlgorithmSpec
from yeto.rl.engine.bridges import (
    DecoupledIslandProgress,
    DecoupledSync,
    LocalOnlySync,
    StrictAvgSync,
    StrictIslandProgress,
)
from yeto.rl.engine.capabilities import CapabilityMismatch
from yeto.rl.engine.driver import (
    DriverError,
    EventTape,
    IslandDriver,
    PolicyIdentityError,
    PublicationError,
    RoundFailedError,
)
from yeto.rl.engine.fake import (
    FAKE_PROFILES,
    LORA_CONFIG_HASH,
    MODEL_REVISION,
    FakeDecoupledSyncer,
    FakeEngine,
    FakeStrictSyncer,
    fake_capabilities,
)

NAME = "base_model.model.layer.lora_A.weight"


def _events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


# decoupling 2.8: every driver test runs against both fake capability profiles
# (what the Miles adapter declares, and the bare five ports).
PROFILE = "miles-like"


@pytest.fixture(autouse=True, params=FAKE_PROFILES)
def capability_profile(request, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "PROFILE", request.param)
    return request.param


def _engine(delta=1.0, **kwargs):
    kwargs.setdefault("capability_profile", PROFILE)
    return FakeEngine(tensors={NAME: torch.zeros(1, 2)}, step_delta=delta, **kwargs)


def _driver(engine, sync, tmp_path, *, name="events.jsonl", **kwargs):
    kwargs.setdefault("capabilities", fake_capabilities(PROFILE))
    return IslandDriver(
        learner_id=kwargs.pop("learner_id", 0),
        rollout=engine.rollout,
        trainer=engine.trainer,
        policy_state=engine.policy_state,
        publisher=engine.publisher,
        placement=engine.placement,
        algorithm=AlgorithmSpec(),
        sync=sync,
        events=EventTape(tmp_path / name, kwargs.get("learner_id", 0)),
        **kwargs,
    )


def _engine_calls(engine):
    return [c for c in engine.calls if c[0] != "export"]


# ---------------------------------------------------------------------------
# 4.1 colocated-serial loop
# ---------------------------------------------------------------------------
def test_serial_colocated_three_rounds_event_order(tmp_path):
    engine = _engine()
    evals = []
    driver = _driver(
        engine,
        LocalOnlySync(3),
        tmp_path,
        evaluate=lambda r: evals.append(r) or {"pass_rate": 0.5},
        eval_interval=2,
    )
    final = driver.run()

    expected = [("publish", 0)]
    for r in range(3):
        expected += [("offload",), ("generate", r), ("onload",), ("train", r), ("publish", r + 1)]
    assert _engine_calls(engine) == expected
    assert final.policy_version == 3
    assert torch.equal(final.tensors[NAME], torch.full((1, 2), 3.0))
    assert evals == [0, 2, 3]
    events = _events(tmp_path / "events.jsonl")
    start = next(e for e in events if e["event"] == "rl_driver_start")
    assert start["rl/algorithm_spec_sha256"] == AlgorithmSpec().sha256()
    phases = [e["phase"] for e in events if e["event"] == "rl_driver_phase"]
    assert phases[:8] == [
        "publish", "eval", "offload", "generate", "onload", "train", "sync", "publish",
    ]
    assert [e["event"] for e in events].count("rl_local_round") == 3


def test_fixed_partition_never_offloads(tmp_path):
    engine = _engine(placement_kind="fixed-partition")
    _driver(engine, LocalOnlySync(2), tmp_path).run()
    assert not any(c[0] in {"offload", "onload"} for c in engine.calls)


def test_zero_grad_fails_the_round_without_commit(tmp_path):
    engine = _engine(zero_grad_rounds={1})
    with pytest.raises(StrictRlInvariantError, match="grad_norm 0") as info:
        _driver(engine, LocalOnlySync(3), tmp_path).run()
    assert info.value.metric == "zero_grad_norm_with_nonzero_advantages"
    publishes = [c for c in engine.calls if c[0] == "publish"]
    assert publishes == [("publish", 0), ("publish", 1)]
    events = _events(tmp_path / "events.jsonl")
    failure = [e for e in events if e["event"] == "rl_strict_failure"]
    assert failure and failure[0]["metric"] == "zero_grad_norm_with_nonzero_advantages"
    assert [e["event"] for e in events].count("rl_local_round") == 1


def test_zero_grad_with_all_zero_advantages_is_not_a_failure(tmp_path):
    engine = _engine(zero_grad_rounds={0}, constant_reward_rounds={0})
    final = _driver(engine, LocalOnlySync(2), tmp_path).run()
    assert final.policy_version == 2


def test_failed_optimizer_step_fails_the_round(tmp_path):
    engine = _engine(failed_step_rounds={0})
    with pytest.raises(RoundFailedError):
        _driver(engine, LocalOnlySync(2), tmp_path).run()
    assert [c for c in engine.calls if c[0] == "publish"] == [("publish", 0)]


def test_partial_publication_blocks_next_generation(tmp_path):
    engine = _engine(unacked_member_rounds={1: "rollout-1"})
    with pytest.raises(PublicationError, match="rollout-1"):
        _driver(engine, LocalOnlySync(3), tmp_path).run()
    assert [c for c in engine.calls if c[0] == "generate"] == [("generate", 0)]


def test_mismatched_group_token_rejected_before_training(tmp_path):
    engine = _engine(stale_token_rounds={1})
    with pytest.raises(PolicyIdentityError, match="r1-g0"):
        _driver(engine, LocalOnlySync(3), tmp_path).run()
    assert ("train", 1) not in engine.calls


def test_capability_mismatch_rejects_before_any_engine_call(tmp_path):
    engine = _engine()
    caps = fake_capabilities(placements={"fixed-partition"})
    with pytest.raises(CapabilityMismatch, match="placement"):
        _driver(engine, LocalOnlySync(1), tmp_path, capabilities=caps).run()
    assert engine.calls == []


def test_trainer_without_grad_norm_is_rejected(tmp_path):
    engine = _engine(grad_norm_reported=False)
    with pytest.raises(DriverError, match="grad_norm"):
        _driver(engine, LocalOnlySync(1), tmp_path).run()
    assert engine.calls == []


# ---------------------------------------------------------------------------
# 4.2 strict-avg through PolicyState / Publisher
# ---------------------------------------------------------------------------
def _strict_config(tmp_path, engine, *, learner_id, rounds, tape="bridge.jsonl"):
    initial = engine.canonical(0)
    return BridgeConfig(
        syncer_addr=("127.0.0.1", 1),
        learner_id=learner_id,
        global_rounds=rounds,
        groups_per_round=engine.groups,
        samples_per_group=engine.samples_per_group,
        local_optimizer_steps=1,
        wan_streams=0,
        expected_specs=initial.specs,
        base_model_revision=MODEL_REVISION,
        lora_config_hash=LORA_CONFIG_HASH,
        layout_hash=initial.layout_hash,
        event_tape=str(tmp_path / tape),
    )


def _strict_syncer(engine, *, learners, rounds):
    return FakeStrictSyncer(
        build_avg_layout(engine.canonical(0).specs), learners=learners, total_steps=rounds
    )


def _strict_driver(tmp_path, engine, syncer, *, learner_id, rounds, progress=None):
    sync = StrictAvgSync(
        _strict_config(tmp_path, engine, learner_id=learner_id, rounds=rounds,
                       tape=f"island-{learner_id}.jsonl"),
        progress=progress,
        client_factory=lambda _bridge: syncer.client(learner_id),
    )
    return _driver(
        engine, sync, tmp_path, name=f"island-{learner_id}.jsonl",
        learner_id=learner_id, progress=progress,
    )


def _run_threads(drivers):
    results, errors = {}, {}

    def run(i, d):
        try:
            results[i] = d.run()
        except BaseException as error:  # noqa: BLE001
            errors[i] = error

    threads = [threading.Thread(target=run, args=(i, d), daemon=True) for i, d in enumerate(drivers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
        assert not t.is_alive()
    return results, errors


def test_two_islands_strict_avg_on_ports_path(tmp_path):
    # Mirrors test_two_fake_miles_islands_run_the_bridge_against_real_syncer.
    engines = (_engine(torch.tensor([1.0, 3.0])), _engine(torch.tensor([3.0, 5.0])))
    syncer = _strict_syncer(engines[0], learners=2, rounds=1)
    drivers = [
        _strict_driver(tmp_path, e, syncer, learner_id=i, rounds=1)
        for i, e in enumerate(engines)
    ]
    results, errors = _run_threads(drivers)
    assert errors == {}
    for i, engine in enumerate(engines):
        assert torch.equal(results[i].tensors[NAME], torch.tensor([[2.0, 4.0]]))
        assert engine.policy_state.applied[-1] == (1, "reset", 1)
        assert engine.published[0] == 1


def test_two_islands_strict_avg_three_rounds_same_hash_and_spec_order(tmp_path):
    engines = (_engine(torch.tensor([1.0, 3.0])), _engine(torch.tensor([3.0, 5.0])))
    syncer = _strict_syncer(engines[0], learners=2, rounds=3)
    drivers = [
        _strict_driver(tmp_path, e, syncer, learner_id=i, rounds=3)
        for i, e in enumerate(engines)
    ]
    results, errors = _run_threads(drivers)
    assert errors == {}
    assert syncer.history == [0, 1, 2, 3]
    assert results[0].policy_tensor_hash() == results[1].policy_tensor_hash()
    assert torch.equal(results[0].tensors[NAME], torch.tensor([[6.0, 12.0]]))
    for engine in engines:
        assert engine.moments[NAME].abs().sum() == 0  # every strict apply resets
    # generate(v) -> train -> export+push -> wait v+1 -> reset-apply -> publish v+1
    events = _events(tmp_path / "island-0.jsonl")
    phases = [
        (e["phase"], e.get("optimizer"))
        for e in events
        if e["event"] == "rl_driver_phase"
        and e["phase"] in {"generate", "train", "export_push", "wait_global", "apply", "publish"}
    ]
    assert phases[:3] == [("apply", "reset"), ("publish", None), ("generate", None)]
    assert phases[3:9] == [
        ("train", None), ("export_push", None), ("wait_global", None),
        ("apply", "reset"), ("publish", None), ("generate", None),
    ]


def _strict_args(tmp_path, engine):
    initial = engine.canonical(0)
    return SimpleNamespace(
        actor_num_gpus_per_node=1, actor_num_nodes=1, advantage_estimator="grpo",
        yeto_rl_model="org/model", yeto_rl_data="org/data",
        yeto_rl_base_model_revision=MODEL_REVISION, yeto_rl_data_revision="d" * 40,
        expert_model_parallel_size=1, yeto_rl_layout_hash=initial.layout_hash, lr=1e-4,
        yeto_rl_lora_config_hash=LORA_CONFIG_HASH, n_samples_per_prompt=2,
        num_steps_per_rollout=1, pipeline_model_parallel_size=1,
        over_sampling_batch_size=1, rollout_batch_size=2, seq_length=128, seed=7,
        rollout_max_response_len=16, custom_generate_function_path=None,
        use_session_server=False, tito_model=None, yeto_rl_reward_sha256="e" * 64,
        yeto_rl_completed_groups_path=str(tmp_path / "island.pt"),
        yeto_rl_event_tape=str(tmp_path / "island-0.jsonl"), yeto_rl_learner_id=0,
    )


def test_strict_progress_replaces_a_killed_learners_record_for_the_same_round(tmp_path):
    # A learner killed mid-round leaves that round's record; the restart
    # regenerates the round and must record the new batch's metrics (legacy
    # recovery rewrites the record too). A record carrying a completed-group
    # queue was written by the rollout process and is kept.
    engine = _engine(torch.tensor([1.0, 3.0]))
    args = _strict_args(tmp_path, engine)
    progress = StrictIslandProgress(args)
    progress.after_generate(rollout_id=1, policy_token="yeto:1:a", metrics={"reward": 0.25})
    progress.after_generate(rollout_id=1, policy_token="yeto:1:a", metrics={"reward": 0.75})
    payload = torch.load(tmp_path / "island.pt", weights_only=True)
    assert payload["rollout_metrics"] == {"reward": 0.75}

    payload["completed_groups"] = [{"group": 0}]
    torch.save(payload, tmp_path / "island.pt")
    progress.after_generate(rollout_id=1, policy_token="yeto:1:a", metrics={"reward": 0.5})
    kept = torch.load(tmp_path / "island.pt", weights_only=True)
    assert kept["rollout_metrics"] == {"reward": 0.75}
    assert kept["completed_groups"] == [{"group": 0}]


def test_single_island_strict_progress_and_publication_tokens(tmp_path):
    # Mirrors test_miles_public_hook_runs_against_real_syncer.
    engine = _engine(torch.tensor([1.0, 3.0]))
    args = _strict_args(tmp_path, engine)
    syncer = _strict_syncer(engine, learners=1, rounds=1)
    driver = _strict_driver(
        tmp_path, engine, syncer, learner_id=0, rounds=1, progress=StrictIslandProgress(args)
    )
    final = driver.run()
    assert torch.equal(final.tensors[NAME], torch.tensor([[1.0, 3.0]]))
    assert [c for c in engine.calls if c[0] == "publish"] == [("publish", 0), ("publish", 1)]
    payload = torch.load(tmp_path / "island.pt", weights_only=True)
    assert payload["schema_version"] == 3
    assert payload["config"] == miles._island_checkpoint_config(args)
    assert payload["local_round_id"] == 1
    assert payload["local_round_stats"]["base_policy_version"] == 0
    assert "tensors" not in payload and "optimizer" not in payload
    events = _events(tmp_path / "island-0.jsonl")
    tokens = [e["rl/policy_token"] for e in events if e["event"] == "rl_publication"]
    assert [t.split(":")[1] for t in tokens] == ["0", "1"]
    local = next(e for e in events if e["event"] == "rl_local_round")
    assert local["grad_norm"] > 0 and local["rl/completed_groups"] == 2


def test_strict_zero_grad_pushes_nothing_and_restart_applies_cut_with_reset(tmp_path):
    engine = _engine(torch.tensor([1.0, 3.0]), zero_grad_rounds={1})
    args = _strict_args(tmp_path, engine)
    syncer = _strict_syncer(engine, learners=1, rounds=3)
    driver = _strict_driver(
        tmp_path, engine, syncer, learner_id=0, rounds=3, progress=StrictIslandProgress(args)
    )
    with pytest.raises(StrictRlInvariantError):
        driver.run()
    assert syncer.history == [0, 1] and syncer.error is None
    payload = torch.load(tmp_path / "island.pt", weights_only=True)
    assert payload["policy_version"] == 1 and payload["local_round_stats"] is None

    # Restart: fresh process, no local LoRA/optimizer; syncer cut v1 is authoritative.
    restarted = _engine(torch.tensor([1.0, 3.0]))
    restarted.moments[NAME].fill_(7.0)
    driver = _strict_driver(
        tmp_path, restarted, syncer, learner_id=0, rounds=3, progress=StrictIslandProgress(args)
    )
    final = driver.run()
    assert restarted.policy_state.applied[0] == (1, "reset", 1)
    assert [c for c in restarted.calls if c[0] == "generate"] == [("generate", 1), ("generate", 2)]
    assert torch.equal(final.tensors[NAME], torch.tensor([[3.0, 9.0]]))
    assert syncer.history == [0, 1, 2, 3]
    assert torch.load(tmp_path / "island.pt", weights_only=True)["local_round_id"] == 3


# ---------------------------------------------------------------------------
# 4.3 / 4.4 decoupled through PolicyState / Publisher
# ---------------------------------------------------------------------------
def _tensors(offset=0.0):
    return {
        "base_model.model.a.lora_A.weight": torch.tensor([[1.0, 2.0]]) + offset,
        "base_model.model.b.lora_A.weight": torch.tensor([[3.0, 4.0]]) + offset,
        "base_model.model.c.lora_A.weight": torch.tensor([[5.0]]) + offset,
        "base_model.model.d.lora_A.weight": torch.tensor([[6.0]]) + offset,
    }


def _state(version, offset=0.0):
    return canonical_state(
        version, _tensors(offset), base_model_revision=MODEL_REVISION,
        lora_config_hash=LORA_CONFIG_HASH,
    )


def _dconfig(initial, *, total=4, horizon=2, budget=None):
    return DecoupledBridgeConfig(
        syncer_addr=("127.0.0.1", 1), learner_id=0, total_fragment_steps=total,
        num_fragments=2, pipeline=2, local_horizon=horizon,
        expected_specs=initial.specs, base_model_revision=MODEL_REVISION,
        lora_config_hash=LORA_CONFIG_HASH, canonical_layout_hash=initial.layout_hash,
        wan_streams=0, learner_budget_steps=budget,
    )


def _dargs(tmp_path):
    initial = _state(0)
    return SimpleNamespace(
        actor_num_gpus_per_node=1, actor_num_nodes=1, advantage_estimator="grpo",
        yeto_rl_model="org/model", yeto_rl_data="org/data",
        yeto_rl_base_model_revision=MODEL_REVISION, yeto_rl_data_revision="c" * 40,
        expert_model_parallel_size=1, yeto_rl_layout_hash=initial.layout_hash, lr=1e-5,
        yeto_rl_lora_config_hash=LORA_CONFIG_HASH, n_samples_per_prompt=2,
        num_steps_per_rollout=1, pipeline_model_parallel_size=1,
        over_sampling_batch_size=2, yeto_rl_reward_sha256="d" * 64,
        yeto_rl_source_sha256="f" * 64, rollout_batch_size=1, seq_length=128, seed=7,
        rollout_max_response_len=32, custom_generate_function_path=None,
        use_session_server=False, tito_model=None, yeto_rl_learner_id=0,
        yeto_rl_sync_preset="decoupled", yeto_rl_num_fragments=2, yeto_rl_pipeline=2,
        yeto_rl_local_horizon=2, yeto_rl_total_sweeps=2, yeto_rl_total_fragment_steps=4,
        yeto_rl_sync_layout_fingerprint="e" * 64, yeto_rl_bridge_config=_dconfig(initial),
        yeto_rl_completed_groups_path=str(tmp_path / "island.pt"),
        yeto_rl_event_tape=str(tmp_path / "events.jsonl"),
    )


def _dengine(offset=0.0, **kwargs):
    return FakeEngine(tensors=_tensors(offset), step_delta=0.25, **kwargs)


def _save(args, snapshot, steps, tokens, groups=()):
    miles._save_decoupled_checkpoint(
        args, snapshot=snapshot, optimizer_steps=steps, action_tokens=tokens,
        rollout_metrics={}, local_round_stats=None, completed_groups=list(groups),
    )


def _stats(rollout_id=0):
    return LocalRoundStats(
        island_id=0, local_round_id=rollout_id + 1, base_policy_version=rollout_id,
        active_groups=1, completed_groups=1, cancelled_groups=0,
        completed_trajectories=2, action_tokens=5, tool_wait_seconds=0.0,
        group_p50_seconds=0.1, group_p95_seconds=0.1, group_p99_seconds=0.1,
        reward_mean=1.0, reward_std=0.0, zero_variance_group_ratio=1.0, mean_kl=0.0,
        ess_ratio=1.0, clip_fraction=0.0, delta_l2_norm=0.0, rollout_seconds=0.1,
        train_seconds=0.1, train_step=rollout_id, loss=-0.4, pg_loss=-0.5,
        grad_norm=1.25, lr=1e-6, pass_rate=0.5,
    )


def _armed(tmp_path, args, engine, bridge, current, snapshot):
    sync = DecoupledSync(args)
    driver = _driver(engine, sync, tmp_path)
    sync._driver = driver
    sync.bridge = bridge
    sync.current = current
    sync.snapshot = snapshot
    args.yeto_rl_policy_token = snapshot.token
    return sync, driver


def test_decoupled_boundary_preserves_optimizer_and_publishes_one_snapshot(tmp_path):
    # Mirrors test_decoupled_hook_preserves_optimizer_and_publishes_one_full_snapshot.
    args = _dargs(tmp_path)
    current = _state(0)
    snapshot = PolicySnapshot.create(0, current, (0, 0))
    _save(args, snapshot, 0, 0)
    engine = _dengine(1.0)
    calls = []

    class Bridge:
        fragment_versions = (1, 0)
        finalizing = False

        def drain_broadcasts(self, local, **progress):
            calls.append("broadcast")
            assert local.policy_version == 1
            assert progress == {"optimizer_steps": 1, "action_tokens": 5}
            return BroadcastBatch(
                _state(1, 2.0),
                (SimpleNamespace(fragment_id=0, version=1, anchor=torch.zeros(3),
                                 payload_bytes=8, queue_seconds=0.25),),
            )

        def commit_broadcasts(self, batch, **progress):
            calls.append("commit")
            assert engine.policy_state.applied[-1] == (1, "preserve", 1)

        def submit_ready(self, local, **progress):
            calls.append("pull")
            return (FragmentSubmission(1, 2, 1, 0, 1, 5, 0.5, 8, 0.1),)

    sync, driver = _armed(tmp_path, args, engine, Bridge(), current, snapshot)
    engine.moments = {k: torch.ones_like(v) for k, v in engine.tensors.items()}
    result = sync.boundary(driver, rollout_id=0, stats=_stats(0))
    driver.publish(result.state, rollout_id=1)

    assert not result.stop
    assert calls == ["broadcast", "commit", "pull"]
    assert engine.policy_state.applied == [(1, "preserve", 1)]
    assert all(m.sum() > 0 for m in engine.moments.values())  # moments preserved
    assert engine.published[0] == 1
    assert args.yeto_rl_policy_token == sync.snapshot.token
    assert torch.equal(engine.tensors["base_model.model.a.lora_A.weight"], torch.tensor([[3.0, 4.0]]))
    payload = miles._load_decoupled_checkpoint(args)
    assert payload["next_rollout_id"] == 1 and payload["fragment_versions"] == [1, 0]
    events = _events(tmp_path / "events.jsonl")
    assert [e["event"] for e in events].count("rl_sync_hook") == 1
    local = next(e for e in events if e["event"] == "rl_local_round")
    assert local["train/grad_norm"] == 1.25
    assert local["sync/fragment_payload_bytes_received"] == 8
    assert local["sync/fragment_payload_bytes_sent"] == 8
    snap = [e for e in events if e["event"] == "rl_policy_snapshot"][-1]
    pub = [e for e in events if e["event"] == "rl_publication"][-1]
    assert snap["rl/policy_token"] == pub["rl/policy_token"]


def test_decoupled_pending_finalization_records_work_and_stops(tmp_path):
    args = _dargs(tmp_path)
    current = _state(0)
    snapshot = PolicySnapshot.create(0, current, (0, 0))
    _save(args, snapshot, 0, 0)

    class Bridge:
        finalizing = True
        final_payload_bytes_received = 16
        acknowledged = None

        def wait_for_final_cut(self, *, policy_version):
            assert policy_version == 1
            return FinalManifest(4, (3, 4)), _state(1, 3.0)

        def acknowledge_finalization(self, manifest):
            Bridge.acknowledged = manifest

    engine = _dengine(1.0)
    sync, driver = _armed(tmp_path, args, engine, Bridge(), current, snapshot)
    result = sync.boundary(driver, rollout_id=0, stats=_stats(0))
    assert result.stop and sync.finished
    assert engine.policy_state.applied == [(1, "preserve", 1)]
    assert Bridge.acknowledged == FinalManifest(4, (3, 4))
    assert miles._load_decoupled_checkpoint(args)["fragment_versions"] == [3, 4]
    events = _events(tmp_path / "events.jsonl")
    assert [e["event"] for e in events].count("rl_local_round") == 1
    assert next(e for e in events if e["event"] == "rl_final_cut")[
        "sync/fragment_payload_bytes_received"
    ] == 16


def test_decoupled_waits_after_submitting_last_fragment_step(tmp_path):
    args = _dargs(tmp_path)
    args.yeto_rl_total_fragment_steps = 2
    current = _state(0)
    snapshot = PolicySnapshot.create(0, current, (0, 0))

    class Bridge:
        finalizing = False
        fragment_versions = (0, 0)

        def drain_broadcasts(self, local, **_):
            return BroadcastBatch(local, ())

        def submit_ready(self, _local, **_):
            return (FragmentSubmission(1, 2, 1, 0, 2, 5, 0.5, 8, 0.1),)

    sync, driver = _armed(tmp_path, args, _dengine(), Bridge(), current, snapshot)
    finalization = []
    sync._finish = lambda *, policy_version, stats: finalization.append(
        (policy_version, stats)
    ) or SimpleNamespace(stop=True)
    assert sync.boundary(driver, rollout_id=0, stats=_stats(0)).stop
    assert finalization[0][0] == 1 and finalization[0][1].delta_l2_norm == 0.5


def test_decoupled_budget_freezes_and_stops_after_final_cut(tmp_path):
    args = _dargs(tmp_path)
    args.yeto_rl_learner_budget_steps = 1
    current = _state(0)
    snapshot = PolicySnapshot.create(0, current, (0, 0))
    _save(args, snapshot, 0, 0)

    class Bridge:
        finalizing = False
        acknowledged = None

        def consolidate_budget(self, frozen, **progress):
            assert frozen.policy_version == 1
            assert progress == {"optimizer_steps": 1, "action_tokens": 5}
            return BudgetConsolidation(
                FinalManifest(2, (1, 2)), _state(1, 3.0),
                (FragmentSubmission(0, 1, 1, 0, 1, 5, 0.5, 8, 0.1),), 16,
            )

        def acknowledge_finalization(self, manifest):
            Bridge.acknowledged = manifest

    engine = _dengine(1.0)
    sync, driver = _armed(tmp_path, args, engine, Bridge(), current, snapshot)
    assert sync.boundary(driver, rollout_id=0, stats=_stats(0)).stop
    assert engine.policy_state.applied == [(1, "preserve", 1)]
    assert Bridge.acknowledged == FinalManifest(2, (1, 2))
    payload = miles._load_decoupled_checkpoint(args)
    assert payload["next_rollout_id"] == 1 and payload["fragment_versions"] == [1, 2]
    events = _events(tmp_path / "events.jsonl")
    assert [e["event"] for e in events].count("rl_fragment_push") == 1


def _recovery_bridge(cut, versions, calls, *, manifest=None):
    class Bridge:
        fragment_versions = versions
        startup_final_manifest = manifest
        final_payload_bytes_received = 16
        acknowledged = None

        def __init__(self, *_args):
            pass

        def start(self):
            pass

        def wait_for_initial_cut(self, **progress):
            calls.append(("cut", progress))
            return InitialCut(cut, versions)

        def commit_initial_cut(self, cut, **progress):
            calls.append(("commit", progress))

        def acknowledge_finalization(self, value):
            Bridge.acknowledged = value

        def close(self):
            pass

    return Bridge


def test_decoupled_restart_applies_newer_uneven_cut_with_reset(tmp_path):
    # Mirrors test_decoupled_recovery_applies_newer_uneven_cut_at_saved_local_progress.
    args = _dargs(tmp_path)
    saved = PolicySnapshot.create(2, _state(2), (1, 0))
    _save(args, saved, 2, 10, groups=[[{"weight_versions": [saved.token]}]])
    calls = []
    engine = _dengine()
    sync = DecoupledSync(
        args, bridge_factory=_recovery_bridge(_state(0, 4.0), (3, 2), calls)
    )
    driver = _driver(engine, sync, tmp_path)
    engine.calls.clear()
    start = sync.start(driver)
    assert calls[0] == ("cut", {"optimizer_steps": 2, "action_tokens": 10})
    assert calls[1][0] == "commit"
    assert engine.policy_state.applied == [(2, "reset", 2)]
    assert all(m.sum() == 0 for m in engine.moments.values())
    assert engine.scheduler_step == 2
    assert start.rollout_id == 2 and not start.finished
    assert sync.snapshot.fragment_versions == (3, 2)
    assert miles._load_decoupled_checkpoint(args)["completed_groups"] == []


def test_decoupled_restart_keeps_groups_for_exact_rebuilt_snapshot(tmp_path):
    args = _dargs(tmp_path)
    recovered = _state(2, 4.0)
    snapshot = PolicySnapshot.create(2, recovered, (1, 0))
    group = [{"status": "completed", "weight_versions": [snapshot.token], "index": i} for i in range(2)]
    _save(args, snapshot, 2, 10, groups=[group])
    sync = DecoupledSync(args, bridge_factory=_recovery_bridge(recovered, (1, 0), []))
    sync.start(_driver(_dengine(), sync, tmp_path))
    restored = miles._load_decoupled_checkpoint(args)
    assert restored["policy_token"] == snapshot.token
    assert len(restored["completed_groups"]) == 1


def test_decoupled_restart_at_terminal_cut_publishes_once_and_stops(tmp_path):
    args = _dargs(tmp_path)
    _save(args, PolicySnapshot.create(2, _state(2), (1, 0)), 2, 10)
    manifest = FinalManifest(4, (3, 4))
    bridge = _recovery_bridge(_state(2, 4.0), manifest.versions, [], manifest=manifest)
    engine = _dengine()
    sync = DecoupledSync(args, bridge_factory=bridge)
    _driver(engine, sync, tmp_path).run()
    assert bridge.acknowledged == manifest
    assert [c for c in engine.calls if c[0] in {"publish", "generate", "train"}] == [("publish", 2)]
    assert miles._load_decoupled_checkpoint(args)["fragment_versions"] == [3, 4]


def test_decoupled_run_until_stop_against_fake_syncer(tmp_path):
    args = _dargs(tmp_path)
    engine = _dengine()
    initial = engine.canonical(0)
    syncer = FakeDecoupledSyncer(
        build_rl_fragment_layout(initial.specs, 2), initial.tensors,
        learners=1, total_steps=4, pipeline=2,
    )
    sync = DecoupledSync(args, client_factory=lambda _bridge: syncer.client(0))
    driver = _driver(
        engine, sync, tmp_path, progress=DecoupledIslandProgress(args), max_rollouts=20
    )
    final = driver.run()

    assert syncer.error is None and syncer.final_manifest == FinalManifest(4, (3, 4))
    assert sync.bridge.client.acked == syncer.final_manifest
    rounds = [c[1] for c in engine.calls if c[0] == "generate"]
    publishes = [c[1] for c in engine.calls if c[0] == "publish"]
    assert rounds == [0, 1, 2, 3, 4]
    assert publishes == [0, 1, 2, 3, 4, 5]  # final cut published exactly once
    assert engine.calls[-1] == ("publish", 5)
    for name, value in syncer.params.items():
        assert torch.equal(final.tensors[name], value)
        assert torch.equal(engine.tensors[name], value)
    modes = [a[1] for a in engine.policy_state.applied]
    assert modes[0] == "reset" and set(modes[1:]) == {"preserve"}
    payload = miles._load_decoupled_checkpoint(args)
    assert payload["next_rollout_id"] == 5 and payload["fragment_versions"] == [3, 4]
    assert "tensors" not in payload
    phases = [
        e["phase"] for e in _events(tmp_path / "events.jsonl") if e["event"] == "rl_driver_phase"
    ]
    assert "drain_bcast" in phases and "drain_pull" in phases


# ---------------------------------------------------------------------------
# fix-decoupled-lr-schedule 2.1: zero learning rate before the final round
# ---------------------------------------------------------------------------
def test_zero_lr_non_final_round_fails_without_commit(tmp_path):
    engine = _engine(zero_lr_rounds={1})
    with pytest.raises(StrictRlInvariantError, match="local round 2") as info:
        _driver(engine, LocalOnlySync(3), tmp_path).run()
    assert info.value.metric == "zero_lr_before_final_round"
    assert "learning rate 0.0" in str(info.value)
    assert [c for c in engine.calls if c[0] == "publish"] == [("publish", 0), ("publish", 1)]
    events = _events(tmp_path / "events.jsonl")
    assert [e["metric"] for e in events if e["event"] == "rl_strict_failure"] == [
        "zero_lr_before_final_round"
    ]
    rounds = [e for e in events if e["event"] == "rl_local_round"]
    assert len(rounds) == 1  # round 2 was never handed to the sync session
    assert rounds[0]["applied_lr"] == 1e-5 and rounds[0]["applied_lrs"] == [1e-5]


def test_zero_lr_on_the_last_local_round_is_not_a_failure(tmp_path):
    engine = _engine(zero_lr_rounds={2})
    final = _driver(engine, LocalOnlySync(3), tmp_path).run()
    assert final.policy_version == 3
    rounds = [e for e in _events(tmp_path / "events.jsonl") if e["event"] == "rl_local_round"]
    assert [r["applied_lr"] for r in rounds] == [1e-5, 1e-5, 0.0]


def test_strict_zero_lr_before_last_round_pushes_nothing(tmp_path):
    engine = _engine(torch.tensor([1.0, 3.0]), zero_lr_rounds={1})
    syncer = _strict_syncer(engine, learners=1, rounds=3)
    driver = _strict_driver(tmp_path, engine, syncer, learner_id=0, rounds=3)
    with pytest.raises(StrictRlInvariantError, match="local round 2") as info:
        driver.run()
    assert info.value.metric == "zero_lr_before_final_round"
    assert syncer.history == [0, 1]  # round 2 never reached the syncer


def test_strict_last_round_with_zero_lr_is_not_a_failure(tmp_path):
    # Strict final round: local_round_id == global_rounds.
    engine = _engine(torch.tensor([1.0, 3.0]), zero_lr_rounds={2})
    syncer = _strict_syncer(engine, learners=1, rounds=3)
    driver = _strict_driver(tmp_path, engine, syncer, learner_id=0, rounds=3)
    assert driver.run().policy_version == 3
    assert syncer.history == [0, 1, 2, 3]


def test_decoupled_zero_lr_before_final_cut_fails(tmp_path):
    args = _dargs(tmp_path)
    engine = _dengine(zero_lr_rounds={2})
    initial = engine.canonical(0)
    syncer = FakeDecoupledSyncer(
        build_rl_fragment_layout(initial.specs, 2), initial.tensors,
        learners=1, total_steps=4, pipeline=2,
    )
    sync = DecoupledSync(args, client_factory=lambda _bridge: syncer.client(0))
    driver = _driver(
        engine, sync, tmp_path, progress=DecoupledIslandProgress(args), max_rollouts=20
    )
    with pytest.raises(StrictRlInvariantError, match="local round 3") as info:
        driver.run()
    assert info.value.metric == "zero_lr_before_final_round"
    assert syncer.final_manifest is None


def test_decoupled_final_round_is_defined_by_the_final_cut(tmp_path):
    args = _dargs(tmp_path)
    sync = DecoupledSync(args)
    sync.bridge = SimpleNamespace(finalizing=False)
    assert not sync.is_final_round(None, rollout_id=0)
    assert not sync.is_final_round(None, rollout_id=50)  # never by round count
    sync.bridge = SimpleNamespace(finalizing=True)  # final cut announced
    assert sync.is_final_round(None, rollout_id=0)
    sync.bridge = SimpleNamespace(finalizing=False)
    args.yeto_rl_learner_budget_steps = 3
    sync.optimizer_steps = 1
    assert not sync.is_final_round(None, rollout_id=1)
    sync.optimizer_steps = 2
    assert sync.is_final_round(None, rollout_id=2)  # this round exhausts the budget


def test_decoupled_zero_lr_after_the_final_cut_is_announced_is_not_a_failure(tmp_path):
    engine = _engine(zero_lr_rounds={1})

    class Finalizing(LocalOnlySync):
        # decoupled semantics: rounds after the syncer announced the final cut
        def is_final_round(self, driver, *, rollout_id):
            return rollout_id >= 1

    final = _driver(engine, Finalizing(2), tmp_path).run()
    assert final.policy_version == 2


# -- failure-path node-loss attribution (rl-multinode-island D9 / tasks 3.3) ------

class _NodeController:
    """Minimal multi-node controller: ``check_nodes`` returns the loss on the
    ``lose_at``-th poll (None before), never returns via safe_point/finalization."""

    admission_open = True
    recovery_required = None

    def __init__(self, topology=(2, 1), lose_at=None, lost="node_lost: n1 (1 of 2 alive)"):
        self.topology = topology
        self.lose_at = lose_at
        self.lost = lost
        self.polls = 0

    def has_pending(self):
        return False

    def check_nodes(self):
        self.polls += 1
        if self.lose_at is not None and self.polls >= self.lose_at:
            return self.lost
        return None


def _node_driver(engine, sync, tmp_path, ctl, *, grace=10.0):
    now = [0.0]
    slept = []

    def clock():
        return now[0]

    def sleep(s):
        slept.append(s)
        now[0] += s

    driver = _driver(engine, sync, tmp_path, controller=ctl, clock=clock)
    driver.sleep = sleep
    driver.node_loss_grace_s = grace
    return driver, slept


def test_failed_publication_is_attributed_to_node_loss_within_grace(tmp_path):
    engine = _engine(unacked_member_rounds={1: "rollout-1"})  # publish of round 1 fails
    ctl = _NodeController(lose_at=4)  # 1 probe before round 0 + 3 failure-path polls
    driver, slept = _node_driver(engine, LocalOnlySync(3), tmp_path, ctl)
    with pytest.raises(DriverError, match="RECOVERY_REQUIRED.*node_lost") as info:
        driver.run()
    assert isinstance(info.value.__cause__, PublicationError)
    assert "rollout-1" in str(info.value.__cause__)
    events = [e for e in _events(tmp_path / "events.jsonl")
              if e["event"] == "rl_reconfiguration" and e["result"] == "RECOVERY_REQUIRED"]
    assert len(events) == 1
    assert events[0]["rollout_id"] == 0 and "node_lost" in events[0]["error"]  # v1 is published at the end of round 0
    assert events[0]["cause"].startswith("PublicationError")
    assert slept == [2.0, 2.0]  # polled at t=0, 2, 4 -> lost on the third failure poll
    assert ("generate", 2) not in engine.calls


def test_failed_publication_without_node_loss_raises_the_original_error(tmp_path):
    engine = _engine(unacked_member_rounds={1: "rollout-1"})
    ctl = _NodeController(lose_at=None)
    driver, slept = _node_driver(engine, LocalOnlySync(3), tmp_path, ctl, grace=5.0)
    with pytest.raises(PublicationError, match="rollout-1"):
        driver.run()
    assert slept == [2.0, 2.0, 1.0] and sum(slept) == 5.0  # bounded by the grace period
    assert not [e for e in _events(tmp_path / "events.jsonl")
                if e["event"] == "rl_reconfiguration"]


def test_failure_on_single_node_island_never_polls_or_sleeps(tmp_path):
    engine = _engine(unacked_member_rounds={1: "rollout-1"})
    ctl = _NodeController(topology=None, lose_at=2)  # would report loss on any failure-path poll
    driver, slept = _node_driver(engine, LocalOnlySync(3), tmp_path, ctl)
    with pytest.raises(PublicationError, match="rollout-1"):
        driver.run()
    assert slept == []
    assert ctl.polls == 1  # only the per-round _probe_nodes before round 0; no failure-path poll


def test_colocated_publish_offloaded_sleeps_before_publish(tmp_path):
    """Miles --offload-train order (train.py): the actor sleeps before update_weights
    and the engines' KV resume, so the publication never shares the GPU with the
    resident training state (S13 FN smoke OOM). The next generate does not offload
    again; the export/apply at the sync boundary still sees the resident trainer."""
    engine = _engine()
    engine.trainer.publish_offloaded = True
    driver = _driver(engine, LocalOnlySync(2), tmp_path)
    final = driver.run()
    if PROFILE == "ports-only":
        # not declared by the backend (traits.publish_while_offloaded): publish resident
        calls = _engine_calls(engine)
        assert not any(a == ("offload",) and b[0] == "publish" for a, b in zip(calls, calls[1:]))
        assert final.policy_version == 2
        return

    expected = [("publish", 0), ("offload",), ("generate", 0), ("onload",), ("train", 0)]
    expected += [("offload",), ("publish", 1), ("generate", 1), ("onload",), ("train", 1),
                 ("offload",), ("publish", 2)]
    assert _engine_calls(engine) == expected
    assert final.policy_version == 2
    events = _events(tmp_path / "events.jsonl")
    phases = [e["phase"] for e in events if e["event"] == "rl_driver_phase"]
    assert phases[:9] == [
        "publish", "offload", "generate", "onload", "train", "sync", "offload", "publish", "generate",
    ]
    assert phases.count("offload") == 3


def test_miles_trainer_group_publish_offloaded_follows_offload_train():
    from types import SimpleNamespace
    from yeto.rl.adapters.miles.trainer import MilesTrainerGroup

    prop = MilesTrainerGroup.publish_offloaded
    assert prop.fget(SimpleNamespace(_args=SimpleNamespace(colocate=True, offload_train=True)))
    assert not prop.fget(SimpleNamespace(_args=SimpleNamespace(colocate=True, offload_train=False)))
    assert not prop.fget(SimpleNamespace(_args=SimpleNamespace(colocate=False, offload_train=True)))
