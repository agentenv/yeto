"""INFRA channels for 1b/2a: per-round trained counts, GSPO clip fraction, non-zero advantages."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from yeto.rl.engine.miles_adapter import rollout_meta_hook as hook
from yeto.rl.engine.miles_adapter import state_plugin
from yeto.rl.engine.miles_adapter import trainer as tr
from yeto.rl.engine.miles_adapter.rollout import handle_from_metadata

H = "a" * 64


def test_step_loss_recorder_keeps_clipfrac_and_skips_non_last_stage():
    state_plugin._STEP_LOSSES.clear()
    state_plugin._record_step_losses(({"train/pg_clipfrac": 0.25, "train/loss": 1.0}, 0.5, "ok"))
    state_plugin._record_step_losses(({}, 0.5, "ok"))  # not last PP stage
    state_plugin._record_step_losses(({"train/loss": 1.0}, 0.5, "ok"))
    assert state_plugin.step_losses(None) == [
        {"pg_clipfrac": 0.25, "loss_tokens": None,
         "metrics": {"train/pg_clipfrac": 0.25, "train/loss": 1.0}},
        {"pg_clipfrac": None, "loss_tokens": None, "metrics": {"train/loss": 1.0}},
    ]
    assert state_plugin.step_losses(None) == []


def test_gspo_masked_fraction_uses_optional_seq_adv(monkeypatch):
    losses = [{"pg_clipfrac": 1.0, "loss_tokens": None}]
    monkeypatch.setitem(sys.modules, "yeto.rl.algos.seq_adv", None)  # module absent
    assert tr.clipfrac_masked_fraction(losses) is None
    fake = types.ModuleType("yeto.rl.algos.seq_adv")
    seen = {}

    def clipfrac_from_losses(step_losses, token_counts=None):
        seen["args"] = (step_losses, token_counts)
        return 1.0

    fake.clipfrac_from_losses = clipfrac_from_losses
    monkeypatch.setitem(sys.modules, "yeto.rl.algos.seq_adv", fake)
    assert tr.clipfrac_masked_fraction(losses) == 1.0
    assert seen["args"] == (losses, None)
    assert tr._mean_clipfrac([{"pg_clipfrac": 0.2}, {"pg_clipfrac": 0.4}]) == pytest.approx(0.3)
    assert tr._mean_clipfrac([{"pg_clipfrac": None}]) is None


def _group():
    return {"group_id": "g0", "sample_ids": ["s0"], "policy_token": "t", "reward_mean": 0.0,
            "reward_std": 0.0, "token_count": 1}


def test_nonzero_advantages_are_attributed_to_their_own_round(tmp_path):
    """Review R2 lag: the dispatcher reports AFTER the all-samples hook; each round's
    handle must carry that round's count (24, 16, 24), never the previous one."""
    from yeto.rl.engine.miles_adapter.rollout import DirMetadataSource

    source = DirMetadataSource(tmp_path)
    sink = source.sink_spec
    seen = []
    for rollout_id, count in ((0, 24), (1, 16), (2, 24)):
        payload = {"schema": hook.METADATA_SCHEMA, "rollout_id": rollout_id, "groups": [_group()],
                   "completed": 1, "aborted": 0}
        hook.put_to_sink(payload, sink)  # all-samples hook first
        sample = SimpleNamespace(rollout_id=rollout_id)
        hook.record_round_metadata(None, [[sample]], sink=sink, nonzero_advantages=count)
        merged = source.take(rollout_id)
        h = handle_from_metadata(merged, rollout_id=rollout_id, policy_version=rollout_id,
                                 policy_hash=H, data_pack=None)
        seen.append(h.nonzero_advantages)
    assert seen == [24, 16, 24]
    # without a report the value is unknown, not stale
    hook.put_to_sink({"schema": hook.METADATA_SCHEMA, "rollout_id": 3, "groups": [_group()],
                      "completed": 1, "aborted": 0}, sink)
    assert handle_from_metadata(source.take(3), rollout_id=3, policy_version=3, policy_hash=H,
                                data_pack=None).nonzero_advantages is None
    with pytest.raises(RuntimeError, match="unknown"):
        hook.record_round_metadata(None, 4, sink=sink, x=1)
    with pytest.raises(RuntimeError, match="non-negative"):
        hook.record_round_metadata(None, 4, sink=sink, nonzero_advantages=-1)


def test_round_metadata_for_another_rollout_is_refused():
    from yeto.rl.engine.miles_adapter.rollout import RolloutMetadataError, merge_round_metadata

    with pytest.raises(RolloutMetadataError, match="arrived with rollout 2"):
        merge_round_metadata({"rollout_id": 2}, [{"schema": hook.ROUND_META_SCHEMA,
                                                  "rollout_id": 1, "nonzero_advantages": 3}], 2)


def test_mismatch_metrics_and_correction_masked_fraction(monkeypatch):
    steps = [
        {"pg_clipfrac": 0.1, "metrics": {"train/train_rollout_kl": 0.02, "train/tis_clipfrac": 0.3,
                                         "train/loss": 1.0}},
        {"pg_clipfrac": 0.1, "metrics": {"train/train_rollout_kl": 0.04, "train/tis_clipfrac": 0.5,
                                         "train/loss": 2.0}},
    ]
    round_metrics = tr.mean_step_metrics(steps)
    assert round_metrics["train/tis_clipfrac"] == pytest.approx(0.4)
    assert set(tr.mismatch_metrics(round_metrics)) == {"train/train_rollout_kl", "train/tis_clipfrac"}
    monkeypatch.setitem(sys.modules, "yeto.rl.algos.mismatch_correction", None)
    assert tr.correction_masked_fraction(object(), round_metrics) is None
    assert tr._has_corrections(object()) is False
    fake = types.ModuleType("yeto.rl.algos.mismatch_correction")
    fake.selected_corrections = lambda spec: ("icepop",)
    fake.masked_fraction_from_metrics = lambda spec, m: m["train/tis_clipfrac"]
    monkeypatch.setitem(sys.modules, "yeto.rl.algos.mismatch_correction", fake)
    spec = SimpleNamespace(correction=object())
    assert tr._has_corrections(spec) is True
    assert tr._has_corrections(object()) is False  # no correction field: robust default
    assert tr.correction_masked_fraction(spec, round_metrics) == pytest.approx(0.4)


def test_driver_round_event_carries_labelled_mismatch(tmp_path):
    import json

    import torch

    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.bridges import LocalOnlySync
    from yeto.rl.engine.driver import EventTape, IslandDriver
    from yeto.rl.engine.execution_profile import ExecutionProfile
    from yeto.rl.engine.fake import FakeEngine, fake_capabilities

    engine = FakeEngine(tensors={"base_model.model.layer.lora_A.weight": torch.zeros(1, 2)},
                        step_delta=1.0, placement_kind="fixed-partition")
    engine.trainer.algorithm_metrics = lambda: {"train/train_rollout_kl": 0.01}
    profile = ExecutionProfile(name="p", execution_mode="partitioned-serial",
                               outer_protocol="none").bind_algorithm(AlgorithmSpec())
    IslandDriver(learner_id=0, rollout=engine.rollout, trainer=engine.trainer,
                 policy_state=engine.policy_state, publisher=engine.publisher,
                 placement=engine.placement,
                 capabilities=fake_capabilities(execution_modes={"partitioned-serial"}),
                 algorithm=AlgorithmSpec(), sync=LocalOnlySync(1),
                 events=EventTape(tmp_path / "e.jsonl", 0), profile=profile).run()
    events = [json.loads(line) for line in (tmp_path / "e.jsonl").read_text().splitlines()]
    (trained,) = [e for e in events if e["event"] == "rl_round_trained"]
    assert trained["mismatch"] == {"train/train_rollout_kl": 0.01}
    assert trained["label/weight_transport"] == "nccl-broadcast"
    assert trained["label/profile_hash"] == profile.contract_hash


def test_driver_reports_each_rounds_own_nonzero_advantages(tmp_path):
    import dataclasses
    import json

    import torch

    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.bridges import LocalOnlySync
    from yeto.rl.engine.driver import EventTape, IslandDriver
    from yeto.rl.engine.fake import FakeEngine, fake_capabilities

    engine = FakeEngine(tensors={"base_model.model.layer.lora_A.weight": torch.zeros(1, 2)},
                        step_delta=1.0)
    counts = {0: 24, 1: 16, 2: 24}
    original = engine.rollout.generate
    engine.rollout.generate = lambda r: dataclasses.replace(original(r), nonzero_advantages=counts[r])
    IslandDriver(learner_id=0, rollout=engine.rollout, trainer=engine.trainer,
                 policy_state=engine.policy_state, publisher=engine.publisher,
                 placement=engine.placement, capabilities=fake_capabilities(),
                 algorithm=AlgorithmSpec(), sync=LocalOnlySync(3),
                 events=EventTape(tmp_path / "e.jsonl", 0)).run()
    events = [json.loads(line) for line in (tmp_path / "e.jsonl").read_text().splitlines()]
    got = {e["rollout_id"]: e["nonzero_advantages"] for e in events if e["event"] == "rl_round_trained"}
    assert got == counts


def test_receipt_label_is_the_role_family_matching_the_layout():
    from yeto.rl.engine.miles_adapter.entry import receipt_role_family
    from yeto.rl.local_learner import _ROLES_BY_ALGORITHM

    for estimator in ("grpo", "gspo", "reinforce_plus_plus", "reinforce_plus_plus_baseline"):
        label = receipt_role_family(SimpleNamespace(advantage_estimator=estimator))
        assert label == "grpo" and label in _ROLES_BY_ALGORITHM
    with pytest.raises(ValueError, match="critic"):
        receipt_role_family(SimpleNamespace(advantage_estimator="ppo"))
    from yeto.rl.contracts import _require_algorithm

    with pytest.raises(ValueError):
        _require_algorithm("gspo")  # contracts keep the role-family vocabulary


def test_round_id_comes_from_the_policy_token_not_trajectory_keys(tmp_path):
    """Multi-segment agentic rollouts: Sample.rollout_id is a trajectory key."""
    from yeto.rl.core import policy_snapshot_token
    from yeto.rl.engine.miles_adapter.rollout import DirMetadataSource

    source = DirMetadataSource(tmp_path)
    sink = source.sink_spec
    source.set_policy_token(policy_snapshot_token(3, H))
    segments = [[SimpleNamespace(rollout_id=1001)], [SimpleNamespace(rollout_id=1002)]]
    assert hook.current_round_id(hook._flat(segments), sink) == 3
    hook.put_to_sink({"schema": hook.METADATA_SCHEMA, "rollout_id": 3, "groups": [_group()],
                      "completed": 1, "aborted": 0}, sink)
    hook.record_round_metadata(None, segments, sink=sink, nonzero_advantages=5)
    assert source.take(3)["nonzero_advantages"] == 5
    # no token (fixtures/legacy): sample fallback
    assert hook.current_round_id([SimpleNamespace(rollout_id=7)], f"dir:{tmp_path}/none") == 7


def test_sample_filter_counts_reach_the_round_event(tmp_path):
    import dataclasses
    import json

    import torch

    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.bridges import LocalOnlySync
    from yeto.rl.engine.driver import EventTape, IslandDriver
    from yeto.rl.engine.fake import FakeEngine, fake_capabilities

    engine = FakeEngine(tensors={"base_model.model.layer.lora_A.weight": torch.zeros(1, 2)},
                        step_delta=1.0)
    original = engine.rollout.generate

    def gen(r):
        b = original(r)
        groups = tuple(dataclasses.replace(g, filtered_samples=i) for i, g in enumerate(b.groups))
        return dataclasses.replace(b, groups=groups)

    engine.rollout.generate = gen
    IslandDriver(learner_id=0, rollout=engine.rollout, trainer=engine.trainer,
                 policy_state=engine.policy_state, publisher=engine.publisher,
                 placement=engine.placement, capabilities=fake_capabilities(),
                 algorithm=AlgorithmSpec(), sync=LocalOnlySync(1),
                 events=EventTape(tmp_path / "e.jsonl", 0)).run()
    (ev,) = [json.loads(l) for l in (tmp_path / "e.jsonl").read_text().splitlines()
             if '"rl_round_trained"' in l]
    assert ev["filtered_samples"] == sum(range(engine.groups))


def test_router_inflight_probe_and_driver_sampler(tmp_path):
    import json
    import time as _time

    import torch

    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.bridges import LocalOnlySync
    from yeto.rl.engine.driver import EventTape, IslandDriver
    from yeto.rl.engine.execution_profile import ExecutionProfile
    from yeto.rl.engine.fake import FakeEngine, fake_capabilities
    from yeto.rl.engine.miles_adapter.rollout import MilesRolloutPool

    pool = MilesRolloutPool(inference_controller=None, rollout_executor=None, metadata=None,
                            expected_policy=lambda: (0, H),
                            args=SimpleNamespace(sglang_router_ip="10.0.0.1", sglang_router_port=3000))
    seen = []
    body = {"inflight": {"http://a": 3, "http://b": 1}, "cordoned": ["http://b"]}
    assert pool.load_sample(http_get=lambda url: seen.append(url) or body) == {
        "active_requests": 4, "workers": 2, "cordoned": 1}
    assert seen == ["http://10.0.0.1:3000/worker_inflight"]

    def missing(url):
        raise OSError("404")

    assert pool.load_sample(http_get=missing) is None  # stock router: unknown, not 0
    assert MilesRolloutPool(inference_controller=None, rollout_executor=None, metadata=None,
                            expected_policy=lambda: (0, H)).load_sample() is None

    engine = FakeEngine(tensors={"base_model.model.layer.lora_A.weight": torch.zeros(1, 2)},
                        step_delta=1.0, placement_kind="fixed-partition")
    original = engine.rollout.generate
    engine.rollout.generate = lambda r: (_time.sleep(0.12), original(r))[1]
    engine.rollout.load_sample = lambda: {"active_requests": 2, "workers": 1, "cordoned": 0}
    profile = ExecutionProfile(name="p", execution_mode="partitioned-serial",
                               outer_protocol="none").bind_algorithm(AlgorithmSpec())
    driver = IslandDriver(learner_id=0, rollout=engine.rollout, trainer=engine.trainer,
                          policy_state=engine.policy_state, publisher=engine.publisher,
                          placement=engine.placement,
                          capabilities=fake_capabilities(execution_modes={"partitioned-serial"}),
                          algorithm=AlgorithmSpec(), sync=LocalOnlySync(1),
                          events=EventTape(tmp_path / "e.jsonl", 0), profile=profile, observe=True)
    driver.load_sample_interval_s = 0.02
    driver.run()
    samples = [json.loads(l) for l in (tmp_path / "e.jsonl").read_text().splitlines()
               if '"rl_load_sample"' in l]
    assert samples and all(s["active_requests"] == 2 and s["profile_hash"] for s in samples)
