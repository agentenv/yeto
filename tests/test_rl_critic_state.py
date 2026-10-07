"""rl-algo-critic-family 4.1-4.4: critic state contract (CPU)."""

from __future__ import annotations

import json

import pytest
import torch

from yeto.rl.critic_state import (
    CriticCheckpointStore,
    CriticLayoutMismatch,
    CriticRoundMismatch,
    CriticRoundReceipt,
    RoleAverageFailed,
    TwoRoleStrictAvg,
    check_critic_layouts,
    critic_layout_hash,
    critic_weights_sha256,
)
from yeto.rl.engine.algorithm import AlgorithmSpec
from yeto.rl.engine.bridges import LocalOnlySync
from yeto.rl.engine.driver import EventTape, IslandDriver
from yeto.rl.engine.fake import FakeEngine, fake_capabilities

A, B = "a" * 64, "b" * 64
SPECS = [("embed.weight", (10, 4), "torch.float32"), ("output_layer.weight", (1, 4), "torch.float32")]
NAME = "base_model.model.layer.lora_A.weight"


def receipt(**kw):
    base = dict(rollout_id=0, actor_layout_hash=A, critic_layout_hash=B,
                critic_param_mode="full", critic_init="copy_actor_backbone")
    return CriticRoundReceipt(**{**base, **kw})


# -- 4.1 ------------------------------------------------------------------------------


def test_critic_layout_hash_is_separate_and_shape_sensitive():
    full = critic_layout_hash(SPECS, value_head="output_layer.weight")
    assert full == critic_layout_hash(list(reversed(SPECS)), value_head="output_layer.weight")
    wider = [SPECS[0], ("output_layer.weight", (1, 8), "torch.float32")]
    assert critic_layout_hash(wider, value_head="output_layer.weight") != full
    lora = critic_layout_hash(SPECS, value_head="output_layer.weight", param_mode="lora",
                              lora={"rank": 8, "alpha": 16, "target_modules": ["q"]})
    assert lora != full
    with pytest.raises(ValueError, match="value head"):
        critic_layout_hash(SPECS, value_head="embed.weight")
    with pytest.raises(ValueError, match="lora shape"):
        critic_layout_hash(SPECS, value_head="output_layer.weight", param_mode="lora")


def test_critic_layout_hash_scalar_head_is_pinned_and_bins_are_identity():
    # [1, hidden] payload must stay byte-for-byte (G1 PASS critic hashes must not drift).
    pinned = [("embed.weight", [4, 4], "bf16"), ("output_layer.weight", [1, 4], "bf16")]
    expected = "5939fb770a1c4a440b1c1f7d8d0da7cadd03f2104979f151226e6ebb16c72868"
    assert critic_layout_hash(pinned, value_head="output_layer.weight") == expected
    assert critic_layout_hash(pinned, value_head="output_layer.weight", value_bins=1) == expected
    # HL-Gauss / classification head: [bins, hidden] accepted only when bins matches.
    binned = [SPECS[0], ("output_layer.weight", (51, 4), "torch.float32")]
    h51 = critic_layout_hash(binned, value_head="output_layer.weight", value_bins=51)
    assert len(h51) == 64 and h51 != critic_layout_hash(SPECS, value_head="output_layer.weight")
    assert h51 == critic_layout_hash(list(reversed(binned)), value_head="output_layer.weight",
                                     value_bins=51)
    with pytest.raises(ValueError, match=r"\[1, hidden\]"):
        critic_layout_hash(binned, value_head="output_layer.weight")
    with pytest.raises(ValueError, match=r"\[51, hidden\]"):
        critic_layout_hash(SPECS, value_head="output_layer.weight", value_bins=51)
    with pytest.raises(ValueError, match="value_bins"):
        critic_layout_hash(SPECS, value_head="output_layer.weight", value_bins=0)


def test_receipt_has_both_layout_hashes_mode_and_init_source():
    event = receipt(critic_init_sha256="c" * 64).to_event()
    assert event["rl/critic/actor_layout_hash"] == A
    assert event["rl/critic/critic_layout_hash"] == B
    assert event["rl/critic/critic_param_mode"] == "full"
    assert event["rl/critic/critic_init"] == "copy_actor_backbone"
    assert event["rl/critic/critic_init_sha256"] == "c" * 64
    with pytest.raises(ValueError, match="critic_layout_hash"):
        receipt(critic_layout_hash="x")


def test_critic_layout_mismatch_refused():
    check_critic_layouts({0: receipt(), 1: receipt()})
    with pytest.raises(CriticLayoutMismatch, match="critic layout differs"):
        check_critic_layouts({0: receipt(), 1: receipt(critic_layout_hash="d" * 64)})
    with pytest.raises(CriticLayoutMismatch, match="actor layout differs"):
        check_critic_layouts({0: receipt(), 1: receipt(actor_layout_hash="d" * 64)})


# -- 4.2 ------------------------------------------------------------------------------


def _islands(n=2):
    """Fake two islands: each trains actor/critic tensors from the committed round."""
    engines = [FakeEngine(tensors={NAME: torch.zeros(1, 2)}, critic=True, step_delta=float(i + 1),
                          critic_tensors={"output_layer.weight": torch.zeros(1, 2)})
               for i in range(n)]
    return engines


def _local(engines, committed):
    out = {}
    for i, e in enumerate(engines):
        e.tensors = {k: v.clone() for k, v in committed.actor.items()}
        e.critic_tensors = {k: v.clone() for k, v in committed.critic.items()}
        e.tensors = {k: v + (i + 1) for k, v in e.tensors.items()}
        e.critic_tensors = {k: v + 10 * (i + 1) for k, v in e.critic_tensors.items()}
        out[i] = (e.tensors, e.critic_tensors)
    return out


def test_two_islands_strict_avg_both_roles_hash_consistent():
    engines = _islands()
    sync = TwoRoleStrictAvg({NAME: torch.zeros(1, 2)}, {"output_layer.weight": torch.zeros(1, 2)})
    committed = sync.round(_local(engines, sync.committed),
                           receipts={0: receipt(), 1: receipt()})
    assert committed.version == 1
    assert torch.equal(committed.actor[NAME], torch.full((1, 2), 1.5))
    assert torch.equal(committed.critic["output_layer.weight"], torch.full((1, 2), 15.0))
    applied = [(critic_weights_sha256(committed.actor), critic_weights_sha256(committed.critic))
               for _ in engines]  # both islands apply the same committed round
    assert len(set(applied)) == 1


def test_critic_average_failure_rolls_back_the_whole_round():
    engines = _islands()

    def broken(states):
        raise RuntimeError("critic syncer lost a fragment")

    sync = TwoRoleStrictAvg({NAME: torch.zeros(1, 2)}, {"output_layer.weight": torch.zeros(1, 2)},
                            average_critic=broken)
    before = (sync.committed.version, sync.committed.actor_sha256, sync.committed.critic_sha256)
    with pytest.raises(RoleAverageFailed, match="not committed, both roles stay at round 0"):
        sync.round(_local(engines, sync.committed))
    assert (sync.committed.version, sync.committed.actor_sha256,
            sync.committed.critic_sha256) == before


def test_layout_mismatch_refuses_the_round_before_averaging():
    sync = TwoRoleStrictAvg({NAME: torch.zeros(1, 2)}, {"output_layer.weight": torch.zeros(1, 2)})
    with pytest.raises(CriticLayoutMismatch):
        sync.round(_local(_islands(), sync.committed),
                   receipts={0: receipt(), 1: receipt(critic_layout_hash="e" * 64)})
    assert sync.committed.version == 0


# -- 4.3 ------------------------------------------------------------------------------


def test_checkpoint_save_restore_hash_and_round_check(tmp_path):
    store = CriticCheckpointStore(tmp_path)
    weights = {"output_layer.weight": torch.arange(4.0).reshape(1, 4)}
    optimizer = {"exp_avg": torch.ones(1, 4), "step": 3}
    manifest = store.save(round_id=2, weights=weights, optimizer=optimizer)
    restored, opt = store.restore(actor_round=2)
    assert critic_weights_sha256(restored) == manifest["weights_sha256"] \
        == critic_weights_sha256(weights)
    assert opt["step"] == 3 and torch.equal(opt["exp_avg"], optimizer["exp_avg"])
    with pytest.raises(CriticRoundMismatch, match="actor round 3 != critic round 2"):
        store.restore(actor_round=3)


def test_checkpoint_refuses_a_tampered_round(tmp_path):
    store = CriticCheckpointStore(tmp_path)
    store.save(round_id=1, weights={"w": torch.ones(2)}, optimizer={})
    torch.save({"w": torch.zeros(2)}, tmp_path / "critic" / "round-1" / "weights.pt")
    with pytest.raises(Exception, match="fails its manifest"):
        store.restore(actor_round=1)


# -- 4.4 ------------------------------------------------------------------------------


def test_tape_records_critic_weight_hash_and_value_metrics(tmp_path):
    engine = FakeEngine(tensors={NAME: torch.zeros(1, 2)}, critic=True,
                        critic_tensors={"output_layer.weight": torch.zeros(1, 2)})
    spec = AlgorithmSpec(advantage_estimator="ppo")
    caps = fake_capabilities().with_unverified(["advantage_estimators:ppo", "execution:critic"])
    IslandDriver(
        learner_id=0, rollout=engine.rollout, trainer=engine.trainer,
        policy_state=engine.policy_state, publisher=engine.publisher,
        placement=engine.placement, capabilities=caps, algorithm=spec,
        sync=LocalOnlySync(2), events=EventTape(tmp_path / "events.jsonl", 0),
    ).run()
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    critic_rows = [r for r in rows if r["event"] == "rl_critic_round"]
    assert [r["rollout_id"] for r in critic_rows] == [0, 1]
    hashes = [r["rl/critic/critic_weights_sha256"] for r in critic_rows]
    assert len(set(hashes)) == 2 and all(len(h) == 64 for h in hashes)
    for r in critic_rows:
        assert r["rl/critic/value_loss"] > 0 and 0 < r["rl/critic/explained_variance"] < 1
        assert r["rl/critic/critic_layout_hash"] != r["rl/critic/actor_layout_hash"]


def test_grpo_tape_has_no_critic_rows(tmp_path):
    engine = FakeEngine(tensors={NAME: torch.zeros(1, 2)})
    IslandDriver(
        learner_id=0, rollout=engine.rollout, trainer=engine.trainer,
        policy_state=engine.policy_state, publisher=engine.publisher,
        placement=engine.placement, capabilities=fake_capabilities(), algorithm=AlgorithmSpec(),
        sync=LocalOnlySync(1), events=EventTape(tmp_path / "events.jsonl", 0),
    ).run()
    assert "rl_critic_round" not in (tmp_path / "events.jsonl").read_text()
