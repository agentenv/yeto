"""rl-algo-critic-family 3.1/3.2: single-island colocated PPO on the ports path (CPU)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from yeto.rl.contracts import LocalStepReceipt
from yeto.rl.engine.algorithm import AlgorithmSpec, AlgorithmSpecError, check_unverified_allowance
from yeto.rl.engine.bridges import LocalOnlySync
from yeto.rl.engine.capabilities import CapabilityMismatch
from yeto.rl.engine.driver import EventTape, IslandDriver
from yeto.rl.engine.fake import FakeEngine, fake_capabilities
from yeto.rl.engine.miles_adapter import algorithm_flags as af
from yeto.rl.engine.miles_adapter.entry import miles_capabilities, receipt_role_family
from yeto.rl.engine.miles_adapter.rollout import policy_token
from yeto.rl.engine.miles_adapter.state_plugin import (
    CRITIC_RECORDERS,
    CRITIC_STATE_SUMMARY,
    GRAD_NORM,
    STEP_LOSSES,
    explained_variance,
)
from yeto.rl.engine.miles_adapter.trainer import MilesTrainerGroup, TrainStepError
from yeto.rl.engine.miles_adapter.trainer_rebuild import SwappableActor, swap_critic
from yeto.rl.engine.ports import GroupMetadata, RolloutBatchHandle
from yeto.rl.local_learner import ComponentIdentity, ParameterLayout, ParameterSpec

NAME = "base_model.model.layer.lora_A.weight"
PPO = AlgorithmSpec(advantage_estimator="ppo")
ALLOW = ("advantage_estimators:ppo", "execution:critic")
H = "a" * 64
TOKEN = policy_token(3, H)


# -- 3.1 gate: unverified allowance, single island ----------------------------------


def _driver(tmp_path, caps, engine):
    return IslandDriver(
        learner_id=0, rollout=engine.rollout, trainer=engine.trainer,
        policy_state=engine.policy_state, publisher=engine.publisher,
        placement=engine.placement, capabilities=caps, algorithm=PPO,
        sync=LocalOnlySync(2), events=EventTape(tmp_path / "events.jsonl", 0),
    )


def test_ppo_refused_without_allowance_with_g1_hint(tmp_path):
    engine = FakeEngine(tensors={NAME: torch.zeros(1, 2)}, critic=True)
    with pytest.raises(CapabilityMismatch, match="--rl-allow-unverified-mechanism execution:critic"):
        _driver(tmp_path, fake_capabilities(), engine).run()
    assert engine.calls == []
    # the Miles adapter does not declare the critic before G1 (design D3)
    caps = miles_capabilities("sha256:" + "0" * 64)
    assert caps.execution.critic is False and "ppo" not in caps.advantage_estimators
    with pytest.raises(CapabilityMismatch, match="execution:critic"):
        caps.check(layout="lora", placement="colocated", execution_mode="colocated-serial",
                   algorithm=PPO)


def test_ppo_single_island_runs_with_allowance_and_reports_value_metrics(tmp_path):
    engine = FakeEngine(tensors={NAME: torch.zeros(1, 2)}, critic=True)
    caps = fake_capabilities().with_unverified(ALLOW)
    _driver(tmp_path, caps, engine).run()
    trains = [c for c in engine.calls if c[0] in ("critic_train", "train")]
    assert trains == [("critic_train", 0), ("train", 0), ("critic_train", 1), ("train", 1)]
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    trained = [e for e in events if e["event"] == "rl_round_trained"]
    assert len(trained) == 2
    for event in trained:
        metrics = event["train_metrics"]
        assert set(metrics) >= {"critic/value_loss", "critic/explained_variance"}
        assert all(isinstance(v, float) and v == v for v in metrics.values())


def test_allowance_refused_with_two_islands_or_outer_sync():
    assert check_unverified_allowance(ALLOW, islands=1, outer_sync=False) == tuple(sorted(ALLOW))
    with pytest.raises(AlgorithmSpecError, match="single-island"):
        check_unverified_allowance(ALLOW, islands=2, outer_sync=False)
    with pytest.raises(AlgorithmSpecError, match="single-island"):
        check_unverified_allowance(ALLOW, islands=1, outer_sync=True)


def test_dry_run_argv_snapshot_ppo():
    flags = [x for name in ALLOW for x in ("--rl-allow-unverified-mechanism", name)]
    refused = af.dry_run(["--dry-run", "--extra", "--advantage-estimator ppo"])
    assert refused["verdict"] == "rejected" and "execution:critic" in refused["error"]
    result = af.dry_run(["--dry-run", "--extra",
                         "--advantage-estimator ppo --value-clip 0.3 --critic-lr 2e-6", *flags])
    assert result["verdict"] == "accepted"
    assert result["miles_argv"] == [
        "--advantage-estimator", "ppo", "--gamma", "1.0", "--lambd", "1.0",
        "--value-clip", "0.3", "--critic-lr", "2e-06", "--num-critic-only-steps", "0"]


def test_receipt_family_and_contract_accept_ppo():
    assert receipt_role_family(PPO) == "ppo"
    assert receipt_role_family(AlgorithmSpec()) == "grpo"
    assert receipt_role_family(AlgorithmSpec(advantage={"estimator": "gspo"})) == "grpo"
    with pytest.raises(ValueError, match="needs_critic"):
        receipt_role_family(AlgorithmSpec(advantage={"estimator": "ppo"}))
    LocalStepReceipt(
        algorithm="ppo", learner_id=0, learner_generation=0, base_policy_version=0,
        base_policy_hash=H, input_batch_hash=H, trajectory_ids=("r0:g:s",), trained_tokens=1,
        optimizer_steps=1, optimizer_step_succeeded=True, parameter_layout_hash=H,
    )


def test_trainer_rebuild_swaps_the_critic():
    critic = SwappableActor(SimpleNamespace(run_plugin=lambda *a: None, name="old"))
    swap_critic(critic, SimpleNamespace(run_plugin=lambda *a: None, name="new"))
    assert critic.target.name == "new" and critic.generation == 1
    swap_critic(None, None)
    with pytest.raises(RuntimeError, match="no critic handle"):
        swap_critic(None, object())
    with pytest.raises(RuntimeError, match="has no critic"):
        swap_critic(critic, None)


# -- 3.1/3.2 Miles trainer order and value metrics ---------------------------------


def _handle():
    return RolloutBatchHandle(
        rollout_id=3, policy_version=3, policy_hash=H,
        groups=(GroupMetadata("g0", ("s0", "s1"), TOKEN, 0.5, 0.5, 10),),
        completed=1, aborted=0, payload="PACK",
    )


class _Group:
    def __init__(self, role, log, *, losses=(), outcome="NORMAL", head_rows=1):
        self.role, self.log, self.losses, self.outcome = role, log, losses, outcome
        self.head_rows = head_rows

    async def train(self, rollout_id, pack, external_data=None):
        self.log.append((self.role, "train", rollout_id, pack, external_data))
        return [SimpleNamespace(outcome=SimpleNamespace(name=self.outcome), role=self.role)]

    async def run_plugin(self, fn_path, kwargs=None):
        self.log.append((self.role, "plugin", fn_path))
        if fn_path == STEP_LOSSES:
            return [list(self.losses)]
        if fn_path == CRITIC_RECORDERS:
            return [True]
        if fn_path == CRITIC_STATE_SUMMARY:
            return [{"rank": 1, "specs": [["0:module.output_layer.weight",
                                           [self.head_rows, 4], "bf16"]],
                     "weights_sha256": "1" * 64},
                    {"rank": 0, "specs": [["0:module.embedding.weight", [8, 4], "bf16"]],
                     "weights_sha256": "0" * 64}]
        if fn_path == GRAD_NORM:
            return [0.5]
        return [[1e-5]]

    async def offload(self):
        self.log.append((self.role, "offload"))


def _trainer(log, critic, released):
    actor = _Group("actor", log)
    return MilesTrainerGroup(
        args=SimpleNamespace(num_steps_per_rollout=1, offload_train=True),
        actor_model=actor, learner_id=0, learner_generation=0,
        parameter_layout_hash=lambda: H, algorithm="ppo", spec=PPO,
        release_refs=lambda args, pack: released.append(("pack", pack)),
        critic_model=critic,
        release_outputs=lambda outputs: released.append(("values", len(outputs))),
    )


def test_critic_trains_first_then_offloads_then_actor_gets_values():
    log, released = [], []
    critic = _Group("critic", log, losses=[
        {"metrics": {"value_loss": 0.4, "value_clipfrac": 0.1, "explained_variance": 0.2}},
        {"metrics": {"train/value_loss": 0.2, "explained_variance": 0.4}},
    ])
    trainer = _trainer(log, critic, released)
    receipt = trainer.train_step(_handle())
    assert receipt.algorithm == "ppo" and receipt.optimizer_step_succeeded
    order = [entry[:2] if entry[1] != "plugin" else entry for entry in log]
    assert order[:6] == [
        ("critic", "plugin", CRITIC_RECORDERS), ("critic", "train"),
        ("critic", "plugin", GRAD_NORM), ("critic", "plugin", STEP_LOSSES),
        ("critic", "offload"), ("actor", "train")]
    actor_train = next(e for e in log if e[:2] == ("actor", "train"))
    assert [o.role for o in actor_train[4]] == ["critic"]  # external_data = critic outputs
    assert released == [("values", 1), ("pack", "PACK")]
    metrics = trainer.round_metrics()
    assert metrics["critic/value_loss"] == pytest.approx(0.4)  # first key form found
    assert metrics["critic/explained_variance"] == pytest.approx(0.3)
    assert metrics["critic/grad_norm"] == 0.5


def test_failed_critic_step_releases_values_and_payload():
    log, released = [], []
    trainer = _trainer(log, _Group("critic", log, outcome="DISCARDED"), released)
    with pytest.raises(TrainStepError, match="critic train step"):
        trainer.train_step(_handle())
    assert ("actor", "train") not in [e[:2] for e in log]
    assert released == [("values", 1), ("pack", "PACK")]


def test_grpo_trainer_unchanged_without_critic():
    log, released = [], []
    trainer = MilesTrainerGroup(
        args=SimpleNamespace(num_steps_per_rollout=1, offload_train=True),
        actor_model=_Group("actor", log), learner_id=0, learner_generation=0,
        parameter_layout_hash=lambda: H,
        release_refs=lambda args, pack: released.append(pack),
    )
    trainer.train_step(_handle())
    assert log[0] == ("actor", "train", 3, "PACK", None)
    assert not any(e[0] == "critic" for e in log) and trainer.last_critic_metrics == {}


def test_explained_variance():
    returns = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert explained_variance(returns, returns) == pytest.approx(1.0)
    assert explained_variance(returns, torch.zeros(4)) == pytest.approx(
        1.0 - float(torch.var(returns, unbiased=False)) / float(torch.var(returns, unbiased=False)))
    assert explained_variance(returns, returns.mean().expand(4)) == pytest.approx(0.0)
    masked = explained_variance(torch.tensor([1.0, 2.0, 9.0]), torch.tensor([1.0, 2.0, 0.0]),
                                torch.tensor([1, 1, 0]))
    assert masked == pytest.approx(1.0)
    assert explained_variance(torch.ones(3), torch.zeros(3)) is None


# -- 3.2 role table ---------------------------------------------------------------------


def _component(role):
    return ComponentIdentity(role=role, model_revision="c" * 40, config_hash="d" * 64)


def test_local_learner_ppo_roles_and_role_lanes():
    specs = [ParameterSpec(role="actor", name="w", shape=(2,), dtype="float32", numel=2),
             ParameterSpec(role="critic", name="w", shape=(2,), dtype="float32", numel=2)]
    both = ParameterLayout.create(algorithm="ppo", components=[_component("actor"),
                                  _component("critic")], specs=specs, num_fragments=1)
    critic_lane = ParameterLayout.create(algorithm="ppo", components=[_component("critic")],
                                         specs=specs[1:], num_fragments=1, stream_role="critic")
    actor_lane = ParameterLayout.create(algorithm="ppo", components=[_component("actor")],
                                        specs=specs[:1], num_fragments=1, stream_role="actor")
    assert len({both.layout_hash, critic_lane.layout_hash, actor_lane.layout_hash}) == 3
    with pytest.raises(ValueError, match="ppo requires exactly these trainable roles"):
        ParameterLayout.create(algorithm="ppo", components=[_component("actor")],
                               specs=specs[:1], num_fragments=1)
    with pytest.raises(ValueError, match="role-scoped"):
        ParameterLayout.create(algorithm="grpo", components=[_component("actor")],
                               specs=specs[:1], num_fragments=1, stream_role="actor")


def test_critic_round_receipt_from_the_critic_processes():
    log, released = [], []
    critic = _Group("critic", log, losses=[{"metrics": {"value_loss": 0.4,
                                                       "explained_variance": 0.2}}])
    trainer = _trainer(log, critic, released)
    trainer._args.yeto_rl_critic_init_sha256 = "c" * 64
    trainer.train_step(_handle())
    receipt = trainer.critic_round_receipt(3)
    assert receipt.actor_layout_hash == H and receipt.critic_layout_hash != H
    assert receipt.critic_param_mode == "full" and receipt.critic_init == "copy_actor_backbone"
    assert receipt.critic_init_sha256 == "c" * 64
    assert receipt.value_loss == pytest.approx(0.4)
    assert receipt.explained_variance == pytest.approx(0.2)
    assert len(receipt.critic_weights_sha256) == 64
    grpo = MilesTrainerGroup(
        args=SimpleNamespace(num_steps_per_rollout=1), actor_model=_Group("actor", log),
        learner_id=0, learner_generation=0, parameter_layout_hash=lambda: H)
    assert grpo.critic_round_receipt(0) is None


def test_critic_round_receipt_accepts_hl_gauss_value_head():
    # S14 8.4 G1 blocker: SAO HL-Gauss critic has a [value_num_bins, hidden] head
    # (fork model_provider._value_head_output_size); the receipt must find it.
    log, released = [], []
    critic = _Group("critic", log, losses=[{"metrics": {"value_loss": 0.4}}], head_rows=51)
    trainer = _trainer(log, critic, released)
    trainer._args.value_loss_type = "classification"
    trainer._args.value_num_bins = 51
    trainer.train_step(_handle())
    receipt = trainer.critic_round_receipt(3)
    assert len(receipt.critic_layout_hash) == 64 and receipt.critic_layout_hash != H
    # scalar args against a binned head (or vice versa) still refuse, with the bins named
    trainer._args.value_loss_type = "mse"
    with pytest.raises(ValueError, match=r"\[1, hidden\]"):
        trainer.critic_round_receipt(3)
    trainer._args.value_loss_type = "classification"
    critic.head_rows = 1
    with pytest.raises(ValueError, match=r"\[51, hidden\]"):
        trainer.critic_round_receipt(3)


def test_value_metrics_recorder_reports_step_level_ev_at_micro_batch_1(monkeypatch):
    """s13-g1-modal-20261007b: at --micro-batch-size 1 with PPO gamma=lambd=1 every
    micro-batch holds one sample whose returns are one constant (Var(G)=0), so a
    per-micro-batch EV is always None. The recorder pools the step's micro-batches
    and the step-loss record carries the step-level explained_variance."""

    import sys
    import types

    from yeto.rl.engine.miles_adapter import state_plugin as sp

    losses_mod = types.ModuleType("miles.backends.training_utils.loss_hub.losses")
    model_mod = types.ModuleType("miles.backends.megatron_utils.model")

    def value_loss_function(args, batch, logits, sum_of_sample_mean):
        return torch.tensor(1.0), {"value_loss": torch.tensor(0.5)}

    losses_mod.value_loss_function = value_loss_function
    micro = [  # (returns, old values, mask): one sample each, constant returns
        (torch.full((3,), 1.0), torch.tensor([0.2, 0.4, 0.6]), torch.tensor([1, 1, 0])),
        (torch.full((2,), -1.0), torch.tensor([-0.5, 0.1]), torch.tensor([1, 1])),
    ]

    def train_one_step(*args, **kwargs):
        reported = {}
        for returns, values, mask in micro:
            batch = {"returns": [returns], "values": [values], "loss_masks": [mask]}
            _, reported = losses_mod.value_loss_function(None, batch, None, None)
            assert "explained_variance" not in reported  # loss dict unchanged
        return reported, 1.25

    model_mod.train_one_step = train_one_step
    pkg = {name: types.ModuleType(name) for name in (
        "miles", "miles.backends", "miles.backends.training_utils",
        "miles.backends.training_utils.loss_hub", "miles.backends.megatron_utils")}
    pkg["miles.backends.training_utils.loss_hub"].losses = losses_mod
    pkg["miles.backends.megatron_utils"].model = model_mod
    for name, module in {**pkg, losses_mod.__name__: losses_mod,
                         model_mod.__name__: model_mod}.items():
        monkeypatch.setitem(sys.modules, name, module)
    for name in ("_STEP_GRAD_NORMS", "_STEP_APPLIED_LRS", "_STEP_LOSSES", "_EV_STATS"):
        monkeypatch.setattr(sp, name, [])  # module-level per-step records: test-local
    monkeypatch.setattr(sp, "_RECORDER_INSTALLED", False)
    monkeypatch.setattr(sp, "_VALUE_METRICS_INSTALLED", False)
    monkeypatch.setattr(sp, "_record_applied_lr", lambda *a: None)
    monkeypatch.setattr(sp, "_arm_grad_audit", lambda *a: None)
    assert sp.install_critic_recorders(None)

    model_mod.train_one_step(optimizer=None)
    for returns, _, _ in micro:  # each micro-batch alone has no defined EV
        assert explained_variance(returns, returns * 0) is None
    (record,) = sp.step_losses(None)
    expected = explained_variance(torch.tensor([1.0, 1.0, -1.0, -1.0]),
                                  torch.tensor([0.2, 0.4, -0.5, 0.1]))
    assert record["metrics"]["explained_variance"] == pytest.approx(expected)
    assert record["metrics"]["value_loss"] == pytest.approx(0.5)
    from yeto.rl.engine.miles_adapter.trainer import critic_round_metrics

    metrics = critic_round_metrics([record], 1.25)
    assert metrics["critic/explained_variance"] == pytest.approx(expected)

    model_mod.train_one_step(optimizer=None)  # stats reset per optimizer step
    (again,) = sp.step_losses(None)
    assert again["metrics"]["explained_variance"] == pytest.approx(expected)
