"""rl-infra-spec 4.7 (CPU, protocol only): trainer DP change + in-pool role transfer (T2R2 <-> T1R3).

Engines, publisher and the fork rebuild are fakes; ranks are the pure-torch
stand-ins of ``tests/rl_reshard_fakes.py``. Not GPU acceptance (A9).
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch

from tests.rl_reshard_fakes import GBS, MBS, default_args, gathered, make_world, train_step
from yeto.rl.elastic_benchmark.capabilities import ResourceConfig, attestation_from_dict
from yeto.rl.engine.cut import AlgorithmIdentity, CutProgress, RestoreExpectation
from yeto.rl.engine.miles_adapter import LoopRunner
from yeto.rl.engine.miles_adapter.trainer import CutContext, MilesTrainerGroup
from yeto.rl.engine.miles_adapter.trainer_rebuild import SwappableActor
from yeto.rl.engine.miles_adapter.trainer_resize import MilesTrainerOps
from yeto.rl.engine.trainer_transition import (
    READY_TO_COMMIT,
    TrainerEdgeRejected,
    TrainerTransition,
    plan_trainer_edge,
    recovery_decision,
)

SHA = "a" * 64
ALGO = AlgorithmIdentity(SHA)
CONFIGS = {
    "T2R2": ResourceConfig("T2R2", 2, 2, placement={"trainer": ["g0", "g1"], "rollout": ["g2", "g3"]}),
    "T1R3": ResourceConfig("T1R3", 1, 3, placement={"trainer": ["g0"], "rollout": ["g1", "g2", "g3"]}),
    "T2R2x": ResourceConfig("T2R2x", 2, 2, parallel=(("tp", 2),),
                            placement={"trainer": ["g0", "g1"], "rollout": ["g2", "g3"]}),
}


def _attestation(hashes=(SHA,), kind="role-transfer"):
    return attestation_from_dict({
        "runtime_fingerprint": "fp", "execution_modes": [],
        "certified_edges": [
            {"source": "T2R2", "target": "T1R3", "kind": kind, "algorithm_spec_sha256": list(hashes)},
            {"source": "T1R3", "target": "T2R2", "kind": kind, "algorithm_spec_sha256": list(hashes)},
        ],
    })


def _spec(estimator="grpo"):
    return SimpleNamespace(loss=SimpleNamespace(aggregation="default", reducer=None, custom_loss=None),
                           advantage=SimpleNamespace(estimator=estimator, whiten=False), sha256=lambda: SHA)


def _plan(src="T2R2", dst="T1R3", **kw):
    base = dict(configs=CONFIGS, attestation=_attestation(), source=src, target=dst, expected_config_epoch=3,
                spec=_spec(), args=default_args(CONFIGS[src].trainer), global_batch_size=GBS,
                micro_batch_size=MBS)
    base.update(kw)
    return plan_trainer_edge(**base)


def test_plan_role_transfer_both_directions():
    p = _plan()
    assert p.direction == "trainer_to_rollout" and p.moved_gpus == ("g1",) and p.add_engines == 1
    assert p.reshard.source["dp"] == 2 and p.reshard.target["dp"] == 1
    q = _plan("T1R3", "T2R2")
    assert q.direction == "rollout_to_trainer" and q.remove_engines == 1


@pytest.mark.parametrize(
    "kw,match",
    [
        ({"attestation": _attestation(hashes=("b" * 64,))}, "not certified"),
        ({"spec": _spec("gspo")}, "GRPO only"),
        ({"target": "T2R2x", "source": "T2R2"}, "not a certified trainer edge"),
        ({"global_batch_size": 3}, "not divisible"),
    ],
)
def test_plan_refusals(kw, match):
    with pytest.raises(TrainerEdgeRejected, match=match):
        _plan(**kw)


# ------------------------------------------------------------------ fakes


class RankGroup:
    def __init__(self, ranks):
        self.ranks = ranks
        self.disposed = False

    async def run_plugin(self, fn_path, kwargs=None):
        module, name = fn_path.rsplit(".", 1)
        fn = getattr(importlib.import_module(module), name)
        return [fn(rank, **(kwargs or {})) for rank in self.ranks]

    async def dispose(self):
        self.disposed = True


class Pool:
    def __init__(self, members):
        self._members = set(members)
        self.drained = set()
        self.log = []
        self.fail_start = False

    def members(self):
        return frozenset(self._members)

    def drain(self, members, deadline):
        self.drained |= set(members)
        return True

    def undrain(self, members):
        self.drained -= set(members)

    def remove_engines(self, members, *, epoch):
        self.log.append(("stop", sorted(members)))
        self._members -= set(members)
        return frozenset(members)

    def plan_add(self, n):
        return frozenset(f"e{len(self._members) + i}" for i in range(n))

    def add_engines(self, n, *, epoch, members):
        if self.fail_start:
            raise RuntimeError("engine start failed")
        self.log.append(("start", sorted(members)))
        self._members |= set(members)
        return frozenset(members)


class Publisher:
    def __init__(self, policy_hash="h"):
        self.hash = policy_hash

    def publish_members(self, state, members, *, epoch, token_rollout_id):
        return SimpleNamespace(members=frozenset(members),
                               manifest=SimpleNamespace(target_policy_hash=self.hash))


class Cursor:
    def data_cursor(self):
        return {"sample_offset": 4, "epoch_id": 0, "sample_group_index": 4, "sample_index": 8}


def _world_trainer(dp, tmp_path, fail=None):
    ranks = make_world(dp)
    g = torch.Generator().manual_seed(7)
    for _ in range(2):
        train_step(ranks, torch.randn(GBS, 6, generator=g))
    actor = SwappableActor(RankGroup(ranks))
    trainer = MilesTrainerGroup(args=ranks[0].args, actor_model=actor, learner_id=0, learner_generation=0,
                                parameter_layout_hash=lambda: "L", runner=LoopRunner(), spec=_spec())
    built = []

    async def rebuild(args, executor, *, old_handles, worker_manager, trainer_pg_view):
        await old_handles["actor"].dispose()
        n = int(args.actor_num_gpus_per_node)
        if fail and fail(n, len(built)):
            err = type("TrainerRebuildError", (RuntimeError,), {})("boom")
            err.stage, err.cleanup_error, err.previous_view, err.view_restored = "create_training_models", None, None, False
            built.append(("fail", n, trainer_pg_view))
            raise err
        built.append(("ok", n, trainer_pg_view))
        return RankGroup(make_world(n, seed=50 + len(built), args=default_args(n))), None

    def context_for(cut_id):
        return CutContext(root=str(tmp_path), cut_id=cut_id, backend_fingerprint="fp",
                          progress=CutProgress(2, 2 * GBS, GBS, 2, 2, "h"), algorithm=ALGO,
                          data=Cursor().data_cursor(), ledger={"carried_over": 0, "ready_unconsumed": 0},
                          outer={"settled": True})

    ops = MilesTrainerOps(
        trainer=trainer, actor=actor, rollout_executor="ex", run=LoopRunner().run, rollout=Cursor(),
        root=str(tmp_path), context_for=context_for,
        expect_for=lambda layout: RestoreExpectation(ALGO, dict(layout), "fp", 2, 2, 5),
        view_for=lambda gpus: tuple(gpus), policy_hash_fn=lambda: "h", certified_for=lambda plan: [SHA],
        worker_manager="wm", rebuild=rebuild,
    )
    return ranks, actor, ops, built


def _transition(plan, ops, pool, publisher=None):
    records = []
    return TrainerTransition(
        plan=plan, tx_id="tx-3-abc", epoch=3, trainer=ops, pool=pool, publisher=publisher or Publisher(),
        published_state=SimpleNamespace(policy_tensor_hash=lambda: "h"), published_version=2,
        record=lambda kind, **f: records.append({"kind": kind, **f}),
        fork_call=lambda op, members, fn: fn(7), fork_epoch=lambda: 7, drain_deadline=1e9,
    ), records


def test_trainer_to_rollout_transfer_reshards_and_starts_an_engine(tmp_path):
    ranks, actor, ops, built = _world_trainer(2, tmp_path)
    pool = Pool({"e0", "e1"})
    tr, records = _transition(_plan(), ops, pool)
    result = tr.run()
    assert result.phase == READY_TO_COMMIT and result.target_members == {"e0", "e1", "e2"}
    assert built == [("ok", 1, ("g0",))] and ops.trainer.actual_layout()["dp"] == 1
    before, after = gathered(ranks), gathered(actor.target.ranks)
    for n in before:
        assert torch.equal(before[n]["tensors"]["exp_avg"], after[n]["tensors"]["exp_avg"])
    kinds = [r.get("phase", r["kind"]) for r in records]
    assert kinds == ["trainer_cut", "TRANSFERRING", "trainer_rebuilt", "INITIALIZING", "add_intent", "VERIFYING"]


def test_rollout_to_trainer_transfer_drains_and_stops_first(tmp_path):
    ranks, actor, ops, built = _world_trainer(1, tmp_path)
    ops.trainer.rebind_args(ranks[0].args)
    pool = Pool({"e0", "e1", "e2"})
    tr, records = _transition(_plan("T1R3", "T2R2", args=default_args(1)), ops, pool)
    result = tr.run()
    assert result.phase == READY_TO_COMMIT and result.target_members == {"e0", "e1"}
    assert pool.log[0] == ("stop", ["e2"]) and built == [("ok", 2, ("g0", "g1"))]
    assert records[0]["phase"] == "QUIESCING"


def test_failed_target_rebuild_falls_back_to_the_old_shape(tmp_path):
    ranks, actor, ops, built = _world_trainer(2, tmp_path, fail=lambda n, i: n == 1)
    pool = Pool({"e0", "e1"})
    tr, _ = _transition(_plan(), ops, pool)
    result = tr.run()
    assert result.phase == "REBUILT_OLD" and pool.members() == {"e0", "e1"}
    assert [b[:2] for b in built] == [("fail", 1), ("ok", 2)]
    assert ops.trainer.actual_layout()["dp"] == 2


def test_engine_start_failure_rolls_the_trainer_back_from_the_same_cut(tmp_path):
    ranks, actor, ops, built = _world_trainer(2, tmp_path)
    pool = Pool({"e0", "e1"})
    pool.fail_start = True
    tr, records = _transition(_plan(), ops, pool)
    result = tr.run()
    assert result.phase == "REBUILT_OLD", result.error
    assert [b[:2] for b in built] == [("ok", 1), ("ok", 2)]
    before, after = gathered(ranks), gathered(actor.target.ranks)
    for n in before:
        assert torch.equal(before[n]["tensors"]["param"], after[n]["tensors"]["param"])
    assert any(r.get("phase") == "REBUILD_OLD" for r in records)


def test_rollback_failure_requires_recovery(tmp_path):
    ranks, actor, ops, built = _world_trainer(2, tmp_path, fail=lambda n, i: i >= 1)
    pool = Pool({"e0", "e1"})
    pool.fail_start = True
    tr, _ = _transition(_plan(), ops, pool)
    assert tr.run().phase == "RECOVERY_REQUIRED"


def test_wrong_policy_ack_rolls_back(tmp_path):
    ranks, actor, ops, built = _world_trainer(2, tmp_path)
    pool = Pool({"e0", "e1"})
    tr, _ = _transition(_plan(), ops, pool, publisher=Publisher("other"))
    result = tr.run()
    assert result.phase == "REBUILT_OLD" and "acknowledge" in result.error
    assert pool.members() == {"e0", "e1"}


def test_recovery_decision_after_controller_crash():
    base = [{"kind": "request", "tx_id": "t"}, {"kind": "phase", "tx_id": "t", "phase": "QUIESCING",
                                                 "edge": "role-transfer"}]
    assert recovery_decision(base, "T2R2", None)["action"] == "resume_old"
    cut = base + [{"kind": "trainer_cut", "tx_id": "t", "cut_id": "c"},
                  {"kind": "phase", "tx_id": "t", "phase": "TRANSFERRING", "edge": "role-transfer"}]
    assert recovery_decision(cut, "T2R2", "older")["action"] == "restore_old"
    assert recovery_decision(cut, "T1R3", "t") == {"action": "restore_target", "tx_id": "t", "cut_id": "c",
                                                   "config": "T1R3"}
    lost = base + [{"kind": "phase", "tx_id": "t", "phase": "TRANSFERRING", "edge": "role-transfer"}]
    assert recovery_decision(lost, "T2R2", None)["action"] == "recovery_required"
    done = cut + [{"kind": "phase", "tx_id": "t", "phase": "SUCCEEDED", "edge": "role-transfer"}]
    assert recovery_decision(done, "T1R3", "t")["action"] == "none"
