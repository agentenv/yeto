"""E1 side of E3 (4.7) wiring, CPU: startup bundle map, MilesRolloutPool.bind_members /
member_gpus, controller commit-CAS failure hint (E3 controller v3 semantics) and
compose wiring of MilesTrainerOps. Fakes model the fork RayWorkerManager API
(set_pg_view / rebind_cell / get_cell_bundles / get_pg_view @ yeto/ports); on the
real fork bind_members additionally depends on gap F-R1 (a stopped rollout cell
declared at startup outside the rollout view)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import NamedTuple

import pytest

from yeto.rl.engine.controller import IslandController, RecoveryRequired, _Tx, Plan
from yeto.rl.engine.journal import EpochConflict, read_epochs, read_journal
from yeto.rl.engine.miles_adapter.bundles import BundleMapError, StartupBundles
from yeto.rl.engine.miles_adapter.rollout import MembershipPlanError, MilesRolloutPool

from test_rl_reconfig_e1 import CONFIGS, FP, _attestation, _profile


class PGInfo(NamedTuple):
    pg: object
    pg_reordered_bundle_indices: list
    pg_reordered_gpu_ids: list


PG = object()
# logical bundles 0..3; reordered bundle index = 10 + logical, GPU id = 7 - logical
FULL = PGInfo(PG, [10, 11, 12, 13], [7, 6, 5, 4])
MAP = {"trainer": [0, 1], "rollout": [2], "standby": [3]}


def _views():
    def sl(ix):
        return PGInfo(PG, [FULL[1][i] for i in ix], [FULL[2][i] for i in ix])
    return {"actor": sl(MAP["trainer"]), "rollout": sl(MAP["rollout"]), "standby": sl(MAP["standby"])}


def test_startup_bundles_map_logical_positions_through_the_role_views():
    b = StartupBundles(pool_gpus=("g0", "g1", "g2", "g3"), views=_views(), placement_map=MAP)
    view = b.view_for(("g1", "g3"))
    assert isinstance(view, PGInfo) and view.pg is PG
    assert view.pg_reordered_bundle_indices == [11, 13] and view.pg_reordered_gpu_ids == [6, 4]
    assert b.gpus_for_bundles([12, 10]) == ("g2", "g0")
    with pytest.raises(BundleMapError):
        b.view_for(("g9",))
    with pytest.raises(BundleMapError):
        b.gpus_for_bundles([99])
    # offset layout (no map): the actor view is the whole group
    plain = StartupBundles(pool_gpus=("a", "b", "c", "d"), views={"actor": FULL}, placement_map=None)
    assert plain.view_for(("d",)).pg_reordered_bundle_indices == [13]


class _Remote:
    def __init__(self, fn):
        self.fn = fn

    def remote(self, *a, **k):
        async def call():
            return self.fn(*a, **k)
        return call()


class FakeManager:
    def __init__(self):
        self.calls = []
        self.bundles = {"c0": [12], "c1": [13], "c2": []}
        self.set_pg_view = _Remote(lambda name, info: self.calls.append(("view", name, info)))
        self.rebind_cell = _Remote(
            lambda cell, *, pg_name, pg_slot_offset: self.calls.append(
                ("rebind", cell, pg_name, pg_slot_offset)))
        self.get_cell_bundles = _Remote(lambda cell: self.bundles[cell])
        self.get_pg_view = _Remote(lambda name: _views()[name])


class FakeController:
    def __init__(self, running=("c0",)):
        self.running = set(running)

    async def get_cell_statuses(self):
        return {c: "Serving" for c in self.running}


def _pool(manager, *, running=("c0",), per=1, bundles=True):
    import yeto.rl.engine.miles_adapter.rollout as rollout_mod

    pool = MilesRolloutPool(
        inference_controller=FakeController(running), rollout_executor=None, metadata=None,
        expected_policy=lambda: (0, "h"), runner=SimpleNamespace(run=asyncio.run),
        declared_cells=("c0", "c1", "c2"), worker_manager=manager,
        bundles=(StartupBundles(pool_gpus=("g0", "g1", "g2", "g3"), views=_views(),
                                placement_map=MAP) if bundles else None),
        gpus_per_engine=per,
    )
    pool.members = lambda: frozenset(rollout_mod.member_id(c) for c in running)
    return pool


def test_bind_members_points_a_fresh_view_at_the_freed_gpus_and_rebinds_in_order():
    m = FakeManager()
    pool = _pool(m)
    view = pool.bind_members(frozenset({"engine:c2", "engine:c1"}), ("g1", "g0"))
    kinds = [c[0] for c in m.calls]
    assert kinds == ["view", "rebind", "rebind"]
    _, name, info = m.calls[0]
    assert name == view and info.pg_reordered_bundle_indices == [11, 10]
    assert m.calls[1][1:] == ("c1", view, 0) and m.calls[2][1:] == ("c2", view, 1)
    again = pool.bind_members(frozenset({"engine:c1"}), ("g1",))
    assert again != view  # a fresh view name per bind


@pytest.mark.parametrize("members, gpus, per, message", [
    ({"engine:c0"}, ("g1",), 1, "running"),
    ({"engine:c1"}, ("g0", "g1"), 1, "need 1 distinct GPUs"),
    ({"engine:c1"}, ("g0", "g0"), 2, "distinct"),
    ({"engine:c9"}, ("g0",), 1, "not declared"),
])
def test_bind_members_refusals_touch_nothing(members, gpus, per, message):
    m = FakeManager()
    with pytest.raises(MembershipPlanError, match=message):
        _pool(m, per=per).bind_members(frozenset(members), gpus)
    assert m.calls == []


def test_bind_members_needs_the_bundle_map():
    with pytest.raises(MembershipPlanError, match="mapping"):
        _pool(FakeManager(), bundles=False).bind_members(frozenset({"engine:c1"}), ("g0",))


def test_member_gpus_reads_the_fork_bindings_of_the_serving_members():
    m = FakeManager()
    pool = _pool(m, running=("c0", "c1"))
    assert pool.member_gpus() == {"engine:c0": ("g2",), "engine:c1": ("g3",)}
    assert pool.member_gpus(frozenset({"engine:c1"})) == {"engine:c1": ("g3",)}
    assert pool.members_on_gpus(("g3",)) == frozenset({"engine:c1"})
    with pytest.raises(MembershipPlanError, match="mapping"):
        _pool(m, bundles=False).member_gpus()


# ---------------------------------------------------------------- commit CAS failure (E3 v3)
def _ctl(tmp_path):
    return IslandController(state_dir=tmp_path, configs=CONFIGS, initial_config="T4R2S2",
                            attestation=_attestation(), profile=_profile(), runtime_fingerprint=FP)


def _tx():
    plan = Plan("h", "T4R2S2", "T4R4S0", "role-transfer", 0, 2, 4, 1.0, None, None)
    return _Tx("tx-1", "r1", {}, plan, 0.0)


def test_failed_commit_cas_without_a_durable_commit_hints_restore_old(tmp_path):
    ctl = _ctl(tmp_path)
    tx = _tx()
    ctl._tx = tx

    def conflict(**_):
        raise EpochConflict("someone else moved the epoch")

    ctl.journal.compare_and_swap = conflict
    with pytest.raises(RecoveryRequired, match="commit CAS failed"):
        ctl._commit(tx, "T4R4S0", {"engine:c0"}, cut_id="tx-1-cut")
    hint = [r for r in read_journal(tmp_path / "reconfig") if r["kind"] == "trainer_recovery_hint"]
    assert hint[0]["action"] == "restore_old" and hint[0]["cut_id"] == "tx-1-cut"
    assert hint[0]["config"] == "T4R2S2" and ctl.recovery_required


def test_failed_commit_cas_after_a_durable_rename_hints_restore_target(tmp_path):
    ctl = _ctl(tmp_path)
    tx = _tx()
    ctl._tx = tx
    real = ctl.journal.compare_and_swap

    def durable_then_fail(**kw):
        real(**kw)  # rename happened ...
        raise OSError("fsync of the directory failed")  # ... then the dir fsync failed

    ctl.journal.compare_and_swap = durable_then_fail
    with pytest.raises(RecoveryRequired):
        ctl._commit(tx, "T4R4S0", {"engine:c0"}, cut_id="tx-1-cut")
    assert read_epochs(tmp_path / "reconfig").last_tx_id == "tx-1"
    hint = [r for r in read_journal(tmp_path / "reconfig") if r["kind"] == "trainer_recovery_hint"]
    assert hint[0]["action"] == "restore_target" and hint[0]["config"] == "T4R4S0"


# ---------------------------------------------------------------- compose wiring
def test_trainer_edges_are_not_wired_without_pool_gpus_or_proxy(tmp_path):
    from yeto.rl.engine.miles_adapter import entry
    from yeto.rl.engine.miles_adapter.trainer_rebuild import SwappableActor

    ctl = _ctl(tmp_path)
    kw = dict(miles_args=SimpleNamespace(global_batch_size=16), launch=None, algorithm=None,
              rollout_executor=None, runner=None, base_model_revision="r", manager=FakeManager())
    driver = SimpleNamespace(rollout=None)
    elastic = SimpleNamespace(controller=ctl, ledger=None, pool_gpus=None)
    actor = SwappableActor(SimpleNamespace(run_plugin=lambda *a: None))
    assert entry._wire_trainer_edges(driver, elastic=elastic, actor_model=actor, **kw) is False
    elastic = SimpleNamespace(controller=ctl, ledger=None, pool_gpus=("g0",))
    assert entry._wire_trainer_edges(driver, elastic=elastic, actor_model=object(), **kw) is False
    assert ctl._trainer_edges is None


def test_trainer_edges_wiring_builds_miles_trainer_ops(tmp_path):
    pytest.importorskip("yeto.rl.engine.miles_adapter.trainer_resize")
    from yeto.rl.engine.miles_adapter import entry
    from yeto.rl.engine.miles_adapter.trainer_rebuild import SwappableActor

    ctl = _ctl(tmp_path)
    m = FakeManager()
    pool = _pool(m, bundles=False)
    driver = SimpleNamespace(rollout=pool, trainer=SimpleNamespace(_args="ARGS"),
                             policy_state=None)
    launch = SimpleNamespace(placement=SimpleNamespace(placement_map=MAP, gpus_per_engine=1))
    ok = entry._wire_trainer_edges(
        driver, elastic=SimpleNamespace(controller=ctl, ledger=None,
                                        pool_gpus=("g0", "g1", "g2", "g3")),
        miles_args=SimpleNamespace(global_batch_size=16, micro_batch_size=2, ref_load=None),
        launch=launch, algorithm=SimpleNamespace(sha256=lambda: "a" * 64),
        actor_model=SwappableActor(SimpleNamespace(run_plugin=lambda *a: None)),
        rollout_executor="ex", runner=SimpleNamespace(run=asyncio.run),
        base_model_revision="r", manager=m)
    assert ok
    ctx = ctl._trainer_edges()
    assert (ctx["args"], ctx["global_batch_size"], ctx["micro_batch_size"]) == ("ARGS", 16, 2)
    assert ctx["ops"].view_for(("g1",)).pg_reordered_bundle_indices == [11]
    assert pool.member_gpus() == {"engine:c0": ("g2",)}
