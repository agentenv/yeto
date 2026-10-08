"""rl-infra-spec 4.2/4.3/4.5: MilesTrainerGroup.save_cut/restore_cut and same-shape rebuild (CPU)."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch

from tests.rl_cut_fakes import GBS, make_rank, params, train_step
from yeto.rl.engine.cut import AlgorithmIdentity, CutError, CutProgress, RestoreExpectation
from yeto.rl.adapters.miles import LoopRunner
from yeto.rl.adapters.miles.trainer import CutContext, MilesTrainerGroup
from yeto.rl.adapters.miles.trainer_rebuild import (
    RecoveryRequired,
    SwappableActor,
    rebuild_preconditions,
    rebuild_same_shape,
)

ALGO = AlgorithmIdentity("a" * 64)
ARGS = SimpleNamespace(actor_num_nodes=1, actor_num_gpus_per_node=1, num_steps_per_rollout=1,
                       global_batch_size=GBS, load="/ref", requested_load=None,
                       start_rollout_id=0)
LAYOUT = {"world": 1, "tp": 1, "pp": 1, "cp": 1, "ep": 1, "dp": 1}


class RankGroup:
    """Actor group whose run_plugin runs the real plugin on in-process CPU ranks."""

    def __init__(self, ranks):
        self.ranks = ranks
        self.disposed = False

    async def run_plugin(self, fn_path, kwargs=None):
        module, name = fn_path.rsplit(".", 1)
        fn = getattr(importlib.import_module(module), name)
        return [fn(rank, **(kwargs or {})) for rank in self.ranks]

    async def dispose(self):
        self.disposed = True


def _trainer(group):
    return MilesTrainerGroup(args=ARGS, actor_model=group, learner_id=0, learner_generation=0,
                             parameter_layout_hash=lambda: "L", runner=LoopRunner())


def _context(tmp_path, cut_id="cut-a", **overrides):
    base = dict(
        root=str(tmp_path), cut_id=cut_id, backend_fingerprint="miles@0af62f4d",
        progress=CutProgress(local_step=2, scheduler_samples=2 * GBS, global_batch_size=GBS,
                             next_rollout_id=2, policy_version=2, policy_hash="h"),
        algorithm=ALGO,
        data={"sample_offset": 4, "epoch_id": 0, "sample_group_index": 4, "sample_index": 32},
        ledger={"carried_over": 0, "ready_unconsumed": 0},
        outer={"settled": True},
    )
    base.update(overrides)
    return CutContext(**base)


def _expect(**o):
    base = dict(algorithm=ALGO, layout=LAYOUT, backend_fingerprint="miles@0af62f4d", local_step=2,
                policy_version=2, epoch=1)
    base.update(o)
    return RestoreExpectation(**base)


def _trained_rank():
    rank = make_rank(0)
    g = torch.Generator().manual_seed(3)
    for _ in range(2):
        train_step(rank, torch.randn(3, 6, generator=g))
    return rank


def test_save_and_restore_cut_through_the_port(tmp_path):
    rank = _trained_rank()
    trainer = _trainer(RankGroup([rank]))
    assert trainer.save_cut(epoch=1, context=_context(tmp_path)) == "cut-a"
    fresh = make_rank(9)
    trainer2 = _trainer(RankGroup([fresh]))
    manifest = trainer2.restore_cut("cut-a", epoch=1, root=str(tmp_path), expect=_expect())
    assert manifest.progress.local_step == 2
    for n, v in params(rank).items():
        assert torch.equal(v, params(fresh)[n])


def test_step_mismatch_between_driver_and_scheduler_is_refused(tmp_path):
    rank = _trained_rank()
    ctx = _context(tmp_path, progress=CutProgress(3, 3 * GBS, GBS, 3, 3, "h"))
    with pytest.raises(CutError, match="scheduler_samples|scheduler at"):
        _trainer(RankGroup([rank])).save_cut(epoch=1, context=ctx)


def test_missing_data_cursor_is_refused(tmp_path):
    with pytest.raises(CutError, match="cursor"):
        _trainer(RankGroup([_trained_rank()])).save_cut(epoch=1, context=_context(tmp_path, data={}))


def test_restore_checks_every_rank_digest(tmp_path, monkeypatch):
    rank = _trained_rank()
    _trainer(RankGroup([rank])).save_cut(epoch=1, context=_context(tmp_path))
    from yeto.rl.adapters.miles import cut_plugin

    real = cut_plugin._restore
    monkeypatch.setattr(cut_plugin, "_restore", lambda *a, **k: {**real(*a, **k), "state_digest": "x"})
    with pytest.raises(CutError, match="state_digest"):
        _trainer(RankGroup([make_rank(1)])).restore_cut("cut-a", epoch=1, root=str(tmp_path), expect=_expect())


def test_shard_count_must_match_workers(tmp_path):
    with pytest.raises(Exception, match="cut shards"):
        _trainer(RankGroup([])).save_cut(epoch=1, context=_context(tmp_path))


# ---------------------------------------------------------------- rebuild




def _rebuild_error(stage, *, cleanup=None, previous_view=None, restored=False):
    err = type("TrainerRebuildError", (RuntimeError,), {})(f"failed at {stage}")
    err.stage, err.cleanup_error, err.previous_view, err.view_restored = stage, cleanup, previous_view, restored
    return err


class Cursor:
    def __init__(self):
        self.value = {"sample_offset": 4, "epoch_id": 0, "sample_group_index": 4, "sample_index": 32}

    def data_cursor(self):
        return dict(self.value)


def _setup(tmp_path):
    rank = _trained_rank()
    group = RankGroup([rank])
    actor = SwappableActor(group)
    trainer = _trainer(actor)
    trainer.save_cut(epoch=1, context=_context(tmp_path))
    return rank, group, actor, trainer


def test_same_shape_rebuild_swaps_the_handle_and_restores(tmp_path):
    rank, old, actor, trainer = _setup(tmp_path)
    fresh = RankGroup([make_rank(4)])
    calls = []

    async def rebuild(args, executor, *, old_handles, worker_manager, trainer_pg_view):
        calls.append((old_handles, trainer_pg_view))
        await old_handles["actor"].dispose()
        return fresh, None

    result = rebuild_same_shape(
        trainer, args=ARGS, rollout_executor="ex", actor=actor, run=LoopRunner().run, worker_manager="wm", rollout=Cursor(),
        rebuild=rebuild,
        restore=lambda: trainer.restore_cut("cut-a", epoch=1, root=str(tmp_path), expect=_expect()),
    )
    assert result.outcome == "RESTORED" and result.generation == 1
    assert old.disposed and actor.target is fresh and calls[0][0] == {"actor": old}
    for n, v in params(rank).items():
        assert torch.equal(v, params(fresh.ranks[0])[n])


def test_failed_rebuild_is_rebuilt_old_from_the_same_cut(tmp_path):
    rank, old, actor, trainer = _setup(tmp_path)
    fresh = RankGroup([make_rank(4)])
    views = []

    async def rebuild(args, executor, *, old_handles, worker_manager, trainer_pg_view):
        views.append(trainer_pg_view)
        if len(views) == 1:
            raise _rebuild_error("create_training_models", previous_view="V0")
        return fresh, None

    result = rebuild_same_shape(
        trainer, args=ARGS, rollout_executor="ex", actor=actor, run=LoopRunner().run, worker_manager="wm", rollout=Cursor(),
        rebuild=rebuild,
        restore=lambda: trainer.restore_cut("cut-a", epoch=1, root=str(tmp_path), expect=_expect()),
    )
    assert result.outcome == "REBUILD_OLD" and views == [None, "V0"]
    assert [a["stage"] for a in result.attempts] == ["create_training_models", "done"]


@pytest.mark.parametrize("failure", ["cleanup", "twice", "foreign"])
def test_unrecoverable_rebuild_requires_recovery(tmp_path, failure):
    _, _, actor, trainer = _setup(tmp_path)

    async def rebuild(args, executor, *, old_handles, worker_manager, trainer_pg_view):
        if failure == "cleanup":
            raise _rebuild_error("start_pools", cleanup=RuntimeError("stop failed"))
        if failure == "foreign":
            raise ValueError("boom")
        raise _rebuild_error("start_pools")

    with pytest.raises(RecoveryRequired) as info:
        rebuild_same_shape(trainer, args=ARGS, rollout_executor="ex", actor=actor, run=LoopRunner().run,
                           worker_manager="wm", rollout=Cursor(), rebuild=rebuild, restore=lambda: None)
    assert info.value.attempts


def test_restore_failure_after_rebuild_requires_recovery(tmp_path):
    _, _, actor, trainer = _setup(tmp_path)
    (tmp_path / "cut-a" / "trainer_tp0_pp0_dp0.pt").write_bytes(b"corrupt")

    async def rebuild(args, executor, **kw):
        return RankGroup([make_rank(4)]), None

    with pytest.raises(RecoveryRequired, match="unusable after rebuild"):
        rebuild_same_shape(
            trainer, args=ARGS, rollout_executor="ex", actor=actor, run=LoopRunner().run, worker_manager="wm", rollout=Cursor(),
            rebuild=rebuild,
            restore=lambda: trainer.restore_cut("cut-a", epoch=1, root=str(tmp_path), expect=_expect()),
        )


def test_rebuild_preconditions():
    # bridge mode: args.load is --ref-load and start_rollout_id is 0 -- not a refusal
    assert rebuild_preconditions(ARGS) == []
    bad = SimpleNamespace(requested_load="/ckpt", use_fault_tolerance=True, indep_dp=False,
                          trainer_controller_addrs=["x"])
    problems = " ".join(rebuild_preconditions(bad))
    for word in ("--load was requested", "fault-tolerance", "independently"):
        assert word in problems


def test_data_cursor_change_across_rebuild_requires_recovery(tmp_path):
    _, _, actor, trainer = _setup(tmp_path)
    cursor = Cursor()

    async def rebuild(args, executor, **kw):
        cursor.value = {**cursor.value, "sample_offset": 0}  # e.g. rollout_executor.load rewound it
        return RankGroup([make_rank(4)]), None

    with pytest.raises(RecoveryRequired, match="data cursor changed"):
        rebuild_same_shape(trainer, args=ARGS, rollout_executor="ex", actor=actor, run=LoopRunner().run,
                           worker_manager="wm", rollout=cursor, rebuild=rebuild,
                           restore=lambda: pytest.fail("restore must not run"))


def test_any_failure_after_swap_requires_recovery(tmp_path):
    _, _, actor, trainer = _setup(tmp_path)

    async def rebuild(args, executor, **kw):
        return RankGroup([make_rank(4)]), None

    def restore():
        raise KeyError("unexpected")

    with pytest.raises(RecoveryRequired):
        rebuild_same_shape(trainer, args=ARGS, rollout_executor="ex", actor=actor, run=LoopRunner().run,
                           worker_manager="wm", rollout=Cursor(), rebuild=rebuild, restore=restore)


def test_layout_is_read_back_from_the_ranks(tmp_path):
    _, _, actor, trainer = _setup(tmp_path)
    other = make_rank(4, coord={"global_rank": 0, "tp": 0, "pp": 0, "dp": 0, "dp_size": 1, "cp_size": 1,
                                "ep_size": 1, "tp_size": 2, "pp_size": 1})

    async def rebuild(args, executor, **kw):
        return RankGroup([other]), None

    with pytest.raises(RecoveryRequired, match="layout changed"):
        rebuild_same_shape(trainer, args=ARGS, rollout_executor="ex", actor=actor, run=LoopRunner().run,
                           worker_manager="wm", rollout=Cursor(), rebuild=rebuild, restore=lambda: None)


def test_restore_into_live_trainer_is_refused(tmp_path):
    rank, _, _, trainer = _setup(tmp_path)  # the same (trained) trainer: scheduler not at 0
    with pytest.raises(Exception, match="freshly built"):
        trainer.restore_cut("cut-a", epoch=1, root=str(tmp_path), expect=_expect())


def test_context_is_checked_before_any_rank_writes(tmp_path):
    rank = _trained_rank()
    with pytest.raises(CutError, match="cursor"):
        _trainer(RankGroup([rank])).save_cut(epoch=1, context=_context(tmp_path, data={}))
    assert not (tmp_path / "cut-a").exists()


def test_missing_optimizer_coverage_is_refused(tmp_path, monkeypatch):
    from yeto.rl.adapters.miles import cut_plugin

    real = cut_plugin._save
    monkeypatch.setattr(cut_plugin, "_save", lambda *a, **k: {**real(*a, **k), "optimizer_names": ["lora_A"]})
    with pytest.raises(CutError, match="no optimizer state"):
        _trainer(RankGroup([_trained_rank()])).save_cut(epoch=1, context=_context(tmp_path))


def test_no_shared_filesystem_refuses_distopt_dp2(tmp_path):
    ranks = [make_rank(0, coord={"global_rank": i, "tp": 0, "pp": 0, "dp": i, "dp_size": 2, "cp_size": 1,
                                 "ep_size": 1, "tp_size": 1, "pp_size": 1}) for i in range(2)]
    for r in ranks:
        r.args = SimpleNamespace(fp16=False, bf16=True, global_batch_size=GBS, use_distributed_optimizer=True)
    args = SimpleNamespace(**{**ARGS.__dict__, "actor_num_gpus_per_node": 2, "use_distributed_optimizer": True})
    t = MilesTrainerGroup(args=args, actor_model=RankGroup(ranks), learner_id=0, learner_generation=0,
                          parameter_layout_hash=lambda: "L", runner=LoopRunner())
    t.save_cut(epoch=1, context=_context(tmp_path, progress=CutProgress(0, 0, GBS, 0, 0, "h")))
    with pytest.raises(CutError, match="shared cut filesystem"):
        t.restore_cut("cut-a", epoch=1, root=str(tmp_path), shared_filesystem=False,
                      expect=_expect(layout={**LAYOUT, "world": 2, "dp": 2}, local_step=0, policy_version=0))


def test_swappable_actor_dispose_resolves_current_target():
    import asyncio

    a, b = RankGroup([]), RankGroup([])
    proxy = SwappableActor(a)
    bound_at_add = proxy.dispose  # what Miles Disposer.add stores
    proxy.swap(b)
    asyncio.run(bound_at_add())
    assert b.disposed and not a.disposed


def test_swappable_actor_forwards():
    a, b = RankGroup([]), RankGroup([])
    proxy = SwappableActor(a)
    assert proxy.ranks is a.ranks
    proxy.swap(b)
    assert proxy.ranks is b.ranks and proxy.generation == 1
    with pytest.raises(RuntimeError):
        proxy.swap(object())


def test_digest_mismatch_names_the_differing_components(tmp_path, monkeypatch):
    rank = _trained_rank()
    _trainer(RankGroup([rank])).save_cut(epoch=1, context=_context(tmp_path))
    from yeto.rl.adapters.miles import cut_plugin

    real = cut_plugin._snapshot
    calls = []

    def skewed(actor, backend, named):
        snap = real(actor, backend, named)
        calls.append(1)
        snap["scheduler"] = {**snap["scheduler"], "lr0": 999}
        return snap

    monkeypatch.setattr(cut_plugin, "_snapshot", skewed)
    with pytest.raises(CutError, match="scheduler/lr0"):
        _trainer(RankGroup([make_rank(1)])).restore_cut("cut-a", epoch=1, root=str(tmp_path), expect=_expect())


def test_rank_diff_reports_leaf_values(tmp_path, monkeypatch):
    rank = _trained_rank()
    _trainer(RankGroup([rank])).save_cut(epoch=1, context=_context(tmp_path))
    from yeto.rl.adapters.miles import cut_plugin

    fresh = make_rank(1)
    real_load = fresh._yeto_cut_backend.load_optimizer

    def lossy(optimizer, named, merged):  # e.g. a loader that drops exp_avg_sq precision
        real_load(optimizer, named, merged)
        for p in optimizer.state:
            optimizer.state[p]["exp_avg_sq"].mul_(1.0001)

    fresh._yeto_cut_backend.load_optimizer = lossy
    with pytest.raises(CutError, match="value:exp_avg_sq"):
        _trainer(RankGroup([fresh])).restore_cut("cut-a", epoch=1, root=str(tmp_path), expect=_expect())


@pytest.mark.parametrize("where", ["load", "after"])
def test_optimizer_diff_splits_load_from_later_changes(tmp_path, where, monkeypatch):
    from yeto.rl.adapters.miles.cut_plugin import RESTORE_DIAGNOSTICS_ENV

    monkeypatch.setenv(RESTORE_DIAGNOSTICS_ENV, "1")
    rank = _trained_rank()
    _trainer(RankGroup([rank])).save_cut(epoch=1, context=_context(tmp_path))
    fresh = make_rank(1)
    backend = fresh._yeto_cut_backend
    real_load, real_export = backend.load_optimizer, backend.export_optimizer
    exports = []

    def load(optimizer, named, merged):
        real_load(optimizer, named, merged)
        if where == "load":
            for p in optimizer.state:
                optimizer.state[p]["exp_avg"].mul_(2)

    def export(optimizer, named):
        exports.append(1)
        if where == "after" and len(exports) == 2:  # the final re-export sees a later change
            for p in optimizer.state:
                optimizer.state[p]["exp_avg_sq"].mul_(3)
        return real_export(optimizer, named)

    backend.load_optimizer, backend.export_optimizer = load, export
    with pytest.raises(CutError) as info:
        _trainer(RankGroup([fresh])).restore_cut("cut-a", epoch=1, root=str(tmp_path), expect=_expect())
    msg = str(info.value)
    cut_to_load, load_to_re = msg.split("cut->after_load ")[1].split("; after_load->reexport ")
    if where == "load":
        assert "'state:exp_avg': {'differ': 2" in cut_to_load
    else:
        assert "'state:exp_avg_sq': {'differ': 2" in load_to_re.split("; rank diff")[0]


@pytest.mark.parametrize("side_effect_free", [False, True])
def test_reading_a_fresh_optimizer_must_not_break_the_restore(tmp_path, side_effect_free):
    """GPU C1 diagnostic 2: a state read before the restore created empty entries, the
    fork-M5 loader skipped its init and dropped exp_avg/exp_avg_sq. Unfixed: restore_cut
    fails closed naming them; with side-effect-free reads: bitwise restore."""
    from tests.rl_cut_fakes import LazyStateDistOptBackend

    rank = _trained_rank()
    _trainer(RankGroup([rank])).save_cut(epoch=1, context=_context(tmp_path))
    fresh = make_rank(1)
    fresh._yeto_cut_backend = LazyStateDistOptBackend(side_effect_free=side_effect_free)
    group = RankGroup([fresh])
    from yeto.rl.adapters.miles.cut_plugin import state_summary

    state_summary(fresh)  # e.g. the harness' "state unchanged" read on the fresh trainer
    trainer = _trainer(group)
    if side_effect_free:
        trainer.restore_cut("cut-a", epoch=1, root=str(tmp_path), expect=_expect())
        for p_saved, p_now in zip(rank.optimizer.param_groups[0]["params"], fresh.optimizer.param_groups[0]["params"]):
            assert torch.equal(rank.optimizer.state[p_saved]["exp_avg"], fresh.optimizer.state[p_now]["exp_avg"])
    else:
        with pytest.raises(CutError, match="missing:exp_avg"):
            trainer.restore_cut("cut-a", epoch=1, root=str(tmp_path), expect=_expect())


def test_after_load_diagnostics_are_opt_in(tmp_path, monkeypatch):
    from yeto.rl.adapters.miles.cut_plugin import RESTORE_DIAGNOSTICS_ENV

    monkeypatch.delenv(RESTORE_DIAGNOSTICS_ENV, raising=False)
    rank = _trained_rank()
    _trainer(RankGroup([rank])).save_cut(epoch=1, context=_context(tmp_path))
    fresh = make_rank(1)
    backend = fresh._yeto_cut_backend
    exports, real_load, real_export = [], backend.load_optimizer, backend.export_optimizer

    def load(optimizer, named, merged):
        real_load(optimizer, named, merged)
        for p in optimizer.state:
            optimizer.state[p]["exp_avg"].mul_(2)

    backend.load_optimizer = load
    backend.export_optimizer = lambda o, n: (exports.append(1), real_export(o, n))[1]
    with pytest.raises(CutError) as info:
        _trainer(RankGroup([fresh])).restore_cut("cut-a", epoch=1, root=str(tmp_path), expect=_expect())
    msg = str(info.value)
    assert "cut->after_load None" in msg and "'state:exp_avg': {'differ': 2" in msg.split("cut->reexport ")[1]
    assert len(exports) == 1  # only the final re-export, no extra after-load export


def test_failure_before_any_write_is_told_apart_from_a_refusal(tmp_path):
    rank = _trained_rank()
    _trainer(RankGroup([rank])).save_cut(epoch=1, context=_context(tmp_path))
    fresh = make_rank(1)
    fresh._yeto_cut_backend.check_optimizer = lambda *a: (_ for _ in ()).throw(KeyError("no such param"))
    with pytest.raises(CutError, match=r"\[failed_before_write\] KeyError"):
        _trainer(RankGroup([fresh])).restore_cut("cut-a", epoch=1, root=str(tmp_path), expect=_expect())
    with pytest.raises(CutError, match=r"\[refused\] .*freshly built"):
        _trainer(RankGroup([rank])).restore_cut("cut-a", epoch=1, root=str(tmp_path), expect=_expect())


def test_unknown_live_cursor_fails_closed(tmp_path):
    """INFRA-E1 f707dc3: data_cursor() is None when the live read fails -> no comparison."""
    _, _, actor, trainer = _setup(tmp_path)

    class Unknown:
        def data_cursor(self):
            return None

    async def rebuild(args, executor, **kw):
        raise AssertionError("nothing may be disposed when the cursor is unknown")

    with pytest.raises(RuntimeError, match="cursor unknown before the rebuild"):
        rebuild_same_shape(trainer, args=ARGS, rollout_executor="ex", actor=actor, run=LoopRunner().run,
                           worker_manager="wm", rollout=Unknown(), rebuild=rebuild, restore=lambda: None)


def test_pp2_distributed_optimizer_cut_is_no_longer_refused(tmp_path):
    """M1/M4 layout (PP2 + --use-distributed-optimizer, DP1): round cuts go through save_cut."""
    ranks = [make_rank(i, coord={"global_rank": i, "tp": 0, "pp": i, "dp": 0, "dp_size": 1, "cp_size": 1,
                                 "ep_size": 1, "tp_size": 1, "pp_size": 2}) for i in range(2)]
    flags = {"use_distributed_optimizer": True, "pipeline_model_parallel_size": 2}
    for r in ranks:
        r.args = SimpleNamespace(fp16=False, bf16=True, global_batch_size=GBS, **flags)
    args = SimpleNamespace(**{**ARGS.__dict__, "actor_num_gpus_per_node": 2, **flags})
    t = MilesTrainerGroup(args=args, actor_model=RankGroup(ranks), learner_id=0, learner_generation=0,
                          parameter_layout_hash=lambda: "L", runner=LoopRunner())
    assert t.save_cut(epoch=1, context=_context(tmp_path, progress=CutProgress(0, 0, GBS, 0, 0, "h"))) == "cut-a"
    fresh = [make_rank(5 + i, coord=r._yeto_cut_backend.coord()) for i, r in enumerate(ranks)]
    for r in fresh:
        r.args = ranks[0].args
    t2 = MilesTrainerGroup(args=args, actor_model=RankGroup(fresh), learner_id=0, learner_generation=0,
                           parameter_layout_hash=lambda: "L", runner=LoopRunner())
    t2.restore_cut("cut-a", epoch=1, root=str(tmp_path),
                   expect=_expect(layout={**LAYOUT, "world": 2, "pp": 2}, local_step=0, policy_version=0))
    for a, b in zip(ranks, fresh, strict=True):
        for n, v in params(a).items():
            assert torch.equal(v, params(b)[n])
