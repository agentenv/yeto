"""rl-infra-spec 4.4 CPU tests: the controller asks the driver, at a safe point,
to replace the trainer behind the ports (save cut -> rebuild -> restore ->
re-publish) without rebind, second initialize/after_local_train, or replaying
outer progress.

Fakes: FakeEngine stands for the trainer (a "rebuild" wipes and restores its
tensors/moments/scheduler); the Miles pieces are exercised with E2's CPU rank
fakes (tests/rl_cut_fakes.py). Protocol evidence only -- the 4.4 acceptance is
the GPU run A6b in evidence/infra-e1/plan-3.8-4.4.md.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from tests.rl_cut_fakes import GBS, make_rank, params, train_step
from yeto.rl.engine.controller import (
    CANCELLED,
    RECOVERY_REQUIRED,
    SUCCEEDED,
    CommandInbox,
    IslandController,
    RebuildRefused,
    Rejected,
)
from yeto.rl.engine.driver import DriverError
from yeto.rl.engine.journal import read_journal
from yeto.rl.engine.miles_adapter import LoopRunner
from yeto.rl.engine.miles_adapter.rebuild_wiring import make_trainer_rebuilder
from yeto.rl.engine.miles_adapter.trainer import MilesTrainerGroup
from yeto.rl.engine.miles_adapter.trainer_rebuild import SwappableActor, rebuild_same_shape

from test_rl_engine_driver import _strict_config, _strict_syncer
from test_rl_reconfig_e1 import _setup
from test_rl_reconfig_x6 import _counting, _plain_island, _run


def _fake_rebuild(engine, log, *, corrupt=False):
    """A same-shape "rebuild": the trainer loses its state, then is restored."""

    def rebuilder(driver, *, epoch, cut_id):
        saved = ({k: v.clone() for k, v in engine.tensors.items()},
                 {k: v.clone() for k, v in engine.moments.items()}, engine.scheduler_step)
        log.append(("save_cut", cut_id, epoch, driver.local_step))

        def rebuild():
            for t in list(engine.tensors.values()) + list(engine.moments.values()):
                t.zero_()  # fresh trainer
            engine.scheduler_step = 0
            tensors, moments, step = saved
            for k, v in tensors.items():
                engine.tensors[k].copy_(v + (1.0 if corrupt else 0.0))
            for k, v in moments.items():
                engine.moments[k].copy_(v)
            engine.scheduler_step = step
            log.append(("restored", cut_id))
            return SimpleNamespace(outcome="RESTORED", generation=1, attempts=[])

        result = driver.rebuild_trainer(rebuild, cut_policy_hash=driver.published_state.policy_tensor_hash())
        return {"outcome": result.outcome, "generation": result.generation}

    return rebuilder


def _island(tmp_path, *, rounds=4, rebuilder=None, **kw):
    driver, ctl, fork, pool, publisher, trained, clock = _setup(tmp_path, rounds=rounds, **kw)
    engine = driver.trainer.engine
    log = []
    ctl.trainer_rebuilder = rebuilder(engine, log) if rebuilder else _fake_rebuild(engine, log)
    return driver, ctl, engine, trained, log


def _at(driver, rollout_id, action):
    orig = driver.safe_point

    def safe_point(rid):
        if rid == rollout_id:
            action()
        return orig(rid)

    driver.safe_point = safe_point


def _events(tmp_path):
    return [json.loads(x) for x in (tmp_path / "events.jsonl").read_text().splitlines()]


def test_rebuild_at_safe_point_keeps_ports_and_outer_progress(tmp_path):
    base, *_ = _setup(tmp_path / "base")
    base_final = base.run()
    base_engine = base.trainer.engine
    driver, ctl, engine, trained, log = _island(tmp_path / "x")
    ports = (driver.rollout, driver.trainer, driver.policy_state, driver.publisher, driver.sync)
    _at(driver, 2, lambda: ctl.request_trainer_rebuild("rb", 0, 60))
    final = driver.run()
    assert ctl.status("rb")["phase"] == SUCCEEDED
    assert final.policy_tensor_hash() == base_final.policy_tensor_hash()
    assert torch.equal(engine.moments[next(iter(engine.moments))],
                       base_engine.moments[next(iter(base_engine.moments))])
    assert (driver.rollout, driver.trainer, driver.policy_state, driver.publisher,
            driver.sync) == ports  # no rebind
    # outer progress is not replayed: same rounds, same generate/train calls, one extra publish
    kinds = lambda calls: [c[0] for c in calls if c[0] in ("generate", "train", "apply")]  # noqa: E731
    assert kinds(engine.calls) == kinds(base_engine.calls)
    pubs = lambda calls: [c for c in calls if c[0] == "publish"]  # noqa: E731
    assert len(pubs(engine.calls)) == len(pubs(base_engine.calls)) + 1
    assert log == [("save_cut", log[0][1], 0, 2), ("restored", log[0][1])]
    ev = _events(tmp_path / "x")
    assert [e["event"] for e in ev].count("rl_driver_start") == 1
    assert [e["event"] for e in ev].count("rl_local_round") == 4
    rebuilt = [e for e in ev if e["event"] == "rl_trainer_rebuilt"]
    assert len(rebuilt) == 1 and rebuilt[0]["policy_version"] == 2
    assert rebuilt[0]["sync/publication_members"] == sorted(driver.rollout.members())
    phases = [r["phase"] for r in read_journal(tmp_path / "x/state/reconfig")
              if r["kind"] == "phase"]
    assert phases == ["VALIDATING", "WAIT_SAFE", "REBUILDING_TRAINER", SUCCEEDED]
    assert driver.config_epoch == 0  # no configuration change


def test_rebuild_on_a_strict_island_keeps_the_bridge(tmp_path):
    rounds = 3
    engine0 = torch.zeros(1, 2)
    del engine0

    def fleet(root, rebuild):
        from yeto.rl.engine.bridges import StrictAvgSync
        from yeto.rl.engine.fake import FakeEngine

        syncer = _strict_syncer(FakeEngine(tensors={"base_model.model.layer.lora_A.weight":
                                                    torch.zeros(1, 2)}),
                                learners=2, rounds=rounds)
        pushes = _counting(syncer)
        sync0 = lambda e: StrictAvgSync(  # noqa: E731
            _strict_config(root, e, learner_id=0, rounds=rounds, tape="b0.jsonl"),
            client_factory=lambda _b: syncer.client(0))
        d0, ctl, engine, trained, log = _island(root / "i0", rounds=rounds, sync_factory=sync0,
                                                outer="strict-avg")
        d1, trained1 = _plain_island(root, syncer, learner_id=1, rounds=rounds)
        if rebuild:
            _at(d0, 1, lambda: ctl.request_trainer_rebuild("rb", 0, 60))
        results, errors = _run([d0, d1])
        assert errors == {}
        return syncer, pushes, results, trained, trained1, ctl, d0

    b = fleet(tmp_path / "b", False)
    x = fleet(tmp_path / "x", True)
    assert x[5].status("rb")["phase"] == SUCCEEDED
    assert x[0].history == b[0].history and sorted(x[1]) == sorted(b[1])
    assert x[3] == b[3] and x[4] == b[4]
    assert x[2][0].policy_tensor_hash() == b[2][0].policy_tensor_hash()
    assert x[6].sync.bridge is not None  # the same bridge object ran every round


def test_refused_rebuild_cancels_and_training_continues(tmp_path):
    def refused(engine, log):
        def rebuilder(driver, *, epoch, cut_id):
            raise RebuildRefused("cut refused: data cursor unknown")
        return rebuilder

    driver, ctl, engine, trained, log = _island(tmp_path, rebuilder=refused)
    _at(driver, 1, lambda: ctl.request_trainer_rebuild("rb", 0, 60))
    driver.run()
    status = ctl.status("rb")
    assert status["phase"] == CANCELLED and len(trained) == 4


def test_restored_policy_mismatch_is_recovery_required_and_stops_consumption(tmp_path):
    driver, ctl, engine, trained, log = _island(
        tmp_path, rebuilder=lambda e, lg: _fake_rebuild(e, lg, corrupt=True))
    _at(driver, 1, lambda: ctl.request_trainer_rebuild("rb", 0, 60))
    with pytest.raises(DriverError, match="RECOVERY_REQUIRED"):
        driver.run()
    assert len(trained) == 1  # nothing consumed after the failed rebuild
    assert ctl.recovery_required and "trainer rebuild failed" in ctl.recovery_required
    with pytest.raises(Rejected):
        ctl.request_trainer_rebuild("again", 0, 60)


def test_restart_during_rebuild_is_recovery_required(tmp_path):
    import shutil

    driver, ctl, engine, trained, log = _island(tmp_path / "run")
    crashed = tmp_path / "crashed"

    def crash(driver_, *, epoch, cut_id):
        # the learner dies here: freeze the journal as the restart will find it
        shutil.copytree(tmp_path / "run" / "state", crashed, ignore=shutil.ignore_patterns("*.lock"))
        raise RebuildRefused("stop the test run")

    ctl.trainer_rebuilder = crash
    _at(driver, 1, lambda: ctl.request_trainer_rebuild("rb", 0, 60))
    driver.run()
    restarted = IslandController(state_dir=crashed, configs=ctl.configs, initial_config="T4R2S2",
                                 attestation=ctl.attestation, profile=ctl.profile,
                                 runtime_fingerprint=ctl.runtime_fingerprint)
    restarted.open(driver.rollout)
    # REBUILDING_TRAINER open at restart: trainer state unknown -> RECOVERY_REQUIRED (4.5)
    assert restarted.recovery_required and "restarted" in restarted.recovery_required
    with pytest.raises(Rejected, match="RECOVERY_REQUIRED"):
        restarted.request_trainer_rebuild("again", 0, 60)
    restarted.close()


def test_rebuild_request_rules(tmp_path):
    driver, ctl, *_ = _island(tmp_path)
    assert ctl.request_trainer_rebuild("rb", 0, 60)["phase"] == "VALIDATING"
    assert ctl.request_trainer_rebuild("rb", 0, 60)["phase"] == "VALIDATING"  # idempotent
    with pytest.raises(Rejected, match="another body"):
        ctl.request_trainer_rebuild("rb", 0, 61)
    with pytest.raises(Rejected, match="in progress"):
        ctl.request("up", "T4R4S0", 0, 60)
    ctl.cancel("rb")
    with pytest.raises(Rejected, match="epoch"):
        ctl.request_trainer_rebuild("rb2", 3, 60)
    ctl.trainer_rebuilder = None
    with pytest.raises(Rejected, match="no trainer rebuilder"):
        ctl.request_trainer_rebuild("rb3", 0, 60)


def test_rebuild_through_the_command_inbox(tmp_path):
    driver, ctl, engine, trained, log = _island(tmp_path, inbox=True)
    CommandInbox(tmp_path / "state" / "inbox").submit(
        "rb", "rebuild", {"kind": "trainer-rebuild", "expected_config_epoch": 0, "deadline_s": 60})
    driver.run()
    assert ctl.status("rb")["phase"] == SUCCEEDED and log


# ---------------------------------------------------------------- Miles wiring (E2 pieces)
ARGS = SimpleNamespace(actor_num_nodes=1, actor_num_gpus_per_node=1, num_steps_per_rollout=1,
                       global_batch_size=GBS, load="/ref", requested_load=None, start_rollout_id=0)


class RankGroup:
    def __init__(self, ranks):
        self.ranks, self.disposed = ranks, False

    async def run_plugin(self, fn_path, kwargs=None):
        import importlib

        module, name = fn_path.rsplit(".", 1)
        return [getattr(importlib.import_module(module), name)(r, **(kwargs or {}))
                for r in self.ranks]

    async def dispose(self):
        self.disposed = True


class _Cursor:
    def data_cursor(self):
        return {"sample_offset": 4, "epoch_id": 0, "sample_group_index": 4, "sample_index": 32}


class _Ledger:
    def cut_summary(self):
        return {"carried_over": 0, "ready_unconsumed": 0, "engine_buffer_length": 0}


class _Driver:
    """The driver surface the rebuilder uses; ``rebuild_trainer`` is the real
    contract's order (checked on the real driver above)."""

    def __init__(self, policy_hash):
        self.published_state = SimpleNamespace(policy_tensor_hash=lambda: policy_hash,
                                               policy_version=2)
        self.published_version = 2
        self.local_step = 2
        self.rounds_completed = 2
        self.expected_token = "tok"
        self.at_safe_point = True
        self.calls = []

    def rebuild_trainer(self, rebuild, *, cut_policy_hash):
        self.calls.append(cut_policy_hash)
        return rebuild()


def _trained_actor():
    rank = make_rank(0)
    g = torch.Generator().manual_seed(3)
    for _ in range(2):
        train_step(rank, torch.randn(3, 6, generator=g))
    return rank, SwappableActor(RankGroup([rank]))


def _miles_rebuilder(tmp_path, trainer, actor, fresh):
    async def fork_rebuild(args, executor, *, old_handles, worker_manager, trainer_pg_view):
        await old_handles["actor"].dispose()
        return fresh, None

    return make_trainer_rebuilder(
        trainer=trainer, rollout=_Cursor(), ledger=_Ledger(), algorithm=SimpleNamespace(
            sha256=lambda: "a" * 64, to_legacy_runtime_attrs=lambda: {}),
        backend_fingerprint="miles@0af62f4d", cut_root=str(tmp_path / "cuts"),
        global_batch_size=GBS,
        rebuild_same_shape=lambda *, restore: rebuild_same_shape(
            trainer, args=ARGS, rollout_executor="ex", actor=actor, run=LoopRunner().run,
            worker_manager="wm", rollout=_Cursor(), rebuild=fork_rebuild, restore=restore),
    )


def test_miles_rebuilder_saves_rebuilds_and_restores_the_cut(tmp_path):
    rank, actor = _trained_actor()
    trainer = MilesTrainerGroup(args=ARGS, actor_model=actor, learner_id=0, learner_generation=0,
                                parameter_layout_hash=lambda: "L", runner=LoopRunner())
    fresh = RankGroup([make_rank(9)])
    rebuilder = _miles_rebuilder(tmp_path, trainer, actor, fresh)
    driver = _Driver("h")
    out = rebuilder(driver, epoch=0, cut_id="rb-0-abc")
    assert out["outcome"] == "RESTORED" and out["generation"] == 1 and driver.calls == ["h"]
    assert actor.target is fresh
    for n, v in params(rank).items():
        assert torch.equal(v, params(fresh.ranks[0])[n])
    from yeto.rl.engine.cut import load_manifest

    manifest = load_manifest(str(tmp_path / "cuts"), "rb-0-abc")
    assert manifest.progress.local_step == 2 and manifest.progress.policy_hash == "h"
    assert manifest.outer["settled"] is True and manifest.ledger["carried_over"] == 0


def test_miles_rebuilder_refuses_before_touching_the_trainer(tmp_path):
    rank, actor = _trained_actor()
    trainer = MilesTrainerGroup(args=ARGS, actor_model=actor, learner_id=0, learner_generation=0,
                                parameter_layout_hash=lambda: "L", runner=LoopRunner())
    old = actor.target
    rebuilder = _miles_rebuilder(tmp_path, trainer, actor, RankGroup([make_rank(9)]))
    driver = _Driver("h")
    driver.local_step = 3  # disagrees with the trainer's scheduler -> save_cut refuses
    with pytest.raises(RebuildRefused, match="cut refused"):
        rebuilder(driver, epoch=0, cut_id="rb-0-abd")
    assert actor.target is old and not old.disposed and driver.calls == []


def test_entry_wires_the_rebuilder_only_for_a_swappable_actor(tmp_path):
    from yeto.rl.engine.miles_adapter import entry

    driver, ctl, *_ = _island(tmp_path)
    ctl.trainer_rebuilder = None
    elastic = SimpleNamespace(controller=ctl, ledger=None)
    kw = dict(elastic=elastic, miles_args=SimpleNamespace(global_batch_size=GBS, ref_load=None),
              algorithm=SimpleNamespace(sha256=lambda: "a" * 64), rollout_executor="ex",
              runner=LoopRunner(), base_model_revision="r")
    entry._wire_trainer_rebuild(driver, actor_model=object(), **kw)
    assert ctl.trainer_rebuilder is None
    entry._wire_trainer_rebuild(driver, actor_model=SwappableActor(RankGroup([])), **kw)
    assert callable(ctl.trainer_rebuilder)
