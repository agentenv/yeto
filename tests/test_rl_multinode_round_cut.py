"""rl-multinode-island M4 (CPU): policy weights continue across a machine replacement.

Island A trains, cuts at the round-boundary safe point (round cut in the store's
``round-cuts``, pointer in the state dir, store synced); its machines are gone (empty
local state dir); island B restores the store, loads the cut into a FRESH trainer
(different initial weights) and continues at the cut's rollout id. The Miles pieces
are E2's CPU rank fakes (real ``MilesTrainerGroup.save_cut/restore_cut`` + manifest
verification); Megatron loading of the same shards is GPU-only (M4 real run).
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from test_rl_reconfig_recovery import _ctl, _island
from test_rl_trainer_rebuild_e1 import ARGS, RankGroup

from tests.rl_cut_fakes import make_rank, params, train_step
from yeto.rl.engine.bridges import LocalOnlySync
from yeto.rl.engine.controller import STORE_MANIFEST
from yeto.rl.engine.cut import load_manifest
from yeto.rl.engine.journal import read_journal
from yeto.rl.engine.miles_adapter import LoopRunner
from yeto.rl.engine.miles_adapter.rebuild_wiring import CutSource
from yeto.rl.engine.miles_adapter.round_cut import POINTER, ROUND_CUTS, RoundCutCheckpoint, RoundCutError
from yeto.rl.engine.miles_adapter.trainer import MilesTrainerGroup
from yeto.rl.engine.miles_adapter.trainer_rebuild import SwappableActor

NEXT = 2  # A trained rollouts 0 and 1; the safe point before rollout 2 is cut


def _hash(rank):
    h = hashlib.sha256()
    for n, v in sorted(params(rank).items()):
        h.update(n.encode())
        h.update(v.detach().numpy().tobytes())
    return h.hexdigest()


class _Ledger:
    def __init__(self, recorded):
        self.recorded = recorded

    def cut_summary(self):
        return {"carried_over": 0, "ready_unconsumed": 0, "engine_buffer_length": 0}

    def batch(self, rid):
        return {"state": "outer_recorded"} if rid < self.recorded else None


class _Pool:
    def data_cursor(self):
        return {"sample_offset": 8, "epoch_id": 0, "sample_group_index": 8, "sample_index": 64}


class _Driver:
    """The driver surface round cuts use (published state, local step, export, emit)."""

    def __init__(self, rank, *, published=None, local_step=0):
        self.rank = rank
        self.published_version = published
        self.published_state = (None if published is None else SimpleNamespace(
            policy_tensor_hash=lambda: _hash(rank), policy_version=published))
        self.local_step = local_step
        self.at_safe_point = True
        self.expected_token = "tok"
        self.events = []

    def export_local(self):
        return SimpleNamespace(policy_tensor_hash=lambda: _hash(self.rank))

    def emit(self, event, **fields):
        self.events.append((event, fields))


def _trainer(rank):
    return MilesTrainerGroup(args=ARGS, actor_model=SwappableActor(RankGroup([rank])), learner_id=0,
                             learner_generation=0, parameter_layout_hash=lambda: "L", runner=LoopRunner())


def _checkpoint(ctl, driver, trainer, ledger):
    source = CutSource(driver=lambda: driver, trainer=trainer, rollout=_Pool(), ledger=ledger,
                       algorithm=SimpleNamespace(sha256=lambda: "a" * 64, to_legacy_runtime_attrs=lambda: {}),
                       backend_fingerprint="miles@0af62f4d", cut_root="", global_batch_size=ARGS.global_batch_size)
    return RoundCutCheckpoint(controller=ctl, source=source, trainer=trainer)


def _island_a(tmp_path, store):
    rank = make_rank(0)
    g = torch.Generator().manual_seed(3)
    for _ in range(NEXT):
        train_step(rank, torch.randn(3, 6, generator=g))
    ctl = _ctl(tmp_path / "a/state", {"t": 1000.0}, checkpoint_store=store)
    driver = _Driver(rank, published=NEXT, local_step=NEXT)
    trainer = _trainer(rank)
    return rank, ctl, driver, _checkpoint(ctl, driver, trainer, _Ledger(NEXT))


def test_round_cut_moves_weights_through_the_store_to_a_fresh_island(tmp_path):
    store = tmp_path / "store"
    rank_a, ctl_a, driver_a, cp_a = _island_a(tmp_path, store)
    info = cp_a.save(driver_a, rollout_id=NEXT)
    assert info["store_synced"] and info["next_rollout_id"] == NEXT and info["local_step"] == NEXT
    assert (store / ROUND_CUTS / info["cut_id"]).is_dir()  # shards in the store (all nodes see it)
    assert json.loads((store / POINTER).read_text())["cut_id"] == info["cut_id"]  # pointer synced
    assert json.loads((store / STORE_MANIFEST).read_text())["reason"] == "round_cut"
    assert cp_a.save(driver_a, rollout_id=NEXT) is None  # one cut per safe point
    ctl_a.close()

    # machine replaced: empty local state dir, fresh trainer with other weights
    rank_b = make_rank(9)
    assert _hash(rank_b) != _hash(rank_a)
    ctl_b = _ctl(tmp_path / "b/state", {"t": 2000.0}, checkpoint_store=store)
    restored = [r for r in read_journal(tmp_path / "b/state/reconfig") if r["kind"] == "checkpoint_store"]
    assert restored and restored[0]["restored_from"]["incarnation"] == ctl_a.incarnation["id"]
    assert not (tmp_path / "b/state" / ROUND_CUTS).exists()  # cuts are read in place
    driver_b = _Driver(rank_b)
    trainer_b = _trainer(rank_b)
    info_b = _checkpoint(ctl_b, driver_b, trainer_b, _Ledger(NEXT)).resume(driver_b)
    assert info_b["next_rollout_id"] == NEXT  # continues, not back at 0
    assert driver_b.local_step == NEXT
    for n, v in params(rank_a).items():
        assert torch.equal(v, params(rank_b)[n])
    manifest = load_manifest(str(store / ROUND_CUTS), info["cut_id"])
    assert manifest.progress.next_rollout_id == NEXT and manifest.outer["settled"] is True
    rec = [r for r in read_journal(tmp_path / "b/state/reconfig") if r["kind"] == "round_cut"]
    assert rec and rec[0]["action"] == "restore" and rec[0]["cut_incarnation"] == ctl_a.incarnation["id"]
    assert driver_b.events[0][0] == "rl_round_cut_restored"
    ctl_b.close()


def test_without_a_pointer_the_run_starts_at_zero_and_a_stale_ledger_is_refused(tmp_path):
    store = tmp_path / "store"
    ctl = _ctl(tmp_path / "x/state", {"t": 1.0}, checkpoint_store=store)
    rank = make_rank(9)
    assert _checkpoint(ctl, _Driver(rank), _trainer(rank), _Ledger(0)).resume(_Driver(rank)) is None
    ctl.close()
    _rank_a, ctl_a, driver_a, cp_a = _island_a(tmp_path, store)
    cp_a.save(driver_a, rollout_id=NEXT)
    ctl_a.close()
    ctl_b = _ctl(tmp_path / "b/state", {"t": 2.0}, checkpoint_store=store)
    rank_b = make_rank(9)
    cp_b = _checkpoint(ctl_b, _Driver(rank_b), _trainer(rank_b), _Ledger(NEXT - 1))
    with pytest.raises(RoundCutError, match="no outer-recorded rollout"):
        cp_b.resume(_Driver(rank_b))
    ctl_b.close()


def test_no_round_cut_sync_once_the_island_is_recovery_required(tmp_path):
    store = tmp_path / "store"
    _rank, ctl, driver, cp = _island_a(tmp_path, store)
    ctl.recovery_required = "node_lost: test"
    assert cp.save(driver, rollout_id=NEXT) is None
    assert ctl.sync_checkpoint_store("round_cut", skip_if_recovery=True) is False
    ctl.close()


def test_safe_point_hook_reports_a_refused_cut_without_stopping(tmp_path):
    sync = LocalOnlySync(4)

    class Boom:
        def save(self, driver, *, rollout_id):
            raise RuntimeError("refused")

    sync.round_cuts = Boom()
    driver = _Driver(make_rank(0))
    sync.at_safe_point(driver, rollout_id=1)
    assert driver.events[0][0] == "rl_round_cut" and driver.events[0][1]["ok"] is False


def test_local_only_sync_starts_at_the_round_cut_and_cuts_at_safe_points(tmp_path):
    driver, ctl, *_ = _island(tmp_path, rounds=5)
    calls = []

    class Stub:
        def resume(self, driver):
            return {"next_rollout_id": 3}

        def save(self, driver, *, rollout_id):
            calls.append(rollout_id)
            return {"cut_id": f"r{rollout_id}"}

    driver.sync.round_cuts = Stub()
    start = driver.sync.start(driver)
    assert start.rollout_id == 3 and start.state.policy_version == 3 and not start.finished
    driver.safe_point(3)  # the driver offers the safe point to the sync's round cut
    assert calls == [3]
    ctl.close()
