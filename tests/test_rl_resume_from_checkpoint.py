"""rl-resume-from-checkpoint (S17 C4), CPU only: store + LATEST + hashes, resume checks,
round cuts without --rl-elastic (save -> machine gone -> resume, real MilesTrainerGroup
cut/restore on E2's CPU rank fakes), launcher/Modal wiring, tape continuity, dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_rl_multinode_round_cut import _Driver, _hash, _Pool, _trainer
from test_rl_reconfig_recovery import _ctl
from test_rl_trainer_rebuild_e1 import ARGS

from tests.rl_cut_fakes import make_rank, params, train_step
from yeto.cloud import modal_ckpt_store as mcs
from yeto.rl.adapters.miles.rebuild_wiring import CutSource
from yeto.rl.adapters.miles.round_cut import POINTER, RoundCutCheckpoint, RoundCutError
from yeto.rl.engine.cut import CutError
from yeto.rl.engine import resume as rs
from yeto.rl.engine.bridges import LocalOnlySync
from yeto.rl.engine.controller import STORE_MANIFEST

FP = {"lr": 5e-6, "seed": 17, "lora_rank": 16, "prompt_data": "gsm8k"}


class _Ledger:
    def __init__(self, recorded):
        self.recorded = recorded
        self._consumed_groups = {f"g{i}": i // 4 for i in range(4 * recorded)}

    def cut_summary(self):
        return {"carried_over": 0, "ready_unconsumed": 0, "engine_buffer_length": 0}

    def batch(self, rid):
        return {"state": "outer_recorded"} if rid < self.recorded else None


class _Vol:
    def __init__(self):
        self.commits = 0
        self.reloads = 0

    def commit(self):
        self.commits += 1

    def reload(self):
        self.reloads += 1


def _store(path, vol=None):
    if vol is None:
        return rs.CheckpointStore(path)
    return mcs.ModalVolumeStore(path, "yeto-ckpt", volume_factory=lambda name: vol)


def _rctl(tmp_path, name, store, keep=2):
    return rs.ResumeController(state_dir=tmp_path / name / "state", store=store, keep=keep)


def _cp(ctl, driver, trainer, ledger, *, backend_fingerprint="miles@0af62f4d", **kw):
    source = CutSource(driver=lambda: driver, trainer=trainer, rollout=_Pool(), ledger=ledger,
                       algorithm=SimpleNamespace(sha256=lambda: "a" * 64, to_legacy_runtime_attrs=lambda: {}),
                       backend_fingerprint=backend_fingerprint, cut_root="", global_batch_size=ARGS.global_batch_size)
    kw.setdefault("fingerprint", dict(FP))
    kw.setdefault("next_lr", lambda d: getattr(d, "last_applied_lrs", None))
    return RoundCutCheckpoint(controller=ctl, source=source, trainer=trainer, **kw)


def _cut(cp, driver, **kw):
    """Save a cut and complete it: a periodic cut (also the store's first, E1) commits in
    the background; the outcome of that commit is folded into the returned pointer."""
    info = cp.save(driver, **kw)
    if info is None or info["store_commit"] != "deferred":
        return info
    done = cp.join_commit(driver)
    assert done["ok"] is True, done
    return {**info, "store_synced": True, "cut_bytes": done["cut_bytes"]}


def _trained_island(tmp_path, store, rounds, *, name="a", seed=3):
    rank = make_rank(0)
    g = torch.Generator().manual_seed(seed)
    for _ in range(rounds):
        train_step(rank, torch.randn(3, 6, generator=g))
    ctl = _rctl(tmp_path, name, store)
    driver = _Driver(rank, published=rounds, local_step=rounds)
    driver.last_applied_lrs = [5e-6]
    return rank, ctl, driver, _cp(ctl, driver, _trainer(rank), _Ledger(rounds))


# ---------------------------------------------------------------- store / LATEST / hashes
def test_snapshot_latest_and_verified_restore(tmp_path):
    state = tmp_path / "s"
    (state / "ledger").mkdir(parents=True)
    (state / "ledger" / "journal.jsonl").write_text('{"kind":"prepared"}\n')
    (state / POINTER).write_text('{"cut_id": "r1"}')
    (state / "journal.lock").write_text("")
    vol = _Vol()
    store = _store(tmp_path / "store", vol)
    rec = rs.write_snapshot(store, state, {"reason": "round_cut", "cut_id": "r1"})
    assert rec["seq"] == 0 and vol.commits == 1 and rec["bytes"] > 0
    latest = rs.read_latest(store.root)
    manifest = rs.verify_snapshot(store.root, latest)
    assert set(manifest["files"]) == {"ledger/journal.jsonl", POINTER}  # lock never copied
    out = rs.restore_snapshot(store.root, tmp_path / "fresh", latest)
    assert out["files"] == 2 and (tmp_path / "fresh" / POINTER).read_text() == '{"cut_id": "r1"}'


def test_tampered_or_torn_store_is_refused_and_previous_latest_survives(tmp_path):
    state = tmp_path / "s"
    state.mkdir()
    (state / POINTER).write_text("v0")
    store = _store(tmp_path / "store")
    rs.write_snapshot(store, state, {"reason": "x"})
    good = rs.read_latest(store.root)
    # a torn second sync: files written into snapshot 1, LATEST never switched
    (store.root / "state" / "00000001").mkdir(parents=True)
    (store.root / "state" / "00000001" / POINTER).write_text("half")
    assert rs.read_latest(store.root) == good
    rs.restore_snapshot(store.root, tmp_path / "r", good)
    assert (tmp_path / "r" / POINTER).read_text() == "v0"
    # the next sync reuses seq 1 cleanly
    (state / POINTER).write_text("v1")
    assert rs.write_snapshot(store, state, {"reason": "x"})["seq"] == 1
    # bit flip in the stored copy -> refused
    (store.root / "state" / "00000001" / POINTER).write_text("v2")
    with pytest.raises(rs.StoreIntegrityError, match="sha256 mismatch|bytes"):
        rs.verify_snapshot(store.root, rs.read_latest(store.root))
    # edited manifest -> refused
    m = store.root / "state" / "00000001" / rs.STATE_MANIFEST
    m.write_text(m.read_text().replace('"x"', '"y"'))
    with pytest.raises(rs.StoreIntegrityError, match="manifest hash"):
        rs.verify_snapshot(store.root, rs.read_latest(store.root))


def test_copy_verified_detects_a_bad_copy(tmp_path, monkeypatch):
    src = tmp_path / "a"
    src.write_bytes(b"abc")
    real = rs.shutil.copyfile
    monkeypatch.setattr(rs.shutil, "copyfile", lambda a, b: (real(a, b), open(b, "ab").write(b"!")))
    with pytest.raises(rs.StoreIntegrityError):
        rs.copy_verified(src, tmp_path / "b")


def test_store_for_picks_the_backend_and_parses_modal_uris():
    assert isinstance(rs.store_for("/x", environ={rs.STORE_IMPL_ENV: mcs.STORE_IMPL, mcs.MODAL_VOLUME_ENV: "v"}),
                      mcs.ModalVolumeStore)
    with pytest.raises(ValueError, match=mcs.MODAL_VOLUME_ENV):
        rs.store_for("/x", environ={rs.STORE_IMPL_ENV: mcs.STORE_IMPL})
    assert type(rs.store_for("/mnt/share", environ={})) is rs.CheckpointStore
    assert isinstance(rs.store_for("~/yeto-checkpoint-store/p", environ={}), rs.BucketMountStore)
    assert mcs.parse_modal_volume_uri("modal-volume://yeto-ckpt/run1/a") == ("yeto-ckpt", "run1/a")
    for bad in ("s3://b/x", "modal-volume://", "modal-volume://a b", "modal-volume://v/../x"):
        with pytest.raises(ValueError):
            mcs.parse_modal_volume_uri(bad)


def test_checks_fingerprint_lr_cadence():
    assert rs.check_fingerprint(FP, dict(FP), allow_change=False) == {}
    changed = {**FP, "lr": 1e-5}
    with pytest.raises(rs.ResumeRefused, match="lr"):
        rs.check_fingerprint(FP, changed, allow_change=False)
    assert rs.check_fingerprint(FP, changed, allow_change=True) == {"lr": {"cut": 5e-6, "now": 1e-5}}
    assert rs.check_lr_continues([5e-6], [5e-6]) is None
    assert "expected" in rs.check_lr_continues([5e-6], [5.000001e-6])
    assert rs.check_lr_continues(None, [1.0]) is None
    assert rs.check_lr_continues([1.0], None) is not None
    assert [r for r in range(8) if rs.should_cut(r, every=2)] == [2, 4, 6]
    assert rs.should_cut(3, every=2, final=True) and not rs.should_cut(0, every=1, final=True)
    with pytest.raises(ValueError):
        rs.should_cut(1, every=0)


# ---------------------------------------------------------------- ResumeController
def test_resume_controller_restores_latest_and_counts_incarnations(tmp_path):
    store = _store(tmp_path / "store")
    c0 = _rctl(tmp_path, "a", store)
    assert c0.incarnation["index"] == 0 and c0.restored is None
    (c0.state_dir / POINTER).write_text(json.dumps({"cut_id": "r000002-x"}))
    assert c0.sync_checkpoint_store("round_cut")
    c1 = _rctl(tmp_path, "b", store)  # a fresh machine
    assert c1.incarnation["index"] == 1 and c1.restored["latest"]["seq"] == 0
    assert json.loads((c1.state_dir / POINTER).read_text())["cut_id"] == "r000002-x"
    # an in-place restart: the local state dir is moved aside, the store wins
    (c1.state_dir / "stray").write_text("newer-than-store")
    c2 = rs.ResumeController(state_dir=c1.state_dir, store=store)
    assert c2.restored["moved_aside"] and not (c2.state_dir / "stray").exists()


def test_cuts_without_latest_are_refused(tmp_path):
    store = _store(tmp_path / "store")
    (store.root / rs.ROUND_CUTS / "r000002-x").mkdir(parents=True)
    with pytest.raises(rs.StoreIntegrityError, match="no LATEST"):
        _rctl(tmp_path, "a", store)


@pytest.mark.parametrize("layout", ["unmarked", "one_marked_one_not", "stray_file", "mark_of_another_cut"])
def test_cuts_without_latest_are_refused_unless_all_are_a_marked_first_cut(tmp_path, layout):
    """E1: only a store whose every cut carries the first-cut mark (never committed)
    is started over; any other store with cuts and no LATEST is refused, untouched."""
    store = _store(tmp_path / "store")
    cuts = store.root / rs.ROUND_CUTS
    (cuts / "r000004-x").mkdir(parents=True)
    if layout == "one_marked_one_not":
        rs.mark_first_cut(store.root, "r000004-x")
        (cuts / "r000008-x").mkdir()
    elif layout == "stray_file":
        rs.mark_first_cut(store.root, "r000004-x")
        (cuts / "notes.txt").write_text("?")
    elif layout == "mark_of_another_cut":
        rs.mark_first_cut(store.root, "r000008-x")
    before = sorted(p.name for p in cuts.iterdir())
    with pytest.raises(rs.StoreIntegrityError, match="no LATEST"):
        _rctl(tmp_path, "a", store)
    assert sorted(p.name for p in cuts.iterdir()) == before


def _first_cut_in_flight(tmp_path, store, every=4):
    """An island at rollout ``every`` whose first (periodic) cut was captured, its
    shards written, and then the process died before the store commit (no join)."""
    rank, ctl, driver, cp = _trained_island(tmp_path, store, every)
    cp.every = every
    info = cp.save(driver, rollout_id=every)
    assert info["store_commit"] == "deferred"
    cp.trainer.finish_cut(cp._deferred["pending"])  # the writer finished; nothing committed
    return rank, ctl, driver, cp, info


def test_first_periodic_cut_with_interval_n_commits_in_the_background(tmp_path):
    """E1: with --rl-cut-every 4 the store's first cut (rollout 4) is in the steady
    rounds: it is deferred like the others, marked until its LATEST is committed."""
    store = _store(tmp_path / "store", _Vol())
    rank, ctl, driver, cp = _trained_island(tmp_path, store, 4)
    cp.every = 4
    calls = []
    start = cp.trainer.start_cut
    cp.trainer.start_cut = lambda **kw: (calls.append(rs.first_cut_mark(store.root, kw["context"].cut_id)
                                                      .is_file()), start(**kw))[1]
    info = cp.save(driver, rollout_id=4)
    assert info["store_commit"] == "deferred" and info["total_s"] is not None
    assert calls == [True]  # marked before the shards were started
    assert rs.read_latest(store.root) is None and cp.pointer() is None
    done = cp.join_commit(driver)
    assert done["ok"] is True and rs.read_latest(store.root)["cut_id"] == info["cut_id"]
    assert not list((store.root / rs.ROUND_CUTS).glob(f"*{rs.FIRST_CUT_MARK}"))
    assert _rctl(tmp_path, "b", store).restored["latest"]["cut_id"] == info["cut_id"]


def test_a_later_cut_is_not_marked(tmp_path):
    store = _store(tmp_path / "store", _Vol())
    rank, ctl, driver, cp = _trained_island(tmp_path, store, 2)
    _cut(cp, driver, rollout_id=2)
    train_step(rank, torch.randn(3, 6, generator=torch.Generator().manual_seed(4)))
    driver.local_step = 3
    driver.published_version = 3
    cp.source.ledger = _Ledger(3)
    info = cp.save(driver, rollout_id=3)
    assert info["store_commit"] == "deferred"
    assert not rs.first_cut_mark(store.root, info["cut_id"]).exists()
    assert cp.join_commit(driver)["ok"] is True


def test_a_run_dead_during_its_first_cut_commit_starts_over_at_zero(tmp_path):
    store = _store(tmp_path / "store", _Vol())
    _rank, _ctl, _driver, _cp_a, info = _first_cut_in_flight(tmp_path, store)
    cuts = store.root / rs.ROUND_CUTS
    assert (cuts / info["cut_id"]).is_dir() and rs.first_cut_mark(store.root, info["cut_id"]).is_file()
    assert rs.read_latest(store.root) is None
    ctl_b = _rctl(tmp_path, "b", store)  # a fresh machine
    assert ctl_b.restored is None and sorted(ctl_b.discarded_first_cut) == sorted(
        [f"{rs.ROUND_CUTS}/{info['cut_id']}", f"{rs.ROUND_CUTS}/{info['cut_id']}{rs.FIRST_CUT_MARK}"])
    assert list(cuts.iterdir()) == []
    journal = [json.loads(line) for line in (ctl_b.state_dir / rs.RESUME_JOURNAL).read_text().splitlines()]
    assert journal[-1]["action"] == "discard_unfinished_first_cut"
    rank_b = make_rank(9)
    driver_b = _Driver(rank_b)
    assert _cp(ctl_b, driver_b, _trainer(rank_b), _Ledger(0)).resume(driver_b) is None  # rollout 0


def test_an_in_place_restart_after_a_dead_first_cut_moves_the_state_dir_aside(tmp_path):
    """The dead run's background commit may have pointed its local state dir at the
    cut before dying; a restart on the same disk must not resume from that pointer."""
    store = _store(tmp_path / "store", _Vol())
    _rank, ctl, _driver, _cp_a, info = _first_cut_in_flight(tmp_path, store)
    (ctl.state_dir / POINTER).write_text(json.dumps({"cut_id": info["cut_id"], "next_rollout_id": 4}))
    ctl_b = rs.ResumeController(state_dir=ctl.state_dir, store=store)
    assert ctl_b.discarded_first_cut and not (ctl_b.state_dir / POINTER).exists()
    assert list(ctl.state_dir.parent.glob(f"{ctl.state_dir.name}.prev-*"))
    assert ctl_b.incarnation["index"] == 0  # the dead run's incarnations went aside too


# ---------------------------------------------------------------- round cut: save -> kill -> resume
def test_save_kill_resume_without_elastic_continues_bitwise(tmp_path):
    """tasks 5.2: fake trainer, save -> the machine is gone -> a fresh trainer with
    other weights resumes; state hash, version, lr and consumed groups continue."""
    vol = _Vol()
    store = _store(tmp_path / "store", vol)
    rank_a, ctl_a, driver_a, cp_a = _trained_island(tmp_path, store, 2)
    info = _cut(cp_a, driver_a, rollout_id=2)
    # E1: the store's first cut commits in the background too (marked until LATEST is committed)
    assert info["store_synced"] is True and info["store_commit"] == "deferred"
    assert not list((store.root / rs.ROUND_CUTS).glob(f"*{rs.FIRST_CUT_MARK}"))
    assert info["cut_bytes"] > 0 and info["lr_at_next_round"] == [5e-6]
    assert info["run_fingerprint"] == FP and info["consumed_groups"] == 8 and info["incarnation_index"] == 0
    assert vol.commits >= 1 and rs.read_latest(store.root)["cut_id"] == info["cut_id"]

    rank_b = make_rank(9)
    assert _hash(rank_b) != _hash(rank_a)
    ctl_b = _rctl(tmp_path, "b", store)
    driver_b = _Driver(rank_b)
    cp_b = _cp(ctl_b, driver_b, _trainer(rank_b), _Ledger(2))
    got = cp_b.resume(driver_b)
    assert got["next_rollout_id"] == 2 and driver_b.local_step == 2
    for n, v in params(rank_a).items():
        assert torch.equal(v, params(rank_b)[n])
    names = [e for e, _ in driver_b.events]
    assert names == ["rl_round_cut_restored", "rl_resume"]
    ev = dict(driver_b.events[1][1])
    assert ev["incarnation"] == 1 and ev["cut_incarnation_index"] == 0 and ev["policy_version"] == 2
    assert ev["checks"]["run_fingerprint"] == "same" and ev["checks"]["policy_hash"] == "same"
    assert ev["checks"]["consumed_prompt_ids_digest"] == "same" and ev["config_diff"] is None
    # first resumed round: lr must continue bitwise
    cp_b.check_first_round(driver_b, rollout_id=2, applied_lrs=(5e-6,))
    assert driver_b.events[-1][1]["ok"] is True


def test_resume_refuses_a_changed_config_unless_allowed(tmp_path):
    store = _store(tmp_path / "store")
    _r, _c, driver_a, cp_a = _trained_island(tmp_path, store, 2)
    _cut(cp_a, driver_a, rollout_id=2)
    rank_b = make_rank(9)
    ctl_b = _rctl(tmp_path, "b", store)
    cp_b = _cp(ctl_b, _Driver(rank_b), _trainer(rank_b), _Ledger(2), fingerprint={**FP, "seed": 1})
    with pytest.raises(rs.ResumeRefused, match="seed"):
        cp_b.resume(_Driver(rank_b))
    ctl_c = _rctl(tmp_path, "c", store)
    driver_c = _Driver(rank_b)
    cp_c = _cp(ctl_c, driver_c, _trainer(rank_b), _Ledger(2), fingerprint={**FP, "seed": 1},
               allow_config_change=True)
    cp_c.resume(driver_c)
    assert driver_c.events[-1][1]["config_diff"] == {"seed": {"cut": 17, "now": 1}}


def test_longer_run_changes_backend_fingerprint_and_lr_only_when_allowed(tmp_path):
    """A longer run changes the Miles argv (--num-rollout, --lr-decay-iters) and so the
    backend fingerprint and the next learning rate: refused by default, recorded when
    --rl-resume-allow-config-change is given."""
    store = _store(tmp_path / "store")
    _r, _c, driver_a, cp_a = _trained_island(tmp_path, store, 2)
    _cut(cp_a, driver_a, rollout_id=2)
    rank_b = make_rank(9)
    cp_b = _cp(_rctl(tmp_path, "b", store), _Driver(rank_b), _trainer(rank_b), _Ledger(2),
               backend_fingerprint="miles@longer")
    with pytest.raises(CutError, match="backend fingerprint differs"):
        cp_b.resume(_Driver(rank_b))
    driver_c = _Driver(rank_b)
    cp_c = _cp(_rctl(tmp_path, "c", store), driver_c, _trainer(rank_b), _Ledger(2),
               backend_fingerprint="miles@longer", allow_config_change=True)
    cp_c.resume(driver_c)
    resumed = [ev for name, ev in driver_c.events if name == "rl_resume"][-1]
    assert resumed["checks"]["backend_fingerprint"] == {
        "result": "changed(allowed)", "cut": "miles@0af62f4d", "run": "miles@longer"}
    cp_c.check_first_round(driver_c, rollout_id=2, applied_lrs=(1e-5,))
    check = [ev for name, ev in driver_c.events if name == "rl_resume_check"][-1]
    assert check["ok"] is False and check["allowed"] is True


def test_resume_refuses_lr_jump_and_ledger_drift(tmp_path):
    store = _store(tmp_path / "store")
    _r, _c, driver_a, cp_a = _trained_island(tmp_path, store, 2)
    _cut(cp_a, driver_a, rollout_id=2)
    rank_b = make_rank(9)
    driver_b = _Driver(rank_b)
    cp_b = _cp(_rctl(tmp_path, "b", store), driver_b, _trainer(rank_b), _Ledger(2))
    cp_b.resume(driver_b)
    with pytest.raises(RoundCutError, match="lr"):
        cp_b.check_first_round(driver_b, rollout_id=2, applied_lrs=(1e-5,))
    ledger = _Ledger(2)
    ledger._consumed_groups["extra"] = 1  # restored ledger is not the one the cut saw
    cp_c = _cp(_rctl(tmp_path, "c", store), _Driver(rank_b), _trainer(rank_b), ledger)
    with pytest.raises(RoundCutError, match="consumed groups"):
        cp_c.resume(_Driver(rank_b))


def test_cut_every_two_keeps_the_newest_two(tmp_path):
    store = _store(tmp_path / "store")
    rank = make_rank(0)
    ctl = _rctl(tmp_path, "a", store, keep=2)
    driver = _Driver(rank, published=0)
    driver.last_applied_lrs = [5e-6]
    cp = _cp(ctl, driver, _trainer(rank), _Ledger(0), every=2)
    saved = []
    g = torch.Generator().manual_seed(1)
    for rid in range(1, 8):
        train_step(rank, torch.randn(3, 6, generator=g))
        driver.published_version = rid
        driver.local_step = rid
        driver.published_state = SimpleNamespace(policy_tensor_hash=lambda: _hash(rank), policy_version=rid)
        cp.source.ledger = _Ledger(rid)
        if _cut(cp, driver, rollout_id=rid) is not None:
            saved.append(rid)
    assert saved == [2, 4, 6]
    assert cp.save(driver, rollout_id=7, final=True)["final"] is True  # last round always cut
    cuts = sorted(p.name for p in (store.root / rs.ROUND_CUTS).iterdir())
    assert [c[:7] for c in cuts] == ["r000006", "r000007"]
    assert len(list((store.root / "state").iterdir())) == 2


def test_colocated_trainer_is_woken_for_the_cut(tmp_path):
    store = _store(tmp_path / "store")
    _r, _c, driver, cp = _trained_island(tmp_path, store, 2)
    calls = []
    cp.trainer.onload = lambda: calls.append("on")
    cp.trainer.offload = lambda: calls.append("off")
    driver._trainer_offloaded = True
    info = cp.save(driver, rollout_id=2)
    assert calls == ["on", "off"] and info["trainer_onloaded_for_cut"] is True


def test_local_only_sync_stop_after_and_final_cut():
    sync = LocalOnlySync(6)
    sync.stop_after = 3
    calls = []

    class Stub:
        def save(self, driver, *, rollout_id, final=False):
            calls.append((rollout_id, final))
            return None

        def check_first_round(self, driver, *, rollout_id, applied_lrs):
            calls.append(("check", rollout_id))

    sync.round_cuts = Stub()
    driver = _Driver(make_rank(0), published=3)
    driver.export_local = lambda: None
    import yeto.rl.engine.bridges as br

    real = br._local_state
    br._local_state = lambda d, v: SimpleNamespace(policy_version=v)
    try:
        stats = SimpleNamespace(applied_lrs=(5e-6,))
        import dataclasses

        br.asdict = lambda s: {} if not dataclasses.is_dataclass(s) else dataclasses.asdict(s)
        assert sync.boundary(driver, rollout_id=1, stats=stats).stop is False
        assert sync.boundary(driver, rollout_id=2, stats=stats).stop is True
        assert any(e == "rl_stop_requested" for e, _ in driver.events)
    finally:
        br._local_state = real
        br.asdict = dataclasses.asdict
    driver.at_safe_point = False  # as after run_round
    sync.finish(driver)
    assert (3, True) in calls and ("check", 1) in calls and driver.at_safe_point is True


# ---------------------------------------------------------------- elastic store copy hashes
def test_elastic_store_copy_is_hashed_and_a_tampered_copy_is_refused(tmp_path):
    store = tmp_path / "store"
    ctl = _ctl(tmp_path / "a/state", {"t": 1.0}, checkpoint_store=store)
    assert ctl.sync_checkpoint_store("test")
    manifest = json.loads((store / STORE_MANIFEST).read_text())
    assert manifest["files"] and all(len(m["sha256"]) == 64 for m in manifest["files"].values())
    ctl.close()
    victim = store / sorted(manifest["files"])[0]
    victim.write_text(victim.read_text() + "x")
    with pytest.raises(rs.StoreIntegrityError):
        _ctl(tmp_path / "b/state", {"t": 2.0}, checkpoint_store=store)


# ---------------------------------------------------------------- launcher / Modal
def test_launcher_modal_store_is_a_volume_or_an_error():
    from test_launch_auto import _args, _specs

    from yeto import launcher

    base = ["--training-mode", "rl", "--rl-single-island-no-sync", "--cluster-prefix", "run"]
    args = _args(["--gpu", "modal:1xh100"] + base + ["--rl-checkpoint-store", "s3://b/x"])
    with pytest.raises(ValueError, match="modal-volume"):
        launcher.modal_checkpoint_store(args)
    with pytest.raises(ValueError, match="modal-volume"):
        launcher.check_cloud_prerequisites(_specs(args.gpu), args=args, modal_ok=True)
    args = _args(["--gpu", "modal:1xh100"] + base + ["--rl-checkpoint-store", "modal-volume://yeto-ckpt/g2"])
    assert launcher.modal_checkpoint_store(args) == ("yeto-ckpt", "/root/yeto-checkpoint-store/g2")
    assert launcher.rl_checkpoint_store_plan(args) == ("/root/yeto-checkpoint-store/g2", None)
    assert launcher.resumes_in_new_container(args)
    aws = _args(["--gpu", "aws:1xh100@us-east-1"] + base + ["--rl-checkpoint-store", "modal-volume://v"])
    with pytest.raises(ValueError, match="only mounts on Modal"):
        launcher.check_cloud_prerequisites(_specs(aws.gpu), args=aws, modal_ok=True)
    assert launcher._check_resume_launch_flags(args) is True
    multi = _args(["--gpu", "modal:1xh100", "--training-mode", "rl", "--rl-checkpoint-store", "modal-volume://v"])
    with pytest.raises(ValueError, match="single-island-no-sync"):
        launcher._check_resume_launch_flags(multi)
    with pytest.raises(ValueError, match="needs --rl-checkpoint-store"):
        launcher._check_resume_launch_flags(_args(["--rl-cut-every", "2"]))


def test_modal_island_config_mounts_the_store_volume(monkeypatch, tmp_path):
    from test_launch_auto import _args, _specs

    from yeto.launcher import build_modal_island_config

    monkeypatch.setenv("HOME", str(tmp_path))
    task = SimpleNamespace(run="true", envs={}, setup="")
    digest = "docker:ghcr.io/x/miles@sha256:" + "d" * 64
    args = _args(["--gpu", "modal:1xh100", "--cluster-prefix", "run", "--training-mode", "rl",
                  "--rl-image", digest, "--rl-single-island-no-sync",
                  "--rl-checkpoint-store", "modal-volume://yeto-ckpt/g2"])
    (spec,) = _specs(args.gpu)
    cfg = build_modal_island_config(args, spec, 0, task, "none")
    assert cfg.checkpoint_store_volume_name == "yeto-ckpt"
    assert cfg.checkpoint_store_mount == "/root/yeto-checkpoint-store"
    assert cfg.envs[mcs.MODAL_VOLUME_ENV] == "yeto-ckpt" and cfg.envs[rs.STORE_IMPL_ENV] == mcs.STORE_IMPL
    cfg.validate()
    from yeto import modal_runner as mr

    assert mr.ModalIslandConfig.from_json(cfg.to_json()) == cfg


def test_learner_resume_switches(tmp_path):
    from test_rl_engine_selection import _learner_argv

    from yeto.rl.adapters.miles import island_entry as learner

    argv = _learner_argv(("--rl-single-island-no-sync", "--rl-resume-store", "/s",
                          "--rl-cut-every", "2", "--rl-cut-keep", "3", "--rl-stop-after-rounds", "3"))
    i = argv.index("--syncer")
    del argv[i:i + 2]
    parsed = learner.parse_args(argv)
    learner._check_ports_infra_switches(parsed)
    miles_args, env = SimpleNamespace(), {}
    learner.apply_ports_infra_switches(parsed, miles_args, env)
    assert miles_args.yeto_rl_resume == {"store": "/s", "state_dir": "~/yeto-rl/resume-state", "every": 2,
                                         "keep": 3, "allow_config_change": False, "stop_after": 3}
    assert miles_args.yeto_rl_elastic_metadata is True and env
    with pytest.raises(SystemExit):  # "--rl-cut-every need --rl-resume-store"
        learner.parse_args(_learner_argv(("--rl-cut-every", "2")))


def test_tape_mirror_never_replaces_a_previous_container(tmp_path):
    from yeto import modal_runner as mr

    src, dst = tmp_path / "out", tmp_path / "vol" / "rank0"
    src.mkdir()
    (src / "rl-island-0.jsonl").write_text('{"e": 1}\n')
    first = mr.TapeSync(str(src), str(dst), lambda: None, interval_s=3600)
    first.sync_once()
    (src / "rl-island-0.jsonl").write_text('{"e": "second container"}\n')  # fresh disk
    second = mr.TapeSync(str(src), str(dst), lambda: None, interval_s=3600)
    second.sync_once()
    assert (dst / "rl-island-0.jsonl").read_text() == '{"e": 1}\n'
    assert (dst / "rl-island-0.inc1.jsonl").read_text() == '{"e": "second container"}\n'
    assert sorted(p.name for p in dst.glob("INCARNATION-*")) == ["INCARNATION-0.txt", "INCARNATION-1.txt"]


def test_dashboard_sees_resumes_segments_and_discarded_rounds():
    from yeto.dashboard.reducer import Reducer

    red = Reducer(run="r")
    t = 1000.0
    recs = [("rl_driver_start", {}, 0)]
    for rid in range(4):
        recs.append(("rl_round_trained", {"rollout_id": rid}, 10 + rid * 10))
    recs += [("rl_driver_start", {}, 100),
             ("rl_resume", {"rollout_id": 2, "incarnation": 1, "cut_id": "r000002-x", "restore_s": 1.5}, 110)]
    for rid in (2, 3):
        recs.append(("rl_round_trained", {"rollout_id": rid}, 120 + rid))
    for event, fields, dt in recs:
        red.feed({"event": event, "learner_id": 0, "time_unix": t + dt, **fields}, source="tape")
    isl = red.island(red.island_ids()[0])
    assert [x["rollout_id"] for x in isl["resumes"]] == [2]
    assert red.discarded_trainings(isl, 2) == 1 and red.discarded_trainings(isl, 1) == 0
    segs = red.resume_segments(isl)
    assert [s["resumed_at"] for s in segs] == [None, 2] and segs[1]["startup_s"] == 22.0


def test_next_lr_matches_the_g1_tape_bitwise():
    """S17 G1 base tape (s1-runs/s17-g1-base, 10 rounds, linear to 0 over 10): applied
    lrs per round; the cut's lr_at_next_round prediction must reproduce them exactly."""
    from yeto.rl.adapters.miles.entry import next_lr_for

    g1 = [1e-05, 9e-06, 8.000000000000001e-06, 7e-06, 6e-06, 5e-06, 4.000000000000001e-06,
          3.0000000000000005e-06, 2e-06, 1e-06]
    f = next_lr_for(SimpleNamespace(lr=1e-5, lr_decay_style="linear", lr_decay_iters=10,
                                    lr_warmup_iters=0, min_lr=0.0))
    assert [f(SimpleNamespace(local_step=k))[0] for k in range(10)] == g1
    c = next_lr_for(SimpleNamespace(lr=5e-6, lr_decay_style="constant", lr_warmup_iters=0, min_lr=0.0))
    assert c(SimpleNamespace(local_step=7)) == [5e-6]
    assert next_lr_for(SimpleNamespace(lr=1e-5, lr_decay_style="linear", lr_decay_iters=None)) is None
    assert next_lr_for(SimpleNamespace(lr=1e-5, lr_decay_style="linear", lr_decay_iters=10,
                                       lr_warmup_iters=5)) is None


def test_preempt_signal_writes_only_a_marker_and_hands_the_signal_on(tmp_path):
    import os
    import signal

    store = _store(tmp_path / "store", _Vol())
    ctl = _rctl(tmp_path, "a", store)
    seen = []
    old = signal.signal(signal.SIGUSR1, lambda n, f: seen.append(n))
    try:
        assert rs.install_preempt_handler(ctl, signals=("SIGUSR1",)) == ["SIGUSR1"]
        os.kill(os.getpid(), signal.SIGUSR1)
        marks = list((store.root / "preempt-notices").glob("*.json"))
        assert len(marks) == 1 and json.loads(marks[0].read_text())["saved"] is False
        assert seen == [signal.SIGUSR1] and store.commits == 1
        assert not (store.root / rs.LATEST).exists()  # no cut attempted in the grace window
    finally:
        signal.signal(signal.SIGUSR1, old)


def test_modal_commit_goes_through_the_runner_interpreter(tmp_path, monkeypatch):
    """G2 A: the learner's Python has no Modal client deps; the commit runs in the
    runner's interpreter with the runner's import path (a fake modal module here)."""
    import sys

    fake = tmp_path / "pkg"
    fake.mkdir()
    log = tmp_path / "calls.txt"
    (fake / "modal.py").write_text(
        "class _V:\n"
        "    def __init__(self, n): self.n = n\n"
        f"    def commit(self): open({str(log)!r}, 'a').write('commit ' + self.n + '\\n')\n"
        f"    def reload(self): open({str(log)!r}, 'a').write('reload ' + self.n + '\\n')\n"
        "class Volume:\n"
        "    @staticmethod\n"
        "    def from_name(n): return _V(n)\n")
    monkeypatch.setenv("YETO_MODAL_PYTHON", sys.executable)
    monkeypatch.setenv("YETO_MODAL_SYSPATH", str(fake))
    store = mcs.ModalVolumeStore(tmp_path / "s", "yeto-ckpt")
    store.reload()
    assert store.commit() >= 0 and store.commits == 1
    assert log.read_text().splitlines() == ["reload yeto-ckpt", "commit yeto-ckpt"]
    monkeypatch.setenv("YETO_MODAL_SYSPATH", str(tmp_path / "nothing"))
    with pytest.raises(RuntimeError, match="runner interpreter failed"):
        store.commit()


def test_resume_wiring_carries_the_runtime_fingerprint(tmp_path):
    """G2 A: the cut refused 'runtime: backend_fingerprint missing' -- the wiring must
    hand the island's runtime fingerprint to the round-cut controller."""
    from yeto.rl.adapters.miles.entry import resume_wiring_for

    cfg = {"store": str(tmp_path / "store"), "state_dir": str(tmp_path / "state"), "every": 2,
           "keep": 2, "allow_config_change": False, "stop_after": None}
    with pytest.raises(ValueError, match="runtime fingerprint"):
        resume_wiring_for(SimpleNamespace(yeto_rl_resume=dict(cfg)), sync=LocalOnlySync(3), environ={})
    w = resume_wiring_for(SimpleNamespace(yeto_rl_resume={**cfg, "runtime_fingerprint": "miles@abc"}),
                          sync=LocalOnlySync(3), environ={})
    assert w.controller.runtime_fingerprint == "miles@abc" and w.every == 2 and w.ledger is not None


# ---------------------------------------------------------------- E1: cut before offload, background cut
class _CountingTrainer:
    """The real MilesTrainerGroup cut path, counting the cut and onload/offload calls."""

    def __init__(self, inner):
        self.inner = inner
        self.calls: list[str] = []

    def save_cut(self, **kw):
        self.calls.append("save_cut")
        return self.inner.save_cut(**kw)

    def start_cut(self, **kw):
        self.calls.append("start_cut")
        return self.inner.start_cut(**kw)

    def finish_cut(self, pending):
        self.calls.append("finish_cut")
        return self.inner.finish_cut(pending)

    def collect_cut(self, pending):
        self.calls.append("collect_cut")
        return self.inner.collect_cut(pending)

    def onload(self):
        self.calls.append("onload")

    def offload(self):
        self.calls.append("offload")

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _publish(driver, version, policy_hash):
    from yeto.rl.engine.driver import policy_token

    driver.published_version = version
    driver.published_state = SimpleNamespace(policy_tensor_hash=lambda: policy_hash, policy_version=version)
    driver.expected_token = policy_token(version, policy_hash)


def _island_with_first_cut(tmp_path, store, keep=2):
    """One trained round, cut inline at v1 (the store's first cut); then a second
    trained round whose cut (v2) is the periodic one under test."""
    rank = make_rank(0)
    g = torch.Generator().manual_seed(3)
    train_step(rank, torch.randn(3, 6, generator=g))
    ctl = _rctl(tmp_path, "a", store, keep=keep)
    driver = _Driver(rank, published=None, local_step=1)
    driver.last_applied_lrs = [5e-6]
    trainer = _CountingTrainer(_trainer(rank))
    cp = _cp(ctl, driver, trainer, _Ledger(2))
    _publish(driver, 1, _hash(rank))
    first = _cut(cp, driver, rollout_id=1)
    assert first["store_synced"] is True
    train_step(rank, torch.randn(3, 6, generator=g))
    driver.local_step = 2
    trainer.calls.clear()
    return rank, driver, trainer, cp, first


def _resume_fresh(tmp_path, store, rank):
    rank_b = make_rank(9)
    driver_b = _Driver(rank_b)
    cp_b = _cp(_rctl(tmp_path, "b", store), driver_b, _trainer(rank_b), _Ledger(2))
    got = cp_b.resume(driver_b)
    for n, v in params(rank).items():
        assert torch.equal(v, params(rank_b)[n])
    return got


def test_prepared_cut_is_written_in_the_background_without_waking_the_trainer(tmp_path):
    store = _store(tmp_path / "store", _Vol())
    rank, driver, trainer, cp, first = _island_with_first_cut(tmp_path, store)
    driver._trainer_offloaded = True
    state = SimpleNamespace(policy_version=2, policy_tensor_hash=lambda: _hash(rank))
    assert cp.prepare(driver, rollout_id=2, state=state) is True
    _publish(driver, 2, _hash(rank))
    info = cp.save(driver, rollout_id=2)
    assert trainer.calls == ["start_cut"]  # captured before the offload: no wake/sleep
    assert info["prepared_before_offload"] is True and info["store_commit"] == "deferred"
    assert cp.pointer()["cut_id"] == first["cut_id"]  # not pointed at until complete
    done = cp.join_commit(driver)
    assert done["ok"] is True and done["cut_id"] == info["cut_id"] and trainer.calls[-1] == "collect_cut"
    assert driver.events[-1][0] == "rl_cut_committed" and driver.events[-1][1]["ok"] is True
    # E1: the event says where the shard was written (the rank's writer process)
    assert driver.events[-1][1]["writer"] == [{"mode": "process"}]
    assert cp.pointer()["cut_id"] == info["cut_id"]
    assert rs.read_latest(store.root)["cut_id"] == info["cut_id"]
    assert _resume_fresh(tmp_path, store, rank)["next_rollout_id"] == 2


def test_prepared_cut_for_another_policy_is_dropped_and_cut_again(tmp_path):
    store = _store(tmp_path / "store", _Vol())
    rank, driver, trainer, cp, _first = _island_with_first_cut(tmp_path, store)
    driver._trainer_offloaded = True
    cp.prepare(driver, rollout_id=2, state=SimpleNamespace(policy_version=2, policy_tensor_hash=lambda: "f" * 64))
    _publish(driver, 2, _hash(rank))  # the publication names other weights
    info = cp.save(driver, rollout_id=2)
    assert trainer.calls == ["start_cut", "finish_cut", "onload", "start_cut", "offload"]
    assert info["prepared_before_offload"] is False and info["policy_hash"] == _hash(rank)
    assert cp.join_commit(driver)["ok"] is True
    assert _resume_fresh(tmp_path, store, rank)["policy_hash"] == _hash(rank)


def test_final_cut_completes_inline_after_the_pending_one(tmp_path):
    store = _store(tmp_path / "store", _Vol())
    rank, driver, trainer, cp, _first = _island_with_first_cut(tmp_path, store)
    _publish(driver, 2, _hash(rank))
    info = cp.save(driver, rollout_id=2, final=True)
    assert info["store_commit"] == "inline" and info["store_synced"] is True
    assert trainer.calls == ["save_cut"] and cp.join_commit(driver) is None


def test_failed_background_commit_keeps_the_previous_cut(tmp_path):
    class _FailingVol(_Vol):
        fail = False

        def commit(self):
            if self.fail:
                raise OSError("volume commit failed")
            super().commit()

    vol = _FailingVol()
    store = _store(tmp_path / "store", vol)
    rank, driver, _trainer_c, cp, first = _island_with_first_cut(tmp_path, store, keep=1)
    _publish(driver, 2, _hash(rank))
    vol.fail = True
    cp.save(driver, rollout_id=2)
    result = cp.join_commit(driver)
    assert result["ok"] is False and "volume commit failed" in result["error"]
    assert driver.events[-1][0] == "rl_cut_committed" and driver.events[-1][1]["ok"] is False
    # nothing was pruned: the previous (committed) cut is still in the store
    assert (store.root / rs.ROUND_CUTS / first["cut_id"]).is_dir()


def test_background_cut_can_be_turned_off(tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_RL_CUT_BACKGROUND_COMMIT", "0")
    store = _store(tmp_path / "store", _Vol())
    rank, driver, trainer, cp, _first = _island_with_first_cut(tmp_path, store)
    _publish(driver, 2, _hash(rank))
    info = cp.save(driver, rollout_id=2)
    assert info["store_commit"] == "inline" and info["store_synced"] is True and trainer.calls == ["save_cut"]


def test_cut_context_takes_the_cursor_read_with_the_in_flight_entries():
    """E1: a rollout that keeps drawing during the cut (Miles fully_async) returns the
    cursor read together with its entries; the context uses that one."""
    rank = make_rank(0)
    driver = _Driver(rank, published=2, local_step=2)
    driver.profile = SimpleNamespace(max_policy_age=1)
    live = {"sample_offset": 9, "epoch_id": 0, "sample_group_index": 9, "sample_index": 9}
    with_entries = {**live, "sample_offset": 12, "sample_group_index": 12, "sample_index": 12}
    pool = SimpleNamespace(data_cursor=lambda: dict(live), export_in_flight=lambda: {
        "entries": [], "buffer_groups": 0, "data_cursor": dict(with_entries)})
    source = CutSource(driver=lambda: driver, trainer=None, rollout=pool, ledger=_Ledger(2),
                       algorithm=SimpleNamespace(sha256=lambda: "a" * 64, to_legacy_runtime_attrs=lambda: {}),
                       backend_fingerprint="miles@0af62f4d", cut_root="", global_batch_size=ARGS.global_batch_size)
    data = source.context("r000002-abc").data
    assert data["sample_offset"] == 12 and data["buffer_length"] == 0


def test_periodic_cut_is_skipped_while_the_previous_commit_runs(tmp_path):
    import threading

    store = _store(tmp_path / "store", _Vol())
    rank, driver, trainer, cp, first = _island_with_first_cut(tmp_path, store)
    gate = threading.Event()
    busy = threading.Thread(target=gate.wait, daemon=True)
    busy.start()
    cp._commit = (busy, {"cut_id": first["cut_id"]})
    _publish(driver, 2, _hash(rank))
    assert cp.save(driver, rollout_id=2) is None and trainer.calls == []
    assert driver.events[-1][0] == "rl_cut_skipped" and cp.skipped_busy == 1
    gate.set()
    busy.join()
    info = cp.save(driver, rollout_id=2, final=True)  # the final cut waits and is taken
    assert info["store_commit"] == "inline"


def test_a_periodic_cut_never_waits_for_the_commit_it_starts(tmp_path):
    import threading

    class _SlowVol(_Vol):
        gate = threading.Event()
        slow = False

        def commit(self):
            if self.slow:
                self.gate.wait(10)
            super().commit()

    vol = _SlowVol()
    store = _store(tmp_path / "store", vol)
    rank, driver, _trainer_c, cp, _first = _island_with_first_cut(tmp_path, store)
    g = torch.Generator().manual_seed(5)
    _publish(driver, 2, _hash(rank))
    assert cp.save(driver, rollout_id=2)["store_commit"] == "deferred"
    vol.slow = True
    train_step(rank, torch.randn(3, 6, generator=g))
    driver.local_step = 3
    _publish(driver, 3, _hash(rank))
    info = cp.save(driver, rollout_id=3)  # completes v2 and starts its (slow) commit
    assert info["store_commit"] == "deferred" and cp._commit[0].is_alive()
    train_step(rank, torch.randn(3, 6, generator=g))
    driver.local_step = 4
    _publish(driver, 4, _hash(rank))
    assert cp.save(driver, rollout_id=4) is None  # v2 still committing: skipped
    assert driver.events[-1][0] == "rl_cut_skipped"
    vol.gate.set()
    final = cp.save(driver, rollout_id=4, final=True)
    assert final["store_commit"] == "inline" and rs.read_latest(store.root)["cut_id"] == final["cut_id"]
    committed = [f["cut_id"] for e, f in driver.events if e == "rl_cut_committed"][1:]  # after v1
    assert committed == ["r000002-" + cp.controller.incarnation["id"], info["cut_id"]]  # v2, then v3


def test_prune_keeps_a_cut_newer_than_every_kept_one(tmp_path):
    """E1: a periodic cut is captured while the previous one commits in the background;
    the commit's prune must not delete the newer cut being written."""
    root = tmp_path / "store"
    for seq, cut_id in ((1, "r000002-aa"), (2, "r000004-aa")):
        (root / rs.STATE_DIR / f"{seq:08d}").mkdir(parents=True)
        (root / rs.STATE_DIR / f"{seq:08d}" / "cut").write_text(cut_id)
    for cut_id in ("r000001-aa", "r000002-aa", "r000004-aa", "r000006-aa"):
        (root / rs.ROUND_CUTS / cut_id).mkdir(parents=True)
    (root / rs.LATEST).write_text(json.dumps({"seq": 2}))
    deleted = rs.prune(root, keep=1, cut_ids_of=lambda p: (p / "cut").read_text())
    assert sorted(deleted) == [f"{rs.ROUND_CUTS}/r000001-aa", f"{rs.ROUND_CUTS}/r000002-aa",
                               f"{rs.STATE_DIR}/00000001"]
    assert sorted(p.name for p in (root / rs.ROUND_CUTS).iterdir()) == ["r000004-aa", "r000006-aa"]


def test_a_prepared_cut_skipped_while_committing_is_collected_at_the_next_cut(tmp_path):
    """E1: the safe point does not wait for the trainer writer of a prepared cut it
    skips (busy commit); the next cut collects that writer and drops its files."""
    import threading

    from yeto.rl.engine.cut import cut_dir

    store = _store(tmp_path / "store", _Vol())
    rank, driver, trainer, cp, first = _island_with_first_cut(tmp_path, store)
    driver._trainer_offloaded = True
    state = SimpleNamespace(policy_version=2, policy_tensor_hash=lambda: _hash(rank))
    assert cp.prepare(driver, rollout_id=2, state=state) is True
    gate = threading.Event()
    busy = threading.Thread(target=gate.wait, daemon=True)
    busy.start()
    cp._commit = (busy, {"cut_id": first["cut_id"]})
    _publish(driver, 2, _hash(rank))
    assert cp.save(driver, rollout_id=2) is None and driver.events[-1][0] == "rl_cut_skipped"
    assert trainer.calls == ["start_cut"]  # the skipped cut's writer is not waited for
    skipped = cut_dir(cp.root, cp._cut_id(2))
    gate.set()
    busy.join()
    info = cp.save(driver, rollout_id=2, final=True)
    # the skipped writer is collected first, then the final cut is taken afresh
    assert trainer.calls[:2] == ["start_cut", "finish_cut"] and cp._abandoned == []
    assert info["store_commit"] == "inline" and info["cut_id"] == skipped.name
    assert rs.read_latest(store.root)["cut_id"] == info["cut_id"]


def test_a_cut_declined_at_prepare_does_not_wake_the_offloaded_trainer(tmp_path):
    """E1: prepare met a running commit; when the commit is done by the safe point the
    periodic cut is skipped instead of waking the offloaded trainer for it."""
    import threading

    store = _store(tmp_path / "store", _Vol())
    rank, driver, trainer, cp, first = _island_with_first_cut(tmp_path, store)
    driver._trainer_offloaded = True
    gate = threading.Event()
    busy = threading.Thread(target=gate.wait, daemon=True)
    busy.start()
    cp._commit = (busy, {"cut_id": first["cut_id"]})
    state = SimpleNamespace(policy_version=2, policy_tensor_hash=lambda: _hash(rank))
    assert cp.prepare(driver, rollout_id=2, state=state) is False
    gate.set()
    busy.join()
    _publish(driver, 2, _hash(rank))
    assert cp.save(driver, rollout_id=2) is None and trainer.calls == []
    assert driver.events[-1][0] == "rl_cut_skipped"
    assert driver.events[-1][1]["reason"] == "not prepared before the trainer offloaded"
    # a cut that was never declined still wakes the trainer (no silent loss of cuts)
    train_step(rank, torch.randn(3, 6, generator=torch.Generator().manual_seed(7)))
    driver.local_step = 3
    _publish(driver, 3, _hash(rank))
    assert cp.save(driver, rollout_id=3)["trainer_onloaded_for_cut"] is True


def test_a_deferred_cut_snapshots_the_state_dir_as_it_was_at_the_cut(tmp_path):
    """E1 (Modal A10G resume e1c): a periodic cut is synced into the store after later
    rounds changed the state dir (the ledger); the snapshot must hold the state dir as
    it was at the cut, else a resume refuses the ledger (consumed groups differ)."""
    store = _store(tmp_path / "store", _Vol())
    rank, driver, _trainer_c, cp, _first = _island_with_first_cut(tmp_path, store)
    marker = Path(cp.controller.state_dir) / "ledger-e1-marker.txt"
    marker.write_text("at the cut")
    _publish(driver, 2, _hash(rank))
    info = cp.save(driver, rollout_id=2)
    assert info["store_commit"] == "deferred"
    marker.write_text("one round later")  # the next round trains before the commit runs
    done = cp.join_commit(driver)
    assert done["ok"] is True
    latest = rs.read_latest(store.root)
    assert latest["cut_id"] == info["cut_id"]
    snap = store.root / rs.STATE_DIR / f"{int(latest['seq']):08d}"
    assert (snap / marker.name).read_text() == "at the cut"
    assert json.loads((snap / POINTER).read_text())["cut_id"] == info["cut_id"]
    assert cp.pointer()["cut_id"] == info["cut_id"]
    assert not list(Path(cp.controller.state_dir).parent.glob(".*.stage-*"))  # staging removed
