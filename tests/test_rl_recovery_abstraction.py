"""rl-code-quality-cleanup stage 3, CPU only: one consistent-cut definition, one
checkpoint store interface (local / bucket mount / Modal Volume / critic), one save
path for the preempt, periodic and manual triggers, and resume after each."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_rl_multinode_round_cut import _Driver, _trainer
from test_rl_reconfig_recovery import _ctl
from test_rl_resume_from_checkpoint import _cp, _Ledger, _rctl, _store, _trained_island, _Vol

from tests.rl_cut_fakes import make_rank, params, train_step
from yeto.cloud import modal_ckpt_store as mcs
from yeto.cloud import preemption as p
from yeto.rl.adapters.miles import inflight_save as s
from yeto.rl.adapters.miles.round_cut import ISLAND
from yeto.rl.critic_state import CriticCheckpointStore
from yeto.rl.engine import recovery as rc
from yeto.rl.engine import resume as rs

CUT_FIELDS = {"schema", "checkpoint_id", "trigger", "outer_step", "island_steps",
              "completed_groups", "completed_groups_digest"}
ENVELOPE = {rc.CUT_KEY, "trigger", "store_kind", "cut_total_s"}


# ---------------------------------------------------------------- 3.1 cut + interface
def test_consistent_cut_round_trips_and_validates():
    cut = rc.ConsistentCut("r000002-x", rc.PERIODIC, outer_step=4, island_steps={"b": 3, "a": 2},
                           completed_groups=8, completed_groups_digest="d" * 64)
    record = cut.to_record()
    assert set(record) == CUT_FIELDS and record["schema"] == rc.CUT_SCHEMA
    assert list(record["island_steps"]) == ["a", "b"]
    assert rc.ConsistentCut.from_record(json.loads(json.dumps(record))) == cut
    with pytest.raises(ValueError):
        rc.ConsistentCut("x", "sometimes")
    with pytest.raises(ValueError):
        rc.ConsistentCut("", rc.MANUAL)
    with pytest.raises(ValueError):
        rc.ConsistentCut.from_record({**record, "schema": "other"})


def test_four_backends_implement_the_store_interface(tmp_path):
    stores = [rs.CheckpointStore(tmp_path / "local"), rs.BucketMountStore(tmp_path / "bucket"),
              mcs.ModalVolumeStore(tmp_path / "vol", "v", volume_factory=lambda n: _Vol()),
              CriticCheckpointStore(tmp_path / "critic")]
    assert [st.kind for st in stores] == ["local", "bucket-mount", "modal-volume", "critic"]
    for st in stores:
        assert isinstance(st, rs.CheckpointStore)
        assert isinstance(st.commit(), float)
        st.reload()
        assert st.describe()["kind"] == st.kind
    critic = stores[-1]
    critic.save(round_id=1, weights={"w": torch.ones(2)}, optimizer={"step": 1})
    assert critic.list("critic") == ["round-1"]
    weights, _ = critic.restore(actor_round=1)
    assert torch.equal(weights["w"], torch.ones(2))


def test_save_commits_unless_the_writer_did(tmp_path):
    vol = _Vol()
    store = _store(tmp_path, vol)
    cut = rc.ConsistentCut("c", rc.MANUAL)
    out = rc.save(cut, store=store, write=lambda record: {"seen": record["checkpoint_id"]})
    assert vol.commits == 1 and out["seen"] == "c" and set(ENVELOPE) <= set(out)
    rc.save(cut, store=store, write=lambda record: {"committed": True})
    assert vol.commits == 1


# ---------------------------------------------------------------- 3.2 / 3.4 triggers
def _reclaim_save(saver):
    notice = p.ReclaimNotice(island="0", cloud="modal", source="test", notified_at=0.0, deadline_s=60.0)
    events = []
    p.handle_notice(notice, save=saver.save, leave=lambda: None, emit=lambda e, **f: events.append(f),
                    last_save_s=saver.last_save_s, margin_s=1.0, clock=lambda: 0.0)
    assert events[0]["saved"] is True, events
    return saver.results[-1]


def test_preempt_periodic_and_manual_write_the_same_cut_format_and_resume(tmp_path):
    vol = _Vol()
    store = _store(tmp_path / "store", vol)
    rank_a, ctl_a, driver_a, cp_a = _trained_island(tmp_path, store, 2)
    driver_a.sync = SimpleNamespace(outer_version=2)

    periodic = cp_a.save(driver_a, rollout_id=2)

    saver = s.InFlightSaver(lambda t: {"entries": [], "buffer_groups": 0}, str(tmp_path / "inflight"),
                            store=_store(tmp_path / "inflight", vol), printer=lambda line: None)
    saver.rehearse(2)
    saver.refresh_cache(2)  # the rollout after round 2 ended (driver phase "train")
    preempt = _reclaim_save(saver)

    train_step(rank_a, torch.randn(3, 6, generator=torch.Generator().manual_seed(4)))
    driver_a.published_version = 3
    driver_a.local_step = 3
    driver_a.sync.outer_version = 3
    cp_a.source.ledger = _Ledger(3)
    manual = cp_a.save(driver_a, rollout_id=3, final=True)

    records = {
        rc.PERIODIC: periodic[rc.CUT_KEY],
        rc.PREEMPT: json.loads(Path(preempt["path"]).read_text())[rc.CUT_KEY],
        rc.MANUAL: manual[rc.CUT_KEY],
    }
    for trigger, record in records.items():
        assert set(record) == CUT_FIELDS and record["schema"] == rc.CUT_SCHEMA
        assert rc.ConsistentCut.from_record(record).trigger == trigger
    assert preempt["trigger"] == rc.PREEMPT and set(ENVELOPE) <= set(preempt)
    assert records[rc.PERIODIC]["island_steps"] == {ISLAND: 2} and records[rc.PERIODIC]["outer_step"] == 2
    assert records[rc.MANUAL]["island_steps"] == {ISLAND: 3} and records[rc.MANUAL]["completed_groups"] == 12
    assert records[rc.PREEMPT]["outer_step"] == 2  # the in-flight set cached after round 2

    # resume after the manual cut: steps and island state continue from that cut
    pointer = json.loads((ctl_a.state_dir / "round-cut.json").read_text())
    assert pointer[rc.CUT_KEY] == records[rc.MANUAL]
    rank_b = make_rank(9)
    ctl_b = _rctl(tmp_path, "b", store)
    driver_b = _Driver(rank_b)
    got = _cp(ctl_b, driver_b, _trainer(rank_b), _Ledger(3)).resume(driver_b)
    cut = rc.ConsistentCut.from_record(got[rc.CUT_KEY])
    assert got["next_rollout_id"] == 3 and driver_b.local_step == cut.island_steps[ISLAND] == 3
    for n, v in params(rank_a).items():
        assert torch.equal(v, params(rank_b)[n])


def test_resume_after_a_preemption_continues_from_the_last_periodic_cut(tmp_path):
    """A reclaim save never replaces the trainer cut: the next launch resumes from
    the newest complete round cut (LATEST) with the same steps."""
    store = _store(tmp_path / "store")
    rank_a, ctl_a, driver_a, cp_a = _trained_island(tmp_path, store, 2)
    periodic = rc.ConsistentCut.from_record(cp_a.save(driver_a, rollout_id=2)[rc.CUT_KEY])
    saver = s.InFlightSaver(lambda t: {"entries": []}, str(tmp_path / "inflight"), printer=lambda line: None)
    saver.rehearse(2)
    _reclaim_save(saver)

    rank_b = make_rank(5)
    ctl_b = _rctl(tmp_path, "b", store)
    driver_b = _Driver(rank_b)
    got = _cp(ctl_b, driver_b, _trainer(rank_b), _Ledger(2)).resume(driver_b)
    assert rs.read_latest(store.root)["cut_id"] == periodic.checkpoint_id == got["cut_id"]
    assert driver_b.local_step == periodic.island_steps[ISLAND] == 2
    assert got["consumed_prompt_ids_digest"] == periodic.completed_groups_digest


def test_same_save_code_on_another_backend_writes_the_same_cut(tmp_path):
    records = []
    for name, vol in (("local", None), ("modal", _Vol())):
        store = _store(tmp_path / name / "store", vol)
        _, _, driver, cp = _trained_island(tmp_path / name, store, 2)
        out = cp.save(driver, rollout_id=2)
        record = dict(out[rc.CUT_KEY])
        record["checkpoint_id"] = record["checkpoint_id"].split("-")[0]  # incarnation differs
        records.append((out["store_synced"], record))
        if vol is not None:
            assert vol.commits >= 1
    assert records[0] == records[1]


# ---------------------------------------------------------------- 3.3 --rl-elastic store
class _CountingStore(rs.CheckpointStore):
    def __init__(self, root):
        super().__init__(root)
        self.commits = 0
        self.reloads = 0

    def commit(self):
        self.commits += 1
        return 0.0

    def reload(self):
        self.reloads += 1


def test_elastic_controller_store_goes_through_the_interface(tmp_path):
    store = _CountingStore(tmp_path / "store")
    ctl = _ctl(tmp_path / "a/state", {"t": 1.0}, checkpoint_store=store)
    assert ctl.store is store and ctl.checkpoint_store == store.root
    commits = store.commits
    assert ctl.sync_checkpoint_store("test") and store.commits == commits + 1
    ctl.close()
    fresh = _ctl(tmp_path / "b/state", {"t": 2.0}, checkpoint_store=store)
    assert store.reloads >= 1 and (fresh.state_dir / "reconfig" / "journal.jsonl").is_file()
    fresh.close()
    by_path = _ctl(tmp_path / "c/state", {"t": 3.0}, checkpoint_store=str(tmp_path / "store2"))
    assert type(by_path.store) is rs.CheckpointStore
    by_path.close()


def test_elastic_commit_runs_outside_the_record_lock_and_its_failure_is_not_a_copy_failure(tmp_path):
    import threading

    seen = {}

    class _LockProbe(_CountingStore):
        def commit(self):
            probe = threading.Thread(target=lambda: seen.setdefault(
                "free", ctl._record_lock.acquire(timeout=1) and (ctl._record_lock.release() or True)))
            probe.start()
            probe.join()
            if seen.get("fail"):
                raise RuntimeError("volume busy")
            return 0.0

    store = _LockProbe(tmp_path / "store")
    ctl = _ctl(tmp_path / "a/state", {"t": 1.0}, checkpoint_store=store)
    assert ctl.sync_checkpoint_store("test") and seen["free"] is True
    assert ctl.last_store_sync["commit_error"] is None
    seen["fail"] = True
    assert ctl.sync_checkpoint_store("test")
    assert ctl.last_store_sync["ok"] is True and "volume busy" in ctl.last_store_sync["commit_error"]
    assert (store.root / rs.FLAT_MANIFEST).is_file()
    ctl.close()


def test_inflight_saver_commit_semantics_per_backend(tmp_path, monkeypatch):
    """Local store: nothing to commit (commit_s None). Modal store: the commit runs in
    the runner interpreter named by YETO_MODAL_PYTHON (a fake modal module here)."""
    import sys

    local = s.InFlightSaver(lambda t: {"entries": []}, str(tmp_path / "l"), printer=lambda line: None)
    assert local.rehearse(1)["commit_s"] is None

    fake = tmp_path / "pkg"
    fake.mkdir()
    log = tmp_path / "calls.txt"
    (fake / "modal.py").write_text(
        "class _V:\n"
        "    def __init__(self, n): self.n = n\n"
        f"    def commit(self): open({str(log)!r}, 'a').write('commit ' + self.n + '\\n')\n"
        "class Volume:\n"
        "    @staticmethod\n"
        "    def from_name(n): return _V(n)\n")
    monkeypatch.setenv("YETO_MODAL_PYTHON", sys.executable)
    monkeypatch.setenv("YETO_MODAL_SYSPATH", str(fake))
    modal = s.InFlightSaver(lambda t: {"entries": []}, str(tmp_path / "m"),
                            store=mcs.ModalVolumeStore(tmp_path / "m", "tape-vol"), printer=lambda line: None)
    out = modal.rehearse(1)
    assert out["commit_error"] is None and out["commit_s"] is not None
    assert log.read_text().splitlines() == ["commit tape-vol"]
