"""rl-resume-from-checkpoint (S17 C4), CPU only: store + LATEST + hashes, resume checks,
round cuts without --rl-elastic (save -> machine gone -> resume, real MilesTrainerGroup
cut/restore on E2's CPU rank fakes), launcher/Modal wiring, tape continuity, dashboard."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from test_rl_multinode_round_cut import _Driver, _hash, _Pool, _trainer
from test_rl_reconfig_recovery import _ctl
from test_rl_trainer_rebuild_e1 import ARGS

from tests.rl_cut_fakes import make_rank, params, train_step
from yeto.rl.adapters.miles.rebuild_wiring import CutSource
from yeto.rl.adapters.miles.round_cut import POINTER, RoundCutCheckpoint, RoundCutError
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
    return rs.ModalVolumeStore(path, "yeto-ckpt", volume_factory=lambda name: vol)


def _rctl(tmp_path, name, store, keep=2):
    return rs.ResumeController(state_dir=tmp_path / name / "state", store=store, keep=keep)


def _cp(ctl, driver, trainer, ledger, **kw):
    source = CutSource(driver=lambda: driver, trainer=trainer, rollout=_Pool(), ledger=ledger,
                       algorithm=SimpleNamespace(sha256=lambda: "a" * 64, to_legacy_runtime_attrs=lambda: {}),
                       backend_fingerprint="miles@0af62f4d", cut_root="", global_batch_size=ARGS.global_batch_size)
    kw.setdefault("fingerprint", dict(FP))
    kw.setdefault("next_lr", lambda d: getattr(d, "last_applied_lrs", None))
    return RoundCutCheckpoint(controller=ctl, source=source, trainer=trainer, **kw)


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
    assert isinstance(rs.store_for("/x", environ={rs.MODAL_VOLUME_ENV: "v"}), rs.ModalVolumeStore)
    assert type(rs.store_for("/mnt/share", environ={})) is rs.CheckpointStore
    assert isinstance(rs.store_for("~/yeto-checkpoint-store/p", environ={}), rs.BucketMountStore)
    assert rs.parse_modal_volume_uri("modal-volume://yeto-ckpt/run1/a") == ("yeto-ckpt", "run1/a")
    for bad in ("s3://b/x", "modal-volume://", "modal-volume://a b", "modal-volume://v/../x"):
        with pytest.raises(ValueError):
            rs.parse_modal_volume_uri(bad)


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


# ---------------------------------------------------------------- round cut: save -> kill -> resume
def test_save_kill_resume_without_elastic_continues_bitwise(tmp_path):
    """tasks 5.2: fake trainer, save -> the machine is gone -> a fresh trainer with
    other weights resumes; state hash, version, lr and consumed groups continue."""
    vol = _Vol()
    store = _store(tmp_path / "store", vol)
    rank_a, ctl_a, driver_a, cp_a = _trained_island(tmp_path, store, 2)
    info = cp_a.save(driver_a, rollout_id=2)
    assert info["store_synced"] and info["cut_bytes"] > 0 and info["lr_at_next_round"] == [5e-6]
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
    cp_a.save(driver_a, rollout_id=2)
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


def test_resume_refuses_lr_jump_and_ledger_drift(tmp_path):
    store = _store(tmp_path / "store")
    _r, _c, driver_a, cp_a = _trained_island(tmp_path, store, 2)
    cp_a.save(driver_a, rollout_id=2)
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
        if cp.save(driver, rollout_id=rid) is not None:
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
    sync.finish(driver)
    assert (3, True) in calls and ("check", 1) in calls


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
    assert launcher._check_resume_launch_flags(args, "ports") is True
    multi = _args(["--gpu", "modal:1xh100", "--training-mode", "rl", "--rl-checkpoint-store", "modal-volume://v"])
    with pytest.raises(ValueError, match="single-island-no-sync"):
        launcher._check_resume_launch_flags(multi, "ports")
    with pytest.raises(ValueError, match="needs --rl-checkpoint-store"):
        launcher._check_resume_launch_flags(_args(["--rl-cut-every", "2"]), "ports")


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
    assert cfg.envs[rs.MODAL_VOLUME_ENV] == "yeto-ckpt"
    cfg.validate()
    from yeto import modal_runner as mr

    assert mr.ModalIslandConfig.from_json(cfg.to_json()) == cfg


def test_learner_resume_switches(tmp_path):
    from test_rl_engine_selection import _learner_argv

    from yeto.rl import learner

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
