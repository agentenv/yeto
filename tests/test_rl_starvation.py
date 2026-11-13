"""Trainer starvation: alert, then stop the trainer only when no sampler is healthy."""

from __future__ import annotations

import json
import types

import pytest

from yeto.rl import starvation as st
from yeto.rl import up

MIN = 60.0
HEALTHY = st.SamplerHealth(1, live=1)
NONE = st.SamplerHealth(0, live=0)


def _cfg(**kw):
    return st.StarvationConfig.from_section(kw)


def test_defaults_are_wide_and_zero_turns_off():
    cfg = _cfg()
    assert (cfg.alert_s, cfg.stop_s, cfg.unhealthy_s) == (30 * MIN, 120 * MIN, 10 * MIN)
    assert not _cfg(alert_min=0, stop_min=0).enabled
    for bad in ({"alert_min": -1}, {"stop_min": "2"}, {"alert_min": 60, "stop_min": 30}, {"nope": 1},
                {"check_s": 0}):
        with pytest.raises(ValueError):
            _cfg(**bad)


def test_alert_once_per_stretch_then_recovered():
    w = st.StarvationWatch(_cfg(), now=0)
    assert w.observe(29 * MIN, new_groups=0, health=HEALTHY).state == st.OK
    v = w.observe(31 * MIN, new_groups=0, health=HEALTHY)
    assert v.state == st.ALERT and v.new_alert and "31 min" in v.reason and "1 healthy" in v.reason
    assert "no stop" in v.reason
    assert not w.observe(40 * MIN, new_groups=None, health=HEALTHY).new_alert
    v = w.observe(41 * MIN, new_groups=3, health=HEALTHY)
    assert v.state == st.OK and v.recovered
    assert w.observe(72 * MIN, new_groups=0, health=HEALTHY).new_alert  # a new stretch alerts again


def test_long_agentic_trials_with_healthy_samplers_never_stop():
    w = st.StarvationWatch(_cfg(), now=0)
    for minute in range(0, 24 * 60, 5):
        assert w.observe(minute * MIN, new_groups=0, health=HEALTHY).state != st.STOP


def test_stop_needs_the_long_threshold_and_no_healthy_sampler_for_a_while():
    w = st.StarvationWatch(_cfg(), now=0)
    assert w.observe(60 * MIN, new_groups=0, health=HEALTHY).state == st.ALERT
    assert w.observe(115 * MIN, new_groups=0, health=NONE).state == st.ALERT  # unhealthy since 115
    v = w.observe(120 * MIN, new_groups=0, health=NONE)
    assert v.state == st.ALERT and "stopped at 120 min" in v.reason  # only 5 min without a sampler
    v = w.observe(125 * MIN, new_groups=0, health=NONE)
    assert v.state == st.STOP and "0 healthy" in v.reason


def test_a_brief_healthy_reading_restarts_the_unhealthy_clock():
    w = st.StarvationWatch(_cfg(), now=0)
    w.observe(100 * MIN, new_groups=0, health=NONE)
    w.observe(121 * MIN, new_groups=0, health=HEALTHY)
    assert w.observe(125 * MIN, new_groups=0, health=NONE).state == st.ALERT
    assert w.observe(135 * MIN, new_groups=0, health=NONE).state == st.STOP


@pytest.mark.parametrize("health", [st.SamplerHealth(None), st.SamplerHealth(0, paused=1, live=1)])
def test_unreadable_pool_state_or_a_paused_sampler_never_stops(health):
    w = st.StarvationWatch(_cfg(), now=0)
    for minute in (0, 60, 200, 400):
        v = w.observe(minute * MIN, new_groups=0, health=health)
    assert v.state == st.ALERT and "no stop" in v.reason


def test_stop_off_alerts_only_and_alert_off_still_stops():
    w = st.StarvationWatch(_cfg(stop_min=0), now=0)
    assert w.observe(500 * MIN, new_groups=0, health=NONE).state == st.ALERT
    w = st.StarvationWatch(_cfg(alert_min=0), now=0)
    w.observe(0, new_groups=0, health=NONE)
    assert w.observe(60 * MIN, new_groups=0, health=NONE).state == st.OK
    assert w.observe(121 * MIN, new_groups=0, health=NONE).state == st.STOP


def test_the_trainer_side_cause_is_named():
    w = st.StarvationWatch(_cfg(), now=0)
    v = w.observe(31 * MIN, new_groups=0, health=HEALTHY, cause="gate paused")
    assert "trainer side: gate paused" in v.reason


# -- yeto run up wiring ---------------------------------------------------------------

class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s


class FakeObjects:
    """The run's buffer on an object store: shard "a" holds ``run.keys``."""

    def __init__(self, run):
        self.run = run
        self.requests = []

    def _shards(self):
        return {"a": self.run.keys, **self.run.shards}

    def list(self, prefix, *, start_after=""):
        from yeto.rl.engine.sample_buffer import shard_prefix

        FakeObjects.calls.append(prefix)
        if self.run.fail(prefix):
            raise OSError("listing refused")
        if prefix.endswith("shard-names/"):
            return [prefix + s for s in sorted(self._shards())]
        for shard, keys in self._shards().items():
            if prefix == shard_prefix(shard):
                return sorted(k for k in (prefix + k for k in keys) if k > start_after)
        return []

    def list_page(self, prefix, *, start_after="", limit=1000):
        return self.list(prefix, start_after=start_after)[:limit], None


FakeObjects.calls = []


@pytest.fixture
def run_up(tmp_path, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
    monkeypatch.setattr(up, "time", types.SimpleNamespace(time=clock.time, sleep=clock.sleep,
                                                         gmtime=lambda *a: __import__("time").gmtime(*a),
                                                         strftime=lambda *a: __import__("time").strftime(*a)))
    from yeto import runs

    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    pool_cfg = tmp_path / "pool.json"
    pool_cfg.write_text(json.dumps({"shard_base": "s"}))
    nc_dir = tmp_path / "nc"
    nc_dir.mkdir()

    def build(starvation=None, members=None, pool_alive=True, keys=None):
        cfg = {"run": "r1", "store": "s3://b/runs/r1", "pool": {"config": str(pool_cfg)},
               "env": {"YETO_NC_STATE_DIR": str(nc_dir)},
               "trainer": {"argv": ["--gpu", "modal:2x8xb300", "--cluster-prefix", "r1-trainer"]},
               "budget": {"wall_h": 1000}}
        if starvation is not None:
            cfg["starvation"] = starvation
        run = up.Run(cfg)
        run.clock = clock
        run.calls = []
        run._cmd = lambda argv, *, log, timeout=None: run.calls.append(argv) or 0
        run.keys = set(keys or ())
        run.shards = {}  # shard -> keys (besides shard "a": run.keys)
        run.fail = lambda prefix: False
        run._objects = lambda: FakeObjects(run)
        run.pool_state = {"members": members if members is not None else []}
        (nc_dir / "pool-s.json").write_text(json.dumps(run.pool_state))
        monkeypatch.setattr("yeto.launch.nc.pool_run.pool_process",
                            lambda cfg: {"alive": run.pool_alive})
        run.pool_alive = pool_alive

        def set_members(ms):
            (nc_dir / "pool-s.json").write_text(json.dumps({"members": ms}))

        run.set_members = set_members
        return run

    return build


def test_sampler_health_reads_the_pool_state(run_up):
    run = run_up(members=[{"ticket": "a", "state": "ready"}, {"ticket": "b", "state": "ready", "paused": True},
                          {"ticket": "c", "state": "provisioning"}, {"ticket": "d", "state": "gone"}])
    h = run.sampler_health()
    assert (h.healthy, h.paused, h.live) == (1, 1, 3)
    run.pool_alive = False
    h = run.sampler_health()
    assert h.healthy == 0 and h.paused == 1 and "pool loop not running" in h.detail


def test_trainer_side_cause_reads_the_gate_pause_from_the_trainer_log(run_up, tmp_path):
    run = run_up()
    log = tmp_path / "trainer.log"
    run.state["procs"] = {"trainer": {"pid": 1, "log": str(log)}}
    log.write_text("x\nWARNING incarnation gate: training paused: lease of shard s-0 unreadable since t\n")
    assert "fail-closed pause" in run.trainer_side_cause()
    with log.open("a") as fh:
        fh.write("WARNING incarnation gate: lease of s-0 read again after 30 s\n")
    assert run.trainer_side_cause() is None


def test_starving_trainer_alerts_then_stops_through_the_stop_flag(run_up, monkeypatch, capsys):
    run = run_up(members=[{"ticket": "a", "state": "ready"}], keys={"k0"})
    alive = {"v": True}
    monkeypatch.setattr(up, "_alive", lambda pid: alive["v"])
    run.state["procs"] = {"trainer": {"pid": 42, "log": "/nonexistent"}}
    monkeypatch.setattr(run, "breach", lambda trainer=True: None)
    monkeypatch.setattr(run, "rollout_stall_alert", lambda: None)
    stopped = []

    def trainer_down(why):
        stopped.append((run.clock.t, why))

    run.trainer_down = trainer_down
    from yeto import runs

    t0 = run.clock.t
    # samplers die after 40 min; the launcher exits 5 min after the STOP flag
    real_sleep = run.clock.sleep

    def sleep(s):
        real_sleep(s)
        if run.clock.t - t0 >= 40 * MIN:
            run.set_members([{"ticket": "a", "state": "gone"}])
        if runs.stop_flag_path("r1-trainer").exists():
            alive["v"] = run.clock.t - t0 < stop_at[0] + 5 * MIN if stop_at else True
    stop_at: list[float] = []
    real_request = runs.request_stop
    monkeypatch.setattr(runs, "request_stop", lambda name: stop_at.append(run.clock.t - t0) or real_request(name))
    monkeypatch.setattr(up.time, "sleep", sleep)
    run.monitor(poll_s=60)
    out = capsys.readouterr().out
    assert "CRITICAL trainer_starvation_alert" in out
    assert "CRITICAL trainer_starvation_stop" in out
    assert stop_at and 120 * MIN <= stop_at[0] <= 125 * MIN
    assert len(stopped) == 1 and stopped[0][1].startswith("trainer_starvation_stop")
    assert stopped[0][0] - t0 - stop_at[0] <= 6 * MIN  # waited for the launcher, not the full grace
    assert run.state["starvation"]["event"] == "trainer_starvation_stop"
    assert not any("pool-stop" in argv for argv in run.calls)  # the pool is kept


def test_new_groups_keep_the_trainer_running(run_up, monkeypatch):
    run = run_up(members=[], keys={"k0"})
    monkeypatch.setattr(up, "_alive", lambda pid: True)
    run.state["procs"] = {"trainer": {"pid": 42, "log": "/nonexistent"}}
    for i in range(60):  # 5 h, a new group every 5 min while the pool reads empty
        run.keys.add(f"k{i + 1:03d}")
        assert run.starvation_check(run.clock.t) is None
        run.clock.t += 5 * MIN
    assert "starvation" not in run.state


def test_pruned_keys_are_no_new_group(run_up, monkeypatch):
    run = run_up(members=[], keys={f"k{i}" for i in range(10)})
    run.state["procs"] = {"trainer": {"pid": 42, "log": "/nonexistent"}}
    why = None
    for _ in range(30):
        run.keys = set(sorted(run.keys)[1:])  # consumed objects pruned, nothing new
        why = run.starvation_check(run.clock.t) or why
        run.clock.t += 5 * MIN
    assert why is not None


def test_starvation_off_and_bad_section_refused_by_preflight(run_up):
    run = run_up(starvation={"alert_min": 0, "stop_min": 0})
    assert run.starvation_check(run.clock.t) is None
    run = run_up(starvation={"stop_min": -1})
    assert "stop_min" in run.starvation_problem()


def _starve(run, polls=40, step=5 * MIN):
    run.state["procs"] = {"trainer": {"pid": 42, "log": "/nonexistent"}}
    why = None
    for _ in range(polls):
        why = run.starvation_check(run.clock.t) or why
        run.clock.t += step
    return why


def test_a_failing_buffer_listing_never_stops(run_up):
    run = run_up(members=[])
    run.fail = lambda prefix: True
    assert _starve(run) is None
    assert "listing failed" in run.state["starvation"]["why"]


def test_a_listing_that_fails_after_a_good_one_never_stops(run_up):
    run = run_up(members=[], keys={"k0"})
    ok = iter([True] * 4)  # the baseline (names, the shard, its page) and one page: then refused
    run.fail = lambda prefix: not next(ok, False)
    assert _starve(run) is None


def test_a_paused_member_in_any_state_blocks_the_stop(run_up):
    run = run_up(members=[{"ticket": "a", "state": "starting", "paused": True}], keys={"k0"})
    assert _starve(run) is None


def test_one_unreadable_pool_blocks_the_stop(run_up):
    run = run_up(members=[], keys={"k0"})
    run.pools = [run.pools[0], types.SimpleNamespace(name="p2", load=lambda: {"shard_base": "t"})]
    first = run._members
    run._members = lambda i: first(0) if i == 0 else None
    assert run.sampler_health().healthy is None
    assert _starve(run) is None


def test_the_stop_lists_the_buffer_again_first(run_up):
    run = run_up(starvation={"check_s": 3000}, members=[], keys={"k0"})  # listings at 0, 50, 100, 150 min
    run.state["procs"] = {"trainer": {"pid": 42, "log": "/nonexistent"}}
    t0 = run.clock.t
    while run.clock.t - t0 < 119 * MIN:
        assert run.starvation_check(run.clock.t) is None
        run.clock.t += MIN
    run.keys.add("k1")  # a group lands after the 100-min listing, right before the stop
    run.clock.t = t0 + 125 * MIN
    assert run.starvation_check(run.clock.t) is None


def _check(run, minutes=5):
    run.clock.t += minutes * MIN
    run._starvation_listed = None  # list on this call
    return run.starvation_check(run.clock.t)


def test_incremental_listing_sees_new_keys_and_new_shards(run_up):
    run = run_up(members=[{"ticket": "a", "state": "ready"}], keys={"k000"})
    assert run.buffer_new_groups() is None  # the baseline
    assert run.buffer_new_groups() == 0
    run.keys.add("k001")
    assert run.buffer_new_groups() == 1  # an old shard's new key
    run.shards["b"] = {"k000", "k001"}
    found = [run.buffer_new_groups() for _ in range(run.SHARD_SCAN_EVERY)]
    assert sum(found) == 2 and found.count(2) == 1  # the new shard: found on the next scan of the names
    FakeObjects.calls.clear()
    run.buffer_new_groups()  # no scan: one page request per known shard
    assert len(FakeObjects.calls) == 2 and not any("shard-names" in c for c in FakeObjects.calls)


def test_a_partly_failed_listing_is_no_evidence(run_up):
    run = run_up(members=[], keys={"k000"})
    run.shards["b"] = {"k000"}
    assert run.buffer_new_groups() is None
    run.fail = lambda prefix: prefix.endswith("/b/")
    assert run.buffer_new_groups() is None  # one shard refused: the whole listing failed
    run.fail = lambda prefix: False
    run.shards["b"].add("k001")
    assert run.buffer_new_groups() == 1  # nothing lost
    run.fail = lambda prefix: prefix.endswith("/b/")
    run.state["procs"] = {"trainer": {"pid": 42, "log": "/nonexistent"}}
    why = None
    for _ in range(40):
        why = _check(run) or why
    assert why is None


def test_a_failed_baseline_is_taken_again(run_up):
    run = run_up(members=[], keys={"k000"})
    run.fail = lambda prefix: True
    assert run.buffer_new_groups() is None
    run.fail = lambda prefix: False
    assert run.buffer_new_groups() is None  # the baseline now
    run.keys.add("k001")
    assert run.buffer_new_groups() == 1


def test_a_replacement_under_way_holds_the_stop_until_its_start_timeout(run_up, tmp_path):
    (tmp_path / "pool.json").write_text(json.dumps({"shard_base": "s", "elastic": {"start_timeout_s": 3600}}))
    run = run_up(members=[], keys={"k000"})
    run.state["procs"] = {"trainer": {"pid": 42, "log": "/nonexistent"}}
    t0 = run.clock.t
    run.set_members([{"ticket": "a", "state": "gone"}])
    assert run.starvation_check(run.clock.t) is None
    for _ in range(23):  # 115 min, no sampler
        assert _check(run) is None
    run.set_members([{"ticket": "b", "state": "starting", "created": run.clock.t}])
    h = run.sampler_health(run.clock.t)
    assert h.recovering == 1 and "recovering: 1 member(s) starting" in h.phrase()
    while run.clock.t - t0 < 170 * MIN:
        assert _check(run) is None  # starting, within its start timeout: no stop
    why = None
    while why is None and run.clock.t - t0 < 300 * MIN:
        why = _check(run)
    assert why is not None and 175 * MIN <= run.clock.t - t0 <= 185 * MIN  # stuck past 3600 s: stop


def test_a_pool_that_reorders_forever_is_stoppable_after_its_start_timeout(run_up, tmp_path):
    (tmp_path / "pool.json").write_text(json.dumps({"shard_base": "s", "elastic": {"start_timeout_s": 1800}}))
    run = run_up(members=[], keys={"k000"})
    run.state["procs"] = {"trainer": {"pid": 42, "log": "/nonexistent"}}
    t0 = run.clock.t
    assert run.starvation_check(run.clock.t) is None
    why = None
    while why is None and run.clock.t - t0 < 400 * MIN:
        run.set_members([{"ticket": f"t{int(run.clock.t)}", "state": "provisioning", "created": run.clock.t}])
        why = _check(run)
    assert why is not None and run.clock.t - t0 <= 125 * MIN  # 30 min of fresh orders: still stoppable at 120


def test_the_alert_names_the_recovery_and_holds_no_stop():
    w = st.StarvationWatch(_cfg(), now=0)
    h = st.SamplerHealth(0, live=1, recovering=2, recover_s=3600)
    v = w.observe(60 * MIN, new_groups=0, health=h)
    assert v.state == st.ALERT and "recovering: 2 member(s) starting" in v.reason and "no stop" in v.reason


def test_trainer_side_cause_names_a_fatal_store_exit(run_up, tmp_path):
    from yeto.rl.engine.incarnation_lease import FATAL_MARK

    run = run_up()
    log = tmp_path / "trainer.log"
    run.state["procs"] = {"trainer": {"pid": 1, "log": str(log)}}
    log.write_text("x\nWARNING incarnation gate: training paused: lease of shard s-0 unreadable since t\n"
                   f"ERROR {FATAL_MARK}: lease of shard s-0 refused by the store (AccessDenied): ...\n")
    cause = run.trainer_side_cause()
    assert "AccessDenied" in cause and "trainer exits" in cause and "fail-closed pause" not in cause


# -- the data end: every pool finished, the shards no sampler ended are ended
def _data_end_run(run_up, tmp_path, monkeypatch, *, finished=True, members=None):
    from yeto.rl.engine.object_store import DirectoryObjectStore

    run = run_up(members=[])
    run.cfg["trainer"]["argv"] += ["--rl-buffer-shards", "s-0,s-1,s-2,s-3"]
    doc = {"members": members if members is not None else [{"ticket": "a", "state": "gone"},
                                                           {"ticket": "b", "state": "gone"}]}
    if finished:
        doc["finished"] = "sampler s-0 finished (exit 0)"
    (tmp_path / "nc" / "pool-s.json").write_text(json.dumps(doc))
    objects = DirectoryObjectStore(tmp_path / "store")
    run._objects = lambda: objects
    _dispense(objects, slots=2, settled=2)
    return run, objects


def _dispense(objects, *, slots, settled):
    """The store dispenser's ledger: ``slots`` tasks, the first ``settled`` trained."""
    from yeto.rl.engine import store_dispense as sd

    ledger = sd.ledger_objects(objects)
    ledger.write("task-dispense/plan.json", json.dumps({"slots": slots, "rows": slots}).encode())
    for p in range(settled):
        ledger.write(f"task-dispense/done/{sd._pkey(p)}.trained", b"{}")


def _write_groups(objects, shard, n, *, end):
    from yeto.rl.engine.buffer_payload import encode_group
    from yeto.rl.engine.sample_buffer import FileSampleBuffer, shard_index
    from yeto.rl.engine.trainer_lifecycle import SampleIndexEntry

    writer = FileSampleBuffer(objects, batch_size=1, index=shard_index(shard))
    for i in range(n):
        ref = writer.write_payload(f"{shard}-{i}", encode_group([{"i": i}]))
        writer.append(SampleIndexEntry(f"{shard}-{i}", 1, ref, 8, float(i + 1)))
    if end:
        writer.mark_data_end()


def test_a_finished_pool_ends_the_shards_no_sampler_ended(run_up, tmp_path, monkeypatch):
    """s-0 ended on the data end, s-1 was fenced (no end record), s-2 / s-3 were never
    ordered: without the seal the trainer's merged index never ends."""
    from yeto.rl.engine.sample_buffer import FileSampleBuffer

    run, objects = _data_end_run(run_up, tmp_path, monkeypatch)
    _write_groups(objects, "s-0", 3, end=True)
    _write_groups(objects, "s-1", 2, end=False)
    trainer = FileSampleBuffer(objects, batch_size=1, merge_shards=True,
                               expected_shards=["s-0", "s-1", "s-2", "s-3"])
    assert not trainer.data_ended  # the defect: the data end never reaches the trainer
    assert run.seal_data_end()
    trainer.refresh()
    assert trainer.data_ended and len(trainer) == 5  # every group kept, then the end
    assert sorted(run.state["data_ended"]["sealed"]) == ["s-1", "s-2", "s-3"]
    assert run.seal_data_end()  # idempotent: nothing written twice


@pytest.mark.parametrize("finished,members", [(False, None), (True, [{"ticket": "a", "state": "ready"}]),
                                              (True, [{"ticket": "a", "state": "draining"}])])
def test_no_seal_while_a_pool_is_not_finished_or_holds_a_member(run_up, tmp_path, monkeypatch,
                                                                finished, members):
    from yeto.rl.engine.sample_buffer import FileSampleBuffer

    run, objects = _data_end_run(run_up, tmp_path, monkeypatch, finished=finished, members=members)
    _write_groups(objects, "s-1", 2, end=False)
    assert not run.seal_data_end()
    trainer = FileSampleBuffer(objects, batch_size=1, merge_shards=True, expected_shards=["s-1"])
    assert not trainer.data_ended and "data_ended" not in run.state


def test_the_data_end_is_no_starvation(run_up, tmp_path, monkeypatch, capsys):
    run, objects = _data_end_run(run_up, tmp_path, monkeypatch)
    _write_groups(objects, "s-0", 1, end=True)
    monkeypatch.setattr(up, "_alive", lambda pid: run.clock.t < t0 + 300 * MIN)
    run.state["procs"] = {"trainer": {"pid": 42, "log": "/nonexistent"}}
    monkeypatch.setattr(run, "breach", lambda trainer=True: None)
    monkeypatch.setattr(run, "rollout_stall_alert", lambda: None)
    monkeypatch.setattr(run, "on_trainer_finished", lambda: True)
    run.trainer_down = lambda why: pytest.fail(f"trainer stopped: {why}")
    t0 = run.clock.t

    def sleep(s):
        run.clock.t += s
        if run.clock.t >= t0 + 300 * MIN:
            run.state["trainer_finished"] = "now"
    monkeypatch.setattr(up.time, "sleep", sleep)
    run.monitor(poll_s=60)
    out = capsys.readouterr().out
    assert "trainer_starvation" not in out and "data end: every sampler pool finished" in out


def test_no_seal_when_a_pool_writes_through_the_syncer(run_up, tmp_path, monkeypatch):
    """A syncer data path: the trainer reads the relay, not the store's shards. An end
    record in the store reaches no trainer, and the seal must not switch the starvation
    watch off."""
    import dataclasses

    run, objects = _data_end_run(run_up, tmp_path, monkeypatch)
    run.pools = [dataclasses.replace(run.pools[0], data_path="syncer")]
    assert not run.seal_data_end()
    assert "data_ended" not in run.state and not objects.list("sample-buffer/")


def test_the_sealed_shards_parse_like_the_trainer(run_up, tmp_path, monkeypatch):
    """The trainer strips the names of --rl-buffer-shards: "s-0, s-1" names s-1, not "_s-1"."""
    from yeto.rl.engine.sample_buffer import FileSampleBuffer

    run, objects = _data_end_run(run_up, tmp_path, monkeypatch)
    run.cfg["trainer"]["argv"] += ["--rl-buffer-shards", "s-0, s-1"]
    assert run.seal_data_end()
    assert sorted(run.state["data_ended"]["sealed"]) == ["s-0", "s-1"]
    trainer = FileSampleBuffer(objects, batch_size=1, merge_shards=True, expected_shards=["s-0", "s-1"])
    trainer.refresh()
    assert trainer.data_ended


def test_a_half_written_pool_state_is_no_data_end(run_up, tmp_path, monkeypatch):
    run, objects = _data_end_run(run_up, tmp_path, monkeypatch)
    (tmp_path / "nc" / "pool-s.json").write_text('{"finished": "x", "members": [')
    assert not run.seal_data_end() and "data_ended" not in run.state


def test_a_group_after_the_seal_fails_the_writer_not_the_trainer(run_up, tmp_path, monkeypatch):
    """A writer left behind after the seal: its append is refused at the writer (the
    end record holds its position), the trainer's merged index stays consistent."""
    from yeto.rl.engine.buffer_payload import encode_group
    from yeto.rl.engine.sample_buffer import FileSampleBuffer, SampleBufferError, shard_index
    from yeto.rl.engine.trainer_lifecycle import SampleIndexEntry

    run, objects = _data_end_run(run_up, tmp_path, monkeypatch)
    zombie = FileSampleBuffer(objects, batch_size=1, index=shard_index("s-1"))  # opened before the seal
    trainer = FileSampleBuffer(objects, batch_size=1, merge_shards=True,
                               expected_shards=["s-0", "s-1", "s-2", "s-3"])
    assert run.seal_data_end()
    ref = zombie.write_payload("late", encode_group([{"i": 0}]))
    with pytest.raises(SampleBufferError, match="after the data end"):
        zombie.append(SampleIndexEntry("late", 1, ref, 8, 1.0))
    trainer.refresh()
    assert trainer.data_ended and len(trainer) == 0


def test_no_seal_when_a_pool_writes_another_store(run_up, tmp_path, monkeypatch):
    """Samplers that write a synced trainer's substore (<store>/trainer0): an end record
    in the run's store reaches no trainer."""
    run, objects = _data_end_run(run_up, tmp_path, monkeypatch)
    env = tmp_path / "r2.env"
    env.write_text("R2_BUCKET=b\n")
    (tmp_path / "pool.json").write_text(json.dumps({"shard_base": "s", "run_prefix": "r1/trainer0",
                                                    "r2_env_file": str(env)}))
    assert not run.seal_data_end() and "data_ended" not in run.state
    (tmp_path / "pool.json").write_text(json.dumps({"shard_base": "s", "run_prefix": "r1",
                                                    "r2_env_file": str(env)}))
    assert run.seal_data_end()


def test_no_seal_while_the_dispenser_holds_tasks(run_up, tmp_path, monkeypatch, capsys):
    """The pool finished (one sampler ended early, the rest went at their lease end) but
    tasks are left: no data end, a CRITICAL alert (once), the starvation watch stays on."""
    from yeto.rl.engine import store_dispense as sd

    run, objects = _data_end_run(run_up, tmp_path, monkeypatch)
    _dispense(objects, slots=5, settled=2)
    assert not run.seal_data_end() and not run.seal_data_end()
    out = capsys.readouterr().out
    assert out.count("CRITICAL data_end_incomplete") == 1 and "3 task(s) are not sampled" in out
    assert "`yeto run up`" in out and run.state["data_end_incomplete"]["left"] == 3
    assert "data_ended" not in run.state and not objects.list("sample-buffer/")
    for p in range(2, 5):  # the pool came back and sampled the rest: sealed now
        sd.ledger_objects(objects).write(f"task-dispense/done/{sd._pkey(p)}.trained", b"{}")
    assert run.seal_data_end() and "data_end_incomplete" not in run.state


def test_no_seal_without_a_dispenser_plan(run_up, tmp_path, monkeypatch):
    import shutil

    run, objects = _data_end_run(run_up, tmp_path, monkeypatch)
    shutil.rmtree(tmp_path / "store" / "task-dispense")
    assert not run.seal_data_end() and "data_ended" not in run.state


def test_no_seal_when_the_dispenser_status_fails(run_up, tmp_path, monkeypatch, capsys):
    from yeto.rl.engine import store_dispense as sd

    run, objects = _data_end_run(run_up, tmp_path, monkeypatch)

    def fail(*a, **k):
        raise OSError("listing failed")
    monkeypatch.setattr(sd, "status", fail)
    assert not run.seal_data_end() and "data_ended" not in run.state
    assert "did not read" in capsys.readouterr().out and not objects.list("sample-buffer/")


def test_the_starvation_watch_stays_on_with_tasks_left(run_up, tmp_path, monkeypatch, capsys):
    run, objects = _data_end_run(run_up, tmp_path, monkeypatch)
    _dispense(objects, slots=5, settled=2)
    monkeypatch.setattr(up, "_alive", lambda pid: True)
    run.state["procs"] = {"trainer": {"pid": 42, "log": "/nonexistent"}}
    monkeypatch.setattr(run, "breach", lambda trainer=True: None)
    monkeypatch.setattr(run, "rollout_stall_alert", lambda: None)
    checked = []
    monkeypatch.setattr(run, "starvation_check", lambda now: checked.append(now) or "starved")
    monkeypatch.setattr(run, "starvation_stop", lambda why: run.state.update(trainer_finished="stopped"))
    monkeypatch.setattr(run, "on_trainer_finished", lambda: True)
    run.monitor(poll_s=60)
    assert checked and "data_ended" not in run.state


def test_a_starvation_stop_downs_the_trainer_by_its_launch_name(run_up, monkeypatch):
    """End to end: starvation -> STOP flag -> the real ``trainer_down`` runs
    ``yeto down <--cluster-prefix>`` (not the run config's ``run``); the pool is kept."""
    from yeto import runs

    run = run_up(members=[{"ticket": "a", "state": "ready"}], keys={"k0"})
    monkeypatch.setattr(up, "_alive", lambda pid: not runs.stop_flag_path("r1-trainer").exists())
    run.state["procs"] = {"trainer": {"pid": 42, "log": "/nonexistent"}}
    monkeypatch.setattr(run, "breach", lambda trainer=True: None)
    monkeypatch.setattr(run, "rollout_stall_alert", lambda: None)
    t0, real_sleep = run.clock.t, run.clock.sleep

    def sleep(s):
        real_sleep(s)
        if run.clock.t - t0 >= 40 * MIN:  # the samplers die
            run.set_members([{"ticket": "a", "state": "gone"}])
        assert run.clock.t - t0 < 24 * 60 * MIN, "no starvation stop"

    monkeypatch.setattr(up.time, "sleep", sleep)
    run.monitor(poll_s=60)
    assert runs.stop_flag_path("r1-trainer").exists() and not runs.stop_flag_path("r1").exists()
    downs = [argv[-2:] for argv in run.calls if "down" in argv]
    assert downs == [["down", "r1-trainer"]]
    assert not any("pool-stop" in argv for argv in run.calls)


def test_the_data_end_seal_takes_an_owned_shard_first(run_up, tmp_path, monkeypatch):
    """A shard under ownership records takes only its owner's end record: the seal
    (no writer is left) claims the shard, then ends it; a fenced sampler's own end
    record would be void."""
    from yeto.rl.engine.sample_buffer import FileSampleBuffer, open_trainer_buffer, shard_index
    from yeto.rl.engine.trainer_lease import TrainerLease

    run, objects = _data_end_run(run_up, tmp_path, monkeypatch)
    TrainerLease.claim(objects, incarnation="trainer-1")
    _write_groups(objects, "s-0", 1, end=True)
    writer = FileSampleBuffer(objects, batch_size=1, index=shard_index("s-1"))
    writer.claim_owner("inc-1")
    writer.claim_owner("pool-fence:t-1", by="pool")  # its sampler was fenced, never ended it
    assert run.seal_data_end()
    trainer = open_trainer_buffer(objects, batch_size=1, shards=["s-0", "s-1", "s-2", "s-3"])
    trainer.refresh()
    assert trainer.data_ended
    owners = FileSampleBuffer(objects, batch_size=1, index=shard_index("s-1"))
    assert owners.owner.incarnation == up.SEAL_OWNER and owners.data_ended
