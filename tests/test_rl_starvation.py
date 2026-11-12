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
