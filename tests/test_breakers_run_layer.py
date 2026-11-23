"""The run layer's breakers (``yeto.breakers.layers.run``): ``run up`` reads them from the
registry, the legacy run config keys stay aliases, and trips go out as ``rl_breaker``."""

from __future__ import annotations

import json
import types

import pytest

from yeto import breakers
from yeto.launch.pool.controller import DEFAULT_STALL_S
from yeto.rl import release_stall as rs
from yeto.rl import starvation as st
from yeto.rl import up, up_consistency

MIN = 60.0


def _cfg(**sections):
    cfg = {"run": "r1", "store": "s3://b/runs/r1", "pool": {"config": "pool.json"},
           "trainer": {"argv": ["--gpu", "modal:2x8xb300", "--cluster-prefix", "r1-trainer"], "usd_per_h": 100}}
    cfg.update(sections)
    return cfg


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s


@pytest.fixture
def make_run(tmp_path, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
    monkeypatch.setattr(up, "time", types.SimpleNamespace(time=clock.time, monotonic=clock.time, sleep=clock.sleep,
                                                         gmtime=lambda *a: __import__("time").gmtime(*a),
                                                         strftime=lambda *a: __import__("time").strftime(*a)))
    from yeto import runs

    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    pool_cfg = tmp_path / "pool.json"
    pool_cfg.write_text(json.dumps({"shard_base": "s"}))

    def build(**sections):
        cfg = _cfg(**sections)
        cfg["pool"] = {"config": str(pool_cfg)}
        run = up.Run(cfg)
        run.clock = clock
        run.calls = []
        run._cmd = lambda argv, *, log, timeout=None: run.calls.append(argv) or 0
        return run

    return build


def _events(run):
    path = run.dir / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


# ----------------------------------------------------------------------------- declarations
def test_the_run_layer_loads_and_its_defaults_are_the_code_constants():
    reg = breakers.default_registry()
    rb = breakers.resolve(breakers.Sources(run_config=_cfg()), layers=("run",))
    assert rb.get("run.wall_clock.hours") == up_consistency.DEFAULT_WALL_H
    assert rb.get("run.budget.force_pools_ratio") == up.FORCE_FACTOR
    assert rb.get("run.first_group.timeout_s") == up.DEFAULT_FIRST_GROUP_TIMEOUT_S
    assert rb.get("run.trainer_silence.floor_s") == up.DEFAULT_SILENCE_MIN * 60
    assert rb.get("run.trainer_silence.stall_factor_n") == up.SILENCE_STALL_FACTOR
    assert rb.get("run.rollout_stall_alert.limit_s") == DEFAULT_STALL_S
    default = st.StarvationConfig()
    for param in ("alert_s", "stop_s", "unhealthy_s", "trainer_wait_max_s", "stop_grace_s", "check_s"):
        assert rb.get(f"run.starvation.{param}") == getattr(default, param), param
    assert rb.get("run.release_stall.check_s") == rs.DEFAULT_CHECK_S
    for name in ("run.trainer_exit", "run.controller_exit"):
        assert reg.get(name).kind == "invariant" and not reg.get(name).can_disable


def test_without_a_section_the_legacy_keys_resolve_as_today():
    rb = breakers.resolve(breakers.Sources(run_config=_cfg(
        budget={"usd": "45", "silence_min": None, "release_nc_on_breach": True},
        starvation={"alert_min": 0, "stop_min": 45, "unhealthy_min": 5},
        rollout={"stall_alert_s": 0}, sandbox={"max_run_containers": 0, "max_containers_per_sampler": 12},
        first_group={"timeout_s": 3600})), layers=("run",))
    assert not rb.has_section and not rb.conflicts
    assert rb.enabled("run.budget") and rb.get("run.budget.usd") == 45.0
    assert rb.get("run.budget.release_pools_on_trip") is True
    assert rb.entry("run.wall_clock").sources["hours"].location == "default (implicit)"
    assert rb.get("run.trainer_silence.floor_s") == 1800.0  # silence_min null: the default
    assert rb.get("run.starvation.alert_s") is None and rb.get("run.starvation.stop_s") == 2700.0
    assert rb.get("run.starvation.unhealthy_s") == 300.0
    assert not rb.enabled("run.rollout_stall_alert")
    assert rb.get("run.sandbox_containers.max_run_n") is None
    assert rb.get("run.sandbox_containers.max_per_sampler_n") == 12
    assert rb.get("run.first_group.timeout_s") == 3600.0


@pytest.mark.parametrize("budget, usd_on, wall", [({}, False, 3.0), ({"usd": None, "wall_h": None}, False, None),
                                                  ({"usd": 0, "wall_h": 1.5}, True, 1.5)])
def test_budget_and_wall_clock_off_spellings(budget, usd_on, wall):
    rb = breakers.resolve(breakers.Sources(run_config=_cfg(budget=budget)), layers=("run",))
    assert rb.enabled("run.budget") is usd_on
    assert (rb.get("run.wall_clock.hours") if rb.enabled("run.wall_clock") else None) == wall


# ----------------------------------------------------------------------------- run up reads the registry
def test_the_section_sets_what_run_up_applies(make_run):
    run = make_run(breakers={"run": {
        "wall_clock": {"hours": 6}, "budget": {"usd": 450, "release_pools_on_trip": True},
        "trainer_silence": {"limit_s": 5400}, "first_group": {"timeout_s": 600},
        "starvation": {"alert_s": 600, "stop_s": None}, "rollout_stall_alert": {"enabled": False},
        "sandbox_containers": {"max_run_n": 30}}})
    assert run.uptime()["wall_h"] == 6
    assert run.breaker_param("run.budget", "usd") == 450 and run.release_nc_on_breach
    assert run.silence_limit_s() == 5400
    cfg = run.starvation_config
    assert (cfg.alert_s, cfg.stop_s) == (600, 0)  # stop_s null: no stop verdict
    assert run.breaker_param("run.rollout_stall_alert", "limit_s") is None
    assert run.container_limit() == 30
    assert run.breaker_param("run.first_group", "timeout_s") == 600


def test_the_silence_limit_is_derived_from_the_stall_timeout_unless_set(make_run):
    run = make_run(budget={"silence_min": 90})
    run.cfg["trainer"]["argv"] += ["--rl-stall-timeout", "600"]
    assert run.silence_limit_s() == 90 * MIN  # floor above 4 x 600 s
    run.cfg["trainer"]["argv"][-1] = "3600"
    assert run.silence_limit_s() == 4 * 3600
    run.cfg["breakers"] = {"run": {"trainer_silence": {"limit_s": 1200}}}
    assert run.silence_limit_s() == 1200


def test_a_silence_breaker_turned_off_never_trips(make_run):
    run = make_run(breakers={"run": {"trainer_silence": {"enabled": False}}})
    run.state["procs"] = {"trainer": {"pid": 1, "log": str(run.dir / "trainer.log")}}
    assert run.silence(0.0) is None
    assert run.silence(10 ** 7) is None


def test_wall_clock_turned_off_by_the_section_never_breaches(make_run, monkeypatch):
    run = make_run(budget={"usd": 10 ** 6}, breakers={"run": {"wall_clock": {"enabled": False}}})
    run.state["t0"] = run.clock.t - 100 * 3600
    monkeypatch.setattr(up.Run, "spend_usd", lambda self: 0.0)
    assert run.breach(trainer=False) is None


def test_the_implicit_wall_clock_line_follows_the_registry_source(make_run, monkeypatch, capsys):
    monkeypatch.setattr("yeto.launch_preflight.modal_sdk_problem", lambda env: None)
    assert make_run(budget={"usd": 450}).preflight()
    assert "WARNING: budget.wall_h is not set" in capsys.readouterr().out
    assert make_run(budget={"usd": 450}, breakers={"run": {"wall_clock": {"hours": 6}}}).preflight()
    assert "budget.wall_h is not set" not in capsys.readouterr().out


def test_a_section_that_disagrees_with_a_legacy_key_is_refused_by_preflight(make_run, monkeypatch, capsys):
    monkeypatch.setattr("yeto.launch_preflight.modal_sdk_problem", lambda env: None)
    run = make_run(budget={"usd": 45}, breakers={"run": {"budget": {"usd": 450}}})
    assert run.preflight() is False and run.calls == []
    out = capsys.readouterr().out
    assert "breakers:" in out and "budget.usd" in out
    run = make_run(budget={"usd": 450}, breakers={"run": {"budget": {"usd": 450}}})  # agreeing: fine
    assert run.preflight() is True


def test_load_config_checks_the_breakers_section(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(_cfg(breakers={"run": {"wall_clock": {"hours": 6}}})))
    assert up.load_config(str(path))["breakers"]["run"]["wall_clock"]["hours"] == 6
    path.write_text(json.dumps(_cfg(breakers={"run": {"wall_clok": {"hours": 6}}})))
    with pytest.raises(ValueError, match=r"^run config: .*wall_clok"):
        up.load_config(str(path))
    path.write_text(json.dumps(_cfg(breakers={"run": {"trainer_exit": {"enabled": False}}})))
    with pytest.raises(ValueError, match="run config: "):
        up.load_config(str(path))


# ----------------------------------------------------------------------------- trips
def test_a_wall_clock_breach_reports_its_stop_reason(make_run, monkeypatch):
    run = make_run(budget={"usd": 10 ** 6, "wall_h": 2})
    run.state["t0"] = run.clock.t - 3 * 3600
    monkeypatch.setattr(up.Run, "spend_usd", lambda self: 0.0)
    assert run.breach(trainer=False) == "wall clock"  # the reason string run up acts on is unchanged
    assert run.state["breaker"]["why"] == "wall clock"
    assert run.state["breaker"]["stop_reason"] == "breaker:run.wall_clock"
    (event,) = _events(run)
    assert event["event"] == "rl_breaker" and event["name"] == "run.wall_clock" and event["state"] == "trip"
    assert event["stop_reason"] == "breaker:run.wall_clock" and event["limit"] == 2
    assert event["config_source"] == "run config budget.wall_h"


def test_a_budget_breach_reports_its_stop_reason(make_run, monkeypatch):
    run = make_run(budget={"usd": 100, "wall_h": None})
    monkeypatch.setattr(up.Run, "spend_usd", lambda self: 150.0)
    assert run.breach(trainer=False) == "budget $150"
    (event,) = _events(run)
    assert (event["name"], event["value"], event["limit"]) == ("run.budget", 150.0, 100.0)
    assert run.state["breaker"]["stop_reason"] == "breaker:run.budget"


def test_a_failed_launcher_trips_the_trainer_exit_invariant(make_run, monkeypatch, tmp_path):
    run = make_run(budget={"wall_h": None})
    log = tmp_path / "trainer.log"
    log.write_text("[yeto] run 'r1-trainer' finished: FAILED (exit code 3)\n")
    run.state["procs"] = {"trainer": {"pid": 1, "log": str(log)}}
    monkeypatch.setattr(up, "_alive", lambda pid: False)
    monkeypatch.setattr(up.Run, "spend_usd", lambda self: 0.0)
    assert run.breach() == "trainer: launcher finished FAILED (exit code 3)"
    assert _events(run)[-1]["name"] == "run.trainer_exit"
    assert run.state["breaker"]["stop_reason"] == "breaker:run.trainer_exit"


def test_the_first_group_timeout_comes_from_the_section_and_is_reported(make_run, monkeypatch):
    run = make_run(budget={"wall_h": None}, first_group={"poll_s": 60},
                   breakers={"run": {"first_group": {"timeout_s": 300}}})
    monkeypatch.setattr(up.Run, "buffer_objects", lambda self: 0)
    monkeypatch.setattr(up.Run, "spend_usd", lambda self: 0.0)
    t0 = run.clock.t
    assert run.wait_first_group() == "first group timed out"
    assert run.clock.t - t0 == 300
    event = _events(run)[-1]
    assert event["name"] == "run.first_group" and event["stop_reason"] == "breaker:run.first_group"


def test_the_forced_pool_stop_factor_is_the_budget_breakers(make_run, monkeypatch):
    run = make_run(budget={"usd": 100, "wall_h": None}, breakers={"run": {"budget": {"force_pools_ratio": 1.5}}})
    spend = {"v": 130.0}
    monkeypatch.setattr(up.Run, "spend_usd", lambda self: spend["v"])
    members = [[{"state": "ready"}], [{"state": "ready"}], []]
    monkeypatch.setattr(up.Run, "pool_members", lambda self: members.pop(0))
    monkeypatch.setattr(up.Run, "pools_phrase", lambda self: "the NC pool")

    def sleep(s):
        spend["v"] = 160.0

    monkeypatch.setattr(up.time, "sleep", sleep)
    run.watch_nc_after_breach("budget $130", poll_s=60)
    stops = [argv for argv in run.calls if "pool-stop" in argv]
    assert len(stops) == 1  # not at 130 (> 1.2 x 100), at 160 (> 1.5 x 100)
    assert run.state["steps"]["nc_forced_stop"]["why"] == "spend $160 > $150"


# ----------------------------------------------------------------------------- starvation
def test_starvation_from_resolved_matches_the_legacy_section():
    for section in ({}, {"alert_min": 0, "stop_min": 45}, {"alert_min": 20, "stop_min": 0, "check_s": 30},
                    {"alert_min": 0, "stop_min": 0, "check_s": 0}, {"trainer_wait_max_min": 60, "stop_grace_min": 3}):
        rb = breakers.resolve(breakers.Sources(run_config=_cfg(starvation=section)), layers=("run",))
        assert st.StarvationConfig.from_resolved(rb) == st.StarvationConfig.from_section(section), section


def test_both_starvation_verdicts_off_turn_the_breaker_off():
    rb = breakers.resolve(breakers.Sources(run_config=_cfg(starvation={"alert_min": 0, "stop_min": 0})),
                          layers=("run",))
    assert not rb.enabled("run.starvation")
    rb = breakers.resolve(breakers.Sources(run_config=_cfg(starvation={"alert_min": 0})), layers=("run",))
    assert rb.enabled("run.starvation")  # the stop verdict is still on


def test_starvation_turned_off_by_the_section_and_a_bad_combination():
    rb = breakers.resolve(breakers.Sources(run_config=_cfg(breakers={"run": {"starvation": {"enabled": False}}})))
    assert not st.StarvationConfig.from_resolved(rb).enabled
    rb = breakers.resolve(breakers.Sources(run_config=_cfg(
        breakers={"run": {"starvation": {"alert_s": 3600, "stop_s": 600}}})))
    with pytest.raises(ValueError, match="run.starvation.stop_s"):
        st.StarvationConfig.from_resolved(rb)


def test_a_starvation_stop_reports_its_stop_reason(make_run, monkeypatch):
    run = make_run(budget={"wall_h": None})
    run.state["procs"] = {"trainer": {"pid": 42, "log": "/nonexistent"}}
    monkeypatch.setattr(up, "_alive", lambda pid: False)
    run.trainer_down = lambda why: None
    run._starvation_trip = (7300.0, 7200.0)
    run.starvation_stop("no new group", poll_s=1)
    assert run.state["starvation"]["event"] == st.STOP_EVENT
    assert run.state["starvation"]["stop_reason"] == "breaker:run.starvation"
    event = _events(run)[-1]
    assert (event["name"], event["value"], event["limit"], event["legacy_event"]) == (
        "run.starvation", 7300.0, 7200.0, st.STOP_EVENT)


# ----------------------------------------------------------------------------- release stall
def _watch(**kw):
    return rs.ReleaseStallWatch(rs.ReleaseStallConfig(**kw))


def test_no_release_stall_verdict_before_the_first_release():
    w = _watch()
    assert w.observe(0, release=None, incarnation=1).state == rs.OK
    assert w.observe(10 ** 6, release=None, incarnation=1).state == rs.OK


def test_release_stall_alerts_once_per_stretch_after_the_default_limit():
    w = _watch()
    w.observe(0, release="r1", incarnation=1)
    assert w.observe(rs.DEFAULT_LIMIT_S - 1, release="r1", incarnation=1).state == rs.OK
    v = w.observe(rs.DEFAULT_LIMIT_S, release="r1", incarnation=1)
    assert v.state == rs.ALERT and v.new_alert
    assert not w.observe(rs.DEFAULT_LIMIT_S + 60, release="r1", incarnation=1).new_alert
    v = w.observe(rs.DEFAULT_LIMIT_S + 120, release="r2", incarnation=1)
    assert v.state == rs.OK and v.recovered


def test_the_release_stall_limit_is_learned_from_the_release_interval():
    w = _watch()
    for i in range(rs.LEARN_RELEASES):
        w.observe(i * 1200.0, release=f"r{i}", incarnation=1)
    assert w.limit_s() == max(rs.MIN_LIMIT_S, rs.INTERVAL_FACTOR * 1200.0)
    w = _watch()
    for i in range(rs.LEARN_RELEASES):
        w.observe(i * 60.0, release=f"r{i}", incarnation=1)
    assert w.limit_s() == rs.MIN_LIMIT_S
    assert _watch(limit_s=900).limit_s() == 900


def test_a_trainer_restart_starts_the_release_clock_again():
    w = _watch()
    w.observe(0, release="r1", incarnation=1)
    assert w.observe(3000, release="r1", incarnation=2).state == rs.OK  # restarted at 3000
    assert w.observe(3000 + rs.DEFAULT_LIMIT_S - 1, release="r1", incarnation=2).state == rs.OK
    assert w.observe(3000 + rs.DEFAULT_LIMIT_S, release="r1", incarnation=2).state == rs.ALERT


def test_a_failed_pointer_read_never_stops():
    w = _watch(stop=True)
    w.observe(0, release="r1", incarnation=1)
    assert w.observe(10 ** 5, release=None, incarnation=1, read=False).state == rs.ALERT
    assert w.observe(10 ** 5 + 1, release="r1", incarnation=1).state == rs.STOP


class FakeReleases:
    def __init__(self):
        self.latest = None
        self.fail = False
        self.reads = 0

    def read(self, key):
        assert key == "policy-releases/LATEST.json"
        self.reads += 1
        if self.fail:
            raise OSError("store down")
        return None if self.latest is None else json.dumps({"seq": self.latest, "path": f"r{self.latest}-v1"}).encode()


def test_run_up_alerts_on_a_release_stall_without_stopping_by_default(make_run, capsys):
    run = make_run()
    store = FakeReleases()
    run._objects = lambda: store
    run.state["procs"] = {"trainer": {"pid": 42, "started": "t"}}
    t = run.clock.t
    assert run.release_stall_check(t) is None
    store.latest = 1
    assert run.release_stall_check(t + 300) is None
    assert run.release_stall_check(t + 300 + rs.DEFAULT_LIMIT_S) is None  # an alert only
    assert "CRITICAL trainer_release_stall_alert" in capsys.readouterr().out
    event = _events(run)[-1]
    assert (event["name"], event["action"], event["state"]) == ("run.release_stall", "warn", "trip")
    assert "stop_reason" not in event
    assert run.state["release_stall"]["event"] == rs.ALERT_EVENT
    store.latest = 2
    run.release_stall_check(t + 600 + rs.DEFAULT_LIMIT_S)
    assert "release_stall" not in run.state and _events(run)[-1]["state"] == "clear"


def test_run_up_reads_the_release_pointer_at_most_every_check_s(make_run):
    run = make_run()
    store = FakeReleases()
    run._objects = lambda: store
    run.state["procs"] = {"trainer": {"pid": 42, "started": "t"}}
    for i in range(10):
        run.release_stall_check(run.clock.t + i * 60)
    assert store.reads == 2  # at 0 and at 300 s (check_s)


def test_release_stall_action_stop_stops_the_trainer_through_the_stop_flag(make_run, monkeypatch):
    run = make_run(budget={"wall_h": None}, breakers={"run": {"release_stall": {"action": "stop", "limit_s": 1800}}})
    store = FakeReleases()
    store.latest = 1
    run._objects = lambda: store
    run.state["procs"] = {"trainer": {"pid": 42, "started": "t", "log": "/nonexistent"}}
    downs = []
    run.trainer_down = downs.append
    monkeypatch.setattr(up, "_alive", lambda pid: True)
    monkeypatch.setattr(run, "breach", lambda trainer=True: None)
    monkeypatch.setattr(run, "starvation_check", lambda now: None)
    monkeypatch.setattr(run, "seal_data_end", lambda: False)
    run.monitor(poll_s=60)
    assert downs and downs[0].startswith(rs.STOP_EVENT)
    assert run.state["release_stall"]["stop_reason"] == "breaker:run.release_stall"
    assert [e["state"] for e in _events(run) if e["name"] == "run.release_stall"] == ["trip"]


def test_the_release_stall_can_be_turned_off(make_run):
    run = make_run(breakers={"run": {"release_stall": {"enabled": False}}})
    run._objects = lambda: pytest.fail("no read while the breaker is off")
    assert run.release_stall_check(run.clock.t) is None
