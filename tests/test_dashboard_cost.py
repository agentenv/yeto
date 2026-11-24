"""Cost view (v7 arithmetic) and the head fleet.jsonl."""

import calendar
import json
import shutil
import time
from pathlib import Path

import pytest

from yeto.dashboard import sources
from yeto.dashboard.fleet import FleetLog, island_id_of, meta_from_task
from yeto.dashboard.reducer import Reducer
from yeto.dashboard.sources import load_all

from dashboard_helpers import T0, local_round

PRICES = {"prices": {"nebius:H200": 3.6, "modal:H100": 3.95}}


def ready(r, name, iid, cloud, gpu, gpus, t=T0):
    r.feed({"event": "island_ready", "island": name, "island_id": iid, "cloud": cloud, "gpu": gpu,
            "gpus": gpus, "time_unix": t})


def test_v7_cost_arithmetic_and_efficiency():
    r = Reducer(prices=PRICES, budget_usd=300.0)
    ready(r, "nebius-h200-b", 1, "nebius", "H200", 8)
    ready(r, "local-4090", 3, "local", "RTX4090", 4)
    ready(r, "verda-x", 2, "verda", "B200", 8)  # not in the table
    r.feed(local_round(1, 1, T0 + 10, reward_mean=0.2, tok_per_s=6100.0))
    r.feed(local_round(1, 2, T0 + 20, reward_mean=0.5, tok_per_s=6100.0))
    c = r.overview(now=T0 + 2.6 * 3600)["cost"]
    rows = {x["id"]: x for x in c["islands"]}
    neb = rows["1"]
    assert neb["rate_usd_h"] == pytest.approx(28.8) and neb["hours"] == pytest.approx(2.6)
    assert neb["cost_usd"] == pytest.approx(74.88)
    assert neb["usd_per_1m_tok"] == pytest.approx(28.8 / (6100 * 3600 / 1e6))
    assert neb["reward_per_usd"] == pytest.approx(0.3 / 74.88) and neb["ranked"]
    assert rows["3"]["cost_usd"] == 0 and rows["3"]["local"] and not rows["3"]["ranked"]
    assert rows["2"]["cost_usd"] is None and not rows["2"]["priced"]  # unpriced, never 0
    assert c["unpriced"] == ["2"]
    assert c["total_usd"] == pytest.approx(74.88) and c["burn_usd_h"] == pytest.approx(28.8)
    assert c["budget_pct"] == pytest.approx(74.88 / 3)
    assert c["hours_to_cap"] == pytest.approx((300 - 74.88) / 28.8)
    assert "not billing data" in c["note"]


def test_relaunch_intervals_accrue_only_ready_time():
    r = Reducer(prices=PRICES)
    ready(r, "n", 0, "nebius", "H200", 1)
    r.feed({"event": "island_lost", "island": "n", "island_id": 0, "time_unix": T0 + 3600})
    ready(r, "n", 0, "nebius", "H200", 1, t=T0 + 7200)
    r.feed({"event": "island_stop", "island": "n", "island_id": 0, "time_unix": T0 + 9000})
    (row,) = r.overview(now=T0 + 99999)["cost"]["islands"]
    assert row["hours"] == pytest.approx(1.5) and not row["running"]


def test_no_fleet_data_means_no_cost():
    r = Reducer()
    r.feed(local_round(0, 1, T0))
    c = r.overview()["cost"]
    assert c["islands"] == [] and c["total_usd"] is None and c["budget_pct"] is None


class Clock:
    def __init__(self):
        self.t = T0

    def __call__(self):
        return self.t


def test_fleet_log_events_and_cost_ticks(tmp_path):
    clock = Clock()
    log = FleetLog(tmp_path / "fleet.jsonl",
                   {"run-l0-eu": {"cloud": "nebius", "gpu": "H200", "gpus": 8},
                    "run-l1-us": {"cloud": "verda", "gpu": "B200", "gpus": 8}},
                   prices=PRICES, budget_usd=500.0, tick_s=300, clock=clock)
    log.event("island_ready", "run-l0-eu")
    log.event("island_ready", "run-l1-us")
    assert log.maybe_tick() is not None
    clock.t += 299
    assert log.maybe_tick() is None
    clock.t += 1801  # 35 min after ready
    tick = log.maybe_tick()
    rows = {x["island"]: x for x in tick["islands"]}
    assert rows["run-l0-eu"]["cost_usd"] == pytest.approx(28.8 * 2100 / 3600)
    assert rows["run-l1-us"]["cost_usd"] is None and rows["run-l1-us"]["rate_usd_h"] is None
    assert tick["total_usd"] == pytest.approx(28.8 * 2100 / 3600) and tick["budget_usd"] == 500.0
    log.event("island_lost", "run-l0-eu", reason="preempted")
    clock.t += 3600
    log.event("island_stop", "run-l1-us")
    tick = log.cost_tick()
    assert {x["island"]: x["wall_h"] for x in tick["islands"]}["run-l0-eu"] == pytest.approx(2100 / 3600)
    lines = [json.loads(x) for x in (tmp_path / "fleet.jsonl").read_text().splitlines()]
    assert [x["event"] for x in lines] == ["island_ready", "island_ready", "cost_tick", "cost_tick",
                                           "island_lost", "island_stop", "cost_tick"]
    assert lines[0]["island_id"] == 0 and lines[0]["gpus"] == 8
    r = Reducer(prices=PRICES)
    load_all(r, [str(tmp_path / "fleet.jsonl")])
    assert r.budget_usd == 500.0 and r.islands["0"]["fleet_state"] == "lost"
    assert r.overview(now=clock.t)["cost"]["total_usd"] == pytest.approx(28.8 * 2100 / 3600)


def test_island_id_and_task_meta():
    assert island_id_of("s9-m4x1-20261005aa-l0-eu-north1") == 0
    assert island_id_of("syncer") is None

    class Res:
        cloud, region, accelerators = "Nebius", "eu-north1", {"H200": 8}

    class Task:
        resources, num_nodes = {Res()}, 2

    assert meta_from_task(Task()) == {"cloud": "nebius", "region": "eu-north1", "gpu": "H200", "gpus": 16}
    assert meta_from_task(object())["gpu"] is None


def test_meta_from_modal_island_config_shape():
    # real Modal runs logged cloud/gpu/gpus = null before this
    class Cfg:
        gpu = "H200"
        gpus_per_node = 8
        num_nodes = 1
        region = None

    assert meta_from_task(Cfg()) == {"cloud": "modal", "region": None, "gpu": "H200", "gpus": 8}
    Cfg.gpu, Cfg.gpus_per_node, Cfg.num_nodes = "H100!", 1, 2
    assert meta_from_task(Cfg()) == {"cloud": "modal", "region": None, "gpu": "H100", "gpus": 2}


def test_fleet_controller_writes_lifecycle(tmp_path):
    from test_controller import RUNNING, SUCCEEDED, FakeOps, make_controller

    ops = FakeOps()
    ops.status_seq["run-l0-x"] = [RUNNING]
    ops.up["run-l0-x"] = False
    ops.relaunch_results["run-l0-x"] = [101]
    ops.after_relaunch["run-l0-x"] = [RUNNING, SUCCEEDED]
    ctl = make_controller(ops, {"run-l0-x": 1})
    clock = Clock()
    ctl.fleet_log = FleetLog(tmp_path / "fleet.jsonl", {"run-l0-x": {"cloud": "nebius", "gpu": "H200",
                                                                     "gpus": 1}},
                             prices=PRICES, clock=clock)
    ctl.run()
    events = [json.loads(x)["event"] for x in (tmp_path / "fleet.jsonl").read_text().splitlines()]
    lifecycle = [e for e in events if e != "cost_tick"]
    assert lifecycle == ["island_ready", "island_lost", "island_ready", "island_stop"]
    assert events[-1] == "cost_tick"


def test_fleet_log_failure_never_breaks_the_controller(tmp_path):
    from test_controller import RUNNING, SUCCEEDED, FakeOps, make_controller

    class Broken:
        def event(self, *a, **k):
            raise RuntimeError("disk full")

        def maybe_tick(self):
            raise RuntimeError("disk full")

    ops = FakeOps()
    ops.status_seq["l0"] = [RUNNING, SUCCEEDED]
    ctl = make_controller(ops, {"l0": 1})
    ctl.fleet_log = Broken()
    assert list(ctl.run()) == ["l0"]


# ------------------------------------------------------------- after a stop the estimate stops
# Two real runs (redacted) whose run.budget breaker stopped the trainer: `run up` signalled the
# launcher and `yeto down` stopped the Modal app, so fleet.jsonl kept only island_ready (no
# island_stop). The page used to bill the island up to "now" for hours after.

STOP_FIXTURES = Path(__file__).parent / "fixtures" / "dashboard" / "breaker_stop"


def _utc(stamp: str) -> float:
    return float(calendar.timegm(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")))


def _stopped_run(tmp_path, run, *, up=True, meta=True, extra=()):
    d = tmp_path / run
    shutil.copytree(STOP_FIXTURES / run, d)
    if not meta:
        (d / "meta.json").unlink()
    with (d / "fleet.jsonl").open("a") as fh:
        for rec in extra:
            fh.write(json.dumps(rec) + "\n")
    r = Reducer(run=run, budget_usd=0.5)
    if up:
        r.up_state = str(d / "up" / "state.json")
    load_all(r, [str(p) for p in sources.discover([str(d)])])
    state = json.loads((d / "up" / "state.json").read_text())
    fleet = [json.loads(x) for x in (d / "fleet.jsonl").read_text().splitlines()]
    ready_ts = next(x["time_unix"] for x in fleet if x["event"] == "island_ready")
    return r, _utc(state["steps"]["trainer_down"]["at"]), ready_ts


def _row(r, now):
    c = r.overview(now=now)["cost"]
    (row,) = c["islands"]
    return c, row


@pytest.mark.parametrize("run", ["c1-bk", "c1-bkr"])
def test_a_breaker_stopped_island_is_billed_up_to_its_trainer_down(tmp_path, run):
    r, down, ready_ts = _stopped_run(tmp_path, run)
    _, live = _row(r, down - 30)
    assert live["running"] and live.get("billed_until") is None
    pinned = []
    for later in (60, 25 * 60, 6 * 3600):
        c, row = _row(r, down + later)
        assert not row["running"] and row["billed_until"] == down
        assert row["billed_until_source"] == "run up trainer_down (breaker:run.budget)"
        assert row["hours"] == pytest.approx((down - ready_ts) / 3600)
        assert c["burn_usd_h"] == 0 and c["hours_to_cap"] is None
        pinned.append(c["total_usd"])
    assert pinned[0] == pinned[1] == pinned[2] and live["cost_usd"] < pinned[0]
    (alert,) = [a for a in r.overview(now=down + 3600)["alerts"] if a["rule"] == "budget"]
    assert "reached in about" not in alert["detail"]  # no ETA once nothing burns


def test_without_the_up_state_the_registry_finish_pins_it(tmp_path):
    r, down, ready_ts = _stopped_run(tmp_path, "c1-bk", up=False)
    finished = json.loads((tmp_path / "c1-bk" / "meta.json").read_text())["finished_at"]
    _, row = _row(r, down + 3600)
    assert (row["running"], row["billed_until"], row["billed_until_source"]) == (False, finished, "run down")
    assert row["hours"] == pytest.approx((finished - ready_ts) / 3600)


def test_with_nothing_saying_it_stopped_the_island_is_still_billed(tmp_path):
    r, down, _ = _stopped_run(tmp_path, "c1-bk", up=False, meta=False)
    _, a = _row(r, down + 60)
    _, b = _row(r, down + 3600)
    assert a["running"] and b["running"] and b["cost_usd"] > a["cost_usd"]


def test_the_island_stop_a_teardown_writes_closes_it_by_itself(tmp_path):
    name = "c1-bk-l0-modal"
    stop = {"event": "island_stop", "island": name, "island_id": 0, "cloud": "modal", "gpu": "H100", "gpus": 1,
            "cpus": 16, "memory_gib": 128.0, "reason": "teardown"}
    _, down, ready_ts = _stopped_run(tmp_path / "probe", "c1-bk", up=False, meta=False)
    r, _, _ = _stopped_run(tmp_path, "c1-bk", up=False, meta=False, extra=[{**stop, "time_unix": down + 5}])
    _, row = _row(r, down + 3600)
    assert not row["running"] and row.get("billed_until") is None  # closed by the fleet tape itself
    assert row["hours"] == pytest.approx((down + 5 - ready_ts) / 3600)


def test_a_relaunch_after_the_stop_is_billed_again(tmp_path):
    name = "c1-bk-l0-modal"
    _, down, _ = _stopped_run(tmp_path / "probe", "c1-bk")
    again = {"event": "island_ready", "island": name, "island_id": 0, "cloud": "modal", "gpu": "H100", "gpus": 1,
             "cpus": 16, "memory_gib": 128.0, "relaunched": True, "time_unix": down + 600}
    r, _, _ = _stopped_run(tmp_path, "c1-bk", extra=[again])
    _, a = _row(r, down + 700)
    _, b = _row(r, down + 3600)
    assert a["running"] and b["running"] and b["cost_usd"] > a["cost_usd"]


def _lifecycle(path):
    recs = [json.loads(line) for line in path.read_text().splitlines() if line.startswith('{"')]
    return [(r["event"], r["island"], r.get("reason")) for r in recs if r["event"].startswith("island_")]


def test_close_open_islands_stops_only_the_open_ones_once(tmp_path, capsys):
    from yeto.dashboard.fleet import close_open_islands

    path = tmp_path / "fleet.jsonl"
    log = FleetLog(path, {"r-l0-modal": {"cloud": "modal", "gpu": "H100", "gpus": 1, "cpus": 16},
                          "r-l1-modal": {"cloud": "modal", "gpu": "H100", "gpus": 1},
                          "r-l2-modal": {"cloud": "modal", "gpu": "H100", "gpus": 1}}, clock=lambda: T0)
    for name in log.meta:
        log.event("island_ready", name)
    log.event("island_stop", "r-l1-modal", reason="SUCCEEDED")
    log.cost_tick()
    with path.open("a") as fh:
        fh.write("{not json\n")
    assert close_open_islands(path, "breaker:run.budget", skip=["r-l2-modal"], clock=lambda: T0 + 60) == ["r-l0-modal"]
    assert close_open_islands(path, "teardown", clock=lambda: T0 + 90) == ["r-l2-modal"]
    assert close_open_islands(path, "teardown") == []  # idempotent
    assert _lifecycle(path)[-3:] == [("island_stop", "r-l1-modal", "SUCCEEDED"),
                                     ("island_stop", "r-l0-modal", "breaker:run.budget"),
                                     ("island_stop", "r-l2-modal", "teardown")]
    stop = json.loads(path.read_text().splitlines()[-2])
    assert (stop["time_unix"], stop["island_id"], stop["cloud"], stop["cpus"]) == (T0 + 60, 0, "modal", 16)
    assert close_open_islands(tmp_path / "missing.jsonl", "teardown") == []
    (tmp_path / "dir.jsonl").mkdir()
    assert close_open_islands(tmp_path / "dir.jsonl", "teardown") == []  # never raises


def test_the_teardown_reason_names_the_breaker_that_stopped_the_run(monkeypatch):
    from yeto import breakers
    from yeto.dashboard.fleet import teardown_stop_reason

    breakers.clear_stop_reason()
    monkeypatch.delenv("YETO_STOP_REASON", raising=False)
    assert teardown_stop_reason() == "teardown"
    monkeypatch.setenv("YETO_STOP_REASON", "breaker:run.budget")
    assert teardown_stop_reason() == "breaker:run.budget"
    breakers.record_stop_reason("breaker:launch.budget")
    try:
        assert teardown_stop_reason() == "breaker:launch.budget"
    finally:
        breakers.clear_stop_reason()


def test_yeto_down_closes_the_islands_a_killed_launcher_left_open(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from yeto import breakers, cli, runs

    breakers.clear_stop_reason()
    monkeypatch.delenv("YETO_STOP_REASON", raising=False)
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path)
    runs.create_run("c1-bk", {"cluster_prefix": "c1-bk"})
    fleet = runs.run_dir("c1-bk") / "fleet.jsonl"
    shutil.copy(STOP_FIXTURES / "c1-bk" / "fleet.jsonl", fleet)
    assert cli.cmd_down(SimpleNamespace(run="c1-bk")) == 0
    assert _lifecycle(fleet) == [("island_ready", "c1-bk-l0-modal", None),
                                 ("island_stop", "c1-bk-l0-modal", "teardown")]
    assert runs.load_run("c1-bk")["state"] == runs.DOWN
