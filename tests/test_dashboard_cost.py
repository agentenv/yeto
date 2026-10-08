"""fleet-dashboard 3.4 / 2.3: cost view (v7 arithmetic) and the head fleet.jsonl."""

import json

import pytest

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
    assert rows["2"]["cost_usd"] is None and not rows["2"]["priced"]  # 未定价, never 0
    assert c["unpriced"] == ["2"]
    assert c["total_usd"] == pytest.approx(74.88) and c["burn_usd_h"] == pytest.approx(28.8)
    assert c["budget_pct"] == pytest.approx(74.88 / 3)
    assert c["hours_to_cap"] == pytest.approx((300 - 74.88) / 28.8)
    assert "非账单" in c["note"]


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
    # real Modal runs (s13-fnsmoke/g1) logged cloud/gpu/gpus = null before this
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
