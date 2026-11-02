"""S29-next P0-c3 / P1-1 / P1-7: the VM watchdog against a fake cloud (no sky, no nebius, no boat)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_t", ROOT / f"tools/probes/s29_spot/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


wd = _load("vm_watchdog")


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class FakeOps:
    """launch results per candidate key (list consumed in order), states per poll."""

    def __init__(self, clock, launch=None, states=None, launch_s=120.0):
        self.clock, self.launch_res, self.states, self.launch_s = clock, launch or {}, list(states or []), launch_s
        self.calls: list[tuple] = []

    def launch(self, cluster, cand, timeout_s):
        self.calls.append(("launch", cluster, cand.key))
        res = self.launch_res.get(cand.key, [False])
        ok = res.pop(0) if res else False
        self.clock.t += self.launch_s if ok else 30.0
        return ok, "" if ok else "Failed to acquire resources"

    def ip(self, cluster):
        return "1.2.3.4"

    def state(self, cluster, region):
        self.calls.append(("state", cluster))
        return self.states.pop(0) if self.states else "RUNNING"

    def down(self, cluster):
        self.calls.append(("down", cluster))

    def sweep(self, run, island):
        self.calls.append(("sweep", run, island))
        return "deleted 3"

    def ready(self, cluster, cmd):
        return True


def _cfg(tmp_path, **kw):
    base = dict(cluster_prefix="s30t-h100", run_id="s30t", island="spot-0", budget_usd=1000.0,
                log=str(tmp_path / "wd.jsonl"), state=str(tmp_path / "wd.state.json"),
                stop_file=str(tmp_path / "wd.stop"), poll_s=60.0, round_sleep_s=300.0, orphan_sweep_min=15.0)
    base.update(kw)
    return wd.Config(**base)


def _events(tmp_path):
    return [json.loads(x) for x in (tmp_path / "wd.jsonl").read_text().splitlines()]


def test_candidate_order_ondemand_h100_then_spot_by_price():
    keys = [c.key for c in wd.candidates()]
    assert keys[0] == "H100:8@eu-north1/od"
    spot = [c for c in wd.candidates() if c.spot]
    assert [c.price_hr for c in spot] == sorted(c.price_hr for c in spot)
    assert {c.gpu for c in spot} == {"H100", "H200"} and len({c.region for c in spot}) >= 3
    assert not any(c.gpu == "H200" and not c.spot for c in wd.candidates())
    assert any(c.gpu == "H200" and not c.spot for c in wd.candidates(allow_ondemand_h200=True))
    assert all(c.spot for c in wd.candidates(spot_only=True))
    # H100 and H200 are not the same price (the old yaml comment said so)
    assert wd.PRICES_8GPU[("H100", "eu-north1")][1] != wd.PRICES_8GPU[("H200", "eu-north1")][1]


def test_prefix_must_be_ours(tmp_path):
    with pytest.raises(ValueError):
        wd.Watchdog(_cfg(tmp_path, cluster_prefix="prod-x"), wd.candidates(), FakeOps(Clock()))


def test_fallback_to_spot_then_preemption_is_delete_and_recreate(tmp_path):
    clk = Clock()
    ops = FakeOps(clk, launch={"H100:8@eu-north1/spot": [True, True]}, states=["RUNNING", "STOPPED"])
    w = wd.Watchdog(_cfg(tmp_path, max_rounds=1), wd.candidates(), ops, clock=clk, sleep=clk.sleep)
    w.acquire()
    assert w.cluster == "s30t-h100-2" and w.cand.key == "H100:8@eu-north1/spot"
    w.watch_once(); w.watch_once()  # RUNNING, then STOPPED -> preempted
    assert ("down", "s30t-h100-2") in ops.calls and w.cluster is None
    w.acquire()
    assert w.cluster == "s30t-h100-4"  # od H100 failed again, spot H100 new cluster; never a start
    ev = [e["event"] for e in _events(tmp_path)]
    assert "preempted" in ev and ev.count("launch_ok") == 2 and "start" not in ev


def test_stuck_starting_is_given_up_after_timeout(tmp_path):
    clk = Clock()
    ops = FakeOps(clk, launch={"H100:8@eu-north1/od": [True]}, states=["STARTING"] * 20)
    w = wd.Watchdog(_cfg(tmp_path, starting_timeout_s=600), wd.candidates(), ops, clock=clk, sleep=clk.sleep)
    w.acquire()
    for _ in range(12):
        clk.sleep(60); w.watch_once()
        if w.cluster is None:
            break
    assert w.cluster is None and any(e["event"] == "starting_timeout" for e in _events(tmp_path))


def test_launch_slower_than_timeout_is_abandoned(tmp_path):
    clk = Clock()
    ops = FakeOps(clk, launch={"H100:8@eu-north1/od": [True]}, launch_s=900.0)
    w = wd.Watchdog(_cfg(tmp_path, launch_timeout_s=600, max_rounds=1), wd.candidates()[:1], ops,
                    clock=clk, sleep=clk.sleep)
    assert w.run() == "max_rounds"
    assert ("down", "s30t-h100-1") in ops.calls


def test_manual_stop_marker_exits_and_leaves_vm(tmp_path):
    clk = Clock()
    ops = FakeOps(clk, launch={"H100:8@eu-north1/od": [True]})
    cfg = _cfg(tmp_path)
    w = wd.Watchdog(cfg, wd.candidates(), ops, clock=clk, sleep=clk.sleep)
    w.acquire()
    Path(cfg.stop_file).touch()
    assert w.run() == "manual_stop"
    assert not any(c[0] == "down" for c in ops.calls)


def test_budget_gate_deletes_and_spend_survives_restart(tmp_path):
    clk = Clock()
    ops = FakeOps(clk, launch={"H100:8@eu-north1/od": [True]})
    cfg = _cfg(tmp_path, budget_usd=40.0)
    w = wd.Watchdog(cfg, wd.candidates(), ops, clock=clk, sleep=clk.sleep)
    assert w.run() == "budget"
    assert ops.calls[-1] == ("down", "s30t-h100-1") and 39.0 <= w.spent <= 41.0
    w2 = wd.Watchdog(cfg, wd.candidates(), FakeOps(clk), clock=clk, sleep=clk.sleep)
    assert w2.spent == pytest.approx(w.spent) and w2.run() == "budget"


def test_orphan_sweep_after_vm_gone_n_minutes(tmp_path):
    clk = Clock()
    ops = FakeOps(clk, launch={"H100:8@eu-north1/od": [True]}, states=["STOPPED"])
    w = wd.Watchdog(_cfg(tmp_path, orphan_sweep_min=15, max_rounds=4), wd.candidates(), ops,
                    clock=clk, sleep=clk.sleep)
    w.acquire(); w.watch_once()  # preempted
    assert w.run() == "max_rounds"  # no capacity anywhere for 4 rounds (> 15 min)
    sweeps = [c for c in ops.calls if c[0] == "sweep"]
    assert sweeps == [("sweep", "s30t", "spot-0")]  # once per outage
    assert any(e["event"] == "orphan_sweep" for e in _events(tmp_path))


def test_cold_start_phases_logged(tmp_path):
    clk = Clock()
    ops = FakeOps(clk, launch={"H100:8@eu-north1/od": [True]})
    w = wd.Watchdog(_cfg(tmp_path, ready_cmd="test -s /root/spot120/ok"), wd.candidates(), ops,
                    clock=clk, sleep=clk.sleep)
    w.acquire()
    ev = {e["event"]: e for e in _events(tmp_path)}
    assert ev["launch_ok"]["launch_s"] == 120.0 and ev["ready"]["cold_start_s"] >= 120.0


def test_spot_plan_tp_and_starts_per_hour():
    sp = _load("spot_plan")
    s = sp.spec(island=0, islands=2, base="/b", data="/d", out="/o", engine_version="x", python="py",
                card_model="H100", tp=8)
    assert s["engine"]["options"]["tp"] == 8 and s["engine"]["server"]["topology"] == "tp8"
    assert int(sp.env(data="/d", public_ip="1.1.1.1")["YETO_BOAT_STARTS_PER_HOUR"]) >= 16


def test_d35_the_stop_marker_ends_a_ready_wait(tmp_path):
    """S30 D35: a ready command that never passes kept the watchdog in its ready
    loop, so the stop marker went unread for an hour."""
    clock = Clock()
    stop = tmp_path / "wd.stop"

    class NeverReady(FakeOps):
        def ready(self, cluster, cmd):
            stop.write_text("")  # the operator stops it while it waits
            return False

    ops = NeverReady(clock, launch={"H100:8@eu-north1/od": [True]})
    dog = wd.Watchdog(_cfg(tmp_path, ready_cmd="test -s x", ready_timeout_s=10_000.0), wd.candidates(), ops,
                      clock=clock, sleep=clock.sleep)
    assert dog.run() == "manual_stop"
    assert ("down", dog.cluster) not in ops.calls  # the VM is left as is


def test_d39_a_vm_reclaimed_during_the_ready_wait_is_relaunched(tmp_path):
    """A spot VM reclaimed before its sampler came up kept the watchdog
    probing ssh until the ready timeout; the ready wait now checks the instance state."""
    clock = Clock()

    class NeverReady(FakeOps):
        def ready(self, cluster, cmd):
            return False

    ops = NeverReady(clock, launch={"H100:8@eu-north1/od": [True, True]}, states=["RUNNING", "STOPPING"])
    dog = wd.Watchdog(_cfg(tmp_path, ready_cmd="test -s x", ready_timeout_s=10_000.0), wd.candidates(), ops,
                      clock=clock, sleep=clock.sleep)
    assert dog.acquire() is None and dog.cluster is None
    assert ("down", "s30t-h100-1") in ops.calls
    ev = [e for e in _events(tmp_path) if e["event"] == "preempted"]
    assert ev and ev[0]["state"] == "STOPPING" and ev[0]["during"] == "ready_wait"
    assert clock() - 1_000_000.0 < 1000  # well before the ready timeout
