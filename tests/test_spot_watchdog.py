"""The VM watchdog against a fake cloud (no sky, no nebius, no boat)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from yeto.rl.spot import spot_plan
from yeto.rl.spot import watchdog as wd


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
        return True

    def sweep(self, run, island):
        self.calls.append(("sweep", run, island))
        return "deleted 3"

    def ready(self, cluster, cmd):
        return True


def _cfg(tmp_path, **kw):
    base = dict(cluster_prefix="spot-t-h100", run_id="runt", island="spot-0", budget_usd=1000.0,
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
    assert w.cluster == "spot-t-h100-2" and w.cand.key == "H100:8@eu-north1/spot"
    w.watch_once(); w.watch_once()  # RUNNING, then STOPPED -> preempted
    assert ("down", "spot-t-h100-2") in ops.calls and w.cluster is None
    w.acquire()
    assert w.cluster == "spot-t-h100-4"  # od H100 failed again, spot H100 new cluster; never a start
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
    assert ("down", "spot-t-h100-1") in ops.calls


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
    assert ops.calls[-1] == ("down", "spot-t-h100-1") and 39.0 <= w.spent <= 41.0
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
    assert sweeps == [("sweep", "runt", "spot-0")]  # once per outage
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
    sp = spot_plan
    s = sp.spec(island=0, islands=2, base="/b", data="/d", out="/o", engine_version="x", python="py",
                card_model="H100", tp=8)
    assert s["engine"]["options"]["tp"] == 8 and s["engine"]["server"]["topology"] == "tp8"
    assert int(sp.env(data="/d", public_ip="1.1.1.1")["YETO_BOAT_STARTS_PER_HOUR"]) >= 16


def test_the_stop_marker_ends_a_ready_wait(tmp_path):
    """A ready command that never passes must not keep the stop marker unread."""
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


def test_a_vm_reclaimed_during_the_ready_wait_is_deleted_before_the_ready_timeout(tmp_path):
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
    assert ("down", "spot-t-h100-1") in ops.calls
    ev = [e for e in _events(tmp_path) if e["event"] == "preempted"]
    assert ev and ev[0]["state"] == "STOPPING" and ev[0]["during"] == "ready_wait"
    assert clock() - 1_000_000.0 < 1000  # well before the ready timeout


def test_run_relaunches_at_once_after_a_reclaim_during_the_ready_wait(tmp_path):
    """The VM reclaimed during its cold start is followed by the next candidate in
    the same round: no round sleep and no round counted against ``max_rounds``."""
    clock = Clock()
    stop = tmp_path / "wd.stop"
    slept: list[float] = []

    class ReadyOnSecond(FakeOps):
        def ready(self, cluster, cmd):
            return cluster.endswith("-2")

    cands = wd.candidates()[:2]
    ops = ReadyOnSecond(clock, launch={cands[0].key: [True], cands[1].key: [True]},
                        states=["STOPPING", "RUNNING"])
    dog = wd.Watchdog(_cfg(tmp_path, ready_cmd="test -s x", max_rounds=1), cands, ops, clock=clock,
                      sleep=lambda s: (slept.append(s), clock.sleep(s), dog.cluster == "spot-t-h100-2"
                                       and stop.write_text("")))
    assert dog.run() == "manual_stop"
    assert dog.cluster == "spot-t-h100-2" and ("down", "spot-t-h100-1") in ops.calls
    assert 300.0 not in slept  # never waited a round
    events = [e["event"] for e in _events(tmp_path)]
    assert "round_failed" not in events and events.count("ready") == 1


def test_a_failed_delete_keeps_the_cluster_billed_and_retries(tmp_path):
    clock = Clock()
    downs = {"spot-t-h100-1": [False, False, True]}

    class FlakyDown(FakeOps):
        def down(self, cluster):
            self.calls.append(("down", cluster))
            res = downs.get(cluster, [True])
            return res.pop(0) if len(res) > 1 else res[0]

    cand = wd.candidates()[0]
    ops = FlakyDown(clock, launch={cand.key: [True]}, states=["STOPPED"])
    dog = wd.Watchdog(_cfg(tmp_path), [cand], ops, clock=clock, sleep=clock.sleep)
    dog.acquire()
    dog.watch_once()  # preempted: the delete fails
    assert dog.cluster is None and "spot-t-h100-1" in dog.doomed
    saved = json.loads((tmp_path / "wd.state.json").read_text())
    assert "spot-t-h100-1" in saved["doomed"]
    spent = dog.spent
    clock.sleep(3600)
    dog.retry_deletes()  # fails again: the hour is billed
    assert dog.spent == pytest.approx(spent + cand.price_hr) and "spot-t-h100-1" in dog.doomed
    # a restarted watchdog keeps retrying it
    again = wd.Watchdog(_cfg(tmp_path), [cand], ops, clock=clock, sleep=clock.sleep)
    assert "spot-t-h100-1" in again.doomed
    again.retry_deletes()
    assert not again.doomed and any(e["event"] == "delete_retry_ok" for e in _events(tmp_path))


def test_the_doomed_cluster_counts_against_the_budget(tmp_path):
    clock = Clock()

    class NeverDown(FakeOps):
        def down(self, cluster):
            self.calls.append(("down", cluster))
            return False

    cand = wd.candidates()[0]
    ops = NeverDown(clock, launch={cand.key: [True]}, states=["STOPPED"])
    dog = wd.Watchdog(_cfg(tmp_path, budget_usd=50.0), [cand], ops, clock=clock, sleep=clock.sleep)
    dog.acquire()
    dog.watch_once()
    assert dog.cluster is None and dog.doomed
    assert dog.run() == "budget"  # the leaked VM alone uses up the budget
    assert dog.spent >= 50.0 - cand.price_hr and dog.doomed


def test_ops_survive_a_hung_subprocess(monkeypatch):
    def hang(*a, **k):
        raise wd.subprocess.TimeoutExpired(a[0], k.get("timeout", 0))

    ops = wd.Ops(yaml="x.yaml", envs=[], secrets=[])
    monkeypatch.setattr(wd.subprocess, "run", hang)
    assert ops.ready("spot-t-h100-1", "true") is False
    assert ops.ip("spot-t-h100-1") == ""
    assert ops.down("spot-t-h100-1") is False

    def missing(*a, **k):
        raise FileNotFoundError("ssh")

    monkeypatch.setattr(wd.subprocess, "run", missing)
    assert ops.ready("spot-t-h100-1", "true") is False and ops.ip("spot-t-h100-1") == ""
    assert ops.down("spot-t-h100-1") is False


def test_ops_down_reports_a_failed_sky_down(monkeypatch):
    class Done:
        def __init__(self, rc):
            self.returncode, self.stdout, self.stderr = rc, "", ""

    ops = wd.Ops(yaml="x.yaml", envs=[], secrets=[])
    monkeypatch.setattr(wd.subprocess, "run", lambda *a, **k: Done(1))
    assert ops.down("spot-t-h100-1") is False
    monkeypatch.setattr(wd.subprocess, "run", lambda *a, **k: Done(0))
    assert ops.down("spot-t-h100-1") is True



def test_price_file_and_own_name_are_parameters(tmp_path):
    prices = tmp_path / "prices.json"
    prices.write_text(json.dumps([["h100", "r1", 30.0, 5.0]]))
    cands = wd.candidates(prices=wd.load_prices(str(prices)))
    assert [c.key for c in cands] == ["H100:8@r1/od", "H100:8@r1/spot"]
    dog = wd.Watchdog(_cfg(tmp_path, cluster_prefix="team-x", own_name=r"^team-"), cands, FakeOps(Clock()))
    assert dog.cfg.own_name == r"^team-"
    assert wd.main(["--cluster-prefix", "spot-x", "--run-id", "r", "--budget-usd", "1", "--prices", str(prices),
                    "--dry-run"]) == 0
    with pytest.raises(SystemExit):  # the sky task is required for a real run
        wd.main(["--cluster-prefix", "spot-x", "--run-id", "r", "--budget-usd", "1", "--dir", str(tmp_path)])


def test_orphan_sweep_error_does_not_kill_the_watchdog(tmp_path, monkeypatch):
    """A hung ``boat_labels sweep`` (subprocess timeout) or a raising sweep hook used to
    propagate out of ``run()``: the watchdog died and the island VM ran on unwatched
    (no budget stop, no preemption recreate)."""
    import subprocess

    def hung(*a, **kw):
        raise subprocess.TimeoutExpired(a[0], kw.get("timeout"))

    monkeypatch.setattr(wd.subprocess, "run", hung)
    with pytest.raises(RuntimeError, match="TimeoutExpired"):  # a failure, retried with backoff
        wd.Ops(yaml="x", envs=[], secrets=[]).sweep("runt", "spot-0")

    class Raising(FakeOps):
        def sweep(self, run, island):
            self.calls.append(("sweep", run, island))
            raise RuntimeError("boat api down")

    clk = Clock()
    ops = Raising(clk, launch={"H100:8@eu-north1/od": [True]}, states=["STOPPED"])
    w = wd.Watchdog(_cfg(tmp_path, orphan_sweep_min=15, max_rounds=6), wd.candidates(), ops,
                    clock=clk, sleep=clk.sleep)
    w.acquire(); w.watch_once()  # preempted
    assert w.run() == "max_rounds"
    events = _events(tmp_path)
    assert any(e["event"] == "orphan_sweep_error" for e in events)
    assert len([c for c in ops.calls if c[0] == "sweep"]) >= 2  # retried, not given up


def test_a_hung_sweep_backs_off_and_does_not_slow_the_relaunch(tmp_path):
    """Each failing sweep blocks 600 s (the subprocess timeout). Without a backoff it ran
    before every candidate's launch, so a pass over the candidates took 600 s per candidate."""
    class Hung(FakeOps):
        def sweep(self, run, island):
            self.calls.append(("sweep", run, island))
            self.clock.t += 600.0
            raise RuntimeError("TimeoutExpired: boat_labels sweep")

    clk = Clock()
    cands = wd.candidates()
    ops = Hung(clk, launch={cands[0].key: [True]}, states=["STOPPED"])
    w = wd.Watchdog(_cfg(tmp_path, orphan_sweep_min=0, max_rounds=3), cands, ops, clock=clk, sleep=clk.sleep)
    w.acquire(); w.watch_once()  # preempted: the outage starts
    assert w.run() == "max_rounds"
    launches = [c for c in ops.calls if c[0] == "launch"][1:]
    sweeps = [c for c in ops.calls if c[0] == "sweep"]
    assert len(launches) == 3 * len(cands)
    assert len(sweeps) <= 6, len(sweeps)  # 60 s, 120 s, 240 s ... not once per candidate
    errors = [e for e in _events(tmp_path) if e["event"] == "orphan_sweep_error"]
    assert [e["retry_in_s"] for e in errors][:3] == [60.0, 120.0, 240.0]
    assert all(e["retry_in_s"] <= wd.SWEEP_BACKOFF_MAX_S for e in errors)
    # the first pass over the candidates waits on one hung sweep, not one per candidate
    first_pass = [e["t"] for e in _events(tmp_path) if e["event"] == "launch_start"][1:len(cands) + 1]
    assert first_pass[-1] - first_pass[0] < 600.0 + 30.0 * len(cands) + 1


def test_ops_launch_survives_a_missing_sky(monkeypatch):
    """No sky executable: launch fails like any other launch instead of raising."""
    def missing(*a, **k):
        raise FileNotFoundError(2, "No such file or directory", "sky")

    monkeypatch.setattr(wd.subprocess, "run", missing)
    ok, msg = wd.Ops(yaml="x.yaml", envs=[], secrets=[]).launch(
        "spot-t-h100-1", wd.candidates()[0], 600.0)
    assert ok is False and "sky" in msg


def test_a_raising_launch_is_a_failed_launch_not_a_crash(tmp_path):
    clk = Clock()

    class Raising(FakeOps):
        def launch(self, cluster, cand, timeout_s):
            self.calls.append(("launch", cluster, cand.key))
            raise PermissionError(13, "Permission denied", "sky")

    ops = Raising(clk)
    w = wd.Watchdog(_cfg(tmp_path, max_rounds=1), wd.candidates()[:2], ops, clock=clk, sleep=clk.sleep)
    assert w.run() == "max_rounds"
    assert [c[0] for c in ops.calls].count("launch") == 2  # every candidate was still tried
    fails = [e for e in _events(tmp_path) if e["event"] == "launch_fail"]
    assert len(fails) == 2 and "Permission denied" in fails[0]["msg"]


def test_a_full_disk_under_the_event_log_does_not_kill_the_watchdog(tmp_path, monkeypatch):
    clk = Clock()
    ops = FakeOps(clk, launch={"H100:8@eu-north1/od": [True]}, states=["RUNNING", "STOPPED"])
    w = wd.Watchdog(_cfg(tmp_path), wd.candidates(), ops, clock=clk, sleep=clk.sleep)
    real_open = open

    def full(path, mode="r", *a, **k):
        if "w" in mode or "a" in mode:
            raise OSError(28, "No space left on device")
        return real_open(path, mode, *a, **k)

    monkeypatch.setattr("builtins.open", full)
    monkeypatch.setattr(wd.Path, "write_text", lambda self, *a, **k: full(self, "w"))
    w.acquire()
    w.watch_once(); w.watch_once()  # RUNNING, then STOPPED -> preempted, deleted
    assert w.cluster is None and ("down", "spot-t-h100-1") in ops.calls
    assert w.log_write_errors > 0
    assert [e["event"] for e in w.history][:2] == ["launch_start", "launch_ok"]  # still kept in memory
    monkeypatch.undo()
    w.event("disk_back")  # the log resumes once the disk has room
    assert _events(tmp_path)[-1]["event"] == "disk_back"
    assert _events(tmp_path)[-1]["log_write_errors"] == w.log_write_errors


def test_a_full_disk_keeps_the_previous_state_file(tmp_path, monkeypatch):
    """The state (spend, live cluster) is replaced whole or not at all, never truncated."""
    clk = Clock()
    ops = FakeOps(clk, launch={"H100:8@eu-north1/od": [True]})
    cfg = _cfg(tmp_path)
    w = wd.Watchdog(cfg, wd.candidates(), ops, clock=clk, sleep=clk.sleep)
    w.acquire()
    before = json.loads(Path(cfg.state).read_text())
    real_write = wd.Path.write_text

    def half(self, data, *a, **k):
        real_write(self, data[: len(data) // 2])
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(wd.Path, "write_text", half)
    w.spent += 5.0
    w.event("x")
    monkeypatch.undo()
    assert json.loads(Path(cfg.state).read_text()) == before
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []
    assert wd.Watchdog(cfg, wd.candidates(), ops, clock=clk, sleep=clk.sleep).cluster == before["cluster"]
