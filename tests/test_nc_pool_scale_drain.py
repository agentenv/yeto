"""Scaling a running async GRPO run's NC sampler pool in and out (``pool-scale`` /
``yeto run scale``): the pool policy (yeto.launch.nc.lease_pool) over the fake National
Compute of test_nc_elastic, with real store-dispenser samplers
(yeto.rl.engine.store_dispense) on an in-memory object store. Nothing here orders,
calls a real tool or opens a socket."""

from __future__ import annotations

import json
import subprocess
import threading

import pytest

from tests.test_nc_elastic import Clock, Cloud, _names, _pool, _run
from tests.test_nc_pool_run import _cfg
from tests.test_store_dispense import FakeObjects, _open, _rows, _slots, _write
from yeto.dashboard.async_rl import AsyncState
from yeto.launch.nc import cli, pool_run
from yeto.launch.pool.island_pool import GONE, READY
from yeto.rl import up
from yeto.rl.engine import store_dispense as sd
from yeto.rl.engine.incarnation_lease import read_lease
from yeto.rl.engine.task_dispense import DispenseEnded

NS = "run/dispense"
DISPENSE = {"backend": "store", "namespace": NS}


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("YETO_NC_ACCOUNTS", "1,4,5")

    def no_subprocess(*a, **k):
        raise AssertionError(f"no subprocess in these tests: {a}")

    monkeypatch.setattr(subprocess, "run", no_subprocess)


class DrainCloud(Cloud):
    """The fake National Compute plus the store dispenser's drain request (NcSshOps's)."""

    def __init__(self, clock, **kw):
        super().__init__(clock, **kw)
        self.ledger_objects = FakeObjects()

    def request_drain(self, m):
        m.drain_seq = sd.request_drain(self.ledger_objects, NS, m.shard, seq=f"{m.ticket}:{self.clock()}")
        self.log.append(("request_drain", m.ticket, m.shard, self.clock()))

    def drained(self, m):
        return m.drain_seq is not None and sd.drain_answered(self.ledger_objects, NS, m.shard, m.drain_seq)

    def cancel_drain(self, m):
        sd.request_drain(self.ledger_objects, NS, m.shard, on=False)
        m.drain_seq = None
        self.log.append(("cancel_drain", m.ticket, m.shard, self.clock()))


def _elastic(tmp_path, *, target=1, accounts=(1, 4, 5), cloud_cls=DrainCloud, pool_kw=None, **extra):
    if "data_slots" not in extra:
        extra.setdefault("sampler", {"dispense": DISPENSE})
    cfg, path = _cfg(tmp_path, target=target, accounts=list(accounts), **extra)
    clock = Clock()
    cloud = cloud_cls(clock)
    pool = _pool(clock, cloud, target=target, accounts=accounts, data_slots=extra.get("data_slots"),
                 **(pool_kw or {}))
    pool.scale_path = pool_run.scale_path(cfg)
    pool.scale_run = cli.run_id(cfg)
    pool.state_path = cli.pool_state_path(cfg)
    pool.state_path.parent.mkdir(parents=True, exist_ok=True)
    _run(pool, clock, clock.t + 1800)
    assert len(_ready(pool)) == target
    pool.save(pool.state_path)
    return cfg, path, clock, cloud, pool


def _ready(pool):
    return [m for m in pool.members if m.state == READY]


def _events(pool, name):
    return [e for e in pool.events.events if e["event"] == f"nc_elastic_{name}"]


class _Blocked(Exception):
    pass


def _idle(sampler):
    """Let a sampler holding nothing scan once (it answers a drain request) and stop it
    at its first wait instead of sleeping."""
    def stop(_s):
        raise _Blocked

    sampler._sleep = stop
    with pytest.raises(_Blocked):
        sampler(0)


def _sampler(path, cloud, shard, **kw):
    """A store-dispenser sampler of ``shard`` as engine_sampler_entry wires it: its claims
    are fenced when its incarnation lease names another incarnation (the pool's fence)."""
    def lease(h):
        return (read_lease(cloud.store, h) or {}).get("incarnation")

    inc = f"{shard}-i0"
    return _open(path, cloud.ledger_objects, holder=shard, inc=inc, groups=2, lease_of_holder=lease,
                 fenced=lambda: lease(shard) not in (None, inc), **kw)


def _save(pool):
    pool.save(pool.state_path)


# -- 1 -> 2 -> 1 while the run trains ---------------------------------------------

def test_one_two_one_drains_the_removed_node_with_no_task_lost_or_given_twice(tmp_path, capsys):
    cfg, path, clock, cloud, pool = _elastic(tmp_path, target=1)
    prompts = _write(tmp_path, _rows(10))
    first = _ready(pool)[0]
    s0 = _sampler(prompts, cloud, first.shard)
    seen = {first.shard: []}
    spots = _slots(s0(0))
    seen[first.shard] += spots
    s0.written(0, spots)

    # scale up: one command, the next tick orders; the new node gets a new shard
    assert pool_run.scale_pool(cfg, path, add=1) == 0
    assert "target 1 -> 2 (scale up)" in capsys.readouterr().out
    _run(pool, clock, clock.t + 1800)
    assert len(_ready(pool)) == 2 and pool.config.target == 2
    second = next(m for m in _ready(pool) if m is not first)
    assert second.shard != first.shard
    s1 = _sampler(prompts, cloud, second.shard)
    seen[second.shard] = []
    held0, held1 = _slots(s0(0)), _slots(s1(0))  # both hold tasks in flight
    seen[first.shard] += held0
    seen[second.shard] += held1
    _save(pool)

    # scale down, removing the new node by ticket; default mode: drain
    orders = len(cloud.kinds("order"))
    assert pool_run.scale_pool(cfg, path, remove=[second.ticket]) == 0
    out = capsys.readouterr().out
    assert "target 2 -> 1 (scale down)" in out and "scale down by drain" in out
    pool.tick()
    assert second.retiring_since is not None and second.state == READY  # still billed, still writing
    assert _events(pool, "scale_draining")[0]["ticket"] == second.ticket
    assert not cloud.kinds("fence") and not cloud.kinds("drain")
    assert s1(0) == []  # it claims nothing new
    s1.written(0, held1)  # its trials in flight are written and trained
    pool.tick()
    assert second.state == READY  # not answered yet: it held claims at its last scan
    _idle(s1)  # holding nothing: it answers the drain request
    pool.tick()
    assert second.state == GONE and cloud.tickets[second.ticket]["state"] == "ended"
    ev = _events(pool, "scaled_down")[-1]
    assert ev["ticket"] == second.ticket and ev["outcome"] == "drained" and ev["ended"] is True
    assert [e[1] for e in cloud.kinds("fence")] == [second.ticket]  # fenced after its groups
    _run(pool, clock, clock.t + 1800)
    assert len(cloud.kinds("order")) == orders and [m.ticket for m in _ready(pool)] == [first.ticket]

    # the remaining sampler takes every other task; nothing lost, nothing twice
    s0.written(0, held0)
    try:
        while True:
            spots = _slots(s0(0))
            seen[first.shard] += spots
            s0.written(0, spots)
    except DispenseEnded:
        pass
    every = [p for v in seen.values() for p in v]
    assert sorted(every) == list(range(10))
    tasks = {t["slot"]: t for t in sd.status(cloud.ledger_objects, NS)["tasks"]}
    assert all(t["state"] == "trained" for t in tasks.values()) and len(tasks) == 10
    assert {tasks[p]["holder"] for p in held1} == {second.shard}


def test_fence_mode_frees_the_removed_nodes_claims_at_once(tmp_path):
    cfg, path, clock, cloud, pool = _elastic(tmp_path, target=2)
    prompts = _write(tmp_path, _rows(4))
    keep, victim = sorted(_ready(pool), key=lambda m: m.created)
    sv = _sampler(prompts, cloud, victim.shard)
    sk = _sampler(prompts, cloud, keep.shard)
    lost = _slots(sv(0))
    assert lost == [0, 1]
    assert pool_run.scale_pool(cfg, path, 1, retire=[victim.ticket], mode="fence") == 0
    pool.tick()
    assert victim.state == GONE and _events(pool, "scaled_down")[0]["outcome"] == "fenced"
    assert not cloud.kinds("request_drain")
    got = _slots(sk(0))  # at once, without waiting for the claim lease
    assert sorted(got) == [0, 1]
    assert not any(sv.still_holds(p) for p in lost)  # its own check before a group write: fenced
    assert sv.written(0, lost)["trained"] == 0  # a late write report of the fenced node counts nothing
    sk.written(0, got)
    tasks = {t["slot"]: t for t in sd.status(cloud.ledger_objects, NS)["tasks"]}
    assert tasks[0]["state"] == "trained" and tasks[0]["holder"] == keep.shard
    assert tasks[0]["failure_reasons"] == ["0:release:fenced"]


def test_a_drain_that_never_answers_is_fenced_at_its_timeout(tmp_path):
    cfg, path, clock, cloud, pool = _elastic(tmp_path, target=2, pool_kw={"scale_drain_timeout_s": 600})
    victim = max(_ready(pool), key=lambda m: m.created)
    assert pool_run.scale_pool(cfg, path, 1) == 0
    pool.tick()
    assert victim.retiring_since is not None
    _run(pool, clock, clock.t + 540)
    assert victim.state == READY
    _run(pool, clock, clock.t + 120)
    assert victim.state == GONE and _events(pool, "scaled_down")[0]["outcome"] == "drain timeout"
    assert _events(pool, "scaled_down")[0]["waited_s"] >= 600


def test_scaling_up_while_a_node_drains_keeps_it_instead_of_ordering(tmp_path):
    cfg, path, clock, cloud, pool = _elastic(tmp_path, target=2)
    victim = max(_ready(pool), key=lambda m: m.created)
    assert pool_run.scale_pool(cfg, path, 1) == 0
    pool.tick()
    assert victim.retiring_since is not None
    _save(pool)
    orders = len(cloud.kinds("order"))
    assert pool_run.scale_pool(cfg, path, add=1) == 0
    pool.tick()
    assert victim.retiring_since is None and victim.state == READY
    assert len(cloud.kinds("order")) == orders and cloud.kinds("cancel_drain")
    assert _events(pool, "scale_drain_cancelled")
    body = json.loads(cloud.ledger_objects.read(sd.drain_key(NS, victim.shard)))
    assert body["drain"] is False  # its sampler claims again


def test_static_data_slots_without_a_dispenser_are_refused_unless_forced(tmp_path, capsys):
    cfg, path, clock, cloud, pool = _elastic(tmp_path, target=2, cloud_cls=Cloud, data_slots=2)
    assert pool_run.scale_pool(cfg, path, 1) == 2
    assert "static data_slots 2" in capsys.readouterr().out and not pool_run.scale_path(cfg).exists()
    victim = max(_ready(pool), key=lambda m: m.created)
    assert pool_run.scale_pool(cfg, path, 1, force_static=True) == 0
    assert "slot pauses" in capsys.readouterr().out
    pool.tick()
    assert victim.state == GONE  # fenced: a slot sampler cannot drain
    assert pool.pending[-1]["data_slot"] == victim.data_slot and pool.pending[-1]["shard"] == victim.shard
    assert _events(pool, "slot_paused")
    _save(pool)
    assert pool_run.scale_pool(cfg, path, 2, force_static=True) == 0
    pool.tick()
    new = pool.members[-1]
    assert new.data_slot == victim.data_slot and new.replaces_shard == victim.shard  # the slot continues


def test_a_pool_without_a_task_source_refuses_any_scale(tmp_path, capsys):
    cfg, path, clock, cloud, pool = _elastic(tmp_path, target=1, cloud_cls=Cloud, sampler={})
    assert pool_run.scale_pool(cfg, path, 2) == 2
    assert "have no dispenser" in capsys.readouterr().out


def test_the_spec_s_dispenser_counts_as_the_task_source(tmp_path):
    spec = {"prompts": {"path": "/work/p.jsonl", "dispense": {"backend": "store", "namespace": NS}}}
    cfg, _path = _cfg(tmp_path, spec_doc=spec)
    assert cli.pool_dispense(cfg)["namespace"] == NS
    cfg, _path = _cfg(tmp_path, spec_doc={"prompts": {"path": "/work/p.jsonl"}})
    assert cli.pool_dispense(cfg) is None


# -- concurrency, quota, breaker -----------------------------------------------------

def test_concurrent_add_requests_both_count(tmp_path):
    cfg, path, clock, cloud, pool = _elastic(tmp_path, target=1)
    rcs = []
    threads = [threading.Thread(target=lambda: rcs.append(pool_run.scale_pool(cfg, path, add=1))) for _ in range(2)]
    [t.start() for t in threads]
    [t.join(10) for t in threads]
    assert rcs == [0, 0]
    doc = pool_run.read_scale(cfg)
    assert doc["target"] == 3 and doc["seq"] == 2
    _run(pool, clock, clock.t + 1800)
    assert len(_ready(pool)) == 3


def test_a_target_past_the_allowed_accounts_is_refused(tmp_path, capsys):
    # account 2 is in the pool config but not in YETO_NC_ACCOUNTS (1,4,5): it never counts
    cfg, path, clock, cloud, pool = _elastic(tmp_path, target=1, accounts=(1, 2, 4, 5))
    assert pool_run.scale_pool(cfg, path, 4) == 2
    out = capsys.readouterr().out
    assert "3 usable accounts [1, 4, 5]" in out and not pool_run.scale_path(cfg).exists()
    assert pool_run.scale_pool(cfg, path, 3, dry_run=True) == 0
    assert "would be accepted" in capsys.readouterr().out and not pool_run.scale_path(cfg).exists()


def test_no_free_account_is_an_event_not_a_silent_gap(tmp_path, monkeypatch):
    cfg, path, clock, cloud, pool = _elastic(tmp_path, target=1)
    monkeypatch.setattr(cloud, "account_free", lambda a: a == 1)  # 4 and 5 busy elsewhere
    assert pool_run.scale_pool(cfg, path, 2) == 0
    pool.tick()
    pool.tick()
    refused = [e for e in _events(pool, "order_refused") if "no free slot" in e["why"]]
    assert len(refused) == 1  # once, not every tick


def test_a_tripped_breaker_refuses_a_scale_up_but_not_a_scale_down(tmp_path, capsys):
    cfg, path, clock, cloud, pool = _elastic(tmp_path, target=2)
    pool.breaker.tripped = True
    _save(pool)
    assert pool_run.scale_pool(cfg, path, 3) == 2
    assert "breaker is tripped" in capsys.readouterr().out
    assert pool_run.scale_pool(cfg, path, 1) == 0


# -- scale while the pool replaces nodes ---------------------------------------------

def test_an_eviction_during_a_drain_keeps_the_draining_node_instead_of_ordering(tmp_path):
    cfg, path, clock, cloud, pool = _elastic(tmp_path, target=2)
    keep, victim = sorted(_ready(pool), key=lambda m: m.created)
    assert pool_run.scale_pool(cfg, path, 1, retire=[victim.ticket]) == 0
    pool.tick()
    assert victim.retiring_since is not None
    orders = len(cloud.kinds("order"))
    cloud.tickets[keep.ticket].update(state="ended", reason="preempted")  # the kept node is evicted
    pool.tick()
    assert keep.state == GONE and "evicted" in _names(pool)
    assert victim.retiring_since is None and victim.state == READY  # it serves the target again
    assert len(cloud.kinds("order")) == orders


def test_an_eviction_of_the_draining_node_orders_nothing(tmp_path):
    cfg, path, clock, cloud, pool = _elastic(tmp_path, target=2)
    keep, victim = sorted(_ready(pool), key=lambda m: m.created)
    assert pool_run.scale_pool(cfg, path, 1, retire=[victim.ticket]) == 0
    pool.tick()
    orders = len(cloud.kinds("order"))
    cloud.tickets[victim.ticket].update(state="ended", reason="preempted")
    pool.tick()
    _run(pool, clock, clock.t + 1800)
    assert victim.state == GONE and _events(pool, "scaled_down")[-1]["outcome"] == "evicted"
    assert len(cloud.kinds("order")) == orders and not pool.pending
    assert [m.ticket for m in _ready(pool)] == [keep.ticket]


def test_a_draining_node_at_its_lease_end_goes_without_a_replacement(tmp_path):
    cfg, path, clock, cloud, pool = _elastic(tmp_path, target=2, pool_kw={"scale_drain_timeout_s": 10**6})
    victim = max(_ready(pool), key=lambda m: m.created)
    assert pool_run.scale_pool(cfg, path, 1, retire=[victim.ticket]) == 0
    pool.tick()
    seen = len(_events(pool, "ordered"))
    keep = next(m for m in _ready(pool) if m is not victim)
    _run(pool, clock, victim.lease_until - pool.config.end_margin_s + 60)
    assert victim.state == GONE and _events(pool, "scaled_down")[-1]["outcome"] == "lease end"
    # the kept node's own lease-end replacement is the only order; none replaces the victim
    assert [e["replaces"] for e in _events(pool, "ordered")[seen:]] == [keep.ticket]
    assert not [p for p in pool.pending if p.get("from") == victim.ticket]


def test_a_node_with_a_replacement_in_flight_drains_and_the_replacement_goes_now(tmp_path):
    cfg, path, clock, cloud, pool = _elastic(tmp_path, target=1)
    old = _ready(pool)[0]
    _run(pool, clock, old.lease_until - pool.config.replace_lead_s + 120)
    new = next(m for m in pool.members if m.replaces == old.ticket)
    assert new.state != READY and old.replaced_by == new.ticket
    _save(pool)
    assert pool_run.scale_pool(cfg, path, 0) == 0
    pool.tick()
    assert new.state == GONE and old.retiring_since is not None and old.state == READY
    assert _events(pool, "scaled_down")[0]["outcome"] == "linked to a draining node"
    sd.StoreLedger(cloud.ledger_objects, NS, rows=1, fingerprint=b"x", holder=old.shard,
                   incarnation="i").answer_drain(old.drain_seq)
    pool.tick()
    assert old.state == GONE and not pool.live() and not pool.stopped


def test_a_draining_member_survives_a_pool_restart(tmp_path):
    from yeto.launch.pool.island_pool import IslandPool

    cfg, path, clock, cloud, pool = _elastic(tmp_path, target=2)
    victim = max(_ready(pool), key=lambda m: m.created)
    assert pool_run.scale_pool(cfg, path, 1) == 0
    pool.tick()
    _save(pool)
    again = IslandPool.restore(pool.state_path, cli.pool_config(cfg, allowed=None), cloud, clock=clock,
                               log=lambda m: None, events=pool.events)
    again.scale_path, again.scale_run = pool.scale_path, pool.scale_run
    v2 = next(m for m in again.members if m.ticket == victim.ticket)
    assert v2.retiring_since is not None and v2.drain_seq
    orders = len(cloud.kinds("order"))
    again.tick()
    assert len(cloud.kinds("order")) == orders and v2.state == READY
    sd.StoreLedger(cloud.ledger_objects, NS, rows=1, fingerprint=b"x", holder=v2.shard,
                   incarnation="i").answer_drain(v2.drain_seq)
    again.tick()
    assert v2.state == GONE


# -- the CLI --------------------------------------------------------------------------

def test_pool_scale_cli_add_remove_and_mode(tmp_path, capsys):
    cfg, path, clock, cloud, pool = _elastic(tmp_path, target=1)
    assert cli.main(["pool-scale", "--config", path, "--add", "1", "--fence"]) == 0
    doc = pool_run.read_scale(cfg)
    assert doc["target"] == 2 and doc["mode"] == "fence"
    with pytest.raises(SystemExit):
        cli.main(["pool-scale", "--config", path, "--target", "2", "--add", "1"])
    _run(pool, clock, clock.t + 1800)
    _save(pool)
    victim = max(_ready(pool), key=lambda m: m.created)
    assert cli.main(["pool-scale", "--config", path, "--remove", victim.ticket]) == 0
    doc = pool_run.read_scale(cfg)
    assert doc["target"] == 1 and doc["retire"] == [victim.ticket] and doc["mode"] is None
    pool.tick()
    assert victim.retiring_since is not None  # the pool's default: drain


# -- yeto run scale / run up guards ---------------------------------------------------

def _run_cfg(tmp_path, monkeypatch, **budget):
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
    pool_cfg = {"target": 1, "accounts": [1, 4, 5], "shard_base": "nc", "run": "run-a"}
    (tmp_path / "pool.json").write_text(json.dumps(pool_cfg))
    return up.Run({"run": "r", "store": "s3://b/runs/r", "pool": {"config": str(tmp_path / "pool.json")},
                   "env": {"YETO_NC_STATE_DIR": str(tmp_path / "state")},
                   "sandbox": {"max_run_containers": 10, "max_containers_per_sampler": 8},
                   "trainer": {"argv": [], "usd_per_h": 0},
                   "budget": {"usd": 100, "nc_usd_per_h": 24, **budget}})


def _pool_state(tmp_path, members, target=None):
    path = tmp_path / "state" / "pool-nc.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"members": members, "target": target or len(members)}))


def test_run_spend_and_the_container_line_follow_the_billed_nodes(tmp_path, monkeypatch):
    import time as _time

    run = _run_cfg(tmp_path, monkeypatch)
    now = _time.time()
    run.state["t0"] = now - 3600
    _pool_state(tmp_path, [{"ticket": "a", "state": "ready", "granted": now - 3600},
                           {"ticket": "b", "state": "ready", "granted": now - 1800},
                           {"ticket": "c", "state": "gone", "granted": now - 3600, "charged_to": now - 2700}],
                target=5)  # a target of 5 bills nothing by itself
    assert run.nc_node_hours(now) == pytest.approx(1.75, abs=0.01)
    assert run.billed_nodes() == 2
    assert run.container_limit() == 16  # 8 per billed node, above the fixed 10


def test_run_scale_refuses_a_scale_up_past_the_budget_or_after_a_breaker(tmp_path, monkeypatch):
    import time as _time

    calls = []
    monkeypatch.setattr(up.subprocess, "run",
                        lambda argv, **k: calls.append(argv) or subprocess.CompletedProcess(argv, 0))
    run = _run_cfg(tmp_path, monkeypatch, usd=90)
    now = _time.time()
    run.state["t0"] = now - 3 * 3600
    _pool_state(tmp_path, [{"ticket": "a", "state": "ready", "granted": now - 3 * 3600}])  # $72 spent
    assert run.scale(add=1) == 2 and not calls  # $72 + one new node hour $24 > $90
    run.cfg["budget"]["usd"] = 200
    assert run.scale(add=1) == 0
    assert calls[-1][-2:] == ["--add", "1"]
    run.state["steps"] = {"trainer": {"at": "2026-11-11T08:00:00Z"}}
    run.state["breaker"] = {"at": "2026-11-11T09:00:00Z", "why": "budget $250"}
    assert run.scale(3) == 2
    assert run.scale(remove=["a"], mode="fence") == 0  # a scale down is never refused here
    assert calls[-1][-3:] == ["--remove", "a", "--fence"]
    assert [s["remove"] for s in run.state["scales"]] == [None, ["a"]]
    run.state["steps"]["trainer"]["at"] = "2026-11-11T10:00:00Z"  # a later run up resumed the trainer
    assert run.scale(3) == 0


def test_run_up_preflight_refuses_a_pool_shard_the_trainer_does_not_read(tmp_path, monkeypatch):
    run = _run_cfg(tmp_path, monkeypatch)
    run.cfg["trainer"]["argv"] = ["--rl-buffer-shards", "nc-0,nc-1"]
    assert "no shard_limit" in run.shard_problem()
    (tmp_path / "pool.json").write_text(json.dumps({"target": 1, "accounts": [1], "shard_base": "nc",
                                                    "run": "run-a", "shard_limit": 3}))
    assert "lacks ['nc-2']" in run.shard_problem()
    run.cfg["trainer"]["argv"] = ["--rl-buffer-shards=nc-0,nc-1,nc-2"]
    assert run.shard_problem() is None
    run.cfg["trainer"]["argv"] = []
    assert run.shard_problem() is None


def test_yeto_run_scale_parser():
    import argparse

    p = argparse.ArgumentParser()
    up.add_parser(p.add_subparsers(dest="cmd"))
    args = p.parse_args(["run", "scale", "--config", "r.json", "--add", "1", "--drain"])
    assert (args.target, args.add, args.mode) == (None, 1, "drain")
    args = p.parse_args(["run", "scale", "--config", "r.json", "--samplers", "0"])
    assert args.target == 0 and args.mode is None


# -- the dashboard ------------------------------------------------------------------

def test_the_dashboard_shows_the_pool_size_through_a_scale(tmp_path):
    cfg, path, clock, cloud, pool = _elastic(tmp_path, target=1)
    assert pool_run.scale_pool(cfg, path, add=1) == 0
    _run(pool, clock, clock.t + 1800)
    _save(pool)
    victim = max(_ready(pool), key=lambda m: m.created)
    assert pool_run.scale_pool(cfg, path, remove=[victim.ticket]) == 0
    pool.tick()
    state = AsyncState()
    for e in pool.events.events:
        state.fold_pool(e)
    mid = state.view()["pool_scale"]
    assert mid["target"] == 1 and mid["draining"] == [victim.shard] and len(mid["ready"]) == 2
    _idle(_sampler(_write(tmp_path, _rows(2)), cloud, victim.shard))
    pool.tick()
    state = AsyncState()
    for e in pool.events.events:
        state.fold_pool(e)
    end = state.view()["pool_scale"]
    assert end["ready"] == [m.shard for m in _ready(pool)] and not end["draining"]
    targets = [p["target"] for p in end["points"]]
    assert targets[-1] == 1 and targets.index(1) > targets.index(2)
    assert state.pool_exit(victim.shard)["event"] == "nc_elastic_scaled_down"


# -- the National Compute ops -------------------------------------------------------

def test_ssh_ops_drain_request_goes_to_the_dispensers_ledger_in_the_run_store(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from yeto.launch.pool.island_pool import Member

    cfg, _path = _cfg(tmp_path, sampler={"dispense": DISPENSE})
    ops = cli.NcSshOps(cfg, run=lambda *a, **k: None, provider=SimpleNamespace())
    objects = FakeObjects()
    monkeypatch.setattr(ops, "_fence_store", lambda: SimpleNamespace(objects=objects))
    m = Member(1, "t9", "nc-3", state=READY)
    ops.request_drain(m)
    assert json.loads(objects.read(sd.drain_key(NS, "nc-3")))["drain"] is True and m.drain_seq
    assert not ops.drained(m)
    sd.StoreLedger(objects, NS, rows=1, fingerprint=b"x", holder="nc-3", incarnation="i").answer_drain(m.drain_seq)
    assert ops.drained(m)
    ops.cancel_drain(m)
    assert json.loads(objects.read(sd.drain_key(NS, "nc-3")))["drain"] is False and m.drain_seq is None
    cfg, _path = _cfg(tmp_path, sampler={})
    bare = cli.NcSshOps(cfg, run=lambda *a, **k: None, provider=SimpleNamespace())
    with pytest.raises(cli.NcError, match="no store dispenser"):
        bare.request_drain(m)


def test_run_up_preflight_refuses_score_centering_without_the_samplers_top_logprobs(tmp_path, monkeypatch):
    run = _run_cfg(tmp_path, monkeypatch)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"sampler_id": "__SHARD__"}))
    (tmp_path / "pool.json").write_text(json.dumps({"target": 1, "accounts": [1], "shard_base": "nc",
                                                    "run": "run-a", "spec": str(spec)}))
    run.cfg["trainer"]["argv"] = ["--rl-loss", "policy_loss"]
    assert run.rollout_fields_problem() is None
    run.cfg["trainer"]["argv"] = ["--rl-loss", "score_centering"]
    assert "top_logprobs_num unset" in run.rollout_fields_problem()
    spec.write_text(json.dumps({"rollout_fields": {"top_logprobs_num": 128}}))
    assert run.rollout_fields_problem() is None
    run.cfg["trainer"]["argv"] = ["--rl-loss=score_centering", "--rl-rollout-top-logprobs-num", "64"]
    assert "top-64" in run.rollout_fields_problem()
    assert run.preflight() is False


def test_run_up_sees_its_finished_trainer_child_as_exited():
    """run up starts the trainer with Popen and never waits on it: the finished launcher
    is a zombie child, which kill(pid, 0) alone reports alive (the trainer's end was
    never seen and release_nc_on_finish never ran)."""
    import os
    import sys
    import time

    from yeto.rl import up

    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    pid = proc.pid
    deadline = time.time() + 30
    while time.time() < deadline:  # exited but not reaped: a zombie
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        if up._alive(pid) is False:
            break
        time.sleep(0.05)
    assert up._alive(pid) is False
    assert up._alive(os.getppid()) is True  # not a child: kill 0 decides
    assert up._alive(None) is False
    proc.returncode = 0  # reaped above; keep Popen.__del__ quiet
