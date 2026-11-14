"""Eviction-tolerant National Compute sampler pool (yeto.launch.nc.elastic +
lease_pool): CPU tests over a fake National Compute and fake nodes."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from yeto.launch.nc import cli, elastic
from yeto.launch.nc import client as ncc
from yeto.launch.nc import tunnel
from yeto.launch.nc.lease_pool import PoolConfig
from yeto.launch.pool import status as el
from yeto.launch.pool.island_pool import DRAINING, FENCING, GONE, READY, IslandPool
from yeto.rl.engine.incarnation_lease import IncarnationGate, IncarnationLease, read_lease
from yeto.rl.engine.sampler_resume import handoff_requested

H = 3600.0


@pytest.fixture(autouse=True)
def _state(tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("NC_MANAGE", "nc-manage")
    monkeypatch.setenv("NC_QUERY", "nc-query")


class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


class Objects:
    """An object API (read / write / list) in memory: the run's store."""

    def __init__(self):
        self.data = {}

    def read(self, key):
        return self.data.get(key)

    def write(self, key, data):
        self.data[key] = bytes(data)

    def list(self, prefix):
        return sorted(k for k in self.data if k.startswith(prefix))

    def delete(self, key):
        self.data.pop(key, None)


class Cloud:
    """National Compute + nodes. Each ticket: account, state, grant time, lease end,
    end reason; each node: reachable, sampler state, joined."""

    def __init__(self, clock, *, grant_after=60.0, lease_s=4 * H, ready_after=1200.0, balance=10**10):
        self.clock, self.grant_after, self.lease_s, self.ready_after = clock, grant_after, lease_s, ready_after
        self.bal = balance
        self.tickets = {}
        self.n = 0
        self.log = []
        self.order_error = None  # exception to raise on order
        self.lose_before_grant = 0  # next N orders end "lost" before a grant
        self.drain_error = 0  # next N drains raise
        self.store = SimpleNamespace(objects=Objects())
        self.ledger = []  # (run, account, ticket, open)

    # PoolOps -------------------------------------------------------------
    def order(self, account):
        if self.order_error is not None:
            raise self.order_error
        self.n += 1
        t = f"t{self.n}"
        now = self.clock()
        rec = {"account": account, "state": "waiting", "created": now, "grant": now + self.grant_after,
               "until": None, "reason": None, "reachable": True, "sampler": "absent", "exit": None,
               "joined_at": None, "preempt_at": None, "lose": self.lose_before_grant > 0, "starts": 0}
        if self.lose_before_grant:
            self.lose_before_grant -= 1
        self.tickets[t] = rec
        self.ledger.append(["run-a", account, t, True])
        self.log.append(("order", t, account, self.clock()))
        return t

    def _advance(self, t):
        rec = self.tickets[t]
        now = self.clock()
        if rec["state"] == "waiting" and now >= rec["grant"]:
            if rec["lose"]:
                rec.update(state="ended", reason="lost")
            else:
                rec.update(state="active", until=rec["grant"] + self.lease_s)
        if rec["sampler"] == "running" and rec["joined_at"] is None and rec["reachable"] \
                and now >= rec["started"] + self.ready_after:
            rec["joined_at"] = now
        return rec

    def poll(self, m):
        rec = self._advance(m.ticket)
        return el.LeaseStatus(rec["state"], rec["until"], rec["reason"], rec["preempt_at"], 8, 3_000_000)

    def start(self, m, shards):
        rec = self.tickets[m.ticket]
        rec.update(sampler="running", started=self.clock(), shards=list(shards), starts=rec["starts"] + 1,
                   continues=m.replaces_shard, slot=m.data_slot, lost_pred=m.predecessor_lost)
        self.log.append(("start", m.ticket, m.shard, self.clock()))

    def restart(self, m, shards):
        rec = self.tickets[m.ticket]
        rec.update(sampler="running", started=self.clock(), joined_at=None, starts=rec["starts"] + 1)
        self.log.append(("restart", m.ticket, self.clock()))

    def ready(self, m):  # not used: probe is present
        raise AssertionError("probe-based readiness")

    def probe(self, m):
        rec = self._advance(m.ticket)
        if not rec["reachable"]:
            return el.Probe(node=el.PROBE_UNREACHABLE)
        return el.Probe(node="ok", sampler=rec["sampler"], exit_code=rec["exit"], joined=rec["joined_at"] is not None)

    def fence(self, m):
        elastic.fence_shard(self.store, m.shard, by=m.ticket, clock=self.clock)
        self.log.append(("fence", m.ticket, m.shard, self.clock()))

    def drain(self, m):
        if self.drain_error:
            self.drain_error -= 1
            raise RuntimeError("nc-manage delete timed out")
        rec = self.tickets[m.ticket]
        if rec["state"] != "ended":
            rec.update(state="ended", reason="released")
        rec["sampler"] = "absent"
        for row in self.ledger:
            if row[2] == m.ticket:
                row[3] = False
        self.log.append(("drain", m.ticket, self.clock()))

    def balance(self):
        return self.bal

    def account_free(self, account):
        return not any(r["account"] == account and r["state"] != "ended" for r in self.tickets.values())

    def run_tickets(self):
        return [(a, t) for run, a, t, open_ in self.ledger if open_ and run == "run-a"]

    # helpers -------------------------------------------------------------
    def kinds(self, kind):
        return [e for e in self.log if e[0] == kind]


def _pool(clock, cloud, *, target=1, accounts=(1, 4), data_slots=None, **elastic):
    cfg = PoolConfig(target=target, accounts=list(accounts), shard_base="nc", data_slots=data_slots,
                     elastic=el.ElasticConfig(**elastic))
    events = el.EventLog(log=lambda m: None, clock=clock, run="run-a")
    return IslandPool(cfg, cloud, clock=clock, log=lambda m: None, events=events)


def _run(pool, clock, until, step=60.0):
    while clock.t < until and not pool.stopped:
        pool.tick()
        clock.t += step


def _names(pool):
    return [e["event"].removeprefix("nc_elastic_") for e in pool.events.events]


def _ready_member(pool, clock, cloud):
    _run(pool, clock, clock.t + 1800)
    ready = [m for m in pool.members if m.state == READY]
    assert ready, _names(pool)
    return ready[0]


# -- classification -----------------------------------------------------------

def test_classify_tells_the_cases_apart():
    cfg = el.ElasticConfig(unreachable_grace_s=300, stall_s=900)
    ok = el.Probe(node="ok", sampler="running", joined=True, progress_age_s=10)
    active = el.LeaseStatus("active", lease_until=10_000)

    def kind(status, probe=ok, now=1000, since=None, joined=True):
        return el.classify(status, probe, now=now, unreachable_since=since, cfg=cfg, joined=joined).kind

    assert kind(active) == el.HEALTHY
    assert kind(el.LeaseStatus("unknown")) == el.CONTROL_BLIP
    assert kind(el.LeaseStatus("ended", 10_000, "preempted")) == el.EVICTED
    assert kind(el.LeaseStatus("ended", 10_000, "lost")) == el.EVICTED
    assert kind(el.LeaseStatus("ended", 10_000, "admin")) == el.EVICTED
    assert kind(el.LeaseStatus("ended", 10_000, "balance")) == el.BALANCE
    assert kind(el.LeaseStatus("ended", 10_000, "released")) == el.RELEASED
    assert kind(el.LeaseStatus("ended", 500, None)) == el.EXPIRED
    assert kind(active, now=10_001) == el.OVERDUE  # the platform leaves it active (and billed)
    down = el.Probe(node=el.PROBE_UNREACHABLE)
    assert kind(active, down, now=1000, since=900) == el.NETWORK_BLIP
    assert kind(active, down, now=1300, since=1000) == el.LOST
    assert kind(el.LeaseStatus("active", 10_000, preempt_at=2000)) == el.PREEMPT_NOTICE
    assert el.effective_end(el.LeaseStatus("active", 10_000, preempt_at=2000)) == 2000
    assert kind(active, el.Probe(node="ok", sampler="exited", exit_code=1, joined=True)) == el.CRASHED
    assert kind(active, el.Probe(node="ok", sampler="exited", exit_code=0, joined=True)) == el.FINISHED
    assert kind(active, el.Probe(node="ok", sampler="running", joined=True, progress_age_s=1000)) == el.CRASHED
    assert kind(active, el.Probe(node="ok", sampler="running"), joined=False) == el.WAITING
    assert kind(el.LeaseStatus("waiting"), None, joined=False) == el.WAITING


def test_probe_parses_the_node_line():
    p = el.Probe.parse("noise\nYETO_NC_PROBE status=running exit=0 joined=1 age=42\n", 0)
    assert (p.node, p.sampler, p.joined, p.progress_age_s) == ("ok", "running", True, 42.0)
    p = el.Probe.parse("YETO_NC_PROBE status=exited exit=137 joined=1 age=-", 0)
    assert (p.sampler, p.exit_code, p.progress_age_s) == ("exited", 137, None)
    assert el.Probe.parse("YETO_NC_PROBE status=absent exit=- joined=0 age=-", 0).sampler == "absent"
    assert el.Probe.parse("", 255).node == el.PROBE_UNREACHABLE
    assert el.Probe.parse("ssh: connect timed out", 1).node == el.PROBE_UNREACHABLE
    script = elastic.probe_script("yeto-sampler-nc-0", "/mnt/nvme/work/sampler/events-x*.jsonl")
    assert "docker inspect" in script and "rm " not in script and "stop" not in script


def test_elastic_config_refuses_unknown_and_bad_values():
    with pytest.raises(ValueError, match="unknown keys"):
        el.ElasticConfig.from_dict({"unreachable_grace": 1})
    with pytest.raises(ValueError, match=">= 0"):
        el.ElasticConfig.from_dict({"start_timeout_s": -1})
    assert el.ElasticConfig.from_dict(None).max_sampler_restarts == 2


# -- eviction, expiry, renewal ------------------------------------------------

def test_eviction_fences_releases_and_replacement_continues_the_shard():
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, target=2, accounts=(1, 4, 5), data_slots=2)
    _run(pool, clock, 1800)
    first = {m.ticket: m for m in pool.members}
    assert all(m.state == READY for m in first.values())
    victim = pool.members[0]
    cloud.tickets[victim.ticket].update(state="ended", reason="preempted")
    pool.tick()
    assert victim.state == GONE and victim.end_reason == "preempted"
    assert ("fence", victim.ticket, victim.shard, clock.t) in cloud.log
    assert handoff_requested(cloud.store, victim.shard)
    assert read_lease(cloud.store, victim.shard)["incarnation"] == f"pool-fence:{victim.ticket}"
    new = pool.members[-1]
    assert new.replaces_shard == victim.shard and new.data_slot == victim.data_slot and new.predecessor_lost
    assert new.account == 1  # the evicted account is free again
    clock.t += 60
    _run(pool, clock, clock.t + 1800)
    assert new.state == READY
    assert cloud.tickets[new.ticket]["continues"] == victim.shard and cloud.tickets[new.ticket]["lost_pred"]
    names = _names(pool)
    assert names.index("evicted") < names.index("shard_fenced") < names.index("released") < names.index("ordered", 2)
    ready = [e for e in pool.events.events if e["event"] == "nc_elastic_member_ready"]
    assert ready[-1]["order_to_ready_s"] > 0  # the replacement time the guide reports


def test_platform_keeps_a_lease_active_past_lease_until_the_pool_drains_it_first():
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, accounts=(1,))
    _run(pool, clock, 4 * H + 1200)
    t1 = cloud.tickets["t1"]
    drains = cloud.kinds("drain")
    assert drains[0][1] == "t1" and drains[0][2] < t1["until"]  # before lease_until, by end_margin
    assert t1["state"] == "ended" and t1["reason"] == "released"
    assert pool.members[1].replaces_shard == pool.members[0].shard  # the next node continues the prompts
    assert "lease_overdue" not in _names(pool)


def test_overdue_lease_is_alerted_and_deleted():
    """The controller was away across lease_until (e.g. restarted late); the ticket
    still reads active: alert + delete, then a replacement."""
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, accounts=(1,))
    m = _ready_member(pool, clock, cloud)
    clock.t = cloud.tickets[m.ticket]["until"] + 30  # no tick in between
    pool.tick()
    names = _names(pool)
    assert "lease_overdue" in names
    assert m.state == GONE and cloud.tickets[m.ticket]["reason"] == "released"
    assert ("fence", m.ticket, m.shard, clock.t) in cloud.log
    assert len(cloud.kinds("order")) == 2


def test_preempt_notice_moves_the_replacement_forward():
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, accounts=(1, 4))
    m = _ready_member(pool, clock, cloud)
    cloud.tickets[m.ticket]["preempt_at"] = clock.t + 1800
    pool.tick()
    assert m.lease_until == clock.t + 1800
    assert m.replaced_by is not None  # ordered now, not 1 h before the 4 h end
    assert "preempt_notice" in _names(pool)


def test_replacement_order_failure_keeps_the_old_member_and_retries():
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, accounts=(1, 4), breaker_failures=50)
    m = _ready_member(pool, clock, cloud)
    cloud.order_error = ncc.NcError("409 account already has an open request")
    _run(pool, clock, cloud.tickets[m.ticket]["until"] - H + 300)
    assert m.state == READY and m.replaced_by is None
    assert "order_failed" in _names(pool)
    cloud.order_error = None
    pool.tick()
    assert m.replaced_by is not None


def test_replacement_lost_before_grant_is_replanned():
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, accounts=(1, 4))
    m = _ready_member(pool, clock, cloud)
    clock.t = cloud.tickets[m.ticket]["until"] - H + 10
    cloud.lose_before_grant = 1
    pool.tick()
    first_try = m.replaced_by
    _run(pool, clock, clock.t + 180)
    assert cloud.tickets[first_try]["reason"] == "lost"
    assert m.replaced_by not in (None, first_try)  # ordered again
    assert pool.breaker.failures == 1


# -- breaker and budget -------------------------------------------------------

def test_consecutive_failures_open_then_trip_the_breaker():
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, accounts=(1,), breaker_failures=2, breaker_cooldown_s=600, breaker_max_trips=2)
    cloud.lose_before_grant = 100
    _run(pool, clock, 200)  # two orders lost -> open
    names = _names(pool)
    assert names.count("evicted") == 2 and "breaker_open" in names
    orders = len(cloud.kinds("order"))
    _run(pool, clock, pool.breaker.open_until - 1)
    assert len(cloud.kinds("order")) == orders  # no order during the cooldown
    _run(pool, clock, clock.t + 3 * H)
    assert "breaker_tripped" in _names(pool)
    assert pool.stopped == "breaker tripped and no member left"
    assert len(cloud.kinds("order")) == 4


def test_breaker_resets_on_a_ready_member():
    b = el.Breaker()
    cfg = el.ElasticConfig(breaker_failures=3)
    b.failure(0, cfg)
    b.failure(0, cfg)
    b.success()
    assert b.failure(0, cfg) is None and b.allows(0)


def test_budget_caps_spend_and_nodes():
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, accounts=(1, 4), max_run_spend_ucr=int(3.5 * 24_000_000))
    _run(pool, clock, 3.5 * H)
    assert len(cloud.kinds("order")) == 1  # one node hour more would pass the cap
    refused = [e for e in pool.events.events if e["event"] == "nc_elastic_order_refused"]
    assert refused and "max_run_spend_ucr" in refused[0]["why"]
    assert 3.0 * 24e6 < pool.budget.spent_ucr < 3.6 * 24e6
    b = el.Budget()
    assert "max_nodes" in b.refusal(el.ElasticConfig(max_nodes=1), billed_nodes=1, target=1, hourly_ucr=1)
    assert "max_orders" in el.Budget(orders=3).refusal(el.ElasticConfig(max_orders=3), billed_nodes=0, target=1,
                                                       hourly_ucr=1)


# -- revived node: double-write fence ----------------------------------------

def _entry(sampler, inc, appended_at):
    return SimpleNamespace(sampler_round={"sampler": sampler, "incarnation": inc}, appended_at=appended_at)


def test_a_lost_node_that_comes_back_cannot_add_groups():
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, accounts=(1, 4), unreachable_grace_s=300)
    m = _ready_member(pool, clock, cloud)
    # the sampler on the node holds its shard's lease
    old = IncarnationLease(cloud.store, m.shard, "inc-old", clock=clock)
    old.claim()
    entries = [_entry(m.shard, "inc-old", clock.t - 600)]
    cloud.tickets[m.ticket]["reachable"] = False
    _run(pool, clock, clock.t + 400)
    assert "node_lost" in _names(pool) and m.state == GONE
    assert cloud.tickets[m.ticket]["reason"] == "released"  # no billing for a dead node
    fenced_at = [e[3] for e in cloud.log if e[0] == "fence"][0]
    # the node comes back: its next lease read fences it, it writes nothing more
    cloud.tickets[m.ticket]["reachable"] = True
    assert old.check_before_write() is True and old.fenced_by == f"pool-fence:{m.ticket}"
    assert handoff_requested(cloud.store, m.shard)
    # a group it wrote anyway after the fence is stale to the trainer; the earlier one is kept
    entries.append(_entry(m.shard, "inc-old", fenced_at + 5))
    gate = IncarnationGate(cloud.store, lambda: entries, monotonic=clock)
    assert gate.verdict(entries[0], 0) is None
    assert gate.verdict(entries[1], 1) is not None
    # its successor continues the shard, told not to wait for a LEAVE that never comes
    succ = pool.members[-1]
    assert succ.replaces_shard == m.shard and succ.predecessor_lost


# -- blips --------------------------------------------------------------------

def test_network_blip_is_not_taken_for_a_lost_node():
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, accounts=(1, 4), unreachable_grace_s=300)
    m = _ready_member(pool, clock, cloud)
    orders = len(cloud.kinds("order"))
    cloud.tickets[m.ticket]["reachable"] = False
    _run(pool, clock, clock.t + 240)  # 4 failed probes, inside the grace
    cloud.tickets[m.ticket]["reachable"] = True
    _run(pool, clock, clock.t + 120)
    names = _names(pool)
    assert names.count("network_blip") == 1 and "node_reachable_again" in names
    assert "node_lost" not in names and m.state == READY
    assert len(cloud.kinds("order")) == orders and not cloud.kinds("fence") and not cloud.kinds("drain")


def test_control_plane_failure_changes_nothing_and_alerts_once():
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, accounts=(1, 4), control_plane_alert_s=600)
    m = _ready_member(pool, clock, cloud)
    good = cloud.poll
    cloud.poll = lambda member: (_ for _ in ()).throw(ncc.NcError("nc-query exit 1: 503"))
    _run(pool, clock, clock.t + 900)
    assert m.state == READY and not cloud.kinds("drain")
    assert _names(pool).count("control_plane_unreachable") == 1
    cloud.poll = good
    pool.tick()
    assert "control_plane_recovered" in _names(pool)


# -- sampler crashes ----------------------------------------------------------

def test_crashed_sampler_restarts_in_place_then_the_node_is_replaced():
    clock = Clock()
    cloud = Cloud(clock, ready_after=300)
    pool = _pool(clock, cloud, accounts=(1, 4), max_sampler_restarts=1)
    m = _ready_member(pool, clock, cloud)
    cloud.tickets[m.ticket].update(sampler="exited", exit=137)
    pool.tick()
    assert cloud.kinds("restart") and m.restarts == 1 and m.state != GONE
    _run(pool, clock, clock.t + 600)
    assert m.state == READY
    cloud.tickets[m.ticket].update(sampler="exited", exit=1)
    pool.tick()
    assert "sampler_crash_loop" in _names(pool) and m.state == GONE
    assert pool.members[-1].replaces_shard == m.shard


def test_a_restart_before_the_first_join_does_not_use_up_the_crash_budget():
    """Setup-time restarts (the sampler died before it ever joined) are not a crash
    loop: the first crash after the join (e.g. an OOM kill) restarts in place."""
    clock = Clock()
    cloud = Cloud(clock, ready_after=300)
    pool = _pool(clock, cloud, accounts=(1, 4), max_sampler_restarts=1)
    while not any(k == "start" for k, *_ in cloud.log):
        pool.tick()
        clock.t += 60
    m = pool.members[0]
    cloud.tickets[m.ticket].update(sampler="exited", exit=1)
    pool.tick()
    assert cloud.kinds("restart") and m.restarts == 1
    _run(pool, clock, clock.t + 900)
    assert m.state == READY and m.restarts == 0
    cloud.tickets[m.ticket].update(sampler="exited", exit=137)
    pool.tick()
    assert len(cloud.kinds("restart")) == 2 and "sampler_crash_loop" not in _names(pool) and m.state != GONE


def test_the_restart_count_resets_after_a_healthy_stretch_but_not_in_a_loop():
    clock = Clock()
    cloud = Cloud(clock, ready_after=300)
    pool = _pool(clock, cloud, accounts=(1, 4), max_sampler_restarts=1, sampler_restart_reset_s=3600)
    m = _ready_member(pool, clock, cloud)
    cloud.tickets[m.ticket].update(sampler="exited", exit=137)
    pool.tick()
    assert m.restarts == 1
    _run(pool, clock, clock.t + 3600 + 900)  # rejoined, then healthy past the reset window
    assert m.state == READY
    cloud.tickets[m.ticket].update(sampler="exited", exit=137)
    pool.tick()
    assert len(cloud.kinds("restart")) == 2 and m.state != GONE and m.restarts == 1
    _run(pool, clock, clock.t + 900)  # rejoined, crashes again inside the window: a loop
    cloud.tickets[m.ticket].update(sampler="exited", exit=137)
    pool.tick()
    assert "sampler_crash_loop" in _names(pool) and m.state == GONE


def test_sampler_restart_reset_s_is_validated():
    with pytest.raises(ValueError, match="sampler_restart_reset_s"):
        el.ElasticConfig.from_dict({"sampler_restart_reset_s": -1})
    assert el.ElasticConfig.from_dict({"sampler_restart_reset_s": None}).sampler_restart_reset_s is None


def test_sampler_exit_zero_ends_the_pool():
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, accounts=(1,))
    m = _ready_member(pool, clock, cloud)
    cloud.tickets[m.ticket].update(sampler="exited", exit=0)
    pool.tick()
    assert pool.stopped and m.state == GONE and len(cloud.kinds("order")) == 1


def test_data_ended_is_never_restarted_or_replaced_even_without_stop_on_finish():
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, target=2, accounts=(1, 4), stop_on_finish=False)
    _run(pool, clock, 1800)
    first, second = [m for m in pool.members if m.state == READY]
    cloud.tickets[first.ticket].update(sampler="exited", exit=0)  # "data ended"
    pool.tick()
    assert first.state == GONE and not cloud.kinds("restart") and pool.finished
    _run(pool, clock, clock.t + 6 * H)  # the other member reaches its lease end
    assert len(cloud.kinds("order")) == 2  # no replacement node, no refill, no lease renewal
    assert second.state == GONE and pool.stopped


def test_start_timeout_retires_the_node():
    clock = Clock()
    cloud = Cloud(clock, ready_after=10 * H)
    pool = _pool(clock, cloud, accounts=(1, 4), start_timeout_s=1800)
    _run(pool, clock, 1900)
    assert "start_timeout" in _names(pool)
    assert cloud.tickets["t1"]["reason"] == "released" and len(cloud.kinds("order")) == 2


# -- release safety -----------------------------------------------------------

def test_failed_drain_is_retried_until_the_ticket_is_gone():
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, accounts=(1,))
    m = _ready_member(pool, clock, cloud)
    cloud.drain_error = 2
    cloud.tickets[m.ticket].update(state="ended", reason="preempted")
    pool.tick()
    assert m.state == DRAINING
    pool.tick()
    assert m.state == DRAINING
    pool.tick()
    assert m.state == GONE and _names(pool).count("drain_failed") == 2


def test_patrol_adopts_this_runs_unowned_tickets_and_never_deletes_them():
    """H1: an open ticket of this run no member owns is adopted, not deleted."""
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, target=1, accounts=(1, 4, 5))
    _ready_member(pool, clock, cloud)
    pool.config.target = 2  # room for it (an adoptee past the target is drained: test below)
    # an open ticket of this run no member owns (an order not saved), one of another run
    cloud.tickets["tx"] = dict(cloud.tickets["t1"], account=4, state="active", sampler="absent", starts=0,
                               joined_at=None)
    cloud.ledger.append(["run-a", 4, "tx", True])
    cloud.tickets["ty"] = dict(cloud.tickets["t1"], account=5, state="active")
    cloud.ledger.append(["run-b", 5, "ty", True])
    pool.tick()
    assert not any(e[0] == "drain" for e in cloud.log)
    adopted = [m for m in pool.members if m.ticket == "tx"]
    assert adopted and adopted[0].shard == "nc-1" and "adopted" in _names(pool)
    assert "orphan_lease" not in _names(pool)
    _run(pool, clock, clock.t + 1800)
    assert adopted[0].state == READY and not any(e[0] == "drain" for e in cloud.log)
    assert all(m.ticket != "ty" for m in pool.members)


def test_patrol_releases_unowned_tickets_only_when_stopping_or_asked():
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, accounts=(1, 4))
    pool.config.release_orphans = True
    _ready_member(pool, clock, cloud)
    cloud.tickets["tx"] = dict(cloud.tickets["t1"], account=4, state="active")
    cloud.ledger.append(["run-a", 4, "tx", True])
    pool.tick()
    assert ("drain", "tx", clock.t) in cloud.log and "orphan_lease" in _names(pool)
    # a stopping pool (pool-stop) releases them too
    clock2 = Clock()
    cloud2 = Cloud(clock2)
    pool2 = _pool(clock2, cloud2, accounts=(1, 4))
    _ready_member(pool2, clock2, cloud2)
    cloud2.tickets["tx"] = dict(cloud2.tickets["t1"], account=4, state="active")
    cloud2.ledger.append(["run-a", 4, "tx", True])
    pool2.stop("pool-stop")
    pool2._patrol(clock2.t)  # noqa: SLF001 - what _drain_all runs after stop
    assert any(e[0] == "drain" and e[1] == "tx" for e in cloud2.log)


def test_crash_between_order_and_save_then_restart_adopts_the_node(tmp_path):
    """H1 (a): the process dies right after an order; the restarted pool adopts the
    ticket (the state saved at the order, else the patrol) and never deletes it."""
    clock = Clock()
    cloud = Cloud(clock)
    path = tmp_path / "pool.json"
    pool = IslandPool.restore(path, PoolConfig(target=1, accounts=[1, 4], shard_base="nc"), cloud,
                              clock=clock, log=lambda m: None,
                              events=el.EventLog(log=lambda m: None, clock=clock, run="run-a"))
    pool.tick()  # orders t1; the state is on disk before the tick ends
    assert json.loads(path.read_text())["members"][0]["ticket"] == "t1"
    # (b) the state file moved away: a fresh pool adopts it too
    path.rename(tmp_path / "moved.json")
    fresh = IslandPool.restore(path, pool.config, cloud, clock=clock, log=lambda m: None,
                               events=el.EventLog(log=lambda m: None, clock=clock, run="run-a"))
    _run(fresh, clock, 1800)
    assert not any(e[0] == "drain" for e in cloud.log)
    assert [m.ticket for m in fresh.live()] == ["t1"] and fresh.live()[0].state == READY
    assert len(cloud.kinds("order")) == 1  # account 1 busy with t1: no second order on it


def test_target_two_replaces_only_the_lost_node_and_leaves_the_healthy_one():
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, target=2, accounts=(1, 4, 5), data_slots=2)
    _run(pool, clock, 1800)
    healthy, victim = pool.members
    assert healthy.state == victim.state == READY and healthy.shard != victim.shard
    before = list(cloud.log)
    cloud.tickets[victim.ticket].update(reachable=False)
    _run(pool, clock, clock.t + 3600)
    after = cloud.log[len(before):]
    assert not any(e[1] == healthy.ticket for e in after if e[0] in ("drain", "restart", "fence", "start"))
    assert healthy.state == READY and healthy.restarts == 0
    new = [m for m in pool.live() if m.ticket not in (healthy.ticket, victim.ticket)]
    assert len(new) == 1 and new[0].replaces_shard == victim.shard and new[0].data_slot == victim.data_slot
    assert new[0].shard not in (healthy.shard, victim.shard)  # its own shard (own incarnation lease)
    assert len(cloud.kinds("order")) == 3


def test_pool_state_roundtrip_keeps_elastic_state(tmp_path):
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, accounts=(1, 4))
    _ready_member(pool, clock, cloud)
    pool.pending.append({"data_slot": None, "shard": "nc-9", "lost": True, "from": "tz"})
    pool.breaker.failures = 2
    path = tmp_path / "pool.json"
    pool.save(path)
    again = IslandPool.restore(path, pool.config, cloud, clock=clock, log=lambda m: None)
    assert again.snapshot() == pool.snapshot()


def test_event_log_writes_jsonl_and_survives_a_failing_hook(tmp_path):
    def bad(doc):
        raise RuntimeError("pager down")

    lines = []
    log = el.EventLog(path=tmp_path / "e.jsonl", run="r", log=lines.append, hook=bad, clock=lambda: 5.0)
    log("node_lost", ticket="t1")
    doc = json.loads((tmp_path / "e.jsonl").read_text())
    assert doc == {"t": 5.0, "event": "nc_elastic_node_lost", "run": "r", "ticket": "t1"}
    assert any("hook failed" in x for x in lines)
    seen = []
    el.command_hook("cat", run=lambda cmd, **kw: seen.append((cmd, kw["input"])))({"event": "x"})
    assert seen == [("cat", '{"event": "x"}')]


# -- client + cli wiring ------------------------------------------------------

class FakeNc:
    def __init__(self, ticket=None):
        self.ticket = ticket
        self.calls = []

    def __call__(self, argv, **kw):
        self.calls.append(list(argv))
        if argv[0] == "nc-query":
            doc = {"ticket": dict(self.ticket or {"state": "none"}, others=[]), "balance": {"balance_ucredits": 10**9}}
        elif argv[1] == "create":
            self.ticket = {"id": "abc123", "state": "queued"}
            doc = {"ok": True, "created": True, "ticket": dict(self.ticket)}
        else:
            raise AssertionError(argv)
        return SimpleNamespace(returncode=0, stdout=json.dumps(doc), stderr="")


def _client(fake, account=1):
    return ncc.NcClient(account, ncc.Ledger(tunnel.state_dir() / "tickets.json"), run=fake, sleep=lambda s: None)


def test_ticket_status_maps_the_order_tool():
    node = {"public_ip": "192.0.2.10", "ssh_user": "ubuntu", "host_key": "k"}
    fake = FakeNc({"id": "abc", "state": "active", "node": node, "lease_until": "2026-11-10T09:35:03Z",
                   "preempt_at": None, "gpus": 8, "rate_ucr_per_gpu_hour": 3_000_000})
    s = _client(fake).ticket_status("abc")
    assert s.state == "active" and s.lease_until == ncc.parse_time("2026-11-10T09:35:03Z") and s.gpus == 8
    fake.ticket = dict(fake.ticket, preempt_at="2026-11-10T08:00:00Z")
    assert _client(fake).ticket_status("abc").preempt_at == ncc.parse_time("2026-11-10T08:00:00Z")
    fake.ticket = {"id": "abc", "state": "ended", "end_reason": "preempted"}
    assert _client(fake).ticket_status("abc").end_reason == "preempted"
    fake.ticket = {"id": "abc", "state": "releasing"}
    assert _client(fake).ticket_status("abc").state == "ended"
    fake.ticket = {"id": "abc", "state": "queued"}
    assert _client(fake).ticket_status("abc").state == "waiting"
    fake.ticket = {"id": "other", "state": "active"}  # another order on the account: ours is over
    assert _client(fake).ticket_status("abc").state == "ended"


def test_create_tags_the_run_and_run_rows_filter_by_it():
    fake = FakeNc()
    c = _client(fake)
    c.create(run="run-a")
    c.ledger.add(1, "zzz", run="run-b")
    c.ledger.add(1, "yyy")
    assert [r["ticket"] for r in c.ledger.run_rows("run-a")] == ["abc123"]
    assert all("--all" not in call for call in fake.calls)


def test_pool_config_guards(tmp_path):
    base = {"target": 1, "accounts": [7], "shard_base": "nc-x", "run": "run-x", "spec": "s.json",
            "run_prefix": "run-x", "r2_env_file": "store.env"}
    pc = cli.pool_config(dict(base, elastic={"unreachable_grace_s": 120}), allowed={7})
    assert pc.elastic.unreachable_grace_s == 120 and pc.accounts == [7]
    with pytest.raises(ValueError, match="YETO_NC_ACCOUNTS"):
        cli.pool_config(base, allowed={1})
    with pytest.raises(ValueError, match="unknown keys"):
        cli.pool_config(dict(base, accounts_all=True), allowed={7})
    with pytest.raises(ValueError, match="repeats"):
        cli.pool_config(dict(base, accounts=[7, 7]), allowed={7})
    with pytest.raises(ValueError, match="non-empty word"):
        cli.pool_config(dict(base, run="a b"), allowed={7})
    assert isinstance(cli.pool_ops(dict(base)), cli.NcSshOps)
    with pytest.raises(ValueError, match="route"):
        cli.pool_ops(dict(base, route="other"))


def test_pool_config_refuses_more_samplers_than_accounts_and_bad_sections():
    base = {"target": 1, "accounts": [1, 2], "shard_base": "nc", "spec": "s.json", "run_prefix": "r",
            "r2_env_file": "store.env"}
    cli.pool_config(dict(base, target=2), allowed={1, 2})
    with pytest.raises(ValueError, match="one node at a time"):
        cli.pool_config(dict(base, target=16), allowed={1, 2})
    with pytest.raises(ValueError, match="target must be"):
        cli.pool_config(dict(base, target=0), allowed={1, 2})
    with pytest.raises(ValueError, match="missing required keys .*'spec'"):
        cli.pool_config({k: v for k, v in base.items() if k != "spec"}, allowed={1, 2})
    with pytest.raises(ValueError, match="data_slots"):
        cli.pool_config(dict(base, target=2, data_slots=1), allowed={1, 2})
    with pytest.raises(ValueError, match="sampler config: unknown"):
        cli.pool_config(dict(base, sampler={"dispence": {}}), allowed={1, 2})
    with pytest.raises(ValueError, match="model_pull"):
        cli.pool_config(dict(base, model_pull={"src": "models/x"}), allowed={1, 2})
    with pytest.raises(ValueError, match=r"node_hooks\[0\]"):
        cli.pool_config(dict(base, node_hooks=[{"cmd": "true"}]), allowed={1, 2})
    with pytest.raises(ValueError, match="backlog"):
        cli.pool_config(dict(base, backlog={"rounds": 2}), allowed={1, 2})


def test_machine_type_replaces_the_model_key_with_a_deprecation_note():
    base = {"target": 1, "accounts": [1], "shard_base": "nc", "spec": "s.json", "run_prefix": "r",
            "r2_env_file": "store.env"}
    assert cli.machine_type(base) == "mi355x" and not cli.pool_config_warnings(base)
    assert cli.machine_type(dict(base, machine_type="mi300x")) == "mi300x"
    legacy = dict(base, model="mi300x")
    cli.pool_config(legacy, allowed={1})
    assert cli.machine_type(legacy) == "mi300x"
    assert "machine_type" in cli.pool_config_warnings(legacy)[0]
    with pytest.raises(ValueError, match="machine_type"):
        cli.pool_config(dict(base, model="a", machine_type="b"), allowed={1})
    with pytest.raises(ValueError, match="Hugging Face model name"):
        cli.pool_config(dict(base, model="<org>/<model>"), allowed={1})


def _ssh_ops(tmp_path, calls, *, result=None, raise_timeout=False, provider=None):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"event_tape": "/work/sampler/events-__RUN_PREFIX__.jsonl",
                                "prompts": {"groups_per_round": 4}}))

    def run(argv, **kw):
        calls.append(argv)
        if raise_timeout:
            raise subprocess.TimeoutExpired(argv, kw.get("timeout"))
        return result or SimpleNamespace(returncode=0, stdout="", stderr="")

    cfg = {"target": 1, "accounts": [7], "shard_base": "nc-x", "run": "run-x", "spec": str(spec),
           "run_prefix": "my-run", "r2_env_file": str(tmp_path / "r2.env")}
    return cli.NcSshOps(cfg, run=run, provider=provider)


def test_ssh_ops_probe_is_bounded_and_read_only(tmp_path):
    calls = []
    ops = _ssh_ops(tmp_path, calls, result=SimpleNamespace(
        returncode=0, stdout="YETO_NC_PROBE status=running exit=0 joined=1 age=3", stderr=""))
    m = SimpleNamespace(ticket="abc", shard="nc-x-0")
    p = ops.probe(m)
    assert p.joined and p.sampler == "running"
    argv = calls[0]
    assert "ConnectTimeout=20" in argv and argv[argv.index("ConnectTimeout=20") + 1] == "yeto-nc-abc"
    assert "/mnt/nvme/work/sampler/events-my-run*.jsonl" in argv[-1]
    assert _ssh_ops(tmp_path, [], raise_timeout=True).probe(m).node == el.PROBE_UNREACHABLE


def test_ssh_ops_drain_deletes_through_the_guarded_client(tmp_path, monkeypatch):
    calls, deleted = [], []

    class P:
        def release(self, account, ticket):
            deleted.append(ticket)
            return {"state": "ended"}

    ops = _ssh_ops(tmp_path, calls, raise_timeout=True, provider=P())  # the node does not answer
    tunnel.state_dir().mkdir(parents=True, exist_ok=True)
    cli.leases_path().write_text(json.dumps({"abc": {}}))
    ops.drain(SimpleNamespace(ticket="abc", shard="nc-x-0", account=7))
    assert deleted == ["abc"] and "docker stop" in calls[0][-1]


def test_ssh_ops_spec_continues_a_lost_shard_without_waiting(tmp_path):
    ops = _ssh_ops(tmp_path, [])
    ops.cfg["elastic"] = {"lost_handoff_wait_s": 30}
    m = SimpleNamespace(shard="nc-x-3", data_slot=None, replaces_shard="nc-x-1", predecessor_lost=True)
    spec = ops._spec(m, ["nc-x-1", "nc-x-3"])
    assert spec["continues_shard"] == "nc-x-1" and spec["handoff_wait_s"] == 30.0
    assert spec["sampler_id"] == "nc-x-3"
    m.predecessor_lost = False
    assert "handoff_wait_s" not in ops._spec(m, ["nc-x-1", "nc-x-3"])


def test_ssh_ops_run_tickets_are_this_runs_accounts_only(tmp_path):
    ops = _ssh_ops(tmp_path, [])
    lg = cli.ledger()
    lg.add(7, "a1", run="run-x")
    lg.add(8, "a2", run="run-x")  # an account outside the pool config
    lg.add(7, "a3", run="run-y")
    assert ops.run_tickets() == [(7, "a1")]


def test_node_hooks_render_and_fail_loudly(tmp_path):
    calls = []
    ops = _ssh_ops(tmp_path, calls)
    ops.cfg["node_hooks"] = [{"name": "tunnel", "where": "local", "command": "echo {alias} {ip} {shard}"}]
    lease = ncc.NodeLease(7, "abc", "192.0.2.10", "ubuntu", "", None, None, 8, None)
    ops._hooks(lease, SimpleNamespace(shard="nc-x-0"))
    assert calls[0] == "echo yeto-nc-abc 192.0.2.10 nc-x-0"
    bad = _ssh_ops(tmp_path, [], result=SimpleNamespace(returncode=3, stdout="", stderr="no route"))
    bad.cfg["node_hooks"] = [{"name": "tunnel", "where": "local", "command": "false"}]
    with pytest.raises(ncc.NcError, match="tunnel"):
        bad._hooks(lease, SimpleNamespace(shard="nc-x-0"))


def test_background_start_keeps_the_other_members_watched():
    import threading

    clock = Clock()
    cloud = Cloud(clock, ready_after=0)
    gate = threading.Event()
    sync_start = cloud.start

    def slow_start(m, shards):
        if m.ticket == "t2":
            gate.wait(5)
            raise RuntimeError("model pull failed")
        sync_start(m, shards)

    cloud.start = slow_start
    cloud.start_in_background = True
    pool = _pool(clock, cloud, target=2, accounts=(1, 4))
    pool.start_retries = 0  # this test is about the watch, not the transient-start retry
    _run(pool, clock, 600)
    t1, t2 = pool.members[0], pool.members[1]
    assert t1.state == READY and t2.state == "starting"  # t2 still in its start
    cloud.tickets["t1"].update(state="ended", reason="preempted")
    pool.tick()
    assert t1.state == GONE  # watched while t2 was starting
    gate.set()
    assert isinstance(pool._starts["t2"].exception(timeout=5), RuntimeError)
    clock.t += 60
    pool.tick()
    assert t2.state == GONE and "start_failed" in _names(pool)


def test_a_background_start_is_dropped_when_its_member_is_gone():
    import threading

    clock = Clock()
    cloud = Cloud(clock, ready_after=0)
    gate = threading.Event()
    sync_start = cloud.start

    def slow_start(m, shards):
        if m.ticket == "t2":
            gate.wait(5)
        sync_start(m, shards)

    cloud.start = slow_start
    cloud.start_in_background = True
    pool = _pool(clock, cloud, target=2, accounts=(1, 4))
    _run(pool, clock, 600)
    t2 = pool.members[1]
    assert t2.state == "starting" and "t2" in pool._starts
    cloud.tickets["t2"].update(state="ended", reason="preempted")  # lost mid-start
    pool.tick()
    assert t2.state == GONE
    gate.set()
    pool._starts.get("t2", pool._executor.submit(lambda: None)).result(timeout=5)
    for _ in range(3):
        clock.t += 60
        pool.tick()
    assert "t2" not in pool._starts  # no future kept for a member that is gone


def test_restore_keeps_a_cut_off_start_whose_sampler_runs(tmp_path):
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, accounts=(1,))
    _run(pool, clock, 300)
    m = pool.members[0]
    assert m.state == "starting" and not m.joined
    path = tmp_path / "pool.json"
    pool.save(path)
    again = IslandPool.restore(path, pool.config, cloud, clock=clock, log=lambda x: None)
    again.tick()
    assert again.members[0].state == "starting" and cloud.tickets[m.ticket]["starts"] == 1
    assert len(cloud.kinds("order")) == 1


def test_restore_reruns_a_start_the_restart_cut_off_before_the_sampler(tmp_path):
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, accounts=(1,))
    _run(pool, clock, 300)
    m = pool.members[0]
    cloud.tickets[m.ticket]["sampler"] = "absent"  # the start died before the sampler
    path = tmp_path / "pool.json"
    pool.save(path)
    again = IslandPool.restore(path, pool.config, cloud, clock=clock, log=lambda x: None)
    again.tick()
    assert cloud.tickets[m.ticket]["starts"] == 2 and len(cloud.kinds("order")) == 1


def test_a_start_rerun_after_a_pool_restart_gets_the_whole_start_window(tmp_path):
    """The pool loop died mid-start (killed monitor, host reboot) and came back 50 min
    after the order with no sampler on the node: the start runs again from scratch. Its
    start timeout counts from that start (as an in-place restart's does), not from the
    order: the rerun is not timed out after a few minutes and the node is not retired."""
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, accounts=(1,))
    _run(pool, clock, 300)
    m = pool.members[0]
    assert m.state == "starting" and not m.joined
    cloud.tickets[m.ticket]["sampler"] = "absent"  # the start died with the pool loop
    path = tmp_path / "pool.json"
    pool.save(path)
    clock.t = 3000.0  # the loop is back 50 min after the order
    again = IslandPool.restore(path, pool.config, cloud, clock=clock, log=lambda x: None,
                               events=el.EventLog(log=lambda x: None, clock=clock, run="run-a"))
    _run(again, clock, 3000.0 + 1500)  # the rerun start needs ready_after (1200 s)
    assert cloud.tickets[m.ticket]["starts"] == 2
    assert "start_timeout" not in _names(again), _names(again)
    assert again.members[0].state == READY and len(cloud.kinds("order")) == 1


def test_a_handed_off_predecessor_exiting_zero_does_not_end_the_pool():
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, accounts=(1, 4))
    m = _ready_member(pool, clock, cloud)
    clock.t = cloud.tickets[m.ticket]["until"] - H + 10
    pool.tick()  # replacement ordered
    clock.t += 120
    pool.tick()  # replacement granted and started: it asks m's sampler to stop
    assert m.handed_off
    cloud.tickets[m.ticket].update(sampler="exited", exit=0)  # the handoff LEAVE
    _run(pool, clock, clock.t + 1800)
    assert not pool.stopped and m.state == GONE
    succ = pool.members[1]
    assert succ.state == READY and succ.replaces_shard == m.shard
    assert "sampler_finished" not in _names(pool)


def test_relay_is_tracked_by_pid_file_not_process_name():
    cmd = tunnel.relay_start_command(["192.0.2.7:22"])
    assert "pgrep" not in cmd and "pkill" not in cmd
    assert "relay.pid" in cmd and "kill -0" in cmd and "/proc/$p/cmdline" in cmd


def test_nc_tools_are_configurable_and_never_assume_a_home_install(monkeypatch):
    monkeypatch.delenv("NC_TOOL_DIR", raising=False)
    ncc.configure_tools(None)
    try:
        with pytest.raises(tunnel.TunnelError, match="NC_TOOL_DIR"):
            tunnel.tool_dir()
        t = ncc.configure_tools({"manage": "/opt/nc/bin/manage --quiet", "query": ["/opt/nc/bin/query"],
                                 "tool_dir": "/opt/nc/tool", "node_bin": "/opt/node/bin/node"})
        assert t.manage == ("/opt/nc/bin/manage", "--quiet") and tunnel.tool_dir() == "/opt/nc/tool"
        seen = []
        _client(lambda argv, **kw: seen.append(argv) or SimpleNamespace(
            returncode=0, stdout='{"ok": true, "ticket": null}', stderr="")).status()
        assert seen[-1][:2] == ["/opt/nc/bin/query", "status"]
        with pytest.raises(ValueError, match="unknown keys"):
            ncc.configure_tools({"path": "x"})
    finally:
        ncc.configure_tools(None)


class OwnProvider:
    """A customer's own supply (e.g. a fixed SSH fleet) behind the provider contract."""

    def __init__(self, cfg, run):
        self.run, self.released = run, []

    def order(self, account):
        return "fleet-1"

    def status(self, account, ticket):
        return el.LeaseStatus("active", lease_until=10**12)

    def release(self, account, ticket):
        self.released.append(ticket)
        return {"state": "ended"}

    def node(self, account, ticket):
        return ncc.NodeLease(account, ticket, "192.0.2.20", "ubuntu", "", None, None, 8, None)

    def balance(self):
        return None

    def account_free(self, account):
        return True

    def run_tickets(self):
        return []


def test_provider_factory_plugs_in_another_supply(tmp_path):
    cfg = {"target": 1, "accounts": [1], "shard_base": "fl", "run": "run-f",
           "provider": {"factory": "test_nc_elastic:OwnProvider"}}
    ops = cli.pool_ops(cfg)
    assert isinstance(ops.provider, OwnProvider) and ops.provider.run == "run-f"
    assert ops.order(1) == "fleet-1" and ops.poll(SimpleNamespace(account=1, ticket="fleet-1")).state == "active"
    assert cli.load_lease("fleet-1").ip == "192.0.2.20"
    with pytest.raises(ValueError, match="'nc', 'daytona', 'sky' and 'modal' are built in"):
        cli.pool_ops(dict(cfg, provider={"kind": "k8s"}))


class FakeNcDelete(FakeNc):
    """FakeNc that also answers ``delete`` (the ticket reads ended right after)."""

    def __call__(self, argv, **kw):
        if argv[0] != "nc-query" and argv[1] == "delete":
            self.calls.append(list(argv))
            self.ticket = dict(self.ticket, state="ended", end_reason="released")
            return SimpleNamespace(returncode=0, stdout=json.dumps({"ok": True, "released": True}), stderr="")
        return super().__call__(argv, **kw)


def test_orders_and_releases_reach_the_runs_cost_ledger(tmp_path, monkeypatch):
    path = tmp_path / "run" / "resources.jsonl"
    monkeypatch.setenv(ncc.COST_LEDGER_ENV, str(path))
    fake = FakeNcDelete()
    c = _client(fake, account=3)
    ticket = c.create(run="run-a")
    c.delete(ticket)
    recs = [json.loads(x) for x in path.read_text().splitlines()]
    assert [(r["op"], r["kind"], r["id"]) for r in recs] == [("create", "nc_lease", ticket),
                                                            ("release", "nc_lease", ticket)]
    assert recs[0]["attrs"]["account"] == 3 and recs[0]["name"] == "run-a"
    assert path.stat().st_mode & 0o077 == 0


def test_no_cost_ledger_and_a_broken_one_never_break_an_order(tmp_path, monkeypatch):
    monkeypatch.delenv(ncc.COST_LEDGER_ENV, raising=False)
    assert _client(FakeNc()).create(run="run-a") == "abc123"
    (tmp_path / "blocker").write_text("")
    monkeypatch.setenv(ncc.COST_LEDGER_ENV, str(tmp_path / "blocker" / "x.jsonl"))  # parent is a file
    assert _client(FakeNc(), account=2).create(run="run-a") == "abc123"


def test_no_order_past_the_shard_limit(tmp_path):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"agentic": {"setup": {"tunnel": {"listen": "10.0.0.1:18081-18088"}}}}))
    assert cli.pool_shard_limit({"spec": str(spec)}) == 8
    assert cli.pool_shard_limit({"spec": str(spec), "shard_limit": 4}) == 4
    assert cli.pool_shard_limit({"spec": str(tmp_path / "missing.json")}) is None

    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud)
    pool.config.shard_limit = 1
    pool.next_shard = 1  # nc-0 is taken: the next order would make nc-1
    pool.tick()
    assert not pool.members and "order_refused" in _names(pool)


def _unowned(cloud, ticket, account):
    cloud.tickets[ticket] = dict(cloud.tickets["t1"], account=account, state="active", sampler="absent",
                                 starts=0, joined_at=None)
    cloud.ledger.append(["run-a", account, ticket, True])


@pytest.mark.parametrize("setup, why", [
    (dict(), "for target 1"),                                         # the target is already met
    (dict(target=2, shard_limit=1), "past shard_limit"),              # no buffer shard / port left
    (dict(target=2, data_slots=2), "data slots is held"),             # it would read every prompt
])
def test_an_adoptee_the_pool_has_no_room_for_is_drained_not_started(setup, why):
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, target=1, accounts=(1, 4, 5))
    if "data_slots" in setup:
        pool.config.data_slots = setup["data_slots"]
    _ready_member(pool, clock, cloud)
    if "data_slots" in setup:  # slot 0 held by the member, slot 1 by a handoff the next order continues
        pool.members[0].data_slot = "0/2"
        pool.pending.append({"data_slot": "1/2", "shard": None, "lost": True, "from": "t0"})
    pool.config.target = setup.get("target", 1)
    pool.config.shard_limit = setup.get("shard_limit")
    orders = len(cloud.kinds("order"))
    _unowned(cloud, "tx", 4)
    pool.tick()
    tx = [m for m in pool.members if m.ticket == "tx"]
    assert tx and tx[0].state == GONE and why in (tx[0].end_reason or "")
    assert ("drain", "tx", clock.t) in cloud.log and "adopt_refused" in _names(pool)
    assert "adopted" not in _names(pool) and tx[0].data_slot is None
    assert not any(e[0] == "start" and e[1] == "tx" for e in cloud.log)
    pool.tick()  # never adopted again once released
    assert len([m for m in pool.members if m.ticket == "tx"]) == 1
    if "target" not in setup:
        assert len(cloud.kinds("order")) == orders


def test_the_metrics_port_comes_from_the_sampler_spec(tmp_path):
    assert el.resolve_metrics_port("spec", {"engine": {"server": {"port": 31234}}}) == 31234
    assert el.resolve_metrics_port("spec", {"engine": {}}) is None
    assert el.resolve_metrics_port(8000, {"engine": {"server": {"port": 31234}}}) == 8000
    assert el.resolve_metrics_port(None, {"engine": {"server": {"port": 31234}}}) is None
    with pytest.raises(ValueError, match="metrics_port"):
        el.ElasticConfig.from_dict({"metrics_port": "auto"})
    calls = []
    ops = _ssh_ops(tmp_path, calls)
    spec = json.loads((tmp_path / "spec.json").read_text())
    spec["engine"] = {"server": {"port": 31234}}
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    ops.probe(SimpleNamespace(ticket="abc", shard="nc-x-0"))
    assert "http://127.0.0.1:31234/metrics" in calls[0][-1] and ops.reads_metrics
    calls = []
    ops = _ssh_ops(tmp_path, calls)
    ops.cfg["elastic"] = {"metrics_port": None}
    ops.probe(SimpleNamespace(ticket="abc", shard="nc-x-0"))
    assert "/metrics" not in calls[0][-1] and not ops.reads_metrics


def test_three_probes_without_engine_metrics_send_one_event():
    clock = Clock()
    cloud = Cloud(clock)
    cloud.reads_metrics = True
    pool = _pool(clock, cloud, target=1)
    m = _ready_member(pool, clock, cloud)
    misses = lambda: [e for e in pool.events.events if e["event"] == "nc_elastic_engine_metrics_unreadable"]  # noqa: E731
    for _ in range(5):
        pool.tick()
    assert len(misses()) == 1 and misses()[0]["ticket"] == m.ticket and m.metrics_misses >= 3
    real = cloud.probe
    cloud.probe = lambda mem: el.Probe(**{**real(mem).__dict__, "gen_tokens": 5.0, "running": 0, "queued": 0})
    pool.tick()
    assert m.metrics_misses == 0 and "engine_metrics_recovered" in _names(pool)


def test_member_ready_carries_the_providers_machine_hour_in_usd():
    """The dashboard prices a node from member_ready's usd_per_h: the provider's
    hourly_cost converted from its currency (NC: ucr)."""
    from yeto.launch.pool.island_pool import IslandPool, Member

    pool = IslandPool.__new__(IslandPool)
    pool.ops = type("Ops", (), {"hourly_cost": lambda self, m: 24_000_000, "currency": lambda self, m: "ucr"})()
    assert pool._usd_per_h(Member(ticket="t", account=1, shard="s")) == 24.0
    pool.ops = object()
    assert pool._usd_per_h(Member(ticket="t", account=1, shard="s")) is None


# -- a transient start failure is retried in place (report: one ssh exit 255 cost a node) ----

def test_one_ssh_exit_255_in_the_start_is_retried_on_the_same_node():
    clock = Clock()
    cloud = Cloud(clock)
    sync_start = cloud.start
    fails = {"n": 1}

    def flaky_start(m, shards):
        if fails["n"]:
            fails["n"] -= 1
            raise subprocess.CalledProcessError(255, ["ssh", "yeto-nc-x"], stderr="read timeout")
        sync_start(m, shards)

    cloud.start = flaky_start
    pool = _pool(clock, cloud, accounts=(1,))
    _run(pool, clock, 3600)
    names = _names(pool)
    assert "start_retry" in names and "start_failed" not in names and "released" not in names
    assert [m.state for m in pool.members] == [READY] and len(cloud.tickets) == 1  # no second order
    retry = next(e for e in pool.events.events if e["event"].endswith("start_retry"))
    assert retry["attempt"] == 1 and retry["retry_in_s"] == pool.start_retry_s


def test_transient_start_failures_are_bounded_and_back_off():
    clock = Clock()
    cloud = Cloud(clock)
    starts = []

    def down(m, shards):
        starts.append(clock())
        raise subprocess.TimeoutExpired(["ssh", "yeto-nc-x"], 60)

    cloud.start = down
    pool = _pool(clock, cloud, accounts=(1,), max_orders=1)
    _run(pool, clock, 3600, step=10.0)
    first = pool.members[0]
    assert first.state == GONE and first.start_retries == pool.start_retries == 3
    assert len([t for t in starts if t <= starts[0] + 600]) == 4  # the first try and 3 retries
    gaps = [b - a for a, b in zip(starts, starts[1:4])]
    assert gaps[0] >= 30 and gaps[1] >= 60 and gaps[2] >= 120  # doubled every retry
    assert _names(pool).count("start_retry") >= 3 and "start_failed" in _names(pool)


def test_a_refused_spec_is_not_retried():
    clock = Clock()
    cloud = Cloud(clock)

    def refuse(m, shards):
        raise ValueError("spec sampler_id 'x' is not the shard")

    cloud.start = refuse
    pool = _pool(clock, cloud, accounts=(1,), max_orders=1)
    _run(pool, clock, 600)
    assert "start_retry" not in _names(pool) and pool.members[0].state == GONE


# -- a stale stop request of an earlier attempt's shard does not stop a new pool's sampler ----

class CloudWithHandoffs(Cloud):
    def clear_stale_handoff(self, m, keep):
        from yeto.launch.pool.data_path import R2DataPath

        return R2DataPath(self.store).clear_stale_handoff(m.shard, keep=keep)


def test_a_new_pool_sets_aside_the_handoff_a_failed_attempt_left_on_its_shard_name():
    clock = Clock()
    cloud = CloudWithHandoffs(clock)
    # attempt 1: the start fails for good; the shard nc-0 is fenced (a handoff record) and released
    cloud.start = lambda m, shards: (_ for _ in ()).throw(ValueError("bad image"))
    first = _pool(clock, cloud, accounts=(1,), max_orders=1)
    _run(first, clock, 600)
    assert first.members[0].shard == "nc-0" and first.members[0].state == GONE
    assert handoff_requested(cloud.store, "nc-0")
    # attempt 2 of the same run: a fresh pool numbers its shards from nc-0 again
    cloud.start = Cloud.start.__get__(cloud)
    second = _pool(clock, cloud, accounts=(1,))
    _run(second, clock, clock.t + 3600)
    m = second.members[0]
    assert m.shard == "nc-0" and m.state == READY
    assert not handoff_requested(cloud.store, "nc-0")  # the old request no longer stops the new sampler
    aside = [k for k in cloud.store.objects.data if k.startswith("sample-buffer/handoffs-aside/nc-0-")]
    assert len(aside) == 1
    record = json.loads(cloud.store.objects.data[aside[0]])
    assert record["successor"] == f"pool-fence:{first.members[0].ticket}" and record["set_aside_reason"]
    assert "stale_handoff_set_aside" in _names(second)


def test_a_handoff_from_this_pools_own_member_or_fence_is_kept():
    from yeto.rl.engine.sampler_resume import request_handoff, set_aside_handoff

    store = SimpleNamespace(objects=Objects())
    request_handoff(store, "nc-0", "nc-1")
    assert set_aside_handoff(store, "nc-0", keep={"nc-1"}) is None and handoff_requested(store, "nc-0")
    request_handoff(store, "nc-2", "pool-fence:t7")
    assert set_aside_handoff(store, "nc-2", keep={"pool-fence:t7"}) is None and handoff_requested(store, "nc-2")
    assert set_aside_handoff(store, "nc-3") is None  # no record: nothing to do
    assert set_aside_handoff(store, "nc-0", keep={"nc-9"})["successor"] == "nc-1"
    assert not handoff_requested(store, "nc-0") and handoff_requested(store, "nc-2")


# -- lease-end handover without a ready replacement ------------------------------

def test_lease_end_without_ready_replacement_alerts_and_keeps_the_started_successor_handoff():
    """The replacement never comes up: the old node still goes at its lease end, but with
    an alert (a capacity gap), and the shard is not fenced under a successor that already
    asked for the handoff."""
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, start_timeout_s=99 * H)
    m = _ready_member(pool, clock, cloud)
    cloud.ready_after = 99 * H
    _run(pool, clock, m.lease_until + 600)
    assert m.state == GONE
    gap = [e for e in pool.events.events if e["event"].endswith("lease_end_without_ready_replacement")]
    assert len(gap) == 1 and gap[0]["replacement"] == m.replaced_by and gap[0]["replacement_state"] == "starting"
    assert not [e for e in cloud.kinds("fence") if e[1] == m.ticket]  # the started successor's handoff stays


def test_lease_end_with_no_replacement_fences_the_shard_and_alerts():
    """No replacement could be ordered (balance below the order floor): the refusal is an
    event, and the node drained at its lease end is fenced first (its claims go back at
    once, the next order continues the shard)."""
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud)
    m = _ready_member(pool, clock, cloud)
    cloud.bal = pool.config.order_floor() - 1
    _run(pool, clock, m.lease_until + 600)
    refused = [e for e in pool.events.events if e["event"].endswith("order_refused")]
    assert len(refused) == 1 and "order floor" in refused[0]["why"] and refused[0]["replaces"] == m.ticket
    gap = [e for e in pool.events.events if e["event"].endswith("lease_end_without_ready_replacement")]
    assert len(gap) == 1 and gap[0]["replacement"] is None
    fences = [e for e in cloud.kinds("fence") if e[1] == m.ticket]
    drains = [e for e in cloud.kinds("drain") if e[1] == m.ticket]
    assert fences and drains and fences[0][3] <= drains[0][2]
    assert pool.pending and pool.pending[0]["shard"] == m.shard and pool.pending[0]["from"] == m.ticket
    cloud.bal = 10**10  # money back: the next order continues the fenced shard
    _run(pool, clock, clock.t + 600)
    new = [x for x in pool.members if x.ticket != m.ticket]
    assert new and new[0].replaces_shard == m.shard


def test_lease_end_fences_when_the_started_successor_failed_meanwhile():
    """The successor started but then failed (its own fence owed, it is ``fencing``): it
    answers no handoff, so the old node at its lease end is fenced before it goes."""
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, start_timeout_s=99 * H)
    m = _ready_member(pool, clock, cloud)
    cloud.ready_after = 99 * H
    while not (m.replaced_by and any(x.ticket == m.replaced_by and x.started for x in pool.members)):
        pool.tick()
        clock.t += 60
    s = next(x for x in pool.members if x.ticket == m.replaced_by)
    fence = cloud.fence

    def failing(x):
        if x.ticket == s.ticket:
            raise RuntimeError("store unreachable")
        fence(x)

    cloud.fence = failing
    pool._retire(s, "crashed", lost=True)
    assert s.state == FENCING
    _run(pool, clock, m.lease_until + 600)
    assert m.state == GONE
    fences = [e for e in cloud.kinds("fence") if e[1] == m.ticket]
    drains = [e for e in cloud.kinds("drain") if e[1] == m.ticket]
    assert fences and drains and fences[0][3] <= drains[0][2]


def test_a_failing_lease_end_fence_does_not_hold_the_node_past_its_lease():
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud)
    m = _ready_member(pool, clock, cloud)
    cloud.bal = pool.config.order_floor() - 1

    def failing(x):
        raise RuntimeError("store unreachable")

    cloud.fence = failing
    _run(pool, clock, m.lease_until + 120)
    assert m.state == GONE and m.fence_pending == "lease end"
    assert [e for e in cloud.kinds("drain") if e[1] == m.ticket][0][2] <= m.lease_until


def test_a_member_evicted_during_a_start_retry_does_not_count_as_a_failure():
    """A granted member waits out a transient start retry (state provisioning again)
    and the provider evicts its node: an eviction of a running node, not a ticket lost
    before its grant, so the breaker does not count it."""
    clock = Clock()
    cloud = Cloud(clock)

    def down(m, shards):
        raise subprocess.CalledProcessError(255, ["ssh", "yeto-nc-x"], stderr="read timeout")

    cloud.start = down
    pool = _pool(clock, cloud, accounts=(1,))
    _run(pool, clock, 120)
    m = pool.members[0]
    assert m.state == "provisioning" and m.granted is not None and m.start_retries == 1
    cloud.tickets[m.ticket].update(state="ended", reason="preempted")
    pool.tick()
    assert m.state == GONE and "evicted" in _names(pool)
    assert pool.breaker.failures == 0


def test_a_background_start_that_fails_because_its_node_was_evicted_is_not_a_failure():
    """The node is evicted while its background start runs; the start then fails (its
    ssh went away). The member is handled as evicted, not counted as a failed start too."""
    clock = Clock()
    cloud = Cloud(clock)
    import threading  # noqa: PLC0415

    gate = threading.Event()

    def start(m, shards):
        gate.wait(5)
        raise RuntimeError("model pull failed: connection closed")

    cloud.start = start
    cloud.start_in_background = True
    pool = _pool(clock, cloud, accounts=(1,))
    pool.start_retries = 0
    _run(pool, clock, 120)
    m = pool.members[0]
    assert m.state == "starting"
    cloud.tickets[m.ticket].update(state="ended", reason="preempted")
    gate.set()
    pool._starts[m.ticket].exception(timeout=5)  # noqa: SLF001
    pool.tick()
    assert m.state == GONE and "evicted" in _names(pool)
    assert pool.breaker.failures == 0


def _background_start_that_fails(cloud):
    import threading  # noqa: PLC0415

    gate = threading.Event()

    def start(m, shards):
        gate.wait(5)
        raise RuntimeError("model pull failed: connection closed")

    cloud.start = start
    cloud.start_in_background = True
    return gate


def test_a_background_start_that_fails_on_a_live_node_still_counts_as_a_failure():
    """The guard against a bad node: the start fails and its ticket is still active, so
    the breaker counts it (else a bad node is ordered again and again)."""
    clock = Clock()
    cloud = Cloud(clock)
    gate = _background_start_that_fails(cloud)
    pool = _pool(clock, cloud, accounts=(1,))
    pool.start_retries = 0
    _run(pool, clock, 120)
    m = pool.members[0]
    gate.set()
    pool._starts[m.ticket].exception(timeout=5)  # noqa: SLF001
    pool.tick()
    assert m.state != "starting" and pool.breaker.failures == 1


def test_a_background_start_failure_whose_lease_status_cannot_be_read_still_counts():
    """The lease status read fails after a failed start: the pool cannot tell an eviction,
    so the failure counts (an unknown status never hides a bad node)."""
    clock = Clock()
    cloud = Cloud(clock)
    gate = _background_start_that_fails(cloud)
    pool = _pool(clock, cloud, accounts=(1,))
    pool.start_retries = 0
    _run(pool, clock, 120)
    m = pool.members[0]
    real_poll = cloud.poll
    cloud.poll = lambda *a, **k: (_ for _ in ()).throw(TimeoutError("control plane down"))
    gate.set()
    pool._starts[m.ticket].exception(timeout=5)  # noqa: SLF001
    pool.tick()
    cloud.poll = real_poll
    assert m.state != "starting" and pool.breaker.failures == 1

def test_a_start_cut_off_by_every_pool_restart_is_still_timed_out(tmp_path):
    """The pool loop keeps dying mid-start (the start itself takes it down): each rerun
    gets a fresh start window only up to max_sampler_restarts times; past that the
    timeout counts on from the earlier start and the node is retired, not kept forever."""
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, accounts=(1,), start_timeout_s=1800)
    _run(pool, clock, 300)
    ticket = pool.members[0].ticket
    path = tmp_path / "pool.json"
    events = []
    for _ in range(8):
        m = next(m for m in pool.members if m.ticket == ticket)
        if m.state == "gone":
            break
        cloud.tickets[ticket]["sampler"] = "absent"
        pool.save(path)
        pool = IslandPool.restore(path, pool.config, cloud, clock=clock, log=lambda x: None,
                                  events=el.EventLog(log=lambda x: None, clock=clock, run="run-a"))
        _run(pool, clock, clock.t + 900)  # dies again before ready_after (1200 s)
        events += _names(pool)
    assert "start_timeout" in events, events
    assert "start_rerun_window_spent" in events


def test_an_account_whose_orders_keep_failing_does_not_starve_the_healthy_ones():
    """An order that fails on one account (not a guard refusal: something was sent) must
    not leave that account first in line every tick: the next order tries another free
    account, so one broken account cannot trip the breaker while a healthy one idles."""
    clock = Clock()
    cloud = Cloud(clock)
    real_order = cloud.order

    def order(account):
        if account == 1:
            cloud.log.append(("order_failed", account, clock()))
            raise ncc.NcError("500 internal error on this account")
        return real_order(account)

    cloud.order = order
    pool = _pool(clock, cloud, accounts=(1, 4), breaker_failures=2, breaker_max_trips=2)
    _run(pool, clock, 1800)
    assert [m.account for m in pool.live()] == [4], _names(pool)
    assert not pool.breaker.tripped
    assert sum(1 for e in cloud.log if e[0] == "order_failed") == 1
