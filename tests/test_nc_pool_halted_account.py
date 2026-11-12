"""NC pool: an account the ticket ledger halted (a delete of this tool hit, or may have
hit, a foreign ticket) costs the pool that account, not every order: the guard's refusal
is no provisioning failure, the other accounts are ordered on, a member on the halted
account is kept until an operator lifts the halt, and the halt is shown."""

from __future__ import annotations

import json

import pytest

from yeto.launch.nc import cli, pool_run, tunnel
from yeto.launch.nc import client as ncc
from yeto.launch.nc.provider import NCProvider
from yeto.launch.pool.island_pool import DRAINING, GONE, READY

from test_nc_elastic import H, Clock, Cloud, _names, _pool, _run, _ssh_ops, _state  # noqa: F401
from test_nc_pool_run import _cfg

HALT = ("account 1 is halted (a delete meant for ticket X was sent and what it released is not known): "
        "no order, delete or adopt on it until clear-halt")


class GuardedCloud(Cloud):
    """``halted``: accounts halted in the ledger (they read free: they hold no ticket; every
    order and delete on them is refused). ``tells``: the ops can read the halt."""

    def __init__(self, clock, *, halted=(), tells=False, **kw):
        super().__init__(clock, **kw)
        self.halted = set(halted)
        self.drain_errors = {}  # ticket -> exception the release raises
        if tells:
            self.slot_halt = lambda account: {"halt": "a delete released a foreign ticket",
                                              "expected_ticket": "tX"} if account in self.halted else None

    def order(self, account):
        if account in self.halted:
            self.log.append(("refused", account, self.clock()))
            raise ncc.NcHalted(HALT)
        return super().order(account)

    def drain(self, m):
        error = self.drain_errors.get(m.ticket)
        if error is None and m.account in self.halted:
            error = ncc.NcHalted(HALT)
        if error is not None:
            self.log.append(("drain_refused", m.ticket, self.clock()))
            raise error
        super().drain(m)


def _events(pool, name):
    return [e for e in pool.events.events if e["event"] == f"nc_elastic_{name}"]


@pytest.mark.parametrize("tells", [False, True])
def test_a_halted_account_costs_the_pool_one_account_not_every_order(tells):
    clock = Clock()
    cloud = GuardedCloud(clock, halted={1}, tells=tells, ready_after=300)
    pool = _pool(clock, cloud, target=2, accounts=(1, 4, 5))
    _run(pool, clock, 6 * H)
    ready = [m for m in pool.members if m.state == READY]
    assert len(ready) == 2 and {m.account for m in ready} == {4, 5}
    assert not pool.breaker.tripped and not pool.breaker.trips and not pool.breaker.failures
    assert "order_failed" not in _names(pool) and "breaker_open" not in _names(pool)
    (halt,) = _events(pool, "slot_halted")  # one alert, with what to do
    assert halt["account"] == 1 and halt["level"] == "alert" and "clear-halt --account 1 --yes" in halt["action"]
    assert "slots_below_target" not in _names(pool)  # two usable accounts for target 2
    if tells:
        assert not cloud.kinds("refused") and halt["expected_ticket"] == "tX"  # never even tried
    else:
        assert len(cloud.kinds("refused")) <= 6 * H / 600 + 1  # asked again every ten minutes at most


def test_both_orders_of_a_tick_go_to_the_other_accounts():
    clock = Clock()
    cloud = GuardedCloud(clock, halted={1})
    pool = _pool(clock, cloud, target=2, accounts=(1, 4, 5))
    pool.tick()
    assert [(e[1], e[2]) for e in cloud.kinds("order")] == [("t1", 4), ("t2", 5)]
    assert len(cloud.kinds("refused")) == 1


def test_a_guard_refusal_that_is_no_halt_is_not_a_failure_either():
    clock = Clock()
    cloud = GuardedCloud(clock, ready_after=300)
    order = cloud.order

    def refuse_on_1(account):
        if account == 1:
            raise ncc.NcRefused("account 1: a delete for ticket tX is not confirmed yet; no other write")
        return order(account)

    cloud.order = refuse_on_1
    pool = _pool(clock, cloud, target=1, accounts=(1, 4))
    _run(pool, clock, 2 * H)
    assert [m.account for m in pool.members if m.state == READY] == [4]
    assert not pool.breaker.failures and _names(pool).count("order_refused_by_guard") == 1
    assert "slot_halted" not in _names(pool) and "order_failed" not in _names(pool)


def test_fewer_usable_accounts_than_the_target_is_a_critical_alert_once():
    clock = Clock()
    cloud = GuardedCloud(clock, halted={1}, tells=True, ready_after=300)
    pool = _pool(clock, cloud, target=2, accounts=(1, 4))
    _run(pool, clock, 2 * H)
    assert [m.account for m in pool.members if m.state == READY] == [4]
    (short,) = _events(pool, "slots_below_target")
    assert short["level"] == "critical" and (short["usable"], short["target"], short["halted"]) == (1, 2, [1])
    assert "gap" in short["detail"] and not pool.breaker.failures and not pool.stopped
    cloud.halted.clear()  # the operator ran clear-halt
    _run(pool, clock, clock.t + 2 * H)
    assert len([m for m in pool.members if m.state == READY]) == 2
    assert "slot_halt_cleared" in _names(pool) and len(_events(pool, "slots_below_target")) == 1


@pytest.mark.parametrize("tells", [False, True])
def test_a_member_on_a_halted_account_is_kept_and_not_drained_again_and_again(tells):
    clock = Clock()
    cloud = GuardedCloud(clock, tells=tells, ready_after=300)
    pool = _pool(clock, cloud, target=1, accounts=(1, 4))
    _run(pool, clock, 1800)
    (m,) = [x for x in pool.members if x.state == READY]
    assert m.account == 1
    cloud.halted.add(1)
    cloud.tickets[m.ticket].update(sampler="exited", exit=1)
    pool.config.elastic.max_sampler_restarts = 0  # the crash retires it: its release is refused
    _run(pool, clock, clock.t + 2 * H)
    assert m.state == DRAINING and cloud.tickets[m.ticket]["state"] == "active"  # kept, still billed
    refused = len(cloud.kinds("drain_refused"))
    assert refused == 1 if tells else refused <= 2 * H / 600 + 2  # not once per tick (120 ticks)
    (blocked,) = _events(pool, "release_blocked")
    assert blocked["ticket"] == m.ticket and blocked["level"] == "alert" and "clear-halt" in blocked["action"]
    assert "drain_failed" not in _names(pool) and len(_events(pool, "slot_halted")) == 1
    replacement = pool.members[-1]
    assert replacement.account == 4 and replacement.state == READY and replacement.replaces_shard == m.shard
    cloud.halted.clear()  # clear-halt
    _run(pool, clock, clock.t + 900)
    assert m.state == GONE and cloud.tickets[m.ticket]["state"] == "ended"


def test_a_deferred_release_is_asked_again_and_is_not_a_failure():
    clock = Clock()
    cloud = GuardedCloud(clock, ready_after=300)
    pool = _pool(clock, cloud, target=1, accounts=(1, 4))
    _run(pool, clock, 1800)
    (m,) = [x for x in pool.members if x.state == READY]
    cloud.drain_errors[m.ticket] = ncc.NcReleaseDeferred("ticket t1 ends on its own in 240 s: no delete is sent now")
    cloud.tickets[m.ticket].update(sampler="exited", exit=1)
    pool.config.elastic.max_sampler_restarts = 0
    _run(pool, clock, clock.t + 300)
    assert m.state == DRAINING and len(cloud.kinds("drain_refused")) == 5  # every tick
    assert _names(pool).count("release_deferred") == 1 and "drain_failed" not in _names(pool)
    assert "slot_halted" not in _names(pool)
    del cloud.drain_errors[m.ticket]
    pool.tick()
    assert m.state == GONE


@pytest.mark.parametrize("error", [ncc.NcForeignTicketReleased("the delete for t1 released ticket t9 of another owner"),
                                   ncc.NcDeleteUnconfirmed("a delete for t1 was sent; what it released is not known")])
def test_a_release_that_hit_a_foreign_ticket_is_a_critical_event_and_the_account_is_left_alone(error):
    clock = Clock()
    cloud = GuardedCloud(clock, ready_after=300)
    pool = _pool(clock, cloud, target=1, accounts=(1, 4))
    _run(pool, clock, 1800)
    (m,) = [x for x in pool.members if x.state == READY]
    cloud.drain_errors[m.ticket] = error
    cloud.tickets[m.ticket].update(sampler="exited", exit=1)
    pool.config.elastic.max_sampler_restarts = 0
    pool.tick()
    cloud.drain_errors.clear()
    cloud.halted.add(1)  # the client halted the account
    _run(pool, clock, clock.t + H)
    (hit,) = _events(pool, "release_hit_foreign_ticket")
    assert hit["level"] == "critical" and hit["kind"] == type(error).__name__ and hit["ticket"] == m.ticket
    assert len(_events(pool, "slot_halted")) == 1 and m.state == DRAINING
    assert len(cloud.kinds("drain_refused")) <= 1 + H / 600 + 2
    assert pool.members[-1].account == 4 and pool.members[-1].state == READY  # the pool goes on elsewhere


# -- the ledger behind it, and what the status commands show --------------------------

def test_the_nc_ops_read_the_halt_from_the_ticket_ledger(tmp_path):
    book = ncc.Ledger(tunnel.state_dir() / "tickets.json")
    provider = NCProvider([7], "run-x", ledger=book, allowed={7})
    ops = _ssh_ops(tmp_path, [], provider=provider)
    assert ops.slot_halt(7) is None
    book.halt(7, "a delete released a foreign ticket", expected_ticket="tA", released_ticket="tB")
    halt = ops.slot_halt(7)
    assert halt["halt"] == "a delete released a foreign ticket" and halt["expected_ticket"] == "tA"
    assert book.clear_halt(7) and ops.slot_halt(7) is None
    assert _ssh_ops(tmp_path, [], provider=object()).slot_halt(7) is None  # another provider: no ledger


def test_pool_status_and_the_run_status_show_halted_accounts_unconfirmed_deletes_and_lost_orders(tmp_path, capsys):
    from yeto.launch.rollout import entry
    from yeto.launch.rollout.pools import PoolRef

    cfg, config_path = _cfg(tmp_path, accounts=[1, 4, 5])
    assert pool_run.account_attention(cfg) == []
    book = cli.ledger()
    book.halt(1, "what a delete released is not known", expected_ticket="tA")
    book.begin_delete(4, "tD")
    order = book.begin_order(5, before=None)
    rows = book.load()
    for row in rows:  # the order call ended without an outcome; the account then showed a ticket
        if row.get("order") == order:
            row.update(call_ended=row["created_at"], observed_ticket="tU")
    book._write(rows)  # noqa: SLF001
    notes = pool_run.account_attention(cfg)
    assert [(n["kind"], n["account"]) for n in notes] == [("halted", 1), ("delete_unconfirmed", 4),
                                                         ("order_without_outcome", 5)]
    assert notes[0]["expected_ticket"] == "tA" and "clear-halt --account 1 --yes" in notes[0]["action"]
    assert notes[1]["ticket"] == "tD" and notes[2]["observed_ticket"] == "tU"
    assert cli.main(["pool-status", "--config", config_path]) == 0
    out = capsys.readouterr()
    assert [n["kind"] for n in json.loads(out.out)["accounts_attention"]] == [n["kind"] for n in notes]
    assert out.err.count("ATTENTION: HALTED: account=1") == 1 and "ATTENTION: DELETE_UNCONFIRMED: account=4" in out.err
    assert "ATTENTION: ORDER_WITHOUT_OUTCOME: account=5" in out.err
    summary = entry.pool_summary(PoolRef("nc", config_path))
    assert len(summary["accounts_attention"]) == 3 and "halted" in summary["warning"]
    assert entry.status([PoolRef("nc", config_path)]) == 1  # a warning: the status exits 1
    assert '"accounts_attention"' in capsys.readouterr().out
