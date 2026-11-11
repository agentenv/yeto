"""Pool slots are named per provider: ``account`` on National Compute only, ``slot``
on Modal / Daytona / sky (logs, events, status, dashboard); ``slots`` is accepted
in the pool config off National Compute, ``accounts`` stays an alias."""

from __future__ import annotations

import pytest

from yeto.dashboard.async_rl import POOL_KEYS
from yeto.launch.nc import cli
from yeto.launch.pool.status import EventLog
from tests.test_rollout_island_pool import Clock, FakeProvider, make_pool, ticks

BASE = {"target": 1, "shard_base": "x", "run": "run-x", "spec": "s.json", "run_prefix": "run-x",
        "r2_env_file": "store.env"}
MODAL = {"kind": "modal", "image": "img", "secret": "s", "bucket": "b"}


def test_nc_pool_config_keeps_accounts_and_refuses_slots():
    pc = cli.pool_config(dict(BASE, accounts=[7]), allowed={7})
    assert pc.accounts == [7] and pc.slot_name == "account"
    with pytest.raises(ValueError, match="not 'slots'"):
        cli.pool_config(dict(BASE, slots=[7]), allowed={7})
    with pytest.raises(ValueError, match=r"missing required keys \['accounts'\]"):
        cli.pool_config(dict(BASE))


@pytest.mark.parametrize("kind", ["modal", "daytona", "sky"])
def test_other_providers_take_slots_or_the_accounts_alias(kind):
    cfg = dict(BASE, provider=dict(MODAL, kind=kind))
    a = cli.pool_config(dict(cfg, slots=[1, 2]))
    b = cli.pool_config(dict(cfg, accounts=[1, 2]))
    assert a.accounts == b.accounts == [1, 2] and a.slot_name == b.slot_name == "slot"
    assert cli.slot_list(dict(cfg, slots=[3])) == [3]
    with pytest.raises(ValueError, match="not both"):
        cli.pool_config(dict(cfg, slots=[1], accounts=[1]))
    with pytest.raises(ValueError, match="2 slots; a slot holds one machine"):
        cli.pool_config(dict(cfg, slots=[1, 2], target=3))


def _pool(slot_name):
    clock = Clock()
    lines = []
    pool, _, _ = make_pool(FakeProvider("p", clock), [1, 2], target=1, clock=clock, slot_name=slot_name)
    pool.log = lines.append
    pool.events = EventLog(log=lines.append, clock=clock)
    ticks(pool, 2, clock)
    return pool, lines


def test_non_nc_pool_logs_and_events_say_slot():
    pool, lines = _pool("slot")
    ordered = next(e for e in pool.events.events if e["event"] == "nc_elastic_ordered")
    assert ordered["slot"] == ordered["account"] == 1 and ordered["slot_name"] == "slot"
    text = "\n".join(lines)
    assert "on slot 1" in text and "slot=1" in text
    assert "account" not in text


def test_nc_pool_logs_and_events_say_account():
    pool, lines = _pool("account")
    ordered = next(e for e in pool.events.events if e["event"] == "nc_elastic_ordered")
    assert ordered["slot_name"] == "account" and ordered["account"] == 1
    text = "\n".join(lines)
    assert "on account 1" in text and "account=1" in text and "slot=1" not in text


def test_event_log_without_slot_name_is_unchanged():
    lines = []
    EventLog(log=lines.append)("ordered", ticket="t", account=2)
    assert lines == ["pool event nc_elastic_ordered ticket=t account=2"]


def test_dashboard_keeps_slot_fields():
    assert {"account", "slot", "slot_name"} <= set(POOL_KEYS)


def test_controller_status_names_members_by_slot_name():
    from yeto.launch.pool.controller import RolloutController
    import inspect

    src = inspect.getsource(RolloutController.status)
    assert "slot_name" in src
