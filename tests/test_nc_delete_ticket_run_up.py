"""``delete-ticket`` of a leftover member of a stopped pool, then ``yeto run up``: the
member is a manual release, not a live member of an unfinished stop, and the restarted
pool fills its target again. Fake provider, no tool run, no socket opened."""

from __future__ import annotations

import json

from tests.test_nc_delete_ticket import FakeClient, _env  # noqa: F401 - the fixture
from tests.test_nc_pool_run import _cfg
from tests.test_rollout_island_pool import Clock, FakeFencer, FakeProvider, FakeRoute, make_pool
from yeto.launch.nc import cli, pool_run
from yeto.launch.pool.controller import nc_pool
from yeto.launch.pool.provider import ProviderPoolOps
from yeto.rl import up


def _stopped_pool_with_a_leftover(tmp_path, *, stopped="run down"):
    cfg, path = _cfg(tmp_path)
    clock = Clock()
    provider = FakeProvider("nc", clock)
    pool, _r, _f = make_pool(provider, [1, 4], target=1, clock=clock)
    pool.tick()
    member = pool.live()[0]
    pool.stopped = stopped  # the drain of an earlier stop left this member
    state = cli.pool_state_path(cfg)
    state.parent.mkdir(parents=True, exist_ok=True)
    pool.save(state)
    return cfg, path, clock, provider, member


def _delete(monkeypatch, path, ticket, provider):
    made = []
    monkeypatch.setattr(cli, "client", lambda a, **k: FakeClient(a, made, {"state": "ended", "end_reason": "released"}))
    assert cli.main(["delete-ticket", "--config", path, "--ticket", ticket, "--yes"]) == 0
    provider.evict(ticket, "released")
    return made


def test_run_up_restarts_a_stopped_pool_whose_leftover_was_deleted_by_hand(tmp_path, monkeypatch):
    cfg, path, clock, provider, member = _stopped_pool_with_a_leftover(tmp_path)
    assert _delete(monkeypatch, path, member.ticket, provider) == [(1, member.ticket)]
    monkeypatch.setattr(pool_run, "check_sampler_spec", lambda *a: None)
    slot = nc_pool(cfg, path, ops=ProviderPoolOps(provider, FakeRoute(), FakeFencer()))
    try:
        up.Run.__new__(up.Run)._restart_stopped_pool("nc", slot.pool)  # no RunUpRefused
        assert slot.pool.stopped is None and slot.pool.config.target == 1
        slot.pool.tick()
        assert [m.ticket for m in slot.pool.live()] != [member.ticket]
        assert len(provider.orders) == 2  # the target filled once, no second order for it
        for _ in range(3):
            clock.t += 60
            slot.pool.tick()
        assert len(provider.orders) == 2 and len(slot.pool.live()) == 1
    finally:
        slot.lock.close()


def test_a_deleted_member_of_a_pool_that_was_not_stopped_lowers_the_target(tmp_path, monkeypatch):
    cfg, path, _clock, provider, member = _stopped_pool_with_a_leftover(tmp_path, stopped=None)
    _delete(monkeypatch, path, member.ticket, provider)
    scale = json.loads(pool_run.scale_path(cfg).read_text())
    assert scale["target"] == 0 and member.ticket in scale["manual_released"]


def test_deleting_the_same_member_twice_lowers_the_target_once(tmp_path, monkeypatch):
    """A rerun of ``delete-ticket`` (the delete answers ``already``) finds the release
    recorded: the target is not lowered a second time."""
    cfg, path, _clock, provider, member = _stopped_pool_with_a_leftover(tmp_path, stopped=None)
    for t in ("a", "b"):  # two other releases recorded, then the target raised to 3 by hand
        pool_run.record_manual_release(cfg, t)
    scale = json.loads(pool_run.scale_path(cfg).read_text())
    pool_run.scale_path(cfg).write_text(json.dumps({**scale, "target": 3, "seq": scale["seq"] + 1}))
    assert pool_run.current_target(cfg, {}) == 3
    _delete(monkeypatch, path, member.ticket, provider)
    _delete(monkeypatch, path, member.ticket, provider)
    scale = json.loads(pool_run.scale_path(cfg).read_text())
    assert scale["target"] == 2 and set(scale["manual_released"]) == {"a", "b", member.ticket}
