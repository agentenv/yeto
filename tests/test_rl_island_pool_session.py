"""PoolAvgSync at the session level against an in-process fake syncer: debug
switches, the finished race, island tape fields, and the launcher wiring.
CPU, no Ray. The real syncer is covered by test_rl_island_pool_e2e.py."""

from __future__ import annotations

import json

import pytest
import torch

from pool_fakes import FakePoolSyncer
from test_rl_engine_driver import _driver, _engine, _strict_config
from yeto.rl.engine.bridges import PoolAvgSync
from yeto.rl.pool_client import (
    DEBUG_DELAY_ENV, DEBUG_PAUSE_ENV, parse_elastic_debug_delay, parse_elastic_debug_pause,
)


def _sync(tmp_path, engine, client, *, rounds=2, learner_id=0):
    return PoolAvgSync(_strict_config(tmp_path, engine, learner_id=learner_id, rounds=rounds),
                       client=client, base_wait_s=5.0)


def _tape(tmp_path):
    return [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]


def test_debug_switches_parse_and_reach_the_island_env():
    from test_rl_launcher import _args
    from yeto import launcher

    assert parse_elastic_debug_pause("1:1:120") == {1: (1, 120.0)}
    assert parse_elastic_debug_pause(" 0:2:5, 1:1:7.5 ") == {0: (2, 5.0), 1: (1, 7.5)}
    assert parse_elastic_debug_pause("") == {} and parse_elastic_debug_pause(None) == {}
    for bad in ("1:120", "1:1:0", "-1:1:5", "1:1:5:9", "a:1:5"):
        with pytest.raises(ValueError):
            parse_elastic_debug_pause(bad)
    assert parse_elastic_debug_delay("1:2.5,3") == {1: 2.5, -1: 3.0}
    pool = ("--rl-sync-island-lease-s", "60")
    assert launcher.elastic_debug_pause_env(_args(pool)) == {}
    assert launcher.elastic_debug_pause_env(_args(pool + ("--rl-elastic-debug-pause", "1:1:120"))) == {
        DEBUG_PAUSE_ENV: "1:1:120"}
    assert launcher.elastic_debug_delay_env(_args(pool + ("--rl-elastic-debug-delay-s", "1:3"))) == {
        DEBUG_DELAY_ENV: "1:3"}
    with pytest.raises(ValueError, match="needs island pool options"):
        launcher.elastic_debug_pause_env(_args(("--rl-elastic-debug-pause", "1:1:120")))


def test_default_off_session_never_pauses_and_pushes_each_round(tmp_path, monkeypatch):
    monkeypatch.delenv(DEBUG_PAUSE_ENV, raising=False)
    fake = FakePoolSyncer()
    engine = _engine(torch.tensor([1.0, 3.0]))
    client = fake.client()
    _driver(engine, _sync(tmp_path, engine, client), tmp_path).run()
    assert client.paused == [] and fake.inits == 1
    assert [(d["base_version"], d["update"].tolist()) for d in fake.deltas] == [
        (0, [1.0, 3.0]), (1, [1.0, 3.0])]
    assert "elastic_debug_pause" not in (tmp_path / "events.jsonl").read_text()


def test_session_pauses_only_the_named_island_once_after_after_v(tmp_path, monkeypatch):
    monkeypatch.setenv(DEBUG_PAUSE_ENV, "0:1:0.05,5:0:9")
    fake = FakePoolSyncer()
    engine = _engine(torch.tensor([1.0, 3.0]))
    client = fake.client()
    _driver(engine, _sync(tmp_path, engine, client, rounds=3), tmp_path).run()
    armed = [e for e in _tape(tmp_path) if e.get("phase") == "elastic_debug_pause_armed"]
    assert len(armed) == 1 and armed[0]["base_version"] == 1 and armed[0]["pause_s"] == 0.05
    assert client.paused == [0.05]
    assert [e["pause_s"] for e in _tape(tmp_path) if e.get("event") == "elastic_debug_pause"] == [0.05]


def test_an_island_that_applied_the_final_cut_stops_without_another_push(tmp_path):
    """S17 M1: the island stops once it applied the final outer version."""
    fake = FakePoolSyncer(final=3)
    engine = _engine(torch.tensor([1.0, 3.0]))
    sync = _sync(tmp_path, engine, fake.client(), rounds=3)
    _driver(engine, sync, tmp_path).run()
    tape = _tape(tmp_path)
    assert sync.base_version == 3 and not [e for e in tape if e["event"] == "elastic_finished"]
    assert sum(e["event"] == "rl_local_round" for e in tape) == 1


def test_push_racing_the_end_takes_the_finished_path(tmp_path):
    """The island's base is not final yet, but the run ended meanwhile: the push is
    refused and the island ends normally on the final cut (elastic_finished)."""
    fake = FakePoolSyncer([0.0, 0.0], version=1, final=1)
    engine = _engine(torch.tensor([1.0, 3.0]))
    sync = _sync(tmp_path, engine, fake.client(), rounds=4)
    fake.final = None  # the start sees a running pool at version 1 ...
    original = sync.start

    def start(driver):
        started = original(driver)
        fake.final = 1  # ... which finishes before the island's first push
        return started
    sync.start = start
    _driver(engine, sync, tmp_path).run()
    done = [e for e in _tape(tmp_path) if e["event"] == "elastic_finished"]
    assert len(done) == 1 and done[0]["final_outer_version"] == 1
    assert fake.leaves == [0] and fake.deltas == []


def test_pool_round_fields_match_the_strict_local_round(tmp_path):
    fake = FakePoolSyncer()
    engine = _engine(torch.tensor([3.0, 4.0]))
    _driver(engine, _sync(tmp_path, engine, fake.client(), rounds=1), tmp_path).run()
    [record] = [e for e in _tape(tmp_path) if e["event"] == "rl_local_round"]
    assert record["rl/local_delta_norm"] == pytest.approx(5.0)
    assert record["sync/base_outer_version"] == 0 and record["sync/bytes_sent"] == 8
    assert record["rl/global_policy_version"] == 0


def test_the_island_records_its_sync_timing(tmp_path):
    fake = FakePoolSyncer(reply_delay_s=0.2)
    engine = _engine(torch.tensor([1.0, 3.0]))
    _driver(engine, _sync(tmp_path, engine, fake.client(), rounds=1), tmp_path).run()
    [timing] = [e for e in _tape(tmp_path) if e["event"] == "rl_sync_timing"]
    assert timing["outer_sync"] == "elastic" and timing["base_version"] == 1
    assert timing["sync/global_ready_ts"] >= timing["sync/submit_ts"]
    assert timing["sync/sync_wait_seconds"] >= 0.0


def test_client_events_reach_the_tape_once(tmp_path, monkeypatch):
    """1b-g: an injected client's tape hook is chained exactly once."""
    monkeypatch.setenv(DEBUG_PAUSE_ENV, "0:1:0.05")
    fake = FakePoolSyncer()
    engine = _engine(torch.tensor([1.0, 3.0]))
    _driver(engine, _sync(tmp_path, engine, fake.client(), rounds=3), tmp_path).run()
    assert [e.get("event") for e in _tape(tmp_path)].count("elastic_debug_pause") == 1


def test_the_miles_entry_selects_the_pool_session(monkeypatch):
    from types import SimpleNamespace

    from yeto.rl.adapters.miles import entry
    from yeto.rl.engine.bridges import StrictAvgSync

    monkeypatch.setattr("yeto.rl.engine.bridges.StrictIslandProgress", lambda args: None)
    base = dict(yeto_rl_bridge_config=SimpleNamespace(groups_per_round=2), rollout_batch_size=2)
    sync, _ = entry.build_sync(SimpleNamespace(**base, yeto_rl_island_pool=True,
                                               yeto_rl_syncer_epoch=3, yeto_rl_island_lease_s=45.0),
                               yeto_policy_sync=True)
    assert isinstance(sync, PoolAvgSync) and sync.syncer_epoch == 3 and sync.lease_s == 45.0
    strict, _ = entry.build_sync(SimpleNamespace(**base), yeto_policy_sync=True)
    assert isinstance(strict, StrictAvgSync)
