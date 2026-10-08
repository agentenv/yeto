"""0.21: an elastic island restart resumes its round id and data cursor. CPU, no Ray.

Before 0.21 ``ElasticAvgSync.start`` always returned round 0, so a restarted island
re-drew the groups it had already trained (or the ledger refused them)."""

from __future__ import annotations

import pytest

from yeto.rl.elastic_client import ElasticClientConfig, ElasticIslandClient
from yeto.rl.engine.bridges import ElasticAvgSync

from test_rl_engine_driver import _strict_config
from test_rl_inter_island_elastic_client import FakeSyncer
from test_rl_restart_data_cursor import (DataSourceState, MilesLikePool, _close, _events,
                                         _records)
from test_rl_reconfig_e1 import _setup


def _island(tmp_path, fake, *, rounds, source, die_at=None, tape="a.jsonl", ledger=True):
    def factory(engine):
        client = ElasticIslandClient(ElasticClientConfig(("x", 0), 0, lease_s=30.0, join_timeout_s=2),
                                     fake.key, connect=fake.connect)
        return ElasticAvgSync(_strict_config(tmp_path, engine, learner_id=0, rounds=rounds, tape=tape),
                              client=client, base_wait_s=5.0)
    driver, ctl, *_ = _setup(tmp_path, rounds=rounds, outer="strict-avg", sync_factory=factory,
                            ledger=ledger)
    pool = MilesLikePool(driver.rollout, source, die_at=die_at)
    driver.rollout = pool
    return driver, ctl, pool


def test_restart_resumes_after_ledger_rounds_and_skips_trained_groups(tmp_path):
    rounds = 4
    first = FakeSyncer()
    driver, ctl, _ = _island(tmp_path, first, rounds=rounds, source=DataSourceState(), die_at=2)
    with pytest.raises(RuntimeError, match="killed at QUIESCING"):
        driver.run()
    _close(driver, ctl)
    assert first.version == 2
    cursor_after_1 = {r["rollout_id"]: r for r in _records(tmp_path) if r["kind"] == "prepared"}[1]["data_cursor"]

    restarted = FakeSyncer(base=list(first.base))  # the syncer kept base_version 2
    restarted.version = 2
    driver, ctl, pool = _island(tmp_path, restarted, rounds=rounds, source=DataSourceState(), tape="b.jsonl")
    driver.run()
    assert pool.seeks == [cursor_after_1]  # ledger path (same as strict)
    prepared = [(r["rollout_id"], r["group_ids"]) for r in _records(tmp_path) if r["kind"] == "prepared"]
    assert prepared == [(0, ["g0", "g1"]), (1, ["g2", "g3"]), (2, ["g4", "g5"]), (3, ["g6", "g7"])]
    assert restarted.version == rounds  # rounds 2 and 3 only (third round = rollout id 2)
    ev = _events(tmp_path)
    assert [e["rollout_id"] for e in ev if e["event"] == "rl_data_cursor_restored"] == [2]
    _close(driver, ctl)


def test_empty_ledger_joining_at_base_version_advances_cursor(tmp_path):
    rounds = 4
    fake = FakeSyncer(base=[0.0, 0.0])
    fake.version = 2
    driver, ctl, pool = _island(tmp_path, fake, rounds=rounds, source=DataSourceState())
    driver.run()
    g = driver.sync.groups_per_round
    assert g == 2 and pool.seeks == [{"sample_offset": 2 * g, "epoch_id": 0,
                                      "sample_group_index": 2 * g, "sample_index": 0}]
    prepared = [(r["rollout_id"], r["group_ids"]) for r in _records(tmp_path) if r["kind"] == "prepared"]
    assert prepared == [(2, ["g4", "g5"]), (3, ["g6", "g7"])]
    restored = [e for e in _events(tmp_path) if e["event"] == "cursor_restored"]
    assert [(e["rollout_id"], e["source"]) for e in restored] == [(2, "base_version")]
    _close(driver, ctl)


def test_fresh_run_still_starts_at_zero(tmp_path):
    fake = FakeSyncer()
    driver, ctl, pool = _island(tmp_path, fake, rounds=2, source=DataSourceState())
    driver.run()
    assert pool.seeks == [] and fake.version == 2
    assert not [e for e in _events(tmp_path) if e["event"] in ("cursor_restored", "rl_data_cursor_restored")]
    _close(driver, ctl)


# 1b (run s15-island1b-20261008b, code d65fe066), l1/rank0/rl-island-1.jsonl lines 3-4: the
# restarted Modal island has no batch ledger (no --rl-elastic) and joined at base 3; no
# cursor event followed and rollouts 3, 4 re-trained rollouts 0, 1's samples.
TAPE_1B = [
    {"base_version": 3, "catch_up": True, "event": "rl_driver_phase", "island_id": 1, "phase": "elastic_join"},
    {"base_version": 3, "event": "rl_driver_phase", "island_id": 1, "ledger_next": 0, "outer_version": 3,
     "phase": "elastic_resume", "rollout_id": 3},
]


def test_1b_regression_no_ledger_island_joining_at_base_3_advances_cursor(tmp_path):
    rounds = 5
    fake = FakeSyncer(base=[0.0, 0.0], catch_up=True)
    fake.version = 3
    driver, ctl, pool = _island(tmp_path, fake, rounds=rounds, source=DataSourceState(), ledger=False)
    assert driver.ledger is None
    driver.run()
    ev = _events(tmp_path)
    phases = [{k: e[k] for k in TAPE_1B[i] if k in e} for i, e in enumerate(
        e for e in ev if e.get("phase") in ("elastic_join", "elastic_resume"))]
    assert [{k: v for k, v in p.items() if k != "island_id"} for p in phases] == [
        {k: v for k, v in t.items() if k != "island_id"} for t in TAPE_1B]
    g = driver.sync.groups_per_round
    assert pool.seeks == [{"sample_offset": 3 * g, "epoch_id": 0, "sample_group_index": 3 * g, "sample_index": 0}]
    assert [(e["rollout_id"], e["source"]) for e in ev if e["event"] == "cursor_restored"] == [(3, "base_version")]
    assert fake.version == rounds  # trained rollouts 3 and 4 only, on groups g6.. (not g0..)
    _close(driver, ctl) if driver.ledger is not None else ctl.close()
