"""S27 G-DI design v2: the rollout engines go away and come back, the trainer stays.

CPU only, FakeEngine driver (as tests/test_s27_rollout_ahead.py) + a fake recovery port."""

from __future__ import annotations

import json

import pytest
import torch

from yeto.rl.engine.engine_outage import EngineOutage, EnginesLost, OutageTimeout, outage_budget_s

from test_s27_rollout_ahead import NAME, _calls, _driver


class _Port:
    """Engines alive unless ``down``; nodes come back after ``ready_after`` polls."""

    def __init__(self, ready_after=2):
        self.down, self.ready_after, self.polls = False, ready_after, 0
        self.restores, self.aborts = 0, 0

    def engines_alive(self):
        return not self.down

    def is_engine_loss(self, error):
        return "engine node died" in str(error)

    def abort_generation(self):
        self.aborts += 1

    def nodes_ready(self):
        self.polls += 1
        if self.polls > self.ready_after:
            self.down = False
            return True
        return False

    def restore_engines(self):
        self.restores += 1
        return {"cells": ["c0", "c1"]}


def _outage(driver, port, wait_s=100.0):
    t = [0.0]

    def sleep(s):
        t[0] += s

    driver.engine_outage = EngineOutage(port, wait_s=wait_s, emit=driver.emit, poll_s=0.01,
                                        heartbeat_s=5.0, clock=lambda: t[0], sleep=sleep)
    return driver.engine_outage


def _outage_events(tmp_path):
    return [json.loads(line) for line in (tmp_path / "e.jsonl").read_text().splitlines()
            if '"rl_engine_outage"' in line]


def _trains(engine):
    return [c[1] for c in engine.calls if c[0] == "train"]


def _lose_once(engine, port, rollout_id, attr="generate", message="engine node died"):
    real = getattr(engine.rollout, attr)
    state = {"done": False}

    def flaky(*args, **kwargs):
        rid = args[0] if attr == "generate" else args[0].rollout_id
        if rid == rollout_id and not state["done"]:
            state["done"] = True
            port.down = True
            if attr != "generate":
                engine.generating.discard(rid)
            raise RuntimeError(message)
        return real(*args, **kwargs)

    setattr(engine.rollout, attr, flaky)


def test_serial_round_survives_an_engine_outage(tmp_path):
    driver, engine = _driver(tmp_path, rounds=4, limit=0, ahead=0)
    port = _Port()
    _outage(driver, port)
    _lose_once(engine, port, 2)
    final = driver.run()
    assert final.policy_version == 4 and torch.equal(final.tensors[NAME], torch.full((1, 2), 4.0))
    assert _trains(engine) == [0, 1, 2, 3]  # every round trained once, the trainer never restarted
    assert port.restores == 1
    publishes = [c[1] for c in engine.calls if c[0] == "publish"]
    assert publishes == [0, 1, 2, 2, 3, 4]  # v2 republished into the restored engines
    phases = [e["phase"] for e in _outage_events(tmp_path)]
    assert phases == ["batch_lost", "lost", "restoring", "restored"]
    restored = [e for e in _outage_events(tmp_path) if e["phase"] == "restored"][0]
    assert restored["policy_version"] == 2 and restored["cells"] == ["c0", "c1"]


def test_publication_is_deferred_while_the_engines_are_gone(tmp_path):
    driver, engine = _driver(tmp_path, rounds=3, limit=0, ahead=0)
    port = _Port(ready_after=1)
    _outage(driver, port)
    real_generate = engine.rollout.generate

    def generate(rollout_id, **kw):
        batch = real_generate(rollout_id, **kw)
        if rollout_id == 1:
            port.down = True  # the engine node dies after batch 1 came back, while round 1 trains
        return batch

    engine.rollout.generate = generate
    final = driver.run()
    assert final.policy_version == 3
    assert _trains(engine) == [0, 1, 2] and port.restores == 1
    # no broadcast into dead engines: v2 is published once, after the restore
    assert [c[1] for c in engine.calls if c[0] == "publish"] == [0, 1, 2, 3]
    phases = [e["phase"] for e in _outage_events(tmp_path)]
    assert phases == ["lost", "publish_deferred", "restoring", "restored"]


def test_rollout_ahead_batch_lost_in_the_join_is_generated_again(tmp_path):
    driver, engine = _driver(tmp_path, rounds=4, limit=1, ahead=1)
    port = _Port()
    _outage(driver, port)
    _lose_once(engine, port, 2, attr="finish_generate")
    final = driver.run()
    assert final.policy_version == 4
    assert _trains(engine) == [0, 1, 2, 3]
    assert port.restores == 1
    gens = [c[1] for c in engine.calls if c[0] == "generate"]
    assert gens.count(2) == 2  # started ahead, lost, generated again after the restore


def test_no_restore_means_the_old_failure_path(tmp_path):
    driver, engine = _driver(tmp_path, rounds=3, limit=0, ahead=0)
    port = _Port(ready_after=10**9)
    _outage(driver, port, wait_s=30.0)
    _lose_once(engine, port, 1)
    with pytest.raises(OutageTimeout, match="did not come back"):
        driver.run()
    assert port.restores == 0
    assert [e["phase"] for e in _outage_events(tmp_path)][-1] == "timeout"


def test_other_errors_are_not_an_outage(tmp_path):
    driver, engine = _driver(tmp_path, rounds=3, limit=0, ahead=0)
    port = _Port()
    _outage(driver, port)
    _lose_once(engine, port, 1, message="reward function crashed")
    port_down = port.down
    with pytest.raises(RuntimeError, match="reward function crashed"):
        driver.run()
    assert port.restores == 0 and port_down is False


def test_monitor_aborts_a_generation_whose_engines_died():
    port = _Port()
    outage = EngineOutage(port, wait_s=10, emit=lambda *a, **k: None, poll_s=0.01)
    import time as _t

    with outage.watch():
        port.down = True
        _t.sleep(0.1)
    assert port.aborts == 1 and outage.lost(None)
    assert outage.lost(EnginesLost("x"))


def test_budget_env():
    assert outage_budget_s({}) is None
    assert outage_budget_s({"YETO_ISLAND_ENGINE_RECOVERY_S": "0"}) is None
    assert outage_budget_s({"YETO_ISLAND_ENGINE_RECOVERY_S": "3600"}) == 3600.0


def test_miles_port_restarts_the_declared_cells():
    from types import SimpleNamespace

    from yeto.rl.adapters.miles.engine_recovery import MilesEngineRecovery

    class Pool:
        def __init__(self):
            self.calls, self.epoch = [], 4

        def members(self):
            return frozenset()  # every engine died with its node

        def membership_status(self):
            return {"epoch": self.epoch}

        def _require_declared(self):
            return ("e0", "e1")

        def remove_engines(self, members, *, epoch):
            self.calls.append(("remove", sorted(members), epoch))

        def add_engines(self, count, *, epoch, members):
            self.calls.append(("add", count, epoch, sorted(members)))
            return members

        def abort(self):
            self.calls.append(("abort",))

    nodes = [{"Alive": True, "Resources": {"GPU": 8}}, {"Alive": False, "Resources": {"GPU": 8}}]
    tables = {"pg1": {"state": "RESCHEDULING"}}
    ray = SimpleNamespace(nodes=lambda: nodes, util=SimpleNamespace(placement_group_table=lambda *a: tables))
    pool = Pool()
    port = MilesEngineRecovery(pool, nodes=2, ray_module=ray)
    assert not port.engines_alive() and port.is_engine_loss(RuntimeError("x")) and not port.nodes_ready()
    port.abort_generation()
    nodes.append({"Alive": True, "Resources": {"GPU": 8}})
    assert port.engines_alive() and not port.nodes_ready()  # bundles not placed again yet
    tables["pg1"]["state"] = "CREATED"
    assert port.nodes_ready()
    record = port.restore_engines()
    assert pool.calls == [("abort",), ("add", 2, 4, ["engine:e0", "engine:e1"])]
    assert record["cells"] == ["engine:e0", "engine:e1"]
