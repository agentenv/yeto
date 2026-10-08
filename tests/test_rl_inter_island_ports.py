"""ports engine + ElasticAvgSync (rl-inter-island-scheduling 0.15). CPU, no Ray.

The end-to-end case runs the real Rust syncer binary when YETO_TEST_ELASTIC_SYNCER
points to one built from the syncer source with elastic support; otherwise skipped.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_rl_engine_driver import NAME, _driver, _engine, _run_threads, _strict_config
from test_rl_inter_island_elastic_client import FakeSyncer
from yeto.rl.elastic_client import DeltaTensor, ElasticClientConfig, ElasticInit, ElasticIslandClient, Leave
from yeto.rl.engine.bridges import ElasticAvgSync, StrictAvgSync
from yeto.rl.engine.miles_adapter import entry


def _sync(tmp_path, engine, client, *, learner_id=0, rounds=2):
    return ElasticAvgSync(_strict_config(tmp_path, engine, learner_id=learner_id, rounds=rounds),
                          client=client, base_wait_s=10.0)


def test_ports_driver_with_elastic_sync_fake_syncer(tmp_path):
    fake = FakeSyncer()
    engine = _engine(torch.tensor([1.0, 3.0]))
    client = ElasticIslandClient(ElasticClientConfig(("x", 0), 0, lease_s=3.0, join_timeout_s=2),
                                 fake.key, connect=fake.connect)
    result = _driver(engine, _sync(tmp_path, engine, client), tmp_path).run()
    time.sleep(0.05)
    inits = [m for m in fake.frames if isinstance(m, ElasticInit)]
    deltas = [m for m in fake.frames if isinstance(m, DeltaTensor)]
    assert inits == [ElasticInit(0, 0, (0.0, 0.0))]
    assert [(d.base_version, d.update) for d in deltas] == [(0, (1.0, 3.0)), (1, (1.0, 3.0))]
    assert isinstance(fake.frames[-1], Leave)
    assert torch.equal(result.tensors[NAME], torch.tensor([[2.0, 6.0]]))


def test_entry_selects_by_mode(monkeypatch):
    import yeto.rl.engine.bridges as b
    monkeypatch.setattr(b, "StrictIslandProgress", lambda args: None)
    base = dict(num_rollout=2, yeto_rl_bridge_config=SimpleNamespace())
    sync, _ = entry.build_sync(SimpleNamespace(**base), yeto_policy_sync=True)
    assert type(sync) is StrictAvgSync  # legacy: attribute absent
    sync, _ = entry.build_sync(SimpleNamespace(**base, yeto_rl_island_scheduling="elastic",
                                               yeto_rl_syncer_epoch=7), yeto_policy_sync=True)
    assert type(sync) is ElasticAvgSync and sync.syncer_epoch == 7


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


@pytest.mark.skipif(not os.environ.get("YETO_TEST_ELASTIC_SYNCER"),
                    reason="needs YETO_TEST_ELASTIC_SYNCER (syncer binary with elastic mode)")
def test_two_ports_islands_against_real_rust_syncer(tmp_path):
    port, key = _free_port(), "ports-e2e"
    out = Path(os.environ.get("YETO_TEST_ELASTIC_OUT", tmp_path))
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jsonl"):  # the syncer tape appends
        old.unlink()
    syncer = subprocess.Popen(
        [os.environ["YETO_TEST_ELASTIC_SYNCER"], "--port", str(port), "--learners", "2",
         "--total-steps", "2", "--event-tape", str(out / "syncer-tape.jsonl"),
         "--island-scheduling-mode", "elastic", "--quorum-theta", "0.75", "--carry-gamma", "0.5",
         "--soft-deadline-s", "30", "--q-min", "1", "--max-carry-lag", "2",
         "--island-lease-s", "6", "--syncer-epoch", "5"],
        env=dict(os.environ, YETO_ISLAND_HMAC_KEY=key),
        stdout=open(out / "syncer.log", "w"), stderr=subprocess.STDOUT)
    try:
        time.sleep(1.0)
        engines = (_engine(torch.tensor([1.0, 3.0])), _engine(torch.tensor([3.0, 5.0])))
        drivers = []
        for i, e in enumerate(engines):
            c = ElasticIslandClient(ElasticClientConfig(("127.0.0.1", port), i, lease_s=6.0,
                                                        syncer_epoch=5), key.encode())
            cfg = _strict_config(out, e, learner_id=i, rounds=2, tape=f"island-{i}.jsonl")
            drivers.append(_driver(e, ElasticAvgSync(cfg, client=c, base_wait_s=30.0), out,
                                   name=f"events-{i}.jsonl", learner_id=i))
        results, errors = _run_threads(drivers)
        assert errors == {}
        assert torch.equal(results[0].tensors[NAME], results[1].tensors[NAME])
        assert syncer.wait(15) == 0
        kinds = [l.split('"kind":"')[1].split('"')[0]
                 for l in (out / "syncer-tape.jsonl").read_text().splitlines()]
        assert kinds.count("outer_step") == 2 and kinds.count("pool_join") == 2
    finally:
        if syncer.poll() is None:
            syncer.kill()


def test_elastic_emits_rl_local_round_like_strict(tmp_path):
    """0.24: one rl_local_round per elastic round with reward/kl/update norm."""
    import inspect
    import json
    import math

    from yeto.rl import bridge as strict_bridge

    fake = FakeSyncer()
    engine = _engine(torch.tensor([1.0, 3.0]))
    client = ElasticIslandClient(ElasticClientConfig(("x", 0), 0, lease_s=3.0, join_timeout_s=2),
                                 fake.key, connect=fake.connect)
    _driver(engine, _sync(tmp_path, engine, client), tmp_path).run()
    rounds = [json.loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines()
              if '"rl_local_round"' in l]
    assert len(rounds) == 2
    assert [r["rl/global_policy_version"] for r in rounds] == [0, 1]
    assert all(math.isclose(r["rl/local_delta_norm"], math.sqrt(10.0), rel_tol=1e-6) for r in rounds)
    for key in ("rl/reward_mean", "rl/current_vs_rollout_kl", "rl/local_delta_norm", "reward_mean"):
        assert key in rounds[0]
    strict_keys = set(__import__("re").findall(r'"(rl/[a-z_0-9]+|sync/bytes_sent)"',
                                               inspect.getsource(strict_bridge.StrictRlBridge._record_submission)))
    assert strict_keys <= set(rounds[0])


def test_debug_delay_switch(tmp_path, monkeypatch):
    """0.25: --rl-elastic-debug-delay-s -> env on the island -> sleep before each delta."""
    import json

    from test_rl_launcher import _args
    from yeto import launcher
    from yeto.rl.elastic_client import parse_elastic_debug_delay

    assert parse_elastic_debug_delay("0:90, 1:5") == {0: 90.0, 1: 5.0}
    assert parse_elastic_debug_delay("30") == {-1: 30.0}
    with pytest.raises(ValueError):
        parse_elastic_debug_delay("0:-1")
    assert launcher.elastic_debug_delay_env(_args()) == {}
    assert launcher.elastic_debug_delay_env(_args(("--rl-island-scheduling", "elastic",
                                                   "--rl-elastic-debug-delay-s", "0:90"))) == {
        "YETO_RL_ELASTIC_DEBUG_DELAY": "0:90"}
    with pytest.raises(ValueError, match="needs --rl-island-scheduling elastic"):
        launcher.elastic_debug_delay_env(_args(("--rl-elastic-debug-delay-s", "0:90")))

    monkeypatch.setenv("YETO_RL_ELASTIC_DEBUG_DELAY", "0:7.5,1:2")
    fake = FakeSyncer()
    engine = _engine(torch.tensor([1.0, 3.0]))
    client = ElasticIslandClient(ElasticClientConfig(("x", 0), 0, lease_s=3.0, join_timeout_s=2),
                                 fake.key, connect=fake.connect)
    sync = _sync(tmp_path, engine, client)
    slept = []
    sync._sleep = slept.append
    _driver(engine, sync, tmp_path).run()
    assert slept == [7.5, 7.5]
    phases = [json.loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines()
              if '"elastic_debug_delay"' in l]
    assert [p["delay_s"] for p in phases] == [7.5, 7.5]
