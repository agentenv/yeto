"""Test switch --rl-elastic-debug-pause: the island's link to the syncer goes silent for S
seconds (no heartbeat, sends/receives held) while the process keeps running, to exercise
lease expiry -> "rejoin required" -> automatic re-JOIN on real hardware. CPU, no Ray.

The end-to-end case runs the real Rust syncer when YETO_TEST_ELASTIC_SYNCER points to it."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest
import torch

from test_rl_elastic_lease_rejoin import MultiConnSyncer, _client, _wait
from test_rl_engine_driver import _driver, _engine, _run_threads, _strict_config
from test_rl_inter_island_elastic_client import FakeSyncer
from test_rl_inter_island_ports import _free_port, _sync
from yeto.rl.elastic_client import (
    DeltaTensor, ElasticClientConfig, ElasticIslandClient, LeaseHeartbeat, parse_elastic_debug_pause,
)

PAUSE_ENV = "YETO_RL_ELASTIC_DEBUG_PAUSE"


def test_parse_and_launcher_env():
    from test_rl_launcher import _args
    from yeto import launcher

    assert parse_elastic_debug_pause("1:1:120") == {1: (1, 120.0)}
    assert parse_elastic_debug_pause(" 0:2:5, 1:1:7.5 ") == {0: (2, 5.0), 1: (1, 7.5)}
    assert parse_elastic_debug_pause("") == {} and parse_elastic_debug_pause(None) == {}
    for bad in ("1:120", "1:1:0", "-1:1:5", "1:1:5:9", "a:1:5"):
        with pytest.raises(ValueError):
            parse_elastic_debug_pause(bad)
    assert launcher.elastic_debug_pause_env(_args()) == {}
    assert launcher.elastic_debug_pause_env(_args(("--rl-island-scheduling", "elastic"))) == {}
    assert launcher.elastic_debug_pause_env(_args(("--rl-island-scheduling", "elastic",
                                                   "--rl-elastic-debug-pause", "1:1:120"))) == {
        PAUSE_ENV: "1:1:120"}
    with pytest.raises(ValueError, match="needs --rl-island-scheduling elastic"):
        launcher.elastic_debug_pause_env(_args(("--rl-elastic-debug-pause", "1:1:120")))


def test_default_off_island_task_envs_unchanged(monkeypatch, tmp_path):
    """Without the flag the island task / Modal cfg carry no new env (same bytes as before)."""
    from test_rl_inter_island_launcher import _fake_sky, _prov
    from test_rl_launcher import _args, _prepare_rl_args
    from yeto import launcher
    from yeto.gpu_spec import parse_gpu_spec

    _fake_sky(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "k")
    base = ("--gpu", "modal:1xh100,modal:1xh100", "--rl-engine", "ports", "--rl-island-scheduling", "elastic")

    def build(extra):
        args = _args(base + extra)
        _prepare_rl_args(_prov(args))
        spec = parse_gpu_spec(args.gpu)[1]
        task = launcher.make_miles_island_task(args, spec, 1, 2, "127.0.0.1:29400")
        return task, launcher.build_modal_island_config(args, spec, 1, task, "1.2.3.4:5000")

    off, off_cfg = build(())
    on, on_cfg = build(("--rl-elastic-debug-pause", "1:1:120"))
    assert PAUSE_ENV not in (off.envs or {}) and PAUSE_ENV not in off_cfg.envs
    assert on.envs[PAUSE_ENV] == "1:1:120" and on_cfg.envs[PAUSE_ENV] == "1:1:120"
    assert {k: v for k, v in on.envs.items() if k != PAUSE_ENV} == off.envs
    assert on.run == off.run and on.setup == off.setup and on_cfg.run_script == off_cfg.run_script


def test_default_off_bridge_never_pauses(tmp_path, monkeypatch):
    monkeypatch.delenv(PAUSE_ENV, raising=False)
    fake = FakeSyncer()
    engine = _engine(torch.tensor([1.0, 3.0]))
    client = ElasticIslandClient(ElasticClientConfig(("x", 0), 0, lease_s=3.0, join_timeout_s=2),
                                 fake.key, connect=fake.connect)
    called = []
    client.pause_link = called.append
    sync = _sync(tmp_path, engine, client)
    _driver(engine, sync, tmp_path).run()
    assert called == [] and client._pause_until == 0.0 and not client.link_paused()
    assert "elastic_debug_pause" not in (tmp_path / "events.jsonl").read_text()
    deltas = [m for m in fake.frames if isinstance(m, DeltaTensor)]
    assert [(d.base_version, d.update) for d in deltas] == [(0, (1.0, 3.0)), (1, (1.0, 3.0))]


def test_bridge_pauses_only_the_named_island_once_after_after_v(tmp_path, monkeypatch):
    monkeypatch.setenv(PAUSE_ENV, "0:1:0.05,5:0:9")
    fake = FakeSyncer()
    engine = _engine(torch.tensor([1.0, 3.0]))
    client = ElasticIslandClient(ElasticClientConfig(("x", 0), 0, lease_s=3.0, join_timeout_s=2),
                                 fake.key, connect=fake.connect)
    sync = _sync(tmp_path, engine, client, rounds=3)
    _driver(engine, sync, tmp_path).run()
    ev = [json.loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines()]
    armed = [e for e in ev if e.get("phase") == "elastic_debug_pause_armed" or e.get("event") == "elastic_debug_pause_armed"]
    assert len(armed) == 1 and armed[0].get("base_version") == 1 and armed[0].get("pause_s") == 0.05
    assert [e["pause_s"] for e in ev if e.get("event") == "elastic_debug_pause"] == [0.05]


def test_pause_silences_heartbeats_and_holds_sends():
    fake, events = MultiConnSyncer(), []
    c = _client(fake, events)  # lease 0.15 s -> heartbeat every 0.05 s
    c.join()
    hb = lambda: sum(isinstance(m, LeaseHeartbeat) for _, m in fake.frames)  # noqa: E731
    assert _wait(lambda: hb() >= 2)
    c.pause_link(0.6)
    time.sleep(0.1)
    n0 = hb()
    t0 = time.monotonic()
    c.delta_tensor(base_version=0, c_tokens=1, c_steps=1, update=(1.0, 1.0))  # held until the pause ends
    assert time.monotonic() - t0 >= 0.4
    assert hb() == n0 or hb() <= n0 + 1  # at most one heartbeat raced the end of the pause
    assert _wait(lambda: hb() >= n0 + 3)  # renewal resumes afterwards
    assert _wait(lambda: any(e.get("event") == "elastic_debug_pause_end" for e in events))
    c.close()


@pytest.mark.skipif(not os.environ.get("YETO_TEST_ELASTIC_SYNCER"),
                    reason="needs YETO_TEST_ELASTIC_SYNCER (syncer binary with elastic mode)")
def test_link_pause_longer_than_lease_rejoins_against_real_rust_syncer(tmp_path, monkeypatch):
    """Island 1 goes silent for 5 s (lease 2 s) after v1: the syncer expires its lease,
    the island's heartbeat is answered "rejoin required", it re-JOINs (catch_up) and then
    takes part in later outer steps; both islands end normally and the syncer exits 0."""
    port, key = _free_port(), "pause-e2e"
    out = Path(os.environ.get("YETO_TEST_ELASTIC_OUT", tmp_path))
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jsonl"):
        old.unlink()
    monkeypatch.setenv(PAUSE_ENV, "1:1:5")
    monkeypatch.setenv("YETO_RL_ELASTIC_DEBUG_DELAY", "0:1.5,1:0.5")  # keep island 0 slow
    rounds = 8
    syncer = subprocess.Popen(
        [os.environ["YETO_TEST_ELASTIC_SYNCER"], "--port", str(port), "--learners", "2",
         "--total-steps", str(rounds), "--event-tape", str(out / "syncer-tape.jsonl"),
         "--island-scheduling-mode", "elastic", "--quorum-theta", "0.75", "--carry-gamma", "0.5",
         "--soft-deadline-s", "2", "--q-min", "1", "--max-carry-lag", "2",
         "--island-lease-s", "2", "--final-grace-s", "10", "--syncer-epoch", "0"],
        env=dict(os.environ, YETO_ISLAND_HMAC_KEY=key),
        stdout=open(out / "syncer.log", "w"), stderr=subprocess.STDOUT)
    try:
        time.sleep(1.0)
        engines = (_engine(torch.tensor([1.0, 3.0])), _engine(torch.tensor([3.0, 5.0])))
        drivers = []
        for i, e in enumerate(engines):
            c = ElasticIslandClient(ElasticClientConfig(("127.0.0.1", port), i, lease_s=2.0),
                                    key.encode())
            cfg = _strict_config(out, e, learner_id=i, rounds=rounds, tape=f"island-{i}.jsonl")
            drivers.append(_driver(e, ElasticAvgSync_(cfg, c), out, name=f"events-{i}.jsonl", learner_id=i))
        results, errors = _run_threads(drivers)
        assert errors == {}, errors
        assert syncer.wait(30) == 0
        rows = [json.loads(l) for l in (out / "syncer-tape.jsonl").read_text().splitlines()]
        leaves = [r for r in rows if r.get("kind") == "pool_leave" and r.get("island_id") == 1
                  and r.get("reason") == "lease_expired"]
        assert len(leaves) == 1, [r for r in rows if r.get("kind", "").startswith("pool")]
        joins = [r for r in rows if r.get("kind") == "pool_join" and r.get("island_id") == 1
                 and r["membership_epoch"] > leaves[0]["membership_epoch"]]
        assert joins and joins[0]["catch_up"] is True
        assert "1 is not a member (rejoin required)" in (out / "syncer.log").read_text()
        ev1 = [json.loads(l) for l in (out / "events-1.jsonl").read_text().splitlines()]
        names = [e.get("event") for e in ev1]
        assert names.count("elastic_debug_pause") == 1 and names.count("elastic_debug_pause_end") == 1
        rj = [e for e in ev1 if e.get("event") == "elastic_rejoin"]
        assert len(rj) == 1 and rj[0]["catch_up"] is True
        # after the re-JOIN island 1 trains and its delta is merged with weight > 0
        idx = rows.index(joins[0])
        later = [r for r in rows[idx:] if r.get("kind") == "outer_step"]
        w1 = [[x[2] for x in (r.get("weights") or []) if x[0] == 1] for r in later]
        assert any(v and v[0] > 0 for v in w1), later
        assert names.count("elastic_finished") <= 1
    finally:
        if syncer.poll() is None:
            syncer.kill()


def ElasticAvgSync_(cfg, client):  # noqa: N802
    from yeto.rl.engine.bridges import ElasticAvgSync

    return ElasticAvgSync(cfg, client=client, base_wait_s=60.0)


def test_self_built_client_events_reach_the_tape_once(tmp_path, monkeypatch):
    """1b-g: a client built by ElasticAvgSync._client() was hooked twice, so every
    elastic_rejoin / elastic_debug_pause appeared twice on the island tape."""
    from yeto.rl.engine import bridges

    fake = FakeSyncer()
    engine = _engine(torch.tensor([1.0, 3.0]))
    sync = _sync(tmp_path, engine, None)
    built = []

    def make(*a, **kw):
        c = ElasticIslandClient(ElasticClientConfig(("x", 0), 0, lease_s=3.0, join_timeout_s=2),
                                fake.key, connect=fake.connect, on_event=kw.get("on_event"))
        built.append(c)
        return c
    import yeto.rl.elastic_client as ec
    monkeypatch.setattr(ec, "ElasticIslandClient", make)
    monkeypatch.setattr(ec, "hmac_key_from_env", lambda: fake.key)
    monkeypatch.setenv(PAUSE_ENV, "0:1:0.05")
    _driver(engine, sync, tmp_path).run()
    assert built and bridges.ElasticAvgSync
    ev = [json.loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [e.get("event") for e in ev].count("elastic_debug_pause") == 1
