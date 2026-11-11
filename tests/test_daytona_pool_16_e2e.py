"""16 Daytona samplers end to end on one Linux host (the process-route world of
``test_daytona_pool_process_e2e``: sandboxes are directories, ``ssh`` runs bash in
them, the sampler is a stub). All 16 are created, preflighted, started and probed
at once; 3 are reclaimed together and refilled; the pool stops them all."""

from __future__ import annotations

import http.server
import json
import shlex
import shutil
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_daytona_pool_process_e2e import STUB_PYTHON, Boxes, Ssh, _kill, _tagged
from test_nc_daytona_provider import FakeSdk

from yeto.launch.nc import cli, node_pool
from yeto.launch.nc.lease_pool import PoolConfig
from yeto.launch.pool import status as el
from yeto.launch.nc.daytona_provider import DaytonaProvider
from yeto.launch.pool.island_pool import GONE, READY, IslandPool

pytestmark = pytest.mark.skipif(not Path("/proc/self/environ").exists() or shutil.which("setsid") is None,
                                reason="needs Linux /proc and setsid")

N = 16
LOG: list[str] = []


class UniqueTokenBoxes(Boxes):
    """Gateway tokens unique per sandbox, as Daytona's are (the fake ssh routes by token)."""

    def create(self, params, timeout=60):
        box = super().create(params, timeout)

        def access(expires_in_minutes=None, box=box):
            token = f"tok-{box.name}-{len(box.tokens) + 1}"
            box.tokens.append((token, expires_in_minutes))
            return SimpleNamespace(token=token, expires_at=None, ssh_command=f"ssh {token}@gateway")

        box.create_ssh_access = access
        return box


@pytest.fixture()
def fleet(tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))
    ssh_config = tmp_path / "home" / ".ssh" / "config"
    monkeypatch.setattr(node_pool, "SSH_CONFIG", ssh_config)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), http.server.SimpleHTTPRequestHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    store_env = tmp_path / "store.env"
    store_env.write_text(f"R2_BUCKET=b\nAWS_ACCESS_KEY_ID=x\nAWS_SECRET_ACCESS_KEY=y\n"
                         f"AWS_ENDPOINT_URL=http://127.0.0.1:{server.server_address[1]}\n")
    store_env.chmod(0o600)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"event_tape": "/work/sampler/events.jsonl", "prompts": {"groups_per_round": 2},
                                "num_rollouts": 8, "max_backlog_groups": 8, "max_prompt_tokens": 64,
                                "sampling": {"max_tokens": 8},
                                "engine": {"factory": "yeto.rl.adapters.sampler_engines.from_spec:build",
                                           "adapter": "fake", "kwargs": {}},
                                "reward": "yeto.rl.numeric_answer_reward:score", "reward_kind": "miles"}))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python3"
    python.write_text(STUB_PYTHON.format(py=shlex.quote(sys.executable)))
    python.chmod(0o755)
    (bin_dir / "df").write_text("#!/bin/bash\necho 'F 1024-blocks Used Available C M'\n"
                                "echo \"sim 8589934592 0 8589934592 0% ${@: -1}\"\n")
    (bin_dir / "df").chmod(0o755)
    (bin_dir / "aws").write_text("#!/bin/bash\nexit 0\n")
    (bin_dir / "aws").chmod(0o755)
    boxes = UniqueTokenBoxes(tmp_path / "boxes")
    slots = list(range(1, 2 * N + 1))
    provider = DaytonaProvider("run-16", slots=slots, image="rocm/vllm:test", sdk=FakeSdk, client=boxes)
    cfg = {"target": N, "accounts": slots, "shard_base": "dt", "run": "run-16", "spec": str(spec),
           "run_prefix": "run-16", "r2_env_file": str(store_env)}
    run = Ssh(boxes, ssh_config, python, bin_dir)
    ops = cli.NcSshOps(cfg, run=run, provider=provider)
    fenced = []
    ops.fence = lambda m: fenced.append(m.shard)
    ops.fence_incarnation = lambda m: fenced.append(m.shard)
    events = el.EventLog(log=lambda m: None, run="run-16")
    pool = IslandPool(PoolConfig(target=N, accounts=slots, shard_base="dt", stop_below_ucr=0,
                                 min_balance_to_order_ucr=0,
                                 elastic=el.ElasticConfig(max_orders=64, unreachable_grace_s=2, engine_stall_s=None)),
                      ops, log=LOG.append, events=events)
    yield pool, boxes, events, fenced
    for name in list(boxes.boxes):
        _kill(name)
    server.shutdown()


def _until(pool, cond, timeout=300):
    t0 = time.monotonic()
    ticks = []
    while time.monotonic() - t0 < timeout:
        pool.tick()
        ticks.append(pool.last_tick_s)
        if cond():
            return ticks
        time.sleep(0.2)
    events = [e for e in pool.events.events if e["event"] != "nc_elastic_drain_failed"]
    raise AssertionError(f"not reached in {timeout}s; events {events[-12:]}; log {LOG[-12:]}")


@pytest.mark.parametrize("parallel", [16, 1])
def test_sixteen_sandboxes_start_together_survive_three_reclaims_and_stop(fleet, parallel):
    pool, boxes, events, fenced = fleet
    pool.watch_parallel = parallel
    t0 = time.monotonic()
    _until(pool, lambda: len([m for m in pool.members if m.state == READY and m.joined]) == N)
    up_s = time.monotonic() - t0
    ready = [m for m in pool.members if m.state == READY]
    assert len({m.shard for m in ready}) == N and len({m.ticket for m in ready}) == N
    # every alias survived the 16 concurrent attaches (one ssh config for all)
    conf = (cli.tunnel.state_dir() / "ssh_config").read_text()
    assert all(f"Host {node_pool.host_alias(m.ticket)}" in conf for m in ready)
    steady = _until(pool, lambda: True)  # one tick: 16 status reads and 16 ssh probes
    for _ in range(3):
        steady += _until(pool, lambda: True)

    victims = ready[2], ready[9], ready[14]
    for m in victims:
        boxes.evict(m.ticket)
    t1 = time.monotonic()
    _until(pool, lambda: all(m.state == GONE for m in victims)
           and len([m for m in pool.members if m.state == READY and m.joined]) == N)
    refill_s = time.monotonic() - t1
    assert sorted(fenced) == sorted(m.shard for m in victims)
    live = [m for m in pool.members if m.state == READY]
    assert sorted(m.replaces_shard for m in live if m.replaces_shard) == sorted(m.shard for m in victims)

    pool.stop("test")
    _until(pool, lambda: all(m.state == GONE for m in pool.members))
    assert all(b.state == "destroyed" for b in boxes.boxes.values())
    assert not any(_tagged(name) for name in boxes.boxes)
    report = {"parallel": parallel, "up_s": round(up_s, 1), "refill_s": round(refill_s, 1), "steady_tick_s": max(steady),
              "orders": len(pool.members)}
    print("DAYTONA16", json.dumps(report))
    assert max(steady) < 30, report
