"""S27 E-4c: a killed full-parameter island process restarts and re-joins a
running asynchronous pool. Real Rust syncer, no Ray.

The restarted process dials with a new connection generation. The syncer
admits it (superseding or re-joining under the membership rules) and resends
every fragment at its current version; the island takes those values as its
anchors, keeps training, and acknowledges the final cut.
"""

from __future__ import annotations

import numpy as np
import pytest

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from test_rl_full_parameter_async import (  # noqa: F401
    P,
    NUMEL,
    _client,
    _delta,
    _f32,
    _hmac_key,
    _initial,
    _start_async_syncer,
    _wait,
    needs_cargo,
    syncer_binary,
)

LEASE = ("--rl-sync-quorum", "1", "--rl-sync-island-lease-s", "3")


def _run_restart(binary, tmp_path, *, total_steps: int, kill_after: int):
    """Learner 1 dies once fragment versions reach ``kill_after``; a new
    process for learner 1 joins, catches up and both islands finish."""

    process, port, profile_hash = _start_async_syncer(binary, tmp_path, *LEASE,
                                                      total_steps=total_steps)
    clients = {0: _client(0, port, profile_hash), 1: _client(1, port, profile_hash)}
    versions = {0: [0] * P, 1: [0] * P}
    seen = {0: set(), 1: set()}
    restarted = {"done": False, "catch_up": None, "pushed_after": 0}

    def drain(learner):
        client = clients[learner]
        client.check_health()
        for update in client.drain_updates():
            fid = update.fragment_id
            versions[learner][fid] = max(versions[learner][fid], update.version)
        for permit in client.drain_pulls():
            key = (permit.global_step, permit.round_attempt)
            if key in seen[learner]:
                continue
            seen[learner].add(key)
            base = versions[learner][permit.fragment_id]
            client.push_fragment(permit.fragment_id, permit.global_step, permit.round_attempt,
                                 base, permit.global_step, 1, 7,
                                 _f32(_delta(learner, permit.global_step)))
            if learner == 1 and restarted["done"]:
                restarted["pushed_after"] += 1

    def restart():
        clients[1].close()  # the process dies: its sockets close, no LEAVE
        fresh = _client(1, port, profile_hash)
        fresh.start()
        caught: dict[int, int] = {}

        def catch_up():
            fresh.check_health()
            for update in fresh.drain_updates():
                caught[update.fragment_id] = max(caught.get(update.fragment_id, 0), update.version)
            return len(caught) == P

        _wait(catch_up, "restarted island got no catch-up", timeout=30)
        clients[1] = fresh
        versions[1] = [caught[f] for f in range(P)]
        seen[1].clear()
        restarted.update(done=True, catch_up=dict(caught))

    def poll():
        drain(0)
        if not restarted["done"] and min(versions[0]) >= kill_after:
            restart()
        drain(1)
        return all(client.finalizing.is_set() for client in clients.values())

    try:
        for client in clients.values():
            client.start()
        for fid in range(P):
            clients[0].send_init(fid, _f32(_initial(fid)))
        _wait(poll, "syncer never finalized", timeout=90)
        finals = []
        for client in clients.values():
            manifest, fragments = client.wait_for_final_fragments(timeout=30)
            finals.append({f.fragment_id: np.frombuffer(bytes(f.data), dtype=np.float32)[-NUMEL:]
                           for f in fragments})
            client.acknowledge_finalization(manifest)
        assert process.wait(timeout=15) == 0
    finally:
        for client in clients.values():
            client.close()
        if process.poll() is None:
            process.terminate()
        log, _ = process.communicate(timeout=10)
    for fid in range(P):
        assert np.array_equal(finals[0][fid], finals[1][fid])
    return restarted, log, (tmp_path / "tape.jsonl").read_text()


@needs_cargo
@pytest.mark.usefixtures("_hmac_key")
def test_a_restarted_island_catches_up_and_finishes(syncer_binary, tmp_path):
    restarted, log, tape = _run_restart(syncer_binary, tmp_path, total_steps=6, kill_after=2)
    assert restarted["done"]
    assert min(restarted["catch_up"].values()) >= 2
    assert restarted["pushed_after"] > 0, "the restarted island never contributed"
    assert '"catch_up":true' in tape


@needs_cargo
@pytest.mark.usefixtures("_hmac_key")
def test_heartbeats_keep_the_lease_while_the_pool_waits_for_init(syncer_binary, tmp_path):
    """s27-x2-pause-a: an island that connected first lost its lease while
    the other island was still starting (no INIT yet), because the syncer
    ignored heartbeats before the initial cut. Here the INIT comes after
    2.5 leases; the first island's heartbeats keep it in the pool."""
    import time

    from yeto.rl.pool_client import LeaseHeartbeat

    lease_s = 3.0
    log_file = tmp_path / "syncer.log"
    process, port, profile_hash = _start_async_syncer(syncer_binary, tmp_path, *LEASE,
                                                      total_steps=2, log_file=log_file)
    early = _client(1, port, profile_hash)
    late = _client(0, port, profile_hash)
    heartbeat = LeaseHeartbeat(early, lease_s)
    try:
        early.start()
        heartbeat.start()
        time.sleep(2.5 * lease_s)  # island 0 warms up its engine
        late.start()
        for fid in range(P):
            late.send_init(fid, _f32(_initial(fid)))
        _wait(lambda: "global parameters initialized" in log_file.read_text(), "no INIT")
        time.sleep(0.5)
        log = log_file.read_text()
        assert "lease expired" not in log, log
        assert heartbeat.errors == []
    finally:
        heartbeat.stop()
        early.close()
        late.close()
        process.terminate()
        process.wait(timeout=10)
