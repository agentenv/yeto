"""An island moved to another source while the fleet trains, end to end: the real Rust
syncer, the trainer worker reconciling the desired state on a local store, fake trainers
(one syncer client per island launch; no cloud, no Ray).

A two-island fleet trains. ``yeto fleet island move`` (its function) changes island 1's
``--gpu`` from Modal to an AWS (sky) cluster in the desired state. The worker ends its
launch with its island STOP and starts the next incarnation (``<prefix>-m1``) on the new
machines under the **same learner id**. To the syncer this is a short absence, not a
removal: the moved island comes back as learner 1, catches up to the newest outer
weights and merges again. Island 0 runs one launch from start to end and no round times
out. With a LEAVE the rounds never wait for the absent island (the moved incarnation starts
once the syncer has handled that LEAVE, as on real machines that take minutes to come up). Without one (the move tears
the old machines down while island 1 holds a PULL) the moved incarnation starts only after
island 1's lease: the round it held waits that one lease (the syncer releases the slot of
an island gone a lease, #788), merges without it, and island 0 goes on. The final weights
equal the f32 Nesterov recomputation of every merged round (1e-6).
"""

from __future__ import annotations

import json
import subprocess
import time

import numpy as np
import pytest

from test_fleet_island_e2e import SyncerTrainers
from test_fleet_island_launches import example
from test_rl_full_parameter_async import _f32, _initial, _start_async_syncer, _wait, needs_cargo
from test_rl_full_parameter_async import syncer_binary  # noqa: F401 - fixture
from test_rl_full_parameter_dynamic_members import (
    LAUNCHED,
    MAX_LEARNERS,
    P,
    POOL,
    _delta,
    _nesterov,
    _outer_steps,
    _values,
)
from yeto.launch import island_lifecycle as lifecycle
from yeto.launch.pool import desired as dz
from yeto.rl import fleet, fleet_islands, island_launches, trainer_worker
from yeto.rl.engine.object_store import DirectoryObjectStore

MOVED_TO = "aws:2xh100@us-east-2"
LEASE_S = float(POOL[POOL.index("--rl-sync-island-lease-s") + 1])  # 5 s
SOFT_DEADLINE_S = float(POOL[POOL.index("--rl-sync-soft-deadline-s") + 1])  # 30 s: far above the lease
TOLERANCE_S = 1.5
RETURN_AFTER_S = LEASE_S + 3  # the moved incarnation starts this long after the teardown


class MovingTrainers(SyncerTrainers):
    """A launch that is a later incarnation of an island (``-m<n>``) catches up first,
    as a re-JOIN does (the syncer pushes it the newest outer weights)."""

    def start(self, prefix: str, argv: list[str]) -> int:
        pid = super().start(prefix, argv)
        k = int(argv[argv.index("--rl-fleet-learner-id") + 1])
        if "-m" in prefix:  # island_launches.island_prefix of incarnation >= 1
            self.catching[k] = {}
        return pid


@needs_cargo
@pytest.mark.parametrize("leave", [True, False], ids=["leave", "torn-down-without-leave"])
def test_a_moved_island_comes_back_under_its_learner_id(syncer_binary, tmp_path, monkeypatch, leave):  # noqa: F811
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "async-rounds-test-key")
    cfg = example(tmp_path, max_islands=MAX_LEARNERS)
    plan = fleet.plan_of(cfg)
    objects = DirectoryObjectStore(tmp_path / "store")
    store = dz.DesiredStore(objects)
    members = island_launches.trainer_members(cfg, plan)
    seed = [*members, *({"kind": "sampler_pool", "name": p.name, "island": i.index, "pool": {"provider": "modal"}}
                        for i in plan.islands for p in i.pools)]
    store.apply({"members": seed}, expect=0)
    before = dz.trainers(store.read())["sglang"]
    assert before["learner_id"] == 1 and before["gpu"].startswith("modal:")

    total_steps, move_at = 10, 3
    process, port, profile_hash = _start_async_syncer(syncer_binary, tmp_path, *POOL, total_steps=total_steps,
                                                      log_file=tmp_path / "syncer.log")
    island_launches.write_syncer_record(objects, addr=f"127.0.0.1:{port}", syncer_epoch=0, learners=LAUNCHED,
                                        max_learners=MAX_LEARNERS, run="r")
    states = lifecycle.IslandStateStore(objects, log=lambda m: None)
    launches = MovingTrainers(port, profile_hash, states, leave=leave)
    worker = trainer_worker.TrainerWorker(store, objects, launches, states, log=lambda m: None)
    versions = {k: [0] * P for k in range(MAX_LEARNERS)}
    seen: dict[int, set] = {k: set() for k in range(MAX_LEARNERS)}
    held: dict[int, list] = {}
    state = {"moved": None, "catch_up": {}}
    beat = [0.0]

    def back() -> bool:  # the moved incarnation is up and caught up
        return launches.started("r-i1-m1") == 1 and 1 in launches.clients and 1 not in launches.catching

    def away() -> bool:  # torn down without LEAVE, not back yet
        return not leave and launches.stopped_at is not None and not back()

    def left() -> bool:  # the syncer handled the departure of island 1's first launch
        tape = tmp_path / "tape.jsonl"
        return tape.exists() and any('"pool_leave"' in line and '"island_id":1,' in line
                                     for line in tape.read_text().splitlines())

    def returning() -> bool:  # may the worker start the moved incarnation yet?
        if launches.stopped_at is None or launches.started("r-i1-m1"):
            return True
        if leave:  # new machines take minutes: the old connection's LEAVE is handled first
            return left()
        return time.monotonic() - launches.stopped_at >= RETURN_AFTER_S  # back after its lease

    def drain(k):
        client = launches.clients[k]
        client.check_health()
        if k in launches.catching:  # the moved island applies its catch-up before it answers a PULL
            caught = launches.catching[k]
            for update in client.drain_updates():
                caught.setdefault(update.fragment_id, (update.version, _values(update.data)))
            held.setdefault(k, []).extend(client.drain_pulls())
            if len(caught) < P:
                return
            versions[k] = [caught[fid][0] for fid in range(P)]
            state["catch_up"] = dict(caught)
            del launches.catching[k]
        for update in client.drain_updates():
            versions[k][update.fragment_id] = max(versions[k][update.fragment_id], update.version)
        later = []
        for permit in held.pop(k, []) + client.drain_pulls():
            if k == 0 and away() and permit.global_step > max(p.global_step for p in launches.unanswered):
                later.append(permit)  # island 0 waits for island 1 to come back before it runs ahead
                continue
            key = (permit.global_step, permit.round_attempt)
            if key in seen[k]:
                continue
            seen[k].add(key)
            base = versions[k][permit.fragment_id]
            client.push_fragment(permit.fragment_id, permit.global_step, permit.round_attempt, base,
                                 permit.global_step, 1, 7, _f32(_delta(k, permit.global_step)))
        if later:
            held.setdefault(k, []).extend(later)

    def move_island():
        doc = store.update(lambda d: fleet_islands.move(d, cfg, "sglang", gpu=MOVED_TO, usd_per_h=6.0),
                           note="move island 1 to aws")
        state["moved"] = (list(versions[0]), dz.trainers(doc)["sglang"])

    def poll():
        if time.monotonic() - beat[0] > 1:  # the islands renew their leases (5 s) while they wait
            beat[0] = time.monotonic()
            for client in launches.clients.values():
                client.heartbeat(0)
        if returning():
            worker.step()
        for k in list(launches.catching) or list(launches.clients):
            if not leave and states.stop_requested(k) is not None:
                continue  # its old machines are being torn down: it answers nothing more
            drain(k)
        if launches.catching:
            return False
        launches.stop_requested(drain)
        if state["moved"] is None and len(launches.clients) == LAUNCHED and min(versions[0]) >= move_at:
            move_island()
        return (state["moved"] is not None and launches.started("r-i1-m1") == 1 and 1 in launches.clients
                and all(c.finalizing.is_set() for c in launches.clients.values()))

    finals = {}
    try:
        worker.step()  # islands 0 and 1 start
        assert sorted(launches.clients) == [0, 1]
        for fid in range(P):
            launches.clients[0].send_init(fid, _f32(_initial(fid)))
        _wait(poll, "the fleet never finalized", timeout=180)
        for k, client in launches.clients.items():
            manifest, fragments = client.wait_for_final_fragments(timeout=30)
            finals[k] = {f.fragment_id: _values(f.data) for f in fragments}
            client.acknowledge_finalization(manifest)
        assert process.wait(timeout=15) == 0
    finally:
        for client in launches.clients.values():
            client.close()
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
            raise

    # the move kept the learner id, its sub-store and its sampler pool; only the machines changed
    moved_at, after = state["moved"]
    assert after["learner_id"] == 1 and after["gpu"] == MOVED_TO and after["prefix"] == before["prefix"]
    assert after["pools"] == before["pools"] and after["argv"] == before["argv"]
    (argv,) = [a for p, a in launches.starts if p == "r-i1-m1"]
    assert argv[argv.index("--gpu") + 1] == MOVED_TO
    assert argv[argv.index("--rl-fleet-learner-id") + 1] == "1"
    assert argv[argv.index("--rl-island-buffer-shards") + 1] == before["argv"][
        before["argv"].index("--rl-island-buffer-shards") + 1]
    # island 0 ran one launch throughout; island 1 ran two: the Modal one and the moved one
    assert launches.started("r-i0") == 1 and launches.started("r-i1") == 1 and launches.left == [1]
    worker.step()
    assert worker.load(1)["state"] == trainer_worker.RUNNING and worker.load(0)["state"] == trainer_worker.RUNNING

    # the syncer saw a short absence of learner 1, then the same learner again
    tape = (tmp_path / "tape.jsonl").read_text().splitlines()
    events = [json.loads(line) for line in tape]
    (gone,) = [e for e in events if e.get("kind") == "pool_leave"]  # the one departure of the run
    assert int(gone["island_id"]) == 1 and gone["reason"] == ("stopped" if leave else "disconnected")
    at_leave = next(i for i, line in enumerate(tape) if '"pool_leave"' in line and '"island_id":1' in line)
    steps = _outer_steps((tmp_path / "tape.jsonl").read_text())
    assert [r["step"] for r in steps] == list(range(1, total_steps * P + 1))
    later = [json.loads(line) for line in tape[at_leave:] if '"kind":"outer_step"' in line]
    assert any(int(e[0]) == 1 for r in later for e in r["raw_weights"]), "the moved island never merged again"
    assert {int(e[0]) for r in steps for e in r["raw_weights"]} == {0, 1}  # no third learner id
    assert sorted(state["catch_up"]) == list(range(P))  # it re-JOINed and was caught up, every fragment
    for fid, (version, values) in state["catch_up"].items():  # it came back on the newest weights
        assert version >= max(moved_at[fid], move_at)
        np.testing.assert_allclose(values, _nesterov(steps, upto=version)[fid], rtol=0, atol=1e-6)
    assert [r for r in steps if r["timed_out"]] == []  # no round reached the soft deadline (30 s)
    opened = {(e["global_step"], e["fragment"]): e["time_unix"] for e in events if e.get("kind") == "round_open"}
    if leave:
        # it answered the round it held, then left: no round waited for it
        back_at = max(e["time_unix"] for e in events if e.get("kind") == "pool_join"
                      and int(e.get("island_id", -1)) == 1)  # the moved incarnation's join
        away_rounds = [r for r in steps if opened[(r["step"], r["fragment"])] <= back_at
                       and r["time_unix"] >= gone["time_unix"]]
        assert away_rounds, "no round was open while island 1 was away"
        for r in away_rounds:
            assert r["time_unix"] - opened[(r["step"], r["fragment"])] < LEASE_S, r
    else:
        # torn down holding a PULL, back only after its lease: the round it held waited one
        # lease from the disconnect (it did wait, > 0), then merged without it
        assert launches.unanswered
        record = {(r["step"], r["fragment"]): r for r in steps}
        for permit in launches.unanswered:
            r = record[(permit.global_step, permit.fragment_id)]
            waited = r["time_unix"] - gone["time_unix"]
            assert LEASE_S - 0.5 <= waited <= LEASE_S + TOLERANCE_S, (waited, r)
            assert r["time_unix"] - opened[(r["step"], r["fragment"])] >= LEASE_S - 0.5
            assert 1 not in {int(e[0]) for e in r["raw_weights"]}  # its slot was released, not answered
    for record in steps:
        assert abs(sum(float(e[2]) for e in record["weights"]) - 1.0) < 1e-9
    expected = _nesterov(steps)
    assert finals.keys() == {0, 1}
    for fid in range(P):
        assert np.array_equal(finals[0][fid], finals[1][fid])
        np.testing.assert_allclose(finals[0][fid], expected[fid], rtol=0, atol=1e-6)
