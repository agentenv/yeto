"""Islands added and removed while a fleet trains, end to end: the real Rust syncer,
the trainer worker reconciling the desired state on a local store, fake trainers (one
syncer client per island launch; no cloud, no Ray).

A two-island fleet trains. ``yeto fleet island add`` (its functions) writes a third
island and its pool into the desired state: the worker starts its launch under learner
id 2 (claimed in the store), the island catches up to the newest outer weights and
merges. ``remove`` then takes island 1 and its pool out: the worker writes island 1's
STOP and its launch ends, and the rounds merge over islands 0 and 2 with renormalized
weights. Island 0 runs one launch from start to end. The final cut equals the f32
Nesterov recomputation of every merged round (1e-6).

Two ways island 1 ends: it answers its round and leaves (LEAVE stopped), or, as the
island STOP does today (the launch tears its machines down), it is gone without a
LEAVE: the round it was asked for waits one island lease from the disconnect (6 s
here), then the syncer releases its slot and the round merges without it, well before
the soft deadline (60 s here).
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from test_fleet_island_launches import example
from test_fleet_trainer_worker import FakeLaunches
from test_rl_full_parameter_async import _f32, _initial, _start_async_syncer, _wait, needs_cargo
from test_rl_full_parameter_async import syncer_binary  # noqa: F401 - fixture
from test_rl_full_parameter_dynamic_members import (
    LAUNCHED,
    MAX_LEARNERS,
    P,
    POOL,
    _client,
    _delta,
    _nesterov,
    _outer_steps,
    _values,
)
from yeto.launch import island_lifecycle as lifecycle
from yeto.launch.learner_ids import allocate_learner_id, claim_key
from yeto.launch.pool import desired as dz
from yeto.launch.pool.fleet_cli import changes
from yeto.rl import fleet, fleet_islands, island_launches, trainer_worker
from yeto.rl.engine.object_store import DirectoryObjectStore


# Without a LEAVE: a short lease (the islands left renew theirs while a round waits) and a
# soft deadline far past it, so the round the torn-down island never answers is released
# by its lease, not cut by the deadline.
NO_LEAVE_LEASE_S = 6
NO_LEAVE_POOL = tuple({"5": str(NO_LEAVE_LEASE_S), "30": "60"}.get(a, a) for a in POOL)


class SyncerTrainers(FakeLaunches):
    """Each island launch is a syncer client of its learner id; a launch ends when its
    island STOP is seen (LEAVE stopped), as the launch's fleet controller ends it."""

    def __init__(self, port: int, profile_hash: str, states: lifecycle.IslandStateStore, *,
                 leave: bool = True) -> None:
        super().__init__()
        self.port, self.profile_hash, self.states, self.leave = port, profile_hash, states, leave
        self.stopped_at: float | None = None
        self.unanswered: list = []
        self.clients: dict[int, object] = {}
        self.prefix_of: dict[int, str] = {}
        self.catching: dict[int, dict] = {}
        self.left: list[int] = []

    def start(self, prefix: str, argv: list[str]) -> int:
        pid = super().start(prefix, argv)
        k = int(argv[argv.index("--rl-fleet-learner-id") + 1])
        assert argv[argv.index("--rl-fleet-syncer") + 1] == f"127.0.0.1:{self.port}"
        client = _client(k, self.port, self.profile_hash)
        client.start()
        self.clients[k], self.prefix_of[k] = client, prefix
        if k >= LAUNCHED:
            self.catching[k] = {}
        return pid

    def stop_requested(self, finish) -> None:
        for k in list(self.clients):
            if self.states.stop_requested(k) is not None:
                if self.leave:
                    finish(k)  # it answers the round it holds a PULL of, then leaves
                else:  # torn down while a round waits for it: it holds a PULL it never answers
                    held = self.clients[k].drain_pulls()
                    if not held:
                        continue
                    self.unanswered = held
                client = self.clients.pop(k)
                if self.leave:
                    client.send_leave(2)  # stopped: the island leaves the pool on purpose
                client.close()  # without LEAVE: its machines torn down, the connection just drops
                self.stopped_at = time.monotonic()
                self.left.append(k)
                self.end(self.prefix_of[k], 0)


@needs_cargo
@pytest.mark.parametrize("leave", [True, False], ids=["leave", "torn-down-without-leave"])
def test_islands_join_and_leave_a_training_fleet_through_the_desired_state(syncer_binary, tmp_path,  # noqa: F811
                                                                           monkeypatch, leave):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "async-rounds-test-key")
    cfg = example(tmp_path, max_islands=MAX_LEARNERS)
    plan = fleet.plan_of(cfg)
    objects = DirectoryObjectStore(tmp_path / "store")
    store = dz.DesiredStore(objects)
    members = island_launches.trainer_members(cfg, plan)
    seed = [*members, *({"kind": "sampler_pool", "name": p.name, "island": i.index, "pool": {"provider": "modal"}}
                        for i in plan.islands for p in i.pools)]
    store.apply({"members": seed}, expect=0)

    total_steps, join_at = 10, 3
    log_file = tmp_path / "syncer.log"
    pool = POOL if leave else NO_LEAVE_POOL
    process, port, profile_hash = _start_async_syncer(syncer_binary, tmp_path, *pool, total_steps=total_steps,
                                                      log_file=log_file)
    island_launches.write_syncer_record(objects, addr=f"127.0.0.1:{port}", syncer_epoch=0, learners=LAUNCHED,
                                        max_learners=MAX_LEARNERS, run="r")
    states = lifecycle.IslandStateStore(objects, log=lambda m: None)
    launches = SyncerTrainers(port, profile_hash, states, leave=leave)
    worker = trainer_worker.TrainerWorker(store, objects, launches, states, log=lambda m: None)
    versions = {k: [0] * P for k in range(MAX_LEARNERS)}
    seen: dict[int, set] = {k: set() for k in range(MAX_LEARNERS)}
    held: dict[int, list] = {}
    state = {"added": None, "removed": False, "catch_up": {}, "pools_after": None}
    beat = [0.0]

    def drain(k):
        client = launches.clients[k]
        client.check_health()
        if k in launches.catching:  # a joiner applies its catch-up before it answers a PULL
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
        for permit in held.pop(k, []) + client.drain_pulls():
            key = (permit.global_step, permit.round_attempt)
            if key in seen[k]:
                continue
            seen[k].add(key)
            base = versions[k][permit.fragment_id]
            client.push_fragment(permit.fragment_id, permit.global_step, permit.round_attempt, base,
                                 permit.global_step, 1, 7, _f32(_delta(k, permit.global_step)))

    def add_island():
        k = allocate_learner_id(objects, launched=LAUNCHED, max_learners=MAX_LEARNERS, island="north")
        entry = dict(cfg["islands"][1]["samplers"][0], name="sglang-north")
        member, pools = fleet_islands.compile_island(cfg, plan, k=k, name="north", gpu="modal:1x2xh100",
                                                     usd_per_h=8.0, samplers=[entry])
        before = store.read()
        doc = store.update(lambda d: fleet_islands.add(d, member, pools), note="add north")
        state["added"] = (k, list(versions[0]), changes(before, doc))

    def remove_island():
        before = store.read()
        doc = store.update(lambda d: fleet_islands.remove(d, "sglang"), note="remove island 1")
        state["removed"] = True
        state["pools_after"] = (sorted(dz.sampler_pools(doc)), changes(before, doc))

    def poll():
        if not leave and time.monotonic() - beat[0] > 1:  # the islands left renew their short lease
            beat[0] = time.monotonic()
            for k, client in launches.clients.items():
                client.heartbeat(0)
        worker.step()
        # a joiner catches up before the fleet moves on (as the dynamic-member test does): the
        # recomputation below has no carry-over of a PULL the joiner answered late
        for k in list(launches.catching) or list(launches.clients):
            if not leave and states.stop_requested(k) is not None:
                continue  # its machines are being torn down: it answers nothing more
            drain(k)
        if launches.catching:
            return False
        launches.stop_requested(drain)
        if state["added"] is None and launches.clients and min(versions[0]) >= join_at:
            add_island()
        if state["added"] is not None and not state["removed"] and 2 in launches.clients \
                and 2 not in launches.catching:
            remove_island()
        return (state["removed"] and 1 not in launches.clients
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
        process.wait(timeout=10)

    # the desired state drove both changes; the pools moved with their islands
    k, joined_at, added_lines = state["added"]
    assert k == 2 and objects.read(claim_key(2)) is not None
    assert "add pool i2-sglang-north (target 2)" in added_lines and "trainer north: added (reconciled by the trainer " \
        "worker)" in added_lines
    pools, removed_lines = state["pools_after"]
    assert pools == ["i0-sglang-l40s", "i0-vllm-h100", "i2-sglang-north"]
    assert any(line.startswith("remove pool i1-sglang-l40s") for line in removed_lines)
    # island 0 ran one launch throughout; island 1 stopped on its STOP and is removed
    assert launches.started("r-i0") == 1 and launches.started("r-i2") == 1 and launches.left == [1]
    worker.step()
    assert worker.load(1)["state"] == trainer_worker.REMOVED and worker.load(0)["state"] == trainer_worker.RUNNING

    steps = _outer_steps((tmp_path / "tape.jsonl").read_text())
    assert [r["step"] for r in steps] == list(range(1, total_steps * P + 1))
    for fid, (version, values) in state["catch_up"].items():  # the joiner started from the newest weights
        assert version >= max(joined_at[fid], join_at)
        np.testing.assert_allclose(values, _nesterov(steps, upto=version)[fid], rtol=0, atol=1e-6)
    assert any(int(e[0]) == 2 for r in steps for e in r["raw_weights"])
    tape = (tmp_path / "tape.jsonl").read_text().splitlines()
    at_leave = next(i for i, line in enumerate(tape) if '"pool_leave"' in line and '"island_id":1' in line)
    after = [json.loads(line) for line in tape[at_leave:] if '"kind":"outer_step"' in line][2 * P:]
    assert after and all(int(e[0]) in (0, 2) for r in after for e in r["raw_weights"])
    events = [json.loads(line) for line in tape]
    (gone,) = [e for e in events if e.get("kind") == "pool_leave" and int(e.get("island_id", -1)) == 1]
    stalled = [r for r in steps if r["timed_out"]]
    for record in steps:
        assert abs(sum(float(e[2]) for e in record["weights"]) - 1.0) < 1e-9
    if leave:
        assert gone["reason"] == "stopped" and stalled == []
    else:
        # torn down holding a PULL (no LEAVE): the syncer sees the connection drop; the round
        # it asked island 1 for waits one lease from the disconnect, then merges over the
        # islands left: the merge stall a remove or move costs, never the soft deadline
        assert gone["reason"] == "disconnected" and launches.unanswered and stalled == []
        asked = {p.global_step for p in launches.unanswered}
        by_step = {r["step"]: r for r in steps}
        for step in asked:
            record = by_step[step]
            assert 1 not in [int(e[0]) for e in record["raw_weights"]]
            assert NO_LEAVE_LEASE_S - 1 <= record["step_seconds"] < NO_LEAVE_LEASE_S + 10
    expected = _nesterov(steps)
    assert finals.keys() == {0, 2}
    for fid in range(P):
        assert np.array_equal(finals[0][fid], finals[2][fid])
        np.testing.assert_allclose(finals[0][fid], expected[fid], rtol=0, atol=1e-6)
