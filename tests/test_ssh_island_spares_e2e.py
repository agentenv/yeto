"""A resident ssh: trainer island replaces a lost node while the fleet trains, end to end
(own-cluster-fleet-stability 4.3): the real Rust syncer, a local store, the fleet
controller over SshIslandOps and the fake SSH cluster, fake trainers (rank 0 of each
incarnation is one syncer client of the island's learner id; no Ray, no GPU).

Two islands train: island 0 is a plain syncer client of learner 0 for the whole run;
island 1 is a two-node ssh: island (learner 1) whose inventory holds one spare node.
Its rank 0 node loses power while it holds a PULL (no LEAVE: the machine is just gone).
The controller judges the node lost (its stream ended, no SSH answer twice), the island
is stopped on its other node, the spare takes rank 0's place and the next incarnation
resumes from the newest checkpoint of its store under the same learner id: to the syncer
a short absence, then a re-JOIN that catches up to the newest outer weights before it
merges again. As in the move test (test_fleet_island_move_e2e) the replacement starts
after island 1's lease, so the round it held merges without it, and island 0 does not run
ahead of the rounds open at the loss while island 1 is away.

Checked: the merge sequence has every step once (1..N, none twice), no fragment is pushed
twice, the resume step is the newest checkpoint's step, the final weights and the
catch-up equal the f32 Nesterov recomputation (1e-6), and island 0 runs on one
connection from start to end."""

from __future__ import annotations

import json
import secrets
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pytest

from test_rl_full_parameter_async import _f32, _initial, _start_async_syncer, _wait, needs_cargo
from test_rl_full_parameter_async import syncer_binary  # noqa: F401 - fixture
from test_rl_full_parameter_dynamic_members import MAX_LEARNERS, P, POOL, _client, _delta, _nesterov, _outer_steps, _values
from test_rl_ssh_resident import Clock, _SyncThread, _cards, b300, make_ops, spec  # noqa: F401 - _cards: autouse
from test_ssh_island_spares import Cluster, FleetLog, _health, spare  # noqa: F401 - _health: autouse
from yeto.launch import island_spares as isp
from yeto.launch.fleet import RUNNING, FleetController
from yeto.launch.island_spares import IslandSpares, NodeRecords, latest_checkpoint_step
from yeto.modal_runner import RoutingOps
from yeto.rl.engine.object_store import DirectoryObjectStore

LEASE_S = float(POOL[POOL.index("--rl-sync-island-lease-s") + 1])  # 5 s
TOLERANCE_S = 1.5
RETURN_AFTER_S = LEASE_S + 3  # the controller sees the loss (and replaces) this long after it
# where the trainer of learner 1 (samplers following it) keeps its newest release: its own
# trainer1/ sub-store; the launcher finds it through resume_keys of the launch's args
CHECKPOINT = "trainer1/policy-releases/LATEST.json"
ARGS = SimpleNamespace(rl_sample_role="trainer", rl_single_island_no_sync=False, rl_island_sample_role=None,
                       rl_sampler_follow="trainer", rl_backend="miles", rl_max_policy_age=2,
                       rl_checkpoint_store="/store", rl_island_buffer_shards=["0:a", "1:b"],
                       rl_island_release_every=None, rl_island_max_policy_age=None)


def _own_checkpoint(store):
    """The fake trainer's resume: its own release, read as it writes it (not via the launcher)."""
    raw = store.read(CHECKPOINT)
    return None if raw is None else json.loads(raw)["version"]


class Sky:
    def __init__(self, clock):
        self.clock = clock

    def now(self):
        return self.clock.t

    def sleep(self, s):
        self.clock.t += s


class TrainerCluster(Cluster):
    """The fake SSH cluster whose rank 0 start of island 1 is a fake trainer: a syncer
    client of learner 1 that resumes from its store's newest checkpoint; its stop (or its
    node's power loss) closes the client without a LEAVE."""

    def __init__(self, names, *, port, profile_hash, store):
        super().__init__(names)
        self.port, self.profile_hash, self.store = port, profile_hash, store
        self.trainers: dict[int, dict] = {}  # incarnation -> {client, node, resume, catching}
        self.current: int | None = None

    def start(self, node, script):
        stream = super().start(node, script)
        if "setsid bash -c" in script and "export SKYPILOT_NODE_RANK=0\n" in script:
            k = self._k(script)
            checkpoint = _own_checkpoint(self.store)
            client = _client(1, self.port, self.profile_hash)
            client.start()
            self.trainers[k] = {"client": client, "node": node.name,
                                "resume": checkpoint,
                                "catching": {} if k > 0 else None, "closed": False}
            self.current = k
        return stream

    def run(self, node, script, *, timeout=300.0):
        done = super().run(node, script, timeout=timeout)
        if "kill -TERM" in script and done.returncode == 0:
            k = self._k(script)
            trainer = self.trainers.get(k)
            if trainer is not None and trainer["node"] == node.name:
                self.close_trainer(k)
        return done

    def close_trainer(self, k):
        trainer = self.trainers[k]
        if not trainer["closed"]:
            trainer["closed"] = True
            trainer["client"].close()  # torn down / power lost: the connection just drops

    def live(self):
        k = self.current
        if k is None or self.trainers[k]["closed"]:
            return None
        return self.trainers[k]


@needs_cargo
def test_an_island_replaces_a_lost_node_and_the_fleet_merges_on(syncer_binary, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", f"spare-test-key-{secrets.token_hex(6)}")
    total_steps, lose_at = 10, 3
    process, port, profile_hash = _start_async_syncer(syncer_binary, tmp_path, *POOL, total_steps=total_steps,
                                                      log_file=tmp_path / "syncer.log")
    store = DirectoryObjectStore(tmp_path / "store")  # island 1's run store: checkpoints, node records
    names = ["b300-0", "b300-1", "b300-3"]
    cluster = TrainerCluster(names, port=port, profile_hash=profile_hash, store=store)
    clock = Clock()
    ops = make_ops(tmp_path, cluster, clock=clock, health_every_s=0, unreachable_polls=2)
    island = spec(2)
    spares = IslandSpares(pool=(*island.nodes, spare(3)), records=NodeRecords(store, run_id="run1", log=lambda m: None),
                          resume_probe=lambda: latest_checkpoint_step(store, isp.resume_keys(ARGS, 1, 2)))
    ops.add("isl1", island, spares=spares)
    log = FleetLog()
    ctl = FleetController(learners={"isl1": (island, 0)}, syncer=None, sky_ops=RoutingOps(Sky(clock), None, ssh_ops=ops),
                          poll_interval=30.0, recover_timeout=600.0, thread_cls=_SyncThread, fixed_roster=False,
                          fleet_log=log)
    rec = ctl.learners["isl1"]

    island0 = _client(0, port, profile_hash)
    generation0 = None
    versions = {k: [0] * P for k in range(MAX_LEARNERS)}
    pushed: list[tuple[int, int, int]] = []  # (learner, fragment, global step) of every push
    seen: dict[int, set] = {0: set(), 1: set()}  # island 1's survive its incarnations (its checkpoint)
    held: dict[int, list] = {0: [], 1: []}
    state = {"lost_at": None, "unanswered": [], "catch_up": {}, "checkpoints": [], "back": False}
    beat = [0.0]
    polled = [0.0]

    def away() -> bool:
        return state["lost_at"] is not None and not state["back"]

    def checkpoint(step: int) -> None:
        if not state["checkpoints"] or state["checkpoints"][-1] < step:
            state["checkpoints"].append(step)
            store.write(CHECKPOINT, json.dumps({"seq": len(state["checkpoints"]), "version": step,
                                                "path": f"r{step:08d}-v{step}"}).encode())

    def drain(k, client, trainer=None):
        client.check_health()
        if trainer is not None and trainer["catching"] is not None:  # the re-JOIN applies its catch-up first
            caught = trainer["catching"]
            for update in client.drain_updates():
                caught.setdefault(update.fragment_id, (update.version, _values(update.data)))
            held[k].extend(client.drain_pulls())
            if len(caught) < P:
                return
            versions[k] = [caught[fid][0] for fid in range(P)]
            state["catch_up"] = dict(caught)
            trainer["catching"] = None
            state["back"] = True
        for update in client.drain_updates():
            versions[k][update.fragment_id] = max(versions[k][update.fragment_id], update.version)
        if k == 1:
            checkpoint(min(versions[1]))  # the trainer's newest checkpoint follows its outer weights
        later = []
        for permit in held[k] + client.drain_pulls():
            if k == 0 and away() and state["unanswered"] and \
                    permit.global_step > max(p.global_step for p in state["unanswered"]):
                later.append(permit)  # island 0 does not run ahead while island 1 is away
                continue
            key = (permit.global_step, permit.round_attempt)
            if key in seen[k]:
                continue
            if k == 1 and state["lost_at"] is None and min(versions[0]) >= lose_at:
                # island 1 holds this PULL when its rank 0 node loses power
                state["unanswered"] = [permit]
                state["lost_at"] = time.monotonic()
                state["newest_at_loss"] = state["checkpoints"][-1] if state["checkpoints"] else None
                cluster.close_trainer(cluster.current)
                cluster.lose(trainer["node"])
                return
            seen[k].add(key)
            base = versions[k][permit.fragment_id]
            client.push_fragment(permit.fragment_id, permit.global_step, permit.round_attempt, base,
                                 permit.global_step, 1, 7, _f32(_delta(k, permit.global_step)))
            pushed.append((k, permit.fragment_id, permit.global_step))
        held[k] = later

    def poll():
        nonlocal generation0
        if time.monotonic() - beat[0] > 1:  # the islands renew their leases (5 s) while they wait
            beat[0] = time.monotonic()
            island0.heartbeat(0)
            if cluster.live() is not None:
                cluster.live()["client"].heartbeat(0)
        if generation0 is None:
            generation0 = island0.connection_generation
        drain(0, island0)
        trainer = cluster.live()
        if trainer is not None:
            drain(1, trainer["client"], trainer)
        if (state["lost_at"] is not None and time.monotonic() - state["lost_at"] >= RETURN_AFTER_S
                and time.monotonic() - polled[0] > 0.1):
            polled[0] = time.monotonic()
            clock.t += 30
            ctl._poll_recorded(rec)  # the fleet controller's poll of island 1
        trainer = cluster.live()
        return (state["back"] and trainer is not None and island0.finalizing.is_set()
                and trainer["client"].finalizing.is_set())

    finals = {}
    try:
        island0.start()
        assert ops.start("isl1") == 0
        for fid in range(P):
            island0.send_init(fid, _f32(_initial(fid)))
        _wait(poll, "the fleet never finalized", timeout=180)
        for k, client in ((0, island0), (1, cluster.live()["client"])):
            manifest, fragments = client.wait_for_final_fragments(timeout=30)
            finals[k] = {f.fragment_id: _values(f.data) for f in fragments}
            client.acknowledge_finalization(manifest)
        assert process.wait(timeout=15) == 0
    finally:
        island0.close()
        for k in list(cluster.trainers):
            cluster.close_trainer(k)
        ops.close()
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
            raise

    # the controller replaced the lost rank 0 node by the spare: same learner id, same layout
    assert rec["state"] == RUNNING and rec["job_id"] == 1 and rec["relaunches"] == 1
    assert [n.name for n in ops.islands["isl1"].spec.nodes] == ["b300-3", "b300-1"]
    assert cluster.trainers[1]["node"] == "b300-3"
    rows = {kind: fields for kind, _, fields in log.rows if kind.startswith("island_node")}
    assert rows["island_node_lost"]["node"] == "b300-0" and rows["island_node_lost"]["rank"] == 0
    replaced = rows["island_node_replaced"]
    assert (replaced["lost"], replaced["spare"], replaced["rank"], replaced["incarnation"]) == ("b300-0", "b300-3", 0, 1)
    assert json.loads(store.read("fleet/nodes/b300-0/quarantine.json"))["island"] == "isl1"
    assert ("kill", "b300-1", 0) in cluster.calls  # the island was stopped on its other node
    assert ("unanswered", "b300-0", "kill") not in cluster.calls  # the lost node was not waited for

    # it resumed from the newest checkpoint of its store, and the event says which
    newest = state["newest_at_loss"]
    assert cluster.trainers[1]["resume"] == replaced["resume_step"] == newest
    assert replaced["resume"] == {"source": "release", "step": newest, "ref": f"r{newest:08d}-v{newest}",
                                  "key": CHECKPOINT}
    assert newest >= lose_at - 1

    # the merge sequence: every global step of every fragment once, no fragment pushed twice
    tape_text = (tmp_path / "tape.jsonl").read_text()
    tape = tape_text.splitlines()
    events = [json.loads(line) for line in tape]
    steps = _outer_steps(tape_text)
    assert [r["step"] for r in steps] == list(range(1, total_steps * P + 1))
    assert len(pushed) == len(set(pushed))
    for record in steps:
        learners = [int(e[0]) for e in record["raw_weights"]]
        assert len(learners) == len(set(learners)), record
        assert abs(sum(float(e[2]) for e in record["weights"]) - 1.0) < 1e-9
    assert {int(e[0]) for r in steps for e in r["raw_weights"]} == {0, 1}  # no third learner id
    assert [r for r in steps if r["timed_out"]] == []

    # the syncer saw island 1 go (no LEAVE: its machine was gone) and come back under its id
    (gone,) = [e for e in events if e.get("kind") == "pool_leave"]
    assert int(gone["island_id"]) == 1 and gone["reason"] == "disconnected"
    at_leave = next(i for i, line in enumerate(tape) if '"pool_leave"' in line)
    later = [json.loads(line) for line in tape[at_leave:] if '"kind":"outer_step"' in line]
    assert any(int(e[0]) == 1 for r in later for e in r["raw_weights"]), "island 1 never merged again"
    record = {(r["step"], r["fragment"]): r for r in steps}
    for permit in state["unanswered"]:  # the round it held waited one lease, then merged without it
        r = record[(permit.global_step, permit.fragment_id)]
        waited = r["time_unix"] - gone["time_unix"]
        assert LEASE_S - 0.5 <= waited <= LEASE_S + TOLERANCE_S, (waited, r)
        assert 1 not in {int(e[0]) for e in r["raw_weights"]}

    # island 0 was never interrupted: one connection, a push in every merged round
    assert island0.connection_generation == generation0
    assert all(0 in {int(e[0]) for e in r["raw_weights"]} for r in steps)

    # the catch-up of the re-JOIN and the final weights: the f32 Nesterov recomputation
    assert sorted(state["catch_up"]) == list(range(P))
    for fid, (version, values) in state["catch_up"].items():
        np.testing.assert_allclose(values, _nesterov(steps, upto=version)[fid], rtol=0, atol=1e-6)
    expected = _nesterov(steps)
    assert finals.keys() == {0, 1}
    for fid in range(P):
        assert np.array_equal(finals[0][fid], finals[1][fid])
        np.testing.assert_allclose(finals[0][fid], expected[fid], rtol=0, atol=1e-6)
    assert isp.NODE_REPLACED in {k for k, _, _ in log.rows}
