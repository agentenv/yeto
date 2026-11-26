"""SSH trainer islands that join, leave and lag a training fleet, end to end
(own-cluster-fleet-stability 9.1 and 9.2): the real Rust syncer, the trainer worker
reconciling the desired state on a local store, and for each island launch the launcher's
own fleet controller over the resident ``ssh:`` island ops (``yeto.launch.ssh_resident``) on
a fake SSH cluster. Rank 0 of each island is its trainer: its start on the fake nodes
connects a syncer client under the island's learner id, its kill drops the connection. No
cloud, no Ray, no GPU.

Two islands (``ssh:2x8xb300@bm``, two nodes each) train. Then, one case each:

* **add**: ``yeto fleet island add`` (its functions) writes a third island into the desired
  state; the worker starts its launch on two more nodes, the island catches up to the newest
  outer weights and its pushes are merged from then on.
* **remove**: ``remove`` takes island 1 out; the worker writes its island STOP, its
  controller drains it, releases its nodes (the kill on each node), and the rounds merge over
  island 0 alone.
* **stuck**: island 1's trainer stops pushing while it stays connected (heartbeats). The
  syncer merges island 0 at the quorum deadline and excuses island 1; island 1's controller
  writes ``island_straggler`` (action ``wait``) once three merge cycles went by without a push
  of it, nothing is stopped or relaunched, and once it pushes again its controller writes
  ``island_straggler_cleared``.

Each case checks that the other islands' processes never restarted (one start per node, no
kill before the end, one launch), that the merged steps run 1..N without a gap or a repeat,
that no push is merged twice, and that the resource ledger holds each SSH incarnation once.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter

import pytest

from test_fleet_island_launches import example_doc
from test_fleet_trainer_worker import FakeLaunches
from test_rl_full_parameter_async import _f32, _initial, _start_async_syncer, _wait, needs_cargo
from test_rl_full_parameter_async import syncer_binary  # noqa: F401 - fixture
from test_rl_full_parameter_dynamic_members import LAUNCHED, MAX_LEARNERS, P, POOL, _client, _delta, _outer_steps
from test_rl_ssh_resident import FakeCluster, b300
from yeto.cost.ledger import ResourceLedger
from yeto.hw import card_tools
from yeto.launch import island_lifecycle as lifecycle
from yeto.launch import island_straggler as st
from yeto.launch.fleet import FleetController
from yeto.launch.learner_ids import allocate_learner_id
from yeto.launch.pool import desired as dz
from yeto.launch.ssh_resident import SshIslandOps
from yeto.launch.ssh_trainer_island import SshTrainerSpec
from yeto.launch.ssh_trainer_ledger import KIND as SSH_TRAINER
from yeto.modal_runner import RoutingOps
from yeto.rl import fleet, fleet_islands, island_launches, trainer_worker
from yeto.rl import up
from yeto.rl.engine.object_store import DirectoryObjectStore

GPU = "ssh:2x8xb300@bm"
NODES = {k: (b300(2 * k), b300(2 * k + 1)) for k in range(MAX_LEARNERS)}  # island k on nodes 2k, 2k+1
CYCLES = 3
# the stuck case: a short soft deadline, so a round that waits for the silent island merges
# island 0's push after 2 s (the quorum deadline of an island pool, #865)
STUCK_POOL = tuple({"30": "2"}.get(a, a) for a in POOL)


@pytest.fixture(autouse=True)
def _ssh_nodes(monkeypatch, tmp_path):
    monkeypatch.setattr(card_tools, "check_cards", lambda family, text, **kw: [])
    monkeypatch.setenv("YETO_RESOURCE_LEDGER", str(tmp_path / "ledger.jsonl"))
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "ssh-island-fleet-test-key")


class TrainerCluster(FakeCluster):
    """The fake nodes of every island; rank 0's start and kill are its trainer's."""

    def __init__(self) -> None:
        super().__init__([n.name for nodes in NODES.values() for n in nodes])
        self.rank0 = {nodes[0].name: k for k, nodes in NODES.items()}
        self.trainer_starts: list[int] = []
        self.trainer_kills: list[int] = []
        self.guard = threading.Lock()

    def start(self, node, script):
        stream = super().start(node, script)
        if "setsid bash -c" in script and node.name in self.rank0:
            with self.guard:
                self.trainer_starts.append(self.rank0[node.name])
        return stream

    def run(self, node, script, *, timeout=300.0):
        out = super().run(node, script, timeout=timeout)
        if "kill -TERM" in script and node.name in self.rank0:
            with self.guard:
                self.trainer_kills.append(self.rank0[node.name])
        return out

    def finish(self, k: int) -> None:
        """Island k's trainer ended its run: every rank exits 0 (rank 0 last)."""
        for node in reversed(NODES[k]):
            if self.runs[node.name].get(0, {}).get("state") == "running":
                self.exit(node.name, 0, 0)


class FastClock:
    """The island controller's sky side: fake time, 30 s per poll in 10 ms."""

    def __init__(self) -> None:
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, seconds):
        time.sleep(0.01)
        self.t += seconds


class FleetLog:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict]] = []

    def event(self, kind, name, **fields):
        self.events.append((kind, name, fields))

    def maybe_tick(self):
        pass

    def cost_tick(self):
        pass

    def kinds(self, kind):
        return [e for e in list(self.events) if e[0] == kind]


class _SyncThread:
    def __init__(self, target, daemon=None):
        self.target = target

    def start(self):
        self.target()

    def join(self, timeout=None):
        pass


class SshIslandLaunches(FakeLaunches):
    """Each island launch is what ``yeto launch --gpu ssh:...`` runs for one island of a
    fleet: the resident ssh: island ops on its inventory nodes and the fleet controller
    supervising it (its island STOP, its record, the straggler watch), on a thread."""

    def __init__(self, tmp_path, cluster: TrainerCluster, objects, tape) -> None:
        super().__init__()
        self.tmp_path, self.cluster, self.objects, self.tape = tmp_path, cluster, objects, tape
        self.controllers: dict[int, FleetController] = {}
        self.logs: dict[int, FleetLog] = {}
        self.errors: list[BaseException] = []

    def start(self, prefix: str, argv: list[str]) -> int:
        pid = super().start(prefix, argv)
        k = int(argv[argv.index("--rl-fleet-learner-id") + 1])
        assert argv[argv.index("--gpu") + 1] == GPU
        name = f"{prefix}-trainer"
        ops = SshIslandOps(run_id=prefix, journal_dir=self.tmp_path / "runs" / prefix / "ssh-islands",
                           events_dir=self.tmp_path / "events" / prefix, ops=self.cluster, log=lambda m: None)
        spec = SshTrainerSpec(nodes=NODES[k], run_id=prefix, workdir="/opt/yeto", run_script="echo island", env={})
        ops.add(name, spec)
        job = ops.start(name)
        log = self.logs[k] = FleetLog()
        recorder = lifecycle.FleetRecorder(lifecycle.IslandStateStore(self.objects, log=lambda m: None),
                                           {name: lifecycle.IslandRef(k, "ssh")})
        ctl = self.controllers[k] = FleetController(
            learners={name: (spec, job)}, syncer=None, sky_ops=RoutingOps(FastClock(), None, ssh_ops=ops),
            poll_interval=30.0, recover_timeout=600.0, thread_cls=_SyncThread, recorder=recorder, elastic=True,
            deterministic_stop=True, fleet_log=log,
            straggler=st.StragglerWatch(st.FileTape(self.tape), cycles=CYCLES, log=lambda m: None))

        def supervise():
            code = 0
            try:
                ctl.run()
            except BaseException as error:  # noqa: BLE001 - reported by the test
                self.errors.append(error)
                code = 1
            self.end(prefix, code)

        threading.Thread(target=supervise, daemon=True).start()
        return pid


def _config(tmp_path):
    doc = example_doc(tmp_path, max_islands=MAX_LEARNERS)
    for island in doc["islands"]:
        island["trainer"].update(gpu=GPU, usd_per_h=0.0)  # own machines
    path = tmp_path / "run.json"
    path.write_text(json.dumps(doc))
    return up.load_config(str(path))


@needs_cargo
@pytest.mark.parametrize("case", ["add", "remove", "stuck"])
def test_ssh_islands_join_leave_and_lag_without_touching_the_other_islands(syncer_binary, tmp_path,  # noqa: F811
                                                                           case):
    cfg = _config(tmp_path)
    plan = fleet.plan_of(cfg)
    objects = DirectoryObjectStore(tmp_path / "store")
    store = dz.DesiredStore(objects)
    members = island_launches.trainer_members(cfg, plan)
    store.apply({"members": [*members, *({"kind": "sampler_pool", "name": p.name, "island": i.index,
                                         "pool": {"provider": "modal"}} for i in plan.islands for p in i.pools)]},
                expect=0)
    total_steps, change_at = 12, 3
    process, port, profile_hash = _start_async_syncer(
        syncer_binary, tmp_path, *(STUCK_POOL if case == "stuck" else POOL), total_steps=total_steps,
        log_file=tmp_path / "syncer.log")
    tape = tmp_path / "tape.jsonl"
    island_launches.write_syncer_record(objects, addr=f"127.0.0.1:{port}", syncer_epoch=0, learners=LAUNCHED,
                                        max_learners=MAX_LEARNERS, run="r")
    states = lifecycle.IslandStateStore(objects, log=lambda m: None)
    cluster = TrainerCluster()
    launches = SshIslandLaunches(tmp_path, cluster, objects, tape)
    worker = trainer_worker.TrainerWorker(store, objects, launches, states, log=lambda m: None)

    clients: dict[int, object] = {}
    started = Counter()
    gone: list[int] = []
    catching: dict[int, dict] = {}
    versions = {k: [0] * P for k in range(MAX_LEARNERS)}
    seen: dict[int, set] = {k: set() for k in range(MAX_LEARNERS)}
    held: dict[int, list] = {}
    state = {"changed": False, "stuck": False, "flagged": False, "cleared": False, "joined_at": None}
    beat = [0.0]

    def follow_the_nodes():
        with cluster.guard:
            starts, kills = list(cluster.trainer_starts), list(cluster.trainer_kills)
        for k in starts[sum(started.values()):]:
            started[k] += 1
            client = _client(k, port, profile_hash)
            client.start()
            clients[k] = client
            if k >= LAUNCHED:
                catching[k] = {}
        for k in kills:
            if k in clients and k not in gone:  # its machines torn down: the connection just drops
                clients.pop(k).close()
                gone.append(k)

    def drain(k):
        client = clients[k]
        client.check_health()
        if k in catching:  # a joiner applies its catch-up before it answers a PULL
            caught = catching[k]
            for update in client.drain_updates():
                caught.setdefault(update.fragment_id, update.version)
            held.setdefault(k, []).extend(client.drain_pulls())
            if len(caught) < P:
                return
            versions[k] = [caught[fid] for fid in range(P)]
            state["joined_at"] = list(versions[k])  # the outer versions its catch-up brought
            del catching[k]
        for update in client.drain_updates():
            versions[k][update.fragment_id] = max(versions[k][update.fragment_id], update.version)
        if case == "stuck" and k == 1 and state["stuck"]:
            return  # connected (its heartbeats go on), delivering nothing
        for permit in held.pop(k, []) + client.drain_pulls():
            key = (permit.global_step, permit.round_attempt)
            if key in seen[k]:
                continue
            seen[k].add(key)
            client.push_fragment(permit.fragment_id, permit.global_step, permit.round_attempt,
                                 versions[k][permit.fragment_id], permit.global_step, 1, 7,
                                 _f32(_delta(k, permit.global_step)))

    def change():
        if case == "add":
            k = allocate_learner_id(objects, launched=LAUNCHED, max_learners=MAX_LEARNERS, island="north")
            entry = dict(cfg["islands"][1]["samplers"][0], name="sglang-north")
            member, pools = fleet_islands.compile_island(cfg, plan, k=k, name="north", gpu=GPU, usd_per_h=0.0,
                                                         samplers=[entry])
            store.update(lambda d: fleet_islands.add(d, member, pools), note="add north")
        elif case == "remove":
            store.update(lambda d: fleet_islands.remove(d, "sglang"), note="remove island 1")
        else:
            state["stuck"] = True
        state["changed"] = True

    def poll():
        assert not launches.errors, launches.errors
        if time.monotonic() - beat[0] > 1:  # every connected island renews its lease
            beat[0] = time.monotonic()
            for client in list(clients.values()):
                client.heartbeat(0)
        worker.step()
        follow_the_nodes()
        for k in list(catching) or list(clients):
            drain(k)
        if catching:
            return False  # a joiner catches up before the fleet moves on
        if not state["changed"] and len(clients) == LAUNCHED and min(versions[0]) >= change_at:
            change()
        if case == "stuck" and state["stuck"] and launches.logs[1].kinds(st.EVENT):
            state["stuck"], state["flagged"] = False, True  # reported: it pushes again
        if case == "stuck" and state["flagged"] and launches.logs[1].kinds(st.CLEARED_EVENT):
            state["cleared"] = True
        done = all(c.finalizing.is_set() for c in clients.values())
        if case == "add":
            return done and 2 in clients and state["joined_at"] is not None
        if case == "remove":
            return done and 1 in gone
        return done and state["cleared"]

    finals = {}
    try:
        worker.step()  # islands 0 and 1 start on their nodes
        follow_the_nodes()
        assert sorted(clients) == [0, 1]
        for fid in range(P):
            clients[0].send_init(fid, _f32(_initial(fid)))
        _wait(poll, f"the fleet never finalized ({case})", timeout=240)
        for k, client in clients.items():
            manifest, _fragments = client.wait_for_final_fragments(timeout=30)
            finals[k] = manifest
            client.acknowledge_finalization(manifest)
        assert process.wait(timeout=15) == 0
        for k in list(clients):
            cluster.finish(k)  # the trainers end their run: every launch ends by its trainer
        _wait(lambda: all(launches.ended(m["prefix"]) is not None for m in worker_members(store, case)),
              "an island launch did not end", timeout=60)
    finally:
        for client in clients.values():
            client.close()
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=10)
    assert not launches.errors, launches.errors

    # -- the other islands were never touched: one start per node, one launch, no kill before the end
    others = {"add": [0, 1], "remove": [0], "stuck": [0, 1]}[case]
    for k in others:
        assert started[k] == 1 and launches.started(f"r-i{k}") == 1
        assert [sum(1 for c in cluster.calls if c[:2] == ("start", n.name)) for n in NODES[k]] == [1, 1]
    kills_before_end = [k for k in cluster.trainer_kills if k in gone]
    assert kills_before_end == ([1] if case == "remove" else [])
    if case == "add":
        assert started[2] == 1 and launches.started("r-i2") == 1
    if case == "remove":
        assert all(sum(1 for c in cluster.calls if c[:2] == ("kill", n.name)) >= 1 for n in NODES[1])
        worker.step()
        assert worker.load(1)["state"] == trainer_worker.REMOVED

    # -- the merges: steps 1..N without a gap or a repeat, no push merged twice
    steps = _outer_steps(tape.read_text())
    assert [r["step"] for r in steps] == list(range(1, total_steps * P + 1))
    merged = [(int(e[0]), pushed) for r in steps for e, pushed in zip(r["raw_weights"], r["pushed_for"])]
    assert len(merged) == len(set(merged)), Counter(merged).most_common(3)
    for record in steps:
        assert abs(sum(float(e[2]) for e in record["weights"]) - 1.0) < 1e-9
    contributors = {int(e[0]) for r in steps for e in r["raw_weights"]}
    assert contributors == ({0, 1, 2} if case == "add" else {0, 1})
    if case == "add":  # the joiner caught up to the merged weights of the add's time, then merged
        first = min(r["step"] for r in steps if 2 in [int(e[0]) for e in r["raw_weights"]])
        assert min(state["joined_at"]) >= change_at and first > max(state["joined_at"])
    if case == "remove":
        leave = next(r for r in _records(tape) if r.get("kind") == "pool_leave" and int(r["island_id"]) == 1)
        after = [r for r in steps if r.get("time_unix", 0) > leave.get("time_unix", 0)][2 * P:]
        assert after and all([int(e[0]) for e in r["raw_weights"]] == [0] for r in after)

    # -- the straggler: written once by island 1's controller, waiting; nobody else reported
    if case == "stuck":
        (event,) = launches.logs[1].kinds(st.EVENT)
        assert event[2]["action"] == st.WAIT and event[2]["missed_cycles"] == CYCLES and event[2]["fragments"] == P
        assert event[2]["absent"] is True
        assert len(launches.logs[1].kinds(st.CLEARED_EVENT)) == 1
        assert states.read(1)["straggler"] is None  # cleared in its record too
        absent = [r for r in steps if r["absent"] == [1]]
        assert any(r["timed_out"] for r in absent)  # merged at the quorum deadline without it
        assert launches.controllers[1].learners["r-i1-trainer"]["relaunches"] == 0
    for k, log in launches.logs.items():
        if case != "stuck" or k != 1:
            assert not log.kinds(st.EVENT), (k, log.kinds(st.EVENT))

    # -- the ledger: each ssh incarnation created once, released at most once
    rows = [r for r in ResourceLedger(tmp_path / "ledger.jsonl").records() if r["kind"] == SSH_TRAINER]
    created = Counter(r["id"] for r in rows if r["op"] == "create")
    released = Counter(r["id"] for r in rows if r["op"] == "release")
    islands = {"add": 3, "remove": 2, "stuck": 2}[case]
    assert sorted(created) == sorted(f"r-i{k}/r-i{k}-trainer-i0" for k in range(islands))
    assert set(created.values()) == {1} and all(n == 1 for n in released.values())


def worker_members(store, case):
    members = list(dz.trainers(store.read()).values())
    if case == "remove":
        members.append({"prefix": "r-i1"})
    return members


def _records(tape):
    return [json.loads(line) for line in tape.read_text().splitlines() if line.strip()]
