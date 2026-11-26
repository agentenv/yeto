"""An on-site control plane moved to another host while the fleet trains, end to end: the
real Rust syncer, ``yeto run up``'s trainer worker on a local store, fake trainers (one
syncer client per island; no cloud). own-cluster-fleet-stability 5.2.

Host A runs the control plane: its trainer worker starts both island launches, each holds
its records lock while its supervisor lives. Host A is lost: its trainer worker and the
island launch supervisors are gone and its local state directory is deleted; the islands
(the syncer clients) train on. ``run up`` on host B, from another state directory, restores
the run's state from the run store, and its trainer worker takes each island over once its
lease ran out: the same launch (same prefix and incarnation), which adopts the running
island. No island process restarts (no second HELLO), every outer step is merged once and
in order, and the final weights equal the f32 Nesterov recomputation (1e-6)."""

from __future__ import annotations

import json
import shutil
import threading
import time

import numpy as np

from test_fleet_island_e2e import SyncerTrainers
from test_fleet_island_runup import FakeController
from test_fleet_service import (
    _env,  # noqa: F401 - autouse: fake secrets, the run registry in tmp, env restored
    onsite_doc,
)
from test_fleet_trainer_worker import FakeLaunches
from test_rl_full_parameter_async import (
    _f32,
    _initial,
    _start_async_syncer,
    _wait,
    needs_cargo,
    syncer_binary,  # noqa: F401 - fixture
)
from test_rl_full_parameter_dynamic_members import (
    LAUNCHED,
    MAX_LEARNERS,
    POOL,
    P,
    _delta,
    _nesterov,
    _outer_steps,
)

from yeto.launch import island_lifecycle as lifecycle
from yeto.launch.pool.store_lock import StoreLock
from yeto.rl import island_launches, up
from yeto.rl.engine.object_store import DirectoryObjectStore

KEY_VALUE = "onsite-takeover-e2e-key"
LOCK_TTL_S = 3.0  # the island launches' records lock lease (120 s in a launch; short here)
START_GRACE_S = 2.0  # a launch's start grace before it may be taken over (600 s in a run; short here)
MARGIN_S = 0.5  # fleet.takeover_margin_s (60 s by default; short here)


class Supervised(SyncerTrainers):
    """Host A's island launches: each starts its island (a syncer client) and holds the
    island's records lock while its supervisor lives (:meth:`lose_host` ends them all)."""

    def __init__(self, *a, objects, **kw) -> None:
        super().__init__(*a, **kw)
        self.objects, self.locks, self.alive_host = objects, {}, threading.Event()
        self.alive_host.set()

    def start(self, prefix: str, argv: list[str]) -> int:
        pid = super().start(prefix, argv)
        k = int(argv[argv.index("--rl-fleet-learner-id") + 1])
        self.locks[k] = StoreLock.acquire(self.objects, lifecycle.lock_prefix(k), holder=f"{prefix} on host-a",
                                          ttl_s=LOCK_TTL_S)
        threading.Thread(target=self._renew, args=(k,), daemon=True).start()
        return pid

    def _renew(self, k: int) -> None:
        while self.alive_host.wait(0) and self.locks.get(k) is not None:
            self.locks[k].renew()
            time.sleep(0.5)

    def lose_host(self) -> None:
        self.alive_host.clear()  # the supervisors die with the host; their leases run out


class Adopting(FakeLaunches):
    """Host B's island launches: a launch of an island whose process runs adopts it (the
    launcher's reattach): no second island process, no second HELLO."""

    def __init__(self, islands: Supervised) -> None:
        super().__init__()
        self.islands, self.adopted = islands, []

    def start(self, prefix: str, argv: list[str]) -> int:
        k = int(argv[argv.index("--rl-fleet-learner-id") + 1])
        assert self.islands.prefix_of.get(k) == prefix and k in self.islands.clients, "would start a second copy"
        self.adopted.append(prefix)
        return super().start(prefix, argv)


@needs_cargo
def test_the_control_plane_moves_host_and_no_island_restarts(syncer_binary, tmp_path, monkeypatch):  # noqa: F811, PLR0915
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", KEY_VALUE)
    monkeypatch.setenv("YETO_RESOURCE_LEDGER", str(tmp_path / "ledger.jsonl"))
    doc = onsite_doc(tmp_path)
    doc["fleet"]["max_islands"] = MAX_LEARNERS
    doc.setdefault("env", {}).pop("HF_TOKEN", None)
    path = tmp_path / "run.json"
    path.write_text(json.dumps(doc))
    objects = DirectoryObjectStore(tmp_path / "store")
    monkeypatch.setattr(up.Run, "_objects", lambda self: objects)
    from yeto.rl import trainer_worker

    monkeypatch.setattr(trainer_worker, "START_GRACE_S", START_GRACE_S)

    total_steps, move_at = 10, 3
    process, port, profile_hash = _start_async_syncer(syncer_binary, tmp_path, *POOL, total_steps=total_steps,
                                                      log_file=tmp_path / "syncer.log")
    island_launches.write_syncer_record(objects, addr=f"127.0.0.1:{port}", syncer_epoch=0, learners=LAUNCHED,
                                        max_learners=MAX_LEARNERS, run="r")
    states = lifecycle.IslandStateStore(objects, log=lambda m: None)
    islands = Supervised(port, profile_hash, states, objects=objects)

    # host A: run up with its state, its trainer worker starts the island launches
    a = up.Run(up.load_config(str(path)), state_root=tmp_path / "host-a")
    a.start_state_mirror(objects, host="host-a")
    a.state.update({"run_t0": time.time(), "trainer_t0": time.time(), "steps": {"first_group": "t"}})
    a.begin_up()
    a.mirror_state(force=True)  # its periodic mirror
    worker = a.attach_trainer_worker(FakeController(), objects, ops=islands)
    worker.host = "host-a"
    versions = {k: [0] * P for k in range(MAX_LEARNERS)}
    seen: dict[int, set] = {k: set() for k in range(MAX_LEARNERS)}
    pushes: list[tuple] = []
    state = {"moved": False, "b": None, "hellos": None}
    held: dict[int, list] = {}

    def taken_over() -> bool:
        # the final rounds wait for host B's takeover, so the run outlasts host A's lapsing leases
        return state["moved"] and all((state["b"][0].load(k) or {}).get("host") == "host-b" for k in (0, 1))

    def drain(k):
        client = islands.clients[k]
        client.check_health()
        for update in client.drain_updates():
            versions[k][update.fragment_id] = max(versions[k][update.fragment_id], update.version)
        held[k] = held.get(k, []) + client.drain_pulls()
        if not taken_over():  # the islands train on while no control plane runs, up to the last rounds
            ready = [p for p in held[k] if p.global_step <= total_steps - 2]
            held[k] = [p for p in held[k] if p.global_step > total_steps - 2]
        else:
            ready, held[k] = held[k], []
        for permit in ready:
            key = (permit.global_step, permit.round_attempt)
            if key in seen[k]:
                continue
            seen[k].add(key)
            pushes.append((k, permit.fragment_id, *key))
            client.push_fragment(permit.fragment_id, permit.global_step, permit.round_attempt,
                                 versions[k][permit.fragment_id], permit.global_step, 1, 7,
                                 _f32(_delta(k, permit.global_step)))

    def move():
        """Host A is lost; run up starts on host B from another state directory."""
        islands.lose_host()
        state["hellos"] = {k: id(c) for k, c in islands.clients.items()}
        shutil.rmtree(tmp_path / "host-a")  # its local run directory is gone with it
        b = up.Run(up.load_config(str(path)), state_root=tmp_path / "host-b")
        b.start_state_mirror(objects, host="host-b")  # restored from the run store
        assert b.state["steps"]["first_group"] == "t" and "procs" not in b.state
        launches = Adopting(islands)
        w = b.attach_trainer_worker(FakeController(), objects, ops=launches)
        w.host, w.takeover_margin_s = "host-b", MARGIN_S
        state["b"], state["moved"] = (w, launches), True

    def poll():
        if not state["moved"]:
            worker.step()
            if min(versions[0]) >= move_at:
                move()
        else:
            state["b"][0].step()
        for k in list(islands.clients):
            drain(k)
        return all(c.finalizing.is_set() for c in islands.clients.values())

    finals = {}
    try:
        worker.step()
        assert sorted(islands.clients) == [0, 1]
        for fid in range(P):
            islands.clients[0].send_init(fid, _f32(_initial(fid)))
        _wait(poll, "the fleet never finalized", timeout=180)
        for k, client in islands.clients.items():
            manifest, fragments = client.wait_for_final_fragments(timeout=30)
            finals[k] = {f.fragment_id: np.frombuffer(f.data, dtype=np.float32) for f in fragments}
            client.acknowledge_finalization(manifest)
        assert process.wait(timeout=15) == 0
    finally:
        for client in islands.clients.values():
            client.close()
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=10)

    w, launches = state["b"]
    # host B took both launches over: the same launches (no -m1, no new incarnation), adopted
    assert sorted(launches.adopted) == ["r-i0", "r-i1"]
    for k in (0, 1):
        record = w.load(k)
        assert record["host"] == "host-b" and record["incarnation"] == 0 and record["relaunches"] == 0
        assert record["taken_over"]["from"] == "host-a"
    # no island process restarted: one launch each on host A, the same clients to the end
    assert islands.started("r-i0") == 1 and islands.started("r-i1") == 1
    assert {k: id(c) for k, c in islands.clients.items()} == state["hellos"]
    tape = (tmp_path / "tape.jsonl").read_text()
    assert tape.count('"kind":"pool_join"') <= LAUNCHED and '"kind":"pool_leave"' not in tape
    # every outer step merged once, in order; nothing pushed twice; the merge is exact
    steps = _outer_steps(tape)
    assert [r["step"] for r in steps] == list(range(1, total_steps * P + 1))
    assert len(pushes) == len(set(pushes))
    expected = _nesterov(steps)
    for fid in range(P):
        np.testing.assert_allclose(finals[0][fid], expected[fid], rtol=0, atol=1e-6)
    restored = json.loads(objects.read(up.STATE_MIRROR))
    assert KEY_VALUE not in json.dumps(restored) and "procs" not in restored["state"]
