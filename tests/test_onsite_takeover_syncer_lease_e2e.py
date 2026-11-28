"""An on-site fleet whose inventory has a ``lease_store`` moved to another control host while
its ssh: island trains, end to end on CPU (no SSH, no cloud): the launches' real argv
(``yeto launch`` parser), their real node-lease path (``launcher._restore_lease_identity``,
``ssh_resident.check``, ``launcher._share_lease_placement``) on a directory lease store, and
``yeto run up``'s real trainer worker on a directory run store.

Host A: the syncer service has recorded the syncer; the trainer worker starts the island
launch, which leases the trainer node and starts the island (a container that keeps running
while no control plane does); the syncer launch watches the service. Host A is lost: its
launches stop renewing their leases and its local run directory is deleted. Host B, from
another run directory, starts run up again: its syncer launch records the syncer for this up
(the trainer worker waits for that), and the trainer worker takes the island over once its
leases ran out: the same launch, which leases the node as the same holder and adopts the
running island.

The syncer launch runs no trainer island: it must never take (or wait for) a trainer node's
lease. Before the fix it leased the island's node as ``<run>/trainer``: on host B the island's
lease had run out with its container still running, the lease probe refused the takeover
(``node_lease_blocked``), the syncer launch waited and never recorded the syncer, and the
trainer worker waited for the syncer: the island was never taken over (B7)."""

from __future__ import annotations

import shutil
import threading
import time
from functools import partial
from pathlib import Path

import pytest

from test_fleet_trainer_worker import fleet
from test_rl_ssh_resident import b300

from yeto import cli, launcher, runs
from yeto.gpu_spec import parse_gpu_spec
from yeto.launch import island_lifecycle as lifecycle
from yeto.launch import node_lease, ssh_resident, ssh_trainer_island
from yeto.launch.node_inventory import NodeList
from yeto.launch.pool.store_lock import StoreLock
from yeto.rl import island_launches as il
from yeto.rl import trainer_worker as tw
from yeto.rl.engine.object_store import DirectoryObjectStore

RUN = "run1"
GPU = "ssh:1x8xb300"
NODE = "b300-0"
LEASE_TTL_S = 2.0  # the node leases (600 s in a launch; short here)
LOCK_TTL_S = 2.0  # the island launch's records lock (120 s in a launch; short here)
START_GRACE_S = 1.0  # tw.START_GRACE_S (600 s in a run)
MARGIN_S = 0.3  # fleet.takeover_margin_s (60 s by default)
LEASE_WAIT_S = 4.0  # --rl-node-lease-wait-s of every launch here (3600 s by default): a wait fails fast
BASE = ["--training-mode", "rl", "--model", "Qwen/Qwen3-0.6B", "--data", "tasks.jsonl",
        "--reward-function", "yeto.rl.numeric_answer_reward:score", "--tuning", "full",
        "--rl-sync-preset", "decoupled", "--rl-node-inventory", "inventory.yaml",
        "--rl-node-lease-wait-s", str(LEASE_WAIT_S), "--rl-ssh-trainer-workdir", "/opt/yeto"]
# the syncer launch as run up starts it (island_launches.syncer_argv): the fleet's --gpu and
# inventory, the syncer role, watching an on-site syncer service
SYNCER_ARGV = [*BASE, "--cluster-prefix", RUN, "--gpu", GPU, "--rl-sample-role", "trainer",
               "--rl-sampler-follow", "trainer", "--rl-fleet-role", "syncer", "--rl-fleet-learners", "1",
               il.SYNCER_SERVICE_FLAG, "10.0.0.5", "--controller", "local"]


def island_member() -> dict:
    return {"kind": "trainer", "name": "A", "learner_id": 0, "gpu": GPU, "usd_per_h": 0.0,
            "argv": [*BASE, "--rl-sample-role", "trainer", "--rl-sampler-follow", "trainer",
                     "--rl-fleet-role", "island", "--rl-fleet-learner-id", "0", "--rl-fleet-learners", "1",
                     "--controller", "local", "--island-lifecycle", "on"],
            "prefix": f"{RUN}-i0", "pools": ["i0-pa"]}


class Site:
    """The trainer node: the island containers that run on it (they outlive a control host)
    and the lease probe (``node_lease.ssh_probe``) that finds them."""

    def __init__(self) -> None:
        self.running: dict[str, int] = {}  # island run (launch prefix) -> container starts
        self.starts = 0

    def start(self, run: str) -> None:
        if run not in self.running:
            self.starts += 1
            self.running[run] = self.starts

    def probe(self, run_script):  # node_lease.ssh_probe's signature
        def probe(node, previous: str) -> str | None:
            run = node_lease.run_of(previous)
            if run in self.running:
                return f"node {node_lease.node_name(node)} still runs run {run} (container:yeto-{run}-n0)"
            return None
        return probe


class Host:
    """One control host: its run directory, its launches (``yeto launch`` processes; each one
    a process of its own, so its own node-lease cache) and what they did."""

    def __init__(self, name: str, tmp_path: Path, site: Site, objects, lease_store: str, monkeypatch) -> None:
        self.name, self.tmp, self.site, self.objects = name, tmp_path, site, objects
        self.lease_store, self.monkeypatch = lease_store, monkeypatch
        self.leases: list = []  # every NodeLeases a launch of this host opened (their keepers)
        self.locks: dict[int, StoreLock] = {}
        self.alive = threading.Event()
        self.alive.set()
        self.live: dict[str, list[str]] = {}
        self.starts: list[tuple[str, list[str]]] = []
        self.errors: list[str] = []

    def enter(self) -> None:
        """This host's process state: its run directory, a fresh lease cache."""
        self.monkeypatch.setattr(runs, "RUNS_DIR", self.tmp / self.name / "runs")
        self.monkeypatch.setattr(ssh_trainer_island, "_TRAINER_LEASES", {})

    def args(self, argv: list[str]):
        args = cli.build_parser().parse_args(["launch", *argv])
        args.rl_node_inventory_nodes = NodeList([b300(0)], lease_store=self.lease_store)  # prepare_launch_args'
        return args

    def lease_preamble(self, args) -> None:
        """The launcher's node-lease steps before any cloud work (``launcher.run``)."""
        env = {launcher.JOURNAL_FROM_STORE_ENV: "1"}
        self.enter()
        launcher._restore_lease_identity(args, environ=env, objects=self.objects)
        ssh_resident.check(args, parse_gpu_spec(args.gpu))
        launcher._share_lease_placement(args, environ=env, objects=self.objects)
        self.leases.extend(ssh_trainer_island._TRAINER_LEASES.values())

    # -- the syncer launch (--rl-fleet-role syncer, fleet.syncer_service)
    def syncer_launch(self) -> None:
        self.lease_preamble(self.args(SYNCER_ARGV))
        # fleet_syncer.watch_service: the service is up; the syncer is recorded for this run up
        il.write_syncer_record(self.objects, addr="10.0.0.5:29400", syncer_epoch=1, learners=1,
                               max_learners=3, run=RUN)

    # -- the trainer worker's launch ops
    def alive_launch(self, prefix: str) -> bool:
        return prefix in self.live

    def start(self, prefix: str, argv: list[str]) -> int:
        self.starts.append((prefix, list(argv)))
        args = self.args(argv)
        try:
            self.lease_preamble(args)
        except (SystemExit, ssh_trainer_island.SshIslandError) as error:
            self.errors.append(str(error))
            return 0
        k = int(args.rl_fleet_learner_id)
        if tw.TAKEOVER_FLAG not in argv:
            self.site.start(prefix)  # a takeover launch adopts the island; it never starts it
        assert prefix in self.site.running, "the takeover launch found no island to adopt"
        self.live[prefix] = list(argv)
        lock = self.locks.get(k)
        if lock is None:  # the takeover claimed the lock for its launch already
            lock = StoreLock.acquire(self.objects, lifecycle.lock_prefix(k), holder=f"{prefix} on {self.name}",
                                     ttl_s=LOCK_TTL_S)
        self.locks[k] = lock
        threading.Thread(target=self._renew, args=(k,), daemon=True).start()
        return 1000 + len(self.starts)

    def _renew(self, k: int) -> None:
        while self.alive.is_set():
            lock = self.locks.get(k)
            if lock is not None:
                try:
                    lock.renew()
                except Exception:  # noqa: BLE001 - lost to a takeover: this launch is fenced
                    return
            time.sleep(0.3)

    def lose(self) -> None:
        """The host is gone: its launches stop renewing their leases and records locks."""
        self.alive.clear()
        for leases in self.leases:
            leases.stop_keeper()
        self.live.clear()


class Ops:
    """The trainer worker's launch ops (``yeto.rl.trainer_worker``) on a :class:`Host`."""

    def __init__(self, host: Host) -> None:
        self.host = host

    def alive(self, prefix: str) -> bool:
        return self.host.alive_launch(prefix)

    def start(self, prefix: str, argv: list[str]) -> int:
        return self.host.start(prefix, argv)

    def ended(self, prefix: str):
        return None

    def request_stop(self, prefix: str, why: str) -> None:
        raise AssertionError(f"nothing is stopped here ({prefix}: {why})")


def _wait(cond, what: str, timeout: float = 30.0, step=None) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if step is not None:
            step()
        if cond():
            return
        time.sleep(0.1)
    raise AssertionError(what)


def _events(lease_store: str) -> list[str]:
    return sorted(str(p.relative_to(lease_store)) for p in Path(lease_store).rglob("*.json") if "/events/" in str(p))


def _holders(lease_store: str) -> set[str]:
    import json

    return {json.loads(p.read_text())["holder"] for p in Path(lease_store).rglob("lease/*.json")}


@pytest.fixture
def site(tmp_path, monkeypatch):
    s = Site()
    monkeypatch.setattr(node_lease, "ssh_probe", lambda run, **kw: s.probe(run))
    monkeypatch.setattr(ssh_trainer_island, "SshOps", lambda *a, **kw: None)
    monkeypatch.setattr(node_lease, "NodeLeases", partial(node_lease.NodeLeases, ttl_s=LEASE_TTL_S))
    monkeypatch.setattr(tw, "START_GRACE_S", START_GRACE_S)
    monkeypatch.setenv("YETO_RESOURCE_LEDGER", str(tmp_path / "ledger.jsonl"))
    from yeto.hw import card_tools

    monkeypatch.setattr(card_tools, "check_cards", lambda family, text, **kw: [])
    return s


def test_takeover_with_a_lease_store_records_the_syncer_and_adopts_the_island(tmp_path, monkeypatch, site):  # noqa: PLR0915
    lease_store = str(tmp_path / "leases")
    objects, desired = fleet(tmp_path, [island_member()], syncer=False)
    states = lifecycle.IslandStateStore(objects, log=lambda m: None)

    # host A: the syncer service recorded the syncer; the worker starts the island launch
    a = Host("host-a", tmp_path, site, objects, lease_store, monkeypatch)
    il.write_syncer_record(objects, addr="10.0.0.5:29400", syncer_epoch=1, learners=1, max_learners=3, run=RUN)
    up_a = time.time() - 1
    worker_a = tw.TrainerWorker(desired, objects, Ops(a), states, log=lambda m: None, host="host-a",
                                syncer_since=up_a)
    worker_a.step()
    assert not a.errors and [p for p, _ in a.starts] == [f"{RUN}-i0"] and site.running == {f"{RUN}-i0": 1}
    island_holder = f"{RUN}-i0/trainer@" + a.leases[0].identity
    _record(a)  # the syncer launch of host A's run up (judged at the end: the takeover is the point)

    # host A is lost (its run directory too); the island trains on
    time.sleep(START_GRACE_S)
    a.lose()
    shutil.rmtree(tmp_path / "host-a")

    # host B: run up from another run directory; the syncer service's record is of an earlier up
    b = Host("host-b", tmp_path, site, objects, lease_store, monkeypatch)
    logs: list[str] = []
    up_b = time.time()
    worker_b = tw.TrainerWorker(desired, objects, Ops(b), states, log=logs.append, host="host-b",
                                syncer_since=up_b, takeover=True, takeover_margin_s=MARGIN_S)
    worker_b.step()
    assert b.starts == [] and any("wait for the fleet's syncer" in m for m in logs)
    syncer = threading.Thread(target=lambda: _record(b), daemon=True)
    time.sleep(LEASE_TTL_S + 0.5)  # the island's node lease ran out, its container still runs
    syncer.start()
    syncer.join(timeout=LEASE_WAIT_S + 10)
    assert not syncer.is_alive(), "the syncer launch hangs"
    assert b.errors == [] and getattr(b, "syncer_error", None) is None, getattr(b, "syncer_error", b.errors)
    record = il.read_syncer_record(objects)
    assert record is not None and float(record["at"]) >= up_b  # the syncer recorded for this up

    _wait(lambda: bool(b.starts), "the island was never taken over", step=worker_b.step)
    (prefix, argv), = b.starts
    assert b.errors == []
    assert prefix == f"{RUN}-i0" and tw.TAKEOVER_FLAG in argv  # the same launch, adopting
    rec = worker_b.load(0)
    assert rec["host"] == "host-b" and rec["incarnation"] == 0 and rec["taken_over"]["from"] == "host-a"
    assert site.starts == 1 and site.running == {f"{RUN}-i0": 1}  # the island never restarted
    # the island's node: leased again by the island's own holder only; nothing blocked
    assert _holders(lease_store) == {island_holder}
    assert not [e for e in _events(lease_store) if "blocked" in e], _events(lease_store)
    b.lose()
    # host A's syncer launch never waited for (nor took) the island's node either
    assert getattr(a, "syncer_error", None) is None, a.syncer_error


def _record(host: Host) -> None:
    try:
        host.syncer_launch()
    except BaseException as error:  # noqa: BLE001 - the test reports it
        host.syncer_error = f"{type(error).__name__}: {error}"
