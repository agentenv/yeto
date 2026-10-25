"""S27 G-EW: the verl engine recovery port (VERL-PARITY #16).

CPU only. verl's separate-mode trainer is a fake with the attributes the port uses
(``standalone_server_manager``, ``standalone_checkpoint_manager``, ``replay_buffer``)
and Ray is a fake module (actors that can die, nodes, placement groups, kill). The
driver and the outage state machine are the real ones (``IslandDriver`` over the
FakeEngine ports of test_s27_rollout_ahead.py, ``engine_outage.EngineOutage``).

Checks:
* the port's verbs: liveness from the replica server actors and the GPU nodes; an
  abort makes verl's replay-buffer wait loop raise; restore starts new replicas of
  the same rank in the old placement groups and swaps the load balancer and the
  checkpoint manager;
* a round whose replicas die during generation (verl's buffer keeps re-issuing the
  failed prompts, so the generation never returns by itself) is aborted, the
  replicas are restored, the policy is republished and every round trains once;
* arming follows the budget environment variable and the separate mode; the
  TEST ONLY kill fires once, in the first process, and kills a real process tree.
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
import torch

from test_s27_rollout_ahead import NAME, _driver
from yeto.rl.adapters.verl import engine_recovery as er
from yeto.rl.engine.engine_outage import ENGINE_RECOVERY_ENV, EngineOutage, EnginesLost


# ---------------------------------------------------------------------------- fake Ray and verl
class _ActorDied(Exception):
    pass


class _Ref:
    def __init__(self, value=None, error=None):
        self.value, self.error = value, error


class _Method:
    def __init__(self, fn):
        self.fn = fn

    def remote(self, *args, **kwargs):
        try:
            return _Ref(self.fn(*args, **kwargs))
        except _ActorDied as error:
            return _Ref(error=error)


class _Ray:
    exceptions = SimpleNamespace(RayActorError=_ActorDied)

    def __init__(self):
        self.dead: set[str] = set()
        self.killed: list[str] = []
        self.gpu_nodes = 1
        self.pg_state = "CREATED"
        self.util = SimpleNamespace(placement_group_table=lambda group: {"state": self.pg_state})

    def nodes(self):
        return [{"Alive": True, "Resources": {"GPU": 1.0}} for _ in range(self.gpu_nodes)] + [
            {"Alive": True, "Resources": {"CPU": 4.0}}]

    def wait(self, refs, timeout=None):
        return list(refs), []

    def get(self, ref, timeout=None):
        if isinstance(ref, _Ref):
            if ref.error is not None:
                raise ref.error
            return ref.value
        return ref

    def kill(self, handle, no_restart=True):
        self.killed.append(handle.name)
        self.dead.add(handle.name)


RAY = _Ray()


class _Actor:
    def __init__(self, name, pid=None):
        self.name, self.pid = name, pid

    def _alive(self):
        if self.name in RAY.dead:
            raise _ActorDied(f"actor {self.name} died")

    @property
    def get_server_address(self):
        return _Method(lambda: (self._alive(), (self.name, 8000))[1])

    @property
    def __ray_call__(self):
        return _Method(lambda fn, *args: (self._alive(), fn(self, *args))[1])


class _Pool:
    def __init__(self, name):
        self.name = name

    def get_placement_groups(self):
        return [f"pg-{self.name}"]


class _Replica:
    started: list = []

    def __init__(self, replica_rank, config, model_config, gpus_per_node, name_suffix=""):
        self.replica_rank, self.config, self.model_config = replica_rank, config, model_config
        self.gpus_per_node, self.name_suffix = gpus_per_node, f"_{name_suffix}" if name_suffix else ""
        self.servers, self.workers, self.resource_pool = [], [], None
        self.aborts = 0

    async def init_standalone(self, resource_pool=None):
        self.resource_pool = resource_pool or _Pool(f"rollout_pool_{self.replica_rank}")
        self.servers = [_Actor(f"server_{self.replica_rank}_0{self.name_suffix}")]
        self.workers = [_Actor(f"worker_{self.replica_rank}_0{self.name_suffix}")]
        _Replica.started.append((self.name_suffix, self.resource_pool))

    @property
    def server_address(self):
        return f"10.0.0.1:{8000 + self.replica_rank}{self.name_suffix}"

    @property
    def server_handle(self):
        return self.servers[0]

    async def abort_all_requests(self, reject_request=False):
        self.servers[0]._alive()
        self.aborts += 1


class _Balancer:
    def __init__(self):
        self.servers = {}
        self.add_servers = _Method(self.servers.update)
        self.remove_servers = _Method(lambda ids: [self.servers.pop(i, None) for i in ids])


class _Checkpoints:
    def __init__(self, replicas):
        self.replicas = list(replicas)

    def remove_replicas(self, replicas):
        self.replicas = [r for r in self.replicas if r not in replicas]

    def add_replicas(self, replicas):
        self.replicas.extend(replicas)


class _ReplayBuffer:
    """verl's sync replay buffer wait loop: it re-issues failed prompts and waits."""

    def __init__(self):
        self.syncs = 0

    def _sync_metadata_from_transfer_queue(self):
        self.syncs += 1

    def sample(self, ready, timeout_s=20.0):
        deadline = time.monotonic() + timeout_s
        while True:
            self._sync_metadata_from_transfer_queue()
            if ready():
                return
            if time.monotonic() > deadline:
                raise AssertionError("the wait loop never ended (no abort)")
            time.sleep(0.005)


def _trainer(replicas=1):
    rollout_config = SimpleNamespace(n_gpus_per_node=1)
    old = [_Replica(rank, rollout_config, "model-config", 1) for rank in range(replicas)]
    for replica in old:
        asyncio.run(replica.init_standalone())
    balancer = _Balancer()
    balancer.servers.update({r.server_address: r.server_handle for r in old})
    manager = SimpleNamespace(rollout_replicas=old, rollout_replica_class=_Replica, rollout_config=rollout_config,
                              model_config="model-config", global_load_balancer=balancer,
                              server_handles=[r.server_handle for r in old],
                              server_addresses=[r.server_address for r in old])
    return SimpleNamespace(standalone_server_manager=manager, standalone_checkpoint_manager=_Checkpoints(old),
                           replay_buffer=_ReplayBuffer())


@pytest.fixture(autouse=True)
def _fresh_ray():
    global RAY
    RAY = _Ray()
    _Replica.started = []
    yield


def _port(trainer, **kw):
    return er.VerlEngineRecovery(trainer, nodes=1, run=asyncio.run, ray_module=RAY, **kw)


# ---------------------------------------------------------------------------- the port's verbs
def test_the_port_reads_liveness_from_the_replica_servers_and_the_gpu_nodes():
    trainer = _trainer()
    port = _port(trainer)
    assert port.engines_alive() and port.nodes_ready()
    RAY.dead.add("server_0_0")
    assert not port.engines_alive()
    assert port.is_engine_loss(RuntimeError("anything")) and port.is_engine_loss(_ActorDied("x"))
    RAY.dead.clear()
    RAY.gpu_nodes = 0
    assert not port.engines_alive() and not port.nodes_ready()
    RAY.gpu_nodes, RAY.pg_state = 1, "RESCHEDULING"
    assert port.engines_alive() and not port.nodes_ready()  # the bundles are not placed again yet
    assert not port.is_engine_loss(ValueError("not an engine error"))


def test_an_abort_ends_verls_wait_loop_and_a_restore_clears_it():
    trainer = _trainer()
    port = _port(trainer)
    trainer.replay_buffer.sample(lambda: True)  # unchanged without an abort
    port.abort_generation()
    with pytest.raises(EnginesLost):
        trainer.replay_buffer.sample(lambda: False)
    port.restore_engines()
    trainer.replay_buffer.sample(lambda: True)
    with pytest.raises(RuntimeError, match="installed twice"):
        _port(trainer)


def test_a_restore_starts_new_replicas_in_the_old_placement_groups():
    trainer = _trainer(replicas=2)
    manager, checkpoints = trainer.standalone_server_manager, trainer.standalone_checkpoint_manager
    old = list(manager.rollout_replicas)
    port = _port(trainer)
    RAY.dead.add("server_1_0")  # one replica died; the other one is killed by the restore
    record = port.restore_engines()
    new = manager.rollout_replicas
    assert [r.replica_rank for r in new] == [0, 1] and all(r.name_suffix == "_yeto1" for r in new)
    assert [r.resource_pool for r in new] == [r.resource_pool for r in old]  # the same placement groups
    assert {"server_0_0", "worker_0_0", "server_1_0", "worker_1_0"} <= set(RAY.killed)
    assert sorted(manager.global_load_balancer.servers) == sorted(r.server_address for r in new)
    assert manager.server_addresses == [r.server_address for r in new]
    assert manager.server_handles == [r.server_handle for r in new]
    assert checkpoints.replicas == new  # the next update_weights builds its NCCL group with them
    assert record["replicas"] == [0, 1] and record["restore"] == 1 and record["addresses"] == manager.server_addresses
    assert port.engines_alive()
    assert port.restore_engines()["restore"] == 2 and manager.rollout_replicas[0].name_suffix == "_yeto2"


# ---------------------------------------------------------------------------- with the real driver
def test_a_round_whose_replicas_die_in_generation_is_restored_and_trained_once(tmp_path):
    driver, engine = _driver(tmp_path, rounds=4, limit=0, ahead=0)
    trainer = _trainer()
    port = _port(trainer)
    driver.engine_outage = EngineOutage(port, wait_s=30.0, emit=driver.emit, poll_s=0.02, heartbeat_s=5.0)
    real = engine.rollout.generate
    state = {"killed": False}

    def alive():
        return all(server.name not in RAY.dead for r in trainer.standalone_server_manager.rollout_replicas
                   for server in r.servers)

    def generate(rollout_id, **kwargs):  # VerlRolloutPool.generate: waits in verl's replay buffer
        if rollout_id == 2 and not state["killed"]:
            state["killed"] = True
            RAY.dead.update({"server_0_0", "worker_0_0"})
        trainer.replay_buffer.sample(alive)
        return real(rollout_id, **kwargs)

    engine.rollout.generate = generate
    final = driver.run()
    assert final.policy_version == 4 and torch.equal(final.tensors[NAME], torch.full((1, 2), 4.0))
    assert [c[1] for c in engine.calls if c[0] == "train"] == [0, 1, 2, 3]  # every round once
    assert [c[1] for c in engine.calls if c[0] == "publish"] == [0, 1, 2, 2, 3, 4]  # v2 into the new replicas
    events = [json.loads(line) for line in (tmp_path / "e.jsonl").read_text().splitlines()
              if '"rl_engine_outage"' in line]
    assert [e["phase"] for e in events] == ["batch_lost", "lost", "restoring", "restored"]
    restored = events[-1]
    assert restored["policy_version"] == 2 and restored["replicas"] == [0] and restored["restore"] == 1
    assert trainer.standalone_server_manager.rollout_replicas[0].name_suffix == "_yeto1"


# ---------------------------------------------------------------------------- arming and the test kill
class _Driver:
    def __init__(self):
        self.events, self.rollout = [], SimpleNamespace(generate=lambda rid, **kw: ("batch", rid),
                                                         members=lambda: frozenset({"m"}))

    def emit(self, event, **fields):
        self.events.append({"event": event, **fields})


def test_arming_follows_the_budget_and_the_separate_mode():
    plan = {"separate": {"trainer_nodes": 1, "rollout_nodes": 1}}
    driver = _Driver()
    assert er.arm_verl_engine_outage(driver, _trainer(), plan, run=asyncio.run, environ={},
                                     ray_module=RAY) is None
    assert not hasattr(driver, "engine_outage")
    with pytest.raises(ValueError, match="separate mode"):
        er.arm_verl_engine_outage(driver, _trainer(), {}, run=asyncio.run,
                                  environ={ENGINE_RECOVERY_ENV: "600"}, ray_module=RAY)
    outage = er.arm_verl_engine_outage(driver, _trainer(), plan, run=asyncio.run,
                                       environ={ENGINE_RECOVERY_ENV: "600"}, ray_module=RAY)
    assert driver.engine_outage is outage and outage.wait_s == 600.0
    assert isinstance(outage.port, er.VerlEngineRecovery)
    assert driver.events == [{"event": "rl_engine_outage", "phase": "armed", "budget_s": 600.0, "nodes": 1,
                              "backend": "verl"}]
    assert not isinstance(driver.rollout, er.TestEngineKill)


@pytest.mark.parametrize("incarnation,wrapped", [("0", True), ("1", False)])
def test_the_test_kill_is_armed_in_the_first_process_only(incarnation, wrapped):
    driver = _Driver()
    er.arm_verl_engine_outage(driver, _trainer(), {"separate": {"trainer_nodes": 1, "rollout_nodes": 1}},
                              run=asyncio.run, ray_module=RAY,
                              environ={ENGINE_RECOVERY_ENV: "600", er.TEST_KILL_ENV: "3:0.5",
                                       "YETO_MODAL_INCARNATION": incarnation})
    assert isinstance(driver.rollout, er.TestEngineKill) is wrapped
    if wrapped:
        assert (driver.rollout.rollout_id, driver.rollout.delay_s) == (3, 0.5)
        assert driver.rollout.members() == frozenset({"m"})  # the rest of the port is the island's


def test_the_test_kill_fires_once_and_kills_the_server_process_tree():
    child = subprocess.Popen([sys.executable, "-c",
                              "import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', "
                              "'import time; time.sleep(120)']); time.sleep(120)"])
    try:
        time.sleep(0.5)
        trainer = _trainer()
        trainer.standalone_server_manager.rollout_replicas[0].servers[0].pid = child.pid
        port = _port(trainer)
        events = []
        kill = er.TestEngineKill(SimpleNamespace(generate=lambda rid, **kw: rid), port, 2, delay_s=0.05,
                                 emit=lambda event, **f: events.append({"event": event, **f}))
        # the server reports its pid; a worker on its node kills the tree
        server = trainer.standalone_server_manager.rollout_replicas[0].servers[0]
        original = er._kill_process_tree
        assert kill.generate(1) == 1 and not kill.fired
        assert kill.generate(2) == 2 and kill.fired
        deadline = time.monotonic() + 20
        while not events and time.monotonic() < deadline:
            time.sleep(0.05)
        assert kill.generate(2) == 2  # fires once
        child.wait(timeout=20)  # ended (psutil reaped it, so its return code reads 0 here)
        record = events[0]
        assert record["event"] == "rl_test_engine_kill" and record["rollout_id"] == 2
        killed = [name for name in record["killed"] if ":" in name]
        assert len(killed) == 2 and f"{child.pid}:python" in killed  # the server and its child
        assert {"server_0_0", "worker_0_0"} <= set(RAY.killed)
        assert server.name in RAY.dead and original is er._kill_process_tree
    finally:
        if child.poll() is None:
            child.kill()


@pytest.fixture(autouse=True)
def _server_pid(monkeypatch):
    """The fake server answers ``os.getpid()`` through ``__ray_call__`` with its own pid."""
    real = _Actor.__ray_call__

    def call(self):
        def run(fn, *args):
            self._alive()
            if getattr(fn, "__name__", "") == "<lambda>" and self.pid is not None:
                return self.pid
            return fn(self, *args)
        return _Method(run)

    monkeypatch.setattr(_Actor, "__ray_call__", property(call))
    yield
    monkeypatch.setattr(_Actor, "__ray_call__", real)


def test_a_no_sync_separate_mode_publication_passes_the_trainer_holds_check():
    """s27-ew-o2 (GPU) failed its first publication: the no-sync local state's export had no
    trainer-weights check, and ``SeparatePublisher`` calls ``check_holds`` (WeightsChanged
    "resident state has no trainer weights check"). The export now carries G-EV's check:
    it holds until the trainer trains another round."""
    from test_s27_verl_full_parameter_sync import COMPONENT, MAX_FRAGMENT_BYTES, _Island, _Model
    from yeto.rl.adapters.verl import full_parameter_stream as vfs
    from yeto.rl.engine.policy_digest import WeightsChanged

    island = _Island(_Model())
    group = vfs.VerlFullParameterGroup(island)
    state = vfs.VerlFullParameterLocalState(group, component=COMPONENT, max_fragment_bytes=MAX_FRAGMENT_BYTES)
    exported = state.export(policy_version=0)
    exported.check_holds()  # what SeparatePublisher.publish calls first
    group.trained_rounds += 1  # the trainer port records one more trained round
    with pytest.raises(WeightsChanged, match="trained 1 round"):
        exported.check_holds()
    state.export(policy_version=1).check_holds()
