import json
import subprocess
import sys

import pytest

from yeto.rl.contracts import InferencePublicationManifest, LocalStepReceipt
from yeto.rl.engine import ports


def test_ports_import_is_light():
    code = (
        "import sys, json; import yeto.rl.engine.ports; "
        "print(json.dumps([m for m in ('torch','ray','miles') "
        "if any(k == m or k.startswith(m + '.') for k in sys.modules)]))"
    )
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert json.loads(out.stdout) == []


def test_ports_reuse_contract_types():
    assert ports.LocalStepReceipt is LocalStepReceipt
    assert ports.InferencePublicationManifest is InferencePublicationManifest
    for name in ports.__all__:
        assert hasattr(ports, name)


def test_protocol_shapes():
    class Pool:
        def generate(self, rollout_id): ...
        def abort(self): ...
        def members(self): ...

    class Trainer:
        def train_step(self, batch): ...
        def onload(self): ...
        def offload(self): ...

    assert isinstance(Pool(), ports.RolloutPool)
    assert isinstance(Trainer(), ports.TrainerGroup)
    assert not isinstance(Pool(), ports.TrainerGroup)
    # E1-E3 verbs are reserved, not required in R0.
    assert not hasattr(ports.RolloutPool, "add_engines")
    assert not hasattr(ports.TrainerGroup, "save_cut")
    assert not hasattr(ports.Placement, "reconfigure")


def test_rollout_handle_metadata_mismatch():
    good = ports.GroupMetadata("g0", ("s0",), "yeto:1:" + "a" * 64, 0.5, 0.1, 10)
    bad = ports.GroupMetadata("g1", ("s1",), "yeto:0:" + "b" * 64, 0.5, 0.1, 10)
    handle = ports.RolloutBatchHandle(1, 1, "a" * 64, (good, bad), 2, 0, payload=object())
    assert handle.mismatched_groups("yeto:1:" + "a" * 64) == (bad,)


# ------------------------------------------------ island Ray address (SkyPilot)


class _TwoRayMachine:
    """A SkyPilot island host: the island's Ray (6379) and SkyPilot's runtime
    Ray (6380) both active.  ``resolve`` mirrors Ray's
    ``canonicalize_bootstrap_address`` for an address-less call such as
    ``ray.util.state.list_nodes()`` inside an actor."""

    ACTIVE = {"10.0.0.7:6379", "10.0.0.7:6380"}

    def __init__(self, raylet_env):
        self.raylet_env = dict(raylet_env)
        self.initialized = False
        self.init_calls = []

    def is_initialized(self):
        return self.initialized

    def init(self, address=None, runtime_env=None):
        self.initialized = True
        self.init_calls.append((address, runtime_env))

    def actor_env(self):
        # Ray workers start from the raylet env; job-level runtime_env
        # env_vars are merged into every actor/task of the job.
        runtime_env = self.init_calls[-1][1] if self.init_calls else None
        return {**self.raylet_env, **((runtime_env or {}).get("env_vars") or {})}

    def resolve(self, env):
        address = env.get("RAY_ADDRESS")
        if address:
            return address
        if len(self.ACTIVE) > 1:
            raise ConnectionError(f"Found multiple active Ray instances: {self.ACTIVE}")
        return next(iter(self.ACTIVE))


def test_actor_without_island_address_hits_multiple_ray_instances():
    machine = _TwoRayMachine(raylet_env={"PATH": "/usr/bin"})
    machine.init()  # the old ports path: driver auto-init, no runtime_env
    with pytest.raises(ConnectionError, match="multiple active Ray"):
        machine.resolve(machine.actor_env())


def test_connect_island_ray_pins_driver_and_actors_to_the_island_ray():
    from yeto.rl.engine.miles_adapter.entry import connect_island_ray

    machine = _TwoRayMachine(raylet_env={"PATH": "/usr/bin", "PYTHONPATH": "/root/miles"})
    driver_env = {
        "RAY_ADDRESS": "10.0.0.7:6379",
        "PYTHONPATH": "/root/miles:/root/sglang/python:/root/sky_workdir",
    }
    assert connect_island_ray(environ=driver_env, ray_module=machine) == "10.0.0.7:6379"
    ((address, runtime_env),) = machine.init_calls
    assert address == "10.0.0.7:6379"
    actor_env = machine.actor_env()
    assert machine.resolve(actor_env) == "10.0.0.7:6379"
    assert actor_env["PYTHONPATH"] == driver_env["PYTHONPATH"]


def test_connect_island_ray_is_a_noop_without_address_and_refuses_late_pin():
    from yeto.rl.engine.miles_adapter.entry import connect_island_ray

    machine = _TwoRayMachine(raylet_env={})
    assert connect_island_ray(environ={}, ray_module=machine) is None
    assert machine.init_calls == []
    machine.initialized = True
    with pytest.raises(RuntimeError, match="RAY_ADDRESS"):
        connect_island_ray(environ={"RAY_ADDRESS": "10.0.0.7:6379"}, ray_module=machine)


def test_connect_island_ray_forwards_elastic_metadata_env_only_when_on():
    """Integ-s2 finding 2: Ray workers (where the rollout metadata hook runs)
    inherit the raylet env, so the --rl-elastic switch must travel in runtime_env."""
    from yeto.rl.engine.miles_adapter.entry import connect_island_ray
    from yeto.rl.engine.miles_adapter.rollout_meta_hook import ELASTIC_METADATA_ENV

    on = _TwoRayMachine(raylet_env={})
    connect_island_ray(environ={"RAY_ADDRESS": "a:6379", ELASTIC_METADATA_ENV: "1"}, ray_module=on)
    assert on.init_calls[0][1]["env_vars"][ELASTIC_METADATA_ENV] == "1"
    off = _TwoRayMachine(raylet_env={})
    connect_island_ray(environ={"RAY_ADDRESS": "a:6379"}, ray_module=off)
    assert ELASTIC_METADATA_ENV not in off.init_calls[0][1]["env_vars"]
