"""Test-only fault injections for the E1 GPU runs (B1 finding 5/6), CPU."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from test_rl_reconfig_e1 import _setup
from yeto.rl.engine.controller import CANCELLED, KILL_AT_ENV, SUCCEEDED, IslandController


class _Killed(BaseException):
    pass


def _kill_run(tmp_path, monkeypatch, phase):
    monkeypatch.setenv(KILL_AT_ENV, phase)
    driver, ctl, fork, pool, *_ = _setup(tmp_path)
    ctl._exit = lambda code: (_ for _ in ()).throw(_Killed(code))
    orig = driver.safe_point

    def safe_point(rid):
        if rid == 1:
            ctl.request("r", "T4R4S0", 0, 60)
        return orig(rid)

    driver.safe_point = safe_point
    with pytest.raises(_Killed):
        driver.run()
    ctl.close()
    driver.ledger.close()
    monkeypatch.delenv(KILL_AT_ENV)
    restarted = IslandController(state_dir=tmp_path / "state", configs=ctl.configs,
                                 initial_config="T4R2S2", attestation=ctl.attestation,
                                 profile=ctl.profile, runtime_fingerprint=ctl.runtime_fingerprint)
    return restarted, pool, fork


def test_kill_after_commit_is_succeeded_after_the_restart(tmp_path, monkeypatch):
    ctl, pool, fork = _kill_run(tmp_path, monkeypatch, "COMMITTED")
    ctl.open(pool)
    assert ctl.status("r")["phase"] == SUCCEEDED
    assert (tmp_path / "state" / "test-killed-at-COMMITTED").exists()


def test_kill_in_quiescing_is_cancelled_after_the_restart(tmp_path, monkeypatch):
    ctl, pool, fork = _kill_run(tmp_path, monkeypatch, "QUIESCING")
    ctl.open(pool)
    assert ctl.status("r")["phase"] == CANCELLED


def test_kill_happens_once_per_state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(KILL_AT_ENV, "QUIESCING")
    driver, ctl, *_ = _setup(tmp_path)
    (tmp_path / "state" / "test-killed-at-QUIESCING").write_text("killed\n")
    ctl._exit = lambda code: pytest.fail("killed twice")
    orig = driver.safe_point
    driver.safe_point = lambda rid: (ctl.request("r", "T4R4S0", 0, 60) if rid == 1 else None,
                                     orig(rid))[1]
    driver.run()
    assert ctl.status("r")["phase"] == SUCCEEDED


def test_stop_failure_injection_goes_through_the_fork_provider(monkeypatch):
    from yeto.rl.adapters.miles.rollout import INJECT_STOP_FAILURES_ENV, MilesRolloutPool

    calls = []

    class Provider:
        async def stop_cells(self, cell_ids):
            calls.append(("provider_stop", tuple(cell_ids)))

    class Fork:  # InferenceController.stop_cells: deregister, then provider stop
        def __init__(self):
            self._engine_provider = Provider()
            self.incomplete = None

        async def stop_cells(self, cells, expected_epoch):
            calls.append(("deregister", tuple(cells)))
            try:
                await self._engine_provider.stop_cells(cell_ids=cells)
            except Exception:
                self.incomplete = ("stop", tuple(cells))
                raise

    monkeypatch.setenv(INJECT_STOP_FAILURES_ENV, "1")
    fork = Fork()
    pool = MilesRolloutPool(inference_controller=fork, rollout_executor=None, metadata=None,
                            expected_policy=lambda: (0, "h"),
                            runner=SimpleNamespace(run=asyncio.run), declared_cells=("c0", "c1"))
    pool.members = lambda: frozenset()
    with pytest.raises(RuntimeError, match="injected provider stop failure"):
        pool.remove_engines(frozenset({"engine:c1"}), epoch=0)
    assert fork.incomplete == ("stop", ("c1",)) and pool.injected_stop_failures == 1
    pool.remove_engines(frozenset({"engine:c1"}), epoch=0)  # the retry reaches the provider
    assert calls == [("deregister", ("c1",)), ("deregister", ("c1",)), ("provider_stop", ("c1",))]


def _perturb_publisher(monkeypatch, eps):
    from yeto.rl.adapters.miles.publish import INJECT_LORA_PERTURB_ENV, MilesPublisher

    if eps is None:
        monkeypatch.delenv(INJECT_LORA_PERTURB_ENV, raising=False)
    else:
        monkeypatch.setenv(INJECT_LORA_PERTURB_ENV, str(eps))
    world = {"trainer": 1.0, "engine": {}, "order": []}

    class Engine:
        def __init__(self, cell):
            self.cell, self.version = cell, None

        async def update_weight_version(self, token):
            self.version = token

        async def get_weight_version(self):
            return self.version

    class Controller:  # the fork's member-publication surface
        async def wait_cells_tracked(self, cells, timeout_seconds):
            world["order"].append("tracked")

        async def start_update_weights(self, members, expected_epoch):
            self.engines = [Engine(c) for c in members]
            return SimpleNamespace(rollout_engines=self.engines,
                                   snapshot_cell_id_to_hashes={c: "h" for c in members})

        async def end_update_weights(self, **_):
            pass

        async def abort_update_weights(self):
            pass

        async def check_weights(self, action):
            # every engine reports a checksum of the LoRA weights it holds
            return [{"w": repr(v)} for v in world["engine"].values()]

        async def admit_cells(self, cells, **_):
            world["order"].append("admit")

        async def start_commit_weight_version(self, **_):
            pass

        async def end_commit_weight_version(self):
            pass

    async def update_weights(*a, members, **k):
        world["order"].append(("update_weights", world["trainer"]))
        for c in members:
            world["engine"][c] = world["trainer"]  # the member receives the trainer's LoRA

    pub = MilesPublisher(args=SimpleNamespace(), actor_model=None, rollout_executor=None,
                         inference_controller=Controller(), update_weights=update_weights,
                         flatten_checksums=lambda raw: list(raw))
    pub._reference = ("tok", {"w": repr(1.0)})  # read-back reference of the published policy
    pub.last_engine_checksums = {}

    def perturb(scale):
        world["order"].append(("perturb", scale))
        world["trainer"] = 1.0 if scale is None else 1.0 + scale

    pub.perturb_trainer = perturb
    return pub, world


def test_lora_perturbation_makes_check_weights_refuse_the_new_engines(monkeypatch):
    from yeto.rl.adapters.miles.publish import PublicationError

    pub, world = _perturb_publisher(monkeypatch, 0.01)
    with pytest.raises(PublicationError, match="read-back differs"):
        asyncio.run(pub._publish_members("tok", ["c2"], 1))
    assert world["order"] == ["tracked", ("perturb", 0.01), ("update_weights", 1.01),
                              ("perturb", None)]  # never admitted
    assert world["trainer"] == 1.0  # the trainer holds the published policy again
    # only once per process: after REBUILD_OLD stops c2, the next member
    # publication is clean and admitted
    world["engine"].pop("c2")
    asyncio.run(pub._publish_members("tok", ["c3"], 2))
    assert world["order"][-1] == "admit"


def test_without_the_injection_the_same_publication_is_admitted(monkeypatch):
    pub, world = _perturb_publisher(monkeypatch, None)
    asyncio.run(pub._publish_members("tok", ["c2"], 1))
    assert world["order"] == ["tracked", ("update_weights", 1.0), "admit"]


def test_lora_perturber_restores_the_trainer_exactly():
    import torch

    from yeto.rl.adapters.miles.entry import lora_perturber
    from yeto.rl.engine.fake import FakeEngine

    engine = FakeEngine(tensors={"base_model.model.layer.lora_A.weight": torch.ones(1, 2)})
    driver = SimpleNamespace(policy_state=engine.policy_state, local_step=0)
    before = engine.policy_state.export().policy_tensor_hash()
    perturb = lora_perturber(driver)
    asyncio.run(perturb(0.5))
    assert engine.policy_state.export().policy_tensor_hash() != before
    assert torch.equal(engine.tensors["base_model.model.layer.lora_A.weight"],
                       torch.full((1, 2), 1.5))
    asyncio.run(perturb(None))
    assert engine.policy_state.export().policy_tensor_hash() == before


def test_lora_perturbation_inside_the_running_loop_with_real_policy_state(monkeypatch):
    """Regression (A4 session 3, E1-B): the hook ran the synchronous policy_state
    (LoopRunner.run_until_complete) inside the async _publish_members and died with
    "This event loop is already running". Drive the real MilesPolicyState + lora_perturber
    over an async actor, in one running loop, as the island does."""
    import torch

    from yeto.rl.adapters.miles.publish import PublicationError
    from yeto.rl.adapters.miles import LoopRunner
    from yeto.rl.adapters.miles.entry import lora_perturber
    from yeto.rl.adapters.miles.state import MilesPolicyState
    from yeto.rl.adapters.miles.state_plugin import APPLY_STATE, EXPORT_STATE

    name = "base_model.model.layer.lora_A.weight"
    held = {"t": torch.ones(1, 2), "step": 0}

    class Actor:  # async like RayWorkerHandle.__getattr__.<locals>.call
        async def run_plugin(self, path, kwargs):
            await asyncio.sleep(0)
            if path == EXPORT_STATE:
                return [{"policy_version": kwargs["policy_version"], "tensors": {name: held["t"].clone()}}]
            assert path == APPLY_STATE
            held["t"] = kwargs["tensors"][name].clone()
            return [{"applied": True}]

    ps = MilesPolicyState(actor_model=Actor(), base_model_revision="a" * 40, config_hash="b" * 64,
                          runner=LoopRunner())
    driver = SimpleNamespace(policy_state=ps, local_step=3)
    pub, world = _perturb_publisher(monkeypatch, 0.25)
    pub.perturb_trainer = lora_perturber(driver)
    seen = []

    async def update_weights(*a, members, **k):
        seen.append(held["t"].clone())
        world["order"].append(("update_weights", float(held["t"][0, 0])))
        for c in members:
            world["engine"][c] = float(held["t"][0, 0])

    pub._update_weights = update_weights

    async def main():
        # reaches check_weights (not "event loop is already running") and is refused
        with pytest.raises(PublicationError, match="read-back differs"):
            await pub._publish_members("tok", ["c2"], 1)

    asyncio.run(main())
    assert torch.equal(seen[0], torch.full((1, 2), 1.25))  # the member got the perturbed adapter
    assert torch.equal(held["t"], torch.ones(1, 2))  # trainer restored exactly
    assert ("update_weights", 1.25) in world["order"] and "admit" not in world["order"]


def test_stop_failure_injection_is_armed_inside_the_fork_actor_through_a_ray_handle(monkeypatch):
    """GPU d123 (chain 2, a65c650) regression: through Miles' RayWorkerHandle the provider is not
    reachable from the learner (``_engine_provider`` answered with a remote-call function and the
    stop failed with AttributeError instead of the fork's incomplete path). The arming must run
    inside the actor via ``__ray_call__``."""
    from yeto.rl.adapters.miles.rollout import INJECT_STOP_FAILURES_ENV, MilesRolloutPool

    calls = []

    class Provider:
        async def stop_cells(self, cell_ids):
            calls.append(("provider_stop", tuple(cell_ids)))

    class Actor:  # the real InferenceController, living inside the Ray actor
        def __init__(self):
            self._engine_provider = Provider()
            self.incomplete = None

        async def stop_cells(self, cells, expected_epoch):
            calls.append(("deregister", tuple(cells)))
            try:
                await self._engine_provider.stop_cells(cell_ids=cells)
            except Exception:
                self.incomplete = ("stop", tuple(cells))
                raise

    actor = Actor()

    class RayCall:
        class remote:
            def __init__(self, fn):
                self._fn = fn

            def __await__(self):
                return asyncio.sleep(0, result=self._fn(actor)).__await__()

    class ActorHandle:  # type name is what _is_ray_handle keys on
        __ray_call__ = RayCall

        def __getattr__(self, name):  # any other name = a remote-call function (RayWorkerHandle behaviour)
            def remote_call(*a, **k):
                return getattr(actor, name)(*a, **k)
            return remote_call

    class RayWorkerHandle:
        def __init__(self):
            self._actor_handle = ActorHandle()

        def __getattr__(self, name):
            return getattr(self._actor_handle, name)

    monkeypatch.setenv(INJECT_STOP_FAILURES_ENV, "1")
    handle = RayWorkerHandle()
    records = []
    pool = MilesRolloutPool(inference_controller=handle, rollout_executor=None, metadata=None,
                            expected_policy=lambda: (0, "h"),
                            runner=SimpleNamespace(run=asyncio.run), declared_cells=("c0", "c1"))
    pool.event_sink = lambda event, **f: records.append((event, f))
    pool.members = lambda: frozenset()
    with pytest.raises(RuntimeError, match="injected provider stop failure"):
        pool.remove_engines(frozenset({"engine:c1"}), epoch=0)
    assert actor.incomplete == ("stop", ("c1",)) and pool.injected_stop_failures == 1
    (event, f), = records
    assert event == "test_injection" and f["kind"] == "stop_failure" and f["where"] == "fork_actor"
    assert f["provider"] == "Provider" and f["applied"] is True
    pool.remove_engines(frozenset({"engine:c1"}), epoch=0)  # N exhausted: the retry reaches the provider
    assert calls == [("deregister", ("c1",)), ("deregister", ("c1",)), ("provider_stop", ("c1",))]
