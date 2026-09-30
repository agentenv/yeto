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
    from yeto.rl.engine.miles_adapter.rollout import INJECT_STOP_FAILURES_ENV, MilesRolloutPool

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


def test_weight_override_runs_after_update_before_check_weights(monkeypatch):
    from yeto.rl.engine.miles_adapter.publish import (
        INJECT_WEIGHT_OVERRIDE_ENV,
        MilesPublisher,
        PublicationError,
    )

    order = []

    class Controller:
        async def wait_cells_tracked(self, cells, timeout_seconds):
            order.append("tracked")

        async def start_update_weights(self, **_):
            raise AssertionError("stop the test after the override")

    async def update_weights(*a, **k):
        order.append("update_weights")

    monkeypatch.setenv(INJECT_WEIGHT_OVERRIDE_ENV, "/vol/other-ckpt")
    pub = MilesPublisher(args=SimpleNamespace(), actor_model=None, rollout_executor=None,
                         inference_controller=Controller(), update_weights=update_weights)

    async def injector(cell, path):
        order.append(("override", cell, path))

    pub.weight_override_injector = injector
    with pytest.raises(AssertionError, match="stop the test"):
        asyncio.run(pub._publish_members("t", ["c3", "c2"], 1))
    assert order == ["tracked", "update_weights", ("override", "c2", "/vol/other-ckpt")]
    order.clear()
    with pytest.raises(AssertionError):
        asyncio.run(pub._publish_members("t", ["c3"], 2))
    assert order == ["tracked", "update_weights"]  # once per process
    del PublicationError
