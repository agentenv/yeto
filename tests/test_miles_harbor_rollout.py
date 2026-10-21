"""Miles Harbor rollout function: boat VMs are deleted at RolloutExecutor.dispose()."""

from __future__ import annotations

import asyncio
import importlib
import sys
import types

import pytest

from yeto.rl.harness.harbor import boat_pool


class _Pool:
    def __init__(self, n: int) -> None:
        self.n, self.calls = n, 0

    def close_now(self) -> int:
        self.calls += 1
        return self.n


@pytest.fixture
def rollout_mod(monkeypatch):
    events: list[str] = []

    class ExampleRolloutFn:
        fail = False

        async def dispose(self) -> None:
            events.append("example")
            if self.fail:
                raise RuntimeError("example dispose failed")

    generate = types.ModuleType("generate")
    generate.RolloutFn = ExampleRolloutFn
    monkeypatch.setitem(sys.modules, "generate", generate)
    monkeypatch.delitem(sys.modules, "yeto.rl.adapters.miles.harbor_rollout", raising=False)
    pool = _Pool(3)
    monkeypatch.setattr(boat_pool, "_exit_pools", {pool})
    return importlib.import_module("yeto.rl.adapters.miles.harbor_rollout"), pool, events


def test_dispose_runs_the_example_then_deletes_pool_vms(rollout_mod):
    mod, pool, events = rollout_mod
    asyncio.run(mod.RolloutFn().dispose())
    assert events == ["example"] and pool.calls == 1


def test_dispose_deletes_pool_vms_when_the_example_fails(rollout_mod):
    mod, pool, _ = rollout_mod
    fn = mod.RolloutFn()
    fn.fail = True
    with pytest.raises(RuntimeError):
        asyncio.run(fn.dispose())
    assert pool.calls == 1


def test_train_args_use_the_yeto_rollout_function():
    from yeto.rl.adapters.miles import harbor_agent

    args = harbor_agent.train_args(tito_model="m", session_server_workers=1).split()
    assert args[args.index("--rollout-function-path") + 1] == "yeto.rl.adapters.miles.harbor_rollout.RolloutFn"
