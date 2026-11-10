"""Provider-agnostic cap on live sandboxes (HARBOR_MAX_SANDBOXES).  CPU only."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from yeto.rl.agentic import AgentTask, HarnessConfigError
from yeto.rl.harness.harbor import capacity
from yeto.rl.harness.harbor.capacity import SandboxLimits, SandboxSlots, parse_caps
from yeto.rl.harness.harbor.trial import HarborHarness


# ------------------------------------------------------------- sandbox cap
def test_parse_caps():
    assert parse_caps("") == {}
    assert parse_caps("8") == {"*": 8}
    assert parse_caps('{"remote-docker": 4, "*": 16}') == {"remote-docker": 4, "*": 16}
    for bad in ("0", "-1", "x", '{"a": 0}', '{"a": "3"}', "true", "[1]"):
        with pytest.raises(HarnessConfigError):
            parse_caps(bad)


def test_remote_docker_refuses_to_run_uncapped():
    with pytest.raises(HarnessConfigError, match="HARBOR_MAX_SANDBOXES"):
        SandboxLimits({}).slots("remote-docker")
    assert SandboxLimits({}).slots("docker") is None
    limits = SandboxLimits({"*": 3})
    assert limits.slots("remote-docker").limit == 3 and limits.slots("boat") is limits.slots("boat")


def test_slots_hold_across_threads_and_event_loops(monkeypatch):
    monkeypatch.setattr(capacity, "POLL_S", 0.001)
    slots = SandboxSlots(3)
    now = {"n": 0, "peak": 0}
    lock = threading.Lock()

    async def one() -> None:
        async with slots.slot():
            with lock:
                now["n"] += 1
                now["peak"] = max(now["peak"], now["n"])
            await asyncio.sleep(0.002)
            with lock:
                now["n"] -= 1

    def loop_thread() -> None:
        async def many() -> None:
            await asyncio.gather(*(one() for _ in range(10)))

        asyncio.run(many())

    threads = [threading.Thread(target=loop_thread) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert now["peak"] == 3 and slots.peak == 3 and slots.held == 0


def test_a_cancelled_waiter_takes_no_slot(monkeypatch):
    monkeypatch.setattr(capacity, "POLL_S", 0.001)
    slots = SandboxSlots(1)

    async def go():
        holder_in, release = asyncio.Event(), asyncio.Event()

        async def holder():
            async with slots.slot():
                holder_in.set()
                await release.wait()

        h = asyncio.ensure_future(holder())
        await holder_in.wait()
        waiter = asyncio.ensure_future(slots.slot().__aenter__())
        await asyncio.sleep(0.01)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        release.set()
        await h
        assert slots.held == 0
        async with slots.slot():
            assert slots.held == 1
        # a cancelled holder also gives its slot back
        release.clear()
        inner = asyncio.ensure_future(holder())
        await asyncio.sleep(0.01)
        assert slots.held == 1
        inner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await inner

    asyncio.run(go())
    assert slots.held == 0


class _FakeTrial:
    live = 0
    peak = 0
    lock = threading.Lock()

    def __init__(self) -> None:
        self.paths = None

    @classmethod
    async def create(cls, config: Any) -> _FakeTrial:
        with cls.lock:
            cls.live += 1
            cls.peak = max(cls.peak, cls.live)
        return cls()

    async def run(self) -> Any:
        await asyncio.sleep(0.003)
        with type(self).lock:
            type(self).live -= 1
        return SimpleNamespace(exception_info=None, verifier_result=SimpleNamespace(rewards={"reward": 1.0}))


def test_harness_caps_live_sandboxes_per_provider(monkeypatch):
    monkeypatch.setattr(capacity, "POLL_S", 0.001)
    _FakeTrial.live = _FakeTrial.peak = 0
    harness = HarborHarness(lambda task, endpoint: object(), trial_cls=_FakeTrial,
                            limits=SandboxLimits({"remote-docker": 2}))
    task = AgentTask(task_id="t", agent="nop", provider="remote-docker")

    async def go():
        return await asyncio.gather(*(harness.run(task, "http://x/v1") for _ in range(12)))

    verdicts = asyncio.run(go())
    assert len(verdicts) == 12 and _FakeTrial.peak == 2 and _FakeTrial.live == 0
    uncapped = HarborHarness(lambda task, endpoint: object(), trial_cls=_FakeTrial, limits=SandboxLimits({}))
    with pytest.raises(HarnessConfigError, match="needs a sandbox cap"):
        asyncio.run(uncapped.run(task, "http://x/v1"))


def test_process_limits_read_the_environment_once(monkeypatch):
    capacity.reset_process_limits()
    monkeypatch.setenv(capacity.MAX_SANDBOXES_ENV, '{"remote-docker": 5}')
    try:
        first = capacity.process_limits()
        monkeypatch.setenv(capacity.MAX_SANDBOXES_ENV, "99")
        assert capacity.process_limits() is first and first.slots("remote-docker").limit == 5
    finally:
        capacity.reset_process_limits()
