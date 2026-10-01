"""rl-infra-spec 1.7 / 3.3 X5: instantaneous tool-wait trajectory count (CPU)."""

from __future__ import annotations

import asyncio
import threading

import pytest

from yeto.rl.engine.timeline import LoadSample, classify_load
from yeto.rl.engine.tool_wait import (
    ToolWaitBoard,
    ToolWaitError,
    async_tool_wait_scope,
    drain_blockers,
    read_tool_wait,
    tool_wait_scope,
)


def _clock():
    t = iter(range(1000))
    return lambda: float(next(t))


def test_count_ages_and_balance():
    board = ToolWaitBoard(clock=_clock())
    assert read_tool_wait(board).in_flight == 0 and read_tool_wait(board).oldest_age_s is None
    board.enter("a")  # t=0
    board.enter("b")  # t=1
    snap = board.snapshot()  # t=2
    assert snap.in_flight == 2 and snap.trajectory_ids == ("a", "b") and snap.oldest_age_s == 2.0
    with pytest.raises(ToolWaitError):
        board.enter("a")
    board.exit("a")
    with pytest.raises(ToolWaitError):
        board.exit("a")
    snap = board.snapshot()
    assert (snap.in_flight, snap.entered_total, snap.exited_total, snap.generation) == (1, 2, 1, 3)


def test_scopes_count_only_inside_and_release_on_error():
    board = ToolWaitBoard()
    with tool_wait_scope(board, "t1"):
        assert board.snapshot().in_flight == 1
    with pytest.raises(RuntimeError):
        with tool_wait_scope(board, "t2"):
            raise RuntimeError("tool failed")
    assert board.snapshot().in_flight == 0

    seen = []

    async def trajectory(i):
        async with async_tool_wait_scope(board, f"a{i}"):
            await asyncio.sleep(0.01)
            seen.append(board.snapshot().in_flight)

    async def main():
        await asyncio.gather(*(trajectory(i) for i in range(3)))

    asyncio.run(main())
    assert max(seen) == 3 and board.snapshot().in_flight == 0


def test_thread_safe_under_concurrent_tools():
    board = ToolWaitBoard()

    def worker(k):
        for i in range(200):
            with tool_wait_scope(board, f"{k}-{i}"):
                pass

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    snap = board.snapshot()
    assert snap.in_flight == 0 and snap.entered_total == snap.exited_total == 1600


def test_drain_needs_zero_requests_and_zero_tool_waits_and_fails_closed():
    board = ToolWaitBoard()
    assert drain_blockers(0, board.snapshot()) == []
    board.enter("t")
    # X5: active requests 0 but a trajectory waits on a tool -> not drained
    assert drain_blockers(0, board.snapshot()) == ["1 trajectories waiting on tools"]
    assert "3 engine requests" in drain_blockers(3, board.snapshot())[0]
    assert drain_blockers(None, None) == ["router in-flight count unknown",
                                          "tool-wait count unknown"]
    # the same count feeds the 1.7 load attribution
    sample = LoadSample(queued_requests=0, active_requests=0,
                        tool_wait_trajectories=board.snapshot().in_flight,
                        ready_groups=0, engine_capacity=4)
    assert classify_load(sample) == "tool-wait"


class _FakeRemoteMethod:
    def __init__(self, fn):
        self.fn = fn

    def remote(self, *args):
        return self.fn(*args)


class _FakeActorHandle:
    """Shape of a Ray actor handle: methods only callable via ``.remote``."""

    def __init__(self, board):
        for name in ("enter", "exit", "snapshot"):
            setattr(self, name, _FakeRemoteMethod(getattr(board, name)))


def test_actor_handle_shape_is_supported():
    board = ToolWaitBoard()
    handle = _FakeActorHandle(board)
    with tool_wait_scope(handle, "x"):
        assert read_tool_wait(handle).in_flight == 1
    assert read_tool_wait(handle).in_flight == 0
