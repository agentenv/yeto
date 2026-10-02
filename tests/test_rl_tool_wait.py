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
    HARNESS_ZERO,
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
    assert drain_blockers(0, board.snapshot(), HARNESS_ZERO) == []
    board.enter("t")
    # X5: active requests 0 but a trajectory waits on a tool -> not drained
    assert drain_blockers(0, board.snapshot(), HARNESS_ZERO) == ["1 trajectories waiting on tools"]
    assert "3 engine requests" in drain_blockers(3, board.snapshot(), HARNESS_ZERO)[0]
    assert drain_blockers(None, None, None) == ["router in-flight count unknown",
                                                "tool-wait count unknown",
                                                "harness counts unknown"]
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


# ------------------------------------------------ 3.3 X5 evidence: tool side-effect journal
def test_side_effect_log_is_append_only_monotonic_and_restart_safe(tmp_path):
    from yeto.rl.engine.tool_wait import ToolSideEffectLog, read_side_effects, side_effect_duplicates

    path = tmp_path / "elastic-state" / "side_effects.jsonl"
    log = ToolSideEffectLog(path, clock=lambda: 100.0, monotonic=lambda: 5.0)
    assert log.records() == [] and log.seq == 0
    r1 = log.record("t1", "c1", seconds=3.0)
    r2 = log.complete("t1", "c1")
    assert (r1["kind"], r1["seq"], r1["trajectory_id"], r1["tool_call_id"], r1["wall_time"], r1["seconds"]) == \
        ("tool_side_effect", 1, "t1", "c1", 100.0, 3.0)
    assert (r2["kind"], r2["seq"]) == ("tool_complete", 2)
    assert [r["seq"] for r in read_side_effects(path)] == [1, 2]
    # a restarted process continues the sequence
    log2 = ToolSideEffectLog(path)
    assert log2.seq == 2 and log2.record("t2", "c2")["seq"] == 3
    recs = read_side_effects(path)
    assert side_effect_duplicates(recs) == []
    log2.record("t1", "c1")  # a replay
    assert side_effect_duplicates(read_side_effects(path)) == [("t1", "c1")]
    # a torn trailing line is ignored, the rest is read
    with open(path, "a") as fh:
        fh.write('{"kind": "tool_side_eff')
    assert len(read_side_effects(path)) == 4
    assert read_side_effects(tmp_path / "missing.jsonl") == []


def test_scopes_journal_before_entering_and_complete_after_exit(tmp_path):
    import asyncio

    from yeto.rl.engine.tool_wait import ToolSideEffectLog, async_tool_wait_scope

    board = ToolWaitBoard()
    log = ToolSideEffectLog(tmp_path / "se.jsonl")
    with tool_wait_scope(board, "t1", log, "call-a"):
        recs = log.records()
        assert [(r["kind"], r["tool_call_id"]) for r in recs] == [("tool_side_effect", "call-a")]
        assert read_tool_wait(board).in_flight == 1
    assert [r["kind"] for r in log.records()] == ["tool_side_effect", "tool_complete"]

    async def run():
        async with async_tool_wait_scope(board, "t2", log):  # call id defaults to the trajectory id
            assert read_tool_wait(board).in_flight == 1
            assert log.records()[-1]["tool_call_id"] == "t2"

    asyncio.run(run())
    assert [r["kind"] for r in log.records()][-2:] == ["tool_side_effect", "tool_complete"]
    assert read_tool_wait(board).in_flight == 0
    with tool_wait_scope(board, "t3"):  # no log: unchanged behaviour
        pass
    assert len(log.records()) == 4
