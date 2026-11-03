"""A reclaimed engine sampler exits within its deadline (CPU).

Fakes: the engine (``FakeEngine``), a harness whose trial blocks in a worker
thread the way a boat sandbox call does (``asyncio.to_thread``) and never
returns, and the boat pool cleanup. Real pieces: ``AgenticRound``,
``ReclaimDeadline``, ``hold_after_reclaim`` and ``main``.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest

from yeto.rl.agentic import TrialVerdict
from yeto.rl.engine import engine_sampler_entry as entry
from yeto.rl.engine.sampler_resume import RECLAIM_MARKER_ENV

from test_agentic_round_streaming import Prompts, _entries, _round


class BlockingSandboxHarness:
    """``fast`` ends at once; ``stuck`` blocks in a worker thread until ``release`` is set."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.blocked = threading.Event()
        self.started: list[str] = []

    async def run(self, task, endpoint):
        self.started.append(task.trajectory_id)
        if "-stuck-" in task.trajectory_id:
            def sandbox_call() -> None:
                self.blocked.set()
                self.release.wait()

            await asyncio.to_thread(sandbox_call)
        return TrialVerdict(rewards={"reward": 1.0})


def test_a_stuck_trial_is_dropped_the_finished_group_kept_and_the_process_exits_in_time(tmp_path, monkeypatch):
    from yeto.rl.harness.harbor import boat_pool

    swept: list[bool] = []
    monkeypatch.setattr(boat_pool, "close_pools_now", lambda: swept.append(True) or 3)
    harness = BlockingSandboxHarness()
    leaving = {"now": False}
    rnd, engine, events = _round(tmp_path, harness, Prompts({0: ["stuck", "fast"], 1: ["next"]}),
                                 reclaimed=lambda: leaving["now"])
    exits: list[tuple[int, float]] = []
    deadline = entry.ReclaimDeadline(deadline_s=1.0, margin_s=0.2, cleanup_s=0.1,
                                     exit_now=lambda code: exits.append((code, time.time())))
    deadline.add_cleanup("boat_sandboxes", boat_pool.close_pools_now)
    deadline.add_cleanup("engine", lambda: threading.Event().wait())  # a cleanup that blocks too
    try:
        rnd(0)  # the fast group is written, the stuck one stays in flight
        assert harness.blocked.wait(5.0)
        marker_at = time.time()
        leaving["now"] = True
        deadline.arm(marker_at)
        rnd(1)  # reclaimed: returns at once, starts nothing new
    finally:
        rnd.close()
    try:
        assert not any("-next-" in s for s in harness.started)
        assert swept, "the round's close deletes the boat sandboxes"
        # the blocked sandbox call still holds an executor worker, which the interpreter joins at exit
        assert any(t.name.startswith("ThreadPoolExecutor") for t in threading.enumerate())
        deadline.join(5.0)
        assert exits and exits[0][0] == entry.RECLAIM_EXIT_CODE
        assert exits[0][1] - marker_at <= 1.0
        assert len(swept) == 2  # the deadline swept the sandboxes once more
        entries, payloads = _entries(tmp_path)
        assert [e.sample_id for e in entries] == ["s-r0-fast"]
        assert all(len(p) == 2 for p in payloads.values())
    finally:
        harness.release.set()


def test_the_deadline_does_nothing_once_cancelled():
    exits: list[int] = []
    deadline = entry.ReclaimDeadline(deadline_s=0.2, margin_s=0.0, exit_now=exits.append).arm()
    deadline.cancel()
    deadline.join(2.0)
    assert exits == [] and not deadline.fired.is_set()


def test_the_hold_ends_at_the_deadline_counted_from_the_marker(tmp_path):
    marker = tmp_path / "marker"
    marker.write_text(json.dumps({"signal": 15, "time": time.time() - 200.0}))
    env = {RECLAIM_MARKER_ENV: str(marker)}
    slept: list[float] = []
    deadline = entry.ReclaimDeadline(exit_now=lambda code: None)
    try:
        assert entry.hold_after_reclaim(env, sleep=slept.append, deadline=deadline) is True
        # 300 s budget, 200 s gone, 15 s margin: the hold is at most 85 s, not 300 s
        assert len(slept) == 1 and 80.0 <= slept[0] <= 85.0
    finally:
        deadline.cancel()


def test_main_exits_with_the_reclaim_code_and_log(tmp_path, monkeypatch, capsys):
    marker = tmp_path / "marker"
    monkeypatch.setenv(RECLAIM_MARKER_ENV, str(marker))
    spec = tmp_path / "spec.json"
    spec.write_text("{}")

    armed = []

    def fake_run(spec_doc, *, reclaim_deadline):
        armed.append(reclaim_deadline)
        marker.write_text(json.dumps({"signal": 15, "time": time.time()}))
        reclaim_deadline.arm(entry.reclaim_seen_at({RECLAIM_MARKER_ENV: str(marker)}))
        return 1

    class Exited(Exception):
        pass

    def fake_exit(code):
        raise Exited(code)

    monkeypatch.setattr(entry, "run", fake_run)
    monkeypatch.setattr(entry.time, "sleep", lambda s: None)
    monkeypatch.setattr(entry.os, "_exit", fake_exit)
    try:
        with pytest.raises(Exited) as caught:
            entry.main(["--spec", str(spec)])
    finally:
        for deadline in armed:
            deadline.cancel()
    assert caught.value.args == (entry.RECLAIM_EXIT_CODE,)
    assert f"{entry.RECLAIM_EXIT_LOG} code={entry.RECLAIM_EXIT_CODE} reason=held" in capsys.readouterr().out


def test_main_without_a_reclaim_returns_zero(tmp_path, monkeypatch):
    monkeypatch.setenv(RECLAIM_MARKER_ENV, str(tmp_path / "marker"))
    spec = tmp_path / "spec.json"
    spec.write_text("{}")
    monkeypatch.setattr(entry, "run", lambda spec_doc, *, reclaim_deadline: 2)
    assert entry.main(["--spec", str(spec)]) == 0


def _armed_run(error):
    armed = []

    def fake_run(spec_doc, *, reclaim_deadline):
        armed.append(reclaim_deadline.arm())
        raise error

    return armed, fake_run


def test_an_error_after_the_reclaim_exits_with_the_reclaim_code_and_names_the_error(tmp_path, monkeypatch,
                                                                                     capsys):
    spec = tmp_path / "spec.json"
    spec.write_text("{}")
    armed, fake_run = _armed_run(ConnectionResetError("engine gone"))
    exits: list[int] = []
    monkeypatch.setattr(entry, "run", fake_run)
    monkeypatch.setattr(entry.os, "_exit", exits.append)
    try:
        with pytest.raises(ConnectionResetError):
            entry.main(["--spec", str(spec)])
    finally:
        for deadline in armed:
            deadline.cancel()
    assert exits == [entry.RECLAIM_EXIT_CODE]
    assert "builtins.ConnectionResetError: engine gone" in capsys.readouterr().out


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(3)])
def test_an_interrupt_after_the_reclaim_is_not_turned_into_a_reclaim_exit(tmp_path, monkeypatch, error):
    spec = tmp_path / "spec.json"
    spec.write_text("{}")
    armed, fake_run = _armed_run(error)
    exits: list[int] = []
    monkeypatch.setattr(entry, "run", fake_run)
    monkeypatch.setattr(entry.os, "_exit", exits.append)
    try:
        with pytest.raises(type(error)):
            entry.main(["--spec", str(spec)])
    finally:
        for deadline in armed:
            deadline.cancel()
    assert exits == []
