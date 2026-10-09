"""S17 N13: per-trajectory grading time (``evaluate_time``) reaches the event."""

from __future__ import annotations

import asyncio

from yeto.rl.engine import timeline
from yeto.rl.engine.miles_adapter.rollout_meta_hook import trajectory_diagnostics
from yeto.rl.harness.codex import codex_openenv_agent_function as F


class _Verifier:
    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.calls = 0

    async def evaluate(self, episode_id):
        self.calls += 1
        await asyncio.sleep(self.delay)
        return {"passed": True, "testsh_rc": 0, "log": "ok"}


def _finish(status, verifier):
    untrusted = {"status": status, "episode_id": "ep", "metrics": {"turns": 2, "evaluate_time": 0.0}}
    return asyncio.run(F.finish_trusted(untrusted, verifier, task_id="t", sample_id="s", key=b"k" * 32))


def test_finish_trusted_records_evaluate_time():
    v = _Verifier(0.05)
    meta = _finish("completed", v)
    et = meta["agent_metrics"]["evaluate_time"]
    assert v.calls == 1 and isinstance(et, float) and 0.04 <= et < 5
    assert meta["agent_metrics"]["turns"] == 2
    out = trajectory_diagnostics(meta)
    assert out["evaluate_time"] == round(et, 3)
    record = {"rollout_id": 0, "policy_version": 0, "sample_index": 0, "group_index": 0, "task_id": "t",
              "trajectory_id": "x", "reward": 1.0, "success": True, "aborted": False, **out}
    assert timeline.validate_trajectory_reward(record) == []
    assert timeline.validate_trajectory_reward({**record, "evaluate_time": "1"})


def test_timeout_skips_grading_and_drops_worker_value():
    v = _Verifier(0.0)
    meta = _finish("timeout", v)
    assert v.calls == 0 and "evaluate_time" not in meta["agent_metrics"]
    assert "evaluate_time" not in trajectory_diagnostics(meta)


def test_hook_ignores_bad_evaluate_time():
    for bad in (True, -1.0, "3", None):
        assert "evaluate_time" not in trajectory_diagnostics({"agent_metrics": {"evaluate_time": bad}})
    assert trajectory_diagnostics({"agent_metrics": {"evaluate_time": 2}})["evaluate_time"] == 2.0
