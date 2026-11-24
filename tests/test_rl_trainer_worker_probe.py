"""A train rank that dies while the round waits for its batch ends the island.

A fleet GPU run: island 0's trainer waited for sampler groups (the driver blocked in
``rollout.generate``, the buffer reader's take in the rollout executor) when one of
its two Megatron train actors was killed. Nothing called the train actors during the
wait, so the island neither failed nor was relaunched; it would have waited until
the sample-wait limit (3600 s) and the next train call. The driver now probes the
train ranks while it waits (``IslandDriver.trainer_probe_s``, Miles
``MilesTrainerGroup.watch_workers``): a dead rank interrupts the wait and the island
leaves with RECOVERY_REQUIRED (INFRA_FAILED), which the launcher relaunches.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from types import SimpleNamespace

import pytest

from yeto.rl.adapters.miles import sampler_island
from yeto.rl.adapters.miles.state_plugin import PING_WORKER
from yeto.rl.adapters.miles.trainer import MilesTrainerGroup
from yeto.rl.engine.driver import IslandRecoveryRequired
from yeto.rl.engine.leave_reason import leave_reason_for
from yeto.rl.engine.loop_runner import LoopRunner
from yeto.rl.leave_code import LeaveReason

from test_rollout_ahead import _driver


class _Actor:
    """The upstream actor group: ``run_plugin`` answers until a rank is killed."""

    def __init__(self):
        self.dead = False
        self.pings = 0

    async def run_plugin(self, fn_path, kwargs=None):
        assert fn_path == PING_WORKER
        self.pings += 1
        await asyncio.sleep(0)
        if self.dead:
            raise RuntimeError("ActorDiedError: The actor died unexpectedly before finishing this task "
                               "(class_name: MegatronTrainRayActor, pid: 3263)")
        return [True, True]


def _group(runner, actor):
    return MilesTrainerGroup(args=SimpleNamespace(), actor_model=actor, learner_id=0, learner_generation=0,
                             parameter_layout_hash=lambda: "layout", runner=runner)


def _kill_after(actor, seconds):
    timer = threading.Timer(seconds, lambda: setattr(actor, "dead", True))
    timer.daemon = True
    timer.start()
    return timer


def test_a_rank_killed_during_the_wait_interrupts_it():
    runner, actor = LoopRunner(), _Actor()
    group = _group(runner, actor)
    _kill_after(actor, 0.2)
    started = time.monotonic()
    with pytest.raises(asyncio.CancelledError):
        with group.watch_workers(0.05, timeout_s=5.0) as loss:
            runner.run(asyncio.sleep(60))  # the take waits for groups that never come
    assert time.monotonic() - started < 10
    assert "ActorDiedError" in loss.reason and "MegatronTrainRayActor" in loss.reason
    runner.run(asyncio.sleep(0))  # the island loop is usable again


def test_live_ranks_are_probed_and_the_wait_is_untouched():
    runner, actor = LoopRunner(), _Actor()
    group = _group(runner, actor)
    with group.watch_workers(0.02, timeout_s=5.0) as loss:
        assert runner.run(asyncio.sleep(0.3, result="batch")) == "batch"
    assert loss.reason is None and actor.pings >= 2


def test_a_probe_without_an_answer_is_no_loss():
    runner = LoopRunner()

    class _Slow(_Actor):
        async def run_plugin(self, fn_path, kwargs=None):
            await asyncio.sleep(60)

    group = _group(runner, _Slow())
    with group.watch_workers(0.02, timeout_s=0.05) as loss:
        runner.run(asyncio.sleep(0.4))
    assert loss.reason is None
    # the loop is not running: the ping cannot be answered, which is no loss either
    assert group.probe_workers(timeout_s=0.05) is None


def _blocking_generate(engine, runner, actor, rollout_id, kill_after_s=0.1):
    """``engine.rollout.generate`` that, for ``rollout_id``, waits on the island loop
    (as Miles' ``executor.get`` does) while a rank is killed."""
    real = engine.rollout.generate

    def generate(rid, **kwargs):
        if rid == rollout_id:
            _kill_after(actor, kill_after_s)
            runner.run(asyncio.sleep(3600))
        return real(rid, **kwargs)

    engine.rollout.generate = generate


def test_the_island_leaves_recoverable_when_a_rank_dies_while_it_waits(tmp_path):
    driver, engine = _driver(tmp_path, rounds=4, ahead=0)
    runner, actor = LoopRunner(), _Actor()
    engine.trainer.watch_workers = _group(runner, actor).watch_workers
    driver.trainer_probe_s = 0.05
    _blocking_generate(engine, runner, actor, rollout_id=2)
    started = time.monotonic()
    with pytest.raises(IslandRecoveryRequired) as raised:
        driver.run()
    assert time.monotonic() - started < 30
    assert leave_reason_for(raised.value) is LeaveReason.INFRA_FAILED  # the launcher relaunches it
    events = [json.loads(line) for line in (tmp_path / "e.jsonl").read_text().splitlines()]
    lost = [e for e in events if e["event"] == "rl_reconfiguration"]
    assert lost and lost[-1]["result"] == "RECOVERY_REQUIRED" and lost[-1]["cause"] == "trainer_worker_lost"
    assert lost[-1]["rollout_id"] == 2 and lost[-1]["where"] == "generate"
    leave = [e for e in events if e["event"] == "island_leave"]
    assert leave and leave[-1]["reason_name"] == "INFRA_FAILED"
    assert [c[1] for c in engine.calls if c[0] == "train"] == [0, 1]  # nothing trained after the loss


def test_a_rank_lost_while_the_batch_generated_ahead_is_joined(tmp_path):
    driver, engine = _driver(tmp_path, rounds=4, ahead=1)
    runner, actor = LoopRunner(), _Actor()
    engine.trainer.watch_workers = _group(runner, actor).watch_workers
    driver.trainer_probe_s = 0.05
    real = engine.rollout.finish_generate

    def finish(pending):
        if pending.rollout_id == 2:
            _kill_after(actor, 0.1)
            runner.run(asyncio.sleep(3600))
        return real(pending)

    engine.rollout.finish_generate = finish
    with pytest.raises(IslandRecoveryRequired):
        driver.run()
    events = [json.loads(line) for line in (tmp_path / "e.jsonl").read_text().splitlines()]
    lost = [e for e in events if e["event"] == "rl_reconfiguration"]
    assert lost[-1]["where"] == "generate_wait" and lost[-1]["rollout_id"] == 2


def test_without_a_probe_interval_the_trainer_is_never_probed(tmp_path):
    driver, engine = _driver(tmp_path, rounds=3, ahead=0)
    calls = []

    def watch_workers(interval_s):  # pragma: no cover - must not be called
        calls.append(interval_s)
        raise AssertionError("probed")

    engine.trainer.watch_workers = watch_workers
    assert driver.trainer_probe_s is None  # the default: unchanged behavior
    driver.run()
    assert calls == []


@pytest.mark.parametrize("raw, expected", [(None, 30.0), ("", 30.0), ("5", 5.0), ("0", None)])
def test_trainer_probe_interval(raw, expected):
    env = {} if raw is None else {sampler_island.TRAINER_PROBE_ENV: raw}
    assert sampler_island.trainer_probe_s(env) == expected


def test_a_negative_probe_interval_is_refused():
    with pytest.raises(ValueError, match="YETO_RL_TRAINER_PROBE_S"):
        sampler_island.trainer_probe_s({sampler_island.TRAINER_PROBE_ENV: "-1"})
