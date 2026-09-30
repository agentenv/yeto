"""rl-infra-spec 2.3: guards of the age-0 eval||train/outer_sync overlap (pure, CPU)."""

from __future__ import annotations

import asyncio

import pytest

from yeto.rl.engine.execution_profile import (
    ExecutionProfile,
    ProfileError,
    ReadinessSnapshot,
    overlap_violation,
    quiescent_cut_blockers,
)
from yeto.rl.engine.overlap import (
    IMPLEMENTED_OVERLAP,
    EvalOverlap,
    OverlapGuardError,
    loop_eval_starter,
    overlap_refusal,
)

SHA = "a" * 64


def _profile(pairs=IMPLEMENTED_OVERLAP, **kw):
    return ExecutionProfile(name="o", execution_mode="partitioned-overlap",
                            outer_protocol="strict-avg", allowed_overlap=pairs,
                            algorithm_spec_sha256=SHA, **kw)


class _Handle:
    def __init__(self, log, rid):
        self.log, self.rid = log, rid

    def result(self):
        self.log.append(("result", self.rid))
        return {"score": 1.0}


def _overlap(log):
    t = iter(range(100))
    return EvalOverlap(lambda rid: (log.append(("start", rid)), _Handle(log, rid))[1],
                       emit=lambda *a, **k: log.append(("emit", a[0])),
                       clock=lambda: float(next(t)))


def test_implemented_pairs_are_legal_at_age_0_and_generation_pairs_are_not():
    p = _profile()
    for a, b in IMPLEMENTED_OVERLAP:
        assert overlap_violation(p, a, b) is None
    for other in ("train", "outer_sync", "publish"):
        assert "age 0" in overlap_violation(p, "generate", other)
    assert "rollout" in overlap_violation(p, "eval", "generate")
    assert overlap_violation(p, "eval", "publish")  # eval[r] waits for publish[r]
    assert overlap_refusal(p) is None
    assert "not the implemented" in overlap_refusal(_profile({("eval", "train")}))
    assert "not the implemented" in overlap_refusal(_profile({("reward", "checkpoint")}))
    with pytest.raises(ProfileError, match="exactly one batch"):
        _profile(max_inflight_batches=2)
    with pytest.raises(ProfileError):
        _profile(max_policy_age=1)


def test_eval_starts_after_generate_and_is_joined_before_publish():
    log = []
    ov = _overlap(log)
    ov.schedule(3, token="t3", published_version=3)
    ov.before_generate(3)  # nothing in flight yet: generation may start
    assert log == []
    ov.after_generate(3, token="t3", published_version=3)
    assert ov.in_flight == 1 and ("start", 3) in log
    with pytest.raises(OverlapGuardError, match="rollout role"):
        ov.before_generate(4)
    done = ov.before_publish(token="t3", published_version=3)
    assert done["rollout_id"] == 3 and done["metrics"] == {"score": 1.0}
    assert done["start"] < done["end"] and ov.in_flight == 0
    assert ov.before_publish(token="t3", published_version=3) is None


def test_queue_is_bounded_and_eval_needs_its_own_publication():
    ov = _overlap([])
    with pytest.raises(OverlapGuardError, match="complete publication"):
        ov.schedule(2, token=None, published_version=2)
    with pytest.raises(OverlapGuardError, match="complete publication"):
        ov.schedule(2, token="t1", published_version=1)
    ov.schedule(2, token="t2", published_version=2)
    with pytest.raises(OverlapGuardError, match="bound 1"):
        ov.schedule(2, token="t2", published_version=2)
    ov.after_generate(2, token="t2", published_version=2)
    with pytest.raises(OverlapGuardError, match="bound 1"):
        ov.schedule(3, token="t3", published_version=3)


def test_eval_never_runs_on_another_version():
    ov = _overlap([])
    ov.schedule(2, token="t2", published_version=2)
    with pytest.raises(OverlapGuardError, match="rollout GPUs hold"):
        ov.after_generate(2, token="t3", published_version=3)
    ov = _overlap([])
    ov.schedule(2, token="t2", published_version=2)
    ov.after_generate(2, token="t2", published_version=2)
    with pytest.raises(OverlapGuardError, match="moved to v3"):
        ov.before_publish(token="t3", published_version=3)


def test_eval_in_flight_blocks_a_quiescent_cut():
    snap = ReadinessSnapshot(rollout_id=1, optimizer_step=1, trained_policy_version=1,
                             published_policy_version=1, publication_complete=True,
                             driver_safe_point=True, eval_in_flight=1)
    assert any("overlapped evals" in r for r in quiescent_cut_blockers(_profile(), snap))


def test_loop_handle_progresses_cooperatively_on_the_island_loop():
    from yeto.rl.engine.miles_adapter import LoopRunner

    runner = LoopRunner(asyncio.new_event_loop())
    order = []

    async def evaluate(rid):
        order.append(("eval-begin", rid))
        await asyncio.sleep(0)
        order.append(("eval-end", rid))
        return {"acc": 0.5}

    async def train():
        order.append("train-begin")
        await asyncio.sleep(0.01)
        order.append("train-end")

    handle = loop_eval_starter(runner, evaluate)(7)
    runner.run(train())  # the trainer drives the same loop; eval progresses meanwhile
    # eval ran to completion inside the trainer's loop drive (before train ended)
    assert order.index(("eval-end", 7)) < order.index("train-end")
    assert order.index("train-begin") < order.index("train-end")
    assert handle.result() == {"acc": 0.5}
    runner.close()
