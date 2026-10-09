"""agentic-rollout-utilization 6.2/6.3: verl adapter fields, refusals, limit 0 unchanged."""

from __future__ import annotations

import pytest

from yeto.rl.engine.policy_age import PolicyAgeError
from yeto.rl.engine.rollout_cutoff import CUTOFF_FIELDS
from yeto.rl.engine.timeline import TRAJECTORY_REWARD_OPTIONAL
from yeto.rl.engine.trajectory_timing import TIMING_FIELDS, phase_totals


def test_verl_and_miles_report_the_same_trajectory_fields():
    from yeto.rl.adapters.verl.rollout_events import trajectory_fields
    from yeto.rl.engine.trajectory_timing import PhaseClock

    verl = trajectory_fields({"generate_sequences": 4.0, "tool_calls": 2.5, "compute_score": 0.5,
                              "num_preempted": 1},
                             {"min_global_steps": 3, "max_global_steps": 4})
    assert verl == {"generation_seconds": 4.0, "tool_seconds": 2.5, "evaluate_time": 0.5,
                    "policy_versions": [3, 4]}
    ticks = iter([0.0, 4.0, 6.5, 6.5])
    clock = PhaseClock(lambda: next(ticks))
    clock.enter_tool()
    clock.exit_tool()
    miles = {**clock.finish(), "evaluate_time": 0.5}
    shared = {"generation_seconds", "tool_seconds", "evaluate_time"}
    assert shared <= set(miles) and shared <= set(verl)
    assert phase_totals(verl) == phase_totals(miles)  # same meaning, same totals
    for key in ("generation_seconds", "tool_seconds", "policy_versions", "evaluate_time"):
        assert key in TRAJECTORY_REWARD_OPTIONAL
    assert {"generation_seconds", "tool_seconds"} <= set(TIMING_FIELDS)
    assert trajectory_fields({"generate_sequences": "x"}, {"min_global_steps": 5,
                                                            "max_global_steps": 4}) == {}


def test_verl_discard_tally_uses_the_miles_cutoff_fields():
    from yeto.rl.adapters.verl.rollout_events import cutoff_fields
    from yeto.rl.engine.rollout_cutoff import cutoff_report
    from types import SimpleNamespace

    fields = cutoff_fields({"groups": 2, "samples": 8, "response_tokens": 640, "unknown_groups": 0})
    report = cutoff_report(SimpleNamespace(groups=(), **fields))
    assert tuple(report.fields()) == CUTOFF_FIELDS
    assert (report.discarded_groups, report.discarded_trajectories, report.discarded_tokens) == (2, 8, 640)


def test_verl_refuses_over_sampling_before_launch():
    from tests.test_rl_launcher import _args
    from yeto.launcher import _prepare_rl_args

    args = _args(["--over-sampling-batch-size", "16"])
    args.rl_backend = "verl"
    with pytest.raises(ValueError, match="verl 后端尚未实现多发截止"):
        _prepare_rl_args(args)


def test_verl_declares_stage_two_and_limit_zero_overrides_are_empty():
    from yeto.rl.adapters.verl.entry import verl_capabilities
    from yeto.rl.adapters.verl.policy_age import SUPPORT, policy_age_overrides

    assert (SUPPORT.stage, SUPPORT.max_policy_age) == (2, 1)  # 6.4b: fully_async path
    assert verl_capabilities("sha256:" + "0" * 64).execution.max_policy_staleness == 0
    assert policy_age_overrides(0) == ()
    with pytest.raises(PolicyAgeError, match="backend 'verl' supports up to stage 2"):
        policy_age_overrides(2, groups_per_round=4)


def test_launcher_accepts_verl_limit_one():
    from tests.test_rl_launcher import _args
    from yeto.launcher import _prepare_rl_args

    args = _args(["--rl-max-policy-age", "1"])
    args.rl_backend = "verl"
    _prepare_rl_args(args)
    assert args.rl_max_policy_age == 1


def test_verl_command_line_unchanged_at_limit_zero():
    """6.3: the verl overrides do not depend on the limit at 0 (byte-identical)."""
    import yeto.rl.adapters.verl.config as vconf

    run = vconf.VerlRunConfig(model_path="/m", train_file="/t", val_file="/v", out_dir="/o")
    from yeto.rl.adapters.verl.policy_age import policy_age_overrides

    assert vconf.build_overrides(run) + list(policy_age_overrides(0)) == vconf.build_overrides(run)
    assert not any("staleness" in o or "partial_rollout" in o for o in vconf.build_overrides(run))
