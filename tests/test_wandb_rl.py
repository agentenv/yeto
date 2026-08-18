"""RL event tape -> W&B.

The RL island's tape is the instrumentation; W&B is a second reader of it.
These pin the projection from tape event to metrics, and that a failing
sink never reaches the RL loop.
"""

import argparse

import pytest

from yeto.rl.wandb_rl import RlTelemetry, event_metrics, tee

LOCAL_ROUND = {
    "event": "rl_local_round",
    "island_id": 1,
    "time_unix": 1.0,
    "rl/rollout_id": 7,
    "rl/reward_mean": 0.42,
    "rl/reward_std": 0.1,
    "rl/action_tokens": 2048,
    "rl/current_vs_rollout_kl": 0.03,
    "rl/ess_ratio": 0.87,
    "rl/clip_fraction": 0.12,
    "rl/zero_variance_group_ratio": 0.0,
    "rl/policy_hash": "deadbeef",
    "rl/fragment_versions": [3, 3, 2, 2],
    "sync/applied_fragments": [0, 1],
    "sync/fragment_payload_bytes_sent": 4096,
}


def test_the_rl_signals_that_matter_survive_the_projection():
    m = event_metrics(LOCAL_ROUND)
    for key in (
        "rl/reward_mean", "rl/reward_std", "rl/action_tokens",
        "rl/current_vs_rollout_kl", "rl/ess_ratio", "rl/clip_fraction",
        "sync/fragment_payload_bytes_sent",
    ):
        assert m[key] == LOCAL_ROUND[key], key
    assert m["rl/rollout_id"] == 7  # the x-axis
    assert m["event/rl_local_round"] == 1


def test_a_zero_valued_metric_is_kept():
    # zero_variance_group_ratio == 0 is the healthy reading, not a missing one.
    assert event_metrics(LOCAL_ROUND)["rl/zero_variance_group_ratio"] == 0.0


def test_identifiers_are_not_logged_as_series():
    m = event_metrics(LOCAL_ROUND)
    # A policy hash has no magnitude; charting it is meaningless.
    assert "rl/policy_hash" not in m
    assert "island_id" not in m
    assert "time_unix" not in m


def test_lists_become_counts():
    m = event_metrics(LOCAL_ROUND)
    assert m["rl/fragment_versions_count"] == 4
    assert m["sync/applied_fragments_count"] == 2
    assert "rl/fragment_versions" not in m


def test_booleans_become_numbers_so_they_can_be_charted():
    m = event_metrics({"event": "e", "rl/rollout_id": 1, "flag": True})
    assert m["flag"] == 1


def test_a_push_event_projects_its_own_shape():
    m = event_metrics({
        "event": "rl_fragment_push", "island_id": 0, "time_unix": 2.0,
        "fragment_id": 2, "global_step": 9, "round_attempt": 1,
        "base_version": 7, "c_steps": 4, "c_tokens": 900,
        "delta_l2_norm": 1.25, "payload_bytes": 512,
        "pull_to_push_seconds": 0.4, "realized_h": 4,
    })
    assert m["delta_l2_norm"] == 1.25
    assert m["payload_bytes"] == 512
    assert m["pull_to_push_seconds"] == 0.4
    assert m["global_step"] == 9


def test_a_strict_failure_is_visible_as_an_event_counter():
    m = event_metrics({
        "event": "rl_strict_failure", "island_id": 0, "time_unix": 3.0,
        "metric": "reward_variance", "value": 1,
        "error": "StrictRlInvariantError: ...",
    })
    assert m["event/rl_strict_failure"] == 1
    assert m["value"] == 1
    assert "error" not in m  # a message, not a measurement


def test_an_event_with_nothing_measurable_is_dropped():
    assert event_metrics({"event": "noise", "island_id": 0, "time_unix": 1.0}) is None


def test_tee_is_inert_without_the_flag():
    calls = []
    telemetry = RlTelemetry()
    telemetry.log = lambda *a, **k: calls.append(a)
    tee(argparse.Namespace(wandb=False), LOCAL_ROUND)
    assert calls == []


def test_a_broken_sink_never_reaches_the_rl_loop(monkeypatch):
    import yeto.rl.wandb_rl as mod

    class _Boom:
        def log(self, args, event):
            raise RuntimeError("wandb is down")

    monkeypatch.setattr(mod, "_TELEMETRY", _Boom())
    # Must not raise: the RL round that produced this event has to continue.
    tee(argparse.Namespace(wandb=True), LOCAL_ROUND)


def test_the_island_run_joins_the_fleet_group(monkeypatch):
    import yeto.rl.wandb_rl as mod

    seen = {}

    def fake_init(args, **kwargs):
        seen.update(kwargs)
        class _Run:
            enabled = True
            def log(self, m): seen.setdefault("logged", []).append(m)
            def finish(self, exit_code=0): pass
        return _Run()

    monkeypatch.setattr(mod, "init", fake_init)
    telemetry = RlTelemetry()
    args = argparse.Namespace(
        wandb=True, yeto_rl_learner_id=3, yeto_rl_sync_preset="decoupled",
        yeto_rl_model="Qwen/Qwen2.5-0.5B", yeto_rl_data="d",
        yeto_rl_base_model_revision="abc", yeto_rl_reward_sha256="sha",
    )
    telemetry.log(args, LOCAL_ROUND)
    assert seen["job_type"] == "rl-learner"
    assert seen["name"] == "learner-3"
    assert seen["config_extra"]["island_backend"] == "rl-miles"
    assert seen["config_extra"]["rl_sync_preset"] == "decoupled"
    assert seen["step_metrics"]["rl/*"] == "rl/rollout_id"
    assert seen["logged"][0]["rl/reward_mean"] == 0.42


def test_the_run_starts_once_across_many_events(monkeypatch):
    import yeto.rl.wandb_rl as mod

    inits = []

    def fake_init(args, **kwargs):
        inits.append(1)
        class _Run:
            enabled = True
            def log(self, m): pass
            def finish(self, exit_code=0): pass
        return _Run()

    monkeypatch.setattr(mod, "init", fake_init)
    telemetry = RlTelemetry()
    args = argparse.Namespace(wandb=True, yeto_rl_learner_id=0)
    for _ in range(5):
        telemetry.log(args, LOCAL_ROUND)
    assert len(inits) == 1
