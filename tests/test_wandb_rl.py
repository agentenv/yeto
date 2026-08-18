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


# --------------------------------------------------------------------------
# the SSH harness builds its own learner command, so it needs its own wiring


def _harness_plan(**learner):
    base = dict(
        model="/m", model_revision="r" * 40, data="/workspace/data/dataset.jsonl",
        reward_function="pkg.mod:f", global_rounds=1, fragments=1, pipeline=1,
        local_horizon=1, total_fragment_steps=1, groups_per_round=2,
        samples_per_group=3, over_sampling_batch_size=2, optimizer_steps=1,
        rollout_max_response_len=4096, tensor_parallel=8, pipeline_parallel=1,
        rollout_num_gpus_per_engine=8, sglang_mem_fraction_static=0.45,
        lora_r=8, lora_targets="attention", inner_lr=1e-5, seq_len=8192,
        seed=1, wan_streams=1, trust_remote_code=True,
        rl_model_recipe="generic", actor_num_nodes=1,
    )
    base.update(learner)
    return {
        "run_id": "fleet-x",
        "syncer_address": "h:29400",
        "reward_sha256": "a" * 64,
        "source_sha256": "b" * 64,
        "islands": [
            {"hosts": ["root@a"], "gpus_per_node": 8},
            {"hosts": ["root@b"], "gpus_per_node": 8},
        ],
        "learner": base,
    }


def _wandb_flags(argv):
    out = []
    for i, a in enumerate(argv):
        if a.startswith("--wandb"):
            out.append(a)
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                out.append(argv[i + 1])
    return out


def test_the_harness_forwards_the_flags_to_both_islands():
    from yeto.rl.ssh_harness import _learner_argv

    plan = _harness_plan(wandb=True, wandb_project="p", wandb_entity="e", wandb_mode="online")
    for learner_id in (0, 1):
        flags = _wandb_flags(_learner_argv(plan, learner_id))
        assert flags == ["--wandb", "--wandb-project", "p", "--wandb-mode", "online",
                         "--wandb-entity", "e"], learner_id


def test_the_harness_command_is_unchanged_without_the_flag():
    from yeto.rl.ssh_harness import _learner_argv

    assert _wandb_flags(_learner_argv(_harness_plan(wandb=False), 0)) == []
    assert _wandb_flags(_learner_argv(_harness_plan(), 0)) == []


def test_an_omitted_entity_is_not_forwarded_as_none():
    from yeto.rl.ssh_harness import _learner_argv

    plan = _harness_plan(wandb=True, wandb_project="p", wandb_entity=None, wandb_mode="offline")
    assert "--wandb-entity" not in _wandb_flags(_learner_argv(plan, 0))


def test_the_plan_never_carries_the_credential():
    """The harness promises to store the env file's path, never its contents.

    WANDB_API_KEY therefore rides in the remote env file next to HF_TOKEN,
    and the plan holds only which project to write to.
    """
    import json

    plan = _harness_plan(wandb=True, wandb_project="p", wandb_entity="e", wandb_mode="online")
    assert "WANDB_API_KEY" not in json.dumps(plan)
    assert not [k for k in plan["learner"] if k.endswith("_key") or k.endswith("_token")]
