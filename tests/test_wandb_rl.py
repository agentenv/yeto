"""RL event tape -> W&B.

The RL island's tape is the instrumentation; W&B is a second reader of it.
These pin the projection from tape event to metrics, and that a failing
sink never reaches the RL loop.
"""

import argparse
import json
from pathlib import Path

import pytest

from yeto.rl.wandb_rl import (
    STEP_KEY,
    TRAIN_STEP_KEY,
    RlTelemetry,
    event_metrics,
    tee,
)

# Captured from a real DeepSeek-V4 island tape, one of each event kind the
# RL path actually emits. Synthetic fixtures hid an x-axis that no real
# event carries; these do not.
REAL_EVENTS = [
    json.loads(line)
    for line in (Path(__file__).parent / "data" / "real_rl_events.jsonl")
    .read_text()
    .splitlines()
    if line.strip()
]

LOCAL_ROUND = {
    "event": "rl_local_round",
    "island_id": 1,
    "time_unix": 1.0,
    "local_round_id": 7,
    "rl/reward_mean": 0.42,
    "rl/reward_std": 0.1,
    "rl/action_tokens": 2048,
    "rl/current_vs_rollout_kl": 0.03,
    "rl/ess_ratio": 0.87,
    "rl/clip_fraction": 0.12,
    "train/step": 7,
    "train/loss": -0.4,
    "train/pg_loss": -0.5,
    "train/grad_norm": 1.25,
    "train/train_rollout_kl": 0.03,
    "train/ess_ratio": 0.87,
    "train/pg_clipfrac": 0.12,
    "train/lr": 1e-6,
    "train/pass_rate": 0.5,
    "rl/zero_variance_group_ratio": 0.0,
    "rl/policy_hash": "deadbeef",
    "rl/fragment_versions": [3, 3, 2, 2],
    "sync/applied_fragments": [0, 1],
    "sync/fragment_payload_bytes_sent": 4096,
}


def test_only_the_allowlisted_loss_curve_survives_the_projection():
    m = event_metrics(LOCAL_ROUND)
    assert m == {
        "train/loss": -0.4,
        "train/pg_loss": -0.5,
        "train/grad_norm": 1.25,
        "train/train_rollout_kl": 0.03,
        "train/ess_ratio": 0.87,
        "train/pg_clipfrac": 0.12,
        "train/lr": 1e-6,
        "rl/reward_mean": 0.42,
        "rl/pass_rate": 0.5,
        TRAIN_STEP_KEY: 7,
        STEP_KEY: 7,
    }


def test_zero_valued_allowlisted_metrics_are_kept():
    event = dict(LOCAL_ROUND, **{"train/pg_loss": 0.0})
    assert event_metrics(event)["train/pg_loss"] == 0.0


def test_identifiers_are_not_logged_as_series():
    m = event_metrics(LOCAL_ROUND)
    for key in (
        "rl/policy_hash",
        "island_id",
        "time_unix",
        "rl/action_tokens",
        "sync/fragment_payload_bytes_sent",
    ):
        assert key not in m


def test_lists_and_unknown_numeric_fields_are_never_projected():
    event = dict(LOCAL_ROUND)
    event.update(
        prompt="private prompt",
        response="private response",
        tools=[{"name": "private-tool"}],
        env={"PRIVATE": "value"},
        secret_numeric=1234,
    )
    m = event_metrics(event)
    assert not {"prompt", "response", "tools", "env", "secret_numeric"} & set(m)
    assert not any(key.endswith("_count") for key in m)


def test_booleans_and_nonfinite_values_are_not_metrics():
    assert (
        event_metrics(
            {
                "event": "rl_local_round",
                "local_round_id": 1,
                "train/step": 1,
                "train/loss": True,
                "rl/reward_mean": float("nan"),
            }
        )
        is None
    )


def test_transport_events_do_not_cross_the_scalar_allowlist():
    assert (
        event_metrics(
            {
                "event": "rl_fragment_push",
                "island_id": 0,
                "time_unix": 2.0,
                "fragment_id": 2,
                "global_step": 9,
                "round_attempt": 1,
                "base_version": 7,
                "c_steps": 4,
                "c_tokens": 900,
                "delta_l2_norm": 1.25,
                "payload_bytes": 512,
                "pull_to_push_seconds": 0.4,
                "realized_h": 4,
            }
        )
        is None
    )


def test_allowlisted_names_on_an_unrecognized_event_are_still_dropped():
    assert (
        event_metrics(
            {
                "event": "untrusted_future_event",
                "local_round_id": 1,
                "train/step": 1,
                "train/loss": 0.25,
                "rl/reward_mean": 1.0,
            }
        )
        is None
    )


def test_recognized_event_kinds_cannot_spoof_each_others_series():
    assert (
        event_metrics(
            {
                "event": "rl_eval_result",
                "policy_version": 1,
                "train/step": 1,
                "train/loss": 0.25,
                "rl/reward_mean": 1.0,
            }
        )
        is None
    )
    assert "eval/reward_mean" not in event_metrics(
        {
            "event": "rl_local_round",
            "local_round_id": 1,
            "rl/reward_mean": 1.0,
            "rl/eval/result": 0.5,
        }
    )


def test_pass_rates_are_bounded_and_huge_integers_are_nonfatal():
    metrics = event_metrics(
        {
            "event": "rl_local_round",
            "local_round_id": 1,
            "train/step": 1,
            "train/loss": 10**10_000,
            "rl/reward_mean": 0.5,
            "train/pass_rate": 1.5,
        }
    )
    assert metrics == {"rl/reward_mean": 0.5, STEP_KEY: 1}


def test_error_text_and_failure_metadata_never_reach_wandb():
    assert (
        event_metrics(
            {
                "event": "rl_strict_failure",
                "island_id": 0,
                "time_unix": 3.0,
                "metric": "reward_variance",
                "value": 1,
                "error": "StrictRlInvariantError: ...",
            }
        )
        is None
    )


def test_an_event_with_nothing_measurable_is_dropped():
    assert event_metrics({"event": "noise", "island_id": 0, "time_unix": 1.0}) is None


def test_tee_is_inert_without_the_flag():
    calls = []
    telemetry = RlTelemetry()
    telemetry.log = lambda *a, **k: calls.append(a)
    tee(argparse.Namespace(wandb=False), LOCAL_ROUND)
    assert calls == []


def test_a_broken_sink_never_reaches_the_rl_loop(monkeypatch, caplog):
    import yeto.rl.wandb_rl as mod

    class _Boom:
        def log(self, args, event):
            raise RuntimeError("private prompt from backend")

    monkeypatch.setattr(mod, "_TELEMETRY", _Boom())
    # Must not raise: the RL round that produced this event has to continue.
    tee(argparse.Namespace(wandb=True), LOCAL_ROUND)
    assert "RuntimeError" in caplog.text
    assert "private prompt" not in caplog.text


def test_the_island_run_joins_the_fleet_group(monkeypatch):
    import yeto.rl.wandb_rl as mod

    seen = {}

    def fake_init(args, **kwargs):
        seen.update(kwargs)

        class _Run:
            enabled = True

            def log(self, m):
                seen.setdefault("logged", []).append(m)

            def finish(self, exit_code=0):
                pass

        return _Run()

    monkeypatch.setattr(mod, "init", fake_init)
    telemetry = RlTelemetry()
    args = argparse.Namespace(
        wandb=True,
        yeto_rl_learner_id=3,
        yeto_rl_sync_preset="decoupled",
        yeto_rl_model="Qwen/Qwen2.5-0.5B",
        yeto_rl_data="d",
        yeto_rl_base_model_revision="abc",
        yeto_rl_reward_sha256="sha",
    )
    telemetry.log(args, LOCAL_ROUND)
    assert seen["job_type"] == "rl-learner"
    assert seen["name"] == "learner-3"
    assert seen["config_override"] == {
        "island_backend": "rl-miles",
        "learner_id": 3,
        "rl_sync_preset": "decoupled",
    }
    assert not {"model", "data", "base_model_revision", "reward_sha256"} & set(
        seen["config_override"]
    )
    assert seen["step_metrics"]["train/*"] == TRAIN_STEP_KEY
    assert seen["step_metrics"]["rl/*"] == STEP_KEY
    assert seen["logged"][0]["rl/reward_mean"] == 0.42


def test_the_run_starts_once_across_many_events(monkeypatch):
    import yeto.rl.wandb_rl as mod

    inits = []

    def fake_init(args, **kwargs):
        inits.append(1)

        class _Run:
            enabled = True

            def log(self, m):
                pass

            def finish(self, exit_code=0):
                pass

        return _Run()

    monkeypatch.setattr(mod, "init", fake_init)
    telemetry = RlTelemetry()
    args = argparse.Namespace(wandb=True, yeto_rl_learner_id=0)
    for _ in range(5):
        telemetry.log(args, LOCAL_ROUND)
    assert len(inits) == 1


def test_a_nonmetric_event_does_not_start_a_wandb_run(monkeypatch):
    import yeto.rl.wandb_rl as mod

    inits = []
    monkeypatch.setattr(mod, "init", lambda *args, **kwargs: inits.append(1))
    telemetry = RlTelemetry()
    telemetry.log(
        argparse.Namespace(wandb=True),
        {"event": "rl_fragment_push", "payload_bytes": 1},
    )
    assert inits == []


# --------------------------------------------------------------------------
# the SSH harness builds its own learner command, so it needs its own wiring


def _harness_plan(**learner):
    base = {
        "model": "/m",
        "model_revision": "r" * 40,
        "data": "/workspace/data/dataset.jsonl",
        "reward_function": "pkg.mod:f",
        "global_rounds": 1,
        "fragments": 1,
        "pipeline": 1,
        "local_horizon": 1,
        "total_fragment_steps": 1,
        "groups_per_round": 2,
        "samples_per_group": 3,
        "over_sampling_batch_size": 2,
        "optimizer_steps": 1,
        "rollout_max_response_len": 4096,
        "tensor_parallel": 8,
        "pipeline_parallel": 1,
        "rollout_num_gpus_per_engine": 8,
        "sglang_mem_fraction_static": 0.45,
        "lora_r": 8,
        "lora_targets": "attention",
        "inner_lr": 1e-5,
        "seq_len": 8192,
        "seed": 1,
        "wan_streams": 1,
        "trust_remote_code": True,
        "rl_model_recipe": "generic",
        "actor_num_nodes": 1,
    }
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

    plan = _harness_plan(
        wandb=True, wandb_project="p", wandb_entity="e", wandb_mode="online"
    )
    for learner_id in (0, 1):
        flags = _wandb_flags(_learner_argv(plan, learner_id))
        assert flags == [
            "--wandb",
            "--wandb-project",
            "p",
            "--wandb-mode",
            "online",
            "--wandb-entity",
            "e",
        ], learner_id


def test_the_harness_command_is_unchanged_without_the_flag():
    from yeto.rl.ssh_harness import _learner_argv

    assert _wandb_flags(_learner_argv(_harness_plan(wandb=False), 0)) == []
    assert _wandb_flags(_learner_argv(_harness_plan(), 0)) == []


def test_an_omitted_entity_is_not_forwarded_as_none():
    from yeto.rl.ssh_harness import _learner_argv

    plan = _harness_plan(
        wandb=True, wandb_project="p", wandb_entity=None, wandb_mode="offline"
    )
    assert "--wandb-entity" not in _wandb_flags(_learner_argv(plan, 0))


def test_the_plan_never_carries_the_credential():
    """The harness promises to store the env file's path, never its contents.

    WANDB_API_KEY therefore rides in the remote env file next to HF_TOKEN,
    and the plan holds only which project to write to.
    """
    import json

    plan = _harness_plan(
        wandb=True, wandb_project="p", wandb_entity="e", wandb_mode="online"
    )
    assert "WANDB_API_KEY" not in json.dumps(plan)
    assert not [k for k in plan["learner"] if k.endswith(("_key", "_token"))]


# --------------------------------------------------------------------------
# against the real tape


@pytest.mark.parametrize("event", REAL_EVENTS[1:], ids=lambda e: e["event"])
def test_real_non_scalar_events_are_dropped(event):
    assert event_metrics(event) is None


def test_the_legacy_real_rollout_round_keeps_only_reward():
    round_event = next(e for e in REAL_EVENTS if e["event"] == "rl_local_round")
    assert event_metrics(round_event) == {
        "rl/reward_mean": round_event["rl/reward_mean"],
        STEP_KEY: round_event["base_policy_version"],
    }


def test_eval_projects_only_reward_and_recognized_pass_rate():
    assert event_metrics(
        {
            "event": "rl_eval_result",
            "policy_version": 9,
            "rollout_id": 0,
            "sample_count": 300,
            "dataset_name": "private-name",
            "rl/eval/result": 0.4,
            "rl/eval/pass_at_1": 0.25,
        }
    ) == {
        "eval/reward_mean": 0.4,
        "eval/pass_rate": 0.25,
        STEP_KEY: 9,
    }


def test_repeated_safe_point_is_logged_once(monkeypatch):
    import yeto.rl.wandb_rl as mod

    logged = []

    class _Run:
        enabled = True

        def log(self, metrics):
            logged.append(metrics)

        def finish(self, exit_code=0):
            pass

    monkeypatch.setattr(mod, "init", lambda *args, **kwargs: _Run())
    telemetry = RlTelemetry()
    args = argparse.Namespace(wandb=True, yeto_rl_learner_id=0)
    telemetry.log(args, LOCAL_ROUND)
    telemetry.log(args, dict(LOCAL_ROUND))
    assert logged == [event_metrics(LOCAL_ROUND)]
