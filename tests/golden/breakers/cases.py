"""The configs the breaker golden lock covers: every example run config, pool and sampler spec,
and hand-made edge configs (no budget, ``--no-cost-guard``, env overrides, legacy off spellings)."""

from __future__ import annotations

from typing import Any

from tests.golden.breakers.capture import Case

K31 = "examples/k31"
AGENTIC_SPEC = f"{K31}/sampler-spec.nc-mi355x-vllm-agentic-262k.json"
K31_POOL = f"{K31}/pool.nc-mi355x-agentic-262k.json"

# a launch argv with nothing but what parses as an RL launch: every breaker flag at its default
BASE_ARGV = ["--training-mode", "rl", "--gpu", "modal:8xh100", "--cluster-prefix", "golden",
             "--model", "Qwen/Qwen3-1.7B", "--data", "HuggingFaceH4/MATH-500", "--rl-sample-role", "trainer",
             "--rl-max-policy-age", "2", "--rollout-batch-size", "8", "--n-samples-per-prompt", "8"]

# examples/README.md: the MATH-500 smoke test (a synchronous launch, no buffer)
MATH_SMOKE_ARGV = ["--training-mode", "rl", "--gpu", "modal:8xh100", "--syncer-region", "nebius/eu-north1",
                   "--cluster-prefix", "math-smoke", "--model", "Qwen/Qwen3-1.7B",
                   "--model-revision", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", "--data", "HuggingFaceH4/MATH-500",
                   "--data-revision", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be", "--rl-prompt-column", "problem",
                   "--rl-label-column", "answer", "--reward-function", "examples/math_reward.py:reward_func",
                   "--tuning", "lora", "--lora-r", "8", "--lora-targets", "attention", "--total-steps", "3",
                   "--rollout-batch-size", "16", "--n-samples-per-prompt", "4", "--rollout-max-response-len", "1024",
                   "--seq-len", "2048", "--inner-lr", "1e-5", "--seed", "17", "--trust-remote-code", "--on-demand"]

# a pool with only what the pool loader needs (no elastic, no dispense)
BARE_POOL: dict[str, Any] = {"run": "golden", "route": "ssh", "target": 1, "accounts": ["0"],
                             "shard_base": "golden", "shard_limit": 4, "machine_type": "mi355x",
                             "spec": AGENTIC_SPEC, "run_prefix": "golden"}


def _run(argv: list[str], **sections: Any) -> dict[str, Any]:
    return {"run": "golden", "store": "s3://golden-bucket/runs/golden",
            "pool": {"config": "<pool.json>"}, "trainer": {"argv": list(argv)}, **sections}


# yeto/rl/up.py module doc: the documented run config
DOC_RUN = _run(BASE_ARGV, sandbox={"max_run_containers": 40},
               first_group={"timeout_s": 5400, "poll_s": 60},
               budget={"usd": 450, "wall_h": 3.0, "nc_usd_per_h": 24, "silence_min": 30,
                       "release_nc_on_breach": False})


def _legacy_off_run(cfg: dict[str, Any]) -> None:
    cfg["budget"] = {"usd": None, "wall_h": None}
    cfg["starvation"] = {"alert_min": 0, "stop_min": 0}
    cfg["rollout"] = {"stall_alert_s": 0}
    cfg["trainer"]["argv"] += ["--rl-stall-timeout", "0", "--idle-warn-minutes", "0", "--rl-max-policy-age", "0"]


def _legacy_off_pool(pool: dict[str, Any]) -> None:
    pool["elastic"] = {"engine_stall_s": None, "fence_max_hold_s": None, "max_sampler_restarts": 0,
                       "sampler_restart_reset_s": None}


def _legacy_off_spec(spec: dict[str, Any]) -> None:
    spec["incarnation_lease"] = False
    spec["reload_max_defer_s"] = None
    spec["agentic"]["drain_timeout_s"] = None
    spec["agentic"]["infra_retries"] = 0


def _custom_run(cfg: dict[str, Any]) -> None:
    cfg["budget"] = {"usd": 120.5, "silence_min": 10, "release_nc_on_breach": True}
    cfg["starvation"] = {"alert_min": 20, "stop_min": 90, "unhealthy_min": 5, "stop_grace_min": 3,
                         "trainer_wait_max_min": 60}
    cfg["rollout"] = {"stall_alert_s": 900}
    cfg["first_group"] = {"timeout_s": 3600}
    cfg["sandbox"] = {"max_containers_per_sampler": 12}
    cfg["trainer"]["argv"] += ["--rl-stall-timeout", "1800", "--rl-sample-wait-s", "5400"]


def _custom_pool(pool: dict[str, Any]) -> None:
    pool["target"] = 3
    pool["sampler"] = {"dispense": {"backend": "store", "namespace": "golden/task-dispense", "lease_s": 1800,
                                    "max_redo": 6, "passes": 2}}
    pool["elastic"] = {"max_run_spend_usd": 250.0, "crash_loop_before_ready": "halt", "stall_s": 3600,
                       "breaker_failures": 5, "breaker_cooldown_s": 300, "breaker_max_trips": 1,
                       "unreachable_grace_s": 120, "overdue_grace_s": 60, "control_plane_alert_s": 900,
                       "start_timeout_s": 1800, "max_orders": 9}


def _custom_spec(spec: dict[str, Any]) -> None:
    spec["max_consecutive_load_failures"] = 5
    spec["write_defer_max_s"] = 1200
    spec["incarnation_lease"] = {"max_unread_s": 120}
    spec["agentic"].update({"min_samples": 3, "drain_timeout_s": 2400, "abandon_grace_s": 60})
    spec["agentic"]["task_defaults"]["wall_clock_s"] = 3900
    spec["agentic"]["task_defaults"]["agent_timeout_s"] = 3600


TRAINER_ENV = {"YETO_SAMPLE_BUFFER_WAIT_S": "7200", "YETO_SAMPLE_BUFFER_MAX_DAMAGED_RUN": "0",
               "YETO_RL_XHW_GUARD": "action=stop,warn_abs_diff=0.2,act_abs_diff=0.3,window=5",
               "YETO_RL_NAN_SKIP": "1"}
SAMPLER_ENV = {"HARBOR_SETUP_RETRIES": "6", "YETO_AGENTENV_SKIP_REACH_CHECK": "1",
               "HARBOR_MAX_SANDBOXES": "16", "YETO_AGENTENV_MAX_ACTIVE": "32",
               "YETO_AGENTENV_STARTS_PER_MINUTE": "20", "YETO_AGENTENV_REQUEST_TIMEOUT_S": "600"}

CASES: list[Case] = [
    # examples/k31
    Case("k31-run-agentic-262k", run=f"{K31}/run.agentic-262k.json"),
    Case("k31-run-baremetal-b300-262k", run=f"{K31}/run.baremetal-b300-262k.json"),
    Case("k31-pool-nc-mi355x-agentic-262k", pool=K31_POOL),
    Case("k31-spec-nc-mi355x-vllm-agentic-262k", spec=AGENTIC_SPEC),
    Case("k31-spec-nc-mi355x-vllm", spec=f"{K31}/sampler-spec.nc-mi355x-vllm.json"),
    Case("k31-spec-nc-mi355x", spec=f"{K31}/sampler-spec.nc-mi355x.json"),
    Case("k31-spec-cpu-fake", spec=f"{K31}/sampler-spec.cpu-fake.json"),
    Case("k31-convertible-spec-cpu-fake", spec=f"{K31}/convertible-spec.cpu-fake.json"),
    # generic examples
    Case("generic-run-up-doc", run=DOC_RUN, pool=BARE_POOL),
    Case("generic-launch-math-smoke", launch_argv=MATH_SMOKE_ARGV),
    Case("ds41-spec-nc-mi355x-vllm", spec="examples/ds41/sampler-spec.nc-mi355x-vllm.json"),
    # edges
    Case("edge-minimal-run", run=_run(BASE_ARGV), pool=BARE_POOL),
    Case("edge-launch-defaults", launch_argv=BASE_ARGV),
    Case("edge-no-cost-guard", launch_argv=[*BASE_ARGV, "--no-cost-guard", "--budget-usd", "100"]),
    Case("edge-launch-budget", launch_argv=[*BASE_ARGV, "--budget-usd", "35", "--budget-warn-usd", "20",
                                            "--budget-stop-grace-minutes", "10", "--idle-warn-minutes", "0",
                                            "--modal-timeout-s", "43200", "--modal-retries", "0"]),
    Case("edge-launch-idle", launch_argv=[*BASE_ARGV, "--budget-usd", "50", "--idle-warn-minutes", "30",
                                          "--idle-stop-minutes", "10", "--load-grace-minutes", "60",
                                          "--max-load-minutes", "90", "--rl-stall-timeout", "3600"]),
    Case("edge-env-overrides", launch_argv=BASE_ARGV, spec=AGENTIC_SPEC,
         trainer_env=TRAINER_ENV, sampler_env=SAMPLER_ENV),
    Case("edge-argv-env-pairs", launch_argv=[*BASE_ARGV, "--modal-env", "YETO_SAMPLE_BUFFER_WAIT_S=10800",
                                             "--island-env", "YETO_RL_XHW_GUARD=on",
                                             "--modal-env", "YETO_SAMPLE_BUFFER_MAX_DAMAGED_RUN=3"]),
    Case("edge-flag-beats-env", launch_argv=[*BASE_ARGV, "--rl-sample-wait-s", "5400"],
         trainer_env={"YETO_SAMPLE_BUFFER_WAIT_S": "7200", "YETO_RL_XHW_GUARD": "off"}),
    Case("edge-legacy-off", run=_run(BASE_ARGV), pool=BARE_POOL, sampler_env={"HARBOR_SETUP_RETRIES": "0"},
         edit=_legacy_off_run, edit_pool=_legacy_off_pool, edit_spec=_legacy_off_spec),
    Case("edge-starvation-stop-only", run=_run(BASE_ARGV, starvation={"alert_min": 0, "stop_min": 45},
                                               budget={"usd": 0, "wall_h": 1.5}), pool=BARE_POOL),
    Case("edge-custom-values", run=_run(BASE_ARGV), pool=BARE_POOL,
         edit=_custom_run, edit_pool=_custom_pool, edit_spec=_custom_spec),
]
