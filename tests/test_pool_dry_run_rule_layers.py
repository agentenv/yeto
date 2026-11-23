"""The pool dry run judges only the breaker layers a pool sees (``pool_run.RULE_LAYERS``).

A pool has no trainer argv: ``launch.stall`` and ``trainer.sample_wait`` resolve to their
defaults there, so a rule that compares them (D4, D5) would warn about values the run does
not use. The pool dry run skips those rules; the run-level check still evaluates them per
pool with the real trainer argv, and the rules on the pool's own limits still fire.
"""

from __future__ import annotations

import json

from yeto import breakers
from yeto.launch.nc import pool_run

# an agentic drain-mode sampler: a 3900 s trial wall clock, 2 infra retries and a 2400 s
# drain cut make its longest legitimate group about 6340 s, more than the 3600 s default
# sample wait; its 7200 s dispense lease trips the pool-only rule D10
SPEC = {
    "reload_max_defer_s": 2400,
    "prompts": {"dispense": {"backend": "store", "namespace": "n", "lease_s": 7200}},
    "agentic": {"weight_switch": "drain", "drain_timeout_s": 2400, "infra_retries": 2,
                "task_defaults": {"agent_timeout_s": 3600, "wall_clock_s": 3900}},
}


def _pool(tmp_path) -> dict:
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(SPEC))
    return {"spec": str(spec)}


def _run_config(tmp_path, pool: dict, *trainer_flags: str) -> dict:
    pool_file = tmp_path / "pool.json"
    pool_file.write_text(json.dumps(pool))
    argv = ["--training-mode", "rl", "--gpu", "modal:8xh100", "--cluster-prefix", "r", "--model", "Qwen/Qwen3-1.7B",
            "--data", "HuggingFaceH4/MATH-500", "--rollout-batch-size", "8", "--n-samples-per-prompt", "8",
            "--rl-sample-role", "trainer", "--rl-max-policy-age", "2", *trainer_flags]
    return {"run": "r", "store": "s3://b/r", "trainer": {"argv": argv}, "pool": {"config": str(pool_file)}}


def _run_level(cfg: dict) -> list:
    sources = breakers.Sources.from_run_config(cfg)
    return breakers.preflight(breakers.resolve(sources, strict=False), run_config=cfg, sources=sources)


def test_the_pool_dry_run_skips_the_rules_on_trainer_and_launcher_limits(tmp_path):
    rules = {f.rule for f in pool_run.breaker_findings(_pool(tmp_path))}
    assert not rules & {"D4", "D5", "D7"}


def test_a_pool_only_rule_still_fires_in_the_pool_dry_run(tmp_path):
    found = pool_run.breaker_findings(_pool(tmp_path))
    assert any(f.rule == "D10" and "lease_s is 7200 s" in f.message for f in found)
    assert any(f.rule == "D1" for f in found)  # the drain cut, read from the spec


def test_the_run_level_check_still_judges_the_sample_wait_per_pool(tmp_path):
    pool = _pool(tmp_path)
    short = [f for f in _run_level(_run_config(tmp_path, pool, "--rl-sample-wait-s", "1800")) if f.rule == "D4"]
    assert len(short) == 1 and short[0].suggested == {"trainer.sample_wait.timeout_s": 6340}
    assert not [f for f in _run_level(_run_config(tmp_path, pool, "--rl-sample-wait-s", "10800")) if f.rule == "D4"]


def test_the_run_level_check_still_judges_the_stall_timer(tmp_path):
    cfg = _run_config(tmp_path, _pool(tmp_path), "--rl-stall-timeout", "100", "--rl-sample-wait-s", "3600")
    assert [f.rule for f in _run_level(cfg) if f.rule == "D5"] == ["D5"]

