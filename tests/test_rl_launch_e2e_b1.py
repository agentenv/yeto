"""Batch-1 GPU findings: launcher -> island learner wiring, end to end (CPU)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rl_e2e_launch import island_run, learner_from_run  # noqa: E402

BASE = ("--rl-placement", "fixed-partition", "--rl-rollout-gpus", "1",
        "--gpu", "aws:2xa100@us-east-1")


def _eval(tmp_path, *extra):
    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text('{"prompt": "q", "label": "a"}\n')
    return ("--rl-eval-interval", "2", "--rl-eval-data", str(heldout),
            "--rl-eval-dataset-name", "held", "--rl-eval-samples-per-prompt", "1", *extra)


def test_eval_sampling_knobs_reach_the_learner_and_the_run_config(tmp_path, monkeypatch):
    run = island_run(BASE + _eval(tmp_path, "--rl-eval-temperature", "0", "--rl-eval-top-p", "1",
                                  "--rl-eval-max-prompt-len", "256",
                                  "--rl-eval-max-response-len", "64",
                                  "--rl-eval-max-context-len", "512"), monkeypatch)
    args, _ = learner_from_run(run, tmp_path / "home")
    assert (args.eval_temperature, args.eval_top_p) == (0.0, 1.0)
    assert (args.eval_max_prompt_len, args.eval_max_response_len, args.eval_max_context_len) == (
        256, 64, 512)
    from yeto.rl.engine.run_config import _resolve_eval

    config = _resolve_eval(args, parameter_mode="lora", prompt_path="/p", eval_prompt_path="/e",
                           yeto_policy_sync=True)
    assert config.temperature == 0.0 and config.max_response_len == 64
    # the shipped heldout file is the one the learner verifies
    from yeto.rl import learner

    args.parameter_mode = "lora"
    assert learner._verify_eval_dataset_identity(args).read_text().startswith('{"prompt"')


def test_eval_knobs_need_the_eval_interval(tmp_path, monkeypatch):
    import pytest

    from test_rl_engine_selection import _cli
    from yeto import launcher

    with pytest.raises(ValueError, match="need --rl-eval-interval"):
        launcher._check_ports_infra_switches(_cli(("--rl-eval-temperature", "0")), "ports")


def test_default_run_carries_no_eval_flags(tmp_path, monkeypatch):
    args, _ = learner_from_run(island_run(BASE, monkeypatch), tmp_path / "home")
    assert args.eval_interval is None and args.eval_temperature is None
