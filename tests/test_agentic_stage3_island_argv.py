"""S26 E9.5 R6 (CPU, no Ray, no GPU): agentic stage 3 through the Miles island entry.

A colocated no-sync island with the Harbor agent, ``--rl-max-policy-age 1`` and
``--rl-resume-store`` derives Miles' suspension between turns (not
``--partial-rollout`` / ``--fully-async``), keeps the round-cut store on the
Miles args, and refuses to start without the gate relay. A full-parameter
island ran this layout only after S27 G-EF: colocated full parameters are
accepted on one island without outer sync and refused otherwise (see
``test_fullparam_colocated_island.py``); a fixed partition derives
``--fully-async``. That is why the S26 R6 GPU run used LoRA.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from yeto.provenance import python_spec_sha256, source_tree_sha256
from yeto.rl.adapters.miles import config as miles_config
from yeto.rl.adapters.miles import island_entry
from yeto.rl.adapters.miles.harbor_agent import AGENT_FUNCTION, HarnessConfigError
from yeto.rl.adapters.miles.policy_age import AGENTIC_GENERATE
from yeto.rl.engine import run_config
from yeto.rl.engine.run_config import full_parameter_layout_problem

REPO = Path(__file__).resolve().parents[1]
REWARD = "yeto.rl.adapters.miles.harbor_agent:reward_func"
SPEC = REPO / "examples" / "rl_algorithms" / "decoupled-ppo-c2-agentic.json"
LORA_TARGETS = ["linear_fc1", "linear_fc2", "linear_proj", "linear_qkv"]


class Qwen3ModelProvider(SimpleNamespace):
    pass


def _provider():  # Qwen3-4B-Instruct-2507 geometry
    return Qwen3ModelProvider(
        hidden_size=2560, num_attention_heads=32, num_layers=36, ffn_hidden_size=9728,
        num_query_groups=8, kv_channels=128, multi_latent_attention=False, num_moe_experts=None,
        seq_length=262144, layernorm_epsilon=1e-6, rotary_base=5000000, rotary_percent=1.0,
        vocab_size=151936, max_position_embeddings=262144, qk_layernorm=True, gated_linear_unit=True,
        share_embeddings_and_output_weights=True, add_bias_linear=False, add_qkv_bias=False,
        activation_func=None)


def _island_argv(tmp_path: Path, *extra: str) -> list[str]:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"prompt": "fix it", "metadata": {"task_id": "t", "agent": "codex"}}) + "\n")
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    return [
        "--model", str(model), "--model-revision", "cdbee75f17c01a7cc42f958dc650907174af0554",
        "--data", str(prompts), "--reward-function", REWARD,
        "--parameter-mode", "lora", "--lora-r", "16", "--lora-targets", "all-linear",
        "--rl-single-island-no-sync", "--learner-id", "0", "--num-learners", "1",
        "--global-rounds", "3", "--groups-per-round", "2", "--over-sampling-batch-size", "4",
        "--samples-per-group", "4", "--optimizer-steps", "1", "--inner-lr", "1e-05", "--seed", "0",
        "--rollout-max-response-len", "22528", "--seq-len", "32768",
        "--rl-algorithm-spec", str(SPEC), "--rl-max-policy-age", "1",
        "--custom-generate-function-path", AGENTIC_GENERATE, "--custom-agent-function-path", AGENT_FUNCTION,
        "--use-session-server", "--session-server-port", "30000", "--session-server-workers", "4",
        "--tito-model", "qwen3",
        "--actor-num-nodes", "1", "--actor-num-gpus-per-node", "2", "--tensor-parallel", "2",
        "--rl-resume-store", str(tmp_path / "store"), "--rl-resume-state-dir", str(tmp_path / "state"),
        "--rl-cut-every", "1",
        "--completed-groups-path", str(tmp_path / "ckpt.pt"), "--event-tape", str(tmp_path / "events.jsonl"),
        "--miles-root", str(tmp_path / "miles"),
        "--reward-sha256", python_spec_sha256(REWARD, base_dir=REPO),
        "--source-sha256", source_tree_sha256(REPO / "yeto"),
        *extra,
    ]


def _miles_argv(args, tmp_path: Path, monkeypatch) -> list[str]:
    monkeypatch.setattr(run_config, "_resolve_ref_load", lambda a, model_path: str(model_path))
    # Qwen3TITOTokenizer's parsers at the fork pin (the real table imports sglang)
    monkeypatch.setattr(miles_config, "_resolve_tito_parsers", lambda model: {"qwen3": ("qwen3", "qwen25")}[model])
    rc = run_config.resolve_rl_run_config(
        args, model_path=str(tmp_path / "model"), rollout_model_path=None,
        prompt_path=str(tmp_path / "prompts.jsonl"), eval_prompt_path=None, provider=_provider(),
        target_modules=LORA_TARGETS, yeto_policy_sync=False)
    return list(island_entry.build_ports_launch(args, rc, ()).argv)


def _value(argv: list[str], flag: str) -> str:
    assert argv.count(flag) == 1, flag
    return argv[argv.index(flag) + 1]


def test_colocated_agentic_island_derives_the_suspension_between_turns(tmp_path, monkeypatch):
    args = island_entry.parse_args(_island_argv(tmp_path))
    argv = _miles_argv(args, tmp_path, monkeypatch)
    assert "--agentic-suspend-between-turns" in argv and _value(argv, "--agentic-suspend-max-rounds") == "1"
    assert "--partial-rollout" not in argv and "--fully-async" not in argv
    assert "--colocate" in argv and "--use-tis" in argv and "--use-session-server" in argv
    assert _value(argv, "--custom-agent-function-path") == AGENT_FUNCTION
    assert _value(argv, "--custom-generate-function-path") == AGENTIC_GENERATE
    assert _value(argv, "--custom-rm-path") == "yeto.rl.adapters.miles.harbor_agent.reward_func"
    assert _value(argv, "--sglang-tool-call-parser") == "qwen25"
    assert (_value(argv, "--rollout-batch-size"), _value(argv, "--over-sampling-batch-size"),
            _value(argv, "--n-samples-per-prompt"), _value(argv, "--global-batch-size")) == ("2", "4", "4", "8")


def test_island_start_needs_the_gate_relay_and_keeps_the_cut_store(tmp_path):
    args = island_entry.parse_args(_island_argv(tmp_path))

    def start(environ):
        miles_args = SimpleNamespace(custom_generate_function_path=AGENTIC_GENERATE,
                                     custom_agent_function_path=AGENT_FUNCTION)
        island_entry.apply_ports_infra_switches(args, miles_args, environ=environ)
        return miles_args

    with pytest.raises(HarnessConfigError, match="gate relay"):
        start({})
    miles_args = start({"YETO_HARBOR_GATE_RELAY_PORT": "31500"})
    assert miles_args.yeto_rl_max_policy_age == 1
    assert miles_args.yeto_rl_resume["store"] == str(tmp_path / "store")
    assert miles_args.yeto_rl_resume["every"] == 1


def test_full_parameter_colocated_suspension_layout_needs_no_outer_sync(tmp_path):
    # colocated (no dedicated engines): refused unless the island has no outer sync (S27 G-EF)
    assert full_parameter_layout_problem(data_parallel=1, actor_num_nodes=1, rollout_gpus=None,
                                         rollout_gpus_per_engine=1) is not None
    assert full_parameter_layout_problem(data_parallel=1, actor_num_nodes=1, rollout_gpus=None,
                                         rollout_gpus_per_engine=1, colocated=True) is not None
    assert full_parameter_layout_problem(data_parallel=1, actor_num_nodes=1, rollout_gpus=None,
                                         rollout_gpus_per_engine=1, colocated=True,
                                         single_island_no_sync=True) is None
    # the island parser takes full parameters on a no-sync island with both roles
    # (S26 E11); the layout check above is what refuses this colocated layout
    args = island_entry.parse_args([a if a != "lora" else "full" for a in _island_argv(tmp_path)])
    assert args.parameter_mode == "full"
