"""The verl island entry parses the one island flag schema (``yeto.rl.engine.run_flags``).

It defines no flag of its own besides the verl-only ones in ``adapters/verl/run_flags``,
and an unknown flag stops the island instead of being dropped (no ``parse_known_args``).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from yeto.rl.adapters.verl import island_entry
from yeto.rl.adapters.verl import run_flags as verl_flags
from yeto.rl.adapters.verl.run_config_rules import refuse_agentic_argv
from yeto.rl.engine import run_flags

REPO = Path(__file__).resolve().parents[1]
REQUIRED = ["--model", "m", "--model-revision", "a" * 40, "--data", "d", "--learner-id", "0",
            "--reward-function", "r", "--global-rounds", "1", "--groups-per-round", "1",
            "--samples-per-group", "2", "--rollout-max-response-len", "8", "--event-tape", "t",
            "--inner-lr", "1e-5", "--seq-len", "64", "--seed", "1"]


def test_the_island_entry_defines_no_flag_and_swallows_none():
    tree = ast.parse((REPO / "yeto/rl/adapters/verl/island_entry.py").read_text())
    calls = [node.func.attr for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    assert "add_argument" not in calls and "parse_known_args" not in calls
    for module in ("full_parameter_stream", "train_options", "eval_port"):
        src = (REPO / f"yeto/rl/adapters/verl/{module}.py").read_text()
        assert "add_argument(" not in src, module


def test_a_misspelled_flag_stops_the_island(capsys):
    with pytest.raises(SystemExit) as exc:
        island_entry.parse_args([*REQUIRED, "--rl-sample-wait", "30"])  # --rl-sample-wait-s
    assert exc.value.code == 2
    assert "unrecognized arguments: --rl-sample-wait" in capsys.readouterr().err
    with pytest.raises(SystemExit):  # no abbreviation either
        island_entry.parse_args([*REQUIRED, "--rl-max-policy", "1"])


def test_the_verl_only_flags_are_defined_once_and_parsed():
    island = {flag.option for flag in run_flags.flags_for(run_flags.ISLAND)}
    own = {flag.option for flag in verl_flags.ISLAND_FLAGS}
    assert not own & island  # the schema does not define them too
    args, unread = island_entry.parse_args([*REQUIRED, "--rl-fsdp-offload", "both", "--rl-release-only-versions",
                                            "2,4", "--rl-onpolicy-old-logprob", "measure"])
    assert (args.rl_fsdp_offload, args.rl_release_only_versions, args.rl_onpolicy_old_logprob) == \
        ("both", [2, 4], "measure")
    assert unread == []


def test_the_verl_island_keeps_its_defaults():
    args, _ = island_entry.parse_args(REQUIRED)
    for dest, value in verl_flags.ISLAND_DEFAULTS.items():
        assert getattr(args, dest) == value, dest
    assert args.completed_groups_path is None and args.source_sha256 is None  # not required here
    assert args.rl_fsdp_offload == "none" and args.parameter_mode == "lora"


def test_schema_flags_the_island_does_not_read_are_recorded_not_dropped_silently():
    # what the launcher sends every island (Miles options among them)
    sent = ["--over-sampling-batch-size", "4", "--rl-distributed-timeout-minutes", "10", "--optimizer-steps", "1",
            "--miles-root", "~/miles", "--rl-per-island-placement", "--wandb", "--wandb-project", "p"]
    args, unread = island_entry.parse_args([*REQUIRED, *sent])
    assert unread[:9] == sent[:9]  # each option with its value, in argv order
    assert "--rl-per-island-placement" in unread and args.wandb_project == "p"
    assert "--model" not in unread and "--seed" not in unread
    assert set(verl_flags.READ) <= {a.dest for a in verl_flags.build_parser()._actions}  # no stale name


def test_the_agentic_options_are_still_refused():
    _, unread = island_entry.parse_args([*REQUIRED, "--custom-agent-function-path", "x.y.run"])
    with pytest.raises(ValueError, match="custom-agent-function-path"):
        refuse_agentic_argv(unread)


def test_the_chat_template_kwargs_read_as_json_either_way():
    args, _ = island_entry.parse_args([*REQUIRED, "--apply-chat-template-kwargs", '{"enable_thinking": false}'])
    assert verl_flags.chat_template_kwargs(args.apply_chat_template_kwargs) == {"enable_thinking": False}
    assert verl_flags.chat_template_kwargs('{"a": 1}') == {"a": 1} and verl_flags.chat_template_kwargs(None) == {}
