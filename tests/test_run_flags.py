"""P0 4.7: ``yeto/rl/engine/run_flags.py`` is the single definition of the RL flags.

The CLI "Miles RL" group and the island entrypoint parser are generated from it;
these tests pin that both parsers contain exactly the schema's flags (plus the
hook-owned ones) and that every CLI/island divergence is declared in the schema.
"""

from __future__ import annotations

import argparse
import collections

import pytest

from yeto import cli
from yeto.rl.adapters.miles import island_entry
from yeto.rl.engine import run_flags
from yeto.rl.engine.run_flags import (
    CLI,
    ISLAND,
    RlFlag,
    cli_flag,
    flags_for,
    island_flag,
    rl_flags,
    rl_option,
    rl_option_pairs,
    shared_flag,
)

# Flags registered by their own hooks, not by the schema.
_WANDB = {"--wandb", "--wandb-project", "--wandb-entity", "--wandb-mode"}


def _recommend_options() -> set[str]:
    parser = argparse.ArgumentParser()
    from yeto.rl.adapters.miles.elastic_hook import add_recommend_arguments

    add_recommend_arguments(parser)
    return _options(parser._actions)


def _options(actions) -> set[str]:
    return {a.option_strings[0] for a in actions if a.option_strings and a.dest != "help"}


def _launch_parser() -> argparse.ArgumentParser:
    return cli.build_parser()._subparsers._group_actions[0].choices["launch"]


def _actions_by_option(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    return {a.option_strings[0]: a for a in parser._actions if a.option_strings}


def test_options_and_dests_are_unique_per_layer():
    for layer in (CLI, ISLAND):
        flags = list(flags_for(layer))
        options = [o for f in flags for o in f.argparse_call(layer)[0]]
        dests = [f.dest for f in flags]
        assert [o for o, n in collections.Counter(options).items() if n > 1] == []
        assert [d for d, n in collections.Counter(dests).items() if n > 1] == []


def test_cli_rl_group_is_the_schema_plus_the_recommend_hook():
    group = next(g for g in _launch_parser()._action_groups if g.title == "Miles RL")
    schema = {f.option for f in flags_for(CLI)}
    assert _options(group._group_actions) == schema | _recommend_options()


def test_island_parser_is_the_schema_plus_the_hooks():
    parser = island_entry.build_parser()
    schema = {f.option for f in flags_for(ISLAND)}
    assert _options(parser._actions) == schema | _recommend_options() | _WANDB


def test_island_divergences_are_the_declared_ones():
    # A new CLI/island difference must be added here on purpose (and justified
    # in the schema), never introduced by a second hand-written definition.
    declared = {f.option: dict(f.island) for f in rl_flags() if f.island}
    assert declared == {
        "--reward-function": {"required": True},
        "--over-sampling-batch-size": {"required": True},
        "--rl-syncer-epoch": {"default": 0},
        "--rl-elastic-quorum-timeout-s": {"type": float},
        "--rollout-max-response-len": {"default": None, "required": True},
    }
    aliases = {f.option: f.cli_aliases for f in rl_flags() if f.cli_aliases}
    assert aliases == {"--rl-deterministic-trainer": ("--deterministic",)}


def test_shared_flags_parse_alike_in_both_parsers():
    cli_actions = _actions_by_option(_launch_parser())
    island_actions = _actions_by_option(island_entry.build_parser())
    for flag in flags_for(ISLAND):
        if CLI not in flag.layers:
            continue
        ours, theirs = cli_actions[flag.option], island_actions[flag.option]
        assert ours.dest == theirs.dest, flag.option
        assert type(ours) is type(theirs), flag.option
        assert ours.choices == theirs.choices, flag.option
        assert ours.nargs == theirs.nargs, flag.option
        if "type" not in flag.island:
            assert ours.type is theirs.type, flag.option


def test_generic_launch_flags_agree_with_their_island_definitions():
    # --model, --seed, --lora-r, ... belong to the generic launch groups of
    # cli.py; the island's definitions live in the schema as island-only flags.
    cli_actions = _actions_by_option(_launch_parser())
    rl_group = next(g for g in _launch_parser()._action_groups if g.title == "Miles RL")
    rl_options = _options(rl_group._group_actions)
    generic = [f for f in flags_for(ISLAND)
               if CLI not in f.layers and f.option in cli_actions and f.option not in rl_options]
    assert {f.option for f in generic} >= {"--model", "--data", "--seed", "--lora-r", "--fragments"}
    for flag in generic:
        ours = cli_actions[flag.option]
        _, kwargs = flag.argparse_call(ISLAND)
        assert ours.dest == flag.dest, flag.option
        assert ours.type is kwargs.get("type"), flag.option
        assert ours.choices == kwargs.get("choices"), flag.option


def test_island_overrides_apply_only_on_the_island():
    flag = next(f for f in rl_flags() if f.option == "--rollout-max-response-len")
    assert flag.argparse_call(CLI)[1]["default"] == 32768
    assert flag.argparse_call(ISLAND)[1] == {"type": int, "default": None, "required": True}
    deterministic = next(f for f in rl_flags() if f.option == "--rl-deterministic-trainer")
    assert deterministic.argparse_call(CLI)[0] == ("--rl-deterministic-trainer", "--deterministic")
    assert deterministic.argparse_call(ISLAND)[0] == ("--rl-deterministic-trainer",)


def test_cli_env_defaults_are_read_when_the_parser_is_built(monkeypatch):
    monkeypatch.setenv("CYBERGYM_URL", "http://example.invalid:1")
    action = _actions_by_option(_launch_parser())["--cybergym-url"]
    assert action.default == "http://example.invalid:1"


def test_rl_option_resolves_per_layer():
    assert rl_option("rl_elastic_cells") == "--rl-elastic-cells"
    # Two layers store rollout_num_gpus under different names.
    assert rl_option("rollout_num_gpus", CLI) == "--rl-rollout-gpus"
    assert rl_option("rollout_num_gpus", ISLAND) == "--rollout-num-gpus"
    assert rl_option_pairs("rl_cut_every", "rl_cut_keep") == (
        ("rl_cut_every", "--rl-cut-every"), ("rl_cut_keep", "--rl-cut-keep"))
    with pytest.raises(KeyError):
        rl_option("rl_no_such_flag")
    with pytest.raises(KeyError):
        rl_option("rl_island_bundle_map", CLI)  # island-only


def test_flag_constructors_reject_inconsistent_layers():
    assert cli_flag("--x").layers == frozenset({CLI})
    assert island_flag("--x").layers == frozenset({ISLAND})
    with pytest.raises(ValueError, match="not a island flag"):
        cli_flag("--x").argparse_call(ISLAND)
    with pytest.raises(ValueError, match="island overrides"):
        RlFlag(("--x",), {}, frozenset({CLI}), island={"required": True})
    with pytest.raises(ValueError, match="CLI aliases"):
        RlFlag(("--x",), {}, frozenset({ISLAND}), cli_aliases=("--y",))
    with pytest.raises(ValueError, match="unknown layers"):
        RlFlag(("--x",), {}, frozenset({"syncer"}))
    assert shared_flag("--x", island={"default": 1}).argparse_call(ISLAND)[1] == {"default": 1}


def test_sha256_hex_arg():
    good = "a" * 64
    assert run_flags.sha256_hex_arg(good) == good
    with pytest.raises(argparse.ArgumentTypeError):
        run_flags.sha256_hex_arg("A" * 64)
