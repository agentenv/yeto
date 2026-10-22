"""P0 4.7: every RL flag is defined once, in the RL flag schema.

Neutral flags live in ``yeto/rl/engine/run_flags.py``; the Miles backend's in
``yeto/rl/adapters/miles/run_flags.py``. The CLI RL group and the island
entrypoint parser are generated from them. These tests pin that both parsers
contain exactly the schema's flags (plus the hook-owned ones), that every
CLI/island divergence is declared, and that the launcher's forwarding tables and
island-command flag literals stay inside the schema.
"""

from __future__ import annotations

import argparse
import ast
import collections
import re
from pathlib import Path

import pytest

from yeto import cli
from yeto.rl.adapters.miles import island_entry
from yeto.rl.engine import run_flags
from yeto.rl.engine.run_flags import (
    CLI,
    ISLAND,
    THEMES,
    RlFlag,
    backend_flags,
    cli_flag,
    flags_for,
    island_flag,
    rl_flags,
    rl_option,
    rl_forward_pairs,
    rl_option_pairs,
    shared_flag,
)

REPO = Path(__file__).resolve().parents[1]

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


def _rl_group() -> argparse._ArgumentGroup:
    return next(g for g in _launch_parser()._action_groups if g.title == "Miles RL")


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
    group = _rl_group()
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
    rl_options = _options(_rl_group()._group_actions)
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


def test_quorum_timeout_is_float_on_both_layers_and_whole_seconds_for_the_syncer():
    import types

    from yeto import launcher

    flag = next(f for f in rl_flags() if f.option == "--rl-elastic-quorum-timeout-s")
    assert flag.argparse_call(CLI)[1]["type"] is flag.argparse_call(ISLAND)[1]["type"] is float
    args = types.SimpleNamespace(rl_elastic=True, rl_elastic_quorum_timeout_s=900.0)
    assert launcher._syncer_quorum_timeout(args) == " --quorum-timeout-s 900"
    with pytest.raises(ValueError, match="whole seconds"):
        launcher._syncer_quorum_timeout(types.SimpleNamespace(rl_elastic=True,
                                                              rl_elastic_quorum_timeout_s=1.5))


# --------------------------------------------------------------------------- split by backend


def test_neutral_schema_names_no_backend():
    source = (REPO / "yeto/rl/engine/run_flags.py").read_text()
    assert re.findall(r"miles|sglang|megatron", source, re.IGNORECASE) == []


def test_each_flag_is_defined_by_exactly_one_side():
    from yeto.rl.engine.run_flags import _NEUTRAL_FLAGS

    neutral = {f.option for flags in _NEUTRAL_FLAGS.values() for f in flags}
    backend = {f.option for flags in backend_flags().values() for f in flags}
    assert neutral and backend and not neutral & backend
    assert set(_NEUTRAL_FLAGS) <= set(THEMES) and set(backend_flags()) <= set(THEMES)
    # The island-internal flags belong to the backend wholesale.
    assert all(CLI in f.layers for flags in _NEUTRAL_FLAGS.values() for f in flags
               if ISLAND in f.layers) and "island" not in _NEUTRAL_FLAGS


def test_rl_image_help_shows_the_pinned_image():
    from yeto.rl.adapters.miles.pins import MILES_NEXT_IMAGE

    assert MILES_NEXT_IMAGE in _actions_by_option(_launch_parser())["--rl-image"].help


def test_island_entry_has_no_hand_written_flags():
    source = (REPO / "yeto/rl/adapters/miles/island_entry.py").read_text()
    assert "add_argument(" not in source


# --------------------------------------------------------------------------- launcher forwarding


def test_forward_tables_name_flags_of_both_layers():
    from yeto import launcher

    cli_dests = {f.dest for f in flags_for(CLI)}
    island_options = {f.option for f in flags_for(ISLAND)}
    for table in (launcher._ELASTIC_PAUSE_FLAGS, launcher._EVAL_OPTIONAL_FLAGS):
        for cli_dest, island_option in table:
            assert cli_dest in cli_dests and island_option in island_options
    assert rl_forward_pairs(("rl_eval_top_p", "eval_top_p")) == (("rl_eval_top_p", "--eval-top-p"),)
    with pytest.raises(KeyError):
        rl_forward_pairs("rl_island_bundle_map")  # island-only: the CLI cannot forward it
    with pytest.raises(KeyError):
        rl_forward_pairs("rl_quorum_theta")  # CLI-only: goes to the syncer, not the island


# Flag literals in the launcher's command builders that are NOT island or launch
# flags: syncer, docker, git, pip, ray, torchrun, sky, ... Pinned so that an island
# flag renamed in the schema but still spelled the old way in the launcher fails.
_NON_RL_LITERALS = frozenset({
    "--abbrev-ref", "--adapter-dir", "--add", "--address", "--allow-unattested-legacy-adapter",
    "--allow-unauthenticated-islands", "--async-checkpoint", "--bcast-delta-full-every", "--bind", "--cap-add", "--carry-gamma",
    "--checkpoint-every", "--checkpoint-path", "--codex-harness-binary", "--config-json",
    "--create", "--dashboard-host", "--dearmor", "--depth", "--detach", "--dtype",
    "--entrypoint", "--expected-island-contract", "--final-ack-timeout-s", "--final-grace-s", "--follow", "--force",
    "--format", "--get", "--global", "--gpus", "--head", "--ignore-installed",
    "--include-dashboard", "--index-url", "--ipc", "--island-lease-s",
    "--island-scheduling-mode", "--json", "--learner-weight", "--learners", "--level",
    "--manifest-path", "--mark-final-checkpoint", "--master", "--matrix-merge",
    "--max-base-lag", "--max-carry-lag", "--shard-count", "--shard-index", "--weighted-fold", "--name", "--net", "--nnodes",
    "--no-build-isolation", "--no-checkout", "--no-deps", "--no-follow", "--node",
    "--node-ip-address", "--nproc", "--num-gpus", "--num-inference-steps", "--output-dir",
    "--password-stdin", "--policy-sweep-fragments", "--porcelain", "--port", "--profile",
    "--prompt", "--q-min",
    "--query-gpu", "--quiet", "--quorum-theta", "--quorum-timeout-s", "--raid-devices",
    "--recovery-bound-steps", "--recovery-donor-timeout-s", "--recovery-max-donor-tries",
    "--release", "--require-profile-binding", "--resume", "--rl-eval", "--rl-test", "--rm", "--run",
    "--runtime", "--seed-column", "--shm-size", "--soft-deadline-s", "--syncer-epoch",
    "--temp-dir", "--ulimit", "--untracked-files", "--wandb-group", "--yes",
    # torchrun, tail and the W4A8 re-quantizer CLI in the SSH trainer island (yeto/launch/ssh_trainer_island.py)
    "--awq-scales", "--loop", "--master-addr", "--master-port", "--no-awq", "--node-rank", "--nproc-per-node",
    "--pid", "--poll-seconds", "--recipe", "--store", "--template", "--frozen-base", "--frozen-pattern",
    # docker run of a runtime: docker SSH trainer node (ssh_trainer_island.py, node_inventory.py example)
    "--device", "--network",
})


def _flag_literals(path: Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            found |= set(re.findall(r"(?<![\w-])--[a-z][a-z0-9-]*[a-z0-9]", node.value))
    return found


def test_launcher_flag_literals_are_schema_flags_or_pinned_non_rl_flags():
    launcher_layer = [REPO / "yeto/launcher.py", *sorted((REPO / "yeto/launch").glob("*.py"))]
    literals = (set().union(*map(_flag_literals, launcher_layer))
                | _flag_literals(REPO / "yeto/rl/adapters/miles/launch_flags.py"))
    island = {o for a in island_entry.build_parser()._actions for o in a.option_strings}
    launch = {o for a in _launch_parser()._actions for o in a.option_strings}
    assert literals - island - launch == _NON_RL_LITERALS
