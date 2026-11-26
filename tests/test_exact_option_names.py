"""yeto's parsers take long options by their full name only (``yeto.exact_argparse``).

argparse's default prefix matching let a misspelled option set another one without
a word: ``yeto launch --rollout-num-gpus 8`` set ``--rollout-num-gpus-per-engine``.
A prefix is refused now, the error names the options probably meant, and the flags
yeto passes on to Miles are untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from yeto import cli
from yeto.exact_argparse import ExactArgumentParser, suggestions
from yeto.launch.nc import cli as nc_cli
from yeto.rl.adapters.miles import config as miles_config
from yeto.rl.adapters.miles import island_entry

REPO = Path(__file__).resolve().parents[1]
LAUNCH = ["launch", "--model", "m", "--data", "d", "--training-mode", "rl"]


def _error(capsys, parse, argv) -> str:
    with pytest.raises(SystemExit) as exc:
        parse(argv)
    assert exc.value.code == 2
    return capsys.readouterr().err


def _parsers(parser: argparse.ArgumentParser, path=("yeto",)):
    yield " ".join(path), parser
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                yield from _parsers(sub, (*path, name))


def test_every_yeto_command_takes_full_option_names_only():
    seen = dict(_parsers(cli.build_parser()))
    assert {"yeto launch", "yeto run up", "yeto fleet show", "yeto cluster probe", "yeto rollout scale",
            "yeto breaker check", "yeto preflight"} <= set(seen)
    assert [name for name, parser in seen.items() if parser.allow_abbrev] == []
    assert island_entry.build_parser().allow_abbrev is False


def test_a_launch_prefix_is_refused_and_the_meant_options_are_named(capsys):
    err = _error(capsys, cli.build_parser().parse_args, [*LAUNCH, "--rollout-num-gpus", "8"])
    assert "unrecognized arguments: --rollout-num-gpus 8" in err
    hint = next(line for line in err.splitlines() if line.strip().startswith("--rollout-num-gpus:"))
    # the option with that destination first, then the one the prefix used to pick
    assert hint.index("--rl-rollout-gpus") < hint.index("--rollout-num-gpus-per-engine")
    # through main() (the `yeto` entry point) too, and with --opt=value
    err = _error(capsys, cli.main, [*LAUNCH, "--rl-max-policy=2"])
    assert "unrecognized arguments: --rl-max-policy=2" in err and "--rl-max-policy-age" in err


def test_a_misspelling_gets_the_closest_spelling(capsys):
    err = _error(capsys, cli.build_parser().parse_args, [*LAUNCH, "--rl-max-polcy-age", "2"])
    assert "did you mean --rl-max-policy-age" in err


def test_full_names_parse_as_before():
    args = cli.build_parser().parse_args([*LAUNCH, "--rollout-num-gpus-per-engine", "4", "--rl-rollout-gpus", "8",
                                          "--rl-max-policy-age=3"])
    assert (args.rollout_num_gpus_per_engine, args.rollout_num_gpus, args.rl_max_policy_age) == (4, 8, 3)
    legacy = cli.parse_args(["--model", "m", "--data", "d", "--rl-rollout-gpus", "2"])
    assert legacy.rollout_num_gpus == 2
    with pytest.raises(SystemExit):
        cli.parse_args(["--model", "m", "--data", "d", "--rollout-num-gpus", "2"])


def test_a_subcommand_hint_comes_from_that_subcommand(capsys):
    err = _error(capsys, cli.build_parser().parse_args, ["fleet", "show", "--config", "c.json", "--gen", "3"])
    assert "--gen: options are taken by their full name; did you mean --generation?" in err
    err = _error(capsys, nc_cli.main, ["pool", "--config", "pool.json", "--dry"])
    assert "unrecognized arguments: --dry" in err and "did you mean --dry-run" in err


def test_short_options_still_combine():
    parser = ExactArgumentParser(prog="t")
    parser.add_argument("-v", action="store_true")
    parser.add_argument("-x", action="store_true")
    parser.add_argument("-n", type=int)
    parser.add_argument("--name-long")
    assert vars(parser.parse_args(["-vxn3"])) == {"v": True, "x": True, "n": 3, "name_long": None}
    assert vars(parser.parse_args(["-vx", "-n", "4", "--name-long=a"]))["n"] == 4
    with pytest.raises(SystemExit):
        parser.parse_args(["--name", "a"])
    export = cli.build_parser().parse_args(["dashboard", "export", "-o", "out.html"])
    assert export.out == "out.html"


def test_suggestions_are_bounded_and_skip_unrelated_names():
    parser = ExactArgumentParser(prog="t")
    for i in range(8):
        parser.add_argument(f"--alpha-{i}")
    parser.add_argument("--zeta")
    hints = suggestions("--alpha", [parser])
    assert len(hints) == 4 and all(h.startswith("--alpha-") for h in hints)
    assert suggestions("--qqqqqqqq", [parser]) == []


def test_the_customer_launch_argv_and_its_miles_flags_pass_through():
    argv = json.loads((REPO / "tests/fixtures/k31_customer_kit/run.json").read_text())["trainer"]["argv"]
    args = cli.build_parser().parse_args(["launch", *argv])
    given = [a for a in argv if a.startswith(miles_config.TRAIN_EXTRA_ENV + "=")]
    extra = [e for e in (*args.island_env, *args.modal_env) if e.startswith(miles_config.TRAIN_EXTRA_ENV + "=")]
    assert len(given) == 1 and extra == given
    # the Miles flags are a value to yeto: Miles gets them as written
    raw = extra[0].split("=", 1)[1]
    assert miles_config.train_extra_argv("kimi_k31", raw) == tuple(raw.split())


def test_the_miles_island_refuses_a_prefix_and_keeps_miles_flags(tmp_path, monkeypatch, capsys):
    from test_agentic_stage3_island_argv import _island_argv, _miles_argv

    full = _island_argv(tmp_path)
    err = _error(capsys, island_entry.parse_args, [*full, "--rl-max-policy", "1"])
    assert "unrecognized arguments: --rl-max-policy 1" in err and "--rl-max-policy-age" in err
    args = island_entry.parse_args(full)
    miles = _miles_argv(args, tmp_path, monkeypatch)
    # Miles' own option names, rendered in full for Miles' parser
    for flag in ("--rollout-num-gpus-per-engine", "--rollout-batch-size", "--n-samples-per-prompt",
                 "--custom-generate-function-path"):
        assert flag in miles
