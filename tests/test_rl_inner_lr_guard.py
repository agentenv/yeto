"""An RL launch that leaves --inner-lr unset is refused: the default is an SFT
learning rate (3e-4), which moved Kimi-K3.1 experts about 2 % per RL step."""
import argparse
from types import SimpleNamespace

import pytest

from yeto import cli
from yeto.launch import rl_args


def _parsed(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--inner-lr", type=float, default=cli._DefaultLR(3e-4))
    return parser.parse_args(argv)


def test_the_default_is_told_from_an_explicit_value():
    assert rl_args.inner_lr_unset(_parsed([]))
    assert float(_parsed([]).inner_lr) == 3e-4
    assert not rl_args.inner_lr_unset(_parsed(["--inner-lr", "3e-4"]))
    assert not rl_args.inner_lr_unset(_parsed(["--inner-lr", "1e-6"]))
    assert not rl_args.inner_lr_unset(SimpleNamespace(inner_lr=1e-6))


def test_the_cli_default_is_the_marked_value(monkeypatch):
    monkeypatch.setattr(cli, "DEFAULT_INNER_LR", cli._DefaultLR(3e-4))
    actions = [a for a in cli.build_parser()._actions if "--inner-lr" in a.option_strings] \
        if hasattr(cli, "build_parser") else []
    for action in actions:
        assert isinstance(action.default, cli._DefaultLR)


def test_an_rl_launch_without_inner_lr_is_refused():
    with pytest.raises(rl_args.UnsetRLLearningRate, match="SFT default 0.0003"):
        rl_args.refuse_unset_inner_lr(_parsed([]))
    rl_args.refuse_unset_inner_lr(_parsed(["--inner-lr", "1e-6"]))
    rl_args.refuse_unset_inner_lr(_parsed(["--inner-lr", "3e-4"]))  # explicit: the user's choice


def test_the_real_cli_parser_refuses_an_rl_launch_without_inner_lr(monkeypatch):
    monkeypatch.setattr(cli, "DEFAULT_INNER_LR", cli._DefaultLR(3e-4))  # conftest unmarks it
    args = cli.build_parser().parse_args(["launch", "--training-mode", "rl", "--model", "m", "--data", "d"])
    assert rl_args.inner_lr_unset(args)
    with pytest.raises(rl_args.UnsetRLLearningRate):
        rl_args.refuse_unset_inner_lr(args)
