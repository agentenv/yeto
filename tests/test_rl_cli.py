"""Tests for the ``yeto rl`` command dispatch."""

import sys
import types

from yeto import cli


def test_rl_command_returns_without_printing_global_help(monkeypatch, capsys):
    called = []
    run_rl_module = types.ModuleType("yeto.rl.run")
    run_rl_module.run_rl = lambda args: called.append(args.env)
    monkeypatch.setitem(sys.modules, "yeto.rl.run", run_rl_module)

    assert cli.main(["rl", "--env", "mock", "--steps", "1"]) == 0
    assert called == ["mock"]
    assert "usage: yeto" not in capsys.readouterr().out


def test_launch_parses_decoupled_rl_initial_adapter_inputs():
    args = cli.parse_args(
        [
            "--gpu",
            "aws:1xa100",
            "--model",
            "org/model",
            "--data",
            "org/data",
            "--training-mode",
            "rl",
            "--rl-initial-adapter",
            "/tmp/final-adapter",
            "--rl-initial-adapter-sha256",
            "A" * 64,
        ]
    )

    assert args.rl_initial_adapter == "/tmp/final-adapter"
    assert args.rl_initial_adapter_sha256 == "A" * 64
