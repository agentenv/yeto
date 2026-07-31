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
