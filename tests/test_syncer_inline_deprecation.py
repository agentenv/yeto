"""The scheduler-inline syncer send flags still work but are deprecated."""

from __future__ import annotations

import types

import pytest

from yeto import cli
from yeto.launch import syncer as launch_syncer


def _launch_parser():
    return cli.build_parser()._subparsers._group_actions[0].choices["launch"]


@pytest.fixture(autouse=True)
def _fresh_warning_state(monkeypatch):
    monkeypatch.setattr(launch_syncer, "_warned_deprecated", set())


@pytest.mark.parametrize("flag", ["--rl-sync-inline-resend", "--rl-sync-inline-broadcast"])
def test_help_marks_inline_flags_deprecated(flag):
    action = next(a for a in _launch_parser()._actions if flag in a.option_strings)
    assert action.help.startswith("DEPRECATED")


def test_inline_flags_pass_through_and_warn_once(capsys):
    args = types.SimpleNamespace(rl_sync_inline_resend=True, rl_sync_inline_broadcast=True)
    first = launch_syncer._syncer_inline_flags(args)
    second = launch_syncer._syncer_inline_flags(args)
    assert first == second == " --inline-resend --inline-broadcast"
    err = capsys.readouterr().err
    assert err.count("--rl-sync-inline-resend is deprecated") == 1
    assert err.count("--rl-sync-inline-broadcast is deprecated") == 1


def test_no_inline_flags_no_warning(capsys):
    args = types.SimpleNamespace(rl_sync_inline_resend=False)
    assert launch_syncer._syncer_inline_flags(args) == ""
    assert capsys.readouterr().err == ""
