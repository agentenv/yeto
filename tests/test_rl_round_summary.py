"""``--rl-round-summary``: the per-round reward / response-length summary on the island tape.

A full-parameter island with asynchronous rounds on Miles records no reward
otherwise (its round event is ``rl_full_parameter_round``; LoRA islands carry
the reward in ``rl_local_round``), so ``auto`` turns the summary on for those
islands only, under the default asynchronous sync mode. Every other launch
keeps its argv: the customer's no-sync buffer trainer, LoRA islands, and every
``--rl-sync-mode strict`` launch (pinned by ``test_rl_sync_mode``).
CPU only.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from yeto.rl.adapters.miles import rollout_meta_hook as rmh

TWO = "modal:2xh100,modal:2xh100"
FULL = ("--tuning", "full", "--rl-sync-preset", "full-parameter", "--rl-placement", "fixed-partition",
        "--rl-rollout-gpus", "1", "--fragments", "4", "--rl-sync-full-parameter-rounds", "async",
        "--local-rl-rounds-per-sync", "2", "--rl-parameter-layout-sha256", "e" * 64)
AUTO_LINE = "[auto] per-round reward/response-length summary on"


@pytest.fixture(autouse=True)
def _island_hmac_key(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "round-summary-test-key")


def _commands(capsys, *extra: str, gpu: str = TWO) -> tuple[list[str], str]:
    from yeto.launcher import _prepare_rl_args, dry_run_plan

    args = _launcher_args("ports", extra, gpu=gpu)
    args.controller = "local"
    _prepare_rl_args(args)
    out = capsys.readouterr().out
    return [island["learner_command"] for island in dry_run_plan(args)["island_requests"]], out


def test_synced_full_parameter_islands_get_the_summary_and_the_launch_says_so(capsys):
    commands, out = _commands(capsys, *FULL)
    assert commands and all(" --rl-batch-summary" in command for command in commands)
    assert AUTO_LINE in out


def test_off_keeps_the_full_parameter_argv(capsys):
    off, out = _commands(capsys, *FULL, "--rl-round-summary", "off")
    auto, _ = _commands(capsys, *FULL)
    assert not any("--rl-batch-summary" in command for command in off)
    assert AUTO_LINE not in out
    assert [c.replace(" --rl-batch-summary", "") for c in auto] == off


@pytest.mark.parametrize("extra", [
    (),  # strict-avg LoRA
    ("--rl-sync-preset", "decoupled", "--fragments", "2", "--pipeline", "2", "--local-rl-rounds-per-sync", "2"),
])
def test_lora_islands_are_unchanged_under_auto(capsys, extra):
    commands, out = _commands(capsys, *extra)
    assert not any("--rl-batch-summary" in command for command in commands)
    assert AUTO_LINE not in out


def test_a_no_sync_island_is_unchanged_under_auto(capsys):
    commands, out = _commands(capsys, "--tuning", "full", "--rl-single-island-no-sync", gpu="modal:1xh100")
    assert not any("--rl-batch-summary" in command for command in commands)
    assert AUTO_LINE not in out


def test_the_strict_sweep_is_unchanged_under_auto(capsys):
    sweep = tuple(x for x in FULL if x not in ("--rl-sync-full-parameter-rounds", "async"))
    sweep = tuple("1" if (i and sweep[i - 1] == "--local-rl-rounds-per-sync") else x for i, x in enumerate(sweep))
    commands, out = _commands(capsys, *sweep, "--rl-sync-full-parameter-rounds", "sweep")
    assert not any("--rl-batch-summary" in command for command in commands)
    assert AUTO_LINE not in out


def test_on_turns_it_on_for_any_island_without_the_auto_line(capsys):
    commands, out = _commands(capsys, "--rl-round-summary", "on")
    assert commands and all(" --rl-batch-summary" in command for command in commands)
    assert AUTO_LINE not in out


def test_the_island_flag_reaches_the_rollout_hook(monkeypatch):
    monkeypatch.delenv(rmh.BATCH_SUMMARY_ENV, raising=False)
    assert not rmh.batch_summary_enabled(SimpleNamespace())
    assert rmh.batch_summary_enabled(SimpleNamespace(yeto_rl_batch_summary=True))
    assert rmh.batch_summary_enabled(SimpleNamespace(yeto_ext={"rl_batch_summary": True}))


def test_the_miles_island_parses_the_flag():
    from yeto.rl.adapters.miles import island_entry

    options = {o for action in island_entry.build_parser()._actions for o in action.option_strings}
    assert "--rl-batch-summary" in options


def test_strict_mode_full_parameter_islands_keep_their_argv(capsys):
    commands, out = _commands(capsys, *FULL, "--rl-sync-mode", "strict")
    assert not any("--rl-batch-summary" in command for command in commands)
    assert AUTO_LINE not in out
