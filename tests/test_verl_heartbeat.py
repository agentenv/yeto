"""A synced verl island heartbeats every 30 s by default, as a Miles island does.

An outer-synced island's rounds wait for the other islands (the first round, a
terminal round in the syncer's grace window) and write nothing meanwhile, which
the dashboard reads as a stalled island. The Miles island heartbeats by default
there; the verl island now follows the same rule (``train_options.heartbeat_plan``):
``--rl-heartbeat-interval`` when given (0: off), else 30 s on a synced island with
an ``[auto]`` line, or with ``--rl-observe-timeline``. A no-sync island keeps its
default. The launcher forwards the option only when given, so the default island
argv is unchanged. CPU only.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from test_verl_decoupled import DECOUPLED_ISLAND, SYNCED, _initial, _island_argv, _plan, _prepared
from verl_island_fakes import ISLAND_ARGV, drive_trainer, run_island_entry, stub_verl_and_ray
from yeto.rl.adapters.verl import island_entry
from yeto.rl.adapters.verl import train_options as to

AUTO = "[auto] rl_heartbeat every 30 s on this outer-synced island (--rl-heartbeat-interval 0 turns it off)"
SYNCER = "127.0.0.1:9"


@pytest.mark.parametrize("synced, observe, value, expected, auto", [
    (True, False, None, 30.0, True),
    (True, True, None, 30.0, False),
    (False, False, None, None, False),  # the customer's single trainer: unchanged
    (False, True, None, 30.0, False),
    (True, False, 0.0, None, False),
    (True, False, 12.0, 12.0, False),
    (False, False, 12.0, 12.0, False),
])
def test_the_heartbeat_rule_is_the_miles_islands(synced, observe, value, expected, auto, capsys):
    plan = to.heartbeat_plan(SimpleNamespace(rl_observe_timeline=observe, rl_heartbeat_interval=value),
                             synced=synced)
    assert plan.get("heartbeat_interval_s") == expected
    assert set(plan) <= {"heartbeat_interval_s"}
    assert (AUTO in capsys.readouterr().out) is auto


def test_a_negative_interval_is_refused():
    with pytest.raises(ValueError, match="--rl-heartbeat-interval must be >= 0"):
        to.heartbeat_plan(SimpleNamespace(rl_heartbeat_interval=-1.0), synced=True)


def test_the_default_launch_forwards_nothing_and_the_decoupled_island_heartbeats():
    argv = _island_argv(_prepared(), 0, SYNCER)
    assert "--rl-heartbeat-interval" not in argv  # the default argv is unchanged
    parsed, _ = island_entry.parse_args(argv)
    assert parsed.rl_heartbeat_interval is None
    assert to.heartbeat_plan(parsed, synced=True) == {"heartbeat_interval_s": 30.0}


@pytest.mark.parametrize("value, expected", [("0", {}), ("12", {"heartbeat_interval_s": 12.0})])
def test_an_explicit_interval_reaches_the_island(value, expected):
    argv = _island_argv(_prepared("--rl-heartbeat-interval", value), 0, SYNCER)
    assert argv[argv.index("--rl-heartbeat-interval") + 1] == repr(float(value))
    parsed, ignored = island_entry.parse_args(argv)
    assert "--rl-heartbeat-interval" not in ignored and parsed.rl_heartbeat_interval == float(value)
    assert to.heartbeat_plan(parsed, synced=True) == expected


def test_the_island_entry_prints_the_auto_line_on_a_synced_lora_island_only(monkeypatch, tmp_path, capsys):
    name, _ = run_island_entry(monkeypatch, tmp_path, *DECOUPLED_ISLAND, stop="eval_section", base=SYNCED)
    assert name == "eval_section" and AUTO in capsys.readouterr().out
    name, _ = run_island_entry(monkeypatch, tmp_path, *DECOUPLED_ISLAND, "--rl-heartbeat-interval", "0",
                               stop="eval_section", base=SYNCED)
    assert name == "eval_section" and AUTO not in capsys.readouterr().out
    name, _ = run_island_entry(monkeypatch, tmp_path, stop="eval_section", base=ISLAND_ARGV)  # no sync
    assert name == "eval_section" and AUTO not in capsys.readouterr().out
    name, message = run_island_entry(monkeypatch, tmp_path, *DECOUPLED_ISLAND, "--rl-heartbeat-interval", "-1",
                                     stop="eval_section", base=SYNCED)
    assert name == "refused" and "--rl-heartbeat-interval must be >= 0" in message


def _lora_plan(tmp_path, **extra) -> dict:
    parsed, _ = island_entry.parse_args(_island_argv(_prepared(), 0, SYNCER))
    return {**_plan(parsed, tmp_path, 0), "sync": "decoupled", "parameter_mode": "lora", "max_policy_age": 0,
            "model_path": str(tmp_path), "lora_rank": 2, "lora_targets": "all-linear", "out_dir": str(tmp_path),
            "event_tape": str(tmp_path / "tape.jsonl"), "thresholds_key": [], **extra}


@pytest.mark.parametrize("extra, expected", [({"heartbeat_interval_s": 30.0}, 30.0), ({}, None)])
def test_the_lora_island_driver_gets_the_planned_heartbeat(monkeypatch, tmp_path, extra, expected):
    from yeto.rl import export

    monkeypatch.setattr(export, "derive_peft_lora_specs", lambda *a, **k: _initial().specs)
    with stub_verl_and_ray():
        driver = drive_trainer(monkeypatch, tmp_path, "sync", plan=_lora_plan(tmp_path, **extra),
                               until="wire_round_cuts")["driver"]
    assert driver.heartbeat_interval_s == expected
    assert driver.observe is False and driver.resource_sample_interval_s is None
