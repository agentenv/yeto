"""The verl island checks ``--rl-expected-algorithm-sha256`` as the Miles island does.

The launcher hashes the run's AlgorithmSpec once (``rl_ports``) and sends it to every
island; a verl island whose own spec (the ``--rl-algorithm-spec`` file, or the default)
hashes differently stops with exit 2 before any data or GPU work.
"""

from __future__ import annotations

import json

import pytest

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from test_verl_full_parameter_launch import _island_argv, _prepared, _verl_args
from test_verl_island_flags import REQUIRED
from yeto.rl.adapters.verl import island_entry
from yeto.rl.engine.algorithm import AlgorithmSpec

DEFAULT = AlgorithmSpec().sha256()


def _expected(argv: list[str]) -> str:
    return argv[argv.index("--rl-expected-algorithm-sha256") + 1]


@pytest.mark.parametrize("make", [
    lambda: _verl_args(),
    lambda: _launcher_args("ports", ("--rl-backend", "verl", "--reward-function", "gsm8k_reward:score"),
                           gpu="modal:1xh200,modal:1xh200"),
], ids=["full-parameter", "lora-two-islands"])
def test_the_launcher_s_verl_islands_pass_the_check(make, _fake_sky, _no_modal_listing):  # noqa: F811
    argv = _island_argv(_prepared(make()), 0)
    args, unread = island_entry.parse_args(argv)
    assert "--rl-expected-algorithm-sha256" not in unread  # read, no longer dropped
    assert island_entry.verify_algorithm_hash(args, island_entry.algorithm_spec_dict(args)) == _expected(argv)


def test_a_different_hash_stops_the_island_with_exit_2(capsys):
    args, _ = island_entry.parse_args([*REQUIRED, "--rl-expected-algorithm-sha256", "f" * 64])
    with pytest.raises(SystemExit) as exc:
        island_entry.verify_algorithm_hash(args, None)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "rl_algorithm_mismatch" in err and DEFAULT in err and "f" * 64 in err


def test_the_spec_file_is_what_is_hashed(tmp_path):
    spec = AlgorithmSpec(kl_coef=0.01)
    path = tmp_path / "algorithm_spec.json"
    path.write_text(spec.canonical_json())
    args, _ = island_entry.parse_args([*REQUIRED, "--rl-algorithm-spec", str(path),
                                       "--rl-expected-algorithm-sha256", spec.sha256().upper()])
    assert island_entry.verify_algorithm_hash(args, island_entry.algorithm_spec_dict(args)) == spec.sha256()
    args, _ = island_entry.parse_args([*REQUIRED, "--rl-algorithm-spec", str(path),
                                       "--rl-expected-algorithm-sha256", DEFAULT])
    with pytest.raises(SystemExit) as exc:  # the file's spec, not the default, is trained
        island_entry.verify_algorithm_hash(args, json.loads(path.read_text()))
    assert exc.value.code == 2


def test_an_island_started_by_hand_warns_and_runs(capsys):
    args, _ = island_entry.parse_args(REQUIRED)
    assert island_entry.verify_algorithm_hash(args, None) == DEFAULT
    assert "no --rl-expected-algorithm-sha256" in capsys.readouterr().err


def test_main_checks_before_any_work(monkeypatch):
    def no_work(*_args, **_kwargs):
        raise AssertionError("reached the island's work")

    monkeypatch.setattr(island_entry, "prepare_data", no_work)
    with pytest.raises(SystemExit) as exc:
        island_entry.main([*REQUIRED, "--rl-single-island-no-sync", "--rl-expected-algorithm-sha256", "0" * 64])
    assert exc.value.code == 2
