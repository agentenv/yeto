"""--rl-print-attestation-fingerprint: the CPU entry prints exactly the
runtime fingerprint the island computes (same launch object, same function)."""

from __future__ import annotations

import inspect
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))

from yeto.rl import learner as rl_learner  # noqa: E402
from yeto.rl.engine.miles_adapter import entry  # noqa: E402


def _launch():
    import argparse

    from test_rl_argv_snapshot import _captured_args
    from yeto.rl.engine import run_config as rc
    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.miles_adapter import config as mc

    (base,), kwargs = _captured_args()
    part = argparse.Namespace(**{**vars(base), "rl_placement": "fixed-partition",
                                 "rollout_num_gpus": 1})
    return mc.translate_run_config(rc.resolve_rl_run_config(part, **kwargs), AlgorithmSpec())


def test_printed_value_is_the_island_value():
    from yeto.rl import MILES_NEXT_COMMIT

    launch = _launch()
    out = io.StringIO()
    args = SimpleNamespace(rl_print_attestation_fingerprint=True, learner_id=1)
    assert rl_learner.print_attestation_fingerprint(args, launch, out) is True
    record = json.loads(out.getvalue())
    assert record["runtime_fingerprint"] == entry.runtime_fingerprint(launch, MILES_NEXT_COMMIT)
    assert record["runtime_fingerprint"] == entry.ports_runtime_fingerprint(launch)
    assert record["miles_argv"] == list(launch.argv) and record["learner_id"] == 1
    assert not rl_learner.print_attestation_fingerprint(SimpleNamespace(), launch, out)


def test_both_paths_hash_the_same_launch_with_the_same_function():
    island = inspect.getsource(entry.preflight_stage)  # IR-1: pre-Ray stage of run_ports_island
    assert "fingerprint = ports_runtime_fingerprint(launch)" in island
    assert "preflight_stage(" in inspect.getsource(entry.run_ports_island)
    run = inspect.getsource(rl_learner.run_miles)
    # printed right after the launch is built and verified, before anything else
    i_build = run.index("ports_launch = build_ports_launch(")
    i_verify = run.index("verify_ports_algorithm(args, miles_args, ports_launch)")
    i_print = run.index("print_attestation_fingerprint(args, ports_launch)")
    i_run = run.index("_run_ports(")
    assert i_build < i_verify < i_print < i_run
    # the island receives this very launch object
    assert "ports_launch,\n" in run[i_run:i_run + 200]
    ports = inspect.getsource(rl_learner._run_ports)
    assert "run_ports_island(" in ports and "launch," in ports


def test_launcher_forwards_the_flag_and_it_is_not_in_the_miles_argv(tmp_path, monkeypatch):
    from rl_e2e_launch import island_run, learner_from_run

    base = ("--rl-placement", "fixed-partition", "--rl-rollout-gpus", "1",
            "--gpu", "aws:2xa100@us-east-1")
    args, _ = learner_from_run(island_run(base + ("--rl-print-attestation-fingerprint",),
                                          monkeypatch), tmp_path / "h")
    assert args.rl_print_attestation_fingerprint
    args2, _ = learner_from_run(island_run(base, monkeypatch), tmp_path / "h2")
    assert not args2.rl_print_attestation_fingerprint
    # a learner-only flag: it is not part of the hashed Miles argv
    assert "--rl-print-attestation-fingerprint" not in " ".join(_launch().argv)
