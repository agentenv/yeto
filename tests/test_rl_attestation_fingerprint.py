"""--rl-print-attestation-fingerprint: the CPU entry prints exactly the
runtime fingerprint the island computes (same launch object, same function)."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))

from yeto.rl.adapters.miles import island_entry as rl_learner  # noqa: E402
from yeto.rl.adapters.miles import entry  # noqa: E402


def _launch():
    import argparse

    from test_rl_argv_snapshot import _captured_args
    from yeto.rl.engine import run_config as rc
    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.adapters.miles import config as mc

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


def test_both_paths_hash_the_same_launch_with_the_same_function(monkeypatch, tmp_path):
    import pytest

    from test_rl_algorithm_provenance import drive_run_miles

    # island path: run_ports_island -> preflight_stage hashes the launch it was given
    hashed, staged = [], []
    monkeypatch.setattr(entry, "ports_runtime_fingerprint",
                        lambda launch: hashed.append(launch) or "fp")
    monkeypatch.setattr(entry, "miles_capabilities",
                        lambda fp, unverified_mechanisms=(): staged.append(fp) or "caps")
    monkeypatch.setattr(entry, "with_partitioned_serial", lambda caps: caps)
    monkeypatch.setattr(entry, "execution_profile_for", lambda *a, **k: "profile")
    monkeypatch.setattr(entry, "expected_algorithm_sha256", lambda a: None)
    monkeypatch.setattr(entry, "preflight", lambda *a: None)
    monkeypatch.setattr(entry, "elastic_wiring_for", lambda a, profile, fingerprint: None)
    monkeypatch.setattr(entry, "resolve_harness_preflight", lambda a: None)
    from yeto.rl.adapters.miles import state

    monkeypatch.setattr(state, "require_run_plugin", lambda: None)
    real_stage = entry.preflight_stage

    class Stop(Exception):
        pass

    def stage(miles_args, launch, algorithm, **kw):
        real_stage(miles_args, launch, algorithm, **kw)
        raise Stop  # nothing after the pre-Ray stage runs here

    monkeypatch.setattr(entry, "preflight_stage", stage)
    launch = SimpleNamespace(argv=["--x"])
    with pytest.raises(Stop):
        entry.run_ports_island(SimpleNamespace(), launch, None, learner_id=0,
                               base_model_revision="r", lora_config_hash="c",
                               layout_hash=None, yeto_policy_sync=False)
    assert hashed == [launch] and staged == ["fp"]
    # CPU entry: printed with the same function as the island
    hashed.clear()
    out = io.StringIO()
    assert rl_learner.print_attestation_fingerprint(
        SimpleNamespace(rl_print_attestation_fingerprint=True), launch, out)
    assert hashed == [launch] and json.loads(out.getvalue())["runtime_fingerprint"] == "fp"
    # run_miles: the launch it prints (after verifying) is the very one the island gets
    calls = drive_run_miles(tmp_path)["calls"]
    assert [c["call"] for c in calls] == ["verify", "print", "run_ports"]
    assert len({c["launch"] for c in calls}) == 1


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
