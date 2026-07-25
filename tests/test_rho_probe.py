"""Contract tests for the short rho probe runner and report."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rho_probe", ROOT / "scripts" / "rho_probe.py"
)
assert SPEC and SPEC.loader
rho_probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rho_probe)


def _args(tmp_path: Path, *extra: str):
    return rho_probe.parse_args(
        [
            "--scale",
            "135m",
            "--h",
            "16",
            "--m",
            "4",
            "--eta",
            "0.175",
            "--mu",
            "0.9",
            "--data",
            str(tmp_path / "train.jsonl"),
            "--eval-data",
            str(tmp_path / "eval.jsonl"),
            "--outer-rounds",
            "20",
            "--bootstrap-replicates",
            "200",
            "--device",
            "cpu",
            "--output",
            str(tmp_path / "probe.json"),
            "--run-dir",
            str(tmp_path / "run"),
            *extra,
        ]
    )


def _telemetry_rows(rounds: int = 20, workers: int = 4) -> list[dict]:
    rows = []
    for step in range(1, rounds + 1):
        fragment = (step - 1) % rho_probe.FRAGMENT_COUNT
        fragment_round = (step - 1) // rho_probe.FRAGMENT_COUNT + 1
        lags = {
            f"lag_{lag}": (
                0.85 - 0.1 * lag + 0.001 * step if fragment_round > lag else None
            )
            for lag in range(1, 5)
        }
        worker_rows = [
            {"learner_id": learner, "l2_norm": 2.0 + 0.1 * learner + 0.01 * step}
            for learner in range(workers)
        ]
        pairs = [
            {
                "learner_a": left,
                "learner_b": right,
                "cosine": 0.4 + 0.01 * step + 0.001 * (left + right),
            }
            for left in range(workers)
            for right in range(left + 1, workers)
        ]
        pair_values = [pair["cosine"] for pair in pairs]
        rows.append(
            {
                "schema": rho_probe.TELEMETRY_SCHEMA,
                "event": "outer_round_fragment",
                "outer_step": step,
                "fragment": fragment,
                "fragment_round": fragment_round,
                "pseudo_gradient": {
                    "definition": "production_merged_anchor_minus_candidate_before_outer_step",
                    "l2_norm": 3.0 + 0.05 * step,
                    "projected_l2_norm": 3.1 + 0.05 * step,
                },
                "autocorrelation": {
                    "estimator": "cosine_of_count_sketch_random_projections",
                    **lags,
                },
                "cross_worker": {
                    "definition": "current_syncer_anchor_minus_admitted_candidate_before_optional_delta_correction",
                    "estimator": "exact_cosine",
                    "worker_count": workers,
                    "pair_count": len(pairs),
                    "defined_pair_count": len(pairs),
                    "mean_cosine": (
                        sum(pair_values) / len(pair_values) if pair_values else None
                    ),
                    "min_cosine": min(pair_values) if pair_values else None,
                    "max_cosine": max(pair_values) if pair_values else None,
                    "workers": worker_rows,
                    "pairs": pairs,
                },
                "sketch": {
                    "method": "count_sketch_v1",
                    "dimension_per_tensor_group": rho_probe.PROJECTION_DIMENSION,
                    "seed": rho_probe.PROJECTION_SEED,
                    "tensor_group_count": 3,
                    "retained_lags": 4,
                },
            }
        )
    return rows


def test_build_compare_command_is_short_exact_and_enables_telemetry(tmp_path):
    args = _args(tmp_path)
    command, derived = rho_probe.build_compare_command(args)
    assert command[command.index("--settings") + 1] == "m4"
    assert command[command.index("--syncer-total-steps") + 1] == "20"
    assert command[command.index("--learner-max-steps") + 1] == "80"
    assert command[command.index("--fixed-window-microsteps") + 1] == "16"
    assert command[command.index("--outer-lr") + 1] == "0.17499999999999999"
    assert command[command.index("--outer-momentum") + 1] == "0.90000000000000002"
    assert "--rho-telemetry" in command
    assert "--strict-quorum" in command
    assert "--barrier-sync" in command
    assert "--version-matched-anchor" in command
    assert derived == {
        "fragment_rounds": 5,
        "learner_max_steps": 80,
        "token_budget": 40960,
        "learner_world_size": 1,
        "gpu_slots": 0,
        "gpu_offset": 0,
        "training_seed": 101101,
    }


def test_mocked_tiny_probe_writes_bootstrapped_report(tmp_path):
    args = _args(tmp_path)
    calls = []

    def fake_runner(command, cwd):
        calls.append((list(command), cwd))
        _output, _run_dir, telemetry = rho_probe._run_paths(args)
        telemetry.parent.mkdir(parents=True, exist_ok=True)
        telemetry.write_text(
            "".join(json.dumps(row) + "\n" for row in _telemetry_rows()),
            encoding="utf-8",
        )

    report = rho_probe.run_probe(args, runner=fake_runner)
    on_disk = json.loads(args.output.read_text(encoding="utf-8"))
    assert report["schema"] == rho_probe.REPORT_SCHEMA
    assert on_disk["schema"] == rho_probe.REPORT_SCHEMA
    assert report["status"] == "complete"
    assert report["config"]["outer_rounds"] == 20
    assert report["telemetry"]["record_count"] == 20
    assert report["telemetry"]["sha256"]
    assert report["work_evidence"] == {
        "full_registered_outer_rounds": True,
        "telemetry_present": True,
        "all_lags_defined": True,
    }
    for lag in ("1", "2", "3", "4"):
        estimate = report["rho"]["lags"][lag]["estimate"]
        interval = report["rho"]["lags"][lag]["ci_95"]
        assert estimate is not None
        assert interval["low"] <= estimate <= interval["high"]
    assert report["norms"]["merged_pseudo_gradient_l2"]["count"] == 20
    assert report["norms"]["worker_pseudo_gradient_l2"]["count"] == 80
    assert report["cross_worker"]["pair_cosine"]["count"] == 120
    assert report["cross_worker"]["undefined_pair_count"] == 0
    assert len(calls) == 1
    assert calls[0][1] == ROOT


def test_telemetry_validation_rejects_incomplete_rounds(tmp_path):
    path = tmp_path / "rho.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in _telemetry_rows()[:-1]),
        encoding="utf-8",
    )
    with pytest.raises(rho_probe.ProbeError, match="expected exactly 20"):
        rho_probe.load_telemetry(path, expected_rounds=20, expected_workers=4)


def test_m8_preset_is_available_for_registered_m_axis():
    compare_spec = importlib.util.spec_from_file_location(
        "compare_diloco_for_rho", ROOT / "scripts" / "compare_diloco.py"
    )
    assert compare_spec and compare_spec.loader
    compare = importlib.util.module_from_spec(compare_spec)
    sys.modules[compare_spec.name] = compare
    compare_spec.loader.exec_module(compare)
    assert compare.PRESETS["m8"].m == 8


def test_gpu_offset_is_forwarded_for_one_cell_per_gpu_probe(tmp_path):
    args = _args(tmp_path, "--gpu-slots", "1", "--gpu-offset", "6")
    command, derived = rho_probe.build_compare_command(args)
    assert command[command.index("--gpu-slots") + 1] == "1"
    assert command[command.index("--gpu-offset") + 1] == "6"
    assert derived["gpu_slots"] == 1
    assert derived["gpu_offset"] == 6


def test_scientific_probe_seed_uses_registered_decimal_concatenation(tmp_path):
    args = _args(tmp_path)
    command, derived = rho_probe.build_compare_command(args)
    assert command[command.index("--training-seed") + 1] == "101101"
    assert derived["training_seed"] == 101101
