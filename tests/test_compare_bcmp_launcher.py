"""Focused pure-logic tests for compare_diloco BC-MP launch plumbing."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parent.parent


def _load_compare():
    name = "compare_diloco_bcmp_launcher"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / "compare_diloco.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


compare = _load_compare()


def _learner_args(**overrides):
    values = {
        "model": "lfm25-230m",
        "lora_r": 16,
        "lora_alpha": 32,
        "seq_len": 128,
        "micro_batch_size": 1,
        "inner_lr": 3e-4,
        "device": "cpu",
        "shard": "ddp",
        "learner_gpus": 0,
        "training_seed": 223223,
        "tuning": "lora",
        "bcmp_shadow_path": False,
        "bcmp_shadow_every": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_bcmp_shadow_paths_are_deterministic_per_async_learner():
    arm_dir = Path("/tmp/run/work/m4")
    args = _learner_args(bcmp_shadow_path=True, bcmp_shadow_every=7)

    command = compare.learner_command(
        args,
        arm_dir,
        learner_id=3,
        num_learners=4,
        syncer="127.0.0.1:9000",
        max_steps=8,
        arm=compare.PRESETS["m4"],
    )

    assert command[command.index("--bcmp-shadow-path") + 1] == str(
        arm_dir / "bcmp_shadow_learner_3.jsonl"
    )
    assert command[command.index("--bcmp-shadow-every") + 1] == "7"


def test_bcmp_shadow_flags_are_absent_when_disabled_or_for_baseline():
    arm_dir = Path("/tmp/run/work/m4")
    disabled = compare.learner_command(
        _learner_args(),
        arm_dir,
        learner_id=0,
        num_learners=4,
        syncer="127.0.0.1:9000",
        max_steps=8,
        arm=compare.PRESETS["m4"],
    )
    baseline = compare.learner_command(
        _learner_args(bcmp_shadow_path=True, bcmp_shadow_every=7),
        Path("/tmp/run/work/baseline"),
        learner_id=0,
        num_learners=1,
        syncer="none",
        max_steps=8,
        arm=None,
    )

    for command in (disabled, baseline):
        assert "--bcmp-shadow-path" not in command
        assert "--bcmp-shadow-every" not in command


def test_skip_baseline_omits_run_record_and_baseline_delta(
    monkeypatch, tmp_path, capsys
):
    work_dir = tmp_path / "work"
    report_dir = tmp_path / "report"
    train_path = work_dir / "train.jsonl"
    eval_path = work_dir / "eval.jsonl"
    baseline_calls = []

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_diloco.py",
            "--data",
            "unused.jsonl",
            "--settings",
            "m2",
            "--skip-baseline",
            "--work-dir",
            str(work_dir),
            "--report-dir",
            str(report_dir),
        ],
    )
    monkeypatch.setattr(compare, "persist_reproducibility_metadata", lambda _: None)
    monkeypatch.setattr(compare, "ensure_syncer", lambda: None)
    monkeypatch.setattr(
        compare,
        "split_data",
        lambda *args, **kwargs: (train_path, eval_path, 10),
    )
    monkeypatch.setattr(compare, "eval_in_subprocess", lambda *args, **kwargs: 1.25)

    def fail_baseline(*args, **kwargs):
        baseline_calls.append((args, kwargs))
        raise AssertionError("skip-baseline must not launch the baseline learner")

    monkeypatch.setattr(compare, "run_baseline", fail_baseline)
    monkeypatch.setattr(
        compare,
        "run_diloco",
        lambda args, arm, work: (work / arm.name / "export", 2.0),
    )

    assert compare.main() == 0
    assert baseline_calls == []

    records = [
        json.loads(line)
        for line in (report_dir / "results.jsonl").read_text().splitlines()
    ]
    assert [record["arm"] for record in records] == ["base (untrained)", "m2"]
    report = (report_dir / "report.md").read_text()
    assert report.startswith("# DiLoCo comparison —")
    assert "baseline (sync" not in report
    assert "| m2 | 2 | 2 | 1.2500 | — |" in report
    assert "synchronous baseline: omitted" in capsys.readouterr().out


def test_skip_baseline_and_injected_loss_are_mutually_exclusive(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_diloco.py",
            "--data",
            "unused.jsonl",
            "--skip-baseline",
            "--baseline-loss",
            "1.0",
            "--dry-run",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        compare.main()
    assert exc_info.value.code == 2


def test_bcmp_shadow_cadence_must_be_positive(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_diloco.py",
            "--data",
            "unused.jsonl",
            "--bcmp-shadow-path",
            "--bcmp-shadow-every",
            "0",
            "--dry-run",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        compare.main()
    assert exc_info.value.code == 2


def test_bcmp_shadow_rejects_plain_sgd_arm_before_launch(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_diloco.py",
            "--data",
            "unused.jsonl",
            "--settings",
            "scaffold_lite",
            "--bcmp-shadow-path",
            "--dry-run",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        compare.main()
    assert exc_info.value.code == 2
