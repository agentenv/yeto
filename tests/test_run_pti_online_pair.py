from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_pti_online_pair.py"
SPEC = importlib.util.spec_from_file_location("run_pti_online_pair", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def _compare_argv(tmp_path: Path) -> list[str]:
    return [
        "--model",
        "/models/qwen",
        "--data",
        "/data/train.parquet",
        "--work-dir",
        str(tmp_path / "work"),
        f"--report-dir={tmp_path / 'report'}",
        "--settings",
        "pti_m4_stock,pti_m4_candidate",
    ]


def test_passes_compare_argv_unchanged_then_runs_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], Path, bool]] = []

    def fake_run(command: list[str], *, cwd: Path, check: bool):
        calls.append((command, cwd, check))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(MOD.subprocess, "run", fake_run)
    compare = _compare_argv(tmp_path)
    output = tmp_path / "report" / "validation.json"
    MOD.run_compare_then_validate(
        compare_argv=compare,
        stock_arm="pti_m4_stock",
        pti_arm="pti_m4_candidate",
        output=output,
        python="/venv/bin/python",
    )
    assert len(calls) == 2
    assert calls[0] == (
        ["/venv/bin/python", str(MOD.COMPARE), *compare],
        MOD.REPO_ROOT,
        True,
    )
    validator = calls[1][0]
    assert validator[:2] == ["/venv/bin/python", str(MOD.VALIDATOR)]
    assert validator[2:] == [
        "--results",
        str((tmp_path / "report" / "results.jsonl").resolve()),
        "--stock-arm-dir",
        str((tmp_path / "work" / "pti_m4_stock").resolve()),
        "--pti-arm-dir",
        str((tmp_path / "work" / "pti_m4_candidate").resolve()),
        "--stock-arm",
        "pti_m4_stock",
        "--pti-arm",
        "pti_m4_candidate",
        "--output",
        str(output.resolve()),
    ]
    assert calls[1][1:] == (MOD.REPO_ROOT, True)


def test_compare_failure_prevents_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def fail_compare(command: list[str], *, cwd: Path, check: bool):
        nonlocal calls
        calls += 1
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(MOD.subprocess, "run", fail_compare)
    with pytest.raises(subprocess.CalledProcessError):
        MOD.run_compare_then_validate(
            compare_argv=_compare_argv(tmp_path),
            stock_arm="pti_m4_stock",
            pti_arm="pti_m4_candidate",
            output=tmp_path / "validation.json",
            python="python",
        )
    assert calls == 1


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        ([], "missing compare argv"),
        (["--settings", "a,b"], "--work-dir exactly once"),
        (
            [
                "--work-dir",
                "work",
                "--report-dir",
                "report",
                "--settings",
                "pti_m4_candidate,pti_m4_stock",
            ],
            "--settings must be exactly",
        ),
    ],
)
def test_rejects_ambiguous_or_mismatched_compare_argv(
    tmp_path: Path, argv: list[str], message: str
) -> None:
    with pytest.raises(MOD.RunnerError, match=message):
        MOD.run_compare_then_validate(
            compare_argv=argv,
            stock_arm="pti_m4_stock",
            pti_arm="pti_m4_candidate",
            output=tmp_path / "validation.json",
            python="python",
        )


def test_cli_splits_arguments_after_double_dash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    received = None

    def fake_run_compare_then_validate(**kwargs):
        nonlocal received
        received = kwargs

    monkeypatch.setattr(
        MOD, "run_compare_then_validate", fake_run_compare_then_validate
    )
    rc = MOD.main(
        [
            "--stock-arm",
            "pti_m1_stock",
            "--pti-arm",
            "pti_m1_candidate",
            "--output",
            str(tmp_path / "out.json"),
            "--python",
            "/venv/python",
            "--",
            "--work-dir",
            "work",
            "--report-dir",
            "report",
            "--settings",
            "pti_m1_stock,pti_m1_candidate",
        ]
    )
    assert rc == 0
    assert received is not None
    assert received["compare_argv"] == [
        "--work-dir",
        "work",
        "--report-dir",
        "report",
        "--settings",
        "pti_m1_stock,pti_m1_candidate",
    ]
