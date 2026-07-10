"""Focused tests for outer-step event-tape diagnostics."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent


def _load():
    name = "summarize_outer_step_diagnostics"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / "summarize_outer_step_diagnostics.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


diagnostics = _load()


def _record(
    step: int,
    fragment: int,
    *,
    gnorm: float,
    step_norm: float,
    cosine: float | None,
    history_ratio: float | None,
    restarted: bool,
) -> dict:
    return {
        "step": step,
        "fragment": fragment,
        "gnorm": gnorm,
        "outer_step_norm": step_norm,
        "outer_direction_cosine": cosine,
        "outer_history_current_ratio": history_ratio,
        "outer_restarted": restarted,
        "responders": [{"id": 0}],
        "ms": 10,
    }


def _write_tape(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def _base_rows() -> list[dict]:
    return [
        _record(
            1,
            0,
            gnorm=2.0,
            step_norm=1.0,
            cosine=None,
            history_ratio=None,
            restarted=False,
        ),
        _record(
            2,
            1,
            gnorm=4.0,
            step_norm=2.0,
            cosine=0.2,
            history_ratio=0.0,
            restarted=True,
        ),
        _record(
            3,
            0,
            gnorm=6.0,
            step_norm=3.0,
            cosine=-0.4,
            history_ratio=1.0,
            restarted=False,
        ),
        _record(
            4,
            1,
            gnorm=8.0,
            step_norm=4.0,
            cosine=0.8,
            history_ratio=3.0,
            restarted=True,
        ),
    ]


def test_cli_reports_overall_halves_fragments_and_writes_outputs(tmp_path, capsys):
    tape_a = _write_tape(tmp_path / "a.jsonl", _base_rows())
    tape_b = _write_tape(tmp_path / "b.jsonl", list(reversed(_base_rows())))
    out_json = tmp_path / "out" / "summary.json"
    out_md = tmp_path / "out" / "summary.md"

    assert diagnostics.main(
        [
            "--run",
            "normalized",
            str(tape_a),
            "--run",
            "restarted",
            str(tape_b),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ]
    ) == 0

    summary = json.loads(out_json.read_text())
    assert summary["schema"] == "outer_step_diagnostics_summary_v1"
    assert summary["run_count"] == 2
    assert "ranking" not in summary
    run = summary["runs"][0]
    overall = run["overall"]
    assert overall["record_count"] == 4
    assert overall["outer_step_norm"]["mean"] == pytest.approx(2.5)
    assert overall["outer_step_norm"]["p50"] == pytest.approx(2.5)
    assert overall["outer_step_norm"]["p95"] == pytest.approx(3.85)
    assert overall["outer_direction_cosine"]["mean"] == pytest.approx(0.2)
    assert overall["outer_direction_cosine"]["p10"] == pytest.approx(-0.28)
    assert overall["outer_direction_cosine"]["undefined_fraction"] == 0.25
    assert overall["outer_history_current_ratio"]["mean"] == pytest.approx(4 / 3)
    assert overall["outer_history_current_ratio"]["p50"] == pytest.approx(1.0)
    assert overall["outer_history_current_ratio"]["p95"] == pytest.approx(2.8)
    assert overall["outer_history_current_ratio"]["undefined_fraction"] == 0.25
    assert overall["restart_count"] == 2
    assert overall["restart_rate"] == 0.5
    assert overall["gnorm_to_outer_step_norm_ratio"] == pytest.approx(2.0)
    assert run["first_half"]["step_start"] == 1
    assert run["first_half"]["step_end"] == 2
    assert run["second_half"]["step_start"] == 3
    assert run["second_half"]["step_end"] == 4
    assert [row["fragment"] for row in run["per_fragment"]] == [0, 1]
    assert run["per_fragment"][0]["record_count"] == 2
    assert summary["runs"][1]["first_half"]["step_start"] == 1
    assert out_md.read_text() == capsys.readouterr().out
    assert "No policy ranking is performed" in out_md.read_text()


def test_all_undefined_optional_diagnostics_are_reported(tmp_path):
    rows = [
        _record(
            step,
            0,
            gnorm=1.0,
            step_norm=1.0,
            cosine=None,
            history_ratio=None,
            restarted=False,
        )
        for step in range(1, 3)
    ]
    run = diagnostics.summarize_run("nesterov", _write_tape(tmp_path / "t.jsonl", rows))

    assert run["overall"]["outer_direction_cosine"] == {
        "defined_count": 0,
        "mean": None,
        "undefined_fraction": 1.0,
        "p10": None,
    }
    assert run["overall"]["outer_history_current_ratio"] == {
        "defined_count": 0,
        "mean": None,
        "undefined_fraction": 1.0,
        "p50": None,
        "p95": None,
    }


def test_zero_total_step_norm_reports_undefined_ratio(tmp_path):
    rows = [
        _record(
            1,
            0,
            gnorm=1.0,
            step_norm=0.0,
            cosine=0.0,
            history_ratio=0.0,
            restarted=False,
        )
    ]
    run = diagnostics.summarize_run("zero", _write_tape(tmp_path / "zero.jsonl", rows))

    assert run["overall"]["gnorm_to_outer_step_norm_ratio"] is None
    assert run["first_half"]["record_count"] == 0
    assert run["second_half"]["record_count"] == 1


@pytest.mark.parametrize(
    "specs, message",
    [
        ([], "at least one --run"),
        ([("", Path("unused"))], "labels must not be empty"),
    ],
)
def test_missing_labels_fail(specs, message):
    with pytest.raises(diagnostics.SummaryError, match=message):
        diagnostics.summarize_runs(specs)


def test_duplicate_labels_fail_before_second_tape_is_read(tmp_path):
    tape = _write_tape(tmp_path / "tape.jsonl", _base_rows())

    with pytest.raises(diagnostics.SummaryError, match="duplicate run label"):
        diagnostics.summarize_runs(
            [("same", tape), ("same", tmp_path / "missing.jsonl")]
        )


def test_missing_empty_and_malformed_tapes_fail(tmp_path):
    with pytest.raises(diagnostics.SummaryError, match="missing tape file"):
        diagnostics.parse_tape(tmp_path / "missing.jsonl")

    empty = _write_tape(tmp_path / "empty.jsonl", [])
    with pytest.raises(diagnostics.SummaryError, match="no records"):
        diagnostics.parse_tape(empty)

    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text("{bad json}\n")
    with pytest.raises(diagnostics.SummaryError, match="malformed JSON"):
        diagnostics.parse_tape(malformed)


def test_noncontiguous_and_duplicate_steps_fail(tmp_path):
    rows = _base_rows()
    rows[2]["step"] = 5
    with pytest.raises(diagnostics.SummaryError, match="unique and contiguous"):
        diagnostics.parse_tape(_write_tape(tmp_path / "gap.jsonl", rows))

    rows = _base_rows()
    rows[2]["step"] = 2
    with pytest.raises(diagnostics.SummaryError, match="unique and contiguous"):
        diagnostics.parse_tape(_write_tape(tmp_path / "duplicate.jsonl", rows))


def test_missing_required_diagnostic_is_inconsistent_schema(tmp_path):
    rows = _base_rows()
    del rows[1]["outer_history_current_ratio"]

    with pytest.raises(diagnostics.SummaryError, match="inconsistent step schema"):
        diagnostics.parse_tape(_write_tape(tmp_path / "missing-field.jsonl", rows))


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("gnorm", float("nan"), "gnorm must be finite"),
        ("outer_step_norm", float("inf"), "outer_step_norm must be finite"),
        ("gnorm", -1.0, "gnorm must be >="),
        ("outer_step_norm", -1.0, "outer_step_norm must be >="),
        ("outer_direction_cosine", 1.1, "outer_direction_cosine must be <="),
        ("outer_history_current_ratio", -0.1, "must be >="),
        ("outer_step_norm", "1.0", "must be a finite number"),
        ("gnorm", True, "must be a finite number"),
    ],
)
def test_malformed_or_nonfinite_numeric_fields_fail(tmp_path, field, value, message):
    rows = _base_rows()
    rows[0][field] = value

    with pytest.raises(diagnostics.SummaryError, match=message):
        diagnostics.parse_tape(_write_tape(tmp_path / "bad-number.jsonl", rows))


@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_invalid_restart_booleans_fail(tmp_path, value):
    rows = _base_rows()
    rows[0]["outer_restarted"] = value

    with pytest.raises(diagnostics.SummaryError, match="must be a boolean"):
        diagnostics.parse_tape(_write_tape(tmp_path / "bad-bool.jsonl", rows))


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("step", True, "invalid step"),
        ("step", 0, "invalid step"),
        ("fragment", True, "invalid fragment"),
        ("fragment", -1, "invalid fragment"),
    ],
)
def test_invalid_step_and_fragment_schema_fail(tmp_path, field, value, message):
    rows = _base_rows()
    rows[0][field] = value

    with pytest.raises(diagnostics.SummaryError, match=message):
        diagnostics.parse_tape(_write_tape(tmp_path / "bad-schema.jsonl", rows))
