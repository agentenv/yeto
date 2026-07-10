"""Focused tests for the strict-run outer-policy matrix summarizer."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent


def _load():
    name = "summarize_outer_policy_matrix"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / "summarize_outer_policy_matrix.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


matrix = _load()


def _write_run(
    root: Path,
    name: str,
    *,
    internal: float = 1.4,
    holdout_a: float = 1.5,
    holdout_b: float = 1.6,
    wall: float = 100.0,
    steps: int = 4,
    quorum: int = 4,
    run_name: str = "m4",
) -> Path:
    run = root / name
    tape_dir = run / "work" / run_name
    tape_dir.mkdir(parents=True)
    (run / "run.log").write_text(
        "\n".join(
            [
                "[compare] model=qwen35-9b budget=350000 tokens",
                f"  {run_name:<10} M={quorum} 950 steps/learner P=4",
                f"[compare] {run_name} eval loss/token: {internal} ({wall}s)",
                "",
            ]
        )
    )
    (run / "holdout_eval.log").write_text(f"EVAL_LOSS {holdout_a}\n")
    (run / "holdout_indices7000_eval.log").write_text(
        f"EVAL_LOSS {holdout_b}\n"
    )
    responders = [{"id": responder_id} for responder_id in range(quorum)]
    (tape_dir / "tape.jsonl").write_text(
        "".join(
            json.dumps({"step": step, "responders": responders}) + "\n"
            for step in range(1, steps + 1)
        )
    )
    return run


def _arm(summary: dict, label: str) -> dict:
    return next(row for row in summary["arms"] if row["label"] == label)


def test_cli_writes_ranked_matrix_and_reference_gains(tmp_path, capsys):
    baseline = _write_run(
        tmp_path, "baseline", internal=1.50, holdout_a=1.48, holdout_b=1.52
    )
    normalized = _write_run(
        tmp_path, "normalized", internal=1.40, holdout_a=1.38, holdout_b=1.42
    )
    restarted = _write_run(
        tmp_path, "restarted", internal=1.43, holdout_a=1.41, holdout_b=1.45
    )
    out_json = tmp_path / "out" / "matrix.json"
    out_md = tmp_path / "out" / "matrix.md"

    assert matrix.main(
        [
            "--arm",
            "baseline",
            str(baseline),
            "--arm",
            "normalized",
            str(normalized),
            "--arm",
            "restarted",
            str(restarted),
            "--reference",
            "baseline",
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ]
    ) == 0

    summary = json.loads(out_json.read_text())
    assert summary["schema"] == "outer_policy_matrix_summary_v1"
    assert summary["reference_label"] == "baseline"
    assert summary["ranking"] == ["normalized", "restarted", "baseline"]
    assert summary["outer_steps"] == 4
    assert summary["quorum"] == 4
    assert _arm(summary, "normalized")["rank"] == 1
    assert _arm(summary, "normalized")["holdout_mean_loss"] == pytest.approx(1.40)
    assert _arm(summary, "normalized")["gain_vs_reference"][
        "holdout_mean_loss"
    ] == pytest.approx(0.10)
    assert _arm(summary, "baseline")["gain_vs_reference"][
        "holdout_mean_loss"
    ] == pytest.approx(0.0)
    assert out_md.read_text() == capsys.readouterr().out
    assert "| 1 | normalized |" in out_md.read_text()


def test_run_log_ignores_timed_synchronous_baseline_result(tmp_path):
    run = _write_run(tmp_path, "with-baseline")
    text = (run / "run.log").read_text().replace(
        "[compare] m4 eval",
        "[compare] baseline eval loss/token: 1.7 (50s)\n[compare] m4 eval",
    )
    (run / "run.log").write_text(text)

    parsed = matrix.parse_run(run)

    assert parsed.run_name == "m4"
    assert parsed.internal_loss == pytest.approx(1.4)


def test_summary_without_reference_uses_null_gains(tmp_path):
    run = _write_run(tmp_path, "only")

    summary = matrix.summarize_policy_matrix([("only", run)])

    assert summary["reference_label"] is None
    assert summary["ranking"] == ["only"]
    assert summary["arms"][0]["gain_vs_reference"] is None
    assert "Reference: `none`" in matrix.markdown(summary)


@pytest.mark.parametrize(
    ("specs", "reference", "message"),
    [
        ([], None, "at least one --arm"),
        ([("", Path("unused"))], None, "labels must not be empty"),
    ],
)
def test_missing_arm_labels_fail_loudly(specs, reference, message):
    with pytest.raises(matrix.SummaryError, match=message):
        matrix.summarize_policy_matrix(specs, reference=reference)


def test_duplicate_and_unknown_reference_labels_fail_loudly(tmp_path):
    run = _write_run(tmp_path, "run")

    with pytest.raises(matrix.SummaryError, match="duplicate arm label"):
        matrix.summarize_policy_matrix([("same", run), ("same", run)])
    with pytest.raises(matrix.SummaryError, match="reference label 'missing'"):
        matrix.summarize_policy_matrix([("present", run)], reference="missing")


def test_missing_holdout_and_duplicate_internal_results_fail(tmp_path):
    missing = _write_run(tmp_path, "missing")
    (missing / "holdout_eval.log").unlink()
    with pytest.raises(matrix.SummaryError, match="missing required file"):
        matrix.parse_run(missing)

    duplicate = _write_run(tmp_path, "duplicate")
    with (duplicate / "run.log").open("a") as handle:
        handle.write("[compare] m4 eval loss/token: 1.3 (99s)\n")
    with pytest.raises(matrix.SummaryError, match="exactly one internal"):
        matrix.parse_run(duplicate)


def test_malformed_holdout_and_tape_fail_loudly(tmp_path):
    bad_holdout = _write_run(tmp_path, "bad-holdout")
    (bad_holdout / "holdout_eval.log").write_text(
        "EVAL_LOSS 1.2\nEVAL_LOSS 1.3\n"
    )
    with pytest.raises(matrix.SummaryError, match="exactly one EVAL_LOSS"):
        matrix.parse_run(bad_holdout)

    bad_tape = _write_run(tmp_path, "bad-tape")
    (bad_tape / "work" / "m4" / "tape.jsonl").write_text("{not json}\n")
    with pytest.raises(matrix.SummaryError, match="malformed JSON"):
        matrix.parse_run(bad_tape)


def test_partial_quorum_is_rejected(tmp_path):
    run = _write_run(tmp_path, "partial")
    tape = run / "work" / "m4" / "tape.jsonl"
    rows = [json.loads(line) for line in tape.read_text().splitlines()]
    rows[2]["responders"] = rows[2]["responders"][:-1]
    tape.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(matrix.SummaryError, match="non-full-quorum step 3"):
        matrix.parse_run(run)


def test_noncontiguous_steps_are_rejected(tmp_path):
    run = _write_run(tmp_path, "gap")
    tape = run / "work" / "m4" / "tape.jsonl"
    rows = [json.loads(line) for line in tape.read_text().splitlines()]
    rows[2]["step"] = 4
    tape.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(matrix.SummaryError, match="must be contiguous"):
        matrix.parse_run(run)


def test_matrix_rejects_nonmatching_steps_or_quorum(tmp_path):
    reference = _write_run(tmp_path, "reference", steps=4, quorum=4)
    short = _write_run(tmp_path, "short", steps=3, quorum=4)
    with pytest.raises(matrix.SummaryError, match="nonmatching outer-step counts"):
        matrix.summarize_policy_matrix([("reference", reference), ("short", short)])

    q3 = _write_run(tmp_path, "q3", steps=4, quorum=3)
    with pytest.raises(matrix.SummaryError, match="nonmatching configured quorums"):
        matrix.summarize_policy_matrix([("reference", reference), ("q3", q3)])


def test_missing_run_directory_and_mismatched_run_name_fail(tmp_path):
    with pytest.raises(matrix.SummaryError, match="run directory does not exist"):
        matrix.parse_run(tmp_path / "absent")

    run = _write_run(tmp_path, "name-mismatch")
    text = (run / "run.log").read_text().replace(
        "[compare] m4 eval", "[compare] other eval"
    )
    (run / "run.log").write_text(text)
    with pytest.raises(matrix.SummaryError, match="for configured arm 'm4'"):
        matrix.parse_run(run)
