import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent


def _load_script():
    name = "summarize_outer_momentum_runs"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / "summarize_outer_momentum_runs.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


summary_script = _load_script()


def _write_run(
    directory: Path,
    *,
    internal: float,
    holdout_a: float,
    holdout_b: float,
    wall_s: float,
    steps: int = 4,
    quorum: int = 4,
    partial_step: int | None = None,
) -> None:
    directory.mkdir(parents=True)
    (directory / "run.log").write_text(
        "[compare] model=qwen35-9b budget=700000 tokens\n"
        f"  m4         M={quorum} 1368 steps/learner P=4\n"
        f"[compare] m4 eval loss/token: {internal} ({wall_s}s)\n",
        encoding="utf-8",
    )
    (directory / "holdout_eval.log").write_text(
        f"warning before result\nEVAL_LOSS {holdout_a}\n", encoding="utf-8"
    )
    (directory / "holdout_indices7000_eval.log").write_text(
        f"EVAL_LOSS {holdout_b}\n", encoding="utf-8"
    )
    tape = directory / "work" / "m4" / "tape.jsonl"
    tape.parent.mkdir(parents=True)
    records = []
    for step in range(1, steps + 1):
        count = quorum - 1 if step == partial_step else quorum
        records.append(
            json.dumps(
                {
                    "step": step,
                    "fragment": (step - 1) % quorum,
                    "responders": [{"id": responder} for responder in range(count)],
                }
            )
        )
    tape.write_text("\n".join(records) + "\n", encoding="utf-8")


def test_cli_summarizes_multiple_paired_runs(tmp_path):
    baseline_1 = tmp_path / "baseline-149"
    treatment_1 = tmp_path / "treatment-149"
    baseline_2 = tmp_path / "baseline-163"
    treatment_2 = tmp_path / "treatment-163"
    _write_run(
        baseline_1,
        internal=1.4356,
        holdout_a=1.432700,
        holdout_b=1.436437,
        wall_s=1504,
    )
    _write_run(
        treatment_1,
        internal=1.3951,
        holdout_a=1.391431,
        holdout_b=1.395241,
        wall_s=1509,
    )
    _write_run(
        baseline_2,
        internal=1.3904,
        holdout_a=1.426981,
        holdout_b=1.431063,
        wall_s=1517,
    )
    _write_run(
        treatment_2,
        internal=1.3497,
        holdout_a=1.387235,
        holdout_b=1.392627,
        wall_s=1491,
    )
    out_json = tmp_path / "out" / "summary.json"
    out_md = tmp_path / "out" / "summary.md"

    assert summary_script.main(
        [
            "--pair",
            "163",
            str(baseline_2),
            str(treatment_2),
            "--pair",
            "149",
            str(baseline_1),
            str(treatment_1),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ]
    ) == 0

    result = json.loads(out_json.read_text(encoding="utf-8"))
    assert result["seeds"] == [149, 163]
    assert result["validation"]["all_runs_full_quorum"] is True
    assert result["per_seed"][0]["outer_steps"] == 4
    assert result["per_seed"][0]["quorum"] == 4
    assert result["per_seed"][0]["loss_improvement"]["holdout_loss"] == pytest.approx(
        0.041269
    )
    assert result["aggregate"]["mean_loss_improvement"][
        "holdout_indices7000_loss"
    ] == pytest.approx((0.041196 + 0.038436) / 2)
    assert result["aggregate"]["all_seeds_treatment_better"]["internal_loss"]
    markdown = out_md.read_text(encoding="utf-8")
    assert "| 149 | 4 | 4 |" in markdown
    assert "Mean two-holdout improvement" in markdown


def test_missing_or_duplicate_eval_result_fails_loudly(tmp_path):
    run = tmp_path / "run"
    _write_run(
        run,
        internal=1.4,
        holdout_a=1.3,
        holdout_b=1.2,
        wall_s=100,
    )
    (run / "holdout_eval.log").write_text(
        "EVAL_LOSS 1.3\nEVAL_LOSS 1.2\n", encoding="utf-8"
    )

    with pytest.raises(summary_script.SummaryError, match="exactly one EVAL_LOSS"):
        summary_script.parse_run(run)

    (run / "holdout_eval.log").unlink()
    with pytest.raises(summary_script.SummaryError, match="missing required file"):
        summary_script.parse_run(run)


def test_non_full_quorum_tape_fails_loudly(tmp_path):
    run = tmp_path / "partial"
    _write_run(
        run,
        internal=1.4,
        holdout_a=1.3,
        holdout_b=1.2,
        wall_s=100,
        partial_step=3,
    )

    with pytest.raises(summary_script.SummaryError, match="non-full-quorum step 3"):
        summary_script.parse_run(run)


def test_nonmatching_outer_step_counts_fail_loudly(tmp_path):
    baseline = tmp_path / "baseline"
    treatment = tmp_path / "treatment"
    _write_run(
        baseline,
        internal=1.4,
        holdout_a=1.3,
        holdout_b=1.2,
        wall_s=100,
        steps=4,
    )
    _write_run(
        treatment,
        internal=1.3,
        holdout_a=1.2,
        holdout_b=1.1,
        wall_s=101,
        steps=5,
    )

    with pytest.raises(summary_script.SummaryError, match="nonmatching outer-step counts"):
        summary_script.summarize_pair(149, baseline, treatment)


def test_tape_steps_must_be_contiguous_and_ordered(tmp_path):
    run = tmp_path / "bad-steps"
    _write_run(
        run,
        internal=1.4,
        holdout_a=1.3,
        holdout_b=1.2,
        wall_s=100,
    )
    tape = run / "work" / "m4" / "tape.jsonl"
    rows = [json.loads(line) for line in tape.read_text(encoding="utf-8").splitlines()]
    rows[2]["step"] = 4
    tape.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    with pytest.raises(summary_script.SummaryError, match="contiguous and ordered"):
        summary_script.parse_run(run)
