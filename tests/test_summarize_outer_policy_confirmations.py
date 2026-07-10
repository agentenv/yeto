"""Focused tests for replicated outer-policy confirmation summaries."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _load():
    name = "summarize_outer_policy_confirmations"
    spec = importlib.util.spec_from_file_location(
        name, SCRIPTS / "summarize_outer_policy_confirmations.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


confirm = _load()


def _write_run(
    root: Path,
    name: str,
    *,
    seed: int,
    internal: float,
    holdout_a: float,
    holdout_b: float,
    wall: float,
    steps: int = 80,
    quorum: int = 4,
    state_payload: bytes | None = None,
    candidate_suffix: bytes = b"",
    extra_losses: dict[str, float] | None = None,
) -> Path:
    run = root / name
    arm_dir = run / "work" / "m4"
    probe_dir = arm_dir / "syncer_probe"
    state_dir = probe_dir / "states"
    candidate_dir = probe_dir / "candidates"
    state_dir.mkdir(parents=True)
    candidate_dir.mkdir(parents=True)

    (run / "run.log").write_text(
        "\n".join(
            [
                "[compare] model=qwen35-9b budget=700000 tokens",
                f"  {'m4':<10} M={quorum} 2500 steps/learner P=4",
                f"[compare] m4 eval loss/token: {internal} ({wall}s)",
                "",
            ]
        )
    )
    (run / "holdout_eval.log").write_text(f"EVAL_LOSS {holdout_a}\n")
    (run / "holdout_indices7000_eval.log").write_text(
        f"EVAL_LOSS {holdout_b}\n"
    )
    for filename, loss in (extra_losses or {}).items():
        (run / filename).write_text(f"EVAL_LOSS {loss}\n")

    responders = [{"id": learner} for learner in range(quorum)]
    (arm_dir / "tape.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "step": step,
                    "fragment": (step - 1) % 4,
                    "responders": responders,
                }
            )
            + "\n"
            for step in range(1, steps + 1)
        )
    )

    state_name = "state_before_step_00000001.ckpt"
    (state_dir / state_name).write_bytes(
        state_payload if state_payload is not None else f"state-{seed}".encode()
    )
    index_rows = []
    for learner in range(4):
        candidate_name = (
            f"candidate_step_00000001_fragment_0000_learner_{learner:04d}.f32"
        )
        (candidate_dir / candidate_name).write_bytes(
            f"candidate-{seed}-{learner}".encode() + candidate_suffix
        )
        index_rows.append(
            {
                "step": 1,
                "fragment": 0,
                "learner_id": learner,
                "state_checkpoint": f"states/{state_name}",
                "candidate_f32": f"candidates/{candidate_name}",
            }
        )
    (probe_dir / "index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in index_rows)
    )
    return run


def _pair(
    root: Path,
    seed: int,
    *,
    ref_losses: tuple[float, float, float] = (1.40, 1.42, 1.44),
    cand_losses: tuple[float, float, float] = (1.38, 1.39, 1.41),
    ref_wall: float = 1500.0,
    cand_wall: float = 1510.0,
    steps: int = 80,
    extra_ref: float = 1.43,
    extra_cand: float = 1.40,
) -> tuple[Path, Path]:
    reference = _write_run(
        root,
        f"ref-{seed}",
        seed=seed,
        internal=ref_losses[0],
        holdout_a=ref_losses[1],
        holdout_b=ref_losses[2],
        wall=ref_wall,
        steps=steps,
        extra_losses={"third_eval.log": extra_ref},
    )
    candidate = _write_run(
        root,
        f"cand-{seed}",
        seed=seed,
        internal=cand_losses[0],
        holdout_a=cand_losses[1],
        holdout_b=cand_losses[2],
        wall=cand_wall,
        steps=steps,
        extra_losses={"third_eval.log": extra_cand},
    )
    return reference, candidate


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_cli_aggregates_confirmations_hashes_and_extra_evals(tmp_path, capsys):
    ref211, cand211 = _pair(tmp_path, 211)
    ref223, cand223 = _pair(
        tmp_path,
        223,
        ref_losses=(1.50, 1.51, 1.52),
        cand_losses=(1.47, 1.48, 1.50),
        ref_wall=1520.0,
        cand_wall=1515.0,
        extra_ref=1.53,
        extra_cand=1.49,
    )
    out_json = tmp_path / "out" / "summary.json"
    out_md = tmp_path / "out" / "summary.md"

    assert confirm.main(
        [
            "--pair",
            "211",
            str(ref211),
            str(cand211),
            "--pair",
            "223",
            str(ref223),
            str(cand223),
            "--reference-label",
            "nesterov_mu05_lr0175",
            "--candidate-label",
            "outer_sgd_lr035",
            "--extra-eval",
            "third_holdout",
            "third_eval.log",
            "--expected-steps",
            "80",
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ]
    ) == 0

    summary = json.loads(out_json.read_text())
    assert summary["schema"] == "outer_policy_confirmations_summary_v1"
    assert summary["seeds"] == [211, 223]
    assert summary["validation"] == {
        "all_extra_evals_parse_exactly_once": True,
        "all_pairs_match_steps_and_quorum": True,
        "all_runs_have_equal_four_fragment_commit_counts": True,
        "all_runs_strict_full_quorum": True,
        "all_step1_state_and_candidate_hashes_match": True,
    }
    seed211 = summary["per_seed"][0]
    assert seed211["fragment_commit_counts"]["reference"] == {
        "0": 20,
        "1": 20,
        "2": 20,
        "3": 20,
    }
    assert seed211["candidate_gain"]["holdout_loss"] == pytest.approx(0.03)
    assert seed211["candidate_gain"]["holdout_indices7000_loss"] == pytest.approx(
        0.03
    )
    assert seed211["candidate_gain"]["holdout_mean_loss"] == pytest.approx(0.03)
    assert seed211["candidate_gain"]["extra_losses"]["third_holdout"] == pytest.approx(
        0.03
    )
    assert seed211["wall_time"]["candidate_minus_reference_s"] == 10.0
    assert seed211["reference"]["hashes"]["state"]["sha256"] == _sha(b"state-211")
    assert seed211["reference"]["hashes"]["candidates_by_learner"]["2"][
        "sha256"
    ] == _sha(b"candidate-211-2")
    aggregate = summary["aggregate"]
    assert aggregate["mean_candidate_gain"]["holdout_mean_loss"] == pytest.approx(
        0.0275
    )
    assert aggregate["seeds_positive_count"]["holdout_mean_loss"] == 2
    assert aggregate["worst_regression"]["holdout_mean_loss"]["regression"] == 0.0
    assert aggregate["primary_holdout_agreement"][
        "all_seeds_agree_on_both_primary_holdouts"
    ] is True
    assert out_md.read_text() == capsys.readouterr().out
    assert "third_holdout" in out_md.read_text()


def test_aggregate_reports_regression_and_primary_disagreement(tmp_path):
    ref1, cand1 = _pair(tmp_path, 1)
    ref2, cand2 = _pair(
        tmp_path,
        2,
        ref_losses=(1.4, 1.4, 1.4),
        cand_losses=(1.5, 1.5, 1.3),
        extra_ref=1.4,
        extra_cand=1.5,
    )

    summary = confirm.summarize_confirmations(
        [(1, ref1, cand1), (2, ref2, cand2)],
        reference_label="ref",
        candidate_label="cand",
        extra_evals=[("third", "third_eval.log")],
        expected_steps=80,
    )

    aggregate = summary["aggregate"]
    assert aggregate["seeds_positive_count"]["holdout_loss"] == 1
    assert aggregate["worst_regression"]["holdout_loss"] == {
        "seed": 2,
        "candidate_gain": pytest.approx(-0.1),
        "regression": pytest.approx(0.1),
    }
    assert aggregate["worst_regression"]["extra_losses"]["third"][
        "regression"
    ] == pytest.approx(0.1)
    assert aggregate["primary_holdout_agreement"][
        "all_seeds_agree_on_both_primary_holdouts"
    ] is False
    assert aggregate["primary_holdout_agreement"][
        "candidate_better_on_both_primary_holdouts_every_seed"
    ] is False


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("state", "syncer-state SHA256 mismatch"),
        ("candidate", "candidate SHA256 mismatch for learner 3"),
    ],
)
def test_step1_hash_mismatches_fail(tmp_path, mutation, message):
    reference, candidate = _pair(tmp_path, 211)
    if mutation == "state":
        state = (
            candidate
            / "work/m4/syncer_probe/states/state_before_step_00000001.ckpt"
        )
        state.write_bytes(b"different state")
    else:
        candidate_file = (
            candidate
            / "work/m4/syncer_probe/candidates/"
            "candidate_step_00000001_fragment_0000_learner_0003.f32"
        )
        candidate_file.write_bytes(b"different candidate")

    with pytest.raises(confirm.ConfirmationError, match=message):
        confirm.summarize_pair(
            211,
            reference,
            candidate,
            extra_evals=[],
            expected_steps=80,
        )


def test_expected_steps_and_fragment_commit_counts_are_enforced(tmp_path):
    reference, candidate = _pair(tmp_path, 211, steps=8)
    with pytest.raises(confirm.ConfirmationError, match="expected 80"):
        confirm.summarize_pair(
            211,
            reference,
            candidate,
            extra_evals=[],
            expected_steps=80,
        )

    reference, candidate = _pair(tmp_path / "imbalanced", 211)
    tape = candidate / "work/m4/tape.jsonl"
    rows = [json.loads(line) for line in tape.read_text().splitlines()]
    rows[-1]["fragment"] = 0
    tape.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(confirm.ConfirmationError, match="fragment commit counts"):
        confirm.summarize_pair(
            211,
            reference,
            candidate,
            extra_evals=[],
            expected_steps=80,
        )


def test_pair_must_match_steps_and_quorum(tmp_path):
    reference, candidate = _pair(tmp_path, 211)
    candidate_short = _write_run(
        tmp_path,
        "candidate-short",
        seed=211,
        internal=1.3,
        holdout_a=1.3,
        holdout_b=1.3,
        wall=100.0,
        steps=76,
    )
    with pytest.raises(confirm.ConfirmationError, match="nonmatching outer-step counts"):
        confirm.summarize_pair(
            211,
            reference,
            candidate_short,
            extra_evals=[],
            expected_steps=None,
        )

    candidate_q3 = _write_run(
        tmp_path,
        "candidate-q3",
        seed=211,
        internal=1.3,
        holdout_a=1.3,
        holdout_b=1.3,
        wall=100.0,
        quorum=3,
    )
    with pytest.raises(confirm.ConfirmationError, match="nonmatching quorums"):
        confirm.summarize_pair(
            211,
            reference,
            candidate_q3,
            extra_evals=[],
            expected_steps=None,
        )


def test_existing_matrix_parser_rejects_partial_quorum(tmp_path):
    reference, candidate = _pair(tmp_path, 211)
    tape = candidate / "work/m4/tape.jsonl"
    rows = [json.loads(line) for line in tape.read_text().splitlines()]
    rows[0]["responders"] = rows[0]["responders"][:-1]
    tape.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(confirm.ConfirmationError, match="non-full-quorum"):
        confirm.summarize_pair(
            211,
            reference,
            candidate,
            extra_evals=[],
            expected_steps=80,
        )


def test_extra_eval_must_exist_and_contain_exactly_one_loss(tmp_path):
    reference, candidate = _pair(tmp_path, 211)
    (candidate / "third_eval.log").write_text("EVAL_LOSS 1.0\nEVAL_LOSS 1.1\n")
    with pytest.raises(confirm.ConfirmationError, match="exactly one EVAL_LOSS"):
        confirm.summarize_pair(
            211,
            reference,
            candidate,
            extra_evals=[("third", "third_eval.log")],
            expected_steps=80,
        )

    (candidate / "third_eval.log").unlink()
    with pytest.raises(confirm.ConfirmationError, match="missing required file"):
        confirm.summarize_pair(
            211,
            reference,
            candidate,
            extra_evals=[("third", "third_eval.log")],
            expected_steps=80,
        )


def test_missing_or_malformed_probe_capture_fails(tmp_path):
    reference, candidate = _pair(tmp_path, 211)
    index = candidate / "work/m4/syncer_probe/index.jsonl"
    index.unlink()
    with pytest.raises(confirm.ConfirmationError, match="missing probe capture index"):
        confirm.summarize_pair(
            211,
            reference,
            candidate,
            extra_evals=[],
            expected_steps=80,
        )

    reference, candidate = _pair(tmp_path / "malformed", 211)
    index = candidate / "work/m4/syncer_probe/index.jsonl"
    index.write_text("{bad json}\n")
    with pytest.raises(confirm.ConfirmationError, match="malformed JSON"):
        confirm.summarize_pair(
            211,
            reference,
            candidate,
            extra_evals=[],
            expected_steps=80,
        )


def test_duplicate_seeds_labels_and_extra_specs_fail(tmp_path):
    reference, candidate = _pair(tmp_path, 211)
    with pytest.raises(confirm.ConfirmationError, match="duplicate pair seeds"):
        confirm.summarize_confirmations(
            [(211, reference, candidate), (211, reference, candidate)],
            reference_label="ref",
            candidate_label="cand",
            extra_evals=[],
            expected_steps=80,
        )
    with pytest.raises(confirm.ConfirmationError, match="labels must differ"):
        confirm.summarize_confirmations(
            [(211, reference, candidate)],
            reference_label="same",
            candidate_label="same",
            extra_evals=[],
            expected_steps=80,
        )
    with pytest.raises(confirm.ConfirmationError, match="duplicate extra-eval label"):
        confirm._validate_extra_specs(
            [("third", "a.log"), ("third", "b.log")]
        )
    with pytest.raises(confirm.ConfirmationError, match="single relative filename"):
        confirm._validate_extra_specs([("third", "nested/a.log")])


def test_cli_rejects_invalid_seed(tmp_path):
    reference, candidate = _pair(tmp_path, 211)
    with pytest.raises(SystemExit, match="invalid seed"):
        confirm.main(
            [
                "--pair",
                "not-a-seed",
                str(reference),
                str(candidate),
                "--reference-label",
                "ref",
                "--candidate-label",
                "cand",
                "--out-json",
                str(tmp_path / "out.json"),
                "--out-md",
                str(tmp_path / "out.md"),
            ]
        )
