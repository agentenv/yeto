#!/usr/bin/env python3
"""Aggregate replicated strict-run policy confirmations with hash validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from typing import Sequence

import summarize_outer_policy_matrix as matrix


PRIMARY_LOSS_FIELDS = (
    "internal_loss",
    "holdout_loss",
    "holdout_indices7000_loss",
    "holdout_mean_loss",
)
EXPECTED_FRAGMENTS = (0, 1, 2, 3)
EXPECTED_LEARNERS = (0, 1, 2, 3)


class ConfirmationError(RuntimeError):
    """Raised when confirmation artifacts are missing, malformed, or mismatched."""


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise ConfirmationError(f"missing required hash artifact: {path}")
    if path.stat().st_size <= 0:
        raise ConfirmationError(f"hash artifact is empty: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ConfirmationError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _capture_path(probe_dir: Path, relative: object, *, field: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise ConfirmationError(
            f"{probe_dir / 'index.jsonl'}: invalid {field} {relative!r}"
        )
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ConfirmationError(
            f"{probe_dir / 'index.jsonl'}: {field} must be relative: {relative!r}"
        )
    resolved_probe = probe_dir.resolve()
    resolved = (probe_dir / candidate).resolve()
    if not resolved.is_relative_to(resolved_probe):
        raise ConfirmationError(
            f"{probe_dir / 'index.jsonl'}: {field} escapes capture directory: "
            f"{relative!r}"
        )
    return resolved


def parse_step1_hashes(run_dir: Path, run_name: str) -> dict:
    probe_dir = run_dir / "work" / run_name / "syncer_probe"
    index_path = probe_dir / "index.jsonl"
    if not index_path.is_file():
        raise ConfirmationError(f"missing probe capture index: {index_path}")
    try:
        lines = index_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConfirmationError(f"cannot read {index_path}: {exc}") from exc

    step1: dict[int, dict] = {}
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConfirmationError(
                f"{index_path}:{line_number}: malformed JSON: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise ConfirmationError(
                f"{index_path}:{line_number}: capture record must be an object"
            )
        step = row.get("step")
        if not isinstance(step, int) or isinstance(step, bool) or step <= 0:
            raise ConfirmationError(
                f"{index_path}:{line_number}: invalid step {step!r}"
            )
        if step != 1:
            continue
        learner = row.get("learner_id")
        if not isinstance(learner, int) or isinstance(learner, bool):
            raise ConfirmationError(
                f"{index_path}:{line_number}: invalid learner_id {learner!r}"
            )
        if learner in step1:
            raise ConfirmationError(
                f"{index_path}: duplicate step-1 learner record for learner {learner}"
            )
        step1[learner] = row

    if set(step1) != set(EXPECTED_LEARNERS):
        raise ConfirmationError(
            f"{index_path}: expected step-1 learners {list(EXPECTED_LEARNERS)}, "
            f"got {sorted(step1)}"
        )

    state_paths = {
        _capture_path(probe_dir, row.get("state_checkpoint"), field="state_checkpoint")
        for row in step1.values()
    }
    if len(state_paths) != 1:
        raise ConfirmationError(
            f"{index_path}: step-1 records reference multiple state checkpoints: "
            f"{sorted(str(path) for path in state_paths)}"
        )
    state_path = state_paths.pop()
    if state_path.name != "state_before_step_00000001.ckpt":
        raise ConfirmationError(
            f"{index_path}: unexpected step-1 state checkpoint name: {state_path.name!r}"
        )

    candidates = {}
    for learner in EXPECTED_LEARNERS:
        path = _capture_path(
            probe_dir,
            step1[learner].get("candidate_f32"),
            field=f"candidate_f32 for learner {learner}",
        )
        candidates[str(learner)] = {
            "path": str(path),
            "sha256": _sha256(path),
        }

    return {
        "state": {"path": str(state_path), "sha256": _sha256(state_path)},
        "candidates_by_learner": candidates,
    }


def parse_fragment_counts(tape_path: Path) -> dict[str, int]:
    try:
        lines = tape_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConfirmationError(f"cannot read {tape_path}: {exc}") from exc
    counts = {str(fragment): 0 for fragment in EXPECTED_FRAGMENTS}
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConfirmationError(
                f"{tape_path}:{line_number}: malformed JSON: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise ConfirmationError(
                f"{tape_path}:{line_number}: tape record must be an object"
            )
        fragment = row.get("fragment")
        if (
            not isinstance(fragment, int)
            or isinstance(fragment, bool)
            or fragment not in EXPECTED_FRAGMENTS
        ):
            raise ConfirmationError(
                f"{tape_path}:{line_number}: expected fragment in "
                f"{list(EXPECTED_FRAGMENTS)}, got {fragment!r}"
            )
        counts[str(fragment)] += 1
    return counts


def _validate_fragment_counts(
    counts: dict[str, int], *, outer_steps: int, expected_steps: int | None, context: str
) -> None:
    target_steps = outer_steps if expected_steps is None else expected_steps
    if target_steps % len(EXPECTED_FRAGMENTS) != 0:
        raise ConfirmationError(
            f"{context}: expected step count {target_steps} is not divisible by "
            f"{len(EXPECTED_FRAGMENTS)} fragments"
        )
    expected_per_fragment = target_steps // len(EXPECTED_FRAGMENTS)
    expected = {str(fragment): expected_per_fragment for fragment in EXPECTED_FRAGMENTS}
    if counts != expected:
        raise ConfirmationError(
            f"{context}: expected fragment commit counts {expected}, got {counts}"
        )


def _parse_extra_losses(
    run_dir: Path, extra_evals: Sequence[tuple[str, str]]
) -> dict[str, float]:
    losses = {}
    for label, filename in extra_evals:
        try:
            losses[label] = matrix.parse_eval_loss(run_dir / filename)
        except matrix.SummaryError as exc:
            raise ConfirmationError(
                f"extra eval {label!r} in {run_dir}: {exc}"
            ) from exc
    return losses


def _run_payload(run: matrix.RunResult, extra_losses: dict, hashes: dict) -> dict:
    return {
        **asdict(run),
        "extra_losses": extra_losses,
        "hashes": hashes,
    }


def summarize_pair(
    seed: int,
    reference_dir: Path,
    candidate_dir: Path,
    *,
    extra_evals: Sequence[tuple[str, str]],
    expected_steps: int | None,
) -> dict:
    if seed < 0:
        raise ConfirmationError(f"seed must be non-negative, got {seed}")
    reference_dir = reference_dir.expanduser().resolve()
    candidate_dir = candidate_dir.expanduser().resolve()
    if reference_dir == candidate_dir:
        raise ConfirmationError(
            f"seed {seed}: reference and candidate directories are identical"
        )
    try:
        reference = matrix.parse_run(reference_dir)
        candidate = matrix.parse_run(candidate_dir)
    except matrix.SummaryError as exc:
        raise ConfirmationError(f"seed {seed}: {exc}") from exc

    if reference.outer_steps != candidate.outer_steps:
        raise ConfirmationError(
            f"seed {seed}: nonmatching outer-step counts: "
            f"reference={reference.outer_steps}, candidate={candidate.outer_steps}"
        )
    if reference.quorum != candidate.quorum:
        raise ConfirmationError(
            f"seed {seed}: nonmatching quorums: "
            f"reference={reference.quorum}, candidate={candidate.quorum}"
        )
    if expected_steps is not None and reference.outer_steps != expected_steps:
        raise ConfirmationError(
            f"seed {seed}: expected {expected_steps} strict full-quorum groups, "
            f"got {reference.outer_steps}"
        )

    reference_tape = reference_dir / "work" / reference.run_name / "tape.jsonl"
    candidate_tape = candidate_dir / "work" / candidate.run_name / "tape.jsonl"
    reference_fragments = parse_fragment_counts(reference_tape)
    candidate_fragments = parse_fragment_counts(candidate_tape)
    _validate_fragment_counts(
        reference_fragments,
        outer_steps=reference.outer_steps,
        expected_steps=expected_steps,
        context=f"seed {seed} reference",
    )
    _validate_fragment_counts(
        candidate_fragments,
        outer_steps=candidate.outer_steps,
        expected_steps=expected_steps,
        context=f"seed {seed} candidate",
    )

    reference_hashes = parse_step1_hashes(reference_dir, reference.run_name)
    candidate_hashes = parse_step1_hashes(candidate_dir, candidate.run_name)
    if reference_hashes["state"]["sha256"] != candidate_hashes["state"]["sha256"]:
        raise ConfirmationError(f"seed {seed}: step-1 syncer-state SHA256 mismatch")
    for learner in map(str, EXPECTED_LEARNERS):
        reference_hash = reference_hashes["candidates_by_learner"][learner]["sha256"]
        candidate_hash = candidate_hashes["candidates_by_learner"][learner]["sha256"]
        if reference_hash != candidate_hash:
            raise ConfirmationError(
                f"seed {seed}: step-1 candidate SHA256 mismatch for learner {learner}"
            )

    reference_extras = _parse_extra_losses(reference_dir, extra_evals)
    candidate_extras = _parse_extra_losses(candidate_dir, extra_evals)
    gains = {
        field: getattr(reference, field) - getattr(candidate, field)
        for field in PRIMARY_LOSS_FIELDS
    }
    extra_gains = {
        label: reference_extras[label] - candidate_extras[label]
        for label, _ in extra_evals
    }
    return {
        "seed": seed,
        "outer_steps": reference.outer_steps,
        "quorum": reference.quorum,
        "fragment_commit_counts": {
            "reference": reference_fragments,
            "candidate": candidate_fragments,
        },
        "reference": _run_payload(reference, reference_extras, reference_hashes),
        "candidate": _run_payload(candidate, candidate_extras, candidate_hashes),
        "candidate_gain": {**gains, "extra_losses": extra_gains},
        "wall_time": {
            "reference_s": reference.wall_time_s,
            "candidate_s": candidate.wall_time_s,
            "candidate_minus_reference_s": candidate.wall_time_s
            - reference.wall_time_s,
            "candidate_relative_change": (
                candidate.wall_time_s - reference.wall_time_s
            )
            / reference.wall_time_s,
        },
        "hashes_match": True,
    }


def _worst_regression(pairs: Sequence[dict], field: str, *, extra: bool = False) -> dict:
    def gain(pair: dict) -> float:
        if extra:
            return float(pair["candidate_gain"]["extra_losses"][field])
        return float(pair["candidate_gain"][field])

    worst = min(pairs, key=gain)
    worst_gain = gain(worst)
    return {
        "seed": int(worst["seed"]),
        "candidate_gain": worst_gain,
        "regression": max(0.0, -worst_gain),
    }


def _sign(value: float) -> int:
    return 1 if value > 0.0 else (-1 if value < 0.0 else 0)


def aggregate_pairs(pairs: Sequence[dict], extra_evals: Sequence[tuple[str, str]]) -> dict:
    if not pairs:
        raise ConfirmationError("at least one --pair is required")
    mean_gains = {
        field: fmean(float(pair["candidate_gain"][field]) for pair in pairs)
        for field in PRIMARY_LOSS_FIELDS
    }
    positive_counts = {
        field: sum(float(pair["candidate_gain"][field]) > 0.0 for pair in pairs)
        for field in PRIMARY_LOSS_FIELDS
    }
    worst = {field: _worst_regression(pairs, field) for field in PRIMARY_LOSS_FIELDS}
    extra_mean = {
        label: fmean(
            float(pair["candidate_gain"]["extra_losses"][label]) for pair in pairs
        )
        for label, _ in extra_evals
    }
    extra_positive = {
        label: sum(
            float(pair["candidate_gain"]["extra_losses"][label]) > 0.0
            for pair in pairs
        )
        for label, _ in extra_evals
    }
    extra_worst = {
        label: _worst_regression(pairs, label, extra=True)
        for label, _ in extra_evals
    }

    gains_a = [float(pair["candidate_gain"]["holdout_loss"]) for pair in pairs]
    gains_b = [
        float(pair["candidate_gain"]["holdout_indices7000_loss"])
        for pair in pairs
    ]
    signs_a = [_sign(value) for value in gains_a]
    signs_b = [_sign(value) for value in gains_b]
    return {
        "mean_candidate_gain": {**mean_gains, "extra_losses": extra_mean},
        "seeds_positive_count": {
            **positive_counts,
            "extra_losses": extra_positive,
        },
        "worst_regression": {**worst, "extra_losses": extra_worst},
        "primary_holdout_agreement": {
            "all_seeds_same_sign_holdout_a": len(set(signs_a)) == 1,
            "all_seeds_same_sign_holdout_b": len(set(signs_b)) == 1,
            "every_seed_a_b_same_sign": all(
                sign_a == sign_b for sign_a, sign_b in zip(signs_a, signs_b)
            ),
            "all_seeds_agree_on_both_primary_holdouts": (
                len(set(signs_a)) == 1
                and len(set(signs_b)) == 1
                and all(sign_a == sign_b for sign_a, sign_b in zip(signs_a, signs_b))
            ),
            "candidate_better_on_both_primary_holdouts_every_seed": all(
                gain_a > 0.0 and gain_b > 0.0
                for gain_a, gain_b in zip(gains_a, gains_b)
            ),
        },
        "wall_time": {
            "reference_mean_s": fmean(
                float(pair["wall_time"]["reference_s"]) for pair in pairs
            ),
            "candidate_mean_s": fmean(
                float(pair["wall_time"]["candidate_s"]) for pair in pairs
            ),
            "mean_candidate_minus_reference_s": fmean(
                float(pair["wall_time"]["candidate_minus_reference_s"])
                for pair in pairs
            ),
            "mean_candidate_relative_change": fmean(
                float(pair["wall_time"]["candidate_relative_change"])
                for pair in pairs
            ),
        },
    }


def _validate_extra_specs(specs: Sequence[Sequence[str]] | None) -> list[tuple[str, str]]:
    validated: list[tuple[str, str]] = []
    seen_labels: set[str] = set()
    seen_filenames: set[str] = set()
    for raw_label, raw_filename in specs or []:
        label = raw_label.strip()
        filename = raw_filename.strip()
        if not label:
            raise ConfirmationError("extra-eval labels must not be empty")
        if label in seen_labels:
            raise ConfirmationError(f"duplicate extra-eval label: {label!r}")
        path = Path(filename)
        if not filename or path.is_absolute() or len(path.parts) != 1 or filename in {".", ".."}:
            raise ConfirmationError(
                f"extra-eval filename must be a single relative filename, got {raw_filename!r}"
            )
        if filename in seen_filenames:
            raise ConfirmationError(f"duplicate extra-eval filename: {filename!r}")
        seen_labels.add(label)
        seen_filenames.add(filename)
        validated.append((label, filename))
    return validated


def summarize_confirmations(
    pair_specs: Sequence[tuple[int, Path, Path]],
    *,
    reference_label: str,
    candidate_label: str,
    extra_evals: Sequence[tuple[str, str]],
    expected_steps: int | None,
) -> dict:
    if not pair_specs:
        raise ConfirmationError("at least one --pair is required")
    if not reference_label.strip() or not candidate_label.strip():
        raise ConfirmationError("reference and candidate labels must not be empty")
    if reference_label.strip() == candidate_label.strip():
        raise ConfirmationError("reference and candidate labels must differ")
    seeds = [seed for seed, _, _ in pair_specs]
    if len(seeds) != len(set(seeds)):
        duplicates = sorted(seed for seed in set(seeds) if seeds.count(seed) > 1)
        raise ConfirmationError(f"duplicate pair seeds: {duplicates}")
    if expected_steps is not None and expected_steps <= 0:
        raise ConfirmationError("--expected-steps must be positive")

    pairs = [
        summarize_pair(
            seed,
            reference_dir,
            candidate_dir,
            extra_evals=extra_evals,
            expected_steps=expected_steps,
        )
        for seed, reference_dir, candidate_dir in sorted(pair_specs)
    ]
    return {
        "schema": "outer_policy_confirmations_summary_v1",
        "reference_label": reference_label.strip(),
        "candidate_label": candidate_label.strip(),
        "gain_definition": "reference loss minus candidate loss; positive favors candidate",
        "expected_steps": expected_steps,
        "expected_fragments": list(EXPECTED_FRAGMENTS),
        "extra_evals": [
            {"label": label, "filename": filename} for label, filename in extra_evals
        ],
        "pair_count": len(pairs),
        "seeds": [int(pair["seed"]) for pair in pairs],
        "validation": {
            "all_runs_strict_full_quorum": True,
            "all_pairs_match_steps_and_quorum": True,
            "all_runs_have_equal_four_fragment_commit_counts": True,
            "all_step1_state_and_candidate_hashes_match": True,
            "all_extra_evals_parse_exactly_once": True,
        },
        "aggregate": aggregate_pairs(pairs, extra_evals),
        "per_seed": pairs,
    }


def _fmt(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def markdown(summary: dict) -> str:
    ref_label = summary["reference_label"]
    cand_label = summary["candidate_label"]
    lines = [
        "# Outer Policy Confirmations",
        "",
        f"Reference: `{ref_label}`",
        f"Candidate: `{cand_label}`",
        "",
        "Positive gain means the candidate has lower loss.",
        "",
        "| Seed | Ref internal | Cand internal | Gain | Ref A | Cand A | Gain | Ref B | Cand B | Gain | Ref mean | Cand mean | Gain | Ref wall (s) | Cand wall (s) | Wall delta (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for pair in summary["per_seed"]:
        ref = pair["reference"]
        cand = pair["candidate"]
        gain = pair["candidate_gain"]
        lines.append(
            "| {seed} | {ri} | {ci} | {ig} | {ra} | {ca} | {ag} | "
            "{rb} | {cb} | {bg} | {rm} | {cm} | {mg} | {rw:.1f} | "
            "{cw:.1f} | {wd:+.1f} |".format(
                seed=pair["seed"],
                ri=_fmt(ref["internal_loss"]),
                ci=_fmt(cand["internal_loss"]),
                ig=_fmt(gain["internal_loss"]),
                ra=_fmt(ref["holdout_loss"]),
                ca=_fmt(cand["holdout_loss"]),
                ag=_fmt(gain["holdout_loss"]),
                rb=_fmt(ref["holdout_indices7000_loss"]),
                cb=_fmt(cand["holdout_indices7000_loss"]),
                bg=_fmt(gain["holdout_indices7000_loss"]),
                rm=_fmt(ref["holdout_mean_loss"]),
                cm=_fmt(cand["holdout_mean_loss"]),
                mg=_fmt(gain["holdout_mean_loss"]),
                rw=pair["wall_time"]["reference_s"],
                cw=pair["wall_time"]["candidate_s"],
                wd=pair["wall_time"]["candidate_minus_reference_s"],
            )
        )

    if summary["extra_evals"]:
        lines.extend(
            [
                "",
                "## Extra Evaluations",
                "",
                "| Seed | Evaluation | Reference | Candidate | Gain |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for pair in summary["per_seed"]:
            for extra in summary["extra_evals"]:
                label = extra["label"]
                lines.append(
                    f"| {pair['seed']} | {label} | "
                    f"{_fmt(pair['reference']['extra_losses'][label])} | "
                    f"{_fmt(pair['candidate']['extra_losses'][label])} | "
                    f"{_fmt(pair['candidate_gain']['extra_losses'][label])} |"
                )

    aggregate = summary["aggregate"]
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "| Metric | Mean candidate gain | Seeds positive | Worst regression | Worst seed |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for field, label in [
        ("internal_loss", "Internal"),
        ("holdout_loss", "Holdout A"),
        ("holdout_indices7000_loss", "Holdout B"),
        ("holdout_mean_loss", "Primary mean"),
    ]:
        worst = aggregate["worst_regression"][field]
        lines.append(
            f"| {label} | {_fmt(aggregate['mean_candidate_gain'][field])} | "
            f"{aggregate['seeds_positive_count'][field]}/{summary['pair_count']} | "
            f"{_fmt(worst['regression'])} | {worst['seed']} |"
        )
    for extra in summary["extra_evals"]:
        label = extra["label"]
        worst = aggregate["worst_regression"]["extra_losses"][label]
        lines.append(
            f"| {label} | "
            f"{_fmt(aggregate['mean_candidate_gain']['extra_losses'][label])} | "
            f"{aggregate['seeds_positive_count']['extra_losses'][label]}/"
            f"{summary['pair_count']} | {_fmt(worst['regression'])} | "
            f"{worst['seed']} |"
        )

    agreement = aggregate["primary_holdout_agreement"]
    lines.extend(
        [
            "",
            "## Agreement And Validation",
            "",
            "- All seeds agree in sign on both primary holdouts: "
            f"`{agreement['all_seeds_agree_on_both_primary_holdouts']}`",
            "- Candidate is better on both primary holdouts for every seed: "
            f"`{agreement['candidate_better_on_both_primary_holdouts_every_seed']}`",
            "- Every run passed strict full-quorum validation: `True`",
            "- Every run has four equally committed fragments: `True`",
            "- Step-1 state and all four learner candidate hashes match: `True`",
            "",
            "## Matched Step-1 Hashes",
            "",
        ]
    )
    for pair in summary["per_seed"]:
        hashes = pair["reference"]["hashes"]
        lines.append(
            f"- Seed `{pair['seed']}` state: `{hashes['state']['sha256']}`"
        )
        for learner in map(str, EXPECTED_LEARNERS):
            lines.append(
                f"- Seed `{pair['seed']}` learner `{learner}` candidate: "
                f"`{hashes['candidates_by_learner'][learner]['sha256']}`"
            )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        action="append",
        nargs=3,
        required=True,
        metavar=("SEED", "REF_DIR", "CANDIDATE_DIR"),
        help="seed and matched strict-run directories; repeat for each seed",
    )
    parser.add_argument("--reference-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument(
        "--extra-eval",
        action="append",
        nargs=2,
        metavar=("LABEL", "FILENAME"),
        help="extra EVAL_LOSS label and filename present inside every arm directory",
    )
    parser.add_argument("--expected-steps", type=int, default=None)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        extra_evals = _validate_extra_specs(args.extra_eval)
        pair_specs = []
        for seed_text, reference_text, candidate_text in args.pair:
            try:
                seed = int(seed_text)
            except ValueError as exc:
                raise ConfirmationError(f"invalid seed {seed_text!r}") from exc
            pair_specs.append((seed, Path(reference_text), Path(candidate_text)))
        summary = summarize_confirmations(
            pair_specs,
            reference_label=args.reference_label,
            candidate_label=args.candidate_label,
            extra_evals=extra_evals,
            expected_steps=args.expected_steps,
        )
    except ConfirmationError as exc:
        raise SystemExit(f"error: {exc}") from exc

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    rendered = markdown(summary)
    args.out_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.out_md.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
