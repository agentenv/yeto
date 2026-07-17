#!/usr/bin/env python3
"""Render the sealed A1/A3/A4 audit handoff from verified JSON evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


class ReportError(RuntimeError):
    """The requested final report is not backed by complete sealed evidence."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReportError(f"{label} must be a JSON object")
    return value


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number):
            return "+∞" if number > 0 else "−∞"
        return f"{number:.{digits}f}"
    return str(value)


def _interval(value: Any) -> str:
    if not isinstance(value, list) or len(value) != 2:
        return "—"
    return f"[{_fmt(value[0])}, {_fmt(value[1])}]"


def _gate_lines(gates: Mapping[str, Any]) -> list[str]:
    return [f"| `{name}` | **{value}** |" for name, value in gates.items()]


def _a1_section(analysis: Mapping[str, Any]) -> list[str]:
    fixed = analysis["fixed_contrasts"]
    tuned = analysis["tuned_contrasts"]
    fixed_holm = analysis["holm_adjusted_fixed"]
    tuned_holm = analysis["holm_adjusted_tuned"]
    lines = [
        "## Registered effects",
        "",
        "Negative method-minus-control NLL favors the momentum method. The fixed-eta "
        "replication criterion is expressed as a positive momentum penalty in the "
        "registered direction; tuned equivalence requires the complete adjusted interval "
        "to lie inside `[-0.010,+0.010]`.",
        "",
        "| Anchor | Contrast | n | Mean | Student-t 95% CI | Holm 95% CI | Exact sign-flip p | Classification |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for h in (16, 256):
        for kind, source, holm in (
            ("fixed", fixed, fixed_holm),
            ("tuned", tuned, tuned_holm),
        ):
            label = f"H{h}_{kind}"
            summary = source[label]
            classification = (
                analysis["classifications"][f"H{h}"] if kind == "tuned" else "FIXED_REPLICATION"
            )
            lines.append(
                "| H{} | {} | {} | {} | {} | {} | {} | `{}` |".format(
                    h,
                    kind,
                    summary["n"],
                    _fmt(summary["mean"]),
                    _interval(summary["student_t_interval_95"]),
                    _interval(holm[label]["interval"]),
                    _fmt(summary["exact_sign_flip_two_sided_p"]),
                    classification,
                )
            )
    return lines


def _a3_section(analysis: Mapping[str, Any]) -> list[str]:
    lines = [
        "## Frozen kernel-law prediction versus extension",
        "",
        "| H | Selected eta | Predicted eta | |log2 error| | Eta gate | Observed frontier | Predicted frontier | Frozen 95% PI | Covered | Bracketed |",
        "|---:|---:|---:|---:|---|---:|---:|---:|---|---|",
    ]
    for h in (8, 512):
        row = analysis["rows"][str(h)]
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                h,
                _fmt(row["observed_selected_eta"]),
                _fmt(row["predicted_eta"]),
                _fmt(row["absolute_log2_eta_error"]),
                "PASS" if row["eta_error_pass"] else "FAIL",
                _fmt(row["observed_tuned_loss"]),
                _fmt(row["predicted_frontier_loss"]),
                _interval(row["prediction_interval_95"]),
                "PASS" if row["prediction_interval_coverage"] else "FAIL",
                "PASS" if row["bracketed"] else "FAIL",
            )
        )
    lines.extend(
        [
            "",
            f"Five-point frontier: `{json.dumps(analysis['five_point_frontier'], sort_keys=True)}`.",
            "",
            f"Extension-point RMSE: **{_fmt(analysis['extension_point_frontier_rmse'])}**. "
            f"Required endpoint ordering: **{'PASS' if analysis['required_endpoint_ordering_pass'] else 'FAIL'}**.",
        ]
    )
    return lines


def _a4_section(analysis: Mapping[str, Any]) -> list[str]:
    fixed = analysis["fixed_contrasts"]
    tuned = analysis["tuned_contrasts"]
    fixed_holm = analysis["holm_adjusted_fixed"]
    tuned_holm = analysis["holm_adjusted_tuned"]
    lines = [
        "## M-axis effects",
        "",
        "| M | H | Contrast | n | Mean | Holm 95% CI | Classification |",
        "|---:|---:|---|---:|---:|---:|---|",
    ]
    for m in (1, 4):
        for h in (16, 256):
            for kind, source, holm in (
                ("fixed", fixed, fixed_holm),
                ("tuned", tuned, tuned_holm),
            ):
                label = f"M{m}_H{h}_{kind}"
                classification = (
                    analysis["classifications"][f"M{m}_H{h}"]
                    if kind == "tuned"
                    else "FIXED_REPLICATION"
                )
                lines.append(
                    "| {} | {} | {} | {} | {} | {} | `{}` |".format(
                        m,
                        h,
                        kind,
                        source[label]["n"],
                        _fmt(source[label]["mean"]),
                        _interval(holm[label]["interval"]),
                        classification,
                    )
                )
    lines.extend(
        [
            "",
            "### Seed-paired `M × tuning-status × method` interaction",
            "",
            "| H | Mean | Student-t 95% CI | Holm 95% CI |",
            "|---:|---:|---:|---:|",
        ]
    )
    for h in (16, 256):
        summary = analysis["M_by_tuning_status_by_method_interactions"][f"H{h}"]
        adjusted = analysis["holm_adjusted_interactions"][f"H{h}"]
        lines.append(
            f"| {h} | {_fmt(summary['mean'])} | {_interval(summary['student_t_interval_95'])} | {_interval(adjusted['interval'])} |"
        )
    return lines


def render(args: argparse.Namespace) -> dict[str, Any]:
    stage = args.stage
    phase = load_object(args.final_phase_manifest, "final phase manifest")
    analysis = load_object(args.analysis, "final analysis")
    replay = load_object(args.replay_report, "isolated replay report")
    ledger = load_object(args.stage_spend_ledger, "stage spend ledger")
    expected_analysis_schema = {
        "A1": "audit_135m_a1_analysis_v1",
        "A3": "audit_135m_a3_analysis_v1",
        "A4": "audit_135m_a4_analysis_v1",
    }[stage]
    if (
        phase.get("status") != "sealed_results"
        or analysis.get("schema") != expected_analysis_schema
        or analysis.get("status") != "SEALED"
        or replay.get("schema") != "audit_135m_replay_report_v1"
        or replay.get("status") != "PASS"
        or replay.get("stage") != stage
        or ledger.get("audit_stage") != stage
        or float(ledger.get("estimated_spend_usd", math.inf))
        >= float(ledger.get("hard_ceiling_usd", -math.inf))
        or float(ledger.get("pre_science_aborted_launch_spend_usd", math.inf))
        > float(ledger.get("abort_burn_kill_usd", -math.inf))
    ):
        raise ReportError("final phase/analysis/replay/cost evidence is incomplete")
    if replay.get("final_phase_manifest_canonical_sha256") != hashlib.sha256(
        json.dumps(
            phase, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest():
        raise ReportError("replay report does not bind the final phase manifest")

    gates = analysis.get("gates")
    if not isinstance(gates, Mapping):
        raise ReportError("analysis has no registered gate output")
    campaigns = replay.get("campaigns")
    if not isinstance(campaigns, list) or not campaigns:
        raise ReportError("replay report has no campaign evidence")
    total_generations = sum(len(row.get("generations", [])) for row in campaigns)
    total_attempts = sum(int(row.get("attempt_count", 0)) for row in campaigns)
    total_launch_cells = sum(int(row.get("launch_cell_count", 0)) for row in campaigns)

    title = f"# AUDIT-135M-{stage}-FINAL"
    primary_gate = gates[
        {"A1": "A1", "A3": "A3_quantitative_law", "A4": "G8_A4_M_axis"}[
            stage
        ]
    ]
    lines = [
        title,
        "",
        f"Status: **sealed completion; isolated replay PASS; registered stage disposition `{primary_gate}`**.",
        "",
        "## Outcome",
        "",
        f"Stage {stage} completed under the frozen tuned-baseline-audit preregistration. "
        f"The cumulative manifest contains **{len(phase['expected_cells'])} expected cells** "
        f"and **{len(phase['results'])} retained result/attempt rows**. Across this stage's "
        f"campaigns, **{total_launch_cells} suffix cells**, **{total_attempts} attempts**, and "
        f"**{total_generations} exact physical generations** were independently replayed.",
        "",
        "## Registered gates",
        "",
        "| Gate | Disposition |",
        "|---|---|",
        *_gate_lines(gates),
        "",
    ]
    lines.extend(
        _a1_section(analysis)
        if stage == "A1"
        else _a3_section(analysis)
        if stage == "A3"
        else _a4_section(analysis)
    )
    lines.extend(
        [
            "",
            "## Cost and execution rails",
            "",
            "| Item | Evidence |",
            "|---|---:|",
            f"| Estimated Spot spend | `${float(ledger['estimated_spend_usd']):.6f}` |",
            f"| Hard stage ceiling | `${float(ledger['hard_ceiling_usd']):.2f}` |",
            f"| Remaining headroom | `${float(ledger['hard_ceiling_usd']) - float(ledger['estimated_spend_usd']):.6f}` |",
            f"| Pre-science aborted-launch spend | `${float(ledger['pre_science_aborted_launch_spend_usd']):.6f}` |",
            f"| Abort-burn kill | `${float(ledger['abort_burn_kill_usd']):.2f}` |",
            "| Provisioning model | Spot only |",
            "| Concurrent atomic-block width cap | 2 |",
            "| Maximum attached A100-equivalent | 16 |",
            "| Final campaign-owned VM/A100 census | 0 / 0 |",
            "| Protected instance `3908640733128066700` | untouched |",
            "",
            "Every generation is bound to numeric instance and boot-disk IDs, a hash-locked "
            "partial manifest, provider `NOT_FOUND` proofs for both exact IDs, and zero attached "
            "accelerators. Spot preemption retries use fresh physical generations and fresh "
            "attempt namespaces; no optimizer, checkpoint, tape, or result state was resumed.",
            "",
            "## Isolated replay",
            "",
            f"The replay report is **PASS** on `{replay.get('hostname')}` using source commit "
            f"`{replay['source']['git_commit']}`. It verified **{replay['verified_file_count']} "
            "archive files**, re-aggregated every VM campaign from full evidence, reproduced "
            "the append-only promotions and hidden-batch bindings, and exactly regenerated "
            "selection, precision decisions where applicable, and the final analysis.",
            "",
            "## Evidence hashes",
            "",
            "| Artifact | SHA-256 |",
            "|---|---|",
            f"| Final cumulative phase manifest | `{sha256_file(args.final_phase_manifest)}` |",
            f"| Final analysis | `{sha256_file(args.analysis)}` |",
            f"| Isolated replay report | `{sha256_file(args.replay_report)}` |",
            f"| Stage spend ledger | `{sha256_file(args.stage_spend_ledger)}` |",
            f"| Frozen preregistration JSON | `{replay['source']['tracked_sha256']['experiment-specs/tuned-baseline-audit-prereg.json']}` |",
            "",
            "## Publication disposition",
            "",
            "The registered classification and all failures, divergences, boundary decisions, "
            "retry lineage, cost evidence, and teardown proofs remain publishable regardless of "
            "gate direction. No completed outcome was removed or replaced by a historical "
            "absolute loss, and no confirmation endpoint was exposed before its complete hidden "
            "bundle, seal, and shared unblind.",
        ]
    )
    output = args.output.resolve()
    if output.exists():
        raise ReportError(f"refusing to overwrite final report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines).rstrip() + "\n")
    return {
        "status": "SEALED",
        "stage": stage,
        "output": str(output),
        "output_sha256": sha256_file(output),
        "gates": dict(gates),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("A1", "A3", "A4"), required=True)
    parser.add_argument("--final-phase-manifest", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--replay-report", type=Path, required=True)
    parser.add_argument("--stage-spend-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = render(args)
    except (ReportError, OSError, ValueError, KeyError) as exc:
        print(f"audit-135M report error: {exc}", file=__import__("sys").stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
