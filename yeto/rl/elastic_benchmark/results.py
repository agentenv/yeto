"""Layered results and the minimal study report.

Each attempt result has four independent layers: execution (did it run),
correctness (ledger reconciles), quality (predeclared held-out rules) and
benefit (predeclared system gain). A study summary is ``incomplete`` whenever a
required matrix item lacks a completed, verified attempt; survivors are never
averaged as if the matrix were whole.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from yeto.benchmark_resume import write_json_atomic
from yeto.rl.elastic_benchmark import evidence
from yeto.rl.elastic_benchmark.plan import StudyPlan

CORRECTNESS = ("passed", "failed", "not_checked")
QUALITY = ("passed", "failed", "insufficient", "censored", "not_checked")
BENEFIT = ("demonstrated", "not_demonstrated", "negative", "not_checked")
_LAYER_VALUES = {"correctness": CORRECTNESS, "quality": QUALITY, "benefit": BENEFIT}


def make_result(
    *,
    execution: str,
    correctness: str = "not_checked",
    quality: str = "not_checked",
    benefit: str = "not_checked",
    reason: str | None = None,
    measurements: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if execution not in evidence.EXECUTION_STATES:
        raise ValueError(f"execution must be one of {evidence.EXECUTION_STATES}")
    layers = {"correctness": correctness, "quality": quality, "benefit": benefit}
    for layer, value in layers.items():
        if value not in _LAYER_VALUES[layer]:
            raise ValueError(f"{layer} must be one of {_LAYER_VALUES[layer]}")
    if execution != "completed" and any(v not in ("not_checked",) for v in layers.values()):
        raise ValueError("only completed attempts may carry correctness/quality/benefit verdicts")
    return {"execution": execution, **layers, "reason": reason, "measurements": measurements or {}}


def summarize(plan: StudyPlan, study_dir: Path) -> dict[str, Any]:
    keys = [item.key for item in plan.items]
    collected = evidence.collect_results(study_dir, keys)
    rows, missing, failed, tampered = [], [], [], []
    for item in plan.items:
        attempts = collected.get(item.key, [])
        verified = _verified_completion(study_dir, plan.study_hash, item.key, attempts, tampered)
        rows.append(_row(item, attempts, verified))
        if item.status != "supported":
            continue
        if verified is None:
            missing.append(item.key.as_dict())
        if any(a.get("execution") == "failed" for a in attempts):
            failed.append(item.key.as_dict())
    status = "complete" if not missing and not tampered else "incomplete"
    return {
        "study_hash": plan.study_hash,
        "status": status,
        "counts": plan.counts(),
        "missing": missing,
        "failed_attempts": failed,
        "tampered": tampered,
        "benefit": _benefit_verdict(rows, status),
        "rows": rows,
    }


def _verified_completion(study_dir, study_hash, key, attempts, tampered) -> dict[str, Any] | None:
    for attempt in attempts:
        if attempt.get("execution") != "completed":
            continue
        attempt_dir = study_dir / key.relative_dir(attempt["attempt"])
        try:
            evidence.verify_evidence_index(attempt_dir, study_hash=study_hash, key=key)
        except evidence.EvidenceError as exc:
            tampered.append({**key.as_dict(), "attempt": attempt["attempt"], "error": str(exc)})
            continue
        return attempt
    return None


def _row(item, attempts, verified) -> dict[str, Any]:
    latest = attempts[-1] if attempts else None
    return {
        **item.key.as_dict(),
        "kind": item.kind,
        "plan_status": item.status,
        "plan_reason": item.reason,
        "attempts": len(attempts),
        "execution": verified["execution"] if verified else (latest["execution"] if latest else "pending"),
        "correctness": verified["correctness"] if verified else "not_checked",
        "quality": verified["quality"] if verified else "not_checked",
        "benefit": verified["benefit"] if verified else "not_checked",
        "reason": (verified or latest or {}).get("reason"),
    }


def _benefit_verdict(rows: list[dict[str, Any]], status: str) -> str:
    """Benefit is only claimed when the matrix is whole and every gate passes."""
    if status != "complete":
        return "not_demonstrated"
    dynamic = [r for r in rows if r["kind"] not in ("legacy-fixed", "target-fixed-default", "target-fixed-sweep")]
    runnable = [r for r in dynamic if r["plan_status"] == "supported"]
    if not runnable:
        return "not_checked"
    if any(r["correctness"] == "failed" for r in runnable):
        return "negative"
    if any(r["benefit"] == "negative" for r in runnable):
        return "negative"
    if all(r["benefit"] == "demonstrated" and r["quality"] == "passed" and r["correctness"] == "passed" for r in runnable):
        return "demonstrated"
    return "not_demonstrated"


def write_summary(summary: dict[str, Any], study_dir: Path) -> Path:
    path = study_dir / "summary.json"
    write_json_atomic(path, summary)
    return path


def render_report(summary: dict[str, Any]) -> str:
    lines = [f"# Elastic RL benchmark report", "", f"study: `{summary['study_hash']}`", f"status: **{summary['status']}**", f"benefit: **{summary['benefit']}**", ""]
    if summary["missing"]:
        lines.append(f"Missing required items: {len(summary['missing'])}. The matrix is incomplete; no survivor mean is reported.")
    if summary["tampered"]:
        lines.append(f"Evidence verification failed for {len(summary['tampered'])} attempt(s); those results are void.")
    if summary["failed_attempts"]:
        lines.append(f"Failed attempts retained: {len(summary['failed_attempts'])}.")
    lines.extend(["", "| arm | config | scenario | seed | plan | execution | correctness | quality | benefit | reason |", "|---|---|---|---|---|---|---|---|---|---|"])
    for row in summary["rows"]:
        lines.append(
            f"| {row['arm']} | {row['config'] or '-'} | {row['scenario']} | {row['seed']} | {row['plan_status']} | "
            f"{row['execution']} | {row['correctness']} | {row['quality']} | {row['benefit']} | {row['reason'] or row['plan_reason'] or ''} |"
        )
    return "\n".join(lines) + "\n"


def write_report(summary: dict[str, Any], study_dir: Path) -> Path:
    path = study_dir / "report.md"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(render_report(summary), encoding="utf-8")
    temporary.replace(path)
    return path
