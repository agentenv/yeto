#!/usr/bin/env python3
"""Aggregate EXPLORATORY Lane B cells and fit loss quadratics in log2(eta)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


LABEL = "EXPLORATORY"
RESULT_ROOT = Path("/root/yeto-results-explore")
ETA0_S2560 = 0.0443

SUMMARY_FIELDS = (
    "label",
    "cell_id",
    "node",
    "gpu",
    "status",
    "grid",
    "h",
    "s",
    "t",
    "mu",
    "d_pred",
    "eta_center",
    "eta_index",
    "eta",
    "seed",
    "training_seed",
    "token_budget",
    "expected_outer_steps",
    "eval_loss",
    "eval_rows",
    "eval_tokens",
    "cell_wall_s",
    "attempt_wall_s",
    "rho_rows",
    "command_hash",
    "result_path",
    "config_path",
    "failure",
)

FIT_FIELDS = (
    "label",
    "curve",
    "grid",
    "h",
    "s",
    "t",
    "mu",
    "completed_cells",
    "expected_cells",
    "fit_status",
    "quadratic_a",
    "quadratic_b",
    "quadratic_c",
    "vertex_log2_eta",
    "eta_star",
    "d_obs",
    "d_pred",
    "d_obs_over_pred",
    "denominator_eta0",
    "loss_eta0",
    "loss_eta1",
    "loss_eta2",
    "loss_eta3",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_csv_atomic(path: Path, fields: Iterable[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def summarize_node(manifest: dict[str, object], node: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cell in manifest["cells"]:
        assignment = cell["assignment"]
        if assignment["node"] != node:
            continue
        cell_id = str(cell["cell_id"])
        attempt = RESULT_ROOT / cell_id / "attempt-1"
        bank_path = attempt / "bank-result.json"
        end_path = attempt / "attempt-end.json"
        status = "PENDING"
        failure = ""
        result: dict[str, object] = {}
        attempt_wall = ""
        rho_rows: int | str = ""
        if bank_path.is_file():
            try:
                bank = json.loads(bank_path.read_text())
                status = str(bank.get("status", "INVALID"))
                result = bank.get("result") or {}
                failure = "; ".join(str(item) for item in bank.get("failures", []))
                rho_rows = int(cell["expected_telemetry_rows"]) if status == "COMPLETED" else ""
            except Exception as exc:
                status = "INVALID"
                failure = f"cannot read bank record: {exc}"
        elif attempt.is_dir():
            status = "RUNNING" if not end_path.is_file() else "UNBANKED"
        if end_path.is_file():
            try:
                attempt_wall = json.loads(end_path.read_text()).get("wall_seconds", "")
            except Exception:
                pass
        rows.append(
            {
                "label": LABEL,
                "cell_id": cell_id,
                "node": node,
                "gpu": assignment["gpu"],
                "status": status,
                "grid": cell["grid"],
                "h": cell["h"],
                "s": cell["s"],
                "t": cell["t"],
                "mu": cell["mu"],
                "d_pred": cell["d_pred"],
                "eta_center": format(float(cell["eta_center"]), ".17g"),
                "eta_index": cell["eta_index"],
                "eta": format(float(cell["eta"]), ".17g"),
                "seed": cell["seed"],
                "training_seed": cell["training_seed"],
                "token_budget": cell["token_budget"],
                "expected_outer_steps": cell["expected_outer_steps"],
                "eval_loss": result.get("eval_loss", ""),
                "eval_rows": result.get("eval_rows", ""),
                "eval_tokens": result.get("eval_tokens", ""),
                "cell_wall_s": result.get("wall_s", ""),
                "attempt_wall_s": attempt_wall,
                "rho_rows": rho_rows,
                "command_hash": cell["command_hash"],
                "result_path": str(bank_path),
                "config_path": str(attempt / "attempt-start.json"),
                "failure": failure,
            }
        )
    return rows


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    for row in rows:
        if row.get("label") != LABEL:
            raise ValueError(f"summary row lacks EXPLORATORY label: {path}")
    return rows


def solve3(matrix: list[list[float]], vector: list[float]) -> list[float]:
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-18:
            raise ValueError("singular quadratic normal equation")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(3):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [augmented[index][3] for index in range(3)]


def fit_quadratic(etas: list[float], losses: list[float]) -> dict[str, object]:
    if len(etas) != 4 or len(losses) != 4:
        return {"status": "INCOMPLETE", "eta_star": None}
    xs = [math.log2(eta) for eta in etas]
    sums = [sum(x**power for x in xs) for power in range(5)]
    matrix = [
        [sums[4], sums[3], sums[2]],
        [sums[3], sums[2], sums[1]],
        [sums[2], sums[1], sums[0]],
    ]
    vector = [
        sum(y * x * x for x, y in zip(xs, losses)),
        sum(y * x for x, y in zip(xs, losses)),
        sum(losses),
    ]
    a, b, c = solve3(matrix, vector)
    vertex = -b / (2.0 * a) if a else math.nan
    convex = a > 0 and math.isfinite(vertex)
    interior = convex and min(xs) < vertex < max(xs)
    if interior:
        status = "INTERIOR"
    elif convex and vertex <= min(xs):
        status = "EXTRAPOLATED_LOW"
    elif convex and vertex >= max(xs):
        status = "EXTRAPOLATED_HIGH"
    else:
        status = "NONCONVEX"
    return {
        "status": status,
        "a": a,
        "b": b,
        "c": c,
        "vertex_log2_eta": vertex if math.isfinite(vertex) else None,
        # Lane B is an explicitly exploratory shape check. Retain a numeric
        # unconstrained convex vertex even when it lies outside the ladder, but
        # make that extrapolation impossible to miss in fit_status and the note.
        "eta_star": 2.0**vertex if convex else None,
    }


def curve_name(h: int, s: int, mu: float) -> str:
    mu_text = format(mu, ".12g")
    return f"H{h}_S{s}_mu{mu_text}"


def fit_curves(
    manifest: dict[str, object], summary_rows: list[dict[str, str]]
) -> list[dict[str, object]]:
    summary_by_id = {row["cell_id"]: row for row in summary_rows}
    curve_cells: dict[tuple[int, int, float], list[dict[str, object]]] = defaultdict(list)
    curve_order: dict[tuple[int, int, float], int] = {}
    for cell in manifest["cells"]:
        key = (int(cell["h"]), int(cell["s"]), float(cell["mu"]))
        curve_cells[key].append(cell)
        curve_order[key] = int(cell["curve_order"])

    fits: list[dict[str, object]] = []
    fit_by_key: dict[tuple[int, int, float], dict[str, object]] = {}
    for key in sorted(curve_cells, key=lambda item: curve_order[item]):
        h, s, mu = key
        cells = curve_cells[key]
        completed = [
            cell
            for cell in cells
            if summary_by_id.get(str(cell["cell_id"]), {}).get("status") == "COMPLETED"
        ]
        means: list[float] = []
        etas: list[float] = []
        complete_grid = True
        for eta_index in range(4):
            eta_cells = sorted(
                (cell for cell in cells if int(cell["eta_index"]) == eta_index),
                key=lambda cell: int(cell["seed"]),
            )
            values: list[float] = []
            for cell in eta_cells:
                row = summary_by_id.get(str(cell["cell_id"]), {})
                if row.get("status") != "COMPLETED" or not row.get("eval_loss"):
                    complete_grid = False
                    continue
                values.append(float(row["eval_loss"]))
            if len(values) != 2:
                complete_grid = False
            if len(values) == 2:
                means.append(sum(values) / 2.0)
                etas.append(float(eta_cells[0]["eta"]))
        fit = fit_quadratic(etas, means) if complete_grid else {
            "status": "INCOMPLETE",
            "a": None,
            "b": None,
            "c": None,
            "vertex_log2_eta": None,
            "eta_star": None,
        }
        sample = cells[0]
        record: dict[str, object] = {
            "label": LABEL,
            "curve": curve_name(h, s, mu),
            "grid": sample["grid"],
            "h": h,
            "s": s,
            "t": sample["t"],
            "mu": mu,
            "completed_cells": len(completed),
            "expected_cells": 8,
            "fit_status": fit["status"],
            "quadratic_a": fit.get("a"),
            "quadratic_b": fit.get("b"),
            "quadratic_c": fit.get("c"),
            "vertex_log2_eta": fit.get("vertex_log2_eta"),
            "eta_star": fit.get("eta_star"),
            "d_obs": None,
            "d_pred": sample["d_pred"],
            "d_obs_over_pred": None,
            "denominator_eta0": None,
            "loss_eta0": means[0] if len(means) == 4 else None,
            "loss_eta1": means[1] if len(means) == 4 else None,
            "loss_eta2": means[2] if len(means) == 4 else None,
            "loss_eta3": means[3] if len(means) == 4 else None,
        }
        fits.append(record)
        fit_by_key[key] = record

    for record in fits:
        mu = float(record["mu"])
        s = int(record["s"])
        eta_star = record["eta_star"]
        denominator: float | None = None
        if mu == 0.0:
            if eta_star is not None:
                record["d_obs"] = 1.0
                record["denominator_eta0"] = eta_star
        elif record["grid"] == "grid1_s_variation":
            baseline = fit_by_key[(int(record["h"]), s, 0.0)]
            denominator = baseline.get("eta_star")
            record["denominator_eta0"] = denominator
            if eta_star is not None and denominator is not None:
                record["d_obs"] = eta_star / (denominator * (1.0 - mu))
        else:
            denominator = ETA0_S2560
            record["denominator_eta0"] = denominator
            if eta_star is not None:
                record["d_obs"] = eta_star / (denominator * (1.0 - mu))
        if record["d_obs"] is not None:
            record["d_obs_over_pred"] = float(record["d_obs"]) / float(record["d_pred"])
    return fits


def fmt(value: object, digits: int = 4) -> str:
    if value is None or value == "":
        return "NA"
    return f"{float(value):.{digits}g}"


def write_note(
    path: Path,
    summary_rows: list[dict[str, str]],
    fits: list[dict[str, object]],
    started_at: str | None,
) -> None:
    counts: dict[str, int] = defaultdict(int)
    for row in summary_rows:
        counts[row["status"]] += 1
    completed = counts.get("COMPLETED", 0)
    invalid = sum(
        count for status, count in counts.items() if status not in ("COMPLETED", "PENDING", "RUNNING")
    )
    lines = [
        "# EXPLORATORY — H200 Lane B finite-T pilot",
        "",
        "> EXPLORATORY shape-check only. Namespace `e1x-*`; no E1/E1v2 evidence is consumed or modified.",
        "",
        f"Updated: {utc_now()}",
        f"Grid start: {started_at or 'recorded by controller'}",
        f"Progress: {completed}/72 completed; {counts.get('RUNNING', 0)} running; "
        f"{counts.get('PENDING', 0)} pending; {invalid} invalid/unbanked.",
        "",
        "## Running quadratic fits",
        "",
        "| Curve | Cells | Fit | eta* | D_obs | D_pred | D_obs/D_pred |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for record in fits:
        lines.append(
            f"| {record['curve']} | {record['completed_cells']}/8 | {record['fit_status']} | "
            f"{fmt(record['eta_star'], 6)} | {fmt(record['d_obs'], 5)} | "
            f"{fmt(record['d_pred'], 5)} | {fmt(record['d_obs_over_pred'], 5)} |"
        )
    lines.extend(["", "## Milestones", "", "GRID STARTED"])
    for record in fits:
        if int(record["completed_cells"]) != 8:
            continue
        lines.append(
            f"CURVE H{record['h']}_S{record['s']}_mu{format(float(record['mu']), '.12g')}: "
            f"D_obs={fmt(record['d_obs'], 6)} pred={fmt(record['d_pred'], 6)}"
        )
    if completed == 72 and invalid == 0:
        lines.append("LANE B DONE")
    lines.extend(
        [
            "",
            "## Fit contract",
            "",
            "Each curve is fit to the two-seed mean loss at each of the four registered eta values: "
            "`loss = a·log2(eta)^2 + b·log2(eta) + c`. Convex vertices outside the ladder retain "
            "numeric eta* and D_obs for this exploratory shape-check, but are explicitly flagged "
            "`EXTRAPOLATED_LOW/HIGH` and are not bracketed optima. Grid 1 uses the fresh "
            "within-S mu=0 eta*; Grid 2 uses the already measured S=2560 mu=0 reference eta*=0.0443.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text("\n".join(lines))
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--node", choices=("h200-n1", "h200-n2"))
    parser.add_argument("--combine", type=Path, nargs="*")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--fits", type=Path)
    parser.add_argument("--note", type=Path)
    parser.add_argument("--started-at")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("label") != LABEL:
        raise SystemExit("manifest is not labeled EXPLORATORY")
    if args.node:
        rows: list[dict[str, object]] = summarize_node(manifest, args.node)
    elif args.combine:
        combined: dict[str, dict[str, str]] = {}
        for path in args.combine:
            for row in read_summary(path):
                if row["cell_id"] in combined:
                    raise SystemExit(f"duplicate combined cell: {row['cell_id']}")
                combined[row["cell_id"]] = row
        rows = [combined[cell["cell_id"]] for cell in manifest["cells"] if cell["cell_id"] in combined]
    else:
        parser.error("specify either --node or --combine")
    write_csv_atomic(args.summary, SUMMARY_FIELDS, rows)
    fits: list[dict[str, object]] = []
    if len(rows) == 72:
        fits = fit_curves(manifest, rows)
        if args.fits:
            write_csv_atomic(args.fits, FIT_FIELDS, fits)
        if args.note:
            write_note(args.note, rows, fits, args.started_at)
    status_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        status_counts[str(row["status"])] += 1
    print(
        json.dumps(
            {
                "label": LABEL,
                "rows": len(rows),
                "status_counts": status_counts,
                "fits": len(fits),
                "summary": str(args.summary),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
