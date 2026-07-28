#!/usr/bin/env python3
"""Seal the registered v7 pilot center and 48/45-cell variant decision."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

try:
    import v7_common as common
except ModuleNotFoundError:  # package import in tests
    from scripts import v7_common as common


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_node_roots(values: list[str]) -> dict[str, Path]:
    roots = {}
    for value in values:
        node, separator, path = value.partition("=")
        if not separator or node not in common.NODES or not path or node in roots:
            raise ValueError(f"invalid --node-root {value!r}")
        roots[node] = Path(path)
    if set(roots) != set(common.NODES):
        raise ValueError("both v7 node roots are required")
    return roots


def load_completed_cell(cell: dict, root: Path, attempt: int) -> dict:
    attempt_root = root / cell["cell_id"] / f"attempt-{attempt}"
    evidence_path = attempt_root / "evidence.json"
    evidence = json.loads(evidence_path.read_text())
    expected_command_hash = (
        cell["command_hash"]
        if attempt == 1
        else cell["registered_retry_commands"][0]["command_hash"]
    )
    bindings = {
        "schema": "yeto_outer_mup_cell_evidence_v1",
        "cell_id": cell["cell_id"],
        "attempt_number": attempt,
        "seed": cell["seed"],
        "training_seed": cell["training_seed"],
        "command_hash": expected_command_hash,
    }
    for key, expected in bindings.items():
        if evidence.get(key) != expected:
            raise ValueError(f"{cell['cell_id']}: evidence {key} mismatch")
    if evidence.get("status") not in ("COMPLETED", "SCIENTIFIC_DIVERGENCE"):
        raise ValueError(
            f"{cell['cell_id']}: pilot evidence is not scientific-terminal"
        )
    results_path = attempt_root / "report" / "results.jsonl"
    result_hash = (
        evidence.get("observed_artifacts", {}).get("results", {}).get("sha256")
    )
    if result_hash != common.sha256_file(results_path):
        raise ValueError(f"{cell['cell_id']}: results hash mismatch")
    rows = [json.loads(line) for line in results_path.read_text().splitlines() if line]
    matching = [row for row in rows if row.get("arm") == "m2"]
    if len(matching) != 1:
        raise ValueError(f"{cell['cell_id']}: expected exactly one m2 result")
    loss = float(matching[0]["eval_loss"])
    attempt_end = json.loads((attempt_root / "attempt-end.json").read_text())
    wall_seconds = float(attempt_end["wall_seconds"])
    if not math.isfinite(wall_seconds) or wall_seconds <= 0.0:
        raise ValueError(f"{cell['cell_id']}: invalid wall time")
    return {
        "cell_id": cell["cell_id"],
        "eta": float(cell["eta"]),
        "eval_loss": loss,
        "wall_seconds": wall_seconds,
        "attempt": attempt,
        "evidence_path": str(evidence_path.resolve()),
        "evidence_sha256": common.sha256_file(evidence_path),
        "results_sha256": result_hash,
        "attempt_end_sha256": common.sha256_file(attempt_root / "attempt-end.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep-manifest", type=Path, required=True)
    parser.add_argument("--node-root", action="append", required=True)
    parser.add_argument(
        "--attempt",
        action="append",
        default=[],
        help="CELL_ID=1|2; omitted cells use attempt 1",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.prep_manifest.read_text())
    if manifest.get("schema") != "yeto_outer_mup_v7_27b_lora_prep_manifest_v1":
        raise SystemExit("input is not the v7 prep manifest")
    roots = parse_node_roots(args.node_root)
    attempts = {}
    for item in args.attempt:
        cell_id, separator, value = item.partition("=")
        if not separator or value not in ("1", "2") or cell_id in attempts:
            raise SystemExit(f"invalid --attempt binding {item!r}")
        attempts[cell_id] = int(value)

    smoke = [cell for cell in manifest["cells"] if cell["stage"] == "SMOKE"]
    pilot = [cell for cell in manifest["cells"] if cell["stage"] == "PILOT"]
    if len(smoke) != 1 or len(pilot) != 3:
        raise SystemExit("prep manifest stage counts changed")
    smoke_record = load_completed_cell(
        smoke[0],
        roots[smoke[0]["assignment"]["node"]],
        attempts.get(smoke[0]["cell_id"], 1),
    )
    pilot_records = [
        load_completed_cell(
            cell,
            roots[cell["assignment"]["node"]],
            attempts.get(cell["cell_id"], 1),
        )
        for cell in pilot
    ]
    pilot_records.sort(key=lambda record: record["eta"])
    selection = common.select_pilot_center(
        [record["eta"] for record in pilot_records],
        [record["eval_loss"] for record in pilot_records],
    )
    max_wall = max(record["wall_seconds"] for record in pilot_records)
    projection = common.select_grid_variant(max_wall)
    grids = common.derive_eta_grids(
        selection["selected_eta_star"], projection["variant"]
    )
    readout = {
        "schema": "yeto_outer_mup_v7_pilot_readout_v1",
        "status": "PASS",
        "created_at_utc": utc_now(),
        "prep_manifest_path": str(args.prep_manifest.resolve()),
        "prep_manifest_sha256": common.sha256_file(args.prep_manifest),
        "source_git_commit": manifest["source"]["git_commit"],
        "smoke": smoke_record,
        "pilot_cells": pilot_records,
        "pilot_selection": selection,
        "fleet_hour_projection": projection,
        "selected_grid": {
            "variant": projection["variant"],
            "eta_grids": grids,
        },
        "note_line": (
            f"PILOT DONE center={selection['selected_eta_star']:.12g} "
            f"projected_fleet_hours={projection['projected_fleet_hours']:.6f} "
            f"variant={projection['variant']}"
        ),
    }
    common.write_json_atomic(args.output, readout)
    print(readout["note_line"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
