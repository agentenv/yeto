#!/usr/bin/env python3
"""Frozen descriptive readout for the reduced v7 27B-LoRA lane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tonight85_common as common
from tonight85_analysis import cell_loss, write_json_atomic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument(
        "--result-root", type=Path, default=Path("/root/yeto-results-tonight85")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--note", type=Path, default=Path("/private/tmp/h200-tonight85-note.md")
    )
    args = parser.parse_args()
    manifest = common.read_json(args.manifest)
    prediction = common.read_json(args.prediction)
    cells = [cell for cell in manifest["cells"] if cell["program"] == "v7_raw"]
    records = []
    for cell in cells:
        try:
            loss, evidence = cell_loss(cell, args.result_root)
        except Exception as exc:
            records.append(
                {
                    "cell_id": cell["cell_id"],
                    "eta": cell["eta"],
                    "seed": cell["seed"],
                    "status": "MISSING_OR_INVALID",
                    "reason": str(exc),
                }
            )
            continue
        records.append(
            {
                "cell_id": cell["cell_id"],
                "eta": cell["eta"],
                "seed": cell["seed"],
                "eval_loss": loss,
                "status": "COMPLETED",
                **evidence,
            }
        )
    completed = [record for record in records if record["status"] == "COMPLETED"]
    means = {}
    for eta in prediction["raw_etas"]:
        values = [record["eval_loss"] for record in completed if record["eta"] == eta]
        means[repr(eta)] = sum(values) / len(values) if values else None
    readout = {
        "schema": "yeto_outer_mup_v7_lean_descriptive_readout_v1",
        "status": "COMPLETE_DESCRIPTIVE"
        if len(completed) == 4
        else "PARTIAL_DESCRIPTIVE",
        "confirmatory_gate": None,
        "scope": "descriptive only under the pre-outcome v7 lean amendment",
        "predicted_eta_star_raw": prediction["predicted_eta_star_raw"],
        "raw_etas": prediction["raw_etas"],
        "two_seed_mean_losses": means,
        "completed_cells": len(completed),
        "registered_cells": 4,
        "cell_records": records,
    }
    write_json_atomic(args.output, readout)
    with args.note.open("a", encoding="utf-8") as destination:
        destination.write(
            f"V7 LEAN DESCRIPTIVE: {readout['status']} ({len(completed)}/4 raw cells)\n"
        )
    print(
        json.dumps(
            {"status": readout["status"], "output": str(args.output)}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
