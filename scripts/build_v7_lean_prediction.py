#!/usr/bin/env python3
"""Seal the reduced-v7 raw spot-check prediction after its three-cell pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tonight85_common as common
import v7_common
from tonight85_analysis import cell_loss


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--result-root", type=Path, default=Path("/root/yeto-results-tonight85")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = common.read_json(args.manifest)
    cells = sorted(
        [cell for cell in manifest["cells"] if cell["program"] == "v7_pilot"],
        key=lambda cell: cell["eta_index"],
    )
    if len(cells) != 3:
        raise SystemExit(f"expected three v7 pilot cells, found {len(cells)}")
    etas = []
    losses = []
    evidence = []
    for cell in cells:
        loss, record = cell_loss(cell, args.result_root)
        etas.append(float(cell["eta"]))
        losses.append(loss)
        evidence.append({"cell_id": cell["cell_id"], "eval_loss": loss, **record})
    selection = v7_common.select_pilot_center(etas, losses)
    mu0 = float(selection["selected_eta_star"])
    d5 = 1.7416157949788522
    raw_prediction = mu0 * 0.1 * d5
    offsets = [-0.5, 0.5]
    payload = {
        "schema": "yeto_outer_mup_v7_lean_sealed_prediction_v1",
        "status": "SEALED_PRE_RAW",
        "created_at_utc": common.utc_now(),
        "raw_loss_seen": False,
        "pilot": {
            "etas": etas,
            "losses": losses,
            "selection": selection,
            "evidence": evidence,
        },
        "D5": d5,
        "formula": "selected_pilot_mu0_eta*(1-0.9)*D5",
        "predicted_eta_star_raw": raw_prediction,
        "raw_offsets_log2": offsets,
        "raw_etas": [raw_prediction * 2.0**offset for offset in offsets],
        "seeds": [701, 709],
        "source_manifest": {
            "path": str(args.manifest),
            "sha256": common.sha256_file(args.manifest),
        },
    }
    payload["prediction_preimage_canonical_sha256"] = common.canonical_sha256(payload)
    common.write_json_atomic(args.output, payload)
    print(
        json.dumps(
            {"output": str(args.output), "sha256": common.sha256_file(args.output)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
