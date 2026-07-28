#!/usr/bin/env python3
"""Seal v11 ratio-transport predictions from the registered mu0 probes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import tonight85_common as common
from tonight85_analysis import cell_loss, fit_quadratic


def ratio_d(t: int) -> tuple[float, dict]:
    gamma, alpha, beta, epsilon = (
        1.3489008177233357,
        -0.9098513603667141,
        -0.10723867757601385,
        0.16020840966569636,
    )

    def surface(T: int, S: int) -> float:
        u = (T - 5.0) / 5.0
        v = math.log2(S / 5120.0)
        return 2.0 ** (gamma + alpha * u + beta * v + epsilon * u * u)

    d5 = surface(5, 2560)
    d10 = surface(10, 5120)
    exponent = math.log2((d10 - 1.0) / (d5 - 1.0))
    value = 1.0 + (d5 - 1.0) * (t / 5.0) ** exponent
    return value, {"D5": d5, "D10": d10, "exponent": exponent}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--result-root", type=Path, default=Path("/root/yeto-results-tonight85")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = common.read_json(args.manifest)
    cells = [cell for cell in manifest["cells"] if cell["program"] == "v11_anchor"]
    if len(cells) != 6:
        raise SystemExit(f"expected six v11 anchor cells, found {len(cells)}")
    coordinates = {}
    for coordinate_id in sorted({cell["coordinate_id"] for cell in cells}):
        selected = sorted(
            [cell for cell in cells if cell["coordinate_id"] == coordinate_id],
            key=lambda cell: cell["eta_index"],
        )
        etas = []
        losses = []
        evidence = []
        for cell in selected:
            loss, record = cell_loss(cell, args.result_root)
            etas.append(float(cell["eta"]))
            losses.append(loss)
            evidence.append({"cell_id": cell["cell_id"], "eval_loss": loss, **record})
        fit = fit_quadratic(etas, losses)
        if not fit["accepted"]:
            raise SystemExit(
                f"{coordinate_id}: registered mu0 probe is unaccepted; truth is blocked"
            )
        t = int(selected[0]["t"])
        d, donor = ratio_d(t)
        prediction = float(fit["eta_star"]) * 0.1 * d
        offsets = [-1.0, -0.5, 0.0, 0.5, 1.0]
        coordinates[coordinate_id] = {
            "T": t,
            "S": int(selected[0]["s"]),
            "mu0_probe_etas": etas,
            "mu0_probe_losses": losses,
            "mu0_fit": fit,
            "ratio_D": d,
            "ratio_donors": donor,
            "formula": "eta_star_raw=eta_star_mu0*(1-0.9)*D(T)",
            "predicted_eta_star_raw": prediction,
            "ground_truth_offsets_log2": offsets,
            "ground_truth_etas": [prediction * 2.0**offset for offset in offsets],
            "anchor_evidence": evidence,
        }
    payload = {
        "schema": "yeto_outer_mup_v11_sealed_predictions_v1",
        "status": "SEALED_PRE_TRUTH",
        "created_at_utc": common.utc_now(),
        "verification_loss_seen": False,
        "source_anchor_manifest": {
            "path": str(args.manifest),
            "sha256": common.sha256_file(args.manifest),
            "source_git_commit": manifest["source"]["git_commit"],
        },
        "registration_contract": {
            "path": "experiment-specs/outer-mup-v11-ratio-transport-prereg.json",
            "sha256": common.sha256_file(
                common.REPO
                / "experiment-specs/outer-mup-v11-ratio-transport-prereg.json"
            ),
        },
        "coordinates": coordinates,
    }
    payload["prediction_preimage_canonical_sha256"] = common.canonical_sha256(payload)
    common.write_json_atomic(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": common.sha256_file(args.output),
                "preimage": payload["prediction_preimage_canonical_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
