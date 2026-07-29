#!/usr/bin/env python3
"""Resolve frozen G6/G8 T=20 selected-rung pairs against an archive listing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


FIELDS = (
    "campaign",
    "scale",
    "T",
    "H",
    "S",
    "convention",
    "momentum_mu",
    "momentum_rung",
    "control_rung",
    "training_seed",
    "momentum_eta",
    "control_eta",
    "momentum_cell_id",
    "control_cell_id",
    "momentum_node",
    "control_node",
    "momentum_archive_member",
    "control_archive_member",
    "momentum_member_listed",
    "control_member_listed",
    "status",
    "reason",
)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def selected_curves(payload: dict, campaign: str) -> list[dict]:
    curves = []
    for curve in payload["curve_fits"]:
        t = int(curve.get("t", curve.get("T")))
        if t != 20:
            continue
        losses = [float(value) for value in curve["seed_mean_losses"]]
        rung = min(range(len(losses)), key=losses.__getitem__)
        expected = 1 if (
            campaign == "G6"
            and curve["arm"] == "raw"
            and int(curve["h"]) == 256
            and int(curve["s"]) == 5120
        ) else 2
        if rung != expected:
            raise ValueError(
                f"{campaign} T=20 selected rung differs from frozen e{expected}: {curve}"
            )
        item = dict(curve)
        item["selected_rung"] = rung
        item["selected_eta"] = float(curve["etas"][rung])
        curves.append(item)
    return curves


def dimensions(record: dict) -> tuple[int, int, int]:
    return (
        int(record.get("t", record.get("T"))),
        int(record.get("h", record.get("H"))),
        int(record.get("s", record.get("S"))),
    )


def curve_dimensions(curve: dict) -> tuple[int, int, int]:
    return (
        int(curve.get("t", curve.get("T"))),
        int(curve.get("h", curve.get("H"))),
        int(curve.get("s", curve.get("S"))),
    )


def exact_eta(left: float, right: float) -> bool:
    return left == right


def member(campaign: str, cell_id: str, attempt: int) -> str:
    results = "yeto-results-v6" if campaign == "G6" else "yeto-results-v8"
    return (
        f"data/{results}/{cell_id}/attempt-{attempt}/work/m4/state.ckpt"
    )


def build_rows(payload: dict, campaign: str, listing: set[str]) -> list[dict]:
    curves = selected_curves(payload, campaign)
    records = payload["cell_records"]
    rows = []
    momentum_curves = [curve for curve in curves if curve["arm"] != "mu0"]
    for curve in momentum_curves:
        dims = curve_dimensions(curve)
        mu = 0.9 if campaign == "G6" else float(curve["mu"])
        controls = [
            candidate
            for candidate in curves
            if candidate["arm"] == "mu0" and curve_dimensions(candidate) == dims
        ]
        if len(controls) != 1:
            raise ValueError(f"{campaign} expected one control curve for {curve}")
        control_curve = controls[0]
        momentum_records = [
            record
            for record in records
            if record["arm"] == curve["arm"]
            and dimensions(record) == dims
            and exact_eta(float(record["eta"]), curve["selected_eta"])
            and (campaign == "G6" or float(record["mu"]) == mu)
        ]
        control_records = [
            record
            for record in records
            if record["arm"] == "mu0"
            and dimensions(record) == dims
            and exact_eta(float(record["eta"]), control_curve["selected_eta"])
        ]
        by_seed = {int(record["seed"]): record for record in control_records}
        if len(by_seed) != len(control_records):
            raise ValueError(f"{campaign} duplicate control seed for {curve}")
        for momentum_record in sorted(momentum_records, key=lambda item: int(item["seed"])):
            seed = int(momentum_record["seed"])
            if seed not in by_seed:
                raise ValueError(f"{campaign} missing control seed {seed} for {curve}")
            control_record = by_seed[seed]
            momentum_member = member(
                campaign, momentum_record["cell_id"], int(momentum_record["attempt"])
            )
            control_member = member(
                campaign, control_record["cell_id"], int(control_record["attempt"])
            )
            momentum_listed = momentum_member in listing
            control_listed = control_member in listing
            nodes = {
                "momentum": str(momentum_record["node"]),
                "control": str(control_record["node"]),
            }
            lost_arms = [name for name, node in nodes.items() if node == "h200-n2"]
            if lost_arms:
                status = "LOST_N2"
                reason = "n2-resident weights not present: " + ",".join(lost_arms)
            elif not momentum_listed or not control_listed:
                status = "MISSING_ARCHIVE_MEMBER"
                absent = []
                if not momentum_listed:
                    absent.append("momentum")
                if not control_listed:
                    absent.append("control")
                reason = "n1 metadata but archive member absent: " + ",".join(absent)
            else:
                status = "PROBEABLE"
                reason = "both same-seed selected-rung terminal weights retained in n1 archive"
            rows.append(
                {
                    "campaign": campaign,
                    "scale": "135M",
                    "T": dims[0],
                    "H": dims[1],
                    "S": dims[2],
                    "convention": (
                        "nesterov_corrected"
                        if curve["arm"] == "corrected"
                        else "nesterov_raw"
                    ),
                    "momentum_mu": mu,
                    "momentum_rung": f"e{curve['selected_rung']}",
                    "control_rung": f"e{control_curve['selected_rung']}",
                    "training_seed": seed,
                    "momentum_eta": curve["selected_eta"],
                    "control_eta": control_curve["selected_eta"],
                    "momentum_cell_id": momentum_record["cell_id"],
                    "control_cell_id": control_record["cell_id"],
                    "momentum_node": nodes["momentum"],
                    "control_node": nodes["control"],
                    "momentum_archive_member": momentum_member,
                    "control_archive_member": control_member,
                    "momentum_member_listed": str(momentum_listed).lower(),
                    "control_member_listed": str(control_listed).lower(),
                    "status": status,
                    "reason": reason,
                }
            )
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g6-readout", required=True, type=Path)
    parser.add_argument("--g8-readout", required=True, type=Path)
    parser.add_argument("--archive-list", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--members-output", required=True, type=Path)
    args = parser.parse_args()

    listing_lines = args.archive_list.read_text(encoding="utf-8").splitlines()
    listing = set(listing_lines)
    if len(listing) != len(listing_lines):
        raise ValueError("archive listing contains duplicate member paths")
    rows = build_rows(load_json(args.g6_readout), "G6", listing)
    rows.extend(build_rows(load_json(args.g8_readout), "G8", listing))
    rows.sort(
        key=lambda row: (
            row["campaign"],
            int(row["H"]),
            row["convention"],
            float(row["momentum_mu"]),
            int(row["training_seed"]),
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    members = sorted(
        {
            row[key]
            for row in rows
            if row["status"] == "PROBEABLE"
            for key in ("momentum_archive_member", "control_archive_member")
        }
    )
    args.members_output.write_text("".join(f"{path}\n" for path in members), encoding="utf-8")

    counts = {status: sum(row["status"] == status for row in rows) for status in sorted({row["status"] for row in rows})}
    print(
        json.dumps(
            {
                "archive_listing_lines": len(listing_lines),
                "archive_listing_sha256": sha256_file(args.archive_list),
                "pair_rows": len(rows),
                "status_counts": counts,
                "unique_extract_members": len(members),
                "output": str(args.output),
                "members_output": str(args.members_output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
