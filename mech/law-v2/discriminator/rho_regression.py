#!/usr/bin/env python3
"""Regress C1-normalized pair residuals on tape-measured lag-1 rho."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


INPUT_FIELDS = (
    "momentum_point_id",
    "campaign",
    "scale",
    "T",
    "H",
    "S",
    "convention",
    "mu",
    "observed_to_law_ratio",
    "c1_prediction",
    "phi",
    "ln_phi",
    "selected_rung",
    "selected_eta",
    "telemetry_run_count",
    "telemetry_lag1_pair_count",
    "rho_energy_weighted",
    "fisher_z_rho",
    "stratum",
    "cell_ids",
    "telemetry_sha256s",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def coefficient_c(convention: str, t: int, mu: float) -> float:
    if mu == 0.0:
        return float(t)
    if convention == "nesterov_raw":
        return t / (1.0 - mu) - mu**2 * (1.0 - mu**t) / (1.0 - mu) ** 2
    if convention == "heavy_ball":
        return t / (1.0 - mu) - mu * (1.0 - mu**t) / (1.0 - mu) ** 2
    if convention == "nesterov_corrected":
        return t / (1.0 - mu)
    raise ValueError(f"unknown convention: {convention}")


def multiplier(convention: str, t: int, mu: float) -> float:
    if convention == "nesterov_raw":
        return (1.0 - mu ** (t + 1)) / (1.0 - mu)
    if convention == "heavy_ball":
        return (1.0 - mu**t) / (1.0 - mu)
    if convention == "nesterov_corrected":
        return 1.0
    raise ValueError(f"unknown convention: {convention}")


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def dimensions(record: dict) -> tuple[int, int, int]:
    return int(record["T"]), int(record["H"]), int(record["S"])


def curve_dimensions(curve: dict) -> tuple[int, int, int]:
    return int(curve["T"]), int(curve["H"]), int(curve["S"])


def selected_curve(payload: dict, row: dict) -> tuple[dict, int, float]:
    arm = {
        "nesterov_corrected": "corrected",
        "nesterov_raw": "raw",
    }.get(row["convention"])
    if arm is None:
        raise LookupError("G8 has no requested convention")
    target = (int(row["T"]), int(row["H"]), int(row["S"]))
    mu = float(row["mu"])
    curves = [
        curve
        for curve in payload["curve_fits"]
        if curve["arm"] == arm
        and curve_dimensions(curve) == target
        and float(curve["mu"]) == mu
    ]
    if len(curves) != 1:
        raise LookupError(f"expected one G8 curve, found {len(curves)}")
    curve = curves[0]
    losses = [float(value) for value in curve["seed_mean_losses"]]
    rung = min(range(len(losses)), key=losses.__getitem__)
    return curve, rung, float(curve["etas"][rung])


def telemetry_path(root: Path, record: dict) -> Path:
    node = {"h200-n1": "n1", "h200-n2": "n2"}.get(record["node"])
    if node is None:
        raise ValueError(f"unknown node: {record['node']}")
    return (
        root
        / node
        / "yeto-results-v8"
        / record["cell_id"]
        / f"attempt-{int(record['attempt'])}"
        / "work"
        / "m4"
        / "rho-telemetry.jsonl"
    )


def run_rho(path: Path, expected_t: int) -> tuple[float, float, int, str]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != "yeto_rho_telemetry_v1":
                raise ValueError(f"{path}:{line_number}: wrong schema")
            if int(row.get("outer_step", -1)) != line_number:
                raise ValueError(f"{path}:{line_number}: nonsequential outer_step")
            if int(row.get("fragment", -1)) != (line_number - 1) % 4:
                raise ValueError(f"{path}:{line_number}: fragment schedule mismatch")
            rows.append(row)
    if len(rows) != 4 * expected_t:
        raise ValueError(f"{path}: {len(rows)} rows, expected {4 * expected_t}")

    numerator = 0.0
    denominator = 0.0
    count = 0
    for index, row in enumerate(rows):
        cosine = row["autocorrelation"].get("lag_1")
        if cosine is None:
            continue
        if index < 4:
            raise ValueError(f"{path}: lag-1 defined without same-fragment predecessor")
        norm = float(row["pseudo_gradient"]["l2_norm"])
        previous_norm = float(rows[index - 4]["pseudo_gradient"]["l2_norm"])
        cosine = float(cosine)
        if not all(math.isfinite(value) for value in (norm, previous_norm, cosine)):
            raise ValueError(f"{path}: nonfinite lag-1 telemetry")
        weight = norm * previous_norm
        numerator += cosine * weight
        denominator += weight
        count += 1
    if count != 4 * (expected_t - 1) or denominator <= 0.0:
        raise ValueError(
            f"{path}: lag-1 count/weight invalid ({count}, {denominator})"
        )
    return numerator, denominator, count, sha256_file(path)


def normal_pvalue(statistic: float) -> float:
    return math.erfc(abs(statistic) / math.sqrt(2.0))


def fixed_effect_regression(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["stratum"]].append(row)
    eligible = {key: values for key, values in groups.items() if len(values) >= 2}
    within_x = []
    within_y = []
    for values in eligible.values():
        mean_x = sum(float(row["fisher_z_rho"]) for row in values) / len(values)
        mean_y = sum(float(row["ln_phi"]) for row in values) / len(values)
        within_x.extend(float(row["fisher_z_rho"]) - mean_x for row in values)
        within_y.extend(float(row["ln_phi"]) - mean_y for row in values)
    n = len(within_x)
    g = len(eligible)
    denominator = sum(value * value for value in within_x)
    if n <= g + 1 or denominator <= 0.0:
        raise ValueError("insufficient within-stratum regression variation")
    beta = sum(x * y for x, y in zip(within_x, within_y)) / denominator
    residuals = [y - beta * x for x, y in zip(within_x, within_y)]
    df_resid = n - g - 1
    sse = sum(value * value for value in residuals)
    ordinary_se = math.sqrt((sse / df_resid) / denominator)
    meat = sum(x * x * error * error for x, error in zip(within_x, residuals))
    hc1_se = math.sqrt((n / df_resid) * meat / denominator**2)
    sum_y2 = sum(value * value for value in within_y)
    pearson = (
        sum(x * y for x, y in zip(within_x, within_y))
        / math.sqrt(denominator * sum_y2)
        if sum_y2 > 0.0
        else None
    )
    return {
        "n": n,
        "strata": g,
        "residual_df": df_resid,
        "slope_ln_phi_per_fisher_z_rho": beta,
        "ordinary_se": ordinary_se,
        "ordinary_z": beta / ordinary_se if ordinary_se > 0.0 else None,
        "ordinary_normal_p_two_sided": (
            normal_pvalue(beta / ordinary_se) if ordinary_se > 0.0 else None
        ),
        "hc1_se": hc1_se,
        "hc1_z": beta / hc1_se if hc1_se > 0.0 else None,
        "hc1_normal_p_two_sided": (
            normal_pvalue(beta / hc1_se) if hc1_se > 0.0 else None
        ),
        "within_stratum_pearson_r": pearson,
        "sse": sse,
    }


def raw_regression(rows: list[dict]) -> dict:
    xs = [float(row["fisher_z_rho"]) for row in rows]
    ys = [float(row["ln_phi"]) for row in rows]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    denominator = sum(value * value for value in centered_x)
    slope = sum(x * y for x, y in zip(centered_x, centered_y)) / denominator
    intercept = mean_y - slope * mean_x
    residuals = [y - intercept - slope * x for x, y in zip(xs, ys)]
    sse = sum(value * value for value in residuals)
    se = math.sqrt((sse / (n - 2)) / denominator)
    pearson = sum(x * y for x, y in zip(centered_x, centered_y)) / math.sqrt(
        denominator * sum(value * value for value in centered_y)
    )
    return {
        "n": n,
        "intercept": intercept,
        "slope_ln_phi_per_fisher_z_rho": slope,
        "ordinary_se": se,
        "ordinary_z": slope / se,
        "ordinary_normal_p_two_sided": normal_pvalue(slope / se),
        "pearson_r": pearson,
        "sse": sse,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--g8-readout", required=True, type=Path)
    parser.add_argument("--telemetry-root", required=True, type=Path)
    parser.add_argument("--inputs-output", required=True, type=Path)
    parser.add_argument("--exclusions-output", required=True, type=Path)
    parser.add_argument("--results-output", required=True, type=Path)
    args = parser.parse_args()

    g8 = read_json(args.g8_readout)
    with args.ledger.open(newline="", encoding="utf-8") as handle:
        ledger = list(csv.DictReader(handle))
    inputs = []
    exclusions = []
    for row in ledger:
        if row["campaign"] != "G8":
            exclusions.append(
                {
                    "momentum_point_id": row["momentum_point_id"],
                    "campaign": row["campaign"],
                    "reason": "no complete frozen readout-to-evacuated-telemetry mapping",
                }
            )
            continue
        try:
            curve, rung, eta = selected_curve(g8, row)
            target = (int(row["T"]), int(row["H"]), int(row["S"]))
            arm = "corrected" if row["convention"] == "nesterov_corrected" else "raw"
            records = [
                record
                for record in g8["cell_records"]
                if record["arm"] == arm
                and dimensions(record) == target
                and float(record["mu"]) == float(row["mu"])
                and float(record["eta"]) == eta
            ]
            if not records:
                raise LookupError("no selected-rung cell records")
            total_num = 0.0
            total_den = 0.0
            total_count = 0
            cell_ids = []
            hashes = []
            for record in sorted(records, key=lambda item: int(item["seed"])):
                path = telemetry_path(args.telemetry_root, record)
                if not path.is_file():
                    continue
                numerator, denominator, count, digest = run_rho(path, int(row["T"]))
                total_num += numerator
                total_den += denominator
                total_count += count
                cell_ids.append(record["cell_id"])
                hashes.append(digest)
            if not cell_ids or total_den <= 0.0:
                raise LookupError("no usable selected-rung telemetry")
            rho = total_num / total_den
            if not -1.0 < rho < 1.0:
                raise ValueError(f"pooled rho outside (-1,1): {rho}")
            t = int(row["T"])
            mu = float(row["mu"])
            convention = row["convention"]
            prediction = (t / coefficient_c(convention, t, mu)) / (
                (1.0 - mu) * multiplier(convention, t, mu)
            )
            phi = float(row["observed_to_law_ratio"]) / prediction
            stratum = "|".join(
                (row["campaign"], row["scale"], convention, row["mu"])
            )
            inputs.append(
                {
                    "momentum_point_id": row["momentum_point_id"],
                    "campaign": row["campaign"],
                    "scale": row["scale"],
                    "T": t,
                    "H": int(row["H"]),
                    "S": int(row["S"]),
                    "convention": convention,
                    "mu": mu,
                    "observed_to_law_ratio": float(row["observed_to_law_ratio"]),
                    "c1_prediction": prediction,
                    "phi": phi,
                    "ln_phi": math.log(phi),
                    "selected_rung": f"e{rung}",
                    "selected_eta": eta,
                    "telemetry_run_count": len(cell_ids),
                    "telemetry_lag1_pair_count": total_count,
                    "rho_energy_weighted": rho,
                    "fisher_z_rho": math.atanh(rho),
                    "stratum": stratum,
                    "cell_ids": ";".join(cell_ids),
                    "telemetry_sha256s": ";".join(hashes),
                }
            )
        except (KeyError, LookupError, TypeError, ValueError) as exc:
            exclusions.append(
                {
                    "momentum_point_id": row["momentum_point_id"],
                    "campaign": row["campaign"],
                    "reason": str(exc),
                }
            )

    inputs.sort(key=lambda row: (row["stratum"], int(row["T"]), int(row["H"])))
    args.inputs_output.parent.mkdir(parents=True, exist_ok=True)
    with args.inputs_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INPUT_FIELDS)
        writer.writeheader()
        writer.writerows(inputs)
    with args.exclusions_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("momentum_point_id", "campaign", "reason")
        )
        writer.writeheader()
        writer.writerows(exclusions)

    result = {
        "schema": "yeto_c3_c4_rho_regression_v1",
        "provenance": {
            "ledger": str(args.ledger.resolve()),
            "ledger_sha256": sha256_file(args.ledger),
            "g8_readout": str(args.g8_readout.resolve()),
            "g8_readout_sha256": sha256_file(args.g8_readout),
            "telemetry_root": str(args.telemetry_root.resolve()),
        },
        "coverage": {
            "ledger_rows": len(ledger),
            "mapped_rows": len(inputs),
            "excluded_rows": len(exclusions),
            "mapped_campaigns": sorted({row["campaign"] for row in inputs}),
            "telemetry_runs": sum(int(row["telemetry_run_count"]) for row in inputs),
        },
        "within_stratum": fixed_effect_regression(inputs),
        "raw_unstratified_diagnostic": raw_regression(inputs),
    }
    args.results_output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
