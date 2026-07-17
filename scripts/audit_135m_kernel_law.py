#!/usr/bin/env python3
"""Mechanical A3 tape gate, finite-history prediction freeze, and G5–G7.

The mechanical gate never uses rho or loss to decide recapture.  The prediction
freeze consumes only already-public H16/H64/H256 curve losses plus loss-free
finite-kernel captures.  H8/H512 endpoint rows must still be null in the
checkpoint preseal, and the output explicitly records that those losses were
not exposed.  The frozen no-H-intercept quadratic is then evaluated after the
complete H8/H512 hidden bundle has sealed and unblinded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts import audit_135m_contract as audit


BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 135_003
POSITIVE_FLOOR = 1.0e-12


class KernelLawError(RuntimeError):
    """A3 kernel coverage, prediction chronology, or gate input is invalid."""


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sealed_at(value: str | None) -> str:
    timestamp = value or utc_now()
    if not timestamp.endswith("Z"):
        raise KernelLawError("seal timestamp must be UTC Z")
    datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    return timestamp


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise KernelLawError(f"{label} must be a UTC Z timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise KernelLawError(f"{label} is not ISO-8601") from exc
    if parsed.tzinfo is None:
        raise KernelLawError(f"{label} lacks timezone information")
    return parsed.astimezone(timezone.utc)


def load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise KernelLawError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise KernelLawError(f"{label} must be a JSON object")
    return value


def write_create_only(path: Path, value: object) -> None:
    if path.exists():
        raise KernelLawError(f"refusing to overwrite create-only artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise KernelLawError(f"{label} must be an array")
    return value


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise KernelLawError(f"{label} must be an object")
    return dict(value)


def _expected_cells(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for raw in _array(manifest.get("expected_cells"), "expected cells"):
        row = _mapping(raw, "expected cell")
        cell_id = row.get("cell_id")
        if not isinstance(cell_id, str) or cell_id in result:
            raise KernelLawError("expected cell IDs are missing/duplicated")
        result[cell_id] = row
    return result


def _final_rows(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    expected = _expected_cells(manifest)
    final: dict[str, tuple[int, int, dict[str, Any]]] = {}
    for order, raw in enumerate(_array(manifest.get("results"), "result rows")):
        row = _mapping(raw, "result row")
        cell_id = row.get("cell_id")
        attempt = row.get("attempt")
        if cell_id not in expected or not isinstance(attempt, int) or isinstance(attempt, bool):
            raise KernelLawError("result row identity/attempt is malformed")
        prior = final.get(str(cell_id))
        if prior is None or (attempt, order) > (prior[0], prior[1]):
            final[str(cell_id)] = (attempt, order, row)
    return {cell_id: value[2] for cell_id, value in final.items()}


def mechanical_gate(args: argparse.Namespace) -> dict[str, Any]:
    audit.load_authority()
    phase = load_object(args.historical_phase_manifest, "historical phase manifest")
    cells = _expected_cells(phase)
    final = _final_rows(phase)
    rows = []
    for h in (16, 64, 256):
        matches = [
            cell_id
            for cell_id, cell in cells.items()
            if cell.get("audit_stage") is None
            and int(cell.get("h", -1)) == h
            and int(cell.get("m", 4)) == 4
            and int(cell.get("seed", -1)) == 347
            and math.isclose(float(cell.get("mu", math.nan)), 0.0, abs_tol=1e-15)
            and math.isclose(
                float(cell.get("eta", math.nan)), 0.021875, abs_tol=1e-15
            )
        ]
        if len(matches) != 1 or matches[0] not in final:
            raise KernelLawError(f"historical matched-eta H={h} cell is missing")
        result = final[matches[0]]
        projection = _mapping(
            result.get("parallel_attempt_projection"), "parallel attempt projection"
        )
        inventory = _mapping(projection.get("artifact_inventory"), "artifact inventory")
        tape = _mapping(inventory.get("raw_tape"), "raw tape artifact")
        tape_path = (args.historical_campaign_root / str(tape["path"])).resolve()
        tape_ok = (
            tape_path.is_file()
            and not tape_path.is_symlink()
            and sha256_file(tape_path) == tape.get("sha256")
            and tape_path.stat().st_size == tape.get("size_bytes")
        )
        compact = inventory.get("finite_kernel_capture")
        compact_ok = False
        if isinstance(compact, Mapping) and isinstance(compact.get("path"), str):
            compact_path = (
                args.historical_campaign_root / str(compact["path"])
            ).resolve()
            compact_ok = (
                compact_path.is_file()
                and sha256_file(compact_path) == compact.get("sha256")
            )
        rows.append(
            {
                "H": h,
                "cell_id": matches[0],
                "raw_event_tape_present_and_hashed": tape_ok,
                "full_merged_delta_or_finite_kernel_capture_present": compact_ok,
                "mechanical_gate_pass": tape_ok and compact_ok,
            }
        )
    recapture = not all(row["mechanical_gate_pass"] for row in rows)
    value = {
        "schema": "audit_135m_a3_mechanical_tape_gate_v1",
        "status": "SEALED",
        "audit_stage": "A3",
        "authority_prereg_sha256": audit.PREREG_JSON_SHA256,
        "decision_inputs": "artifact_presence_hash_coverage_only",
        "loss_or_rho_used_to_trigger": False,
        "historical_phase_manifest_raw_sha256": sha256_file(
            args.historical_phase_manifest
        ),
        "rows": rows,
        "matched_eta_recapture_required": recapture,
        "authorized_recapture": {
            "H": [16, 64, 256],
            "mu": 0.0,
            "eta": 0.021875,
            "seed": 347,
        }
        if recapture
        else None,
        "sealed_at_utc": _sealed_at(args.sealed_at_utc),
    }
    value["gate_canonical_sha256"] = canonical_sha256(value)
    write_create_only(args.output, value)
    return {
        "status": "SEALED",
        "recapture_required": recapture,
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
    }


def _analysis_attempts(source: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    attempts = {
        str(row["attempt_id"]): _mapping(row, "attempt")
        for row in _array(source.get("attempts"), "attempts")
    }
    analysis = _mapping(source.get("analysis_rounds"), "analysis rounds")
    result = {}
    for cell_id, raw in analysis.items():
        selected = _mapping(raw, "analysis round")
        attempt = attempts.get(str(selected.get("attempt_id")))
        if attempt is None or attempt.get("cell_id") != cell_id:
            raise KernelLawError("analysis round does not join to its attempt")
        result[str(cell_id)] = attempt
    return result


def _capture_rows(
    *,
    source: Mapping[str, Any],
    bound: Mapping[str, Any],
    campaign_root: Path,
) -> list[dict[str, Any]]:
    expected = _expected_cells(bound)
    attempts = _analysis_attempts(source)
    rows = []
    for cell_id, cell in expected.items():
        if cell.get("finite_kernel_capture_required") is not True:
            continue
        attempt = attempts.get(cell_id)
        if attempt is None or attempt.get("status") != "COMPLETED" or attempt.get("loss") is not None:
            raise KernelLawError("kernel capture cell is not completed and loss-free")
        inventory = _mapping(attempt.get("artifact_inventory"), "capture inventory")
        entry = _mapping(inventory.get("finite_kernel_capture"), "kernel capture entry")
        path = (campaign_root / str(entry["path"])).resolve()
        if (
            not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != entry.get("sha256")
            or path.stat().st_size != entry.get("size_bytes")
        ):
            raise KernelLawError("kernel capture artifact hash/size differs")
        capture = load_object(path, "finite-kernel capture")
        preimage = dict(capture)
        digest = preimage.pop("capture_canonical_sha256", None)
        if (
            capture.get("schema") != "audit_135m_finite_kernel_capture_v1"
            or capture.get("status") != "SEALED"
            or capture.get("loss_exposed") is not False
            or digest != canonical_sha256(preimage)
            or capture.get("state_transition_replay_exact") is not True
            or capture.get("all_registered_updates_covered") is not True
        ):
            raise KernelLawError("finite-kernel capture integrity differs")
        rows.append(
            {
                "cell_id": cell_id,
                "H": int(cell["h"]),
                "seed": int(cell["seed"]),
                "eta": float(cell["eta"]),
                "capture_raw_sha256": sha256_file(path),
                "capture_canonical_sha256": digest,
                "scientific_ended_at_utc": attempt["scientific_ended_at"],
                "K_H": int(capture["K_H"]),
                "V_H_psd": float(capture["V_H_psd"]),
                "ordered_update_registry_hash": capture[
                    "ordered_update_registry_hash"
                ],
                "all_registered_updates_covered": True,
                "state_transition_replay_exact": True,
            }
        )
    return rows


def _historical_curve_points(phase: Mapping[str, Any]) -> list[dict[str, Any]]:
    cells = _expected_cells(phase)
    final = _final_rows(phase)
    points = []
    for cell_id, cell in cells.items():
        if (
            cell.get("audit_stage") is None
            and int(cell.get("h", -1)) in (16, 64, 256)
            and int(cell.get("m", 4)) == 4
            and int(cell.get("seed", -1)) == 347
            and math.isclose(float(cell.get("mu", math.nan)), 0.0, abs_tol=1e-15)
        ):
            row = final.get(cell_id)
            loss = None if row is None else row.get("loss")
            if (
                row is None
                or row.get("status") != "COMPLETED"
                or isinstance(loss, bool)
                or not isinstance(loss, (int, float))
                or not math.isfinite(float(loss))
            ):
                raise KernelLawError("historical mu=0 curve contains unresolved work")
            points.append(
                {
                    "cell_id": cell_id,
                    "H": int(cell["h"]),
                    "eta": float(cell["eta"]),
                    "loss": float(loss),
                }
            )
    counts = {h: sum(row["H"] == h for row in points) for h in (16, 64, 256)}
    if any(count < 3 for count in counts.values()):
        raise KernelLawError(f"historical curve coverage is too small: {counts}")
    return sorted(points, key=lambda row: (row["H"], row["eta"]))


def _fit_positive_quadratic(
    points: Sequence[Mapping[str, Any]], kv: Mapping[int, tuple[float, float]]
) -> dict[str, Any]:
    x = np.asarray(
        [
            [1.0, -kv[int(row["H"])][0] * float(row["eta"]), kv[int(row["H"])][1] * float(row["eta"]) ** 2]
            for row in points
        ],
        dtype=np.float64,
    )
    y = np.asarray([float(row["loss"]) for row in points], dtype=np.float64)

    def candidate(fixed_a: float | None, fixed_b: float | None) -> np.ndarray:
        adjusted = y.copy()
        columns = [np.ones(len(points), dtype=np.float64)]
        positions = [0]
        if fixed_a is None:
            columns.append(x[:, 1])
            positions.append(1)
        else:
            adjusted -= x[:, 1] * fixed_a
        if fixed_b is None:
            columns.append(x[:, 2])
            positions.append(2)
        else:
            adjusted -= x[:, 2] * fixed_b
        design = np.column_stack(columns)
        fitted, *_ = np.linalg.lstsq(design, adjusted, rcond=None)
        result = np.zeros(3, dtype=np.float64)
        result[1] = POSITIVE_FLOOR if fixed_a is None else fixed_a
        result[2] = POSITIVE_FLOOR if fixed_b is None else fixed_b
        for position, value in zip(positions, fitted):
            result[position] = value
        return result

    candidates = [np.linalg.lstsq(x, y, rcond=None)[0]]
    candidates.extend(
        (
            candidate(POSITIVE_FLOOR, None),
            candidate(None, POSITIVE_FLOOR),
            candidate(POSITIVE_FLOOR, POSITIVE_FLOOR),
        )
    )
    eligible = [row for row in candidates if row[1] > 0.0 and row[2] > 0.0]
    if not eligible:
        raise KernelLawError("positive constrained law fit has no eligible solution")
    coefficient = min(eligible, key=lambda row: float(np.sum((y - x @ row) ** 2)))
    prediction = x @ coefficient
    residual = y - prediction
    return {
        "coefficient": coefficient,
        "prediction": prediction,
        "residual": residual,
        "sse": float(np.sum(residual**2)),
        "rmse": float(math.sqrt(np.mean(residual**2))),
        "x": x,
        "y": y,
    }


def _prediction(coefficient: np.ndarray, *, k_h: float, v_h: float) -> tuple[float, float]:
    _l_init, a, b = coefficient
    eta = float(a * k_h / (2.0 * b * v_h))
    frontier = float(coefficient[0] - a * a * k_h * k_h / (4.0 * b * v_h))
    return eta, frontier


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    audit.load_authority()
    mechanical = load_object(args.mechanical_gate, "A3 mechanical tape gate")
    if (
        mechanical.get("schema") != "audit_135m_a3_mechanical_tape_gate_v1"
        or mechanical.get("status") != "SEALED"
        or mechanical.get("loss_or_rho_used_to_trigger") is not False
        or mechanical.get("matched_eta_recapture_required") is not True
        or mechanical.get("authorized_recapture")
        != {"H": [16, 64, 256], "mu": 0.0, "eta": 0.021875, "seed": 347}
    ):
        raise KernelLawError(
            "A3 recapture freeze lacks the exact loss-blind mechanical authorization"
        )
    historical = load_object(args.historical_phase_manifest, "historical phase")
    recapture_source = load_object(args.recapture_campaign_manifest, "A3 recapture campaign")
    recapture_bound = load_object(args.recapture_bound_manifest, "A3 recapture bound")
    current_preseal = load_object(args.current_checkpoint_preseal, "A3 current preseal")
    current_bound = load_object(args.current_bound_manifest, "A3 current bound")
    if any(
        row.get("loss") is not None
        for row in _array(current_preseal.get("attempts"), "A3 current attempts")
        if row.get("status") == "COMPLETED"
    ):
        raise KernelLawError("H8/H512 endpoint loss was exposed before prediction freeze")
    recaptures = _capture_rows(
        source=recapture_source,
        bound=recapture_bound,
        campaign_root=args.recapture_campaign_root,
    )
    current = _capture_rows(
        source=current_preseal,
        bound=current_bound,
        campaign_root=args.current_campaign_root,
    )
    captures = recaptures + current
    by_h: dict[int, list[dict[str, Any]]] = {}
    for row in captures:
        by_h.setdefault(int(row["H"]), []).append(row)
    if set(by_h) != {8, 16, 64, 256, 512}:
        raise KernelLawError(f"finite-kernel horizon coverage differs: {sorted(by_h)}")
    if any(len(by_h[h]) != (2 if h in (8, 512) else 1) for h in by_h):
        raise KernelLawError("finite-kernel seed replicate coverage differs")
    kv = {
        h: (
            float(by_h[h][0]["K_H"]),
            float(np.mean([row["V_H_psd"] for row in by_h[h]])),
        )
        for h in by_h
    }
    if any(len({row["K_H"] for row in rows}) != 1 for rows in by_h.values()):
        raise KernelLawError("kernel replicate K_H values differ")
    points = _historical_curve_points(historical)
    fit = _fit_positive_quadratic(points, kv)
    coefficient = fit["coefficient"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = {8: [], 512: []}
    accepted = 0
    mammen_low = (1.0 - math.sqrt(5.0)) / 2.0
    mammen_high = (1.0 + math.sqrt(5.0)) / 2.0
    probability_low = (math.sqrt(5.0) + 1.0) / (2.0 * math.sqrt(5.0))
    for _draw in range(BOOTSTRAP_DRAWS):
        weights = {
            h: (mammen_low if rng.random() < probability_low else mammen_high)
            for h in (16, 64, 256)
        }
        synthetic = []
        for index, point in enumerate(points):
            row = dict(point)
            row["loss"] = float(
                fit["prediction"][index]
                + weights[int(point["H"])] * fit["residual"][index]
            )
            synthetic.append(row)
        refit = _fit_positive_quadratic(synthetic, kv)
        coef = refit["coefficient"]
        if coef[1] <= POSITIVE_FLOOR or coef[2] <= POSITIVE_FLOOR:
            continue
        accepted += 1
        for h in (8, 512):
            sampled_v = float(by_h[h][int(rng.integers(0, len(by_h[h])))] ["V_H_psd"])
            boot[h].append(_prediction(coef, k_h=kv[h][0], v_h=sampled_v))
    if accepted < BOOTSTRAP_DRAWS // 2:
        raise KernelLawError("too few positive constrained bootstrap fits")
    predictions = {}
    for h in (8, 512):
        eta, frontier = _prediction(coefficient, k_h=kv[h][0], v_h=kv[h][1])
        samples = np.asarray(boot[h], dtype=np.float64)
        predictions[str(h)] = {
            "K_H": kv[h][0],
            "V_H_psd_mean": kv[h][1],
            "V_H_psd_by_seed": {
                str(row["seed"]): row["V_H_psd"] for row in by_h[h]
            },
            "eta_hat_star": eta,
            "frontier_hat": frontier,
            "eta_prediction_interval_95": [
                float(np.quantile(samples[:, 0], 0.025)),
                float(np.quantile(samples[:, 0], 0.975)),
            ],
            "frontier_prediction_interval_95": [
                float(np.quantile(samples[:, 1], 0.025)),
                float(np.quantile(samples[:, 1], 0.975)),
            ],
        }
    sealed_at = _sealed_at(args.sealed_at_utc)
    chronology_floor = max(
        [_parse_utc(current_preseal.get("sealed_at_utc"), "A3 checkpoint preseal time")]
        + [
            _parse_utc(row.get("scientific_ended_at_utc"), "kernel capture completion")
            for row in captures
        ]
    )
    if _parse_utc(sealed_at, "A3 prediction freeze time") <= chronology_floor:
        raise KernelLawError(
            "A3 prediction freeze must follow the preseal and every kernel capture"
        )
    value = {
        "schema": "audit_135m_a3_prediction_freeze_v1",
        "status": "SEALED",
        "audit_stage": "A3",
        "authority_prereg_sha256": audit.PREREG_JSON_SHA256,
        "loss_exposed_for_H8_H512": False,
        "mechanical_gate_raw_sha256": sha256_file(args.mechanical_gate),
        "historical_phase_manifest_raw_sha256": sha256_file(
            args.historical_phase_manifest
        ),
        "recapture_campaign_manifest_raw_sha256": sha256_file(
            args.recapture_campaign_manifest
        ),
        "recapture_bound_manifest_raw_sha256": sha256_file(
            args.recapture_bound_manifest
        ),
        "current_checkpoint_preseal_raw_sha256": sha256_file(
            args.current_checkpoint_preseal
        ),
        "current_bound_manifest_raw_sha256": sha256_file(args.current_bound_manifest),
        "kernel_estimator": {
            "rho": "full_finite_energy_weighted_lag_kernel",
            "psd_regularization": (
                "capture_sidecar_toeplitz_eigenvalue_clip_1e-8_and_diagonal_renormalization"
            ),
            "V_H": "sum_f[K_f+2*sum_{k=1}^{K_f-1}(K_f-k)*rho_k(H,f)]",
            "replicate_combination_H8_H512": "arithmetic_mean_of_seed_level_V_H",
        },
        "kernel_integrity": {
            "all_five_horizons_present": True,
            "all_state_transition_replays_exact": True,
            "all_registered_updates_covered": True,
            "captures": captures,
            "captures_hash": canonical_sha256(captures),
        },
        "model": "L_hat(H,eta)=L_init-a*K_H*eta+b*V_H*eta^2",
        "constraints": {"a_positive": True, "b_positive": True},
        "fit_curve_points": points,
        "fit_curve_points_hash": canonical_sha256(points),
        "coefficients": {
            "L_init": float(coefficient[0]),
            "a": float(coefficient[1]),
            "b": float(coefficient[2]),
        },
        "fit_rmse": fit["rmse"],
        "uncertainty": {
            "procedure": "H_cluster_Mammen_wild_bootstrap_plus_H8_H512_seed_V_resampling",
            "draws_requested": BOOTSTRAP_DRAWS,
            "draws_accepted": accepted,
            "rng_seed": BOOTSTRAP_SEED,
        },
        "predictions": predictions,
        "sealed_at_utc": sealed_at,
    }
    value["prediction_freeze_canonical_sha256"] = canonical_sha256(value)
    write_create_only(args.output, value)
    return {
        "status": "SEALED",
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "fit_rmse": fit["rmse"],
        "predictions": predictions,
        "loss_exposed_for_H8_H512": False,
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    prediction = load_object(args.prediction_freeze, "A3 prediction freeze")
    selection = load_object(args.selection_evidence, "A3 selection evidence")
    if (
        prediction.get("schema") != "audit_135m_a3_prediction_freeze_v1"
        or prediction.get("status") != "SEALED"
        or prediction.get("loss_exposed_for_H8_H512") is not False
        or selection.get("schema") != "audit_135m_a3_frontier_selection_evidence_v1"
    ):
        raise KernelLawError("A3 prediction/selection identity differs")
    if datetime.fromisoformat(
        str(selection.get("sealed_at_utc")).replace("Z", "+00:00")
    ) <= datetime.fromisoformat(
        str(prediction.get("sealed_at_utc")).replace("Z", "+00:00")
    ):
        raise KernelLawError("A3 selection does not follow the prediction freeze")
    observed = {}
    for curve in _array(selection.get("curves"), "A3 selected curves"):
        h = int(curve["H"])
        loss = curve.get("selected_pooled_mean")
        if h not in (8, 512) or not isinstance(loss, (int, float)):
            raise KernelLawError("A3 selected frontier has a nonfinite endpoint")
        observed[h] = {
            "eta": float(curve["selected_eta"]),
            "loss": float(loss),
            "bracketed": bool(curve["final_interior_with_worse_neighbors"]),
        }
    if set(observed) != {8, 512}:
        raise KernelLawError("A3 final selection lacks H8/H512")
    gate_rows = {}
    for h in (8, 512):
        predicted = prediction["predictions"][str(h)]
        eta_error = abs(math.log2(observed[h]["eta"] / predicted["eta_hat_star"]))
        interval = predicted["frontier_prediction_interval_95"]
        gate_rows[str(h)] = {
            "observed_selected_eta": observed[h]["eta"],
            "predicted_eta": predicted["eta_hat_star"],
            "absolute_log2_eta_error": eta_error,
            "eta_error_pass": observed[h]["bracketed"] and eta_error <= 0.5,
            "observed_tuned_loss": observed[h]["loss"],
            "predicted_frontier_loss": predicted["frontier_hat"],
            "prediction_interval_95": interval,
            "prediction_interval_coverage": interval[0]
            <= observed[h]["loss"]
            <= interval[1],
            "bracketed": observed[h]["bracketed"],
        }
    rmse = math.sqrt(
        sum(
            (
                observed[h]["loss"]
                - prediction["predictions"][str(h)]["frontier_hat"]
            )
            ** 2
            for h in (8, 512)
        )
        / 2.0
    )
    existing = audit.load_authority()["experiments"]["A3"]["existing_minima"]
    frontier = {
        "8": observed[8]["loss"],
        "16": float(existing["16"]["nll"]),
        "64": float(existing["64"]["nll"]),
        "256": float(existing["256"]["nll"]),
        "512": observed[512]["loss"],
    }
    ordering = frontier["8"] <= frontier["16"] and frontier["512"] >= frontier["256"]
    g5 = all(
        prediction["kernel_integrity"][key]
        for key in (
            "all_five_horizons_present",
            "all_state_transition_replays_exact",
            "all_registered_updates_covered",
        )
    )
    g6 = all(row["eta_error_pass"] for row in gate_rows.values())
    g7 = (
        all(row["prediction_interval_coverage"] for row in gate_rows.values())
        and rmse <= 0.010
        and ordering
    )
    report = {
        "schema": "audit_135m_a3_analysis_v1",
        "status": "SEALED",
        "audit_stage": "A3",
        "authority_prereg_sha256": audit.PREREG_JSON_SHA256,
        "prediction_freeze_raw_sha256": sha256_file(args.prediction_freeze),
        "selection_evidence_raw_sha256": sha256_file(args.selection_evidence),
        "rows": gate_rows,
        "extension_point_frontier_rmse": rmse,
        "five_point_frontier": frontier,
        "required_endpoint_ordering_pass": ordering,
        "gates": {
            "G5_A3_kernel_integrity": "PASS" if g5 else "FAIL",
            "G6_A3_eta_prediction": "PASS" if g6 else "FAIL",
            "G7_A3_frontier_prediction": "PASS" if g7 else "FAIL",
            "A3_quantitative_law": "PASS" if g5 and g6 and g7 else "FAIL_DROP_LAW_CLAIM",
        },
        "sealed_at_utc": _sealed_at(args.sealed_at_utc),
    }
    report["analysis_canonical_sha256"] = canonical_sha256(report)
    write_create_only(args.output, report)
    return {
        "status": "SEALED",
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "gates": report["gates"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    gate = sub.add_parser("mechanical-gate")
    gate.add_argument("--historical-phase-manifest", type=Path, required=True)
    gate.add_argument("--historical-campaign-root", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)
    gate.add_argument("--sealed-at-utc")

    frozen = sub.add_parser("freeze")
    frozen.add_argument("--mechanical-gate", type=Path, required=True)
    frozen.add_argument("--historical-phase-manifest", type=Path, required=True)
    frozen.add_argument("--recapture-campaign-manifest", type=Path, required=True)
    frozen.add_argument("--recapture-bound-manifest", type=Path, required=True)
    frozen.add_argument("--recapture-campaign-root", type=Path, required=True)
    frozen.add_argument("--current-checkpoint-preseal", type=Path, required=True)
    frozen.add_argument("--current-bound-manifest", type=Path, required=True)
    frozen.add_argument("--current-campaign-root", type=Path, required=True)
    frozen.add_argument("--output", type=Path, required=True)
    frozen.add_argument("--sealed-at-utc")

    analysis = sub.add_parser("analyze")
    analysis.add_argument("--prediction-freeze", type=Path, required=True)
    analysis.add_argument("--selection-evidence", type=Path, required=True)
    analysis.add_argument("--output", type=Path, required=True)
    analysis.add_argument("--sealed-at-utc")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    actions = {
        "mechanical-gate": mechanical_gate,
        "freeze": freeze,
        "analyze": analyze,
    }
    try:
        result = actions[args.action](args)
    except (KernelLawError, OSError, ValueError, KeyError, audit.AuditContractError) as exc:
        print(f"A3 kernel-law error: {exc}", file=__import__("sys").stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
