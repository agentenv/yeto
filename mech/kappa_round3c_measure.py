#!/usr/bin/env python3
"""Frozen Round-3C trajectory identification measurement.

The scientific specification is ``mech/kappa-zeroshot-round3c-protocol.md``.
This program opens only the frozen banked mu=0 loss tapes, the two Round-3
trajectory manifests/checkpoints/tapes, and their eight Lane-E spectrum JSON
files. It never opens a rerun endpoint report or a banked corrected-arm loss.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import struct
import sys
from pathlib import Path
from statistics import fmean
from types import ModuleType
from typing import Sequence


PROTOCOL_SHA256 = "7e61343200d5248e24bb35e25c09c6499c9076f8c44f251185f5666c932ef2da"
ROUND2_ANALYZER_SHA256 = (
    "4902dfff95b74bb089eb7fa10a483d9a3082d529c7b39966cda718310a9f6b16"
)
ROUND2_RESULT_SHA256 = (
    "28358ada20742685128003d19cf63b8fb807a0e3851ccc5b602b712cd617b0e7"
)
SPECTRUM_ADAPTER_SHA256 = (
    "857c88c2a227c32f983c5d206c48d43f49792cdda2f797db691df1386e46d8bd"
)
EVAL_SHA256 = "533838a0564b13519956a044d23ed8db6705ddc7ae5f0ddb96538f49460bcebc"
CKPT_MAGIC = 0xD1705A7E
ARMS = ("mu0", "corrected")
AGES = (5, 10, 15, 20)
STEPS = tuple(4 * age for age in AGES)
TARGET_INTERVAL = (0.9932, 0.9938)
EXPECTED_MODEL_FILES = {
    "config.json": "1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843",
    "generation_config.json": "2056c988e990b0d13670f63f2f3b87b3b6d07edaf7a3416998ba27dab2d8a059",
    "merges.txt": "0b54e8aa4e53d5383e2e4bc635a56b43f9647f7b13832d5d9ecd8f82dac4f510",
    "model-id.txt": "658bb1a9a4cd06ae39e30e597bd9585330a0da295e265e2a99ec9f207a3b8424",
    "model-revision.txt": "c701d5cd6ed2c1d78e9fbfae6f5f09a08b8759be48cffb2bbf8c8246ba6f1d40",
    "model.safetensors": "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1",
    "special_tokens_map.json": "e786b595b9a23148bf1630df78d9037a048ea671e48bfd3549a1e3c233742bb3",
    "tokenizer.json": "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
    "tokenizer_config.json": "4bb9af56a342753d39374f4016a16574cab299fe088e896f425ce3c433f61424",
    "vocab.json": "82b84012e3add4d01d12ba14442026e49b8cbbaead1f79ecf3d919784f82dc79",
}


class GateFailure(RuntimeError):
    """A preregistered input/model/execution gate failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(32 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_regular(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise GateFailure(f"{label} is not a regular non-symlink file: {path}")


def require_sha(path: Path, expected: str, label: str) -> None:
    require_regular(path, label)
    observed = sha256_file(path)
    if observed != expected:
        raise GateFailure(f"{label} SHA-256 mismatch: {observed} != {expected}: {path}")


def load_module(path: Path) -> ModuleType:
    require_sha(path, ROUND2_ANALYZER_SHA256, "Round-2 analyzer")
    spec = importlib.util.spec_from_file_location("kappa_round2_frozen", path)
    if spec is None or spec.loader is None:
        raise GateFailure(f"cannot import frozen Round-2 analyzer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_model(model_root: Path) -> dict[str, str]:
    if not model_root.is_dir() or model_root.is_symlink():
        raise GateFailure(f"model root is not a regular directory: {model_root}")
    observed = {}
    for name, expected in EXPECTED_MODEL_FILES.items():
        path = model_root / name
        require_sha(path, expected, f"model file {name}")
        observed[name] = expected
    return observed


def frozen_loss_fit(
    args, round2: ModuleType
) -> tuple[dict[str, object], dict[str, object]]:
    cells = round2.locate_exact_cells(args.loss_root)
    evidence = round2.verify_loss_cells(cells)
    losses, counts = round2.parse_losses(cells)
    steps, values = round2.aggregate_loss_curve(losses, round2.SEEDS)
    fitted = round2.floor_profile_fit(steps, values)

    require_sha(args.round2_result, ROUND2_RESULT_SHA256, "Round-2 result")
    crosscheck = json.loads(args.round2_result.read_text())["loss_floor"]
    if float(fitted["ell_inf"]) != float(crosscheck["ell_inf"]):
        raise GateFailure(
            "Round-3 floor does not exactly reproduce the frozen Round-2 floor"
        )
    if float(fitted["beta"]) != float(crosscheck["beta"]):
        raise GateFailure(
            "Round-3 beta does not exactly reproduce the frozen Round-2 beta"
        )

    result = {
        **fitted,
        **counts,
        "cells": sorted(cells),
        "evidence": evidence,
        "round2_exact_crosscheck": True,
    }
    provenance = {
        "analyzer": str(args.round2_analyzer.resolve()),
        "analyzer_sha256": ROUND2_ANALYZER_SHA256,
        "prior_result": str(args.round2_result.resolve()),
        "prior_result_sha256": ROUND2_RESULT_SHA256,
    }
    return result, provenance


def checkpoint_header(path: Path) -> int:
    with path.open("rb") as handle:
        header = handle.read(12)
    if len(header) != 12:
        raise GateFailure(f"short checkpoint header: {path}")
    magic, step = struct.unpack("<IQ", header)
    if magic != CKPT_MAGIC:
        raise GateFailure(f"checkpoint magic mismatch: {path}")
    return int(step)


def load_trajectories(
    root: Path,
) -> tuple[dict[tuple[str, int], dict[str, object]], list[dict[str, object]]]:
    expected: dict[tuple[str, int], dict[str, object]] = {}
    manifests = []
    for arm in ARMS:
        arm_root = root / arm
        if not arm_root.is_dir() or arm_root.is_symlink():
            raise GateFailure(f"missing regular trajectory directory: {arm_root}")
        manifest_path = arm_root / "trajectory-manifest.json"
        require_regular(manifest_path, f"{arm} trajectory manifest")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != "yeto_mechanism_trajectory_archive_v1":
            raise GateFailure(f"{arm} manifest schema mismatch")
        if manifest.get("requested_global_steps") != list(STEPS):
            raise GateFailure(f"{arm} manifest requested steps mismatch")
        if manifest.get("complete") is not True:
            raise GateFailure(f"{arm} trajectory is not complete")
        files = manifest.get("files")
        if not isinstance(files, list) or len(files) != len(STEPS):
            raise GateFailure(f"{arm} manifest file count mismatch")
        arm_records = []
        for age, step, record in zip(AGES, STEPS, files):
            if int(record.get("global_step", -1)) != step:
                raise GateFailure(f"{arm} manifest step mismatch at age {age}")
            checkpoint = arm_root / Path(str(record.get("path", ""))).name
            require_regular(checkpoint, f"{arm} age-{age} checkpoint")
            size = checkpoint.stat().st_size
            digest = sha256_file(checkpoint)
            if size != int(record.get("size_bytes", -1)):
                raise GateFailure(f"{arm} age-{age} checkpoint size mismatch")
            if digest != record.get("sha256"):
                raise GateFailure(f"{arm} age-{age} checkpoint manifest hash mismatch")
            if checkpoint_header(checkpoint) != step:
                raise GateFailure(f"{arm} age-{age} checkpoint header mismatch")
            item = {
                "arm": arm,
                "age": age,
                "global_step": step,
                "path": str(checkpoint.resolve()),
                "size_bytes": size,
                "sha256": digest,
            }
            expected[(arm, age)] = item
            arm_records.append(item)
        manifests.append(
            {
                "arm": arm,
                "path": str(manifest_path.resolve()),
                "sha256": sha256_file(manifest_path),
                "checkpoints": arm_records,
            }
        )
    if set(expected) != {(arm, age) for arm in ARMS for age in AGES}:
        raise GateFailure("trajectory panel does not equal the frozen eight inputs")
    return expected, manifests


def probe_path(root: Path, arm: str, age: int) -> Path:
    step = 4 * age
    return root / arm / f"age_{age:02d}_step_{step:08d}.json"


def load_spectra(
    root: Path,
    checkpoints: dict[tuple[str, int], dict[str, object]],
    model_root: Path,
) -> tuple[
    dict[tuple[str, int], float], list[dict[str, object]], list[dict[str, object]]
]:
    curvatures = {}
    records = []
    invalid = []
    for arm in ARMS:
        for age in AGES:
            path = probe_path(root, arm, age)
            require_regular(path, f"{arm} age-{age} spectrum")
            payload = json.loads(path.read_text())
            failures = []
            if payload.get("schema") != "yeto_checkpoint_spectrum_probe_v1":
                failures.append("schema")
            if payload.get("status") != "COMPLETE":
                failures.append("status")
            provenance = payload.get("provenance", {})
            checkpoint = payload.get("checkpoint", {})
            probe = payload.get("probe", {})
            frozen_checkpoint = checkpoints[(arm, age)]
            if provenance.get("checkpoint_sha256") != frozen_checkpoint["sha256"]:
                failures.append("checkpoint_sha256")
            if provenance.get("data_sha256") != EVAL_SHA256:
                failures.append("data_sha256")
            if int(provenance.get("seed", -1)) != 20260727:
                failures.append("probe_seed")
            try:
                if Path(str(provenance.get("model"))).resolve() != model_root.resolve():
                    failures.append("model_path")
            except (OSError, RuntimeError):
                failures.append("model_path")
            if int(checkpoint.get("global_step", -1)) != 4 * age:
                failures.append("global_step")
            versions = [int(value) for value in checkpoint.get("fragment_versions", [])]
            if versions != list(range(4 * age - 3, 4 * age + 1)):
                failures.append("fragment_versions")
            fragment_numel = checkpoint.get("fragment_numel", [])
            if len(fragment_numel) != 4 or any(
                int(value) <= 0 for value in fragment_numel
            ):
                failures.append("fragment_layout")
            settings = {
                "seq_len": 128,
                "panels": 4,
                "batch_size": 1,
                "max_rows": 128,
                "train_on": "assistant",
                "block_steps": 4,
                "krylov_rank": 8,
            }
            for key, expected in settings.items():
                if probe.get(key) != expected:
                    failures.append(f"probe_{key}")

            try:
                ritz = [float(value) for value in probe["ritz_values"]]
                coords = [float(value) for value in probe["gradient_ritz_coordinates"]]
            except (KeyError, TypeError, ValueError):
                ritz, coords = [], []
                failures.append("spectrum_vectors")
            if len(ritz) != 8 or len(coords) != 8:
                failures.append("rank")
            if any(not math.isfinite(value) or value == 0.0 for value in ritz):
                failures.append("ritz_finite_nonzero")
            if any(not math.isfinite(value) for value in coords):
                failures.append("coords_finite")

            mass = sum(value * value for value in coords)
            inverse_form = (
                sum(value * value / lam for value, lam in zip(coords, ritz))
                if len(ritz) == len(coords) == 8
                and all(math.isfinite(value) and value != 0.0 for value in ritz)
                and all(math.isfinite(value) for value in coords)
                else math.nan
            )
            effective = (
                mass / inverse_form
                if mass > 0.0 and inverse_form > 0.0 and math.isfinite(inverse_form)
                else math.nan
            )
            if not mass > 0.0 or not math.isfinite(mass):
                failures.append("gradient_mass")
            if not inverse_form > 0.0 or not math.isfinite(inverse_form):
                failures.append("inverse_quadratic_form")
            if not effective > 0.0 or not math.isfinite(effective):
                failures.append("effective_curvature")

            record = {
                "arm": arm,
                "age": age,
                "global_step": 4 * age,
                "path": str(path.resolve()),
                "result_sha256": sha256_file(path),
                "checkpoint_sha256": provenance.get("checkpoint_sha256"),
                "ritz_values": ritz,
                "gradient_ritz_coordinates": coords,
                "gradient_mass": mass if math.isfinite(mass) else None,
                "inverse_quadratic_form": (
                    inverse_form if math.isfinite(inverse_form) else None
                ),
                "lambda_eff": effective if math.isfinite(effective) else None,
                "checks_pass": not failures,
                "failures": failures,
            }
            records.append(record)
            if failures:
                invalid.append(record)
            else:
                curvatures[(arm, age)] = effective
    return curvatures, records, invalid


def slope_fit(x: Sequence[float], y: Sequence[float]) -> dict[str, float]:
    if len(x) != len(y) or len(x) < 2:
        raise GateFailure("slope fit has the wrong vector lengths")
    xbar = fmean(x)
    ybar = fmean(y)
    sxx = sum((value - xbar) ** 2 for value in x)
    if not sxx > 0.0:
        raise GateFailure("slope fit has no age variation")
    slope = sum((a - xbar) * (b - ybar) for a, b in zip(x, y)) / sxx
    intercept = ybar - slope * xbar
    residuals = [b - intercept - slope * a for a, b in zip(x, y)]
    sse = sum(value * value for value in residuals)
    syy = sum((value - ybar) ** 2 for value in y)
    return {
        "intercept": intercept,
        "slope": slope,
        "sse": sse,
        "r2": 1.0 - sse / syy if syy > 0.0 else 1.0,
        "n": len(x),
    }


def curvature_fit(values: dict[tuple[str, int], float]) -> dict[str, object]:
    required = {(arm, age) for arm in ARMS for age in AGES}
    if set(values) != required:
        raise GateFailure("cannot fit curvature without all eight frozen values")
    x = [math.log(age) for age in AGES]
    per_arm = {}
    arm_y = {}
    for arm in ARMS:
        y = [math.log(values[(arm, age)]) for age in AGES]
        fit = slope_fit(x, y)
        per_arm[arm] = {**fit, "gamma": -fit["slope"]}
        arm_y[arm] = y
    mean_y = [
        fmean((arm_y["mu0"][index], arm_y["corrected"][index]))
        for index in range(len(AGES))
    ]
    common = slope_fit(x, mean_y)
    gamma = -common["slope"]
    if not math.isfinite(gamma):
        raise GateFailure("common curvature gamma is nonfinite")
    return {
        "estimator": "unweighted OLS with arm fixed effects",
        "ages": list(AGES),
        "gamma": gamma,
        "common_slope": common,
        "arm_specific_diagnostics": per_arm,
        "n": 8,
    }


def schedule_audit(trajectory_root: Path) -> dict[str, object]:
    records = []
    total = 0
    for arm in ARMS:
        path = trajectory_root / arm / "tape.jsonl"
        require_regular(path, f"{arm} schedule tape")
        rows = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        if len(rows) != 80:
            raise GateFailure(f"{arm} schedule tape has {len(rows)} rows, expected 80")
        for expected_step, row in enumerate(rows, 1):
            step = int(row.get("step", -1))
            fragment = int(row.get("fragment", -1))
            if step != expected_step:
                raise GateFailure(
                    f"{arm} schedule step mismatch at row {expected_step}"
                )
            if fragment != (step - 1) % 4:
                raise GateFailure(f"{arm} fragment mismatch at step {step}")
            total += 1
        records.append(
            {
                "arm": arm,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "rows": 80,
            }
        )
    return {
        "status": "PASS",
        "rows_checked": total,
        "fragments": 4,
        "rule": "fragment=(step-1)%4; per-fragment age=floor((step-1)/4)+1",
        "records": records,
    }


def label(kappa: float) -> str:
    low, high = TARGET_INTERVAL
    if low <= kappa <= high:
        return "HIT"
    width = high - low
    distance = low - kappa if kappa < low else kappa - high
    return "NEAR" if distance <= 2.0 * width else "MISS"


def write_result(path: Path, result: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def format_note(result: dict[str, object]) -> str:
    loss = result.get("loss_floor", {})
    curvature = result.get("curvature", {})
    prediction = result.get("prediction", {})
    schedule = result.get("schedule", {})
    d_pred = prediction.get("D_pred", "NA")
    if isinstance(d_pred, list):
        d_pred = {str(row["T"]): row["D_pred"] for row in d_pred}
    return (
        "KAPPA ROUND3C: <"
        f"{result['status']}, ell_inf={loss.get('ell_inf', 'NA')}, "
        f"beta={loss.get('beta', 'NA')}, curvature={curvature.get('valid_probe_count', 0)}/8, "
        f"gamma={curvature.get('gamma', 'NA')}, alpha={prediction.get('alpha', 'NA')}, "
        f"D_pred={d_pred}, q_pred={prediction.get('q_pred', 'NA')}, "
        f"kappa_pred={prediction.get('kappa_pred', 'NA')}, "
        f"schedule={schedule.get('status', 'NA')}, free_choices=EMPTY>"
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--round2-analyzer", required=True, type=Path)
    parser.add_argument("--round2-result", required=True, type=Path)
    parser.add_argument("--loss-root", action="append", required=True, type=Path)
    parser.add_argument("--trajectory-root", required=True, type=Path)
    parser.add_argument("--spectrum-root", required=True, type=Path)
    parser.add_argument("--spectrum-adapter", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--eval-data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"refusing to overwrite --output: {args.output}")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    base = {
        "schema": "yeto_kappa_round3_identification_v1",
        "protocol": {
            "path": str(args.protocol.resolve()),
            "sha256": PROTOCOL_SHA256,
            "remaining_free_choices": [],
            "post_hoc_instrumentation_not_gated_science": True,
        },
        "target_interval": list(TARGET_INTERVAL),
    }
    partial: dict[str, object] = {}
    try:
        require_sha(args.protocol, PROTOCOL_SHA256, "Round-3 protocol")
        require_sha(
            args.spectrum_adapter, SPECTRUM_ADAPTER_SHA256, "Lane-E spectrum adapter"
        )
        require_sha(args.eval_data, EVAL_SHA256, "fixed evaluation data")
        model_files = verify_model(args.model_root)
        round2 = load_module(args.round2_analyzer)
        loss, loss_provenance = frozen_loss_fit(args, round2)
        partial["loss_floor"] = loss
        checkpoints, manifests = load_trajectories(args.trajectory_root)
        partial["trajectory"] = {
            "manifests": manifests,
            "checkpoint_count": len(checkpoints),
        }
        curvatures, spectrum_records, invalid = load_spectra(
            args.spectrum_root, checkpoints, args.model_root
        )
        curvature: dict[str, object] = {
            "adapter": str(args.spectrum_adapter.resolve()),
            "adapter_sha256": SPECTRUM_ADAPTER_SHA256,
            "records": spectrum_records,
            "valid_probe_count": len(curvatures),
            "invalid_probe_count": len(invalid),
            "invalid_records": invalid,
        }
        partial["curvature"] = curvature
        schedule = schedule_audit(args.trajectory_root)
        partial["schedule"] = schedule
        if invalid:
            raise GateFailure(
                f"{len(invalid)} of eight harmonic curvature inputs failed"
            )
        fitted_curvature = curvature_fit(curvatures)
        curvature.update(fitted_curvature)
        beta = float(loss["beta"])
        prediction = round2.predict(beta, float(fitted_curvature["gamma"]))
        partial["prediction"] = prediction
        result = {
            **base,
            **partial,
            "status": label(float(prediction["kappa_pred"])),
            "input_provenance": {
                "loss": loss_provenance,
                "model_files": model_files,
                "eval_data": str(args.eval_data.resolve()),
                "eval_sha256": EVAL_SHA256,
            },
            "adjudication": {
                "label": label(float(prediction["kappa_pred"])),
                "point_kappa": float(prediction["kappa_pred"]),
                "rule": "HIT in [0.9932,0.9938]; NEAR outside target within 0.0012; otherwise MISS",
            },
        }
    except Exception as exc:  # Every post-freeze input/tooling failure is VOID.
        result = {
            **base,
            **partial,
            "status": "VOID",
            "reason": str(exc),
            "adjudication": {
                "label": "VOID",
                "closed_numeric_vocabulary_not_entered": True,
            },
        }
    write_result(args.output, result)
    print(format_note(result))
    print(
        json.dumps(
            {"status": result["status"], "output": str(args.output)}, sort_keys=True
        )
    )
    return 0 if result["status"] != "VOID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
