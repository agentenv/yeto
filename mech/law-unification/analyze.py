#!/usr/bin/env python3
"""Reproducible banked-data audit of the finite-age outer-rate law.

This script is deliberately post-hoc and read-only with respect to campaign
data.  It consumes only the frozen source snapshots beside it, reconstructs
per-curve eta-star confidence intervals where a readout contains seed-level
cells, assembles the complete inclusion/exclusion ledgers, fits the registered
law, tests its requested sharing restrictions, and writes the paper artifacts.

Run from the repository root with NumPy, SciPy, and Matplotlib available.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


REPLICATES = 10_000
Z975 = 1.959963984540054
SEEDS_BY_SOURCE = {
    "G4C": 20260726,
    "G6": 20260727,
    "G8": 20260728,
    "G12": 20260741,
}
SCALE_PARAMETERS = {"135M": 135_000_000, "1.7B": 1_711_376_384, "7B": 7_615_616_512}
SCALE_ORDER = ("135M", "1.7B", "7B")
CONVENTION_ORDER = ("mu0", "nesterov_raw", "nesterov_corrected", "heavy_ball")
SOURCE_HASHES = {
    "disambiguation_source.md": "4be85f66125472fca267f284912442d5bf4ce6e25dbaae9dbca8f381e4af4835",
    "g12-readout.json": "0162dbdd2a78492c0b167e2dd516396ab7ea0d1bcb34ac0293abf6abdfc67ae5",
    "g4c-readout.json": "16bab85dce3a83f42c8b1c91be8d91eb1ec1372cfba572f2ffd9b491044b04aa",
    "g5b-readout.json": "4552330266cf1ab6b59c6e658117872732e6990c65f9ac7b73168db41ca1a446",
    "g6-readout.json": "7e7e9cd8901520cc8f0ca822151af92c4c6183ee99bf02199bc1452a2763af8c",
    "g8-readout.json": "9884f0775b30a35964e7df878bd0569b62f24af23619d90b4ff346d1afae596c",
    "scale_predictions.json": "6faab136f8d84347660e0891ed25b43a30a672dc774c56112b8092098402fbc6",
    "supplemental-1p7b-readout.json": "b572c017250414a96adaba472f75e4452933e26bc21e4ff9f207d032095b648a",
    "two_param_master_D.csv": "f1b132a5b4580a396da344a959f195c747d3b759d6db54243d303316eed77427",
    "two_param_master_eta.csv": "610f8e302d3a3ba7e888e268ad832d1be7e5015c0b93ab000208c57d0cf5d649",
}


class AuditError(RuntimeError):
    """A frozen-input or analysis invariant failed."""


@dataclass
class Fit:
    name: str
    groups: list[str]
    coefficients: np.ndarray
    covariance: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray
    rss: float
    df_resid: int
    rank: int
    n: int
    weighted: bool
    aic: float
    bic: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"{path}: expected a JSON object")
    return value


def load_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            return list(csv.DictReader(source))
    except OSError as exc:
        raise AuditError(f"cannot read {path}: {exc}") from exc


def finite(value: Any, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{context}: expected a number, got {value!r}") from exc
    if not math.isfinite(result):
        raise AuditError(f"{context}: expected a finite number")
    return result


def optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return finite(value, "optional value")


def convention_for(arm: str, mu: float) -> str:
    if abs(mu) < 1e-15:
        return "mu0"
    if arm in ("raw", "snoo-b"):
        return "nesterov_raw"
    if arm == "corrected":
        return "nesterov_corrected"
    if arm in ("heavy_ball", "mu09"):
        return "heavy_ball"
    raise AuditError(f"cannot map arm={arm!r}, mu={mu} to a convention")


def finite_age_multiplier(t: int, mu: float, convention: str) -> float:
    if abs(mu) < 1e-15 or convention in ("mu0", "nesterov_corrected"):
        return 1.0
    if convention == "nesterov_raw":
        return (1.0 - mu ** (t + 1)) / (1.0 - mu)
    if convention == "heavy_ball":
        return (1.0 - mu**t) / (1.0 - mu)
    raise AuditError(f"unknown convention {convention!r}")


def eta_factor(t: int, mu: float, convention: str) -> float:
    return (1.0 - mu) * finite_age_multiplier(t, mu, convention)


def log_ci_se(low: float, high: float) -> float:
    if low <= 0.0 or high <= low:
        return math.nan
    return (math.log(high) - math.log(low)) / (2.0 * Z975)


def point_id(campaign: str, scale: str, t: int, mu: float, arm: str, h: int | None, s: int | None) -> str:
    fields = [campaign, scale, f"T{t}", f"mu{mu:g}", arm]
    if h is not None:
        fields.append(f"H{h}")
    if s is not None:
        fields.append(f"S{s}")
    return ":".join(fields).replace(" ", "_")


def make_row(
    *,
    campaign: str,
    source_family: str,
    scale: str,
    t: int,
    mu: float,
    arm: str,
    eta_star: float,
    ci_low: float,
    ci_high: float,
    h: int | None,
    s: int | None,
    status: str,
    n_seeds: int,
    bootstrap_replicates: int,
    bootstrap_invalid: int,
    exploratory: bool,
    dependence_group: str,
    source_file: str,
    ci_method: str,
    note: str = "",
) -> dict[str, Any]:
    convention = convention_for(arm, mu)
    multiplier = finite_age_multiplier(t, mu, convention)
    factor = (1.0 - mu) * multiplier
    se = log_ci_se(ci_low, ci_high)
    return {
        "point_id": point_id(campaign, scale, t, mu, arm, h, s),
        "campaign": campaign,
        "source_family": source_family,
        "scale": scale,
        "parameters": SCALE_PARAMETERS[scale],
        "T": int(t),
        "H": "" if h is None else int(h),
        "S": "" if s is None else int(s),
        "mu": float(mu),
        "arm": arm,
        "convention": convention,
        "M": multiplier,
        "law_factor_1_minus_mu_times_M": factor,
        "eta_star": eta_star,
        "eta_ci_low": ci_low,
        "eta_ci_high": ci_high,
        "se_log_eta": se if math.isfinite(se) else "",
        "fit_status": status,
        "n_seeds": n_seeds,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_invalid": bootstrap_invalid,
        "noise_eligible": bool(math.isfinite(se) and se > 0.0 and bootstrap_invalid == 0),
        "exploratory": bool(exploratory),
        "dependence_group": dependence_group,
        "source_file": source_file,
        "source_sha256": SOURCE_HASHES[source_file],
        "ci_method": ci_method,
        "note": note,
    }


def quadratic_projector(etas: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    xs = np.log2(np.asarray(etas, dtype=float))
    design = np.column_stack((xs * xs, xs, np.ones_like(xs)))
    if np.linalg.matrix_rank(design) != 3:
        raise AuditError(f"singular eta grid {etas}")
    return xs, np.linalg.pinv(design)


def python_draws(seed: int, sample_size: int) -> np.ndarray:
    rng = random.Random(seed)
    return np.asarray(
        [[rng.randrange(sample_size) for _ in range(sample_size)] for _ in range(REPLICATES)],
        dtype=np.int16,
    )


def bootstrap_eta_intervals(
    records: Sequence[dict[str, Any]],
    group_key: Callable[[dict[str, Any]], tuple[Any, ...]],
    *,
    rng_seed: int,
    acceptance: str = "strict",
) -> dict[tuple[Any, ...], dict[str, Any]]:
    """Paired seed bootstrap of every eta-curve in one sealed readout."""

    groups: defaultdict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[group_key(record)].append(record)
    all_seeds = sorted({int(record["seed"]) for record in records})
    draws = python_draws(rng_seed, len(all_seeds))
    output: dict[tuple[Any, ...], dict[str, Any]] = {}
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        etas = sorted({float(record["eta"]) for record in group})
        seeds = sorted({int(record["seed"]) for record in group})
        if seeds != all_seeds:
            raise AuditError(f"{key}: seed set differs from paired campaign seed set")
        by_cell: dict[tuple[int, float], float] = {}
        for record in group:
            cell_key = (int(record["seed"]), float(record["eta"]))
            if cell_key in by_cell:
                raise AuditError(f"{key}: duplicate seed/eta cell {cell_key}")
            by_cell[cell_key] = finite(record["eval_loss"], f"{key} loss")
        expected = {(seed, eta) for seed in seeds for eta in etas}
        if set(by_cell) != expected:
            raise AuditError(f"{key}: incomplete eta-by-seed grid")
        losses = np.asarray([[by_cell[(seed, eta)] for eta in etas] for seed in seeds])
        xs, projector = quadratic_projector(etas)
        point_coeff = projector @ losses.mean(axis=0)
        point_vertex = -point_coeff[1] / (2.0 * point_coeff[0])
        point_eta = 2.0**point_vertex
        sampled_means = losses[draws].mean(axis=1)
        coeff = sampled_means @ projector.T
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            vertices = -coeff[:, 1] / (2.0 * coeff[:, 0])
            samples = np.exp2(vertices)
        accepted = np.isfinite(samples) & (coeff[:, 0] > 0.0)
        low_x, high_x = float(xs.min()), float(xs.max())
        if acceptance == "strict":
            accepted &= (vertices > low_x + 1e-12) & (vertices < high_x - 1e-12)
        elif acceptance == "near_0p5":
            accepted &= (vertices > low_x - 0.5) & (vertices < high_x + 0.5)
        else:
            raise AuditError(f"unknown acceptance rule {acceptance}")
        valid = samples[accepted]
        if not len(valid):
            raise AuditError(f"{key}: no valid bootstrap eta refits")
        ci_low, ci_high = np.quantile(valid, (0.025, 0.975))
        output[key] = {
            "eta_star_recomputed": float(point_eta),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "valid": int(accepted.sum()),
            "invalid": int(REPLICATES - accepted.sum()),
        }
    return output


def assemble_two_parameter(source_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = source_dir / "two_param_master_eta.csv"
    rows = load_csv(path)
    if len(rows) != 50:
        raise AuditError(f"two-parameter master: expected 50 rows, got {len(rows)}")
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for source in rows:
        campaign_raw = source["campaign"]
        campaign = f"TP-{campaign_raw}"
        arm = source["arm"]
        mu = finite(source["mu"], "two-parameter mu")
        t, h, s = int(source["T"]), int(source["H"]), int(source["S"])
        eta = optional_float(source["eta_star"])
        ci_low = optional_float(source["eta_ci_low"])
        ci_high = optional_float(source["eta_ci_high"])
        eligible = (
            source["interior"] == "True"
            and source["fit_status"] == "INTERIOR"
            and eta is not None
            and ci_low is not None
            and ci_high is not None
        )
        if not eligible:
            excluded.append(
                {
                    "exclusion_id": point_id(campaign, "135M", t, mu, arm, h, s),
                    "campaign": campaign,
                    "scale": "135M",
                    "T": t,
                    "H": h,
                    "S": s,
                    "mu": mu,
                    "arm": arm,
                    "eta_star": "" if eta is None else eta,
                    "eta_ci_low": "" if ci_low is None else ci_low,
                    "eta_ci_high": "" if ci_high is None else ci_high,
                    "fit_status": source["fit_status"],
                    "reason": (
                        "canonical master fit is not an accepted interior optimum; "
                        f"status={source['fit_status']}"
                    ),
                    "source_file": path.name,
                    "source_sha256": SOURCE_HASHES[path.name],
                }
            )
            continue
        if arm.startswith("snoo-"):
            raise AuditError("an interior v3 SNOO row would conflict with the final G5B estimator")
        dependence = f"{campaign_raw}:T{t}:H{h}:S{s}:mu{mu:g}"
        note = ""
        if campaign_raw == "v3" and abs(mu) < 1e-15:
            dependence = f"v3-shared-mu0:T{t}:H{h}:S{s}"
            note = "A/A-equivalent mu=0 control appears under raw and corrected labels."
        included.append(
            make_row(
                campaign=campaign,
                source_family="two_parameter_master",
                scale="135M",
                t=t,
                mu=mu,
                arm=arm,
                eta_star=float(eta),
                ci_low=float(ci_low),
                ci_high=float(ci_high),
                h=h,
                s=s,
                status=source["fit_status"],
                n_seeds=int(source["n_seeds"]),
                bootstrap_replicates=REPLICATES,
                bootstrap_invalid=int(source["bootstrap_invalid"]),
                exploratory=campaign_raw == "pilot",
                dependence_group=dependence,
                source_file=path.name,
                ci_method="source-reported paired training-seed bootstrap, equal-tailed 95%",
                note=note,
            )
        )
    if len(included) != 38 or len(excluded) != 12:
        raise AuditError(
            f"two-parameter status accounting changed: included={len(included)}, excluded={len(excluded)}"
        )
    return included, excluded


DISAMBIGUATION_RE = re.compile(
    r"^CURVE H(?P<H>\d+)_S(?P<S>\d+)_(?P<arm>mu0|corr|raw): "
    r"eta\*=(?P<eta>[0-9.eE+-]+).*status=(?P<status>\w+) "
    r"eta95=\[(?P<low>[0-9.eE+-]+),(?P<high>[0-9.eE+-]+)\] .*"
    r"bootstrap_valid=(?P<valid>\d+)/(?P<total>\d+)$"
)


def assemble_disambiguation(source_dir: Path) -> list[dict[str, Any]]:
    path = source_dir / "disambiguation_source.md"
    matches = [
        match
        for line in path.read_text(encoding="utf-8").splitlines()
        if (match := DISAMBIGUATION_RE.match(line))
    ]
    if len(matches) != 6:
        raise AuditError(f"disambiguation: expected 6 curve lines, got {len(matches)}")
    output = []
    for match in matches:
        data = match.groupdict()
        arm_token = data["arm"]
        arm = {"mu0": "mu0", "corr": "corrected", "raw": "raw"}[arm_token]
        mu = 0.0 if arm_token == "mu0" else 0.9
        h, s = int(data["H"]), int(data["S"])
        if s // h != 5 or s % h:
            raise AuditError("disambiguation curve does not have T=5")
        output.append(
            make_row(
                campaign="DISAMBIG",
                source_family="fixed_T5_disambiguation",
                scale="135M",
                t=5,
                mu=mu,
                arm=arm,
                eta_star=float(data["eta"]),
                ci_low=float(data["low"]),
                ci_high=float(data["high"]),
                h=h,
                s=s,
                status=data["status"],
                n_seeds=3,
                bootstrap_replicates=int(data["total"]),
                bootstrap_invalid=int(data["total"]) - int(data["valid"]),
                exploratory=True,
                dependence_group=f"disambig:H{h}:S{s}:{arm}",
                source_file=path.name,
                ci_method="source-reported paired training-seed bootstrap, equal-tailed 95%",
                note="Exploratory fixed-T=5 outer-horizon disambiguation lane.",
            )
        )
    return output


def assemble_seed_readouts(source_dir: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    path = source_dir / "g4c-readout.json"
    g4c = load_json(path)
    if g4c.get("schema") != "yeto_outer_mup_v4c_g4c_readout_v2" or len(g4c["curve_fits"]) != 4:
        raise AuditError("G4C canonical readout schema/count mismatch")
    g4c_ci = bootstrap_eta_intervals(
        g4c["cell_records"], lambda r: (int(r["t"]), int(r["s"]), float(r["mu"])),
        rng_seed=SEEDS_BY_SOURCE["G4C"], acceptance="near_0p5",
    )
    for fit in g4c["curve_fits"]:
        t, s, mu = int(fit["t"]), int(fit["s"]), float(fit["mu"])
        ci = g4c_ci[(t, s, mu)]
        if not math.isclose(ci["eta_star_recomputed"], float(fit["eta_star"]), rel_tol=1e-10):
            raise AuditError(f"G4C point refit differs at T={t}, mu={mu}")
        arm = "mu0" if mu == 0.0 else "raw"
        output.append(
            make_row(
                campaign="G4C",
                source_family="v4_family_final_combined",
                scale="1.7B", t=t, mu=mu, arm=arm,
                eta_star=float(fit["eta_star"]), ci_low=ci["ci_low"], ci_high=ci["ci_high"],
                h=s // t, s=s, status=str(fit["status"]), n_seeds=5,
                bootstrap_replicates=REPLICATES, bootstrap_invalid=ci["invalid"],
                exploratory=False, dependence_group=f"G4C:T{t}:mu{mu:g}",
                source_file=path.name,
                ci_method="recomputed paired 5-seed bootstrap with canonical 0.5-bit near-bracket rule",
                note="Final G4/G4B/G4C combined-grid estimator; earlier cumulative fits are superseded.",
            )
        )

    path = source_dir / "g6-readout.json"
    g6 = load_json(path)
    if g6.get("schema") != "yeto_outer_mup_v6_g6_readout_v1" or len(g6["curve_fits"]) != 36:
        raise AuditError("G6 canonical readout schema/count mismatch")
    g6_ci = bootstrap_eta_intervals(
        g6["cell_records"],
        lambda r: (int(r["t"]), int(r["s"]), int(r["h"]), str(r["arm"])),
        rng_seed=SEEDS_BY_SOURCE["G6"], acceptance="strict",
    )
    for fit in g6["curve_fits"]:
        t, s, h, arm = int(fit["t"]), int(fit["s"]), int(fit["h"]), str(fit["arm"])
        ci = g6_ci[(t, s, h, arm)]
        if not math.isclose(ci["eta_star_recomputed"], float(fit["eta_star"]), rel_tol=2e-10):
            raise AuditError(f"G6 point refit differs at T={t}, S={s}, arm={arm}")
        mu = 0.0 if arm == "mu0" else 0.9
        output.append(
            make_row(
                campaign="G6", source_family="factorial_T_by_S", scale="135M",
                t=t, mu=mu, arm=arm, eta_star=float(fit["eta_star"]),
                ci_low=ci["ci_low"], ci_high=ci["ci_high"], h=h, s=s,
                status=str(fit["status"]), n_seeds=3, bootstrap_replicates=REPLICATES,
                bootstrap_invalid=ci["invalid"], exploratory=False,
                dependence_group=f"G6:T{t}:S{s}:{arm}", source_file=path.name,
                ci_method="recomputed paired 3-seed strict-interior bootstrap, equal-tailed 95%",
            )
        )

    path = source_dir / "g8-readout.json"
    g8 = load_json(path)
    if g8.get("schema") != "yeto_outer_mup_v8_phase_diagram_readout_v1" or len(g8["curve_fits"]) != 15:
        raise AuditError("G8 canonical readout schema/count mismatch")
    g8_ci = bootstrap_eta_intervals(
        g8["cell_records"],
        lambda r: (int(r["T"]), int(r["S"]), int(r["H"]), str(r["arm"]), float(r["mu"])),
        rng_seed=SEEDS_BY_SOURCE["G8"], acceptance="strict",
    )
    for fit in g8["curve_fits"]:
        t, s, h = int(fit["T"]), int(fit["S"]), int(fit["H"])
        arm, mu = str(fit["arm"]), float(fit["mu"])
        ci = g8_ci[(t, s, h, arm, mu)]
        if not math.isclose(ci["eta_star_recomputed"], float(fit["eta_star"]), rel_tol=2e-10):
            raise AuditError(f"G8 point refit differs at T={t}, mu={mu}, arm={arm}")
        output.append(
            make_row(
                campaign="G8", source_family="phase_mu_scan", scale="135M",
                t=t, mu=mu, arm=arm, eta_star=float(fit["eta_star"]),
                ci_low=ci["ci_low"], ci_high=ci["ci_high"], h=h, s=s,
                status=str(fit["status"]), n_seeds=3, bootstrap_replicates=REPLICATES,
                bootstrap_invalid=ci["invalid"], exploratory=False,
                dependence_group=f"G8:T{t}:mu{mu:g}:{arm}", source_file=path.name,
                ci_method="recomputed paired 3-seed strict-interior bootstrap, equal-tailed 95%",
            )
        )

    path = source_dir / "g12-readout.json"
    g12 = load_json(path)
    if len(g12.get("fits", {})) != 6 or g12.get("gate", {}).get("verdict") != "PASS":
        raise AuditError("G12 canonical heavy-ball readout mismatch")
    g12_ci = bootstrap_eta_intervals(
        g12["cell_records"], lambda r: (int(r["t"]), str(r["arm"])),
        rng_seed=SEEDS_BY_SOURCE["G12"], acceptance="strict",
    )
    for fit_key, fit in sorted(g12["fits"].items()):
        match = re.fullmatch(r"T(\d+)_mu(0|09)", fit_key)
        if not match:
            raise AuditError(f"G12 unexpected fit key {fit_key}")
        t = int(match.group(1))
        source_arm = "mu0" if match.group(2) == "0" else "mu09"
        mu, arm = (0.0, "mu0") if source_arm == "mu0" else (0.9, "heavy_ball")
        ci = g12_ci[(t, source_arm)]
        if not math.isclose(ci["eta_star_recomputed"], float(fit["eta_star"]), rel_tol=2e-10):
            raise AuditError(f"G12 point refit differs at T={t}, arm={source_arm}")
        s = 2560
        output.append(
            make_row(
                campaign="G12", source_family="heavy_ball_scan", scale="135M",
                t=t, mu=mu, arm=arm, eta_star=float(fit["eta_star"]),
                ci_low=ci["ci_low"], ci_high=ci["ci_high"], h=s // t, s=s,
                status=str(fit["status"]), n_seeds=3, bootstrap_replicates=REPLICATES,
                bootstrap_invalid=ci["invalid"], exploratory=False,
                dependence_group=f"G12:T{t}:{arm}", source_file=path.name,
                ci_method="recomputed paired 3-seed strict-interior bootstrap, equal-tailed 95%",
                note="Heavy-ball terminal multiplier uses geometric prefix through T-1.",
            )
        )
    return output


def assemble_snoo_and_scale(source_dir: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    path = source_dir / "g5b-readout.json"
    g5b = load_json(path)
    curves = g5b.get("combined_eta_curves", {})
    intervals = g5b.get("paired_bootstrap", {}).get("eta_star_ci_95", {})
    if set(curves) != {"a", "b", "c"} or set(intervals) != {"a", "b", "c"}:
        raise AuditError("G5B final combined SNOO curve/interval set mismatch")
    for key in ("a", "b", "c"):
        fit, ci = curves[key], intervals[key]
        mu = 0.9 if key == "b" else 0.0
        arm = f"snoo-{key}"
        output.append(
            make_row(
                campaign="G5B", source_family="SNOO_final_combined", scale="135M",
                t=5, mu=mu, arm=arm, eta_star=float(fit["eta_star"]),
                ci_low=float(ci["low"]), ci_high=float(ci["high"]), h=512, s=2560,
                status=str(fit["status"]), n_seeds=5, bootstrap_replicates=REPLICATES,
                bootstrap_invalid=int(g5b["paired_bootstrap"]["invalid_unbracketed_replicates"]),
                exploratory=False,
                dependence_group=("G5B:shared-mu0:a-c" if key in ("a", "c") else "G5B:b"),
                source_file=path.name,
                ci_method="source-reported final combined paired 5-seed bootstrap, equal-tailed 95%",
                note="Final combined G5B estimator supersedes earlier SNOO analyses.",
            )
        )

    path = source_dir / "supplemental-1p7b-readout.json"
    g9a = load_json(path)
    fits = g9a.get("stage_1p7b_curve_fits", {})
    targets = g9a.get("g9a_bootstrap", {}).get("targets", {})
    if set(fits) != {"raw", "corrected"} or set(targets) != {"raw", "corrected"}:
        raise AuditError("G9A supplemental 1.7B curve/interval set mismatch")
    for arm in ("raw", "corrected"):
        fit, target = fits[arm], targets[arm]["eta_star_ci_95"]
        output.append(
            make_row(
                campaign="G9A", source_family="prospective_1p7B_grid", scale="1.7B",
                t=10, mu=0.9, arm=arm, eta_star=float(fit["eta_star"]),
                ci_low=float(target["low"]), ci_high=float(target["high"]), h=512, s=5120,
                status=str(fit["status"]), n_seeds=2, bootstrap_replicates=REPLICATES,
                bootstrap_invalid=int(g9a["g9a_bootstrap"]["invalid_replicates"]),
                exploratory=False, dependence_group=f"G9A:{arm}", source_file=path.name,
                ci_method="source-reported paired 2-seed bootstrap, equal-tailed 95%",
                note="Prospective 1.7B verification component.",
            )
        )

    path = source_dir / "scale_predictions.json"
    scale = load_json(path)
    slots = scale.get("slots", [])
    slot_7b = [slot for slot in slots if int(slot.get("actual_parameters", 0)) == SCALE_PARAMETERS["7B"]]
    if len(slot_7b) != 1:
        raise AuditError("final scale panel does not contain exactly one 7B slot")
    values = slot_7b[0].get("values", {})
    expected = {"mu0_T5", "raw_T5"}
    if set(values) != expected:
        raise AuditError(f"final 7B value set changed: {set(values)}")
    for key in ("mu0_T5", "raw_T5"):
        value = values[key]
        arm = "mu0" if key == "mu0_T5" else "raw"
        mu = 0.0 if arm == "mu0" else 0.9
        eta = float(value["observed_eta_star"])
        output.append(
            make_row(
                campaign="G9B", source_family="final_7B_relative_grid", scale="7B",
                t=5, mu=mu, arm=arm, eta_star=eta,
                ci_low=eta, ci_high=eta, h=512, s=2560,
                status="INTERIOR_SINGLETON_SEED", n_seeds=1, bootstrap_replicates=REPLICATES,
                bootstrap_invalid=0, exploratory=False,
                dependence_group=f"G9B:{arm}", source_file=path.name,
                ci_method="source-reported empirical singleton-seed point mass; not noise-calibrating",
                note=(
                    "One retained seed (907); included in equal-point fit/figure but excluded from "
                    "inverse-variance and seed-noise tests. Raw joint readout unavailable in evacuation "
                    "archive; this derived artifact binds upstream SHA-256 "
                    f"{scale['provenance']['g9_readout_sha256']}."
                ),
            )
        )
    return output


def append_superseded_and_scope_exclusions(excluded: list[dict[str, Any]]) -> None:
    blank = {"eta_star": "", "eta_ci_low": "", "eta_ci_high": ""}
    for stage in ("G4", "G4B"):
        for t, s in ((5, 2560), (20, 10240)):
            for mu, arm in ((0.0, "mu0"), (0.9, "raw")):
                excluded.append(
                    {
                        "exclusion_id": f"{stage}:T{t}:mu{mu:g}",
                        "campaign": stage, "scale": "1.7B", "T": t,
                        "H": s // t, "S": s, "mu": mu, "arm": arm,
                        **blank, "fit_status": "SUPERSEDED_CUMULATIVE_ESTIMATOR",
                        "reason": "same scientific cells/curve as the final five-seed G4C combined estimator; excluded to prevent cumulative double counting",
                        "source_file": "g4c-readout.json",
                        "source_sha256": SOURCE_HASHES["g4c-readout.json"],
                    }
                )
    for key, mu in (("a", 0.0), ("b", 0.9), ("c", 0.0)):
        excluded.append(
            {
                "exclusion_id": f"G5:{key}", "campaign": "G5", "scale": "135M",
                "T": 5, "H": 512, "S": 2560, "mu": mu, "arm": f"snoo-{key}",
                **blank, "fit_status": "SUPERSEDED_CUMULATIVE_ESTIMATOR",
                "reason": "earlier SNOO estimator is superseded by the final combined G5B fit; excluded to prevent cumulative double counting",
                "source_file": "g5b-readout.json", "source_sha256": SOURCE_HASHES["g5b-readout.json"],
            }
        )
    excluded.append(
        {
            "exclusion_id": "G13-G13B:external-transfer", "campaign": "G13/G13B",
            "scale": "OTHER", "T": "", "H": "", "S": "", "mu": "", "arm": "",
            **blank, "fit_status": "OUT_OF_SCOPE_EXTERNAL_TRANSFER",
            "reason": "different model family and corpus; not a point on the requested SmolLM/Qwen scale axis",
            "source_file": "", "source_sha256": "",
        }
    )


def assemble_all(source_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    included, excluded = assemble_two_parameter(source_dir)
    included.extend(assemble_disambiguation(source_dir))
    included.extend(assemble_seed_readouts(source_dir))
    included.extend(assemble_snoo_and_scale(source_dir))
    append_superseded_and_scope_exclusions(excluded)
    if len(included) != 112:
        counts = Counter(row["campaign"] for row in included)
        raise AuditError(f"expected 112 included optima, got {len(included)}: {dict(counts)}")
    ids = [row["point_id"] for row in included]
    if len(ids) != len(set(ids)):
        duplicates = [item for item, count in Counter(ids).items() if count > 1]
        raise AuditError(f"duplicate included point IDs: {duplicates}")
    included.sort(
        key=lambda row: (
            SCALE_ORDER.index(row["scale"]), row["campaign"], int(row["T"]),
            float(row["mu"]), row["convention"], str(row["S"]), row["arm"],
        )
    )
    excluded.sort(key=lambda row: (str(row["campaign"]), str(row["T"]), str(row["mu"]), str(row["arm"])))
    return included, excluded


def response_vector(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [math.log(float(row["eta_star"])) - math.log(float(row["law_factor_1_minus_mu_times_M"])) for row in rows],
        dtype=float,
    )


def design_matrix(
    rows: Sequence[dict[str, Any]],
    *,
    slope_by: str = "shared",
) -> tuple[np.ndarray, list[str]]:
    scales = [scale for scale in SCALE_ORDER if any(row["scale"] == scale for row in rows)]
    columns: list[np.ndarray] = []
    labels: list[str] = []
    for scale in scales:
        columns.append(np.asarray([1.0 if row["scale"] == scale else 0.0 for row in rows]))
        labels.append(f"log_eta0[{scale}]")
    if slope_by == "shared":
        categories: list[Any] = ["shared"]
        getter = lambda row: "shared"
    elif slope_by == "mu":
        categories = sorted({float(row["mu"]) for row in rows})
        getter = lambda row: float(row["mu"])
    elif slope_by == "scale":
        categories = scales
        getter = lambda row: row["scale"]
    elif slope_by == "convention":
        categories = [name for name in CONVENTION_ORDER if any(row["convention"] == name for row in rows)]
        getter = lambda row: row["convention"]
    else:
        raise AuditError(f"unknown slope-sharing dimension {slope_by}")
    for category in categories:
        columns.append(
            np.asarray([float(row["T"]) if getter(row) == category else 0.0 for row in rows])
        )
        label_value = f"{category:g}" if isinstance(category, float) else str(category)
        labels.append(f"log_q[{label_value}]")
    matrix = np.column_stack(columns)
    rank = int(np.linalg.matrix_rank(matrix))
    if rank != matrix.shape[1]:
        raise AuditError(
            f"rank-deficient {slope_by} design: rank={rank}, columns={matrix.shape[1]}, labels={labels}"
        )
    return matrix, labels


def fit_linear(
    name: str,
    rows: Sequence[dict[str, Any]],
    *,
    slope_by: str = "shared",
    weighted: bool = False,
) -> Fit:
    y = response_vector(rows)
    x, labels = design_matrix(rows, slope_by=slope_by)
    if weighted:
        se = np.asarray([float(row["se_log_eta"]) for row in rows])
        if not np.all(np.isfinite(se) & (se > 0.0)):
            raise AuditError(f"{name}: weighted fit received unusable standard errors")
        xw, yw = x / se[:, None], y / se
        beta, _, rank, _ = np.linalg.lstsq(xw, yw, rcond=None)
        fitted = x @ beta
        residual = y - fitted
        rss = float(np.sum((residual / se) ** 2))
        covariance = np.linalg.pinv(xw.T @ xw)
        information_constant = float(np.sum(np.log(2.0 * math.pi * se * se)))
        aic = rss + information_constant + 2.0 * len(beta)
        bic = rss + information_constant + math.log(len(rows)) * len(beta)
    else:
        beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
        fitted = x @ beta
        residual = y - fitted
        rss = float(residual @ residual)
        sigma2 = rss / (len(rows) - int(rank))
        covariance = sigma2 * np.linalg.pinv(x.T @ x)
        safe_rss = max(rss, np.finfo(float).tiny)
        likelihood_term = len(rows) * (math.log(2.0 * math.pi) + 1.0 + math.log(safe_rss / len(rows)))
        aic = likelihood_term + 2.0 * len(beta)
        bic = likelihood_term + math.log(len(rows)) * len(beta)
    return Fit(
        name=name, groups=labels, coefficients=beta, covariance=covariance,
        fitted=fitted, residuals=residual, rss=rss, df_resid=len(rows) - int(rank),
        rank=int(rank), n=len(rows), weighted=weighted, aic=float(aic), bic=float(bic),
    )


def coefficient_interval(fit: Fit, index: int) -> tuple[float, float]:
    estimate = float(fit.coefficients[index])
    se = math.sqrt(max(0.0, float(fit.covariance[index, index])))
    critical = Z975 if fit.weighted else float(stats.t.ppf(0.975, fit.df_resid))
    return estimate - critical * se, estimate + critical * se


def serialize_fit(fit: Fit) -> dict[str, Any]:
    coefficients = []
    for index, (label, estimate) in enumerate(zip(fit.groups, fit.coefficients)):
        low, high = coefficient_interval(fit, index)
        record: dict[str, Any] = {
            "name": label, "estimate": float(estimate), "se": math.sqrt(max(0.0, float(fit.covariance[index, index]))),
            "ci_95": [float(low), float(high)],
        }
        if label.startswith("log_q["):
            record.update(
                {"q": math.exp(float(estimate)), "q_ci_95": [math.exp(low), math.exp(high)]}
            )
        elif label.startswith("log_eta0["):
            record.update(
                {"eta0": math.exp(float(estimate)), "eta0_ci_95": [math.exp(low), math.exp(high)]}
            )
        coefficients.append(record)
    return {
        "name": fit.name, "weighted": fit.weighted, "n": fit.n, "rank": fit.rank,
        "df_resid": fit.df_resid, "rss_or_chi2": fit.rss, "aic": fit.aic, "bic": fit.bic,
        "coefficients": coefficients,
    }


def nested_test(restricted: Fit, full: Fit) -> dict[str, Any]:
    if restricted.n != full.n or restricted.weighted != full.weighted:
        raise AuditError("nested model comparison inputs differ")
    df = full.rank - restricted.rank
    if df <= 0:
        raise AuditError("nested full model did not add identifiable coefficients")
    improvement = max(0.0, restricted.rss - full.rss)
    if full.weighted:
        statistic = improvement
        p_value = float(stats.chi2.sf(statistic, df))
        test = "delta_chi_square"
    else:
        statistic = (improvement / df) / (full.rss / full.df_resid)
        p_value = float(stats.f.sf(statistic, df, full.df_resid))
        test = "partial_F"
    return {
        "test": test, "statistic": float(statistic), "df_num": int(df),
        "df_den": None if full.weighted else int(full.df_resid), "p_value": p_value,
        "restricted_rss_or_chi2": restricted.rss, "full_rss_or_chi2": full.rss,
        "survives_p_ge_0p05": bool(p_value >= 0.05),
    }


def eligible_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if bool(row["noise_eligible"])]


def sharing_audit(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    noise = eligible_rows(rows)
    tests: dict[str, Any] = {}
    for dimension in ("mu", "convention"):
        restricted_ols = fit_linear(f"OLS shared q for {dimension} test", rows)
        full_ols = fit_linear(f"OLS q by {dimension}", rows, slope_by=dimension)
        restricted_wls = fit_linear(f"WLS shared q for {dimension} test", noise, weighted=True)
        full_wls = fit_linear(f"WLS q by {dimension}", noise, slope_by=dimension, weighted=True)
        tests[dimension] = {
            "scope": "all scales and all eligible points",
            "ols": nested_test(restricted_ols, full_ols),
            "wls": nested_test(restricted_wls, full_wls),
            "full_ols": serialize_fit(full_ols),
            "full_wls": serialize_fit(full_wls),
        }
    scale_rows = [row for row in rows if row["scale"] in ("135M", "1.7B")]
    scale_noise = eligible_rows(scale_rows)
    restricted_ols = fit_linear("OLS shared q for scale test", scale_rows)
    full_ols = fit_linear("OLS q by scale", scale_rows, slope_by="scale")
    restricted_wls = fit_linear("WLS shared q for scale test", scale_noise, weighted=True)
    full_wls = fit_linear("WLS q by scale", scale_noise, slope_by="scale", weighted=True)
    tests["scale"] = {
        "scope": "135M and 1.7B only; the two 7B points share T=5, so a 7B slope is not identifiable",
        "ols": nested_test(restricted_ols, full_ols),
        "wls": nested_test(restricted_wls, full_wls),
        "full_ols": serialize_fit(full_ols),
        "full_wls": serialize_fit(full_wls),
    }
    for value in tests.values():
        value["survives_both_ols_and_wls"] = bool(
            value["ols"]["survives_p_ge_0p05"] and value["wls"]["survives_p_ge_0p05"]
        )
    return tests


def goodness(rows: Sequence[dict[str, Any]], fit: Fit) -> dict[str, Any]:
    if not fit.weighted or fit.n != len(rows):
        raise AuditError("goodness requires the matching WLS fit")
    se = np.asarray([float(row["se_log_eta"]) for row in rows])
    standardized = fit.residuals / se
    response_low = np.asarray(
        [math.log(float(row["eta_ci_low"])) - math.log(float(row["law_factor_1_minus_mu_times_M"])) for row in rows]
    )
    response_high = np.asarray(
        [math.log(float(row["eta_ci_high"])) - math.log(float(row["law_factor_1_minus_mu_times_M"])) for row in rows]
    )
    covered = (fit.fitted >= response_low) & (fit.fitted <= response_high)
    p_value = float(stats.chi2.sf(fit.rss, fit.df_resid))
    return {
        "n_noise_eligible": len(rows), "chi2": fit.rss, "df": fit.df_resid,
        "heterogeneity_p": p_value, "coverage_count": int(covered.sum()),
        "coverage_fraction": float(covered.mean()), "max_abs_standardized_residual": float(np.max(np.abs(standardized))),
        "median_abs_standardized_residual": float(np.median(np.abs(standardized))),
        "passes_noise_p_ge_0p05": bool(p_value >= 0.05),
        "passes_coverage_ge_0p90": bool(covered.mean() >= 0.90),
    }


def coefficient_by_name(fit: Fit, label: str) -> tuple[float, float, float]:
    index = fit.groups.index(label)
    low, high = coefficient_interval(fit, index)
    return float(fit.coefficients[index]), float(low), float(high)


def shared_q_record(fit: Fit) -> dict[str, float]:
    estimate, low, high = coefficient_by_name(fit, "log_q[shared]")
    return {
        "log_q": estimate, "log_q_ci_low": low, "log_q_ci_high": high,
        "q": math.exp(estimate), "q_ci_low": math.exp(low), "q_ci_high": math.exp(high),
    }


def annotate_predictions(rows: list[dict[str, Any]], primary: Fit) -> None:
    if primary.n != len(rows) or primary.weighted:
        raise AuditError("prediction annotation requires the all-point primary OLS fit")
    eta0 = {
        scale: math.exp(coefficient_by_name(primary, f"log_eta0[{scale}]")[0])
        for scale in SCALE_ORDER
    }
    q = shared_q_record(primary)["q"]
    for row, fitted, residual in zip(rows, primary.fitted, primary.residuals):
        factor = float(row["law_factor_1_minus_mu_times_M"])
        eta_pred = math.exp(float(fitted)) * factor
        scale_eta0 = eta0[row["scale"]]
        normalized = float(row["eta_star"]) / (scale_eta0 * factor)
        normalized_low = float(row["eta_ci_low"]) / (scale_eta0 * factor)
        normalized_high = float(row["eta_ci_high"]) / (scale_eta0 * factor)
        row.update(
            {
                "eta0_hat_primary": scale_eta0,
                "q_hat_primary": q,
                "eta_pred_primary": eta_pred,
                "normalized_rate": normalized,
                "normalized_ci_low": normalized_low,
                "normalized_ci_high": normalized_high,
                "fitted_curve_q_to_T": q ** int(row["T"]),
                "residual_log": float(residual),
                "residual_bits": float(residual / math.log(2.0)),
                "residual_ci_low_bits": math.log2(float(row["eta_ci_low"]) / eta_pred),
                "residual_ci_high_bits": math.log2(float(row["eta_ci_high"]) / eta_pred),
                "ci_covers_primary_prediction": bool(
                    float(row["eta_ci_low"]) <= eta_pred <= float(row["eta_ci_high"])
                ),
                "standardized_residual_primary": (
                    float(residual / float(row["se_log_eta"])) if row["noise_eligible"] else ""
                ),
            }
        )


def summary_record(kind: str, value: str, subset: Sequence[dict[str, Any]]) -> dict[str, Any]:
    residuals = np.asarray([float(row["residual_bits"]) for row in subset])
    eligible = [row for row in subset if row["noise_eligible"]]
    return {
        "group_kind": kind, "group_value": value, "n": len(subset),
        "n_noise_eligible": len(eligible), "mean_residual_bits": float(residuals.mean()),
        "sd_residual_bits": float(residuals.std(ddof=1)) if len(residuals) > 1 else 0.0,
        "median_residual_bits": float(np.median(residuals)),
        "rmse_residual_bits": float(np.sqrt(np.mean(residuals**2))),
        "max_abs_residual_bits": float(np.max(np.abs(residuals))),
        "primary_ci_coverage_fraction": (
            float(np.mean([bool(row["ci_covers_primary_prediction"]) for row in eligible])) if eligible else ""
        ),
    }


def residual_summaries(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result = [summary_record("all", "all", rows)]
    for field in ("campaign", "scale", "convention", "mu"):
        values = sorted({row[field] for row in rows}, key=str)
        for value in values:
            result.append(summary_record(field, str(value), [row for row in rows if row[field] == value]))
    return result


def correlation_record(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    return {
        "n": len(x), "pearson_r": float(pearson.statistic), "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic), "spearman_p": float(spearman.pvalue),
    }


def g6_slope_analysis(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    subset = [row for row in rows if row["campaign"] == "G6"]
    ages = sorted({int(row["T"]) for row in subset})
    columns = [np.asarray([1.0 if int(row["T"]) == age else 0.0 for row in subset]) for age in ages]
    columns.append(np.asarray([math.log2(int(row["S"]) / 2560.0) for row in subset]))
    x = np.column_stack(columns)
    y = np.asarray([float(row["residual_bits"]) for row in subset])
    beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ beta
    df = len(y) - int(rank)
    covariance = (residual @ residual / df) * np.linalg.pinv(x.T @ x)
    index = len(beta) - 1
    se = math.sqrt(float(covariance[index, index]))
    t_stat = float(beta[index] / se)
    critical = float(stats.t.ppf(0.975, df))
    return {
        "formula": "global-law residual bits ~ T fixed effects + log2(S/2560)",
        "n": len(subset), "df_resid": df,
        "slope_bits_per_S_doubling": float(beta[index]), "se": se,
        "ci_95": [float(beta[index] - critical * se), float(beta[index] + critical * se)],
        "t": t_stat, "p_value": float(2.0 * stats.t.sf(abs(t_stat), df)),
        "T_fixed_effects": {str(age): float(beta[i]) for i, age in enumerate(ages)},
        "rmse_bits": float(np.sqrt(np.mean(residual**2))),
    }


def paired_cancellation_diagnostic(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Test the law in matched cells where eta0(scale) and q**T cancel exactly."""

    controls: defaultdict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["convention"] == "mu0":
            key = (row["campaign"], row["scale"], row["T"], row["H"], row["S"])
            controls[key].append(row)
    records = []
    skipped = []
    for row in rows:
        if row["convention"] == "mu0":
            continue
        key = (row["campaign"], row["scale"], row["T"], row["H"], row["S"])
        candidates = controls.get(key, [])
        if not candidates:
            skipped.append({"point_id": row["point_id"], "reason": "no included matched mu=0 control"})
            continue
        control_etas = {round(float(candidate["eta_star"]), 15) for candidate in candidates}
        if len(control_etas) != 1:
            skipped.append({"point_id": row["point_id"], "reason": "multiple non-equivalent matched mu=0 controls"})
            continue
        control = candidates[0]
        factor = float(row["law_factor_1_minus_mu_times_M"])
        ratio = float(row["eta_star"]) / (float(control["eta_star"]) * factor)
        conservative_low = float(row["eta_ci_low"]) / (float(control["eta_ci_high"]) * factor)
        conservative_high = float(row["eta_ci_high"]) / (float(control["eta_ci_low"]) * factor)
        records.append(
            {
                "momentum_point_id": row["point_id"], "control_point_id": control["point_id"],
                "campaign": row["campaign"], "scale": row["scale"], "T": row["T"],
                "H": row["H"], "S": row["S"], "mu": row["mu"],
                "convention": row["convention"], "observed_to_law_ratio": ratio,
                "log2_observed_to_law": math.log2(ratio),
                "conservative_marginal_ci_low": conservative_low,
                "conservative_marginal_ci_high": conservative_high,
                "law_required_ratio": 1.0,
                "note": "eta0(scale) and q**T cancel; CI is conservative marginal endpoint combination, not a paired-bootstrap ratio CI",
            }
        )
    summaries = []
    for convention in ("nesterov_raw", "nesterov_corrected", "heavy_ball"):
        ages = sorted({int(record["T"]) for record in records if record["convention"] == convention})
        for age in ages:
            values = np.asarray(
                [record["log2_observed_to_law"] for record in records if record["convention"] == convention and int(record["T"]) == age]
            )
            summaries.append(
                {
                    "convention": convention, "T": age, "n": len(values),
                    "mean_log2_observed_to_law": float(values.mean()),
                    "median_log2_observed_to_law": float(np.median(values)),
                    "min_log2_observed_to_law": float(values.min()),
                    "max_log2_observed_to_law": float(values.max()),
                }
            )
    return {
        "estimand": "eta_momentum / [eta_mu0 * (1-mu) * M(T,mu,convention)]",
        "law_required_value": 1.0, "law_required_log2_value": 0.0,
        "n_unambiguous_pairs": len(records), "records": records, "summaries_by_convention_and_T": summaries,
        "skipped": skipped,
    }


def residual_structure(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    residual = [float(row["residual_bits"]) for row in rows]
    correlations = {}
    for field in ("T", "H", "S"):
        correlations[field] = correlation_record([float(row[field]) for row in rows], residual)
    largest = sorted(rows, key=lambda row: abs(float(row["residual_bits"])), reverse=True)[:15]
    return {
        "correlations_with_primary_residual_bits": correlations,
        "paired_cancellation_test": paired_cancellation_diagnostic(rows),
        "g6_omitted_S_test": g6_slope_analysis(rows),
        "largest_absolute_residuals": [
            {
                "point_id": row["point_id"], "campaign": row["campaign"], "scale": row["scale"],
                "T": row["T"], "S": row["S"], "mu": row["mu"], "convention": row["convention"],
                "residual_bits": row["residual_bits"], "ci_bits": [row["residual_ci_low_bits"], row["residual_ci_high_bits"]],
            }
            for row in largest
        ],
    }


def brief_fit(fit: Fit) -> dict[str, Any]:
    result = shared_q_record(fit)
    result.update(
        {
            "n": fit.n, "df_resid": fit.df_resid,
            "rmse_log": float(np.sqrt(np.mean(fit.residuals**2))),
            "rmse_bits": float(np.sqrt(np.mean((fit.residuals / math.log(2.0)) ** 2))),
        }
    )
    return result


def sensitivity_audit(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    confirmatory = [row for row in rows if not row["exploratory"]]
    seen: set[str] = set()
    deduplicated = []
    for row in rows:
        key = str(row["dependence_group"])
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(row)
    leave_one_out = []
    for campaign in sorted({row["campaign"] for row in rows}):
        subset = [row for row in rows if row["campaign"] != campaign]
        record = brief_fit(fit_linear(f"leave out {campaign}", subset))
        record["omitted_campaign"] = campaign
        leave_one_out.append(record)
    wls = fit_linear("global fixed-effect WLS", eligible_rows(rows), weighted=True)
    return {
        "confirmatory_only": brief_fit(fit_linear("confirmatory-only OLS", confirmatory)),
        "one_per_dependence_group": brief_fit(fit_linear("dependence-deduplicated OLS", deduplicated)),
        "inverse_variance_fixed_effect": brief_fit(wls),
        "leave_one_campaign_out": leave_one_out,
        "leave_one_campaign_out_q_range": [
            min(record["q"] for record in leave_one_out), max(record["q"] for record in leave_one_out)
        ],
    }


def convention_strata(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for convention in CONVENTION_ORDER:
        subset = [row for row in rows if row["convention"] == convention]
        noise = eligible_rows(subset)
        campaigns = sorted({row["campaign"] for row in subset})
        ages = sorted({int(row["T"]) for row in subset})
        record: dict[str, Any] = {
            "n_points": len(subset), "n_noise_eligible": len(noise),
            "campaigns": campaigns, "n_campaigns": len(campaigns), "ages": ages,
            "eligible_for_partial": bool(len(subset) >= 8 and len(campaigns) >= 2 and len(ages) >= 3),
        }
        if len(noise) > len({row["scale"] for row in noise}) + 1:
            wls = fit_linear(f"{convention} WLS", noise, weighted=True)
            record["wls"] = serialize_fit(wls)
            record["q"] = shared_q_record(wls)
            record["goodness"] = goodness(noise, wls)
            record["meets_noise_and_coverage_bar"] = bool(
                record["eligible_for_partial"]
                and record["goodness"]["passes_noise_p_ge_0p05"]
                and record["goodness"]["passes_coverage_ge_0p90"]
            )
        else:
            record["meets_noise_and_coverage_bar"] = False
        result[convention] = record
    return result


def decide_verdict(
    global_goodness: dict[str, Any], sharing: dict[str, Any], strata: dict[str, Any]
) -> dict[str, Any]:
    sharing_rejections = [name for name, record in sharing.items() if not record["survives_both_ols_and_wls"]]
    global_pass = bool(
        global_goodness["passes_noise_p_ge_0p05"]
        and global_goodness["passes_coverage_ge_0p90"]
        and not sharing_rejections
    )
    partial_survivors = [
        name for name, record in strata.items() if record.get("meets_noise_and_coverage_bar", False)
    ]
    if global_pass:
        verdict = "COLLAPSES"
    elif partial_survivors:
        verdict = "PARTIAL"
    else:
        verdict = "FAILS"
    broken = [name for name, record in strata.items() if not record.get("meets_noise_and_coverage_bar", False)]
    return {
        "verdict": verdict, "global_pass": global_pass,
        "sharing_rejections": sharing_rejections, "partial_surviving_conventions": partial_survivors,
        "breaking_or_nonqualifying_conventions": broken,
        "rule": (
            "COLLAPSES iff WLS heterogeneity p>=.05, marginal CI coverage>=90%, and all requested "
            "sharing tests survive in both OLS and WLS; PARTIAL iff a prespecified convention with "
            ">=8 points, >=2 campaigns, and >=3 ages meets the same noise/coverage bar; otherwise FAILS."
        ),
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    if not rows:
        raise AuditError(f"refusing to write empty table {path}")
    if fieldnames is None:
        ordered: list[str] = []
        for row in rows:
            for key in row:
                if key not in ordered:
                    ordered.append(key)
        fieldnames = ordered
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def short_float(value: Any, digits: int = 4) -> str:
    if value in (None, ""):
        return "--"
    return f"{float(value):.{digits}g}"


def write_latex_table(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    lines = [
        r"% Generated by mech/law-unification/analyze.py; requires booktabs,longtable.",
        r"\begingroup\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{longtable}{lllrrrrll}",
        r"\caption{Complete included tuned-optimum ledger. Confidence intervals are 95\%.}\label{tab:law-optima}\\",
        r"\toprule",
        r"Campaign & Scale & Convention & $T$ & $H$ & $S$ & $\mu$ & $\eta^*$ [95\% CI] & Status \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Campaign & Scale & Convention & $T$ & $H$ & $S$ & $\mu$ & $\eta^*$ [95\% CI] & Status \\",
        r"\midrule",
        r"\endhead",
        r"\midrule\multicolumn{9}{r}{Continued on next page}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for row in rows:
        eta = f"{short_float(row['eta_star'])} [{short_float(row['eta_ci_low'])}, {short_float(row['eta_ci_high'])}]"
        values = [
            row["campaign"], row["scale"], row["convention"], row["T"], row["H"], row["S"],
            short_float(row["mu"], 3), eta, row["fit_status"],
        ]
        lines.append(" & ".join(latex_escape(value) for value in values) + r" \\")
    lines.extend([r"\end{longtable}", r"\endgroup", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def source_manifest(source_dir: Path, rows: Sequence[dict[str, Any]], excluded: Sequence[dict[str, Any]]) -> dict[str, Any]:
    roles = {
        "two_param_master_eta.csv": "canonical 135M two-parameter optimum/CI ledger",
        "two_param_master_D.csv": "paired displacement master table; audited provenance, not a second optimum unit",
        "disambiguation_source.md": "six exploratory fixed-T=5 H/S disambiguation optima",
        "g4c-readout.json": "final canonical five-seed 1.7B v4-family cell records and fits",
        "g5b-readout.json": "final combined SNOO fits and confidence intervals",
        "g6-readout.json": "complete 135M T-by-S factorial cell records and fits",
        "g8-readout.json": "complete 135M mu=.8/.95 phase scan cell records and fits",
        "g12-readout.json": "complete 135M heavy-ball scan cell records and fits",
        "supplemental-1p7b-readout.json": "prospective 1.7B T=10 raw/corrected fits and intervals",
        "scale_predictions.json": "final SHA-bound G9 joint panel, including two 7B singleton-seed optima",
    }
    files = []
    for name in sorted(SOURCE_HASHES):
        path = source_dir / name
        observed = sha256_file(path)
        if observed != SOURCE_HASHES[name]:
            raise AuditError(f"source hash mismatch for {path}: {observed}")
        files.append(
            {"name": name, "sha256": observed, "bytes": path.stat().st_size, "role": roles[name]}
        )
    return {
        "schema": "law_unification_source_manifest_v1",
        "audit_date": "2026-07-28", "banked_data_only": True,
        "source_files": files,
        "included_points": len(rows), "excluded_ledger_rows": len(excluded),
        "included_by_campaign": dict(sorted(Counter(row["campaign"] for row in rows).items())),
        "included_by_scale": dict(sorted(Counter(row["scale"] for row in rows).items())),
        "g9_7b_provenance": {
            "canonical_derived_artifact": "scale_predictions.json",
            "upstream_joint_readout_sha256": "4d42a5f133684822f0fc81c69c3348610c7b3824836b1ecf937abb6e9515bcaf",
            "raw_joint_readout_available_in_evacuation_archive": False,
            "archive_lookup_result": "root/g9-joint-readout.json not found",
            "handling": "retain two final one-seed optima in OLS/figure; exclude point-mass CIs from WLS/noise calibration",
        },
        "bootstrap_reconstruction": {
            "replicates": REPLICATES, "paired_seed_resampling": True,
            "rng_seeds": SEEDS_BY_SOURCE,
            "G4C_acceptance": "canonical positive-curvature vertex within 0.5 log2 bits of bracket",
            "G6_G8_G12_acceptance": "positive-curvature strict-interior vertex",
        },
    }


def fit_summary_rows(primary: Fit, wls: Fit, sharing: dict[str, Any], strata: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    def append_serialized(family: str, serialized: dict[str, Any]) -> None:
        for coefficient in serialized["coefficients"]:
            if "q" not in coefficient:
                continue
            output.append(
                {
                    "fit_family": family, "fit_name": serialized["name"],
                    "weighting": "WLS" if serialized["weighted"] else "OLS",
                    "slope_stratum": coefficient["name"].removeprefix("log_q[").removesuffix("]"),
                    "n": serialized["n"], "q": coefficient["q"],
                    "q_ci_low": coefficient["q_ci_95"][0], "q_ci_high": coefficient["q_ci_95"][1],
                    "rss_or_chi2": serialized["rss_or_chi2"], "df_resid": serialized["df_resid"],
                    "aic": serialized["aic"], "bic": serialized["bic"],
                }
            )

    append_serialized("global_shared", serialize_fit(primary))
    append_serialized("global_shared", serialize_fit(wls))
    for dimension, record in sharing.items():
        append_serialized(f"q_by_{dimension}", record["full_ols"])
        append_serialized(f"q_by_{dimension}", record["full_wls"])
    for convention, record in strata.items():
        if "wls" in record:
            append_serialized(f"convention_stratum_{convention}", record["wls"])
    return output


def configure_plot_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans", "font.size": 9.0,
            "axes.titlesize": 11.0, "axes.labelsize": 10.0,
            "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
            "legend.fontsize": 7.5, "axes.spines.top": False,
            "axes.spines.right": False, "axes.grid": False,
            "pdf.fonttype": 42, "ps.fonttype": 42,
        }
    )


def save_figure(fig: mpl.figure.Figure, output_dir: Path, stem: str, title: str) -> None:
    fig.savefig(
        output_dir / f"{stem}.png", dpi=240,
        metadata={"Title": title, "Software": "mech/law-unification/analyze.py"},
    )
    fig.savefig(
        output_dir / f"{stem}.pdf",
        metadata={
            "Title": title, "Author": "Law-unification audit",
            "Subject": "Banked-data finite-age outer-rate audit",
            "Creator": "mech/law-unification/analyze.py", "CreationDate": None, "ModDate": None,
        },
    )


def deterministic_jitter(identifier: str, width_bits: float = 0.09) -> float:
    integer = int(hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:8], 16)
    return ((integer / 0xFFFFFFFF) - 0.5) * width_bits


def campaign_colors(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    campaigns = sorted({row["campaign"] for row in rows})
    palette = list(mpl.colormaps["tab20"].colors)
    return {campaign: palette[index % len(palette)] for index, campaign in enumerate(campaigns)}


def collapse_figure(
    rows: Sequence[dict[str, Any]], primary: Fit, verdict: dict[str, Any], output_dir: Path
) -> None:
    configure_plot_style()
    colors = campaign_colors(rows)
    markers = {"mu0": "o", "nesterov_raw": "^", "nesterov_corrected": "s", "heavy_ball": "D"}
    sizes = {"135M": 24.0, "1.7B": 52.0, "7B": 88.0}
    q_record = shared_q_record(primary)
    q, q_low, q_high = q_record["q"], q_record["q_ci_low"], q_record["q_ci_high"]

    fig = plt.figure(figsize=(11.7, 7.7))
    grid = fig.add_gridspec(2, 1, height_ratios=(3.1, 1.25), hspace=0.06)
    ax = fig.add_subplot(grid[0])
    residual_ax = fig.add_subplot(grid[1], sharex=ax)
    fig.subplots_adjust(left=0.09, right=0.735, bottom=0.10, top=0.92)

    ages = np.geomspace(1.8, 180.0, 500)
    ax.fill_between(ages, q_low**ages, q_high**ages, color="#333333", alpha=0.10, linewidth=0)
    ax.plot(ages, q**ages, color="#111111", linewidth=2.0, label=fr"shared fit $q^T$, $q={q:.5f}$")
    residual_ax.axhline(0.0, color="#111111", linewidth=1.35)
    residual_ax.axhspan(-0.10, 0.10, color="#777777", alpha=0.08, zorder=0)

    for row in rows:
        x = float(row["T"]) * 2.0 ** deterministic_jitter(str(row["point_id"]))
        color = colors[row["campaign"]]
        marker = markers[row["convention"]]
        size = sizes[row["scale"]]
        y = float(row["normalized_rate"])
        y_low, y_high = float(row["normalized_ci_low"]), float(row["normalized_ci_high"])
        if y_high > y_low:
            ax.errorbar(
                x, y, yerr=np.asarray([[y - y_low], [y_high - y]]), fmt="none",
                ecolor=color, elinewidth=0.75, alpha=0.45, capsize=1.2, zorder=1,
            )
        ax.scatter(
            [x], [y], s=size, marker=marker, facecolor=color,
            edgecolor="black" if row["scale"] == "7B" else "white",
            linewidth=0.9 if row["scale"] == "7B" else 0.45, alpha=0.90, zorder=3,
        )
        r = float(row["residual_bits"])
        r_low, r_high = float(row["residual_ci_low_bits"]), float(row["residual_ci_high_bits"])
        if r_high > r_low:
            residual_ax.errorbar(
                x, r, yerr=np.asarray([[r - r_low], [r_high - r]]), fmt="none",
                ecolor=color, elinewidth=0.75, alpha=0.45, capsize=1.2, zorder=1,
            )
        residual_ax.scatter(
            [x], [r], s=size, marker=marker, facecolor=color,
            edgecolor="black" if row["scale"] == "7B" else "white",
            linewidth=0.9 if row["scale"] == "7B" else 0.45, alpha=0.90, zorder=3,
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlim(1.65, 190)
    ticks = [2, 5, 10, 20, 40, 160]
    ax.set_xticks(ticks, labels=[str(value) for value in ticks])
    ax.tick_params(axis="x", labelbottom=False)
    residual_ax.set_xticks(ticks, labels=[str(value) for value in ticks])
    ax.set_ylabel(r"normalized tuned rate  $\eta^*/[\widehat{\eta}_0(\mathrm{scale})(1-\mu)M]$")
    residual_ax.set_ylabel(r"residual  $\log_2(\eta^*_{obs}/\eta^*_{fit})$  [bits]")
    residual_ax.set_xlabel(r"effective per-fragment update age  $T$")
    ax.grid(which="major", axis="both", color="#d8d8d8", linewidth=0.55, alpha=0.65)
    residual_ax.grid(which="major", axis="both", color="#d8d8d8", linewidth=0.55, alpha=0.65)
    residual_extent = max(
        0.25,
        max(abs(float(row["residual_ci_low_bits"])) for row in rows),
        max(abs(float(row["residual_ci_high_bits"])) for row in rows),
    )
    residual_ax.set_ylim(-1.06 * residual_extent, 1.06 * residual_extent)
    ax.set_title(
        f"Finite-age law unification: {verdict['verdict']}  |  q={q:.5f}, equal-point OLS, n={len(rows)}",
        loc="left", fontweight="bold",
    )
    ax.text(
        0.012, 0.025,
        "Shading: 95% parameter interval for shared q. Point bars: seed-based 95% CIs.\n"
        "7B outlined points are one-seed estimates with point-mass empirical intervals.",
        transform=ax.transAxes, fontsize=7.5, color="#333333", va="bottom",
    )

    campaign_handles = [
        mpl.lines.Line2D([0], [0], marker="o", linestyle="none", markersize=5.5,
                         markerfacecolor=colors[name], markeredgecolor="white", label=name)
        for name in sorted(colors)
    ]
    convention_labels = {
        "mu0": "mu=0", "nesterov_raw": "raw Nesterov",
        "nesterov_corrected": "corrected Nesterov", "heavy_ball": "heavy-ball",
    }
    convention_handles = [
        mpl.lines.Line2D([0], [0], marker=marker, linestyle="none", markersize=6,
                         markerfacecolor="#777777", markeredgecolor="white", label=convention_labels[name])
        for name, marker in markers.items()
    ]
    scale_handles = [
        mpl.lines.Line2D([0], [0], marker="o", linestyle="none",
                         markersize=math.sqrt(size) * 0.95, markerfacecolor="#999999",
                         markeredgecolor="white", label=scale)
        for scale, size in sizes.items()
    ]
    legend_campaign = ax.legend(
        handles=campaign_handles, title="Campaign (color)", loc="upper left",
        bbox_to_anchor=(1.01, 1.005), frameon=False, ncol=2, columnspacing=0.8,
        handletextpad=0.35, borderaxespad=0.0,
    )
    ax.add_artist(legend_campaign)
    legend_convention = ax.legend(
        handles=convention_handles, title="Convention (marker)", loc="upper left",
        bbox_to_anchor=(1.01, 0.55), frameon=False, borderaxespad=0.0,
    )
    ax.add_artist(legend_convention)
    ax.legend(
        handles=scale_handles, title="Scale (area)", loc="upper left",
        bbox_to_anchor=(1.01, 0.29), frameon=False, borderaxespad=0.0,
    )
    save_figure(fig, output_dir, "collapse", "Finite-age outer-rate law collapse")
    plt.close(fig)


def residual_structure_figure(rows: Sequence[dict[str, Any]], structure: dict[str, Any], output_dir: Path) -> None:
    configure_plot_style()
    colors = campaign_colors(rows)
    convention_colors = dict(zip(CONVENTION_ORDER, ("#4C78A8", "#F58518", "#54A24B", "#B279A2")))
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.3))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.10, top=0.91, hspace=0.36, wspace=0.30)

    ax = axes[0, 0]
    convention_data = [[float(row["residual_bits"]) for row in rows if row["convention"] == name] for name in CONVENTION_ORDER]
    boxes = ax.boxplot(convention_data, positions=np.arange(4), widths=0.48, patch_artist=True,
                       showfliers=False, medianprops={"color": "black", "linewidth": 1.3})
    for patch, name in zip(boxes["boxes"], CONVENTION_ORDER):
        patch.set_facecolor(convention_colors[name]); patch.set_alpha(0.28)
    for index, name in enumerate(CONVENTION_ORDER):
        subset = [row for row in rows if row["convention"] == name]
        xs = [index + deterministic_jitter(str(row["point_id"]), 0.32) for row in subset]
        ax.scatter(xs, [row["residual_bits"] for row in subset], s=14,
                   color=convention_colors[name], alpha=0.65, edgecolor="white", linewidth=0.3)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(range(4), ["mu=0", "raw\nNesterov", "corrected\nNesterov", "heavy-ball"])
    ax.set_ylabel("global-law residual [bits]")
    ax.set_title("A. Convention structure", loc="left", fontweight="bold")

    ax = axes[0, 1]
    paired_g6 = [
        record for record in structure["paired_cancellation_test"]["records"]
        if record["campaign"] == "G6"
    ]
    for convention in ("nesterov_raw", "nesterov_corrected"):
        subset = [record for record in paired_g6 if record["convention"] == convention]
        for record in subset:
            x = float(record["T"]) * 2.0 ** deterministic_jitter(record["momentum_point_id"], 0.11)
            ax.scatter(x, record["log2_observed_to_law"], s=28,
                       marker="^" if convention == "nesterov_raw" else "s",
                       color=convention_colors[convention], edgecolor="white", linewidth=0.4, alpha=0.76)
        ages = sorted({int(record["T"]) for record in subset})
        medians = [
            float(np.median([record["log2_observed_to_law"] for record in subset if int(record["T"]) == age]))
            for age in ages
        ]
        ax.plot(ages, medians, marker="^" if convention == "nesterov_raw" else "s",
                color=convention_colors[convention], linewidth=1.6,
                label="raw Nesterov" if convention == "nesterov_raw" else "corrected Nesterov")
    ax.axhline(0, color="black", linewidth=1.1, linestyle="--", label="law requirement")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 5, 10, 20], ["2", "5", "10", "20"])
    ax.set_xlabel("G6 effective age T")
    ax.set_ylabel(r"$\log_2\{\eta_\mu/[\eta_{\mu=0}(1-\mu)M]\}$ [bits]")
    ax.set_title("B. Matched-control cancellation test", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7.5)

    ax = axes[1, 0]
    campaigns = sorted({row["campaign"] for row in rows})
    for position, campaign in enumerate(campaigns):
        subset = [row for row in rows if row["campaign"] == campaign]
        values = np.asarray([float(row["residual_bits"]) for row in subset])
        ys = [position + deterministic_jitter(str(row["point_id"]), 0.30) for row in subset]
        ax.scatter(values, ys, s=15, color=colors[campaign], alpha=0.66, edgecolor="white", linewidth=0.3)
        ax.plot([values.mean()], [position], marker="|", markersize=13, color="black", markeredgewidth=1.5)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(range(len(campaigns)), campaigns)
    ax.set_xlabel("global-law residual [bits]")
    ax.set_title("C. Campaign offsets", loc="left", fontweight="bold")

    ax = axes[1, 1]
    g6 = [row for row in rows if row["campaign"] == "G6"]
    grand = float(np.mean([row["residual_bits"] for row in g6]))
    means_t = {age: float(np.mean([row["residual_bits"] for row in g6 if row["T"] == age])) for age in {row["T"] for row in g6}}
    adjusted = [float(row["residual_bits"]) - means_t[row["T"]] + grand for row in g6]
    x_values = np.asarray([math.log2(int(row["S"]) / 2560.0) for row in g6])
    for row, x, y in zip(g6, x_values, adjusted):
        ax.scatter(x, y, marker={"mu0": "o", "nesterov_raw": "^", "nesterov_corrected": "s"}[row["convention"]],
                   s=28, color=convention_colors[row["convention"]], edgecolor="white", linewidth=0.4, alpha=0.8)
    slope = structure["g6_omitted_S_test"]["slope_bits_per_S_doubling"]
    line_x = np.asarray([0.0, 2.0])
    line_y = grand + slope * (line_x - x_values.mean())
    ax.plot(line_x, line_y, color="black", linewidth=1.7)
    ax.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_xticks([0, 1, 2], ["2560", "5120", "10240"])
    ax.set_xlabel("G6 local-work scale S")
    ax.set_ylabel("residual adjusted for T fixed effects [bits]")
    ax.set_title("D. Local work is not the first-order break", loc="left", fontweight="bold")
    ax.text(0.03, 0.04, f"slope={slope:+.3f} bits/doubling\np={structure['g6_omitted_S_test']['p_value']:.2g}",
            transform=ax.transAxes, fontsize=8, va="bottom")

    fig.suptitle("Residual structure of the one-q finite-age law", x=0.08, ha="left", fontsize=13, fontweight="bold")
    save_figure(fig, output_dir, "residual_structure", "Residual structure of finite-age outer-rate law")
    plt.close(fig)


def p_text(value: float) -> str:
    if value == 0.0 or value < 1e-300:
        return "<1e-300"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def p_clause(value: float) -> str:
    rendered = p_text(value)
    return f"p{rendered}" if rendered.startswith("<") else f"p={rendered}"


def write_writeup(
    path: Path,
    rows: Sequence[dict[str, Any]],
    excluded: Sequence[dict[str, Any]],
    primary: Fit,
    wls: Fit,
    goodness_record: dict[str, Any],
    sharing: dict[str, Any],
    strata: dict[str, Any],
    verdict: dict[str, Any],
    structure: dict[str, Any],
    sensitivities: dict[str, Any],
) -> None:
    q = shared_q_record(primary)
    q_wls = shared_q_record(wls)
    eta0_parts = []
    for scale in SCALE_ORDER:
        estimate, low, high = coefficient_by_name(primary, f"log_eta0[{scale}]")
        eta0_parts.append(f"{scale} {math.exp(estimate):.5g} [{math.exp(low):.5g}, {math.exp(high):.5g}]")
    sharing_parts = []
    for dimension in ("mu", "scale", "convention"):
        record = sharing[dimension]
        status = "survives" if record["survives_both_ols_and_wls"] else "rejected"
        sharing_parts.append(
            f"{dimension}: OLS {p_clause(record['ols']['p_value'])}, "
            f"WLS {p_clause(record['wls']['p_value'])} ({status})"
        )
    strata_lines = []
    for convention in CONVENTION_ORDER:
        record = strata[convention]
        if "goodness" in record:
            good = record["goodness"]
            q_value = record["q"]["q"]
            strata_lines.append(
                f"| {convention} | {record['n_points']} | {record['n_campaigns']} | "
                f"{len(record['ages'])} | {q_value:.5f} | {p_text(good['heterogeneity_p'])} | "
                f"{100*good['coverage_fraction']:.1f}% | "
                f"{'yes' if record['meets_noise_and_coverage_bar'] else 'no'} |"
            )
        else:
            strata_lines.append(
                f"| {convention} | {record['n_points']} | {record['n_campaigns']} | "
                f"{len(record['ages'])} | -- | -- | -- | no |"
            )
    g6 = structure["g6_omitted_S_test"]
    paired_g6 = [
        record for record in structure["paired_cancellation_test"]["records"]
        if record["campaign"] == "G6"
    ]
    paired_parts = []
    for convention in ("nesterov_raw", "nesterov_corrected"):
        label = "raw" if convention == "nesterov_raw" else "corrected"
        ages = sorted({int(record["T"]) for record in paired_g6 if record["convention"] == convention})
        values = []
        for age in ages:
            median = float(np.median([
                record["log2_observed_to_law"] for record in paired_g6
                if record["convention"] == convention and int(record["T"]) == age
            ]))
            values.append(f"T={age}: {median:+.3f}")
        paired_parts.append(f"{label} ({', '.join(values)})")
    largest = structure["largest_absolute_residuals"][0]
    included_counts = ", ".join(
        f"{campaign}={count}" for campaign, count in sorted(Counter(row["campaign"] for row in rows).items())
    )
    verdict_explanation = {
        "COLLAPSES": "The residuals meet the frozen seed-noise and coverage thresholds, and none of the requested q-sharing tests rejects.",
        "PARTIAL": (
            "The global law fails, but the following prespecified convention strata clear the frozen noise/coverage bar: "
            + ", ".join(verdict["partial_surviving_conventions"]) + "."
        ),
        "FAILS": "The global law fails and no prespecified optimizer-convention stratum clears the frozen PARTIAL threshold.",
    }[verdict["verdict"]]
    text = rf"""# Law unification verdict: {verdict['verdict']}

**Decision.** {verdict_explanation} This is not a cosmetic rejection: the best-fitting shared curve has fixed-effect heterogeneity $\chi^2={goodness_record['chi2']:.1f}$ on {goodness_record['df']} degrees of freedom ({p_clause(goodness_record['heterogeneity_p'])}) and covers only {goodness_record['coverage_count']}/{goodness_record['n_noise_eligible']} ({100*goodness_record['coverage_fraction']:.1f}%) of the noise-eligible marginal 95% intervals, versus the frozen 90% requirement. The honest v2 keystone label is therefore **{verdict['verdict']}**, not a law-level collapse.

## Evidence and estimand

The ledger contains all {len(rows)} accepted banked tuned optima: {included_counts}. It excludes no accepted point because it disagrees with the hypothesis. The separate exclusion ledger has {len(excluded)} rows: 12 non-interior/nonconvex/extrapolated two-parameter fits, 11 superseded cumulative estimators (G4/G4B and G5), and one out-of-scope G13/G13B external transfer. Exact A/A-equivalent controls remain visible with dependence labels. The two final 7B points use the sole retained seed and point-mass empirical intervals; they enter equal-point OLS and the figure, but not WLS or seed-noise calibration. The final SHA-bound scale artifact is used because the raw joint G9 readout was not found under its expected evacuation-archive path.

For each point the audited response is $z=\log\eta^*-\log[(1-\mu)M]$. The primary equal-point model is $z=\log\eta_0(\mathrm{{scale}})+T\log q$. Raw Nesterov uses $M=(1-\mu^{{T+1}})/(1-\mu)$, G12 heavy-ball uses $M=(1-\mu^T)/(1-\mu)$, corrected Nesterov uses $M=1$, and $\mu=0$ uses $M=1$. No campaign, $S$, or $H$ term is admitted to the keystone fit.

## Global fit and sharing tests

Equal-point OLS gives **q={q['q']:.6f}** (95% CI {q['q_ci_low']:.6f}-{q['q_ci_high']:.6f}); fitted $\eta_0$ values are {', '.join(eta0_parts)}. The fixed-effect inverse-variance sensitivity gives q={q_wls['q']:.6f} ({q_wls['q_ci_low']:.6f}-{q_wls['q_ci_high']:.6f}), showing that narrow seed intervals do not merely sharpen the same descriptive fit. Leave-one-campaign-out OLS spans q={sensitivities['leave_one_campaign_out_q_range'][0]:.6f}-{sensitivities['leave_one_campaign_out_q_range'][1]:.6f}; confirmatory-only OLS gives q={sensitivities['confirmatory_only']['q']:.6f}; and one-per-dependence-group OLS gives q={sensitivities['one_per_dependence_group']['q']:.6f}.

Requested slope-sharing tests: {'; '.join(sharing_parts)}. The scale test is restricted to 135M and 1.7B because two 7B points at one age cannot identify a 7B slope. A restriction can be descriptively stable in OLS yet still fail under the seed-precision audit; the report calls sharing successful only when both prespecified tests survive.

| Convention | points | campaigns | ages | WLS q | heterogeneity p | 95% coverage | PARTIAL bar? |
|---|---:|---:|---:|---:|---:|---:|:---:|
{chr(10).join(strata_lines)}

## Residual mechanism target

The decisive residual structure is convention by age. In matched G6 cells, dividing a momentum optimum by its same-$T$/same-$S$ $\mu=0$ control and by $(1-\mu)M$ cancels both $\eta_0(\mathrm{{scale}})$ and $q^T$; the law therefore requires exactly zero log2 bits at every age. Observed median bits are {'; '.join(paired_parts)}. The raw arm reverses across age and misses by nearly three bits at $T=20$, so neither a different shared q nor a scale intercept can repair it.

By contrast, within the balanced G6 factorial and after $T$ fixed effects, each doubling of local-work scale $S$ shifts the residual by only **{g6['slope_bits_per_S_doubling']:+.4f} bits** (95% CI {g6['ci_95'][0]:+.4f} to {g6['ci_95'][1]:+.4f}, {p_clause(g6['p_value'])}). Thus local work is not the first-order break in this audit. The largest absolute global residual is {largest['point_id']} at {largest['residual_bits']:+.3f} bits. Residual summaries by campaign, scale, exact $\mu$, and convention are in `residual_summary.csv`; the full matched cancellation ledger, correlations, and 15 largest misses are in `residual_structure.json` and `paired_cancellation.csv`.

The next mechanism target is the convention-specific age/history transform itself - especially why the raw and heavy-ball tuned-rate ratio decays with age while the proposed multiplier grows toward one - not another refit of one universal scalar q. A law-paper rebrand is not supported by these banked data unless a revised transform is prospectively specified and independently tested.
"""
    path.write_text(text, encoding="utf-8")


def write_figure_caption(path: Path, rows: Sequence[dict[str, Any]], primary: Fit, verdict: dict[str, Any]) -> None:
    q = shared_q_record(primary)
    path.write_text(
        rf"""**Figure: banked-data test of the finite-age outer-rate law.** The top panel shows all {len(rows)} accepted tuned optima after division by the fitted scale intercept and the code-true $(1-\mu)M$ factor; the black curve is the equal-point OLS estimate $q^T$ with $q={q['q']:.6f}$ and shading is its 95% parameter interval. Vertical bars are seed-based 95% intervals. Color identifies campaign, marker shape identifies optimizer convention, and marker area identifies model scale. The two outlined 7B points have one retained seed and therefore point-mass empirical intervals; they are shown and included in OLS but excluded from inverse-variance/noise calibration. The lower panel gives $\log_2(\eta^*_{{obs}}/\eta^*_{{fit}})$. Under the frozen criteria the verdict is **{verdict['verdict']}**. The visible residual structure is quantified separately rather than absorbed into the keystone fit.
""",
        encoding="utf-8",
    )


def write_readme(path: Path) -> None:
    path.write_text(
        """# Reproducing the law-unification audit

This directory is self-contained apart from its declared Python packages. It reads only the frozen files in `sources/`; it launches no training and makes no network calls.

From the repository root:

```bash
uv run --no-project -p python3.13 --with numpy --with scipy --with matplotlib \\
  python mech/law-unification/analyze.py
```

The script verifies every source SHA-256, reconstructs the G4C/G6/G8/G12 eta-star intervals with the original paired seed-resampling seeds, assembles both ledgers, runs OLS/WLS and nested sharing tests, applies the verdict rule frozen in `INCLUSION.md`, and regenerates the CSV/JSON/TeX/PNG/PDF/writeup artifacts. It also writes the requested one-line handoff to `/private/tmp/h200-law-note.md`; pass `--law-note PATH` to change that location.

The primary estimate is equal-point OLS in natural-log space. Fixed-effect WLS uses `SE(log eta) = [log(CI_high)-log(CI_low)]/(2*1.959964)` only for nondegenerate intervals with zero invalid bootstrap refits. This intentionally prevents singleton 7B intervals or qualified conditional intervals from receiving infinite or misleading weight.
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent,
        help="artifact directory (default: directory containing this script)",
    )
    parser.add_argument(
        "--law-note", type=Path, default=Path("/private/tmp/h200-law-note.md"),
        help="requested external one-line verdict note",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    source_dir = output_dir / "sources"
    if not (output_dir / "INCLUSION.md").is_file():
        raise AuditError(f"missing frozen inclusion policy in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, excluded = assemble_all(source_dir)
    manifest = source_manifest(source_dir, rows, excluded)
    primary = fit_linear("global equal-point OLS shared q", rows)
    noise = eligible_rows(rows)
    wls = fit_linear("global fixed-effect inverse-variance WLS shared q", noise, weighted=True)
    annotate_predictions(rows, primary)
    sharing = sharing_audit(rows)
    global_goodness = goodness(noise, wls)
    strata = convention_strata(rows)
    verdict = decide_verdict(global_goodness, sharing, strata)
    structure = residual_structure(rows)
    sensitivities = sensitivity_audit(rows)
    residual_summary = residual_summaries(rows)

    results = {
        "schema": "law_unification_fit_results_v1", "audit_date": "2026-07-28",
        "law": "eta_star = eta0(scale) * (1-mu) * M(T,mu,convention) * q**T",
        "n_points": len(rows), "n_noise_eligible": len(noise),
        "counts": {
            "campaign": dict(sorted(Counter(row["campaign"] for row in rows).items())),
            "scale": dict(sorted(Counter(row["scale"] for row in rows).items())),
            "convention": dict(sorted(Counter(row["convention"] for row in rows).items())),
            "exact_mu": {str(key): value for key, value in sorted(Counter(row["mu"] for row in rows).items())},
        },
        "primary_ols": serialize_fit(primary),
        "fixed_effect_wls": serialize_fit(wls),
        "global_noise_and_coverage": global_goodness,
        "sharing_tests": sharing,
        "convention_strata": strata,
        "sensitivities": sensitivities,
        "verdict": verdict,
        "inclusion_policy_sha256": sha256_file(output_dir / "INCLUSION.md"),
    }

    write_csv(output_dir / "tuned_optima.csv", rows)
    write_csv(output_dir / "excluded_points.csv", excluded)
    write_latex_table(output_dir / "tuned_optima.tex", rows)
    write_csv(output_dir / "fit_summary.csv", fit_summary_rows(primary, wls, sharing, strata))
    write_csv(output_dir / "residual_summary.csv", residual_summary)
    write_csv(
        output_dir / "paired_cancellation.csv",
        structure["paired_cancellation_test"]["records"],
    )
    sharing_rows = []
    for dimension, record in sharing.items():
        for method in ("ols", "wls"):
            test = record[method]
            sharing_rows.append(
                {
                    "dimension": dimension, "scope": record["scope"], "method": method.upper(),
                    "test": test["test"], "statistic": test["statistic"], "df_num": test["df_num"],
                    "df_den": "" if test["df_den"] is None else test["df_den"],
                    "p_value": test["p_value"], "survives_p_ge_0p05": test["survives_p_ge_0p05"],
                    "survives_both_ols_and_wls": record["survives_both_ols_and_wls"],
                }
            )
    write_csv(output_dir / "sharing_tests.csv", sharing_rows)
    convention_rows = []
    for convention, record in strata.items():
        good = record.get("goodness", {})
        convention_rows.append(
            {
                "convention": convention, "n_points": record["n_points"],
                "n_noise_eligible": record["n_noise_eligible"], "n_campaigns": record["n_campaigns"],
                "n_ages": len(record["ages"]), "ages": ";".join(str(value) for value in record["ages"]),
                "eligible_for_partial": record["eligible_for_partial"],
                "q_wls": record.get("q", {}).get("q", ""),
                "heterogeneity_p": good.get("heterogeneity_p", ""),
                "coverage_fraction": good.get("coverage_fraction", ""),
                "meets_noise_and_coverage_bar": record["meets_noise_and_coverage_bar"],
            }
        )
    write_csv(output_dir / "convention_strata.csv", convention_rows)
    write_json(output_dir / "fit_results.json", results)
    write_json(output_dir / "residual_structure.json", structure)
    write_json(output_dir / "source_manifest.json", manifest)
    write_writeup(
        output_dir / "WRITEUP.md", rows, excluded, primary, wls, global_goodness,
        sharing, strata, verdict, structure, sensitivities,
    )
    write_figure_caption(output_dir / "FIGURE_CAPTION.md", rows, primary, verdict)
    write_readme(output_dir / "README.md")
    collapse_figure(rows, primary, verdict, output_dir)
    residual_structure_figure(rows, structure, output_dir)

    q = shared_q_record(primary)["q"]
    args.law_note.write_text(
        f"LAW: {verdict['verdict']}, q={q:.6f}, n_points={len(rows)}\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "verdict": verdict["verdict"], "q": q, "n_points": len(rows),
                "n_noise_eligible": len(noise), "heterogeneity_p": global_goodness["heterogeneity_p"],
                "coverage": global_goodness["coverage_fraction"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
