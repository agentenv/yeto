#!/usr/bin/env python3
"""CPU replay skeleton for exact-state midpoint optimizer candidates.

This deliberately refuses endpoint-only captures. It computes MTRF and MSTP
directions from exact H/2/end parameters and Adam state, audits a provided
production baseline, and scores only optional sealed next-direction or CRN
finite-loss outcomes. It does not run a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np


EPS = 1e-12
REQUIRED_ARRAYS = (
    "theta0",
    "theta_mid",
    "theta_end",
    "exp_avg_mid",
    "exp_avg_end",
    "metric_mid",
    "metric_end",
    "step_mid",
    "step_end",
    "weights",
    "bounds",
    "lr_mass_first",
    "lr_mass_second",
    "baseline_direction",
)
OPTIONAL_TARGETS = (
    "next_direction",
    "loss_baseline_k0",
    "loss_mtrf_k0",
    "loss_mstp_k0",
    "loss_baseline_k8",
    "loss_mtrf_k8",
    "loss_mstp_k8",
)
INDEX_REQUIRED_FIELDS = (
    "schema",
    "boundary_id",
    "seed",
    "fragment",
    "worker_ids",
    "responder_order",
    "h",
    "accepted_mid_steps",
    "accepted_end_steps",
    "state_npz",
    "optimizer_metadata_json",
    "crn_manifest_json",
    "source_commit",
    "image_digest",
    "model_digest",
    "data_digest",
    "analysis_config_digest",
)


def finite(x: float) -> float:
    if not math.isfinite(x):
        raise ValueError(f"non-finite value {x}")
    return float(x)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=np.float64)))


def cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    na, nb = norm(a), norm(b)
    if na <= EPS or nb <= EPS:
        return None
    return finite(
        np.clip(np.dot(a.astype(np.float64), b.astype(np.float64)) / (na * nb), -1, 1)
    )


def graft(raw: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict]:
    nr, nt = norm(raw), norm(target)
    if nr <= EPS or nt <= EPS or not np.isfinite(raw).all():
        return target.copy(), {"fallback": True, "norm_rel_error": 0.0}
    out = np.asarray(raw * np.float32(nt / nr), dtype=np.float32)
    return out, {
        "fallback": False,
        "norm_rel_error": finite(abs(norm(out) - nt) / max(nt, EPS)),
    }


def weighted_average(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    total = float(np.sum(weights, dtype=np.float64))
    if total <= 0:
        raise ValueError("nonpositive total weight")
    out = np.zeros(values.shape[1], dtype=np.float32)
    for idx in range(values.shape[0]):
        out += np.float32(float(weights[idx]) / total) * values[idx]
    return out


def rda_tensor(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, dict]:
    total = float(np.sum(weights, dtype=np.float64))
    if total <= 0:
        raise ValueError("nonpositive total weight")
    norms = np.asarray([norm(v) for v in values], dtype=np.float64)
    radial = float(np.dot(norms, weights.astype(np.float64)) / total)
    direction = np.zeros(values.shape[1], dtype=np.float32)
    for idx, value in enumerate(values):
        if norms[idx] > 0:
            direction += np.float32(float(weights[idx]) / total / norms[idx]) * value
    consensus = norm(direction)
    if consensus <= EPS:
        return weighted_average(values, weights), {"fallback": True}
    direction *= np.float32(radial / consensus)
    return direction, {"fallback": False}


def rda(
    values: np.ndarray, weights: np.ndarray, bounds: np.ndarray
) -> tuple[np.ndarray, dict]:
    out = np.empty(values.shape[1], dtype=np.float32)
    fallbacks = 0
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        sl = slice(int(lo), int(hi))
        out[sl], meta = rda_tensor(values[:, sl], weights)
        fallbacks += int(meta["fallback"])
    return out, {"fallbacks": fallbacks, "tensor_count": len(bounds) - 1}


def bias_correct(value: np.ndarray, beta: float, steps: np.ndarray) -> np.ndarray:
    steps = np.asarray(steps, dtype=np.float64).reshape(-1)
    if value.shape[0] != steps.size:
        raise ValueError("step vector does not match workers")
    correction = 1.0 - np.power(beta, steps)
    if np.any(correction <= 0):
        raise ValueError("invalid Adam bias-correction step")
    return np.asarray(value / correction[:, None], dtype=np.float32)


def adam_views(
    record: dict[str, np.ndarray], beta1: float, beta2: float, eps: float
) -> dict[str, np.ndarray]:
    mm = bias_correct(record["exp_avg_mid"], beta1, record["step_mid"])
    me = bias_correct(record["exp_avg_end"], beta1, record["step_end"])
    vm = bias_correct(record["metric_mid"], beta2, record["step_mid"])
    ve = bias_correct(record["metric_end"], beta2, record["step_end"])
    if np.any(vm < 0) or np.any(ve < 0):
        raise ValueError("negative Adam metric state")
    pm = np.asarray(1.0 / (np.sqrt(vm.astype(np.float64)) + eps), dtype=np.float32)
    pe = np.asarray(1.0 / (np.sqrt(ve.astype(np.float64)) + eps), dtype=np.float32)
    return {"m_mid": mm, "m_end": me, "p_mid": pm, "p_end": pe}


def candidate_mtrf(
    record: dict[str, np.ndarray],
    views: dict[str, np.ndarray],
    baseline: np.ndarray,
) -> tuple[np.ndarray, dict]:
    theta0, mid, end = record["theta0"], record["theta_mid"], record["theta_end"]
    weights, bounds = record["weights"], record["bounds"]
    a, b = theta0 - mid, mid - end
    g = a + b
    l1 = np.asarray(record["lr_mass_first"], dtype=np.float64).reshape(-1)
    l2 = np.asarray(record["lr_mass_second"], dtype=np.float64).reshape(-1)
    if l1.size == 1:
        l1 = np.repeat(l1, g.shape[0])
    if l2.size == 1:
        l2 = np.repeat(l2, g.shape[0])
    if np.any(l1 <= 0) or np.any(l2 <= 0):
        raise ValueError("nonpositive half LR mass")
    pbar_end = weighted_average(views["p_end"], weights)
    worker_candidate = np.empty_like(g)
    accepted = 0
    fallbacks = 0
    max_error = 0.0
    for worker in range(g.shape[0]):
        f1 = a[worker] / (np.float32(l1[worker]) * views["p_mid"][worker])
        f2 = b[worker] / (np.float32(l2[worker]) * views["p_end"][worker])
        trend = np.float32(2.0) * f2 - f1
        moment_trend = np.float32(2.0) * views["m_end"][worker] - views["m_mid"][worker]
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            sl = slice(int(lo), int(hi))
            z = pbar_end[sl] * trend[sl]
            q = pbar_end[sl] * moment_trend[sl]
            if (
                np.isfinite(z).all()
                and np.isfinite(q).all()
                and float(np.dot(z.astype(np.float64), q.astype(np.float64))) > 0
            ):
                chosen, meta = graft(z, g[worker, sl])
                worker_candidate[worker, sl] = chosen
                accepted += int(not meta["fallback"])
                fallbacks += int(meta["fallback"])
                max_error = max(max_error, float(meta["norm_rel_error"]))
            else:
                worker_candidate[worker, sl] = g[worker, sl]
    merged, merge_meta = rda(worker_candidate, weights, bounds)
    final = np.empty_like(merged)
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        sl = slice(int(lo), int(hi))
        final[sl], meta = graft(merged[sl], baseline[sl])
        fallbacks += int(meta["fallback"])
        max_error = max(max_error, float(meta["norm_rel_error"]))
    return final, {
        "accepted_worker_tensors": accepted,
        "worker_tensor_attempts": g.shape[0] * (len(bounds) - 1),
        "fallbacks": fallbacks + merge_meta["fallbacks"],
        "max_norm_rel_error": finite(max_error),
    }


def candidate_mstp(
    record: dict[str, np.ndarray],
    views: dict[str, np.ndarray],
    baseline: np.ndarray,
) -> tuple[np.ndarray, dict]:
    theta0, mid, end = record["theta0"], record["theta_mid"], record["theta_end"]
    weights, bounds = record["weights"], record["bounds"]
    a, b = theta0 - mid, mid - end
    merge_a, audit_a = rda(a, weights, bounds)
    merge_b, audit_b = rda(b, weights, bounds)
    direction_mid = weighted_average(views["p_mid"] * views["m_mid"], weights)
    direction_end = weighted_average(views["p_end"] * views["m_end"], weights)
    raw = np.float32(2.0) * direction_end - direction_mid
    final = np.empty_like(baseline)
    acted = 0
    stationary = 0
    fallbacks = audit_a["fallbacks"] + audit_b["fallbacks"]
    max_error = 0.0
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        sl = slice(int(lo), int(hi))
        q, qm = graft(raw[sl], baseline[sl])
        delta = q - baseline[sl]
        radius = min(norm(baseline[sl]), norm(merge_b[sl] - merge_a[sl]))
        nd = norm(delta)
        alpha = 0.0 if nd <= EPS else min(1.0, radius / nd)
        candidate = baseline[sl] + np.float32(alpha) * delta
        final[sl], fm = graft(candidate, baseline[sl])
        acted += int(alpha > 0 and norm(final[sl] - baseline[sl]) > EPS)
        stationary += int(radius <= EPS)
        fallbacks += int(qm["fallback"]) + int(fm["fallback"])
        max_error = max(
            max_error, float(qm["norm_rel_error"]), float(fm["norm_rel_error"])
        )
    return final, {
        "acted_tensors": acted,
        "stationary_tensors": stationary,
        "tensor_count": len(bounds) - 1,
        "fallbacks": fallbacks,
        "max_norm_rel_error": finite(max_error),
    }


def action_metrics(candidate: np.ndarray, baseline: np.ndarray) -> dict:
    relative = norm(candidate - baseline) / max(norm(baseline), EPS)
    c = cosine(candidate, baseline)
    angle = 0.0 if c is None else math.degrees(math.acos(float(np.clip(c, -1, 1))))
    return {"relative_action": finite(relative), "angle_deg": finite(angle)}


def load_record(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in REQUIRED_ARRAYS if key not in data]
        if missing:
            raise ValueError(f"{path}: missing {missing}")
        record = {key: np.asarray(data[key]) for key in data.files}
    shapes = {
        record[key].shape
        for key in (
            "theta0",
            "theta_mid",
            "theta_end",
            "exp_avg_mid",
            "exp_avg_end",
            "metric_mid",
            "metric_end",
        )
    }
    if len(shapes) != 1:
        raise ValueError(f"{path}: worker-state shapes disagree: {shapes}")
    workers, width = record["theta0"].shape
    if record["weights"].reshape(-1).size != workers:
        raise ValueError("weights do not match workers")
    bounds = record["bounds"].astype(np.int64).reshape(-1)
    if bounds[0] != 0 or bounds[-1] != width or np.any(bounds[1:] <= bounds[:-1]):
        raise ValueError("invalid tensor bounds")
    record["weights"] = record["weights"].astype(np.float64).reshape(-1)
    record["bounds"] = bounds
    for key in (
        "theta0",
        "theta_mid",
        "theta_end",
        "exp_avg_mid",
        "exp_avg_end",
        "metric_mid",
        "metric_end",
        "baseline_direction",
    ):
        record[key] = record[key].astype(np.float32, copy=False)
    return record


def replay_record(path: Path, beta1: float, beta2: float, eps: float) -> dict:
    record = load_record(path)
    g_workers = record["theta0"] - record["theta_end"]
    reconstructed, production_meta = rda(g_workers, record["weights"], record["bounds"])
    baseline = record["baseline_direction"].reshape(-1)
    if baseline.shape != reconstructed.shape:
        raise ValueError("baseline direction shape mismatch")
    baseline_error = float(np.max(np.abs(reconstructed - baseline)))
    views = adam_views(record, beta1, beta2, eps)
    mtrf, mtrf_audit = candidate_mtrf(record, views, baseline)
    mstp, mstp_audit = candidate_mstp(record, views, baseline)
    candidates = {"mtrf": mtrf, "mstp": mstp}
    result = {
        "record": str(path),
        "record_sha256": sha256_file(path),
        "production": {"max_abs_error": finite(baseline_error), **production_meta},
        "candidates": {},
    }
    for name, candidate in candidates.items():
        entry = {
            "audit": mtrf_audit if name == "mtrf" else mstp_audit,
            "action": action_metrics(candidate, baseline),
            "direction_sha256": hashlib.sha256(candidate.tobytes()).hexdigest(),
        }
        if "next_direction" in record:
            target = record["next_direction"].astype(np.float32).reshape(-1)
            base_cos, cand_cos = cosine(baseline, target), cosine(candidate, target)
            if base_cos is None or cand_cos is None:
                raise ValueError("zero next-direction target")
            entry["next_direction"] = {
                "baseline_cosine": base_cos,
                "candidate_cosine": cand_cos,
                "gain": finite(cand_cos - base_cos),
            }
        loss_key0, loss_key8 = f"loss_{name}_k0", f"loss_{name}_k8"
        if all(key in record for key in ("loss_baseline_k0", loss_key0)):
            entry["finite_loss_k0_gain"] = finite(
                float(record["loss_baseline_k0"]) - float(record[loss_key0])
            )
        if all(key in record for key in ("loss_baseline_k8", loss_key8)):
            entry["finite_loss_k8_gain"] = finite(
                float(record["loss_baseline_k8"]) - float(record[loss_key8])
            )
        result["candidates"][name] = entry
    return result


def audit_index(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty index")
    missing_by_row = [sorted(set(INDEX_REQUIRED_FIELDS) - set(row)) for row in rows]
    missing = sorted({field for fields in missing_by_row for field in fields})
    return {
        "index": str(path),
        "rows": len(rows),
        "identifiable": not missing,
        "required_fields": list(INDEX_REQUIRED_FIELDS),
        "missing_union": missing,
        "rows_with_missing_fields": sum(bool(fields) for fields in missing_by_row),
        "decision": "READY_FOR_EXACT_REPLAY" if not missing else "UNIDENTIFIABLE",
    }


def self_test() -> None:
    workers, width = 4, 8
    theta0 = np.zeros((workers, width), dtype=np.float32)
    half = np.tile(np.linspace(0.01, 0.08, width, dtype=np.float32), (workers, 1))
    theta_mid, theta_end = theta0 - half, theta0 - np.float32(2.0) * half
    metric = np.ones_like(theta0)
    moment = half.copy()
    weights = np.ones(workers, dtype=np.float64)
    bounds = np.asarray([0, 4, 8], dtype=np.int64)
    baseline, _ = rda(theta0 - theta_end, weights, bounds)
    record = {
        "theta0": theta0,
        "theta_mid": theta_mid,
        "theta_end": theta_end,
        "exp_avg_mid": moment,
        "exp_avg_end": moment,
        "metric_mid": metric,
        "metric_end": metric,
        "step_mid": np.full(workers, 8),
        "step_end": np.full(workers, 16),
        "weights": weights,
        "bounds": bounds,
        "lr_mass_first": np.ones(workers),
        "lr_mass_second": np.ones(workers),
        "baseline_direction": baseline,
    }
    views = adam_views(record, 0.9, 0.999, 1e-8)
    mtrf, _ = candidate_mtrf(record, views, baseline)
    mstp, audit = candidate_mstp(record, views, baseline)
    assert norm(mtrf - baseline) / norm(baseline) < 1e-6
    assert norm(mstp - baseline) / norm(baseline) < 1e-6
    assert audit["stationary_tensors"] == 2
    print("self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--audit-index", type=Path)
    parser.add_argument("--record", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.audit_index is not None:
        result = audit_index(args.audit_index)
    elif args.record:
        result = {
            "schema": "exact_state_midpoint_replay_v1",
            "records": [
                replay_record(path, args.beta1, args.beta2, args.eps)
                for path in args.record
            ],
            "targets_present": {
                key: sum(key in load_record(path) for path in args.record)
                for key in OPTIONAL_TARGETS
            },
        }
    else:
        parser.error("provide --self-test, --audit-index, or one or more --record")
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.out is None:
        print(payload, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)


if __name__ == "__main__":
    main()
