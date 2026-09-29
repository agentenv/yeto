#!/usr/bin/env python3
"""Legacy-vs-ports RL engine equivalence harness (openspec rl-engine-ports 6.1-6.3).

Layered acceptance (spec ``rl-engine-selection`` / "等价性验收"). All thresholds
are module constants, fixed before any ports experiment and written into the
report header; the harness never derives them from the ports data.

Tier 1 -- round 1, strict (strict-avg gated; decoupled reported only).
    Both paths start from the same weights and the same prompts, so for the
    primary seed ``completed_groups``, ``action_tokens`` and ``reward_mean``
    must be equal, and ``|grad_norm_p - grad_norm_l| / |grad_norm_l| <= 3%``.

Tier 2 -- teacher forcing.
    Legacy's recorded round-1 rollouts (``--save-debug-rollout-data`` dumps)
    are replayed through ``yeto.rl.teacher_forcing.replay_generate`` into a
    1-round run on EACH engine. Gated per island: loss
    (``|d| <= max(1e-6, 1e-3 * |loss_l|)``), grad_norm (rel <= 3%) and the
    pre-optimizer LoRA gradient (``round-1.grad.f32``, written by
    ``yeto.rl.grad_audit`` when ``YETO_RL_AUDIT_GRADS=1``) ANCHORED ON AN FP32
    REFERENCE (design D12, second data-driven revision, approved by the user):
    ``yeto.rl.fp32_reference`` recomputes the gradient on CPU in float32 from
    the same initial LoRA (audit ``base.f32``), the same replayed batch and the
    engines' GRPO loss, and ports must be as close to it as legacy is:
    ``relL2(ports, fp32) <= relL2(legacy, fp32) + 0.05`` and
    ``cos(ports, fp32) >= cos(legacy, fp32) - 0.005``. A missing input or a
    failed fp32 computation makes the tier INCOMPLETE. The cross-path
    legacy-vs-ports gradient distance / cosine / worst tensors and the LoRA
    update (rel L2, cosine, sign-flip fraction, norm ratio) are REPORTED only:
    two independent bf16 errors of 8-14 % each compose into a cross-path
    distance that says nothing about which engine is wrong.
    Preconditions: the legacy-TF run reproduces the recording (reward /
    tokens / groups equal to the reference run) and both runs start from the
    same LoRA (audit base rel L2 <= 1e-6, same layout hash).

Tier 3 -- distribution, rounds >= 2.
    Per seed, each of reward_mean, loss, grad_norm, action_tokens is averaged
    over every (island, round >= 2): one value per metric per seed. Legacy vs
    ports seeds (5 each by default) go through an exact two-sided two-sample
    permutation test (all C(10,5)=252 splits, statistic = mean difference);
    Bonferroni alpha = 0.05/4 = 0.0125; any p < 0.0125 FAILS. The report adds
    Cohen's d and the per-seed values, and states the power limit (with 5 vs 5
    the smallest p is 2/252, i.e. only complete separation rejects).

Tier 4 -- hash, within each path only.
    For every policy version applied on >= 2 islands, all islands' post-sync
    hash must agree, for every seed of both paths. Hashes are never compared
    across paths.

Decoupled (``--preset decoupled``) additionally: every adapter directory must
load with standard PEFT, and the relative L2 distance between the legacy and
ports final LoRA (primary seed) is reported next to the legacy seed-to-seed
distance, without a hard threshold (trajectories diverge from round 2).

loss / grad_norm: the rl_local_round events may carry None; values are then
filled from the Miles ``log_utils ... step k: {...}`` lines in
``island-<i>/miles.log`` (the n-th local round <-> the n-th Miles step; one
optimizer step per round), and the source of every filled value is recorded.

Subcommands:
  plan      print the commands of a ``run`` (nothing executed)
  run       launch legacy/ports x seeds (+ teacher forcing) and analyze
  analyze   analyze existing benchmark_rl work dirs (the usual GPU hand-off)
  fake      CPU-only harness proof (real IslandDriver on the fake engine +
            SYNTHETIC legacy tapes; labeled FAKE, never equivalence evidence)
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import math
import os
import random
import re
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Fixed acceptance thresholds (see spec rl-engine-selection and design D12)
# ---------------------------------------------------------------------------
ROUND1_EXACT = ("completed_groups", "action_tokens", "reward_mean")
ROUND1_GRAD_NORM_REL = 0.03
TF_LOSS_ABS = 1e-6
TF_LOSS_REL = 1e-3
TF_GRAD_NORM_REL = 0.03
TF_FP32_REL_L2_MARGIN = 0.05  # relL2(ports, fp32) <= relL2(legacy, fp32) + margin
TF_FP32_COS_MARGIN = 0.005  # cos(ports, fp32) >= cos(legacy, fp32) - margin
TF_GRAD_WORST = 5  # per-tensor rows listed in the report
TF_BASE_REL_L2 = 1e-6
DIST_METRICS = ("reward_mean", "loss", "grad_norm", "action_tokens")
DIST_MIN_SEEDS = 3
DIST_ALPHA = 0.05  # Bonferroni over DIST_METRICS -> 0.0125 each
DEFAULT_SEEDS = (17, 18, 19, 20, 21)

FLOAT_METRICS = ("reward_mean", "loss", "grad_norm")
COUNT_METRICS = ("completed_groups", "action_tokens")
HASH_METRIC = "post_sync_hash"
ALL_METRICS = FLOAT_METRICS + COUNT_METRICS

DEFAULT_LAUNCH_CMD = (
    "python scripts/benchmark_rl.py {launch_args} --seeds {seed} --rl-engine {engine} "
    "--work-dir {run_dir}/work --report-dir {run_dir}/report"
)
TF_GENERATE = "yeto.rl.teacher_forcing.replay_generate"
REPLAY_ENV = "YETO_RL_REPLAY_ROLLOUTS"
GRAD_AUDIT_ENV = "YETO_RL_AUDIT_GRADS"  # == yeto.rl.grad_audit.GRAD_AUDIT_ENV

Key = tuple[int, int]  # (island_id, round)


# ---------------------------------------------------------------------------
# Event + log extraction
# ---------------------------------------------------------------------------
def read_events(paths: list[Path]) -> list[dict[str, Any]]:
    events = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def extract_rounds(events: list[dict[str, Any]]) -> dict[Key, dict[str, Any]]:
    """Per-(island, round) metrics from rl_local_round + rl_policy_apply."""

    rounds: dict[Key, dict[str, Any]] = {}
    hashes = extract_hashes(events)
    for e in events:
        if e.get("event") == "rl_local_round":
            island, r = int(e.get("island_id", 0)), int(e["local_round_id"])
            row = rounds.setdefault((island, r), {})
            for m in ALL_METRICS:
                row[m] = e.get(m, e.get(f"rl/{m}"))
    for key, row in rounds.items():
        row[HASH_METRIC] = hashes.get(key)
    return rounds


def extract_hashes(events: list[dict[str, Any]]) -> dict[Key, str]:
    """(island, policy_version) -> global policy hash applied at that version.

    Decoupled partial fragment applies mix global fragments into local progress,
    so islands legitimately differ there; they are skipped except the island's
    last applied version (the final cut)."""

    applies = [e for e in events
               if e.get("event") == "rl_policy_apply" and e.get("sync/global_policy_hash")]
    last: dict[int, int] = {}
    for e in applies:
        i = int(e.get("island_id", 0))
        last[i] = max(last.get(i, -1), int(e["policy_version"]))
    return {(int(e.get("island_id", 0)), int(e["policy_version"])): e["sync/global_policy_hash"]
            for e in applies
            if not e.get("partial_fragment_apply")
            or int(e["policy_version"]) == last[int(e.get("island_id", 0))]}


_STEP = re.compile(r"log_utils\.py:\d+ - step (\d+): (\{.*\})")


def miles_steps(log_text: str) -> dict[int, dict[str, Any]]:
    """Miles trainer step logs (rank-0 MegatronTrainRayActor), keyed by step."""

    out: dict[int, dict[str, Any]] = {}
    for line in log_text.splitlines():
        m = _STEP.search(line)
        if m and "MegatronTrainRayActor" in line:
            try:
                out[int(m.group(1))] = ast.literal_eval(m.group(2))
            except (ValueError, SyntaxError):
                continue
    return out


def fill_from_miles(rounds: dict[Key, dict[str, Any]], island: int,
                    steps: dict[int, dict[str, Any]]) -> dict[str, str]:
    """Fill None loss/grad_norm of ``island`` from Miles steps (n-th round <-> n-th step)."""

    sources: dict[str, str] = {}
    ordered = sorted(r for (i, r) in rounds if i == island)
    step_ids = sorted(steps)
    for n, r in enumerate(ordered):
        if n >= len(step_ids):
            break
        step = steps[step_ids[n]]
        row = rounds[(island, r)]
        for m, key in (("loss", "train/loss"), ("grad_norm", "train/grad_norm")):
            if row.get(m) is None and key in step:
                row[m] = float(step[key])
                sources[f"{island}/{r}/{m}"] = f"miles.log step {step_ids[n]}"
    return sources


def load_arm(arm_dir: str | Path) -> dict[str, Any]:
    """One benchmark_rl arm dir (``.../seed-N/yeto-<arm>-m<k>``)."""

    arm = Path(arm_dir)
    tapes = sorted(arm.glob("island-*/events.jsonl"))
    if not tapes:
        raise FileNotFoundError(f"no island-*/events.jsonl under {arm}")
    events = read_events(tapes)
    rounds = extract_rounds(events)
    sources: dict[str, str] = {}
    for isl in sorted(arm.glob("island-*")):
        log = isl / "miles.log"
        if log.is_file():
            island = int(isl.name.split("-")[1])
            sources.update(fill_from_miles(
                rounds, island, miles_steps(log.read_text(errors="ignore"))))
    return {"dir": str(arm), "rounds": rounds, "hashes": extract_hashes(events),
            "filled_from_miles_log": sources}


_SEED_DIR = re.compile(r"seed-(\d+)$")


def discover_arms(path: str | Path) -> dict[int, Path]:
    """seed -> arm dir. ``path`` may be an arm dir, a seed dir, a work dir or a run dir."""

    root = Path(path)
    if list(root.glob("island-*/events.jsonl")):
        m = _SEED_DIR.search(root.parent.name)
        return {int(m.group(1)) if m else -1: root}
    found: dict[int, Path] = {}
    for tape in sorted(root.glob("**/island-*/events.jsonl")):
        arm = tape.parent.parent
        m = _SEED_DIR.search(arm.parent.name)
        if not m:
            continue
        seed = int(m.group(1))
        if seed in found and found[seed] != arm:
            raise ValueError(f"two arms for seed {seed} under {root}: {found[seed]} and {arm}")
        found[seed] = arm
    if not found:
        raise FileNotFoundError(f"no benchmark arm (seed-*/*/island-*/events.jsonl) under {root}")
    return found


def load_runs(paths: list[str]) -> dict[int, dict[str, Any]]:
    runs: dict[int, dict[str, Any]] = {}
    for p in paths:
        for seed, arm in discover_arms(p).items():
            if seed in runs:
                raise ValueError(f"seed {seed} given twice")
            runs[seed] = load_arm(arm)
    return runs


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------
def rel_diff(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    a, b = float(a), float(b)
    if a == b:
        return 0.0
    return abs(b - a) / abs(a) if a != 0 else math.inf


def read_f32(path: str | Path):
    import numpy as np

    return np.fromfile(str(path), dtype="<f4")


def rel_l2(ref, other) -> float:
    import numpy as np

    ref = np.asarray(ref, dtype=np.float64)
    other = np.asarray(other, dtype=np.float64)
    if ref.shape != other.shape:
        raise ValueError(f"shape mismatch {ref.shape} vs {other.shape}")
    denom = float(np.linalg.norm(ref))
    num = float(np.linalg.norm(other - ref))
    if denom == 0.0:
        return 0.0 if num == 0.0 else math.inf
    return num / denom


def cosine(a, b) -> float | None:
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return None if na == 0 or nb == 0 else float(a @ b) / (na * nb)


# ---------------------------------------------------------------------------
# Tier 1: round 1 strict
# ---------------------------------------------------------------------------
def round1_check(legacy: dict[Key, dict[str, Any]], ports: dict[Key, dict[str, Any]]) -> dict:
    rows = []
    islands = sorted({i for (i, r) in set(legacy) | set(ports) if r == 1})
    for i in islands:
        a, b = legacy.get((i, 1)), ports.get((i, 1))
        if a is None or b is None:
            rows.append({"island": i, "metric": "round 1 present", "legacy": a is not None,
                         "ports": b is not None, "rule": "both", "pass": False})
            continue
        for m in ROUND1_EXACT:
            ok = a.get(m) is not None and a.get(m) == b.get(m)
            rows.append({"island": i, "metric": m, "legacy": a.get(m), "ports": b.get(m),
                         "rule": "exact", "pass": ok})
        d = rel_diff(a.get("grad_norm"), b.get("grad_norm"))
        rows.append({"island": i, "metric": "grad_norm", "legacy": a.get("grad_norm"),
                     "ports": b.get("grad_norm"), "rel_diff": d,
                     "rule": f"rel <= {ROUND1_GRAD_NORM_REL}",
                     "pass": d is not None and d <= ROUND1_GRAD_NORM_REL})
    return {"passed": bool(rows) and all(r["pass"] for r in rows), "rows": rows}


# ---------------------------------------------------------------------------
# Tier 3: distribution over seeds
# ---------------------------------------------------------------------------
def seed_summary(rounds: dict[Key, dict], *, min_round: int = 2,
                 metrics=DIST_METRICS) -> dict[str, float | None]:
    """One value per metric for a seed: mean over every (island, round >= min_round)."""

    out: dict[str, float | None] = {}
    keys = sorted(k for k in rounds if k[1] >= min_round)
    for m in metrics:
        vals = [rounds[k].get(m) for k in keys]
        out[m] = (None if not vals or any(v is None for v in vals)
                  else math.fsum(float(v) for v in vals) / len(vals))
    return out


def permutation_test(x: list[float], y: list[float]) -> float:
    """Exact two-sided two-sample permutation p-value, statistic = mean(y) - mean(x)."""

    from itertools import combinations

    pooled = list(x) + list(y)
    n, total = len(pooled), math.fsum(pooled)
    obs = abs(math.fsum(y) / len(y) - math.fsum(x) / len(x))
    tol = 1e-12 * max(1.0, max(abs(v) for v in pooled))
    hits = count = 0
    for idx in combinations(range(n), len(y)):
        sy = math.fsum(pooled[i] for i in idx)
        d = abs(sy / len(y) - (total - sy) / len(x))
        hits += d >= obs - tol
        count += 1
    return hits / count


def effect_size(x: list[float], y: list[float]) -> float | None:
    """(mean_y - mean_x) / pooled SD (Cohen's d)."""

    nx, ny = len(x), len(y)
    mx, my = math.fsum(x) / nx, math.fsum(y) / ny
    var = (math.fsum((v - mx) ** 2 for v in x) + math.fsum((v - my) ** 2 for v in y)) / (nx + ny - 2)
    if var == 0:
        return 0.0 if mx == my else math.inf
    return (my - mx) / math.sqrt(var)


def power_limits(n_x: int, n_y: int, alpha: float) -> dict[str, Any]:
    """Smallest attainable p, and the Gaussian shift (in SD) detected with 80% power."""

    n_splits = math.comb(n_x + n_y, n_y)
    min_p = 2 / n_splits if n_x == n_y else 1 / n_splits
    rng = random.Random(0)
    d80 = None
    if min_p < alpha:
        for tenth in range(5, 81):
            d, trials, hits = tenth / 10, 4000, 0
            for _ in range(trials):
                xs = [rng.gauss(0, 1) for _ in range(n_x)]
                ys = [rng.gauss(d, 1) for _ in range(n_y)]
                hits += min(ys) > max(xs) or max(ys) < min(xs)
            if hits / trials >= 0.8:
                d80 = d
                break
    return {"splits": n_splits, "min_p": min_p,
            "reject_only_if_fully_separated": n_x == n_y == 5,
            "min_effect_80pct_power_sd": d80}


def distribution_check(legacy_by_seed: dict[int, dict[Key, dict]],
                       ports_by_seed: dict[int, dict[Key, dict]], *,
                       min_round: int = 2, metrics=DIST_METRICS) -> dict:
    if len(legacy_by_seed) < DIST_MIN_SEEDS or len(ports_by_seed) < DIST_MIN_SEEDS:
        return {"passed": False, "rows": [], "incomplete": True, "error":
                f"need >= {DIST_MIN_SEEDS} seeds per path (legacy {len(legacy_by_seed)}, "
                f"ports {len(ports_by_seed)})"}
    alpha = DIST_ALPHA / len(metrics)
    ls = {s: seed_summary(r, min_round=min_round, metrics=metrics) for s, r in legacy_by_seed.items()}
    ps = {s: seed_summary(r, min_round=min_round, metrics=metrics) for s, r in ports_by_seed.items()}
    rows = []
    for m in metrics:
        lv = {s: v[m] for s, v in sorted(ls.items())}
        pv = {s: v[m] for s, v in sorted(ps.items())}
        row = {"metric": m, "legacy_values": lv, "ports_values": pv, "alpha": alpha}
        if any(v is None for v in list(lv.values()) + list(pv.values())):
            row.update({"pass": False, "note": "missing value (no rounds >= 2 or None) in some seed"})
        else:
            x, y = list(lv.values()), list(pv.values())
            p = permutation_test(x, y)
            row.update({"legacy_mean": math.fsum(x) / len(x), "ports_mean": math.fsum(y) / len(y),
                        "p_value": p, "effect_size": effect_size(x, y), "pass": p >= alpha})
        rows.append(row)
    stats = {"n_legacy_seeds": len(ls), "n_ports_seeds": len(ps), "alpha_family": DIST_ALPHA,
             "alpha_per_metric": alpha, **power_limits(len(ls), len(ps), alpha)}
    return {"passed": bool(rows) and all(r["pass"] for r in rows), "rows": rows, "stats": stats}


# ---------------------------------------------------------------------------
# Tier 4: within-path hash consistency
# ---------------------------------------------------------------------------
def hash_check(hashes: dict[Key, str]) -> dict:
    by_version: dict[int, dict[int, str]] = {}
    for (island, version), h in hashes.items():
        by_version.setdefault(version, {})[island] = h
    rows = []
    for version, per in sorted(by_version.items()):
        if len(per) < 2:
            continue
        rows.append({"version": version, "hashes": per, "pass": len(set(per.values())) == 1})
    return {"passed": bool(rows) and all(r["pass"] for r in rows), "rows": rows}


# ---------------------------------------------------------------------------
# Tier 2: teacher forcing
# ---------------------------------------------------------------------------
def load_tf_arm(arm_dir: str | Path, *, round_id: int = 1) -> dict[int, dict[str, Any]]:
    """Per island: round-1 metrics (+ Miles log fill) and the round audit files."""

    arm = load_arm(arm_dir)
    out: dict[int, dict[str, Any]] = {}
    for (island, r), row in arm["rounds"].items():
        if r != round_id:
            continue
        audit = Path(arm["dir"]) / f"island-{island}" / "audit"
        stem = f"round-{round_id:08d}"
        meta_path = audit / f"{stem}.json"
        out[island] = {**{m: row.get(m) for m in ALL_METRICS},
                       "audit_base": str(audit / f"{stem}.base.f32"),
                       "audit_delta": str(audit / f"{stem}.delta.f32"),
                       "audit_grad": str(audit / f"{stem}.grad.f32"),
                       "audit_meta": (json.loads(meta_path.read_text())
                                      if meta_path.is_file() else None)}
    return out


def read_grad(path: str | Path) -> tuple[dict[str, Any], Any]:
    """(index, flat f32) of a ``*.grad.f32`` written by ``yeto.rl.grad_audit``."""
    import numpy as np

    path = Path(path)
    index = json.loads(path.with_name(path.name[: -len(".f32")] + ".json").read_text())
    flat = np.fromfile(str(path), dtype="<f4")
    if flat.size != index["numel"]:
        raise ValueError(f"{path}: {flat.size} values, index says {index['numel']}")
    return index, flat


def grad_tensors(index: dict[str, Any], flat) -> dict[str, Any]:
    return {s["name"]: flat[s["offset"]: s["offset"] + s["numel"]] for s in index["specs"]}


def grad_compare(legacy_path: str | Path, ports_path: str | Path, *,
                 worst: int = TF_GRAD_WORST) -> dict[str, Any]:
    """Concatenated (canonical name order) rel L2 / cosine and the worst tensors."""
    import numpy as np

    li, lf = read_grad(legacy_path)
    pi, pf = read_grad(ports_path)
    lt, pt = grad_tensors(li, lf), grad_tensors(pi, pf)
    if set(lt) != set(pt):
        return {"error": "tensor names differ", "only_legacy": sorted(set(lt) - set(pt))[:5],
                "only_ports": sorted(set(pt) - set(lt))[:5]}
    names = sorted(lt)
    shapes = {n for n in names if [s for s in li["specs"] if s["name"] == n][0]["shape"]
              != [s for s in pi["specs"] if s["name"] == n][0]["shape"]}
    if shapes:
        return {"error": "tensor shapes differ", "tensors": sorted(shapes)[:5]}
    gl = np.concatenate([lt[n] for n in names])
    gp = np.concatenate([pt[n] for n in names])
    per = sorted(({"name": n, "rel_l2": rel_l2(lt[n], pt[n]), "cosine": cosine(lt[n], pt[n])}
                  for n in names), key=lambda r: -r["rel_l2"])
    meta = {k: (li.get(k), pi.get(k)) for k in ("grad_norm_l2", "clip_coefficient",
                                                 "optimizer_grad_norm", "grad_source", "grad_dtype")}
    return {"rel_l2": rel_l2(gl, gp), "cosine": cosine(gl, gp), "tensors": len(names),
            "numel": int(gl.size), "worst": per[:worst], "meta": meta}


def update_stats(dl, dp) -> dict[str, Any]:
    """Reported-only LoRA update comparison (design D12)."""
    import numpy as np

    dl = np.asarray(dl, dtype=np.float64)
    dp = np.asarray(dp, dtype=np.float64)
    nl = float(np.linalg.norm(dl))
    return {"rel_l2": rel_l2(dl, dp), "cosine": cosine(dl, dp),
            "sign_flip": float(np.mean(np.sign(dl) != np.sign(dp))) if dl.size else None,
            "norm_ratio": float(np.linalg.norm(dp)) / nl if nl else None}


def fp32_anchor_compare(legacy_path: str | Path, ports_path: str | Path,
                        ref: dict[str, Any]) -> dict[str, Any]:
    """Legacy and ports LoRA gradients vs the fp32 reference (canonical name order)."""
    import numpy as np

    order = ref["order"]
    out: dict[str, Any] = {}
    for label, path in (("legacy", legacy_path), ("ports", ports_path)):
        idx, flat = read_grad(path)
        t = grad_tensors(idx, flat)
        if set(t) != set(order):
            return {"error": f"{label} gradient tensor names differ from the fp32 reference"}
        shapes = {s["name"]: s["shape"] for s in idx["specs"]}
        bad = [n for n in order if list(shapes[n]) != list(ref["shapes"][n])]
        if bad:
            return {"error": f"{label} gradient shapes differ from the fp32 reference: {bad[:3]}"}
        g = np.concatenate([t[n] for n in order])
        out[label] = {"rel_l2": rel_l2(ref["flat"], g), "cosine": cosine(ref["flat"], g),
                      "norm": float(np.linalg.norm(g.astype(np.float64)))}
    out["fp32_norm"] = ref["grad_norm"]
    return out


def fp32_gate(anchor: dict[str, Any]) -> dict[str, Any]:
    """Option A (design D12): ports at least as close to fp32 as legacy, within margins."""
    lg, pt = anchor["legacy"], anchor["ports"]
    rel_ok = pt["rel_l2"] <= lg["rel_l2"] + TF_FP32_REL_L2_MARGIN
    cos_ok = (pt["cosine"] is not None and lg["cosine"] is not None
              and pt["cosine"] >= lg["cosine"] - TF_FP32_COS_MARGIN)
    return {"rel_ok": rel_ok, "cos_ok": cos_ok}


def teacher_forcing_check(reference: dict[int, dict], legacy_tf: dict[int, dict],
                          ports_tf: dict[int, dict], *, require_update: bool = True,
                          fp32: dict[int, dict] | None = None) -> dict:
    """``require_update=False`` (decoupled) reports a missing gradient/round audit instead
    of failing; the LoRA-gradient gate then comes from the strict-avg TF run.

    ``fp32``: island -> ``yeto.rl.fp32_reference.compute_island`` result, or
    ``{"error": ...}`` when an input was missing / the computation failed
    (-> INCOMPLETE). ``None`` means the reference was not computed at all."""
    rows = []

    def add(island, check, legacy, ports, value, rule, ok, gate=True, **extra):
        rows.append({"island": island, "check": check, "legacy": legacy, "ports": ports,
                     "value": value, "rule": rule, "gate": gate, "pass": ok, **extra})

    islands = sorted(set(reference) | set(legacy_tf) | set(ports_tf))
    for i in islands:
        ref, lg, pt = reference.get(i), legacy_tf.get(i), ports_tf.get(i)
        if ref is None or lg is None or pt is None:
            add(i, "island present in reference/legacy-TF/ports-TF", ref is not None,
                pt is not None, None, "all three", False)
            continue
        # Preconditions: replay fidelity (both TF runs saw the recorded batch).
        for m in ROUND1_EXACT:
            add(i, f"fidelity legacy-TF {m} == reference", ref.get(m), lg.get(m), None,
                "exact", ref.get(m) is not None and ref.get(m) == lg.get(m))
            add(i, f"fidelity ports-TF {m} == reference", ref.get(m), pt.get(m), None,
                "exact", ref.get(m) is not None and ref.get(m) == pt.get(m))
        # Informational: legacy replay vs the original legacy step (legacy is deterministic).
        add(i, "legacy-TF grad_norm vs reference (info)", ref.get("grad_norm"),
            lg.get("grad_norm"), rel_diff(ref.get("grad_norm"), lg.get("grad_norm")),
            "reported", None, gate=False)
        # Gates.
        la, pa = lg.get("loss"), pt.get("loss")
        if la is None or pa is None:
            add(i, "loss", la, pa, None, "present", False)
        else:
            d = abs(float(pa) - float(la))
            bound = max(TF_LOSS_ABS, TF_LOSS_REL * abs(float(la)))
            add(i, "loss", la, pa, d, f"|d| <= max({TF_LOSS_ABS}, {TF_LOSS_REL}*|loss_l|)",
                d <= bound)
        d = rel_diff(lg.get("grad_norm"), pt.get("grad_norm"))
        add(i, "grad_norm", lg.get("grad_norm"), pt.get("grad_norm"), d,
            f"rel <= {TF_GRAD_NORM_REL}", d is not None and d <= TF_GRAD_NORM_REL)
        grads = [lg.get("audit_grad"), pt.get("audit_grad")]
        if not all(g and Path(g).is_file() for g in grads):
            add(i, "LoRA gradient (grad audit f32)", bool(grads[0] and Path(grads[0]).is_file()),
                bool(grads[1] and Path(grads[1]).is_file()), None,
                (f"grad audit present ({GRAD_AUDIT_ENV}=1)" if require_update
                 else "not available (see strict-avg TF)"),
                False if require_update else None, gate=require_update)
        else:
            ref32 = (fp32 or {}).get(i)
            if ref32 is None or "error" in ref32:
                why = ("not computed" if ref32 is None else ref32["error"])
                add(i, "fp32 reference gradient", None, None, None,
                    "fp32 reference computed from recorded inputs", False, incomplete=True,
                    detail=why)
            else:
                a = fp32_anchor_compare(grads[0], grads[1], ref32)
                if "error" in a:
                    add(i, "LoRA gradient tensors aligned with fp32 reference", None, None, None,
                        "same names/shapes", False, detail=a["error"])
                else:
                    gate = fp32_gate(a)
                    lr, pr = a["legacy"]["rel_l2"], a["ports"]["rel_l2"]
                    lc, pc = a["legacy"]["cosine"], a["ports"]["cosine"]
                    add(i, "LoRA gradient relL2 vs fp32", lr, pr, pr - lr,
                        f"ports <= legacy + {TF_FP32_REL_L2_MARGIN}", gate["rel_ok"])
                    add(i, "LoRA gradient cosine vs fp32", lc, pc,
                        None if pc is None or lc is None else pc - lc,
                        f"ports >= legacy - {TF_FP32_COS_MARGIN}", gate["cos_ok"])
                    add(i, "LoRA gradient norm legacy/ports/fp32 (info)", a["legacy"]["norm"],
                        a["ports"]["norm"], a["fp32_norm"], "reported", None, gate=False)
                    add(i, "fp32 reference loss (info)", None, None, ref32.get("loss"),
                        "reported", None, gate=False)
            g = grad_compare(grads[0], grads[1])
            if "error" in g:
                add(i, "LoRA gradient tensors aligned", None, None, None, "same names/shapes",
                    False, detail=g)
            else:
                add(i, "cross-path LoRA gradient rel L2 (info)", None, None, g["rel_l2"],
                    "reported", None, gate=False, cosine=g["cosine"])
                for k, (x, y) in g["meta"].items():
                    add(i, f"grad audit {k} (info)", x, y, None, "reported", None, gate=False)
                for w in g["worst"]:
                    add(i, f"worst tensor {w['name']} (info)", None, None, w["rel_l2"],
                        "reported", None, gate=False, cosine=w["cosine"])
        lm, pm = lg.get("audit_meta"), pt.get("audit_meta")
        files = [lg["audit_base"], lg["audit_delta"], pt["audit_base"], pt["audit_delta"]]
        if lm is None or pm is None or not all(Path(f).is_file() for f in files):
            add(i, "LoRA base/update (audit f32)", lm is not None, pm is not None, None,
                "audit present" if require_update else "not available (see strict-avg TF)",
                False if require_update else None, gate=require_update)
            continue
        add(i, "layout_hash equal", lm.get("layout_hash"), pm.get("layout_hash"), None,
            "exact", lm.get("layout_hash") == pm.get("layout_hash"))
        base = rel_l2(read_f32(lg["audit_base"]), read_f32(pt["audit_base"]))
        add(i, "initial LoRA rel L2", None, None, base, f"<= {TF_BASE_REL_L2}",
            base <= TF_BASE_REL_L2)
        u = update_stats(read_f32(lg["audit_delta"]), read_f32(pt["audit_delta"]))
        add(i, "LoRA update rel L2 (info)", None, None, u["rel_l2"], "reported", None,
            gate=False, cosine=u["cosine"])
        add(i, "LoRA update sign-flip fraction (info)", None, None, u["sign_flip"], "reported",
            None, gate=False)
        add(i, "LoRA update norm ratio ports/legacy (info)", None, None, u["norm_ratio"],
            "reported", None, gate=False)
    gated = [r for r in rows if r["gate"]]
    failed = [r for r in gated if not r["pass"]]
    return {"passed": bool(gated) and not failed, "rows": rows,
            "incomplete": bool(failed) and all(r.get("incomplete") for r in failed),
            "fp32_provenance": {str(k): v.get("provenance") if "error" not in v else v
                                for k, v in (fp32 or {}).items()}}


# ---------------------------------------------------------------------------
# Decoupled: PEFT load + final LoRA distance
# ---------------------------------------------------------------------------
def load_adapter(adapter_dir: str | Path) -> dict[str, Any]:
    from safetensors.numpy import load_file

    path = Path(adapter_dir) / "adapter_model.safetensors"
    return load_file(str(path))


def peft_check(adapter_dir: str | Path, *, base_model: str | None = None) -> dict:
    """Standard PEFT can read the adapter (config + weights; full model if ``base_model``)."""

    adapter_dir = Path(adapter_dir)
    try:
        from peft import PeftConfig

        cfg = PeftConfig.from_pretrained(str(adapter_dir))
        tensors = load_adapter(adapter_dir)
        if not tensors:
            raise ValueError("adapter has no tensors")
        if base_model:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True)
            PeftModel.from_pretrained(model, str(adapter_dir))
        return {"dir": str(adapter_dir), "pass": True, "peft_type": str(cfg.peft_type),
                "tensors": len(tensors), "full_model_load": bool(base_model)}
    except Exception as error:  # noqa: BLE001 - reported
        return {"dir": str(adapter_dir), "pass": False, "error": f"{type(error).__name__}: {error}"}


def adapter_rel_l2(ref_dir: str | Path, other_dir: str | Path) -> float:
    import numpy as np

    a, b = load_adapter(ref_dir), load_adapter(other_dir)
    if set(a) != set(b):
        raise ValueError("adapters have different tensor names")
    num = sum(float(np.sum((b[k].astype(np.float64) - a[k].astype(np.float64)) ** 2)) for k in a)
    den = sum(float(np.sum(a[k].astype(np.float64) ** 2)) for k in a)
    return math.sqrt(num) / math.sqrt(den) if den else math.inf


def decoupled_extras(legacy_runs: dict[int, dict], ports_runs: dict[int, dict], *,
                     primary_seed: int, base_model: str | None) -> dict:
    peft = [peft_check(Path(run["dir"]) / "adapter", base_model=base_model)
            for runs in (legacy_runs, ports_runs) for run in runs.values()]
    out: dict[str, Any] = {"peft": peft, "passed": bool(peft) and all(p["pass"] for p in peft)}
    try:
        lp = Path(legacy_runs[primary_seed]["dir"]) / "adapter"
        out["final_lora_rel_l2_legacy_vs_ports"] = adapter_rel_l2(
            lp, Path(ports_runs[primary_seed]["dir"]) / "adapter")
        others = [s for s in sorted(legacy_runs) if s != primary_seed]
        if others:
            out["final_lora_rel_l2_legacy_seed_to_seed"] = {
                f"{primary_seed}->{s}": adapter_rel_l2(lp, Path(legacy_runs[s]["dir"]) / "adapter")
                for s in others}
    except Exception as error:  # noqa: BLE001 - reported, not gated
        out["final_lora_error"] = f"{type(error).__name__}: {error}"
    return out


# ---------------------------------------------------------------------------
# Verdict + report
# ---------------------------------------------------------------------------
def evaluate(*, preset: str, legacy_runs: dict[int, dict], ports_runs: dict[int, dict],
             primary_seed: int, tf: dict | None, extras: dict | None = None) -> dict:
    tiers: dict[str, Any] = {}
    if primary_seed in legacy_runs and primary_seed in ports_runs:
        r1 = round1_check(legacy_runs[primary_seed]["rounds"], ports_runs[primary_seed]["rounds"])
    else:
        r1 = {"passed": False, "rows": [], "error": f"primary seed {primary_seed} missing"}
    r1["gated"] = preset == "strict-avg"
    tiers["1_round1"] = r1
    tiers["2_teacher_forcing"] = (tf if tf is not None else
                                  {"passed": False, "rows": [], "error": "not run", "incomplete": True,
                                   "gated": True})
    tiers["2_teacher_forcing"].setdefault("gated", True)
    dist = distribution_check({s: r["rounds"] for s, r in legacy_runs.items()},
                              {s: r["rounds"] for s, r in ports_runs.items()})
    dist["gated"] = True
    tiers["3_distribution"] = dist
    hash_rows, hash_ok = [], True
    for path, runs in (("legacy", legacy_runs), ("ports", ports_runs)):
        for seed, run in sorted(runs.items()):
            h = hash_check(run["hashes"])
            hash_ok &= h["passed"]
            hash_rows += [{"path": path, "seed": seed, **row} for row in h["rows"]] or [
                {"path": path, "seed": seed, "version": None, "hashes": {}, "pass": False}]
    tiers["4_hash"] = {"passed": hash_ok and bool(hash_rows), "rows": hash_rows, "gated": True}
    if preset == "decoupled":
        extras = extras or {"passed": False, "error": "not run", "incomplete": True}
        extras["gated"] = True
        tiers["5_peft"] = extras
    gated = [t for t in tiers.values() if t.get("gated")]
    passed = all(t["passed"] for t in gated)
    failed = [t for t in gated if not t["passed"] and not t.get("incomplete")]
    verdict = "PASS" if passed else ("FAIL" if failed else "INCOMPLETE")
    return {"verdict": verdict, "passed": passed, "tiers": tiers}


def thresholds() -> dict[str, Any]:
    return {"round1_exact": list(ROUND1_EXACT), "round1_grad_norm_rel": ROUND1_GRAD_NORM_REL,
            "tf_loss": f"|d| <= max({TF_LOSS_ABS}, {TF_LOSS_REL}*|loss_legacy|)",
            "tf_grad_norm_rel": TF_GRAD_NORM_REL,
            "tf_lora_grad": (f"pre-optimizer LoRA gradient (all tensors concatenated) anchored on "
                             f"a CPU fp32 reference: relL2(ports, fp32) <= relL2(legacy, fp32) + "
                             f"{TF_FP32_REL_L2_MARGIN} and cos(ports, fp32) >= cos(legacy, fp32) - "
                             f"{TF_FP32_COS_MARGIN}; fp32 missing/failed -> INCOMPLETE; "
                             f"cross-path legacy-vs-ports distance reported only"),
            "tf_lora_update": "reported only (rel L2, cosine, sign-flip fraction, norm ratio)",
            "tf_initial_lora_rel_l2": TF_BASE_REL_L2,
            "distribution": (f"per seed: mean over (island, round >= 2) of {list(DIST_METRICS)}; "
                             f"exact two-sided permutation test legacy vs ports (mean diff), "
                             f"FAIL if p < {DIST_ALPHA}/{len(DIST_METRICS)}"),
            "hash": "within path: all islands equal per policy version; never cross-path"}


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def render_report(meta: dict, result: dict) -> str:
    L = []
    if meta.get("fake"):
        L += ["# FAKE REPORT -- CPU fake engine + SYNTHETIC legacy tapes",
              "# This validates the harness only; it is NOT equivalence evidence.", ""]
    L += ["# RL engine equivalence report (legacy vs ports, layered)", ""]
    for k, v in meta.items():
        L.append(f"- {k}: {v}")
    L += ["", "## Thresholds (fixed in scripts/rl_engine_equivalence.py before the experiment)", ""]
    L += [f"- {k}: {v}" for k, v in thresholds().items()]
    verdict = result["verdict"] + (" (FAKE)" if meta.get("fake") else "")
    L += ["", f"## Result: {verdict}", ""]
    for name, t in result["tiers"].items():
        mark = "PASS" if t["passed"] else "FAIL"
        L.append(f"- tier {name}: {mark}{'' if t.get('gated') else ' (reported, not gated)'}"
                 f"{' -- ' + t['error'] if t.get('error') else ''}")
    t = result["tiers"]["1_round1"]
    L += ["", "## Tier 1: round 1 (primary seed)", "",
          "| island | metric | legacy | ports | rel diff | rule | pass |", "|---|---|---|---|---|---|---|"]
    for r in t["rows"]:
        L.append(f"| {r['island']} | {r['metric']} | {_fmt(r['legacy'])} | {_fmt(r['ports'])} | "
                 f"{_fmt(r.get('rel_diff', '-'))} | {r['rule']} | {'ok' if r['pass'] else 'FAIL'} |")
    t = result["tiers"]["2_teacher_forcing"]
    L += ["", "## Tier 2: teacher forcing (legacy round-1 rollouts replayed on both engines)", "",
          "| island | check | legacy | ports | value | rule | pass |", "|---|---|---|---|---|---|---|"]
    for r in t["rows"]:
        mark = "info" if not r["gate"] else ("ok" if r["pass"] else "FAIL")
        extra = f" (cos {r['cosine']:.6f})" if r.get("cosine") is not None else ""
        L.append(f"| {r['island']} | {r['check']} | {_fmt(r['legacy'])} | {_fmt(r['ports'])} | "
                 f"{_fmt(r['value'])}{extra} | {r['rule']} | {mark} |")
    for isl, prov in sorted((t.get("fp32_provenance") or {}).items()):
        L += ["", f"### fp32 reference inputs, island {isl}", ""]
        if not prov or "error" in prov:
            L.append(f"- INCOMPLETE: {(prov or {}).get('error', 'not computed')}")
            continue
        L.append(f"- replay batch: {prov['replay']['path']} sha256 {prov['replay']['sha256']}")
        L.append(f"- initial LoRA: {prov['initial_lora']['path']} sha256 "
                 f"{prov['initial_lora']['sha256']} (layout {prov['initial_lora']['layout_hash']})")
        m = prov["model"]
        L.append(f"- model: {m['id']}@{m['revision']} ({m['dir']})")
        for name, h in m["files_sha256"].items():
            L.append(f"  - {name} sha256 {h}")
        L.append(f"- loss: {prov['loss']}; {prov['dtype']} on {prov['device']}, torch {prov['torch']}")
    t = result["tiers"]["3_distribution"]
    L += ["", "## Tier 3: distribution over seeds (rounds >= 2, per-seed means, permutation test)", ""]
    if t.get("stats"):
        s = t["stats"]
        L += [f"- seeds: legacy {s['n_legacy_seeds']}, ports {s['n_ports_seeds']}; exact enumeration of "
              f"{s['splits']} splits; alpha {s['alpha_family']} Bonferroni -> {s['alpha_per_metric']:.4g} per metric",
              f"- power limitation: smallest attainable p = {s['min_p']:.4g}"
              + ("; with 5 vs 5 the test rejects only when the two groups are completely separated"
                 if s["reject_only_if_fully_separated"] else "")
              + (f"; a Gaussian mean shift of about {s['min_effect_80pct_power_sd']} pooled SD is needed for "
                 "80% power, so smaller real differences usually PASS" if s["min_effect_80pct_power_sd"]
                 else "; this sample size cannot reach the per-metric alpha at all"), ""]
    L += ["| metric | legacy per seed | ports per seed | legacy mean | ports mean | effect size | p | pass |",
          "|---|---|---|---|---|---|---|---|"]
    for r in t["rows"]:
        lv = ", ".join(f"{k}:{_fmt(v)}" for k, v in r["legacy_values"].items())
        pv = ", ".join(f"{k}:{_fmt(v)}" for k, v in r["ports_values"].items())
        L.append(f"| {r['metric']} | {lv} | {pv} | {_fmt(r.get('legacy_mean'))} | "
                 f"{_fmt(r.get('ports_mean'))} | {_fmt(r.get('effect_size'))} | {_fmt(r.get('p_value'))} | "
                 f"{'ok' if r['pass'] else 'FAIL'}{' (' + r['note'] + ')' if r.get('note') else ''} |")
    t = result["tiers"]["4_hash"]
    L += ["", "## Tier 4: within-path hash consistency", "",
          "| path | seed | policy version | islands agree |", "|---|---|---|---|"]
    for r in t["rows"]:
        L.append(f"| {r['path']} | {r['seed']} | {r['version']} | {'ok' if r['pass'] else 'FAIL'} |")
    if "5_peft" in result["tiers"]:
        t = result["tiers"]["5_peft"]
        L += ["", "## Decoupled: PEFT load and final LoRA distance", ""]
        for p in t.get("peft", []):
            L.append(f"- {p['dir']}: {'ok' if p['pass'] else 'FAIL ' + p.get('error', '')}")
        for k in ("final_lora_rel_l2_legacy_vs_ports", "final_lora_rel_l2_legacy_seed_to_seed",
                  "final_lora_error"):
            if k in t:
                L.append(f"- {k}: {t[k]}")
        L.append("- final LoRA distance is reported without a hard threshold: sampling diverges "
                 "from round 2, so it measures trajectory divergence, not trainer error; compare it "
                 "with the legacy seed-to-seed distance.")
    return "\n".join(L) + "\n"


def write_report(out: Path, meta: dict, result: dict, raw: dict) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    report = out / "report.md"
    report.write_text(render_report(meta, result), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(
        {"meta": meta, "thresholds": thresholds(), **result, "raw": raw},
        indent=2, default=str), encoding="utf-8")
    return report


def _raw(runs: dict[int, dict]) -> dict:
    return {str(s): {"dir": r["dir"], "filled_from_miles_log": r["filled_from_miles_log"],
                     "rounds": {f"{k[0]}/{k[1]}": v for k, v in sorted(r["rounds"].items())}}
            for s, r in sorted(runs.items())}


_MILES_ARG = re.compile(r"^  (\w+) \.{3,} (.*)$")
LOSS_ARGS = ("advantage_estimator", "rewards_normalization", "grpo_std_normalization",
             "eps_clip", "eps_clip_high", "calculate_per_token_loss", "global_batch_size",
             "clip_grad", "lora_alpha", "kl_coef", "use_kl_loss", "entropy_coef", "use_tis",
             "use_rollout_logprobs", "normalize_advantages")


def miles_loss_args(log_path: str | Path) -> dict[str, str]:
    """First value of each :data:`LOSS_ARGS` in a Miles argument dump (``miles.log``)."""
    out: dict[str, str] = {}
    for line in Path(log_path).read_text(errors="ignore").splitlines():
        m = _MILES_ARG.match(line)
        if m and m.group(1) in LOSS_ARGS and m.group(1) not in out:
            out[m.group(1)] = m.group(2).strip()
    return out


def fp32_loss_config(legacy_args: dict[str, str], ports_args: dict[str, str]):
    """LossConfig matching both engines, or a reason the fp32 reference cannot model them."""
    from yeto.rl.fp32_reference import LossConfig

    diff = {k: (legacy_args.get(k), ports_args.get(k)) for k in LOSS_ARGS
            if legacy_args.get(k) != ports_args.get(k)}
    if diff:
        return f"engines' loss arguments differ: {diff}"
    a = legacy_args
    missing = [k for k in ("advantage_estimator", "eps_clip", "global_batch_size") if k not in a]
    if missing:
        return f"loss arguments missing from miles.log: {missing}"
    unsupported = {k: a.get(k) for k, ok in (
        ("advantage_estimator", a.get("advantage_estimator") in ("grpo", "gspo")),
        ("calculate_per_token_loss", a.get("calculate_per_token_loss", "False") == "False"),
        ("use_kl_loss", a.get("use_kl_loss", "False") == "False"),
        ("entropy_coef", float(a.get("entropy_coef", "0") or 0) == 0.0),
        ("use_tis", a.get("use_tis", "False") == "False"),
        ("normalize_advantages", a.get("normalize_advantages", "False") == "False"),
    ) if not ok}
    if unsupported:
        return f"loss configuration not modelled by the fp32 reference: {unsupported}"
    hi = a.get("eps_clip_high")
    return LossConfig(
        rewards_normalization=a.get("rewards_normalization", "True") == "True",
        grpo_std_normalization=a.get("grpo_std_normalization", "True") == "True",
        eps_clip=float(a["eps_clip"]),
        eps_clip_high=float(hi) if hi not in (None, "None") else float(a["eps_clip"]),
        num_samples=int(a["global_batch_size"]),
        clip_grad=float(a.get("clip_grad", "1.0")),
        lora_alpha=float(a["lora_alpha"]) if a.get("lora_alpha") not in (None, "None") else None)


def compute_fp32_references(args, ref_arm: Path, legacy_arm: Path, ports_arm: Path,
                            legacy_tf: dict[int, dict]) -> dict[int, dict]:
    """island -> fp32 reference (or ``{"error": ...}`` -> INCOMPLETE)."""
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from yeto.rl import fp32_reference as f32
    except Exception as error:  # noqa: BLE001
        return {i: {"error": f"fp32 reference unavailable: {error}"} for i in legacy_tf}
    cfg_json = legacy_arm.parents[2] / "report" / "config.json"
    arguments = (json.loads(cfg_json.read_text()).get("arguments", {})
                 if cfg_json.is_file() else {})
    model = args.fp32_model or arguments.get("model")
    out: dict[int, dict] = {}
    loaded = None
    for i, lg in sorted(legacy_tf.items()):
        try:
            revision = (args.fp32_revision or (lg.get("audit_meta") or {}).get("base_model_revision")
                        or arguments.get("model_revision"))
            if not model or not revision:
                raise f32.ReferenceInputError("model id / revision unknown")
            logs = [legacy_arm / f"island-{i}" / "miles.log", ports_arm / f"island-{i}" / "miles.log"]
            if not all(p.is_file() for p in logs):
                raise f32.ReferenceInputError(f"miles.log (engine loss arguments) missing: {logs}")
            cfg = fp32_loss_config(miles_loss_args(logs[0]), miles_loss_args(logs[1]))
            if isinstance(cfg, str):
                raise f32.ReferenceInputError(cfg)
            pattern = args.tf_replay or str(ref_arm / "rollouts" / "island-{island}"
                                            / f"{args.tf_rollout_id}.pt")
            hits = sorted(glob.glob(pattern.replace("{island}", str(i))))
            if len(hits) != 1:
                raise f32.ReferenceInputError(
                    f"replay batch for island {i}: {len(hits)} files match {pattern}")
            stem = Path(lg["audit_base"]).name[: -len(".base.f32")]
            meta = Path(lg["audit_base"]).with_name(stem + ".json")
            if loaded is None:
                mdir = f32.resolve_model_dir(model, revision, local_path=args.fp32_model_path,
                                             cache_dir=args.fp32_cache_dir)
                loaded = f32.load_base_model(mdir)
            t0 = time.time()
            res = f32.compute_island(
                replay=hits[0], base_f32=lg["audit_base"], base_meta=str(meta), model=model,
                revision=revision, cfg=cfg, model_path=args.fp32_model_path,
                cache_dir=args.fp32_cache_dir, num_threads=args.fp32_threads, loaded_model=loaded)
            res["provenance"]["seconds"] = round(time.time() - t0, 1)
            if args.fp32_save_dir:
                d = Path(args.fp32_save_dir) / f"island-{i}"
                d.mkdir(parents=True, exist_ok=True)
                res["flat"].tofile(str(d / "fp32ref.grad.f32"))
                res["provenance"]["saved_grad"] = {
                    "path": str(d / "fp32ref.grad.f32"),
                    "sha256": f32.sha256_file(d / "fp32ref.grad.f32")}
            print(f"fp32 reference island {i}: norm {res['grad_norm']:.6f} "
                  f"({res['provenance']['seconds']} s)", flush=True)
            out[i] = res
        except Exception as error:  # noqa: BLE001 - reported as INCOMPLETE
            out[i] = {"error": f"{type(error).__name__}: {error}"}
    return out


def analyze(args) -> int:
    legacy = load_runs(args.legacy)
    ports = load_runs(args.ports)
    tf = None
    if args.tf_reference or args.tf_legacy or args.tf_ports:
        if not (args.tf_reference and args.tf_legacy and args.tf_ports):
            raise SystemExit("--tf-reference, --tf-legacy and --tf-ports go together")
        ref_arm, l_arm, p_arm = (_one_arm(args.tf_reference), _one_arm(args.tf_legacy),
                                 _one_arm(args.tf_ports))
        legacy_tf = load_tf_arm(l_arm)
        fp32 = (None if getattr(args, "no_fp32", False)
                else compute_fp32_references(args, ref_arm, l_arm, p_arm, legacy_tf))
        tf = teacher_forcing_check(load_tf_arm(ref_arm), legacy_tf, load_tf_arm(p_arm),
                                   require_update=args.preset == "strict-avg", fp32=fp32)
    extras = None
    if args.preset == "decoupled":
        extras = decoupled_extras(legacy, ports, primary_seed=args.primary_seed,
                                  base_model=args.peft_base_model)
    result = evaluate(preset=args.preset, legacy_runs=legacy, ports_runs=ports,
                      primary_seed=args.primary_seed, tf=tf, extras=extras)
    meta = {"mode": "real", "fake": False, "preset": args.preset, "config": args.config,
            "primary_seed": args.primary_seed, "legacy_seeds": sorted(legacy),
            "ports_seeds": sorted(ports), "created_unix": time.time()}
    report = write_report(Path(args.out_dir), meta, result,
                          {"legacy": _raw(legacy), "ports": _raw(ports)})
    print(f"{result['verdict']}: {report}")
    return 0 if result["passed"] else 1


def _one_arm(path: str) -> Path:
    arms = discover_arms(path)
    if len(arms) != 1:
        raise SystemExit(f"{path}: expected exactly one arm, found seeds {sorted(arms)}")
    return next(iter(arms.values()))


# ---------------------------------------------------------------------------
# plan / run (real launches; never invoked by tests)
# ---------------------------------------------------------------------------
def build_plan(args) -> dict[str, Any]:
    out = Path(args.out_dir)
    steps = []
    for engine in ("legacy", "ports"):
        for seed in args.seeds:
            run_dir = out / f"{engine}-s{seed}"
            steps.append({"name": run_dir.name, "engine": engine, "seed": seed,
                          "run_dir": str(run_dir), "env": {},
                          "command": shlex.split(args.launch_cmd.format(
                              engine=engine, seed=seed, run_dir=run_dir,
                              launch_args=args.launch_args))})
    if args.teacher_forcing:
        ref = out / f"legacy-s{args.primary_seed}"
        replay = str(ref / "work" / f"seed-{args.primary_seed}" / "*" / "rollouts" / "island-*"
                     / f"{args.tf_rollout_id}.pt")
        tf_args = (f"{args.tf_launch_args or args.launch_args} --global-rounds 1 "
                   f"--custom-generate-function-path {TF_GENERATE}")
        for engine in ("legacy", "ports"):
            run_dir = out / f"tf-{engine}"
            steps.append({"name": run_dir.name, "engine": engine, "seed": args.primary_seed,
                          "run_dir": str(run_dir),
                          "env": {REPLAY_ENV: replay, GRAD_AUDIT_ENV: "1"},
                          "command": shlex.split(args.launch_cmd.format(
                              engine=engine, seed=args.primary_seed, run_dir=run_dir,
                              launch_args=tf_args))})
    return {"preset": args.preset, "steps": steps, "thresholds": thresholds(),
            "report": str(out / "report.md")}


def run(args) -> int:
    the_plan = build_plan(args)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(the_plan, indent=2), encoding="utf-8")
    for step in the_plan["steps"]:
        Path(step["run_dir"]).mkdir(parents=True, exist_ok=True)
        with open(Path(step["run_dir"]) / "launch.log", "wb") as log:
            rc = subprocess.run(step["command"], env={**os.environ, **step["env"]},
                                stdout=log, stderr=subprocess.STDOUT).returncode
        (Path(step["run_dir"]) / "rc").write_text(f"{rc}\n")
        if rc != 0:
            print(f"step {step['name']} failed rc={rc}; see {step['run_dir']}/launch.log")
            return rc
    ns = argparse.Namespace(
        legacy=[str(out / f"legacy-s{s}") for s in args.seeds],
        ports=[str(out / f"ports-s{s}") for s in args.seeds],
        tf_reference=str(out / f"legacy-s{args.primary_seed}") if args.teacher_forcing else None,
        tf_legacy=str(out / "tf-legacy") if args.teacher_forcing else None,
        tf_ports=str(out / "tf-ports") if args.teacher_forcing else None,
        preset=args.preset, primary_seed=args.primary_seed, out_dir=args.out_dir,
        config=args.launch_args, peft_base_model=args.peft_base_model,
        tf_replay=None, tf_rollout_id=args.tf_rollout_id, fp32_model=None, fp32_revision=None,
        fp32_model_path=None, fp32_cache_dir=None, fp32_threads=None, fp32_save_dir=None,
        no_fp32=False)
    return analyze(ns)


# ---------------------------------------------------------------------------
# Fake mode (CPU harness proof)
# ---------------------------------------------------------------------------
FAKE_TENSOR = "base_model.model.layer.lora_A.weight"


def run_fake_ports(run_dir: Path, *, rounds: int, islands: int = 2) -> list[Path]:
    """Real IslandDriver + StrictAvgSync on the CPU fake engine."""

    import torch

    from yeto.rl.bridge import BridgeConfig
    from yeto.rl.core import build_avg_layout
    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.bridges import StrictAvgSync
    from yeto.rl.engine.driver import EventTape, IslandDriver
    from yeto.rl.engine.fake import (
        LORA_CONFIG_HASH, MODEL_REVISION, FakeEngine, FakeStrictSyncer, fake_capabilities,
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    engines = [FakeEngine(tensors={FAKE_TENSOR: torch.zeros(1, 2)},
                          step_delta=torch.tensor([1.0 + 2 * i, 3.0 + 2 * i]))
               for i in range(islands)]
    initial = engines[0].canonical(0)
    syncer = FakeStrictSyncer(build_avg_layout(initial.specs), learners=islands,
                              total_steps=rounds)
    drivers, paths = [], []
    for i, engine in enumerate(engines):
        island_dir = run_dir / f"island-{i}"
        island_dir.mkdir(parents=True, exist_ok=True)
        tape = island_dir / "events.jsonl"
        paths.append(tape)
        config = BridgeConfig(
            syncer_addr=("127.0.0.1", 1), learner_id=i, global_rounds=rounds,
            groups_per_round=engine.groups, samples_per_group=engine.samples_per_group,
            local_optimizer_steps=1, wan_streams=0, expected_specs=initial.specs,
            base_model_revision=MODEL_REVISION, lora_config_hash=LORA_CONFIG_HASH,
            layout_hash=initial.layout_hash, event_tape=str(tape),
        )
        sync = StrictAvgSync(config, client_factory=lambda _b, i=i: syncer.client(i))
        drivers.append(IslandDriver(
            learner_id=i, rollout=engine.rollout, trainer=engine.trainer,
            policy_state=engine.policy_state, publisher=engine.publisher,
            placement=engine.placement, algorithm=AlgorithmSpec(), sync=sync,
            events=EventTape(tape, i), capabilities=fake_capabilities(),
        ))
    errors: dict[int, BaseException] = {}

    def go(i, d):
        try:
            d.run()
        except BaseException as error:  # noqa: BLE001
            errors[i] = error

    threads = [threading.Thread(target=go, args=(i, d), daemon=True) for i, d in enumerate(drivers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
        if t.is_alive():
            raise RuntimeError("fake ports island did not finish")
    if errors:
        raise RuntimeError(f"fake ports run failed: {errors}")
    return paths


def write_fake_legacy(arm_dir: Path, ports_arm: Path, *, offset: float, noise: float) -> None:
    """SYNTHETIC legacy arm: the ports tapes with round >= 2 floats/tokens shifted by
    ``offset`` (in units of ``noise``) and round-1 grad_norm scaled by 1 + offset/100."""

    for tape in sorted(ports_arm.glob("island-*/events.jsonl")):
        lines = []
        for line in tape.read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            e["synthetic_legacy"] = True
            if e.get("event") == "rl_local_round":
                r = int(e["local_round_id"])
                for m in FLOAT_METRICS + ("action_tokens",):
                    e.pop(f"rl/{m}", None)
                if r == 1:
                    e["grad_norm"] = float(e["grad_norm"]) * (1 + offset / 100)
                else:
                    for m in FLOAT_METRICS:
                        if e.get(m) is not None:
                            e[m] = float(e[m]) + offset * noise
                    e["action_tokens"] = int(e["action_tokens"]) + int(offset)
            if e.get("event") == "rl_policy_apply" and e.get("policy_version", 0) > 0:
                e["sync/global_policy_hash"] = f"synthetic-legacy-{e['policy_version']}"
            lines.append(json.dumps(e, sort_keys=True))
        dest = arm_dir / tape.parent.name / "events.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fake(args) -> int:
    out = Path(args.out_dir)
    seeds = list(args.seeds)
    offsets = [(i - (len(seeds) - 1) / 2) for i in range(len(seeds))]  # symmetric around 0
    for seed, off in zip(seeds, offsets):
        arm = out / f"ports-s{seed}" / "work" / f"seed-{seed}" / "yeto-federated-m2"
        run_fake_ports(arm, rounds=args.rounds)
        write_fake_legacy(out / f"legacy-s{seed}" / "work" / f"seed-{seed}" / "yeto-federated-m2",
                          arm, offset=off, noise=args.fake_noise)
    legacy = load_runs([str(out / f"legacy-s{s}") for s in seeds])
    ports = load_runs([str(out / f"ports-s{s}") for s in seeds])
    result = evaluate(preset="strict-avg", legacy_runs=legacy, ports_runs=ports,
                      primary_seed=seeds[0], tf=None)
    meta = {"mode": "fake", "fake": True, "preset": "strict-avg",
            "config": f"fake strict-avg islands=2 rounds={args.rounds}",
            "primary_seed": seeds[0], "legacy_seeds": seeds, "ports_seeds": seeds,
            "created_unix": time.time()}
    report = write_report(out, meta, result, {"legacy": _raw(legacy), "ports": _raw(ports)})
    print(f"{result['verdict']} (FAKE): {report}")
    return 0 if result["verdict"] in ("PASS", "INCOMPLETE") else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _seeds(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--out-dir", default="rl-engine-equivalence")
        sp.add_argument("--preset", choices=["strict-avg", "decoupled"], default="strict-avg")
        sp.add_argument("--primary-seed", type=int, default=DEFAULT_SEEDS[0])
        sp.add_argument("--peft-base-model", default=None,
                        help="decoupled: also load each adapter onto this base model")

    for name in ("plan", "run"):
        sp = sub.add_parser(name)
        common(sp)
        sp.add_argument("--launch-cmd", default=DEFAULT_LAUNCH_CMD,
                        help="template; placeholders {engine} {seed} {run_dir} {launch_args}")
        sp.add_argument("--launch-args", required=True,
                        help="shared benchmark_rl arguments (no --seeds/--rl-engine/--work-dir)")
        sp.add_argument("--seeds", type=_seeds, default=list(DEFAULT_SEEDS))
        sp.add_argument("--teacher-forcing", action="store_true")
        sp.add_argument("--tf-launch-args", default=None,
                        help="launch args for the TF runs (default: --launch-args)")
        sp.add_argument("--tf-rollout-id", type=int, default=0)

    sp = sub.add_parser("analyze")
    common(sp)
    sp.add_argument("--legacy", nargs="+", required=True, help="legacy run/work/arm dirs")
    sp.add_argument("--ports", nargs="+", required=True, help="ports run/work/arm dirs")
    sp.add_argument("--tf-reference", help="legacy run whose rollouts were replayed")
    sp.add_argument("--tf-legacy", help="legacy teacher-forcing run")
    sp.add_argument("--tf-ports", help="ports teacher-forcing run")
    sp.add_argument("--config", default="", help="free-text config description for the report")
    sp.add_argument("--tf-replay", default=None,
                    help="glob of the replayed dump per island ({island} placeholder); default "
                         "<tf-reference arm>/rollouts/island-{island}/<tf-rollout-id>.pt")
    sp.add_argument("--tf-rollout-id", type=int, default=0)
    sp.add_argument("--fp32-model", default=None,
                    help="HF model id (default: TF run report/config.json)")
    sp.add_argument("--fp32-revision", default=None,
                    help="40-hex revision (default: audit base_model_revision)")
    sp.add_argument("--fp32-model-path", default=None,
                    help="local snapshot dir named after the revision (instead of the HF cache)")
    sp.add_argument("--fp32-cache-dir", default=None, help="HF cache dir (offline lookup)")
    sp.add_argument("--fp32-threads", type=int, default=None)
    sp.add_argument("--fp32-save-dir", default=None,
                    help="also write each island's fp32 reference gradient here")
    sp.add_argument("--no-fp32", action="store_true",
                    help="skip the fp32 reference (tier 2 then reports INCOMPLETE)")

    sp = sub.add_parser("fake")
    sp.add_argument("--out-dir", default="rl-engine-equivalence-fake")
    sp.add_argument("--seeds", type=_seeds, default=list(DEFAULT_SEEDS))
    sp.add_argument("--rounds", type=int, default=3)
    sp.add_argument("--fake-noise", type=float, default=1e-3)
    args = p.parse_args(argv)
    if getattr(args, "seeds", None) is not None and getattr(args, "primary_seed", None) is not None:
        if args.primary_seed not in args.seeds:
            p.error("--primary-seed must be one of --seeds")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.cmd == "plan":
        print(json.dumps(build_plan(args), indent=2))
        return 0
    if args.cmd == "run":
        return run(args)
    if args.cmd == "analyze":
        return analyze(args)
    return fake(args)


if __name__ == "__main__":
    sys.exit(main())
