"""Per-bucket evaluation metrics (rl-eval-difficulty-buckets D4, tasks 3.2).

Input: the deduplicated per-unit result records of one policy version (and
optionally of version 0 for the paired difference). A record carries at least
``task_id``, ``trial``, ``eval_bucket``, ``success`` (bool), ``reward`` and
``end_reason`` (completed / max_turns / max_seq_len / timed_out / infra_error).

``infra_error`` records never count in the pass rate (D6.d); they are reported
as a fraction of their own. Standard errors come from a task-level bootstrap
(resample tasks, keep each task's trials together) with a fixed seed, so the
same records always give the same numbers.
"""

from __future__ import annotations

import hashlib
import math
import random
import statistics
from collections import defaultdict
from typing import Any, Iterable, Mapping

END_REASONS = ("completed", "max_turns", "max_seq_len", "timed_out", "infra_error")
TRUNCATED = ("max_seq_len",)
BOOTSTRAP_ROUNDS = 1000
BOOTSTRAP_SEED = 20261008


def _seed(bucket: str, salt: str) -> int:
    return int(hashlib.sha256(f"{BOOTSTRAP_SEED}/{bucket}/{salt}".encode()).hexdigest()[:16], 16)


def _by_task(records: Iterable[Mapping[str, Any]]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if r.get("end_reason") == "infra_error":
            continue
        out[str(r["task_id"])].append(1.0 if r.get("success") else 0.0)
    return dict(out)


def _task_means(per_task: Mapping[str, list[float]]) -> dict[str, float]:
    return {t: statistics.fmean(v) for t, v in per_task.items() if v}


def bootstrap_se(values: Mapping[str, float], *, seed: int, rounds: int = BOOTSTRAP_ROUNDS) -> float | None:
    """Standard error of the mean over tasks (tasks resampled with replacement)."""
    keys = sorted(values)
    if len(keys) < 2:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(rounds):
        draw = [values[keys[rng.randrange(len(keys))]] for _ in keys]
        means.append(statistics.fmean(draw))
    return statistics.pstdev(means)


def bucket_metrics(records: Iterable[Mapping[str, Any]],
                   baseline: Iterable[Mapping[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    """``{bucket: {n, n_tasks, pass_rate, pass_se, reward_mean, truncated_frac,
    max_turns_frac, timed_out_frac, infra_error_frac[, paired_diff, paired_se, paired_tasks]}}``.

    Pass rate = mean over tasks of the task's success fraction (each task
    weighs the same whatever its trial count, so version 0 with 4 trials and
    later versions with 2 trials compare on one scale)."""
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in records:
        groups[str(r.get("eval_bucket") or "unknown")].append(r)
    base_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in baseline or ():
        base_groups[str(r.get("eval_bucket") or "unknown")].append(r)
    out: dict[str, dict[str, Any]] = {}
    for bucket in sorted(groups):
        rows = groups[bucket]
        n = len(rows)
        reasons = [str(r.get("end_reason") or "completed") for r in rows]
        means = _task_means(_by_task(rows))
        counted = [r for r in rows if r.get("end_reason") != "infra_error"]
        rewards = [float(r["reward"]) for r in counted if r.get("reward") is not None
                   and math.isfinite(float(r["reward"]))]
        m: dict[str, Any] = {
            "n": n,
            "n_tasks": len({str(r["task_id"]) for r in rows}),
            "pass_rate": statistics.fmean(means.values()) if means else None,
            "pass_se": bootstrap_se(means, seed=_seed(bucket, "pass")),
            "reward_mean": statistics.fmean(rewards) if rewards else None,
            "truncated_frac": sum(x in TRUNCATED for x in reasons) / n if n else None,
            "max_turns_frac": reasons.count("max_turns") / n if n else None,
            "timed_out_frac": reasons.count("timed_out") / n if n else None,
            "infra_error_frac": reasons.count("infra_error") / n if n else None,
        }
        if baseline is not None:
            base = _task_means(_by_task(base_groups.get(bucket, ())))
            common = sorted(set(base) & set(means))
            diffs = {t: means[t] - base[t] for t in common}
            m["paired_tasks"] = len(common)
            m["paired_diff"] = statistics.fmean(diffs.values()) if diffs else None
            m["paired_se"] = bootstrap_se(diffs, seed=_seed(bucket, "paired"))
        out[bucket] = m
    return out


def flatten(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """``{"bucket/<bucket>/<metric>": value}`` for the ``rl_eval`` event."""
    return {f"bucket/{b}/{k}": v for b, row in metrics.items() for k, v in row.items()}
