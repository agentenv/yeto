"""Per-run summary from ``parsed.json`` (the definitions every arm shares).

* Steady window: the rounds after the first (the first carries engine start-up,
  CUDA-graph capture and caches); ``steady_wall_s`` = the sum of their
  ``round_s`` (step end minus previous step end).
* Throughput per card: steady tokens / steady wall / every card of the arm
  (rollout and trainer cards both count).
* Segments: per-round medians over the steady window; ``unaccounted_s`` = round
  minus the named segments (offload/onload, reward, advantage, orchestration).
* GPU: NVML samples inside the steady window; ``bubble_frac`` = the share of
  card samples with utilization below :data:`BUBBLE_UTIL_PCT`.
* Cost: ``usd_per_card_hour`` from ``prices.json`` x cards x container seconds
  (whole run) and x steady wall (steady state), per 1k trained samples.

A value whose inputs are missing is ``None`` and its name is listed in
``missing``; nothing is filled in.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

PRICES = Path(__file__).with_name("prices.json")
BUBBLE_UTIL_PCT = 5.0
SEGMENTS = ("gen_s", "train_s", "weight_sync_s", "sampler_update_s", "wait_s")


def price(gpu: str, path: Path = PRICES) -> float:
    table = json.loads(path.read_text())
    if gpu not in table:
        raise KeyError(f"no price for GPU {gpu!r} in {path}")
    return float(table[gpu]["usd_per_card_hour"])


def _vals(rows: list[dict], key: str) -> list[float]:
    return [float(r[key]) for r in rows if r.get(key) is not None]


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 4) if values else None


def _sum_all(rows: list[dict], key: str) -> float | None:
    """Sum over rows, or None when any row lacks the value (a partial sum would mislead)."""
    vals = [r.get(key) for r in rows]
    return None if not vals or any(v is None for v in vals) else float(sum(vals))


def summarize(parsed: dict[str, Any], *, prices_path: Path = PRICES) -> dict[str, Any]:
    rounds = sorted(parsed["rounds"], key=lambda r: r["round"])
    cards = int(parsed["cards"]["total"])
    steady = rounds[1:]
    s: dict[str, Any] = {"arm": parsed["arm"], "run_id": parsed["run_id"], "rounds": len(rounds),
                         "rounds_expected": parsed["spec"]["rounds"], "cards": cards, "gpu": parsed["gpu"],
                         "rc": parsed.get("rc")}
    wall = _sum_all(steady, "round_s")
    s["steady_wall_s"] = round(wall, 3) if wall is not None else None
    for key, out in (("gen_tokens", "gen_tok_per_card_s"), ("train_tokens", "train_tok_per_card_s"),
                     ("samples", "samples_per_s")):
        total = _sum_all(steady, key)
        denom = (wall * (cards if key != "samples" else 1)) if wall else None
        s[out] = round(total / denom, 3) if total is not None and denom else None
    s["round_s_median"] = _median(_vals(steady, "round_s"))
    s["round_s_range"] = [min(_vals(steady, "round_s")), max(_vals(steady, "round_s"))] if _vals(steady, "round_s") \
        else None
    for seg in SEGMENTS:
        s[f"{seg}_median"] = None if parsed["sources"].get(seg) in ("n/a",) else _median(_vals(steady, seg))
    named = [seg for seg in SEGMENTS if parsed["sources"].get(seg) != "n/a"]
    rest = [r["round_s"] - sum(r[seg] for seg in named) for r in steady
            if r.get("round_s") is not None and all(r.get(seg) is not None for seg in named)]
    s["unaccounted_s_median"] = _median(rest)
    s["segments_raw_median"] = _raw_medians(parsed.get("segments_raw") or [])
    # reward against wall time and samples (every round; t from the first step end)
    t0 = rounds[0].get("t_end_unix") if rounds else None
    cum, curve = 0.0, []
    for r in rounds:
        cum = cum + r["samples"] if r.get("samples") is not None and cum is not None else None
        curve.append({"round": r["round"], "t_s": round(r["t_end_unix"] - t0, 3)
                      if r.get("t_end_unix") is not None and t0 is not None else None,
                      "samples": cum, "reward_mean": r.get("reward_mean")})
    s["reward_curve"] = curve
    s["reward_last3_mean"] = _median(_vals(rounds[-3:], "reward_mean"))  # median of the last three
    # GPU (NVML samples inside the steady window)
    lo = rounds[0].get("t_end_unix") if rounds else None
    hi = rounds[-1].get("t_end_unix") if rounds else None
    nv = [x for x in parsed.get("nvml") or [] if lo is not None and hi is not None and lo <= x["t_unix"] <= hi]
    util = [x["util"] for x in nv if x.get("util") is not None]
    s["nvml_samples"] = len(util)
    s["nvml_interval_s"] = parsed.get("nvml_interval_s")
    s["gpu_util_mean"] = round(statistics.fmean(util), 2) if util else None
    s["bubble_frac"] = round(sum(u < BUBBLE_UTIL_PCT for u in util) / len(util), 4) if util else None
    mem = [x["mem_mb"] for x in parsed.get("nvml") or [] if x.get("mem_mb") is not None]
    s["gpu_mem_peak_mb"] = max(mem) if mem else None
    # cost
    rate = price(parsed["gpu"], prices_path)
    s["usd_per_card_hour"] = rate
    start, end = parsed.get("container_start_unix"), parsed.get("container_end_unix")
    total_samples = _sum_all(rounds, "samples")
    if start and end:
        s["usd_run"] = round(rate * cards * (end - start) / 3600, 4)
        s["usd_per_1k_samples_run"] = round(s["usd_run"] / total_samples * 1000, 4) if total_samples else None
    else:
        s["usd_run"] = s["usd_per_1k_samples_run"] = None
    steady_samples = _sum_all(steady, "samples")
    s["usd_per_1k_samples_steady"] = round(rate * cards * wall / 3600 / steady_samples * 1000, 4) \
        if wall and steady_samples else None
    # off-policy
    s["is_clip_frac_mean"] = round(statistics.fmean(_vals(rounds, "is_clip_frac")), 5) \
        if _vals(rounds, "is_clip_frac") else None
    dropped, generated = _vals(rounds, "dropped_samples"), _vals(rounds, "generated_samples")
    s["dropped_total"] = max(dropped) if dropped and "count/" in parsed["sources"].get("dropped_samples", "") \
        else (sum(dropped) if dropped else None)  # verl's fully_async counters are cumulative
    s["drop_frac"] = round(s["dropped_total"] / max(generated), 4) if s["dropped_total"] is not None and generated \
        and max(generated) else None
    s["version_lag"] = [r.get("version_lag") for r in rounds if r.get("version_lag") is not None] or None
    s["sources"] = parsed["sources"]
    na = {k for k, v in parsed["sources"].items() if v == "n/a"}
    s["not_applicable"] = sorted(k for k in s if s[k] is None and _field_of(k) in na)
    s["missing"] = sorted(k for k, v in s.items() if v is None and k != "rc" and k not in s["not_applicable"])
    return s


# summary key -> the round field it is computed from (for the not-applicable list)
_DERIVED_FROM = {"dropped_total": "dropped_samples", "drop_frac": "dropped_samples"}


def _field_of(key: str) -> str:
    return _DERIVED_FROM.get(key, key.removesuffix("_median"))


def _raw_medians(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({k for r in rows[1:] for k in r})
    return {k: _median([float(r[k]) for r in rows[1:] if r.get(k) is not None]) for k in keys}
