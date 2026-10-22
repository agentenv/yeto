"""E11 whole-cluster comparison: reward against samples and time on one card total.

Every arm becomes one list of :class:`Point` (one per trained batch, in end-time
order). A single-island arm gives one point per round (``parsed.json`` of an
``eff-run`` directory); the 2-island yeto arm (``logs/E11/run2i.sh``) gives one
point per island round, read from each island's tape, so its two islands
interleave by time.

Definitions (SESSION26/gpu-review/E11.md §3, fixed before the runs):

* reward of a point = mean reward of the batch's rollout samples;
* smoothed reward = sample-weighted mean over the latest ``window`` samples
  (3 single-island rounds), evaluated at every point;
* target T = sample-weighted mean reward of the first ``2 * round_samples``
  samples of every run, pooled, plus ``margin``;
* time to T = first point whose smoothed reward >= T, measured from the run's
  launch (``t0``: the local launch command) and from the end of its first batch
  (no cold start); GPU hours = that time x the run's card total.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .parse import _tape_events, _utc_file, parse_yeto


@dataclass(frozen=True)
class Point:
    t_end: float
    samples: int
    reward: float


@dataclass
class Curve:
    run_id: str
    arm: str
    cards: int
    t0: float  # local launch start (unix s)
    t_last: float  # the run's GPU containers ended (or its last point)
    points: list[Point]
    usd_extra: float = 0.0  # e.g. the 2-island arm's head VM


def _from_parsed(run_dir: Path) -> Curve:
    parsed = json.loads((run_dir / "parsed.json").read_text())
    pts = [Point(float(r["t_end_unix"]), int(r["samples"]), float(r["reward_mean"]))
           for r in sorted(parsed["rounds"], key=lambda r: r["round"])
           if r.get("t_end_unix") is not None and r.get("samples") and r.get("reward_mean") is not None]
    t0 = _launch_start(run_dir) or float(parsed["container_start_unix"])
    end = parsed.get("container_end_unix") or (pts[-1].t_end if pts else t0)
    return Curve(parsed["run_id"], parsed["arm"], int(parsed["cards"]["total"]), t0, float(end), pts)


def _launch_start(run_dir: Path) -> float | None:
    """The local launch time: miles_local.sh's start_utc.txt, else the driver's first log line."""
    utc = _utc_file(run_dir / "raw" / "start_utc.txt")
    if utc:
        return utc
    log = run_dir / "driver.log"
    if not log.is_file():
        return None
    try:
        first = log.read_text().split(" ", 1)[0]
        return datetime.strptime(first, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def _from_islands(run_dir: Path, *, cards: int, head_usd_per_hour: float) -> Curve:
    """A run2i.sh directory: one tape per island under ``tape-direct/``."""
    pts: list[Point] = []
    tapes = sorted((run_dir / "tape-direct").rglob("rl-island-*.jsonl"))
    if not tapes:
        raise ValueError(f"{run_dir}: no island tapes under tape-direct/")
    for tape in tapes:
        island = parse_yeto(_tape_events(tape.parent), mode="async")
        pts += [Point(float(r["t_end_unix"]), int(r["samples"]), float(r["reward_mean"]))
                for r in island.rounds
                if r.get("t_end_unix") is not None and r.get("samples") and r.get("reward_mean") is not None]
    pts.sort(key=lambda p: p.t_end)
    t0 = _utc_file(run_dir / "start_utc.txt")
    end = _utc_file(run_dir / "end_utc.txt") or (pts[-1].t_end if pts else t0)
    hours = (end - t0) / 3600 if t0 and end else 0.0
    return Curve(run_dir.name, "yeto-miles-2islands", cards, float(t0), float(end), pts,
                 usd_extra=hours * head_usd_per_hour)


def load(run_dir: Path, *, cards: int = 4, head_usd_per_hour: float = 0.0) -> Curve:
    run_dir = Path(run_dir)
    if (run_dir / "parsed.json").is_file():
        return _from_parsed(run_dir)
    return _from_islands(run_dir, cards=cards, head_usd_per_hour=head_usd_per_hour)


def smoothed(points: list[Point], window: int) -> list[float]:
    """Sample-weighted mean reward over the latest ``window`` samples at every point."""
    out = []
    for i in range(len(points)):
        n = s = 0.0
        for p in reversed(points[:i + 1]):
            take = min(p.samples, window - n)
            s += p.reward * take
            n += take
            if n >= window:
                break
        out.append(s / n if n else float("nan"))
    return out


def start_reward(curves: list[Curve], samples: int) -> float:
    """Pooled sample-weighted mean reward of every run's first ``samples`` samples."""
    n = s = 0.0
    for c in curves:
        left = samples
        for p in c.points:
            take = min(p.samples, left)
            s += p.reward * take
            n += take
            left -= take
            if left <= 0:
                break
    return s / n if n else float("nan")


def metrics(c: Curve, *, target: float, window: int, usd_per_card_hour: float) -> dict[str, Any]:
    sm = smoothed(c.points, window)
    hit = next((i for i, v in enumerate(sm) if v >= target), None)
    total = sum(p.samples for p in c.points)
    run_s = c.t_last - c.t0
    out: dict[str, Any] = {
        "run_id": c.run_id, "arm": c.arm, "cards": c.cards, "points": len(c.points), "samples": total,
        "first_batch_s": c.points[0].t_end - c.t0 if c.points else None,
        "train_s": c.points[-1].t_end - c.points[0].t_end if len(c.points) > 1 else None,
        "samples_per_s": ((total - c.points[0].samples) / (c.points[-1].t_end - c.points[0].t_end)
                          if len(c.points) > 1 else None),
        "reward_last3": sm[-1] if sm else None,
        "run_s": run_s, "usd_run": run_s / 3600 * c.cards * usd_per_card_hour + c.usd_extra,
        "target": target, "reached": hit is not None,
    }
    if hit is not None:
        p = c.points[hit]
        out.update(to_target_s=p.t_end - c.t0, to_target_after_first_s=p.t_end - c.points[0].t_end,
                   to_target_samples=sum(q.samples for q in c.points[:hit + 1]))
        out["to_target_gpu_h"] = out["to_target_s"] / 3600 * c.cards
        out["to_target_usd"] = out["to_target_gpu_h"] * usd_per_card_hour
    return out


def curve_table(c: Curve, window: int) -> list[dict[str, Any]]:
    sm, cum = smoothed(c.points, window), 0
    rows = []
    for p, v in zip(c.points, sm):
        cum += p.samples
        rows.append({**asdict(p), "t_from_launch_s": p.t_end - c.t0, "cum_samples": cum, "smoothed": v})
    return rows
