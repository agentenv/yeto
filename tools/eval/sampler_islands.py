"""Sampler-island measurements from island event tapes (S26 E4, E5).

Input: one run directory per call. The tapes are read from
``<run>/tape-direct/<app>/l<N>/rank0/*.jsonl`` (one file per incarnation, the
layout ``modal volume get`` of the tape volume gives), else from
``<run>/tapes/l<N>/*.jsonl``. ``l<N>`` is the island; its role comes from its
events (``rl_sampler_*``: sampler; ``rl_round_trained``: trainer).

Two reports (JSON on stdout):

* ``scale`` (E4): per sampler island the rounds, groups, generation seconds,
  waits and preemptions (LEAVE PREEMPTED to the next JOIN of the same sampler and
  to its first new round); for the run the sampler production rate and the
  trainer's consumption rate (groups per second). ``--baseline RUN`` adds the
  speedup and efficiency against a baseline run (``N`` sampler islands vs the
  baseline's), or the effective-throughput ratio of a preempted run vs an
  unpreempted run of the same size.
* ``latency`` (E5): per release and sampler island the phases of one policy
  update: the trainer's publish to its own engines and its release phases
  (export, digest, commits), the wait until the sampler first read the release,
  and the sampler's load phases. ``wait_s`` compares two hosts' wall clocks
  (released on the trainer, seen on the sampler); it includes the clock offset.

Rates use the trainer's rounds after its first one (the first round includes
the start-up). Every number is computed from the tapes; a missing field is
reported as None, never estimated.

CLI::

    python -m tools.eval.sampler_islands scale RUN [--baseline RUN] [--samples-per-group K]
    python -m tools.eval.sampler_islands latency RUN
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SAMPLER = "sampler"
TRAINER = "trainer"
_ISLAND_DIR = re.compile(r"l(\d+)")


# ---------------------------------------------------------------- reading
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Every JSON object line of ``path`` (a line cut by a live pull is skipped)."""
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and "event" in row:
            rows.append(row)
    return rows


def _island_dirs(run: Path) -> dict[int, list[Path]]:
    found: dict[int, list[Path]] = defaultdict(list)
    for base in sorted((run / "tape-direct").glob("*")) + [run / "tapes"]:
        if not base.is_dir():
            continue
        for directory in sorted(base.iterdir()):
            match = _ISLAND_DIR.fullmatch(directory.name)
            if match and directory.is_dir():
                found[int(match.group(1))].extend(sorted(directory.rglob("*.jsonl")))
        if found:
            break
    return found


_LOG_EVENT = re.compile(r"\[[^\]]*-l(\d+)-modal\] YETO_RL_EVENT (\{.*\})\s*$")


def launch_log_events(run: Path) -> dict[int, list[dict[str, Any]]]:
    """Events the launch log echoed (``<run>/launch.ts.log``), by island. A container
    stopped right after an event may echo it here before its tape file is saved."""
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    log = Path(run) / "launch.ts.log"
    if not log.is_file():
        return out
    for line in log.read_text(errors="replace").splitlines():
        match = _LOG_EVENT.search(line)
        if not match:
            continue
        try:
            row = json.loads(match.group(2))
        except ValueError:
            continue
        if isinstance(row, dict) and "event" in row:
            out[int(match.group(1))].append(row)
    return out


def load_run(run: Path) -> dict[int, list[dict[str, Any]]]:
    """{island: events sorted by time} from the tapes and the launch log,
    duplicates (same event, time and incarnation) dropped."""
    out: dict[int, list[dict[str, Any]]] = {}
    echoed = launch_log_events(Path(run))
    dirs = _island_dirs(Path(run))
    for island in sorted(set(dirs) | set(echoed)):
        files = dirs.get(island, [])
        seen, rows = set(), []
        for path in [*files, None]:
            for row in (read_jsonl(path) if path is not None else echoed.get(island, [])):
                key = (row.get("event"), row.get("time_unix"), row.get("incarnation"))
                if key not in seen:
                    seen.add(key)
                    rows.append(row)
        out[island] = sorted(rows, key=lambda r: float(r.get("time_unix") or 0.0))
    return out


def island_role(events: Iterable[dict[str, Any]]) -> str | None:
    names = {e.get("event") for e in events}
    if "rl_round_trained" in names:
        return TRAINER
    if any(str(n).startswith("rl_sampler_") for n in names):
        return SAMPLER
    return None


def _of(events: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [e for e in events if e.get("event") == name]


def _spread(values: list[float]) -> dict[str, float] | None:
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    return {"n": len(values), "median": round(statistics.median(values), 3),
            "min": round(min(values), 3), "max": round(max(values), 3)}


# ---------------------------------------------------------------- E4: scale
def sampler_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    # a round cut by a preemption ends with groups 0 (nothing written): not a round
    every = _of(events, "rl_sampler_round")
    rounds = [r for r in every if int(r.get("groups") or 0) > 0]
    generate = [r.get("generate_s") for r in rounds]
    starts = [float(r["time_unix"]) - float(r["generate_s"]) for r in rounds
              if r.get("generate_s") is not None]
    joins, leaves = _of(events, "rl_sampler_join"), _of(events, "rl_sampler_leave")
    preemptions = []
    for leave in (x for x in leaves if x.get("reason") == "PREEMPTED"):
        t = float(leave["time_unix"])
        rejoin = next((j for j in joins if float(j["time_unix"]) > t), None)
        after = t if rejoin is None else float(rejoin["time_unix"])
        first_round = next((r for r in rounds if float(r["time_unix"]) > after), None)
        preemptions.append({
            "leave_unix": round(t, 3), "incarnation": leave.get("incarnation"),
            "in_flight_rollout_id": leave.get("in_flight_rollout_id"),
            "rejoin_s": None if rejoin is None else round(float(rejoin["time_unix"]) - t, 3),
            "first_new_round_s": None if first_round is None else round(float(first_round["time_unix"]) - t, 3),
        })
    return {
        "incarnations": sorted({str(j.get("incarnation")) for j in joins}),
        "rounds": len(rounds),
        "cut_rounds": len(every) - len(rounds),
        "cut_generate_s": round(sum(float(r.get("generate_s") or 0.0) for r in every
                                    if int(r.get("groups") or 0) == 0), 3),
        "groups": sum(int(r.get("groups") or 0) for r in rounds),
        "generate_s": None if any(g is None for g in generate) or not generate
        else round(sum(float(g) for g in generate), 3),
        "round_generate_s": _spread(generate),
        "backlog_wait_s": round(sum(float(e.get("waited_s") or 0.0)
                                    for e in _of(events, "rl_sampler_admitted")), 3),
        "policy_wait_s": round(sum(float(e.get("waited_s") or 0.0)
                                   for e in _of(events, "rl_sampler_policy_ready")), 3),
        "first_round_start_unix": round(min(starts), 3) if starts else None,
        "last_round_end_unix": round(float(rounds[-1]["time_unix"]), 3) if rounds else None,
        "preemptions": preemptions,
    }


def driver_phase_seconds(events: list[dict[str, Any]]) -> dict[str, float]:
    """Seconds per driver phase: ``rl_driver_phase`` marks a phase start; a phase
    lasts until the next mark (the last mark has no end and is not counted). On a
    buffer trainer ``generate`` is the wait for a round from the buffer."""
    marks = _of(events, "rl_driver_phase")
    out: dict[str, float] = defaultdict(float)
    for mark, following in zip(marks, marks[1:]):
        out[str(mark.get("phase"))] += float(following["time_unix"]) - float(mark["time_unix"])
    return {name: round(value, 3) for name, value in out.items()}


def trainer_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    trained = _of(events, "rl_round_trained")
    groups = [int(e.get("trained_groups") or 0) for e in trained]
    phases = driver_phase_seconds(events)
    total = sum(phases.values())
    out: dict[str, Any] = {"rounds": len(trained), "groups": sum(groups),
                           "steady_groups_per_s": None, "steady_window_s": None,
                           "phase_s": phases,
                           # share of the driver's time spent waiting for samples
                           "buffer_wait_share": round(phases.get("generate", 0.0) / total, 4)
                           if total > 0 else None}
    if len(trained) >= 2:
        window = float(trained[-1]["time_unix"]) - float(trained[0]["time_unix"])
        if window > 0:
            out["steady_window_s"] = round(window, 3)
            out["steady_groups_per_s"] = round(sum(groups[1:]) / window, 4)
    return out


def scale_report(run: Path, *, samples_per_group: int | None = None) -> dict[str, Any]:
    islands = load_run(run)
    samplers = {m: sampler_summary(ev) for m, ev in islands.items() if island_role(ev) == SAMPLER}
    trainers = {m: trainer_summary(ev) for m, ev in islands.items() if island_role(ev) == TRAINER}
    starts = [s["first_round_start_unix"] for s in samplers.values() if s["first_round_start_unix"]]
    ends = [s["last_round_end_unix"] for s in samplers.values() if s["last_round_end_unix"]]
    groups = sum(s["groups"] for s in samplers.values())
    production = None
    if starts and ends and max(ends) > min(starts):
        production = round(groups / (max(ends) - min(starts)), 4)
    trainer = next(iter(trainers.values()), None)
    report = {
        "run": Path(run).name, "sampler_islands": len(samplers), "samplers": samplers,
        "trainer": trainer,
        "sampler_groups": groups,
        "production_groups_per_s": production,
        "consumption_groups_per_s": None if trainer is None else trainer["steady_groups_per_s"],
        "preemptions": sum(len(s["preemptions"]) for s in samplers.values()),
    }
    if samples_per_group:
        k = int(samples_per_group)
        for key in ("production_groups_per_s", "consumption_groups_per_s"):
            value = report[key]
            report[key.replace("groups", "samples")] = None if value is None else round(value * k, 4)
    return report


def compare(run: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    """Speedup of ``run`` over ``baseline`` (consumption and production rates).
    Same sampler count: ``effective_ratio`` (a preempted run vs an unpreempted
    one); more samplers: ``speedup`` and ``efficiency`` = speedup / island ratio."""
    out: dict[str, Any] = {"baseline": baseline["run"]}
    for key in ("consumption_groups_per_s", "production_groups_per_s"):
        a, b = run.get(key), baseline.get(key)
        out[key.replace("_groups_per_s", "_ratio")] = None if not a or not b else round(a / b, 4)
    n, n0 = run["sampler_islands"], baseline["sampler_islands"]
    ratio = out["consumption_ratio"]
    if n == n0:
        out["effective_ratio"] = ratio
    elif n0:
        out["speedup"] = ratio
        out["efficiency"] = None if ratio is None else round(ratio / (n / n0), 4)
    return out


# ---------------------------------------------------------------- E5: latency
def _sum(timings: dict[str, Any] | None) -> float | None:
    if not timings:
        return None
    return round(sum(float(v) for v in timings.values()), 3)


def latency_report(run: Path) -> dict[str, Any]:
    islands = load_run(run)
    released: dict[int, dict[str, Any]] = {}
    for events in (ev for ev in islands.values() if island_role(ev) == TRAINER):
        for e in _of(events, "rl_policy_released"):
            released[int(e["seq"])] = e
    rows = []
    for island, events in sorted(islands.items()):
        if island_role(events) != SAMPLER:
            continue
        for loaded in _of(events, "rl_sampler_policy_loaded"):
            rel = released.get(int(loaded.get("seq", -1)))
            if rel is None:
                continue
            t_rel, t_load = float(rel["time_unix"]), float(loaded["time_unix"])
            seen = loaded.get("seen_at")
            follower = loaded.get("follower_timings") or None
            row = {
                "island": island, "seq": int(loaded["seq"]),
                "policy_version": loaded.get("policy_version"), "weights": loaded.get("weights"),
                "release_bytes": rel.get("release_bytes"),
                "trainer_publish_s": rel.get("publish_s"),
                "trainer_release_s": _sum(rel.get("release_timings")),
                "trainer_release_phases": rel.get("release_timings"),
                "wait_s": None if seen is None else round(float(seen) - t_rel, 3),
                "sampler_load_s": _sum(follower),
                "sampler_follower_phases": follower,
                "sampler_load_phases": loaded.get("load_timings"),
                "fused_stage": loaded.get("fused_stage"),
                "released_to_loaded_s": round(t_load - t_rel, 3),
            }
            parts = (row["trainer_publish_s"], row["trainer_release_s"], row["released_to_loaded_s"])
            row["end_to_end_s"] = None if any(p is None for p in parts) else round(sum(parts), 3)
            rows.append(row)
    keys = ("trainer_publish_s", "trainer_release_s", "wait_s", "sampler_load_s",
            "released_to_loaded_s", "end_to_end_s")
    phases: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for group in ("trainer_release_phases", "sampler_follower_phases", "sampler_load_phases"):
            for name, value in (row.get(group) or {}).items():
                phases[f"{group.split('_')[0]}.{name}"].append(value)
    # the first release is loaded before the first round (start-up, not a policy update)
    later = [r for r in rows if r["seq"] > 1]
    return {
        "run": Path(run).name, "loads": len(rows), "rows": rows,
        "summary_after_first": {k: _spread([r[k] for r in later]) for k in keys},
        "phases": {name: _spread(values) for name, values in sorted(phases.items())},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.eval.sampler_islands")
    sub = parser.add_subparsers(dest="command", required=True)
    scale = sub.add_parser("scale")
    scale.add_argument("run", type=Path)
    scale.add_argument("--baseline", type=Path, default=None)
    scale.add_argument("--samples-per-group", type=int, default=None)
    latency = sub.add_parser("latency")
    latency.add_argument("run", type=Path)
    args = parser.parse_args(argv)
    if args.command == "scale":
        report = scale_report(args.run, samples_per_group=args.samples_per_group)
        if args.baseline is not None:
            report["compare"] = compare(report, scale_report(args.baseline,
                                                             samples_per_group=args.samples_per_group))
    else:
        report = latency_report(args.run)
    json.dump(report, sys.stdout, indent=1, sort_keys=False)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
