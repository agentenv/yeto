"""Check a multi-island run's data split from its island tapes (S27 task assignment).

Usage: ``python -m tools.eval.data_split_check TAPE.jsonl [TAPE.jsonl ...]``
(every island tape of the run, every incarnation; ``--rl-observe-timeline``
writes the ``rl_trajectory_reward`` records this reads). Prints one JSON report
and exits 1 when a check fails.

Checks:

1. ``slots``: each island's ``rl_driver_start`` names its ``data_slot``; the
   slots of the islands are distinct and have one count N.
2. ``disjoint``: no row (``data_row``, else ``task_id``) is trained by two
   islands. This covers "no duplicate task in one round" and is stricter.
3. ``own_slot``: every trained ``data_row`` of island k satisfies
   ``row % N == k``.
4. ``relaunch``: every process of an island after its first one either
   restored a data position (``rl_data_position_restored``) or reported none
   (``rl_data_position_unknown``, a warning); its first trained row is not a
   row of a round that an earlier process brought to its sync boundary
   (``rl_full_parameter_round`` / ``rl_local_round``). The rows of a round that
   stopped before its boundary are drawn again by design (``data_position``).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

START, REWARD = "rl_driver_start", "rl_trajectory_reward"
RESTORED, UNKNOWN = "rl_data_position_restored", "rl_data_position_unknown"
# the boundary event of a round: (event, field, offset to its rollout id)
BOUNDARIES = {"rl_full_parameter_round": ("policy_version", -1), "rl_local_round": ("rollout_id", 0)}


def read_records(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    out = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue  # a torn last line
    return sorted(out, key=lambda r: float(r.get("time_unix", 0.0)))


def _row(record: Mapping[str, Any]) -> Any:
    return record["data_row"] if record.get("data_row") is not None else (record.get("task_id") or None)


def check(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """The report of :func:`main` for tape records in time order."""
    slots: dict[int, set[str]] = defaultdict(set)
    processes: dict[int, list[dict[str, Any]]] = defaultdict(list)  # one entry per rl_driver_start
    for record in records:
        island = int(record.get("island_id", -1))
        event = record.get("event")
        if event == START:
            slots[island].add(str(record.get("data_slot")))
            processes[island].append({"rows": [], "by_round": defaultdict(list), "bounded": set(),
                                      "restored": False, "unknown": False})
        elif not processes[island]:
            continue  # records before the first start (none expected)
        elif event == REWARD and _row(record) is not None:
            processes[island][-1]["rows"].append(_row(record))
            processes[island][-1]["by_round"][int(record.get("rollout_id", -1))].append(_row(record))
        elif event in BOUNDARIES and record.get(BOUNDARIES[event][0]) is not None:
            field, offset = BOUNDARIES[event]
            processes[island][-1]["bounded"].add(int(record[field]) + offset)
        elif event == RESTORED:
            processes[island][-1]["restored"] = True
        elif event == UNKNOWN:
            processes[island][-1]["unknown"] = True
    problems: list[str] = []
    named = {island: next(iter(s)) for island, s in slots.items() if len(s) == 1}
    counts = {slot.split("/")[1] for slot in named.values() if "/" in slot}
    if len(named) != len(slots) or len(counts) != 1 or len(set(named.values())) != len(named):
        problems.append(f"slots: islands do not hold distinct slots of one count: {dict(slots)}")
    owners: dict[Any, set[int]] = defaultdict(set)
    for island, runs in processes.items():
        for run in runs:
            for row in run["rows"]:
                owners[row].add(island)
    shared = sorted((str(row) for row, islands in owners.items() if len(islands) > 1))
    if shared:
        problems.append(f"disjoint: {len(shared)} rows trained by more than one island: {shared[:10]}")
    for island, slot in named.items():
        index, _, count = slot.partition("/")
        if not count:
            continue
        foreign = sorted({row for run in processes[island] for row in run["rows"]
                          if isinstance(row, int) and row % int(count) != int(index)})
        if foreign:
            problems.append(f"own_slot: island {island} trained rows outside slot {slot}: {foreign[:10]}")
    relaunches = []
    for island, runs in processes.items():
        seen: set[Any] = set()
        for n, run in enumerate(runs):
            if n:
                first = run["rows"][0] if run["rows"] else None
                relaunches.append({"island": island, "process": n, "restored": run["restored"],
                                   "unknown": run["unknown"], "first_row": first})
                if not (run["restored"] or run["unknown"]):
                    problems.append(f"relaunch: island {island} process {n} neither restored nor "
                                    "reported its data position")
                if first is not None and first in seen:
                    problems.append(f"relaunch: island {island} process {n} started at row {first}, "
                                    "trained before")
            seen.update(row for r in run["bounded"] for row in run["by_round"].get(r, ()))
    return {"ok": not problems, "problems": problems, "slots": {str(k): v for k, v in named.items()},
            "rows_per_island": {str(k): sum(len(r["rows"]) for r in v) for k, v in processes.items()},
            "relaunches": relaunches}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tapes", nargs="+")
    report = check(read_records(parser.parse_args(argv).tapes))
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
