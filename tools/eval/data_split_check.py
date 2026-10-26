"""Check a multi-island run's data split from its island tapes (S27 task assignment).

Usage: ``python -m tools.eval.data_split_check TAPE.jsonl [TAPE.jsonl ...]``
(every island tape of the run, every incarnation; ``--rl-observe-timeline``
writes the records this reads). Prints one JSON report and exits 1 when a
check fails.

Rows of a round: the round's ``rl_round_data_rows`` (every trained row, as
runs) when the process wrote one; else its ``rl_trajectory_reward`` records,
which stop at 256 samples per round. ``rows_source`` in the report counts the
rounds of each island by source.

Checks:

1. ``slots``: each island's ``rl_driver_start`` names its ``data_slot``; the
   slots of the islands are distinct and have one count N.
2. ``disjoint``: no row (``data_row``, else ``task_id``) is trained by two
   islands. This covers "no duplicate task in one round" and is stricter.
3. ``round_rows``: every ``rl_round_data_rows`` is well formed and holds every
   ``data_row`` of the same round's ``rl_trajectory_reward`` records.
4. ``own_slot``: every trained ``data_row`` of island k satisfies
   ``row % N == k``.
5. ``relaunch``: every process of an island after its first one either
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
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from yeto.rl.engine.data_split import rows_of_runs
from yeto.rl.engine.timeline import validate_round_data_rows

START, REWARD, ROUND_ROWS = "rl_driver_start", "rl_trajectory_reward", "rl_round_data_rows"
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


def _new_process() -> dict[str, Any]:
    return {"reward_rows": defaultdict(list), "round_rows": {}, "bounded": set(),
            "restored": False, "unknown": False}


def _read_processes(records: Iterable[Mapping[str, Any]], problems: list[str]
                    ) -> tuple[dict[int, set[str]], dict[int, list[dict[str, Any]]]]:
    """The data slots each island named and its processes (one per
    ``rl_driver_start``), in time order."""
    slots: dict[int, set[str]] = defaultdict(set)
    processes: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        island = int(record.get("island_id", -1))
        event = record.get("event")
        if event == START:
            slots[island].add(str(record.get("data_slot")))
            processes[island].append(_new_process())
            continue
        if not processes[island]:
            continue  # records before the first start (none expected)
        run = processes[island][-1]
        if event == REWARD and _row(record) is not None:
            run["reward_rows"][int(record.get("rollout_id", -1))].append(_row(record))
        elif event == ROUND_ROWS:
            malformed = validate_round_data_rows(record)
            if malformed:
                problems.append(f"round_rows: island {island} round {record.get('rollout_id')}: {malformed}")
            else:
                run["round_rows"][record["rollout_id"]] = rows_of_runs(record["rows"])
        elif event in BOUNDARIES and record.get(BOUNDARIES[event][0]) is not None:
            field, offset = BOUNDARIES[event]
            run["bounded"].add(int(record[field]) + offset)
        elif event == RESTORED:
            run["restored"] = True
        elif event == UNKNOWN:
            run["unknown"] = True
    return slots, processes


def _merge_rounds(island: int, run: dict[str, Any], problems: list[str]) -> None:
    """Sets ``by_round`` (each round's round record, else its reward records),
    ``rows`` (in round order) and ``sources`` of one process."""
    run["by_round"], run["sources"] = {}, Counter()
    for rollout_id in sorted(set(run["reward_rows"]) | set(run["round_rows"])):
        rewarded = run["reward_rows"].get(rollout_id, [])
        full = run["round_rows"].get(rollout_id)
        if full is None:
            run["by_round"][rollout_id] = rewarded
            run["sources"]["rewards"] += 1
            continue
        missing = sorted({row for row in rewarded if isinstance(row, int)} - set(full))
        if missing:
            problems.append(f"round_rows: island {island} round {rollout_id}: rewarded rows "
                            f"missing from {ROUND_ROWS}: {missing[:10]}")
        run["by_round"][rollout_id] = full
        run["sources"]["rounds"] += 1
    run["rows"] = [row for rows in run["by_round"].values() for row in rows]


def _slot_problems(slots: Mapping[int, set[str]]) -> tuple[dict[int, str], list[str]]:
    named = {island: next(iter(s)) for island, s in slots.items() if len(s) == 1}
    counts = {slot.split("/")[1] for slot in named.values() if "/" in slot}
    if len(named) != len(slots) or len(counts) != 1 or len(set(named.values())) != len(named):
        return named, [f"slots: islands do not hold distinct slots of one count: {dict(slots)}"]
    return named, []


def _disjoint_problems(processes: Mapping[int, list[dict[str, Any]]]) -> list[str]:
    owners: dict[Any, set[int]] = defaultdict(set)
    for island, runs in processes.items():
        for run in runs:
            for row in run["rows"]:
                owners[row].add(island)
    shared = sorted((str(row) for row, islands in owners.items() if len(islands) > 1))
    return [f"disjoint: {len(shared)} rows trained by more than one island: {shared[:10]}"] if shared else []


def _own_slot_problems(named: Mapping[int, str], processes: Mapping[int, list[dict[str, Any]]]) -> list[str]:
    problems = []
    for island, slot in named.items():
        index, _, count = slot.partition("/")
        if not count:
            continue
        foreign = sorted({row for run in processes[island] for row in run["rows"]
                          if isinstance(row, int) and row % int(count) != int(index)})
        if foreign:
            problems.append(f"own_slot: island {island} trained rows outside slot {slot}: {foreign[:10]}")
    return problems


def _relaunches(processes: Mapping[int, list[dict[str, Any]]], problems: list[str]) -> list[dict[str, Any]]:
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
    return relaunches


def check(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """The report of :func:`main` for tape records in time order."""
    problems: list[str] = []
    slots, processes = _read_processes(records, problems)
    for island, runs in processes.items():
        for run in runs:
            _merge_rounds(island, run, problems)
    named, slot_problems = _slot_problems(slots)
    problems += slot_problems + _disjoint_problems(processes) + _own_slot_problems(named, processes)
    relaunches = _relaunches(processes, problems)
    sources = {island: sum((run["sources"] for run in runs), Counter()) for island, runs in processes.items()}
    return {"ok": not problems, "problems": problems, "slots": {str(k): v for k, v in named.items()},
            "rows_per_island": {str(k): sum(len(r["rows"]) for r in v) for k, v in processes.items()},
            "rows_source": {str(k): dict(v) for k, v in sources.items()},
            "relaunches": relaunches}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tapes", nargs="+")
    report = check(read_records(parser.parse_args(argv).tapes))
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
