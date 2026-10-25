"""Offline check of a task-dispenser run (S27 P2): no row given twice, none lost.

Input: the syncer's ``task-dispense.jsonl`` (``--dispense-log``; every syncer
incarnation of the run appends to it), optionally the island tapes.

The check replays the log and keeps, for every position, where it is:
fresh (never given), held by one island process (granted, not reported),
queued (returned, requeued, released), or settled (trained, dropped, redo
exhausted). It fails when:

* a position is granted while another process holds it, or granted from
  neither the fresh range nor the queue;
* a report or reclaim names a position the process does not hold;
* a position is reported trained twice;
* a restart (``dispense_restore``) does not match the last checkpoint cut in
  the log (``dispense_checkpoint``). A restart returns to that cut: positions
  settled after it are counted as rolled back and are given out again.

It reports (without failing) what is still held or queued at the end of the
log and, per island, the positions it was granted and trained and the passes
it read. With ``--tapes`` and ``--seed`` it also maps every trained position
to its row (``task_dispense.PositionMap``) and compares, per island, the rows
the island tapes trained (``rl_round_data_rows``) with the rows of the
positions that island reported trained: a tape row that no report covers is
a problem.

    python -m tools.eval.dispense_check task-dispense.jsonl [--tapes rl-island-*.jsonl --seed S]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

SETTLED_KINDS = ("trained", "dropped")


def read_records(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # a torn last line
    return out


def _positions(ranges: Iterable[Iterable[int]]) -> list[int]:
    return [p for a, b in ranges for p in range(int(a), int(b))]


class _State:
    def __init__(self) -> None:
        self.next = 0
        self.holder: dict[int, tuple[int, str]] = {}  # position -> (island, incarnation)
        self.queued: set[int] = set()
        self.settled: dict[int, str] = {}  # position -> trained / dropped / exhausted
        self.trained_by: dict[int, int] = {}  # position -> island


def check(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    problems: list[str] = []
    st = _State()
    cut: _State | None = None
    rows: int | None = None
    per_island: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"granted": 0, "trained": 0, "dropped": 0, "requeued": 0, "released": 0,
                 "reclaimed": 0, "blocks": 0, "incarnations": set(), "passes": set()})
    restores = rolled_back = 0
    trained_reports: Counter[int] = Counter()
    for n, record in enumerate(records):
        kind = record.get("kind")
        if not isinstance(kind, str) or not kind.startswith("dispense_"):
            continue
        island = record.get("island_id")
        who = (island, record.get("incarnation"))
        if kind == "dispense_open":
            rows = int(record.get("rows") or 0) or rows
            per_island[island]["incarnations"].add(record.get("incarnation"))
        elif kind == "dispense_grant":
            info = per_island[island]
            info["blocks"] += 1
            for p in _positions(record.get("ranges") or []):
                if p in st.holder:
                    problems.append(f"line {n}: position {p} granted to island {island} while island "
                                    f"{st.holder[p][0]} holds it")
                elif p in st.queued:
                    st.queued.discard(p)
                elif p == st.next:
                    st.next += 1
                elif p > st.next:
                    problems.append(f"line {n}: position {p} granted past the next fresh position {st.next}")
                    st.next = p + 1
                else:
                    problems.append(f"line {n}: position {p} granted again although it is "
                                    f"{st.settled.get(p, 'not queued')}")
                st.holder[p] = who
                info["granted"] += 1
                if rows:
                    info["passes"].add(p // rows)
        elif kind == "dispense_report":
            report = record.get("report")
            exhausted = set(_positions(record.get("redo_exhausted") or []))
            info = per_island[island]
            for p in _positions(record.get("ranges") or []):
                if st.holder.get(p) != who:
                    problems.append(f"line {n}: island {island} reports position {p} it does not hold")
                    continue
                del st.holder[p]
                if report == "trained":
                    trained_reports[p] += 1
                    if p in st.trained_by:
                        problems.append(f"line {n}: position {p} trained twice (islands "
                                        f"{st.trained_by[p]} and {island})")
                    st.trained_by[p] = island
                    st.settled[p] = "trained"
                    info["trained"] += 1
                elif report == "dropped":
                    st.settled[p] = "dropped"
                    info["dropped"] += 1
                elif report == "requeue" and p in exhausted:
                    st.settled[p] = "exhausted"
                    info["requeued"] += 1
                elif report in ("requeue", "release"):
                    st.queued.add(p)
                    info["requeued" if report == "requeue" else "released"] += 1
                else:
                    problems.append(f"line {n}: unknown report kind {report!r}")
        elif kind == "dispense_reclaim":
            for p in _positions(record.get("ranges") or []):
                if st.holder.get(p) != who:
                    problems.append(f"line {n}: reclaim of position {p} that island {island} does not hold")
                    continue
                del st.holder[p]
                st.queued.add(p)
                per_island[island]["reclaimed"] += 1
        elif kind == "dispense_checkpoint":
            if int(record.get("next", -1)) != st.next:
                problems.append(f"line {n}: checkpoint next {record.get('next')} != replayed {st.next}")
            cut = copy.deepcopy(st)
        elif kind == "dispense_restore":
            restores += 1
            base = cut if cut is not None else _State()
            if int(record.get("next", -1)) != base.next:
                problems.append(f"line {n}: restore next {record.get('next')} != the last checkpoint "
                                f"cut {base.next}")
            rolled_back += sum(1 for p in st.settled if p not in base.settled)
            st = copy.deepcopy(base)
            st.queued |= set(st.holder)  # a restart retires every process of the cut
            st.holder.clear()
            if int(record.get("queued", -1)) != len(st.queued):
                problems.append(f"line {n}: restore queued {record.get('queued')} != the cut's "
                                f"{len(st.queued)}")
    accounted = len(st.settled) + len(st.queued) + len(st.holder)
    if accounted != st.next:
        problems.append(f"accounting: settled {len(st.settled)} + queued {len(st.queued)} + held "
                        f"{len(st.holder)} != given out {st.next}")
    islands = {
        str(island): {**{k: v for k, v in info.items() if k not in ("incarnations", "passes")},
                      "incarnations": len(info["incarnations"]),
                      "passes": sorted(info["passes"])}
        for island, info in sorted(per_island.items(), key=lambda kv: (kv[0] is None, kv[0]))
        if island is not None
    }
    return {
        "ok": not problems,
        "problems": problems[:50],
        "problem_count": len(problems),
        "rows": rows,
        "given_out": st.next,
        "trained": sum(1 for v in st.settled.values() if v == "trained"),
        "dropped": sum(1 for v in st.settled.values() if v == "dropped"),
        "redo_exhausted": sum(1 for v in st.settled.values() if v == "exhausted"),
        "queued_at_end": len(st.queued),
        "held_at_end": len(st.holder),
        "restores": restores,
        "rolled_back_after_restore": rolled_back,
        "islands": islands,
        "trained_positions_by_island": {str(i): sorted(p for p, j in st.trained_by.items() if j == i)
                                        for i in sorted({j for j in st.trained_by.values()})},
    }


def check_tapes(summary: Mapping[str, Any], tapes: Iterable[Mapping[str, Any]], *, seed: int,
                island_of_tape: Mapping[int, int] | None = None) -> list[str]:
    """Rows the island tapes trained (``rl_round_data_rows``) against the rows of
    the positions each island reported trained (as multisets)."""
    from yeto.rl.engine.data_split import rows_of_runs
    from yeto.rl.engine.task_dispense import PositionMap

    rows = summary.get("rows")
    if not rows:
        return ["the dispenser log names no row count"]
    pmap = PositionMap(int(rows), seed=int(seed))
    reported: dict[int, Counter[int]] = {
        int(i): Counter(pmap.row(p) for p in ps)
        for i, ps in summary.get("trained_positions_by_island", {}).items()}
    taped: dict[int, Counter[int]] = defaultdict(Counter)
    island = None
    for record in tapes:
        event = record.get("event") or record.get("kind")
        if event == "rl_driver_start":
            island = record.get("learner_id", record.get("island_id"))
        if event == "rl_round_data_rows" and island is not None:
            taped[int(island)].update(rows_of_runs(record.get("rows") or []))
    problems = []
    for isl, counts in sorted(taped.items()):
        missing = counts - reported.get(isl, Counter())
        if missing:
            sample = sorted(missing)[:10]
            problems.append(f"island {isl}: {sum(missing.values())} trained tape rows have no "
                            f"trained report (e.g. rows {sample})")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", nargs="+", help="task-dispense.jsonl (every syncer incarnation)")
    parser.add_argument("--tapes", nargs="*", default=(), help="island tapes (optional)")
    parser.add_argument("--seed", type=int, default=None, help="the run seed (needed with --tapes)")
    parser.add_argument("--positions", action="store_true", help="print the trained positions too")
    args = parser.parse_args(argv)
    summary = check(read_records(args.log))
    if args.tapes:
        if args.seed is None:
            parser.error("--tapes needs --seed")
        tape_problems = check_tapes(summary, read_records(args.tapes), seed=args.seed)
        summary["tape_problems"] = tape_problems
        summary["ok"] = summary["ok"] and not tape_problems
    if not args.positions:
        summary.pop("trained_positions_by_island", None)
    json.dump(summary, sys.stdout, indent=2, sort_keys=True)
    print()
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
