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
positions that island reported trained: a tape row of a committed round
(``rl_dispense_commit`` names the round) that no report covers is a problem.
A round the tape trained but never committed (the process died between the
train step and the commit; the dispenser gave its rows out again) is
counted, not a problem. A tape without ``rl_dispense_commit`` records (before
the commit existed) counts every round as committed.

Streams (S27 P3, ``dispense_streams``): a position's stream is its high bits;
fresh positions count per stream. Per stream (source and task label) the
summary has the positions granted, trained, dropped, released and reclaimed,
per island the positions of each stream it trained, and the mix: each
stream's share of the fresh positions given out against its weight, and the
largest gap in positions over every prefix of grants (``mix_error``; below 1
when every island takes every stream, Tijdeman's bound). With streams the tape
check needs the run's prompt rows (``--prompts`` and the island flags
``--mix`` / ``--by-label``) to map positions to rows.

Sampler islands (S27 P4): their island ids are ``SAMPLER_HOLDER_BASE`` plus a
hash of the sampler id; a sampler reports a position trained when its group
is in the sample buffer. ``--tapes`` of the buffer trainers then compare the
rows the trainers trained (all trainers together) with the rows the samplers
reported, and count the rows the trainers passed over as too stale
(``rl_buffer_stale_skipped`` with ``stale_positions``).

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


STREAM_SHIFT = 40  # syncer/src/dispense.rs
LOCAL_MASK = (1 << STREAM_SHIFT) - 1


class _State:
    def __init__(self) -> None:
        self.next: dict[int, int] = {}  # stream -> next fresh local position
        self.holder: dict[int, tuple[int, str]] = {}  # position -> (island, incarnation)
        self.queued: set[int] = set()
        self.settled: dict[int, str] = {}  # position -> trained / dropped / exhausted
        self.trained_by: dict[int, int] = {}  # position -> island

    @property
    def given(self) -> int:
        return sum(self.next.values())


def _mix_gap(given: Mapping[int, int], weights: Mapping[int, int]) -> float:
    """The largest |given_s - n w_s / W| over the streams (positions)."""
    total_w = sum(weights.values())
    n = sum(given.values())
    if not total_w or not n:
        return 0.0
    return max(abs(given.get(s, 0) - n * w / total_w) for s, w in weights.items())


def check(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    problems: list[str] = []
    st = _State()
    cut: _State | None = None
    rows: int | None = None
    streams: dict[int, dict[str, Any]] = {}  # stream -> {name, rows, weight}
    per_island: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"granted": 0, "trained": 0, "dropped": 0, "requeued": 0, "released": 0,
                 "reclaimed": 0, "blocks": 0, "incarnations": set(), "passes": set(),
                 "trained_by_stream": Counter(), "accept": None})
    per_stream: dict[int, Counter[str]] = defaultdict(Counter)
    restores = rolled_back = 0
    mix_error = 0.0

    def stream_rows(stream: int) -> int | None:
        if stream in streams:
            return int(streams[stream]["rows"])
        return rows if stream == 0 and len(streams) <= 1 else None

    for n, record in enumerate(records):
        kind = record.get("kind")
        if not isinstance(kind, str) or not kind.startswith("dispense_"):
            continue
        island = record.get("island_id")
        who = (island, record.get("incarnation"))
        if kind == "dispense_streams":
            streams = {int(x["stream"]): {"name": x.get("name", ""), "rows": int(x["rows"]),
                                          "weight": int(x["weight"])} for x in record.get("streams") or ()}
        elif kind == "dispense_open":
            rows = int(record.get("rows") or 0) or rows
            per_island[island]["incarnations"].add(record.get("incarnation"))
            per_island[island]["accept"] = record.get("accept")
        elif kind == "dispense_grant":
            info = per_island[island]
            info["blocks"] += 1
            for p in _positions(record.get("ranges") or []):
                stream, local = p >> STREAM_SHIFT, p & LOCAL_MASK
                fresh = st.next.get(stream, 0)
                if p in st.holder:
                    problems.append(f"line {n}: position {p} granted to island {island} while island "
                                    f"{st.holder[p][0]} holds it")
                elif p in st.queued:
                    st.queued.discard(p)
                elif local == fresh:
                    st.next[stream] = fresh + 1
                elif local > fresh:
                    problems.append(f"line {n}: position {p} granted past the next fresh position "
                                    f"{fresh} of stream {stream}")
                    st.next[stream] = local + 1
                else:
                    problems.append(f"line {n}: position {p} granted again although it is "
                                    f"{st.settled.get(p, 'not queued')}")
                st.holder[p] = who
                info["granted"] += 1
                per_stream[stream]["granted"] += 1
                count = stream_rows(stream)
                if count:
                    info["passes"].add(local // count)
            if len(streams) > 1:
                mix_error = max(mix_error, _mix_gap(st.next, {s: v["weight"] for s, v in streams.items()}))
        elif kind == "dispense_report":
            report = record.get("report")
            exhausted = set(_positions(record.get("redo_exhausted") or []))
            info = per_island[island]
            for p in _positions(record.get("ranges") or []):
                if st.holder.get(p) != who:
                    problems.append(f"line {n}: island {island} reports position {p} it does not hold")
                    continue
                del st.holder[p]
                stream = p >> STREAM_SHIFT
                if report == "trained":
                    if p in st.trained_by:
                        problems.append(f"line {n}: position {p} trained twice (islands "
                                        f"{st.trained_by[p]} and {island})")
                    st.trained_by[p] = island
                    st.settled[p] = "trained"
                    info["trained"] += 1
                    info["trained_by_stream"][stream] += 1
                    per_stream[stream]["trained"] += 1
                elif report == "dropped":
                    st.settled[p] = "dropped"
                    info["dropped"] += 1
                    per_stream[stream]["dropped"] += 1
                elif report == "requeue" and p in exhausted:
                    st.settled[p] = "exhausted"
                    info["requeued"] += 1
                    per_stream[stream]["redo_exhausted"] += 1
                elif report in ("requeue", "release"):
                    st.queued.add(p)
                    name = "requeued" if report == "requeue" else "released"
                    info[name] += 1
                    per_stream[stream][name] += 1
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
                per_stream[p >> STREAM_SHIFT]["reclaimed"] += 1
        elif kind == "dispense_checkpoint":
            if int(record.get("next", -1)) != st.given:
                problems.append(f"line {n}: checkpoint next {record.get('next')} != replayed {st.given}")
            cut = copy.deepcopy(st)
        elif kind == "dispense_restore":
            restores += 1
            base = cut if cut is not None else _State()
            if int(record.get("next", -1)) != base.given:
                problems.append(f"line {n}: restore next {record.get('next')} != the last checkpoint "
                                f"cut {base.given}")
            rolled_back += sum(1 for p in st.settled if p not in base.settled)
            st = copy.deepcopy(base)
            st.queued |= set(st.holder)  # a restart retires every process of the cut
            st.holder.clear()
            if int(record.get("queued", -1)) != len(st.queued):
                problems.append(f"line {n}: restore queued {record.get('queued')} != the cut's "
                                f"{len(st.queued)}")
    accounted = len(st.settled) + len(st.queued) + len(st.holder)
    if accounted != st.given:
        problems.append(f"accounting: settled {len(st.settled)} + queued {len(st.queued)} + held "
                        f"{len(st.holder)} != given out {st.given}")
    names = {s: (v["name"] or f"stream{s}") for s, v in streams.items()}
    islands = {
        str(island): {**{k: v for k, v in info.items()
                         if k not in ("incarnations", "passes", "trained_by_stream", "accept")},
                      "incarnations": len(info["incarnations"]),
                      "passes": sorted(info["passes"]),
                      **({"accept": info["accept"],
                          "trained_by_stream": {names.get(s, str(s)): c for s, c in
                                                sorted(info["trained_by_stream"].items())}}
                         if len(streams) > 1 else {})}
        for island, info in sorted(per_island.items(), key=lambda kv: (kv[0] is None, kv[0]))
        if island is not None
    }
    summary: dict[str, Any] = {
        "ok": not problems,
        "problems": problems[:50],
        "problem_count": len(problems),
        "rows": rows,
        "given_out": st.given,
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
    if len(streams) > 1:
        total_w = sum(v["weight"] for v in streams.values())
        given = sum(st.next.values()) or 1
        trained = sum(per_stream[s]["trained"] for s in streams) or 1
        summary["streams"] = {
            names[s]: {"stream": s, "rows": v["rows"], "weight_share": round(v["weight"] / total_w, 6),
                       "fresh": st.next.get(s, 0), "fresh_share": round(st.next.get(s, 0) / given, 6),
                       "trained_share": round(per_stream[s]["trained"] / trained, 6),
                       **{k: per_stream[s][k] for k in ("granted", "trained", "dropped", "released",
                                                        "reclaimed", "requeued", "redo_exhausted")}}
            for s, v in sorted(streams.items())}
        summary["mix_error"] = round(mix_error, 6)
    return summary


def tape_rows(tapes: Iterable[Mapping[str, Any]]) -> tuple[dict[int, Counter[int]], dict[int, Counter[int]]]:
    """``(committed, uncommitted)`` trained tape rows per island (module doc).
    Records are read in order; ``rl_driver_start`` starts a new process."""
    from yeto.rl.engine.data_split import rows_of_runs

    # the Modal tape of a relaunched island starts with the records of the
    # incarnations before it (s27-dh-a1: rl-island-1.inc1.jsonl): count each once
    records, seen = [], set()
    for r in tapes:
        if r.get("island_id") is None:
            continue
        key = json.dumps(r, sort_keys=True)
        if key not in seen:
            seen.add(key)
            records.append(r)
    with_commits = {int(r["island_id"]) for r in records if r.get("event") == "rl_dispense_commit"}
    committed: dict[int, Counter[int]] = defaultdict(Counter)
    uncommitted: dict[int, Counter[int]] = defaultdict(Counter)
    waiting: dict[int, dict[int, Counter[int]]] = defaultdict(dict)  # island -> round -> rows
    for record in records:  # every island tape record carries its island_id
        island, event = int(record["island_id"]), record.get("event")
        if event == "rl_driver_start":
            for counts in waiting.pop(island, {}).values():
                uncommitted[island].update(counts)
        elif event == "rl_round_data_rows":
            counts = Counter(rows_of_runs(record.get("rows") or []))
            if island not in with_commits:
                committed[island].update(counts)
            else:
                waiting[island].setdefault(int(record.get("rollout_id", -1)), Counter()).update(counts)
        elif event == "rl_dispense_commit":  # a commit names the rounds it reported
            for round_id in record.get("rounds") or ():
                counts = waiting[island].pop(int(round_id), None)
                if counts:
                    committed[island].update(counts)
    for island, rounds in waiting.items():
        for counts in rounds.values():
            uncommitted[island].update(counts)
    return dict(committed), dict(uncommitted)


def stale_positions(tapes: Iterable[Mapping[str, Any]]) -> list[int]:
    """The dispenser positions of the groups buffer trainers passed over as too
    stale (``rl_buffer_stale_skipped.stale_positions``, S27 P4)."""
    from yeto.rl.engine.data_split import rows_of_runs

    out: list[int] = []
    for record in tapes:
        if record.get("event") == "rl_buffer_stale_skipped":
            out.extend(rows_of_runs(record.get("stale_positions") or []))
    return out


def check_tapes(summary: Mapping[str, Any], tapes: Iterable[Mapping[str, Any]], *,
                seed: int, info: dict[str, Any] | None = None, table: Any = None) -> list[str]:
    """Rows the island tapes trained in committed rounds (``rl_round_data_rows``)
    against the rows of the positions each island reported trained (as multisets).
    ``info`` receives the uncommitted tape rows per island. ``table``: the
    run's :class:`~yeto.rl.engine.task_dispense.StreamTable` (needed with
    streams). With sampler islands (S27 P4) the samplers' reports are compared
    with the rows of all buffer trainer tapes together."""
    from yeto.rl.engine.task_dispense import SAMPLER_HOLDER_BASE, PositionMap

    tapes = list(tapes)
    rows = summary.get("rows")
    if not rows:
        return ["the dispenser log names no row count"]
    if summary.get("streams") and table is None:
        return ["the dispenser log has streams: the tape check needs --prompts (and --mix / --by-label)"]
    pmap = PositionMap(int(rows), seed=int(seed)) if table is None else PositionMap(seed=int(seed), table=table)
    reported: dict[int, Counter[int]] = {
        int(i): Counter(pmap.row(p) for p in ps)
        for i, ps in summary.get("trained_positions_by_island", {}).items()}
    committed, uncommitted = tape_rows(tapes)
    if info is not None:
        info["tape_rows_committed"] = {str(i): sum(c.values()) for i, c in sorted(committed.items())}
        info["tape_rows_uncommitted"] = {str(i): sum(c.values()) for i, c in sorted(uncommitted.items())}
    problems = []
    samplers = sorted(i for i in reported if i >= SAMPLER_HOLDER_BASE)
    if samplers:
        pool: Counter[int] = Counter()
        for i in samplers:
            pool.update(reported[i])
        trained: Counter[int] = Counter()
        for counts in committed.values():
            trained.update(counts)
        stale = stale_positions(tapes)
        if info is not None:
            sampled = {int(p) for i in samplers
                       for p in summary.get("trained_positions_by_island", {}).get(str(i), ())}
            info["sampler_rows_reported"] = sum(pool.values())
            info["trainer_rows_trained"] = sum(trained.values())
            info["stale_positions"] = len(stale)
            info["stale_positions_not_sampled"] = sum(1 for p in stale if p not in sampled)
        missing = trained - pool
        if missing:
            problems.append(f"buffer trainers: {sum(missing.values())} trained tape rows were reported by "
                            f"no sampler (e.g. rows {sorted(missing)[:10]})")
        return problems
    for isl, counts in sorted(committed.items()):
        missing = counts - reported.get(isl, Counter())
        if missing:
            sample = sorted(missing)[:10]
            problems.append(f"island {isl}: {sum(missing.values())} trained tape rows of committed "
                            f"rounds have no trained report (e.g. rows {sample})")
    return problems


def prompt_table(path: str, *, mix: str | None, by_label: bool) -> Any:
    """The run's StreamTable from its prompt rows (the island flags' meaning)."""
    from yeto.rl.engine.task_dispense import StreamTable

    return StreamTable.from_prompts(path, mix=mix.split(",") if mix else None, by_label=by_label)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", nargs="+", help="task-dispense.jsonl (every syncer incarnation)")
    parser.add_argument("--tapes", nargs="*", default=(), help="island tapes (optional)")
    parser.add_argument("--seed", type=int, default=None, help="the run seed (needed with --tapes)")
    parser.add_argument("--positions", action="store_true", help="print the trained positions too")
    parser.add_argument("--prompts", default=None,
                        help="the run's prompt rows (--data, or the launcher's data-mix.jsonl): "
                        "needed by --tapes when the run has streams")
    parser.add_argument("--mix", default=None, help="the --rl-dispense-mix weights of the run")
    parser.add_argument("--by-label", action="store_true", help="the run split its streams by task label")
    args = parser.parse_args(argv)
    summary = check(read_records(args.log))
    if args.tapes:
        if args.seed is None:
            parser.error("--tapes needs --seed")
        table = (prompt_table(args.prompts, mix=args.mix, by_label=args.by_label)
                 if args.prompts else None)
        if table is not None and summary.get("streams"):
            logged = [(v["rows"], v["stream"]) for v in summary["streams"].values()]
            if sorted(logged, key=lambda x: x[1]) != [(len(s.rows), i) for i, s in enumerate(table.streams)]:
                parser.error("--prompts / --mix / --by-label do not give the logged streams")
        tape_problems = check_tapes(summary, read_records(args.tapes), seed=args.seed, info=summary,
                                    table=table)
        summary["tape_problems"] = tape_problems
        summary["ok"] = summary["ok"] and not tape_problems
    if not args.positions:
        summary.pop("trained_positions_by_island", None)
    json.dump(summary, sys.stdout, indent=2, sort_keys=True)
    print()
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
