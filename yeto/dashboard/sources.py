"""Tape discovery and incremental tailing for the dashboard.

Each source keeps its own byte offset (independent of the W&B forwarder's
``OffsetStore``). Only complete lines are consumed; a torn tail waits for
the next poll. Offsets are passed to ``Reducer.feed`` so a re-read never
double counts.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterator

_ISLAND_RE = (re.compile(r"rl-island-(\d+)"), re.compile(r"island-(\d+)"), re.compile(r"-l(\d+)(?:-|$)"))


def island_hint(path: Path) -> str | None:
    """Best-effort island id for a stream that does not carry ``island_id``
    (the controller journal): a sibling ``rl-island-N.jsonl`` or an
    ``island-N`` / ``-lN-`` component in the path."""
    siblings = sorted(path.parent.glob("rl-island-*.jsonl"))
    if len(siblings) == 1:
        m = _ISLAND_RE[0].search(siblings[0].name)
        if m:
            return m.group(1)
    for part in reversed(path.parts):
        for rx in _ISLAND_RE:
            m = rx.search(part)
            if m:
                return m.group(1)
    return None


_HOST_RE = re.compile(r"^l(\d+)$")


def host_island_hint(path: Path) -> str | None:
    """Island for a ``modal-hostmem-*.jsonl`` (no island_id): the ``l<N>`` path
    component next to ``rank<R>``, else a sibling ``rl-island-<N>.jsonl``."""
    parts = path.parts
    for i in range(len(parts) - 2, 0, -1):
        if parts[i].startswith("rank") and _HOST_RE.match(parts[i - 1]):
            return _HOST_RE.match(parts[i - 1]).group(1)
    siblings = sorted(path.parent.glob("rl-island-*.jsonl"))
    if len(siblings) == 1:
        m = _ISLAND_RE[0].search(siblings[0].name)
        if m:
            return m.group(1)
    return None


# non-JSONL files of ms-swift SFT islands and the syncer (see sft.py)
_SFT_EXTRA = ("syncer.log", "args.json", "island.json", "cmdline.txt")


def make_source(path):
    """A ``TapeSource``, or an ``SftSource`` for ms-swift / syncer native files."""
    from .sft import SftSource, role

    kind = role(Path(path))
    return SftSource(path, kind) if kind else TapeSource(path)


SYNCER_TAPE = "yeto-tape.jsonl"


def syncer_shard_tapes(path: Path) -> list[Path]:
    """The other shards' tapes of a fragment-sharded syncer next to its shard-0 tape
    (``yeto-tape.shard<i>.jsonl``, ``yeto.launch.syncer._shard_path``)."""
    return sorted(path.parent.glob(path.stem + ".shard*" + path.suffix)) if path.name == SYNCER_TAPE else []


def discover(paths: list[str], max_depth: int = 2) -> list[Path]:
    """Files given directly (a syncer tape with its shard tapes) plus ``*.jsonl`` up to
    ``max_depth`` below each dir."""
    out: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        p = Path(os.path.expanduser(raw))
        if p.is_file():
            cands = [p] + syncer_shard_tapes(p)
        elif p.is_dir():
            cands = []
            for depth in range(max_depth + 1):
                for pat in ("*.jsonl",) + _SFT_EXTRA:
                    cands += sorted(p.glob("/".join(["*"] * depth + [pat])))
        else:
            cands = []
        for c in cands:
            r = c.resolve()
            if c != p and c.suffix != ".jsonl" and not _sft_role(c):
                continue
            if r not in seen and r.is_file():
                seen.add(r)
                out.append(c)
    return out


def _sft_role(path: Path) -> str | None:
    from .sft import role

    return role(path)


_CLUSTER_TAPE_RE = re.compile(r"-l(\d+)-[A-Za-z0-9]+\.jsonl$")


def cluster_tape_island(path: Path) -> str | None:
    """The island of a launcher per-cluster event tape
    (``~/.yeto/runs/<run>/events/<prefix>-l<N>-<cloud>.jsonl``): N. The records'
    ``island_id`` is the island's *learner id*, which is not unique when a launch
    holds islands that each run as learner 0 of 1 (the sampler islands of a synced
    group or a pair), so this file's island names them."""
    if path.parent.name != "events":
        return None
    m = _CLUSTER_TAPE_RE.search(path.name)
    return m.group(1) if m else None


COST_EVENTS_FILE = "cost-events.jsonl"  # yeto.cost.guard.EVENTS_FILE (the dashboard does not import the guard)
UP_EVENTS_FILE = "events.jsonl"  # the ``run up`` monitor's tape, next to its state.json


def breakers_only(path: Path) -> bool:
    """A tape that feeds only the Breakers panel (``Reducer.feed_breaker``): the cost
    guard's events and the ``run up`` monitor's tape."""
    return path.name == COST_EVENTS_FILE or (path.name == UP_EVENTS_FILE and (path.parent / "state.json").is_file())


class TapeSource:
    def __init__(self, path: str | os.PathLike, island: str | None = None):
        self.path = Path(path)
        self.name = str(self.path)
        self.breakers_only = breakers_only(self.path)
        cluster = cluster_tape_island(self.path) if island is None else None
        self.island_fixed = cluster is not None
        self.island = island if island is not None else cluster if cluster is not None else (
            island_hint(self.path) if "journal" in self.path.name
            else host_island_hint(self.path) if "hostmem" in self.path.name else None)
        self.node = node_hint(self.path)
        self.offset = 0
        self.bad_lines = 0

    def read_new(self) -> Iterator[tuple[int, dict]]:
        try:
            size = self.path.stat().st_size
        except OSError:
            return
        if size < self.offset:  # truncated / rotated: start over
            self.offset = 0
        if size == self.offset:
            return
        with open(self.path, "rb") as fh:
            fh.seek(self.offset)
            data = fh.read(size - self.offset)
        pos = self.offset
        for line in data.splitlines(keepends=True):
            if not line.endswith(b"\n"):
                break  # torn tail; re-read next poll
            pos += len(line)
            text = line.strip()
            if not text:
                continue
            try:
                rec = json.loads(text)
            except (json.JSONDecodeError, UnicodeDecodeError):
                self.bad_lines += 1
                continue
            if isinstance(rec, dict):
                yield pos, rec
        self.offset = pos

    def pump(self, reducer) -> int:
        n = 0
        for off, rec in self.read_new():
            if self.breakers_only:
                n += reducer.feed_breaker(rec, source=self.name, offset=off)
                continue
            if reducer.feed(rec, source=self.name, offset=off, island=self.island, node=self.node,
                            island_fixed=self.island_fixed):
                n += 1
        return n


def load_all(reducer, paths: list[str]) -> list[TapeSource]:
    sources = [make_source(p) for p in discover(paths)]
    for s in sources:
        s.pump(reducer)
    for rec in operator_stops_near(paths):
        reducer.feed_operator_stop(rec)
    return sources


def run_sources(run: str) -> list[str]:
    """Default tape locations for a run registered under ``~/.yeto/runs``."""
    from .. import runs as runs_mod

    try:
        from ..launcher import RL_SYNCER_EVENT_TAPE
    except Exception:  # pragma: no cover - launcher needs sky; keep the known default
        RL_SYNCER_EVENT_TAPE = "~/yeto-output/yeto-tape.jsonl"

    d = runs_mod.run_dir(run)
    # the run's own paths are kept even before they exist: serve re-discovers them every
    # poll, so a dashboard started before the run follows them once they appear; the cost
    # guard's and the ``run up`` monitor's tapes feed the Breakers panel only; the host-wide
    # syncer tape too: a syncer that starts after the dashboard (or a tape the launcher copies
    # from the syncer's host later) is followed once it appears, with its shards
    up = Path(os.environ.get("YETO_STATE_DIR") or "~/.yeto/rl-up").expanduser() / run
    return [str(d / "events"), str(d / "fleet.jsonl"), str(d / COST_EVENTS_FILE), str(up / UP_EVENTS_FILE),
            str(Path(os.path.expanduser(RL_SYNCER_EVENT_TAPE)))]


_RUN_FROM_TAPE_DIR = re.compile(r"^yeto-(.+)$")
_RUN_FROM_ISLAND = re.compile(r"^(.+?)-l\d+(?:-[a-z0-9]+)?(?:\.jsonl)?$")


def run_meta_near(path: str | os.PathLike, max_up: int = 4) -> dict | None:
    """``meta.json`` of the ``~/.yeto/runs/<run>/`` directory a tape lives in
    (looked up at most ``max_up`` parents), or None."""
    p = Path(path)
    for d in [p, *list(p.parents)[:max_up]]:
        m = d / "meta.json"
        if m.is_file():
            try:
                data = json.loads(m.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                return None
            if isinstance(data, dict) and isinstance(data.get("name"), str):
                return data
    return None


def infer_run_name(sources: list[str], island_names: list[str] | None = None) -> str | None:
    """Run name when ``--run`` was not given: the run directory's
    meta.json, else a ``yeto-<run>`` Modal tape dir, else an island name /
    learner tape file ``<run>-l<N>[-<cloud>]``. None when nothing matches."""
    for s in sources:
        meta = run_meta_near(s)
        if meta:
            return meta["name"]
    for s in sources:
        for part in Path(s).parts:
            m = _RUN_FROM_TAPE_DIR.match(part)
            if m:
                return m.group(1)
    for name in list(island_names or []) + [Path(s).name for s in sources]:
        m = _RUN_FROM_ISLAND.match(name or "")
        if m and not m.group(1).startswith(("rl-island", "modal-hostmem")):
            return m.group(1)
    return None


_NODE_RE = re.compile(r"^rank(\d+)$")


def node_hint(path: Path) -> str | None:
    """Node rank of a per-node stream (``.../rank<R>/...``)."""
    for part in reversed(path.parts):
        m = _NODE_RE.match(part)
        if m:
            return m.group(1)
    return None


# -- our own stop of a run ------------------------------------------------
# Authority, in order: (1) ``STOP_ISSUED_utc.txt`` lines "<utc> <cause>" written by the
# run's stop scripts right BEFORE they stop the cloud app (the gate,
# early and watchdog stoppers); (2) for runs older than that file, the legacy marker files
# GATE*_STOPPED / EARLY_STOPPED_TIME / WATCHDOG_FIRED, whose content (a utc stamp) or
# mtime is when the stop command RETURNED -- an upper bound on the issue time, so an
# island failure in the few seconds before it is conservatively NOT counted as ours.
STOP_FILE = "STOP_ISSUED_utc.txt"
TEARDOWN_FILE = "teardown.json"
LEGACY_STOP_MARKERS = (("GATE*_STOPPED", "gate"), ("EARLY_STOPPED_TIME", "early"), ("WATCHDOG_FIRED", "watchdog"))


def _utc_stamp(text: str) -> float | None:
    import datetime as _dt

    try:
        return _dt.datetime.strptime(text.strip(), "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=_dt.timezone.utc).timestamp()
    except ValueError:
        return None


def stop_records_in(d: Path) -> list[dict]:
    """``operator_stop`` records for one run directory (see the authority note above)."""
    out: list[dict] = []
    f = d / STOP_FILE
    if f.is_file():
        for line in f.read_text(errors="replace").splitlines():
            parts = line.split()
            t = _utc_stamp(parts[0]) if parts else None
            if t is not None:
                out.append({"event": "operator_stop", "time_unix": t,
                            "cause": parts[1] if len(parts) > 1 else "unknown",
                            "marker": STOP_FILE, "path": str(f)})
    td = d / TEARDOWN_FILE  # the launcher tore the run down (its last act): every island is stopped
    if td.is_file():
        out.append({"event": "operator_stop", "time_unix": td.stat().st_mtime, "cause": "teardown",
                    "marker": TEARDOWN_FILE + " (mtime)", "path": str(td)})
    if any(r["marker"] == STOP_FILE for r in out):
        return out
    for pattern, cause in LEGACY_STOP_MARKERS:
        for p in sorted(d.glob(pattern)):
            if not p.is_file():
                continue
            try:
                t = _utc_stamp(p.read_text(errors="replace").splitlines()[0]) if p.stat().st_size else None
            except (OSError, IndexError):
                t = None
            out.append({"event": "operator_stop", "time_unix": t if t is not None else p.stat().st_mtime,
                        "cause": cause, "marker": p.name + ("" if t is not None else " (mtime)"),
                        "path": str(p)})
    return out


def operator_stops_near(paths: list[str], max_up: int = 5) -> list[dict]:
    """Stop records from the given dirs/files and up to ``max_up`` parents
    (a run's tapes live under ``<s1-runs>/<run>/{runs,tape-direct}/...``)."""
    seen: set[Path] = set()
    out: list[dict] = []
    for raw in paths:
        p = Path(os.path.expanduser(raw)).resolve()
        start = p if p.is_dir() else p.parent
        for d in [start, *list(start.parents)[:max_up]]:
            if d in seen:
                continue
            seen.add(d)
            out += stop_records_in(d)
    return out


_UP_STOPS: dict[str, tuple] = {}


def up_state_stops(path: str | os.PathLike | None) -> list[dict]:
    """The stops a ``run up`` monitor recorded in its state file (``--up-state``): every
    top-level entry with a ``stop_reason`` (a breaker trip that stopped the trainer:
    ``{"at": <utc>, "why": ..., "stop_reason": ...}``). Re-read when the file changes."""
    if not path:
        return []
    p = Path(os.path.expanduser(str(path)))
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return []
    hit = _UP_STOPS.get(str(p))
    if hit and hit[0] == mtime:
        return [dict(x) for x in hit[1]]
    try:
        state = json.loads(p.read_text())
    except (OSError, ValueError):
        return []
    out = []
    for key, row in (state.items() if isinstance(state, dict) else ()):
        if isinstance(row, dict) and row.get("stop_reason"):
            at = row.get("at")
            out.append({"cause": str(row["stop_reason"]), "time_unix": _utc_stamp(at) if isinstance(at, str) else None,
                        "marker": f"{p.name} {key}", "path": str(p), "why": row.get("why"), "kind": "up"})
    _UP_STOPS[str(p)] = (mtime, out)
    return [dict(x) for x in out]
