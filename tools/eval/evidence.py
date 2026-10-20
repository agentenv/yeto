"""Raw run data, read the same way for both backends.

A run directory holds ``cell.json`` (the cell and run id), ``driver.json`` and
``raw/`` (everything pulled). :class:`Evidence` turns the raw data into:

* ``islands``: event lists keyed by component (verl: ``trainer0``, ``sampler-0``,
  ``island-a``...; Miles: the launcher's island labels ``l0``, ``l1``, and
  ``s:l0`` for a separate sampler launch), each sorted by time and de-duplicated
  across sources (launch log echo, per-container tapes, the tape Volume);
* ``index``: the sample buffer index (shared-directory JSONL or object files);
* ``lanes``: per-lane read-position logs of synced trainers;
* return codes, GPU names and card seconds.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .verdict import events, find, find_all, jsonl, objects

TRAINER_EVENTS = ("rl_round_trained", "rl_verl_island_done")
SAMPLER_EVENTS = ("rl_sampler_round", "rl_sampler_start")


def _merge(target: list[dict], rows: Iterable[dict], seen: set) -> None:
    for row in rows:
        key = (row.get("event"), row.get("time_unix"), row.get("seq"), row.get("rollout_id"))
        if row.get("event") and key not in seen:
            seen.add(key)
            target.append(row)


@dataclass
class Evidence:
    run_dir: Path
    backend: str
    islands: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    index: list[dict[str, Any]] = field(default_factory=list)
    lanes: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    shards: dict[str, list[dict[str, Any]]] = field(default_factory=dict)  # sampler shards of the index
    rc: dict[str, Any] = field(default_factory=dict)
    gpu_names: dict[str, int] = field(default_factory=dict)
    store: Path | None = None
    files: dict[str, Path] = field(default_factory=dict)

    # ------------------------------------------------------------- queries
    def of(self, component: str, name: str) -> list[dict[str, Any]]:
        return list(events(self.islands.get(component, []), name))

    def all(self, name: str) -> list[dict[str, Any]]:
        return [e for tape in self.islands.values() for e in events(tape, name)]

    def trainers(self) -> list[str]:
        return sorted(k for k, tape in self.islands.items() if any(e.get("event") in TRAINER_EVENTS for e in tape))

    def samplers(self) -> list[str]:
        return sorted(k for k, tape in self.islands.items() if any(e.get("event") in SAMPLER_EVENTS for e in tape))

    def entries(self) -> list[dict[str, Any]]:
        """Index records that are groups (not end markers)."""
        return [e for e in self.index if not e.get("end") and ("behavior_version" in e or "sample_id" in e)]


# ------------------------------------------------------------------ loaders
def load(run_dir: Path) -> Evidence:
    run_dir = Path(run_dir)
    cell = json.loads((run_dir / "cell.json").read_text())
    ev = Evidence(run_dir=run_dir, backend=cell["backend"])
    raw = run_dir / "raw"
    (_load_verl if ev.backend == "verl" else _load_miles)(ev, raw, cell["run_id"])
    for tape in ev.islands.values():
        tape.sort(key=lambda e: float(e.get("time_unix") or e.get("t") or 0.0))
    if ev.store is not None:
        buffer = ev.store / "sample-buffer"
        if not buffer.is_dir():  # miles_head.sh pulls index.jsonl and lanes/ into the store dir itself
            buffer = ev.store
        ev.index = jsonl(buffer / "index.jsonl") or objects(buffer / "index")
        for path in sorted((buffer / "shards").glob("*.jsonl")) if (buffer / "shards").is_dir() else []:
            ev.shards[path.stem] = jsonl(path)
        lanes = buffer / "lanes"
        for path in sorted(lanes.glob("*-of-*")) if lanes.is_dir() else []:
            m = re.match(r"(\d+)-of-\d+", path.name)
            if m:
                ev.lanes[int(m.group(1))] = jsonl(path) if path.is_file() else objects(path)
    return ev


def _load_verl(ev: Evidence, raw: Path, run_id: str) -> None:
    root = raw / run_id if (raw / run_id).is_dir() else raw
    for comp in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "store") if root.is_dir() else []:
        rows: list[dict] = []
        seen: set = set()
        for name in ("trainer-tape.jsonl", "sampler-tape.jsonl", "syncer-tape.jsonl"):
            _merge(rows, jsonl(comp / name), seen)
        for tape in find_all(comp / "yeto-output", "rl-island-*.jsonl"):
            _merge(rows, jsonl(tape), seen)
        if comp.name == "syncer":
            rows = [dict(r, event=r.get("event") or r.get("kind")) for r in jsonl(comp / "syncer-tape.jsonl")]
        ev.islands[comp.name] = rows
        meta = comp / "meta.json"
        if meta.is_file():
            data = json.loads(meta.read_text())
            # a sampler or a phase returns without "rc": its error (if any) is the outcome
            ev.rc[comp.name] = data["rc"] if "rc" in data else (data["error"] if data.get("error") else 0)
            for name in data.get("gpu") or []:
                ev.gpu_names[name] = ev.gpu_names.get(name, 0) + 1
        ev.files[comp.name] = comp
    ev.store = root / "store"


_EVENT_LINE = r"^(?:\S+ )?\[{prefix}-l(\d+)-modal\] YETO_RL_EVENT (\{{.*\}})\s*$"


def _launch_events(path: Path, prefix: str) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = defaultdict(list)
    if not path.is_file():
        return out
    pattern = re.compile(_EVENT_LINE.format(prefix=re.escape(prefix)))
    for line in path.read_text(errors="replace").splitlines():
        m = pattern.match(line)
        if m:
            try:
                out[int(m.group(1))].append(json.loads(m.group(2)))
            except ValueError:
                pass
    return out


def _load_miles(ev: Evidence, raw: Path, run_id: str) -> None:
    seen: dict[str, set] = defaultdict(set)
    for app, tag, log in ((run_id, "", "launch.ts.log"), (f"{run_id}-s", "s:", "slaunch.ts.log")):
        for island, rows in _launch_events(raw / log, app).items():
            _merge(ev.islands.setdefault(f"{tag}l{island}", []), rows, seen[f"{tag}l{island}"])
        # the tape Volume holds every incarnation's tape (rl-island-0.inc<N>.jsonl after a
        # replacement container); the launch log only echoes the first container
        for tape in find_all(raw / "tape-direct" / f"yeto-{app}", "rl-island-*.jsonl"):
            label = next((part for part in tape.parts if re.fullmatch(r"l\d+", part)), None)
            if label is None:
                continue
            key = f"{tag}{label}"
            _merge(ev.islands.setdefault(key, []), jsonl(tape), seen[key])
    for name, key in (("rc", "launch"), ("src", "sampler_launch")):
        path = raw / name
        if path.is_file():
            ev.rc[key] = int(path.read_text().strip() or -1)
    head_rc = raw / "rc.txt"
    if head_rc.is_file():
        m = re.search(r"rc=(\d+)", head_rc.read_text())
        ev.rc["launch"] = int(m.group(1)) if m else -1
    nvml = raw / "nvml.csv"
    if nvml.is_file():
        for row in csv.reader(nvml.read_text(errors="replace").splitlines()):
            if len(row) > 2:
                name = row[2].strip()
                ev.gpu_names[name] = ev.gpu_names.get(name, 0) + 1
    store = raw / "store"
    # `modal volume get <vol> <P>/sample-buffer <dst>` lands as <dst>/sample-buffer
    ev.store = store
    ev.files["raw"] = raw


def gpu_ok(ev: Evidence, want: str) -> bool:
    return bool(ev.gpu_names) and all(want in name for name in ev.gpu_names)


def finite(values: Iterable[Any]) -> bool:
    import math

    vals = list(values)
    return bool(vals) and all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals)


def numeric_sum(rows: Iterable[Mapping[str, Any]], needle: str) -> float:
    """Sum of every numeric field whose (flattened) name contains ``needle``."""
    total = 0.0

    def walk(value: Any, name: str) -> None:
        nonlocal total
        if isinstance(value, Mapping):
            for k, v in value.items():
                walk(v, f"{name}/{k}")
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and needle in name:
            total += float(value)

    for row in rows:
        walk(row, "")
    return total


__all__ = ["Evidence", "load", "gpu_ok", "finite", "numeric_sum", "find"]
