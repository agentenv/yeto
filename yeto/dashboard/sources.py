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


def discover(paths: list[str], max_depth: int = 2) -> list[Path]:
    """Files given directly plus ``*.jsonl`` up to ``max_depth`` below each dir."""
    out: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        p = Path(os.path.expanduser(raw))
        if p.is_file():
            cands = [p]
        elif p.is_dir():
            cands = []
            for depth in range(max_depth + 1):
                cands += sorted(p.glob("/".join(["*"] * depth + ["*.jsonl"])))
        else:
            cands = []
        for c in cands:
            r = c.resolve()
            if r not in seen and r.is_file():
                seen.add(r)
                out.append(c)
    return out


class TapeSource:
    def __init__(self, path: str | os.PathLike, island: str | None = None):
        self.path = Path(path)
        self.name = str(self.path)
        self.island = island if island is not None else (
            island_hint(self.path) if "journal" in self.path.name
            else host_island_hint(self.path) if "hostmem" in self.path.name else None)
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
            if reducer.feed(rec, source=self.name, offset=off, island=self.island):
                n += 1
        return n


def load_all(reducer, paths: list[str]) -> list[TapeSource]:
    sources = [TapeSource(p) for p in discover(paths)]
    for s in sources:
        s.pump(reducer)
    return sources


def run_sources(run: str) -> list[str]:
    """Default tape locations for a run registered under ``~/.yeto/runs``."""
    from .. import runs as runs_mod

    try:
        from ..launcher import RL_SYNCER_EVENT_TAPE
    except Exception:  # pragma: no cover - launcher needs sky; keep the known default
        RL_SYNCER_EVENT_TAPE = "~/yeto-output/yeto-tape.jsonl"

    d = runs_mod.run_dir(run)
    out = [str(d / "events"), str(d / "fleet.jsonl")]
    syncer = Path(os.path.expanduser(RL_SYNCER_EVENT_TAPE))
    if syncer.exists():
        out.append(str(syncer))
    return [p for p in out if Path(p).exists()]
