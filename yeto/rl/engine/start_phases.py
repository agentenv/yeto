"""Trainer start phases (rl-trainer-cold-start-cost L0, design D4).

A trainer start passes seven fixed phases, in this order:

``submitted``          the launcher built the island task (launcher clock)
``container_ready``    the island setup script began (container clock)
``setup_done``         the island setup script ended
``ray_ready``          ``ray start --head`` returned on node 0
``actor_import_done``  every trainer actor process finished its class import
``actor_init_done``    the trainer actors finished ``init`` (model + optimizer)
``first_step_done``    the first optimizer step returned

Each phase is one ``trainer_start_phase`` event; after ``first_step_done``
one ``trainer_start_summary`` event carries ``total_s`` =
``first_step_done - submitted`` (``T_s`` of the scheduler).

Times are Unix wall-clock seconds: the phases come from three processes on
two machines (launcher, setup shell, island), so a monotonic clock cannot
order them.  Clock skew between the launcher and the container goes into
``submitted -> container_ready`` only.

The first three container phases are written by the shell (:func:`shell_mark`)
into :data:`MARK_DIR`; the island replays them (:meth:`StartPhases.replay`).
Stdlib only: the launcher, the island and the head all import this module.
"""

from __future__ import annotations

import json
import os
import shlex
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

START_PHASES = (
    "submitted",
    "container_ready",
    "setup_done",
    "ray_ready",
    "actor_import_done",
    "actor_init_done",
    "first_step_done",
)
START_PHASE_EVENT = "trainer_start_phase"
START_SUMMARY_EVENT = "trainer_start_summary"
# Launcher -> island: the Unix time the island task was built.
SUBMITTED_ENV = "YETO_START_SUBMITTED_UNIX"
# Setup/run shell -> island: one file per phase, content = Unix time.
MARK_DIR = "$HOME/.yeto-start-phases"
SHELL_PHASES = ("container_ready", "setup_done", "ray_ready")
# Setup shell -> island: events the setup shell could not write to the tape
# (no tape exists before the island starts), one JSON object per line.
SETUP_EVENTS_FILE = "setup-events.jsonl"
# Run manifest key (L0.2): {learner_id: summary}.
MANIFEST_KEY = "trainer_start"

Emit = Callable[..., None]


def shell_mark(phase: str) -> str:
    """One shell line that records ``phase`` now; it never fails the script."""

    if phase not in SHELL_PHASES:
        raise ValueError(f"{phase!r} is not a shell-recorded start phase: {SHELL_PHASES}")
    return (
        f'{{ mkdir -p "{MARK_DIR}" && date +%s.%N > "{MARK_DIR}/{phase}"; }} '
        ">/dev/null 2>&1 || true"
    )


def shell_event(event: str, fields: Mapping[str, Any] | None = None,
                int_fields: Mapping[str, str] | None = None) -> str:
    """Shell lines that append one setup event and print it to stderr.

    ``fields`` are fixed values (JSON-encoded now); ``int_fields`` map a field
    to a shell expression that expands to an integer at run time (``"$rc"``).
    The lines never fail the script.
    """

    static = json.dumps({"event": event, **(fields or {})}, sort_keys=True)
    body, args = static[:-1].replace("%", "%%"), ""
    for name, expr in (int_fields or {}).items():
        body += f", {json.dumps(name)}: %d"
        args += f' "{expr}"'
    line = shlex.quote(body + "}\\n")
    return (
        f'{{ mkdir -p "{MARK_DIR}" && printf {line}{args} >> "{MARK_DIR}/{SETUP_EVENTS_FILE}"; }} '
        ">/dev/null 2>&1 || true\n"
        f"printf {shlex.quote('[yeto-setup] event ' + body + '}' + chr(92) + 'n')}{args} >&2 || true"
    )


def shell_reset() -> str:
    """Clear marks of an earlier start in the same home (relaunch on one machine)."""

    return f'rm -rf "{MARK_DIR}" >/dev/null 2>&1 || true'


class StartPhases:
    """Records the phases in order and emits the events.

    ``emit(event, **fields)`` writes one event.  A phase out of order, or a
    phase given twice, is an error: the order is the contract the scheduler
    reads.  A phase that could not be measured is skipped with
    :meth:`missing`; the summary names it.
    """

    def __init__(self, emit: Emit, *, clock: Callable[[], float] = time.time) -> None:
        self._emit = emit
        self._clock = clock
        self.times: dict[str, float] = {}
        self.missing_phases: list[str] = []
        self.summary: dict[str, Any] | None = None
        self._setup_events_seen = 0

    def _next_index(self) -> int:
        done = [*self.times, *self.missing_phases]
        return max((START_PHASES.index(p) for p in done), default=-1) + 1

    def _check_next(self, phase: str) -> None:
        if phase not in START_PHASES:
            raise ValueError(f"unknown start phase {phase!r}; known: {START_PHASES}")
        if phase in self.times or phase in self.missing_phases:
            raise ValueError(f"start phase {phase!r} recorded twice")
        if START_PHASES.index(phase) < self._next_index():
            raise ValueError(f"start phase {phase!r} out of order after {list(self.times)}")

    def mark(self, phase: str, at: float | None = None) -> None:
        self._check_next(phase)
        t = self._clock() if at is None else float(at)
        submitted = self.times.get("submitted")
        self.times[phase] = t
        self._emit(
            START_PHASE_EVENT,
            phase=phase,
            index=START_PHASES.index(phase),
            t_unix=t,
            since_submitted_s=None if submitted is None else round(t - submitted, 3),
        )
        if phase == START_PHASES[-1]:
            self._summarize()

    def missing(self, phase: str, reason: str) -> None:
        self._check_next(phase)
        self.missing_phases.append(phase)
        self._emit(START_PHASE_EVENT, phase=phase, index=START_PHASES.index(phase),
                   t_unix=None, missing=reason)

    def mark_once(self, phase: str) -> None:
        """``mark`` unless already recorded (first train step of a long loop)."""

        if phase not in self.times and phase not in self.missing_phases:
            self.mark(phase)

    def replay(self, environ: Mapping[str, str] | None = None,
               mark_dir: str | os.PathLike[str] | None = None) -> None:
        """Record ``submitted`` and the shell phases from env and mark files."""

        environ = os.environ if environ is None else environ
        directory = Path(os.path.expandvars(MARK_DIR) if mark_dir is None else mark_dir)
        submitted = environ.get(SUBMITTED_ENV)
        if submitted:
            self.mark("submitted", float(submitted))
        else:
            self.missing("submitted", f"{SUBMITTED_ENV} not set")
        for phase in SHELL_PHASES:
            try:
                self.mark(phase, float((directory / phase).read_text().strip()))
            except (OSError, ValueError) as error:
                self.missing(phase, f"no mark file: {error}")
        self.forward_setup_events(directory)

    def forward_setup_events(self, mark_dir: str | os.PathLike[str] | None = None) -> int:
        """Emit setup-shell events not yet forwarded; returns how many were new.

        Called at replay and again later: a background step of the setup (the
        model prefetch) can fail after the island has started.
        """

        directory = Path(os.path.expandvars(MARK_DIR) if mark_dir is None else mark_dir)
        records = read_tape_records([directory / SETUP_EVENTS_FILE])
        new = records[self._setup_events_seen:]
        self._setup_events_seen = len(records)
        for record in new:
            fields = {k: v for k, v in record.items() if k != "event"}
            self._emit(str(record.get("event", "setup_event")), source="setup", **fields)
        return len(new)

    def _summarize(self) -> None:
        first, last = START_PHASES[0], START_PHASES[-1]
        total = (round(self.times[last] - self.times[first], 3)
                 if first in self.times and last in self.times else None)
        self.summary = {
            "phases": {p: self.times.get(p) for p in START_PHASES},
            "missing": list(self.missing_phases),
            "total_s": total,
        }
        self._emit(START_SUMMARY_EVENT, **self.summary)


def summaries_from_events(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """``{learner_id: summary}`` from tape records (last summary per island wins)."""

    out: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("event") != START_SUMMARY_EVENT:
            continue
        learner = str(record.get("island_id", record.get("learner_id", "?")))
        out[learner] = {k: record.get(k) for k in ("phases", "missing", "total_s")}
    return out


def read_tape_records(paths: Iterable[os.PathLike[str] | str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        try:
            lines = Path(path).read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def write_manifest_summary(manifest_path: os.PathLike[str] | str,
                           summaries: Mapping[str, Mapping[str, Any]]) -> bool:
    """Merge ``summaries`` into the run manifest under :data:`MANIFEST_KEY`."""

    if not summaries:
        return False
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    manifest.setdefault(MANIFEST_KEY, {}).update({k: dict(v) for k, v in summaries.items()})
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return True


def t_s_from_manifest(manifest: Mapping[str, Any]) -> float | None:
    """``T_s`` of a finished run: the slowest island start (``None`` if unknown)."""

    totals = [s.get("total_s") for s in (manifest.get(MANIFEST_KEY) or {}).values()]
    totals = [float(t) for t in totals if t is not None]
    return max(totals) if totals else None
