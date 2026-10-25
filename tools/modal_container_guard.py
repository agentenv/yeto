#!/usr/bin/env python3
"""Container guard and cost guard of one Modal run (reader of ``modal container list``).

The S27 run scripts (s1-runs/s27-*/run*.sh) each carried a copy of a shell loop
that stopped the app as soon as more distinct containers than ``NODES`` had been
seen ("an island container was rescheduled"). On s27-ch-d0a (2026-10-25
00:56:15Z) two new containers of the island function appeared while the two
original containers still trained (step 66 at 00:56:11Z); the loop stopped a
healthy run. This module replaces that loop. Standard library only, so a run
script can start it with ``/usr/bin/python3 -I`` from its code snapshot.

Rules (one poll every ``--interval-s``):

* Which containers are the island's own (the original containers) and which are extra.
  With ``--progress-log`` the islands name their containers: the first line
  ``[modal-island N] rank R container <id>`` of each (island, rank) slot gives the
  container of that slot; at most ``--nodes`` slots count. The islands are all named
  when ``--nodes`` slots are named.

  * A listed container that no island named, while the islands are not all named, is
    pending. It is no extra container and no rule below counts it. Modal builds a new
    image inside the app and lists the build container there (s27-ex-1: 12 min of
    image build, two build containers); the old rule took the first ``--nodes``
    containers for the islands' and stopped the app when the real ones came. A
    pending container is billed like any container. It is recorded
    (``unnamed_seen``; ``unnamed_claimed`` when an island names it later;
    ``unnamed_gone`` when it leaves unnamed, and its log is saved). One that stays
    pending for more than ``--unnamed-warn-s`` gets one WARNING in ``poll/guard.log``.
  * When the islands are all named, every other container is an extra container,
    pending ones too (an ``extra_seen`` event with ``unnamed_s``; the time limits
    below count from that poll).
  * A container that prints the line of a slot that already has a container is an extra
    container at once: the island script ran again in a new container.
  * An original container that no container list showed yet is not "gone".
  * Without ``--progress-log``, or with ``--positional-originals``, no names are known:
    the first ``--nodes`` started containers of the app (by ``start_time``, then id) are
    the original containers (a queued task without a start time is left out, as in the
    launcher's own poller), and a build container before the islands is taken for one.

  Every extra container is recorded when it appears and when it leaves
  (``poll/extra-containers.jsonl``: id, ``start_time``, poll time, the original
  containers present, the progress line count), and its recent log lines are
  saved (``poll/extra-<id>.log``, ``modal app logs --container`` with the function
  id, over a time window around its start, once when it appears and once when it
  leaves), so the next run has evidence where it came from.
* The app is stopped (kind ``resched``) only when an extra container is present
  and one of these holds:

  1. an original container is gone (a replacement island: mixed state), unless its
     island finished: the progress log has the island's container line
     (``[modal-island N] rank R container <id>``) and its ``rl_learner_finalized``
     event (s27-ek-b: island 1 finished and its container left while an idle extra
     container of island 1's function was listed);
  2. no progress line (``--progress-re`` in ``--progress-log``) for
     ``--extra-stall-s`` seconds, counted from the later of the last progress
     line and the time the present extra containers appeared;
  3. an extra container has been present for more than ``--extra-max-s`` seconds.

  Without ``--progress-log`` rule 2 is off; rules 1 and 3 still apply.
* Expected island relaunches (``--replacement-containers K``, default 0: off). A
  run that kills an island on purpose (fault injection) and has the launcher
  start it again gets new containers once the island's original containers are
  gone. While at most K extra containers were ever seen, rules 1 and 3 do not
  apply; the app is stopped only when more than ``--nodes`` started containers
  are present for more than ``--extra-max-s`` seconds (old and new containers
  billed together) or by rule 2. The (K+1)-th extra container brings back the
  three rules.
* Cost guard (kind ``cost``): the estimate is the sum over polls of (containers
  in the app, extra containers included) x ``--rate`` x seconds since the last
  poll, plus ``--head-rate`` x seconds since the guard started. A failed
  ``container list`` counts the last known number. At ``--cost-alert`` the guard
  writes ``COST_ALERT`` once; at the cap it stops the app. ``COST_CAP_OVERRIDE``
  in the run directory, when present, replaces ``--cost-cap`` (the main agent
  writes it).

Files in ``--run-dir`` (the names the run scripts used): ``containers.log``
(``<utc> n=<N> <ids>``, one line per successful poll), ``poll/seen.txt``
(distinct ids), ``poll/current.txt`` (ids of the last poll, for per-container
samplers), ``first_container_utc.txt``, ``cost.now``, ``COST_ALERT``,
``stopped.log``, ``RESCHED_FIRED`` / ``COST_FIRED`` (the reason), ``poll/guard.log``
and ``poll/guard-state.json``.

Stopping runs ``--stop-cmd`` through ``bash -c`` once, with ``{kind}`` replaced by
the stop kind and the environment variables ``GUARD_KIND`` and ``GUARD_REASON``;
its output goes to ``poll/guard-stop.out``. The guard then exits with code 3.
It exits with code 0 when ``--while-pid`` ends or after ``--max-polls`` polls.

Use in a run script (replaces the container poller's set / cost part)::

    ( cd /tmp; exec /usr/bin/python3 -I $R/yeto/tools/modal_container_guard.py \\
        --run-dir $R --app $APP --nodes $NODES --rate $RATE --head-rate ${HEAD_RATE:-0} \\
        --cost-cap $COST_CAP --cost-alert $COST_ALERT --modal $MODAL --while-pid $SUB \\
        --progress-log $R/launch.stream.log --progress-re "$PROGRESS_RE" \\
        --stop-cmd "bash $R/teardown.sh {kind}" ) > $R/poll/guard.out 2>&1 9>&- &
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

RESCHED, COST = "resched", "cost"
MARKERS = {RESCHED: "RESCHED_FIRED", COST: "COST_FIRED"}
EXIT_STOPPED = 3
DEFAULT_EXTRA_MAX_S = 600.0
DEFAULT_EXTRA_STALL_S = 300.0


def utc(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def started(row: dict[str, Any]) -> bool:
    """A listed task that has started (a queued one shows no start time, or "Pending").
    The container rules use started tasks only, as the launcher's own poller does
    (``ModalOps.list_container_ids``); the cost guard counts every listed task."""
    value = row.get("start_time")
    return value is not None and str(value).strip().lower() not in ("", "pending")


def normalize(row: dict[str, Any]) -> dict[str, Any]:
    """``{"Container ID": ...}`` (older CLI output) -> ``{"container_id": ...}``."""
    return {str(k).strip().lower().replace(" ", "_"): v for k, v in row.items()}


@dataclass
class GuardConfig:
    nodes: int
    rate: float  # USD per container-hour
    head_rate: float = 0.0  # USD per hour, from the guard start
    cost_cap: float | None = None
    cost_alert: float | None = None
    extra_max_s: float = DEFAULT_EXTRA_MAX_S
    extra_stall_s: float = DEFAULT_EXTRA_STALL_S
    replacements: int = 0  # extra containers of expected island relaunches (0: none expected)


@dataclass(frozen=True)
class IslandNames:
    """The containers that the progress log names (``[modal-island N] rank R container <id>``):
    ``owners``, the first container of each (island, rank) slot in log order, and ``others``,
    named containers that are not the first of their slot (the island script ran again)."""

    owners: tuple[str, ...] = ()
    others: frozenset[str] = frozenset()


@dataclass
class Decision:
    """What one poll found: events to record, the cost estimate, a stop."""

    events: list[dict[str, Any]] = field(default_factory=list)
    cost: float = 0.0
    alert: bool = False  # the estimate reached the alert line at this poll (first time)
    stop: tuple[str, str] | None = None  # (kind, reason)


class ContainerGuard:
    """The rules of the module doc on plain data (no I/O, the caller passes the time).

    ``observe(now, containers, progress, finished=..., names=...)``: ``containers`` is the
    app's rows of ``modal container list --json`` (dicts with ``container_id`` and
    ``start_time``), or None when the list failed; ``progress`` is the number of
    progress lines so far, or None without a progress source; ``finished`` holds the
    container ids of islands that finished and ``names`` the containers the islands
    named (:class:`IslandEnds`). ``names=None``: no names are known, the first
    ``nodes`` started containers are the originals."""

    def __init__(self, config: GuardConfig, *, started: float) -> None:
        if config.nodes < 1:
            raise ValueError("nodes must be >= 1")
        if config.replacements < 0:
            raise ValueError("replacements must be >= 0")
        self.config = config
        self.started = float(started)
        self.originals: list[str] = []
        self.listed: set[str] = set()  # originals that a container list showed
        self.extras: dict[str, dict[str, Any]] = {}  # present extra id -> first_seen, start_time
        self.extras_gone: list[dict[str, Any]] = []
        self.unnamed: dict[str, dict[str, Any]] = {}  # pending id -> first_seen, start_time
        self.unnamed_gone: list[dict[str, Any]] = []
        self.by_name = False  # the last observe had names
        self.originals_gone: set[str] = set()
        self.seen: list[str] = []
        self.present: list[str] = []
        self.last_count = 0
        self.last_poll = float(started)
        self.containers_usd = 0.0
        self.progress = 0
        self.last_progress = float(started)
        self.alerted = False
        self.stopped: tuple[str, str] | None = None
        self.over_since: float | None = None  # first poll with more than nodes started containers
        self.finished: set[str] = set()  # container ids of finished islands (last observe)

    def cap(self, override: float | None = None) -> float | None:
        return self.config.cost_cap if override is None else override

    def observe(self, now: float, containers: list[dict[str, Any]] | None,
                progress: int | None = None, *, cap_override: float | None = None,
                finished: set[str] | frozenset[str] = frozenset(),
                names: IslandNames | None = None) -> Decision:
        decision = Decision()
        self.finished = set(finished)
        self.by_name = names is not None
        if containers is not None:
            self._update_set(now, [c for c in containers if started(c)], decision, names)
        count = len(containers) if containers is not None else self.last_count  # queued ones too
        self.containers_usd += max(0.0, now - self.last_poll) / 3600.0 * count * self.config.rate
        self.last_poll, self.last_count = now, count
        decision.cost = round(self.containers_usd + max(0.0, now - self.started) / 3600.0
                              * self.config.head_rate, 2)
        if progress is not None and progress > self.progress:
            self.progress, self.last_progress = progress, now
        if self.stopped is not None:
            return decision
        alert = self.config.cost_alert
        if alert is not None and decision.cost >= alert and not self.alerted:
            self.alerted = decision.alert = True
        decision.stop = self._container_stop(now, progress) or self._cost_stop(decision.cost, cap_override)
        self.stopped = decision.stop
        return decision

    def _update_set(self, now: float, containers: list[dict[str, Any]], decision: Decision,
                    names: IslandNames | None) -> None:
        rows = {str(c["container_id"]): c for c in containers if c.get("container_id")}
        by_name = names is not None
        if names is not None:  # the islands' own containers are the ones the log names
            for cid in names.owners[:self.config.nodes]:
                if cid not in self.originals:
                    self.originals.append(cid)
        named = set(names.owners) | set(names.others) if names is not None else set()
        known = set(self.seen)
        new = sorted((cid for cid in rows if cid not in known),
                     key=lambda cid: (str(rows[cid].get("start_time") or "~"), cid))
        for cid in new:
            self.seen.append(cid)
            if by_name and cid in self.originals:  # named before the list showed it
                self.listed.add(cid)
            elif not by_name and len(self.originals) < self.config.nodes:
                self.originals.append(cid)
                self.listed.add(cid)
            elif by_name and cid not in named and len(self.originals) < self.config.nodes:
                # not named, islands not all named: maybe a build container, maybe an island
                # container whose line is not in the log yet. Not an extra container.
                self.unnamed[cid] = {"first_seen": now, "start_time": rows[cid].get("start_time")}
                decision.events.append({"event": "unnamed_seen", "utc": utc(now), "container_id": cid,
                                        "start_time": rows[cid].get("start_time"),
                                        "named": list(self.originals), "progress_lines": self.progress})
            else:
                self._add_extra(now, cid, rows, decision, rows[cid].get("start_time"))
        for cid in list(self.unnamed):
            info = self.unnamed[cid]
            held = round(now - info["first_seen"], 1)
            if cid in self.originals:  # the log names it now: an island container
                del self.unnamed[cid]
                self.listed.add(cid)
                decision.events.append({"event": "unnamed_claimed", "utc": utc(now), "container_id": cid,
                                        "present_s": held})
            elif cid not in rows:  # left without a name: never an island container
                del self.unnamed[cid]
                gone = {"event": "unnamed_gone", "utc": utc(now), "container_id": cid,
                        "start_time": info["start_time"], "present_s": held}
                self.unnamed_gone.append(gone)
                decision.events.append(gone)
            elif cid in named or len(self.originals) >= self.config.nodes:
                # the islands are all named (or it ran island code for a taken slot): an extra container
                del self.unnamed[cid]
                self._add_extra(now, cid, rows, decision, info["start_time"], unnamed_s=held)
        for cid in [cid for cid in self.extras if cid not in rows]:
            info = self.extras.pop(cid)
            gone = {"event": "extra_gone", "utc": utc(now), "container_id": cid,
                    "start_time": info["start_time"], "present_s": round(now - info["first_seen"], 1)}
            self.extras_gone.append(gone)
            decision.events.append(gone)
        for cid in self.originals:
            if cid in self.listed and cid not in rows and cid not in self.originals_gone:
                self.originals_gone.add(cid)
                decision.events.append({"event": "original_gone", "utc": utc(now), "container_id": cid,
                                        "extras_present": sorted(self.extras),
                                        "island_finished": cid in self.finished})
        self.present = list(rows)
        if len(rows) - len(self.unnamed) > self.config.nodes:  # a pending container is not counted
            self.over_since = now if self.over_since is None else self.over_since
        else:
            self.over_since = None

    def _add_extra(self, now: float, cid: str, rows: dict[str, dict[str, Any]], decision: Decision,
                   start_time: Any, unnamed_s: float | None = None) -> None:
        self.extras[cid] = {"first_seen": now, "start_time": start_time}
        event = {"event": "extra_seen", "utc": utc(now), "container_id": cid, "start_time": start_time,
                 "originals_present": [o for o in self.originals if o in rows],
                 "progress_lines": self.progress}
        if unnamed_s is not None:
            event["unnamed_s"] = unnamed_s
        decision.events.append(event)

    def _container_stop(self, now: float, progress: int | None) -> tuple[str, str] | None:
        if not self.extras:
            return None
        extras = sorted(self.extras)
        expected = len(self.extras) + len(self.extras_gone) <= self.config.replacements
        if expected:
            return self._replacement_stop(now, progress, extras)
        missing = [o for o in self.originals
                   if o in self.listed and o not in self.present and o not in self.finished]
        if missing:
            return RESCHED, (f"original container(s) {missing} gone while extra container(s) {extras} "
                             "are in the app (a replacement island)")
        since = min(info["first_seen"] for info in self.extras.values())
        if progress is not None:
            quiet = now - max(self.last_progress, since)
            if quiet > self.config.extra_stall_s:
                return RESCHED, (f"no progress line for {quiet:.0f} s (limit {self.config.extra_stall_s:.0f} s) "
                                 f"while extra container(s) {extras} are in the app")
        oldest = max(now - info["first_seen"] for info in self.extras.values())
        if oldest > self.config.extra_max_s:
            return RESCHED, (f"extra container(s) {extras} in the app for {oldest:.0f} s "
                             f"(limit {self.config.extra_max_s:.0f} s)")
        return None

    def _replacement_stop(self, now: float, progress: int | None, extras: list[str]) -> tuple[str, str] | None:
        """Every extra container so far is an expected relaunch (``replacements``)."""
        if progress is not None:
            since = min(info["first_seen"] for info in self.extras.values())
            quiet = now - max(self.last_progress, since)
            if quiet > self.config.extra_stall_s:
                return RESCHED, (f"no progress line for {quiet:.0f} s (limit {self.config.extra_stall_s:.0f} s) "
                                 f"while relaunched container(s) {extras} are in the app")
        if self.over_since is not None and now - self.over_since > self.config.extra_max_s:
            return RESCHED, (f"{len(self.present)} started containers (more than {self.config.nodes}) for "
                             f"{now - self.over_since:.0f} s (limit {self.config.extra_max_s:.0f} s): "
                             f"relaunched container(s) {extras} run beside old ones")
        return None

    def _cost_stop(self, cost: float, cap_override: float | None) -> tuple[str, str] | None:
        cap = self.cap(cap_override)
        if cap is not None and cost >= cap:
            return COST, f"estimated cost {cost} >= {cap}"
        return None

    def state(self) -> dict[str, Any]:
        return {"originals": self.originals, "extras_present": self.extras, "extras_gone": self.extras_gone,
                "unnamed_present": self.unnamed, "unnamed_gone": self.unnamed_gone, "by_name": self.by_name,
                "originals_gone": sorted(self.originals_gone), "present": self.present,
                "originals_present": [o for o in self.originals if o in self.present],
                "seen": self.seen, "containers_usd": round(self.containers_usd, 2),
                "progress_lines": self.progress, "last_progress_utc": utc(self.last_progress),
                "stopped": self.stopped}


class ProgressCounter:
    """Counts the lines of ``path`` that match ``pattern`` (complete lines only, read incrementally)."""

    def __init__(self, path: str | os.PathLike[str], pattern: str) -> None:
        self.path = Path(path)
        self.pattern = re.compile(pattern.encode())
        self.offset = 0
        self.count = 0

    def read(self) -> int:
        try:
            with open(self.path, "rb") as fh:
                fh.seek(self.offset)
                data = fh.read()
        except FileNotFoundError:
            return self.count
        end = data.rfind(b"\n")
        if end < 0:
            return self.count
        self.offset += end + 1
        self.count += sum(1 for line in data[:end].split(b"\n") if self.pattern.search(line))
        return self.count


class IslandEnds:
    """Reads ``path`` incrementally (complete lines only) for the islands' container lines
    and ``rl_learner_finalized`` events. ``read()`` returns the container ids of the
    islands that finished; ``names()`` returns the containers the islands named."""

    CONTAINER = re.compile(rb"\[modal-island (\d+)\] rank (\d+) container (ta-[A-Za-z0-9]+)")
    FINALIZED = re.compile(rb'"event"\s*:\s*"rl_learner_finalized"')
    ISLAND_ID = re.compile(rb'"island_id"\s*:\s*(\d+)')

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.offset = 0
        self.island_of: dict[str, str] = {}  # container id -> island
        self.slot_of: dict[str, tuple[str, str]] = {}  # container id -> (island, rank)
        self.owner_of: dict[tuple[str, str], str] = {}  # (island, rank) -> its first container, in log order
        self.finalized: set[str] = set()  # islands

    def read(self) -> set[str]:
        try:
            with open(self.path, "rb") as fh:
                fh.seek(self.offset)
                data = fh.read()
        except FileNotFoundError:
            data = b""
        end = data.rfind(b"\n")
        if end >= 0:
            self.offset += end + 1
            for line in data[:end].split(b"\n"):
                found = self.CONTAINER.search(line)
                if found:
                    island, rank, cid = (g.decode() for g in found.groups())
                    self.island_of.setdefault(cid, island)
                    self.slot_of.setdefault(cid, (island, rank))
                    self.owner_of.setdefault((island, rank), cid)
                if self.FINALIZED.search(line):
                    island = self.ISLAND_ID.search(line)
                    if island:
                        self.finalized.add(island.group(1).decode())
        return {cid for cid, island in self.island_of.items() if island in self.finalized}

    def names(self) -> IslandNames:
        """The containers named so far (``read()`` first)."""
        owners = tuple(self.owner_of.values())
        return IslandNames(owners=owners, others=frozenset(self.slot_of) - frozenset(owners))


def list_containers(command: list[str], app: str, *, timeout_s: float = 60.0,
                    cwd: str | None = None) -> tuple[list[dict[str, Any]] | None, str]:
    """The app's rows of ``modal container list --json`` (None and a reason on a failure)."""
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s, cwd=cwd)
    except (OSError, subprocess.TimeoutExpired) as error:
        return None, f"{type(error).__name__}: {error}"[:300]
    if done.returncode != 0:
        return None, f"rc={done.returncode}: {done.stderr.strip()[-300:]}"
    try:
        rows = json.loads(done.stdout)
    except ValueError as error:
        return None, f"bad json: {error}"
    if not isinstance(rows, list):
        return None, "bad json: not a list"
    rows = [normalize(r) for r in rows if isinstance(r, dict)]
    return [r for r in rows if r.get("app_name") == app], ""


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class RunFiles:
    """The guard's files in the run directory (see the module doc)."""

    def __init__(self, run_dir: str | os.PathLike[str]) -> None:
        self.run = Path(run_dir)
        self.poll = self.run / "poll"
        self.poll.mkdir(parents=True, exist_ok=True)

    def append(self, name: str, line: str) -> None:
        with open(self.run / name, "a", encoding="utf-8") as fh:
            fh.write(line.rstrip("\n") + "\n")

    def write(self, name: str, text: str) -> None:
        target = self.run / name
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, target)

    def log(self, now: float, text: str) -> None:
        self.append("poll/guard.log", f"{utc(now)} {text}")

    def cap_override(self) -> float | None:
        try:
            return float((self.run / "COST_CAP_OVERRIDE").read_text().strip())
        except (OSError, ValueError):
            return None


def _log_window(start_time: Any, now: float) -> list[str]:
    """``--since/--until`` around a container's start, in UTC with the offset written out
    (the CLI reads a time without offset as local time). s27-ek-b: without a window the
    query over the whole app failed with "Logs query hit resource limit"."""
    begin = now - 600.0
    if start_time:
        try:
            begin = datetime.fromisoformat(str(start_time).replace("Z", "+00:00")).timestamp() - 60.0
        except ValueError:
            pass
    fmt = "%Y-%m-%dT%H:%M:%S+00:00"
    return ["--since", time.strftime(fmt, time.gmtime(begin)), "--until", time.strftime(fmt, time.gmtime(now + 60.0))]


def _diagnose_extra(args: argparse.Namespace, files: RunFiles, row_app_id: str | None, cid: str,
                    start_time: Any = None, now: float | None = None) -> None:
    """Save the extra container's log lines with its function id (background, bounded);
    best effort."""
    if args.no_diagnose or not row_app_id:
        return
    command = ["timeout", str(int(args.diagnose_timeout_s)), *shlex.split(args.modal), "app", "logs", row_app_id,
               "--container", cid, "--timestamps", "--show-function-id", "--show-function-call-id",
               "--show-container-id", *_log_window(start_time, time.time() if now is None else now),
               "--tail", "200"]
    try:
        with open(files.poll / f"extra-{cid}.log", "ab") as out:
            subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             cwd=args.list_cwd, start_new_session=True)
    except OSError as error:
        files.log(time.time(), f"extra {cid}: log capture failed: {error}")


def unnamed_to_warn(guard: ContainerGuard, now: float, warned: set[str], after_s: float) -> list[str]:
    """Pending containers (no island named them) listed for more than ``after_s`` seconds, once each.
    An island container is pending for a few seconds only, until its line reaches the log."""
    due = [cid for cid, info in guard.unnamed.items() if cid not in warned and now - info["first_seen"] > after_s]
    warned.update(due)
    return due


def _stop(args: argparse.Namespace, files: RunFiles, now: float, kind: str, reason: str) -> None:
    files.append("stopped.log", f"{utc(now)} {reason}")
    files.write(MARKERS[kind], f"{utc(now)} {reason}\n")
    files.log(now, f"stop {kind}: {reason}")
    if not args.stop_cmd:
        return
    command = args.stop_cmd.replace("{kind}", kind)
    env = {**os.environ, "GUARD_KIND": kind, "GUARD_REASON": reason}
    with open(files.poll / "guard-stop.out", "ab") as out:
        try:
            done = subprocess.run(["bash", "-c", command], stdout=out, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL, env=env, timeout=args.stop_timeout_s)
            files.log(time.time(), f"stop command rc={done.returncode}")
        except subprocess.TimeoutExpired:
            files.log(time.time(), f"stop command still running after {args.stop_timeout_s:.0f} s")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--app", required=True, help="the app_name of the run's containers")
    ap.add_argument("--nodes", type=int, required=True, help="containers of the run (the original ones)")
    ap.add_argument("--rate", type=float, required=True, help="USD per container-hour")
    ap.add_argument("--head-rate", type=float, default=0.0, help="USD per hour of the head (default 0)")
    ap.add_argument("--cost-cap", type=float, default=None)
    ap.add_argument("--cost-alert", type=float, default=None)
    ap.add_argument("--extra-max-s", type=float, default=DEFAULT_EXTRA_MAX_S)
    ap.add_argument("--extra-stall-s", type=float, default=DEFAULT_EXTRA_STALL_S)
    ap.add_argument("--replacement-containers", type=int, default=0,
                    help="extra containers of expected island relaunches (fault injection runs; default 0)")
    ap.add_argument("--progress-log", default=None,
                    help="log of the run; its '[modal-island N] rank R container <id>' lines name the "
                         "island containers, its --progress-re lines count as progress")
    ap.add_argument("--progress-re", default=None)
    ap.add_argument("--modal", default="modal", help="the modal CLI (default: modal on PATH)")
    ap.add_argument("--list-cmd", default=None,
                    help="command printing the container list JSON (default: <modal> container list --json)")
    ap.add_argument("--list-cwd", default="/tmp")
    ap.add_argument("--list-timeout-s", type=float, default=60.0)
    ap.add_argument("--interval-s", type=float, default=30.0)
    ap.add_argument("--while-pid", type=int, default=None, help="poll while this process lives")
    ap.add_argument("--max-polls", type=int, default=None)
    ap.add_argument("--stop-cmd", default=None, help="bash command run once to stop the app ({kind} replaced)")
    ap.add_argument("--stop-timeout-s", type=float, default=1800.0)
    ap.add_argument("--positional-originals", action="store_true",
                    help="with --progress-log: still take the first --nodes started containers as the "
                         "originals (for a log without island container lines)")
    ap.add_argument("--unnamed-warn-s", type=float, default=120.0,
                    help="log one WARNING for a container that no island has named after this many seconds")
    ap.add_argument("--no-diagnose", action="store_true", help="do not save the log lines of extra containers")
    ap.add_argument("--diagnose-timeout-s", type=float, default=120.0)
    args = ap.parse_args(argv)
    if (args.progress_log is None) != (args.progress_re is None):
        ap.error("--progress-log and --progress-re go together")
    if args.while_pid is None and args.max_polls is None:
        ap.error("give --while-pid or --max-polls")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    files = RunFiles(args.run_dir)
    started = time.time()
    guard = ContainerGuard(GuardConfig(nodes=args.nodes, rate=args.rate, head_rate=args.head_rate,
                                       cost_cap=args.cost_cap, cost_alert=args.cost_alert,
                                       extra_max_s=args.extra_max_s, extra_stall_s=args.extra_stall_s,
                                       replacements=args.replacement_containers),
                           started=started)
    progress = None if args.progress_log is None else ProgressCounter(args.progress_log, args.progress_re)
    ends = None if args.progress_log is None else IslandEnds(args.progress_log)
    app_ids: dict[str, str] = {}  # extra container id -> app id (for the second log capture)
    warned: set[str] = set()  # pending containers already named in a WARNING
    command = shlex.split(args.list_cmd) if args.list_cmd else [*shlex.split(args.modal), "container", "list",
                                                                  "--json"]
    by_name = ends is not None and not args.positional_originals
    files.log(started, f"start app={args.app} nodes={args.nodes} rate={args.rate} head_rate={args.head_rate} "
                       f"cap={args.cost_cap} alert={args.cost_alert} extra_max_s={args.extra_max_s} "
                       f"extra_stall_s={args.extra_stall_s if progress else 'off'} "
                       f"originals={'by name' if by_name else 'first started'}")
    polls = 0
    while True:
        if args.while_pid is not None and not pid_alive(args.while_pid):
            files.log(time.time(), f"pid {args.while_pid} ended")
            return 0
        if args.max_polls is not None and polls >= args.max_polls:
            return 0
        rows, why = list_containers(command, args.app, timeout_s=args.list_timeout_s, cwd=args.list_cwd)
        now = time.time()
        polls += 1
        if rows is None:
            files.log(now, f"container list failed ({why}); cost counts the last {guard.last_count} container(s)")
        finished = set() if ends is None else ends.read()  # read() first: names() uses what it read
        decision = guard.observe(now, rows, None if progress is None else progress.read(),
                                 cap_override=files.cap_override(), finished=finished,
                                 names=ends.names() if by_name else None)
        if rows is not None:
            ids = [str(r["container_id"]) for r in rows if r.get("container_id")]
            files.append("containers.log", f"{utc(now)} n={len(ids)} {' '.join(ids)}".rstrip())
            files.write("poll/current.txt", "".join(f"{cid}\n" for cid in ids))
            files.write("poll/seen.txt", "".join(f"{cid}\n" for cid in guard.seen))
            if ids and not (files.run / "first_container_utc.txt").exists():
                files.write("first_container_utc.txt", utc(now) + "\n")
        for event in decision.events:
            files.append("poll/extra-containers.jsonl", json.dumps(event, sort_keys=True))
            files.log(now, json.dumps(event, sort_keys=True))
            if event["event"] in ("extra_seen", "extra_gone", "unnamed_seen", "unnamed_gone"):
                cid = event["container_id"]
                app_id = app_ids.get(cid) or next(
                    (r.get("app_id") for r in rows or () if r.get("container_id") == cid), None)
                if app_id:
                    app_ids[cid] = app_id
                if event["event"] != "unnamed_seen":  # an island container is unnamed for a few seconds
                    _diagnose_extra(args, files, app_id, cid, event.get("start_time"), now)
        for cid in unnamed_to_warn(guard, now, warned, args.unnamed_warn_s):
            files.log(now, f"WARNING: container {cid} is not named by the progress log after "
                           f"{args.unnamed_warn_s:.0f} s; {len(guard.originals)} of {args.nodes} island containers "
                           "are named. It counts as no extra container until all are named (a build container, "
                           "or an island that has not printed its line)")
        files.write("cost.now", f"{utc(now)} est_cost_usd={decision.cost} "
                                f"containers_usd={round(guard.containers_usd, 2)} "
                                f"distinct_containers={len(guard.seen)} extra_present={len(guard.extras)}\n")
        if decision.alert and not (files.run / "COST_ALERT").exists():
            files.write("COST_ALERT", f"{utc(now)} estimated cost {decision.cost} >= alert "
                                      f"{args.cost_alert}: report to the main agent\n")
        files.write("poll/guard-state.json", json.dumps(guard.state(), indent=1, sort_keys=True, default=str))
        if decision.stop is not None:
            _stop(args, files, now, *decision.stop)
            return EXIT_STOPPED
        if args.max_polls is not None and polls >= args.max_polls:
            return 0
        time.sleep(args.interval_s)


if __name__ == "__main__":
    sys.exit(main())
