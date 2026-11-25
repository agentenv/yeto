"""Adopt the running trainer islands of a launcher that died (``yeto.launch.island_lifecycle``).

A launcher restarted on the same run reads each island's record in the run store
(``fleet/trainers/<k>/state.json``) before it orders anything, and asks the island's
source whether the incarnation the record names still runs:

* **sky**: the clusters of the run, found by the run's cluster-name prefix with a
  refreshed status (``sky.status(refresh=...)``), must hold the recorded cluster UP
  and its recorded job alive (``sky.queue``);
* **Modal**: the recorded function call (``modal.FunctionCall.from_id``) must still be
  running; the call id is persisted in the record when the island starts;
* **SSH**: the resident island's journal decides (``ssh_resident.SshIslandOps.start``:
  every node running -> reattach), unchanged; it is reported here as adopted.

An adopted island keeps its machine, its job and its incarnation: nothing is ordered
for it and the trainer is not restarted. An island whose recorded machine is gone (or
whose record is terminal, or that has none) is launched as usual: one order, with the
next incarnation number. A record in ``ordering`` or ``recovering`` is probed too: the
machine it names may have started before the launcher died.

A probe that cannot tell (the cloud listing or the job table does not answer) raises
:class:`AdoptionUnknown`: an order then could start the island twice, so the launcher
orders nothing and says to try again.

Standard library only; the cloud calls are injected (:class:`SkyProbe`, :class:`ModalProbe`).
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from . import island_lifecycle as lc

# records in these states may name a machine that still runs
ADOPTABLE = frozenset({lc.ORDERING, lc.STARTING, lc.RUNNING, lc.RECOVERING, lc.DRAINING})


class AdoptionUnknown(RuntimeError):
    """Whether a recorded island machine still runs is unknown (its source did not answer)."""


@dataclass(frozen=True)
class Adoption:
    """The plan for one launch island: ``adopt`` (keep ``job_id`` on ``cluster``) or not,
    the incarnation it runs as, and why."""

    name: str
    island: int
    adopt: bool
    incarnation: int
    relaunches: int = 0
    job_id: Any = None
    cluster: str | None = None
    why: str = ""


class SkyProbe:
    """Whether a recorded sky cluster and job still run: one refreshed listing of the
    run's clusters (by name prefix), then the job table of the recorded cluster."""

    def __init__(self, prefix: str, *, status: Callable[[], Iterable[Any]] | None = None,
                 job_alive: Callable[[str, Any], bool] | None = None,
                 live_job: Callable[[str], Any] | None = None) -> None:
        self.prefix = str(prefix)
        self._status = status or self._sky_status
        self._job_alive = job_alive or self._sky_job_alive
        self._live_job = live_job or self._sky_live_job
        self._clusters: dict[str, str] | None = None
        self.found: dict[str, int] = {}  # cluster -> the live job found for a record without one

    @staticmethod
    def _sky_status() -> list[Any]:
        import sky

        # FORCE: the cloud's view, not a cached UP of a cluster that is gone
        mode = getattr(sky, "StatusRefreshMode", None)
        refresh = getattr(mode, "FORCE", None) or getattr(mode, "AUTO", True)
        return list(sky.get(sky.status(refresh=refresh)))

    @staticmethod
    def _sky_live_job(cluster: str) -> Any:
        """The id of a job on ``cluster`` that is queued, setting up or running (None: none;
        a job table that does not read raises)."""
        import sky

        live = []
        for record in sky.get(sky.queue(cluster)):
            rid = record.get("job_id") if isinstance(record, dict) else getattr(record, "job_id", None)
            status = record.get("status") if isinstance(record, dict) else getattr(record, "status", None)
            if rid is not None and status is not None and not status.is_terminal():
                live.append(int(rid))
        return max(live) if live else None

    @staticmethod
    def _sky_job_alive(cluster: str, job_id: Any) -> bool:
        """The recorded job is queued, setting up or running (a job table that does not read
        raises: unknown is not "gone")."""
        import sky

        records = sky.get(sky.queue(cluster))
        for record in records:
            rid = record.get("job_id") if isinstance(record, dict) else getattr(record, "job_id", None)
            status = record.get("status") if isinstance(record, dict) else getattr(record, "status", None)
            if rid == int(job_id):
                return status is not None and not status.is_terminal()
        return False

    def clusters(self) -> dict[str, str]:
        """The run's clusters (name -> status), read once."""
        if self._clusters is None:
            out = {}
            for record in self._status():
                name = record.get("name") if isinstance(record, dict) else getattr(record, "name", None)
                status = record.get("status") if isinstance(record, dict) else getattr(record, "status", None)
                if name and str(name).startswith(self.prefix):
                    out[str(name)] = str(getattr(status, "value", status))
            self._clusters = out
        return self._clusters

    def __call__(self, handle: Mapping[str, Any]) -> tuple[bool, str]:
        cluster, job_id = handle.get("cluster"), handle.get("job_id")
        if not cluster:
            return False, "no cluster in the record"
        status = self.clusters().get(str(cluster))
        if status is None:
            return False, f"cluster {cluster} is gone"
        if job_id is None:
            # ordered, its job not recorded yet (the launcher died mid-launch): a cluster that is
            # still being provisioned cannot tell; an UP one tells by its job table
            if "INIT" in status.upper():
                raise AdoptionUnknown(f"cluster {cluster} is {status} and its record names no job yet")
            if "UP" not in status.upper():
                return False, f"cluster {cluster} is {status}"
            live = self._live_job(str(cluster))
            if live is None:
                return False, f"cluster {cluster} UP with no live job"
            self.found[str(cluster)] = live
            return True, f"cluster {cluster} UP, job {live} running (found in its job table)"
        if "UP" not in status.upper():
            return False, f"cluster {cluster} is {status}"
        if not self._job_alive(str(cluster), job_id):
            return False, f"job {job_id} on {cluster} is not running"
        return True, f"cluster {cluster} UP, job {job_id} running"


class ModalProbe:
    """Whether a recorded Modal function call still runs (``status(call_id)`` is
    ``RUNNING``; ``modal_runner.ModalOps.status`` reads ``FunctionCall.from_id``)."""

    def __init__(self, status: Callable[[str], str]) -> None:
        self._status = status

    def __call__(self, handle: Mapping[str, Any]) -> tuple[bool, str]:
        call_id = handle.get("call_id")
        if not call_id:
            return False, "no function call in the record"
        status = self._status(str(call_id))
        if status != "RUNNING":
            return False, f"function call {call_id} is {status}"
        return True, f"function call {call_id} running"


def plan(records: Mapping[int, Mapping[str, Any]], islands: Mapping[str, lc.IslandRef], *,
         probes: Mapping[str, Callable[[Mapping[str, Any]], tuple[bool, str]]],
         log: Callable[[str], None] | None = None) -> dict[str, Adoption]:
    """The adoption plan of every launch island (name -> :class:`Adoption`). ``probes``:
    source -> probe(handle) -> (running, why). A source without a probe (``ssh``: its
    journal decides at its start) is not adopted here."""
    say = log or (lambda m: print(m, file=sys.stderr, flush=True))
    out = {}
    for name, ref in islands.items():
        record = records.get(ref.island)
        if record is None:
            out[name] = Adoption(name, ref.island, False, 0, why="no record")
            continue
        incarnation = int(record.get("incarnation") or 0)
        relaunches = int(record.get("relaunches") or 0)
        handle = record.get("handle") or {}
        nxt = Adoption(name, ref.island, False, incarnation + 1, relaunches + 1)
        state = record.get("state")
        probe = probes.get(ref.source)
        if record.get("source") not in (None, ref.source) or record.get("name") not in (None, name):
            why = f"recorded as {record.get('name')} on {record.get('source')}, now {name} on {ref.source}"
        elif state not in ADOPTABLE:
            why = f"its record is {state}"
        elif probe is None:
            why = f"{ref.source}: decided at its start"
            nxt = Adoption(name, ref.island, False, incarnation, relaunches)
        else:
            try:
                running, why = probe(handle)
            except Exception as error:  # noqa: BLE001 - unknown is not "gone": nothing is ordered
                raise AdoptionUnknown(f"{name}: island {ref.island} ({ref.source}): whether its "
                                      f"incarnation {incarnation} still runs is unknown "
                                      f"({type(error).__name__}: {error})"[:400]) from error
            if running:
                job_id = handle.get("call_id") if ref.source == "modal" else handle.get("job_id")
                if job_id is None:  # a sky record ordered without its job: the job the probe found
                    job_id = getattr(probe, "found", {}).get(str(handle.get("cluster") or name))
                out[name] = Adoption(name, ref.island, True, incarnation, relaunches, job_id=job_id,
                                     cluster=handle.get("cluster") or name, why=why)
                say(f"[launcher] {name}: adopting island {ref.island} incarnation {incarnation} ({why}); "
                    "no new order, the trainer keeps running")
                continue
        out[name] = Adoption(nxt.name, nxt.island, False, nxt.incarnation, nxt.relaunches, why=why)
        if record is not None and state in ADOPTABLE and probe is not None:
            say(f"[launcher] {name}: not adopted ({why}); it is launched again as incarnation {nxt.incarnation}")
    return out
