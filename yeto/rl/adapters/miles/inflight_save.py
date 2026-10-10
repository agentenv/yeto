"""rl-spot-cost-saving 4.1 (minimal): write the rollout's in-flight trajectories to
persistent storage within a time limit when a reclaim notice arrives.

Opt-in through ``YETO_SPOT_INFLIGHT_SAVE_DIR`` (a directory on a Modal Volume,
e.g. under the tape mount). When set, the learner:

* writes its pid to ``YETO_SPOT_INFLIGHT_PID_FILE`` (default
  ``/tmp/yeto-learner.pid``; diagnostics only). The Modal function process does
  not signal the learner or the Ray processes (they keep running until the
  platform kills the container; the rollout executor must answer the export);
* after every trained round writes a rehearsal copy (``rehearsal-r<k>.json``):
  this measures the save time (``last_save_s`` for the reclaim plan, and the
  time distribution 4.4 asks for);
* on the reclaim notice runs :class:`yeto.cloud.preemption.ModalExitHandler` with
  :meth:`InFlightSaver.save` (25 s cap; a save that overruns is abandoned and
  only its ``.tmp`` file is left, never a half written ``.json``). The notice is
  a marker file (``YETO_SPOT_RECLAIM_MARKER``) the Modal function process writes
  on SIGINT/SIGTERM; a daemon thread in the learner watches it. (S19 run
  s19-agentic5-b-20261010b: a signal forwarded to the learner was not handled
  within the ~20 s before the kill -- a Python signal handler runs only on the
  main thread, which was inside a training call.) Each step prints
  ``[yeto] rl_inflight_save_progress {...}``.

The export is :func:`.in_flight.export_in_flight` (same entries as a cut). Agentic
trajectories suspended between model turns are written as references only: by
design they cannot be continued in another container (``agentic_session_lost``),
so 4.2/4.3 (continue / rebuild) are not part of this module.

Every result is printed as ``[yeto] rl_inflight_save {json}`` (the container log
reaches the launcher even when the container dies right after) and emitted as
the tape event ``rl_inflight_save``.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

ENV_DIR = "YETO_SPOT_INFLIGHT_SAVE_DIR"
ENV_VOLUME = "YETO_SPOT_INFLIGHT_SAVE_VOLUME"   # Modal Volume to commit after a write
ENV_PID_FILE = "YETO_SPOT_INFLIGHT_PID_FILE"
DEFAULT_PID_FILE = "/tmp/yeto-learner.pid"
EVENT = "rl_inflight_save"
ROUND_EVENT = "rl_round_trained"
PROGRESS = "rl_inflight_save_progress"
ENV_MARKER = "YETO_SPOT_RECLAIM_MARKER"  # written by the Modal function process on its signal
DEFAULT_MARKER = "/tmp/yeto-reclaim-requested"
MARKER_POLL_S = 0.2


def marker_file(environ: Mapping[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    return env.get(ENV_MARKER) or DEFAULT_MARKER


def watch_marker(path: str, on_marker: Callable[[str], Any], *, poll_s: float = MARKER_POLL_S,
                 stop: Any = None, sleep: Callable[[float], Any] = time.sleep,
                 exists: Callable[[str], bool] = os.path.exists) -> None:
    """Daemon-thread loop: call ``on_marker(path)`` once when the marker appears.
    Independent of the main thread (a Python signal handler only runs there, and
    the main thread may sit in a long training call until the platform kill)."""
    while stop is None or not stop.is_set():
        if exists(path):
            on_marker(path)
            return
        sleep(poll_s)


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return bool(env.get(ENV_DIR))


def pid_file(environ: Mapping[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    return env.get(ENV_PID_FILE) or DEFAULT_PID_FILE


def volume_commit(volume_name: str) -> Callable[[], None]:
    def commit() -> None:
        import modal

        modal.Volume.from_name(volume_name).commit()

    return commit


def write_in_flight(exported: Mapping[str, Any], directory: str | os.PathLike[str], name: str, *,
                    commit: Callable[[], Any] | None = None,
                    clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """Write ``exported`` as ``<directory>/<name>.json`` (tmp + fsync + rename +
    dir fsync), then commit the volume. A failed commit is reported, not raised
    (the file is on the mount; Modal also commits when the container exits)."""
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    entries = list(exported.get("entries") or [])
    agentic = sum(1 for e in entries if str(e.get("session_ref") or "").startswith("codex-suspended:"))
    body = json.dumps({"schema": "yeto.inflight_save/v1", "written_at": time.time(), **dict(exported)},
                      sort_keys=True).encode("utf-8")
    t0 = clock()
    final = folder / f"{name}.json"
    tmp = folder / f".{name}.json.tmp"
    with open(tmp, "wb") as fh:
        fh.write(body)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, final)
    dir_fd = os.open(folder, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    t1 = clock()
    commit_error = None
    if commit is not None:
        try:
            commit()
        except Exception as exc:  # noqa: BLE001 - reported, the file is written
            commit_error = f"{type(exc).__name__}: {str(exc)[:160]}"
    t2 = clock()
    return {"path": str(final), "bytes": len(body), "entries": len(entries),
            "agentic_entries": agentic, "buffered_entries": len(entries) - agentic,
            "buffer_groups": exported.get("buffer_groups"),
            "write_s": round(t1 - t0, 4), "commit_s": None if commit is None else round(t2 - t1, 4),
            "commit_error": commit_error}


class InFlightSaver:
    """Export + write, timed. ``export(timeout_s)`` returns the export mapping."""

    def __init__(self, export: Callable[[float], Mapping[str, Any]], directory: str, *,
                 emit: Callable[..., Any] | None = None, commit: Callable[[], Any] | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 printer: Callable[[str], Any] | None = None) -> None:
        self.export, self.directory, self.emit = export, directory, emit
        self.commit, self.clock = commit, clock
        self.printer = printer or (lambda line: print(line, flush=True))
        self.last_save_s: float | None = None
        self.results: list[dict[str, Any]] = []

    def progress(self, kind: str, step: str, t0: float, **fields: Any) -> None:
        """Reclaim path only: one line per step, so a save cut short by the kill
        still shows how far it got (the result line is printed only at the end)."""
        if kind == "reclaim":
            self.printer(f"[yeto] {PROGRESS} " + json.dumps(
                {"step": step, "elapsed_s": round(self.clock() - t0, 4), **fields}, sort_keys=True, default=str))

    def _save(self, kind: str, name: str, budget_s: float, **extra: Any) -> dict[str, Any]:
        t0 = self.clock()
        self.progress(kind, "export_start", t0, budget_s=round(budget_s, 3))
        exported = self.export(budget_s)
        export_s = round(self.clock() - t0, 4)
        self.progress(kind, "export_done", t0, entries=len(exported.get("entries") or []))
        out = write_in_flight(exported, self.directory, name, commit=self.commit, clock=self.clock)
        self.progress(kind, "written", t0, write_s=out["write_s"], commit_s=out["commit_s"])
        out.update(kind=kind, export_s=export_s, total_s=round(self.clock() - t0, 4), **extra)
        self.results.append(out)
        self.printer(f"[yeto] {EVENT} {json.dumps(out, sort_keys=True)}")
        if self.emit is not None:
            try:
                self.emit(EVENT, **out)
            except Exception:  # noqa: BLE001 - the tape must never block the save
                pass
        return out

    def rehearse(self, round_id: Any, budget_s: float = 25.0) -> dict[str, Any] | None:
        """After a trained round: a timed copy; its total time becomes ``last_save_s``.
        A failure is printed and leaves ``last_save_s`` unchanged."""
        try:
            out = self._save("rehearsal", f"rehearsal-r{round_id}", budget_s, round=round_id)
        except Exception as exc:  # noqa: BLE001 - a rehearsal must never stop training
            self.printer(f"[yeto] {EVENT} " + json.dumps(
                {"kind": "rehearsal", "round": round_id, "error": f"{type(exc).__name__}: {str(exc)[:160]}"}))
            return None
        self.last_save_s = out["total_s"]
        return out

    def save(self, budget_s: float) -> dict[str, Any]:
        """Reclaim save (called by :func:`yeto.cloud.preemption.handle_notice`)."""
        out = self._save("reclaim", f"reclaim-{int(time.time())}", budget_s, save_budget_s=round(budget_s, 3))
        return {"dropped_trajectories": 0, **out}


def ray_export(executor: Any) -> Callable[[float], Mapping[str, Any]]:
    """Export through the rollout executor actor with a timeout (thread safe:
    no shared event loop, so it can run from the reclaim helper thread)."""
    from .in_flight import export_in_flight
    from .rollout import _executor_target

    def export(timeout_s: float) -> Mapping[str, Any]:
        kind, target = _executor_target(executor)
        if kind == "local":
            from types import SimpleNamespace

            return export_in_flight(SimpleNamespace(data_source=target))
        if kind != "ray":
            raise RuntimeError(f"rollout executor {type(executor).__name__} is not a Ray actor")
        import ray

        return ray.get(target.__ray_call__.remote(export_in_flight), timeout=max(0.1, timeout_s))

    return export


def wrap_emit_with_rehearsal(driver: Any, saver: InFlightSaver) -> None:
    """Rehearse after each ``rl_round_trained`` (driver.emit wrapped on the instance)."""
    original = driver.emit

    def emit(event: str, **fields: Any) -> None:
        original(event, **fields)
        if event == ROUND_EVENT:
            saver.rehearse(fields.get("rollout_id", fields.get("round")))

    saver.emit = original
    driver.emit = emit


def install(driver: Any, executor: Any, *, island: str, environ: Mapping[str, str] | None = None,
            start: bool = True) -> Any | None:
    """Wire 4.1 on this learner when ``YETO_SPOT_INFLIGHT_SAVE_DIR`` is set; None otherwise."""
    env = os.environ if environ is None else environ
    directory = env.get(ENV_DIR)
    if not directory:
        return None
    from yeto.cloud import preemption as p

    volume = env.get(ENV_VOLUME)
    saver = InFlightSaver(ray_export(executor), directory,
                          commit=volume_commit(volume) if volume else None)
    wrap_emit_with_rehearsal(driver, saver)
    def emit_reclaim(event: str, **fields: Any) -> None:
        print(f"[yeto] {event} {json.dumps(fields, sort_keys=True, default=str)}", flush=True)
        saver.emit(event, **fields)

    handler = p.ModalExitHandler(island, save=saver.save, leave=lambda: None, emit=emit_reclaim,
                                 last_save_s=lambda: saver.last_save_s, role="train")
    marker = marker_file(env)

    def on_marker(path: str) -> None:
        print(f"[yeto] {PROGRESS} " + json.dumps({"step": "marker_seen", "marker": path,
                                                  "last_save_s": saver.last_save_s}), flush=True)
        handler.handle(source="modal_signal")

    if start:
        try:  # a marker left from an earlier process in this container is stale
            os.unlink(marker)
        except OSError:
            pass
        Path(pid_file(env)).write_text(f"{os.getpid()}\n", encoding="utf-8")
        import threading

        threading.Thread(target=watch_marker, args=(marker, on_marker), daemon=True,
                         name="yeto-reclaim-marker").start()
    handler.saver = saver
    handler.marker = marker
    return handler
