"""RolloutController: the rollout island pools run inside the launcher's controller
process, next to the trainer fleet.

One controller drives any number of :class:`~.island_pool.IslandPool` (one per
provider config, or one pool over a :class:`~.provider.MultiProvider`) on one
background thread: a pool tick reads every member's status and probes its node
(an ssh round trip each, up to the probe timeout), so it must never run on the
trainer fleet's poll loop. What it shares with the trainer's ``FleetController``
(``FleetController(rollout=...)``):

* the STOP flag (``yeto stop-run``): seen by either, it drains every pool
  (sampler stopped, machine released and confirmed ended) before the run ends;
* the run budget: ``budget()`` (None: an order may be placed; a string: why not)
  gates every order of every pool, next to each pool's own budget;
* the event tape: every pool event is also written to the run's ``fleet.jsonl``
  (``FleetLog.event("rollout_event", <pool>, pool_event=..., ...)``);
* the stall view: :meth:`progress` counts pool events and sampler progress (a probe
  that saw a sampler's tape grow -- a group or a trainer-wait heartbeat -- or its
  engine generate), and :meth:`stalled` says when every ready sampler with a progress
  signal has been silent for ``stall_s`` while none waits for the trainer: an alert
  (the fleet prints it and writes ``rollout_stalled``), never a kill;
* the cleanup: a started controller is registered (:func:`active`) until it is
  closed; :func:`drain_active` drains every registered one (the launcher's
  teardown and the reaper), and a process exit drains what is still running.

The controller keeps the subprocess mode's files: each pool saves its state file
after every tick, reads its scale control file (``scale_path``) every tick, and a
pool built by :func:`nc_pool` holds the same lock as ``python -m yeto.launch.nc
pool``, so the two modes never drive one pool at once.
"""

from __future__ import annotations

import atexit
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .island_pool import READY, IslandPool, Member

DEFAULT_STALL_S = 1800.0  # every ready sampler silent this long: the rollout stall alert

_ACTIVE: list[RolloutController] = []
_ACTIVE_LOCK = threading.Lock()


@dataclass
class PoolSlot:
    """One pool the controller drives: its state file (saved after every tick; None:
    not saved), the lock it holds while driven (closed at the end), what to run after
    each of its ticks (``after_tick(pool)`` -> a reason to stop this pool, or None: the
    subprocess loop's ``pool-stop`` / ``pool-retry`` request files) and what to call
    once it is drained (``on_close(left_members)``, e.g. a billing banner)."""

    pool: IslandPool
    state_path: Path | None = None
    lock: Any = None
    on_close: Callable[[list[Member]], None] | None = None
    after_tick: Callable[[IslandPool], str | None] | None = None
    left: list[Member] = field(default_factory=list)
    closed: bool = False


class RolloutController:
    def __init__(self, pools: Mapping[str, IslandPool | PoolSlot], *, stop_flag: str | os.PathLike[str] | None = None,
                 fleet_log: Any = None, budget: Callable[[], str | None] | None = None, interval: float = 60.0,
                 log: Callable[[str], None] = print, thread_cls=threading.Thread, drain_on_exit: bool = True,
                 drain_wait_s: float = 10.0, sleep: Callable[[float], None] = time.sleep,
                 stall_s: float = DEFAULT_STALL_S):
        self.slots = {name: p if isinstance(p, PoolSlot) else PoolSlot(p) for name, p in pools.items()}
        self.stop_flag = None if stop_flag is None else Path(stop_flag)
        self.fleet_log, self.budget, self.interval, self.log = fleet_log, budget, float(interval), log
        self.thread_cls, self.drain_on_exit, self.drain_wait_s, self.sleep = thread_cls, drain_on_exit, drain_wait_s, sleep
        self.stall_s = float(stall_s)
        self.stopped: str | None = None
        self.error: BaseException | None = None
        self.ticks = 0
        self._wake = threading.Event()
        self._thread: Any = None
        self._lock = threading.RLock()  # one tick or one stop at a time
        for name, slot in self.slots.items():
            pool = slot.pool
            if budget is not None and pool.gate is None:
                pool.gate = budget
            if pool.events is not None:
                pool.events.sinks.append(self._sink(name))

    @property
    def pools(self) -> dict[str, IslandPool]:
        return {name: s.pool for name, s in self.slots.items()}

    # -- events -> fleet.jsonl
    def _sink(self, name: str) -> Callable[[dict[str, Any]], None]:
        def sink(doc: dict[str, Any]) -> None:
            if self.fleet_log is None:
                return
            fields = {k: v for k, v in doc.items() if k not in ("t", "event")}
            self.fleet_log.event("rollout_event", name, pool_event=doc.get("event"), **fields)

        return sink

    # -- one pass
    def tick(self) -> None:
        """One tick of every pool that is not stopped; the STOP flag drains them all.
        A tick that raises is an event (``tick_failed``), never the end of the pools."""
        with self._lock:
            if self.stopped:
                return
            if self.stop_flag is not None and self.stop_flag.exists():
                self._stop_locked(f"STOP flag {self.stop_flag}")
                return
            for name, slot in self.slots.items():
                pool = slot.pool
                if pool.stopped:
                    continue
                try:
                    pool.tick()
                except Exception as error:  # noqa: BLE001 - one bad tick never ends a pool with billed nodes
                    if pool.events is not None:
                        pool.events("tick_failed", error=repr(error)[:300])
                why = None
                if slot.after_tick is not None:
                    try:
                        why = slot.after_tick(pool)
                    except Exception as error:  # noqa: BLE001
                        self.log(f"rollout: after_tick failed: {error!r}")
                if why:
                    self.log(f"rollout: {why}: draining the pool")
                    slot.left = pool.drain_all(why, wait_s=self.drain_wait_s, sleep=self.sleep)
                self._save(slot)
                if why:
                    # Like the exit of a pool loop: drop the pid file and the lock, so a
                    # waiting ``pool-stop`` sees the pool gone (it would otherwise wait for
                    # this process, the ``yeto run up`` that waits for it, and time out).
                    self._close_slot(name, slot)
            self.ticks += 1

    def _save(self, slot: PoolSlot) -> None:
        if slot.state_path is None:
            return
        if slot.pool.state_path == slot.state_path:  # logs, state_save_failing / _recovered events
            slot.pool._save_now()  # noqa: SLF001
            return
        try:
            slot.pool.save(slot.state_path)
        except Exception as error:  # noqa: BLE001 - the next tick saves again
            self.log(f"rollout: could not save {slot.state_path}: {error!r}")

    # -- the thread
    def start(self) -> RolloutController:
        """Run :meth:`tick` every ``interval`` on a background thread (idempotent)."""
        if self._thread is not None:
            return self
        with _ACTIVE_LOCK:
            if self not in _ACTIVE:
                _ACTIVE.append(self)
        self._thread = self.thread_cls(target=self._loop, name="rollout-controller", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        try:
            while not self.stopped and not self.all_stopped():
                self.tick()
                if self._wait():
                    self._wake.clear()
        except BaseException as error:  # noqa: BLE001 - kept for check(); the pools stay drainable
            self.error = error
            self.log(f"rollout: controller thread died: {error!r}")

    def _wait(self) -> bool:
        """Wait for the next tick; a control file change of any pool (pause, resume,
        scale) ends the wait within a second. True: woken by :attr:`_wake`."""
        left = self.interval
        while left > 0:
            step = min(1.0, left)
            if self._wake.wait(step):
                return True
            if any(getattr(s.pool, "control_changed", lambda: False)()
                   for s in self.slots.values() if not s.pool.stopped):
                return False
            left -= step
        return self._wake.is_set()

    def all_stopped(self) -> bool:
        return all(s.pool.stopped for s in self.slots.values())

    def check(self) -> str | None:
        """Why the controller needs attention (its thread died), None when healthy."""
        if self.error is not None:
            return f"rollout controller thread died: {self.error!r}"
        return None

    def wake(self) -> None:
        self._wake.set()

    # -- stop / drain
    def stop(self, why: str, *, join_s: float = 120.0) -> dict[str, list[Member]]:
        """Drain every pool (sampler stop, release, confirmed ended; failed drains retried),
        stop the thread, close the locks. Returns the members still billed, by pool."""
        self._wake.set()
        with self._lock:
            left = self._stop_locked(why)
        thread = self._thread
        if thread is not None and thread is not threading.current_thread() and hasattr(thread, "join"):
            thread.join(join_s)
        return left

    close = stop

    def _stop_locked(self, why: str) -> dict[str, list[Member]]:
        if self.stopped is None:
            self.stopped = why
            self.log(f"rollout: stopping every pool: {why}")
        out: dict[str, list[Member]] = {}
        try:
            for name, slot in self.slots.items():
                pool = slot.pool
                if pool.live() or not pool.stopped:
                    slot.left = pool.drain_all(why, wait_s=self.drain_wait_s, sleep=self.sleep)
                self._save(slot)
                out[name] = list(slot.left)
                self._close_slot(name, slot)
        finally:
            # a drain that raised is not run a second time by the atexit hook (at
            # interpreter shutdown its clients may be closed and the retry hangs): the
            # caller reports it and `yeto run down` releases what is left
            with _ACTIVE_LOCK:
                if self in _ACTIVE:
                    _ACTIVE.remove(self)
        return out

    def _close_slot(self, name: str, slot: PoolSlot) -> None:
        """``on_close`` once, then release the pool lock (idempotent)."""
        if slot.closed and not slot.left:
            return
        slot.closed = True
        if slot.on_close is not None:
            try:
                slot.on_close(list(slot.left))
            except Exception as error:  # noqa: BLE001
                self.log(f"rollout: on_close of pool {name} failed: {error!r}")
        if slot.lock is not None:
            try:
                slot.lock.close()
            except Exception:  # noqa: BLE001
                pass
            slot.lock = None

    # -- views
    def progress(self) -> int:
        """Pool events plus sampler progress so far (all pools): a counter that moves while
        the rollout side works (a group written, a trainer-wait heartbeat, tokens)."""
        events = sum(len(s.pool.events.events) for s in self.slots.values() if s.pool.events is not None)
        return events + self.sampler_progress()

    def sampler_progress(self) -> int:
        """Probes, over every member of every pool, that saw its sampler move."""
        return sum(int(m.progress_ticks) for s in self.slots.values() for m in s.pool.members)

    def stalled(self, now: float | None = None) -> str | None:
        """Why the rollout side looks stuck, None otherwise: every ready, unpaused sampler
        with a progress signal has been silent for ``stall_s``, and none of them waits for
        the trainer (a sampler blocked on the trainer heartbeats; its newest tape line is
        the wait). Samplers without a signal (e.g. Modal calls) give no verdict."""
        if self.stall_s <= 0 or self.stopped:
            return None
        watched = [m for s in self.slots.values() if not s.pool.stopped for m in s.pool.live()
                   if m.state == READY and not m.paused and m.progress_at is not None]
        if not watched or any(m.trainer_wait for m in watched):
            return None
        if now is None:
            now = next(iter(self.slots.values())).pool.clock()
        quiet = now - max(float(m.progress_at) for m in watched)
        if quiet < self.stall_s:
            return None
        return (f"no sampler wrote a group or a heartbeat for {quiet:.0f}s ({len(watched)} ready, "
                f"rollout stall alert {self.stall_s:.0f}s)")

    def status(self) -> dict[str, Any]:
        """Per pool: target, live / ready members, pending handoffs, why it is stopped,
        halted or finished (``yeto rollout status``)."""
        out: dict[str, Any] = {}
        for name, slot in self.slots.items():
            pool = slot.pool
            out[name] = {
                "target": pool.config.target, "stopped": pool.stopped, "halted": pool.halted,
                "finished": pool.finished, "pending": list(pool.pending),
                "breaker": {"tripped": pool.breaker.tripped, "trips": pool.breaker.trips},
                "slot_name": pool.config.slot_name,
                "members": [{pool.config.slot_name: m.account, "ticket": m.ticket, "shard": m.shard, "state": m.state,
                             "lease_until": m.lease_until, "continues_shard": m.replaces_shard}
                            for m in pool.live()],
            }
        return {"stopped": self.stopped, "pools": out}


# -- cleanup registry -----------------------------------------------------------

def active() -> list[RolloutController]:
    with _ACTIVE_LOCK:
        return list(_ACTIVE)


def drain_active(why: str) -> dict[str, list[Member]]:
    """Drain every started, not yet stopped controller (teardown, reaper, process exit)."""
    out: dict[str, list[Member]] = {}
    for controller in active():
        if controller.drain_on_exit or why != "process exit":
            for name, left in controller.stop(why).items():
                out[name] = left
    return out


atexit.register(lambda: drain_active("process exit"))


# -- the National Compute / Daytona pool of a pool.json --------------------------

def nc_pool(cfg: dict[str, Any], config_path: str, *, ops: Any = None, lock: bool = True) -> PoolSlot:
    """The pool ``python -m yeto.launch.nc pool --config <pool.json>`` runs, built the
    same way (state file restored, event log, scale control file, the pool lock) for a
    :class:`RolloutController`. Raises ``SystemExit`` when another loop holds the lock."""
    from ..nc import cli, pool_run  # noqa: PLC0415

    external = not pool_run.is_national_compute(cfg)
    pc = cli.pool_config(cfg, allowed=None if external else cli.allowed_accounts())
    if cfg.get("route", "ssh") == "ssh":
        pool_run.check_sampler_spec(cfg, pc)
    held = None
    if lock:
        held = pool_run.acquire_lock(pool_run.lock_path(cfg), run=cli.run_id(cfg), role="controller")
        if held is None:
            raise SystemExit(f"pool {cfg['shard_base']}: another instance holds {pool_run.lock_path(cfg)}; "
                             f"stop it with: {pool_run.stop_command(config_path)}")
    events = pool_run.event_log(cfg)
    state = cli.pool_state_path(cfg)
    pool = IslandPool.restore(state, pc, ops if ops is not None else cli.pool_ops(cfg), log=cli._log,  # noqa: SLF001
                              events=events)
    pool.scale_path, pool.scale_run = pool_run.scale_path(cfg), cli.run_id(cfg)
    pool.apply_scale()
    pool_run._write_pid(cfg, config_path)  # noqa: SLF001 - pool-status / pool-scale see a live loop
    pool_run._remove(pool_run.stop_path(cfg))  # noqa: SLF001

    def after_tick(p: IslandPool) -> str | None:
        if pool_run.retry_path(cfg).exists():
            pool_run._retry_in_loop(p, config_path)  # noqa: SLF001
        return "pool-stop requested" if pool_run.stop_path(cfg).exists() else None

    def close(left: list[Member]) -> None:
        pool_run._remove(pool_run.pid_path(cfg))  # noqa: SLF001
        pool_run._remove(pool_run.stop_path(cfg))  # noqa: SLF001
        if left:
            pool_run._print_banner(pool_run.billing_banner(  # noqa: SLF001
                config_path, [m.__dict__ for m in left], why="the rollout controller stopped with members left",
                national=not external, kind=pool_run.provider_kind(cfg)))

    return PoolSlot(pool, state_path=state, lock=held, on_close=close, after_tick=after_tick)
