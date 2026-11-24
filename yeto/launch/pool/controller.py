"""RolloutController: the rollout island pools run inside the launcher's controller
process, next to the trainer fleet.

One controller drives any number of :class:`~.island_pool.IslandPool` (one per
provider config, or one pool over a :class:`~.provider.MultiProvider`) off the
trainer fleet's poll loop: a pool tick reads every member's status and probes its
node (an ssh round trip each, up to the probe timeout).

With ``pool_workers`` (the store control plane turns it on; off, one thread ticks
every pool in turn, as before) a started controller is a supervisor: every pool ticks
on its own worker thread, so
one pool whose provider call hangs (a release blocked for minutes) never holds the
others; :meth:`RolloutController.status` marks it ``blocked`` (and ``pool_blocked`` is
written once) while its tick runs past ``blocked_s``. A worker that dies (anything
escaping its tick) is restarted by the supervisor; a pool on the store control plane
reloads its state from the store first. Pools on the store control plane release
machines in the background (``IslandPool.async_release``): a tick only decides.
:meth:`RolloutController.tick` still ticks every pool once, in order (tests, tools).

:meth:`RolloutController.detach` stops driving without draining: the samplers keep
running and writing their shards, every pool's state is saved and its store lock
released, and the next controller adopts the members under their shards (no restart,
no new shard). A controller with ``drain_on_exit`` false detaches at process exit.

What it shares with the trainer's ``FleetController`` (``FleetController(rollout=...)``):

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
DEFAULT_BLOCKED_S = 300.0  # a pool tick running this long: the pool is reported blocked
_STATE_RANK = {"draining": 1, "gone": 2}  # member states a stale store copy may not undo
RESTART_RESET_S = 600.0  # a worker that ran this long before it died: its restart backoff starts over

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
    # the supervisor's view of the pool's worker
    tick_lock: Any = field(default_factory=threading.Lock)  # one tick (or the stop's drain) at a time
    worker: Any = None
    wake: Any = field(default_factory=threading.Event)
    tick_began: float | None = None  # monotonic start of the tick in flight (None: idle)
    worker_error: BaseException | None = None
    restarts: int = 0
    blocked_reported: bool = False
    restart_at: float | None = None  # a dead worker is started again at this monotonic time
    worker_started_at: float | None = None
    # the stop or detach gave up waiting for the tick in flight ("stop" / "detach"): the
    # pool is detached (nothing more is ordered, started or fenced), and its state save and
    # store lock release run when that tick returns, so whatever it did is recorded
    close_pending: str | None = None


class RolloutController:
    def __init__(self, pools: Mapping[str, IslandPool | PoolSlot], *, stop_flag: str | os.PathLike[str] | None = None,
                 fleet_log: Any = None, budget: Callable[[], str | None] | None = None, interval: float = 60.0,
                 log: Callable[[str], None] = print, thread_cls=threading.Thread, drain_on_exit: bool = True,
                 drain_wait_s: float = 10.0, sleep: Callable[[float], None] = time.sleep,
                 stall_s: float = DEFAULT_STALL_S, blocked_s: float = DEFAULT_BLOCKED_S,
                 restart_backoff_s: float = 5.0, detach_flag: str | os.PathLike[str] | None = None,
                 pool_workers: bool = False):
        self.slots = {name: p if isinstance(p, PoolSlot) else PoolSlot(p) for name, p in pools.items()}
        self.stop_flag = None if stop_flag is None else Path(stop_flag)
        self.fleet_log, self.budget, self.interval, self.log = fleet_log, budget, float(interval), log
        self.thread_cls, self.drain_on_exit, self.drain_wait_s, self.sleep = thread_cls, drain_on_exit, drain_wait_s, sleep
        self.stall_s = float(stall_s)
        self.blocked_s, self.restart_backoff_s = float(blocked_s), float(restart_backoff_s)
        self.detach_flag = None if detach_flag is None else Path(detach_flag)
        # True: the supervisor and one worker per pool (module doc); False (the default): one
        # thread ticks every pool in turn, as before
        self.pool_workers = bool(pool_workers)
        self.detached: str | None = None
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
            if getattr(pool, "control", None) is not None:
                pool.async_release = True  # the store control plane: a tick only decides

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
        """One tick of every pool that is not stopped, in order; the STOP flag drains them
        all. A tick that raises is an event (``tick_failed``), never the end of the pools."""
        with self._lock:
            if self.stopped or self.detached:
                return
            if self._flags():
                return
            for name, slot in self.slots.items():
                self._tick_slot(name, slot)
            self.ticks += 1

    def _flags(self) -> bool:
        """The STOP flag drains every pool, the DETACH flag detaches; True: one of them acted."""
        if self.stop_flag is not None and self.stop_flag.exists():
            self._stop_locked(f"STOP flag {self.stop_flag}")
            return True
        if self.detach_flag is not None and self.detach_flag.exists():
            self.detach(f"DETACH flag {self.detach_flag}")
            return True
        return False

    def _tick_slot(self, name: str, slot: PoolSlot) -> None:
        """One tick of one pool (its worker, or :meth:`tick`), under the pool's tick lock."""
        with slot.tick_lock:
            pool = slot.pool
            if self.stopped or self.detached or pool.stopped or pool.detached:
                if pool.detached and not slot.closed:
                    self._close_slot(name, slot)  # its store lock went to another controller
                return
            slot.tick_began = time.monotonic()
            try:
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
            finally:
                if slot.blocked_reported and pool.events is not None:  # the dashboard's blocked view ends
                    pool.events("pool_unblocked", pool=name, tick_s=round(time.monotonic() - slot.tick_began))
                slot.tick_began = None
                slot.blocked_reported = False
                if slot.close_pending is not None and not slot.closed:
                    self._close_after_tick(name, slot)

    def _close_after_tick(self, name: str, slot: PoolSlot) -> None:
        """The tick the stop / detach gave up on returned: save what it did, then close."""
        how, slot.close_pending = slot.close_pending, None
        pool = slot.pool
        if how == "detach":
            self._save_detached(slot)
            slot.left = []
        else:
            self._save_detached(slot)
            slot.left = pool.live()
            self.log(f"rollout: pool {name}: its blocked tick returned; {len(slot.left)} member(s) left billed")
        self._close_slot(name, slot)

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

    # -- the supervisor and the pool workers
    def start(self) -> RolloutController:
        """Start the supervisor thread (idempotent): it starts one worker per pool, restarts
        a worker that died, reports a blocked pool and watches the STOP / DETACH flags."""
        if self._thread is not None:
            return self
        with _ACTIVE_LOCK:
            if self not in _ACTIVE:
                _ACTIVE.append(self)
        self._thread = self.thread_cls(target=self._loop, name="rollout-controller", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        if not self.pool_workers:
            self._serial_loop()
            return
        try:
            while not self.stopped and not self.detached and not self.all_stopped():
                with self._lock:
                    if not self.stopped and not self.detached and self._flags():
                        break
                self.supervise()
                if self._wake.wait(min(1.0, self.interval)):
                    self._wake.clear()
        except BaseException as error:  # noqa: BLE001 - kept for check(); the pools stay drainable
            self.error = error
            self.log(f"rollout: controller thread died: {error!r}")

    def _serial_loop(self) -> None:
        """Without ``pool_workers``: every pool in turn on this thread (one tick at a time)."""
        try:
            while not self.stopped and not self.detached and not self.all_stopped():
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

    def supervise(self, now: float | None = None) -> None:
        """One supervisor pass: start or restart the worker of every driven pool and report
        a pool whose tick has run longer than ``blocked_s``."""
        now = time.monotonic() if now is None else now
        for name, slot in self.slots.items():
            pool = slot.pool
            if self.stopped or self.detached or pool.stopped or pool.detached:
                continue
            worker = slot.worker
            if worker is None or not _alive(worker):
                if worker is not None and slot.restart_at is None:
                    self._worker_died(name, slot, now)
                if slot.restart_at is not None and now < slot.restart_at:
                    continue  # its backoff: the supervisor never waits it out (the other pools go on)
                slot.restart_at = None
                slot.worker = self.thread_cls(target=self._work, args=(name, slot), name=f"pool-{name}",
                                              daemon=True)
                slot.worker_started_at = now
                slot.worker.start()
            began = slot.tick_began
            if began is not None and now - began > self.blocked_s and not slot.blocked_reported:
                slot.blocked_reported = True
                self.log(f"rollout: pool {name} is blocked: its tick has run {now - began:.0f}s (the other "
                         "pools tick on)")
                if pool.events is not None:
                    pool.events("pool_blocked", pool=name, tick_s=round(now - began), blocked_s=self.blocked_s)

    def _worker_died(self, name: str, slot: PoolSlot, now: float) -> None:
        """The worker of ``name`` died: report it, keep or reload the pool's state, and
        schedule the restart (backoff ``restart_backoff_s`` times the restarts in a row, at
        most 60 s; a worker that ran ``RESTART_RESET_S`` before it died starts the count over)."""
        if slot.worker_started_at is not None and now - slot.worker_started_at >= RESTART_RESET_S:
            slot.restarts = 0
        slot.restarts += 1
        error = slot.worker_error
        slot.worker_error = None
        self.log(f"rollout: pool {name} worker died ({error!r}); restarting it (restart {slot.restarts})")
        pool = slot.pool
        if pool.events is not None:
            pool.events("worker_restarted", error=repr(error)[:300], restarts=slot.restarts)
        control = getattr(pool, "control", None)
        if control is not None:
            self._restore_after_death(name, pool, control)
        if self.restart_backoff_s > 0:
            slot.restart_at = now + min(self.restart_backoff_s * slot.restarts, 60.0)

    def _restore_after_death(self, name: str, pool: IslandPool, control: Any) -> None:
        """A store pool whose worker died: the store copy replaces what a tick cut off may
        have left in memory, except a member the memory holds as draining or gone (released
        after that save: never live again). Every member the memory holds that the store
        copy does not (ordered after the last successful save: a save that failed, or the
        crash came first) is kept, so no billed machine drops out of the pool; the result
        is saved at once. The futures of background starts / releases are kept: they track threads
        still running (dropping them would start or release a member twice)."""
        held = {m.ticket: m for m in pool.members}
        try:
            doc = control.state.load()
            if doc is not None:
                pool._load(doc, control.state.key)  # noqa: SLF001
        except Exception as reload_error:  # noqa: BLE001 - the pool goes on with what it holds
            self.log(f"rollout: pool {name}: state not reloaded from the store ({reload_error!r})")
            return
        for i, m in enumerate(pool.members):  # a member the memory saw drain or end after that save stays so
            mine = held.get(m.ticket)
            if mine is not None and _STATE_RANK.get(mine.state, 0) > _STATE_RANK.get(m.state, 0):
                pool.members[i] = mine
        known = {m.ticket for m in pool.members}
        extra = [m for ticket, m in held.items() if ticket not in known]
        if extra:
            pool.members.extend(extra)
            self.log(f"rollout: pool {name}: kept {len(extra)} member(s) the store copy did not hold yet")
            pool._event("worker_state_merged", tickets=[m.ticket for m in extra])  # noqa: SLF001
        pool._save_now()  # noqa: SLF001

    def _work(self, name: str, slot: PoolSlot) -> None:
        """A pool's worker: tick, wait ``interval`` (a control file change or :meth:`wake`
        ends the wait), until the pool or the controller stops."""
        try:
            while not self.stopped and not self.detached and not slot.pool.stopped and not slot.pool.detached:
                self._tick_slot(name, slot)
                if self._wait_slot(slot):
                    slot.wake.clear()
        except BaseException as error:  # noqa: BLE001 - the supervisor restarts the worker
            slot.worker_error = error

    def _wait_slot(self, slot: PoolSlot) -> bool:
        left = self.interval
        while left > 0 and not self.stopped and not self.detached:
            step = min(1.0, left)
            if slot.wake.wait(step):
                return True
            if getattr(slot.pool, "control_changed", lambda: False)():
                return False
            left -= step
        return slot.wake.is_set()

    def all_stopped(self) -> bool:
        return all(s.pool.stopped or s.pool.detached for s in self.slots.values())

    def check(self) -> str | None:
        """Why the controller needs attention (its thread died), None when healthy."""
        if self.error is not None:
            return f"rollout controller thread died: {self.error!r}"
        return None

    def wake(self) -> None:
        self._wake.set()
        for slot in self.slots.values():
            slot.wake.set()

    # -- stop / drain
    def stop(self, why: str, *, join_s: float = 120.0) -> dict[str, list[Member]]:
        """Drain every pool (sampler stop, release, confirmed ended; failed drains retried),
        stop the thread, close the locks. Returns the members still billed, by pool."""
        self.wake()
        with self._lock:
            left = self._stop_locked(why, join_s=join_s)
        thread = self._thread
        if thread is not None and thread is not threading.current_thread() and hasattr(thread, "join"):
            thread.join(join_s)
        return left

    close = stop

    def _stop_locked(self, why: str, *, join_s: float = 120.0) -> dict[str, list[Member]]:
        if self.stopped is None:
            self.stopped = why
            self.log(f"rollout: stopping every pool: {why}")
        self.wake()
        out: dict[str, list[Member]] = {}
        deadline = time.monotonic() + max(0.0, join_s)  # one wait for all pools, not join_s per stuck pool
        try:
            for name, slot in self.slots.items():
                pool = slot.pool
                if pool.detached:  # another controller drives its machines: never drained from here
                    self._close_slot(name, slot)
                    continue
                # the pool's tick in flight finishes first (its worker then sees the stop); a
                # tick blocked past join_s is not raced: its members are reported as left, the
                # pool is detached (the tick, once it returns, orders / starts / fences nothing
                # more) and its store lock is released only when that tick returns
                if not slot.tick_lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
                    self.log(f"rollout: CRITICAL: pool {name} is blocked in a tick: not drained; its members "
                             "may still bill (`yeto run down` releases them)")
                    self._give_up_on_tick(name, slot, "stop", why)
                    slot.left = pool.live()
                    out[name] = list(slot.left)
                    continue
                try:
                    if pool.live() or not pool.stopped:
                        slot.left = pool.drain_all(why, wait_s=self.drain_wait_s, sleep=self.sleep)
                    self._save(slot)
                finally:
                    slot.tick_lock.release()
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
        if slot.close_pending is not None:
            return  # a tick still in flight: closed when it returns (_close_after_tick)
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

    def detach(self, why: str, *, join_s: float = 120.0) -> dict[str, list[str]]:
        """Stop driving every pool without draining (module doc): the workers end after
        their tick, each pool's state is saved, its store lock released (the next
        controller takes it at once) and its local lock closed. The machines keep running;
        returns the tickets left running, by pool."""
        if self.stopped:
            return {}
        if self.detached is None:
            self.detached = why
            self.log(f"rollout: detaching every pool (no drain): {why}")
        self.wake()
        out: dict[str, list[str]] = {}
        deadline = time.monotonic() + max(0.0, join_s)
        for name, slot in self.slots.items():
            pool = slot.pool
            got = slot.tick_lock.acquire(timeout=max(0.0, deadline - time.monotonic()))
            if not got and not pool.stopped and not slot.closed:
                self._give_up_on_tick(name, slot, "detach", why)
            try:
                if not pool.stopped:
                    if getattr(pool, "detach", None) is not None:
                        pool.detach(why)
                    if got:
                        self._save_detached(slot)
                out[name] = [m.ticket for m in pool.live()]
            finally:
                if got:
                    slot.tick_lock.release()
            slot.left = []
            self._close_slot(name, slot)
        with _ACTIVE_LOCK:  # never drained at exit
            if self in _ACTIVE:
                _ACTIVE.remove(self)
        return out

    def _give_up_on_tick(self, name: str, slot: PoolSlot, how: str, why: str) -> None:
        """The stop / detach does not wait any longer for the tick in flight of ``name``."""
        slot.close_pending = how
        if getattr(slot.pool, "detach", None) is not None:
            slot.pool.detach(f"{how} gave up on a blocked tick: {why}")
        if slot.pool.events is not None:
            slot.pool.events("pool_tick_abandoned", during=how, tick_s=None if slot.tick_began is None
                             else round(time.monotonic() - slot.tick_began))
        if slot.tick_lock.acquire(blocking=False):  # the tick returned meanwhile: nobody else closes it
            try:
                if slot.close_pending is not None:
                    self._close_after_tick(name, slot)
            finally:
                slot.tick_lock.release()

    def _save_detached(self, slot: PoolSlot) -> None:
        """Save a pool that is detached (its own save path skips nothing it must keep)."""
        pool = slot.pool
        control = getattr(pool, "control", None)
        from .store_lock import LockLost, LockUnconfirmed  # noqa: PLC0415

        if control is not None:
            try:
                control.state.save(pool.snapshot())
            except (LockLost, LockUnconfirmed) as error:  # another controller took over: never a skipped close
                self.log(f"rollout: detach: state of {pool.config.shard_base} not saved to the store ({error}); "
                         "the controller that holds it now drives its members")
            except Exception as error:  # noqa: BLE001 - the last tick's save holds
                self.log(f"rollout: detach: state of {pool.config.shard_base} not saved: {error!r}")
        if slot.state_path is not None:
            try:
                pool.save(slot.state_path)
            except Exception as error:  # noqa: BLE001
                self.log(f"rollout: detach: local state of {pool.config.shard_base} not saved: {error!r}")

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
                **({"detached": pool.detached} if pool.detached else {}),
                **({"blocked_s": round(time.monotonic() - slot.tick_began)}
                   if slot.tick_began is not None and time.monotonic() - slot.tick_began > self.blocked_s else {}),
                **({"worker_restarts": slot.restarts} if slot.restarts else {}),
                "finished": pool.finished, "pending": list(pool.pending),
                "breaker": {"tripped": pool.breaker.tripped, "trips": pool.breaker.trips},
                "slot_name": pool.config.slot_name,
                "members": [{pool.config.slot_name: m.account, "ticket": m.ticket, "shard": m.shard, "state": m.state,
                             "lease_until": m.lease_until, "continues_shard": m.replaces_shard}
                            for m in pool.live()],
            }
        return {"stopped": self.stopped, **({"detached": self.detached} if self.detached else {}), "pools": out}


def _alive(thread: Any) -> bool:
    alive = getattr(thread, "is_alive", None)
    return True if alive is None else bool(alive())


# -- cleanup registry -----------------------------------------------------------

def active() -> list[RolloutController]:
    with _ACTIVE_LOCK:
        return list(_ACTIVE)


def drain_active(why: str) -> dict[str, list[Member]]:
    """Drain every started, not yet stopped controller (teardown, reaper, process exit). At
    a process exit a controller with ``drain_on_exit`` false is not drained; one with pools
    on the store control plane detaches (their store locks released for the next one)."""
    out: dict[str, list[Member]] = {}
    for controller in active():
        if controller.drain_on_exit or why != "process exit":
            for name, left in controller.stop(why).items():
                out[name] = left
        elif any(getattr(s.pool, "control", None) is not None for s in controller.slots.values()):
            controller.detach(why, join_s=10.0)
    return out


atexit.register(lambda: drain_active("process exit"))


# -- the National Compute / Daytona pool of a pool.json --------------------------

def nc_pool(cfg: dict[str, Any], config_path: str, *, ops: Any = None, lock: bool = True,
            control_objects: Any = None, name: str | None = None, lock_ttl_s: float | None = None) -> PoolSlot:
    """The pool ``python -m yeto.launch.nc pool --config <pool.json>`` runs, built the
    same way (state file restored, event log, scale control file, the pool lock) for a
    :class:`RolloutController`. Raises ``SystemExit`` when another loop holds the lock.

    ``control_objects`` (the run store's object API; None: off, as before): the store
    control plane (:mod:`.store_state`). The pool's state is ``fleet/pools/<name>/state.json``
    (the local state file imported once when the store has none, then written as a mirror
    only), guarded by the pool's store lock, and its shard numbers come from the store's
    allocator. Another controller holding the store lock raises ``SystemExit`` too."""
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
    try:
        pool_run.claim_state(cfg)  # another run's: archived when stopped, else refused
    except pool_run.PoolStateOwnerError as error:
        if held is not None:
            held.close()
        raise SystemExit(f"pool {cfg['shard_base']}: refused: {error}") from None
    pool_ops = ops if ops is not None else cli.pool_ops(cfg)
    if control_objects is None:
        pool = IslandPool.restore(state, pc, pool_ops, log=cli._log, events=events)  # noqa: SLF001
    else:
        pool = _store_pool(cfg, pc, pool_ops, control_objects, name or str(cfg["shard_base"]), state, events,
                           held, lock_ttl_s)
    pool.owner = pool_run.state_owner(cfg)  # this run's from now on (a file with no owner is stamped)
    pool.scale_path, pool.scale_run = pool_run.scale_path(cfg), cli.run_id(cfg)
    pool.apply_scale()
    pool_run.attach_offer_ledger(cfg, pool)  # every pool feeds the offer ledger (leads from it: replace_lead_auto)
    pool_run._write_pid(cfg, config_path)  # noqa: SLF001 - pool-status / pool-scale see a live loop
    pool_run._remove(pool_run.stop_path(cfg))  # noqa: SLF001

    def after_tick(p: IslandPool) -> str | None:
        if pool_run.retry_path(cfg).exists():
            pool_run._retry_in_loop(p, config_path)  # noqa: SLF001
        return "pool-stop requested" if pool_run.stop_path(cfg).exists() else None

    def close(left: list[Member]) -> None:
        if pool.control is not None:
            pool.control.close()  # the store lock: released (a detach or a drain alike)
        pool_run._remove(pool_run.pid_path(cfg))  # noqa: SLF001
        pool_run._remove(pool_run.stop_path(cfg))  # noqa: SLF001
        if left:
            pool_run._print_banner(pool_run.billing_banner(  # noqa: SLF001
                config_path, [m.__dict__ for m in left], why="the rollout controller stopped with members left",
                national=not external, kind=pool_run.provider_kind(cfg)))

    return PoolSlot(pool, state_path=state, lock=held, on_close=close, after_tick=after_tick)


def _member_table(ops: Any, name: str, shard_base: str) -> Any:
    """The pool's member table writer on the store its samplers write their shards to (the
    trainer's buffer store: ``ops.data_path().records``); None when the ops have none."""
    from yeto.rl.engine.member_table import MemberTableWriter  # noqa: PLC0415

    data_path = getattr(ops, "data_path", None)
    try:
        records = data_path().records if callable(data_path) else None
    except Exception:  # noqa: BLE001 - no store credentials yet: no table (the trainer keeps its list)
        return None
    objects = getattr(records, "objects", None)
    if objects is None or not callable(getattr(objects, "list", None)):
        return None
    return MemberTableWriter(objects, name, shard_base)


def _store_pool(cfg: dict[str, Any], pc: Any, ops: Any, objects: Any, name: str, state: Path, events: Any,
                held: Any, ttl_s: float | None) -> IslandPool:
    """:func:`nc_pool`'s pool on the store control plane (its ``control_objects``)."""
    from ..nc import cli  # noqa: PLC0415
    from . import store_state  # noqa: PLC0415

    kw = {} if ttl_s is None else {"ttl_s": float(ttl_s)}
    try:
        control = store_state.open_pool_control(objects, name, str(cfg["shard_base"]), **kw)
    except store_state.PoolLocked as error:
        if held is not None:
            held.close()
        raise SystemExit(f"pool {cfg['shard_base']}: refused: {error}") from None
    try:
        doc = control.state.load(legacy=state)
        control.members = _member_table(ops, name, str(cfg["shard_base"]))
        if pc.index_limit is None:
            # the allocator replaces next_shard. With the member table the trainer follows every
            # shard the pool publishes: only the tunnel ports bound the index; without one the
            # trainer's --rl-buffer-shards list still does
            pc.index_limit = (cli.pool_shard_limit(dict(cfg, shard_limit=None)) if control.members is not None
                              else pc.shard_limit)
        pool = IslandPool.from_doc(doc, pc, ops, control=control, where=control.state.key,
                                   log=cli._log, events=events)  # noqa: SLF001
    except BaseException:
        control.close()
        if held is not None:
            held.close()
        raise
    pool.state_path = state  # the local mirror (status, run up's readers)
    try:
        pool.end_unused_shards()  # numbers issued before a crash / detach that no order took
    except BaseException as error:  # noqa: BLE001 - a lost lock detaches the pool at its first tick anyway
        pool.log(f"pool {name}: unused shards not checked ({error!r})")
    if control.state.imported_from:
        pool.log(f"pool {name}: imported the local state {control.state.imported_from} into the store once "
                 f"({control.state.key}); the store holds it from now on")
        pool._event("state_imported", source=control.state.imported_from, key=control.state.key)  # noqa: SLF001
    return pool
