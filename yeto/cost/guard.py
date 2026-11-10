"""The run's cost breaker: budget, idle shutdown, NC lease expiry.

The launcher runs one :class:`CostGuard` thread per run. Every ``poll_s`` it

1. reads the new lines of the run's event tapes (idle detection input);
2. totals the cost of every resource in the run's ledger (:mod:`.billing`);
3. at ``budget_warn_usd`` emits ``cost_budget_warn`` (once);
4. at ``budget_usd`` emits ``cost_budget_exceeded`` and calls ``graceful_stop``
   (default: the run's STOP flag - the trainer loop stops the trainer with a cut,
   the fleet ends and the launcher tears down every resource); if the run has not
   ended ``stop_grace_s`` later it calls ``force_stop`` (default: release every
   ledger resource by exact id, :mod:`.cleanup`);
5. idle (see :mod:`.idle`): ``idle_warn`` after N minutes, ``idle_shutdown`` and
   the same graceful -> forced stop after N + M minutes;
6. an NC lease still open ``nc_overdue_grace_s`` after its ``lease_until`` is
   reported (``nc_lease_overdue``, once): it keeps billing past its end, but the
   NC sampler pool owns its leases (its own lease-end drain, ``pool-stop``,
   ``yeto cleanup --include-nc``); a trainer-side guard never releases one
   unless ``release_overdue_nc`` is set.

Every event goes to ``<run dir>/cost-events.jsonl`` and to stdout
(``[cost-guard] ...``), so the launcher log and the dashboard see it.
"""

from __future__ import annotations

import glob
import json
import os
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .billing import CostSummary, ModalBilling, Rates, _ts, resource_costs
from .idle import IdleConfig, IdleDetector
from .ledger import Resource, ResourceLedger

EVENTS_FILE = "cost-events.jsonl"
PREFIX = "YETO_RL_EVENT "


@dataclass
class GuardConfig:
    budget_usd: float | None = None
    budget_warn_usd: float | None = None  # default: 80 % of budget_usd
    poll_s: float = 60.0
    stop_grace_s: float = 20 * 60
    nc_overdue_grace_s: float = 5 * 60
    # the NC pool owns its leases: a trainer's guard only alerts on an overdue one
    release_overdue_nc: bool = False
    idle: IdleConfig = field(default_factory=IdleConfig)

    @property
    def warn_usd(self) -> float | None:
        if self.budget_warn_usd is not None:
            return self.budget_warn_usd
        return None if self.budget_usd is None else 0.8 * self.budget_usd


class TapeTail:
    """New records of every file matching the globs, from where the last read stopped."""

    def __init__(self, globs: Iterable[str]) -> None:
        self.globs = list(globs)
        self.offsets: dict[str, int] = {}

    def read(self) -> list[tuple[str, dict[str, Any]]]:
        out = []
        for pattern in self.globs:
            for path in sorted(set(glob.glob(pattern, recursive=True))):
                if path.endswith(".gz"):
                    continue
                try:
                    size = os.path.getsize(path)
                    off = self.offsets.get(path, 0)
                    if size < off:  # rewritten file: start over
                        off = 0
                    if size == off:
                        continue
                    with open(path, "rb") as fh:
                        fh.seek(off)
                        data = fh.read(size - off)
                except OSError:
                    continue
                cut = data.rfind(b"\n")
                if cut < 0:
                    continue  # no complete line yet
                self.offsets[path] = off + cut + 1
                for raw in data[:cut].decode("utf-8", "replace").splitlines():
                    i = raw.find(PREFIX)
                    text = raw[i + len(PREFIX):] if i >= 0 else raw
                    try:
                        rec = json.loads(text)
                    except ValueError:
                        continue
                    if isinstance(rec, dict) and "event" in rec:
                        out.append((path, rec))
        return out


def record_time(rec: dict[str, Any], default: float) -> float:
    for k in ("time_unix", "ts", "time", "t", "wall_time"):
        v = rec.get(k)
        if v is None:
            continue
        try:
            t = _ts(v)
        except ValueError:
            continue
        if t is not None and t > 1e9:
            return t
    return default


class CostGuard:
    def __init__(self, run: str, config: GuardConfig, *, ledger: ResourceLedger,
                 tapes: TapeTail | None = None, rates: Rates | None = None,
                 billing: ModalBilling | None = None,
                 graceful_stop: Callable[[str], None] | None = None,
                 force_stop: Callable[[str], None] | None = None,
                 release_resource: Callable[[Resource], None] | None = None,
                 run_ended: Callable[[], bool] = lambda: False,
                 events_path: str | os.PathLike | None = None,
                 clock: Callable[[], float] = time.time, log: Callable[[str], None] = print,
                 start: float | None = None) -> None:
        self.run = run
        self.config = config
        self.ledger = ledger
        self.tapes = tapes or TapeTail([])
        self.rates = rates or Rates()
        self.billing = billing
        self.graceful_stop = graceful_stop or self._default_graceful_stop
        self.force_stop = force_stop or self._default_force_stop
        self.release_resource = release_resource or self._default_release
        self.run_ended = run_ended
        self.clock = clock
        self.log = log
        self.start_ts = clock() if start is None else start
        self.events_path = Path(events_path) if events_path else ledger.path.parent / EVENTS_FILE
        self.idle = IdleDetector(config.idle, self.start_ts)
        self.warned_budget = False
        self.stop_reason: str | None = None
        self.stop_at: float | None = None
        self.forced = False
        self.idle_warned = False
        self.unpriced_warned: set[str] = set()
        self.overdue_released: set[str] = set()
        self.last_summary: CostSummary | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- defaults -------------------------------------------------------------------

    def _default_graceful_stop(self, reason: str) -> None:
        from .. import runs

        runs.request_stop(self.run)

    def _default_force_stop(self, reason: str) -> None:
        from .cleanup import cleanup_run

        cleanup_run(self.run, log=self.log)

    def _default_release(self, r: Resource) -> None:
        from .cleanup import ProductionCleanupOps, cleanup

        for a in cleanup(self.ledger, ProductionCleanupOps(), keys=(r.key,)):
            self.log(f"[cost-guard] {a.key}: {a.action} {a.detail}")

    # -- events --------------------------------------------------------------------

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        rec = {"event": event, "run": self.run, "time_unix": self.clock(), **fields}
        try:
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.events_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
        except OSError as exc:
            self.log(f"[cost-guard] cannot write {self.events_path}: {exc}")
        if event != "cost_tick":
            self.log(f"[cost-guard] {event}: " + ", ".join(f"{k}={v}" for k, v in fields.items()))
        return rec

    # -- one tick ------------------------------------------------------------------

    def _request_stop(self, reason: str, now: float) -> None:
        if self.stop_at is not None:
            return
        self.stop_reason, self.stop_at = reason, now
        self.emit("cost_guard_stop", reason=reason, mode="graceful",
                  force_after_s=self.config.stop_grace_s)
        try:
            self.graceful_stop(reason)
        except Exception as exc:  # noqa: BLE001 - the forced stop still follows
            self.log(f"[cost-guard] graceful stop failed ({exc}); forcing now")
            self.stop_at = now - self.config.stop_grace_s

    def tick(self) -> CostSummary:
        now = self.clock()
        for path, rec in self.tapes.read():
            self.idle.observe(rec, record_time(rec, now), source=path)
        resources = self.ledger.resources()
        # A reused run name keeps its ledger: what an earlier launch of that name
        # created and released before this guard started is not this launch's spend
        # (``yeto cost --run`` still reports the whole ledger). A resource still open
        # bills now and counts, whoever created it.
        counted = [r for r in resources if r.released_at is None or r.released_at >= self.start_ts]
        summary = resource_costs(counted, now, rates=self.rates, billing=self.billing)
        self.last_summary = summary
        total = summary.total_usd
        self.emit("cost_tick", total_usd=round(total, 4), budget_usd=self.config.budget_usd,
                  open=[i.key for i in summary.items if i.open],
                  unpriced=[i.key for i in summary.unpriced], billing_error=summary.billing_error)
        for i in summary.unpriced:
            if i.open and i.key not in self.unpriced_warned:
                self.unpriced_warned.add(i.key)
                self.emit("cost_unpriced", resource=i.key, note=i.note,
                          detail="this resource is NOT counted against the budget")
        cfg = self.config
        if cfg.warn_usd is not None and total >= cfg.warn_usd and not self.warned_budget:
            self.warned_budget = True
            self.emit("cost_budget_warn", total_usd=round(total, 2), warn_usd=cfg.warn_usd,
                      budget_usd=cfg.budget_usd)
        if cfg.budget_usd is not None and total >= cfg.budget_usd and self.stop_at is None:
            self.emit("cost_budget_exceeded", total_usd=round(total, 2), budget_usd=cfg.budget_usd)
            self._request_stop(f"budget ${cfg.budget_usd:.2f} reached (${total:.2f})", now)
        verdict = self.idle.verdict(now)
        if verdict.state in ("warn", "stop") and not self.idle_warned:
            self.idle_warned = True
            self.emit("idle_warn", reason=verdict.reason, idle_min=round(verdict.idle_s / 60, 1),
                      shutdown_in_min=round(max(0.0, cfg.idle.idle_after_s + cfg.idle.stop_after_s
                                                - verdict.idle_s) / 60, 1))
        elif verdict.state in ("ok", "loading") and self.idle_warned and self.stop_at is None:
            self.idle_warned = False
            self.emit("idle_cleared", state=verdict.state)
        if verdict.state == "stop" and self.stop_at is None:
            self.emit("idle_shutdown", reason=verdict.reason, idle_min=round(verdict.idle_s / 60, 1))
            self._request_stop(f"idle: {verdict.reason}", now)
        for r in resources:
            if r.kind != "nc_lease" or not r.open or r.key in self.overdue_released:
                continue
            until = _ts(r.attrs.get("lease_until"))
            if until is not None and now >= until + cfg.nc_overdue_grace_s:
                self.overdue_released.add(r.key)
                release = cfg.release_overdue_nc
                self.emit("nc_lease_overdue", resource=r.key, lease_until=r.attrs.get("lease_until"),
                          overdue_min=round((now - until) / 60, 1), action="release" if release else "alert",
                          detail=None if release else "kept: the NC pool owns it (pool-stop / "
                          f"yeto cleanup --run {self.run} --include-nc)")
                if not release:
                    continue
                try:
                    self.release_resource(r)
                except Exception as exc:  # noqa: BLE001
                    self.emit("nc_lease_release_failed", resource=r.key, error=str(exc))
        if self.stop_at is not None and not self.forced and not self.run_ended() \
                and now - self.stop_at >= cfg.stop_grace_s:
            self.forced = True
            self.emit("cost_guard_stop", reason=self.stop_reason, mode="forced",
                      detail=f"run still up {cfg.stop_grace_s / 60:.0f} min after the graceful stop")
            try:
                self.force_stop(self.stop_reason or "forced")
            except Exception as exc:  # noqa: BLE001
                self.emit("cost_guard_force_failed", error=str(exc))
        return summary

    # -- thread --------------------------------------------------------------------

    def start(self) -> "CostGuard":
        def loop() -> None:
            while not self._stop.is_set():
                try:
                    self.tick()
                except Exception as exc:  # noqa: BLE001 - the guard never takes the run down
                    self.log(f"[cost-guard] tick failed: {type(exc).__name__}: {exc}")
                self._stop.wait(self.config.poll_s)

        self._thread = threading.Thread(target=loop, name=f"yeto-cost-guard-{self.run}", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)


def guard_config_from_args(args) -> GuardConfig:
    g = lambda name, default=None: getattr(args, name, default)  # noqa: E731
    m = lambda name, default: float(g(name, default) if g(name) is not None else default) * 60  # noqa: E731
    return GuardConfig(
        budget_usd=g("budget_usd"), budget_warn_usd=g("budget_warn_usd"),
        poll_s=float(g("cost_poll_seconds") or 60), stop_grace_s=m("budget_stop_grace_minutes", 20),
        idle=IdleConfig(idle_after_s=m("idle_warn_minutes", 15), stop_after_s=m("idle_stop_minutes", 15),
                        load_grace_s=m("load_grace_minutes", 45), max_load_s=m("max_load_minutes", 120)))


def guard_enabled(args) -> bool:
    """On for every CLI launch (the parser sets the idle flags) unless --no-cost-guard;
    a programmatic launch without the cost flags runs it only when it sets a budget."""
    if getattr(args, "no_cost_guard", False):
        return False
    return hasattr(args, "idle_warn_minutes") or getattr(args, "budget_usd", None) is not None
