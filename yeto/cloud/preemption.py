"""Reclaim (spot preemption) notices: one event shape, one handling order.

rl-spot-cost-saving tasks 1.3-1.7, design D3. Every cloud feeds the same
:class:`ReclaimNotice`; :func:`handle_notice` then runs the fixed order

    remaining time -> save if it fits -> leave -> write ``spot_reclaim`` event

Rules (spec "回收通知处理顺序"):

* save only if ``remaining_s > last_save_s + margin_s``;
* no measured save time -> not enough time (skip the save, write the reason);
* the whole handler never runs past the cloud grace (Modal: 30 s - 5 s margin).

Sources: Modal (interrupt signal, :class:`ModalExitHandler`), AWS (instance
metadata poll, :class:`AwsMetadataPoller`). Nebius / Verda: unchecked, so the
caller treats them as "no notice" (the island just disappears; lease expiry).

The event carries ``billing`` and ``role`` so phase 2 (one on-demand anchor
island + spot islands that may be dropped, user decision 2026-10-09) reuses it
unchanged; :func:`summarize` groups by cloud and region for cost scoring.
"""

from __future__ import annotations

import json
import signal as _signal
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Mapping

from yeto.cloud import capabilities

RECLAIM_EVENT = "spot_reclaim"
DEFAULT_MARGIN_S = 10.0
MODAL_HANDLER_CAP_S = 25.0   # 30 s Modal grace - 5 s margin (D3)
AWS_POLL_S = 5.0
AWS_ACTION_URL = "http://169.254.169.254/latest/meta-data/spot/instance-action"
AWS_TOKEN_URL = "http://169.254.169.254/latest/api/token"

SKIP_NO_MEASUREMENT = "skip save: no measured save time"
SKIP_NOT_ENOUGH = "skip save: not enough time"


@dataclass
class ReclaimNotice:
    island: str
    cloud: str
    region: str | None = None
    billing: str = "spot"            # spot | on_demand
    role: str = "eval"               # eval | droppable | anchor | train (phase 2 uses droppable/anchor)
    source: str = "unknown"          # modal_signal | aws_metadata | test
    notified_at: float = 0.0         # wall clock (time.time) of the notice
    deadline_s: float | None = None  # seconds from notified_at until the kill; None = unknown


@dataclass(frozen=True)
class SavePlan:
    save: bool
    remaining_s: float | None
    reason: str


def plan_response(remaining_s: float | None, last_save_s: float | None,
                  margin_s: float = DEFAULT_MARGIN_S) -> SavePlan:
    """Pure decision: save or skip (spec scenarios: enough / not enough / no measurement)."""
    if last_save_s is None:
        return SavePlan(False, remaining_s, SKIP_NO_MEASUREMENT)
    if remaining_s is None or remaining_s <= last_save_s + margin_s:
        return SavePlan(False, remaining_s, SKIP_NOT_ENOUGH)
    return SavePlan(True, remaining_s, "save")


def handle_notice(notice: ReclaimNotice, *, save: Callable[[float], Any] | None,
                  leave: Callable[[], Any], emit: Callable[..., Any],
                  last_save_s: float | None, margin_s: float = DEFAULT_MARGIN_S,
                  cap_s: float | None = None, clock: Callable[[], float] = time.time,
                  write_marker: Callable[[str], Any] | None = None) -> dict[str, Any]:
    """Run the fixed order and return the event written.

    ``save(budget_s)`` must return within ``budget_s`` (it may return a mapping
    with ``dropped_trajectories`` / ``dropped_tokens``); it runs in a helper
    thread and is abandoned if it overruns. ``leave()`` returns truthy when the
    coordinator confirmed the LEAVE. ``cap_s`` bounds the whole handler."""
    start = clock()
    deadline = notice.deadline_s
    if cap_s is not None:
        deadline = cap_s if deadline is None else min(deadline, cap_s)
    remaining = None if deadline is None else max(0.0, deadline - (start - notice.notified_at))
    plan = plan_response(remaining, last_save_s, margin_s) if save is not None else \
        SavePlan(False, remaining, "skip save: nothing to save")
    saved, save_s, extra, outcome = False, None, {}, plan.reason
    if plan.save:
        budget = remaining - margin_s
        box: dict[str, Any] = {}

        def _run() -> None:
            try:
                box["out"] = save(budget)
            except Exception as exc:  # noqa: BLE001 - a failed save must not block the leave
                box["error"] = f"{type(exc).__name__}: {exc}"

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()
        worker.join(timeout=budget)
        save_s = round(clock() - start, 3)
        if worker.is_alive():
            outcome = "save timed out"
        elif "error" in box:
            outcome = f"save failed: {box['error']}"
        else:
            saved, outcome = True, "saved"
            if isinstance(box.get("out"), Mapping):
                extra = {k: box["out"][k] for k in ("dropped_trajectories", "dropped_tokens") if k in box["out"]}
    if not saved and write_marker is not None:
        try:
            write_marker(outcome)
        except Exception:  # noqa: BLE001
            pass
    try:
        ack = leave()
        left = None if ack is None else bool(ack)  # None: no coordinator (eval island)
    except Exception:  # noqa: BLE001
        left = False
    event = {**asdict(notice), "remaining_s": None if remaining is None else round(remaining, 3),
             "saved": saved, "save_s": save_s, "outcome": outcome, "leave_confirmed": left,
             "handler_s": round(clock() - start, 3),
             "dropped_trajectories": extra.get("dropped_trajectories"),
             "dropped_tokens": extra.get("dropped_tokens")}
    emit(RECLAIM_EVENT, **event)
    return event


# --- Modal: interrupt signal -> flag + bounded save + leave ---------------------


class ModalExitHandler:
    """Install on the island's main thread. On SIGINT/SIGTERM: set ``stop`` and
    run :func:`handle_notice` with a 25 s cap. The island loop polls
    :attr:`stop` and stops taking new work. Call :meth:`handle` directly from an
    ``@modal.exit`` hook as well (it runs once)."""

    def __init__(self, island: str, *, save: Callable[[float], Any] | None, leave: Callable[[], Any],
                 emit: Callable[..., Any], last_save_s: Callable[[], float | None] | float | None,
                 region: str | None = None, role: str = "eval", margin_s: float = 5.0,
                 cap_s: float = MODAL_HANDLER_CAP_S, clock: Callable[[], float] = time.time,
                 write_marker: Callable[[str], Any] | None = None) -> None:
        self.island, self.region, self.role = island, region, role
        self.save, self.leave, self.emit = save, leave, emit
        self._last_save = last_save_s
        self.margin_s, self.cap_s, self.clock = margin_s, cap_s, clock
        self.write_marker = write_marker
        self.stop = threading.Event()
        self.event: dict[str, Any] | None = None
        self._lock = threading.Lock()

    def install(self, signals: Iterable[str] = ("SIGINT", "SIGTERM")) -> list[str]:
        if threading.current_thread() is not threading.main_thread():
            return []
        done = []
        for name in signals:
            num = getattr(_signal, name, None)
            if num is not None:
                _signal.signal(num, lambda n, f, _name=name: self.handle(source="modal_signal"))
                done.append(name)
        return done

    def handle(self, source: str = "modal_signal") -> dict[str, Any] | None:
        self.stop.set()
        with self._lock:
            if self.event is not None:
                return self.event
            last = self._last_save() if callable(self._last_save) else self._last_save
            grace = capabilities.notice_seconds("modal")
            notice = ReclaimNotice(self.island, "modal", self.region, "spot", self.role, source,
                                   self.clock(), grace)
            self.event = handle_notice(notice, save=self.save, leave=self.leave, emit=self.emit,
                                       last_save_s=last, margin_s=self.margin_s, cap_s=self.cap_s,
                                       clock=self.clock, write_marker=self.write_marker)
            return self.event


# --- AWS: instance metadata poll -----------------------------------------------


def _aws_fetch(timeout: float = 2.0) -> tuple[int, str]:
    """IMDSv2 read of spot/instance-action -> (http status, body)."""
    req = urllib.request.Request(AWS_TOKEN_URL, method="PUT",
                                 headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"})
    try:
        token = urllib.request.urlopen(req, timeout=timeout).read().decode()
        req = urllib.request.Request(AWS_ACTION_URL, headers={"X-aws-ec2-metadata-token": token})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, ""


def parse_aws_action(status: int, body: str, *, now: float) -> float | None:
    """404 -> None (no notice). 200 with action stop/terminate -> seconds left."""
    if status != 200 or not body:
        return None
    try:
        doc = json.loads(body)
    except ValueError:
        return None
    if doc.get("action") not in ("stop", "terminate", "hibernate"):
        return None
    when = doc.get("time")
    if not when:
        return capabilities.notice_seconds("aws")
    from datetime import datetime

    try:
        ts = datetime.fromisoformat(str(when).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return capabilities.notice_seconds("aws")
    return max(0.0, ts - now)


@dataclass
class AwsMetadataPoller:
    """Poll every 5 s; on the first stop/terminate notice call ``on_notice(notice)`` once."""

    island: str
    on_notice: Callable[[ReclaimNotice], Any]
    region: str | None = None
    role: str = "eval"
    fetch: Callable[[], tuple[int, str]] = _aws_fetch
    interval_s: float = AWS_POLL_S
    clock: Callable[[], float] = time.time
    fired: bool = False
    errors: int = 0
    _stop: threading.Event = field(default_factory=threading.Event)

    def poll_once(self) -> ReclaimNotice | None:
        if self.fired:
            return None
        try:
            status, body = self.fetch()
        except Exception:  # noqa: BLE001 - metadata hiccup: try again next tick
            self.errors += 1
            return None
        now = self.clock()
        left = parse_aws_action(status, body, now=now)
        if left is None:
            return None
        self.fired = True
        notice = ReclaimNotice(self.island, "aws", self.region, "spot", self.role, "aws_metadata", now, left)
        self.on_notice(notice)
        return notice

    def run(self, sleep: Callable[[float], Any] | None = None) -> None:
        while not self.fired and not self._stop.is_set():
            self.poll_once()
            if sleep is not None:
                sleep(self.interval_s)
            else:
                self._stop.wait(self.interval_s)

    def start(self) -> threading.Thread:
        t = threading.Thread(target=self.run, daemon=True, name=f"aws-reclaim-{self.island}")
        t.start()
        return t

    def stop(self) -> None:
        self._stop.set()


# --- summary for cost scoring (task 1.6) -----------------------------------------


def summarize(events: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str | None], dict[str, Any]]:
    """Group ``spot_reclaim`` events by (cloud, region): count, intervals, losses."""
    groups: dict[tuple[str, str | None], list[Mapping[str, Any]]] = {}
    for ev in events:
        if ev.get("event", RECLAIM_EVENT) != RECLAIM_EVENT:
            continue
        groups.setdefault((ev["cloud"], ev.get("region")), []).append(ev)
    out = {}
    for key, evs in groups.items():
        evs = sorted(evs, key=lambda e: e.get("notified_at") or 0.0)
        times = [e.get("notified_at") or 0.0 for e in evs]
        out[key] = {
            "count": len(evs),
            "intervals_s": [round(b - a, 3) for a, b in zip(times, times[1:])],
            "saved": sum(bool(e.get("saved")) for e in evs),
            "dropped_trajectories": sum(e.get("dropped_trajectories") or 0 for e in evs),
            "dropped_tokens": sum(e.get("dropped_tokens") or 0 for e in evs),
        }
    return out


# --- training-island spot admission (task 1.7) ------------------------------------


def training_spot_admission(cloud: str, *, measured_save_s: float | None,
                            margin_s: float = DEFAULT_MARGIN_S) -> tuple[bool, str]:
    """Spec "训练岛默认不用 spot": allow only if notice checked, save+margin < notice,
    and the cloud has a durable store."""
    notice = capabilities.notice_seconds(cloud)
    if notice is None:
        return False, f"{cloud}: reclaim notice seconds unchecked; training islands stay on-demand"
    if measured_save_s is None:
        return False, f"{cloud}: no measured checkpoint save time; training islands stay on-demand"
    if measured_save_s + margin_s >= notice:
        return False, (f"{cloud}: notice {notice:g} s is not enough to save a checkpoint "
                       f"(measured {measured_save_s:g} s + margin {margin_s:g} s)")
    if not capabilities.durable_store(cloud):
        return False, f"{cloud}: no durable store"
    return True, "ok"
