"""Alert policy: transitions, severity routing, cooldown, re-notify, ack, mute,
quiet hours, and merging everything one poll produced into a single card.

* P0 conditions notify on appearance and then every ``p0_renotify_s`` while
  they hold, until acked; they also pass quiet hours and a plain ``/mute``.
* P1 conditions notify on appearance, at most once per ``p1_cooldown_s`` per
  subject; P1 events held back by the cooldown accumulate and go out together.
* P2 notifies on appearance, never pages.
* A P1 / P2 appearance held back by a mute, quiet hours, an ack or the
  cooldown is not lost: it goes out at the first poll after the hold ends,
  if the condition is still active (its suppression is recorded once).
* ``/ack`` mutes the subjects of one card (or all active ones) for ``ack_mute_s``.
* The first snapshot only seeds: P0 conditions already true are re-notified
  after ``p0_renotify_s`` unless acked; nothing else from before the start is
  announced (the startup progress card shows it).
"""

from __future__ import annotations

import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Callable

from .config import Policy, Thresholds
from .render import Message, b, fdur
from .rules import P0, P1, P2, SEV_NAME, Condition, Event, EventTracker, conditions

SEV_ICON = {P0: "\U0001f534", P1: "\U0001f7e0", P2: "⚪"}
SEV_LEVEL = {P0: "p0", P1: "p1", P2: "p2"}
HISTORY_MAX = 5000
CARDS_MAX = 500


@dataclass
class Notice:
    sev: int
    key: tuple
    title: str
    lines: list[str] = field(default_factory=list)
    island: str | None = None
    kind: str = "fire"  # fire | renotify | resolve | event
    count: int = 1
    since: float | None = None
    held: bool = False  # an earlier attempt was already recorded as suppressed


def key_str(key: tuple) -> str:
    return "/".join(str(k) for k in key)


class Monitor:
    def __init__(self, th: Thresholds, policy: Policy, *, clock: Callable[[], float] = time.time):
        self.th = th
        self.policy = policy
        self.clock = clock
        self.seeded = False
        self.active: dict[tuple, Condition] = {}
        self.since: dict[tuple, float] = {}
        self.notified: dict[tuple, float] = {}  # last delivered notice per subject (survives flaps)
        self.alerted: set[tuple] = set()  # active subjects whose appearance was delivered
        # active P1 / P2 subjects whose appearance is not delivered yet -> suppression recorded
        self.unsent: dict[tuple, bool] = {}
        self.acked: dict[tuple, tuple[float, str]] = {}  # subject -> (until, by)
        self.pending: dict[tuple, Event] = {}  # P1 events held back by the cooldown
        self.tracker = EventTracker()
        self.mute_until = 0.0
        self.mute_all = False
        self.fail_streak = 0
        self.dashboard_down = False
        self.history: deque = deque(maxlen=HISTORY_MAX)  # (ts, sev, key, title, kind, delivered)
        self.cards: OrderedDict[int, list[Notice]] = OrderedDict()
        self._card_seq = 0

    # -- suppression ------------------------------------------------------------
    def quiet(self, now: float) -> bool:
        qh = self.policy.quiet_hours
        if not qh:
            return False
        t = time.localtime(now)
        m = t.tm_hour * 60 + t.tm_min
        a, z = qh
        return a <= m < z if a < z else (m >= a or m < z)

    def muted(self, sev: int, now: float) -> bool:
        return now < self.mute_until and (sev > P0 or self.mute_all)

    def acked_now(self, key: tuple, now: float) -> bool:
        ack = self.acked.get(key)
        return ack is not None and now < ack[0]

    def deliverable(self, n: Notice, now: float) -> bool:
        if n.kind in ("fire", "renotify", "event") and self.acked_now(n.key, now):
            return False
        if self.muted(n.sev, now):
            return False
        return not (n.sev > P0 and n.kind != "resolve" and self.quiet(now))

    # -- one poll -----------------------------------------------------------------
    def evaluate(self, view: dict) -> list[Notice]:
        now = self.clock()
        out: list[Notice] = []
        if self.dashboard_down:
            out.append(Notice(P1, ("dashboard", "down"), "dashboard reachable again",
                              [f"after {self.fail_streak} failed polls"], kind="resolve"))
        self.fail_streak, self.dashboard_down = 0, False
        cur = conditions(view, self.th)
        if not self.seeded:
            for key, c in cur.items():
                self.since[key] = now
                if c.sev == P0:  # re-notified after p0_renotify_s unless acked
                    self.notified[key] = now
                    self.alerted.add(key)
            self.active = cur
            self.tracker.events(view, seed=True)
            self.seeded = True
            return []
        candidates: list[Notice] = []
        for key, c in cur.items():
            if key not in self.active:
                self.since[key] = now
                if c.sev != P0:
                    self.unsent[key] = False
            last = self.notified.get(key, float("-inf"))
            if c.sev == P0:
                due = now - last >= self.policy.p0_renotify_s
                if due:
                    kind = "renotify" if key in self.alerted else "fire"
                    candidates.append(Notice(c.sev, key, c.title, c.lines, c.island, kind, since=self.since[key]))
            elif key in self.unsent and (c.sev == P2 or now - last >= self.policy.p1_cooldown_s):
                since = self.since[key]
                lines = c.lines + ([f"active for {fdur(now - since)}, held back until now (mute, quiet hours,"
                                    f" ack or cooldown)"] if now > since else [])
                candidates.append(Notice(c.sev, key, c.title, lines, c.island, "fire", since=since,
                                         held=self.unsent[key]))
        for key, c in self.active.items():
            if key not in cur:
                self.since.pop(key, None)
                self.acked.pop(key, None)
                self.unsent.pop(key, None)
                if key in self.alerted:
                    self.alerted.discard(key)
                    if c.resolve:
                        candidates.append(Notice(c.sev, key, c.title, c.lines, c.island, "resolve"))
        self.active = cur
        for ev in self.tracker.events(view):
            candidates.extend(self._event(ev, now))
        for key in list(self.pending):  # cooled-down P1 events still waiting
            if now - self.notified.get(key, float("-inf")) >= self.policy.p1_cooldown_s:
                ev = self.pending.pop(key)
                candidates.append(Notice(ev.sev, key, ev.title, ev.lines, ev.island, "event", ev.count))
        delivered = []
        for n in out + candidates:
            ok = self.deliverable(n, now)
            if ok or not n.held:  # a held-back appearance retried every poll is recorded once
                self.history.append((now, n.sev, n.key, n.title, n.kind, ok))
            if not ok:
                if n.key in self.unsent and n.kind == "fire":
                    self.unsent[n.key] = True
                continue
            if n.kind != "resolve":
                self.notified[n.key] = now
            if n.kind in ("fire", "renotify"):
                self.alerted.add(n.key)
                self.unsent.pop(n.key, None)
            delivered.append(n)
        return delivered

    def _event(self, ev: Event, now: float) -> list[Notice]:
        if ev.sev == P1:
            held = self.pending.get(ev.key)
            if held is not None:
                ev = Event(ev.sev, ev.key, ev.title, held.lines + ev.lines, ev.island, held.count + ev.count)
            if now - self.notified.get(ev.key, float("-inf")) < self.policy.p1_cooldown_s:
                self.pending[ev.key] = ev
                return []
            self.pending.pop(ev.key, None)
        return [Notice(ev.sev, ev.key, ev.title, ev.lines, ev.island, "event", ev.count)]

    def fetch_failed(self, exc: BaseException) -> list[Notice]:
        self.fail_streak += 1
        if self.dashboard_down or self.fail_streak < self.th.dashboard_down_after:
            return []
        self.dashboard_down = True
        n = Notice(P1, ("dashboard", "down"), "dashboard unreachable",
                   [f"{self.fail_streak} failed polls in a row: {exc}"[:300]])
        now = self.clock()
        ok = self.deliverable(n, now)
        self.history.append((now, n.sev, n.key, n.title, n.kind, ok))
        return [n] if ok else []

    # -- operator actions -----------------------------------------------------------
    def ack(self, keys: list[tuple] | None, by: str) -> list[tuple]:
        """Mute subjects (active conditions or event subjects such as rejected pushes) for ``ack_mute_s``."""
        now = self.clock()
        keys = [k for k, c in self.active.items() if c.resolve] if keys is None else list(dict.fromkeys(keys))
        for k in keys:
            self.acked[k] = (now + self.policy.ack_mute_s, by)
            self.pending.pop(k, None)
        return keys

    def mute(self, seconds: float, *, include_p0: bool = False) -> None:
        self.mute_until = self.clock() + seconds
        self.mute_all = include_p0

    def unmute(self) -> None:
        self.mute_until, self.mute_all = 0.0, False

    # -- cards ------------------------------------------------------------------------
    def card(self, notices: list[Notice], *, link: str = "") -> Message | None:
        """All notices of one poll as one message; registers the card for /ack and /details."""
        if not notices:
            return None
        self._card_seq += 1
        cid = self._card_seq
        self.cards[cid] = list(notices)
        while len(self.cards) > CARDS_MAX:
            self.cards.popitem(last=False)
        firing = sorted((n for n in notices if n.kind != "resolve"), key=lambda n: n.sev)
        resolved = [n for n in notices if n.kind == "resolve"]
        now = self.clock()
        lines: list[str] = []
        for n in firing:
            head = f"{SEV_ICON[n.sev]} {b(SEV_NAME[n.sev])} {n.title}"
            if n.count > 1:
                head += f" (+{n.count})"
            if n.kind == "renotify":
                head += f" - still firing for {fdur(now - (n.since or now))}, not acked"
            lines.append(head)
            lines += [f"    {x}" for x in n.lines if x]
        for n in resolved:
            lines.append(f"✅ resolved: {n.title}")
        if firing:
            top = firing[0]
            level = SEV_LEVEL[top.sev]
            title = f"{SEV_NAME[top.sev]} {top.title}" + (f" (+{len(notices) - 1} more)" if len(notices) > 1 else "")
        else:
            level = "ok"
            title = f"resolved: {resolved[0].title}" + (f" (+{len(resolved) - 1} more)" if len(resolved) > 1 else "")
        buttons: list[tuple[str, str]] = []
        if any(n.sev <= P1 for n in firing):
            buttons.append(("Ack 2h", f"/ack c{cid}"))
        buttons.append(("Details", f"/details c{cid}"))
        if link:
            buttons.append(("Open dashboard", link))
        mention = any(n.sev == P0 and n.kind in ("fire", "renotify", "event") for n in firing)
        return Message(title, lines, level, buttons, mention=mention)

    def card_notices(self, ref: str) -> list[Notice] | None:
        ref = ref.strip().lower().lstrip("c")
        return self.cards.get(int(ref)) if ref.isdigit() else None

    # -- summaries --------------------------------------------------------------------
    def active_counts(self) -> dict[int, int]:
        out = {P0: 0, P1: 0, P2: 0}
        for c in self.active.values():
            if c.sev in out and c.resolve:
                out[c.sev] += 1
        return out

    def history_since(self, t0: float) -> list[tuple]:
        return [h for h in self.history if h[0] >= t0]

    def alerts_message(self, *, window_s: float = 86400.0) -> Message:
        now = self.clock()
        lines = []
        act = sorted(((k, c) for k, c in self.active.items() if c.resolve), key=lambda kc: kc[1].sev)
        lines.append(b(f"Active ({len(act)})"))
        for k, c in act:
            ack = self.acked.get(k)
            tag = f" | acked by {ack[1]} until {time.strftime('%H:%M', time.localtime(ack[0]))}" \
                if ack and now < ack[0] else ""
            age = fdur(now - self.since.get(k, now))
            lines.append(f"{SEV_ICON[c.sev]} {SEV_NAME[c.sev]} {c.title} | for {age}{tag}")
        if not act:
            lines.append("none")
        hist = self.history_since(now - window_s)
        counts = {s: sum(1 for h in hist if h[1] == s and h[4] != "resolve") for s in (P0, P1, P2)}
        lines += ["", b(f"Last {fdur(window_s)}") + f": P0 {counts[P0]} | P1 {counts[P1]} | P2 {counts[P2]}"
                  f" | suppressed {sum(1 for h in hist if not h[5])}"]
        for ts, sev, _key, title, kind, ok in list(hist)[-10:]:
            lines.append(f"{time.strftime('%m-%d %H:%M', time.localtime(ts))} {SEV_NAME.get(sev, '?')} {kind}"
                         f" {title}" + ("" if ok else " (suppressed)"))
        if now < self.mute_until:
            lines.append(f"muted until {time.strftime('%m-%d %H:%M', time.localtime(self.mute_until))}"
                         + (" (P0 too)" if self.mute_all else " (P0 still pages)"))
        if self.policy.quiet_hours:
            a, z = self.policy.quiet_hours
            lines.append(f"quiet hours {a // 60:02d}:{a % 60:02d}-{z // 60:02d}:{z % 60:02d} (P0 only)"
                         + (" - now quiet" if self.quiet(now) else ""))
        return Message("yeto alerts", lines, "report", [("Ack all", "/ack all"), ("Status", "/status")])
