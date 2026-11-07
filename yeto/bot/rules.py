"""Alert rules over one ``/api/view`` snapshot.

*Conditions* are states that hold while true (an island is stale, HBM is at the
limit); the monitor alerts on their transitions. *Events* are one-shot facts
discovered since the previous snapshot (new syncer log lines, a new eval point).

Severities: 0 = P0 (page: red, @owner, re-notify until acked), 1 = P1 (orange,
cooldown per subject), 2 = P2 (grey, informational).

Data the dashboard does not expose yet is wired as hooks that fire as soon as
the view carries it: ``disk`` samples (``[ts, free_bytes, total_bytes]`` under
``sft.syncer.disk`` or an island's ``disk``), syncer log categories ``oom`` /
``checkpoint_failed`` / ``island_checkpoint``, and per-island merge weights.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from .config import Thresholds
from .render import (cell, column, eval_refs, fbytes, fdur, fsig, last_round_ts, latest_rss, nonfinite, num,
                     ref_cmp, refs, round_intervals, step_cols)

P0, P1, P2 = 0, 1, 2
SEV_NAME = {P0: "P0", P1: "P1", P2: "P2"}

# syncer log / tape event categories -> (severity, label)
EVENT_RULES: dict[str, tuple[int, str]] = {
    "oom": (P0, "out of memory"),
    "checkpoint_failed": (P0, "checkpoint write failed"),
    "rejected": (P1, "pushes rejected"),
    "timeout": (P1, "quorum timeouts"),
    "not_carried": (P1, "pushes not carried"),
    "error": (P1, "syncer ERROR lines"),
    "pool_leave": (P1, "island left the pool"),
    "checkpoint": (P2, "syncer checkpoint written"),
    "island_checkpoint": (P2, "island checkpoint saved"),
    "joined": (P2, "learner (re)joined"),
    "pool_join": (P2, "island joined the pool"),
}


@dataclass
class Condition:
    sev: int
    title: str
    lines: list[str] = field(default_factory=list)
    island: str | None = None
    resolve: bool = True  # announce when it clears


@dataclass
class Event:
    sev: int
    key: tuple
    title: str
    lines: list[str] = field(default_factory=list)
    island: str | None = None
    count: int = 1


def island_name(isl: dict) -> str:
    return isl.get("name") or f"island {isl.get('id')}"


def hbm_limit(view: dict, th: Thresholds) -> float | None:
    return th.hbm_limit_gib if th.hbm_limit_gib is not None else num(refs(view).get("sft_hbm_limit_gib"))


def hbm_warn(view: dict, th: Thresholds) -> float | None:
    """Explicit ``hbm_warn_gib``, else ``hbm_warn_frac`` of the HBM limit; None without a limit."""
    if th.hbm_warn_gib is not None:
        return th.hbm_warn_gib
    limit = hbm_limit(view, th)
    return None if limit is None else th.hbm_warn_frac * limit


def rss_warn(view: dict, th: Thresholds) -> float | None:
    return th.rss_warn_bytes if th.rss_warn_bytes is not None else num(refs(view).get("sft_syncer_rss_limit_bytes"))


def stuck_limit(sft: dict, th: Thresholds) -> float | None:
    """Explicit ``stuck_round_s``, else the median recent round interval + 2 x grace."""
    if th.stuck_round_s is not None:
        return th.stuck_round_s
    gaps = round_intervals(sft)
    if len(gaps) >= 3:
        return statistics.median(gaps) + 2 * th.grace_s
    return None


def missed_streaks(rounds: list[dict]) -> dict[str, int]:
    """Island -> rounds missed in a row, counted back from the newest round."""
    streak: dict[str, int] = {}
    for r in rounds:
        missed = {str(x) for x in r.get("missed") or []}
        for iid in list(streak):
            if iid not in missed:
                streak.pop(iid)
        for iid in missed:
            streak[iid] = streak.get(iid, 0) + 1
    return streak


def conditions(view: dict, th: Thresholds) -> dict[tuple, Condition]:
    out: dict[tuple, Condition] = {}
    sft = (view or {}).get("sft") or {}
    hbm, warn = hbm_limit(view, th), hbm_warn(view, th)
    ix = step_cols(sft)
    names = {}
    for isl in sft.get("islands") or []:
        names[str(isl.get("id"))] = island_name(isl)
        _island(isl, ix, hbm, warn, th, out)
    if sft:
        _syncer(sft, rss_warn(view, th), th, out)
    for iid, n in missed_streaks((view or {}).get("rounds_syncer") or []).items():
        if n >= th.missed_rounds:
            out[(iid, "missed")] = Condition(P1, f"{names.get(iid, 'island ' + iid)} not merged for {n} rounds",
                                             [f"missed the last {n} syncer rounds in a row"], iid)
    if th.forward_dashboard_alerts:
        for a in ((view or {}).get("overview") or {}).get("alerts") or []:
            text = f"{a.get('title')} {a.get('detail')}".lower()
            if "out of memory" in text or " oom" in f" {text}":
                out[("oom", a.get("island"))] = Condition(P0, f"OOM: {a.get('title')}", [str(a.get("detail") or "")],
                                                          a.get("island"))
            elif a.get("sev") == 0:
                out[("dash", a.get("rule"), a.get("island"))] = Condition(
                    P0, f"dashboard: {a.get('title')}", [str(a.get("detail") or "")], a.get("island"))
    return out


def _island(isl: dict, ix: dict[str, int], hbm: float | None, hbm_warn_at: float | None, th: Thresholds,
            out: dict) -> None:
    """Island conditions; a rule whose column the view does not carry stays silent."""
    iid, name, rows = str(isl.get("id")), island_name(isl), isl.get("rows") or []
    c_step, c_loss, c_gn, c_lr, c_hbm = (ix.get(k) for k in ("step", "loss", "grad_norm", "lr", "hbm_gib"))
    st = isl.get("status")
    last_step = cell(rows[-1], c_step) if rows else None
    last_step = "-" if last_step is None else last_step
    if st == "done":
        out[(iid, "done")] = Condition(P2, f"{name} finished", [f"step {last_step}/{isl.get('steps_total')}"], iid,
                                       resolve=False)
        return
    if st == "stale":
        out[(iid, "stale")] = Condition(P0, f"{name} stale (process dead or hung?)",
                                        [f"no new step for {fdur(isl.get('age_s'))} (last step {last_step})"], iid)
    # ms-swift logged the step (lr is set); a dashboard null loss on such a step is NaN
    logged = [r for r in rows if cell(r, c_lr) is not None] if c_loss is not None else []
    if logged:
        last = logged[-1]
        step, loss = cell(last, c_step), cell(last, c_loss)
        if loss is None or nonfinite(loss):
            out[(iid, "loss_nan")] = Condition(P0, f"{name} loss NaN/Inf",
                                               [f"step {step} logged a non-finite loss"
                                                f" (grad_norm {fsig(cell(last, c_gn))})"], iid)
        else:
            prev = column(logged[:-1], c_loss)[-th.loss_window:]
            if len(prev) >= th.loss_min_points:
                mean, sd = statistics.fmean(prev), statistics.pstdev(prev)
                limit = mean + th.loss_sigma * sd
                if num(loss) is not None and loss > limit and sd > 0:
                    out[(iid, "loss_spike")] = Condition(
                        P1, f"{name} loss spike", [f"step {step} loss {fsig(loss, 4)} > mean"
                                                   f" {fsig(mean, 4)} + {th.loss_sigma:g} sd ({fsig(sd, 2)}) of the"
                                                   f" previous {len(prev)} steps"], iid)
        gn = num(cell(last, c_gn))
        if gn is not None and gn > th.grad_norm_max:
            out[(iid, "grad_norm")] = Condition(P1, f"{name} grad_norm high",
                                                [f"step {step} grad_norm {fsig(gn)} > {th.grad_norm_max:g}"],
                                                iid)
    hbm_rows = [r for r in rows if num(cell(r, c_hbm)) is not None]
    if hbm_rows:
        step, peak = cell(hbm_rows[-1], c_step), hbm_rows[-1][c_hbm]
        if hbm is not None and peak >= hbm:
            out[(iid, "hbm")] = Condition(P0, f"{name} HBM peak at the limit",
                                          [f"step {step} peak {fsig(peak)} GiB >= {fsig(hbm)} GiB"], iid)
        elif hbm_warn_at is not None and peak > hbm_warn_at:
            out[(iid, "hbm_warn")] = Condition(P1, f"{name} HBM peak high",
                                               [f"step {step} peak {fsig(peak)} GiB > {fsig(hbm_warn_at)} GiB"
                                                + (f" (limit {fsig(hbm)})" if hbm is not None else "")], iid)
    for node, ser in (isl.get("host") or {}).items():
        if ser and num(ser[-1][1]) is not None and ser[-1][1] < th.host_avail_min_bytes:
            out[(iid, "host", node)] = Condition(P1, f"{name} host {node} memory low",
                                                 [f"available {fbytes(ser[-1][1])} < {fbytes(th.host_avail_min_bytes)}"
                                                  + (f" of {fbytes(ser[-1][2])}" if len(ser[-1]) > 2 else "")], iid)
    _disk(isl.get("disk"), (iid, "disk"), name, th, out, iid)


def _disk(samples, key: tuple, who: str, th: Thresholds, out: dict, island: str | None = None) -> None:
    """Hook: fires once the dashboard exposes ``[ts, free_bytes, total_bytes]`` samples."""
    if not samples:
        return
    last = samples[-1]
    free, total = (num(last[1]), num(last[2])) if len(last) > 2 else (None, None)
    if free is not None and total and free / total < th.disk_free_min_frac:
        out[key] = Condition(P0, f"{who} disk almost full",
                             [f"free {fbytes(free)} of {fbytes(total)} ({free / total:.1%})"], island)


def _syncer(sft: dict, rss_limit: float | None, th: Thresholds, out: dict) -> None:
    syn = sft.get("syncer") or {}
    cur, total = syn.get("outer_step"), syn.get("total_steps")
    finished = isinstance(cur, int) and isinstance(total, int) and cur >= total
    lrt, now = last_round_ts(sft), num(sft.get("now"))
    limit = stuck_limit(sft, th)
    if not finished and lrt is not None and now is not None and limit is not None and now - lrt > limit:
        out[("syncer", "stuck")] = Condition(P0, "syncer outer step not advancing",
                                             [f"outer step {cur}/{total or '?'} unchanged for {fdur(now - lrt)}"
                                              f" (limit {fdur(limit)} = median round + 2 x grace)"])
    rss = latest_rss(sft)
    if rss is not None and rss_limit and rss > rss_limit:
        out[("syncer", "rss")] = Condition(P1, "syncer RSS high", [f"RSS {fbytes(rss)} > {fbytes(rss_limit)}"])
    _disk(syn.get("disk"), ("syncer", "disk"), "syncer host", th, out)


class EventTracker:
    """High-water marks for one-shot events (syncer events by timestamp, eval by step)."""

    def __init__(self) -> None:
        self.event_hw: float | None = None
        self.eval_step: int | None = None

    def events(self, view: dict, *, seed: bool = False) -> list[Event]:
        sft = (view or {}).get("sft") or {}
        out = self._syncer_events(sft, seed) + self._eval_events(sft, seed)
        return [] if seed else out

    def _syncer_events(self, sft: dict, seed: bool) -> list[Event]:
        evs = [e for e in (sft.get("syncer") or {}).get("events") or []
               if len(e) > 1 and e[1] in EVENT_RULES and num(e[0]) is not None]
        newest = max((e[0] for e in evs), default=None)
        hw = self.event_hw
        if newest is not None and (hw is None or newest > hw):
            self.event_hw = newest
        if seed or hw is None and not evs:
            return []
        fresh = [e for e in evs if hw is None or e[0] > hw]
        by_cat: dict[str, list] = {}
        for e in fresh:
            by_cat.setdefault(e[1], []).append(e)
        out = []
        for cat, items in by_cat.items():
            sev, label = EVENT_RULES[cat]
            who: dict = {}
            for e in items:
                lid = e[2] if len(e) > 2 and e[2] is not None else None
                if lid is not None:
                    who[lid] = who.get(lid, 0) + 1
            lines = [", ".join(f"learner {k} x{v}" for k, v in sorted(who.items(), key=lambda kv: str(kv[0])))] \
                if who else []
            if cat == "checkpoint":
                secs = [num(e[4]) for e in items if len(e) > 4 and num(e[4]) is not None]
                if secs:
                    lines.append(f"write took {fdur(secs[-1])}")
            out.append(Event(sev, ("syncer", cat), label, lines, count=len(items)))
        return out

    def _eval_events(self, sft: dict, seed: bool) -> list[Event]:
        rows = (sft.get("eval") or {}).get("rows") or []
        steps = [r["step"] for r in rows if isinstance(r.get("step"), int)]
        if not steps:
            return []
        newest, prev_seen = max(steps), self.eval_step
        if prev_seen is not None and newest <= prev_seen:
            return []
        self.eval_step = newest
        if seed:
            return []
        row = next(r for r in rows if r.get("step") == newest)
        older = [r for r in rows if isinstance(r.get("step"), int) and r["step"] < newest]
        prev = max(older, key=lambda r: r["step"]) if older else None
        # "worse" compares the loss with the previous eval point, not with any named reference
        d_prev = num(ref_cmp(row, "prev").get("dloss"))
        if d_prev is None and prev is not None and None not in (num(row.get("loss")), num(prev.get("loss"))):
            d_prev = row["loss"] - prev["loss"]
        lines = [f"loss {fsig(row.get('loss'), 4)}" + (f" | vs previous point {d_prev:+.4f}" if d_prev is not None
                                                        else "")]
        for name in eval_refs([row]):
            d = num(ref_cmp(row, name).get("dloss"))
            if d is not None and name != "prev":
                lines.append(f"vs {name} {d:+.4f}")
        if d_prev is not None and d_prev > 0:
            return [Event(P1, ("eval", "worse"), f"eval step {newest} worse than step {prev['step'] if prev else '?'}",
                          lines)]
        return [Event(P2, ("eval", "new"), f"new eval point: step {newest}", lines)]
