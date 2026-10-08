"""CPU fake-island harness for inter-island scheduling (rl-inter-island-scheduling, stage 0).

N fake islands run as ``multiprocessing`` processes (no Ray, no GPU, no model);
the coordinator runs in the calling process and owns a
:class:`~yeto.rl.engine.island_ledger.CrossIslandLedger`. It demonstrates:

* P4 stepping -- advance when the arrived capacity reaches ``theta`` of the
  total, or at the soft deadline T_soft; late deltas are carried over with a
  ``gamma**lag`` discount;
* heartbeat leases -- a crashed island (silent exit) is removed when its lease
  expires, its uncommitted delta is dropped and recorded;
* catch-up -- an island joining mid-run contributes weight 0 in its first round;
* cross-island sample groups judged ACCEPT / ACCEPT_IS / REJECT;
* pool_join / pool_leave / pool_epoch durably written to a reconfig journal.

Output: ``<out_dir>/tape.json`` (ledger events + harness timeline) and the
journal under ``<out_dir>/journal``. Timings are wall-clock; tests assert on
protocol outcomes, not on exact times.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .island_ledger import (CrossIslandLedger, DeltaEntry, IslandSchedulingMode, LedgerError,
                            SampleGroup, StalenessPolicy, parse_mode)
from .pause_advice import slow_island_advice
from .journal import Journal, append_pool_event, replay_pool


@dataclass(frozen=True)
class FakeIsland:
    island_id: str
    work_s: float = 0.05
    tokens_per_round: int = 1000
    steps_per_round: int = 4
    crash_after_rounds: int | None = None  # silent exit (no LEAVE) after N submits
    join_at_version: int = 0  # started once the coordinator reaches this version
    behavior_logprob: bool = True


def _island_main(spec: FakeIsland, inbox: Any, outbox: Any, hb_s: float) -> None:
    send = lambda *m: outbox.put((spec.island_id, time.time(), *m))  # noqa: E731
    send("join")
    done = 0
    while True:
        try:
            msg = inbox.get(timeout=hb_s)
        except queue.Empty:
            send("heartbeat")
            continue
        if msg[0] == "stop":
            return
        _, version, policy_hash = msg
        t_end = time.time() + spec.work_s
        while time.time() < t_end:  # "train", heartbeating meanwhile
            time.sleep(min(hb_s, max(0.0, t_end - time.time())))
            send("heartbeat")
        send("delta", {"island_id": spec.island_id, "outer_version": version,
                       "inner_step": spec.steps_per_round, "policy_hash": policy_hash,
                       "c_tokens": spec.tokens_per_round, "c_steps": spec.steps_per_round})
        send("samples", {"island_id": spec.island_id, "outer_version": version,
                         "inner_step": spec.steps_per_round, "policy_hash": policy_hash, "n": 8,
                         "behavior_logprob": [-0.5] * 4 if spec.behavior_logprob else None})
        done += 1
        if spec.crash_after_rounds is not None and done >= spec.crash_after_rounds:
            outbox.close()
            outbox.join_thread()  # flush the last delta so it is pending at the coordinator
            os._exit(0)  # crash: no LEAVE, no more heartbeats


def run_fake_islands(islands: Sequence[FakeIsland], out_dir: str | Path, *, rounds: int = 5,
                     theta: float = 1.0, quorum_min: int = 1, gamma: float = 0.5,
                     quorum_timeout_s: float = 0.8, lease_s: float = 0.5, hb_s: float = 0.05,
                     policy: StalenessPolicy = StalenessPolicy(),
                     deadline_s: float = 60.0,
                     mode: "str | IslandSchedulingMode | None" = None) -> dict[str, Any]:
    """``mode`` defaults to legacy (fixed members, all-arrive stepping); pass
    ``"elastic"`` for the inter-island scheduling semantics."""
    mode = parse_mode(mode)
    elastic = mode is IslandSchedulingMode.ELASTIC
    if not elastic and any(s.join_at_version > 0 for s in islands):
        raise ValueError("legacy mode has fixed members: join_at_version must be 0")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    outbox = ctx.Queue()
    ledger = CrossIslandLedger(policy=policy, theta=theta, quorum_min=quorum_min,
                               gamma=gamma, lease_s=lease_s, mode=mode)
    failure: str | None = None
    base_sent: dict[str, float] = {}
    round_wall: dict[str, float] = {}
    advised: set[str] = set()

    def pool(tx_kind: str, **f: Any) -> None:
        if elastic:  # legacy: pool_* records are not written
            append_pool_event(journal, tx_kind, mode=mode, **f)
    timeline: list[dict[str, Any]] = []
    procs: dict[str, Any] = {}
    inboxes: dict[str, Any] = {}
    pool_epoch = 0

    def log(kind: str, **f: Any) -> None:
        timeline.append({"t": time.time(), "kind": kind, "outer_version": ledger.outer_version, **f})

    def start(spec: FakeIsland) -> None:
        inboxes[spec.island_id] = ctx.Queue()
        p = ctx.Process(target=_island_main, args=(spec, inboxes[spec.island_id], outbox, hb_s),
                        daemon=True)
        p.start()
        procs[spec.island_id] = p
        log("island_started", island_id=spec.island_id)

    def broadcast_base(ids) -> None:
        for i in ids:
            base_sent.setdefault(i, time.time())
            inboxes[i].put(("base", ledger.outer_version, ledger.published[ledger.outer_version]))

    t0 = time.time()
    with Journal(out / "journal") as journal:
        pool("pool_epoch", pool_epoch=pool_epoch,
                          membership_epoch=ledger.membership_epoch)
        for spec in islands:
            if spec.join_at_version <= 0:
                start(spec)
        round_start = time.time()
        while (failure is None and ledger.outer_version < rounds
               and time.time() - t0 < deadline_s):
            for spec in islands:
                if spec.island_id not in procs and ledger.outer_version >= spec.join_at_version:
                    start(spec)
            try:
                island_id, ts, kind, *payload = outbox.get(timeout=hb_s)
            except queue.Empty:
                kind = None
            now = time.time()
            if kind == "join":
                ev = ledger.join(island_id, now=now)
                pool("pool_join", pool_epoch=pool_epoch, island_id=island_id,
                                  membership_epoch=ledger.membership_epoch,
                                  catch_up=ev["catch_up"], base_version=ev["base_version"])
                broadcast_base([island_id])
            elif kind == "heartbeat" and island_id in ledger.members:
                ledger.heartbeat(island_id, now=now)
            elif kind == "delta" and island_id in ledger.members:
                ledger.heartbeat(island_id, now=now)
                ledger.submit(DeltaEntry(**payload[0]))
                if island_id in base_sent:  # time from a base being sent to its delta
                    round_wall[island_id] = now - base_sent.pop(island_id)
            elif kind == "samples" and island_id in ledger.members:
                raw = dict(payload[0])
                lp = raw.pop("behavior_logprob")
                group = SampleGroup(**raw, behavior_logprob=None if lp is None else tuple(lp))
                for consumer in sorted(ledger.members):
                    if consumer != island_id:
                        ledger.judge(group, consumer_island=consumer)
            for gone in ledger.expire_leases(now=now):
                pool("pool_leave", pool_epoch=pool_epoch, island_id=gone,
                                  membership_epoch=ledger.membership_epoch, reason="lease_expired")
            timed_out = now - round_start > quorum_timeout_s
            try:
                step = ledger.try_advance(timed_out=timed_out)
            except LedgerError as exc:  # legacy strict timeout fails closed
                failure = str(exc)
                log("failed", reason=failure)
                break
            if step is not None:
                log("outer_step", **{k: step[k] for k in ("arrived", "cap_arrived", "cap_total", "timed_out", "absent", "carried_in")})
                round_start = time.time()
                if elastic:  # 0.11: suggestion only, recorded in the tape for a human
                    live = {i: w for i, w in round_wall.items() if i in ledger.members}
                    for adv in slow_island_advice(live, now=time.time()):
                        target = adv.target_resource_intent["island_id"]
                        if target not in advised:
                            advised.add(target)
                            log("pause_advice", source=adv.source, reason=adv.reason,
                                veto=adv.veto, expires_at=adv.expires_at,
                                target_resource_intent=dict(adv.target_resource_intent))
                broadcast_base(sorted(ledger.members))
            elif timed_out:
                round_start = time.time()  # idle round (below quorum_min): restart the window
        pool_epoch += 1
        pool("pool_epoch", pool_epoch=pool_epoch,
                          membership_epoch=ledger.membership_epoch, reason="run_end")
        replayed = replay_pool(journal.records, mode)
    for i, q in inboxes.items():
        q.put(("stop",))
    for p in procs.values():
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
    result = {
        "schema": "yeto.rl.fake-islands-tape/v1",
        "config": {"islands": [asdict(s) for s in islands], "rounds": rounds,
                   "theta": theta, "gamma": gamma, "quorum_min": quorum_min,
                   "quorum_timeout_s": quorum_timeout_s, "lease_s": lease_s, "hb_s": hb_s,
                   "policy": asdict(policy), "mode": mode.value},
        "failure": failure,
        "final": {"outer_version": ledger.outer_version, "membership_epoch": ledger.membership_epoch,
                  "members": sorted(ledger.members), "wall_s": time.time() - t0},
        "journal_replay": asdict(replayed),
        "ledger_events": ledger.events,
        "timeline": timeline,
    }
    (out / "tape.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    return result
