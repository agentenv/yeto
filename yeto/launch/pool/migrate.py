"""Moving a running machine from one pool to another (capacity broker phase 2, off by default).

The broker (:mod:`.broker`, ``migrate: true``) may move a node an island no longer needs
to an island short of nodes, instead of releasing one machine and ordering another (a
K3.1 move loads one release, minutes; a new machine is a cold start, about 28 minutes).
:class:`PoolMover` is the broker's :class:`~.broker.LaneMover` over the run's
:class:`~.island_pool.IslandPool` objects. It only *asks*; each pool does its half on
its own tick, so no pool waits for another and no lock is held across a tick:

1. **reserve** (destination, :meth:`PoolMoves.expect_move_in`): a slot for the machine
   and a placeholder that counts toward the destination's target, so its fill step
   orders nothing for the node on its way. The target goes up by one.
2. **drain the old lane** (source, :meth:`PoolMoves.move_out`): the "drain" scale down
   the pool already has (the sampler claims no new task; it goes once its trials in
   flight are written, or after ``scale_drain_timeout_s``), then the shard fence (its
   claims go back to the ledger at once), then the sampler stops
   (``ops.stop_sampler``). Nothing is released: ``ops.drain`` is never called. The
   target goes down by one. The member ends ``gone`` with ``end_reason`` "moved to
   <pool>" and the machine is offered in the :class:`MoveBox`.
3. **load the release and start the new lane** (destination, on its tick): it claims
   the machine in the box (an exclusive create: exactly one of "taken" and the
   source's "release after the timeout" wins), takes it as a new member under a new
   shard of its own, asks ``ops.load_release`` when the ops have it, and starts the
   sampler on its normal start path. Spend is counted by the source up to the handover
   and by the destination from it.

What can go wrong, and what then happens:

* the source cannot drain or fence before the machine's lease ends (or the machine is
  lost): the move is cancelled, the machine goes as in any scale down; the
  destination's placeholder runs out after ``timeout_s`` and its fill step orders a
  machine (the behaviour without phase 2);
* nobody takes a handed-over machine within ``timeout_s`` (the destination stopped,
  the controller died): the source wins the box claim and releases it;
* a restart in between: both pools keep their half in their state document (key
  ``moves``, written only while there is one, so a pool that never moved a machine
  writes the same document as before) and the box lives in the run store
  (:data:`BOX_PREFIX`) on the store control plane.

The offer ledger sees one machine: the source's ``member_moved_out`` takes the ticket
out of its shard without counting an end, the destination's ``member_moved_in`` carries
its offer and its order / ready times without counting an order or a start
(:mod:`.offer_ledger`).

A pool's machine can move when its ops can stop a sampler without releasing the machine
(``stop_sampler``), the member is ready and plain (not paused, retiring, fencing,
replacing or replaced) and the pool has no static data slots. The destination keeps the
machine's slot when it has it free, else asks ``ops.move_slot(account, ticket, free)``
(providers whose slot is only a counter, e.g. sky); without either the move is refused.
National Compute accounts and Modal calls are never moved.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

MOVE_TIMEOUT_S = 1800.0  # a placeholder / a handed-over machine nobody took runs out after this
BOX_PREFIX = "fleet/moves/"
OUT, IN = "out", "in"
DRAINING_OUT, HANDED_OVER, TAKEN, RELEASING, RELEASED = "draining", "handed_over", "taken", "releasing", "released"


class MoveRefused(RuntimeError):
    """A pool refused its half of a move (the reason is the message); nothing changed."""


class MoveBusy(RuntimeError):
    """A pool's tick lock was held; the broker plans the move again next time."""


# -- the box the two pools hand a machine over through ------------------------------


class MoveBox:
    """Where a handed-over machine waits for its new pool: ``offer`` (the source writes
    the machine's record), ``get``, and ``claim`` (an exclusive create: the first claimer
    wins, the same claimer again is True). In memory by default; on ``objects`` (the run
    store's object API: ``read`` / ``write`` / ``create``) it outlives the controller."""

    def __init__(self, objects: Any = None, prefix: str = BOX_PREFIX) -> None:
        self.objects, self.prefix = objects, prefix
        self._offers: dict[str, dict[str, Any]] = {}
        self._claims: dict[str, str] = {}
        self._lock = threading.Lock()

    def _key(self, ticket: str, name: str) -> str:
        return f"{self.prefix}{ticket}/{name}.json"

    def offer(self, ticket: str, record: Mapping[str, Any]) -> None:
        if self.objects is not None:
            self.objects.write(self._key(ticket, "offer"), json.dumps(dict(record), default=str).encode())
            return
        with self._lock:
            self._offers[ticket] = dict(record)

    def get(self, ticket: str) -> dict[str, Any] | None:
        if self.objects is not None:
            data = self.objects.read(self._key(ticket, "offer"))
            return json.loads(data) if data else None
        with self._lock:
            got = self._offers.get(ticket)
            return dict(got) if got is not None else None

    def claim(self, ticket: str, by: str) -> bool:
        if self.objects is not None:
            if self.objects.create(self._key(ticket, "claim"), json.dumps({"by": by}).encode()):
                return True
            return self.claimed_by(ticket) == by
        with self._lock:
            return self._claims.setdefault(ticket, by) == by

    def claimed_by(self, ticket: str) -> str | None:
        if self.objects is not None:
            data = self.objects.read(self._key(ticket, "claim"))
            return str(json.loads(data).get("by")) if data else None
        with self._lock:
            return self._claims.get(ticket)


# -- a pool's half of a move (mixed into IslandPool) ---------------------------------


class PoolMoves:
    """The move steps on :class:`~.island_pool.IslandPool` (it holds ``moves`` and
    ``move_box``). ``moves[OUT][ticket]``: a machine of this pool on its way out;
    ``moves[IN][ticket]``: a placeholder for one on its way in."""

    moves: dict[str, dict[str, dict[str, Any]]]
    move_box: Any

    @property
    def move_name(self) -> str:
        return str(self.config.shard_base)  # type: ignore[attr-defined]

    # -- what may move ------------------------------------------------------------
    def move_blocker(self, m: Any) -> str | None:
        """Why ``m`` cannot move now (None: it can)."""
        from .island_pool import READY  # noqa: PLC0415

        why = self._move_pool_blocker()
        if why is not None:
            return why
        if getattr(self.ops, "stop_sampler", None) is None:  # type: ignore[attr-defined]
            return "the pool's ops cannot stop a sampler without releasing its machine"
        if self.config.data_slots is not None:  # type: ignore[attr-defined]
            return "the pool has static data slots"
        if m.state != READY or m.paused or m.retiring_since is not None or m.fence_pending is not None:
            return f"member {m.ticket} is {m.state}{' (paused)' if m.paused else ''}, not plainly ready"
        if m.replaces or m.replaced_by:
            return f"member {m.ticket} is part of a replacement"
        if m.ticket in self.moves.get(OUT, {}):
            return f"member {m.ticket} is already moving"
        return None

    def _move_pool_blocker(self) -> str | None:
        for name in ("stopped", "finished", "halted", "detached"):
            if getattr(self, name, None):
                return f"the pool is {name} ({getattr(self, name)})"
        return None

    def _move_set_target(self, target: int) -> None:
        if self.scale_path is not None:  # type: ignore[attr-defined]
            self.record_target(target, source="broker-move")  # type: ignore[attr-defined]
        else:
            self.config.target = int(target)  # type: ignore[attr-defined]

    # -- the destination: reserve, take, start ------------------------------------
    def expect_move_in(self, ticket: str, *, source: str, account: Any, timeout_s: float = MOVE_TIMEOUT_S) -> Any:
        """Reserve a slot and a placeholder for machine ``ticket`` of pool ``source``; the
        target goes up by one. Returns the slot. :class:`MoveRefused` when the pool cannot
        take it (not running, no slot)."""
        why = self._move_pool_blocker()
        if why is not None:
            raise MoveRefused(why)
        if ticket in self.moves.get(IN, {}):
            return self.moves[IN][ticket]["account"]
        limit = self.config.shard_limit  # type: ignore[attr-defined]
        if self.control is None and limit is not None and self.next_shard >= limit:  # type: ignore[attr-defined]
            raise MoveRefused(f"no shard left under shard_limit {limit}")
        held = {m.account for m in self.live()} | {r["account"] for r in self.moves.get(IN, {}).values()}  # type: ignore[attr-defined]
        free = [a for a in self.config.accounts if a not in held]  # type: ignore[attr-defined]
        pick = getattr(self.ops, "move_slot", None)  # type: ignore[attr-defined]
        if account in free:
            slot = account
        elif pick is not None:
            slot = pick(account, ticket, free)
        else:
            slot = None
        if slot is None:
            raise MoveRefused(f"no free slot for {ticket} (slot {account!r}; free {free})")
        now = self.clock()  # type: ignore[attr-defined]
        self.moves.setdefault(IN, {})[ticket] = {"source": source, "account": slot, "since": now,
                                                 "until": now + float(timeout_s)}
        self._move_set_target(self.config.target + 1)  # type: ignore[attr-defined]
        self._event("move_in_expected", ticket=ticket, source=source, account=slot,  # type: ignore[attr-defined]
                    target=self.config.target, timeout_s=float(timeout_s))  # type: ignore[attr-defined]
        self._save_now()  # type: ignore[attr-defined]
        return slot

    def cancel_move_in(self, ticket: str, why: str) -> None:
        """Drop the placeholder of ``ticket`` and the target it added (a move refused at its source)."""
        if self.moves.get(IN, {}).pop(ticket, None) is None:
            return
        self._move_set_target(max(0, self.config.target - 1))  # type: ignore[attr-defined]
        self._event("move_in_cancelled", ticket=ticket, why=why, target=self.config.target)  # type: ignore[attr-defined]
        self._prune_moves()
        self._save_now()  # type: ignore[attr-defined]

    def moves_expected(self) -> int:
        """Placeholders the fill step counts as members (a machine on its way in)."""
        return len(self.moves.get(IN, {}))

    def _take_moves(self, now: float) -> None:
        box = None if self._move_pool_blocker() is not None else self.move_box  # a pool not running takes nothing
        for ticket, rec in list(self.moves.get(IN, {}).items()):
            offer = box.get(ticket) if box is not None else None
            if offer is not None and offer.get("to") == self.move_name and box.claim(ticket, self.move_name):
                self._move_in(ticket, rec, offer, now)
            elif now >= float(rec["until"]):
                del self.moves[IN][ticket]
                self._event("move_in_expired", ticket=ticket, source=rec.get("source"),  # type: ignore[attr-defined]
                            waited_s=round(now - float(rec["since"])), target=self.config.target)  # type: ignore[attr-defined]
                self._save_now()  # type: ignore[attr-defined]

    def _move_in(self, ticket: str, rec: Mapping[str, Any], offer: Mapping[str, Any], now: float) -> None:
        from .island_pool import Member  # noqa: PLC0415

        number = self._store_shard() if self.control is not None else None  # type: ignore[attr-defined]
        if isinstance(number, str):  # no shard number now (the allocator did not answer): next tick
            self.log(f"pool: machine {ticket} moving in waits for a shard number: {number}")  # type: ignore[attr-defined]
            return
        if number is not None:
            self.next_shard = number  # type: ignore[attr-defined]
        else:
            self._skip_ended_shards()  # type: ignore[attr-defined]
        try:
            data_slot = self._free_slot()  # type: ignore[attr-defined]
        except RuntimeError:
            data_slot = None
        at = float(offer.get("at") or now)
        shard = f"{self.config.shard_base}-{self.next_shard}"  # type: ignore[attr-defined]
        self.next_shard += 1  # type: ignore[attr-defined]
        self._reserved_shard = None  # type: ignore[attr-defined]
        member = Member(rec["account"], ticket, shard, created=float(offer.get("created") or at), data_slot=data_slot,
                        lease_until=offer.get("lease_until"), granted=at, charged_to=at,
                        hourly_ucr=offer.get("hourly_ucr"))
        self.members.append(member)  # type: ignore[attr-defined]
        del self.moves[IN][ticket]
        self._event("member_moved_in", ticket=ticket, account=member.account, shard=shard,  # type: ignore[attr-defined]
                    source=offer.get("from"), from_shard=offer.get("shard"), offer=offer.get("offer"),
                    ordered_at=offer.get("ordered_at"), ready_at=offer.get("ready_at"),
                    waited_s=round(now - float(rec["since"])))
        load = getattr(self.ops, "load_release", None)  # type: ignore[attr-defined]
        if load is not None:
            try:
                load(member)
            except Exception as error:  # noqa: BLE001 - the start pulls the release itself
                self._event("move_release_load_failed", ticket=ticket, shard=shard,  # type: ignore[attr-defined]
                            error=repr(error)[:300])
        self._save_now()  # type: ignore[attr-defined] - publishes the member table (store control plane)

    # -- the source: drain, fence, stop, hand over ---------------------------------
    def move_out(self, ticket: str, *, to: str, timeout_s: float = MOVE_TIMEOUT_S) -> None:
        """Start moving member ``ticket`` to pool ``to``: the target goes down by one and the
        member drains ("drain" scale down; fenced at once without ``request_drain``).
        :class:`MoveRefused` when it cannot move."""
        m = next((x for x in self.live() if x.ticket == ticket), None)  # type: ignore[attr-defined]
        if m is None:
            raise MoveRefused(f"no live member {ticket}")
        why = self.move_blocker(m)
        if why is not None:
            raise MoveRefused(why)
        now = self.clock()  # type: ignore[attr-defined]
        known = self._ledger_record(ticket)
        self.moves.setdefault(OUT, {})[ticket] = {
            "to": to, "state": DRAINING_OUT, "since": now, "timeout_s": float(timeout_s),
            "account": m.account, "shard": m.shard,
            "lease_until": m.lease_until, "hourly_ucr": m.hourly_ucr, "created": m.created,
            "offer": known.get("offer"), "ordered_at": known.get("ordered", m.created),
            "ready_at": known.get("ready", m.ready_since)}
        self._move_set_target(max(0, self.config.target - 1))  # type: ignore[attr-defined]
        self._event("move_out_started", ticket=ticket, account=m.account, shard=m.shard, to=to,  # type: ignore[attr-defined]
                    target=self.config.target)  # type: ignore[attr-defined]
        if not self._request_drain(m, now):  # type: ignore[attr-defined] - no drain: fenced at once
            self._finish_retire(m, now, outcome="moving (not draining)", alone=True)  # type: ignore[attr-defined]
        self._save_now()  # type: ignore[attr-defined]

    def _ledger_record(self, ticket: str) -> dict[str, Any]:
        ledger = getattr(self, "offer_ledger", None)
        own = getattr(ledger, "own", None)
        if own is None:
            return {}
        lock = getattr(ledger, "_lock", None)
        if lock is None:
            return dict(own.tickets.get(ticket) or {})
        with lock:
            return dict(own.tickets.get(ticket) or {})

    def moving_out(self, m: Any) -> bool:
        rec = self.moves.get(OUT, {}).get(m.ticket)
        return rec is not None and rec["state"] == DRAINING_OUT

    def _finish_move(self, m: Any, now: float, *, outcome: str, must_release: bool) -> None:
        """:meth:`_finish_retire` of a member moving out: fence, then hand over. Its lease
        over / its ticket ended (``must_release``): no move, it goes as a scale down."""
        from .island_pool import FENCING  # noqa: PLC0415

        m.retiring_since = None
        if must_release:
            self._cancel_move_out(m.ticket, why=outcome)
            self._finish_retire(m, now, outcome=outcome, alone=True, must_release=True)  # type: ignore[attr-defined]
            return
        if not self._fence(m, "move"):  # type: ignore[attr-defined]
            m.state, m.fence_then, m.fence_lost = FENCING, "move", False
            return
        self._hand_over(m, now, outcome=outcome)

    def _move_fence_retried(self, m: Any, now: float, *, over: bool) -> None:
        """:meth:`_retry_fences` of a member moving out whose fence failed before."""
        if self._fence(m, m.fence_pending):  # type: ignore[attr-defined]
            m.fence_then = None
            self._hand_over(m, now, outcome="fenced")
        elif over:  # its lease is over meanwhile (or the fence hold ran out): released, no move
            m.fence_then = None
            self._cancel_move_out(m.ticket, why="the fence did not succeed before the machine's end")
            self._drain(m, "move cancelled")  # type: ignore[attr-defined]

    def _hand_over(self, m: Any, now: float, *, outcome: str) -> None:
        from .island_pool import GONE  # noqa: PLC0415

        rec = self.moves[OUT][m.ticket]
        try:
            self.ops.stop_sampler(m)  # type: ignore[attr-defined]
        except Exception as error:  # noqa: BLE001 - fenced: its sampler takes and writes nothing more
            self._event("move_stop_failed", ticket=m.ticket, shard=m.shard, error=repr(error)[:300])  # type: ignore[attr-defined]
        self._charge(m, now)  # type: ignore[attr-defined]
        m.state, m.end_reason, m.drain_seq = GONE, f"moved to {rec['to']}", None
        rec.update(state=HANDED_OVER, at=now, lease_until=m.lease_until, hourly_ucr=m.hourly_ucr)
        record = {**rec, "ticket": m.ticket, "from": self.move_name}
        if self.move_box is not None:
            try:
                self.move_box.offer(m.ticket, record)
            except Exception as error:  # noqa: BLE001 - offered again next tick
                rec["unoffered"] = True
                self.log(f"pool: hand-over of {m.ticket} not written: {error!r}"[:300])  # type: ignore[attr-defined]
        else:
            rec["unoffered"] = True
        self._event("member_moved_out", ticket=m.ticket, account=m.account, shard=m.shard, to=rec["to"],  # type: ignore[attr-defined]
                    outcome=outcome, waited_s=round(now - float(rec["since"])), offer=rec.get("offer"),
                    target=self.config.target)  # type: ignore[attr-defined]
        self._save_now()  # type: ignore[attr-defined]

    def _cancel_move_out(self, ticket: str, *, why: str) -> None:
        if self.moves.get(OUT, {}).pop(ticket, None) is not None:
            self._event("move_out_cancelled", ticket=ticket, why=why)  # type: ignore[attr-defined]
            self._prune_moves()

    def _settle_moves(self, now: float, timeout_s: float | None = None) -> None:
        """Every tick: a move whose member went another way is cancelled; a handed-over
        machine is confirmed taken, or released once nobody took it within the move's
        timeout (``timeout_s`` overrides it: 0 at a stop); placeholders take their machine
        or run out."""
        from .island_pool import DRAINING, FENCING, GONE, Member  # noqa: PLC0415

        by_ticket = {m.ticket: m for m in self.members}  # type: ignore[attr-defined]
        box = self.move_box
        for ticket, rec in list(self.moves.get(OUT, {}).items()):
            state = rec["state"]
            if state == DRAINING_OUT:
                m = by_ticket.get(ticket)
                if m is None or m.state in (GONE, DRAINING) or (m.retiring_since is None and m.state != FENCING):
                    self._cancel_move_out(ticket, why="the member went another way" if m is None or m.state in
                                          (GONE, DRAINING) else "its drain was cancelled")
                continue
            if state == HANDED_OVER:
                if rec.get("unoffered") and box is not None:
                    try:
                        box.offer(ticket, {**rec, "ticket": ticket, "from": self.move_name})
                        rec.pop("unoffered", None)
                    except Exception:  # noqa: BLE001 - offered again next tick
                        pass
                taker = box.claimed_by(ticket) if box is not None else None
                if taker == rec["to"]:
                    rec["state"] = TAKEN
                    self._event("move_out_taken", ticket=ticket, to=rec["to"])  # type: ignore[attr-defined]
                    self._save_now()  # type: ignore[attr-defined]
                wait = float(rec.get("timeout_s", MOVE_TIMEOUT_S)) if timeout_s is None else timeout_s
                stopping = timeout_s is not None
                if taker != rec["to"] and now - float(rec["at"]) >= wait:
                    if box is None and not stopping:
                        # no box (restored without a mover): its new pool may hold it, so it is
                        # not released here; the pool's stop releases it (alerted once)
                        if not rec.get("stranded"):
                            rec["stranded"] = True
                            self._event("move_out_stranded", level="alert", ticket=ticket,  # type: ignore[attr-defined]
                                        to=rec["to"], waited_s=round(now - float(rec["at"])))
                    elif box is None or box.claim(ticket, self.move_name):
                        rec["state"] = RELEASING
                        self._event("move_out_abandoned", ticket=ticket, to=rec["to"],  # type: ignore[attr-defined]
                                    waited_s=round(now - float(rec["at"])))
                state = rec["state"]
            if state == RELEASING:
                stray = Member(rec["account"], ticket, shard="", state=DRAINING, created=now)
                try:
                    self.ops.drain(stray)  # type: ignore[attr-defined]
                except Exception as error:  # noqa: BLE001 - tried again next tick
                    self._event("drain_failed", ticket=ticket, account=rec["account"], why="move abandoned",  # type: ignore[attr-defined]
                                error=repr(error)[:300])
                    continue
                rec["state"] = RELEASED
                self._event("released", ticket=ticket, account=rec["account"], why="move abandoned")  # type: ignore[attr-defined]
                self._save_now()  # type: ignore[attr-defined]
        self._take_moves(now)

    def _stop_moves(self) -> None:
        """The pool stops: placeholders go; a machine handed over and not taken yet is
        released now unless its new pool claimed it (a draining one was released by the stop)."""
        self.moves.pop(IN, None)
        now = self.clock()  # type: ignore[attr-defined]
        for ticket, rec in list(self.moves.get(OUT, {}).items()):
            if rec["state"] == DRAINING_OUT:
                self._cancel_move_out(ticket, why="the pool stopped")
        if self.moves.get(OUT):
            self._settle_moves(now, timeout_s=0.0)

    def moved_away(self) -> set[str]:
        """Tickets of this pool's moves out: its patrol neither adopts nor releases them."""
        return set(self.moves.get(OUT, {}))

    def forget_moves(self, listed: Iterable[str]) -> None:
        """Drop finished moves out whose machine the provider no longer lists (patrol)."""
        listed = set(listed)
        out = self.moves.get(OUT, {})
        for ticket in [t for t, r in out.items() if r["state"] in (TAKEN, RELEASED) and t not in listed]:
            del out[ticket]
        self._prune_moves()

    def _prune_moves(self) -> None:
        for key in [k for k, v in self.moves.items() if not v]:
            del self.moves[key]

    def moves_doc(self) -> dict[str, Any]:
        self._prune_moves()
        return {"moves": json.loads(json.dumps(self.moves, default=str))} if self.moves else {}

    def load_moves(self, doc: Mapping[str, Any]) -> None:
        moves = doc.get("moves")
        self.moves = {k: {str(t): dict(r) for t, r in v.items()} for k, v in moves.items()} \
            if isinstance(moves, Mapping) else {}


# -- the broker's mover ---------------------------------------------------------------


@dataclass
class PoolMover:
    """:class:`~.broker.LaneMover` over the run's pools (pool name, the broker offers'
    ``pool``, -> its :class:`~.island_pool.IslandPool` or the controller's ``PoolSlot``).

    :meth:`begin` (called by :func:`~.broker.run_migration` before the steps) reserves
    the machine at the destination and starts the source's drain, each under that pool's
    tick lock taken without waiting (in pool-name order; :class:`MoveBusy` when either is
    held: the broker plans the move again next time). The pools finish the move on their
    own ticks (module doc), so ``drain_lane`` / ``load_release`` / ``start_lane`` only
    confirm that each half is under way."""

    pools: Mapping[str, Any]
    box: MoveBox = field(default_factory=MoveBox)
    timeout_s: float = MOVE_TIMEOUT_S
    _moves: dict[str, tuple[str, str]] = field(default_factory=dict)  # ticket -> (source, destination)

    def __post_init__(self) -> None:
        """Hand every pool the box. A box in memory starts empty (a new controller): it is
        seeded from the pools' saved states, so a machine its new pool took before the
        restart is claimed by that pool (its old pool never releases it) and one handed over
        but not taken yet is offered again."""
        for name in self.pools:
            self.pool(name).move_box = self.box
        outs: dict[str, tuple[Any, dict[str, Any]]] = {}
        for name in self.pools:
            pool = self.pool(name)
            for ticket, rec in pool.moves.get(OUT, {}).items():
                if rec["state"] == HANDED_OVER:
                    outs[ticket] = (pool, rec)
        for name in self.pools:
            pool = self.pool(name)
            for m in pool.live():
                if m.ticket in outs and outs[m.ticket][0] is not pool:
                    self.box.claim(m.ticket, pool.move_name)
        for ticket, (pool, rec) in outs.items():
            if self.box.claimed_by(ticket) is None and self.box.get(ticket) is None:
                self.box.offer(ticket, {**rec, "ticket": ticket, "from": pool.move_name})

    def pool(self, name: str) -> Any:
        entry = self.pools[name]
        return getattr(entry, "pool", entry)

    def _lock(self, name: str) -> Any:
        return getattr(self.pools[name], "tick_lock", None)

    def begin(self, move: Any) -> None:
        src, dst = str(move.from_offer).split("/", 1)[0], str(move.to_offer).split("/", 1)[0]
        if src not in self.pools or dst not in self.pools or src == dst:
            raise MoveRefused(f"no pools {src!r} -> {dst!r} to move {move.ticket} between")
        held: list[Any] = []
        try:
            for name in sorted({src, dst}):
                lock = self._lock(name)
                if lock is None:
                    continue
                if not lock.acquire(blocking=False):
                    raise MoveBusy(f"pool {name} is in a tick")
                held.append(lock)
            source, target = self.pool(src), self.pool(dst)
            member = next((m for m in source.live() if m.ticket == move.ticket), None)
            if member is None:
                raise MoveRefused(f"pool {src} has no live member {move.ticket}")
            why = source.move_blocker(member)
            if why is not None:
                raise MoveRefused(why)
            target.expect_move_in(move.ticket, source=source.move_name, account=member.account,
                                  timeout_s=self.timeout_s)
            try:
                source.move_out(move.ticket, to=target.move_name, timeout_s=self.timeout_s)
            except Exception as error:
                target.cancel_move_in(move.ticket, why=f"the source refused: {error}"[:300])
                raise
            self._moves[move.ticket] = (src, dst)
        finally:
            for lock in reversed(held):
                lock.release()

    def drain_lane(self, ticket: str, island: str) -> None:  # noqa: ARG002 - the move names its pools
        src, _dst = self._of(ticket)
        if ticket not in self.pool(src).moves.get(OUT, {}):
            raise MoveRefused(f"pool {src} is not moving {ticket} out")

    def load_release(self, ticket: str, island: str) -> None:  # noqa: ARG002
        self._of(ticket)  # the destination loads it when it takes the machine (ops.load_release)

    def start_lane(self, ticket: str, island: str) -> None:  # noqa: ARG002
        _src, dst = self._of(ticket)
        if ticket not in self.pool(dst).moves.get(IN, {}):
            raise MoveRefused(f"pool {dst} holds no place for {ticket}")

    def _of(self, ticket: str) -> tuple[str, str]:
        if ticket not in self._moves:
            raise MoveRefused(f"no move of {ticket} begun")
        return self._moves[ticket]


def nodes_from_pools(pools: Mapping[str, Any], *, island_of: Mapping[str, str],
                     offer_of: Mapping[str, str | Callable[[Any], str]] | None = None) -> list[Any]:
    """The broker's :class:`~.broker.Node` list (``Snapshot.nodes``): every live member of
    every pool (pool name -> IslandPool or PoolSlot), its offer key ``<pool>/<offer>``
    (``offer_of[pool]``: the offer name, or a function of the member; default ``"default"``),
    the pool's island, its lease end and whether it may move now (:meth:`PoolMoves.move_blocker`)."""
    from .broker import Node  # noqa: PLC0415

    out = []
    for name, entry in sorted(pools.items()):
        pool = getattr(entry, "pool", entry)
        how = (offer_of or {}).get(name, "default")
        for m in pool.live():
            offer = how(m) if callable(how) else how
            out.append(Node(m.ticket, f"{name}/{offer}", island_of[name], lease_end=m.lease_until,
                            movable=pool.move_blocker(m) is None))
    return out
