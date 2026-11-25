"""Moving a running machine from one pool to another (capacity broker phase 2, off by default).

The broker (:mod:`.broker`, ``migrate: true``) may move a node an island no longer needs
to an island short of nodes, instead of releasing one machine and ordering another (a
K3.1 move loads one release, minutes; a new machine is a cold start, about 28 minutes).
:class:`PoolMover` is the broker's :class:`~.broker.LaneMover` over the controller's
pools. It only *asks*; each pool does its half on its own tick, so no pool waits for
another and no lock is held across a tick:

1. **reserve** (destination, :meth:`PoolMoves.expect_move_in`): a slot for the machine
   (``ops.take_foreign``: only a provider that can read and release a ticket it did not
   order is a destination) and a placeholder that counts toward the destination's
   target, so its fill step orders nothing for the node on its way. Target + 1.
2. **drain the old lane** (source, :meth:`PoolMoves.move_out`): the "drain" scale down
   the pool already has (the sampler claims no new task; it goes once its trials in
   flight are written, or after ``scale_drain_timeout_s``), then the shard fence (its
   claims go back to the ledger at once), then ``ops.stop_sampler`` (retried; a sampler
   that does not stop in :data:`MOVE_STOP_TRIES` cancels the move and the machine is
   released). Nothing is released otherwise: ``ops.drain`` is never called. Target - 1.
   The member ends ``gone`` ("moved to <pool>"); once that is saved the machine is
   offered in the :class:`MoveBox`.
3. **take the machine, load the release, start the new lane** (destination, its tick):
   a shard number first, then the placeholder is saved as "taking", then the claim in
   the box (an exclusive create: exactly one of "taken" and the source's "release after
   the timeout" wins), then the member under the new shard; ``ops.load_release`` when
   the route has one; the sampler starts on the normal start path. Spend is counted by
   the source up to the handover and by the destination from it.

The box is the run store's (``fleet/moves/<move id>/``) on the store control plane, else
a ``moves/`` directory next to the local state files: it outlives any controller, and a
pool restored without a mover finds it again. A pool with neither (no state kept) never
moves a machine. Keys carry the move id (ticket and start time), so a machine moved back
later starts a fresh claim. Each move records where its box is: a pool that finds a
different box (a store-plane pool restored from its local mirror by ``pool-stop``) never
claims or releases on it; the move is reported unresolved (a critical alert, and the
machine is listed with the members a stop leaves billed).

What can go wrong, and what then happens:

* the source cannot drain, fence or stop before the machine's lease ends (or the machine
  is lost): the move is cancelled, the machine goes as in any scale down; the
  destination's placeholder runs out after ``timeout_s`` and its fill step orders a
  machine (the behaviour without phase 2);
* nobody takes a handed-over machine within ``timeout_s`` (the destination stopped, its
  controller died): the source wins the box claim and releases it (``move_out_abandoned``,
  an alert), at once when the source stops;
* a restart anywhere in between: both pools keep their half in their state document (key
  ``moves``, written only while there is one, so a pool that never moved a machine
  writes the same document as before) and resume from it.

The offer ledger sees one machine: the source's ``member_moved_out`` takes the ticket
out of its shard without counting an end, the destination's ``member_moved_in`` carries
its offer and its order / ready times without counting an order or a start
(:mod:`.offer_ledger`).

A machine can move when its pool's ops can stop a sampler without releasing the machine
(``stop_sampler``), the member is ready and plain (not paused, retiring, fencing,
replacing or replaced) and the pool has no static data slots. National Compute accounts
and Modal calls are never moved (their ops have no ``stop_sampler``).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

MOVE_TIMEOUT_S = 1800.0  # a placeholder / a handed-over machine nobody took runs out after this
MOVE_STOP_TRIES = 3  # stop_sampler attempts before the move is cancelled (the machine released)
UNRESOLVED_ALERT_S = 3600.0  # a move out its pool cannot settle (no box): alerted again this often
BOX_PREFIX = "fleet/moves/"
LOCAL_BOX_DIR = "moves"
OUT, IN = "out", "in"
# a move out: draining -> stopping (a stop failed, retried) -> handed_over -> taken | releasing -> released
DRAINING_OUT, STOPPING, HANDED_OVER, TAKEN, RELEASING, RELEASED = (
    "draining", "stopping", "handed_over", "taken", "releasing", "released")
OUT_RANK = {DRAINING_OUT: 0, STOPPING: 1, HANDED_OVER: 2, TAKEN: 3, RELEASING: 3, RELEASED: 4}
# a move in: waiting (the placeholder) -> taking (saved before the claim) -> a member
WAITING, TAKING = "waiting", "taking"


class MoveRefused(RuntimeError):
    """A pool refused its half of a move (the reason is the message); nothing changed."""


class MoveBusy(MoveRefused):
    """A pool's tick lock was held; the broker plans the move again next time."""


def move_id(ticket: str, at: float | None = None) -> str:
    return f"{ticket}.{int((time.time() if at is None else at) * 1000)}"


# -- the box the two pools hand a machine over through ------------------------------


class MoveBox:
    """Where a handed-over machine waits for its new pool, on an object API (``read`` /
    ``write`` / ``create``): ``offer`` (the source writes the machine's record), ``get``,
    and ``claim`` (an exclusive create: the first claimer wins, the same claimer again is
    True). A store whose create is not atomic is refused."""

    def __init__(self, objects: Any, prefix: str = BOX_PREFIX, *, where: str | None = None) -> None:
        from .store_lock import require_atomic_create  # noqa: PLC0415

        require_atomic_create(objects)
        self.objects, self.prefix = objects, prefix
        self.where = where or str(getattr(objects, "root", None) or getattr(objects, "uri", None) or id(objects))

    def _key(self, move: str, name: str) -> str:
        return f"{self.prefix}{move}/{name}.json"

    def offer(self, move: str, record: Mapping[str, Any]) -> None:
        self.objects.write(self._key(move, "offer"), json.dumps(dict(record), default=str).encode())

    def get(self, move: str) -> dict[str, Any] | None:
        data = self.objects.read(self._key(move, "offer"))
        return json.loads(data) if data else None

    def claim(self, move: str, by: str) -> bool:
        from .store_lock import require_atomic_create  # noqa: PLC0415

        won = bool(self.objects.create(self._key(move, "claim"), json.dumps({"by": by}).encode()))
        require_atomic_create(self.objects)  # a create that fell back to check-then-write: not trusted
        return won or self.claimed_by(move) == by

    def claimed_by(self, move: str) -> str | None:
        data = self.objects.read(self._key(move, "claim"))
        return str(json.loads(data).get("by")) if data else None


def box_of(pool: Any) -> MoveBox | None:
    """The pool's hand-over box: ``move_box`` when set, else the run store's on the store
    control plane, else ``moves/`` next to its local state file (None: the pool keeps no
    state, it never moves a machine)."""
    if getattr(pool, "move_box", None) is not None:
        return pool.move_box
    control = getattr(pool, "control", None)
    objects = getattr(getattr(control, "state", None), "objects", None)
    if objects is not None:
        pool.move_box = MoveBox(objects)
    elif getattr(pool, "state_path", None) is not None:
        from yeto.rl.engine.store_dispense import DirObjects  # noqa: PLC0415

        root = pool.state_path.parent / LOCAL_BOX_DIR
        pool.move_box = MoveBox(DirObjects(root), prefix="", where=str(root))
    return pool.move_box


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

    def _box(self) -> MoveBox | None:
        try:
            return box_of(self)
        except Exception as error:  # noqa: BLE001 - no box: no move (reported by the caller)
            self.log(f"pool: no hand-over box: {error!r}"[:300])  # type: ignore[attr-defined]
            return None

    def _rec_box(self, rec: Mapping[str, Any]) -> MoveBox | None:
        """The box of one move: the pool's, only when it is the one the move began on (every
        move records it; a record without it is not trusted either)."""
        box = self._box()
        if box is None or rec.get("box") != box.where:
            return None
        return box

    def _move_guard(self) -> None:
        """Before a box write: the pool's store lock (a fenced controller writes nothing)."""
        guard = getattr(self, "_guard_store", None)
        if self.control is not None and guard is not None:  # type: ignore[attr-defined]
            guard()

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
        if self.moves.get(OUT, {}).get(m.ticket, {}).get("state") in (DRAINING_OUT, STOPPING):
            return f"member {m.ticket} is already moving"
        return None

    def _move_pool_blocker(self) -> str | None:
        for name in ("stopped", "finished", "halted", "detached"):
            if getattr(self, name, None):
                return f"the pool is {name} ({getattr(self, name)})"
        if self._box() is None:
            return "the pool keeps no state, so it has no hand-over box"
        return None

    def _move_set_target(self, target: int) -> None:
        if self.scale_path is not None:  # type: ignore[attr-defined]
            self.record_target(target, source="broker-move")  # type: ignore[attr-defined]
        else:
            self.config.target = int(target)  # type: ignore[attr-defined]

    # -- the destination: reserve, take, start ------------------------------------
    def expect_move_in(self, ticket: str, *, move: str, source: str, account: Any,
                       timeout_s: float = MOVE_TIMEOUT_S, box: str | None = None) -> Any:
        """Reserve a slot and a placeholder for machine ``ticket`` of pool ``source`` (move
        ``move``); the target goes up by one. Returns the slot. :class:`MoveRefused` when
        the pool cannot take it (not running, its provider cannot manage a ticket it did
        not order, no slot, the placeholder not saved)."""
        why = self._move_pool_blocker()
        if why is not None:
            raise MoveRefused(why)
        rec = self.moves.get(IN, {}).get(ticket)
        if rec is not None and rec["move"] == move:
            return rec["account"]
        if rec is not None:
            raise MoveRefused(f"pool {self.move_name} already waits for {ticket} (move {rec['move']})")
        take = getattr(self.ops, "take_foreign", None)  # type: ignore[attr-defined]
        if take is None:
            raise MoveRefused(f"pool {self.move_name}'s provider cannot manage a machine it did not order")
        limit = self.config.shard_limit  # type: ignore[attr-defined]
        if self.control is None and limit is not None and self.next_shard >= limit:  # type: ignore[attr-defined]
            raise MoveRefused(f"no shard left under shard_limit {limit}")
        held = {m.account for m in self.live()} | {r["account"] for r in self.moves.get(IN, {}).values()}  # type: ignore[attr-defined]
        free = [a for a in self.config.accounts if a not in held]  # type: ignore[attr-defined]
        slot = take(account, ticket, free)
        if slot is None:
            raise MoveRefused(f"no free slot for {ticket} (slot {account!r}; free {free})")
        now = self.clock()  # type: ignore[attr-defined]
        self.moves.setdefault(IN, {})[ticket] = {"move": move, "state": WAITING, "source": source, "account": slot,
                                                 "since": now, "until": now + float(timeout_s),
                                                 "box": box or self._box().where}
        old_target = self.config.target  # type: ignore[attr-defined]
        self._move_set_target(old_target + 1)
        if not self._save_now():  # type: ignore[attr-defined]
            self.moves[IN].pop(ticket, None)
            self._prune_moves()
            self._move_set_target(old_target)
            raise MoveRefused(f"pool {self.move_name} could not save the placeholder")
        self._event("move_in_expected", ticket=ticket, move=move, source=source, account=slot,  # type: ignore[attr-defined]
                    target=self.config.target, timeout_s=float(timeout_s))  # type: ignore[attr-defined]
        return slot

    def cancel_move_in(self, ticket: str, why: str) -> None:
        """Drop the placeholder of ``ticket`` and the target it added (a move refused at its source)."""
        rec = self.moves.get(IN, {}).get(ticket)
        if rec is None or rec["state"] != WAITING:
            return
        del self.moves[IN][ticket]
        self._move_set_target(max(0, self.config.target - 1))  # type: ignore[attr-defined]
        self._event("move_in_cancelled", ticket=ticket, move=rec["move"], why=why,  # type: ignore[attr-defined]
                    target=self.config.target)  # type: ignore[attr-defined]
        self._prune_moves()
        self._save_now()  # type: ignore[attr-defined]

    def moves_expected(self) -> int:
        """Placeholders the fill step counts as members (a machine on its way in)."""
        return len(self.moves.get(IN, {}))

    def _take_moves(self, now: float, *, stopping: bool = False) -> None:
        """Each placeholder: complete a take this pool claimed (also when it no longer runs:
        the machine is then its member, released by its own rules), claim an offered
        machine (only while running), or run out."""
        running = self._move_pool_blocker() is None and not stopping
        for ticket, rec in list(self.moves.get(IN, {}).items()):
            move = rec["move"]
            box = self._rec_box(rec)
            if rec["state"] == TAKING and box is None:  # it may have claimed it: neither taken nor dropped blind
                self._unresolved(ticket, rec, now, "move_in_unresolved")
                continue
            if rec["state"] == TAKING:
                taker = box.claimed_by(move)
                if taker == self.move_name:
                    self._move_in(ticket, rec, now)
                    continue
                if taker is None and running:
                    self._move_guard()
                    if box.claim(move, self.move_name):
                        self._move_in(ticket, rec, now)
                        continue
                    taker = box.claimed_by(move)
                if taker is not None or not running:  # its old pool released it, or no take any more
                    self._drop_placeholder(ticket, rec, now, "move_in_lost" if taker else "move_in_dropped")
                continue
            offer = box.get(move) if box is not None and running else None
            if offer is not None and offer.get("to") == self.move_name:
                self._begin_take(ticket, rec, offer, box, now)
            elif now >= float(rec["until"]) or not running:
                self._drop_placeholder(ticket, rec, now, "move_in_expired" if running else "move_in_dropped")

    def _drop_placeholder(self, ticket: str, rec: Mapping[str, Any], now: float, event: str) -> None:
        self.moves[IN].pop(ticket, None)
        self._prune_moves()
        self._event(event, ticket=ticket, move=rec["move"], source=rec.get("source"),  # type: ignore[attr-defined]
                    waited_s=round(now - float(rec["since"])), target=self.config.target)  # type: ignore[attr-defined]
        self._save_now()  # type: ignore[attr-defined]

    def _begin_take(self, ticket: str, rec: dict[str, Any], offer: Mapping[str, Any], box: MoveBox,
                    now: float) -> None:
        """A shard number, then "taking" saved, then the claim: a claim is never won by a
        pool that has no record of it (a crash after the claim completes the take)."""
        if self.control is not None:  # type: ignore[attr-defined]
            number = self._store_shard()  # type: ignore[attr-defined]
            if isinstance(number, str):  # the allocator did not answer: next tick
                self.log(f"pool: machine {ticket} moving in waits for a shard number: {number}")  # type: ignore[attr-defined]
                return
        else:
            self._skip_ended_shards()  # type: ignore[attr-defined]
            number = self.next_shard  # type: ignore[attr-defined]
        rec.update(state=TAKING, number=int(number), offer=dict(offer))
        if not self._save_now():  # type: ignore[attr-defined]
            rec["state"] = WAITING  # claimed only once "taking" is saved
            rec.pop("number", None)
            rec.pop("offer", None)
            return
        self._move_guard()
        if box.claim(rec["move"], self.move_name):
            self._move_in(ticket, rec, now)
        else:  # its old pool released it after the timeout
            self._drop_placeholder(ticket, rec, now, "move_in_lost")

    def _move_in(self, ticket: str, rec: Mapping[str, Any], now: float) -> None:
        from .island_pool import Member  # noqa: PLC0415

        offer = rec["offer"]
        number = int(rec["number"])
        try:
            data_slot = self._free_slot()  # type: ignore[attr-defined]
        except RuntimeError:
            data_slot = None
        at = float(offer.get("at") or now)
        shard = f"{self.config.shard_base}-{number}"  # type: ignore[attr-defined]
        self.next_shard = max(self.next_shard, number + 1)  # type: ignore[attr-defined]
        if self._reserved_shard == number:  # type: ignore[attr-defined]
            self._reserved_shard = None  # type: ignore[attr-defined]
        member = Member(rec["account"], ticket, shard, created=float(offer.get("created") or at), data_slot=data_slot,
                        lease_until=offer.get("lease_until"), granted=at, charged_to=at,
                        hourly_ucr=offer.get("hourly_ucr"))
        self.members.append(member)  # type: ignore[attr-defined]
        del self.moves[IN][ticket]
        self._prune_moves()
        self._event("member_moved_in", ticket=ticket, move=rec["move"], account=member.account,  # type: ignore[attr-defined]
                    shard=shard, source=offer.get("from"), from_shard=offer.get("shard"), offer=offer.get("offer"),
                    ordered_at=offer.get("ordered_at"), ready_at=offer.get("ready_at"),
                    waited_s=round(now - float(rec["since"])))
        load = getattr(self.ops, "load_release", None)  # type: ignore[attr-defined]
        if load is not None and self._move_pool_blocker() is None:
            try:
                load(member)
            except Exception as error:  # noqa: BLE001 - the start pulls the release itself
                self._event("move_release_load_failed", ticket=ticket, shard=shard,  # type: ignore[attr-defined]
                            error=repr(error)[:300])
        self._save_now()  # type: ignore[attr-defined] - publishes the member table (store control plane)

    # -- the source: drain, fence, stop, hand over ---------------------------------
    def move_out(self, ticket: str, *, move: str, to: str, timeout_s: float = MOVE_TIMEOUT_S,
                 box: str | None = None) -> None:
        """Start moving member ``ticket`` to pool ``to`` (move ``move``): the target goes down
        by one and the member drains ("drain" scale down; fenced at once without
        ``request_drain``). :class:`MoveRefused` when it cannot move. A save that fails is
        safe: until a later save holds, a restart finds the member serving as before."""
        m = next((x for x in self.live() if x.ticket == ticket), None)  # type: ignore[attr-defined]
        if m is None:
            raise MoveRefused(f"no live member {ticket}")
        why = self.move_blocker(m)
        if why is not None:
            raise MoveRefused(why)
        now = self.clock()  # type: ignore[attr-defined]
        known = self._ledger_record(ticket)
        self.moves.setdefault(OUT, {})[ticket] = {
            "move": move, "to": to, "state": DRAINING_OUT, "since": now, "timeout_s": float(timeout_s),
            "account": m.account, "shard": m.shard, "lease_until": m.lease_until, "hourly_ucr": m.hourly_ucr,
            "created": m.created, "offer": known.get("offer"), "ordered_at": known.get("ordered", m.created),
            "ready_at": known.get("ready", m.ready_since), "stop_failures": 0, "box": box or self._box().where}
        self._move_set_target(max(0, self.config.target - 1))  # type: ignore[attr-defined]
        self._event("move_out_started", ticket=ticket, move=move, account=m.account, shard=m.shard,  # type: ignore[attr-defined]
                    to=to, target=self.config.target)  # type: ignore[attr-defined]
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
        """Fenced: stop the sampler (a failure: ``stopping``, tried again each tick), then the
        member is gone, saved, and only then offered in the box."""
        from .island_pool import FENCING, GONE  # noqa: PLC0415

        rec = self.moves[OUT][m.ticket]
        try:
            self.ops.stop_sampler(m)  # type: ignore[attr-defined]
        except Exception as error:  # noqa: BLE001 - the old engine may still hold the GPUs: not handed over
            rec["stop_failures"] = int(rec.get("stop_failures", 0)) + 1
            rec["state"] = STOPPING
            m.state = FENCING  # fenced and kept: not watched, not counted, nothing ordered for it
            self._event("move_stop_failed", ticket=m.ticket, shard=m.shard, failures=rec["stop_failures"],  # type: ignore[attr-defined]
                        error=repr(error)[:300])
            if rec["stop_failures"] >= MOVE_STOP_TRIES:
                self._cancel_move_out(m.ticket, why=f"its sampler did not stop in {MOVE_STOP_TRIES} tries")
                self._drain(m, "move cancelled")  # type: ignore[attr-defined]
            self._save_now()  # type: ignore[attr-defined]
            return
        self._charge(m, now)  # type: ignore[attr-defined]
        m.state, m.end_reason, m.drain_seq = GONE, f"moved to {rec['to']}", None
        rec.update(state=HANDED_OVER, at=now, lease_until=m.lease_until, hourly_ucr=m.hourly_ucr, unoffered=True)
        self._event("member_moved_out", ticket=m.ticket, move=rec["move"], account=m.account, shard=m.shard,  # type: ignore[attr-defined]
                    to=rec["to"], outcome=outcome, waited_s=round(now - float(rec["since"])), offer=rec.get("offer"),
                    target=self.config.target)  # type: ignore[attr-defined]
        self._offer_saved(m.ticket, rec)

    def _offer_saved(self, ticket: str, rec: dict[str, Any]) -> None:
        """Offer a handed-over machine once the pool's state holds it as gone (a restart never
        finds it serving here while another pool takes it)."""
        box = self._rec_box(rec)
        if box is None or not self._save_now():  # type: ignore[attr-defined]
            return
        try:
            self._move_guard()
            box.offer(rec["move"], {**rec, "ticket": ticket, "from": self.move_name})
        except Exception as error:  # noqa: BLE001 - offered again next tick
            self.log(f"pool: hand-over of {ticket} not written: {error!r}"[:300])  # type: ignore[attr-defined]
            return
        rec.pop("unoffered", None)
        self._save_now()  # type: ignore[attr-defined]

    def _cancel_move_out(self, ticket: str, *, why: str) -> None:
        rec = self.moves.get(OUT, {}).pop(ticket, None)
        if rec is not None:
            self._event("move_out_cancelled", ticket=ticket, move=rec.get("move"), why=why)  # type: ignore[attr-defined]
            self._prune_moves()

    def _settle_moves(self, now: float, *, stopping: bool = False) -> None:
        """Every tick: a move whose member went another way is cancelled; a sampler that did
        not stop is stopped again; a handed-over machine is confirmed taken, or released
        once nobody took it within the move's timeout (at once when ``stopping``);
        placeholders take their machine or run out."""
        from .island_pool import DRAINING, FENCING, GONE, Member  # noqa: PLC0415

        by_ticket = {m.ticket: m for m in self.members}  # type: ignore[attr-defined]
        for ticket, rec in list(self.moves.get(OUT, {}).items()):
            state = rec["state"]
            m = by_ticket.get(ticket)
            box = self._rec_box(rec)
            if state == DRAINING_OUT:
                if m is None or m.state in (GONE, DRAINING) or (m.retiring_since is None and m.state != FENCING):
                    gone = m is None or m.state in (GONE, DRAINING)
                    self._cancel_move_out(ticket, why="the member went another way" if gone else "its drain was cancelled")
                continue
            if state == STOPPING:
                if m is None or m.state in (GONE, DRAINING):
                    self._cancel_move_out(ticket, why="the member went another way")
                else:
                    self._hand_over(m, now, outcome="stopped late")
                continue
            if state == HANDED_OVER:
                if box is None:
                    self._unresolved(ticket, rec, now, "move_out_unresolved")
                    continue
                if rec.get("unoffered"):
                    self._offer_saved(ticket, rec)
                    if rec.get("unoffered"):
                        continue
                taker = box.claimed_by(rec["move"])
                if taker == rec["to"]:
                    rec["state"] = TAKEN
                    self._event("move_out_taken", ticket=ticket, move=rec["move"], to=rec["to"])  # type: ignore[attr-defined]
                    self._save_now()  # type: ignore[attr-defined]
                    continue
                if stopping or now - float(rec["at"]) >= float(rec.get("timeout_s", MOVE_TIMEOUT_S)):
                    self._move_guard()
                    if box.claim(rec["move"], self.move_name):
                        rec["state"] = RELEASING
                        self._event("move_out_abandoned", level="alert", ticket=ticket, move=rec["move"],  # type: ignore[attr-defined]
                                    to=rec["to"], waited_s=round(now - float(rec["at"])), stopping=stopping)
                        self._save_now()  # type: ignore[attr-defined]
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
        self._take_moves(now, stopping=stopping)

    def _unresolved(self, ticket: str, rec: dict[str, Any], now: float, event: str) -> None:
        """A move whose box the pool cannot open (none, or not the one the move began on): the
        machine is neither released nor forgotten (the other pool may hold it); a critical
        alert every UNRESOLVED_ALERT_S, and :meth:`moves_left` lists it."""
        last = rec.get("alerted_at")
        if last is None or now - float(last) >= UNRESOLVED_ALERT_S:
            rec["alerted_at"] = now
            self._event(event, level="critical", ticket=ticket, move=rec["move"],  # type: ignore[attr-defined]
                        other=rec.get("to") or rec.get("source"), box=rec.get("box"),
                        waited_s=round(now - float(rec.get("at") or rec["since"])),
                        action=f"find which pool holds {ticket}; release it there or by hand")

    def moves_left(self) -> list[Any]:
        """Machines of unsettled moves this pool may still be billed for (a hand-over nobody
        confirmed, a release that failed, a take whose box is gone), as members for the
        billing banner of a stop."""
        from .island_pool import DRAINING, Member  # noqa: PLC0415

        out = [(t, r, f"moving to {r['to']}, not settled ({r['state']})") for t, r in self.moves.get(OUT, {}).items()
               if r["state"] in (HANDED_OVER, RELEASING)]
        out += [(t, r, f"moving in from {r.get('source')}, not settled") for t, r in self.moves.get(IN, {}).items()
                if r["state"] == TAKING]
        return [Member(r["account"], t, str(r.get("shard") or ""), state=DRAINING, end_reason=why)
                for t, r, why in out]

    def _stop_moves(self) -> None:
        """The pool stops (before its members are drained): a take it claimed becomes a
        member (released with the others), placeholders go, a draining move is cancelled
        (its member drains with the others), and a machine handed over and not taken yet is
        released unless its new pool claimed it."""
        now = self.clock()  # type: ignore[attr-defined]
        for ticket, rec in list(self.moves.get(OUT, {}).items()):
            if rec["state"] in (DRAINING_OUT, STOPPING):
                self._cancel_move_out(ticket, why="the pool stopped")
        self._settle_moves(now, stopping=True)

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
        """``moves`` of a state document; a record that does not read as a move is dropped
        (logged), never a reason the pool cannot load."""
        moves = doc.get("moves")
        self.moves = {}
        if not isinstance(moves, Mapping):
            return
        for kind in (OUT, IN):
            for ticket, rec in (moves.get(kind) or {}).items() if isinstance(moves.get(kind), Mapping) else ():
                if isinstance(rec, Mapping) and isinstance(rec.get("move"), str) and isinstance(rec.get("state"), str) \
                        and "account" in rec and "since" in rec:
                    self.moves.setdefault(kind, {})[str(ticket)] = dict(rec)
                else:
                    self.log(f"pool: a {kind} move record of {ticket} does not read: dropped ({rec!r})"[:300])  # type: ignore[attr-defined]

    def merge_moves(self, held: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> None:
        """After a reload from the store (a worker restarted): ``held`` is what the memory
        held before it. A move out keeps whichever copy is further along; the placeholders
        are the memory's (newer), minus any whose machine is a member by now."""
        out = {t: dict(r) for t, r in self.moves.get(OUT, {}).items()}
        for t, r in held.get(OUT, {}).items():
            if t not in out or OUT_RANK.get(r["state"], 0) >= OUT_RANK.get(out[t]["state"], 0):
                out[t] = dict(r)
        members = {m.ticket for m in self.members if m.state != "gone"}  # type: ignore[attr-defined]
        moves_in = {t: dict(r) for t, r in held.get(IN, {}).items() if t not in members}
        self.moves = {k: v for k, v in ((OUT, out), (IN, moves_in)) if v}


# -- the broker's mover ---------------------------------------------------------------


@dataclass
class PoolMover:
    """:class:`~.broker.LaneMover` over the controller's pools (pool name, the broker
    offers' ``pool``, -> its ``PoolSlot``: the pool and its tick lock).

    :meth:`begin` (called by :func:`~.broker.run_migration` before the steps) reserves
    the machine at the destination and starts the source's drain, each under that pool's
    tick lock taken without waiting (in pool-name order; :class:`MoveBusy` when either is
    held, or a pool has no lock: the broker plans the move again next time). The pools
    finish the move on their own ticks (module doc), so ``drain_lane`` / ``load_release``
    / ``start_lane`` only confirm that each half is under way."""

    pools: Mapping[str, Any]
    timeout_s: float = MOVE_TIMEOUT_S
    clock: Callable[[], float] = time.time
    _moves: dict[str, tuple[str, str, str]] = field(default_factory=dict)  # ticket -> (move, source, destination)

    def pool(self, name: str) -> Any:
        entry = self.pools[name]
        return getattr(entry, "pool", entry)

    def begin(self, move: Any) -> None:
        from .store_lock import LockLost, LockUnconfirmed  # noqa: PLC0415

        src, dst = str(move.from_offer).split("/", 1)[0], str(move.to_offer).split("/", 1)[0]
        if src not in self.pools or dst not in self.pools or src == dst:
            raise MoveRefused(f"no pools {src!r} -> {dst!r} to move {move.ticket} between")
        held: list[Any] = []
        try:
            for name in sorted({src, dst}):
                lock = getattr(self.pools[name], "tick_lock", None)
                if lock is None:
                    raise MoveBusy(f"pool {name} has no tick lock: nothing is changed without it")
                if not lock.acquire(blocking=False):
                    raise MoveBusy(f"pool {name} is in a tick")
                held.append(lock)
            self._begin(move, src, dst)
        except (LockLost, LockUnconfirmed) as error:  # a pool lost its store lock: it detaches itself
            raise MoveRefused(f"store lock: {error}") from None
        finally:
            for lock in reversed(held):
                lock.release()

    def _begin(self, move: Any, src: str, dst: str) -> None:
        from .store_lock import LockLost, LockUnconfirmed  # noqa: PLC0415

        source, target = self.pool(src), self.pool(dst)
        member = next((m for m in source.live() if m.ticket == move.ticket), None)
        if member is None:
            raise MoveRefused(f"pool {src} has no live member {move.ticket}")
        why = source.move_blocker(member)
        if why is not None:
            raise MoveRefused(why)
        a, b = source._box(), target._box()  # noqa: SLF001
        if a is None or b is None or a.where != b.where:
            raise MoveRefused(f"pools {src} and {dst} do not share a hand-over box")
        mid = move_id(move.ticket, self.clock())
        target.expect_move_in(move.ticket, move=mid, source=source.move_name, account=member.account,
                              timeout_s=self.timeout_s, box=a.where)
        try:
            source.move_out(move.ticket, move=mid, to=target.move_name, timeout_s=self.timeout_s, box=a.where)
        except (Exception, LockLost, LockUnconfirmed) as error:
            target.cancel_move_in(move.ticket, why=f"the source refused: {error}"[:300])
            raise
        self._moves[move.ticket] = (mid, src, dst)

    def drain_lane(self, ticket: str, island: str) -> None:  # noqa: ARG002 - the move names its pools
        mid, src, _dst = self._of(ticket)
        if self.pool(src).moves.get(OUT, {}).get(ticket, {}).get("move") != mid:
            self._moves.pop(ticket, None)
            raise MoveRefused(f"pool {src} is not moving {ticket} out")

    def load_release(self, ticket: str, island: str) -> None:  # noqa: ARG002
        self._of(ticket)  # the destination loads it when it takes the machine (ops.load_release)

    def start_lane(self, ticket: str, island: str) -> None:  # noqa: ARG002
        mid, _src, dst = self._of(ticket)
        self._moves.pop(ticket, None)  # the last step: the pools finish the move on their own
        if self.pool(dst).moves.get(IN, {}).get(ticket, {}).get("move") != mid:
            raise MoveRefused(f"pool {dst} holds no place for {ticket}")

    def _of(self, ticket: str) -> tuple[str, str, str]:
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
