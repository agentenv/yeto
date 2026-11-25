"""The rollout controller's reconcile of the fleet's desired state (``desired``).

:class:`FleetReconciler` runs in the controller's supervisor pass. It reads the newest
generation of ``fleet/desired.json`` (at most every ``poll_s``) and, when it is newer
than the one applied, compares its sampler pools with the applied ones:

* a pool added: ``hooks.add(name, member)`` builds its slot (for ``yeto run up``: the
  pool config written like a ``samplers`` entry, on the store control plane) and the
  supervisor starts its worker. The trainer consumes its shards from the member table,
  without a restart;
* a target changed (or a one-shot ``retire`` list): ``hooks.set_target`` (``pool-scale``
  for a pool with a scale control file, else :meth:`IslandPool.set_target`);
* a pool reconfigured (anything of its entry but ``target`` changed): ``hooks.reconfigure``
  gives the pool its new config and ops; :meth:`IslandPool.start_roll` then replaces the
  members one at a time (new members start with the new config);
* a pool removed: its target goes to 0 and the scale-down drains its members; once none
  is left, every shard it ever assigned is ended for good (the allocator's end record and
  the member table's ``retired``: the trainer's data end never waits for them), the pool
  is stopped (its machines released, confirmed ended) and dropped (``hooks.removed``).

What was applied (the generation, its pools, the pools still being removed) is kept in
the records ``fleet/applied/<seq>.json`` (``DesiredStore.write_applied``). A controller that starts (after
a detach, a crash) builds the applied pools and the removals in progress and starts from
that generation, so every change made to the desired state meanwhile is applied: a pool
removed is drained and retired, a target change and a new config are applied.

The reconcile never waits for a pool's tick: what needs the pool's tick lock (a new config,
the end of a removal) is tried without blocking and again at the next pass. The end of a
removal runs on the pool's own worker (``PoolSlot.pending_finish``), so a provider call
that hangs there never holds the supervisor.

Trainer members (``kind: trainer``) are left to the trainer worker of the
``unified-trainer-islands`` change. A pool's own target changes between two
generations (a manual release it detected, an automatic scaler) are not undone: only a
change of the desired state is acted on.
"""

from __future__ import annotations

import os
import socket
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import desired as dz


@dataclass
class ReconcileHooks:
    """What the reconciler cannot do alone. ``add(name, member) -> PoolSlot``;
    ``set_target(name, slot, target, retire, mode, generation=g) -> None`` (None: the pool's
    own :meth:`set_target`); ``reconfigure(name, slot, member) -> None`` (gives the pool its new
    config and ops; called under the pool's tick lock; raises to refuse); ``removed(name) ->
    None`` (after the pool is gone)."""

    add: Callable[[str, dict[str, Any]], Any] | None = None
    set_target: Callable[..., None] | None = None
    reconfigure: Callable[[str, Any, dict[str, Any]], None] | None = None
    removed: Callable[[str], None] | None = None


@dataclass
class _Removal:
    since: float
    member: dict[str, Any] = field(default_factory=dict)  # its entry in the generation before the removal
    generation: int = 0


@dataclass
class FleetReconciler:
    store: dz.DesiredStore
    hooks: ReconcileHooks = field(default_factory=ReconcileHooks)
    poll_s: float = 10.0
    log: Callable[[str], None] = print
    clock: Callable[[], float] = time.monotonic
    applied: int = 0
    applied_pools: dict[str, dict[str, Any]] = field(default_factory=dict)
    removing: dict[str, _Removal] = field(default_factory=dict)
    pending_rolls: dict[str, tuple[dict[str, Any], int]] = field(default_factory=dict)  # tick lock was busy
    _polled_at: float | None = None
    _resume: list[str] = field(default_factory=list)  # restored removals: target 0 asked again
    _dirty: bool = False  # the applied state changed and is not in the store yet
    _claim: bool = True  # the first write claims the applied state for this controller
    _seq: int | None = None  # the applied record this controller wrote last
    writer: str = field(default_factory=lambda: f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}")
    fenced: str | None = None  # another controller took the fleet over: this one applies nothing more
    errors: int = 0

    def start_from(self, doc: dict[str, Any] | None, applied: dict[str, Any] | None = None) -> None:
        """Where the controller starts: what was applied before (``DesiredStore.read_applied``: its
        generation, pools and removals in progress), else the generation the controller was
        built from (``doc``: its pools are that generation's)."""
        if applied is not None:
            self.applied = int(applied.get("generation") or 0)
            self.applied_pools = dz.sampler_pools(applied)
            self.removing = {m["name"]: _Removal(since=self.clock(), member=m, generation=self.applied)
                             for m in applied.get("removing") or () if m.get("kind") == "sampler_pool"}
            self._resume = sorted(self.removing)
            self._dirty = True  # claimed at the first pass
        elif doc is not None:
            self.applied = int(doc.get("generation") or 0)
            self.applied_pools = dz.sampler_pools(doc)
            self._dirty = True

    def step(self, controller: Any, *, force: bool = False) -> None:
        """One reconcile pass: a newer generation is applied; removals in progress advance."""
        self._persist()
        if self.fenced is not None:
            return
        for name in self._resume:  # a removal another controller began: drained to 0 again
            self._set_target(controller, name, 0, (), None, self.removing[name].generation if name in self.removing
                             else self.applied)
        self._resume = []
        for name, (member, gen) in list(self.pending_rolls.items()):
            self._reconfigure(controller, name, member, gen)
        self._advance_removals(controller)
        now = self.clock()
        if not force and self._polled_at is not None and now - self._polled_at < self.poll_s:
            return
        self._polled_at = now
        try:
            doc = self.store.read()
        except Exception as error:  # noqa: BLE001 - the next pass reads again; nothing changes meanwhile
            self.errors += 1
            self.log(f"fleet: desired state not read ({error!r})")
            return
        if doc is None or int(doc["generation"]) <= self.applied:
            return
        if not self._still_ours():
            return
        self.apply(controller, doc)

    def _still_ours(self) -> bool:
        """Before acting on a new generation: no other controller claimed the fleet since."""
        if self._claim:
            return True
        try:
            current = self.store.read_applied()
        except Exception as error:  # noqa: BLE001 - not known: the next pass checks again
            self.log(f"fleet: applied state not read ({error!r}); the new generation waits")
            return False
        if current is not None and current.get("writer") not in (None, self.writer):
            self.fenced = f"the applied state is written by {current.get('writer')} now (not {self.writer})"
            self.log(f"fleet: CRITICAL: {self.fenced}: this controller applies the desired state no more")
            return False
        return True

    def apply(self, controller: Any, doc: dict[str, Any]) -> None:
        gen = int(doc["generation"])
        new = dz.sampler_pools(doc)
        old = self.applied_pools
        self.log(f"fleet: desired state generation {gen} (applied {self.applied})")
        for name in sorted(set(new) - set(old)):
            self._add(controller, name, new[name], gen)
        for name in sorted(set(old) - set(new)):
            self._remove(controller, name, old[name], gen)
        for name in sorted(set(new) & set(old)):
            if dz.pool_shape(new[name]) != dz.pool_shape(old[name]):
                self._reconfigure(controller, name, new[name], gen)
            target = dz.pool_target(new[name])
            if target is not None and (target != dz.pool_target(old[name]) or new[name].get("retire")):
                self._set_target(controller, name, target, new[name].get("retire") or (), new[name].get("mode"),
                                 gen)
        self.applied, self.applied_pools = gen, new
        self._dirty = True
        self._persist()

    def _persist(self) -> None:
        """The applied state when it changed (a store that does not answer: the next pass)."""
        if not self._dirty or self.fenced is not None:
            return
        try:
            self._seq = self.store.write_applied(
                self.applied, self.applied_pools, {name: r.member for name, r in self.removing.items() if r.member},
                writer=self.writer, claim=self._claim)
            self._dirty, self._claim = False, False
        except (dz.AppliedFenced, dz.AppliedUnsafe) as error:
            self.fenced = str(error)
            self.log(f"fleet: CRITICAL: {error}: this controller applies the desired state no more")
        except Exception as error:  # noqa: BLE001
            self.log(f"fleet: applied state not written ({error!r}); tried again at the next pass")

    # -- the changes
    def _add(self, controller: Any, name: str, member: dict[str, Any], gen: int) -> None:
        if name in self.removing:  # brought back while it drained: driven again at its new target
            self.removing.pop(name)
            self.log(f"fleet: pool {name} is back in generation {gen}: its removal stops")
            target = dz.pool_target(member)
            if target is not None:
                self._set_target(controller, name, target, (), None, gen)
            return
        if name in controller.slots and not controller.slots[name].pool.stopped:
            return  # already driven (built from this generation)
        if self.hooks.add is None:
            self.log(f"fleet: pool {name} added in generation {gen} but this controller cannot build pools")
            return
        try:
            slot = self.hooks.add(name, member)
        except BaseException as error:  # noqa: BLE001 - reported; the other pools go on (a SystemExit: refused)
            self.log(f"fleet: pool {name} (generation {gen}) not started: {error!r}")
            controller.note(name, "pool_add_failed", generation=gen, error=repr(error)[:300])
            return
        slot = controller.add_slot(name, slot)
        slot.pool._event("pool_added", generation=gen)  # noqa: SLF001
        self.log(f"fleet: pool {name} added (generation {gen})")

    def _set_target(self, controller: Any, name: str, target: int, retire: Any, mode: Any, gen: int) -> None:
        slot = controller.slots.get(name)
        if slot is None:
            return
        try:
            if self.hooks.set_target is not None:
                self.hooks.set_target(name, slot, target, list(retire), mode, generation=gen)
            else:
                slot.pool.set_target(target, retire=retire, mode=mode, source="desired", generation=gen)
        except Exception as error:  # noqa: BLE001
            self.log(f"fleet: pool {name}: target {target} (generation {gen}) not applied: {error!r}")
            controller.note(name, "target_not_applied", generation=gen, target=target, error=repr(error)[:300])
        controller.wake()

    def _reconfigure(self, controller: Any, name: str, member: dict[str, Any], gen: int) -> None:
        """The new config and ops are swapped in under the pool's tick lock (never while a tick
        runs with the old ones); a busy lock is tried again at the next pass."""
        slot = controller.slots.get(name)
        if slot is None:
            self.pending_rolls.pop(name, None)
            return
        if not slot.tick_lock.acquire(blocking=False):
            self.pending_rolls[name] = (member, gen)
            return
        self.pending_rolls.pop(name, None)
        try:
            if self.hooks.reconfigure is not None:
                self.hooks.reconfigure(name, slot, member)
            slot.pool.start_roll(f"desired state generation {gen}")
        except Exception as error:  # noqa: BLE001 - the pool keeps its old config
            self.log(f"fleet: pool {name}: new config (generation {gen}) refused: {error!r}")
            controller.note(name, "reconfigure_refused", generation=gen, error=repr(error)[:300])
        finally:
            slot.tick_lock.release()
        controller.wake()

    def _remove(self, controller: Any, name: str, member: dict[str, Any], gen: int) -> None:
        self.pending_rolls.pop(name, None)
        slot = controller.slots.get(name)
        if slot is None:
            return
        self.removing[name] = _Removal(since=self.clock(), member=member, generation=gen)
        self.log(f"fleet: pool {name} removed in generation {gen}: target 0, draining")
        slot.pool._event("pool_removing", generation=gen)  # noqa: SLF001
        self._set_target(controller, name, 0, (), None, gen)

    def _advance_removals(self, controller: Any) -> None:
        """A removed pool with no member left (the scale-down drained them on its worker):
        its end runs on its worker (``pending_finish``) at its next tick; a pool that no
        worker ticks any more (stopped) ends here, when its tick lock is free."""
        for name in list(self.removing):
            slot = controller.slots.get(name)
            if slot is None:
                self.removing.pop(name, None)
                self._dirty = True
                continue
            pool = slot.pool
            if pool.detached:
                continue
            busy = [m for m in pool.live() if m.state != "gone"]
            if busy and not pool.stopped:
                continue  # the scale-down drains them (fence, claims back, release)
            if not pool.stopped:
                if getattr(slot, "pending_finish", None) is None:
                    slot.pending_finish = lambda n=name, s=slot: self.finish_removal(controller, n, s)
                    controller.wake()
                continue
            if not slot.tick_lock.acquire(blocking=False):
                continue
            try:
                self.finish_removal(controller, name, slot)
            finally:
                slot.tick_lock.release()

    def finish_removal(self, controller: Any, name: str, slot: Any) -> bool:
        """Every shard of the drained pool ends for good, then the pool stops and is dropped.
        Runs under the pool's tick lock. False: not finished (tried again at the next pass)."""
        if name not in self.removing:
            return False  # brought back meanwhile
        pool = slot.pool
        control = getattr(pool, "control", None)
        shards = {m.shard for m in pool.members if m.shard}
        if control is not None:
            from .store_lock import LockLost, LockUnconfirmed  # noqa: PLC0415

            pool._reserved_shard = None  # noqa: SLF001 - the pool orders nothing more: an unused number
            try:  # first: a number issued but never ordered gets its in-shard data end (and end record)
                pool.end_unused_shards()
            except LockLost as error:  # another controller holds the pool: it finishes the removal
                pool.detach(f"store lock lost while removing the pool: {error}")
                return False
            except LockUnconfirmed as error:
                self.log(f"fleet: pool {name}: removal waits for the store ({error})")
                return False
            try:  # every number the allocator issued to the pool, a never ordered one too
                shards |= {str(d["shard"]) for d in control.allocator.issued() if d.get("shard")}
            except Exception as error:  # noqa: BLE001
                self.log(f"fleet: pool {name}: issued shard numbers not read ({error!r}); tried again")
                return False
            for shard in sorted(shards):
                try:
                    control.allocator.mark_ended(shard, why="pool removed from the desired state")
                except Exception as error:  # noqa: BLE001
                    self.log(f"fleet: pool {name}: end record of {shard} not written ({error!r}); tried again")
                    return False
        shards_sorted = sorted(shards)
        pool.retired_shards = sorted(set(pool.retired_shards) | set(shards_sorted))
        if not pool._save_now() or not pool._publish_members():  # noqa: SLF001
            # the member table must retire them before the slot goes (the trainer's data end)
            self.log(f"fleet: pool {name}: state / member table not written; the removal is tried again")
            return False
        pool._event("shards_retired", shards=shards_sorted)  # noqa: SLF001
        left = pool.drain_all("removed from the fleet's desired state", wait_s=controller.drain_wait_s,
                              sleep=controller.sleep)
        controller._save(slot)  # noqa: SLF001
        slot.left = list(left)
        controller._close_slot(name, slot)  # noqa: SLF001
        controller.drop_slot(name)
        self.removing.pop(name, None)
        self._dirty = True
        if self.hooks.removed is not None:
            try:
                self.hooks.removed(name)
            except Exception as error:  # noqa: BLE001
                self.log(f"fleet: pool {name}: removal hook failed ({error!r})")
        self.log(f"fleet: pool {name} removed: {len(shards_sorted)} shard(s) retired, "
                 f"{len(left)} member(s) left billed")
        return True
