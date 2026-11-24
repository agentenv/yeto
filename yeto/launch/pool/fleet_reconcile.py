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

Trainer members (``kind: trainer``) are left to the trainer worker of the
``unified-trainer-islands`` change. A pool's own target changes between two
generations (a manual release it detected, an automatic scaler) are not undone: only a
change of the desired state is acted on.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import desired as dz


@dataclass
class ReconcileHooks:
    """What the reconciler cannot do alone. ``add(name, member) -> PoolSlot``;
    ``set_target(name, slot, target, retire, mode) -> None`` (None: the pool's own
    :meth:`set_target`); ``reconfigure(name, slot, member) -> None`` (gives the pool its new
    config and ops); ``removed(name) -> None`` (after the pool is gone)."""

    add: Callable[[str, dict[str, Any]], Any] | None = None
    set_target: Callable[..., None] | None = None
    reconfigure: Callable[[str, Any, dict[str, Any]], None] | None = None
    removed: Callable[[str], None] | None = None


@dataclass
class _Removal:
    since: float
    shards_ended: bool = False


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
    _polled_at: float | None = None
    errors: int = 0

    def start_from(self, doc: dict[str, Any] | None) -> None:
        """The generation the controller was built from (its pools are that generation's)."""
        if doc is not None:
            self.applied = int(doc.get("generation") or 0)
            self.applied_pools = dz.sampler_pools(doc)

    def step(self, controller: Any, *, force: bool = False) -> None:
        """One reconcile pass: a newer generation is applied; removals in progress advance."""
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
        self.apply(controller, doc)

    def apply(self, controller: Any, doc: dict[str, Any]) -> None:
        gen = int(doc["generation"])
        new = dz.sampler_pools(doc)
        old = self.applied_pools
        self.log(f"fleet: desired state generation {gen} (applied {self.applied})")
        for name in sorted(set(new) - set(old)):
            self._add(controller, name, new[name], gen)
        for name in sorted(set(old) - set(new)):
            self._remove(controller, name, gen)
        for name in sorted(set(new) & set(old)):
            if name in self.removing:  # brought back while draining: driven again
                self.removing.pop(name)
            if dz.pool_shape(new[name]) != dz.pool_shape(old[name]):
                self._reconfigure(controller, name, new[name], gen)
            target = dz.pool_target(new[name])
            if target is not None and (target != dz.pool_target(old[name]) or new[name].get("retire")):
                self._set_target(controller, name, target, new[name].get("retire") or (), new[name].get("mode"),
                                 gen)
        self.applied, self.applied_pools = gen, new

    # -- the changes
    def _add(self, controller: Any, name: str, member: dict[str, Any], gen: int) -> None:
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
                self.hooks.set_target(name, slot, target, list(retire), mode)
            else:
                slot.pool.set_target(target, retire=retire, mode=mode, source="desired", generation=gen)
        except Exception as error:  # noqa: BLE001
            self.log(f"fleet: pool {name}: target {target} (generation {gen}) not applied: {error!r}")
            controller.note(name, "target_not_applied", generation=gen, target=target, error=repr(error)[:300])
        controller.wake()

    def _reconfigure(self, controller: Any, name: str, member: dict[str, Any], gen: int) -> None:
        slot = controller.slots.get(name)
        if slot is None:
            return
        try:
            if self.hooks.reconfigure is not None:
                self.hooks.reconfigure(name, slot, member)
        except Exception as error:  # noqa: BLE001 - the pool keeps its old config
            self.log(f"fleet: pool {name}: new config (generation {gen}) refused: {error!r}")
            controller.note(name, "reconfigure_refused", generation=gen, error=repr(error)[:300])
            return
        with slot.tick_lock:
            slot.pool.start_roll(f"desired state generation {gen}")
        controller.wake()

    def _remove(self, controller: Any, name: str, gen: int) -> None:
        slot = controller.slots.get(name)
        if slot is None:
            return
        self.removing[name] = _Removal(since=self.clock())
        self.log(f"fleet: pool {name} removed in generation {gen}: target 0, draining")
        slot.pool._event("pool_removing", generation=gen)  # noqa: SLF001
        self._set_target(controller, name, 0, (), None, gen)

    def _advance_removals(self, controller: Any) -> None:
        for name in list(self.removing):
            slot = controller.slots.get(name)
            if slot is None:
                self.removing.pop(name)
                continue
            pool = slot.pool
            if pool.detached:
                continue
            busy = [m for m in pool.live() if m.state != "gone"]
            if busy and not pool.stopped:
                continue  # the scale-down drains them (fence, claims back, release)
            with slot.tick_lock:
                self._finish_removal(controller, name, slot)
            self.removing.pop(name)

    def _finish_removal(self, controller: Any, name: str, slot: Any) -> None:
        """Every shard of the drained pool ends for good, then the pool stops and is dropped."""
        pool = slot.pool
        shards = {m.shard for m in pool.members if m.shard}
        control = getattr(pool, "control", None)
        if control is not None:
            try:  # every number the allocator issued to the pool, a never ordered one too
                shards |= {str(d["shard"]) for d in control.allocator.issued() if d.get("shard")}
            except Exception as error:  # noqa: BLE001 - the members' shards are retired anyway
                self.log(f"fleet: pool {name}: issued shard numbers not read ({error!r})")
        shards = sorted(shards)
        if control is not None:
            for shard in shards:
                try:
                    control.allocator.mark_ended(shard, why="pool removed from the desired state")
                except Exception as error:  # noqa: BLE001 - the member table still retires it
                    self.log(f"fleet: pool {name}: end record of {shard} not written ({error!r})")
            from .store_lock import LockLost, LockUnconfirmed  # noqa: PLC0415

            try:
                pool.end_unused_shards()
            except LockLost as error:  # another controller holds the pool: it finishes the removal
                pool.detach(f"store lock lost while removing the pool: {error}")
                return
            except LockUnconfirmed as error:  # the allocator's records hold; the member table retires them
                self.log(f"fleet: pool {name}: unused shards not ended now ({error})")
        pool.retired_shards = sorted(set(pool.retired_shards) | set(shards))
        pool._event("shards_retired", shards=shards)  # noqa: SLF001
        pool._save_now()  # noqa: SLF001 - publishes the member table with every shard retired
        left = pool.drain_all("removed from the fleet's desired state", wait_s=controller.drain_wait_s,
                              sleep=controller.sleep)
        controller._save(slot)  # noqa: SLF001
        slot.left = list(left)
        controller._close_slot(name, slot)  # noqa: SLF001
        controller.drop_slot(name)
        if self.hooks.removed is not None:
            try:
                self.hooks.removed(name)
            except Exception as error:  # noqa: BLE001
                self.log(f"fleet: pool {name}: removal hook failed ({error!r})")
        self.log(f"fleet: pool {name} removed: {len(shards)} shard(s) retired, "
                 f"{len(left)} member(s) left billed")
