"""Lease-aware, eviction-tolerant sampler pool on National Compute.

A National Compute lease is granted until ``lease_until`` (4 h, no renewal) but
is **not** ended there: it stays active and billed until deleted. An account
holds one order at a time. The pool keeps ``target`` samplers writing
one buffer:

* a member whose lease ends within ``replace_lead_s`` (bootstrap + model pull +
  engine start, plus margin) gets a replacement ordered on a free account; the
  replacement writes a new shard (``<base>-<n>``) of the same buffer and
  continues the prompts of the shard it replaces (``continues_shard``);
* once the replacement is ready, the old member is drained (sampler stopped,
  node released, ticket deleted and confirmed ``ended``); a member is drained
  anyway ``end_margin_s`` before its lease end, ready replacement or not: the
  platform does **not** end a ticket at ``lease_until`` (it stays ``active`` and
  billed), so the pool never leaves that to the platform;
* with no free account (one account, one order), the replacement waits until
  the old member is gone: a gap rather than a second order on the same account;
* when the shared balance falls below ``stop_below_ucr``, nothing new is ordered
  and every member is drained; ``min_balance_to_order_ucr`` guards each order.

Eviction tolerance (:mod:`.elastic`; every knob in :class:`~.elastic.ElasticConfig`):

* every tick reads each member's ticket and, once its sampler is started, probes
  the node (one ssh round trip). :func:`~.elastic.classify` tells an eviction
  (ticket ended ``preempted``/``lost``/``admin``), a preemption notice
  (``preempt_at``), an expiry, an overdue lease (``active`` past
  ``lease_until``), a lost node (probes failing for ``unreachable_grace_s``), a
  network blip (shorter), a crashed sampler and a failed status call apart;
* a gone member (evicted, expired, overdue, lost, crash-looping, never started)
  is retired: its shard is fenced (:func:`~.elastic.fence_shard`, so a node that
  comes back adds nothing the trainer takes), its ticket is deleted (a lost node
  is still billed) and confirmed ended, and its data slot and shard go to a
  pending handoff that the next order continues;
* a crashed sampler on a reachable node is restarted in place (new incarnation,
  newest release) up to ``max_sampler_restarts`` times;
* a stalled engine (container up, requests running or queued in vLLM, its
  generation-token counter flat for ``engine_stall_s``; an idle engine waiting
  for the trainer is not stalled) is fenced (its claims free at once) and
  restarted in place on the same node; ``max_stall_restart_failures`` restarts in
  a row with no token progress between them retire the node;
* a drain that fails is retried every tick until the ticket reads ended;
* orders are gated by a circuit breaker (consecutive provisioning failures) and
  a budget (run spend, billed nodes, orders); every decision is an
  ``nc_elastic_*`` event;
* the patrol **adopts** any open ticket the ledger records for this run that no
  live member owns (an order the process did not live to save, a state file
  moved away, an older pool instance): it becomes a member and is watched like
  any other, never deleted as an orphan. Only a stopping pool (``pool-stop``)
  or ``release_orphans`` deletes such a ticket, or a ticket the pool has no room
  for (past ``shard_limit``, every data slot held, ``target`` already met): it is
  drained with an ``adopt_refused`` warning;
* the state is saved right after every order (no window in which a billed
  ticket exists that no state file knows), and a restart keeps a member whose
  sampler start was cut off as STARTING when its sampler container runs (the
  start is not run again: no second model pull, no new incarnation).

:class:`LeasePool` is pure policy over :class:`PoolOps` (the provision, start,
probe, fence, drain and balance calls) and a clock, so the tests drive it with
fakes; :mod:`.cli` wires the National Compute implementation.

The policy itself is provider-neutral and lives in
:mod:`yeto.launch.pool.island_pool` (:class:`LeasePool` is
:class:`~yeto.launch.pool.island_pool.IslandPool`); this module re-exports its
names and adds the National Compute :class:`PoolConfig` (a node hour at the
MI355X rate in ucr for the budget gate and the order floor).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..pool.island_pool import (  # noqa: F401 - re-exported
    DRAINING,
    GONE,
    METRICS_MISS_ALERT,
    PROVISIONING,
    READY,
    STARTING,
    IslandPool,
    Member,
    PoolOps,
    is_config_error,
)
from ..pool.island_pool import PoolConfig as IslandPoolConfig
from .client import hourly_cost_ucr


@dataclass
class PoolConfig(IslandPoolConfig):
    """:class:`~yeto.launch.pool.island_pool.PoolConfig` of a National Compute pool:
    one node hour is :func:`~.client.hourly_cost_ucr` (8 MI355X GPUs) unless
    ``hourly_cost`` is set."""

    def hourly(self) -> float:
        return self.hourly_cost or hourly_cost_ucr()


LeasePool = IslandPool
