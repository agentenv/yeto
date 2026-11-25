"""Breakers of the pool layer: the rollout island pool controller, any provider
(:mod:`yeto.launch.pool.island_pool`, knobs in :class:`~yeto.launch.pool.status.ElasticConfig`).

Every default is read from the pool's own dataclasses, so the registry cannot drift
from the code. The elastic fields stay the native knobs (aliases ``pool_elastic``);
``breakers.pool.<entry>`` applies to every pool of a run, ``breakers.pools.<name>.<entry>``
to one. Entries that the pool cannot switch off today are declared ``can_disable=False``.
"""

from __future__ import annotations

from dataclasses import fields

from yeto.breakers.spec import Alias, Breaker, Param
from yeto.launch.pool.island_pool import PoolConfig
from yeto.launch.pool.status import ElasticConfig

_E = ElasticConfig()
_LEASE = {f.name: f.default for f in fields(PoolConfig) if f.name in ("replace_lead_s", "end_margin_s")}
_OWNER = "yeto.launch.pool.island_pool"


def _elastic(path: str, param: str | None, **kw) -> Alias:
    return Alias("pool_elastic", path, param, **kw)


BREAKERS: tuple[Breaker, ...] = (
    Breaker(
        name="pool.provision_breaker", layer="pool", kind="breaker",
        params=(Param("failures_n", int, _E.breaker_failures, unit="n", lo=1,
                      doc="provisioning failures in a row that open the breaker"),
                Param("cooldown_s", float, _E.breaker_cooldown_s, unit="s", lo=0,
                      doc="first cooldown; doubles per trip"),
                Param("max_cooldown_s", float, _E.breaker_max_cooldown_s, unit="s", lo=0),
                Param("max_trips_n", int, _E.breaker_max_trips, unit="n", lo=1,
                      doc="trips after which the pool orders nothing again (it stops once empty)")),
        actions=("halt",), default_action="halt", can_disable=False,
        aliases=(_elastic("breaker_failures", "failures_n"), _elastic("breaker_cooldown_s", "cooldown_s"),
                 _elastic("breaker_max_cooldown_s", "max_cooldown_s"),
                 _elastic("breaker_max_trips", "max_trips_n")),
        legacy_events=("nc_elastic_breaker_open", "rl_pool_breaker_open", "nc_elastic_breaker_tripped",
                       "rl_pool_breaker_tripped"),
        gauge="count", owner=_OWNER,
        doc="an order that fails, a ticket lost before it is granted or a node whose sampler never comes up "
            "counts a failure; each trip is a cooldown (warn), the last one halts ordering (trip)"),
    Breaker(
        name="pool.spend_cap", layer="pool", kind="breaker",
        params=(Param("usd", float, _E.max_run_spend_usd, unit="usd", lo=0, nullable=True,
                      inherit_from="run.budget.usd", doc="the pool's spend over the whole run, in USD"),
                Param("ucr", int, _E.max_run_spend_ucr, unit="ucr", lo=0, nullable=True,
                      doc="the same cap in the National Compute cost unit; both may be set")),
        actions=("pause",), default_action="pause", default_enabled=False, enabled_by=("usd", "ucr"),
        aliases=(_elastic("max_run_spend_usd", "usd"), _elastic("max_run_spend_ucr", "ucr")),
        depends_on=("D11",),
        legacy_events=("nc_elastic_order_refused", "rl_pool_order_refused"),
        gauge="level", owner=_OWNER,
        doc="refuses a new order when the spend plus one node hour passes the cap (the nodes up keep running)"),
    Breaker(
        name="pool.node_cap", layer="pool", kind="breaker",
        params=(Param("max_nodes_n", int, _E.max_nodes, unit="n", lo=1, nullable=True,
                      doc="billed nodes at once (null: 2 x target)"),
                Param("max_orders_n", int, _E.max_orders, unit="n", lo=0, nullable=True,
                      doc="orders over the run (null: no cap)")),
        actions=("pause",), default_action="pause", can_disable=False,
        aliases=(_elastic("max_nodes", "max_nodes_n"), _elastic("max_orders", "max_orders_n")),
        legacy_events=("nc_elastic_order_refused", "rl_pool_order_refused"), gauge="count", owner=_OWNER,
        doc="refuses a new order past the node or order cap"),
    Breaker(
        name="pool.crash_loop", layer="pool", kind="breaker",
        actions=("replace", "halt"), default_action="replace", can_disable=False,
        aliases=(_elastic("crash_loop_before_ready", "action"),),
        legacy_events=("nc_elastic_sampler_crash_loop", "rl_pool_sampler_crash_loop"), gauge="count", owner=_OWNER,
        doc="a sampler that crashed past pool.sampler_restarts on a reachable node; one that never joined: "
            "replace the node, or halt the pool like a configuration error (nothing is ordered until "
            "pool-retry / pool-stop; a node that joined before is always replaced)"),
    Breaker(
        name="pool.sampler_restarts", layer="pool", kind="retry_budget",
        params=(Param("max_n", int, _E.max_sampler_restarts, unit="n", lo=0,
                      doc="in-place restarts of a crashed sampler before the node is replaced (0: none, "
                          "not off)"),
                Param("reset_s", float, _E.sampler_restart_reset_s, unit="s", lo=0, nullable=True,
                      doc="healthy time after the join that starts a fresh count (null: never)")),
        actions=("replace",), default_action="replace", can_disable=False,
        aliases=(_elastic("max_sampler_restarts", "max_n"), _elastic("sampler_restart_reset_s", "reset_s")),
        depends_on=("D10",),
        legacy_events=("nc_elastic_sampler_restart", "rl_pool_sampler_restart"),
        gauge="count", owner=_OWNER),
    Breaker(
        name="pool.engine_stall", layer="pool", kind="breaker",
        params=(Param("limit_s", float, _E.engine_stall_s, unit="s", lo=0,
                      doc="requests running or queued while the generation-token counter stands still"),
                Param("max_restart_failures_n", int, _E.max_stall_restart_failures, unit="n", lo=1,
                      doc="in-place restarts in a row without token progress before the node is retired")),
        actions=("replace",), default_action="replace",
        aliases=(_elastic("engine_stall_s", "limit_s", off_values=(None,)),
                 _elastic("max_stall_restart_failures", "max_restart_failures_n")),
        legacy_events=("nc_elastic_engine_stalled", "rl_pool_engine_stalled"), gauge="elapsed", owner=_OWNER,
        doc="fences and restarts a stalled engine in place, then retires the node"),
    Breaker(
        name="pool.sampler_stall", layer="pool", kind="breaker",
        params=(Param("limit_s", float, _E.stall_s, unit="s", lo=0, nullable=True,
                      doc="no event-tape progress of a joined sampler (set it above the longest round)"),),
        actions=("replace",), default_action="replace", default_enabled=False, enabled_by=("limit_s",),
        aliases=(_elastic("stall_s", "limit_s", off_values=(None,)),),
        gauge="elapsed", owner=_OWNER, doc="the sampler counts as crashed"),
    Breaker(
        name="pool.start_timeout", layer="pool", kind="breaker",
        params=(Param("limit_s", float, _E.start_timeout_s, unit="s", lo=0,
                      doc="from the order to a joined sampler (model pull + engine start)"),),
        actions=("replace",), default_action="replace", can_disable=False,
        aliases=(_elastic("start_timeout_s", "limit_s"),),
        depends_on=("D14",), gauge="elapsed", owner=_OWNER,
        doc="a provisioning failure: the node is replaced and the provision breaker counts it"),
    Breaker(
        name="pool.unreachable", layer="pool", kind="breaker",
        params=(Param("grace_s", float, _E.unreachable_grace_s, unit="s", lo=0,
                      doc="failed node probes in a row before a node counts as lost"),),
        actions=("replace",), default_action="replace", can_disable=False,
        aliases=(_elastic("unreachable_grace_s", "grace_s"),),
        legacy_events=("nc_elastic_lost", "rl_pool_lost"), gauge="elapsed", owner=_OWNER),
    Breaker(
        name="pool.fence_hold", layer="pool", kind="breaker",
        params=(Param("limit_s", float, _E.fence_max_hold_s, unit="s", lo=0,
                      doc="a node kept because its shard fence fails is released after this long"),),
        aliases=(_elastic("fence_max_hold_s", "limit_s", off_values=(None,)),),
        gauge="elapsed", owner=_OWNER, doc="releases the node (the fence is still retried)"),
    Breaker(
        name="pool.lease_overdue", layer="pool", kind="breaker",
        params=(Param("grace_s", float, _E.overdue_grace_s, unit="s", lo=0,
                      doc="past lease_until before a ticket still active is released (0 is a real value)"),),
        actions=("replace",), default_action="replace", can_disable=False,
        aliases=(_elastic("overdue_grace_s", "grace_s"),),
        legacy_events=("nc_elastic_lease_overdue", "rl_pool_lease_overdue"), gauge="elapsed", owner=_OWNER,
        doc="a lease-bounded provider only; deletes this pool's own ticket"),
    Breaker(
        name="pool.control_plane_alert", layer="pool", kind="alert",
        params=(Param("limit_s", float, _E.control_plane_alert_s, unit="s", lo=0,
                      doc="failed status calls in a row before an alert (the pool never acts on them)"),),
        actions=("warn",), default_action="warn", can_disable=False,
        aliases=(_elastic("control_plane_alert_s", "limit_s"),),
        gauge="elapsed", owner=_OWNER),
    Breaker(
        name="pool.fence_alert", layer="pool", kind="alert",
        params=(Param("failures_n", int, _E.fence_alert_failures, unit="n", lo=1),
                Param("limit_s", float, _E.fence_alert_s, unit="s", lo=0)),
        actions=("warn",), default_action="warn", can_disable=False,
        aliases=(_elastic("fence_alert_failures", "failures_n"), _elastic("fence_alert_s", "limit_s")),
        legacy_events=("nc_elastic_fence_stuck", "rl_pool_fence_stuck"), gauge="count", owner=_OWNER,
        doc="a shard fence that keeps failing (it is retried every tick)"),
    Breaker(
        name="pool.lease_end", layer="pool", kind="breaker",
        params=(Param("replace_lead_s", float, _LEASE["replace_lead_s"], unit="s", lo=0,
                      doc="before lease_until a replacement is ordered on a free slot (make before break)"),
                Param("end_margin_s", float, _LEASE["end_margin_s"], unit="s", lo=0,
                      doc="before lease_until the node is drained and released")),
        actions=("replace",), default_action="replace", can_disable=False,
        aliases=(Alias("pool_config", "replace_lead_s", "replace_lead_s"),
                 Alias("pool_config", "end_margin_s", "end_margin_s")),
        legacy_events=("nc_elastic_lease_end_without_ready_replacement",
                       "rl_pool_lease_end_without_ready_replacement"),
        gauge="elapsed", owner=_OWNER,
        doc="lease-bounded providers only (National Compute); a provider without leases never drains for it"),
)
