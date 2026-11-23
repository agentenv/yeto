"""Breakers of the launch layer: the host that runs ``yeto launch``.

The cost guard thread (``yeto.cost.guard``: budget, idle shutdown, NC lease overdue), the
fleet controller's stall clock (``yeto.launch.fleet``), the trainer scheduler's failure
budget (``yeto.launch.trainer_loop``), the provider's own function timeout and the
teardown check. Every knob here is a ``yeto launch`` flag today; those flags stay as
aliases. ``--no-cost-guard`` is the ``cost_guard`` switch group (budget, idle, NC lease
alert: the guard thread does not run).

Defaults are literals: the runtime modules are not imported here (registry loads stay
light). ``tests/test_breakers_launch.py`` compares each one with its code constant or
argparse default.
"""

from __future__ import annotations

from yeto.breakers.spec import Alias, Breaker, Param, SwitchGroup

MIN = 60.0  # the legacy flags count minutes; the registry counts seconds


BREAKERS: tuple[Breaker, ...] = (
    Breaker(
        name="launch.budget", layer="launch", kind="breaker",
        params=(
            Param("usd", float, None, unit="usd", lo=0, nullable=True, inherit_from="run.budget.usd",
                  doc="stop the run when the launch's ledger spend reaches this; null: no budget"),
            Param("warn_usd", float, None, unit="usd", lo=0, nullable=True,
                  doc="cost_budget_warn at this spend; null: 80 % of usd"),
            Param("stop_grace_s", float, 20 * MIN, unit="s", lo=0,
                  doc="after the graceful stop (STOP flag), release every ledger resource by force "
                  "when the run is still up this much later"),
        ),
        actions=("stop",), default_action="stop", default_enabled=False, enabled_by=("usd",),
        aliases=(
            # "off": how a disabled launch.budget is written back (the parser reads it as no budget)
            Alias("launch_flag", "--budget-usd", "usd", off_values=("off", "none")),
            Alias("launch_flag", "--budget-warn-usd", "warn_usd"),
            Alias("launch_flag", "--budget-stop-grace-minutes", "stop_grace_s", scale=MIN),
        ),
        depends_on=("D11", "D12"),
        legacy_events=("cost_budget_warn", "cost_budget_exceeded", "cost_guard_stop"),
        gauge="level", owner="yeto.cost.guard",
        doc="the launch's spend (its resource ledger, priced per resource); with a breakers: "
        "section that sets run.budget.usd it inherits that value unless written here"),
    Breaker(
        name="launch.idle", layer="launch", kind="breaker",
        params=(
            # 0 would be written back as --idle-warn-minutes 0, the legacy "off": enabled: false
            # is the only off, so the smallest limit is one second
            Param("warn_s", float, 15 * MIN, unit="s", lo=1,
                  doc="idle_warn after this long without progress / with 0 % GPU"),
            Param("stop_after_warn_s", float, 15 * MIN, unit="s", lo=0,
                  doc="stop the run this long after idle_warn when still idle"),
            Param("load_grace_s", float, None, unit="s", lo=0, nullable=True,
                  doc="no idle verdict this long after the start / a loading event; null: the "
                  "recipe default (45 min, 90 min for a Kimi-K3.1 --tuning full launch)"),
            Param("max_load_s", float, 120 * MIN, unit="s", lo=0,
                  doc="loading events alone cannot exempt a run longer than this since its last "
                  "progress"),
        ),
        actions=("stop",), default_action="stop",
        aliases=(
            Alias("launch_flag", "--idle-warn-minutes", "warn_s", off_values=(0, 0.0), scale=MIN),
            Alias("launch_flag", "--idle-stop-minutes", "stop_after_warn_s", scale=MIN),
            Alias("launch_flag", "--load-grace-minutes", "load_grace_s", scale=MIN),
            Alias("launch_flag", "--max-load-minutes", "max_load_s", scale=MIN),
        ),
        legacy_events=("idle_warn", "idle_cleared", "idle_shutdown", "cost_guard_stop"),
        gauge="elapsed", owner="yeto.cost.guard",
        doc="no progress event (or 0 % GPU on every reporting GPU) outside a loading phase"),
    Breaker(
        name="launch.nc_lease_overdue", layer="launch", kind="alert",
        params=(Param("grace_s", float, 5 * MIN, unit="s", lo=0,
                      doc="alert when an NC lease is still open this long after its lease_until"),),
        actions=("warn",), default_action="warn",
        aliases=(Alias("launch_flag", "--nc-lease-overdue-minutes", "grace_s", scale=MIN),),
        legacy_events=("nc_lease_overdue",), gauge="elapsed", owner="yeto.cost.guard",
        doc="alert only: the NC pool owns its leases; off with the cost guard"),
    Breaker(
        name="launch.stall", layer="launch", kind="breaker",
        params=(Param("timeout_s", float, 900.0, unit="s", lo=1,
                      doc="stop the run when no island event arrives for this long and not every "
                      "island finalized; the clock starts at the first island event"),),
        actions=("stop",), default_action="stop",
        aliases=(Alias("launch_flag", "--rl-stall-timeout", "timeout_s", off_values=(0, 0.0)),),
        depends_on=("D5", "D6"), legacy_events=("RunStalled",), gauge="elapsed",
        owner="yeto.launch.fleet",
        doc="RunStalled, exit 6; the launcher tears the run down"),
    Breaker(
        name="launch.trainer_failures", layer="launch", kind="retry_budget",
        params=(Param("max_n", int, 3, unit="n", lo=1,
                      doc="trainer incarnations that may fail in a row (no trained round between "
                      "them); a trained round resets the count"),),
        actions=("stop",), default_action="stop", can_disable=False,
        aliases=(Alias("launch_flag", "--rl-trainer-max-failures", "max_n"),),
        legacy_events=("TrainerLoopGaveUp",), gauge="count", owner="yeto.launch.trainer_loop",
        doc="the trainer scheduler releases the trainer and stops the run (exit 8)"),
    Breaker(
        name="launch.provider_timeout", layer="launch", kind="breaker",
        params=(
            Param("timeout_s", int, 24 * 3600, unit="s", lo=1,
                  doc="Modal islands: the function timeout (the provider's hard stop)"),
            Param("retries_n", int, 10, unit="n", lo=0,
                  doc="Modal islands: function retries after a failed island container "
                  "(--no-island-relaunch forces 0)"),
        ),
        actions=("stop",), default_action="stop", can_disable=False,
        aliases=(
            Alias("launch_flag", "--modal-timeout-s", "timeout_s"),
            Alias("launch_flag", "--modal-retries", "retries_n"),
        ),
        depends_on=("D13",), owner="yeto.modal_runner",
        doc="enforced by the provider, not by yeto: no rl_breaker event"),
    Breaker(
        name="launch.teardown_verify", layer="launch", kind="invariant",
        actions=("stop",), default_action="stop", can_disable=False,
        native="yeto.launch.modal_islands MODAL_STOP_VERIFY_ATTEMPTS / yeto.launch.teardown "
        "terminate_and_verify",
        owner="yeto.launcher",
        doc="after teardown every Modal app row of the run is stopped with 0 tasks (else exit 5) "
        "and every sky cluster is confirmed gone (else the head stays up)"),
)

SWITCH_GROUPS: tuple[SwitchGroup, ...] = (
    SwitchGroup(
        "cost_guard", ("launch.budget", "launch.idle", "launch.nc_lease_overdue"),
        aliases=(Alias("launch_flag", "--no-cost-guard", None, nargs=0),),
        doc="the cost guard thread: budget stop, idle shutdown and the NC lease alert (the "
        "resource ledger is still written)"),
)
