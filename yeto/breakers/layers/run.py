"""Breakers of the run layer: the ``yeto run up`` monitor (``yeto.rl.up``, ``yeto.rl.starvation``,
``yeto.rl.release_stall``).

The legacy run config keys (``budget.*``, ``first_group.timeout_s``, ``starvation.*``,
``rollout.stall_alert_s``, ``sandbox.max_*containers*``) stay as read-only aliases: ``run up``
reads the resolved values, so nothing is written back into the run config.
"""

from __future__ import annotations

from typing import Any

from yeto.breakers.spec import Alias, Breaker, Param
from yeto.launch.pool.controller import DEFAULT_STALL_S
from yeto.rl import release_stall, starvation, up, up_consistency


def _minutes(value: Any) -> float:
    return float(value) * 60


def _minutes_or_off(value: Any) -> float | None:
    """``starvation.alert_min`` / ``stop_min``: 0 turns that verdict off (None)."""
    return None if float(value) == 0 else _minutes(value)


def _count_or_unset(value: Any) -> int | None:
    """``sandbox.max_run_containers`` / ``max_containers_per_sampler``: 0 reads as unset."""
    return int(value) if value else None


BREAKERS: tuple[Breaker, ...] = (
    Breaker(
        name="run.budget", layer="run", kind="breaker",
        params=(Param("usd", float, None, unit="usd", nullable=True,
                      doc="estimated spend cap of the run (trainer plus every sampler pool, over every up)"),
                Param("warn_ratio", float, 0.8, unit="ratio", lo=0.0, hi=1.0,
                      doc="the warn line launch and the dashboard inherit (D12); run up itself does not warn"),
                Param("force_pools_ratio", float, up.FORCE_FACTOR, unit="ratio", lo=1.0,
                      doc="after a budget or wall-clock trip that kept the pools, force pool-stop once the spend "
                          "(or the wall time) passes this multiple of its limit"),
                Param("release_pools_on_trip", bool, False,
                      doc="on a budget or wall-clock trip run the full down (every sampler pool, the sandboxes) "
                          "instead of stopping the trainer only")),
        actions=("stop",), default_action="stop", default_enabled=False, enabled_by=("usd",),
        aliases=(Alias("run_config", "budget.usd", "usd", off_values=(None,), to_canonical=float,
                       materialize=False),
                 Alias("run_config", "budget.release_nc_on_breach", "release_pools_on_trip", materialize=False)),
        depends_on=("D11", "D12"), gauge="level", owner="yeto.rl.up",
        doc="stops the trainer when the estimated spend passes the cap; the pools stay unless "
            "release_pools_on_trip"),
    Breaker(
        name="run.wall_clock", layer="run", kind="breaker",
        params=(Param("hours", float, up_consistency.DEFAULT_WALL_H, unit="h", lo=0.0,
                      doc="hours of this run up (since it started the pools) before the trainer is stopped"),),
        actions=("stop",), default_action="stop", implicit=True,
        aliases=(Alias("run_config", "budget.wall_h", "hours", off_values=(None,), to_canonical=float,
                       materialize=False),),
        depends_on=("D13",), gauge="elapsed", owner="yeto.rl.up",
        doc="stops the trainer after this many hours of one run up; on when not written (3 h). Its pool "
            "release and forced pool-stop follow run.budget.release_pools_on_trip / force_pools_ratio"),
    Breaker(
        name="run.trainer_silence", layer="run", kind="breaker",
        params=(Param("limit_s", float, None, unit="s", lo=0.0, nullable=True,
                      doc="silence limit; null: max(stall_factor_n x the trainer's --rl-stall-timeout, floor_s)"),
                Param("floor_s", float, up.DEFAULT_SILENCE_MIN * 60, unit="s", lo=0.0,
                      doc="the derived limit is never below this"),
                Param("stall_factor_n", int, up.SILENCE_STALL_FACTOR, unit="n", lo=1,
                      doc="the derived limit in stall timeouts")),
        actions=("stop",), default_action="stop",
        aliases=(Alias("run_config", "budget.silence_min", "floor_s", to_canonical=_minutes, materialize=False),),
        depends_on=("D6",), gauge="elapsed", owner="yeto.rl.up",
        doc="stops the trainer when its launcher log and progress paths stop growing while the launcher runs"),
    Breaker(
        name="run.first_group", layer="run", kind="breaker",
        params=(Param("timeout_s", float, up.DEFAULT_FIRST_GROUP_TIMEOUT_S, unit="s", lo=0.0,
                      doc="seconds run up waits for the first sample-buffer object before the trainer starts"),),
        actions=("stop",), default_action="stop",
        aliases=(Alias("run_config", "first_group.timeout_s", "timeout_s", to_canonical=float,
                       materialize=False),),
        depends_on=("D14",), gauge="elapsed", owner="yeto.rl.up",
        doc="ends run up (exit 1, the pools kept) when no group reaches the store in time"),
    Breaker(
        name="run.starvation", layer="run", kind="breaker",
        params=(Param("alert_s", float, starvation.DEFAULT_ALERT_MIN * 60, unit="s", lo=0.0, nullable=True,
                      doc="no new group this long: an alert (null: no alert)"),
                Param("stop_s", float, starvation.DEFAULT_STOP_MIN * 60, unit="s", lo=0.0, nullable=True,
                      doc="no new group this long and no healthy sampler: stop the trainer (null: no stop)"),
                Param("unhealthy_s", float, starvation.DEFAULT_UNHEALTHY_MIN * 60, unit="s", lo=0.0,
                      doc="no healthy sampler at least this long before a stop"),
                Param("trainer_wait_max_s", float, starvation.DEFAULT_TRAINER_WAIT_MAX_MIN * 60, unit="s", lo=0.0,
                      doc="every sampler waiting for the trainer holds the clock this long at most"),
                Param("stop_grace_s", float, starvation.DEFAULT_STOP_GRACE_MIN * 60, unit="s", lo=0.0,
                      doc="after the STOP flag, wait this long for the launcher before yeto down"),
                Param("check_s", float, starvation.DEFAULT_CHECK_S, unit="s", lo=0.0,
                      doc="list the buffer at most this often (> 0 while a verdict is on)")),
        actions=("stop",), default_action="stop", off_when_null=("alert_s", "stop_s"),
        aliases=(Alias("run_config", "starvation.alert_min", "alert_s", to_canonical=_minutes_or_off, materialize=False),
                 Alias("run_config", "starvation.stop_min", "stop_s", to_canonical=_minutes_or_off, materialize=False),
                 Alias("run_config", "starvation.unhealthy_min", "unhealthy_s", scale=60, materialize=False),
                 Alias("run_config", "starvation.trainer_wait_max_min", "trainer_wait_max_s", scale=60,
                       materialize=False),
                 Alias("run_config", "starvation.stop_grace_min", "stop_grace_s", scale=60, materialize=False),
                 Alias("run_config", "starvation.check_s", "check_s", materialize=False)),
        depends_on=("D7",), gauge="elapsed", owner="yeto.rl.starvation",
        legacy_events=(starvation.ALERT_EVENT, starvation.RECOVERED_EVENT, starvation.STOP_EVENT),
        doc="the trainer runs but no new group reaches the buffer: alert, then stop through the STOP flag "
            "when no sampler is healthy"),
    Breaker(
        name="run.rollout_stall_alert", layer="run", kind="alert",
        params=(Param("limit_s", float, DEFAULT_STALL_S, unit="s", lo=0.0,
                      doc="every ready sampler silent this long (none waiting for the trainer)"),),
        actions=("warn",), default_action="warn",
        aliases=(Alias("run_config", "rollout.stall_alert_s", "limit_s", off_values=(0,), materialize=False),),
        gauge="elapsed", owner="yeto.launch.pool.controller",
        doc="the in-process controller logs a rollout stall (controller: in_process only)"),
    Breaker(
        name="run.sandbox_containers", layer="run", kind="alert",
        params=(Param("max_run_n", int, None, unit="n", lo=0, nullable=True,
                      doc="sandbox containers of this run on the docker hosts"),
                Param("max_per_sampler_n", int, None, unit="n", lo=0, nullable=True,
                      doc="raises the line to this many per billed sampler node")),
        actions=("warn",), default_action="warn", default_enabled=False, enabled_by=("max_run_n", "max_per_sampler_n"),
        aliases=(Alias("run_config", "sandbox.max_run_containers", "max_run_n", to_canonical=_count_or_unset,
                       materialize=False),
                 Alias("run_config", "sandbox.max_containers_per_sampler", "max_per_sampler_n",
                       to_canonical=_count_or_unset, materialize=False)),
        gauge="level", owner="yeto.rl.up",
        doc="a warning line when the run's sandbox containers pass the larger of the two limits"),
    Breaker(
        name="run.release_stall", layer="run", kind="breaker",
        params=(Param("limit_s", float, None, unit="s", lo=1.0, nullable=True,
                      doc="no new policy release this long after the previous one (or the trainer's restart); "
                          f"null: {release_stall.limit_phrase(None)}"),
                Param("check_s", float, release_stall.DEFAULT_CHECK_S, unit="s", lo=1.0,
                      doc="read the store's LATEST release at most this often"),
                Param("stop_grace_s", float, starvation.DEFAULT_STOP_GRACE_MIN * 60, unit="s", lo=0.0,
                      doc="action stop: after the STOP flag, wait this long for the launcher before yeto down")),
        actions=("warn", "stop"), default_action="warn",
        gauge="elapsed", owner="yeto.rl.release_stall",
        doc="after the first release, no new release for the limit: an alert (stop: the STOP flag); "
            "a trainer restart starts the clock again"),
    Breaker(
        name="run.trainer_exit", layer="run", kind="invariant", can_disable=False,
        actions=("stop",), default_action="stop", native="the trainer launcher's exit (yeto launch)",
        owner="yeto.rl.up",
        doc="a launcher that exits without a result, or finishes FAILED / non-zero, is a breach"),
    Breaker(
        name="run.controller_exit", layer="run", kind="invariant", can_disable=False,
        actions=("stop",), default_action="stop", native="controller: in_process",
        owner="yeto.launch.pool.controller",
        doc="the in-process rollout controller stopped driving the pools: a breach"),
)
