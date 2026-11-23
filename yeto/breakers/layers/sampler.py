"""Breakers of the sampler layer (the engine sampler process, its agentic round
and its store dispenser client: ``yeto.rl.engine.*``).

Every entry here describes a limit the sampler enforces today; the sampler reads
the resolved values through :mod:`yeto.rl.engine.sampler_breakers`. The spec
aliases are the sampler spec's native keys (``engine_sampler_entry`` module doc),
so a spec without a ``breakers`` section resolves to exactly what the sampler read
before. The defaults are literals (the runtime modules import numpy through the
sample buffer); ``tests/test_breakers_sampler_layer.py`` compares each one with
its code constant.

The store dispenser's lease and redo limit used to be configured in three places
(the spec's ``prompts.dispense``, the pool config's ``sampler.dispense`` and the
pool CLI's ``--dispense-*`` flags); each is one entry with three aliases now.
"""

from __future__ import annotations

from typing import Any

from yeto.breakers.spec import Alias, Breaker, Param

# defaults (code constants of the runtime modules; the layer test checks each)
LOAD_FAILURES_N = 3  # EngineSamplerConfig.max_consecutive_load_failures
MAX_READ_OUTAGE_S = 1800.0  # sampler_loop.DEFAULT_MAX_READ_OUTAGE_S
MAX_DAMAGED_LATEST_READS = 120  # sampler_loop.DEFAULT_MAX_DAMAGED_LATEST_READS
WRITE_DEFER_MAX_S = 600.0  # engine_sampler.DEFAULT_WRITE_DEFER_MAX_S
MAX_UNREAD_S = 60.0  # incarnation_lease.DEFAULT_MAX_UNREAD_S
RELOAD_MAX_DEFER_S = 600.0  # engine_sampler.DEFAULT_RELOAD_MAX_DEFER_S
ABANDON_GRACE_S = 120.0  # agentic_round.ABANDON_GRACE_S
MIN_SAMPLES = 2  # agentic_round.round_factory
INFRA_RETRIES = 2  # agentic_round.DEFAULT_INFRA_RETRIES
TRIAL_WALL_CLOCK_S = 7200.0  # yeto.rl.agentic.types.AgentBudget.wall_clock_s
DISPENSE_MAX_REDO = 2  # store_dispense.DEFAULT_MAX_REDO
DISPENSE_LEASE_S = 900.0  # store_dispense.DEFAULT_LEASE_S
VRAM_HIGH_FRAC = 0.97  # vram_guard.DEFAULT_HIGH_FRAC
VRAM_REALERT_S = 600.0  # vram_guard.DEFAULT_REALERT_S
INFRA_STREAK_N = 5  # infra_streak.DEFAULT_MAX_N
INFRA_STREAK_COOLDOWN_S = 300.0  # infra_streak.DEFAULT_COOLDOWN_S
OWNER = "yeto.rl.engine"


# spec numbers convert as the sampler converts them (a null never reaches to_canonical)
_float, _int = float, int


def _spec(path: str, param: str | None, **kw: Any) -> Alias:
    return Alias("spec", path, param, **kw)


def _backlog_groups(value: Any) -> int:
    """A spec's ``max_backlog_groups`` as the sampler runs with it: ``"auto"`` is the
    cap ``yeto run up`` derived, else the fallback (``sample_roles.resolve_auto_backlog``,
    a light module, imported when a spec is read)."""
    from yeto.rl.engine.sample_roles import resolve_auto_backlog  # noqa: PLC0415

    return int(resolve_auto_backlog(value)[0])


BREAKERS: tuple[Breaker, ...] = (
    Breaker(
        name="sampler.load_failures", layer="sampler", kind="breaker",
        params=(Param("max_consecutive_n", int, LOAD_FAILURES_N, unit="n", lo=1,
                      doc="failed loads or post-load checks in a row"),),
        actions=("stop",), default_action="stop", can_disable=False,
        aliases=(_spec("max_consecutive_load_failures", "max_consecutive_n", to_canonical=_int),),
        legacy_events=("sampler_load_check_failed", "SamplerLoadFatal"), gauge="count",
        owner=f"{OWNER}.sampler_checks",
        doc="the sampler exits (SamplerLoadFatal) after this many failed weight loads in a row"),
    Breaker(
        name="sampler.store_read_outage", layer="sampler", kind="breaker",
        params=(Param("limit_s", float, MAX_READ_OUTAGE_S, unit="s", lo=0, nullable=True,
                      doc="longest a store poll (LATEST, release download, buffer index) may keep failing"),
                Param("max_damaged_latest_n", int, MAX_DAMAGED_LATEST_READS, unit="n", lo=1,
                      doc="reads in a row of a LATEST that is no release record")),
        actions=("stop",), default_action="stop", can_disable=False,
        aliases=(_spec("max_read_outage_s", "limit_s", to_canonical=_float),
                 _spec("max_damaged_latest_reads", "max_damaged_latest_n", to_canonical=_int)),
        depends_on=("D20",), legacy_events=("rl_sampler_store_read_failed", "StoreReadOutage"),
        gauge="elapsed", owner=f"{OWNER}.sampler_loop",
        doc="the sampler exits (StoreReadOutage) when the store has not answered a poll for limit_s; "
            "null: no bound"),
    Breaker(
        name="sampler.write_defer", layer="sampler", kind="guard",
        params=(Param("limit_s", float, WRITE_DEFER_MAX_S, unit="s", lo=0,
                      doc="longest group writes wait for a store that does not say whose the task is"),),
        actions=("drop",), default_action="drop", can_disable=False,
        aliases=(_spec("write_defer_max_s", "limit_s", to_canonical=_float),),
        depends_on=("D10",), legacy_events=("rl_sampler_write_deferred",), gauge="elapsed",
        owner=f"{OWNER}.engine_sampler",
        doc="past the limit a waiting group is given up and its task released (not counted as a redo)"),
    Breaker(
        name="sampler.incarnation_lease", layer="sampler", kind="guard",
        params=(Param("max_unread_s", float, MAX_UNREAD_S, unit="s", lo=0, nullable=True,
                      doc="longest the incarnation lease may stay unread before group writes wait"),),
        actions=("pause",), default_action="pause",
        aliases=(_spec("incarnation_lease", None, off_values=(False, None), on_values=(True,)),
                 _spec("incarnation_lease.max_unread_s", "max_unread_s", to_canonical=_float)),
        legacy_events=("rl_sampler_write_deferred",), gauge="elapsed",
        owner=f"{OWNER}.incarnation_lease",
        doc="group writes wait while the lease is unread for longer than max_unread_s (null: never); "
            "off (incarnation_lease: false) also turns the incarnation fence off"),
    Breaker(
        name="sampler.drain", layer="sampler", kind="guard",
        params=(Param("timeout_s", float, None, unit="s", lo=0, nullable=True,
                      doc="agentic.drain_timeout_s: longest a drain-mode weight switch waits on running trials"),
                Param("reload_max_defer_s", float, RELOAD_MAX_DEFER_S, unit="s", lo=0, nullable=True,
                      doc="longest a ready release waits on running trials (> 0); also caps the drain"),
                Param("abandon_grace_s", float, ABANDON_GRACE_S, unit="s", lo=0,
                      doc="after the cut, the longest wait for a trial that ignores the cancel")),
        actions=("drop",), default_action="drop", can_disable=False,
        aliases=(_spec("agentic.drain_timeout_s", "timeout_s", to_canonical=_float),
                 _spec("reload_max_defer_s", "reload_max_defer_s", to_canonical=_float),
                 _spec("agentic.abandon_grace_s", "abandon_grace_s", to_canonical=_float)),
        depends_on=("D1", "D3", "D4"), legacy_events=("rl_agentic_drain",), gauge="elapsed",
        owner=f"{OWNER}.agentic_round",
        doc="a drain-mode weight switch cancels the trials still running at min(timeout_s, "
            "reload_max_defer_s) (agentic_round.effective_drain_timeout); both null: no cut"),
    Breaker(
        name="sampler.min_samples", layer="sampler", kind="guard",
        params=(Param("min_n", int, MIN_SAMPLES, unit="n", lo=1, doc="kept trials a group needs"),),
        actions=("drop",), default_action="drop", can_disable=False,
        aliases=(_spec("agentic.min_samples", "min_n", to_canonical=_int),),
        depends_on=("D16",), legacy_events=("rl_agentic_group",), gauge="count",
        owner=f"{OWNER}.agentic_round",
        doc="a group with fewer kept trials is not written; its task goes out again (counted as a redo)"),
    Breaker(
        name="sampler.infra_retries", layer="sampler", kind="retry_budget",
        params=(Param("max_n", int, INFRA_RETRIES, unit="n", lo=0,
                      doc="runs again of a trial that failed on infrastructure before the policy ran"),),
        actions=("drop",), default_action="drop", can_disable=False,
        aliases=(_spec("agentic.infra_retries", "max_n", to_canonical=_int),),
        depends_on=("D4", "D9"), legacy_events=("rl_agentic_trial_retry",), gauge="count",
        owner=f"{OWNER}.agentic_round",
        doc="per trial slot, at once (no backoff); 0: no retry. A spent budget loses the trial"),
    Breaker(
        name="sampler.trial_wall_clock", layer="sampler", kind="guard",
        params=(Param("limit_s", float, TRIAL_WALL_CLOCK_S, unit="s", lo=0,
                      doc="outer cap of one trial (> agent_timeout_s)"),),
        actions=("drop",), default_action="drop", can_disable=False,
        aliases=(_spec("agentic.task_defaults.wall_clock_s", "limit_s", to_canonical=_float),),
        depends_on=("D2", "D3", "D4"), legacy_events=("rl_agentic_group",), gauge="elapsed",
        owner="yeto.rl.agentic.types",
        doc="a trial past it ends ABORTED (wall_clock_cap), dropped and not retried; a task row's own "
            "wall_clock_s overrides it for that task"),
    Breaker(
        name="sampler.dispense_redo", layer="sampler", kind="retry_budget",
        params=(Param("max_n", int, DISPENSE_MAX_REDO, unit="n",
                      doc="infra / expired attempts of a slot before it is settled exhausted"),),
        actions=("drop",), default_action="drop", can_disable=False,
        aliases=(_spec("prompts.dispense.max_redo", "max_n", to_canonical=_int),
                 Alias("pool_dispense", "max_redo", "max_n", to_canonical=_int),
                 Alias("pool_cli_flag", "--dispense-max-redo", "max_n", to_canonical=_int)),
        depends_on=("D8", "D9"), legacy_events=("rl_dispense_settle",), gauge="count",
        owner=f"{OWNER}.store_dispense",
        doc="the first sampler's value is kept in plan.json and every later sampler adopts it"),
    Breaker(
        name="sampler.dispense_lease", layer="sampler", kind="guard",
        params=(Param("lease_s", float, DISPENSE_LEASE_S, unit="s", lo=0,
                      doc="claim lease (> 0), renewed every lease / 3"),),
        actions=("drop",), default_action="drop", can_disable=False,
        aliases=(_spec("prompts.dispense.lease_s", "lease_s", to_canonical=_float),
                 Alias("pool_dispense", "lease_s", "lease_s", to_canonical=_float),
                 Alias("pool_cli_flag", "--dispense-lease-s", "lease_s", to_canonical=_float)),
        depends_on=("D10",), legacy_events=("rl_dispense_expired", "rl_dispense_lease"), gauge="elapsed",
        owner=f"{OWNER}.store_dispense",
        doc="a claim not renewed for lease_s ends expired (counted as a redo) and is dispensed again. "
            "Each sampler uses its own spec's lease (plan.json keeps the first one for reference); a "
            "row's metadata.dispense_lease_s overrides it for that task"),
    Breaker(
        name="sampler.vram_guard", layer="sampler", kind="alert",
        params=(Param("high_frac_ratio", float, VRAM_HIGH_FRAC, unit="ratio", lo=0, hi=1,
                      doc="a card above this used fraction alerts (> 0)"),
                Param("realert_s", float, VRAM_REALERT_S, unit="s", lo=0, doc="alert again while high"),
                Param("interval_s", float, None, unit="s", lo=0, nullable=True,
                      doc="card sample period (null: the telemetry default, 60 s)")),
        actions=("warn",), default_action="warn", can_disable=False,
        aliases=(_spec("vram_guard.high_frac", "high_frac_ratio", to_canonical=_float),
                 _spec("vram_guard.realert_s", "realert_s", to_canonical=_float),
                 _spec("vram_guard.interval_s", "interval_s", to_canonical=_float)),
        legacy_events=("rl_sampler_vram_high", "rl_sampler_vram_ok"), gauge="ratio",
        owner=f"{OWNER}.vram_guard",
        doc="rl_sampler_vram_high / rl_sampler_vram_ok on the tape of a card sampler"),
    Breaker(
        name="sampler.infra_streak", layer="sampler", kind="breaker",
        params=(Param("max_n", int, INFRA_STREAK_N, unit="n", lo=1,
                      doc="agentic groups in a row lost to infrastructure (below_min_samples)"),
                Param("cooldown_s", float, INFRA_STREAK_COOLDOWN_S, unit="s", lo=0,
                      doc="action pause: claim again after this long without an infra-ok group")),
        actions=("warn", "pause"), default_action="warn",
        aliases=(_spec("agentic.infra_streak.enabled", None, off_values=(False,), on_values=(True,)),
                 _spec("agentic.infra_streak.max_n", "max_n", to_canonical=_int),
                 _spec("agentic.infra_streak.cooldown_s", "cooldown_s", to_canonical=_float),
                 _spec("agentic.infra_streak.action", "action")),
        depends_on=("D9",), gauge="count", owner=f"{OWNER}.infra_streak",
        doc="warn: rl_breaker only. pause: the store dispenser claims no new task (trials in flight "
            "go on) until a group is written or cooldown_s has passed"),
    Breaker(
        name="sampler.loop_guard", layer="sampler", kind="guard",
        params=(Param("max_period_n", int, 64, unit="n", lo=1), Param("repeats_n", int, 8, unit="n", lo=2)),
        actions=("drop",), default_action="drop", default_enabled=False,
        owner=f"{OWNER}.loop_guard",
        doc="cuts a looping completion (finish loop_abort); off by default, set only by code "
            "(EngineSamplerConfig.loop_guard); thresholds not measured"),
    Breaker(
        name="sampler.token_alignment", layer="sampler", kind="invariant", can_disable=False,
        native="always on (yeto.rl.agentic.finish)", legacy_events=("rl_agentic_trial_lost",),
        owner="yeto.rl.agentic.finish",
        doc="a trial whose recorded calls do not align with its tokens ends ABORTED (token_alignment)"),
    Breaker(
        name="sampler.compaction_startup", layer="sampler", kind="invariant", can_disable=False,
        native="spec agentic.compaction", owner="yeto.rl.compaction",
        doc="a compaction config whose budgets do not fit max_model_len refuses the sampler's start"),
    Breaker(name="sampler.max_prompt_len", layer="sampler", kind="semantic", native="spec agentic.max_prompt_len",
            owner=f"{OWNER}.agentic_round", doc="a first prompt longer than this ends the trial ABORTED"),
    Breaker(name="sampler.agent_timeout", layer="sampler", kind="semantic",
            native="spec agentic.task_defaults.agent_timeout_s", depends_on=("D2",), owner="yeto.rl.agentic.types",
            doc="a trial past it ends POLICY_LIMIT and is scored (kept)"),
    Breaker(name="sampler.dispense_passes", layer="sampler", kind="semantic", native="spec prompts.dispense.passes",
            depends_on=("D8",), owner=f"{OWNER}.store_dispense", doc="passes over the data"),
    Breaker(name="sampler.max_backlog_groups", layer="sampler", kind="semantic", native="spec max_backlog_groups",
            params=(Param("groups_n", int, None, unit="n", lo=1, nullable=True,
                          doc="the cap; \"auto\" in the spec shows the derived value"),),
            aliases=(_spec("max_backlog_groups", "groups_n", to_canonical=_backlog_groups, materialize=False),),
            depends_on=("D19",), owner=f"{OWNER}.sampler_loop",
            doc="unread groups at which the sampler pauses generating (rl_sampler_backlog_full); "
                "\"auto\": derived by yeto run up from the release cadence (sample_roles)"),
    Breaker(name="sampler.max_compactions", layer="sampler", kind="semantic",
            native="spec agentic.compaction.max_compactions", owner="yeto.rl.compaction",
            doc="compactions of one trial before ContextFull (POLICY_LIMIT, kept)"),
)
