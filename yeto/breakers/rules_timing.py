"""Dependency rules between timeouts that must nest (D1-D7, D9, D10, D13, D14, D20).

A trial runs inside its wall clock, inside the sandbox TTL; a group takes up to
a trial plus its infra retries plus a drain, and the trainer's sample wait must
outlast that (and a declared outage); the launcher's stall timer must outlast
the sample wait, and the run monitor's silence backstop the stall timer. Each
rule checks one link of that chain (:mod:`yeto.breakers.rules`).
"""

from __future__ import annotations

import math
from collections.abc import Iterator

from yeto.breakers.rules import Ctx, finding, fmt_s, rule
from yeto.breakers.spec import Finding

VERIFIER_MIN_S = 60.0    # D2: the least time a verifier needs after the agent timeout
VERIFIER_GOOD_S = 300.0  # D2: headroom below this is a warning
# D9: one failed sandbox start (a 502 or a create error) before any retry backoff;
# a gateway that answers at all answers in seconds
INFRA_ATTEMPT_S = 20.0
MODAL_MAX_TIMEOUT_S = 86400.0  # D13: the longest timeout Modal accepts for a function
FIRST_GROUP_MIN_S = 600.0  # D14: a sampler's first group after it is ready (warm-up + one round)
# D5: a buffer trainer emits rl_sample_wait every max(60 s, stall / 4) while it waits for groups
# (yeto.rl.engine.trainer_breakers.wait_heartbeat_s; a test keeps the two equal)
WAIT_HEARTBEAT_MIN_S = 60.0
WAIT_HEARTBEAT_STALL_FRACTION = 0.25


def wait_heartbeat_s(stall: float) -> float:
    """Seconds between a waiting buffer trainer's heartbeats under a stall clock of ``stall``."""
    return max(WAIT_HEARTBEAT_MIN_S, stall * WAIT_HEARTBEAT_STALL_FRACTION)


def drain_cut(ctx: Ctx) -> float | None:
    """The effective drain cut of a drain-mode pool: min(``timeout_s``,
    ``reload_max_defer_s``), either one unset (None: no cut, or not a drain)."""
    if ctx.pool is None or not ctx.pool.drain or not ctx.enabled("sampler.drain"):
        return None
    known = [v for v in (ctx.number("sampler.drain.timeout_s"), ctx.number("sampler.drain.reload_max_defer_s"))
             if v is not None]
    return min(known) if known else None


def reach_check_s(ctx: Ctx) -> float:
    """What one failed reachability self-check costs (AgentENV only; 0 when it is off)."""
    if ctx.pool is None or ctx.pool.provider != "agentenv" or not ctx.enabled("harness.reach_check"):
        return 0.0
    attempts = ctx.number("harness.reach_check.attempts_n") or 0.0
    per_try = (ctx.number("harness.reach_check.timeout_s") or 0.0) + (ctx.number("harness.reach_check.interval_s")
                                                                         or 0.0)
    return attempts * per_try


def trial_infra_s(ctx: Ctx, *, setup_retries: int | None = None) -> float:
    """How long one trial attempt takes to fail on infrastructure: every harness
    setup attempt (a failed start, plus the reach check) and the backoff between them."""
    retries = int(ctx.number("harness.setup_retries.max_n") or 0) if setup_retries is None else setup_retries
    first = ctx.number("harness.setup_retries.backoff_initial_s") or 0.0
    cap = ctx.number("harness.setup_retries.backoff_max_s")
    backoff = sum(min(cap if cap is not None else math.inf, first * 2 ** k) for k in range(retries))
    return (retries + 1) * (INFRA_ATTEMPT_S + reach_check_s(ctx)) + backoff


def longest_group_s(ctx: Ctx) -> float | None:
    """The longest a legitimate agentic group takes: a trial's wall clock, its infra
    retries, and a drain cut (None: not an agentic pool, or no wall clock)."""
    if ctx.pool is None or not ctx.pool.agentic:
        return None
    wall = ctx.number("sampler.trial_wall_clock.limit_s", on=False)
    if wall is None:
        return None
    retries = int(ctx.number("sampler.infra_retries.max_n") or 0)
    return wall + retries * trial_infra_s(ctx) + (drain_cut(ctx) or 0.0)


@rule("D1", "drain_cut", scope="pool")
def drain_cut_rule(ctx: Ctx) -> Iterator[Finding]:
    """The effective drain cut is min(drain_timeout_s, reload_max_defer_s): a warning
    when drain_timeout_s is larger (it never takes effect) or a compacting sampler
    drains, else the cut as a note."""
    cut = drain_cut(ctx)
    if cut is None:
        return
    timeout = ctx.number("sampler.drain.timeout_s")
    defer = ctx.number("sampler.drain.reload_max_defer_s")
    values = {"sampler.drain.timeout_s": timeout, "sampler.drain.reload_max_defer_s": defer, "cut_s": cut}
    from yeto.rl.engine.agentic_round import COMPACTION_DRAIN_FLOOR_S, drain_cut_warning  # noqa: PLC0415

    warning = None if defer is None else drain_cut_warning(timeout, defer, compaction=ctx.pool.compaction)
    if warning is not None:
        suggested = ({"sampler.drain.timeout_s": defer} if timeout is None or timeout > defer
                     else {"sampler.drain.reload_max_defer_s": max(defer, COMPACTION_DRAIN_FLOOR_S)})
        yield finding("D1", "warn", ctx.where(f"sampler.drain: {warning}"), items=("sampler.drain",), values=values,
                      suggested=suggested)
        return
    yield finding("D1", "info", ctx.where(
        f"sampler.drain: the effective drain cut is {fmt_s(cut)} (min of drain_timeout_s "
        f"{'unset' if timeout is None else f'{timeout:g}'}, reload_max_defer_s "
        f"{'unset' if defer is None else f'{defer:g}'}): trials running longer after a release are lost"),
        items=("sampler.drain",), values=values)


@rule("D2", "verifier_headroom", scope="pool")
def verifier_headroom(ctx: Ctx) -> Iterator[Finding]:
    """trial_wall_clock > agent_timeout_s + 60 s; a warning below 300 s of headroom."""
    if ctx.pool is None or not ctx.pool.agentic or ctx.pool.agent_timeout_s is None:
        return
    wall = ctx.number("sampler.trial_wall_clock.limit_s", on=False)
    if wall is None:
        return
    agent = ctx.pool.agent_timeout_s
    headroom = wall - agent
    if headroom >= VERIFIER_GOOD_S:
        return
    level = "error" if headroom <= VERIFIER_MIN_S else "warn"
    yield finding("D2", level, ctx.where(
        f"sampler.trial_wall_clock {wall:g} s leaves {headroom:g} s after agentic.task_defaults.agent_timeout_s "
        f"{agent:g} s (needed: > {VERIFIER_MIN_S:g} s, good: >= {VERIFIER_GOOD_S:g} s): a trial that reaches the "
        "agent timeout ends ABORTED (wall_clock_cap, dropped) instead of POLICY_LIMIT (scored) because the "
        "verifier has no time to score it; raise agentic.task_defaults.wall_clock_s"),
        items=("sampler.trial_wall_clock",), values={"wall_clock_s": wall, "agent_timeout_s": agent},
        suggested={"sampler.trial_wall_clock.limit_s": agent + VERIFIER_GOOD_S})


@rule("D3", "sandbox_ttl", scope="pool")
def sandbox_ttl(ctx: Ctx) -> Iterator[Finding]:
    """The sandbox TTL outlasts a trial's wall clock plus the abandon grace."""
    if ctx.pool is None or not ctx.pool.agentic:
        return
    wall = ctx.number("sampler.trial_wall_clock.limit_s", on=False)
    grace = ctx.number("sampler.drain.abandon_grace_s", on=False) or 0.0
    ttl = ctx.number("harness.sandbox_ttl.limit_s", on=False)
    if wall is None:
        return
    if ttl is not None:
        if ttl > wall + grace:
            return
        yield finding("D3", "error", ctx.where(
            f"harness.sandbox_ttl {ttl:g} s <= sampler.trial_wall_clock {wall:g} s + abandon_grace_s {grace:g} s: "
            "the provider can reclaim a sandbox under a live trial; raise harness.sandbox_ttl.limit_s or leave it "
            "unset (AgentENV then derives it per trial)"),
            items=("harness.sandbox_ttl", "sampler.trial_wall_clock", "sampler.drain"),
            values={"ttl_s": ttl, "wall_clock_s": wall, "abandon_grace_s": grace},
            suggested={"harness.sandbox_ttl.limit_s": wall + grace + VERIFIER_GOOD_S})
        return
    margin = ctx.number("harness.sandbox_ttl.margin_s", on=False)
    if ctx.pool.provider != "agentenv" or margin is None or margin > grace:
        return
    yield finding("D3", "error", ctx.where(
        f"harness.sandbox_ttl.margin_s {margin:g} s <= sampler.drain abandon_grace_s {grace:g} s: an AgentENV "
        f"sandbox lives the trial's wall clock plus the margin, so a trial abandoned after a drain can lose its "
        "sandbox; raise margin_s"),
        items=("harness.sandbox_ttl", "sampler.drain"), values={"margin_s": margin, "abandon_grace_s": grace},
        suggested={"harness.sandbox_ttl.margin_s": grace + VERIFIER_GOOD_S})


@rule("D4", "sample_wait", scope="pool")
def sample_wait(ctx: Ctx) -> Iterator[Finding]:
    """trainer.sample_wait >= max(longest legitimate group time, outage_tolerance_s)."""
    wait = ctx.number("trainer.sample_wait.timeout_s")
    if wait is None:
        return
    longest = longest_group_s(ctx)
    tolerance = ctx.facts.outage_tolerance_s
    need = max(v for v in (longest, tolerance, 0.0) if v is not None)
    if wait >= need or need == 0:
        return
    parts = []
    if longest is not None:
        parts.append(f"the longest group takes about {fmt_s(longest)} (trial wall clock + infra retries + drain cut)")
    if tolerance is not None:
        parts.append(f"breakers.deps.outage_tolerance_s is {fmt_s(tolerance)}")
    yield finding("D4", "warn", ctx.where(
        f"trainer.sample_wait {fmt_s(wait)} < {fmt_s(need)}: {'; '.join(parts)}; a round that sees no new group "
        "that long ends the trainer (SampleBufferRolloutError); raise --rl-sample-wait-s"),
        items=("trainer.sample_wait", "sampler.trial_wall_clock", "sampler.infra_retries", "sampler.drain"),
        values={"sample_wait_s": wait, "longest_group_s": longest, "outage_tolerance_s": tolerance},
        suggested={"trainer.sample_wait.timeout_s": math.ceil(need)})


@rule("D5", "stall_covers_sample_wait")
def stall_covers_sample_wait(ctx: Ctx) -> Iterator[Finding]:
    """launch.stall >= trainer.sample_wait, else the sample wait never fires; unless the
    trainer reads the sample buffer: it then emits an island event every
    :func:`wait_heartbeat_s` while it waits, and two of them fit in the stall window
    (one lost log line is not a hang)."""
    stall = ctx.number("launch.stall.timeout_s")
    wait = ctx.number("trainer.sample_wait.timeout_s")
    if stall is None or wait is None or stall >= wait:
        return
    trainer = ctx.facts.trainer
    if trainer is not None and trainer.buffer_trainer and stall >= 2 * wait_heartbeat_s(stall):
        return
    yield finding("D5", "warn",
                  f"launch.stall {fmt_s(stall)} < trainer.sample_wait {fmt_s(wait)}: the buffer reader emits no island "
                  "event while it waits for groups, so the launcher's stall timer ends the run (RunStalled, exit 6) "
                  "before the sample wait can; raise --rl-stall-timeout to at least --rl-sample-wait-s",
                  items=("launch.stall", "trainer.sample_wait"), values={"stall_s": stall, "sample_wait_s": wait},
                  suggested={"launch.stall.timeout_s": wait})


def silence_limit_s(ctx: Ctx, stall: float) -> float | None:
    """``run.trainer_silence``: its explicit limit, else max(stall_factor_n x stall, floor_s)."""
    explicit = ctx.number("run.trainer_silence.limit_s")
    if explicit is not None:
        return explicit
    factor = ctx.number("run.trainer_silence.stall_factor_n")
    floor = ctx.number("run.trainer_silence.floor_s")
    known = [v for v in ((None if factor is None else factor * stall), floor) if v is not None]
    return max(known) if known else None


@rule("D6", "silence_backstop")
def silence_backstop(ctx: Ctx) -> Iterator[Finding]:
    """run.trainer_silence >= launch.stall when the stall timer is on; with it off,
    the silence backstop must be on with an explicit limit."""
    if not ctx.facts.run_up or not ctx.registered("launch.stall") or not ctx.registered("run.trainer_silence"):
        return
    if not ctx.enabled("launch.stall"):
        explicit = (ctx.number("run.trainer_silence.limit_s") is not None
                    and ctx.source("run.trainer_silence.limit_s") not in (None, "default"))
        if ctx.enabled("run.trainer_silence") and explicit:
            return
        yield finding("D6", "error",
                      "--rl-stall-timeout 0 disables the trainer's hang detection (launch.stall) and the run monitor's "
                      "silence backstop is derived from it (4x this timeout): run up refuses it; set "
                      "--rl-stall-timeout, or set breakers.run.trainer_silence.limit_s explicitly",
                      items=("launch.stall", "run.trainer_silence"),
                      suggested={"run.trainer_silence.limit_s": 3600})
        return
    stall = ctx.number("launch.stall.timeout_s")
    silence = None if stall is None or not ctx.enabled("run.trainer_silence") else silence_limit_s(ctx, stall)
    if stall is None or silence is None or silence >= stall:
        return
    yield finding("D6", "error",
                  f"run.trainer_silence {fmt_s(silence)} < launch.stall {fmt_s(stall)}: the run monitor would stop a "
                  "trainer the launcher still waits for; raise breakers.run.trainer_silence.limit_s",
                  items=("run.trainer_silence", "launch.stall"), values={"silence_s": silence, "stall_s": stall},
                  suggested={"run.trainer_silence.limit_s": 4 * stall})


@rule("D7", "starvation_order")
def starvation_order(ctx: Ctx) -> Iterator[Finding]:
    """starvation alert_s < stop_s; which of starvation stop and sample wait fires first."""
    if not ctx.facts.run_up:
        return
    alert, stop = ctx.number("run.starvation.alert_s"), ctx.number("run.starvation.stop_s")
    if alert is not None and stop is not None and alert >= stop:
        yield finding("D7", "warn",
                      f"run.starvation: alert_s {fmt_s(alert)} >= stop_s {fmt_s(stop)}: the run stops without an "
                      "alert first; set alert_min below stop_min",
                      items=("run.starvation",), values={"alert_s": alert, "stop_s": stop},
                      suggested={"run.starvation.alert_s": stop / 2})
    wait = ctx.number("trainer.sample_wait.timeout_s")
    if stop is None or wait is None:
        return
    first = ("the starvation stop (STOP flag: the run ends cleanly, pools down)" if stop < wait else
             "the trainer's sample wait (SampleBufferRolloutError: the trainer exits)")
    yield finding("D7", "info",
                  f"with no group arriving, {first} fires first: run.starvation.stop_s {fmt_s(stop)}, "
                  f"trainer.sample_wait {fmt_s(wait)}",
                  items=("run.starvation", "trainer.sample_wait"), values={"stop_s": stop, "sample_wait_s": wait})


def slot_outage_s(ctx: Ctx, *, setup_retries: int | None = None) -> float | None:
    """How long a gateway outage one dispense slot rides out before it is exhausted:
    (dispense_redo + 1) x (infra_retries + 1) x one trial's infra failure time."""
    redo = ctx.number("sampler.dispense_redo.max_n", on=False)
    if redo is None:
        return None
    retries = int(ctx.number("sampler.infra_retries.max_n") or 0)
    return (int(redo) + 1) * (retries + 1) * trial_infra_s(ctx, setup_retries=setup_retries)


@rule("D9", "slot_outage", scope="pool")
def slot_outage(ctx: Ctx) -> Iterator[Finding]:
    """The outage one dispense slot survives, against outage_tolerance_s when declared."""
    if ctx.pool is None or not ctx.pool.agentic or ctx.pool.dispense is None:
        return
    survives = slot_outage_s(ctx)
    if survives is None:
        return
    redo = int(ctx.number("sampler.dispense_redo.max_n", on=False) or 0)
    retries = int(ctx.number("sampler.infra_retries.max_n") or 0)
    setup = int(ctx.number("harness.setup_retries.max_n") or 0)
    shape = f"dispense_redo {redo}, infra_retries {retries}, setup_retries {setup}"
    values = {"slot_outage_s": round(survives, 1), "outage_tolerance_s": ctx.facts.outage_tolerance_s}
    tolerance = ctx.facts.outage_tolerance_s
    if tolerance is None or survives >= tolerance:
        yield finding("D9", "info", ctx.where(
            f"one dispense slot rides out about {survives / 60:.0f} min of sandbox gateway outage ({shape}) before "
            "it is exhausted" + ("" if tolerance is not None else
                                 "; declare breakers.deps.outage_tolerance_s to check it against a target")),
            items=("sampler.dispense_redo", "sampler.infra_retries", "harness.setup_retries"), values=values)
        return
    need = next((n for n in range(setup + 1, 31) if (slot_outage_s(ctx, setup_retries=n) or 0) >= tolerance), None)
    advice = (f"set HARBOR_SETUP_RETRIES (harness.setup_retries.max_n) to {need} (exponential backoff)"
              if need is not None else "raise prompts.dispense.max_redo and HARBOR_SETUP_RETRIES")
    yield finding("D9", "warn", ctx.where(
        f"one dispense slot rides out about {survives / 60:.0f} min of sandbox gateway outage ({shape}), below "
        f"breakers.deps.outage_tolerance_s {fmt_s(tolerance)}: such an outage burns the task list (every slot "
        f"exhausted, the sampler ends with FINISHED); {advice}"),
        items=("sampler.dispense_redo", "sampler.infra_retries", "harness.setup_retries"), values=values,
        suggested={"harness.setup_retries.max_n": need} if need is not None else {})


@rule("D10", "lease_vs_write_defer", scope="pool")
def lease_vs_write_defer(ctx: Ctx) -> Iterator[Finding]:
    """dispense_lease > write_defer; a note on a lease that delays re-dispensing."""
    if ctx.pool is None or ctx.pool.dispense is None:
        return
    lease = ctx.number("sampler.dispense_lease.lease_s", on=False)
    defer = ctx.number("sampler.write_defer.limit_s")
    if lease is None:
        return
    if defer is not None and lease <= defer:
        yield finding("D10", "warn", ctx.where(
            f"sampler.dispense_lease {fmt_s(lease)} <= sampler.write_defer {fmt_s(defer)}: during a store outage "
            "the claim expires while the group's write is still deferred, and the task is dispensed again (an "
            "expired attempt counts toward max_redo); raise prompts.dispense.lease_s above write_defer_max_s"),
            items=("sampler.dispense_lease", "sampler.write_defer"), values={"lease_s": lease, "write_defer_s": defer},
            suggested={"sampler.dispense_lease.lease_s": max(900.0, defer + 300.0)})
    from yeto.rl.engine.store_dispense import lease_warning  # noqa: PLC0415

    note = lease_warning(lease)
    if note is not None:
        yield finding("D10", "info", ctx.where(note), items=("sampler.dispense_lease",), values={"lease_s": lease},
                      suggested={"sampler.dispense_lease.lease_s": 900.0})


@rule("D13", "provider_timeout")
def provider_timeout(ctx: Ctx) -> Iterator[Finding]:
    """launch.provider_timeout >= run.wall_clock (a Modal trainer)."""
    if not ctx.facts.run_up or ctx.facts.trainer is None or not ctx.facts.trainer.modal:
        return
    timeout = ctx.number("launch.provider_timeout.timeout_s", on=False)
    hours = ctx.number("run.wall_clock.hours")
    if timeout is None or hours is None or timeout >= hours * 3600:
        return
    values = {"provider_timeout_s": timeout, "wall_clock_h": hours}
    items = ("launch.provider_timeout", "run.wall_clock")
    if timeout >= MODAL_MAX_TIMEOUT_S:
        yield finding("D13", "info",
                      f"run.wall_clock {hours:g} h is longer than Modal's longest function timeout "
                      f"({fmt_s(MODAL_MAX_TIMEOUT_S)}): the trainer function is ended at least every "
                      f"{fmt_s(timeout)} and the run relies on --modal-retries re-running it (it resumes from its "
                      "last cut)", items=items, values=values)
        return
    yield finding("D13", "warn",
                  f"launch.provider_timeout {fmt_s(timeout)} < run.wall_clock {hours:g} h: Modal ends the trainer "
                  "function at its timeout before the run's wall clock; raise --modal-timeout-s",
                  items=items, values=values,
                  suggested={"launch.provider_timeout.timeout_s": int(min(hours * 3600, MODAL_MAX_TIMEOUT_S))})


@rule("D14", "first_group", scope="pool")
def first_group(ctx: Ctx) -> Iterator[Finding]:
    """run.first_group >= pool.start_timeout + the first group's minimum time."""
    if not ctx.facts.run_up:
        return
    wait = ctx.number("run.first_group.timeout_s")
    start = ctx.number("pool.start_timeout.limit_s")
    if wait is None or start is None or wait >= start + FIRST_GROUP_MIN_S:
        return
    yield finding("D14", "warn", ctx.where(
        f"run.first_group {fmt_s(wait)} < pool.start_timeout {fmt_s(start)} + {fmt_s(FIRST_GROUP_MIN_S)} (a "
        "sampler's first group once ready): run up gives up waiting for the first group while the pool still "
        "lets a node start; raise first_group.timeout_s"),
        items=("run.first_group", "pool.start_timeout"), values={"first_group_s": wait, "start_timeout_s": start},
        suggested={"run.first_group.timeout_s": start + 2 * FIRST_GROUP_MIN_S})


@rule("D20", "store_read_outage", scope="pool")
def store_read_outage(ctx: Ctx) -> Iterator[Finding]:
    """sampler.store_read_outage >= outage_tolerance_s, when declared."""
    tolerance = ctx.facts.outage_tolerance_s
    limit = ctx.number("sampler.store_read_outage.limit_s")
    if tolerance is None or limit is None or limit >= tolerance:
        return
    yield finding("D20", "warn", ctx.where(
        f"sampler.store_read_outage {fmt_s(limit)} < breakers.deps.outage_tolerance_s {fmt_s(tolerance)}: a store "
        "read outage that long ends the sampler (StoreReadOutage); raise breakers.sampler.store_read_outage.limit_s"),
        items=("sampler.store_read_outage",), values={"limit_s": limit, "outage_tolerance_s": tolerance},
        suggested={"sampler.store_read_outage.limit_s": tolerance})

