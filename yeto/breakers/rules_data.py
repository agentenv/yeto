"""Dependency rules on groups and tasks (D8, D15, D16, D19).

Whether the samplers can feed the trainer at all: enough dispense slots for the
groups the trainer trains (D8), a policy age the algorithm spec tolerates (D15),
a group the estimator can normalize (D16), and backlog caps that hold the groups
the trainer reads between two releases (D19).
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

from yeto.breakers.rules import Ctx, finding, rule
from yeto.breakers.rules_facts import AUTO, GROUP_BASELINE_ESTIMATORS, Backlog, TrainerFacts
from yeto.breakers.spec import Finding

COVERAGE_MARGIN = 0.10  # D8: fewer spare slots than this share of the needed groups is a warning


@rule("D8", "dispense_coverage")
def dispense_coverage(ctx: Ctx) -> Iterator[Finding]:
    """Training rows x passes >= --total-steps x --rollout-batch-size; a warning below 10 % margin."""
    trainer = ctx.facts.trainer
    if trainer is None or not trainer.single_buffer_trainer or trainer.steps is None \
            or trainer.groups_per_step is None:
        return
    pools = [p for p in ctx.facts.pools if p.data_path == "r2"]
    if not pools or any(p.dispense is None or p.dispense.rows is None for p in pools):
        return  # a pool without a store dispenser, or rows not readable here: not checkable
    plans = {p.dispense.namespace: p.dispense for p in pools}  # pools of one namespace share its slots
    slots = sum(d.rows * d.passes for d in plans.values())
    need = trainer.steps * trainer.groups_per_step
    margin = (slots - need) / need
    if margin >= COVERAGE_MARGIN:
        return
    shape = " + ".join(f"{d.rows} rows x {d.passes} passes" for d in plans.values())
    rows = sum(d.rows for d in plans.values())
    passes = math.ceil(need * (1 + COVERAGE_MARGIN) / rows) if rows else None
    head = (f"dispense coverage: {shape} = {slots} slots for --total-steps {trainer.steps} x --rollout-batch-size "
            f"{trainer.groups_per_step} = {need} groups")
    if slots < need:
        message = (f"{head}: the dispenser ends before the trainer has its groups (the trainer then waits until "
                   "--rl-sample-wait-s ends it); raise prompts.dispense.passes, add rows, or lower --total-steps")
        level = "error"
    else:
        message = (f"{head} (margin {margin:.1%} < {COVERAGE_MARGIN:.0%}): every slot exhausted by infra failures "
                   "or a dropped group (below_min_samples) is a group the run never gets; raise "
                   "prompts.dispense.passes")
        level = "warn"
    yield finding("D8", level, message, items=("sampler.dispense_redo",),
                  values={"slots": slots, "groups": need, "margin": round(margin, 4)},
                  suggested={"prompts.dispense.passes": passes} if passes else {})


def policy_age_problem(age: int, tolerated: int, *, backend: str | None = None) -> str | None:
    """Why ``--rl-max-policy-age age`` is refused by an algorithm spec tolerating
    ``tolerated`` versions (None: allowed; 0 is no age limit). The rule the Miles
    island applies in its config translation, checked before any cloud resource."""
    if age == 0 or tolerated >= age:
        return None
    return (f"--rl-max-policy-age {age} needs an algorithm spec with "
            f"execution.max_policy_staleness >= {age} (and a TIS correction); the run's spec "
            f"tolerates {tolerated} (--rl-algorithm-spec). Start from a shipped spec, e.g. "
            "examples/k31/algorithm-spec.async-age2-tis.json (max_policy_staleness 2, TIS), "
            "or lower --rl-max-policy-age"
            + ("; with --rl-backend verl set correction.lower to 0.0 in a copy of it (verl TIS "
               "clamps the upper side only and refuses lower > 0)"
               if (backend or "miles") == "verl" else ""))


@rule("D15", "policy_age")
def policy_age(ctx: Ctx) -> Iterator[Finding]:
    """trainer.policy_age <= the algorithm spec's max_policy_staleness."""
    trainer = ctx.facts.trainer
    age = ctx.number("trainer.policy_age.max_versions")
    if trainer is None or age is None or trainer.max_policy_staleness is None:
        return
    problem = policy_age_problem(int(age), trainer.max_policy_staleness, backend=trainer.backend)
    if problem is not None:
        yield finding("D15", "error", problem, items=("trainer.policy_age",),
                      values={"max_versions": int(age), "max_policy_staleness": trainer.max_policy_staleness},
                      suggested={"trainer.policy_age.max_versions": trainer.max_policy_staleness})


@rule("D16", "min_samples", scope="pool")
def min_samples(ctx: Ctx) -> Iterator[Finding]:
    """min_samples <= sampling.n, and >= 2 for an estimator with a group baseline."""
    if ctx.pool is None or not ctx.pool.agentic:
        return
    want = ctx.number("sampler.min_samples.min_n", on=False)
    if want is None:
        return
    n = ctx.pool.sampling_n
    if n is not None and want > n:
        yield finding("D16", "error", ctx.where(
            f"sampler.min_samples {want:g} > sampling.n {n}: no group can keep enough trials, every group is dropped "
            "and its task requeued; set agentic.min_samples to at most sampling.n"),
            items=("sampler.min_samples",), values={"min_n": want, "sampling_n": n},
            suggested={"sampler.min_samples.min_n": min(2, n)})
    estimator = None if ctx.facts.trainer is None else ctx.facts.trainer.estimator
    if estimator in GROUP_BASELINE_ESTIMATORS and want < 2:
        yield finding("D16", "error", ctx.where(
            f"sampler.min_samples {want:g} with the {estimator} estimator: a group of one trial has no baseline "
            "(its advantage is 0, the group trains nothing); set agentic.min_samples to at least 2"),
            items=("sampler.min_samples",), values={"min_n": want, "estimator": estimator},
            suggested={"sampler.min_samples.min_n": 2})


def backlog_findings(trainer: TrainerFacts, shapes: Sequence[Backlog]) -> tuple[list[str], list[str]]:
    """``(problems, warnings)`` of the samplers' backlog caps (``shapes``: each r2
    pool's) against a single buffer trainer's release cadence.

    A sampler's backlog counts from the read position of the newest release it
    sees, 0 before the first (``SamplerLoop.backlog``): until a release lands, the
    samplers together write at most the sum of their caps (plus the groups each
    can write past it). A pool with ``max_backlog_groups: "auto"`` sizes its caps
    from the release cadence: the check is then satisfied.

    * problem: even with every pool at its most samplers, each at its cap plus
      the most it writes past it, the trainer cannot read the ``release_every x
      groups_per_round`` groups up to its first release; it would wait for groups
      until ``--rl-sample-wait-s`` ends the run, so nothing should be ordered;
    * warning: the target samplers hold fewer than
      :func:`yeto.rl.engine.sample_roles.release_backlog_need` groups: they wait
      at their caps between releases. The value to set is that need over the
      samplers, rounded up, and over one sampler fewer (one down or being replaced)."""
    from yeto.rl.engine import sample_roles  # noqa: PLC0415

    groups, every, need = trainer.groups_per_step, trainer.release_every, trainer.backlog_need
    if groups is None or need is None or not shapes or any(s.cap == AUTO for s in shapes):
        return [], []
    rounds = trainer.record_rounds
    first = every * groups
    most = sum(s.most * (int(s.cap) + s.over) for s in shapes)
    caps = ", ".join(f"{s.target} x {s.cap}" for s in shapes)
    samplers = sum(s.target for s in shapes)
    held = sum(s.target * int(s.cap) for s in shapes)
    fewer = max(1, samplers - 1)
    suggest, suggest_fewer = -(-need // samplers), -(-need // fewer)
    advice = (f"set max_backlog_groups to at least {suggest} per sampler"
              + (f" ({suggest_fewer} to keep up with one of the {samplers} samplers down or being replaced)"
                 if samplers > 1 else "")
              + f" = ceil(((release_every {every} + m {rounds} + 1) x {groups} + "
                f"{sample_roles.RELEASE_BACKLOG_MARGIN_GROUPS}) / samplers), or release more often")
    if most < first:
        return [(f"the samplers ({caps} max_backlog_groups; at most {most} groups with every node a pool may run, "
                 f"each past its cap) count their backlog from the newest release's read position, 0 before the "
                 f"first release, but the trainer reads --rl-release-every {every} x --rollout-batch-size {groups} "
                 f"= {first} groups before it can release: it would wait for groups and never release; "
                 + advice)], []
    if held >= need:
        return [], []
    upload = (f"m = {rounds}: the round after a publication plus a staged upload assumed "
              f"{sample_roles.ASSUMED_UPLOAD_ROUNDS} rounds long; re-derive with the measured upload / step time"
              if rounds else "m = 0: releases written inline")
    return [], [f"the samplers ({caps} max_backlog_groups = {held} groups) count their backlog from the newest "
                f"release's read position, and the trainer reads up to {need} groups between two releases "
                f"(--rl-release-every {every}, --rollout-batch-size {groups}, {upload}): the samplers would wait "
                f"at their caps between releases; " + advice]


@rule("D19", "backlog_cadence", max_level="error")
def backlog_cadence(ctx: Ctx) -> Iterator[Finding]:
    """The samplers' backlog caps hold the groups a single trainer reads between two
    releases (refused when they cannot reach the first release at all)."""
    trainer = ctx.facts.trainer
    if trainer is None or not trainer.single_buffer_trainer:
        return  # synced trainers size their samplers by the pool's backlog section
    shapes = [p.backlog for p in ctx.facts.pools if p.data_path == "r2" and p.backlog is not None]
    problems, warnings = backlog_findings(trainer, shapes)
    samplers = sum(s.target for s in shapes) or 1
    suggested = {} if trainer.backlog_need is None else {"max_backlog_groups": -(-trainer.backlog_need // samplers)}
    for text in problems:
        yield finding("D19", "error", text, values={"need": trainer.backlog_need}, suggested=suggested)
    for text in warnings:
        yield finding("D19", "warn", text, values={"need": trainer.backlog_need}, suggested=suggested)
