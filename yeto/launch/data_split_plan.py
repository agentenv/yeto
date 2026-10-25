"""The data split plan of a launch (S27 task assignment P1 / P2).

Kept apart from ``island_tasks`` (the launch checks) so that the syncer
command builder (``syncer``) can read the plan without importing the checks,
which import ``syncer`` themselves (``tests/test_launch_layout.py``: no
import cycle among the ``yeto/launch`` modules).
"""

from __future__ import annotations

from ..gpu_spec import parse_gpu_spec


def _data_split_islands(args) -> tuple[list[int], bool]:
    """``(learner ids of the same-card training islands, whether the launch has
    sample roles)``. With sample roles (samplers, buffer trainers, convertible
    nodes) no island of the launch both generates and trains on ``--data``."""

    from ..rl.engine import sample_roles

    count = len(parse_gpu_spec(args.gpu)) if getattr(args, "gpu", None) else 1  # rl_island_count
    roles = sample_roles.island_roles(args, count)
    if roles or sample_roles.sample_role(args) != sample_roles.BOTH:
        return [], True
    return list(range(count)), False


def data_split_plan(args):
    """The launch's :class:`~yeto.rl.engine.data_split.DataSplitPlan` (cached by
    :func:`_check_data_split`; computed here for a caller that skipped the launch
    checks, e.g. a dry run or a unit test that builds one island task)."""

    plan = getattr(args, "_data_split_plan", None)
    return _plan_data_split(args) if plan is None else plan


def _plan_data_split(args):
    from ..rl.engine.data_split import plan_data_split

    islands, roles = _data_split_islands(args)
    return plan_data_split(getattr(args, "rl_data_split", None), islands=islands,
                           tasks=bool(getattr(args, "rl_island_tasks", None)), sample_roles=roles,
                           external_learners=int(getattr(args, "external_learners", 0) or 0),
                           dispense_blockers=dispense_blockers(args))


def dispense_blockers(args) -> list[str]:
    """Why this launch cannot use the task dispenser (S27 P2); empty: it can.
    ``--rl-data-split auto`` falls back to the row slots (stride) on any of
    these; an explicit ``dispense`` is refused on them (``island_tasks``)."""

    out = []
    backend = getattr(args, "rl_backend", None) or "miles"
    if backend != "miles":
        out.append(f"--rl-backend {backend} (the dispenser is wired for Miles only)")
    if getattr(args, "rl_single_island_no_sync", False):
        out.append("--rl-single-island-no-sync (no syncer)")
    if getattr(args, "rl_elastic", False):
        out.append("--rl-elastic")
    shards = getattr(args, "syncer_shards", None)
    if shards is not None and int(shards) > 1:
        out.append(f"--syncer-shards {int(shards)}")
    if (int(getattr(args, "rl_max_policy_age", 0) or 0) >= 1
            and not int(getattr(args, "rl_rollout_ahead", 0) or 0)):
        out.append("Miles fully_async rollouts (--rl-max-policy-age without --rl-rollout-ahead)")
    for flag, name in (("--rl-cut-every", "rl_cut_every"), ("--rl-cut-auto-mtbf-s", "rl_cut_auto_mtbf_s")):
        if getattr(args, name, None) is not None:
            out.append(f"{flag} (island round cuts keep their own data position)")
    if getattr(args, "eval_only", False):
        out.append("--eval-only")
    return out


def dispense_syncer_flags(args) -> str:
    """The actor syncer's ``--dispense`` flags (S27 P2); empty without the dispenser."""

    plan = data_split_plan(args)
    if not plan.dispense:
        return ""
    flags = (" --dispense"
             f" --dispense-lease-s {float(getattr(args, 'rl_dispense_lease_s', 300.0))!r}"
             f" --dispense-max-redo {int(getattr(args, 'rl_dispense_max_redo', 1))}"
             " --dispense-log ~/yeto-output/task-dispense.jsonl")
    passes = getattr(args, "rl_data_passes", None)
    if passes is not None:
        flags += f" --dispense-passes {int(passes)}"
    return flags
