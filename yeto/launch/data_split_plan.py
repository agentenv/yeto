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


def sampler_dispense(args) -> tuple[list[int], str | None]:
    """S27 P4: ``(sampler islands that would claim rows, why the launch cannot
    dispense to samplers or None)``. A synced group's actor syncer dispenses to
    its sampler islands; a launch of synced trainers only dispenses to sampler
    launches that join it (``--rl-dispense-addr``); a sampler launch of its own
    claims from the dispenser it names. A store-coupled pair or a convertible
    pool has no syncer."""

    from ..rl.engine import sample_roles

    count = len(parse_gpu_spec(args.gpu)) if getattr(args, "gpu", None) else 1
    addr = getattr(args, "rl_dispense_addr", None)
    if sample_roles.is_convertible_pool(args):
        return [], "a convertible pool has no syncer to run the dispenser"
    if sample_roles.is_pair(args):
        return [], ("a store-coupled pair has no syncer to run the dispenser; use a synced group "
                    "(--rl-sample-role trainer --rl-island-sample-role M:sampler) or sampler "
                    "launches with --rl-dispense-addr")
    if sample_roles.is_synced_group(args):
        if addr:
            return [], ("--rl-dispense-addr: a synced group's samplers claim from its own actor "
                        "syncer (the launcher sets the address)")
        return sample_roles.sampler_islands(args, count), None
    role = sample_roles.sample_role(args)
    if role == sample_roles.SAMPLER:
        if count != 1:
            return [], "a sampler launch of its own has one island"
        if not addr:
            return [], ("a sampler launch of its own has no syncer: give --rl-dispense-addr "
                        "HOST:PORT, the actor syncer of the run it joins")
        return [0], None
    if sample_roles.is_synced_trainer(args):
        if addr:
            return [], "--rl-dispense-addr belongs to a sampler launch"
        return [], None  # samplers of other launches claim from this launch's syncer
    return [], f"--rl-data-split dispense does not cover {role} islands"


def _plan_data_split(args):
    from ..rl.engine.data_split import DISPENSE, plan_data_split

    islands, roles = _data_split_islands(args)
    samplers, blocker = ([], None)
    if roles and getattr(args, "rl_data_split", None) == DISPENSE:
        samplers, blocker = sampler_dispense(args)
    elif getattr(args, "rl_dispense_addr", None):
        raise ValueError("--rl-dispense-addr needs --rl-data-split dispense on a sampler launch")
    return plan_data_split(getattr(args, "rl_data_split", None), islands=islands,
                           tasks=bool(getattr(args, "rl_island_tasks", None)), sample_roles=roles,
                           external_learners=int(getattr(args, "external_learners", 0) or 0),
                           samplers=samplers, sampler_blocker=blocker,
                           dispense_blockers=dispense_blockers(args))


def island_backends(args) -> set[str]:
    """The training backends of the launch: ``--rl-backend`` and every island's own
    (rl-mixed-island-placement), when the launch has islands."""

    backends = {getattr(args, "rl_backend", None) or "miles"}
    try:
        from .topology import rl_island_args

        count = len(parse_gpu_spec(args.gpu)) if getattr(args, "gpu", None) else 0
        backends |= {getattr(rl_island_args(args, k), "rl_backend", None) or "miles" for k in range(count)}
    except Exception:  # noqa: BLE001 - the launch checks report a bad placement themselves
        pass
    return backends


def dispense_blockers(args) -> list[str]:
    """Why this launch cannot use the task dispenser (S27 P2); empty: it can.
    ``--rl-data-split auto`` falls back to the row slots (stride) on any of
    these; an explicit ``dispense`` is refused on them (``island_tasks``)."""

    from ..rl.engine.selection import FULL_PARAMETER_SYNC_PRESETS, SAO_STREAMING_SYNC_PRESET

    out = []
    if getattr(args, "rl_sync_preset", None) == SAO_STREAMING_SYNC_PRESET:
        out.append(f"--rl-sync-preset {SAO_STREAMING_SYNC_PRESET} (no GPU check with the dispenser)")
    # S27 G-EU (VERL-PARITY #31): verl training islands of the synchronous trainer take
    # their rows from the dispenser (adapters/verl/task_dispense.py), as Miles islands do
    backends = island_backends(args)
    if len(backends) > 1:
        out.append(f"training islands of different backends ({', '.join(sorted(backends))}: their "
                   "prompt files differ)")
    if "verl" in backends and int(getattr(args, "rl_max_policy_age", 0) or 0) >= 1:
        out.append("verl fully_async (--rl-max-policy-age > 0: verl's rollouter draws the rows)")
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
    if (getattr(args, "rl_checkpoint_store", None) and not getattr(args, "rl_peer_recovery", False)
            and not getattr(args, "rl_export_hf_rounds", None)
            and str(getattr(args, "rl_sync_preset", "")) in FULL_PARAMETER_SYNC_PRESETS):
        # S27 G-DS catch-up: the store keeps only the islands' data positions, which the
        # dispenser would make unused (the launch would then refuse the store)
        out.append("--rl-checkpoint-store kept for the islands' data positions (catch-up without "
                   "peer recovery)")
    return out


def dispense_syncer_flags(args) -> str:
    """The actor syncer's ``--dispense`` flags (S27 P2); empty without the dispenser."""

    plan = data_split_plan(args)
    if not plan.dispense or getattr(args, "rl_dispense_addr", None):
        return ""  # a sampler launch of its own claims from another launch's dispenser
    flags = (" --dispense"
             f" --dispense-lease-s {float(getattr(args, 'rl_dispense_lease_s', 300.0))!r}"
             f" --dispense-max-redo {int(getattr(args, 'rl_dispense_max_redo', 1))}"
             " --dispense-log ~/yeto-output/task-dispense.jsonl")
    passes = getattr(args, "rl_data_passes", None)
    if passes is not None:
        flags += f" --dispense-passes {int(passes)}"
    return flags
