"""Per-island task data: plan, launch checks and staging.

Two plans say which rows of ``--data`` a training island reads:

* ``--rl-island-tasks``: the user's task list per island (one prompt file each);
* ``--rl-data-split`` (``yeto.rl.engine.data_split``): without task lists, each
  same-card training island of a multi-island launch reads its own row slot
  (``--rl-data-slot``); ``shared`` keeps every row on every island and warns.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ..gpu_spec import parse_gpu_spec
from ..rl.engine.run_flags import CLI, rl_option


ISLAND_TASKS_DIR = "island-tasks"


def island_task_plan(args):
    """rl-per-island-tasks: island -> :class:`IslandTaskData`, or None without the flag.

    Computed once per launch by :func:`_check_island_tasks` (cached on
    ``args``). Islands are only built for the ``--gpu`` entries (elastic adds
    no islands), so the plan covers every learner id the launch can build.
    The head's second hop prepares its own args and rebuilds the same bytes
    from its staged ``--data``.
    """

    if not getattr(args, "rl_island_tasks", None):
        return None
    plan = getattr(args, "_island_task_plan", None)
    if plan is None:
        raise ValueError(f"{rl_option('rl_island_tasks', CLI)} was not checked; "
                         "prepare the launch arguments first")
    return plan


def _check_island_tasks(args, eval_interval: int | None) -> None:
    """``--rl-island-tasks``: refuse a bad spec before any cloud call (P3-Q1..Q3),
    then cache the plan on ``args``.

    Rewrites the flag to its inline form so a JSON-file spec reaches the head.
    """

    if not getattr(args, "rl_island_tasks", None):
        return
    from ..datasource import kind
    from ..rl.island_tasks import (
        format_island_tasks,
        parse_island_tasks,
        plan_island_tasks,
        task_count_warning,
    )

    flag = rl_option("rl_island_tasks", CLI)
    data = getattr(args, "data", None)
    if not (isinstance(data, str) and kind(data) == "local" and data.endswith(".jsonl")
            and Path(data).expanduser().is_file()):
        raise ValueError(f"{flag} needs --data to be one local JSONL file")
    if getattr(args, "external_learners", 0):
        raise ValueError(f"{flag} does not cover --external-learners")
    if eval_interval is not None:
        # P3-Q3: eval uses the full set on the eval island only.
        raise ValueError(f"{flag}: training islands do not run eval; drop "
                         f"{rl_option('rl_eval_interval', CLI)} and evaluate on the eval island")
    spec = parse_island_tasks(args.rl_island_tasks, flag=flag)
    args.rl_island_tasks = format_island_tasks(spec)
    args._island_task_plan = plan_island_tasks(
        data, spec, range(len(parse_gpu_spec(args.gpu))), flag=flag)
    warning = task_count_warning(args._island_task_plan, flag=flag)
    if warning:
        print(f"[launcher] WARNING {warning}", file=sys.stderr, flush=True)
    shared = overlap_warning({island: data.task_ids for island, data in args._island_task_plan.items()},
                             flag=flag)
    if shared:
        print(f"[launcher] WARNING {shared}", file=sys.stderr, flush=True)


def overlap_warning(task_lists, *, flag: str) -> str | None:
    """The launch warning when two islands' task lists name the same tasks, or None."""

    from ..rl.engine.data_split import overlapping_tasks

    pairs = overlapping_tasks(task_lists)
    if not pairs:
        return None
    listed = ", ".join(f"islands {a} and {b}: {n}" for (a, b), n in sorted(pairs.items()))
    return (f"{flag}: some tasks are on more than one island ({listed} shared tasks); "
            "those tasks are trained by each of these islands")


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


def _check_data_split(args) -> None:
    """``--rl-data-split``: plan which rows each training island reads, refuse a
    request the launch cannot honour, print the plan and cache it on ``args``.

    Several islands that read the same rows get a loud warning (P0): every round
    they train the same prompts."""

    from ..rl.engine.data_split import DATA_SPLIT_FLAG

    for name in ("rl_data_slot", "rl_data_position_store", "rl_data_dispense"):
        if getattr(args, name, None):
            raise ValueError(f"{rl_option(name, CLI)} is set by the launcher, not a launch "
                             f"option; use {DATA_SPLIT_FLAG}")
    plan = args._data_split_plan = _plan_data_split(args)
    _check_dispense(args, plan)
    print(f"[launcher] data split ({plan.mode}): {plan.describe()}", file=sys.stderr, flush=True)
    warning = plan.warning()
    if warning:
        print(f"[launcher] WARNING {warning}", file=sys.stderr, flush=True)
    warning = _relaunch_position_warning(args, plan)
    if warning:
        print(f"[launcher] WARNING {warning}", file=sys.stderr, flush=True)


def _check_dispense(args, plan) -> None:
    """S27 P2: refuse what the dispenser does not cover yet (fail closed)."""

    from ..rl.engine.data_split import DATA_SPLIT_FLAG
    from .syncer import syncer_shard_count

    dispense_only = {"rl_data_passes": None, "rl_dispense_max_redo": 1,
                     "rl_dispense_lease_s": 300.0, "rl_dispense_prefetch_blocks": 2}
    if not plan.dispense:
        named = [name for name, default in dispense_only.items()
                 if getattr(args, name, default) not in (default, None)]
        if named:
            raise ValueError(f"{', '.join(rl_option(n, CLI) for n in named)} need "
                             f"{DATA_SPLIT_FLAG} dispense")
        return
    passes = getattr(args, "rl_data_passes", None)
    if passes is not None and int(passes) < 1:
        raise ValueError("--rl-data-passes must be >= 1")
    if int(getattr(args, "rl_dispense_max_redo", 1)) < 0:
        raise ValueError("--rl-dispense-max-redo must be >= 0")
    if not float(getattr(args, "rl_dispense_lease_s", 300.0)) > 0:
        raise ValueError("--rl-dispense-lease-s must be positive")
    if int(getattr(args, "rl_dispense_prefetch_blocks", 2)) < 1:
        raise ValueError("--rl-dispense-prefetch-blocks must be >= 1")
    if getattr(args, "rl_single_island_no_sync", False) or getattr(args, "rl_elastic", False):
        raise ValueError(f"{DATA_SPLIT_FLAG} dispense needs a synced launch (the dispenser runs "
                         "in the actor syncer); not --rl-single-island-no-sync / --rl-elastic")
    if syncer_shard_count(args) > 1:
        raise ValueError(f"{DATA_SPLIT_FLAG} dispense with --syncer-shards > 1 is not supported yet")
    if (int(getattr(args, "rl_max_policy_age", 0) or 0) >= 1
            and not int(getattr(args, "rl_rollout_ahead", 0) or 0)):
        raise ValueError(f"{DATA_SPLIT_FLAG} dispense does not cover Miles fully_async rollouts "
                         "(--rl-max-policy-age without --rl-rollout-ahead) yet")
    for name in ("rl_cut_every", "rl_cut_auto_mtbf_s"):
        if getattr(args, name, None) is not None:
            raise ValueError(f"{DATA_SPLIT_FLAG} dispense and island round cuts "
                             f"({rl_option(name, CLI)}) exclude each other: the dispenser, not the "
                             "island cut, keeps the data position")
    if (getattr(args, "rl_backend", None) or "miles") != "miles":
        raise ValueError(f"{DATA_SPLIT_FLAG} dispense is wired for the Miles backend only (verl: later)")


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


def _relaunch_position_warning(args, plan) -> str | None:
    """An island pool of full-parameter islands keeps each island's data
    position in ``--rl-checkpoint-store``; without one a relaunched island
    starts at the first row of its rows again."""

    from .syncer import island_pool, rl_dense_sweep

    if (len(plan.islands) < 2 or not rl_dense_sweep(args) or not island_pool(args)
            or getattr(args, "rl_checkpoint_store", None)):
        return None
    return ("a full-parameter island that is relaunched into the running pool starts again "
            "at the first of its rows (it keeps no data position without --rl-checkpoint-store)")


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
                           external_learners=int(getattr(args, "external_learners", 0) or 0))


def _refuse_island_tasks(args, path: str) -> None:
    """Per-island tasks exist on the ports RL island path only."""

    if getattr(args, "rl_island_tasks", None):
        raise ValueError(f"{rl_option('rl_island_tasks', CLI)} applies to ports RL islands "
                         f"only, not the {path} path")


def island_task_data(args, learner_id: int) -> tuple[str, str, dict[str, str]]:
    """The island's ``--data`` value, its data flags and file mounts.

    Without ``--rl-island-tasks`` every island gets the shared ``--data``; a
    same-card training island of a split launch also gets its row slot
    (``--rl-data-slot I/N``, :func:`data_split_plan`). With task lists, the
    island's own prompt file is written under the run directory and mounted in
    place of ``--data`` (``--data-sha256``).
    """

    from .. import runs
    from ..datasource import learner_data_arg, learner_file_mounts
    from ..rl.engine import sample_roles

    plan = island_task_plan(args)
    if plan is None:
        split = data_split_plan(args)
        both = sample_roles.sample_role(args) == sample_roles.BOTH
        slot = split.slot(learner_id) if both else None
        flags = f" {rl_option('rl_data_slot')} {slot.env()}" if slot is not None else ""
        if both and split.dispense:
            flags += f" {rl_option('rl_data_dispense')}"
            prefetch = int(getattr(args, "rl_dispense_prefetch_blocks", 2))
            if prefetch != 2:
                flags += f" {rl_option('rl_dispense_prefetch_blocks')} {prefetch}"
        return learner_data_arg(args.data), flags, learner_file_mounts(args.data)
    if learner_id not in plan:
        raise ValueError(f"{rl_option('rl_island_tasks', CLI)} has no task list for learner "
                         f"{learner_id} (planned islands: {sorted(plan)})")
    island = plan[learner_id]
    path = runs.run_dir(args.cluster_prefix) / ISLAND_TASKS_DIR / f"island-{learner_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(island.content)
    return (learner_data_arg(str(path)),
            f" {rl_option('data_sha256')} {island.sha256}",
            learner_file_mounts(str(path)))
