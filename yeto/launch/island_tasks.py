"""Per-island task data (``--rl-island-tasks``): plan, launch checks and staging."""

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


def _refuse_island_tasks(args, path: str) -> None:
    """Per-island tasks exist on the ports RL island path only."""

    if getattr(args, "rl_island_tasks", None):
        raise ValueError(f"{rl_option('rl_island_tasks', CLI)} applies to ports RL islands "
                         f"only, not the {path} path")


def island_task_data(args, learner_id: int) -> tuple[str, str, dict[str, str]]:
    """The island's ``--data`` value, its SHA256 flag and file mounts.

    Without ``--rl-island-tasks`` every island gets the shared ``--data``. With
    it, the island's own prompt file is written under the run directory and
    mounted in place of ``--data``.
    """

    from .. import runs
    from ..datasource import learner_data_arg, learner_file_mounts

    plan = island_task_plan(args)
    if plan is None:
        return learner_data_arg(args.data), "", learner_file_mounts(args.data)
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
