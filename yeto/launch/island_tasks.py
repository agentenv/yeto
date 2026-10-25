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
from .data_split_plan import (  # noqa: F401 - re-exported
    _data_split_islands,
    _plan_data_split,
    data_split_plan,
    dispense_syncer_flags,
)


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


def _check_data_split(args) -> None:
    """``--rl-data-split``: plan which rows each training island reads, refuse a
    request the launch cannot honour, print the plan and cache it on ``args``.

    Several islands that read the same rows get a loud warning (P0): every round
    they train the same prompts."""

    from ..rl.engine.data_split import DATA_SPLIT_FLAG

    for name in ("rl_data_slot", "rl_data_position_store", "rl_data_dispense",
                 "rl_dispense_by_label", "rl_dispense_labels"):
        if getattr(args, name, None):
            raise ValueError(f"{rl_option(name, CLI)} is set by the launcher, not a launch "
                             f"option; use {DATA_SPLIT_FLAG}")
    if getattr(args, "rl_dispense_mix", None) and getattr(args, "data_mix_record", None) is None:
        raise ValueError(f"{rl_option('rl_dispense_mix', CLI)} is set by the launcher from "
                         "--data-mix, not a launch option")
    plan = args._data_split_plan = _plan_data_split(args)
    _check_dispense(args, plan)
    _check_streams(args, plan)
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
    if passes is not None and not plan.islands:
        raise ValueError("--rl-data-passes: sampler islands do not end on the dispenser's data end "
                         "yet (S27 P4); give the samplers their rounds instead")
    if passes is not None and _strict_roster(args):
        raise ValueError("--rl-data-passes needs islands that may end on their own when the data "
                         "ends (asynchronous rounds or an island pool); a strict roster trains "
                         "every island in every round")
    if int(getattr(args, "rl_dispense_max_redo", 1)) < 0:
        raise ValueError("--rl-dispense-max-redo must be >= 0")
    if not float(getattr(args, "rl_dispense_lease_s", 300.0)) > 0:
        raise ValueError("--rl-dispense-lease-s must be positive")
    if int(getattr(args, "rl_dispense_prefetch_blocks", 2)) < 1:
        raise ValueError("--rl-dispense-prefetch-blocks must be >= 1")
    joins = bool(getattr(args, "rl_dispense_addr", None))  # a sampler launch of its own (P4)
    if ((getattr(args, "rl_single_island_no_sync", False) and not joins)
            or getattr(args, "rl_elastic", False)):
        raise ValueError(f"{DATA_SPLIT_FLAG} dispense needs a synced launch (the dispenser runs "
                         "in the actor syncer); not --rl-single-island-no-sync / --rl-elastic "
                         "(a sampler launch of its own names its dispenser with --rl-dispense-addr)")
    if syncer_shard_count(args) > 1:
        raise ValueError(f"{DATA_SPLIT_FLAG} dispense with --syncer-shards > 1 is not supported yet")
    if (plan.islands and int(getattr(args, "rl_max_policy_age", 0) or 0) >= 1
            and not int(getattr(args, "rl_rollout_ahead", 0) or 0)):
        raise ValueError(f"{DATA_SPLIT_FLAG} dispense does not cover Miles fully_async rollouts "
                         "(--rl-max-policy-age without --rl-rollout-ahead) yet")
    for name in ("rl_cut_every", "rl_cut_auto_mtbf_s"):
        # S27 P4: a buffer trainer's cut keeps its buffer position, not a data position
        if plan.islands and getattr(args, name, None) is not None:
            raise ValueError(f"{DATA_SPLIT_FLAG} dispense and island round cuts "
                             f"({rl_option(name, CLI)}) exclude each other: the dispenser, not the "
                             "island cut, keeps the data position")
    from .data_split_plan import island_backends

    backends = island_backends(args)
    if plan.islands and len(backends) > 1:
        # S27 G-EU: a Miles island and a verl island write different prompt files, so the
        # dispenser would refuse the second fingerprint (sampler islands claim through the
        # engine-neutral DispensedPrompts / the Miles data source either way)
        raise ValueError(f"{DATA_SPLIT_FLAG} dispense: the training islands run different backends "
                         f"({', '.join(sorted(backends))}); their prompt files differ")


def dispensing_islands(args, plan) -> list[int]:
    """The islands of the launch that claim rows from the dispenser (by island
    index in ``--gpu``; the same-card training islands' learner ids)."""

    return list(plan.islands) + list(getattr(plan, "samplers", ()) or ())


def island_task_labels(args) -> dict[int, list[str] | None]:
    """``--rl-island-task-labels`` per island ({} without it); checked by
    :func:`_check_streams` (cached on ``args``)."""

    cached = getattr(args, "_island_task_labels", None)
    if cached is not None:
        return cached
    spec = getattr(args, "rl_island_task_labels", None)
    if not spec:
        return {}
    from ..rl.engine.task_dispense import parse_island_labels

    return parse_island_labels(spec)


def _data_rows(args):
    """The rows of the run's ``--data`` (the launcher reads them for the label
    check; an HF id is loaded as the island loads it)."""

    from ..data import load_rows, prompt_split  # the split the island reads

    revision = getattr(args, "data_revision", None)
    split = prompt_split(args.data, revision)
    if split == "train":
        return load_rows(args.data, revision=revision)
    return load_rows(args.data, split=split, revision=revision)


def _check_streams(args, plan) -> None:
    """S27 P3: ``--data-mix`` and ``--rl-island-task-labels`` need the
    dispenser; every row label must be taken by an island of the launch and
    every listed label must be on a row (fail before any cloud spend)."""

    from ..rl.engine.data_split import DATA_SPLIT_FLAG
    from ..rl.engine.task_dispense import UNLABELED, row_label, unmatched_labels

    mix = getattr(args, "rl_dispense_mix", None)
    spec = getattr(args, "rl_island_task_labels", None)
    if not (mix or spec):
        return
    if not plan.dispense:
        named = "--data-mix" if mix else rl_option("rl_island_task_labels", CLI)
        raise ValueError(f"{named} needs {DATA_SPLIT_FLAG} dispense (the dispenser mixes the "
                         "sources and keeps the label streams)")
    if mix:
        record = getattr(args, "data_mix_record", None) or {}
        sources = record.get("sources") or []
        print(f"[launcher] data mix ({len(sources)} sources, weights {mix}): the dispenser "
              "interleaves one stream per source (per source and task label with task "
              "labels); a prefix of the rows given out is within one row of the weights when "
              "every island takes every stream", file=sys.stderr, flush=True)
    if not spec:
        return
    labels = island_task_labels(args)
    islands = dispensing_islands(args, plan)
    unknown = sorted(set(labels) - set(islands))
    if unknown:
        raise ValueError(f"{rl_option('rl_island_task_labels', CLI)} names island(s) {unknown}; "
                         f"the islands that claim rows are {islands}")
    counts: dict[str, int] = {}
    for row in _data_rows(args):
        label = row_label(row)
        counts[label] = counts.get(label, 0) + 1
    shown = {UNLABELED if k == "" else k: v for k, v in sorted(counts.items())}
    listed = {("" if label == UNLABELED else label) for accepted in labels.values() if accepted
              for label in accepted}
    absent = sorted(UNLABELED if k == "" else k for k in listed - set(counts))
    if absent:
        raise ValueError(f"{rl_option('rl_island_task_labels', CLI)}: no row of --data has task "
                         f"label(s) {absent} (the rows have {shown})")
    orphan = unmatched_labels(counts, islands, labels)
    if orphan:
        raise ValueError(f"{rl_option('rl_island_task_labels', CLI)}: no island takes the rows "
                         f"with task label(s) {[UNLABELED if k == '' else k for k in orphan]} "
                         f"({sum(counts[k] for k in orphan)} rows); give one island these labels "
                         "or leave an island unlisted (it takes every row)")
    args._island_task_labels = labels
    described = "; ".join(f"island {m}: {'every row' if labels.get(m) is None else ','.join(labels[m])}"
                          for m in islands)
    print(f"[launcher] task labels {shown}: {described}", file=sys.stderr, flush=True)
    if len({tuple(v) if v else None for v in labels.values()}) > 1 or len(labels) < len(islands):
        print("[launcher] WARNING islands with different task labels train different rows: the "
              "merge weights an island by tokens^2/steps, so islands of long answers weigh "
              "more; compare against --rl-sync-learner-weight equal", file=sys.stderr, flush=True)


def sampler_dispense_island(args, plan) -> int | None:
    """S27 P4: the island index of a sampler island that claims rows (``args``:
    the island args), or None. A synced group's sampler carries it
    (``sample_roles``: ``rl_sampler_dispense_island``); a sampler launch of its
    own is its island 0."""

    from ..rl.engine import sample_roles

    if not plan.dispense or sample_roles.sample_role(args) != sample_roles.SAMPLER:
        return None
    island = getattr(args, "rl_sampler_dispense_island", None)
    if island is not None:
        return int(island)
    return 0 if getattr(args, "rl_dispense_addr", None) else None


def sampler_dispense_addr(args) -> str:
    """The dispenser address a sampler island dials: the one its launch was
    given (a sampler launch of its own) or the group's actor syncer, which the
    island reads from ``$SYNCER_ADDR`` (set on every island of a synced launch)."""

    import shlex

    addr = getattr(args, "rl_dispense_addr", None)
    return shlex.quote(str(addr)) if addr else "$SYNCER_ADDR"


def stream_island_flags(args, island: int) -> str:
    """The island flags of the dispenser's streams (S27 P3): the mix weights and
    the label split for every claiming island, its own labels when it has some."""

    flags = ""
    mix = getattr(args, "rl_dispense_mix", None)
    if mix:
        flags += f" {rl_option('rl_dispense_mix')} {mix}"
    if getattr(args, "rl_island_task_labels", None):
        flags += f" {rl_option('rl_dispense_by_label')}"
        accepted = island_task_labels(args).get(int(island))
        if accepted is not None:
            flags += f" {rl_option('rl_dispense_labels')} {','.join(accepted)}"
    return flags


def _strict_roster(args) -> bool:
    """Whether every island must train every round (the island bridge's
    ``ends_on_data_end`` is False): a full-parameter strict sweep, or
    strict-avg without an island pool (also with a critic)."""

    from ..rl.engine.selection import FULL_PARAMETER_SYNC_PRESETS
    from .syncer import island_pool

    preset = str(getattr(args, "rl_sync_preset", "strict-avg"))
    if preset in FULL_PARAMETER_SYNC_PRESETS:
        return str(getattr(args, "rl_sync_full_parameter_rounds", "sweep")) != "async"
    if preset == "strict-avg":
        return not island_pool(args)
    return False


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
        island = learner_id if both else sampler_dispense_island(args, split)
        if split.dispense and (both or island is not None):
            flags += f" {rl_option('rl_data_dispense')}"
            if not both:  # S27 P4: a sampler names its dispenser (it has no --syncer)
                flags += f" {rl_option('rl_dispense_addr')} {sampler_dispense_addr(args)}"
            prefetch = int(getattr(args, "rl_dispense_prefetch_blocks", 2))
            if prefetch != 2:
                flags += f" {rl_option('rl_dispense_prefetch_blocks')} {prefetch}"
            flags += stream_island_flags(args, island)
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
