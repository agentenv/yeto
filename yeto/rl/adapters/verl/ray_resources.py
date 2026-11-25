"""The island Ray's resources against what the verl trainer asks for.

A verl island's trainer actor asks Ray for :data:`TRAINER_ACTOR_CPUS` CPU slots
(``fully_async_runner.YetoFullyAsyncTrainer`` and the buffer trainer), its task runner
for :data:`TASK_RUNNER_CPUS`, and its worker groups for every card through placement
groups. Ray leaves a request it cannot place pending, with no error: an island on an
8-vCPU sky VM (AWS g5.2xlarge) stopped after "Starting create trainer" with
``Pending Demands: {'CPU': 10.0}``, its GPU idle, until someone killed it.

Ray CPUs are scheduling tokens, so the launcher starts a verl island's Ray with
``max(node vCPUs, ISLAND_RAY_MIN_CPUS)`` of them (``launch_flags``). What is still short
(a Ray started by hand, another workload on the node) is a configuration fault of the
island, not a fault a relaunch cures:

* :func:`cpu_shortfall`: before verl starts, the island's Ray holds fewer CPU slots than
  the trainer actor and its task runner need -- a stop at once;
* :func:`await_actor` / :func:`get_watching_placements`: a trainer actor not placed, or a
  placement group of its workers still pending, after :func:`schedule_timeout_s` -- a stop.

Each raises :class:`RayResourcesUnsatisfiable` with what is asked and what the Ray has; it
is a LEAVE ``CONFIG_MISMATCH`` (``multinode.TopologyError``), so a fleet island ends with
the deterministic-stop exit (11, ``config_errors.island_stop_exit``) and is not relaunched.

No Ray import at module level: the island entry reads it before verl runs.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from typing import Any

from yeto.rl.engine.multinode import TopologyError

# The trainer actor's and the task runner's ``ray.remote(num_cpus=...)``.
TRAINER_ACTOR_CPUS = 10
TASK_RUNNER_CPUS = 1
SCHEDULE_TIMEOUT_ENV = "YETO_RL_RAY_SCHEDULE_TIMEOUT_S"
SCHEDULE_TIMEOUT_S = 300.0
FIX = ("start the island Ray with more CPU slots (ray start --num-cpus N; Ray CPUs are scheduling "
       "tokens, not threads; the yeto launcher gives a verl island max(vCPUs, 32)) or free the node")


class RayResourcesUnsatisfiable(TopologyError):
    """The island's Ray cannot place what the verl trainer asks for (module doc)."""


def schedule_timeout_s(environ: Mapping[str, str] | None = None) -> float:
    """Seconds a trainer actor or a worker placement group may stay pending
    (:data:`SCHEDULE_TIMEOUT_ENV`, default :data:`SCHEDULE_TIMEOUT_S`)."""
    raw = (os.environ if environ is None else environ).get(SCHEDULE_TIMEOUT_ENV, "")
    try:
        value = float(raw)
    except ValueError:
        return SCHEDULE_TIMEOUT_S
    return value if value > 0 else SCHEDULE_TIMEOUT_S


def _fmt(resources: Mapping[str, Any] | None) -> str:
    keep = {k: float(v) for k, v in dict(resources or {}).items()
            if k in ("CPU", "GPU", "NPU") or k.startswith("accelerator_type:")}
    return "{" + ", ".join(f"'{k}': {v:g}" for k, v in sorted(keep.items())) + "}"


def trainer_cpus() -> float:
    """CPU slots the trainer actor and its task runner hold at once."""
    return float(TRAINER_ACTOR_CPUS + TASK_RUNNER_CPUS)


def cpu_shortfall(ray_info: Mapping[str, Any]) -> str | None:
    """The refusal when the island Ray (``island_entry.ensure_ray_cpus``'s record: the
    resources after a restart, else before) holds fewer CPU slots than :func:`trainer_cpus`;
    None when they fit or the resources are unknown."""
    resources = ray_info.get("after") or ray_info.get("before")
    if not resources or "CPU" not in resources:
        return None
    have = float(resources.get("CPU") or 0)
    need = trainer_cpus()
    if have >= need:
        return None
    return (f"the island Ray has {have:g} CPU slots ({_fmt(resources)}) but the verl trainer needs "
            f"{need:g} (trainer actor {TRAINER_ACTOR_CPUS} + task runner {TASK_RUNNER_CPUS}); Ray would "
            f"leave the trainer pending with no error. Fix: {FIX}")


def await_actor(actor: Any, *, demand: Mapping[str, float], timeout_s: float | None = None,
                ray_mod: Any = None) -> None:
    """Wait until ``actor`` is placed (its ``__ray_ready__``), at most ``timeout_s``
    (:func:`schedule_timeout_s`); raises :class:`RayResourcesUnsatisfiable` after it."""
    if ray_mod is None:
        import ray as ray_mod  # noqa: PLC0415 - the trainer process
    limit = schedule_timeout_s() if timeout_s is None else float(timeout_s)
    ready, _ = ray_mod.wait([actor.__ray_ready__.remote()], timeout=limit)
    if ready:
        return
    raise RayResourcesUnsatisfiable(
        f"the verl trainer actor asked Ray for {_fmt(demand)} and was not placed within {limit:g} s; "
        f"the island Ray has {_fmt(ray_mod.cluster_resources())} in all, "
        f"{_fmt(ray_mod.available_resources())} free. Fix: {FIX}")


def get_watching_placements(ref: Any, *, timeout_s: float | None = None, poll_s: float = 10.0,
                            ray_mod: Any = None, clock: Callable[[], float] = time.monotonic) -> Any:
    """``ray.get(ref)`` (the trainer's ``init_workers``, which may take long: a model load)
    while no placement group stays ``PENDING`` longer than ``timeout_s``
    (:func:`schedule_timeout_s`): one that does is a worker group Ray cannot place
    (cards or CPU slots it does not have), and :class:`RayResourcesUnsatisfiable` is raised."""
    if ray_mod is None:
        import ray as ray_mod  # noqa: PLC0415 - the trainer process
    limit = schedule_timeout_s() if timeout_s is None else float(timeout_s)
    pending_since: dict[str, float] = {}
    while True:
        ready, _ = ray_mod.wait([ref], timeout=poll_s)
        if ready:
            return ray_mod.get(ref)
        now = clock()
        table = ray_mod.util.placement_group_table() or {}
        pending = {str(pid): info for pid, info in table.items()
                   if str((info or {}).get("state") or "").upper() == "PENDING"}
        pending_since = {pid: pending_since.get(pid, now) for pid in pending}
        stuck = sorted(pid for pid, since in pending_since.items() if now - since > limit)
        if stuck:
            bundles = [list(((pending[pid] or {}).get("bundles") or {}).values()) for pid in stuck]
            asked: dict[str, float] = {}
            for group in bundles:
                for bundle in group:
                    for key, value in dict(bundle or {}).items():
                        asked[key] = asked.get(key, 0.0) + float(value)
            raise RayResourcesUnsatisfiable(
                f"{len(stuck)} placement group(s) of the verl trainer's workers stayed pending for "
                f"more than {limit:g} s, asking for {_fmt(asked)} in all (bundles {bundles}); the island "
                f"Ray has {_fmt(ray_mod.cluster_resources())} in all, {_fmt(ray_mod.available_resources())} "
                f"free. Fix: {FIX}")
