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
* :func:`await_actor` / :func:`get_watching_placements`: the trainer actor, or a placement
  group of its workers, asks for more than the island Ray has in all (it can never be
  placed) -- a stop after :data:`INFEASIBLE_GRACE_S`; Ray reports it still waiting for its
  resources (actor ``PENDING_CREATION``, group ``PENDING``) after :func:`schedule_timeout_s`
  (default 1800 s) -- a stop. A trainer that is only slow (a cold start: the model download,
  the FSDP build, a first import) is never stopped: its actor is placed, and its wait has no
  limit here.

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
SCHEDULE_TIMEOUT_S = 1800.0
# A request larger than the Ray's total is refused after this long (the totals are read on
# two polls, so a resource read mid-start is not taken for the Ray's size)
INFEASIBLE_GRACE_S = 30.0
FIX = ("start the island Ray with more CPU slots (ray start --num-cpus N; Ray CPUs are scheduling "
       "tokens, not threads; the yeto launcher gives a verl island max(vCPUs, 32)) or free the node")


class RayResourcesUnsatisfiable(TopologyError):
    """The island's Ray cannot place what the verl trainer asks for (module doc)."""


def schedule_timeout_s(environ: Mapping[str, str] | None = None) -> float:
    """Seconds a trainer actor or a worker placement group Ray reports as still waiting
    for its resources may stay so (:data:`SCHEDULE_TIMEOUT_ENV`, default
    :data:`SCHEDULE_TIMEOUT_S`): the backstop of a request that fits the Ray's total but
    never gets placed (the slots held by something else)."""
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


def exceeds(demand: Mapping[str, float], total: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    """``{resource: (asked, total)}`` of each resource ``demand`` asks for more of than the
    island Ray has in all (empty: it fits the totals)."""
    return {key: (float(value), float(total.get(key) or 0)) for key, value in demand.items()
            if float(value) > float(total.get(key) or 0)}


def _actor_state(actor: Any) -> str | None:
    """Ray's state of ``actor`` (``PENDING_CREATION``, ``ALIVE``, ...; the state API), None
    when it cannot be read (no dashboard): unknown, never taken for pending."""
    try:
        from ray.util.state import get_actor  # noqa: PLC0415 - the trainer process

        found = get_actor(actor._actor_id.hex())
        return None if found is None else str(getattr(found, "state", None) or "") or None
    except Exception:  # noqa: BLE001 - a state read never stops the island
        return None


def await_actor(actor: Any, *, demand: Mapping[str, float], timeout_s: float | None = None,
                poll_s: float = 10.0, grace_s: float = INFEASIBLE_GRACE_S, ray_mod: Any = None,
                clock: Callable[[], float] = time.monotonic,
                state_of: Callable[[Any], str | None] = _actor_state) -> None:
    """Wait until ``actor`` is ready (its ``__ray_ready__``: placed and constructed). Stops
    (:class:`RayResourcesUnsatisfiable`) only on Ray's own evidence that it cannot be placed:
    ``demand`` exceeds the island Ray's total (after ``grace_s``), or Ray still reports it
    ``PENDING_CREATION`` after ``timeout_s`` (:func:`schedule_timeout_s`). A placed actor
    whose construction is slow is waited for without a limit (module doc)."""
    if ray_mod is None:
        import ray as ray_mod  # noqa: PLC0415 - the trainer process
    limit = schedule_timeout_s() if timeout_s is None else float(timeout_s)
    ready_ref = actor.__ray_ready__.remote()
    started = clock()
    over_since: float | None = None
    warned = False
    while True:
        ready, _ = ray_mod.wait([ready_ref], timeout=poll_s)
        if ready:
            return
        now = clock()
        total = ray_mod.cluster_resources()
        over = exceeds(demand, total)
        over_since = (over_since if over_since is not None else now) if over else None
        if over and now - over_since >= grace_s:
            raise RayResourcesUnsatisfiable(
                f"the verl trainer actor asks Ray for {_fmt(demand)}, more than the island Ray has in all "
                f"({_fmt(total)}; {', '.join(f'{k} {a:g} > {t:g}' for k, (a, t) in sorted(over.items()))}): "
                f"it can never be placed. Fix: {FIX}")
        if now - started >= limit:
            state = state_of(actor)
            if state == "PENDING_CREATION":
                raise RayResourcesUnsatisfiable(
                    f"the verl trainer actor asked Ray for {_fmt(demand)} and Ray still reports it "
                    f"PENDING_CREATION after {limit:g} s: the island Ray has {_fmt(total)} in all, "
                    f"{_fmt(ray_mod.available_resources())} free. Fix: {FIX}")
            if not warned:  # slow, or a state Ray does not tell: waited for, never stopped
                warned = True
                print(f"[yeto-verl] the trainer actor is not ready after {now - started:.0f} s "
                      f"(Ray state {state or 'unknown'}); waiting", flush=True)


def _bundle_sum(groups: list[list[Mapping[str, Any]]]) -> dict[str, float]:
    asked: dict[str, float] = {}
    for group in groups:
        for bundle in group:
            for key, value in dict(bundle or {}).items():
                asked[key] = asked.get(key, 0.0) + float(value)
    return asked


def get_watching_placements(ref: Any, *, timeout_s: float | None = None, poll_s: float = 10.0,
                            grace_s: float = INFEASIBLE_GRACE_S, ray_mod: Any = None,
                            clock: Callable[[], float] = time.monotonic) -> Any:
    """``ray.get(ref)`` (the trainer's ``init_workers``, which may take long: a model load)
    while watching the placement groups Ray holds ``PENDING``: one whose bundles ask for more
    than the island Ray has in all is refused after ``grace_s``, one still pending after
    ``timeout_s`` (:func:`schedule_timeout_s`) too (:class:`RayResourcesUnsatisfiable`). A
    group that is placed is never a fault, however long the init then takes."""
    if ray_mod is None:
        import ray as ray_mod  # noqa: PLC0415 - the trainer process
    limit = schedule_timeout_s() if timeout_s is None else float(timeout_s)
    pending_since: dict[str, float] = {}
    over_since: dict[str, float] = {}
    while True:
        ready, _ = ray_mod.wait([ref], timeout=poll_s)
        if ready:
            return ray_mod.get(ref)
        now = clock()
        table = ray_mod.util.placement_group_table() or {}
        pending = {str(pid): info for pid, info in table.items()
                   if str((info or {}).get("state") or "").upper() == "PENDING"}
        pending_since = {pid: pending_since.get(pid, now) for pid in pending}
        total = ray_mod.cluster_resources() if pending else {}
        bundles = {pid: list(((info or {}).get("bundles") or {}).values()) for pid, info in pending.items()}
        over_since = {pid: over_since.get(pid, now) for pid in pending if exceeds(_bundle_sum([bundles[pid]]), total)}
        infeasible = sorted(pid for pid, since in over_since.items() if now - since >= grace_s)
        stuck = sorted(pid for pid, since in pending_since.items() if now - since > limit)
        if infeasible or stuck:
            named = infeasible or stuck
            asked = _bundle_sum([bundles[pid] for pid in named])
            why = ("ask for more than the island Ray has in all: they can never be placed" if infeasible else
                   f"stayed pending for more than {limit:g} s")
            raise RayResourcesUnsatisfiable(
                f"{len(named)} placement group(s) of the verl trainer's workers {why}, asking for "
                f"{_fmt(asked)} in all (bundles {[bundles[pid] for pid in named]}); the island Ray has "
                f"{_fmt(total)} in all, {_fmt(ray_mod.available_resources())} free. Fix: {FIX}")
