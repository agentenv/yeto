"""Trainer scheduler loop on the head (rl-trainer-start-stop L5.3b).

``--rl-trainer-scheduler s1`` runs :class:`~yeto.rl.engine.trainer_loop.TrainerLoop`
on its own thread next to the fleet controller's poll loop. Off (the default), the
launcher behaves as before: :func:`run_with_trainer_loop` only calls ``run``.

Inputs and outputs of the loop on the head:

* journal: ``<run dir>/trainer-loop/journal.jsonl`` (``yeto.rl.engine.journal``).
  A head restart with the same ``--cluster-prefix`` reopens it and continues;
* ``T_s``: the configured ``--rl-trainer-start-seconds`` until a start-phase
  summary event is available (``start_events`` of the cloud binding);
* ``A_hard``: ``--rl-max-policy-age``;
* the trainer island itself: a :class:`CloudTrainerBinding` from
  :data:`binding_factory`. It acquires a trainer island per incarnation and
  releases it, and gives the sample buffer and the cut log of the run.

The launcher holds back the island ``--rl-trainer-island`` (index into ``--gpu``):
the fleet controller does not start it; the binding starts one per incarnation
(``yeto.launch.trainer_island``): a Modal island or a sky island (not Verda). The
head reads the cuts and the buffer and writes the trainer commands in the
``--rl-checkpoint-store``, so the store is one the head reads by API
(``modal-volume://``, ``s3://`` or ``oss://``) that the trainer island and every
sampler island reach (``storage.store_unreachable_reason``). :func:`check` refuses anything
else before any launch.

A convertible pool (every island declared ``--rl-node-class islandM=convertible``,
rl-trainer-start-stop L7.5) turns the loop on by the declaration alone: no island is
held back, a convertible node becomes the trainer (``yeto.launch.convertible_pool``),
and ``T_s`` defaults to the conversion time
(:data:`~.convertible_pool.DEFAULT_CONVERT_START_SECONDS`) until one is measured.
"""

from __future__ import annotations

import dataclasses
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from yeto.rl.engine.convertible import NodeSpec, parse_node_spec, validate_nodes
from yeto.rl.engine.journal import Journal
from yeto.rl.engine.trainer_loop import (
    DEFAULT_MAX_CONSECUTIVE_FAILURES,
    TrainerLoop,
    TrainerLoopGaveUp,
    TrainerLoopThread,
    next_incarnation,
)
from yeto.rl.engine.trainer_scheduler import S1Config, S1Scheduler

from .trainer_island import CloudTrainerBinding, TrainerIsland

OFF = "off"
S1 = "s1"
MODES = (OFF, S1)
DEFAULT_INTERVAL_S = 30.0
DEFAULT_START_SECONDS = 600.0
DEFAULT_DRAIN_SECONDS = 3600.0
DEFAULT_READY_SECONDS = 1800.0
TRAINER_PRECISION = "bf16"  # the trainer's precision; a convertible node must match it
JOURNAL_DIR = "trainer-loop"

T = TypeVar("T")


def default_binding(args, island: TrainerIsland | None) -> CloudTrainerBinding:
    if island is None:
        raise SystemExit("--rl-trainer-scheduler: no trainer island was held back")
    from .trainer_island import modal_binding, sky_binding

    binding = modal_binding if island.cloud == "modal" else sky_binding
    return binding(args, island, journal_dir=journal_dir(args))


# (args, held-back trainer island) -> CloudTrainerBinding
binding_factory: Callable[[Any, TrainerIsland | None], CloudTrainerBinding] | None = default_binding


def add_arguments(group) -> None:
    group.add_argument(
        "--rl-trainer-scheduler", choices=MODES, default=OFF,
        help="start and stop the trainer island by scheduler S1 from the sample-buffer "
        "backlog (rl-trainer-start-stop); off (default) keeps the trainer up for the run")
    group.add_argument(
        "--rl-trainer-scheduler-interval", type=float, default=DEFAULT_INTERVAL_S,
        help="seconds between two scheduler decisions on the head")
    group.add_argument(
        "--rl-trainer-island", type=int, default=None, metavar="M",
        help="with --rl-trainer-scheduler: the island (index into --gpu) the scheduler starts "
        "and stops; the fleet does not start it. Every island must reach the "
        "--rl-checkpoint-store: modal-volume:// on Modal islands, s3:// or oss:// on sky "
        "islands (not Verda)")
    group.add_argument(
        "--rl-trainer-command-poll-seconds", type=float, default=None,
        help="seconds between two reads of the command file on the trainer island and of "
        "the island events on the head (default 5)")
    group.add_argument(
        "--rl-trainer-drain-seconds", type=float, default=DEFAULT_DRAIN_SECONDS,
        help="after the fleet (the samplers) ended: max seconds the scheduler keeps the "
        "trainer to consume the rest of the buffer before it stops it (default 3600)")
    group.add_argument(
        "--rl-trainer-ready-seconds", type=float, default=DEFAULT_READY_SECONDS,
        help="max seconds from a trainer start to the island's ready event (provisioning, "
        "start, cut restore) before the start counts as failed (default "
        f"{DEFAULT_READY_SECONDS:.0f})")
    group.add_argument(
        "--rl-trainer-max-failures", type=int, default=DEFAULT_MAX_CONSECUTIVE_FAILURES,
        help="trainer incarnations that may fail in a row (no trained round between them) "
        "before the scheduler stops, releases the trainer and fails the run (default "
        f"{DEFAULT_MAX_CONSECUTIVE_FAILURES}); a trained round resets the count")
    group.add_argument(
        "--rl-trainer-start-seconds", type=float, default=None,
        help="configured trainer start time T_s (seconds); a measured start replaces it "
        f"(default {DEFAULT_START_SECONDS:g}; a convertible pool: the conversion time, "
        "60 s until one is measured)")
    group.add_argument(
        "--rl-trainer-start-seconds-fixed", action="store_true",
        help="keep --rl-trainer-start-seconds as T_s after a start was measured (an override "
        "for check runs; the measured start is still journaled)")
    group.add_argument(
        "--rl-node-class", action="append", default=[], metavar="NAME=CLASS[:spot][:PRECISION]",
        help="node class of one node (rollout, trainer, convertible; rl-trainer-start-stop "
        "D7). A node without ':spot' follows --spot/--on-demand. A convertible node must be "
        f"on-demand and at the trainer precision ({TRAINER_PRECISION}). Repeat per node. "
        "Declaring every --gpu island convertible (NAME islandM) runs a convertible pool: "
        "the trainer scheduler converts the best node to trainer and back, no other switch")


def node_specs(args) -> list[NodeSpec]:
    """``--rl-node-class`` entries; a node is spot if it says so or the run is spot."""
    run_spot = bool(getattr(args, "spot", False))
    specs = []
    for text in getattr(args, "rl_node_class", None) or ():
        spec = parse_node_spec(text)
        if run_spot and not spec.spot:
            spec = NodeSpec(spec.name, spec.node_class, spot=True,
                            precision=spec.precision, capacity=spec.capacity)
        specs.append(spec)
    return specs


def check_node_classes(args) -> list[NodeSpec]:
    """Refuse a launch whose node classes break the class rule (L7.1)."""
    try:
        specs = node_specs(args)
        validate_nodes(specs, trainer_precision=TRAINER_PRECISION)
    except ValueError as exc:  # NodeClassError, or a bad capacity number
        raise SystemExit(f"--rl-node-class: {exc}") from None
    return specs


def convertible_pool(args) -> bool:
    from . import convertible_pool as pool

    return pool.enabled(args)


def enabled(args) -> bool:
    """The trainer loop runs: ``--rl-trainer-scheduler`` or a convertible pool."""
    return (getattr(args, "rl_trainer_scheduler", OFF) or OFF) != OFF or convertible_pool(args)


def scheduler_mode(args) -> str:
    """The scheduler that decides (S1 is the only one; a convertible pool uses it)."""
    mode = getattr(args, "rl_trainer_scheduler", OFF) or OFF
    return S1 if mode == OFF and convertible_pool(args) else mode


def start_seconds(args) -> float:
    """Configured ``T_s``: the flag, else the conversion time of a convertible pool,
    else the cold-start default."""
    value = getattr(args, "rl_trainer_start_seconds", None)
    if value is not None:
        return float(value)
    if convertible_pool(args):
        from .convertible_pool import DEFAULT_CONVERT_START_SECONDS

        return DEFAULT_CONVERT_START_SECONDS
    return DEFAULT_START_SECONDS


def journal_dir(args) -> Path:
    from yeto import runs

    return runs.run_dir(args.cluster_prefix) / JOURNAL_DIR


def held_island(args) -> int | None:
    """Index (into ``--gpu``) of the island the scheduler owns; None with the switch off
    and for a convertible pool (a node converts; nothing is held back)."""
    return args.rl_trainer_island if enabled(args) and not convertible_pool(args) else None


def check(args) -> None:
    """Refuse bad node classes, and the switch without a trainer island binding
    (before any launch)."""
    check_node_classes(args)
    if not enabled(args):
        return
    if convertible_pool(args):
        from . import convertible_pool as pool

        pool.check(args)
    elif binding_factory is None:
        raise SystemExit(f"--rl-trainer-scheduler {args.rl_trainer_scheduler}: no trainer "
                         "island binding to the cloud layer")
    if args.rl_trainer_scheduler_interval <= 0:
        raise SystemExit("--rl-trainer-scheduler-interval must be positive")
    if getattr(args, "rl_trainer_drain_seconds", DEFAULT_DRAIN_SECONDS) < 0:
        raise SystemExit("--rl-trainer-drain-seconds must be >= 0")
    if getattr(args, "rl_trainer_max_failures", DEFAULT_MAX_CONSECUTIVE_FAILURES) < 1:
        raise SystemExit("--rl-trainer-max-failures must be >= 1")
    if getattr(args, "rl_trainer_ready_seconds", DEFAULT_READY_SECONDS) <= 0:
        raise SystemExit("--rl-trainer-ready-seconds must be positive")
    if binding_factory is default_binding and not convertible_pool(args):
        _check_trainer_island(args)


def _check_trainer_island(args) -> None:
    from yeto.gpu_spec import parse_gpu_spec

    m = getattr(args, "rl_trainer_island", None)
    if m is None:
        raise SystemExit("--rl-trainer-scheduler needs --rl-trainer-island M (the island it "
                         "starts and stops)")
    specs = parse_gpu_spec(args.gpu)
    if not 0 <= m < len(specs):
        raise SystemExit(f"--rl-trainer-island {m}: --gpu has {len(specs)} island(s)")
    cloud = specs[m].cloud
    if cloud == "verda":
        raise SystemExit(f"--rl-trainer-island {m} is a Verda island; the scheduler launches "
                         "a cluster per trainer start, which the Verda launch path does not do")
    from .storage import HEAD_READABLE_STORE_SCHEMES, store_unreachable_reason

    store = str(getattr(args, "rl_checkpoint_store", None) or "")
    if store.partition("://")[0] not in HEAD_READABLE_STORE_SCHEMES or "://" not in store:
        raise SystemExit(f"--rl-trainer-scheduler needs a --rl-checkpoint-store the head reads "
                         f"({', '.join(s + '://' for s in HEAD_READABLE_STORE_SCHEMES)}): the head "
                         f"reads the cuts and the sample buffer there and writes the trainer "
                         f"commands there (got {store or 'none'!r})")
    from yeto.rl.engine.sample_roles import SAMPLER, island_roles

    samplers = [i for i, role in island_roles(args, len(specs)).items() if role == SAMPLER]
    for i in [m, *sorted(samplers)]:
        reason = store_unreachable_reason(store, specs[i].cloud)
        if reason is not None:
            role = "trainer island" if i == m else "sampler island"
            raise SystemExit(f"--rl-trainer-scheduler: {role} {i} ({specs[i].cloud}) does not reach "
                             f"--rl-checkpoint-store {store}: {reason}")


def build(args, binding: CloudTrainerBinding, journal: Journal) -> TrainerLoop:
    """Scheduler S1 + lifecycle + loop over ``journal``; restored from its records."""
    records = journal.records
    scheduler = S1Scheduler(
        S1Config(a_hard=int(getattr(args, "rl_max_policy_age", 0) or 0),
                 start_seconds=start_seconds(args),
                 fixed_start_seconds=bool(getattr(args, "rl_trainer_start_seconds_fixed", False))),
        start_events=binding.start_events, record=journal.append)
    if binding.wrap_scheduler is not None:
        scheduler = binding.wrap_scheduler(scheduler, journal.append)
    loop = TrainerLoop(scheduler=scheduler, lifecycle=binding.lifecycle(next_incarnation(records)),
                       buffer=binding.buffer, cuts=binding.cuts, record=journal.append,
                       data_ended=binding.data_ended, release_orphan=binding.release_orphan,
                       spec=binding.spec, needs_first_policy=binding.needs_first_policy,
                       ready_timeout_s=float(getattr(
                           args, "rl_trainer_ready_seconds", DEFAULT_READY_SECONDS)),
                       max_consecutive_failures=int(getattr(
                           args, "rl_trainer_max_failures", DEFAULT_MAX_CONSECUTIVE_FAILURES)))
    if loop.restore(records):
        print(f"[launcher] trainer scheduler continues from {journal.dir}", flush=True)
    if binding.after_restore is not None:
        binding.after_restore()
    return loop


def run_with_trainer_loop(args, run: Callable[[], T], *, island: TrainerIsland | None = None,
                          factory: Callable[[Any, TrainerIsland | None], CloudTrainerBinding]
                          | None = None,
                          abort: Callable[[BaseException], None] | None = None) -> T:
    """``run()`` (the fleet controller); with the switch on, the trainer loop ticks
    on its own thread meanwhile. ``island``: the held-back trainer island
    (:func:`held_island`).

    When ``run`` returns, the fleet (the samplers) has ended: the data ended. The
    loop keeps ticking up to ``--rl-trainer-drain-seconds`` so the trainer consumes
    the rest of the buffer and the loop finishes (backlog under one batch); then
    the trainer is stopped. When ``run`` raises, the trainer is stopped at once.

    ``abort(error)``: called once from the loop thread when the loop gives up
    (:class:`TrainerLoopGaveUp`); the fleet controller then ends ``run`` with that
    error so the samplers stop too. Without it the error is raised after ``run``."""
    if not enabled(args):
        return run()
    if factory is None and convertible_pool(args):
        from . import convertible_pool as pool

        def factory(a: Any, _island: TrainerIsland | None) -> CloudTrainerBinding:
            return pool.binding(a, journal_dir=journal_dir(a), nodes=node_specs(a),
                                precision=TRAINER_PRECISION)
    factory = binding_factory if factory is None else factory
    if factory is None:
        check(args)  # raises
    binding = factory(args, island)
    fleet_ended = threading.Event()
    cloud_data_ended = binding.data_ended
    binding = dataclasses.replace(
        binding, data_ended=lambda: fleet_ended.is_set() or bool(cloud_data_ended()))
    with Journal(journal_dir(args)) as journal:
        def on_error(error: BaseException) -> None:
            if abort is not None and isinstance(error, TrainerLoopGaveUp):
                abort(error)

        runner = TrainerLoopThread(build(args, binding, journal),
                                   args.rl_trainer_scheduler_interval, on_error=on_error,
                                   on_finished=binding.on_finished).start()
        try:
            result = run()
            if isinstance(runner.error, TrainerLoopGaveUp):
                raise runner.error
            fleet_ended.set()
            drain_s = float(getattr(args, "rl_trainer_drain_seconds", DEFAULT_DRAIN_SECONDS))
            if not runner.join(drain_s):
                print(f"[launcher] trainer scheduler: buffer not drained after {drain_s:.0f} s "
                      "since the fleet ended; stopping the trainer", file=sys.stderr, flush=True)
            return result
        finally:
            runner.stop()
            if binding.on_finished is not None and not runner.loop.finished:
                binding.on_finished()  # the run ended early: end the nodes too
            if runner.error is not None:
                print(f"[launcher] trainer scheduler stopped on error: {runner.error!r}",
                      file=sys.stderr, flush=True)
