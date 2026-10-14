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
the fleet controller does not start it; the binding starts one per incarnation.
Only a Modal trainer island is bound (``yeto.launch.trainer_island``); :func:`check`
refuses anything else before any launch.
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
from yeto.rl.engine.trainer_loop import TrainerLoop, TrainerLoopThread, next_incarnation
from yeto.rl.engine.trainer_scheduler import S1Config, S1Scheduler

from .trainer_island import CloudTrainerBinding, TrainerIsland

OFF = "off"
S1 = "s1"
MODES = (OFF, S1)
DEFAULT_INTERVAL_S = 30.0
DEFAULT_START_SECONDS = 600.0
DEFAULT_DRAIN_SECONDS = 3600.0
TRAINER_PRECISION = "bf16"  # the trainer's precision; a convertible node must match it
JOURNAL_DIR = "trainer-loop"

T = TypeVar("T")


def default_binding(args, island: TrainerIsland | None) -> CloudTrainerBinding:
    if island is None or island.cloud != "modal":
        raise SystemExit("--rl-trainer-scheduler: the trainer island must be a Modal island "
                         f"(got {None if island is None else island.cloud}); sky is not bound yet")
    from .trainer_island import modal_binding

    return modal_binding(args, island, journal_dir=journal_dir(args))


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
        "and stops; the fleet does not start it. Must be a Modal island with a "
        "modal-volume:// --rl-checkpoint-store")
    group.add_argument(
        "--rl-trainer-command-poll-seconds", type=float, default=None,
        help="seconds between two reads of the command file on the trainer island and of "
        "the island events on the head (default 5)")
    group.add_argument(
        "--rl-trainer-drain-seconds", type=float, default=DEFAULT_DRAIN_SECONDS,
        help="after the fleet (the samplers) ended: max seconds the scheduler keeps the "
        "trainer to consume the rest of the buffer before it stops it (default 3600)")
    group.add_argument(
        "--rl-trainer-start-seconds", type=float, default=DEFAULT_START_SECONDS,
        help="configured trainer start time T_s (seconds); a measured start replaces it")
    group.add_argument(
        "--rl-trainer-start-seconds-fixed", action="store_true",
        help="keep --rl-trainer-start-seconds as T_s after a start was measured (an override "
        "for check runs; the measured start is still journaled)")
    group.add_argument(
        "--rl-node-class", action="append", default=[], metavar="NAME=CLASS[:spot][:PRECISION]",
        help="node class of one node (rollout, trainer, convertible; rl-trainer-start-stop "
        "D7). A node without ':spot' follows --spot/--on-demand. A convertible node must be "
        f"on-demand and at the trainer precision ({TRAINER_PRECISION}). Repeat per node")


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


def enabled(args) -> bool:
    return (getattr(args, "rl_trainer_scheduler", OFF) or OFF) != OFF


def journal_dir(args) -> Path:
    from yeto import runs

    return runs.run_dir(args.cluster_prefix) / JOURNAL_DIR


def held_island(args) -> int | None:
    """Index (into ``--gpu``) of the island the scheduler owns; None with the switch off."""
    return args.rl_trainer_island if enabled(args) else None


def check(args) -> None:
    """Refuse bad node classes, and the switch without a trainer island binding
    (before any launch)."""
    check_node_classes(args)
    if not enabled(args):
        return
    if binding_factory is None:
        raise SystemExit(f"--rl-trainer-scheduler {args.rl_trainer_scheduler}: no trainer "
                         "island binding to the cloud layer")
    if args.rl_trainer_scheduler_interval <= 0:
        raise SystemExit("--rl-trainer-scheduler-interval must be positive")
    if getattr(args, "rl_trainer_drain_seconds", DEFAULT_DRAIN_SECONDS) < 0:
        raise SystemExit("--rl-trainer-drain-seconds must be >= 0")
    if binding_factory is default_binding:
        _check_modal_island(args)


def _check_modal_island(args) -> None:
    from yeto.gpu_spec import parse_gpu_spec

    m = getattr(args, "rl_trainer_island", None)
    if m is None:
        raise SystemExit("--rl-trainer-scheduler needs --rl-trainer-island M (the island it "
                         "starts and stops)")
    specs = parse_gpu_spec(args.gpu)
    if not 0 <= m < len(specs):
        raise SystemExit(f"--rl-trainer-island {m}: --gpu has {len(specs)} island(s)")
    if specs[m].cloud != "modal":
        raise SystemExit(f"--rl-trainer-island {m} is a {specs[m].cloud} island; only a Modal "
                         "trainer island is bound to the scheduler (sky: not implemented)")
    if not str(getattr(args, "rl_checkpoint_store", None) or "").startswith("modal-volume://"):
        raise SystemExit("--rl-trainer-scheduler needs a modal-volume:// --rl-checkpoint-store: "
                         "the head reads the cuts and the sample buffer there and writes the "
                         "trainer commands there")


def build(args, binding: CloudTrainerBinding, journal: Journal) -> TrainerLoop:
    """Scheduler S1 + lifecycle + loop over ``journal``; restored from its records."""
    records = journal.records
    scheduler = S1Scheduler(
        S1Config(a_hard=int(getattr(args, "rl_max_policy_age", 0) or 0),
                 start_seconds=float(args.rl_trainer_start_seconds),
                 fixed_start_seconds=bool(getattr(args, "rl_trainer_start_seconds_fixed", False))),
        start_events=binding.start_events, record=journal.append)
    loop = TrainerLoop(scheduler=scheduler, lifecycle=binding.lifecycle(next_incarnation(records)),
                       buffer=binding.buffer, cuts=binding.cuts, record=journal.append,
                       data_ended=binding.data_ended, release_orphan=binding.release_orphan,
                       spec=binding.spec)
    if loop.restore(records):
        print(f"[launcher] trainer scheduler continues from {journal.dir}", flush=True)
    if binding.after_restore is not None:
        binding.after_restore()
    return loop


def run_with_trainer_loop(args, run: Callable[[], T], *, island: TrainerIsland | None = None,
                          factory: Callable[[Any, TrainerIsland | None], CloudTrainerBinding]
                          | None = None) -> T:
    """``run()`` (the fleet controller); with the switch on, the trainer loop ticks
    on its own thread meanwhile. ``island``: the held-back trainer island
    (:func:`held_island`).

    When ``run`` returns, the fleet (the samplers) has ended: the data ended. The
    loop keeps ticking up to ``--rl-trainer-drain-seconds`` so the trainer consumes
    the rest of the buffer and the loop finishes (backlog under one batch); then
    the trainer is stopped. When ``run`` raises, the trainer is stopped at once."""
    if not enabled(args):
        return run()
    factory = binding_factory if factory is None else factory
    if factory is None:
        check(args)  # raises
    binding = factory(args, island)
    fleet_ended = threading.Event()
    cloud_data_ended = binding.data_ended
    binding = dataclasses.replace(
        binding, data_ended=lambda: fleet_ended.is_set() or bool(cloud_data_ended()))
    with Journal(journal_dir(args)) as journal:
        runner = TrainerLoopThread(build(args, binding, journal),
                                   args.rl_trainer_scheduler_interval).start()
        try:
            result = run()
            fleet_ended.set()
            drain_s = float(getattr(args, "rl_trainer_drain_seconds", DEFAULT_DRAIN_SECONDS))
            if not runner.join(drain_s):
                print(f"[launcher] trainer scheduler: buffer not drained after {drain_s:.0f} s "
                      "since the fleet ended; stopping the trainer", file=sys.stderr, flush=True)
            return result
        finally:
            runner.stop()
            if runner.error is not None:
                print(f"[launcher] trainer scheduler stopped on error: {runner.error!r}",
                      file=sys.stderr, flush=True)
