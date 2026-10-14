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
  releases it (sky / Modal launch layer), and gives the sample buffer and the cut
  log of the run.

No binding factory is set yet: with the switch on, the launcher refuses before
anything runs.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from yeto.rl.engine.convertible import NodeSpec, parse_node_spec, validate_nodes
from yeto.rl.engine.journal import Journal
from yeto.rl.engine.trainer_lifecycle import CutLog, DurableSampleBuffer, TrainerLifecycle
from yeto.rl.engine.trainer_loop import TrainerLoop, TrainerLoopThread, next_incarnation
from yeto.rl.engine.trainer_scheduler import S1Config, S1Scheduler

OFF = "off"
S1 = "s1"
MODES = (OFF, S1)
DEFAULT_INTERVAL_S = 30.0
DEFAULT_START_SECONDS = 600.0
TRAINER_PRECISION = "bf16"  # the trainer's precision; a convertible node must match it
JOURNAL_DIR = "trainer-loop"

T = TypeVar("T")


@dataclass
class CloudTrainerBinding:
    """What the loop needs from the cloud layer for one run.

    ``lifecycle(first_incarnation)`` builds the trainer lifecycle (e.g.
    ``DriverLifecycle`` whose ``build_driver`` acquires a trainer island and whose
    ``release`` gives it back). ``release_orphan(incarnation)`` gives back the island
    a dead head started. ``data_ended`` tells the samplers finished.
    ``after_restore()`` runs after the journal is read back and an orphan released
    (e.g. reset the head's buffer to the watermark of the newest cut)."""

    lifecycle: Callable[[int], TrainerLifecycle]
    buffer: DurableSampleBuffer
    cuts: CutLog
    release_orphan: Callable[[int], None] | None = None
    data_ended: Callable[[], bool] = lambda: False
    start_events: Callable[[], Iterable[Mapping[str, Any]]] | None = None
    spec: Mapping[str, Any] | None = None
    after_restore: Callable[[], None] | None = None


# (args) -> CloudTrainerBinding. None until the cloud layer lands (L5.4).
binding_factory: Callable[[Any], CloudTrainerBinding] | None = None


def add_arguments(group) -> None:
    group.add_argument(
        "--rl-trainer-scheduler", choices=MODES, default=OFF,
        help="start and stop the trainer island by scheduler S1 from the sample-buffer "
        "backlog (rl-trainer-start-stop); off (default) keeps the trainer up for the run")
    group.add_argument(
        "--rl-trainer-scheduler-interval", type=float, default=DEFAULT_INTERVAL_S,
        help="seconds between two scheduler decisions on the head")
    group.add_argument(
        "--rl-trainer-start-seconds", type=float, default=DEFAULT_START_SECONDS,
        help="configured trainer start time T_s (seconds); a measured start replaces it")
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


def check(args) -> None:
    """Refuse bad node classes, and the switch without a trainer island binding
    (before any launch)."""
    check_node_classes(args)
    if not enabled(args):
        return
    if binding_factory is None:
        raise SystemExit(f"--rl-trainer-scheduler {args.rl_trainer_scheduler}: no trainer "
                         "island binding to the cloud layer yet (rl-trainer-start-stop L5.4)")
    if args.rl_trainer_scheduler_interval <= 0:
        raise SystemExit("--rl-trainer-scheduler-interval must be positive")


def build(args, binding: CloudTrainerBinding, journal: Journal) -> TrainerLoop:
    """Scheduler S1 + lifecycle + loop over ``journal``; restored from its records."""
    records = journal.records
    scheduler = S1Scheduler(
        S1Config(a_hard=int(getattr(args, "rl_max_policy_age", 0) or 0),
                 start_seconds=float(args.rl_trainer_start_seconds)),
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


def run_with_trainer_loop(args, run: Callable[[], T], *,
                          factory: Callable[[Any], CloudTrainerBinding] | None = None) -> T:
    """``run()`` (the fleet controller); with the switch on, the trainer loop ticks
    on its own thread meanwhile and stops the trainer when ``run`` returns."""
    if not enabled(args):
        return run()
    factory = binding_factory if factory is None else factory
    if factory is None:
        check(args)  # raises
    binding = factory(args)
    with Journal(journal_dir(args)) as journal:
        runner = TrainerLoopThread(build(args, binding, journal),
                                   args.rl_trainer_scheduler_interval).start()
        try:
            return run()
        finally:
            runner.stop()
            if runner.error is not None:
                print(f"[launcher] trainer scheduler stopped on error: {runner.error!r}",
                      file=sys.stderr, flush=True)
