"""Miles/Megatron translation of the core LR schedule (yeto-framework-decoupling
task 4.1, audit E11). The schedule (decay style, horizon) is decided in the core
(:func:`yeto.rl.engine.run_config.resolve_lr_schedule`); only the flag spelling
lives here, moved verbatim from ``run_config.py``.
"""

from __future__ import annotations

from ..run_config import LrSchedule

LR_SCHEDULE_FLAGS = ("--lr-decay-style", "--lr-decay-iters", "--lr-warmup-iters", "--min-lr")


def lr_schedule_argv(schedule: "LrSchedule | None") -> tuple[str, ...]:
    """Miles/Megatron flags for ``schedule``; shared verbatim by both engines."""

    if schedule is None:
        return ()
    return (
        "--lr-decay-style", schedule.decay_style,
        "--lr-decay-iters", str(schedule.decay_iters),
        "--lr-warmup-iters", "0",
        "--min-lr", "0",
    )
