"""Miles/Megatron translation of the core LR schedule (yeto-framework-decoupling
task 4.1, audit E11). The schedule (decay style, horizon) is decided in the core
(:func:`yeto.rl.engine.run_config.resolve_lr_schedule`); only the flag spelling
lives here, moved verbatim from ``run_config.py``.
"""

from __future__ import annotations

from yeto.rl.engine.run_config import LrSchedule

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


def miles_lr_schedule_sha256(miles_args) -> str | None:
    """S17 N17: engine-neutral hash of the LR schedule a Miles namespace applies
    (``--lr``, ``--lr-decay-style``, ``--lr-decay-iters``, ``--lr-warmup-iters``,
    ``--min-lr``).  It goes into the ExecutionProfile contract hash and is bound
    into the identity the syncer compares, so islands with different schedules
    (e.g. one with ``--rl-lr-schedule constant``, one linear) are refused.
    None when the namespace carries no schedule (Miles left it implicit)."""
    from yeto.rl.engine.backend_identity import lr_schedule_sha256

    style = getattr(miles_args, "lr_decay_style", None)
    lr = getattr(miles_args, "lr", None)
    if style is None or lr is None:
        return None
    return lr_schedule_sha256(style, getattr(miles_args, "lr_decay_iters", None), lr,
                              getattr(miles_args, "lr_warmup_iters", 0) or 0,
                              getattr(miles_args, "min_lr", 0) or 0)
