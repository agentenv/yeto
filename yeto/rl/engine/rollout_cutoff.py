"""agentic-rollout-utilization stage 0: over-sample, cut off at the target, discard.

Backend-neutral (no torch/ray/miles). A rollout submits ``submitted`` groups
(over-sampling); the round ends as soon as ``target`` groups have completed and
been kept; what is still in flight is cut off. With a policy-age limit of 0
(the default, deterministic mode) cut-off trajectories are discarded -- never
trained, never carried over -- and reported on one ``rl_rollout_cutoff`` event:
how many groups/trajectories were discarded and how many response tokens they
had already generated. Dynamic-filter drops are a separate count
(``filtered_groups``), so the two kinds of discard never mix (design decision 4).

Engines do the cut-off themselves (Miles ``generate_rollout`` aborts in-flight
groups); adapters translate what the engine reports into
:class:`CutoffReport`. :func:`cutoff_at_target` is the reference semantics used
by the fake engine and tests.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

CUTOFF_EVENT = "rl_rollout_cutoff"
# Event fields shared by every backend (Miles, verl): spec "两种训练后端行为一致".
CUTOFF_FIELDS = (
    "submitted_groups",
    "target_groups",
    "filtered_groups",
    "discarded_groups",
    "discarded_trajectories",
    "discarded_tokens",
    "discarded_unknown_groups",
    "mechanism",
)


@dataclass(frozen=True)
class Completion:
    """One submitted group, in the order it would finish (fake/reference engine)."""

    group_id: str
    trajectories: int
    tokens: int
    keep: bool = True  # False: dropped by the dynamic filter


@dataclass(frozen=True)
class CutoffOutcome:
    kept: tuple[Completion, ...]
    filtered: tuple[Completion, ...]
    discarded: tuple[Completion, ...]


def cutoff_at_target(finish_order: Sequence[Completion], target: int) -> CutoffOutcome:
    """Keep the first ``target`` completions the filter accepts; everything still
    in flight at that point is cut off. Fewer than ``target`` acceptable groups:
    the round cannot fill (``ValueError``; the engine would draw more)."""

    if type(target) is not int or target <= 0:
        raise ValueError("cut-off target must be a positive int")
    kept: list[Completion] = []
    filtered: list[Completion] = []
    for i, c in enumerate(finish_order):
        (kept if c.keep else filtered).append(c)
        if len(kept) == target:
            return CutoffOutcome(tuple(kept), tuple(filtered), tuple(finish_order[i + 1:]))
    raise ValueError(f"only {len(kept)} of {target} groups were kept; the round did not fill")


@dataclass(frozen=True)
class CutoffReport:
    """What one rollout's cut-off threw away. None = the engine did not report it."""

    submitted_groups: int | None
    target_groups: int
    filtered_groups: int | None
    discarded_groups: int | None
    discarded_trajectories: int | None = None
    discarded_tokens: int | None = None
    discarded_unknown_groups: int | None = None
    mechanism: str | None = None

    @classmethod
    def from_outcome(cls, outcome: CutoffOutcome, *, mechanism: str = "reference") -> CutoffReport:
        return cls(
            submitted_groups=len(outcome.kept) + len(outcome.filtered) + len(outcome.discarded),
            target_groups=len(outcome.kept),
            filtered_groups=len(outcome.filtered),
            discarded_groups=len(outcome.discarded),
            discarded_trajectories=sum(c.trajectories for c in outcome.discarded),
            discarded_tokens=sum(c.tokens for c in outcome.discarded),
            discarded_unknown_groups=0,
            mechanism=mechanism,
        )

    def fields(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in CUTOFF_FIELDS}


def cutoff_report(batch: Any) -> CutoffReport | None:
    """The cut-off a :class:`~yeto.rl.engine.ports.RolloutBatchHandle` reports, or
    None when nothing was cut off / nothing is known (default runs emit nothing)."""

    groups = getattr(batch, "aborted_in_flight_groups", None)
    trajectories = getattr(batch, "aborted_in_flight_trajectories", None)
    if not groups and trajectories is None:
        return None
    return CutoffReport(
        submitted_groups=getattr(batch, "submitted_groups", None),
        target_groups=len(getattr(batch, "groups", ()) or ()),
        filtered_groups=getattr(batch, "filtered", None),
        discarded_groups=None if groups is None else int(groups),
        discarded_trajectories=trajectories,
        discarded_tokens=getattr(batch, "aborted_in_flight_tokens", None),
        discarded_unknown_groups=getattr(batch, "aborted_in_flight_unknown_groups", None),
        mechanism=getattr(batch, "abort_mechanism", None),
    )


def discard_stats_fields(stats: Mapping[str, Any] | None) -> dict[str, int]:
    """Translate an engine's ``{groups, samples, response_tokens, unknown_groups}``
    discard tally (Miles ``args.rollout_abort_discard_stats``; verl the same keys)
    into rollout-metadata fields. Missing/invalid: {} (unknown, never guessed)."""

    if not isinstance(stats, Mapping):
        return {}
    keys = {"groups": "aborted_in_flight_groups",
            "samples": "aborted_in_flight_trajectories",
            "response_tokens": "aborted_in_flight_tokens",
            "unknown_groups": "aborted_in_flight_unknown_groups"}
    out: dict[str, int] = {}
    for src, dst in keys.items():
        value = stats.get(src)
        if type(value) is not int or value < 0:
            return {}
        out[dst] = value
    return out

