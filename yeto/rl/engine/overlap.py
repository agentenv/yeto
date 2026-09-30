"""Legal age-0 train/inference overlap: eval || train/outer_sync (rl-infra-spec 2.3).

Under the only certified contract (on-policy, ``max_policy_age == 0``) next-round
generation needs this round's published policy, so generation can never overlap
train/outer_sync/publish (``execution_profile.overlap_violation``). What *is*
independent is evaluation of an already-published policy: it reads the rollout
engines, feeds nothing into training, and the trainer does not touch the rollout
GPUs while it trains or runs the bridge. This module implements exactly that
overlap for ``partitioned-overlap`` and nothing else:

    publish(v_r) -> generate(r) -> [ train(r) -> outer_sync(r) ]  -> join -> publish(v_r+1)
                                   [ eval(v_r) on rollout GPUs  ]

Guards (acceptance X9, checked on every transition, independent of timing):

* bounded: at most ONE eval in flight (``EVAL_QUEUE_BOUND``); a second start is
  refused, never queued;
* the eval measures the policy it was scheduled for: it starts only while the
  published version/token are still the scheduled ones and is joined before any
  later publication (a delayed publish therefore waits for the eval; it never
  lets the eval or a generation run on another version);
* the rollout role is exclusive: no eval in flight while generation starts;
* a full-island quiescent cut and a pause see ``eval_in_flight`` (see
  ``ReadinessSnapshot``) and refuse while it is non-zero.

The driver owns the schedule (hooks after generate / before publish); this class
owns the guard state so it is testable without an engine. Pure: no torch/ray.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .execution_profile import ExecutionProfile

# The only overlap pairs the driver implements (2.3). Other legal pairs of the
# dependency table (reward||checkpoint, ...) have no producer on the ports path.
IMPLEMENTED_OVERLAP = frozenset({("eval", "train"), ("eval", "outer_sync")})
EVAL_QUEUE_BOUND = 1


class OverlapGuardError(RuntimeError):
    """An overlapped eval would observe or block the wrong policy/role."""


class EvalHandle(Protocol):
    def result(self) -> Mapping[str, float]: ...


EvalStarter = Callable[[int], EvalHandle]


def overlap_refusal(profile: ExecutionProfile | None) -> str | None:
    """Why the driver cannot run this ``partitioned-overlap`` profile, or None."""
    if profile is None or profile.execution_mode != "partitioned-overlap":
        return None
    if profile.allowed_overlap != IMPLEMENTED_OVERLAP:
        return (
            f"allowed_overlap {sorted(profile.allowed_overlap)} is not the implemented "
            f"set {sorted(IMPLEMENTED_OVERLAP)} (rl-infra-spec 2.3: only eval||train and "
            "eval||outer_sync exist on the ports path)"
        )
    if profile.max_inflight_batches != 1:
        return "age-0 overlap keeps exactly one training batch in flight"
    return None


@dataclass
class _Pending:
    rollout_id: int
    token: str
    handle: EvalHandle
    started_at: float


class EvalOverlap:
    """Guard state for the eval||train/outer_sync schedule of one island."""

    def __init__(
        self,
        starter: EvalStarter,
        *,
        emit: Callable[..., None],
        clock: Callable[[], float],
    ) -> None:
        self.starter = starter
        self.emit = emit
        self.clock = clock
        self.due: tuple[int, str] | None = None
        self.pending: _Pending | None = None

    @property
    def in_flight(self) -> int:
        return int(self.pending is not None)

    def schedule(self, rollout_id: int, *, token: str | None, published_version: int | None) -> None:
        """An eval point of the just-published policy ``v_rollout_id``; started after generate."""
        if token is None or published_version != rollout_id:
            raise OverlapGuardError(
                f"eval of v{rollout_id} scheduled without its complete publication"
            )
        if self.due is not None or self.pending is not None:
            raise OverlapGuardError(
                f"eval queue bound {EVAL_QUEUE_BOUND} exceeded at v{rollout_id}"
            )
        self.due = (rollout_id, token)

    def before_generate(self, rollout_id: int) -> None:
        if self.pending is not None:
            raise OverlapGuardError(
                f"eval of v{self.pending.rollout_id} still holds the rollout role at "
                f"generation of rollout {rollout_id}"
            )

    def after_generate(self, rollout_id: int, *, token: str | None,
                       published_version: int | None) -> None:
        """Start the due eval now that generation released the rollout GPUs."""
        if self.due is None:
            return
        due_id, due_token = self.due
        if due_id != rollout_id or token != due_token or published_version != due_id:
            raise OverlapGuardError(
                f"eval scheduled for v{due_id} ({due_token}) but rollout GPUs hold "
                f"v{published_version} ({token})"
            )
        self.due = None
        started = self.clock()
        handle = self.starter(due_id)
        self.pending = _Pending(due_id, due_token, handle, started)
        self.emit("rl_eval_overlap_start", policy_version=due_id,
                  **{"rl/policy_token": due_token})

    def before_publish(self, *, token: str | None, published_version: int | None) -> dict | None:
        """Join the in-flight eval; it must still see the policy it was started on.

        Returns ``{"rollout_id", "token", "metrics", "start", "end"}`` or None.
        """
        if self.due is not None:
            # generation never ran after the eval point (it failed or the run
            # stopped); nothing overlapped, the caller runs the eval serially.
            return None
        pending, self.pending = self.pending, None
        if pending is None:
            return None
        metrics = dict(pending.handle.result())
        end = self.clock()
        if published_version != pending.rollout_id or token != pending.token:
            raise OverlapGuardError(
                f"publication moved to v{published_version} while eval of "
                f"v{pending.rollout_id} was in flight"
            )
        return {
            "rollout_id": pending.rollout_id,
            "token": pending.token,
            "metrics": metrics,
            "start": pending.started_at,
            "end": end,
        }


class LoopEvalHandle:
    """Adapter-side handle: an asyncio task on the island's single ``LoopRunner``.

    The task progresses whenever the driver drives the same loop (the trainer's
    ``train_step``/bridge calls go through ``runner.run``), i.e. cooperatively
    during train/outer_sync; ``result`` drives the loop until it finishes.
    """

    def __init__(self, runner: Any, coro: Any) -> None:
        self._runner = runner
        self._task = runner.loop.create_task(coro)

    def result(self) -> Mapping[str, float]:
        return self._runner.run(self._task) or {}


def loop_eval_starter(runner: Any, evaluate: Callable[[int], Any]) -> EvalStarter:
    return lambda rollout_id: LoopEvalHandle(runner, evaluate(rollout_id))
