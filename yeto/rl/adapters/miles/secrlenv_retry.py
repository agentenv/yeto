"""SecRLEnv same-task retry on the ports engine (P0-1.7b.4-Q1).

The SecRLEnv group check (``yeto.rl.harness.codex.reward.check_group``)
verifies each group's signed episode outcomes. A non-trainable group (an
infrastructure failure, or a sample that did not complete) is rejected and the
check asks for one exact rerun of the same task. This module holds the
Miles-independent half of that contract; :mod:`.secrlenv_miles` binds it to
Miles' data source and rollout function classes.

Rollout contract (unchanged from the removed legacy rollout):

* the round draws exactly ``rollout_batch_size`` new task groups; the
  ``over_sampling_batch_size = rollout_batch_size + 1`` slot is reserved for
  the retry, it is not over-sampling;
* at most ``SECRLENV_INFRASTRUCTURE_REPLACEMENTS`` (1) retry per round; a
  refill without a queued retry, a second retry, or a retry whose task or
  503-retry ledger does not match the drawn task ends the run with a
  sanitized :class:`SecRLEnvRolloutTerminalError`;
* on a terminal error the in-flight work is aborted and the run checks that no
  episode is left behind before it raises.

Miles marks a rollout boundary for the data source: its rollout function
draws the round's groups with ``get_samples`` and returns the aborted groups
with ``add_samples`` once the round ends.
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from yeto.rl import SECRLENV_AGENTS, SECRLENV_INFRASTRUCTURE_REPLACEMENTS
from yeto.rl.engine.algorithm import SECRLENV_GROUP_CHECK_FILTER
from yeto.rl.harness.codex.reward import (
    INFRASTRUCTURE_503_RETRIES_KEY,
    OUTCOME_KEY,
    SecRLEnvReplacementExhausted,
    SecRLEnvRolloutCleanupError,
    SecRLEnvRolloutTerminalError,
    SecRLEnvUntrustedEvidence,
    UntrustedOutcome,
)

DATA_SOURCE_PATH = "yeto.rl.adapters.miles.secrlenv_miles.SecRLEnvRetryDataSource"
ROLLOUT_FUNCTION_PATH = "yeto.rl.adapters.miles.secrlenv_miles.SecRLEnvRolloutFn"
# Read by the group check (reward.check_group) to queue the retry.
RETRY_CALLBACK_ATTR = "_yeto_secrlenv_retry_callback"
RETRY_STATS_ATTR = "_yeto_secrlenv_retry_stats"
# The group check's per-round decision memo; reset at each round start.
FILTER_STATE_ATTR = "_yeto_secrlenv_filter_state"
METRIC_PREFIX = "rl/secrlenv/"
TERMINAL_CLEANUP_TIMEOUT_S = 240.0
_STAT_NAMES = (
    "replacement_attempts",
    "nontrainable_groups",
    "infrastructure_groups",
    "aborted_groups",
    "pending_groups",
    "failed_groups",
)


def rollout_argv(*, custom_agent: str | None, dynamic_filter: str | None,
                 rollout_batch_size: int, over_sampling_batch_size: int,
                 max_policy_age: int) -> tuple[str, ...]:
    """Miles flags that bind the SecRLEnv retry; empty for any other agent.

    Raises ``ValueError`` when the agent and the group check are not used
    together, when the refill slot is not exactly one group, or when the run
    carries rollouts across rounds (the retry slot is counted per round).
    """

    secrlenv = custom_agent in SECRLENV_AGENTS
    if secrlenv != (dynamic_filter == SECRLENV_GROUP_CHECK_FILTER):
        raise ValueError(
            f"the SecRLEnv agents and the {SECRLENV_GROUP_CHECK_FILTER!r} filter go "
            f"together (agent {custom_agent!r}, filter {dynamic_filter!r})"
        )
    if not secrlenv:
        return ()
    if over_sampling_batch_size != rollout_batch_size + 1:
        raise ValueError(
            "the SecRLEnv agents reserve exactly one refill slot for the same-task "
            f"retry: over-sampling must be {rollout_batch_size + 1} "
            f"(got {over_sampling_batch_size})"
        )
    if max_policy_age:
        raise ValueError(
            "the SecRLEnv same-task retry is counted per round; it does not support "
            f"a policy-age limit (got {max_policy_age})"
        )
    return (
        "--data-source-path", DATA_SOURCE_PATH,
        "--rollout-function-path", ROLLOUT_FUNCTION_PATH,
    )


def _task_indices(group: object) -> tuple[int, ...] | None:
    """The sample indexes of a group (nested sample lists use their first item)."""

    if not isinstance(group, list) or not group:
        return None
    values = []
    for item in group:
        while isinstance(item, list):
            if not item:
                return None
            item = item[0]
        try:
            values.append(int(item.index))
        except (AttributeError, TypeError, ValueError):
            return None
    return tuple(values)


def _sample_statuses(group: object) -> list[str | None]:
    values: list[str | None] = []
    pending = list(group) if isinstance(group, list) else []
    while pending:
        item = pending.pop(0)
        if isinstance(item, list):
            pending[0:0] = item
            continue
        status = getattr(item, "status", None)
        value = getattr(status, "value", status)
        values.append(value if isinstance(value, str) else None)
    return values


def _strict_int(value: Any) -> bool:
    return type(value) is int


class SecRLEnvTaskRetry:
    """Data-source mixin: reserve the spare refill slot for one same-task retry.

    Put it before a Miles data source class (``RolloutDataSourceWithBuffer``):
    it overrides ``get_samples``/``add_samples`` and defers the rest.
    """

    def __init__(self, args: Any) -> None:
        super().__init__(args)
        target = getattr(args, "rollout_batch_size", None)
        capacity = getattr(args, "over_sampling_batch_size", None)
        limit = getattr(args, "yeto_rl_secrlenv_max_infrastructure_replacements", None)
        if (
            not _strict_int(target)
            or target <= 0
            or capacity != target + 1
            or not _strict_int(limit)
            or limit != SECRLENV_INFRASTRUCTURE_REPLACEMENTS
        ):
            raise SecRLEnvReplacementExhausted(
                "SecRLEnv rollout replacement contract is invalid"
            )
        self._retry_args = args
        self._retry_target = target
        self._retry_capacity = capacity
        self._retry_limit = limit
        self._round_open = False
        self._reset_round()
        setattr(args, RETRY_CALLBACK_ATTR, self.request_retry)

    def _reset_round(self) -> None:
        self._templates: dict[tuple[int, ...], list[Any]] = {}
        self._queued: list[list[Any]] = []
        self._seen_evidence: set[object] = set()
        self._retries_used = 0
        self._retry_stats = dict.fromkeys(_STAT_NAMES, 0)
        setattr(self._retry_args, RETRY_STATS_ATTR, self._retry_stats)
        setattr(self._retry_args, FILTER_STATE_ATTR, None)

    def get_samples(self, num_samples: int) -> list[list[Any]]:
        if not _strict_int(num_samples) or num_samples != self._retry_capacity:
            raise SecRLEnvReplacementExhausted("SecRLEnv rollout refill contract is invalid")
        if not self._round_open:
            return self._open_round()
        if not self._queued:
            raise SecRLEnvReplacementExhausted(
                "SecRLEnv rollout refill had no authenticated task retry"
            )
        groups, self._queued = self._queued, []
        return groups

    def _open_round(self) -> list[list[Any]]:
        self._reset_round()
        groups = super().get_samples(self._retry_target)
        if not isinstance(groups, list) or len(groups) != self._retry_target:
            raise SecRLEnvReplacementExhausted(
                "SecRLEnv rollout did not receive its exact task batch"
            )
        for group in groups:
            key = _task_indices(group)
            if key is None:
                raise SecRLEnvReplacementExhausted("SecRLEnv rollout task identity is unavailable")
            self._templates[key] = copy.deepcopy(group)
        self._round_open = True
        return groups

    def add_samples(self, samples: list[list[Any]]) -> None:
        """Miles returns the round's aborted groups here once the round ends."""

        super().add_samples(samples)
        self._round_open = False

    def retry_stats(self) -> dict[str, int]:
        return dict(self._retry_stats)

    def request_retry(
        self,
        group: list[Any],
        *,
        evidence_key: object,
        reason: str,
        infrastructure_503_retries: Mapping[int, int],
    ) -> None:
        """Queue one exact rerun of ``group``'s task (called by the group check)."""

        if evidence_key in self._seen_evidence:
            return
        self._seen_evidence.add(evidence_key)
        if self._retries_used >= self._retry_limit or self._queued:
            raise SecRLEnvReplacementExhausted("SecRLEnv rollout replacement budget was exhausted")
        key = _task_indices(group)
        template = self._templates.get(key) if key is not None else None
        if template is None:
            raise SecRLEnvReplacementExhausted(
                "SecRLEnv rollout replacement task identity did not match"
            )
        if (
            not isinstance(infrastructure_503_retries, Mapping)
            or set(infrastructure_503_retries) != set(key)
            or any(
                not _strict_int(index) or not _strict_int(value) or value not in {0, 1}
                for index, value in infrastructure_503_retries.items()
            )
        ):
            raise SecRLEnvUntrustedEvidence("SecRLEnv rollout retry ledger authentication failed")
        retry = copy.deepcopy(template)
        for sample in retry:
            reset = getattr(sample, "reset_for_retry", None)
            if not callable(reset):
                raise SecRLEnvReplacementExhausted("SecRLEnv rollout sample cannot be reset safely")
            reset()
            metadata = getattr(sample, "metadata", None)
            if isinstance(metadata, dict):
                metadata.pop(OUTCOME_KEY, None)
                metadata[INFRASTRUCTURE_503_RETRIES_KEY] = (
                    infrastructure_503_retries[int(sample.index)]
                )
        statuses = _sample_statuses(group)
        stats = self._retry_stats
        stats["nontrainable_groups"] += 1
        stats["aborted_groups"] += int("aborted" in statuses)
        stats["pending_groups"] += int("pending" in statuses)
        stats["failed_groups"] += int("failed" in statuses)
        stats["infrastructure_groups"] += int(reason == "secrlenv_infrastructure_failure")
        self._retries_used += 1
        stats["replacement_attempts"] = self._retries_used
        self._queued.append(retry)


def retry_metrics(stats: Mapping[str, int] | None) -> dict[str, float]:
    """The round's retry counters as rollout metrics (``rl/secrlenv/<name>``)."""

    if not isinstance(stats, Mapping):
        return {}
    return {f"{METRIC_PREFIX}{name}": float(stats.get(name, 0)) for name in _STAT_NAMES}


TERMINAL_ERRORS = (SecRLEnvRolloutTerminalError, UntrustedOutcome)


async def raise_after_terminal_cleanup(
    error: BaseException,
    *,
    abort: Callable[[], Awaitable[Any]],
    require_clean: Callable[[], None],
    timeout_s: float = TERMINAL_CLEANUP_TIMEOUT_S,
) -> None:
    """Abort the in-flight work after a terminal error, then raise it sanitized.

    ``abort`` stops the engines and the agent's episodes; ``require_clean``
    raises if an episode is left behind. A cleanup failure or timeout raises
    :class:`SecRLEnvRolloutCleanupError` instead. Error text never carries
    episode payloads (``from None``).
    """

    if isinstance(error, SecRLEnvRolloutTerminalError):
        terminal: SecRLEnvRolloutTerminalError = error
    else:
        terminal = SecRLEnvUntrustedEvidence("SecRLEnv rollout evidence authentication failed")
    try:
        await asyncio.wait_for(abort(), timeout=timeout_s)
        require_clean()
    except Exception:
        raise SecRLEnvRolloutCleanupError("SecRLEnv terminal rollout cleanup failed") from None
    raise terminal from None
