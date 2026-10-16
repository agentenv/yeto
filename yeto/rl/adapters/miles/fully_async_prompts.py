"""Prompt accounting for Miles fully_async rollouts (P0-S).

Under ``--fully-async`` Miles draws prompt groups from its data source in a
background worker; groups in flight and finished groups waiting in the output
buffer are invisible to the round that trains. :class:`PromptLedger` wraps the
data source (``get_samples`` / ``add_samples``) inside the rollout executor
process and keeps every drawn group until it is trained or returned to the
data source. It serves two purposes:

* round accounting: groups drawn since the last report and Miles' stale-group
  count (reported unknown, with a warning, when Miles does not expose it);
* cuts (decision P0-S-Q3): :func:`requeue_untrained` stops the background
  worker and puts every drawn, untrained group back into the data source
  buffer, reset for a fresh generation. The cut then carries them like any
  buffered group, so the resumed run generates them again; nothing in flight
  is saved.

Runs inside the rollout executor actor (``__ray_call__``). Import-light.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any

logger = logging.getLogger(__name__)

_LEDGER_ATTR = "_yeto_prompt_ledger"
_LEDGER: "PromptLedger | None" = None  # the one ledger of this rollout process
WORKER_STOP_TIMEOUT_S = 30.0


class PromptLedgerError(RuntimeError):
    """The fully_async prompts cannot be accounted for (the cut is refused)."""


def _group_key(group: Sequence[Any]) -> int:
    first = group[0]
    while isinstance(first, list):
        first = first[0]
    key = getattr(first, "group_index", None)
    if key is None:
        raise PromptLedgerError("a drawn prompt group has no group_index")
    return int(key)


class PromptLedger:
    """Drawn prompt groups that are neither trained nor back in the data source."""

    def __init__(self, data_source: Any, executor: Any = None) -> None:
        self.data_source = data_source
        self.executor = executor
        self.untrained: dict[int, list[Any]] = {}
        self.drawn_since_report = 0
        self._get_samples: Callable[..., Any] = data_source.get_samples
        self._add_samples: Callable[..., Any] = data_source.add_samples
        data_source.get_samples = self._drawn
        data_source.add_samples = self._returned

    def _drawn(self, *args: Any, **kwargs: Any) -> Any:
        groups = self._get_samples(*args, **kwargs)
        for group in groups:
            self.untrained[_group_key(group)] = group
        self.drawn_since_report += len(groups)
        return groups

    def _returned(self, groups: Iterable[Any], *args: Any, **kwargs: Any) -> Any:
        groups = list(groups)
        for group in groups:
            self.untrained.pop(_group_key(group), None)
        return self._add_samples(groups, *args, **kwargs)

    def mark_trained(self, groups: Iterable[Sequence[Any]]) -> None:
        for group in groups:
            self.untrained.pop(_group_key(group), None)

    def take_round_counts(self) -> dict[str, Any]:
        """Groups drawn since the last report and stale groups Miles dropped
        this round (``None`` = unknown: Miles did not expose the count).

        Miles' ``_metric_stale_groups`` is a per-step counter: its
        ``get_metrics`` resets it right after this hook runs (in ``_drain``),
        so the value read here is already this round's count, not a running
        total."""
        counts: dict[str, Any] = {"submitted_groups": self.drawn_since_report}
        self.drawn_since_report = 0
        rollout_fn = getattr(self.executor, "generate_rollout", None)
        stale = getattr(getattr(rollout_fn, "_output", None), "_metric_stale_groups", None)
        if not isinstance(stale, int):
            logger.warning("fully_async stale-group count is not exposed by Miles; reported unknown")
            counts["stale_dropped_groups"] = None
            return counts
        counts["stale_dropped_groups"] = stale
        return counts


def install(executor: Any) -> None:
    """Wrap the executor's data source once (idempotent)."""
    global _LEDGER
    source = getattr(executor, "data_source", None)
    source = getattr(source, "__self__", source)
    if source is None:
        raise PromptLedgerError("the rollout executor has no data source")
    ledger = getattr(source, _LEDGER_ATTR, None)
    if ledger is None:
        ledger = PromptLedger(source, executor)
        setattr(source, _LEDGER_ATTR, ledger)
    _LEDGER = ledger


def current() -> PromptLedger | None:
    return _LEDGER


def requeue_untrained(executor: Any, *, timeout_s: float = WORKER_STOP_TIMEOUT_S) -> int:
    """Stop the background worker and return every drawn, untrained group to
    the data source, reset for a fresh generation. Returns the group count.

    The worker's group generation tasks are cancelled and awaited too: the new
    worker submits a full budget again, so a task left running would keep
    generating outside the in-flight budget (S24 T-TB2: about one budget leaked
    per cut). A task that does not settle in time still holds its own
    (original) samples; the data source gets reset copies, so nothing it
    writes reaches them. The rollout function starts a new worker on its next
    call."""
    source = getattr(executor, "data_source", None)
    ledger = getattr(getattr(source, "__self__", source), _LEDGER_ATTR, None)
    if ledger is None:
        raise PromptLedgerError("fully_async prompts were not tracked; the cut would lose them")
    rollout_fn = getattr(executor, "generate_rollout", None)
    _stop_worker(rollout_fn, timeout_s)
    groups = [copy.deepcopy(group) for _, group in sorted(ledger.untrained.items())]
    for group in groups:
        for sample in group:
            sample.reset_for_retry()
    if groups:
        ledger.data_source.add_samples(groups)  # the ledger forgets them
    if rollout_fn is not None:
        rollout_fn._worker = None
        rollout_fn._output = None
    return len(groups)


def _stop_worker(rollout_fn: Any, timeout_s: float) -> None:
    worker = getattr(rollout_fn, "_worker", None)
    if worker is None or worker.done():
        return
    loop = worker.get_loop()
    if getattr(loop, "_thread_id", None) == threading.get_ident():
        raise PromptLedgerError("cannot stop the fully_async worker from its own event loop")
    loop.call_soon_threadsafe(worker.cancel)
    deadline = time.monotonic() + timeout_s
    while not worker.done():
        if time.monotonic() > deadline:
            raise PromptLedgerError("the fully_async worker did not stop in time")
        time.sleep(0.05)
    # Cancelling the worker does not cancel the group tasks it submitted
    # (asyncio.wait never cancels what it waits on).
    stopped = asyncio.run_coroutine_threadsafe(_cancel_generation(rollout_fn, timeout_s), loop)
    left = stopped.result(timeout=timeout_s + 5.0)
    if left:
        logger.warning("fully_async: %d group generation task(s) did not stop in %.0f s", left, timeout_s)


def _is_generation_of(task: "asyncio.Task[Any]", rollout_fn: Any) -> bool:
    coro = task.get_coro()
    frame = getattr(coro, "cr_frame", None)
    return (getattr(coro, "__qualname__", "").endswith("._generate_group")
            and frame is not None and frame.f_locals.get("self") is rollout_fn)


async def _cancel_generation(rollout_fn: Any, timeout_s: float) -> int:
    """On the rollout loop: cancel this rollout function's group generation tasks
    (each cancels its sample tasks; an agent's ``finally`` ends its trajectory)
    and wait for them; returns how many are still running after ``timeout_s``."""
    tasks = [t for t in asyncio.all_tasks() if not t.done() and _is_generation_of(t, rollout_fn)]
    for task in tasks:
        task.cancel()
    if not tasks:
        return 0
    _done, pending = await asyncio.wait(tasks, timeout=timeout_s)
    logger.info("fully_async: cancelled %d group generation task(s) at the cut", len(tasks))
    return len(pending)
