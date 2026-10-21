"""Round cuts on the verl fully_async path (S26 E9.2; S25 C4 refused them).

The cut itself is the sync path's (:mod:`.cut`: one yeto file per FSDP rank of
the trainer actor's worker group, reached through ``yeto_call_workers``). What
differs is the data cursor: verl's ``FullyAsyncRollouter`` draws prompts on its
own, ahead of the trainer (the MessageQueue and the requests in flight hold
prompts no round has consumed yet). At a cut those prompts are not saved:

* the cursor is ``{"sample_offset": prompts the trainer took from the queue}``
  (kept or discarded by the policy-age gate), counted by
  :class:`FullyAsyncDataCursor`, never the rollouter's own draw count;
* on resume, before the rollouter's generation loop starts, the rollouter skips
  that many prompts of its dataloader (:func:`skip_rollouter_prompts`, run in
  the rollouter actor through ``__ray_call__``): the skip draws through the same
  continuous iterator, so the epoch/shuffle order of an uninterrupted run is
  kept; prompts that were queued or in flight at the cut are generated again by
  the restored policy (each prompt is trained at most once; one that was queued
  at the cut is generated twice in total).

Pure Python; the rollouter is duck-typed (``_create_continuous_iterator``,
``global_steps``), so the unit tests use a fake. Not yet run on a GPU.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any

CURSOR_KEY = "sample_offset"


def skip_rollouter_prompts(rollouter: Any, prompts: int) -> dict[str, int]:
    """Runs INSIDE the rollouter actor before its ``fit``: its continuous iterator
    starts ``prompts`` prompts in (drawn and discarded), and its step counter
    (``global_steps``, one per fed prompt; the feed stops at
    ``total_rollout_steps``) counts them as fed."""
    prompts = int(prompts)
    if prompts < 0:
        raise ValueError(f"cannot skip {prompts} prompts")
    if getattr(rollouter, "_yeto_skipped", None) is not None:
        raise RuntimeError("the rollouter's prompts were already skipped once")
    original: Callable[[], Iterator[Any]] = rollouter._create_continuous_iterator

    def skipped() -> Iterator[Any]:
        iterator = original()
        for drawn in range(prompts):
            if next(iterator, None) is None:
                raise RuntimeError(f"the rollouter's data ended after {drawn} of {prompts} skipped prompts")
        yield from iterator

    rollouter._create_continuous_iterator = skipped
    rollouter._yeto_skipped = prompts
    rollouter.global_steps = int(rollouter.global_steps) + prompts
    return {CURSOR_KEY: prompts}


class FullyAsyncDataCursor:
    """The fully_async pool's data cursor (module doc): prompts consumed by rounds."""

    def __init__(self, seek_rollouter: Callable[[int], Mapping[str, int]] | None) -> None:
        self.seek_rollouter = seek_rollouter
        self.consumed = 0
        self.started = False

    def cursor(self) -> dict[str, int]:
        return {CURSOR_KEY: int(self.consumed)}

    def consume(self, prompts: int) -> None:
        self.consumed += int(prompts)

    def seek(self, cursor: Mapping[str, int]) -> dict[str, int]:
        if set(cursor) != {CURSOR_KEY}:
            raise ValueError(f"verl fully_async data cursor is {{{CURSOR_KEY!r}}} only, got {dict(cursor)}")
        target = int(cursor[CURSOR_KEY])
        if self.started:
            raise ValueError("the fully_async data cursor moves only before the rollouter starts")
        if target < self.consumed:
            raise ValueError(f"verl fully_async data cursor cannot move back from {self.consumed} to {target}")
        if target == self.consumed:
            return self.cursor()
        if self.seek_rollouter is None:
            raise ValueError("this fully_async island cannot move its rollouter's data (no seek hook)")
        landed = int(self.seek_rollouter(target - self.consumed)[CURSOR_KEY])
        self.consumed += landed
        return self.cursor()


def optimizer_steps_per_round(config: Mapping[str, Any], *, groups_per_round: int) -> int:
    """Optimizer steps of one fully_async ``update_actor`` (the resume first-step
    check): ``ppo_epochs`` x (prompts per round / ``ppo_mini_batch_size``). verl's
    ``data.train_batch_size`` is 0 on this path, so the round size is yeto's
    ``groups_per_round``. Not yet checked against a live run (the first resumed
    round refuses a wrong value)."""
    actor = ((config.get("actor_rollout_ref") or {}).get("actor") or {})
    epochs = int(actor.get("ppo_epochs", 1) or 1)
    mini = int(actor.get("ppo_mini_batch_size") or 0)
    if mini <= 0 or int(groups_per_round) % mini:
        raise ValueError(f"groups_per_round {groups_per_round} is not a whole number of "
                         f"ppo_mini_batch_size {mini}")
    return epochs * (int(groups_per_round) // mini)
