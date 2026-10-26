"""verl trainer island over the durable sample buffer: the feed (S25 C2).

A verl *buffer trainer* trains on groups that a separate sampler island wrote
to the sample buffer (``yeto.rl.engine.sample_buffer``; the engine sampler
island ``yeto.rl.engine.engine_sampler`` or the Miles sampler island). It reuses
the fully_async consumption unchanged: the yeto ``IslandDriver`` and the ports
of :mod:`.fully_async_ports` (``RoundCollector`` with the ``max_policy_age``
judgement, carry-over and discard reports, the trainer step with the off-policy
correction). Only the source of the queued samples differs: verl's
``MessageQueue`` (filled by its rollouter) is replaced by :class:`BufferFeed`.

:class:`BufferFeed` answers the three queue calls of the rollout port
(``yeto_pull_sample`` / ``yeto_keep_last`` / ``yeto_drop_last``) in the driver
process, so everything that decides which group is trained runs here, on plain
data, without verl or Ray:

* ``pull``: the next buffer entry (one GRPO group), waiting up to ``wait_s`` for
  the sampler; its :class:`~.fully_async_round.SampleMeta` with every response's
  behavior version translated to verl's ``param_version`` through the island's
  :class:`~.fully_async_translate.VersionMap` (a version the trainer never
  published here maps to None: the collector discards the group as unknown);
* ``keep``: the group's padded rows (:func:`group_rows`) go to the trainer actor
  (``add_group``), which wraps them as a verl ``RolloutSample``;
* ``drop``: nothing reaches the actor; the group stays consumed.

Every pulled group is recorded once (``record``) with its buffer position, so a
run can show that no group was consumed twice. The buffer positions are those of
the trainer's read cursor (``FileSampleBuffer.select``).

Synced trainers (S25 C3, ``yeto.rl.engine.merged_release``): the buffer is a
lane reader (positions ``p % N == m`` only) and the feed keeps the lane cursor
(``cursor_log``): it continues after the last recorded round.

**What a relaunch reads again** (S27 G-EQ4 changed this). ``close_round`` holds the
round's positions as a *pending* record and ``commit_round`` writes it only after
the train call returns, so:

* never again the groups of a round that trained;
* **again** the groups of the round that was collected but not trained when the
  process stopped -- which is what :mod:`yeto.rl.engine.data_position` promises for
  a full-parameter island ("again, the rows of the round that was running when the
  process stopped").

Until G-EQ4 the record was written when collection ended, *before* the round
trained. That guaranteed "trained at most once" but **lost** a round's groups
whenever the island stopped between collection and the optimizer step: measured on
npu1, a kill during round 4 left the record ``cursor 32, positions [28, 29, 30, 31]``
for a round that never trained, and the relaunch started at 32 (S27 G-EQ4,
VERL-ASCEND-RL.md section 36).

The remaining window is the opposite one and is **not** closed here: if the process
stops between the train call returning and the record reaching the store, the
relaunch reads those groups again and trains them twice. Closing it needs the
cursor record and the round's release to land in **one** atomic commit, which this
store API does not offer. So the guarantee this module gives is "a group is trained
**at least** once, and at most once unless the island dies in that window" -- the
dispenser's "generated at least once, trained at most once" holds per *generation*,
not across a crash in that window. A reader that must never train a group twice has
to make the release and the cursor record one commit first.
The groups carry the *merged* version as behavior version;
:class:`MergedVersions` turns it into the island's local policy version (the
first local version that applied that merge, whose weights are the merged
policy), and the version map turns that into verl's ``param_version``. So the
``max_policy_age`` judgement counts local versions: equal to merged versions
when every round merges, stricter when an island trains several rounds per
merge (elastic sync).

:func:`group_rows` builds the rows the way verl's agent loop postprocess pads
them (``AgentLoopWorker._postprocess``): prompts left-padded to the prompt
length, responses right-padded to the response length, the reward on the last
response token (``rm_scores``), the behavior log-probs as ``rollout_log_probs``
and the sample's source record as ``provenance`` (the per-source off-policy
metrics read it).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from yeto.rl.engine.buffer_payload import (
    SampleBufferRolloutError,
    complete_prompt_tokens,
    decode_group,
    group_entry_fields,
)

from .fully_async_translate import VerlTranslationError, VersionMap

TRAINER_CONSUMER = "trainer"  # the buffer's single consumer name (as the Miles reader)
QUEUE_METHODS = ("yeto_pull_sample", "yeto_keep_last", "yeto_drop_last")
UID_PREFIX = "buf"


class BufferFeedError(SampleBufferRolloutError):
    """A buffer group cannot become verl rows."""


class MergedVersions:
    """Synced trainers: merged (syncer outer) version -> the first local policy
    version of this island that applied it (module doc). Merged versions only
    increase; a local version pairs with one merged version."""

    def __init__(self) -> None:
        self._local: dict[int, int] = {}

    def record(self, merged: int, local: int) -> bool:
        """Record one publication; True when ``merged`` is new on this island."""
        merged, local = int(merged), int(local)
        if merged in self._local:
            return False
        if self._local and (merged < max(self._local) or local <= self._local[max(self._local)]):
            raise BufferFeedError(f"merged v{merged} at local v{local} after merged v{max(self._local)} "
                                  f"at local v{self._local[max(self._local)]}: versions must increase")
        self._local[merged] = local
        return True

    def local(self, merged: int) -> int | None:
        return self._local.get(int(merged))

    def to_dict(self) -> dict[str, int]:
        return {str(k): v for k, v in sorted(self._local.items())}


@dataclass(frozen=True)
class RowShape:
    """The padded widths of the trainer's rows (``data.max_prompt_length`` and
    ``data.max_response_length`` of the verl config) and the pad token id."""

    prompt_length: int
    response_length: int
    pad_token_id: int

    def __post_init__(self) -> None:
        if self.prompt_length < 1 or self.response_length < 1:
            raise ValueError(f"row widths must be positive, got {self.prompt_length}/{self.response_length}")


def sample_version(sample: Mapping[str, Any]) -> int:
    """The oldest engine weight version of one sample (its behavior version)."""
    return group_entry_fields([sample])[0]


def group_rows(samples: Sequence[Mapping[str, Any]], *, uid: str, param_versions: Sequence[int],
               shape: RowShape, tokenize: Callable[[str], Sequence[int]] | None = None) -> dict[str, Any]:
    """The verl rows of one group as plain lists (module doc).

    ``param_versions``: verl's ``param_version`` of each sample's behavior
    version (``min_global_steps`` = ``max_global_steps``: a buffer sample comes
    from one engine version). ``tokenize`` rebuilds the prompt ids of a sample
    whose engine did not return them (``complete_prompt_tokens``). Returns
    ``{"tensors": {name: rows}, "non_tensor": {name: values}, "meta_info": {...}}``.
    Raises :class:`BufferFeedError` when a sample does not fit the widths."""
    if len(param_versions) != len(samples):
        raise BufferFeedError(f"{len(param_versions)} param versions for {len(samples)} samples")
    tensors: dict[str, list[list[Any]]] = {k: [] for k in (
        "prompts", "responses", "response_mask", "input_ids", "attention_mask", "position_ids",
        "rollout_log_probs", "rm_scores")}
    provenance, finish = [], []
    pad = int(shape.pad_token_id)
    for sample in samples:
        sample = complete_prompt_tokens(sample, tokenize)
        tokens = [int(t) for t in sample.get("tokens") or ()]
        length = int(sample.get("response_length") or 0)
        logprobs = [float(x) for x in sample.get("rollout_log_probs") or ()]
        where = f"group {uid} sample {sample.get('index')}"
        if length < 1 or len(logprobs) != length or len(tokens) <= length:
            raise BufferFeedError(f"{where}: {len(tokens)} tokens, response_length {length}, "
                                  f"{len(logprobs)} log-probs")
        prompt, response = tokens[:-length], tokens[-length:]
        if len(prompt) > shape.prompt_length or length > shape.response_length:
            raise BufferFeedError(f"{where}: prompt {len(prompt)} / response {length} tokens exceed "
                                  f"the rows ({shape.prompt_length} / {shape.response_length})")
        left = shape.prompt_length - len(prompt)
        right = shape.response_length - length
        attention = [0] * left + [1] * (len(prompt) + length) + [0] * right
        tensors["prompts"].append([pad] * left + prompt)
        tensors["responses"].append(response + [pad] * right)
        tensors["response_mask"].append([1] * length + [0] * right)
        tensors["input_ids"].append([pad] * left + prompt + response + [pad] * right)
        tensors["attention_mask"].append(attention)
        tensors["position_ids"].append(_position_ids(attention))
        tensors["rollout_log_probs"].append(logprobs + [0.0] * right)
        tensors["rm_scores"].append([0.0] * (length - 1) + [float(sample.get("reward") or 0.0)]
                                    + [0.0] * right)
        metadata = sample.get("metadata") or {}
        provenance.append(dict(metadata.get("provenance") or {}) or None)
        finish.append(metadata.get("finish_reason"))
    n = len(samples)
    versions = [int(v) for v in param_versions]
    return {
        "tensors": tensors,
        "non_tensor": {"uid": [uid] * n, "min_global_steps": versions, "max_global_steps": list(versions),
                       "provenance": provenance, "finish_reason": finish, "__num_turns__": [1] * n},
        # verl's assemble step pops one ``metrics`` record per row (addition_process)
        "meta_info": {"metrics": [{"generate_sequences": 0.0, "tool_calls": 0.0}] * n,
                      "reward_extra_keys": []},
    }


def _position_ids(attention: Sequence[int]) -> list[int]:
    """verl's ``compute_position_id_with_mask``: cumulative sum minus one, floored at 0."""
    out, seen = [], 0
    for bit in attention:
        seen += int(bit)
        out.append(max(seen - 1, 0))
    return out


@dataclass
class _Pulled:
    """The group the collector is judging: its rows' inputs and its record."""

    position: int
    samples: list[dict[str, Any]]
    param_versions: tuple[int | None, ...]
    record: dict[str, Any]


class BufferFeed:
    """The rollout port's queue calls over the sample buffer (module doc).

    ``buffer``: the trainer's ``FileSampleBuffer``; ``vmap``: returns the
    island's live version map; ``add_group(rows)``: hands kept rows to the
    trainer actor; ``record(fields)``: one record per pulled group."""

    def __init__(self, buffer: Any, *, vmap: Callable[[], VersionMap], shape: RowShape,
                 add_group: Callable[[dict[str, Any]], Any], record: Callable[[dict[str, Any]], None],
                 tokenize: Callable[[str], Sequence[int]] | None = None, wait_s: float = 3600.0,
                 poll_s: float = 2.0, consumer: str = TRAINER_CONSUMER,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 cursor_log: Any = None, merged: MergedVersions | None = None) -> None:
        """``cursor_log``: a synced trainer's lane cursor log
        (``merged_release.lane_cursor_log``); ``merged``: a synced trainer's
        merged version pairing (module doc)."""
        self.buffer = buffer
        self.vmap = vmap
        self.shape = shape
        self.add_group = add_group
        self.record = record
        self.tokenize = tokenize
        self.wait_s = float(wait_s)
        self.poll_s = float(poll_s)
        self.consumer = consumer
        self.clock = clock
        self.sleep = sleep
        self._last: _Pulled | None = None
        self.pulled_positions: list[int] = []
        self.cursor_log = cursor_log
        self.merged = merged
        self._round_kept: list[int] = []  # trained positions of the round being collected
        # the closed round's cursor record, written only once that round has trained
        # (:meth:`commit_round`, S27 G-EQ4)
        self._pending: tuple[int, list[int], int] | None = None
        if cursor_log is not None:
            self.buffer.reset_to(cursor_log.last_cursor())

    def __call__(self, name: str, *args: Any) -> Any:
        """``call(name)`` for the three queue methods (:data:`QUEUE_METHODS`)."""
        if name == "yeto_pull_sample":
            return self.pull()
        if name == "yeto_keep_last":
            return self.keep()
        if name == "yeto_drop_last":
            return self.drop()
        raise ValueError(f"{name} is not a queue call of the buffer feed")

    def _next(self) -> Any | None:
        """The next selected entry; None after the sampler's data end."""
        started = self.clock()
        while True:
            selected = self.buffer.select(self.consumer, 1)
            if selected:
                return selected[0]
            if self.buffer.data_ended:
                return None
            if self.clock() - started > self.wait_s:
                raise BufferFeedError(f"no buffer group within {self.wait_s:g} s "
                                      f"(read position {self.buffer.read_cursor})")
            self.sleep(self.poll_s)

    def _local(self, version: int) -> int | None:
        if self.merged is not None:
            local = self.merged.local(version)
            if local is None:  # a merge this island never applied: unknown
                return None
            version = local
        try:
            return self.vmap().local(version)
        except VerlTranslationError:
            return None

    def pull(self) -> dict[str, Any] | None:
        if self._last is not None:
            raise BufferFeedError("pull before the previous group was kept or dropped")
        item = self._next()
        if item is None:
            return None
        samples = decode_group(self.buffer.read_payload(item.entry.payload_ref))
        if not samples:
            raise BufferFeedError(f"buffer position {item.position}: empty group")
        versions = [sample_version(s) for s in samples]
        local = tuple(self._local(v) for v in versions)
        lengths = tuple(int(s.get("response_length") or 0) for s in samples)
        position = int(item.position)
        self._last = _Pulled(position, samples, local, {
            "position": position, "sample_id": item.entry.sample_id,
            "behavior_versions": versions, "param_versions": list(local),
            "source": (item.entry.provenance or {}).get("sampler_id"),
            "samples": len(samples), "response_tokens": sum(lengths)})
        self.pulled_positions.append(position)
        return {"uid": f"{UID_PREFIX}{int(item.position)}", "min_steps": local, "max_steps": local,
                "calls": (None,) * len(samples), "lengths": lengths,
                "rewards": tuple(float(s.get("reward") or 0.0) for s in samples)}

    def keep(self) -> None:
        last = self._take_last()
        if None in last.param_versions:  # the collector keeps only groups whose versions it knows
            raise BufferFeedError(f"buffer position {last.position}: kept with an unknown version "
                                  f"{last.param_versions}")
        self.add_group(group_rows(last.samples, uid=f"{UID_PREFIX}{last.position}",
                                  param_versions=last.param_versions, shape=self.shape,
                                  tokenize=self.tokenize))
        self._round_kept.append(last.position)
        self.record({**last.record, "kept": True})

    def drop(self) -> None:
        self.record({**self._take_last().record, "kept": False})

    def _take_last(self) -> _Pulled:
        if self._last is None:
            raise BufferFeedError("keep/drop without a pulled group")
        last, self._last = self._last, None
        return last

    def close_round(self, rollout_id: int) -> None:
        """The round's collection ended. Hold the round's positions and read position
        as a *pending* cursor record; :meth:`commit_round` writes it once the round
        has trained (module doc, "What a relaunch reads again")."""
        if self._last is not None:
            raise BufferFeedError("round closed with a pulled group not kept or dropped")
        kept, self._round_kept = self._round_kept, []
        self._pending = (int(rollout_id), kept, int(self.buffer.read_cursor))

    def commit_round(self) -> tuple[int, list[int], int] | None:
        """Write the pending cursor record: the round trained, so a relaunch must not
        read its groups again. Returns the record written, or None when there is none
        (a round that never closed, or one already committed). S27 G-EQ4."""
        pending, self._pending = self._pending, None
        if pending is None or self.cursor_log is None:
            return pending
        rollout_id, kept, cursor = pending
        self.cursor_log.append(rollout_id, kept, cursor)
        return pending

    def read_cursor(self) -> int:
        """The buffer position after the last pulled group (released to the
        samplers for their backlog cap)."""
        return int(self.buffer.read_cursor)

    def read_shards(self) -> dict[str, int] | None:
        """The read cursor per sampler shard (None without merged shards): a
        sampler that writes its own shard counts its backlog from this
        (``SamplerLoop.backlog``), not from the merged position."""
        return self.buffer.shard_positions(self.buffer.read_cursor) or None


ASSEMBLE_METHOD = "yeto_assemble"  # the rollout port's call after a round's collection
TRAIN_METHOD = "yeto_train_step"  # the trainer's step; it returns when the round has trained


class BufferCall:
    """The island's ``call(method, *args)``: the queue calls go to the feed, every
    other call to the trainer actor (``actor_call``). ``round_id()``: the round
    being collected; its collection ends at the assemble call, where the feed
    closes the round (:meth:`BufferFeed.close_round`), and the feed writes that
    round's cursor record after the train call returns
    (:meth:`BufferFeed.commit_round`, S27 G-EQ4)."""

    def __init__(self, feed: BufferFeed, actor_call: Callable[..., Any],
                 round_id: Callable[[], int] | None = None) -> None:
        self.feed = feed
        self.actor_call = actor_call
        self.round_id = round_id

    def __call__(self, name: str, *args: Any, **kwargs: Any) -> Any:
        if name in QUEUE_METHODS:
            return self.feed(name, *args)
        if name == ASSEMBLE_METHOD and self.round_id is not None:
            self.feed.close_round(self.round_id())
        out = self.actor_call(name, *args, **kwargs)
        if name == TRAIN_METHOD:
            # the round trained: only now may a relaunch skip its groups. A train call
            # that raises leaves the record pending, so the groups are read again.
            self.feed.commit_round()
        return out
