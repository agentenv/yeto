"""verl trainer island over the durable sample buffer: the feed.

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
  behavior version put through the island's consumption map
  (``FullyAsyncIsland.consume_map``, a :class:`~.fully_async_translate.VersionMap`
  pairing the outer version with itself on this path: a buffered group's behavior
  version is a store release version, which is an outer version, not a verl
  ``param_version``). A version that map does not hold becomes None and the
  collector discards the group as unknown; at a relaunch the map is seeded from
  the store's release records, so the versions an earlier process published are
  known (``buffer_ports`` module doc, "relaunch");
* ``keep``: the group's padded rows (:func:`group_rows`) go to the trainer actor
  (``add_group``), which wraps them as a verl ``RolloutSample``;
* ``drop``: nothing reaches the actor; the group stays consumed.

Every pulled group is recorded once (``record``) with its buffer position, so a
run can show that no group was consumed twice. The buffer positions are those of
the trainer's read cursor (``FileSampleBuffer.select``).

Synced trainers (``yeto.rl.engine.merged_release``): the buffer is a
lane reader (positions ``p % N == m`` only) and the feed keeps the lane cursor
(``cursor_log``): it continues after the last recorded round.

**What a relaunch reads again** (changed by the commit-after-step fix). ``close_round`` holds the
round's positions as a *pending* record and ``commit_round`` writes it only after
the train call returns, so:

* never again the groups of a round that trained;
* **again** the groups of the round that was collected but not trained when the
  process stopped -- which is what :mod:`yeto.rl.engine.data_position` promises for
  a full-parameter island ("again, the rows of the round that was running when the
  process stopped").

Before that fix the record was written when collection ended, *before* the round
trained. That guaranteed "trained at most once" but **lost** a round's groups
whenever the island stopped between collection and the optimizer step: measured on
npu1, a kill during round 4 left the record ``cursor 32, positions [28, 29, 30, 31]``
for a round that never trained, and the relaunch started at 32
(VERL-ASCEND-RL.md section 36).

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

import os
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from yeto.rl.engine.buffer_payload import (
    SampleBufferRolloutError,
    complete_prompt_tokens,
    decode_group,
    group_entry_fields,
)

from yeto.rl.engine.incarnation_lease import STALE_REASON

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
        # A multi-turn (agentic) sample carries its response mask (tool output
        # and template tokens are 0); a single-turn sample trains every response token
        mask = sample.get("loss_mask")
        mask = [1] * length if mask is None else [int(m) for m in mask]
        if len(mask) != length:
            raise BufferFeedError(f"{where}: loss_mask has {len(mask)} entries, response_length {length}")
        tensors["response_mask"].append(mask + [0] * right)
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


def oversized(samples: Sequence[Mapping[str, Any]], shape: RowShape,
              tokenize: Callable[[str], Sequence[int]] | None = None) -> str | None:
    """Why one sample of the group does not fit ``shape`` (None: all fit)."""
    for sample in samples:
        sample = complete_prompt_tokens(sample, tokenize)
        length = int(sample.get("response_length") or 0)
        prompt = len(sample.get("tokens") or ()) - length
        if prompt > shape.prompt_length or length > shape.response_length:
            return (f"sample {sample.get('index')}: prompt {prompt} / response {length} tokens, "
                    f"rows {shape.prompt_length} / {shape.response_length}")
    return None


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


DROP_WINDOW_ENV = "YETO_BUFFER_DROP_WINDOW"  # groups per drop-rate window (default 20)
DROP_WARN_RATIO_ENV = "YETO_BUFFER_DROP_WARN_RATIO"  # warn above this drop rate (default 0.5)
DROP_STALL_WARN_ENV = "YETO_BUFFER_DROP_STALL_WARN_S"  # no trainable group for this long (default 600)


def _env_number(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        return default


def print_feed_event(event: str, **fields: Any) -> None:
    """The feed's default emitter: one ``YETO_RL_EVENT <json>`` stdout record."""
    from yeto.rl.event_echo import format_record

    print(format_record({"island_id": fields.pop("island_id", None), "time_unix": time.time(),
                         "event": str(event), **fields}), flush=True)


INCARNATION_FENCE_ENV = "YETO_BUFFER_INCARNATION_FENCE"  # "0": no stale-incarnation gate
LEASE_ROOT_ENV = "YETO_INCARNATION_LEASE_ROOT"  # the store holding the sampler leases (default: the buffer's)


def default_incarnation_gate(buffer: Any, environ: Mapping[str, str] | None = None) -> Any:
    """The feed's stale-incarnation gate over the buffer's store (or
    :data:`LEASE_ROOT_ENV`); None when off (:data:`INCARNATION_FENCE_ENV` ``0``) or
    the buffer has no store / entries (test doubles)."""
    from yeto.rl.engine.incarnation_lease import IncarnationGate

    env = os.environ if environ is None else environ
    if str(env.get(INCARNATION_FENCE_ENV, "1")).strip() == "0":
        return None
    entries = getattr(buffer, "entries", None)
    store = getattr(buffer, "store", None)
    if env.get(LEASE_ROOT_ENV):
        from yeto.rl.engine.resume import store_for

        store = store_for(str(env[LEASE_ROOT_ENV]), environ=env)
    if store is None or not callable(entries):
        return None
    return IncarnationGate(store, entries)


class BufferFeed:
    """The rollout port's queue calls over the sample buffer (module doc).

    ``buffer``: the trainer's ``FileSampleBuffer``; ``vmap``: returns the
    island's live consumption map (module doc, ``pull``); ``add_group(rows)``:
    hands kept rows to the trainer actor; ``record(fields)``: one record per
    pulled group."""

    def __init__(self, buffer: Any, *, vmap: Callable[[], VersionMap], shape: RowShape,
                 add_group: Callable[[dict[str, Any]], Any], record: Callable[[dict[str, Any]], None],
                 tokenize: Callable[[str], Sequence[int]] | None = None, wait_s: float = 3600.0,
                 poll_s: float = 2.0, consumer: str = TRAINER_CONSUMER,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 cursor_log: Any = None, merged: MergedVersions | None = None,
                 group_size: int | None = None, emit: Callable[..., None] | None = None,
                 drop_window: int | None = None, drop_warn_ratio: float | None = None,
                 drop_stall_warn_s: float | None = None, incarnation_gate: Any = "auto") -> None:
        """``cursor_log``: a synced trainer's lane cursor log
        (``merged_release.lane_cursor_log``); ``merged``: a synced trainer's
        merged version pairing (module doc). ``emit(event, **fields)`` reports a
        dropped group (``rl_buffer_group_dropped``, with running counts), a drop
        rate above ``drop_warn_ratio`` over the last ``drop_window`` groups
        (``rl_buffer_drop_rate_high``) and ``drop_stall_warn_s`` seconds of drops
        without a trainable group (``rl_buffer_feed_starved``); the thresholds
        default to :data:`DROP_WINDOW_ENV` / :data:`DROP_WARN_RATIO_ENV` /
        :data:`DROP_STALL_WARN_ENV`. ``incarnation_gate``: an
        ``incarnation_lease.IncarnationGate``; a group of a superseded sampler
        incarnation written after the new one claimed the lease is dropped as
        ``stale_incarnation``; ``"auto"`` builds it over the buffer's store
        (:func:`default_incarnation_gate`), None turns it off."""
        self.buffer = buffer
        self.gate = default_incarnation_gate(buffer) if incarnation_gate == "auto" else incarnation_gate
        self.emit = print_feed_event if emit is None else emit
        self.drop_window = max(1, int(drop_window if drop_window is not None
                                      else _env_number(DROP_WINDOW_ENV, 20)))
        self.drop_warn_ratio = float(drop_warn_ratio if drop_warn_ratio is not None
                                     else _env_number(DROP_WARN_RATIO_ENV, 0.5))
        self.drop_stall_warn_s = float(drop_stall_warn_s if drop_stall_warn_s is not None
                                       else _env_number(DROP_STALL_WARN_ENV, 600.0))
        self.dropped: dict[str, int] = {}
        self.accepted = 0
        self._outcomes: deque[bool] = deque(maxlen=self.drop_window)  # True: dropped
        self._last_accepted_at = clock()
        self._stall_warned_at: float | None = None
        self._streak = 0
        self.group_size = None if group_size is None else int(group_size)
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
        # (:meth:`commit_round`)
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
        while True:
            item = self._next()
            if item is None:
                return None
            stale = None if self.gate is None else self.gate.verdict(item.entry, int(item.position))
            if stale is not None:  # never trained, never read
                self.record({"position": int(item.position), "sample_id": item.entry.sample_id,
                             "kept": False, "dropped": STALE_REASON, "detail": stale,
                             "source": (item.entry.provenance or {}).get("sampler_id")})
                self.pulled_positions.append(int(item.position))
                self._dropped(STALE_REASON, int(item.position), stale)
                continue
            samples = decode_group(self.buffer.read_payload(item.entry.payload_ref))
            if not samples:
                raise BufferFeedError(f"buffer position {item.position}: empty group")
            too_long = oversized(samples, self.shape, self.tokenize)
            if too_long is None and self.group_size is not None and len(samples) != self.group_size:
                # An agentic group keeps >= min_samples trials, so it can be short;
                # verl's update splits the round into mini-batches of groups x n samples and
                # asserted (run s29r120-d: 15 % 16). A short group is dropped and recorded.
                reason = "short_group"
                too_long = f"{len(samples)} samples, the trainer trains groups of {self.group_size}"
            else:
                reason = "over_length"
            if too_long is None:
                self._accepted()
                break
            # A group with a sample wider than the rows (an agentic prompt past
            # max_prompt_length, a response past max_response_length) is dropped and recorded,
            # not fatal: before, group_rows raised in keep() and the island left INFRA_FAILED
            # (run s29r120-b: prompt 8200 > 8192). RL never truncates a trajectory.
            self.record({"position": int(item.position), "sample_id": item.entry.sample_id,
                         "kept": False, "dropped": reason, "detail": too_long,
                         "source": (item.entry.provenance or {}).get("sampler_id"), "samples": len(samples)})
            self.pulled_positions.append(int(item.position))
            self._dropped(reason, int(item.position), too_long)
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

    def _accepted(self) -> None:
        self.accepted += 1
        self._outcomes.append(False)
        if self._stall_warned_at is not None:
            print(f"[yeto-buffer-feed] a trainable group again after {self._streak} dropped groups",
                  flush=True)
        self._streak = 0
        self._last_accepted_at = self.clock()
        self._stall_warned_at = None

    def _dropped(self, reason: str, position: int, detail: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1
        self._outcomes.append(True)
        self._streak += 1
        total = sum(self.dropped.values())
        self.emit("rl_buffer_group_dropped", reason=reason, position=position, detail=detail,
                  count=self.dropped[reason], dropped_total=total, accepted_total=self.accepted)
        if len(self._outcomes) == self.drop_window:
            rate = sum(self._outcomes) / self.drop_window
            if rate > self.drop_warn_ratio:
                print(f"[yeto-buffer-feed] WARNING: {sum(self._outcomes)} of the last {self.drop_window} "
                      f"buffer groups were dropped (rate {rate:.0%} > {self.drop_warn_ratio:.0%}); "
                      f"dropped so far {dict(self.dropped)}, accepted {self.accepted}", flush=True)
                self.emit("rl_buffer_drop_rate_high", rate=rate, window=self.drop_window,
                          threshold=self.drop_warn_ratio, dropped=dict(self.dropped), accepted_total=self.accepted)
                self._outcomes.clear()  # one warning per window
        now = self.clock()
        idle = now - self._last_accepted_at
        if idle >= self.drop_stall_warn_s and (
                self._stall_warned_at is None or now - self._stall_warned_at >= self.drop_stall_warn_s):
            self._stall_warned_at = now
            print(f"[yeto-buffer-feed] WARNING: no trainable group for {idle:.0f} s: the last {self._streak} "
                  f"buffer groups were all dropped (last: {reason}, {detail}); training makes no progress "
                  f"until a group fits the trainer's rows and group size", flush=True)
            self.emit("rl_buffer_feed_starved", idle_s=idle, streak=self._streak, reason=reason,
                      dropped=dict(self.dropped), accepted_total=self.accepted)

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
        (a round that never closed, or one already committed)."""
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
    (:meth:`BufferFeed.commit_round`)."""

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
