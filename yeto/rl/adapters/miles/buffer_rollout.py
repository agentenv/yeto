"""Miles rollout through the durable sample buffer (rl-trainer-start-stop L4.5).

Two Miles rollout functions (``--rollout-function-path``, classes in
:mod:`.buffer_rollout_miles`) connect a sampler island and a trainer island
through a :class:`~yeto.rl.engine.sample_buffer.FileSampleBuffer`:

* sampler (``BufferWriterRolloutFn``): Miles generates the round as usual; every
  group is written to the buffer as one payload plus one index entry
  (:class:`GroupWriter`). The payload keeps each sample whole, so the per-token
  behavior log-probs (``rollout_log_probs``) and the per-token engine versions
  (``weight_versions``) reach the trainer for the D2 version gate and decoupled
  PPO. The entry's ``behavior_version`` is the OLDEST engine version in the group
  (the start version; P0-S schema 4).
* trainer (``BufferReaderRolloutFn``): the round is selected from the buffer, not
  generated (:class:`BatchReader`). Groups older than ``max_policy_age`` versions
  are passed over (consumed, not trained) and counted.

The consumed watermark: the reader remembers the buffer position after each
rollout id. The round cut before rollout ``N`` (all rollouts below ``N`` are
trained) commits the position after rollout ``N - 1`` and writes it into the cut
record (``RoundCutCheckpoint.buffer_watermark``); a restore resets the reader to
the watermark of the restored cut (``RoundCutCheckpoint.buffer_resume``). The
driver reaches the reader in the rollout executor's process with
:func:`executor_watermark_before` / :func:`executor_reset` (``__ray_call__``,
``MilesRolloutPool._in_executor``); :func:`wire_sample_buffer` connects both.

This module does not import Miles (it is tested without it).

Limits (first implementation):

* one trainer reads a buffer; on a shared directory (``index.jsonl``) one
  sampler island writes it (``flock`` holds only on one host);
* the buffer lives in the checkpoint store (:func:`~yeto.rl.engine.resume.store_for`
  at ``BUFFER_ROOT_ENV``): a Modal Volume or a shared directory that both islands
  mount (visibility is the store's ``commit``/``reload``), or an ``s3://`` store
  read and written by object key (``CheckpointStore.objects``), not through its
  bucket mount;
* the trainer does no evaluation through the buffer (eval rounds are refused);
* ``--fully-async`` is not supported on either side (it selects its own rollout
  function).
"""

from __future__ import annotations

import base64
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from yeto.rl.engine.policy_versions import engine_version_number
from yeto.rl.engine.trainer_lifecycle import SampleIndexEntry

SAMPLER, TRAINER, CONVERTIBLE = "sampler", "trainer", "convertible"
ROLES = (SAMPLER, TRAINER, CONVERTIBLE)

BUFFER_ROOT_ENV = "YETO_SAMPLE_BUFFER_ROOT"  # checkpoint store path holding sample-buffer/
SAMPLER_ID_ENV = "YETO_SAMPLE_BUFFER_SAMPLER_ID"  # prefix of the sample ids a sampler writes
WAIT_ENV = "YETO_SAMPLE_BUFFER_WAIT_S"  # reader: max seconds to wait for a full round
POLL_ENV = "YETO_SAMPLE_BUFFER_POLL_S"  # reader: seconds between index reads while waiting
MAX_POLICY_AGE_ENV = "YETO_SAMPLE_BUFFER_MAX_POLICY_AGE"
SHARD_ENV = "YETO_SAMPLE_BUFFER_SHARD"  # convertible node: its own shard of the index
# the rollout functions run in Ray workers, which inherit the raylet's environment:
# connect_island_ray forwards these to every worker
ISLAND_ENVS = (BUFFER_ROOT_ENV, SAMPLER_ID_ENV, WAIT_ENV, POLL_ENV, MAX_POLICY_AGE_ENV, SHARD_ENV)

DEFAULT_WAIT_S = 3600.0
DEFAULT_POLL_S = 2.0
TRAINER_CONSUMER = "trainer"

_MODULE = "yeto.rl.adapters.miles.buffer_rollout_miles"
ROLLOUT_FUNCTION_PATHS = {
    SAMPLER: f"{_MODULE}.BufferWriterRolloutFn",
    TRAINER: f"{_MODULE}.BufferReaderRolloutFn",
    CONVERTIBLE: f"{_MODULE}.ConvertibleRolloutFn",
}
# the rollout functions that read the buffer (the trainer keeps a watermark)
READER_FUNCTION_PATHS = (ROLLOUT_FUNCTION_PATHS[TRAINER], ROLLOUT_FUNCTION_PATHS[CONVERTIBLE])
PAYLOAD_SCHEMA = "yeto.rl.sample-buffer-payload/v1"
_NDARRAY = "__ndarray__"


class SampleBufferRolloutError(RuntimeError):
    pass


# ---------------------------------------------------------------------- argv
def rollout_argv(role: str | None, *, max_policy_age: int, fully_async: bool = False,
                 rollout_function_path: str | None = None) -> tuple[str, ...]:
    """The Miles flags of an island that writes (``sampler``) or reads (``trainer``)
    the sample buffer; ``()`` when ``role`` is None. Raises ``ValueError`` on a
    combination the buffer rollout does not support."""
    if role is None:
        return ()
    if role not in ROLES:
        raise ValueError(f"sample buffer role {role!r}: expected one of {ROLES}")
    if fully_async:
        raise ValueError("the sample buffer rollout does not support --fully-async "
                         "(it selects its own rollout function)")
    if rollout_function_path is not None:
        raise ValueError(f"the sample buffer rollout replaces the rollout function; "
                         f"{rollout_function_path!r} was also given")
    if role in (TRAINER, CONVERTIBLE) and int(max_policy_age) < 1:
        raise ValueError("a trainer that reads the sample buffer trains on samples of earlier "
                         f"versions: it needs max_policy_age >= 1 (got {max_policy_age})")
    return ("--rollout-function-path", ROLLOUT_FUNCTION_PATHS[role])


def island_env(role: str, *, store_root: str, sampler_id: str | None = None,
               max_policy_age: int | None = None, wait_s: float | None = None,
               shard: str | None = None) -> dict[str, str]:
    """The environment the launch layer exports on the island (read in the rollout
    process). A convertible node writes like a sampler, into its own ``shard`` of
    the index, and reads like a trainer."""
    if role not in ROLES:
        raise ValueError(f"sample buffer role {role!r}: expected one of {ROLES}")
    env = {BUFFER_ROOT_ENV: str(store_root)}
    if role in (SAMPLER, CONVERTIBLE):
        if not sampler_id:
            raise ValueError("a sampler needs a sampler id (sample id prefix)")
        env[SAMPLER_ID_ENV] = str(sampler_id)
    if role == CONVERTIBLE:
        if not shard:
            raise ValueError("a convertible node needs its buffer shard name")
        env[SHARD_ENV] = str(shard)
    if role in (TRAINER, CONVERTIBLE):
        if max_policy_age is None:
            raise ValueError("a trainer needs max_policy_age")
        env[MAX_POLICY_AGE_ENV] = str(int(max_policy_age))
        if wait_s is not None:
            env[WAIT_ENV] = f"{float(wait_s):g}"
    return env


def open_buffer(environ: Mapping[str, str] | None = None, *, batch_size: int = 1,
                shard: str | None = None, create_index: bool = True) -> Any:
    """The island's buffer, from :data:`BUFFER_ROOT_ENV` (store chosen by ``store_for``);
    with ``shard``, the writer of that shard of the index; ``create_index=False``
    for a reader whose index another host writes."""
    from yeto.rl.engine.resume import store_for
    from yeto.rl.engine.sample_buffer import INDEX, FileSampleBuffer, shard_index

    env = os.environ if environ is None else environ
    root = env.get(BUFFER_ROOT_ENV)
    if not root:
        raise SampleBufferRolloutError(f"{BUFFER_ROOT_ENV} is not set")
    return FileSampleBuffer(store_for(root, environ=env), batch_size=batch_size,
                            index=INDEX if shard is None else shard_index(shard),
                            create_index=create_index)


# ---------------------------------------------------------------------- payload
def _encode_value(value: Any) -> Any:
    tolist = getattr(value, "tolist", None)
    if callable(tolist) and hasattr(value, "dtype") and hasattr(value, "shape"):
        import numpy

        array = numpy.ascontiguousarray(value)
        return {_NDARRAY: base64.b64encode(array.tobytes()).decode("ascii"),
                "dtype": str(array.dtype), "shape": list(array.shape)}
    if hasattr(value, "value") and type(value).__module__ != "builtins":  # enum
        return value.value
    raise TypeError(f"sample field of type {type(value).__name__} is not serializable")


def _decode_hook(obj: dict[str, Any]) -> Any:
    if _NDARRAY in obj:
        import numpy

        data = base64.b64decode(obj[_NDARRAY])
        return numpy.frombuffer(data, dtype=obj["dtype"]).reshape(obj["shape"]).copy()
    return obj


def encode_group(samples: Sequence[Mapping[str, Any]]) -> bytes:
    """One payload: the group's samples as ``Sample.to_dict()`` mappings."""
    record = {"schema": PAYLOAD_SCHEMA, "samples": [dict(s) for s in samples]}
    return json.dumps(record, default=_encode_value, sort_keys=True, allow_nan=True).encode()


def decode_group(data: bytes) -> list[dict[str, Any]]:
    record = json.loads(data, object_hook=_decode_hook)
    if record.get("schema") != PAYLOAD_SCHEMA:
        raise SampleBufferRolloutError(f"payload schema {record.get('schema')!r} != {PAYLOAD_SCHEMA}")
    return list(record["samples"])


def _sample_versions(sample: Mapping[str, Any]) -> list[int]:
    versions = []
    for call in sample.get("weight_versions") or ():
        for span in call or ():
            versions.append(span["version"])
    if not versions and (sample.get("metadata") or {}).get("weight_version") is not None:
        versions.append(sample["metadata"]["weight_version"])
    numbers = [engine_version_number(v) for v in versions]
    if not numbers or None in numbers:
        raise SampleBufferRolloutError(
            f"sample {sample.get('index')}: no numeric engine weight version ({versions!r}); "
            "the buffer needs the behavior version of every sample")
    return [int(n) for n in numbers]


def group_entry_fields(samples: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    """``(behavior_version, tokens)`` of a group: the oldest engine version of any
    token, and the response tokens. Every sample must carry per-token behavior
    log-probs (decoupled PPO needs them)."""
    if not samples:
        raise SampleBufferRolloutError("empty group")
    oldest, tokens = None, 0
    for sample in samples:
        length = int(sample.get("response_length") or 0)
        logprobs = sample.get("rollout_log_probs")
        if logprobs is None or len(logprobs) != length:
            raise SampleBufferRolloutError(
                f"sample {sample.get('index')}: rollout_log_probs "
                f"{'missing' if logprobs is None else len(logprobs)} != response_length {length}")
        version = min(_sample_versions(sample))
        oldest = version if oldest is None else min(oldest, version)
        tokens += length
    return int(oldest), tokens


def renumber_group(samples: Sequence[dict[str, Any]], *, position: int,
                   samples_per_group: int) -> list[dict[str, Any]]:
    """Give a group read from the buffer an identity unique in the buffer.

    Miles numbers groups and samples per sampler process from 0, so groups from two
    sampler rounds (or a restarted sampler) repeat them; the trainer's batch checks
    and trajectory ids need them unique. The buffer position is unique for the
    buffer's life and the same after a restore re-reads it, so ``group_index`` is
    the position and ``index`` is ``position * samples_per_group + j``. The
    sampler's numbers stay in ``metadata["yeto_sampler_identity"]``."""
    if len(samples) > int(samples_per_group):
        raise SampleBufferRolloutError(
            f"group at buffer position {position} has {len(samples)} samples > "
            f"samples per group {samples_per_group}")
    out = []
    for j, sample in enumerate(samples):
        sample = dict(sample)
        metadata = dict(sample.get("metadata") or {})
        metadata["yeto_sampler_identity"] = {"group_index": sample.get("group_index"),
                                             "index": sample.get("index")}
        sample.update(metadata=metadata, group_index=int(position),
                      index=int(position) * int(samples_per_group) + j)
        out.append(sample)
    return out


# ---------------------------------------------------------------------- sampler side
class GroupWriter:
    """Writes finished groups to the buffer: payload first, then the index entry."""

    def __init__(self, buffer: Any, sampler_id: str) -> None:
        if not sampler_id:
            raise ValueError("sampler_id must be non-empty")
        self.buffer = buffer
        self.sampler_id = str(sampler_id)

    def write(self, samples: Sequence[Mapping[str, Any]], *, group_key: str) -> SampleIndexEntry:
        behavior_version, tokens = group_entry_fields(samples)
        sample_id = f"{self.sampler_id}-{group_key}"
        ref = self.buffer.write_payload(sample_id, encode_group(samples))
        entry = SampleIndexEntry(sample_id=sample_id, behavior_version=behavior_version,
                                 payload_ref=ref, tokens=tokens)
        self.buffer.append(entry)
        return entry


# ---------------------------------------------------------------------- trainer side
@dataclass
class TakeReport:
    rollout_id: int
    groups: int
    stale_groups: int  # passed over: older than max_policy_age (consumed, not trained)
    start: int  # buffer position of the first selected entry
    end: int  # buffer position after the last selected entry
    behavior_versions: dict[int, int] = field(default_factory=dict)
    wait_s: float = 0.0

    def metrics(self) -> dict[str, float]:
        return {"sample_buffer/groups": float(self.groups),
                "sample_buffer/stale_groups": float(self.stale_groups),
                "sample_buffer/start": float(self.start), "sample_buffer/end": float(self.end),
                "sample_buffer/wait_s": float(self.wait_s)}


class BatchReader:
    """Selects the trainer's rounds from the buffer (single consumer)."""

    def __init__(self, buffer: Any, *, max_policy_age: int, wait_s: float = DEFAULT_WAIT_S,
                 poll_s: float = DEFAULT_POLL_S, consumer: str = TRAINER_CONSUMER,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if int(max_policy_age) < 1:
            raise ValueError("max_policy_age must be >= 1")
        self.buffer = buffer
        self.max_policy_age = int(max_policy_age)
        self.wait_s = float(wait_s)
        self.poll_s = float(poll_s)
        self.consumer = consumer
        self.clock = clock
        self.sleep = sleep
        self._ends: dict[int, int] = {}  # rollout id -> buffer position after its round

    def take(self, rollout_id: int, groups: int, *, current_version: int | None
             ) -> tuple[list[tuple[int, list[dict[str, Any]]]], TakeReport]:
        """``groups`` ``(buffer position, decoded group)`` pairs for ``rollout_id``.
        Waits up to ``wait_s`` for them; after the data end returns what is left
        (raises when nothing is)."""
        if int(groups) < 1:
            raise ValueError("groups must be positive")
        rollout_id = int(rollout_id)
        started = self.clock()
        start = self.buffer.read_cursor
        out: list[tuple[int, list[dict[str, Any]]]] = []
        versions: dict[int, int] = {}
        stale = 0
        floor = None if current_version is None else int(current_version) - self.max_policy_age
        while len(out) < groups:
            selected = self.buffer.select(self.consumer, groups - len(out))
            for item in selected:
                if floor is not None and item.entry.behavior_version < floor:
                    stale += 1
                    continue
                out.append((item.position,
                            decode_group(self.buffer.read_payload(item.entry.payload_ref))))
                versions[item.entry.behavior_version] = versions.get(item.entry.behavior_version, 0) + 1
            if len(out) >= groups or selected:
                continue
            if self.buffer.data_ended:
                break
            if self.clock() - started > self.wait_s:
                raise SampleBufferRolloutError(
                    f"rollout {rollout_id}: {len(out)}/{groups} groups after {self.wait_s:g} s "
                    f"(buffer position {self.buffer.read_cursor})")
            self.sleep(self.poll_s)
        end = self.buffer.read_cursor
        if not out:
            raise SampleBufferRolloutError(f"rollout {rollout_id}: the sample buffer ended "
                                           f"with no trainable group (position {end})")
        self._ends[rollout_id] = end
        return out, TakeReport(rollout_id=rollout_id, groups=len(out), stale_groups=stale,
                               start=start, end=end, behavior_versions=dict(sorted(versions.items())),
                               wait_s=round(self.clock() - started, 3))

    def watermark_before(self, rollout_id: int) -> int:
        """Commit and return the position after the last round below ``rollout_id``
        (every round below it is trained when the cut before ``rollout_id`` runs)."""
        done = [rid for rid in self._ends if rid < int(rollout_id)]
        if done:
            self.buffer.commit_consumed(self._ends[max(done)])
            for rid in done:
                del self._ends[rid]
        return int(self.buffer.watermark)

    def reset(self, watermark: int) -> int:
        """Continue at the restored cut's watermark (rounds after it are selected again)."""
        self.buffer.reset_to(int(watermark))
        self._ends.clear()
        return int(self.buffer.watermark)


# ---------------------------------------------------------------------- executor link
_ACTIVE: dict[str, BatchReader] = {}
# the mode of a convertible node's rollout function (``ConvertibleRolloutFn``):
# SAMPLER generates into the node's shard, TRAINER reads the buffer
_MODE: dict[str, str] = {CONVERTIBLE: SAMPLER}


def convertible_mode() -> str:
    return _MODE[CONVERTIBLE]


def executor_set_mode(_executor: Any, mode: str) -> str:
    """Runs INSIDE the rollout executor actor (``__ray_call__``): switch a
    convertible node's rollout function between generating (``sampler``) and
    reading the buffer (``trainer``). Returns the previous mode."""
    if mode not in (SAMPLER, TRAINER):
        raise SampleBufferRolloutError(f"convertible rollout mode {mode!r}: expected "
                                       f"{SAMPLER} or {TRAINER}")
    previous, _MODE[CONVERTIBLE] = _MODE[CONVERTIBLE], mode
    return previous


def register_reader(reader: BatchReader) -> None:
    """Called by the reader rollout function in the rollout executor's process."""
    _ACTIVE[TRAINER] = reader


def _active_reader() -> BatchReader:
    reader = _ACTIVE.get(TRAINER)
    if reader is None:
        raise SampleBufferRolloutError("no sample buffer reader in this rollout process")
    return reader


def executor_watermark_before(_executor: Any, rollout_id: int) -> int:
    """Runs INSIDE the rollout executor actor (``__ray_call__``)."""
    return _active_reader().watermark_before(int(rollout_id))


def executor_read_cursor(_executor: Any) -> int | None:
    """Runs INSIDE the rollout executor actor: the buffer position after the last
    selected group (None before the reader exists)."""
    reader = _ACTIVE.get(TRAINER)
    return None if reader is None else int(reader.buffer.read_cursor)


def executor_reset(_executor: Any, watermark: int) -> int:
    """Runs INSIDE the rollout executor actor (``__ray_call__``)."""
    return _active_reader().reset(int(watermark))


class BufferCutLink:
    """Driver side: the round cut's watermark and restore hooks over ``in_executor``
    (``MilesRolloutPool._in_executor``: runs ``fn(executor, *args)`` in the actor)."""

    def __init__(self, in_executor: Callable[..., Any]) -> None:
        self.in_executor = in_executor

    def watermark(self, rollout_id: int) -> int:
        return int(self.in_executor(executor_watermark_before, int(rollout_id)))

    def resume(self, watermark: int) -> int:
        return int(self.in_executor(executor_reset, int(watermark)))


def wire_sample_buffer(checkpoint: Any, rollout: Any) -> BufferCutLink:
    """Give a ``RoundCutCheckpoint`` the trainer's buffer watermark and restore hook."""
    in_executor = getattr(rollout, "_in_executor", None)
    if not callable(in_executor):
        raise SampleBufferRolloutError(f"rollout pool {type(rollout).__name__} cannot run "
                                       "code in the rollout executor")
    link = BufferCutLink(in_executor)
    checkpoint.buffer_watermark = link.watermark
    checkpoint.buffer_resume = link.resume
    return link
