"""Fakes for the store fault tests (read path, zombie writers, clocks, load).

:class:`FaultS3` is a boto3 S3 client double under the real
``yeto.cloud.s3_store.S3ObjectStore``: one server clock (``now``: every object's
``LastModified`` and every listing's ``Date`` header), paged listings that read
the bucket page by page (a write between two pages is seen by the later page
only, as on S3), per-call counters and one-shot hooks that run right before a
call (``before``) or make it fail (``fail``), so a test places another actor's
step at an exact point of a writer's sequence.

:class:`Sampler` wires the real modules of one engine sampler process the way
``engine_sampler_entry.run`` does (shard buffer, ``GroupWriter``, incarnation
lease, store dispenser, ``EngineRound._write_scored``); :func:`trainer` and
:func:`drain` are the trainer's merging buffer and ``BatchReader``.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from email.utils import formatdate
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError

from yeto.cloud.s3_store import S3ObjectStore
from yeto.launch.nc.elastic import ObjectRecords
from yeto.launch.pool.data_path import R2DataPath
from yeto.rl.adapters.miles.buffer_rollout import BatchReader, SampleBufferRolloutError
from yeto.rl.engine import store_dispense as sd
from yeto.rl.engine.buffer_payload import GroupWriter
from yeto.rl.engine.engine_sampler import EngineRound
from yeto.rl.engine.incarnation_lease import IncarnationLease, read_lease
from yeto.rl.engine.sample_buffer import FileSampleBuffer, open_trainer_buffer, shard_index

BUCKET = "bkt"
URI = f"s3://{BUCKET}"
NAMESPACE = "task-dispense"


def client_error(code: str, op: str = "Op") -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, op)


class FaultS3:
    """The boto3 S3 client calls ``S3ObjectStore`` makes, over a dict (module doc)."""

    class exceptions:
        class NoSuchKey(Exception):
            pass

    def __init__(self, *, page_size: int = 1000) -> None:
        self.now = 1_000_000.0
        self.page_size = int(page_size)
        self.objects: dict[str, tuple[bytes, float]] = {}
        self.calls: Counter[str] = Counter()
        self._hooks: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ hooks
    def before(self, op: str, match: str | Callable[[str], bool], action: Callable[[str], None], *,
               times: int = 1, skip: int = 0) -> None:
        """Run ``action(key)`` right before the next ``times`` calls of ``op`` whose key
        matches (a substring or a predicate), after letting ``skip`` of them pass."""
        self._hooks.append({"op": op, "match": match, "action": action, "times": int(times), "skip": int(skip)})

    def fail(self, op: str, match: str | Callable[[str], bool], error: BaseException | Callable[[], BaseException],
             *, times: int = 1, skip: int = 0) -> None:
        def action(_key: str) -> None:
            raise error() if callable(error) and not isinstance(error, BaseException) else error

        self.before(op, match, action, times=times, skip=skip)

    def clear_hooks(self) -> None:
        self._hooks.clear()

    def _enter(self, op: str, key: str) -> None:
        self.calls[op] += 1
        for hook in list(self._hooks):
            match = hook["match"]
            if hook["op"] != op or not (match(key) if callable(match) else match in key):
                continue
            if hook["skip"] > 0:
                hook["skip"] -= 1
                continue
            hook["times"] -= 1
            if hook["times"] <= 0:
                self._hooks.remove(hook)
            hook["action"](key)

    # ------------------------------------------------------------------ calls
    def get_object(self, Bucket, Key, Range=None, IfMatch=None):  # noqa: N803 - boto3 names
        self._enter("get", Key)
        if Key not in self.objects:
            raise self.exceptions.NoSuchKey(Key)
        data = self.objects[Key][0]
        return {"Body": type("Body", (), {"read": lambda self: data})()}

    def put_object(self, Bucket, Key, Body, IfNoneMatch=None, **_):  # noqa: N803
        self._enter("create" if IfNoneMatch is not None else "put", Key)
        if IfNoneMatch is not None and Key in self.objects:
            raise client_error("PreconditionFailed", "PutObject")
        self.objects[Key] = (bytes(Body), self.now)

    def head_object(self, Bucket, Key, **_):  # noqa: N803
        self._enter("head", Key)
        if Key not in self.objects:
            raise client_error("404", "HeadObject")
        return {"ContentLength": len(self.objects[Key][0]), "ETag": '"x"'}

    def delete_object(self, Bucket, Key):  # noqa: N803
        self._enter("delete", Key)
        self.objects.pop(Key, None)

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        fake = self

        class Paginator:
            def paginate(self, Bucket, Prefix, StartAfter=""):  # noqa: N803
                after = StartAfter
                while True:
                    fake._enter("list", Prefix)  # one request per page, over the bucket as it is now
                    keys = sorted(k for k in fake.objects if k.startswith(Prefix) and k > after)
                    page = keys[:fake.page_size]
                    yield {"ResponseMetadata": {"HTTPHeaders": {"date": formatdate(fake.now, usegmt=True)}},
                           "Contents": [{"Key": k, "LastModified": datetime.fromtimestamp(
                               fake.objects[k][1], tz=timezone.utc)} for k in page if k in fake.objects]}
                    if len(keys) <= fake.page_size:
                        return
                    after = page[-1]

        return Paginator()

    # ---------------------------------------------------------------- helpers
    def store(self) -> S3ObjectStore:
        return S3ObjectStore(URI, client=self)

    def put_json(self, key: str, doc: Any) -> None:
        self.objects[key] = ((json.dumps(doc, sort_keys=True) + "\n").encode(), self.now)

    def json(self, key: str) -> Any:
        return json.loads(self.objects[key][0])

    def keys(self, prefix: str = "") -> list[str]:
        return sorted(k for k in self.objects if k.startswith(prefix))


def sample(index: int, group: int, version: int = 1, *, reward: Any = 0.5, metadata: dict | None = None) -> dict:
    return {"index": index, "group_index": group, "prompt": "p", "response": "r", "label": None,
            "tokens": [1, 2, 3, 4, 5], "response_length": 3, "rollout_log_probs": [-0.1, -0.2, -0.3],
            "reward": reward, "status": "completed",
            "metadata": {"weight_version": str(version), "prompt_tokens_known": True, **(metadata or {})}}


def group(g: int, version: int = 1, *, n: int = 2, reward: Any = None, metadata: dict | None = None) -> list[dict]:
    """``n`` samples with distinct rewards (a constant ``reward`` when given)."""
    return [sample(n * g + j, g, version, reward=0.25 * (j + 1) if reward is None else reward, metadata=metadata)
            for j in range(n)]


def shard_buffer(s3: FaultS3, shard: str, *, clock: Callable[[], float] | None = None) -> FileSampleBuffer:
    return FileSampleBuffer(s3.store(), batch_size=1, index=shard_index(shard),
                            clock=clock or (lambda: s3.now))


def trainer(s3: FaultS3, shards: tuple[str, ...] | None = ("s0", "s1")) -> FileSampleBuffer:
    """The trainer's merging buffer (``--rl-buffer-shards`` = ``shards``)."""
    if shards is None:
        return FileSampleBuffer(s3.store(), batch_size=1, merge_shards=True)
    return open_trainer_buffer(s3.store(), batch_size=1, shards=shards)


def reader(buffer: Any, **kwargs: Any) -> BatchReader:
    kwargs.setdefault("group_size", 2)
    return BatchReader(buffer, max_policy_age=8, wait_s=0.0, poll_s=0.0, prefetch=False,
                       sleep=lambda s: None, **kwargs)


def drain(batch_reader: BatchReader, *, start: int = 0, limit: int = 64) -> tuple[list[Any], list[Any]]:
    """Take one group per round until the buffer has no more: ``(entries trained, reports)``."""
    trained, reports = [], []
    for rollout_id in range(start, start + limit):
        try:
            taken, report = batch_reader.take(rollout_id, 1, current_version=1)
        except SampleBufferRolloutError:
            break
        entries = batch_reader.buffer.entries()
        trained += [entries[position] for position, _ in taken]
        reports.append(report)
    return trained, reports


def slot_of(entry: Any) -> int | None:
    return (entry.sampler_round or {}).get("dispense_position")


def ledger_states(s3: FaultS3, namespace: str = NAMESPACE) -> dict[int, sd.TaskState]:
    keys, _now = s3.store().list_meta(namespace + "/")
    return sd.parse_ledger(keys, namespace)


def trained_slots(s3: FaultS3, namespace: str = NAMESPACE) -> set[int]:
    return {p for p, state in ledger_states(s3, namespace).items() if state.done == "trained"}


def write_prompts(tmp_path: Path, rows: int) -> Path:
    path = tmp_path / "prompts.jsonl"
    path.write_text("".join(json.dumps({"text": f"q{i}", "metadata": {"task_id": f"t{i}"}}) + "\n"
                            for i in range(rows)))
    return path


class _Provenance:
    def key(self) -> str:
        return "test"

    def to_record(self) -> dict[str, str]:
        return {}


class Sampler:
    """One engine sampler process over the shared bucket (module doc).

    ``skew``: this node's wall clock minus the server's (``appended_at`` of its
    groups, ``claimed_at`` of its lease); ``mono``: its monotonic clock (the lease's
    ``write_check_s`` cache and the dispenser's scan pacing)."""

    def __init__(self, s3: FaultS3, prompts: Path, holder: str, incarnation: str, *, skew: float = 0.0,
                 lease_s: float = 900.0, max_redo: int = 2, groups_per_round: int = 1,
                 lease: bool = True, claim: bool = True) -> None:
        self.s3, self.holder, self.incarnation = s3, holder, incarnation
        self.skew = float(skew)
        self.mono = 0.0
        self.objects = s3.store()
        self.records = ObjectRecords(self.objects)
        self.buffer = shard_buffer(s3, holder, clock=self.wall)
        self.writer = GroupWriter(self.buffer, holder)
        self.lease = (IncarnationLease(self.records, holder, incarnation, clock=self.wall,
                                       monotonic=lambda: self.mono) if lease else None)

        def lease_of_holder(sampler: str) -> str | None:  # engine_sampler_entry.build_prompts
            record = read_lease(self.records, sampler)
            return None if not record or record.get("incarnation") is None else str(record["incarnation"])

        config = {"backend": "store", "namespace": NAMESPACE, "lease_s": lease_s, "max_redo": max_redo,
                  "scan_s": 0.0, "max_scan_s": 0.0}
        self.prompts = sd.StoreDispensedPrompts.open(
            prompts, config, store=self.objects, holder=holder, incarnation=incarnation, input_key="text",
            label_key=None, groups_per_round=groups_per_round, renew=False, sleep=lambda s: None,
            clock=lambda: self.mono, lease_of_holder=lease_of_holder if lease else None)
        self.prompts.recover(self.buffer.entries())
        if self.lease is not None and claim:
            self.lease.claim(self.buffer.entries())
            self.prompts.fenced = lambda: self.lease.fenced
        self.round = object.__new__(EngineRound)  # only what _write_scored / _write_group read
        self.round.writer, self.round.prompts = self.writer, self.prompts
        self.round.incarnation, self.round.guard, self.round.data_cursor = incarnation, None, None
        self.round.reclaimed = lambda: False
        self.round.write_fenced = self.lease.check_before_write if self.lease is not None else (lambda: False)
        self.round.dropped_groups = 0
        self.round.provenance = _Provenance()
        self.round.emit = lambda *args, **fields: None
        self.rounds = 0

    def wall(self) -> float:
        return self.s3.now + self.skew

    def draw(self) -> list[Any]:
        """Claim the next tasks (``StoreDispensedPrompts.__call__``)."""
        return list(self.prompts(self.rounds))

    def write(self, items: list[Any], *, version: int = 1) -> int:
        """Score and write the groups of ``items`` (``EngineRound._write_scored``: the
        lease check, ``still_holds``, payload, index record, then the settlement);
        returns the groups written."""
        scored = [(item, group(i, version, metadata=dict(item.metadata))) for i, item in enumerate(items)]
        written = self.round._write_scored(self.rounds, version, scored, finish={}, tokens=0, record={})
        self.rounds += 1
        return written

    def held(self) -> set[int]:
        return {p for p, h in self.prompts._held.items() if not h.lost}


def pool_fence(s3: FaultS3, shard: str, *, skew: float = 0.0, by: str = "ticket-1") -> dict[str, Any]:
    """The pool retires ``shard`` (``R2DataPath.fence_shard``: handoff record, lease
    claim as ``pool-fence:<by>``, then ``store_dispense.retire``); ``skew``: the
    control machine's wall clock minus the server's."""
    objects = s3.store()
    path = R2DataPath(ObjectRecords(objects), (objects, NAMESPACE), clock=lambda: s3.now + skew)
    return path.fence_shard(shard, by=by)
