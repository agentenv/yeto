"""A fake boto3 S3 client with a virtual clock and seeded fault injection.

``yeto.cloud.s3_store.S3ObjectStore`` runs unchanged on top of it, so the real
retry verdicts (``ResponseMetadata.RetryAttempts``), pagination, multipart and
ETag-pinned range reads are what the simulation exercises.

One :class:`S3Server` holds the bucket; every participant (a sampler process, a
trainer, the pool) talks to it through its own :class:`FaultyS3Client`. A client
call is one SDK call: the SDK's own retries happen inside it (:meth:`FaultyS3Client._send`),
as botocore's do. Per call the :class:`Injector` may

* lose the answer of an attempt that the server applied (the retry of a
  conditional put is then refused as "the key exists", with ``RetryAttempts`` >= 1);
* lose the request (connection reset, 5xx, 429): not applied, retried;
* fail every attempt (an outage; the first attempt may have been applied) and
  raise a transport error, as botocore does once its retries are used up;
* fail every request of a partitioned node;
* answer slowly (the virtual clock moves);
* cut a GET body short (the read raises; botocore does not retry a body);
* kill the calling process (:class:`Crash`) before or after the server applied it;
* let another participant run in the middle of the call (``preempt``).

Times: the server stamps ``LastModified`` and the ``Date`` header from the
simulation clock; a participant's own clock may be skewed (:class:`ActorClock`).
"""

from __future__ import annotations

import datetime
import hashlib
import random
import threading
from dataclasses import dataclass, field
from email.utils import format_datetime
from typing import Any, Callable

from botocore.exceptions import (
    ClientError,
    ConnectionClosedError,
    EndpointConnectionError,
    IncompleteReadError,
    ReadTimeoutError,
)

READ_OPS = frozenset({"get", "head", "list", "mpu_list"})
MAX_ATTEMPTS = 8  # yeto.cloud.s3_store.RETRIES
ENDPOINT = "https://sim.invalid"
EPOCH = 1_800_000_000.0  # the simulation starts here (server time, seconds)


class Crash(BaseException):
    """The calling process was killed at this store operation. A ``BaseException``:
    no ``except Exception`` of the code under test may survive it."""


class SimClock:
    """The one clock of a simulation (server time)."""

    def __init__(self, start: float = EPOCH) -> None:
        self.now = float(start)

    def advance(self, seconds: float) -> None:
        self.now += max(0.0, float(seconds))


class ActorClock:
    """A participant's view of the time: its wall clock is skewed, its monotonic
    clock is not."""

    def __init__(self, clock: SimClock, skew_s: float = 0.0) -> None:
        self.clock, self.skew_s = clock, float(skew_s)

    def time(self) -> float:
        return self.clock.now + self.skew_s

    def monotonic(self) -> float:
        return self.clock.now - EPOCH

    def sleep(self, seconds: float) -> None:
        self.clock.advance(seconds)


@dataclass
class FaultConfig:
    """Per-call fault probabilities (all 0: a perfect store)."""

    lost_response: float = 0.0  # applied, the answer lost, the SDK retries
    lost_request: float = 0.0  # not applied (reset / 5xx / 429), the SDK retries
    outage: float = 0.0  # every attempt fails; the call raises
    retry_fault: float = 0.2  # a retry fails again (not applied)
    slow: float = 0.0  # the call takes up to ``slow_max_s``
    slow_max_s: float = 20.0
    body_cut: float = 0.0  # a GET body ends early
    preempt: float = 0.0  # another participant acts in the middle of the call
    crash: float = 0.0  # per scheduler action: the actor dies at one of its next calls
    partition: float = 0.0  # per scheduler step: a node loses the store for a while
    partition_max_s: float = 90.0
    control: float = 0.0  # per scheduler step: a pool action (retire, drain, fence, restart)
    trainer_restart: float = 0.0  # per scheduler step: the trainer is restarted from its last cut
    read_outage: float = 0.0  # per scheduler step: reads (GET / HEAD / LIST) fail for a while
    read_outage_max_s: float = 600.0
    clock_skew_s: float = 0.0  # each process's wall clock is off by up to this much
    two_trainers: bool = False  # a restarted trainer's predecessor keeps running for a while

    @classmethod
    def preset(cls, name: str) -> "FaultConfig":
        return cls(**PRESETS[name])


PRESETS: dict[str, dict[str, Any]] = {
    "none": {},
    # the SDK-visible network faults only: every process stays up
    "network": {"lost_response": 0.03, "lost_request": 0.03, "outage": 0.004, "slow": 0.02,
                "body_cut": 0.01, "preempt": 0.05},
    # lost answers only (the PR #354 class), densely
    "lost-response": {"lost_response": 0.08, "preempt": 0.03},
    # processes die and restart, nodes are partitioned, the pool acts
    "process": {"crash": 0.03, "partition": 0.01, "control": 0.02, "trainer_restart": 0.01,
                "preempt": 0.05, "clock_skew_s": 5.0},
    "all": {"lost_response": 0.02, "lost_request": 0.02, "outage": 0.003, "slow": 0.02, "body_cut": 0.01,
            "preempt": 0.05, "crash": 0.02, "partition": 0.008, "control": 0.02, "trainer_restart": 0.008,
            "clock_skew_s": 5.0},
    # the store refuses reads for up to ``read_outage_max_s`` (one node or all); writes pass
    "read-outage": {"read_outage": 0.02},
    # the pool takes partitioned nodes for gone and fences their shards while the old
    # process still runs; every clock is up to a minute off
    "fence-zombie": {"partition": 0.04, "partition_max_s": 240.0, "control": 0.08, "preempt": 0.05,
                     "clock_skew_s": 60.0},
    # the unfixed design gap: two trainers merge the same shards
    "two-trainers": {"trainer_restart": 0.03, "two_trainers": True, "preempt": 0.05},
}


def _client_error(code: str, op: str, status: int, attempts: int = 0) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code},
                        "ResponseMetadata": {"HTTPStatusCode": status, "RetryAttempts": int(attempts)}}, op)


class _NoSuchKey(ClientError):
    pass


@dataclass
class _Object:
    data: bytes
    modified: float

    @property
    def etag(self) -> str:
        return '"' + hashlib.md5(self.data).hexdigest() + '"'


class S3Server:
    """The bucket: objects with their server-side modification time, multipart
    uploads in progress, and what the oracle wants to know afterwards."""

    def __init__(self, clock: SimClock, *, page_size: int = 7) -> None:
        self.clock = clock
        self.page_size = int(page_size)
        self.objects: dict[str, _Object] = {}
        self.uploads: dict[str, dict[int, bytes]] = {}
        self._upload_seq = 0
        self.lock = threading.RLock()
        self.requests = 0
        # conditional puts this server applied whose answer was lost and whose retry it
        # then refused: key -> actor (the writer owns the object without having been told)
        self.own_retry_refusals: dict[str, str] = {}
        self.failed_calls: set[tuple[str, str, str]] = set()  # (actor, op, key) of calls that raised

    # -- the oracle's direct view (no client, no faults) ------------------------
    def get(self, key: str) -> bytes | None:
        obj = self.objects.get(key)
        return None if obj is None else obj.data

    def keys(self, prefix: str = "") -> list[str]:
        return sorted(k for k in self.objects if k.startswith(prefix))


class Injector:
    """Decides the faults of every client call from one seeded generator."""

    def __init__(self, rng: random.Random, clock: SimClock, faults: FaultConfig,
                 trace: Callable[..., None] = lambda *a, **k: None) -> None:
        self.rng, self.clock, self.faults, self.trace = rng, clock, faults, trace
        self.enabled = True
        self.partitioned: dict[str, float] = {}  # node -> server time the partition heals
        self.read_outage: dict[str, float] = {}  # node (or "*": all) -> server time reads work again
        self.crash_in: dict[str, tuple[int, bool]] = {}  # actor -> (calls left, after the apply)
        self.preempt: Callable[[str], None] | None = None  # run another participant now
        self._depth = 0
        self._lock = threading.RLock()
        self.counts: dict[str, int] = {}

    def _count(self, name: str) -> None:
        self.counts[name] = self.counts.get(name, 0) + 1

    def is_partitioned(self, node: str) -> bool:
        until = self.partitioned.get(node)
        if until is None:
            return False
        if self.clock.now >= until:
            del self.partitioned[node]
            self.trace("heal", node=node)
            return False
        return True

    def _reads_down(self, node: str) -> bool:
        for name in (node, "*"):
            until = self.read_outage.get(name)
            if until is not None:
                if self.clock.now < until:
                    return True
                del self.read_outage[name]
                self.trace("reads_back", node=name)
        return False

    def begin(self, actor: str, node: str, op: str, key: str) -> bool:
        """Called once per SDK call; returns whether the process dies right after
        the server applied it. Raises :class:`Crash` when it dies before."""
        with self._lock:
            self.clock.advance(0.005)
            due = self.crash_in.get(actor)
            if due is not None:
                left, after = due
                if left <= 0:
                    del self.crash_in[actor]
                    self._count("crash")
                    self.trace("crash", actor=actor, op=op, key=key, after_apply=after)
                    if not after:
                        raise Crash(f"{actor} killed before {op} {key}")
                    return True
                self.crash_in[actor] = (left - 1, after)
            if not self.enabled:
                return False
            f = self.faults
            if f.slow and self.rng.random() < f.slow:
                self._count("slow")
                self.clock.advance(self.rng.uniform(1.0, f.slow_max_s))
            run = (f.preempt and self.preempt is not None and self._depth == 0
                   and self.rng.random() < f.preempt)
        if run:
            self._depth += 1
            try:
                self._count("preempt")
                self.preempt(actor)
            finally:
                self._depth -= 1
        return False

    def attempt(self, actor: str, node: str, op: str, key: str, attempt: int,
                mutates: bool) -> tuple[str, bool] | None:
        """The fault of one attempt: ``(kind, applied)`` or None (it is answered).
        ``kind``: ``partition`` / ``outage`` end the call, the others are retried."""
        with self._lock:
            if self.is_partitioned(node):
                self._count("partitioned_call")
                return "partition", False
            if op in READ_OPS and self._reads_down(node):
                self._count("read_outage_call")
                return "outage", False
            if not self.enabled:
                return None
            f, r = self.faults, self.rng.random()
            if attempt:
                return (self.rng.choice(("reset", "5xx", "429")), False) if r < f.retry_fault else None
            if r < f.lost_response:
                self._count("lost_response")
                self.trace("lost_response", actor=actor, op=op, key=key)
                return "reset", True
            r -= f.lost_response
            if r < f.lost_request:
                self._count("lost_request")
                return self.rng.choice(("reset", "5xx", "429")), False
            r -= f.lost_request
            if r < f.outage:
                applied = bool(mutates and self.rng.random() < 0.5)
                self._count("outage")
                self.trace("outage", actor=actor, op=op, key=key, applied=applied)
                return "outage", applied
            return None

    def body_cut(self, actor: str, key: str) -> bool:
        with self._lock:
            if self.enabled and self.faults.body_cut and self.rng.random() < self.faults.body_cut:
                self._count("body_cut")
                self.trace("body_cut", actor=actor, key=key)
                return True
            return False

    def backoff(self, exhausted: bool) -> None:
        with self._lock:
            self.clock.advance(self.rng.uniform(5.0, 40.0) if exhausted else self.rng.uniform(0.2, 2.0))


def _transport_error(kind: str, op: str, attempts: int) -> Exception:
    if kind == "partition":
        return EndpointConnectionError(endpoint_url=ENDPOINT)
    if kind in ("5xx", "429"):
        code, status = ("SlowDown", 429) if kind == "429" else ("InternalError", 500)
        return _client_error(code, op, status, attempts)
    if kind == "outage":
        return ReadTimeoutError(endpoint_url=ENDPOINT)
    return ConnectionClosedError(endpoint_url=ENDPOINT)


class _Body:
    def __init__(self, data: bytes, cut: bool) -> None:
        self._data, self._cut = data, cut

    def read(self) -> bytes:
        if self._cut:
            raise IncompleteReadError(actual_bytes=len(self._data) // 2, expected_bytes=len(self._data))
        return self._data


class FaultyS3Client:
    """The boto3 S3 client calls ``S3ObjectStore`` makes, for one participant.
    ``actor``: the process (its crash countdown); ``node``: the host (its partition).
    ``injector=None``: a perfect client (the oracle's)."""

    class exceptions:
        NoSuchKey = _NoSuchKey

    def __init__(self, server: S3Server, actor: str, injector: Injector | None = None, *,
                 node: str | None = None) -> None:
        self.server, self.actor, self.injector = server, actor, injector
        self.node = node or actor

    # -- one SDK call: attempts, retries, the answer ----------------------------
    def _send(self, op: str, key: str, apply: Callable[[], Any], *, mutates: bool = False,
              conditional: bool = False) -> Any:
        inj = self.injector
        die_after = inj.begin(self.actor, self.node, op, key) if inj is not None else False
        attempt = 0
        applied_unanswered = False
        while True:
            fault = None if inj is None else inj.attempt(self.actor, self.node, op, key, attempt, mutates)
            if fault is None:
                try:
                    with self.server.lock:
                        self.server.requests += 1
                        out = apply()
                except ClientError as error:
                    error.response.setdefault("ResponseMetadata", {})["RetryAttempts"] = attempt
                    if conditional and applied_unanswered and error.response["Error"]["Code"] == "PreconditionFailed":
                        self.server.own_retry_refusals[key] = self.actor
                        if inj is not None:
                            inj.trace("create_refused_after_own_retry", actor=self.actor, key=key)
                    raise
                if die_after:
                    raise Crash(f"{self.actor} killed after {op} {key}")
                if isinstance(out, dict):
                    out.setdefault("ResponseMetadata", {}).update(RetryAttempts=attempt, HTTPStatusCode=200)
                return out
            kind, applied = fault
            if applied:
                try:
                    with self.server.lock:
                        self.server.requests += 1
                        apply()
                    applied_unanswered = True
                except ClientError:
                    pass  # the lost answer was an error
            attempt += 1
            exhausted = kind in ("partition", "outage") or attempt >= MAX_ATTEMPTS
            inj.backoff(exhausted)
            if exhausted:
                if die_after:
                    raise Crash(f"{self.actor} killed during {op} {key}")
                self.server.failed_calls.add((self.actor, op, key))
                raise _transport_error(kind, op, attempt - 1)

    # -- objects ------------------------------------------------------------------
    def get_object(self, Bucket: str, Key: str, Range: str | None = None, IfMatch: str | None = None) -> dict:
        def apply() -> dict:
            obj = self.server.objects.get(Key)
            if obj is None:
                raise _NoSuchKey({"Error": {"Code": "NoSuchKey", "Message": Key},
                                  "ResponseMetadata": {"HTTPStatusCode": 404}}, "GetObject")
            if IfMatch is not None and IfMatch != obj.etag:
                raise _client_error("PreconditionFailed", "GetObject", 412)
            data = obj.data
            if Range is not None:
                first, last = (int(x) for x in Range.removeprefix("bytes=").split("-"))
                data = data[first:last + 1]
            return {"_data": data, "ETag": obj.etag, "ContentLength": len(data)}

        out = self._send("get", Key, apply)
        cut = self.injector is not None and self.injector.body_cut(self.actor, Key)
        if cut:
            self.server.failed_calls.add((self.actor, "get", Key))
        out["Body"] = _Body(out.pop("_data"), cut)
        return out

    def put_object(self, Bucket: str, Key: str, Body: bytes, IfNoneMatch: str | None = None,
                   ChecksumSHA256: str | None = None) -> dict:
        data = bytes(Body)

        def apply() -> dict:
            if IfNoneMatch is not None and Key in self.server.objects:
                raise _client_error("PreconditionFailed", "PutObject", 412)
            self.server.objects[Key] = _Object(data, self.server.clock.now)
            return {"ETag": self.server.objects[Key].etag}

        return self._send("create" if IfNoneMatch is not None else "put", Key, apply, mutates=True,
                          conditional=IfNoneMatch is not None)

    def head_object(self, Bucket: str, Key: str, ChecksumMode: str | None = None) -> dict:
        def apply() -> dict:
            obj = self.server.objects.get(Key)
            if obj is None:
                raise _client_error("404", "HeadObject", 404)
            return {"ContentLength": len(obj.data), "ETag": obj.etag}

        return self._send("head", Key, apply)

    def delete_object(self, Bucket: str, Key: str) -> dict:
        def apply() -> dict:
            self.server.objects.pop(Key, None)
            return {}

        return self._send("delete", Key, apply, mutates=True)

    # -- multipart ----------------------------------------------------------------
    def create_multipart_upload(self, Bucket: str, Key: str, ChecksumAlgorithm: str | None = None) -> dict:
        def apply() -> dict:
            self.server._upload_seq += 1
            upload_id = f"u{self.server._upload_seq}"
            self.server.uploads[upload_id] = {}
            return {"UploadId": upload_id}

        return self._send("mpu_create", Key, apply, mutates=True)

    def upload_part(self, Bucket: str, Key: str, UploadId: str, PartNumber: int, Body: bytes,
                    ChecksumSHA256: str | None = None) -> dict:
        data = bytes(Body)

        def apply() -> dict:
            if UploadId not in self.server.uploads:
                raise _client_error("NoSuchUpload", "UploadPart", 404)
            self.server.uploads[UploadId][int(PartNumber)] = data
            return {"ETag": '"' + hashlib.md5(data).hexdigest() + '"'}

        return self._send("mpu_part", Key, apply, mutates=True)

    def list_parts(self, Bucket: str, Key: str, UploadId: str, PartNumberMarker: int = 0, MaxParts: int = 2) -> dict:
        def apply() -> dict:
            if UploadId not in self.server.uploads:
                raise _client_error("NoSuchUpload", "ListParts", 404)
            parts = self.server.uploads[UploadId]
            numbers = sorted(n for n in parts if n > int(PartNumberMarker))
            page, rest = numbers[:MaxParts], numbers[MaxParts:]
            return {"Parts": [{"PartNumber": n, "ETag": '"' + hashlib.md5(parts[n]).hexdigest() + '"',
                               "Size": len(parts[n])} for n in page],
                    "IsTruncated": bool(rest), **({"NextPartNumberMarker": page[-1]} if rest else {})}

        return self._send("mpu_list", Key, apply)

    def complete_multipart_upload(self, Bucket: str, Key: str, UploadId: str, MultipartUpload: dict) -> dict:
        def apply() -> dict:
            parts = self.server.uploads.get(UploadId)
            if parts is None:
                raise _client_error("NoSuchUpload", "CompleteMultipartUpload", 404)
            numbers = [int(p["PartNumber"]) for p in MultipartUpload["Parts"]]
            if numbers != list(range(1, len(numbers) + 1)) or any(n not in parts for n in numbers):
                raise _client_error("InvalidPart", "CompleteMultipartUpload", 400)
            for p in MultipartUpload["Parts"]:
                if p["ETag"] != '"' + hashlib.md5(parts[int(p["PartNumber"])]).hexdigest() + '"':
                    raise _client_error("InvalidPart", "CompleteMultipartUpload", 400)
            del self.server.uploads[UploadId]
            self.server.objects[Key] = _Object(b"".join(parts[n] for n in numbers), self.server.clock.now)
            return {}

        return self._send("mpu_complete", Key, apply, mutates=True)

    def abort_multipart_upload(self, Bucket: str, Key: str, UploadId: str) -> dict:
        def apply() -> dict:
            self.server.uploads.pop(UploadId, None)
            return {}

        return self._send("mpu_abort", Key, apply, mutates=True)

    # -- listing ------------------------------------------------------------------
    def get_paginator(self, name: str) -> Any:
        assert name == "list_objects_v2"
        client = self

        class Paginator:
            def paginate(self, Bucket: str, Prefix: str = "", StartAfter: str = ""):
                after = StartAfter
                while True:
                    def apply(after: str = after) -> dict:
                        server = client.server
                        keys = sorted(k for k in server.objects if k.startswith(Prefix) and k > after)
                        page = keys[:server.page_size]
                        stamp = datetime.datetime.fromtimestamp(int(server.clock.now), datetime.timezone.utc)
                        out: dict[str, Any] = {
                            "IsTruncated": len(keys) > len(page),
                            "ResponseMetadata": {"HTTPHeaders": {"date": format_datetime(stamp, usegmt=True)}}}
                        if page:
                            out["Contents"] = [
                                {"Key": k, "Size": len(server.objects[k].data),
                                 "LastModified": datetime.datetime.fromtimestamp(server.objects[k].modified,
                                                                                 datetime.timezone.utc)}
                                for k in page]
                        return out

                    page = client._send("list", Prefix, apply)
                    yield page
                    if not page["IsTruncated"]:
                        return
                    after = page["Contents"][-1]["Key"]

        return Paginator()
