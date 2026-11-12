"""Cloud layer: S3 (and S3-compatible, e.g. MinIO) object API of an ``s3://`` checkpoint store.

An island mounts an ``s3://BUCKET[/PREFIX]`` store with sky Storage MOUNT
(goofys) for its cuts. The state that one host rewrites while another reads it
(sample buffer, trainer commands, policy release pointer) goes through the object
API instead (:mod:`yeto.rl.engine.object_store` says why). The launch layer
exports on the island:

* ``YETO_RL_STORE_IMPL=yeto.cloud.s3_store:store_from_env`` and
  :data:`URI_ENV` (the store URI): ``resume.store_for`` then builds an
  :class:`S3Store` (the bucket mount plus ``objects``);
* :data:`STAGE_ENV` on a one-node island: the store's ``root`` is then a local
  directory mirrored to the object store (``object_store.TreeMirror``): round
  cuts, snapshots and policy releases are written to local disk and uploaded by
  ``commit`` (pointer files last), and ``reload`` downloads the newest committed
  cut. The bucket mount carries nothing of the run then;
* the store credentials, :data:`CREDENTIAL_ENVS`, and optionally
  :data:`ENDPOINT_ENV` / :data:`REGION_ENV`. They are copied from the launcher's
  environment, never from its cloud credential files: an island must not hold
  cloud credentials (secret-handling-hardening), so the store credential is a
  separate one, limited to the store's bucket by its issuer. The island never
  falls back to the default AWS credential chain.

The head (launcher) reads and writes the same keys with :func:`head_objects`: the
store credentials when its environment has them, else its own AWS credentials.

Large files (policy releases of hundreds of GB) move in parts: a multipart
upload and parallel ranged downloads (:class:`TransferSettings`, from
:data:`PART_MB_ENV` / :data:`PART_CONCURRENCY_ENV`). Alibaba Cloud OSS reuses
this client through its S3-compatible API (``yeto.cloud.oss_store``).
"""

from __future__ import annotations

import functools
import os
import pickle
import subprocess
import sys
import random
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from yeto.rl.engine.object_store import DOWNLOAD_SUFFIX, ObjectStoreError, TreeMirror, check_key
from yeto.rl.engine.resume import STORE_IMPL_ENV, BucketMountStore

# The retryable failures of one streamed range read: the whole class of transfer
# errors, not one exception at a time (an R2 body cut short has surfaced as botocore
# ReadTimeoutError, ResponseStreamingError and urllib3 IncompleteRead so far).
import http.client as _http_client

RANGE_READ_RETRYABLE: tuple[type[BaseException], ...] = (
    ConnectionError, TimeoutError, _http_client.HTTPException)  # incl. ConnectionResetError, IncompleteRead
try:  # botocore ships with boto3
    from botocore.exceptions import (ConnectionError as _BotoConnectionError, HTTPClientError,
                                     IncompleteReadError, ResponseStreamingError)
    from urllib3.exceptions import HTTPError as _Urllib3Error  # ProtocolError, IncompleteRead, timeouts
    RANGE_READ_RETRYABLE += (_BotoConnectionError, HTTPClientError, IncompleteReadError,
                             ResponseStreamingError, _Urllib3Error)
except ImportError:  # pragma: no cover - a store without boto3 never reads
    pass
RANGE_READ_ATTEMPTS = 5
# A download file (``S3ObjectStore._partial``) is named by where its writer runs
# (:func:`_host_token`), its pid and the call. The next download of the same target
# removes it once it has no writer:
# * the same pid namespace and the pid is gone, or the pid is this process's but
#   the file is none of its downloads (an earlier process that had the pid):
#   after DEAD_PARTIAL_GRACE_S without a write;
# * the same pid namespace and the pid is alive: after STALE_PARTIAL_S without a
#   write (the pid went to another process);
# * a writer that cannot be asked about (another host or container, a host whose
#   pid namespace cannot be read, a name of another format): after
#   UNKNOWN_PARTIAL_S without a write.
# Never on the pid alone: a running download marks its file every
# PARTIAL_HEARTBEAT_S (one part can take minutes to arrive before it is written),
# so a file inside these windows may have a writer whatever its name says.
STALE_PARTIAL_S = 3600.0
UNKNOWN_PARTIAL_S = 600.0
DEAD_PARTIAL_GRACE_S = 120.0
PARTIAL_HEARTBEAT_S = 20.0
_LIVE_PARTIALS: set[str] = set()  # the download files this process is writing
_LIVE_PARTIALS_LOCK = threading.Lock()


@functools.lru_cache(maxsize=1)
def _pid_namespace() -> str | None:
    """What tells this pid namespace from every other one: the boot of the kernel
    and the namespace's inode (Linux). None when it cannot be read (macOS, no
    /proc): a pid in a file's name then proves nothing. The host name does not
    do: containers on the host network, or started with one fixed name, share it
    while each has its own pids, often the same ones."""
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        return f"{boot}:{os.stat('/proc/self/ns/pid').st_ino}"
    except (OSError, ValueError):
        return None


def _host_token() -> str:
    """Where this process runs, in a download file's name: a pid means something
    only next to it."""
    import hashlib
    import socket

    return hashlib.sha256(f"{socket.gethostname()}\n{_pid_namespace() or ''}".encode("utf-8", "replace")).hexdigest()[:8]


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:  # EPERM: a process of another user
        return True
    return True


def _read_retry_pause(attempt: int) -> None:
    """Wait before try ``attempt + 2`` of a body read: exponential, with jitter (16
    samplers cut off at once do not come back in lock step)."""
    time.sleep(random.uniform(0.5, 1.0) * min(30.0, 2.0 ** attempt))


# A range read the store refused for its request rate, after botocore's own retries
# (R2: ``503 ServiceUnavailable: Reduce your concurrent request rate for the same
# object``; S3: ``503 SlowDown``; a 429). The download then lowers the range reads
# this process runs at once and tries the range again (:class:`RangeGate`), instead
# of failing the whole file: 16 samplers that pull one release together are 8000
# concurrent reads, up to 256 on one file.
THROTTLE_CODES = frozenset({"ServiceUnavailable", "SlowDown", "TooManyRequests", "RequestLimitExceeded",
                            "Throttling", "ThrottlingException", "RequestThrottled", "503", "429"})
THROTTLE_STATUS = frozenset({429, 503})
THROTTLED_READ_ATTEMPTS = 6  # tries of one refused range (each one botocore's 9 requests)
# The two retry layers multiply: a range refused every time is 6 x 9 = 54 requests and
# waits at most 6 x 91 s (botocore) + 1 + 2 + 4 + 8 + 16 s (ours) = 577 s before the
# download fails; ours exist to lower the concurrency (RangeGate), not to retry more.
# A non-throttle error is never retried here (body errors: RANGE_READ_ATTEMPTS).
RANGE_RESTORE_READS = 8  # range reads without a refusal per read given back to the limit


def is_throttled(error: BaseException) -> bool:
    """``error`` is the store refusing a request for its rate (a botocore
    ``ClientError`` by code or HTTP status)."""
    response = getattr(error, "response", None)
    if not isinstance(response, Mapping):
        return False
    status = (response.get("ResponseMetadata") or {}).get("HTTPStatusCode")
    return error_code(error) in THROTTLE_CODES or status in THROTTLE_STATUS


class RangeGate:
    """How many range reads one process runs at once. Unlimited until the store
    refuses a read for its rate; each refusal halves the reads then in flight
    (never below one), and every :data:`RANGE_RESTORE_READS` reads without a
    refusal give one back (the limit is never dropped again: going back to
    every thread at once would ask for the next refusal). One gate per store
    object: every file and thread of a download in this process shares it."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self.limit: int | None = None  # None: no limit
        self.active = 0
        self.peak = 0  # the most reads in flight since the last refusal
        self.cuts = 0  # refusals that lowered the limit
        self._good = 0

    def __enter__(self) -> "RangeGate":
        with self._cond:
            while self.limit is not None and self.active >= self.limit:
                self._cond.wait()
            self.active += 1
            self.peak = max(self.peak, self.active)
        return self

    def __exit__(self, *exc: object) -> None:
        with self._cond:
            self.active -= 1
            self._cond.notify()

    def refused(self) -> int:
        """A read was refused: halve what ran at once; the new limit."""
        with self._cond:
            running = self.peak if self.limit is None else min(self.limit, self.peak)
            self.limit = max(1, running // 2)
            self.peak, self._good = self.active, 0
            self.cuts += 1
            return self.limit

    def succeeded(self) -> None:
        with self._cond:
            if self.limit is None:
                return
            self._good += 1
            if self._good >= RANGE_RESTORE_READS:
                self._good = 0
                self.limit += 1
                self._cond.notify()


# botocore's retries of one request. "standard" mode retries a 500 / 502 / 503 / 504
# and the throttling codes (SlowDown, ...) with full jitter: the wait before retry
# ``n`` is ``random() * min(2 ** (n - 1), 20)`` s, so 16 samplers refused at once by
# the store do not come back in lock step. ``max_attempts`` here counts retries (the
# client config's meaning): 8 retries are 9 requests and 8 waits, 45.5 s on average
# and 91 s at most (measured against an endpoint that always answers 503: mean 45.8 s,
# p95 65 s). A client also holds a retry quota (500 units, 5 per retry, refilled by
# successes): after about 100 retries without a success between them a refused
# request fails at once (measured: the 14th refused call in a row was not retried).
# "adaptive" mode adds a client-side rate limit, but only after a throttling *code*
# (SlowDown, Throttling, ...): R2's ``503 ServiceUnavailable: Reduce your concurrent
# request rate`` is not one, so it would never slow this client down (measured).
# :func:`client_config` reads the overrides.
RETRIES = {"mode": "standard", "max_attempts": 8}
RETRY_MODES = ("standard", "adaptive", "legacy")
# Seconds to open a connection / to wait for the next bytes of a response. botocore's
# 60 s connect timeout times 9 requests keeps one call to an unreachable endpoint
# busy for 9 minutes, longer than the dispenser's lease renewal period (lease / 3 =
# 300 s); a TCP + TLS handshake that has not finished in 10 s is retried instead.
CONNECT_TIMEOUT_S = 10.0
READ_TIMEOUT_S = 60.0  # botocore's default; a 64 MiB range read is many socket reads

STORE_IMPL = "yeto.cloud.s3_store:store_from_env"  # value of resume.STORE_IMPL_ENV
URI_ENV = "YETO_RL_STORE_S3_URI"
ENDPOINT_ENV = "YETO_RL_STORE_S3_ENDPOINT_URL"  # S3-compatible endpoint (MinIO); unset: AWS
REGION_ENV = "YETO_RL_STORE_S3_REGION"
KEY_ID_ENV = "YETO_RL_STORE_S3_ACCESS_KEY_ID"
SECRET_ENV = "YETO_RL_STORE_S3_SECRET_ACCESS_KEY"
TOKEN_ENV = "YETO_RL_STORE_S3_SESSION_TOKEN"
# a named AWS profile (~/.aws/config, ~/.aws/credentials) of a launcher-side client:
# its keys, endpoint_url and region; AWS_PROFILE is read too
PROFILE_ENV = "YETO_RL_STORE_S3_PROFILE"
STAGE_ENV = "YETO_RL_STORE_S3_STAGE"  # a local directory: cuts and releases by object API
PART_MB_ENV = "YETO_RL_STORE_PART_MB"  # multipart part size (MiB); shared by the S3 and OSS stores
PART_CONCURRENCY_ENV = "YETO_RL_STORE_PART_CONCURRENCY"  # parts of one file moved in parallel
FILE_CONCURRENCY_ENV = "YETO_RL_STORE_FILE_CONCURRENCY"  # files of one push tier moved in parallel
MAX_CONNECTIONS_ENV = "YETO_RL_STORE_MAX_CONNECTIONS"  # HTTP connection pool of the client (default: the threads)
DOWNLOAD_PROCESSES_ENV = "YETO_RL_STORE_DOWNLOAD_PROCESSES"  # processes of a many-file download (read_files)
CONNECT_TIMEOUT_ENV = "YETO_RL_STORE_CONNECT_TIMEOUT_S"  # seconds to open a connection (default 10)
READ_TIMEOUT_ENV = "YETO_RL_STORE_READ_TIMEOUT_S"  # seconds without a byte of a response (default 60)
RETRY_MODE_ENV = "YETO_RL_STORE_RETRY_MODE"  # botocore retry mode: standard (default) / adaptive / legacy
MAX_ATTEMPTS_ENV = "YETO_RL_STORE_MAX_ATTEMPTS"  # requests of one call, the first included (default 9)
PART_CHECKSUM_ENV = "YETO_RL_STORE_PART_CHECKSUM"  # SHA-256 per uploaded part: 1 / 0 / auto (default)
UPLOAD_ATTEMPTS_ENV = "YETO_RL_STORE_UPLOAD_ATTEMPTS"  # tries of one multipart upload, resumed between them
CREDENTIAL_ENVS = (KEY_ID_ENV, SECRET_ENV, TOKEN_ENV)
SECRET_ENVS = frozenset({SECRET_ENV, TOKEN_ENV})  # shipped as task secrets, not plain envs
SETTING_ENVS = (ENDPOINT_ENV, REGION_ENV, PART_MB_ENV, PART_CONCURRENCY_ENV, FILE_CONCURRENCY_ENV,
                MAX_CONNECTIONS_ENV, DOWNLOAD_PROCESSES_ENV, PART_CHECKSUM_ENV, UPLOAD_ATTEMPTS_ENV,
                CONNECT_TIMEOUT_ENV, READ_TIMEOUT_ENV, RETRY_MODE_ENV, MAX_ATTEMPTS_ENV)
DEFAULT_STAGE = "~/.cache/yeto/store-stage"
# what a staged store pulls at start (the newest committed cut: LATEST, its snapshot, the
# cuts) and the pointer files it uploads last (resume.LATEST, policy_release.LATEST)
STAGED_SUBTREES = ("LATEST", "state/", "round-cuts/")
POINTER_FILES = frozenset({"LATEST", "LATEST.json"})
# record files: uploaded after the data files next to them, before the pointers
# (policy_release.RECORD lists the release's weight files)
RECORD_FILES = frozenset({"RELEASE.json"})
# top-level directories their writer commits by path (policy releases of every scheme,
# policy_release.PolicyReleaseStore): a cut's commit neither waits for nor uploads them
INDEPENDENT_DIRS = ("policy-releases",)

# error codes of a conditional write that lost to an existing object (S3; OSS
# answers FileAlreadyExists to x-oss-forbid-overwrite)
_CREATE_LOST = frozenset({"PreconditionFailed", "ConditionalRequestConflict", "FileAlreadyExists"})
_MISSING = frozenset({"404", "NoSuchKey", "NotFound"})
_MIB = 1 << 20
MIN_PART_BYTES = 5 * _MIB  # S3 and OSS: every part but the last is at least 5 MiB
MAX_PARTS = 10_000  # S3 and OSS: parts of one multipart upload
# the resume journal of a multipart upload, next to its source file (a stage push
# skips it: object_store.TMP_SUFFIX); removed when the upload completes
UPLOAD_JOURNAL_SUFFIX = ".upload.tmp"
_FALSE = frozenset({"0", "false", "no", "off"})
_TRUE = frozenset({"1", "true", "yes", "on"})
# endpoints known to refuse x-amz-checksum-sha256 on UploadPart (NotImplemented):
# PART_CHECKSUM_ENV=auto sends no part checksums to them
NO_PART_CHECKSUM_HOSTS = (".r2.cloudflarestorage.com",)
_PART_CHECKSUM_WARNED = False


def part_checksum_default(environ: Mapping[str, str]) -> bool:
    """:data:`PART_CHECKSUM_ENV`: ``1`` sends a SHA-256 with every part, ``0`` none;
    unset or ``auto`` sends them except to an endpoint known not to take them
    (:data:`NO_PART_CHECKSUM_HOSTS`, Cloudflare R2). An endpoint that still answers
    ``NotImplemented`` turns them off for that store (:meth:`S3ObjectStore.write_file`)."""
    raw = str(environ.get(PART_CHECKSUM_ENV) or "auto").strip().lower()
    if raw in _FALSE:
        return False
    if raw in _TRUE:
        return True
    if raw != "auto":
        raise ValueError(f"{PART_CHECKSUM_ENV}={raw!r}: expected 1, 0 or auto")
    return not _is_r2(store_endpoint(environ))


def _warn_part_checksums_off(uri: str, error: Exception) -> None:
    global _PART_CHECKSUM_WARNED
    if _PART_CHECKSUM_WARNED:
        return
    _PART_CHECKSUM_WARNED = True
    print(f"yeto: warning: {uri} does not take SHA-256 part checksums ({error.__class__.__name__}: "
          f"NotImplemented); uploading without them (set {PART_CHECKSUM_ENV}=0 to skip this probe)",
          file=sys.stderr, flush=True)


def error_code(error: BaseException) -> str:
    """The store's error code of a botocore ``ClientError``, ``""`` for anything else.
    A connection-level error (botocore ``HTTPClientError``: ``ReadTimeoutError``,
    ``ConnectionClosedError``) has ``response = None``, not a dict: it has no code,
    and asking for one must not replace it with an ``AttributeError``."""
    response = getattr(error, "response", None)
    if not isinstance(response, Mapping):
        return ""
    return str((response.get("Error") or {}).get("Code", ""))


def _b64(digest: bytes) -> str:
    import base64

    return base64.b64encode(digest).decode("ascii")


def multipart_etag(part_etags: Sequence[Any]) -> str | None:
    """The ETag S3 and R2 report for a multipart object whose parts have
    ``part_etags`` (in part order; each the MD5 of its part, as UploadPart answers
    without SSE-KMS / SSE-C): the MD5 of the concatenated raw part MD5s, then
    ``-<parts>`` (measured on R2, a one-part upload included). Unquoted; None when
    a part ETag is not an MD5, so nothing can be derived."""
    import hashlib

    raw = b""
    for etag in part_etags:
        text = str(etag or "").strip('"')
        if len(text) != 32:
            return None
        try:
            raw += bytes.fromhex(text)
        except ValueError:
            return None
    return f"{hashlib.md5(raw).hexdigest()}-{len(part_etags)}" if part_etags else None


def composite_sha256(part_digests: Sequence[str]) -> str:
    """The checksum S3 reports for a multipart object uploaded with SHA-256 part
    checksums (a COMPOSITE checksum): base64 SHA-256 of the concatenated raw part
    digests, then ``-<parts>``. ``part_digests``: base64, in part order."""
    import base64
    import hashlib

    raw = b"".join(base64.b64decode(d) for d in part_digests)
    return f"{_b64(hashlib.sha256(raw).digest())}-{len(part_digests)}"


class _UploadJournal:
    """What a multipart upload of one file has sent: its UploadId and, per part
    number, the ETag and SHA-256 (base64) the store acknowledged. Saved atomically
    after each part, next to the source (:data:`UPLOAD_JOURNAL_SUFFIX`), so an upload
    cut short (a failed part, a killed process) continues where it stopped. It
    belongs to one version of the source: a different key, size, mtime, part size or
    checksum mode makes it stale."""

    def __init__(self, source: str | os.PathLike[str], identity: dict[str, Any]) -> None:
        import threading

        self.path = Path(str(source) + UPLOAD_JOURNAL_SUFFIX)
        self.identity = identity
        self.upload_id: str | None = None
        self.parts: dict[int, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.writable = True

    def load(self) -> dict[str, Any] | None:
        """The journal on disk: its contents when it belongs to this upload, a dict
        with only ``stale_upload_id`` when it belongs to another, None when absent."""
        import json

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        if data.get("identity") != self.identity:
            return {"stale_upload_id": data.get("upload_id")}
        return data

    def save(self) -> None:
        import json

        if not self.writable:
            return
        data = {"identity": self.identity, "upload_id": self.upload_id,
                "parts": {str(n): v for n, v in sorted(self.parts.items())}}
        tmp = self.path.with_name(self.path.name.removesuffix(".tmp") + ".w.tmp")  # skipped by a push too
        try:
            tmp.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:  # a read-only source directory: the upload goes on without resume
            self.writable = False

    def record(self, number: int, part: dict[str, Any]) -> None:
        with self._lock:
            self.parts[int(number)] = part
            self.save()

    def remove(self) -> None:
        self.path.unlink(missing_ok=True)


@dataclass(frozen=True)
class TransferSettings:
    """How large files move: a file larger than one part is uploaded in parts of
    ``part_bytes`` (raised so a file needs at most :data:`MAX_PARTS`) and
    downloaded by ranges of the same size; ``concurrency`` parts of a file move at
    once, and a staged store's push moves ``file_concurrency`` files at once
    (memory: up to ``file_concurrency * concurrency`` parts).

    Defaults from an R2 upload measurement (GPU-A, 2 GiB files, 64 MiB parts, one
    16-core container): one file tops out near 100 MB/s whatever its part
    concurrency (76 / 109 / 102 MB/s at 8 / 32 / 64 parts); 4 and 8 files of 16
    parts each reach 255 and 273 MB/s.

    ``download_processes``: a many-file download (:meth:`S3ObjectStore.read_files`,
    a policy release) splits its files over that many processes. One Python process
    tops out near 700 MB/s of R2 range reads (MI355X node, 64 threads, nothing
    written); 4 / 8 processes of 16 threads read 1.7 / 3.0 GB/s. 4 stays well under
    the ~7 GB/s that dropped an MI355X node off the network (GPU-B section 4.1)."""

    part_bytes: int = 64 * _MIB
    concurrency: int = 16
    file_concurrency: int = 8
    download_processes: int = 4
    # SHA-256 per uploaded part: the store refuses a part whose body does not match
    # it, and the object's composite checksum is checked after the upload
    part_checksum: bool = True
    # tries of one multipart upload in one call; each continues the previous one
    # (list_parts), and after the last the upload is kept for the next call
    upload_attempts: int = 3

    def __post_init__(self) -> None:
        if self.part_bytes < MIN_PART_BYTES:
            raise ValueError(f"part size {self.part_bytes} B is below the {MIN_PART_BYTES} B minimum")
        if self.concurrency < 1:
            raise ValueError("part concurrency must be >= 1")
        if self.file_concurrency < 1:
            raise ValueError("file concurrency must be >= 1")
        if self.download_processes < 1:
            raise ValueError("download processes must be >= 1")
        if self.upload_attempts < 1:
            raise ValueError("upload attempts must be >= 1")

    def part_for(self, size: int) -> int:
        return max(self.part_bytes, -(-int(size) // MAX_PARTS))

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> "TransferSettings":
        default = cls()
        return cls(part_bytes=int(float(environ.get(PART_MB_ENV) or default.part_bytes / _MIB) * _MIB),
                   concurrency=int(environ.get(PART_CONCURRENCY_ENV) or default.concurrency),
                   file_concurrency=int(environ.get(FILE_CONCURRENCY_ENV) or default.file_concurrency),
                   download_processes=int(environ.get(DOWNLOAD_PROCESSES_ENV) or default.download_processes),
                   part_checksum=part_checksum_default(environ),
                   upload_attempts=int(environ.get(UPLOAD_ATTEMPTS_ENV) or default.upload_attempts))


def parse_s3_uri(uri: str, scheme: str = "s3") -> tuple[str, str]:
    """``<scheme>://BUCKET[/PREFIX]`` -> (bucket, prefix). Raises ValueError."""
    got, _, rest = str(uri).partition("://")
    bucket, _, prefix = rest.strip("/").partition("/")
    if got != scheme or not bucket:
        raise ValueError(f"{uri!r} is not a store URI {scheme}://BUCKET[/PREFIX]")
    return bucket, prefix.strip("/")


# The store credential as a Modal Secret carries it (the AWS names;
# yeto.launch.storage.modal_store_secret_name). An island reads that pair only when
# the launch says so (:data:`CREDENTIALS_ENV` = :data:`CREDENTIALS_AWS_ENV`): the
# launcher's own AWS_* variables are never a store credential (secret-handling
# hardening), and the default credential chain stays unused either way.
CREDENTIALS_ENV = "YETO_RL_STORE_S3_CREDENTIALS"
CREDENTIALS_AWS_ENV = "aws-env"  # read AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (a Modal Secret)
AWS_KEY_ID_ENV = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ENV = "AWS_SECRET_ACCESS_KEY"
AWS_TOKEN_ENV = "AWS_SESSION_TOKEN"
AWS_ENDPOINT_ENV = "AWS_ENDPOINT_URL"
AWS_REGION_ENVS = ("AWS_REGION", "AWS_DEFAULT_REGION")


def store_credentials(environ: Mapping[str, str]) -> tuple[str, str, str | None] | None:
    """``(key id, secret, session token or None)``: the store credential
    (:data:`KEY_ID_ENV` / :data:`SECRET_ENV`); with :data:`CREDENTIALS_ENV` set to
    :data:`CREDENTIALS_AWS_ENV` the AWS-named pair a Secret put in the environment.
    None when no complete pair applies."""
    if environ.get(KEY_ID_ENV) and environ.get(SECRET_ENV):
        return environ[KEY_ID_ENV], environ[SECRET_ENV], environ.get(TOKEN_ENV) or None
    if (environ.get(CREDENTIALS_ENV) == CREDENTIALS_AWS_ENV and environ.get(AWS_KEY_ID_ENV)
            and environ.get(AWS_SECRET_ENV)):
        return environ[AWS_KEY_ID_ENV], environ[AWS_SECRET_ENV], environ.get(AWS_TOKEN_ENV) or None
    return None


def missing_credentials_message() -> str:
    return (f"s3 store: no store credentials. Set {KEY_ID_ENV} and {SECRET_ENV} (a key limited to the "
            f"store's bucket), or {PROFILE_ENV} (or --profile) to an AWS profile that holds them; on "
            f"an island with a Modal Secret, {CREDENTIALS_ENV}={CREDENTIALS_AWS_ENV} reads "
            f"{AWS_KEY_ID_ENV} and {AWS_SECRET_ENV}; an island uses the store credentials only, never the default AWS chain")


def has_store_credentials(environ: Mapping[str, str]) -> bool:
    return store_credentials(environ) is not None


AWS_PROFILE_ENV = "AWS_PROFILE"
AWS_ENDPOINT_S3_ENV = "AWS_ENDPOINT_URL_S3"


def store_profile(environ: Mapping[str, str]) -> str | None:
    """The AWS profile a store client reads: :data:`PROFILE_ENV`, else ``AWS_PROFILE``."""
    return environ.get(PROFILE_ENV) or environ.get(AWS_PROFILE_ENV) or None


@functools.lru_cache(maxsize=16)
def _profile_config_cached(profile: str, config_file: str | None, credentials_file: str | None) -> dict[str, Any]:
    import botocore.session

    try:
        config = dict(botocore.session.Session(profile=profile).get_scoped_config())
    except Exception as error:  # noqa: BLE001 - botocore ProfileNotFound
        if type(error).__name__ == "ProfileNotFound":
            return {"_missing": True}
        raise
    s3 = config.get("s3") if isinstance(config.get("s3"), dict) else {}
    return {"endpoint_url": s3.get("endpoint_url") or config.get("endpoint_url"), "region": config.get("region")}


def profile_config(environ: Mapping[str, str]) -> dict[str, Any]:
    """``endpoint_url`` and ``region`` of the store's AWS profile (:func:`store_profile`;
    the ``s3`` section's endpoint first), ``{}`` without a profile, ``{"_missing": True}``
    when the profile is not configured on this host (an island)."""
    profile = store_profile(environ)
    if not profile:
        return {}
    try:
        return _profile_config_cached(profile, environ.get("AWS_CONFIG_FILE"),
                                      environ.get("AWS_SHARED_CREDENTIALS_FILE"))
    except ImportError:  # no botocore: nothing to read
        return {}


def store_endpoint(environ: Mapping[str, str]) -> str | None:
    """The S3-compatible endpoint: :data:`ENDPOINT_ENV`, else ``AWS_ENDPOINT_URL_S3`` /
    ``AWS_ENDPOINT_URL``, else the ``endpoint_url`` of the store's AWS profile; None
    for AWS S3."""
    return (environ.get(ENDPOINT_ENV) or environ.get(AWS_ENDPOINT_S3_ENV) or environ.get(AWS_ENDPOINT_ENV)
            or profile_config(environ).get("endpoint_url") or None)


def _is_r2(endpoint: str | None) -> bool:
    from urllib.parse import urlparse

    host = (urlparse(endpoint).hostname or "") if endpoint else ""
    return any(host.endswith(suffix) for suffix in NO_PART_CHECKSUM_HOSTS)


def store_region(environ: Mapping[str, str]) -> str | None:
    """:data:`REGION_ENV`, else the AWS region variables, else the store profile's
    region, else ``auto`` for a Cloudflare R2 endpoint; None otherwise (boto3's
    default). ``yeto.launch.storage.store_client_environ`` sets ``auto`` for an
    endpoint given by ``--rl-object-store-endpoint`` (what R2 expects)."""
    for name in (REGION_ENV, *AWS_REGION_ENVS):
        if environ.get(name):
            return environ[name]
    region = profile_config(environ).get("region")
    if region:
        return str(region)
    return "auto" if _is_r2(store_endpoint(environ)) else None


def _read_files_here(objects: "S3ObjectStore", items: Sequence[tuple[str, str]], threads: int) -> list[str]:
    def one(item: tuple[str, str]) -> str | None:
        return None if objects.read_file(*item) else item[0]

    with ThreadPoolExecutor(max(1, min(int(threads), len(items) or 1))) as pool:
        return [key for key in pool.map(one, items) if key is not None]


def _read_files_child() -> None:
    """One child process of :meth:`S3ObjectStore.read_files`: its own client; the
    arguments come pickled on stdin (from the parent only), the missing keys go back
    pickled on stdout."""
    uri, client_factory, transfer, threads, items = pickle.loads(sys.stdin.buffer.read())
    objects = S3ObjectStore(uri, client=client_factory(), transfer=TransferSettings(**transfer))
    missing = _read_files_here(objects, items, threads)
    sys.stdout.buffer.write(pickle.dumps(missing))
    sys.stdout.buffer.flush()


READ_FILES_CHILD = "from yeto.cloud.s3_store import _read_files_child; _read_files_child()"


BOTOCORE_DEFAULT_POOL = 10  # botocore's max_pool_connections


def client_pool_size(environ: Mapping[str, str]) -> int:
    """HTTP connections of the store's boto3 client: :data:`MAX_CONNECTIONS_ENV`, else
    one per transfer thread (``file_concurrency * concurrency``). botocore's default
    pool of 10 serialized the 8 x 16 range reads of a release download (MI355X: a
    207 GB Kimi-K3.1 release far below the node's R2 bandwidth)."""
    raw = environ.get(MAX_CONNECTIONS_ENV)
    if raw:
        return max(1, int(raw))
    t = TransferSettings.from_env(environ)
    return max(BOTOCORE_DEFAULT_POOL, t.file_concurrency * t.concurrency)


def _seconds(environ: Mapping[str, str], name: str, default: float) -> float:
    raw = environ.get(name)
    value = float(raw) if raw else default
    if value <= 0:
        raise ValueError(f"{name}={raw!r}: expected seconds > 0")
    return value


def client_config(environ: Mapping[str, str]) -> dict[str, Any]:
    """The botocore ``Config`` arguments of the store's client: the connection pool
    (:func:`client_pool_size`), the timeouts (:data:`CONNECT_TIMEOUT_ENV` /
    :data:`READ_TIMEOUT_ENV`) and the retries (:data:`RETRIES`).

    Retry mode: :data:`RETRY_MODE_ENV`, else ``AWS_RETRY_MODE``, else "standard".
    Requests of one call (the first included): :data:`MAX_ATTEMPTS_ENV`, else
    ``AWS_MAX_ATTEMPTS``, else 9. Each is read on its own: ``AWS_MAX_ATTEMPTS`` alone
    keeps "standard" mode (botocore's own default mode is "legacy")."""
    mode = environ.get(RETRY_MODE_ENV) or environ.get("AWS_RETRY_MODE") or RETRIES["mode"]
    if mode not in RETRY_MODES:
        raise ValueError(f"retry mode {mode!r} ({RETRY_MODE_ENV} / AWS_RETRY_MODE): expected one of {RETRY_MODES}")
    raw = environ.get(MAX_ATTEMPTS_ENV) or environ.get("AWS_MAX_ATTEMPTS")
    attempts = int(raw) if raw else int(RETRIES["max_attempts"]) + 1
    if attempts < 1:
        raise ValueError(f"{MAX_ATTEMPTS_ENV} / AWS_MAX_ATTEMPTS={raw!r}: expected an integer >= 1")
    return {"max_pool_connections": client_pool_size(environ),
            "connect_timeout": _seconds(environ, CONNECT_TIMEOUT_ENV, CONNECT_TIMEOUT_S),
            "read_timeout": _seconds(environ, READ_TIMEOUT_ENV, READ_TIMEOUT_S),
            "retries": {"mode": mode, "total_max_attempts": attempts}}


def _client(environ: Mapping[str, str], *, require_credentials: bool) -> Any:
    import boto3
    from botocore.config import Config

    creds = store_credentials(environ)
    profile = store_profile(environ)
    if profile and profile_config(environ).get("_missing"):
        if environ.get(PROFILE_ENV) or (creds is None and require_credentials):
            raise ObjectStoreError(f"s3 store: AWS profile {profile!r} ({PROFILE_ENV if environ.get(PROFILE_ENV) else AWS_PROFILE_ENV}) "
                                   "is not configured on this host (~/.aws/config, ~/.aws/credentials)")
        profile = None  # AWS_PROFILE of another host: not this store's
    if creds is not None:
        key, secret, token = creds
        session = boto3.session.Session(aws_access_key_id=key, aws_secret_access_key=secret,
                                        aws_session_token=token)
    elif profile:
        session = boto3.session.Session(profile_name=profile)
    elif require_credentials:
        raise ObjectStoreError(missing_credentials_message())
    else:
        session = boto3.session.Session()
    return session.client("s3", endpoint_url=store_endpoint(environ), region_name=store_region(environ),
                          config=Config(**client_config(environ)))


class S3ObjectStore:
    """:class:`~yeto.rl.engine.object_store.ObjectStore` over ``s3://BUCKET[/PREFIX]``:
    a key is ``PREFIX/<key>``. ``client``: a boto3 S3 client (or a test double).

    ``create`` is a conditional put (``If-None-Match: *``). An endpoint that does
    not implement it answers ``NotImplemented``; the store then checks for the key
    and puts, which is safe only while one writer appends (the sample buffer's
    current limit: one sampler island). A refused create that the SDK had sent
    more than once may be this writer's own (:meth:`_created_by_retry`).

    ``buffer_shards``: the sample buffer must not rely on ``create`` between
    writers on several hosts; each sampler island then writes its own shard
    (``sample_buffer`` module doc). False here: S3 has conditional writes."""

    scheme = "s3"
    buffer_shards = False
    # the endpoint takes x-amz-checksum-sha256 on parts and reports a composite
    # checksum (HEAD, ChecksumMode=ENABLED); OSS does not
    part_checksums = True

    def __init__(self, uri: str, *, client: Any, transfer: TransferSettings | None = None,
                 client_factory: Callable[[], Any] | None = None) -> None:
        """``client_factory``: a picklable callable that builds a client like ``client``
        in another process (:meth:`read_files`); None keeps every read in this process."""
        self.bucket, self.prefix = parse_s3_uri(uri, self.scheme)
        self.uri = f"{self.scheme}://{self.bucket}" + (f"/{self.prefix}" if self.prefix else "")
        self.client = client
        self.transfer = transfer or TransferSettings()
        self.client_factory = client_factory
        self.range_gate = RangeGate()  # range reads of this process, lowered when the store refuses
        self.conditional_writes = True
        # set when the endpoint answered a part checksum with NotImplemented
        self.part_checksums_refused = False

    def _full(self, name: str) -> str:
        return f"{self.prefix}/{name}" if self.prefix else name

    def _key(self, key: str) -> str:
        return self._full(check_key(key))

    def _code(self, error: Exception) -> str:
        return error_code(error)

    def read(self, key: str) -> bytes | None:
        """The object's bytes, None when it is missing. A body cut short (a reset
        connection, a stalled stream) is read again, up to
        :data:`RANGE_READ_ATTEMPTS` times: botocore retries the request, never the
        body it streams afterwards, and one reset while the trainer reads a payload
        or a sampler its lease would otherwise end the caller. Each try is one GET
        of one version: an object replaced meanwhile is read as it is now, one
        deleted meanwhile is missing. An error answer (403, a 503 after the SDK's
        retries) is raised as before."""
        full = self._key(key)
        for attempt in range(RANGE_READ_ATTEMPTS):
            try:
                response = self.client.get_object(Bucket=self.bucket, Key=full)
            except self.client.exceptions.NoSuchKey:
                return None
            try:
                return response["Body"].read()
            except RANGE_READ_RETRYABLE:
                if attempt + 1 == RANGE_READ_ATTEMPTS:
                    raise
                _read_retry_pause(attempt)
        raise AssertionError("unreachable")  # pragma: no cover

    def write(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=self._key(key), Body=data)

    def create(self, key: str, data: bytes) -> bool:
        if self.conditional_writes:
            try:
                self.client.put_object(Bucket=self.bucket, Key=self._key(key), Body=data, IfNoneMatch="*")
                return True
            except Exception as error:  # noqa: BLE001 - botocore ClientError, by code
                code = self._code(error)
                if code in _CREATE_LOST:
                    return self._created_by_retry(error, key, data)
                if code != "NotImplemented":
                    raise
                self.conditional_writes = False
        if self.exists(key):
            return False
        self.write(key, data)
        return True

    def _created_by_retry(self, error: Exception, key: str, data: bytes) -> bool:
        """Whether a conditional put refused as "the key exists" created the key itself.

        The SDK sends a request again when its answer does not arrive (a read
        timeout, a reset connection, a 5xx after the store applied the put). The
        first send then created the object and the retry is refused, like a create
        that lost to another writer (measured on R2: ``PreconditionFailed`` with
        ``RetryAttempts`` 1, the object holding this writer's bytes). Taken for a
        loss, a buffer writer appended its record a second time at the next
        position and a dispenser claim stayed held by a sampler that did not know
        it. So a refusal that follows a retry reads the object back: the same bytes
        are this create (every record of this store names its writer or is the same
        for every writer). A refusal on the first send costs no extra request."""
        meta = getattr(error, "response", {}).get("ResponseMetadata") or {}
        if not int(meta.get("RetryAttempts") or 0):
            return False
        return self.read(key) == bytes(data)

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except Exception as error:  # noqa: BLE001 - head_object reports a missing key as 404
            if self._code(error) in _MISSING:
                return False
            raise

    def _head(self, key: str) -> dict[str, Any] | None:
        try:
            return self.client.head_object(Bucket=self.bucket, Key=self._key(key))
        except Exception as error:  # noqa: BLE001 - botocore ClientError, by code
            if self._code(error) in _MISSING:
                return None
            raise

    def size(self, key: str) -> int | None:
        head = self._head(key)
        return None if head is None else int(head["ContentLength"])

    def _checksums(self) -> bool:
        return bool(self.part_checksums and self.transfer.part_checksum and not self.part_checksums_refused)

    def checksum(self, key: str) -> str | None:
        """The SHA-256 checksum the store keeps for object ``key`` (base64; a
        multipart object's is composite, ``<b64>-<parts>``), None when the object is
        missing or has none (uploaded without checksums, an endpoint without them)."""
        if not self._checksums():
            return None
        try:
            head = self.client.head_object(Bucket=self.bucket, Key=self._key(key), ChecksumMode="ENABLED")
        except Exception as error:  # noqa: BLE001 - botocore ClientError, by code
            if self._code(error) in _MISSING:
                return None
            raise
        value = head.get("ChecksumSHA256")
        return str(value) if value else None

    def write_file(self, key: str, source: str | os.PathLike[str]) -> str | None:
        """Upload a file; one larger than a part goes as a multipart upload whose
        parts are read from the file as they are sent (never the whole file in
        memory). Returns the object's SHA-256 checksum (:meth:`checksum`), None
        without checksums.

        Integrity: each part carries its SHA-256 (the store refuses a part whose body
        does not match), and the composite checksum the store reports afterwards must
        equal the one of the parts sent (:func:`composite_sha256`).

        Resume: the UploadId and every acknowledged part are journaled next to the
        source (:class:`_UploadJournal`). A failed try continues from the parts
        ``list_parts`` and the journal agree on (same ETag, same checksum), up to
        ``upload_attempts`` tries; after the last the upload and its journal are kept,
        so a later call (the next push, a restarted process) continues it. A journal
        of another version of the file aborts its upload and starts again.

        A CompleteMultipartUpload that the store applied but did not answer (the
        SDK's retry, or the next try, then finds ``NoSuchUpload``): when every part
        is journaled and the upload is gone, the object under the key is compared
        with the journal (:meth:`_completed`) and, when it is that upload, the file
        is not sent again (a 100-200 GB release file otherwise goes up twice).

        An endpoint that answers a checksum with ``NotImplemented`` (Cloudflare R2
        on UploadPart) turns checksums off for this store, with one warning per
        process, and the file is uploaded again without them."""
        try:
            return self._write_file(key, source)
        except Exception as error:  # noqa: BLE001 - botocore ClientError, by code
            if not self._checksums() or self._code(error) != "NotImplemented":
                raise
            self.part_checksums_refused = True
            _warn_part_checksums_off(self.uri, error)
            return self._write_file(key, source)

    def _write_file(self, key: str, source: str | os.PathLike[str]) -> str | None:
        import hashlib

        checks = self._checksums()
        size = os.path.getsize(source)
        part = self.transfer.part_for(size)
        full = self._key(key)
        if size <= part:
            with open(source, "rb") as fh:
                body = fh.read()
            extra = {"ChecksumSHA256": _b64(hashlib.sha256(body).digest())} if checks else {}
            self.client.put_object(Bucket=self.bucket, Key=full, Body=body, **extra)
            return extra.get("ChecksumSHA256")
        count = -(-size // part)
        st = os.stat(source)
        journal = _UploadJournal(source, {"bucket": self.bucket, "key": full, "size": int(size),
                                          "mtime_ns": int(st.st_mtime_ns), "part_bytes": int(part),
                                          "checksum": checks})
        completed = self._open_upload(journal, full, part, size, checks, count=count)
        for attempt in range(1, 0 if completed else self.transfer.upload_attempts + 1):
            try:
                self._send_parts(journal, full, source, part, size, count, checks)
                parts = [{"PartNumber": n, "ETag": journal.parts[n]["ETag"],
                          **({"ChecksumSHA256": journal.parts[n]["sha256"]} if checks else {})}
                         for n in range(1, count + 1)]
                self.client.complete_multipart_upload(Bucket=self.bucket, Key=full, UploadId=journal.upload_id,
                                                      MultipartUpload={"Parts": parts})
                break
            except Exception as error:  # noqa: BLE001 - botocore ClientError, by code
                if checks and self._code(error) == "NotImplemented":
                    try:  # the endpoint takes no part checksums: write_file starts again without
                        self.client.abort_multipart_upload(Bucket=self.bucket, Key=full,
                                                           UploadId=journal.upload_id)
                    except Exception:  # noqa: BLE001 - already gone
                        pass
                    journal.remove()
                    raise
                last = attempt == self.transfer.upload_attempts
                if last and len(journal.parts) < count:
                    raise  # the upload and its journal stay: the next call continues them
                try:
                    listed = self._listed_parts(full, journal.upload_id, part, size)
                    if listed is None and self._completed(full, journal.parts, count, size, checks):
                        break  # the complete took effect and its answer was lost
                except Exception:  # noqa: BLE001 - the store is still failing
                    if last:
                        raise error from None
                    raise
                if last:
                    raise  # as above; the next call looks at the object again when the upload is gone
                if listed is None:  # the upload is gone (completed elsewhere, expired, aborted)
                    journal.upload_id, journal.parts = None, {}
                    self._open_upload(journal, full, part, size, checks, fresh=True)
                else:
                    journal.parts = self._agreed(journal.parts, listed, checks)
                    journal.save()
        journal.remove()
        if not checks:
            return None
        expected = composite_sha256([journal.parts[n]["sha256"] for n in range(1, count + 1)])
        reported = self.checksum(key)
        if reported is not None and reported != expected:
            raise ObjectStoreError(f"{key}: the store reports checksum {reported}, the parts sent make {expected}")
        return expected

    def _open_upload(self, journal: _UploadJournal, full: str, part: int, size: int, checks: bool, *,
                     fresh: bool = False, count: int = 0) -> bool:
        """Continue the journaled upload of this file version, or create one. True
        when the journaled upload is gone and the object is what it completed
        (:meth:`_completed`; a process killed, or out of tries, between the complete
        and its answer): nothing is left to send."""
        state = None if fresh else journal.load()
        if state is not None and state.get("upload_id") and "stale_upload_id" not in state:
            listed = self._listed_parts(full, str(state["upload_id"]), part, size)
            saved = {int(n): v for n, v in dict(state.get("parts") or {}).items()}
            if listed is not None:
                journal.upload_id = str(state["upload_id"])
                journal.parts = self._agreed(saved, listed, checks)
                journal.save()
                return False
            if count and self._completed(full, saved, count, size, checks):
                journal.upload_id, journal.parts = str(state["upload_id"]), saved
                return True
        stale = (state or {}).get("stale_upload_id")
        if stale:
            try:  # another version of the file: its parts are never completed
                self.client.abort_multipart_upload(Bucket=self.bucket, Key=full, UploadId=str(stale))
            except Exception:  # noqa: BLE001 - already gone
                pass
        extra = {"ChecksumAlgorithm": "SHA256"} if checks else {}
        journal.upload_id = self.client.create_multipart_upload(Bucket=self.bucket, Key=full, **extra)["UploadId"]
        journal.parts = {}
        journal.save()
        return False

    def _completed(self, full: str, parts: Mapping[int, Mapping[str, Any]], count: int, size: int,
                   checks: bool) -> bool:
        """Whether the object under ``full`` is the multipart upload of the journaled
        ``parts``, for an upload the store does not know any more: a complete that
        took effect answers the next one (the SDK's own retry included) with
        ``NoSuchUpload`` (measured on R2), like an aborted or expired upload.

        Proof, not a guess: every part 1..``count`` is journaled, the object has the
        file's size, and it carries what only these parts produce: the composite
        SHA-256 of their checksums when the store reports one, else the multipart
        ETag of their ETags (:func:`multipart_etag`; R2 takes no part checksums).
        Anything else (no object, another size, part ETags that are not MD5s, an
        endpoint with another ETag scheme such as OSS) is False: the file is
        uploaded again, as before."""
        if set(parts) != set(range(1, count + 1)):
            return False
        try:
            head = self.client.head_object(Bucket=self.bucket, Key=full,
                                           **({"ChecksumMode": "ENABLED"} if checks else {}))
        except Exception as error:  # noqa: BLE001 - botocore ClientError, by code
            if self._code(error) in _MISSING:
                return False
            raise
        if int(head.get("ContentLength", -1)) != int(size):
            return False
        ordered = [parts[n] for n in range(1, count + 1)]
        reported = head.get("ChecksumSHA256")
        if checks and reported and all(p.get("sha256") for p in ordered):
            return str(reported) == composite_sha256([p["sha256"] for p in ordered])
        expected = multipart_etag([p.get("ETag") for p in ordered])
        return expected is not None and str(head.get("ETag") or "").strip('"') == expected

    def _send_parts(self, journal: _UploadJournal, full: str, source: str | os.PathLike[str], part: int,
                    size: int, count: int, checks: bool) -> None:
        import hashlib

        missing = [n for n in range(1, count + 1) if n not in journal.parts]
        if not missing:
            return
        with open(source, "rb") as fh:
            def send(number: int) -> None:
                offset = (number - 1) * part
                body = os.pread(fh.fileno(), min(part, size - offset), offset)
                digest = _b64(hashlib.sha256(body).digest()) if checks else None
                extra = {"ChecksumSHA256": digest} if checks else {}
                done = self.client.upload_part(Bucket=self.bucket, Key=full, UploadId=journal.upload_id,
                                               PartNumber=number, Body=body, **extra)
                journal.record(number, {"ETag": done["ETag"], "sha256": digest, "size": len(body)})

            with ThreadPoolExecutor(min(self.transfer.concurrency, len(missing))) as pool:
                for future in [pool.submit(send, n) for n in missing]:
                    future.result()

    def _listed_parts(self, full: str, upload_id: str, part: int, size: int) -> dict[int, dict[str, Any]] | None:
        """The parts the store holds for ``upload_id`` (number -> ETag, size,
        checksum), None when the upload does not exist any more."""
        out: dict[int, dict[str, Any]] = {}
        marker = 0
        while True:
            try:
                page = self.client.list_parts(Bucket=self.bucket, Key=full, UploadId=upload_id,
                                              PartNumberMarker=marker)
            except Exception as error:  # noqa: BLE001 - botocore ClientError, by code
                if self._code(error) in {"NoSuchUpload", *_MISSING}:
                    return None
                raise
            for p in page.get("Parts") or ():
                out[int(p["PartNumber"])] = {"ETag": p.get("ETag"), "size": p.get("Size"),
                                             "sha256": p.get("ChecksumSHA256")}
            if not page.get("IsTruncated"):
                return out
            marker = int(page["NextPartNumberMarker"])

    @staticmethod
    def _agreed(journaled: Mapping[int, Mapping[str, Any]], listed: Mapping[int, Mapping[str, Any]],
                checks: bool) -> dict[int, dict[str, Any]]:
        """The journaled parts the store holds as journaled: same ETag, same size,
        and (with checksums) the same SHA-256 when the store lists one. A part the
        store holds but the journal never recorded is sent again."""
        out = {}
        for n, mine in journaled.items():
            theirs = listed.get(int(n))
            if theirs is None or theirs.get("ETag") != mine.get("ETag"):
                continue
            if theirs.get("size") is not None and mine.get("size") is not None \
                    and int(theirs["size"]) != int(mine["size"]):
                continue
            if checks and (not mine.get("sha256") or (theirs.get("sha256") and theirs["sha256"] != mine["sha256"])):
                continue
            out[int(n)] = dict(mine)
        return out

    def read_file(self, key: str, target: str | os.PathLike[str]) -> bool:
        """Download object ``key`` to ``target``; a large one by parallel ranges,
        each pinned to the object's ETag (a replaced object fails the download
        instead of mixing two versions). False when the key is missing."""
        head = self._head(key)
        if head is None:
            return False
        size, etag = int(head["ContentLength"]), head.get("ETag")
        part = self.transfer.part_for(size)
        full = self._key(key)
        if size <= part:
            # One GET reads one version; no pin, so a rewritten pointer file is read
            # as it is now instead of failing the pull.
            data = self.read(key)
            if data is None:  # deleted since the HEAD (a prune)
                return False
            self._drop_stale_partials(target)
            partial = self._partial(target)
            try:  # never a cut-short file at the final name (a killed process, a full disk)
                Path(partial).write_bytes(data)
                os.replace(partial, target)
            finally:
                self._release_partial(partial)
            return True
        pin = {"IfMatch": etag} if etag else {}
        self._drop_stale_partials(target)
        partial = self._partial(target)  # renamed onto target once whole
        try:
            self._read_ranges(key, full, partial, size, part, pin)
            os.replace(partial, target)
        finally:  # a failed download leaves nothing (its name is this call's alone)
            self._release_partial(partial)
        return True

    @staticmethod
    def _partial(target: str | os.PathLike[str]) -> str:
        """The file a download of ``target`` is written to: next to it, named by this
        host, process and call, so two downloads of one target (two processes sharing
        a cache directory, a retry beside a stalled try) never truncate or interleave
        each other's bytes; the last rename wins with a whole file. It ends in
        :data:`DOWNLOAD_SUFFIX`: what a killed process left is recognised and removed
        (:meth:`_drop_stale_partials`, ``policy_release``), and a stage push skips it."""
        import uuid

        partial = f"{target}.{_host_token()}-{os.getpid()}-{uuid.uuid4().hex[:8]}{DOWNLOAD_SUFFIX}"
        with _LIVE_PARTIALS_LOCK:
            _LIVE_PARTIALS.add(os.path.abspath(partial))
        return partial

    @staticmethod
    def _release_partial(partial: str) -> None:
        Path(partial).unlink(missing_ok=True)
        with _LIVE_PARTIALS_LOCK:
            _LIVE_PARTIALS.discard(os.path.abspath(partial))

    @staticmethod
    def _partial_abandoned(path: Path, owner: str) -> bool:
        """Whether download file ``path`` (``owner``: the part of its name between
        the target's name and :data:`DOWNLOAD_SUFFIX`) has no writer any more."""
        host, _, rest = owner.partition("-")
        pid, _, _ = rest.partition("-")
        idle = time.time() - path.stat().st_mtime
        if _pid_namespace() is None or host != _host_token() or not pid.isdigit():
            return idle > UNKNOWN_PARTIAL_S  # nobody here to ask
        if int(pid) == os.getpid():  # this process, or an earlier one that had its pid
            with _LIVE_PARTIALS_LOCK:
                mine = os.path.abspath(path) in _LIVE_PARTIALS
            return not mine and idle > DEAD_PARTIAL_GRACE_S
        return idle > (STALE_PARTIAL_S if _pid_alive(int(pid)) else DEAD_PARTIAL_GRACE_S)

    @classmethod
    def _drop_stale_partials(cls, target: str | os.PathLike[str]) -> None:
        """Remove what killed downloads of ``target`` left next to it (a file of up
        to the object's size each; a sampler whose release prefetch is cut off again
        and again would fill its disk): :meth:`_partial_abandoned`. The file of a
        download still running, in this process or another, stays: it was written
        or marked (:meth:`_read_ranges`) a moment ago."""
        path = Path(target)
        head = path.name + "."
        try:
            with os.scandir(path.parent) as entries:
                names = [e.name for e in entries if e.name.startswith(head) and e.name.endswith(DOWNLOAD_SUFFIX)]
        except OSError:
            return
        for name in names:
            other = path.with_name(name)
            try:
                if cls._partial_abandoned(other, name[len(head):-len(DOWNLOAD_SUFFIX)]):
                    other.unlink(missing_ok=True)
            except OSError:  # raced with that download's rename or another cleanup
                pass

    def _read_ranges(self, key: str, full: str, partial: str, size: int, part: int, pin: Mapping[str, Any]) -> None:
        with open(partial, "wb") as fh:
            fh.truncate(size)

            def fetch(offset: int) -> None:
                end = min(offset + part, size) - 1
                attempt = refusals = 0
                while True:
                    try:
                        with self.range_gate:
                            body = self.client.get_object(Bucket=self.bucket, Key=full,
                                                          Range=f"bytes={offset}-{end}", **pin)["Body"].read()
                    except RANGE_READ_RETRYABLE:
                        # botocore retries the request, not the streamed body: a stalled
                        # body read of one part would otherwise fail the whole download
                        attempt += 1
                        if attempt == RANGE_READ_ATTEMPTS:
                            raise
                        _read_retry_pause(attempt - 1)
                        continue
                    except Exception as error:  # noqa: BLE001 - a ClientError, by code
                        refusals += 1
                        if not is_throttled(error) or refusals == THROTTLED_READ_ATTEMPTS:
                            raise
                        self._range_refused(error)
                        _read_retry_pause(refusals - 1)
                        continue
                    self.range_gate.succeeded()
                    break
                if len(body) != end - offset + 1:
                    raise ObjectStoreError(f"{self.uri}/{key}: range {offset}-{end} returned {len(body)} B")
                os.pwrite(fh.fileno(), body, offset)

            done = threading.Event()

            def mark() -> None:  # "this file has a writer", while parts are on their way
                while not done.wait(PARTIAL_HEARTBEAT_S):
                    try:
                        os.utime(partial)
                    except OSError:
                        return

            marker = threading.Thread(target=mark, name="yeto-get-mark", daemon=True)
            marker.start()
            try:
                with ThreadPoolExecutor(self.transfer.concurrency) as pool:
                    list(pool.map(fetch, range(0, size, part)))
            finally:
                done.set()
                marker.join()

    def _range_refused(self, error: BaseException) -> None:
        limit = self.range_gate.refused()
        print(f"[s3-store] WARNING: {self.uri}: the store refused a range read for its request rate "
              f"({error_code(error) or type(error).__name__}); this process now runs at most {limit} range "
              f"read(s) at once and raises that again as reads succeed", file=sys.stderr, flush=True)

    def read_files(self, items: Sequence[tuple[str, str | os.PathLike[str]]]) -> list[str]:
        """Download each ``(key, target)`` like :meth:`read_file`; returns the keys
        that are missing (nothing written for them). With a ``client_factory`` and
        ``transfer.download_processes`` > 1 the files are split over that many
        child interpreters, each moving ``ceil(file_concurrency / processes)`` files at
        once; otherwise ``file_concurrency`` threads of this process move them. The
        children are plain ``python -c`` subprocesses fed by stdin, not
        ``multiprocessing``: spawn re-imports the caller's ``__main__`` (an unguarded
        script runs again) and fork copies a process that may hold threads or a GPU
        runtime."""
        items = [(key, str(target)) for key, target in items]
        procs = min(int(self.transfer.download_processes), len(items))
        if self.client_factory is None or procs <= 1:
            return _read_files_here(self, items, self.transfer.file_concurrency)
        threads = max(1, -(-int(self.transfer.file_concurrency) // procs))
        groups = [items[i::procs] for i in range(procs)]
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
        children = []
        for group in groups:
            child = subprocess.Popen([sys.executable, "-c", READ_FILES_CHILD], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, env=env)
            children.append(child)
            child.stdin.write(pickle.dumps((self.uri, self.client_factory, asdict(self.transfer), threads, group)))
            child.stdin.close()
        missing, failed = [], []
        for child in children:
            out = child.stdout.read()
            if child.wait() != 0:
                failed.append(child.returncode)
                continue
            missing += pickle.loads(out)
        if failed:
            raise ObjectStoreError(f"{self.uri}: {len(failed)} of {len(children)} download processes "
                                   f"failed (exit {failed}); their stderr is above")
        return missing

    def list(self, prefix: str, *, start_after: str = "") -> list[str]:
        strip = len(self.prefix) + 1 if self.prefix else 0
        kwargs = {"Bucket": self.bucket, "Prefix": self._full(prefix)}
        if start_after:
            kwargs["StartAfter"] = self._key(start_after)
        keys = []
        for page in self.client.get_paginator("list_objects_v2").paginate(**kwargs):
            # a key ending in "/" is a directory marker a bucket mount (goofys) wrote: not an object
            keys += [item["Key"][strip:] for item in page.get("Contents", ()) if not item["Key"].endswith("/")]
        return keys

    def list_page(self, prefix: str, *, start_after: str = "", limit: int = 1000) -> tuple[list[str], str | None]:
        """One listing request (``ListObjectsV2``): up to ``limit`` keys after
        ``start_after`` and the key to continue from (None: that was the last page).
        A reader that knows which keys it can skip pages through a prefix itself."""
        strip = len(self.prefix) + 1 if self.prefix else 0
        kwargs = {"Bucket": self.bucket, "Prefix": self._full(prefix), "MaxKeys": int(limit)}
        if start_after:
            kwargs["StartAfter"] = self._full(start_after)
        page = self.client.list_objects_v2(**kwargs)
        contents = page.get("Contents", ())
        keys = [item["Key"][strip:] for item in contents if not item["Key"].endswith("/")]
        return keys, (contents[-1]["Key"][strip:] if page.get("IsTruncated") and contents else None)

    def list_meta(self, prefix: str, *, start_after: str = "",
                  end: str = "") -> tuple[list[tuple[str, float]], float]:
        """``([(key, LastModified epoch s)], server time)`` under ``prefix``: the
        object times and the clock they compare to both come from the server
        (the ``Date`` header of the listing), never from this host's clock.

        ``start_after`` / ``end``: only the keys ``start_after < key < end`` (byte
        order). The listing starts there (``StartAfter``) and stops at the first
        page that reaches ``end``, so a short range of a large prefix costs one
        request, not one per 1000 keys of the prefix."""
        from email.utils import parsedate_to_datetime

        strip = len(self.prefix) + 1 if self.prefix else 0
        kwargs = {"Bucket": self.bucket, "Prefix": self._full(prefix)}
        if start_after:
            kwargs["StartAfter"] = self._full(start_after)
        stop = self._full(end) if end else None
        out: list[tuple[str, float]] = []
        now: float | None = None
        for page in self.client.get_paginator("list_objects_v2").paginate(**kwargs):
            date = (page.get("ResponseMetadata") or {}).get("HTTPHeaders", {}).get("date")
            if date:
                stamp = parsedate_to_datetime(date).timestamp()
                now = stamp if now is None else max(now, stamp)
            contents = page.get("Contents", ())
            out += [(item["Key"][strip:], item["LastModified"].timestamp())
                    for item in contents
                    if not item["Key"].endswith("/") and (stop is None or item["Key"] < stop)]
            if stop is not None and contents and contents[-1]["Key"] >= stop:
                break
        if now is None:  # no Date header: the newest object time is a lower bound of the server clock
            now = max((t for _, t in out), default=0.0)
        return out, now

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=self._key(key))


class S3Store(BucketMountStore):
    """The bucket mount of an ``s3://`` store, plus its object API (``objects``,
    built on first use from the store credentials). With ``stage`` (a local
    directory), ``root`` is that directory, mirrored to the object store by
    ``commit`` / ``reload`` instead of the mount (see the module doc)."""

    def __init__(self, root: str | os.PathLike[str], uri: str, *, environ: Mapping[str, str],
                 client: Any = None, stage: str | os.PathLike[str] | None = None) -> None:
        bucket, prefix = parse_s3_uri(uri)
        if stage is not None:
            root = Path(stage).expanduser() / bucket / prefix
        super().__init__(root)
        self.uri = uri
        self.staged = stage is not None
        self._environ = dict(environ)
        self._client = client
        self._objects: S3ObjectStore | None = None
        self._mirror: TreeMirror | None = None

    @property
    def objects(self) -> S3ObjectStore:  # type: ignore[override]
        if self._objects is None:
            client = self._client or _client(self._environ, require_credentials=True)
            # another process builds its own client from the same settings (read_files);
            # an injected client (tests) has no such factory
            factory = None if self._client is not None else functools.partial(
                _client, dict(self._environ), require_credentials=True)
            self._objects = S3ObjectStore(self.uri, client=client,
                                          transfer=TransferSettings.from_env(self._environ),
                                          client_factory=factory)
        return self._objects

    def _tree(self) -> TreeMirror:
        if self._mirror is None:
            self._mirror = TreeMirror(self.root, self.objects, subtrees=STAGED_SUBTREES,
                                      pointers=POINTER_FILES, records=RECORD_FILES,
                                      independent=INDEPENDENT_DIRS,
                                      file_concurrency=self.objects.transfer.file_concurrency)
        return self._mirror

    @property
    def put_renames(self) -> bool:  # type: ignore[override]
        return self.staged  # a stage directory is local disk; the bucket mount is not

    @property
    def put_links(self) -> bool:  # type: ignore[override]
        return self.staged  # a 207 GB release export is linked into the stage, not copied

    def commit(self, paths: Sequence[str] | None = None) -> float:
        return self._tree().push(paths) if self.staged else super().commit(paths)

    @property
    def cut_commits_beside_buffer_reads(self) -> bool:  # type: ignore[override]
        return self.staged  # the buffer goes by the object API; a push takes the cut's paths only

    def push_local_file(self, relative: str) -> int | None:
        """Upload one file of the stage now (a trainer rank's cut shard, on whichever
        node wrote it: a multi-node island has one stage per node); returns the size
        the object store reports, None when this store has no stage."""
        if not self.staged:
            return None
        return self._tree().push_file(relative)

    def local_file_checksum(self, relative: str) -> str | None:
        """The checksum the object store keeps for a file :meth:`push_local_file`
        uploaded (``S3ObjectStore.checksum``), None without one."""
        if not self.staged:
            return None
        return self._tree().object_checksum(relative)

    def evict_local_file(self, relative: str) -> int:
        """Drop the stage copy of a file :meth:`push_local_file` uploaded (a cut shard:
        a K3.1 cut is ~1.35 TB per node); the bucket keeps it. Returns the bytes freed."""
        return self._tree().evict_file(relative) if self.staged else 0

    def delete_remote_tree(self, relative: str) -> list[str]:
        """Delete every object under ``relative``, including what other nodes uploaded."""
        if not self.staged:
            return []
        return self._tree().delete_tree(relative)

    def evict_local(self, relative: str) -> None:
        """Drop the stage copy of a pushed directory (a 207 GB Kimi-K3.1 release
        kept per release filled the trainer's disk); the bucket keeps it."""
        if self.staged:
            self._tree().evict_tree(relative)

    def remove_tree(self, relative: str) -> None:
        if self.staged:  # an evicted directory has no local files left to prune
            self.delete_remote_tree(relative)
        super().remove_tree(relative)

    def reload(self) -> None:
        if self.staged:
            self._tree().pull()

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "uri": self.uri, "staged": self.staged}


def store_from_env(path: str | os.PathLike[str], *, environ: Mapping[str, str], client: Any = None) -> S3Store:
    """``resume.store_for`` hook: the store URI comes from :data:`URI_ENV`."""
    uri = environ.get(URI_ENV)
    if not uri:
        raise ValueError(f"s3 checkpoint store selected but {URI_ENV} is not set")
    return S3Store(path, uri, environ=environ, client=client, stage=environ.get(STAGE_ENV) or None)


def head_objects(uri: str, *, environ: Mapping[str, str] | None = None, client: Any = None) -> S3ObjectStore:
    """The launcher's object API of the store."""
    env = os.environ if environ is None else environ
    return S3ObjectStore(uri, client=client or _client(env, require_credentials=False),
                         transfer=TransferSettings.from_env(env))


def island_envs(uri: str, environ: Mapping[str, str] | None = None, *, stage: bool = False,
                credentials: bool = True) -> dict[str, str]:
    """The environment of an island that uses the store's object API: the store
    hook, the URI, the local stage directory when ``stage`` (one-node islands),
    and the store settings and credentials of the launcher's environment. Raises
    ValueError without store credentials. ``credentials=False``: the island gets
    its credential from a Modal Secret with the AWS names (:data:`CREDENTIALS_ENV`
    tells it so); none is required here or copied."""
    env = dict(os.environ if environ is None else environ)
    parse_s3_uri(uri)
    if credentials and not (env.get(KEY_ID_ENV) and env.get(SECRET_ENV)) and env.get(PROFILE_ENV):
        env.update(profile_credentials(env))  # the profile the launch names: its key goes to the island
    if credentials and not (env.get(KEY_ID_ENV) and env.get(SECRET_ENV)):
        raise ValueError(f"an island that uses the s3 store {uri} needs store credentials: set "
                         f"{KEY_ID_ENV} and {SECRET_ENV} (a key limited to the store's bucket), or "
                         f"{PROFILE_ENV} to an AWS profile holding one, on the launcher; the island "
                         "never uses the launcher's cloud credentials")
    out = {STORE_IMPL_ENV: STORE_IMPL, URI_ENV: uri, **({STAGE_ENV: DEFAULT_STAGE} if stage else {})}
    out.update({name: env[name] for name in SETTING_ENVS if env.get(name)})
    # an endpoint / region found in the profile: the island has no profile to read them from
    endpoint, region = store_endpoint(env), store_region(env)
    if endpoint and ENDPOINT_ENV not in out:
        out[ENDPOINT_ENV] = endpoint
    if region and REGION_ENV not in out:
        out[REGION_ENV] = region
    if credentials:
        out.update({name: env[name] for name in CREDENTIAL_ENVS if env.get(name)})
    else:
        out[CREDENTIALS_ENV] = CREDENTIALS_AWS_ENV
    return out


def profile_credentials(environ: Mapping[str, str]) -> dict[str, str]:
    """:data:`KEY_ID_ENV` / :data:`SECRET_ENV` (/ :data:`TOKEN_ENV`) from the store's
    AWS profile (:data:`PROFILE_ENV`); ``{}`` when it has none."""
    profile = environ.get(PROFILE_ENV)
    if not profile or profile_config(environ).get("_missing"):
        return {}
    import boto3

    creds = boto3.session.Session(profile_name=profile).get_credentials()
    if creds is None:
        return {}
    frozen = creds.get_frozen_credentials()
    out = {KEY_ID_ENV: frozen.access_key, SECRET_ENV: frozen.secret_key}
    if frozen.token:
        out[TOKEN_ENV] = frozen.token
    return out


# ------------------------------------------------------------------ CLI helpers
def add_store_client_arguments(parser: Any) -> None:
    """``--endpoint-url`` and ``--profile`` of a CLI that opens an ``s3://`` store."""
    parser.add_argument("--endpoint-url", default=None, metavar="URL",
                        help=f"S3-compatible endpoint (R2, MinIO); default: {ENDPOINT_ENV}, AWS_ENDPOINT_URL, "
                             "the profile's endpoint_url, else AWS S3")
    parser.add_argument("--profile", default=None, metavar="NAME",
                        help=f"AWS profile with the store's key, endpoint_url and region "
                             f"(default: {PROFILE_ENV}, AWS_PROFILE)")


def apply_store_client_arguments(args: Any, environ: Any = None) -> None:
    """Put ``--endpoint-url`` / ``--profile`` into ``environ`` (default: the process
    environment), where every store client of this process reads them."""
    env = os.environ if environ is None else environ
    if getattr(args, "endpoint_url", None):
        env[ENDPOINT_ENV] = str(args.endpoint_url)
    if getattr(args, "profile", None):
        env[PROFILE_ENV] = str(args.profile)


_AUTH_CODES = frozenset({"InvalidAccessKeyId", "SignatureDoesNotMatch", "AccessDenied", "Unauthorized",
                         "InvalidToken", "ExpiredToken", "403", "401"})


def describe_store_error(error: BaseException, uri: str = "", environ: Mapping[str, str] | None = None) -> str | None:
    """A message naming what to set for a store error of credentials or endpoint;
    None for any other error."""
    env = os.environ if environ is None else environ
    name = type(error).__name__
    code = error_code(error)
    endpoint = store_endpoint(env)
    where = f"endpoint {endpoint}" if endpoint else "AWS S3 (no endpoint set)"
    hint_endpoint = (f"set {ENDPOINT_ENV} or AWS_ENDPOINT_URL (or --endpoint-url, or endpoint_url in the "
                     f"AWS profile) to the store's endpoint, e.g. https://<account>.r2.cloudflarestorage.com")
    target = f"s3 store {uri}" if uri else "s3 store"
    if name in ("NoCredentialsError", "PartialCredentialsError"):
        return f"{target}: {missing_credentials_message()}"
    if name == "ProfileNotFound":
        return f"{target}: AWS profile {store_profile(env)!r} not found; set {PROFILE_ENV} (or --profile) to a configured profile"
    if name in ("EndpointConnectionError", "ConnectTimeoutError", "InvalidEndpointConfigurationError") \
            or isinstance(error, ValueError) and "endpoint" in str(error).lower():
        return f"{target}: cannot reach {where} ({error}); {hint_endpoint}"
    if code in _AUTH_CODES:
        extra = "" if endpoint else f"; a key of an S3-compatible store (R2) is refused by AWS: {hint_endpoint}"
        return (f"{target}: {where} refused the credentials ({code}). Check {KEY_ID_ENV} / {SECRET_ENV} "
                f"or the profile in {PROFILE_ENV} / AWS_PROFILE{extra}")
    if code == "AuthorizationHeaderMalformed" or (code in ("InvalidRegionName", "InvalidRequest")
                                                  and endpoint and _is_r2(endpoint)):
        return f"{target}: {where} refused the region ({code}); set {REGION_ENV}=auto for Cloudflare R2"
    return None
