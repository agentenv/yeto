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
  cloud credentials (secret-handling-hardening D4), so the store credential is a
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

import os
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yeto.rl.engine.object_store import ObjectStoreError, TreeMirror, check_key
from yeto.rl.engine.resume import STORE_IMPL_ENV, BucketMountStore

STORE_IMPL = "yeto.cloud.s3_store:store_from_env"  # value of resume.STORE_IMPL_ENV
URI_ENV = "YETO_RL_STORE_S3_URI"
ENDPOINT_ENV = "YETO_RL_STORE_S3_ENDPOINT_URL"  # S3-compatible endpoint (MinIO); unset: AWS
REGION_ENV = "YETO_RL_STORE_S3_REGION"
KEY_ID_ENV = "YETO_RL_STORE_S3_ACCESS_KEY_ID"
SECRET_ENV = "YETO_RL_STORE_S3_SECRET_ACCESS_KEY"
TOKEN_ENV = "YETO_RL_STORE_S3_SESSION_TOKEN"
STAGE_ENV = "YETO_RL_STORE_S3_STAGE"  # a local directory: cuts and releases by object API
PART_MB_ENV = "YETO_RL_STORE_PART_MB"  # multipart part size (MiB); shared by the S3 and OSS stores
PART_CONCURRENCY_ENV = "YETO_RL_STORE_PART_CONCURRENCY"  # parts moved in parallel
CREDENTIAL_ENVS = (KEY_ID_ENV, SECRET_ENV, TOKEN_ENV)
SECRET_ENVS = frozenset({SECRET_ENV, TOKEN_ENV})  # shipped as task secrets, not plain envs
SETTING_ENVS = (ENDPOINT_ENV, REGION_ENV, PART_MB_ENV, PART_CONCURRENCY_ENV)
DEFAULT_STAGE = "~/.cache/yeto/store-stage"
# what a staged store pulls at start (the newest committed cut: LATEST, its snapshot, the
# cuts) and the pointer files it uploads last (resume.LATEST, policy_release.LATEST)
STAGED_SUBTREES = ("LATEST", "state/", "round-cuts/")
POINTER_FILES = frozenset({"LATEST", "LATEST.json"})
# record files: uploaded after the data files next to them, before the pointers
# (policy_release.RECORD lists the release's weight files)
RECORD_FILES = frozenset({"RELEASE.json"})

# error codes of a conditional write that lost to an existing object (S3; OSS
# answers FileAlreadyExists to x-oss-forbid-overwrite)
_CREATE_LOST = frozenset({"PreconditionFailed", "ConditionalRequestConflict", "FileAlreadyExists"})
_MISSING = frozenset({"404", "NoSuchKey", "NotFound"})
_MIB = 1 << 20
MIN_PART_BYTES = 5 * _MIB  # S3 and OSS: every part but the last is at least 5 MiB
MAX_PARTS = 10_000  # S3 and OSS: parts of one multipart upload


@dataclass(frozen=True)
class TransferSettings:
    """How large files move: a file larger than one part is uploaded in parts of
    ``part_bytes`` (raised so a file needs at most :data:`MAX_PARTS`) and
    downloaded by ranges of the same size; ``concurrency`` parts move at once
    (memory: up to ``concurrency`` parts)."""

    part_bytes: int = 64 * _MIB
    concurrency: int = 8

    def __post_init__(self) -> None:
        if self.part_bytes < MIN_PART_BYTES:
            raise ValueError(f"part size {self.part_bytes} B is below the {MIN_PART_BYTES} B minimum")
        if self.concurrency < 1:
            raise ValueError("part concurrency must be >= 1")

    def part_for(self, size: int) -> int:
        return max(self.part_bytes, -(-int(size) // MAX_PARTS))

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> "TransferSettings":
        default = cls()
        return cls(part_bytes=int(float(environ.get(PART_MB_ENV) or default.part_bytes / _MIB) * _MIB),
                   concurrency=int(environ.get(PART_CONCURRENCY_ENV) or default.concurrency))


def parse_s3_uri(uri: str, scheme: str = "s3") -> tuple[str, str]:
    """``<scheme>://BUCKET[/PREFIX]`` -> (bucket, prefix). Raises ValueError."""
    got, _, rest = str(uri).partition("://")
    bucket, _, prefix = rest.strip("/").partition("/")
    if got != scheme or not bucket:
        raise ValueError(f"{uri!r} is not a store URI {scheme}://BUCKET[/PREFIX]")
    return bucket, prefix.strip("/")


def has_store_credentials(environ: Mapping[str, str]) -> bool:
    return bool(environ.get(KEY_ID_ENV) and environ.get(SECRET_ENV))


def _client(environ: Mapping[str, str], *, require_credentials: bool) -> Any:
    import boto3

    if has_store_credentials(environ):
        session = boto3.session.Session(aws_access_key_id=environ[KEY_ID_ENV],
                                        aws_secret_access_key=environ[SECRET_ENV],
                                        aws_session_token=environ.get(TOKEN_ENV) or None)
    elif require_credentials:
        raise ObjectStoreError(f"s3 store: {KEY_ID_ENV} and {SECRET_ENV} are not set (an island "
                               "uses the store credentials only, never the default AWS chain)")
    else:
        session = boto3.session.Session()
    return session.client("s3", endpoint_url=environ.get(ENDPOINT_ENV) or None,
                          region_name=environ.get(REGION_ENV) or None)


class S3ObjectStore:
    """:class:`~yeto.rl.engine.object_store.ObjectStore` over ``s3://BUCKET[/PREFIX]``:
    a key is ``PREFIX/<key>``. ``client``: a boto3 S3 client (or a test double).

    ``create`` is a conditional put (``If-None-Match: *``). An endpoint that does
    not implement it answers ``NotImplemented``; the store then checks for the key
    and puts, which is safe only while one writer appends (the sample buffer's
    current limit: one sampler island).

    ``buffer_shards``: the sample buffer must not rely on ``create`` between
    writers on several hosts; each sampler island then writes its own shard
    (``sample_buffer`` module doc). False here: S3 has conditional writes."""

    scheme = "s3"
    buffer_shards = False

    def __init__(self, uri: str, *, client: Any, transfer: TransferSettings | None = None) -> None:
        self.bucket, self.prefix = parse_s3_uri(uri, self.scheme)
        self.uri = f"{self.scheme}://{self.bucket}" + (f"/{self.prefix}" if self.prefix else "")
        self.client = client
        self.transfer = transfer or TransferSettings()
        self.conditional_writes = True

    def _full(self, name: str) -> str:
        return f"{self.prefix}/{name}" if self.prefix else name

    def _key(self, key: str) -> str:
        return self._full(check_key(key))

    def _code(self, error: Exception) -> str:
        return str(getattr(error, "response", {}).get("Error", {}).get("Code", ""))

    def read(self, key: str) -> bytes | None:
        try:
            return self.client.get_object(Bucket=self.bucket, Key=self._key(key))["Body"].read()
        except self.client.exceptions.NoSuchKey:
            return None

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
                    return False
                if code != "NotImplemented":
                    raise
                self.conditional_writes = False
        if self.exists(key):
            return False
        self.write(key, data)
        return True

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

    def write_file(self, key: str, source: str | os.PathLike[str]) -> None:
        """Upload a file; one larger than a part goes as a multipart upload whose
        parts are read from the file as they are sent (never the whole file in
        memory). A failed upload is aborted, so no part is left billed."""
        size = os.path.getsize(source)
        part = self.transfer.part_for(size)
        full = self._key(key)
        if size <= part:
            with open(source, "rb") as fh:
                self.client.put_object(Bucket=self.bucket, Key=full, Body=fh.read())
            return
        upload_id = self.client.create_multipart_upload(Bucket=self.bucket, Key=full)["UploadId"]
        try:
            with open(source, "rb") as fh:
                def send(number: int) -> dict[str, Any]:
                    offset = (number - 1) * part
                    body = os.pread(fh.fileno(), min(part, size - offset), offset)
                    done = self.client.upload_part(Bucket=self.bucket, Key=full, UploadId=upload_id,
                                                   PartNumber=number, Body=body)
                    return {"PartNumber": number, "ETag": done["ETag"]}

                with ThreadPoolExecutor(self.transfer.concurrency) as pool:
                    parts = list(pool.map(send, range(1, -(-size // part) + 1)))
            self.client.complete_multipart_upload(Bucket=self.bucket, Key=full, UploadId=upload_id,
                                                  MultipartUpload={"Parts": parts})
        except BaseException:
            self.client.abort_multipart_upload(Bucket=self.bucket, Key=full, UploadId=upload_id)
            raise

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
            Path(target).write_bytes(data)
            return True
        pin = {"IfMatch": etag} if etag else {}
        with open(target, "wb") as fh:
            fh.truncate(size)

            def fetch(offset: int) -> None:
                end = min(offset + part, size) - 1
                body = self.client.get_object(Bucket=self.bucket, Key=full, Range=f"bytes={offset}-{end}",
                                              **pin)["Body"].read()
                if len(body) != end - offset + 1:
                    raise ObjectStoreError(f"{self.uri}/{key}: range {offset}-{end} returned {len(body)} B")
                os.pwrite(fh.fileno(), body, offset)

            with ThreadPoolExecutor(self.transfer.concurrency) as pool:
                list(pool.map(fetch, range(0, size, part)))
        return True

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
            self._objects = S3ObjectStore(self.uri, client=client,
                                          transfer=TransferSettings.from_env(self._environ))
        return self._objects

    def _tree(self) -> TreeMirror:
        if self._mirror is None:
            self._mirror = TreeMirror(self.root, self.objects, subtrees=STAGED_SUBTREES,
                                      pointers=POINTER_FILES, records=RECORD_FILES)
        return self._mirror

    def commit(self) -> float:
        return self._tree().push() if self.staged else super().commit()

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


def island_envs(uri: str, environ: Mapping[str, str] | None = None, *, stage: bool = False) -> dict[str, str]:
    """The environment of an island that uses the store's object API: the store
    hook, the URI, the local stage directory when ``stage`` (one-node islands),
    and the store settings and credentials of the launcher's environment. Raises
    ValueError without store credentials."""
    env = os.environ if environ is None else environ
    parse_s3_uri(uri)
    if not has_store_credentials(env):
        raise ValueError(f"an island that uses the s3 store {uri} needs store credentials: set "
                         f"{KEY_ID_ENV} and {SECRET_ENV} (a key limited to the store's bucket) on the "
                         "launcher; the island never uses the launcher's cloud credentials")
    out = {STORE_IMPL_ENV: STORE_IMPL, URI_ENV: uri, **({STAGE_ENV: DEFAULT_STAGE} if stage else {})}
    out.update({name: env[name] for name in (*SETTING_ENVS, *CREDENTIAL_ENVS) if env.get(name)})
    return out
