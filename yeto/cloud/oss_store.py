"""Cloud layer: Alibaba Cloud OSS checkpoint store (rl-cross-site-sample-store).

A cross-site run (trainer on one site, sampler islands on others) shares the
sample buffer, the round cuts and the policy releases through one OSS bucket,
which every site reaches; no file system is mounted on two sites. The store
talks to OSS through its S3-compatible API with the client of
:mod:`yeto.cloud.s3_store` (same object API, same multipart transfer). Like the
staged S3 store, the store's ``root`` is a local directory mirrored to the
bucket (:class:`~yeto.rl.engine.object_store.TreeMirror`): the island writes
files as on a local disk, ``commit`` uploads them (data files, then release
records, then pointer files) and ``reload`` downloads the newest committed cut.
State that one host rewrites while another reads it goes by object key
(``CheckpointStore.objects``). There is no bucket mount.

OSS has no file lock and no append. The sample buffer therefore keeps one shard
per sampler island (``OssObjectStore.buffer_shards``; ``sample_buffer`` module
doc). Conditional creates use ``x-oss-forbid-overwrite`` instead of
``If-None-Match`` (OSS documents the former; how it answers the latter is not
verified).

The launch layer exports on the island:

* ``YETO_RL_STORE_IMPL=yeto.cloud.oss_store:store_from_env`` and :data:`URI_ENV`
  (``oss://BUCKET[/PREFIX]``);
* :data:`ENDPOINT_ENV` (required: the bucket region's S3-compatible endpoint,
  e.g. ``https://oss-cn-hangzhou.aliyuncs.com``), optionally :data:`REGION_ENV`,
  :data:`STAGE_ENV` and the part settings of ``s3_store``;
* the credentials: :data:`KEY_ID_ENV` + :data:`SECRET_ENV` (+ :data:`TOKEN_ENV`
  for an STS key), or :data:`CREDENTIALS_FILE_ENV`, the path of a JSON file
  ``{"access_key_id", "access_key_secret"[, "security_token"]}`` readable by its
  owner only. The store never falls back to another credential source.

Not tested against OSS: request signing, virtual-hosted addressing, the
forbid-overwrite header and multipart limits follow the OSS documentation only
(the CPU tests use a fake S3 client).
"""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from yeto.cloud.s3_store import (PART_CONCURRENCY_ENV, PART_MB_ENV, POINTER_FILES, RECORD_FILES,
                                 STAGED_SUBTREES, S3ObjectStore, TransferSettings, parse_s3_uri)
from yeto.rl.engine.object_store import ObjectStoreError, TreeMirror
from yeto.rl.engine.resume import STORE_IMPL_ENV, CheckpointStore

SCHEME = "oss"
STORE_IMPL = "yeto.cloud.oss_store:store_from_env"  # value of resume.STORE_IMPL_ENV
URI_ENV = "YETO_RL_STORE_OSS_URI"
ENDPOINT_ENV = "YETO_RL_STORE_OSS_ENDPOINT_URL"
REGION_ENV = "YETO_RL_STORE_OSS_REGION"
KEY_ID_ENV = "YETO_RL_STORE_OSS_ACCESS_KEY_ID"
SECRET_ENV = "YETO_RL_STORE_OSS_ACCESS_KEY_SECRET"
TOKEN_ENV = "YETO_RL_STORE_OSS_SECURITY_TOKEN"
CREDENTIALS_FILE_ENV = "YETO_RL_STORE_OSS_CREDENTIALS_FILE"
STAGE_ENV = "YETO_RL_STORE_OSS_STAGE"
DEFAULT_STAGE = "~/.cache/yeto/oss-store-stage"
CREDENTIAL_ENVS = (KEY_ID_ENV, SECRET_ENV, TOKEN_ENV)
SECRET_ENVS = frozenset({SECRET_ENV, TOKEN_ENV})  # shipped as task secrets, not plain envs
SETTING_ENVS = (ENDPOINT_ENV, REGION_ENV, CREDENTIALS_FILE_ENV, PART_MB_ENV, PART_CONCURRENCY_ENV)
FORBID_OVERWRITE_HEADER = "x-oss-forbid-overwrite"


def parse_oss_uri(uri: str) -> tuple[str, str]:
    """``oss://BUCKET[/PREFIX]`` -> (bucket, prefix). Raises ValueError."""
    return parse_s3_uri(uri, SCHEME)


def is_oss_uri(value: str) -> bool:
    return str(value).startswith(SCHEME + "://")


class OssObjectStore(S3ObjectStore):
    """:class:`~yeto.cloud.s3_store.S3ObjectStore` over ``oss://BUCKET[/PREFIX]``.
    ``create`` sends ``x-oss-forbid-overwrite: true``; a lost create answers
    ``FileAlreadyExists``. The sample buffer still writes one shard per sampler
    island (``buffer_shards``), so no record depends on that header."""

    scheme = SCHEME
    buffer_shards = True

    def create(self, key: str, data: bytes) -> bool:
        if not self.conditional_writes:
            return super().create(key, data)
        condition = {"ForbidOverwrite": True} if _accepts_forbid(self.client) else {"IfNoneMatch": "*"}
        try:
            self.client.put_object(Bucket=self.bucket, Key=self._key(key), Body=data, **condition)
            return True
        except Exception as error:  # noqa: BLE001 - botocore ClientError, by code
            if self._code(error) in ("FileAlreadyExists", "PreconditionFailed"):
                return False
            if self._code(error) != "NotImplemented":
                raise
            self.conditional_writes = False
        return super().create(key, data)


def _accepts_forbid(client: Any) -> bool:
    """Whether ``client`` takes ``ForbidOverwrite`` (a client from :func:`oss_client`
    maps it to the OSS header; a test double declares it)."""
    return bool(getattr(client, "accepts_forbid_overwrite", False))


def _add_forbid_overwrite(client: Any) -> Any:
    """Let ``put_object(ForbidOverwrite=True)`` send ``x-oss-forbid-overwrite``:
    boto3 has no such parameter, so the call carries it in its context and a
    request hook turns it into the header before signing."""
    events = client.meta.events

    def lift(params: dict[str, Any], context: dict[str, Any], **_: Any) -> None:
        if params.pop("ForbidOverwrite", False):
            context["yeto_forbid_overwrite"] = True

    def header(request: Any, **_: Any) -> None:
        if request.context.get("yeto_forbid_overwrite"):
            request.headers[FORBID_OVERWRITE_HEADER] = "true"

    events.register("before-parameter-build.s3.PutObject", lift)
    events.register("before-sign.s3.PutObject", header)
    client.accepts_forbid_overwrite = True
    return client


def credentials(environ: Mapping[str, str]) -> tuple[str, str, str | None]:
    """(access key id, secret, security token or None) from the environment or the
    credentials file. Raises ObjectStoreError without them; never returns them in
    an error message."""
    if environ.get(KEY_ID_ENV) and environ.get(SECRET_ENV):
        return environ[KEY_ID_ENV], environ[SECRET_ENV], environ.get(TOKEN_ENV) or None
    path = environ.get(CREDENTIALS_FILE_ENV)
    if not path:
        raise ObjectStoreError(f"oss store: set {KEY_ID_ENV} and {SECRET_ENV}, or {CREDENTIALS_FILE_ENV}")
    file = Path(path).expanduser()
    try:
        mode = file.stat().st_mode
        doc = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ObjectStoreError(f"oss store: cannot read the credentials file {file}: "
                               f"{type(error).__name__}") from None
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise ObjectStoreError(f"oss store: credentials file {file} is accessible by group or others "
                               "(chmod 600 it)")
    if not (isinstance(doc, dict) and doc.get("access_key_id") and doc.get("access_key_secret")):
        raise ObjectStoreError(f"oss store: credentials file {file} needs access_key_id and access_key_secret")
    return str(doc["access_key_id"]), str(doc["access_key_secret"]), doc.get("security_token") or None


def oss_client(environ: Mapping[str, str]) -> Any:
    """A boto3 S3 client for the OSS endpoint: virtual-hosted bucket addressing
    (OSS refuses path style), SigV4, and checksums only where the API requires
    them (newer boto3 adds CRC headers that S3-compatible services may refuse)."""
    import boto3
    from botocore.config import Config

    endpoint = environ.get(ENDPOINT_ENV)
    if not endpoint:
        raise ObjectStoreError(f"oss store: {ENDPOINT_ENV} is not set (the bucket region's endpoint)")
    key_id, secret, token = credentials(environ)
    session = boto3.session.Session(aws_access_key_id=key_id, aws_secret_access_key=secret,
                                    aws_session_token=token)
    config = Config(signature_version="s3v4", s3={"addressing_style": "virtual"},
                    request_checksum_calculation="when_required",
                    response_checksum_validation="when_required")
    return _add_forbid_overwrite(session.client("s3", endpoint_url=endpoint,
                                                region_name=environ.get(REGION_ENV) or None, config=config))


class OssStore(CheckpointStore):
    """The checkpoint store of an ``oss://`` URI: a local stage directory (``root``)
    mirrored to the bucket, plus the object API (``objects``, built on first use)."""

    kind = "oss"

    def __init__(self, uri: str, *, environ: Mapping[str, str], stage: str | os.PathLike[str] = DEFAULT_STAGE,
                 client: Any = None) -> None:
        bucket, prefix = parse_oss_uri(uri)
        super().__init__(Path(stage).expanduser() / bucket / prefix)
        self.uri = uri
        self._environ = dict(environ)
        self._client = client
        self._objects: OssObjectStore | None = None
        self._mirror: TreeMirror | None = None

    @property
    def objects(self) -> OssObjectStore:  # type: ignore[override]
        if self._objects is None:
            self._objects = OssObjectStore(self.uri, client=self._client or oss_client(self._environ),
                                           transfer=TransferSettings.from_env(self._environ))
        return self._objects

    def _tree(self) -> TreeMirror:
        if self._mirror is None:
            self._mirror = TreeMirror(self.root, self.objects, subtrees=STAGED_SUBTREES,
                                      pointers=POINTER_FILES, records=RECORD_FILES)
        return self._mirror

    def commit(self) -> float:
        return self._tree().push()

    def reload(self) -> None:
        self._tree().pull()

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "uri": self.uri}


def store_from_env(path: str | os.PathLike[str], *, environ: Mapping[str, str], client: Any = None) -> OssStore:
    """``resume.store_for`` hook: the store URI comes from :data:`URI_ENV` (``path``,
    the island's store path, is replaced by the stage directory)."""
    uri = environ.get(URI_ENV)
    if not uri:
        raise ValueError(f"oss checkpoint store selected but {URI_ENV} is not set")
    return OssStore(uri, environ=environ, stage=environ.get(STAGE_ENV) or DEFAULT_STAGE, client=client)


def head_objects(uri: str, *, environ: Mapping[str, str] | None = None, client: Any = None) -> OssObjectStore:
    """The launcher's object API of the store (the same credentials as the islands)."""
    env = os.environ if environ is None else environ
    return OssObjectStore(uri, client=client or oss_client(env), transfer=TransferSettings.from_env(env))


def has_credentials(environ: Mapping[str, str]) -> bool:
    return bool((environ.get(KEY_ID_ENV) and environ.get(SECRET_ENV)) or environ.get(CREDENTIALS_FILE_ENV))


def island_envs(uri: str, environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment of an island that uses the store: the store hook, the URI,
    and the store settings and credentials of the launcher's environment (a
    credentials file path is passed as is: the file must exist on the island).
    Raises ValueError without an endpoint or credentials."""
    env = os.environ if environ is None else environ
    parse_oss_uri(uri)
    if not env.get(ENDPOINT_ENV):
        raise ValueError(f"the oss store {uri} needs {ENDPOINT_ENV} on the launcher")
    if not has_credentials(env):
        raise ValueError(f"an island that uses the oss store {uri} needs store credentials: set "
                         f"{KEY_ID_ENV} and {SECRET_ENV}, or {CREDENTIALS_FILE_ENV}, on the launcher")
    out = {STORE_IMPL_ENV: STORE_IMPL, URI_ENV: uri}
    out.update({name: env[name] for name in (*SETTING_ENVS, *CREDENTIAL_ENVS, STAGE_ENV) if env.get(name)})
    return out
