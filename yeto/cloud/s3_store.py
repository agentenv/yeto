"""Cloud layer: S3 (and S3-compatible, e.g. MinIO) object API of an ``s3://`` checkpoint store.

An island mounts an ``s3://BUCKET[/PREFIX]`` store with sky Storage MOUNT
(goofys) for its cuts. The state that one host rewrites while another reads it
(sample buffer, trainer commands, policy release pointer) goes through the object
API instead (:mod:`yeto.rl.engine.object_store` says why). The launch layer
exports on the island:

* ``YETO_RL_STORE_IMPL=yeto.cloud.s3_store:store_from_env`` and
  :data:`URI_ENV` (the store URI): ``resume.store_for`` then builds an
  :class:`S3Store` (the bucket mount plus ``objects``);
* the store credentials, :data:`CREDENTIAL_ENVS`, and optionally
  :data:`ENDPOINT_ENV` / :data:`REGION_ENV`. They are copied from the launcher's
  environment, never from its cloud credential files: an island must not hold
  cloud credentials (secret-handling-hardening D4), so the store credential is a
  separate one, limited to the store's bucket by its issuer. The island never
  falls back to the default AWS credential chain.

The head (launcher) reads and writes the same keys with :func:`head_objects`: the
store credentials when its environment has them, else its own AWS credentials.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from yeto.rl.engine.object_store import ObjectStoreError, check_key
from yeto.rl.engine.resume import STORE_IMPL_ENV, BucketMountStore

STORE_IMPL = "yeto.cloud.s3_store:store_from_env"  # value of resume.STORE_IMPL_ENV
URI_ENV = "YETO_RL_STORE_S3_URI"
ENDPOINT_ENV = "YETO_RL_STORE_S3_ENDPOINT_URL"  # S3-compatible endpoint (MinIO); unset: AWS
REGION_ENV = "YETO_RL_STORE_S3_REGION"
KEY_ID_ENV = "YETO_RL_STORE_S3_ACCESS_KEY_ID"
SECRET_ENV = "YETO_RL_STORE_S3_SECRET_ACCESS_KEY"
TOKEN_ENV = "YETO_RL_STORE_S3_SESSION_TOKEN"
CREDENTIAL_ENVS = (KEY_ID_ENV, SECRET_ENV, TOKEN_ENV)
SECRET_ENVS = frozenset({SECRET_ENV, TOKEN_ENV})  # shipped as task secrets, not plain envs
SETTING_ENVS = (ENDPOINT_ENV, REGION_ENV)

# S3 error codes of a conditional write that lost to an existing object
_CREATE_LOST = frozenset({"PreconditionFailed", "ConditionalRequestConflict"})


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """``s3://BUCKET[/PREFIX]`` -> (bucket, prefix). Raises ValueError."""
    scheme, _, rest = str(uri).partition("://")
    bucket, _, prefix = rest.strip("/").partition("/")
    if scheme != "s3" or not bucket:
        raise ValueError(f"{uri!r} is not an s3://BUCKET[/PREFIX] store")
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
    current limit: one sampler island)."""

    def __init__(self, uri: str, *, client: Any) -> None:
        self.bucket, self.prefix = parse_s3_uri(uri)
        self.uri = f"s3://{self.bucket}" + (f"/{self.prefix}" if self.prefix else "")
        self.client = client
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
            if self._code(error) in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def list(self, prefix: str, *, start_after: str = "") -> list[str]:
        strip = len(self.prefix) + 1 if self.prefix else 0
        kwargs = {"Bucket": self.bucket, "Prefix": self._full(prefix)}
        if start_after:
            kwargs["StartAfter"] = self._key(start_after)
        keys = []
        for page in self.client.get_paginator("list_objects_v2").paginate(**kwargs):
            keys += [item["Key"][strip:] for item in page.get("Contents", ())]
        return keys

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=self._key(key))


class S3Store(BucketMountStore):
    """The bucket mount of an ``s3://`` store, plus its object API (``objects``,
    built on first use from the store credentials)."""

    def __init__(self, root: str | os.PathLike[str], uri: str, *, environ: Mapping[str, str],
                 client: Any = None) -> None:
        super().__init__(root)
        self.uri = uri
        parse_s3_uri(uri)
        self._environ = dict(environ)
        self._client = client
        self._objects: S3ObjectStore | None = None

    @property
    def objects(self) -> S3ObjectStore:  # type: ignore[override]
        if self._objects is None:
            client = self._client or _client(self._environ, require_credentials=True)
            self._objects = S3ObjectStore(self.uri, client=client)
        return self._objects

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "uri": self.uri}


def store_from_env(path: str | os.PathLike[str], *, environ: Mapping[str, str], client: Any = None) -> S3Store:
    """``resume.store_for`` hook: the store URI comes from :data:`URI_ENV`."""
    uri = environ.get(URI_ENV)
    if not uri:
        raise ValueError(f"s3 checkpoint store selected but {URI_ENV} is not set")
    return S3Store(path, uri, environ=environ, client=client)


def head_objects(uri: str, *, environ: Mapping[str, str] | None = None, client: Any = None) -> S3ObjectStore:
    """The launcher's object API of the store."""
    env = os.environ if environ is None else environ
    return S3ObjectStore(uri, client=client or _client(env, require_credentials=False))


def island_envs(uri: str, environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment of an island that uses the store's object API: the store
    hook, the URI, and the store settings and credentials of the launcher's
    environment. Raises ValueError without store credentials."""
    env = os.environ if environ is None else environ
    parse_s3_uri(uri)
    if not has_store_credentials(env):
        raise ValueError(f"an island that uses the s3 store {uri} needs store credentials: set "
                         f"{KEY_ID_ENV} and {SECRET_ENV} (a key limited to the store's bucket) on the "
                         "launcher; the island never uses the launcher's cloud credentials")
    out = {STORE_IMPL_ENV: STORE_IMPL, URI_ENV: uri}
    out.update({name: env[name] for name in (*SETTING_ENVS, *CREDENTIAL_ENVS) if env.get(name)})
    return out
