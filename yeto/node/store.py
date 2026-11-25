"""The run store as the node runtime and the pool routes reach it.

One place for what the routes each wrote for themselves:

* :data:`STORE_URI_SHELL`: the shell line a node runs to build the run store's
  URI from its env file (``R2_BUCKET``, ``AWS_ENDPOINT_URL``) and
  ``YETO_NC_RUN_PREFIX``: the bucket and the endpoint are only known on the node;
* :func:`run_store_uri`: the same URI on the launcher, from the pool's store env file;
* :func:`open_objects`: the object API of a store URI (``s3://``, ``r2://``) or of a
  store directory, with the env file's credentials;
* :func:`pool_fence_store`: the run store as fence records for a pool config
  (``r2_env_file``, ``run_prefix``, ``bucket_env``);
* :func:`reclaim_marker_env` / :func:`sampler_route_env`: the reclaim marker and the
  release download's spread key every route gives its sampler.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

STORE_URI_SHELL = ('export YETO_RL_STORE_S3_URI="s3://${R2_BUCKET}/runs/${YETO_NC_RUN_PREFIX}" '
                   'YETO_RL_STORE_S3_ENDPOINT_URL="$AWS_ENDPOINT_URL"')
RECLAIM_MARKER_ENV = "YETO_SPOT_RECLAIM_MARKER"
RELEASE_SPREAD_KEY_ENV = "YETO_RL_RELEASE_SPREAD_KEY"


class StoreConfigError(RuntimeError):
    pass


def read_env_file(path: str | os.PathLike[str]) -> dict[str, str]:
    """``KEY=VALUE`` lines (``export`` and quotes allowed; ``#`` comments skipped)."""
    out: dict[str, str] = {}
    for line in Path(path).expanduser().read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.removeprefix("export ").strip()] = value.strip().strip("'\"")
    return out


def run_store_uri(env_file: str | os.PathLike[str], run_prefix: str, *, bucket_env: str = "R2_BUCKET") -> str:
    """``s3://<bucket>/runs/<run prefix>`` of a pool (the bucket from its store env file)."""
    bucket = read_env_file(env_file).get(bucket_env)
    if not bucket:
        raise StoreConfigError(f"no {bucket_env} in the store env file")
    return f"s3://{bucket}/runs/{str(run_prefix).strip('/')}"


def reclaim_marker_env(shard: str, sampler_dir: str) -> dict[str, str]:
    """``YETO_SPOT_RECLAIM_MARKER`` of a sampler: SIGTERM writes the marker and the sampler
    leaves in order (stops the engine, frees its claims, exits) instead of dying in its round."""
    return {RECLAIM_MARKER_ENV: f"{sampler_dir.rstrip('/')}/{shard}.reclaim"}


def sampler_route_env(run_prefix: str, shard: str, sampler_dir: str) -> dict[str, str]:
    """The two per-shard variables every pool route gives its sampler: the reclaim marker
    (SIGTERM is the orderly leave, ``sampler_resume.install_marker_signals``) and the
    release download's spread key (``policy_release.download_delay_s``: a node's host
    name need not be distinct)."""
    return {**reclaim_marker_env(shard, sampler_dir), RELEASE_SPREAD_KEY_ENV: f"{str(run_prefix).strip('/')}/{shard}"}


def store_environ(environ: Mapping[str, str] | None = None,
                  env_file: str | os.PathLike[str] | None = None) -> dict[str, str]:
    """``environ`` plus the env file, with the S3 store's defaults (aws-env credentials,
    the AWS endpoint as the store endpoint, region ``auto``)."""
    from yeto.cloud import s3_store  # noqa: PLC0415

    env = dict(os.environ if environ is None else environ)
    if env_file:
        env.update(read_env_file(env_file))
    env.setdefault(s3_store.CREDENTIALS_ENV, s3_store.CREDENTIALS_AWS_ENV)
    if env.get(s3_store.AWS_ENDPOINT_ENV):
        env.setdefault(s3_store.ENDPOINT_ENV, env[s3_store.AWS_ENDPOINT_ENV])
    env.setdefault(s3_store.REGION_ENV, "auto")
    return env


def open_objects(store: str, *, env_file: str | os.PathLike[str] | None = None,
                 environ: Mapping[str, str] | None = None) -> Any:
    """The object API of the run store ``store``: an ``s3://`` / ``r2://`` URI, or a
    local directory (one host, tests)."""
    if "://" in str(store):
        from yeto.cloud import s3_store  # noqa: PLC0415

        return s3_store.head_objects(s3_store.s3_uri(store), environ=store_environ(environ, env_file))
    from yeto.rl.engine.object_store import DirectoryObjectStore  # noqa: PLC0415

    return DirectoryObjectStore(Path(store).expanduser())


def pool_fence_store(cfg: Mapping[str, Any], *, error: type[Exception] = StoreConfigError) -> Any:
    """The run store of a pool config as fence records (``r2_env_file``, ``run_prefix``,
    ``bucket_env``): the one copy of what the NC, ssh and Modal pool routes each built
    (``elastic.s3_fence_store`` of ``s3://<bucket>/runs/<run prefix>``). ``error``: the
    route's error type for an env file without the bucket."""
    from yeto.launch.nc import elastic  # noqa: PLC0415 - looked up per call (tests replace it)

    bucket_env = cfg.get("bucket_env", "R2_BUCKET")
    bucket = elastic.read_env_file(cfg["r2_env_file"]).get(bucket_env)
    if not bucket:
        raise error(f"no {bucket_env} in the store env file")
    return elastic.s3_fence_store(f"s3://{bucket}/runs/{str(cfg['run_prefix']).strip('/')}", cfg["r2_env_file"])
