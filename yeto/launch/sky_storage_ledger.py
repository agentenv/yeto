"""Buckets sky creates for a sky island's storage mounts, in the run's resource ledger.

A sky island with the per-island checkpoint volume (``--spot``: ``island_keeps_spot_volume``)
mounts a ``sky.Storage`` with a yeto-chosen name and no source
(``storage._rl_checkpoint_storage_name``): sky creates that bucket while it launches the
cluster, before any instance exists. A launch that then fails (no capacity, a spot quota)
leaves no cluster, so ``sky down`` never runs and the empty bucket stays (VC r4: one S3
bucket per failed AWS move). Each such bucket is recorded here (kind ``sky_storage``, id =
the bucket name) when its island is ordered, and released at the launcher's teardown and by
``yeto cleanup`` / ``yeto run down``, so ``ledger_check`` sees it open until it is gone.

Only a bucket whose name is exactly the name yeto derives from the run's cluster prefix is
recorded or deleted (:func:`yeto_owned`); a mount with a source (an existing bucket, the
checkpoint store) is never recorded: sky did not create it.

Release: ``sky storage delete <name>`` (sky's own record and credentials). When sky has no
record of it, a non-persistent storage was deleted by ``sky down`` of its cluster; an AWS
one is checked on S3 by its exact name (and emptied and deleted if it is still there).
"""

from __future__ import annotations

import re
from typing import Any, Callable

KIND = "sky_storage"
_RL_SUFFIX = re.compile(r".+-[0-9a-f]{8}-rl-(\d+)")


def yeto_owned(name: str | None, cluster_prefix: str | None) -> bool:
    """Whether ``name`` is the bucket name yeto derives for an island of ``cluster_prefix``
    (``_rl_checkpoint_storage_name(cluster_prefix, n)``, the prefix's hash included)."""
    from .storage import _rl_checkpoint_storage_name

    m = _RL_SUFFIX.fullmatch(str(name or ""))
    return bool(m and cluster_prefix) and name == _rl_checkpoint_storage_name(str(cluster_prefix), int(m.group(1)))


def created_storages(task: Any) -> list[tuple[str, dict[str, Any]]]:
    """``(name, attrs)`` of every storage mount of ``task`` that sky creates: a named
    ``sky.Storage`` without a source."""
    out = []
    for mount, storage in sorted((getattr(task, "storage_mounts", None) or {}).items()):
        name, source = getattr(storage, "name", None), getattr(storage, "source", None)
        if name and not source:
            out.append((str(name), {"mount": str(mount), "persistent": bool(getattr(storage, "persistent", False))}))
    return out


def record(ledger: Any, cluster: str, task: Any, *, cluster_prefix: str, cloud: str | None,
           region: str | None) -> list[str]:
    """Record the buckets sky will create for ``task`` (cluster ``cluster``) before it is
    ordered; returns their names. A storage this run's prefix does not name is skipped."""
    from ..cost.ledger import record_safely

    names = []
    for name, attrs in created_storages(task):
        if not yeto_owned(name, cluster_prefix):
            continue
        record_safely("created", KIND, name, name, ledger=ledger, cluster=cluster,
                      cluster_prefix=cluster_prefix, cloud=cloud, region=region, **attrs)
        names.append(name)
    return names


def _sky_not_found(error: BaseException) -> bool:
    return isinstance(error, ValueError) and "not found" in str(error).lower()


def _s3_client(region: str | None) -> Any:
    """An S3 client on AWS itself: the explicit regional endpoint wins over an
    ``AWS_ENDPOINT_URL`` the run exported for its R2 / MinIO store."""
    import boto3

    region = region or "us-east-1"
    return boto3.client("s3", region_name=region, endpoint_url=f"https://s3.{region}.amazonaws.com")


def _error_code(error: BaseException) -> str:
    response = getattr(error, "response", None) or {}
    return str((response.get("Error") or {}).get("Code") or "")


def delete_s3_bucket(name: str, client: Any) -> str:
    """Empty and delete the S3 bucket ``name``; ``"absent"`` when it does not exist,
    ``"deleted"`` when it did. Other errors raise."""
    try:
        client.head_bucket(Bucket=name)
    except Exception as error:  # noqa: BLE001 - botocore ClientError
        if _error_code(error) in ("404", "NoSuchBucket", "NotFound"):
            return "absent"
        raise
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=name):
        keys = [{"Key": o["Key"]} for o in page.get("Contents") or []]
        if keys:
            client.delete_objects(Bucket=name, Delete={"Objects": keys, "Quiet": True})
    client.delete_bucket(Bucket=name)
    return "deleted"


def delete(name: str, attrs: dict[str, Any], *, sky: Any = None,
           s3_client: Callable[[str | None], Any] | None = None) -> tuple[bool, str]:
    """Release the bucket ``name`` recorded with ``attrs``: ``(released, what happened)``."""
    if not yeto_owned(name, attrs.get("cluster_prefix")):
        return False, f"refused: {name!r} is not the bucket name of run prefix {attrs.get('cluster_prefix')!r}"
    if sky is None:
        import sky
    try:
        sky.stream_and_get(sky.storage_delete(name))
        return True, f"sky storage delete {name}"
    except Exception as error:  # noqa: BLE001
        if not _sky_not_found(error):
            return False, f"sky storage delete {name}: {type(error).__name__}: {str(error)[:200]}"
    cloud = str(attrs.get("cloud") or "").lower()
    if cloud != "aws":  # sky down deleted the non-persistent storage of its cluster
        return True, f"sky has no storage {name} (deleted with its cluster)"
    try:
        done = delete_s3_bucket(name, (s3_client or _s3_client)(attrs.get("region")))
    except Exception as error:  # noqa: BLE001
        return False, (f"S3 bucket {name}: {type(error).__name__}: {str(error)[:200]}; delete it by hand: "
                       f"aws s3 rb s3://{name} --force")
    return True, f"S3 bucket {name}: {done}"


def release(ledger: Any, name: str, attrs: dict[str, Any], *, by: str, sky: Any = None,
            s3_client: Callable[[str | None], Any] | None = None) -> bool:
    """:func:`delete`, then the ledger's release (or release_failed) of ``name``."""
    ok, what = delete(name, attrs, sky=sky, s3_client=s3_client)
    print(f"[launcher] {'released' if ok else 'NOT released'} sky storage: {what}", flush=True)
    try:
        if ok:
            ledger.released(KIND, name, name, by=by, result=what)
        else:
            ledger.release_failed(KIND, name, name, what)
    except Exception:  # noqa: BLE001 - the ledger never blocks a teardown
        pass
    return ok
