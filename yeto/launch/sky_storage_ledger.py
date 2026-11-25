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

What the bucket holds: the island's completed groups (``--rl-completed-groups-path``),
mounted with ``sync_on_reconstruction`` so a relaunch of the same cluster (the fleet
controller's recovery, which reuses the task and the name) re-syncs them. It is a
non-persistent storage (``persistent=False``): sky deletes it, contents included, when it
downs the cluster. The release keeps to that rule and never deletes more than sky would:

* only at a real teardown (``yeto down`` / the launcher's end / ``yeto cleanup``), never at a
  relaunch, never with ``--keep`` or a hands-off launcher; and only once every island
  cluster of the run is confirmed terminated (a cluster that may still mount it keeps it);
* a ``persistent`` storage is never deleted (sky down keeps it): ``release_skipped``;
* ``sky storage delete <name>`` while sky has its record: what sky down does for it;
* sky has no record: on another cloud than AWS, sky down deleted it with its cluster; on
  AWS the exact bucket name is looked at on S3 (an explicit regional AWS endpoint, never
  the run's R2 / MinIO endpoint, and never without a region): absent is released, an
  empty bucket (a launch that never got an instance) is deleted, a bucket that holds
  objects is NOT emptied: ``release_skipped`` (``nonempty``), left for a person.
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


def open_island_clusters(ledger: Any) -> list[str]:
    """The run's island sky clusters still open in ``ledger`` (not a sampler pool's): any
    of them may still mount an island bucket, so no bucket is touched while one is listed."""
    from ..cost.cleanup import pool_owned

    return sorted(str(c.id) for c in ledger.open_resources() if c.kind == "sky_cluster" and not pool_owned(c))


def _sky_not_found(error: BaseException) -> bool:
    return isinstance(error, ValueError) and "not found" in str(error).lower()


RELEASED, SKIPPED, FAILED = "released", "skipped", "failed"
_AWS_S3_ENDPOINT = re.compile(r"https://s3\.[a-z]{2}(-[a-z]+)+-\d\.amazonaws\.com")


def aws_s3_endpoint(region: str | None) -> str | None:
    """The regional AWS S3 endpoint of ``region``; None without a (well-formed) region."""
    endpoint = f"https://s3.{region}.amazonaws.com" if region else ""
    return endpoint if _AWS_S3_ENDPOINT.fullmatch(endpoint) else None


def _s3_client(region: str | None) -> Any:
    """An S3 client on AWS itself: the explicit regional endpoint wins over an
    ``AWS_ENDPOINT_URL`` the run exported for its R2 / MinIO store."""
    import boto3

    endpoint = aws_s3_endpoint(region)
    if endpoint is None:
        raise ValueError(f"no AWS region for the bucket (got {region!r})")
    return boto3.client("s3", region_name=region, endpoint_url=endpoint)


def _error_code(error: BaseException) -> str:
    response = getattr(error, "response", None) or {}
    return str((response.get("Error") or {}).get("Code") or "")


def delete_empty_s3_bucket(name: str, client: Any) -> str:
    """``"absent"`` when the S3 bucket ``name`` does not exist, ``"deleted"`` when it was
    empty and is now deleted, ``"nonempty"`` when it holds objects (left as it is: never
    emptied here). Other errors raise."""
    try:
        client.head_bucket(Bucket=name)
    except Exception as error:  # noqa: BLE001 - botocore ClientError
        if _error_code(error) in ("404", "NoSuchBucket", "NotFound"):
            return "absent"
        raise
    listed = client.list_objects_v2(Bucket=name, MaxKeys=1)
    if listed.get("KeyCount") or listed.get("Contents"):
        return "nonempty"
    client.delete_bucket(Bucket=name)
    return "deleted"


def delete(name: str, attrs: dict[str, Any], *, sky: Any = None,
           s3_client: Callable[[str | None], Any] | None = None) -> tuple[str, str]:
    """Release the bucket ``name`` recorded with ``attrs`` (its clusters are gone):
    ``(RELEASED | SKIPPED | FAILED, what happened)``."""
    if not yeto_owned(name, attrs.get("cluster_prefix")):
        return FAILED, f"refused: {name!r} is not the bucket name of run prefix {attrs.get('cluster_prefix')!r}"
    if attrs.get("persistent"):
        return SKIPPED, f"persistent storage {name}: sky down keeps it, so does yeto; delete it by hand if unused"
    if sky is None:
        import sky
    try:
        sky.stream_and_get(sky.storage_delete(name))
        return RELEASED, f"sky storage delete {name}"
    except Exception as error:  # noqa: BLE001
        if not _sky_not_found(error):
            return FAILED, f"sky storage delete {name}: {type(error).__name__}: {str(error)[:200]}"
    cloud = str(attrs.get("cloud") or "").lower()
    if cloud != "aws":  # sky down deleted the non-persistent storage of its cluster
        return RELEASED, f"sky has no storage {name} (deleted with its cluster)"
    region = attrs.get("region")
    if aws_s3_endpoint(region) is None:
        return FAILED, f"S3 bucket {name}: no AWS region recorded ({region!r}); not looked up"
    try:
        client = (s3_client or _s3_client)(region)
        endpoint = str(getattr(getattr(client, "meta", None), "endpoint_url", "") or "")
        if endpoint and endpoint.rstrip("/") != aws_s3_endpoint(region):
            return FAILED, f"S3 bucket {name}: refused, the client talks to {endpoint}, not AWS S3 in {region}"
        done = delete_empty_s3_bucket(name, client)
    except Exception as error:  # noqa: BLE001
        return FAILED, (f"S3 bucket {name}: {type(error).__name__}: {str(error)[:200]}; check it, then: "
                        f"aws s3 rb s3://{name} (AWS region {region})")
    if done == "nonempty":
        return SKIPPED, (f"S3 bucket {name} holds objects (sky has no record of it): not emptied; check it and "
                         f"delete it by hand: aws s3 rb s3://{name} --force (AWS region {region})")
    return RELEASED, f"S3 bucket {name}: {done}"


def release(ledger: Any, name: str, attrs: dict[str, Any], *, by: str, sky: Any = None,
            s3_client: Callable[[str | None], Any] | None = None) -> str:
    """:func:`delete`, then the ledger's release / release_skipped / release_failed."""
    status, what = delete(name, attrs, sky=sky, s3_client=s3_client)
    print(f"[launcher] sky storage {status}: {what}", flush=True)
    record_outcome(ledger, name, status, what, by=by)
    return status


def record_outcome(ledger: Any, name: str, status: str, what: str, *, by: str) -> None:
    try:
        if status == RELEASED:
            ledger.released(KIND, name, name, by=by, result=what)
        elif status == SKIPPED:
            ledger.release_skipped(KIND, name, name, what)
        else:
            ledger.release_failed(KIND, name, name, what)
    except Exception:  # noqa: BLE001 - the ledger never blocks a teardown
        pass
