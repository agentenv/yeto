"""Which of this machine's cloud credential files sky copies onto a yeto node.

sky copies the local credential file of every cloud it knows onto every node it
provisions, whatever cloud the node is on (``sky.check.get_cloud_credential_file_mounts``):
``~/.aws/credentials`` with every profile in it, ``~/.kube/config``, the Nebius token
files and so on. A cloud is left out only when the sky config sets its
``remote_identity`` to ``NO_UPLOAD`` (AWS's default is ``LOCAL_CREDENTIALS``). Anyone
who can run code on such a node can use those keys.

A yeto node does not need them. A trainer or sampler island reaches the run store
with the store's own credential (``YETO_RL_STORE_S3_*`` in the task's secrets), pulls a
private image with the registry login in the task's secrets (an ECR login is the
12-hour password ``aws ecr get-login-password`` prints), and the syncer VM's tape
uploader uses the store credential too. A head VM that launches islands gets the
credentials it needs as explicit file mounts (``launch.credentials.head_cloud_credentials``),
not through this upload.

So every yeto ``sky.launch`` is submitted with ``remote_identity: NO_UPLOAD`` for each
cloud the sky config accepts it for, except:

* the node's own cloud when that is not AWS: sky's autostop / autodown on the node
  calls that cloud's API with that file. An AWS node does not need the file: sky
  gives it an IAM instance profile (``skypilot-v1``, or :data:`AWS_INSTANCE_PROFILE_ENV`);
* AWS on a non-AWS node that mounts an AWS S3 bucket with sky Storage: the mount
  reads ``~/.aws/credentials`` there. The launch keeps the upload for that node and
  prints a warning naming the risk;
* a cloud whose ``remote_identity`` this machine's sky config already sets;
* :data:`UPLOAD_ENV` = ``1``: sky's default, every file uploaded (printed).

The request snapshots the client config when it is submitted, so the override only
has to hold for the ``sky.launch`` call (the pattern of
``storage.model_store_sky_config``). The shared API server's own config is not changed.
"""

from __future__ import annotations

import contextlib
import functools
import os
import re
import sys
from typing import Any, Callable, Iterator, Mapping

NO_UPLOAD = "NO_UPLOAD"
AWS = "aws"
# "1": sky's default (every local cloud credential file goes to every node)
UPLOAD_ENV = "YETO_SKY_UPLOAD_CLOUD_CREDENTIALS"
# an IAM instance profile for nodes that can only land on AWS, in place of sky's
# skypilot-v1 profile (EC2 and S3 full access); see docs/CLOUDS.md for a minimal policy
AWS_INSTANCE_PROFILE_ENV = "YETO_SKY_AWS_INSTANCE_PROFILE"
_INSTANCE_PROFILE_NAME = re.compile(r"[\w+=,.@-]{1,128}")


@functools.lru_cache(maxsize=1)
def identity_clouds() -> tuple[str, ...]:
    """The clouds of sky's registry whose config takes ``remote_identity`` (sky's schema)."""
    from sky.utils import registry, schemas

    props = schemas.get_config_schema().get("properties") or {}
    return tuple(sorted(name for name in registry.CLOUD_REGISTRY
                        if "remote_identity" in ((props.get(name) or {}).get("properties") or {})))


def task_clouds(task) -> set[str] | None:
    """Lower-case clouds the task's resources can land on; None when any of them
    leaves the cloud open (or the task has no resources)."""
    clouds: set[str] = set()
    found = getattr(task, "resources", None) or ()
    for resources in (found if isinstance(found, (set, frozenset, list, tuple)) else (found,)):
        cloud = getattr(resources, "cloud", None)
        if cloud is None:
            return None
        clouds.add(str(cloud).lower())
    return clouds or None


def _store_value(store_type) -> str:
    return str(getattr(store_type, "value", store_type)).upper()


def aws_s3_mounts(task, clouds: set[str] | None) -> list[str]:
    """Mount paths of the task's sky Storage that may read ``~/.aws/credentials`` on a
    non-AWS node: an ``s3://`` source, an S3 store, or no source and no store (sky picks
    the node cloud's store only when that cloud's storage is enabled on the API server,
    else the first enabled one, which may be AWS S3). [] on a node that can only be AWS."""
    if clouds == {AWS}:
        return []
    found = []
    for mount, storage in sorted((getattr(task, "storage_mounts", None) or {}).items()):
        source = getattr(storage, "source", None)
        sources = source if isinstance(source, (list, tuple)) else [source]
        stores = getattr(storage, "stores", None) or {}
        if (any(isinstance(s, str) and s.startswith("s3://") for s in sources)
                or any(_store_value(t) == "S3" for t in stores)
                or (source is None and not stores)):
            found.append(str(mount))
    return found


def _configured(cloud: str) -> bool:
    from sky import skypilot_config

    return skypilot_config.get_nested((cloud, "remote_identity"), None) is not None


def identity_config(task, name: str, environ: Mapping[str, str] | None = None, *,
                    log: Callable[[str], None] = print,
                    configured: Callable[[str], bool] | None = None,
                    clouds_with_identity: tuple[str, ...] | None = None) -> dict[str, Any]:
    """The client-side sky config override of one launch of ``task`` as cluster
    ``name`` (module docstring); {} when nothing is left out. Raises ValueError for a
    malformed :data:`AWS_INSTANCE_PROFILE_ENV`."""
    environ = os.environ if environ is None else environ
    if environ.get(UPLOAD_ENV) == "1":
        log(f"[launcher] WARNING: {name}: {UPLOAD_ENV}=1: sky copies this machine's cloud credential files "
            "(~/.aws/credentials with every profile in it, and the other clouds') onto its nodes; anyone "
            "who can run code there can use them")
        return {}
    profile = (environ.get(AWS_INSTANCE_PROFILE_ENV) or "").strip()
    if profile and not _INSTANCE_PROFILE_NAME.fullmatch(profile):
        raise ValueError(f"{AWS_INSTANCE_PROFILE_ENV}={profile!r} is not an IAM instance profile name")
    configured = _configured if configured is None else configured
    clouds = task_clouds(task)
    own = set() if clouds is None else clouds - {AWS}
    if clouds is None:
        # the node's cloud is open: its own credential may be the one autodown needs,
        # so only AWS (an AWS node has its instance profile) is left out
        candidates: tuple[str, ...] = (AWS,)
    else:
        candidates = clouds_with_identity if clouds_with_identity is not None else identity_clouds()
    config: dict[str, Any] = {}
    for cloud in candidates:
        if cloud in own or configured(cloud):
            continue
        if cloud == AWS:
            mounts = aws_s3_mounts(task, clouds)
            if mounts:
                where = "/".join(sorted(own)) if own else "non-AWS"
                log(f"[launcher] WARNING: {name}: sky copies this machine's ~/.aws/credentials (every profile "
                    f"in it) onto the {where} nodes: they mount the AWS S3 bucket(s) at {', '.join(mounts)} "
                    "with sky Storage, which reads that file there. Anyone who can run code on those nodes "
                    "can use those keys. To avoid it: run the island on AWS (its instance profile mounts "
                    "the bucket), or use a store on an S3-compatible endpoint with a bucket-scoped key "
                    "(reached by object API, no mount)")
                continue
            if profile and clouds == {AWS}:
                config[AWS] = {"remote_identity": profile}
                continue
            if profile:
                log(f"[launcher] {name}: {AWS_INSTANCE_PROFILE_ENV} applies only to a node that can only "
                    "land on AWS; this one can land elsewhere too, so an AWS node gets sky's own profile")
        config[cloud] = {"remote_identity": NO_UPLOAD}
    return config


@contextlib.contextmanager
def launch_identity(task, name: str, environ: Mapping[str, str] | None = None, *,
                    log: Callable[[str], None] = print) -> Iterator[dict[str, Any]]:
    """Hold :func:`identity_config` in the client sky config around one ``sky.launch``
    submit; yields the override. Without the real sky package (a test double in its
    place) nothing is overridden."""
    loaded = sys.modules.get("sky")
    if loaded is not None and not hasattr(loaded, "__path__"):
        yield {}
        return
    try:
        from sky import skypilot_config
        from sky.utils import registry, schemas  # noqa: F401 - identity_clouds reads them
    except ImportError:
        yield {}
        return
    config = identity_config(task, name, environ, log=log)
    if not config:
        yield config
        return
    with skypilot_config.override_skypilot_config(config):
        yield config
