"""Remote storage and mount paths: checkpoint stores, model stores and
initial-adapter paths.
"""

from __future__ import annotations

import hashlib
import os
import re
import shlex
from pathlib import PurePosixPath

from .syncer import island_pool


ELASTIC_ISLAND_STATE_DIR = "~/yeto-rl/elastic-state"
# rl-multinode-island Q4 (C5): where a --rl-checkpoint-store bucket is mounted on the island
ELASTIC_CHECKPOINT_STORE_MOUNT = "~/yeto-checkpoint-store"  # not under the spot ~/yeto-rl mount
NODE0_LOCAL_CHECKPOINT_WARNING = (
    "[launcher] warning: --rl-checkpoint-store not set: the island state dir (journal, cuts, "
    "ledger) stays on node0's local disk; losing a node loses the consistent checkpoint and "
    "the island cannot be rebuilt on other machines (rl-multinode-island Q4)"
)


def modal_checkpoint_store(args) -> tuple[str, str] | None:
    """``(volume name, path in the container)`` for a ``modal-volume://`` store; None
    without a store and for an ``s3://`` store (no Volume: the island reaches it by
    object API, :func:`store_object_envs`). Any other store form on a Modal island is
    an error: Modal mounts no bucket and a container path does not outlive the
    container (it used to be dropped silently, rl-resume-from-checkpoint 1.2)."""
    value = getattr(args, "rl_checkpoint_store", None)
    if not value:
        return None
    if is_object_store_uri(value):
        return None
    from ..cloud.modal_ckpt_store import MODAL_VOLUME_SCHEME, parse_modal_volume_uri

    if not str(value).startswith(MODAL_VOLUME_SCHEME + "://"):
        raise ValueError(f"--rl-checkpoint-store {value!r} on a Modal island: Modal mounts neither "
                         "buckets nor host paths here; use modal-volume://NAME[/PREFIX] (a v1 Volume) "
                         "or s3://BUCKET[/PREFIX] (an S3 or S3-compatible store reached by object API)")
    name, prefix = parse_modal_volume_uri(str(value))
    return name, MODAL_CHECKPOINT_STORE_MOUNT + (f"/{prefix}" if prefix else "")


MODAL_CHECKPOINT_STORE_MOUNT = "/root/yeto-checkpoint-store"


def modal_store_secret_name(args, spec=None) -> str | None:
    """``--modal-store-secret NAME``: the Modal Secret that holds the object store
    credentials (``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``, and for an
    S3-compatible store ``AWS_ENDPOINT_URL`` / ``AWS_REGION``), attached to the
    island function so the values never pass through the launcher. None when
    unset. Refused for a non-Modal island when ``spec`` is given (a sky island
    cannot read a Modal Secret)."""
    name = getattr(args, "modal_store_secret", None)
    if not name:
        return None
    if spec is not None and getattr(spec, "cloud", None) != "modal":
        raise ValueError(f"--modal-store-secret attaches a Modal Secret to a Modal island; "
                         f"island cloud {getattr(spec, 'cloud', None)!r} cannot use it")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", str(name)):
        raise ValueError(f"--modal-store-secret {name!r} is not a Modal Secret name")
    return str(name)


# --rl-checkpoint-store schemes a Modal island reaches without a mount: the whole store
# (cuts, LATEST, sample buffer, releases) goes through the object API from a local stage
# (yeto.cloud.s3_store STAGE_ENV); the credentials come from --modal-store-secret or the
# launcher's store credential.
MODAL_OBJECT_STORE_SCHEMES = ("s3",)


def is_object_store_uri(value) -> bool:
    return str(value).partition("://")[0] in MODAL_OBJECT_STORE_SCHEMES and "://" in str(value)


def resumes_in_new_container(args) -> bool:
    """A Modal re-run of the island (preemption, or a whole clustered restart) is a
    resume, not a failure, when the run has a non-elastic modal-volume:// or s3://
    store (rl-resume-from-checkpoint 2.5): the store carries LATEST and the new
    container starts from the newest complete cut. A store of
    ``--rl-checkpoint-store-export-only`` holds no resume state: a re-run is a failure."""
    store = str(getattr(args, "rl_checkpoint_store", None) or "")
    return ((store.startswith("modal-volume://") or is_object_store_uri(store))
            and not getattr(args, "rl_elastic", False)
            and bool(getattr(args, "rl_single_island_no_sync", False))
            and not getattr(args, "rl_checkpoint_store_export_only", False))


def droppable_rejoins_in_new_container(args, modal_cfg) -> bool:
    """rl-spot-cost-saving 5.2: on an island-pool run a Modal re-run of a droppable island
    (reclaim / ``modal container stop``) is expected: it rejoins from the syncer base.
    Every other island keeps the fail-closed container guard."""
    if not island_pool(args):
        return False
    from ..cloud import droppable

    doc = droppable.env_role(dict(getattr(modal_cfg, "envs", None) or {}))
    return bool(doc) and doc.get("role") == droppable.DROPPABLE


def rl_checkpoint_store_plan(args) -> tuple[str, str | None] | None:
    """``(path on the island, bucket URI or None)`` for ``--rl-checkpoint-store``:
    a ``scheme://`` URI is mounted at :data:`ELASTIC_CHECKPOINT_STORE_MOUNT` (sky
    Storage MOUNT, persistent); an absolute / ``~/`` path is used as is (a share
    the replacement machine also mounts). None when the flag is not given."""
    value = getattr(args, "rl_checkpoint_store", None)
    if not value:
        return None
    value = str(value)
    if value.startswith("modal-volume://"):
        from ..cloud.modal_ckpt_store import parse_modal_volume_uri

        _name, prefix = parse_modal_volume_uri(value)
        return MODAL_CHECKPOINT_STORE_MOUNT + (f"/{prefix}" if prefix else ""), None
    if value.startswith("oss://"):  # no mount: a local stage mirrored by object (yeto.cloud.oss_store)
        from ..cloud.oss_store import DEFAULT_STAGE, parse_oss_uri

        bucket, prefix = parse_oss_uri(value)
        return f"{DEFAULT_STAGE}/{bucket}" + (f"/{prefix}" if prefix else ""), None
    if "://" in value:
        scheme, _, rest = value.partition("://")
        if not scheme.isalnum() or not rest.strip("/") or rest.startswith("/"):
            raise ValueError(f"--rl-checkpoint-store {value!r} is not a bucket URI (scheme://bucket[/prefix])")
        # sky Storage MOUNT only mounts a bucket root ("MOUNT mode does not support
        # mounting specific files"): mount the bucket, use the prefix as a subdir.
        bucket, _, prefix = rest.strip("/").partition("/")
        path = ELASTIC_CHECKPOINT_STORE_MOUNT + (f"/{prefix.strip('/')}" if prefix.strip("/") else "")
        return path, f"{scheme}://{bucket}"
    if not (value.startswith("/") or value.startswith("~/")) or ".." in PurePosixPath(value).parts:
        raise ValueError("--rl-checkpoint-store must be a bucket URI or an absolute / ~/ path on the island")
    return value.rstrip("/") or "/", None
def _store_root(value: str) -> str:
    """The part of a store URI the island mounts as a whole: ``modal-volume://NAME``
    or ``scheme://bucket``."""
    scheme, _, rest = str(value).partition("://")
    return f"{scheme}://{rest.strip('/').partition('/')[0]}"


def island_store_path(args, value: str, *, flag: str) -> str:
    """The island path of a second store option (``flag``, e.g.
    ``--rl-peer-recovery-store``) given as a path or a URI.

    The island mounts only the run's ``--rl-checkpoint-store`` (Modal: its Volume at
    :data:`MODAL_CHECKPOINT_STORE_MOUNT`; sky: its bucket at
    :data:`ELASTIC_CHECKPOINT_STORE_MOUNT`). So a URI must name the same Volume or
    bucket, and becomes the same path :func:`rl_checkpoint_store_plan` gives its
    prefix. Before this check the URI reached the island as it was and the island's
    ``CheckpointStore`` took it for a relative directory on its local disk (a
    donor's 473 GB snapshot then never left the donor container).
    A path must be absolute or ``~/``; with a ``modal-volume://`` checkpoint store
    it must lie under the Volume mount (any other container path is local)."""
    value = str(value)
    store = str(getattr(args, "rl_checkpoint_store", None) or "")
    if "://" in value:
        if not store or "://" not in store or _store_root(value) != _store_root(store):
            raise ValueError(
                f"{flag} {value!r}: an island reaches a Volume or bucket only through the mount of "
                f"--rl-checkpoint-store ({store or 'not set'}); name the same "
                f"{_store_root(value).partition('://')[0]} store, or give a path under its mount")
        plan = rl_checkpoint_store_plan(type("StoreArgs", (), {"rl_checkpoint_store": value})())
        return plan[0]
    if not (value.startswith("/") or value.startswith("~/")) or ".." in PurePosixPath(value).parts:
        raise ValueError(f"{flag} {value!r} must be a store URI or an absolute / ~/ path on the island")
    path = value.rstrip("/") or "/"
    if store.startswith("modal-volume://"):
        mount = PurePosixPath(MODAL_CHECKPOINT_STORE_MOUNT)
        if PurePosixPath(path) != mount and mount not in PurePosixPath(path).parents:
            raise ValueError(
                f"{flag} {value!r}: on a Modal island only {MODAL_CHECKPOINT_STORE_MOUNT} (the "
                "--rl-checkpoint-store Volume) is shared between containers; a path elsewhere is "
                "the container's local disk")
    return path


def store_object_envs(args, cloud: str, num_nodes: int = 1) -> dict[str, str]:
    """The environment of a sky island with a sample role (sampler or trainer)
    and an ``s3://`` store: the sample buffer, the trainer commands and the policy
    release pointer then go through the store's object API, not the bucket mount
    (``yeto.cloud.s3_store``); on a one-node island the round cuts and policy
    releases too (staged on local disk, uploaded by object). An ``oss://`` store
    has no mount: every island that uses it gets the OSS store settings
    (``yeto.cloud.oss_store``). {} otherwise (Modal Volume, shared path, no role).
    Raises ValueError when the launcher has no store credentials."""
    from ..rl.engine import sample_roles

    store = str(getattr(args, "rl_checkpoint_store", None) or "")
    if cloud == "modal":
        if not is_object_store_uri(store):
            return {}  # a Modal Volume (or no store): nothing goes by object API
        from ..cloud.s3_store import island_envs

        # No bucket mount on Modal: every node stages the store on its local disk and
        # moves the whole store by object (one stage per container). The credentials
        # ride the named Secret when --modal-store-secret is given; else the launcher's
        # store credential is shipped like on sky.
        return island_envs(store, store_client_environ(args), stage=True,
                           credentials=modal_store_secret_name(args) is None)
    if store.startswith("oss://"):
        from ..cloud.oss_store import island_envs as oss_island_envs

        return oss_island_envs(store)
    if not store.startswith("s3://"):
        return {}
    if not (sample_roles.is_pair(args) or sample_roles.sample_role(args) != sample_roles.BOTH):
        return {}
    from ..cloud.s3_store import island_envs

    return island_envs(store, store_client_environ(args), stage=int(num_nodes) == 1)


def store_client_environ(args, environ=None) -> dict[str, str]:
    """The environment the S3 store client of the head and the islands reads, with
    ``--rl-object-store-endpoint`` applied: the endpoint, and ``region=auto`` when an
    endpoint is set and no region is configured (what Cloudflare R2 expects; AWS S3
    ignores it). ``environ`` defaults to the process environment."""
    from ..cloud.s3_store import ENDPOINT_ENV, PROFILE_ENV, REGION_ENV, store_endpoint, store_region

    env = dict(os.environ if environ is None else environ)
    if getattr(args, "rl_object_store_profile", None):
        env[PROFILE_ENV] = str(args.rl_object_store_profile)
    endpoint = getattr(args, "rl_object_store_endpoint", None) or env.get("AWS_ENDPOINT_URL")
    if not endpoint and not env.get(ENDPOINT_ENV) and env.get(PROFILE_ENV):
        endpoint = store_endpoint(env)  # the profile's endpoint_url
        if endpoint and store_region(env):
            env.setdefault(REGION_ENV, store_region(env))
    if endpoint and not env.get(ENDPOINT_ENV):
        env[ENDPOINT_ENV] = str(endpoint)
        env.setdefault(REGION_ENV, env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION") or "auto")
    return env


def apply_object_store_endpoint(args) -> None:
    """Export ``--rl-object-store-endpoint`` into the launcher's own environment
    (:data:`~yeto.cloud.s3_store.ENDPOINT_ENV` and ``AWS_ENDPOINT_URL``), so every
    launcher-side boto3 client (the head's object API, the code pack store) talks to
    the same endpoint as the islands. A variable already set wins."""
    from ..cloud.s3_store import ENDPOINT_ENV, PROFILE_ENV, REGION_ENV

    env = store_client_environ(args)
    for name in (ENDPOINT_ENV, REGION_ENV, PROFILE_ENV):
        if env.get(name):
            os.environ.setdefault(name, env[name])
    if env.get(ENDPOINT_ENV):
        os.environ.setdefault("AWS_ENDPOINT_URL", env[ENDPOINT_ENV])


# --rl-checkpoint-store schemes the head reads and writes by object or volume API
# (cuts, sample buffer, trainer commands): what a trainer scheduler needs
HEAD_READABLE_STORE_SCHEMES = ("modal-volume", "s3", "oss")


def store_unreachable_reason(store: str, cloud: str) -> str | None:
    """Why an island on ``cloud`` cannot use the ``--rl-checkpoint-store`` URI
    ``store``; None when it can. A static check on the store kind only: whether
    the island's network reaches the endpoint is checked on the node (preflight),
    not here."""
    scheme = str(store).partition("://")[0] if "://" in str(store) else ""
    if scheme == "modal-volume":
        return None if cloud == "modal" else "a Modal Volume is mounted only into Modal containers"
    if cloud == "modal":
        if scheme in MODAL_OBJECT_STORE_SCHEMES:
            return None  # reached by object API from a local stage (store_object_envs)
        return ("Modal mounts neither buckets nor host paths here; use modal-volume://NAME[/PREFIX] "
                "or s3://BUCKET[/PREFIX]")
    if scheme in ("s3", "oss"):
        return None
    if not scheme:
        return "a host path is shared only by the hosts that mount it"
    return f"no object API for {scheme}:// stores"


RL_INITIAL_ADAPTER_PATH = "~/yeto-rl-initial-adapter"
# the launcher's --rl-expert-selection file, mounted into every island (Kimi K3.1)
RL_INIT_RELEASE_DIR = "~/yeto-init-release"  # --rl-init-from-release: the prepared model directory
RL_EXPERT_SELECTION_PATH = "~/yeto-rl-expert-selection.json"  # not under ~/yeto-rl: a Volume mount on Modal
RL_HEAD_INITIAL_ADAPTER_PATH = "~/yeto-rl-initial-adapter-src"


def _rl_checkpoint_mount(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not (value.startswith("~/") or value.startswith("/"))
        or value.endswith("/")
        or ".." in path.parts
        or str(path.parent) in {".", "~", "/"}
    ):
        raise ValueError(
            "Spot RL --rl-completed-groups-path must be a file in an "
            "absolute or ~/ subdirectory"
        )
    return str(path.parent)


def _rl_checkpoint_storage_name(cluster_prefix: str, learner_id: int) -> str:
    stem = re.sub(r"[^a-z0-9-]+", "-", cluster_prefix.lower()).strip("-") or "yeto"
    suffix = f"-{hashlib.sha256(cluster_prefix.encode()).hexdigest()[:8]}-rl-{learner_id}"
    return stem[: 63 - len(suffix)].rstrip("-") + suffix


MODEL_STORE_MOUNT = "/mnt/yeto-models"


def model_store_filesystem(uri, cloud, region):
    """Nebius filesystem id for ``--model-store nebius-fs://<id>`` on a
    ``nebius@region`` island, else None (with a WARNING when a store was
    asked for but cannot be attached there)."""
    if not uri:
        return None
    scheme, sep, fs_id = uri.partition("://")
    if scheme != "nebius-fs" or not sep or not fs_id.startswith("computefilesystem-"):
        raise ValueError(f"--model-store: expected nebius-fs://computefilesystem-..., got {uri!r}")
    if cloud != "nebius" or not region:
        print(f"[launcher] WARNING: --model-store {uri} needs a nebius@<region> island "
              f"(got {cloud}@{region}); models download from the Hub")
        return None
    return fs_id.rstrip("/")


def model_store_sky_config(fs_id, region):
    """Client-side sky config override that attaches the store filesystem.
    sky 0.13 reads Nebius filesystems only from config
    (nebius.region_configs.<region>.filesystems, sky/clouds/nebius.py), and
    mounts each one with virtiofs during node setup."""
    from sky import skypilot_config

    regional = dict(skypilot_config.get_nested(("nebius", "region_configs", region), {}) or {})
    regional["filesystems"] = [
        # READ_WRITE: Nebius rejects READ_ONLY filesystem attachments
        # ("readOnly mode attachment is not supported for filesystems yet",
        # 2026-10-05).  Nothing writes here: HF_HUB_CACHE points at the store
        # only on a completed-marker hit, when every file is already cached.
        {"filesystem_id": fs_id, "attach_mode": "READ_WRITE", "mount_path": MODEL_STORE_MOUNT}
    ]
    return {"nebius": {"region_configs": {region: regional}}}


def model_store_env(model, revision, mount=MODEL_STORE_MOUNT):
    """Shell: point HF_HUB_CACHE at the store when it holds a COMPLETED
    snapshot of exactly ``model@revision`` (marker written by
    scripts/populate_nebius_model_store.sh after a size check); otherwise
    WARN and leave the Hub download path untouched.  Never fails."""
    if not revision:
        return (f"echo '[yeto-model-store] WARNING: no --model-revision; not using "
                f"{mount}, downloading {model} from the Hub' >&2")
    marker = f"{mount}/yeto-complete/{model.replace('/', '--')}@{revision}.json"
    snap = f"{mount}/hub/models--{model.replace('/', '--')}/snapshots/{revision}"
    return (
        f"if [ -f {shlex.quote(marker)} ] && [ -d {shlex.quote(snap)} ]; then "
        f"export HF_HUB_CACHE={mount}/hub YETO_MODEL_STORE_HIT=1; "
        f"else echo '[yeto-model-store] WARNING: {mount} has no completed {model}@{revision}; "
        "downloading from the Hub' >&2; fi"
    )


def modal_volume_store_env(model, revision, mount=MODEL_STORE_MOUNT, cache="/root/yeto-hub"):
    """Shell: serve ``model@revision`` from a Modal model Volume mounted at
    ``mount`` (``hf/<name>/<rev[:8]>/`` + MANIFEST.json, written by the
    model upload tool) through a symlinked HF hub cache, so the
    learner's snapshot_download finds every file locally.  Never fails: on a
    mismatch it WARNs and leaves the Hub download path untouched."""
    if not revision or len(revision) != 40:
        return (f"echo '[yeto-model-volume] WARNING: needs a full 40-hex --model-revision; "
                f"downloading {model} from the Hub' >&2")
    snap = f"{mount}/hf/{model.split('/')[-1]}/{revision[:8]}"
    repo_dir = f"{cache}/models--{model.replace('/', '--')}"
    check = (f"python3 -c \"import json,sys; m=json.load(open(sys.argv[1])); "
             f"sys.exit(0 if m.get('revision')==sys.argv[2] and m.get('repo')==sys.argv[3] and m.get('all_ok') else 1)\" "
             f"{shlex.quote(snap + '/MANIFEST.json')} {revision} {shlex.quote(model)}")
    return (
        f"if [ -f {shlex.quote(snap + '/MANIFEST.json')} ] && {check}; then "
        f"mkdir -p {shlex.quote(repo_dir)}/snapshots {shlex.quote(repo_dir)}/refs && "
        f"ln -sfn {shlex.quote(snap)} {shlex.quote(repo_dir + '/snapshots/' + revision)} && "
        f"echo {revision} > {shlex.quote(repo_dir + '/refs/main')} && "
        f"export HF_HUB_CACHE={cache} YETO_MODEL_STORE_HIT=1 && "
        f"echo '[yeto-model-volume] serving {model}@{revision} from {snap}' >&2; "
        f"else echo '[yeto-model-volume] WARNING: {snap} has no matching MANIFEST; "
        "downloading from the Hub' >&2; fi"
    )


def nebius_baked_image_id(image, cloud, region, baked=None):
    """``image_id`` for an RL island: the docker image alone, or on Nebius a
    ``{region: computeimage-..., "docker": image}`` dict when a VM image with
    exactly that docker digest pre-pulled was baked for ``region``
    (``NEBIUS_BAKED_IMAGES``; COLDSTART-PLAN.md #3).  A region without an
    explicit ``@region`` keeps the plain image: a region-keyed dict would pin
    sky to the baked regions.  A baked entry for a *different* digest is never
    used (it would add a non-default base disk and still pull everything); it
    only produces a WARNING so the stale bake is noticed."""
    if baked is None:
        from yeto.rl import NEBIUS_BAKED_IMAGES as baked
    if cloud != "nebius" or not region or not isinstance(image, str):
        return image
    if os.environ.get("YETO_NEBIUS_NO_BAKED_IMAGE"):  # diagnostic opt-out: stock VM image + docker pull
        print(f"[launcher] nebius/{region}: YETO_NEBIUS_NO_BAKED_IMAGE set; using the stock VM image")
        return image
    if not image.startswith("docker:") or "@sha256:" not in image:
        return image
    digest = "sha256:" + image.rsplit("@sha256:", 1)[1]
    vm_image = baked.get(digest, {}).get(region)
    if vm_image:
        print(f"[launcher] nebius/{region}: VM image {vm_image} has {digest[:19]} pre-pulled")
        return {region: vm_image, "docker": image}
    stale = sorted(d for d, regions in baked.items() if region in regions)
    if stale:
        print(
            f"[launcher] WARNING: nebius/{region} has baked VM images only for "
            f"{', '.join(d[:19] for d in stale)}, not {digest[:19]}; using the stock "
            "image and a full docker pull (re-run scripts/bake_nebius_image.sh)"
        )
    return image
