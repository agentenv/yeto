"""Remote storage and mount paths: checkpoint stores, model stores and
initial-adapter paths.
"""

from __future__ import annotations

import hashlib
import os
import re
import shlex
from pathlib import PurePosixPath

from .syncer import island_scheduling_mode


ELASTIC_ISLAND_STATE_DIR = "~/yeto-rl/elastic-state"
# rl-multinode-island Q4 (C5): where a --rl-checkpoint-store bucket is mounted on the island
ELASTIC_CHECKPOINT_STORE_MOUNT = "~/yeto-checkpoint-store"  # not under the spot ~/yeto-rl mount
NODE0_LOCAL_CHECKPOINT_WARNING = (
    "[launcher] warning: --rl-checkpoint-store not set: the island state dir (journal, cuts, "
    "ledger) stays on node0's local disk; losing a node loses the consistent checkpoint and "
    "the island cannot be rebuilt on other machines (rl-multinode-island Q4)"
)


def modal_checkpoint_store(args) -> tuple[str, str] | None:
    """``(volume name, path in the container)`` for a ``modal-volume://`` store, None
    without a store. Any other store form on a Modal island is an error: Modal mounts no
    bucket and a container path does not outlive the container (it used to be dropped
    silently, rl-resume-from-checkpoint 1.2)."""
    value = getattr(args, "rl_checkpoint_store", None)
    if not value:
        return None
    from ..cloud.modal_ckpt_store import MODAL_VOLUME_SCHEME, parse_modal_volume_uri

    if not str(value).startswith(MODAL_VOLUME_SCHEME + "://"):
        raise ValueError(f"--rl-checkpoint-store {value!r} on a Modal island: Modal mounts neither "
                         "buckets nor host paths here; use modal-volume://NAME[/PREFIX] (a v1 Volume)")
    name, prefix = parse_modal_volume_uri(str(value))
    return name, MODAL_CHECKPOINT_STORE_MOUNT + (f"/{prefix}" if prefix else "")


MODAL_CHECKPOINT_STORE_MOUNT = "/root/yeto-checkpoint-store"


def resumes_in_new_container(args) -> bool:
    """A Modal re-run of the island (preemption) is a resume, not a failure, when the run
    has a non-elastic modal-volume:// store (rl-resume-from-checkpoint 2.5)."""
    store = str(getattr(args, "rl_checkpoint_store", None) or "")
    return (store.startswith("modal-volume://") and not getattr(args, "rl_elastic", False)
            and bool(getattr(args, "rl_single_island_no_sync", False)))


def droppable_rejoins_in_new_container(args, modal_cfg) -> bool:
    """rl-spot-cost-saving 5.2: on an elastic run a Modal re-run of a droppable island
    (reclaim / ``modal container stop``) is expected: it rejoins from the syncer base.
    Every other island keeps the fail-closed container guard."""
    if island_scheduling_mode(args) != "elastic":
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
def store_object_envs(args, cloud: str) -> dict[str, str]:
    """The environment of a sky island with a sample role (sampler or trainer)
    and an ``s3://`` store: the sample buffer, the trainer commands and the policy
    release pointer then go through the store's object API, not the bucket mount
    (``yeto.cloud.s3_store``). {} otherwise (Modal Volume, shared path, no role).
    Raises ValueError when the launcher has no store credentials."""
    from ..rl.engine import sample_roles

    store = str(getattr(args, "rl_checkpoint_store", None) or "")
    if cloud == "modal" or not store.startswith("s3://"):
        return {}
    if not (sample_roles.is_pair(args) or sample_roles.sample_role(args) != sample_roles.BOTH):
        return {}
    from ..cloud.s3_store import island_envs

    return island_envs(store)


RL_INITIAL_ADAPTER_PATH = "~/yeto-rl-initial-adapter"
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
    ``mount`` (``hf/<name>/<rev[:8]>/`` + MANIFEST.json, written by
    infra-drafts/fn-modal/dl_hf.py) through a symlinked HF hub cache, so the
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
