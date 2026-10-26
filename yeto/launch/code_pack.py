"""Launch side of the run code pack (rl-identity-tiering L9.4-L9.6).

With ``--rl-code-pack git-patch`` (also the ``auto`` default when the checkpoint
store is a ``scheme://`` URI) the launcher resolves the run's pack before it
writes the run manifest: the first start makes and stores one; every later start
(resume, a new launcher process for the same run) takes the pack the old manifest
names. The unpacked pack becomes the workdir of every island (sky ``workdir`` and the
Modal ``add_local_dir``), so a local tree that changes after the first start does not
reach replacement or resumed islands. Islands check the tree against
``--code-pack-digest`` before they start.

With ``--rl-code-pack release`` (L9.12) the run pack is the ``--rl-image`` itself: a
release image built by ``scripts/build_yeto_release_image.sh`` (yeto in an image
layer at ``/opt/yeto/src``, with ``~/sky_workdir`` linked to it). Its digest is the
pack digest; nothing is uploaded and no local tree is shipped (no sky ``workdir``,
no Modal ``add_local_dir``). Islands check the image tree against its release
manifest (``--code-pack-mode release``). The syncer head still runs from the local
tree; release mode covers the islands only.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import REPO_ROOT

CODE_PACK_OFF = "off"
CODE_PACK_GIT_PATCH = "git-patch"
CODE_PACK_AUTO = "auto"
CODE_PACK_RELEASE = "release"
CODE_PACK_CACHE = Path("~/.cache/yeto/codepacks")


def is_release(args) -> bool:
    pack = getattr(args, "code_pack", None)
    return pack is not None and pack.mode == CODE_PACK_RELEASE


def code_workdir(args) -> str | None:
    """The directory shipped as the island workdir: the run pack tree, else the repo;
    None in release mode (the code is in the island image)."""

    if is_release(args):
        return None
    return str(getattr(args, "code_pack_tree", None) or REPO_ROOT)


def code_pack_island_flag(args) -> str:
    pack = getattr(args, "code_pack", None)
    if pack is None:
        return ""
    return f" --code-pack-digest {pack.digest}" + (" --code-pack-mode release" if is_release(args) else "")


def code_pack_mode(args) -> str:
    """The run's pack mode. ``auto`` (the default) packs when the checkpoint store is a
    ``scheme://`` URI (bucket, ``modal-volume://``): the launcher can write the pack
    there. A path store is often visible on the islands only, so ``auto`` leaves it
    off; ``git-patch`` asks for it explicitly."""

    mode = getattr(args, "rl_code_pack", None) or CODE_PACK_OFF
    if mode != CODE_PACK_AUTO:
        return mode
    store = str(getattr(args, "rl_checkpoint_store", None) or "")
    return CODE_PACK_GIT_PATCH if "://" in store else CODE_PACK_OFF


def _previous_manifest(cluster_prefix: str) -> dict | None:
    from .. import runs

    path = runs.run_dir(cluster_prefix) / "run_manifest.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def _semantic_sha256(args) -> str | None:
    raw = getattr(args, "rl_algorithm_spec_json", None)
    if not raw:
        return None
    from ..rl.engine.algorithm import AlgorithmSpec

    return AlgorithmSpec.from_dict(json.loads(raw)).semantic_sha256()


def prepare_run_code_pack(args, *, store=None, repo: str | Path | None = None,
                          cache_root: str | Path | None = None) -> None:
    """Resolve the run pack and point the launch at it (sets ``args.code_pack``,
    ``args.code_pack_tree`` and re-attests ``args.source_sha256`` on the pack tree).
    No-op unless the pack mode (:func:`code_pack_mode`) is git-patch or release on an
    RL launch. In release mode ``args.code_pack_tree`` is None and
    ``args.source_sha256`` stays the launcher's value (the island checks the image
    tree against the image's release manifest instead)."""

    mode = code_pack_mode(args)
    upgrade = getattr(args, "rl_code_pack_upgrade", None)
    if mode == CODE_PACK_OFF or getattr(args, "training_mode", "sft") != "rl":
        if upgrade is not None:
            raise ValueError("--rl-code-pack-upgrade requires --rl-code-pack git-patch or release "
                             "(or auto with a bucket / modal-volume:// checkpoint store)")
        return
    from .. import codepack
    from ..provenance import source_tree_sha256

    if mode == CODE_PACK_RELEASE:
        if (getattr(args, "rl_backend", None) or "miles") != "miles":
            raise ValueError("--rl-code-pack release needs the miles backend (the release "
                             "image is built on the miles-ports engine image)")
        try:
            resolved = codepack.resolve_run_pack(
                previous=_previous_manifest(args.cluster_prefix), repo=repo or REPO_ROOT,
                store=None, cache_root=Path(cache_root or CODE_PACK_CACHE).expanduser(),
                semantic_sha256=_semantic_sha256(args), upgrade_reason=upgrade,
                release_image=str(getattr(args, "rl_image", None) or ""))
        except codepack.CodePackError as e:
            raise ValueError(f"--rl-code-pack release: {e}") from None
        args.code_pack = resolved.pack
        args.code_pack_tree = None
        return

    if store is None:
        store_uri = getattr(args, "rl_checkpoint_store", None)
        if not store_uri:
            raise ValueError("--rl-code-pack git-patch requires --rl-checkpoint-store "
                             "(the pack is stored next to the run's checkpoints)")
        store = codepack.pack_store_for(store_uri)
    resolved = codepack.resolve_run_pack(
        previous=_previous_manifest(args.cluster_prefix), repo=repo or REPO_ROOT,
        store=store, cache_root=Path(cache_root or CODE_PACK_CACHE).expanduser(),
        semantic_sha256=_semantic_sha256(args), upgrade_reason=upgrade)
    # sky syncs a workdir minus its .gitignore matches; the pack holds no ignored
    # untracked file, and .skyignore (when present) replaces .gitignore, so a
    # force-added ignored file still ships. The extra file is not part of the digest.
    skyignore = resolved.tree / ".skyignore"
    if not skyignore.exists():
        skyignore.write_text("/.git\n", encoding="utf-8")
    args.code_pack = resolved.pack
    args.code_pack_tree = str(resolved.tree)
    args.source_sha256 = source_tree_sha256(resolved.tree / "yeto")
