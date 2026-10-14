"""Launch side of the run code pack (rl-identity-tiering L9.4-L9.6).

With ``--rl-code-pack git-patch`` the launcher resolves the run's pack before it
writes the run manifest: the first start makes and stores one; every later start
(resume, a new launcher process for the same run) takes the pack the old manifest
names. The unpacked pack becomes the workdir of every island (sky ``workdir`` and the
Modal ``add_local_dir``), so a local tree that changes after the first start does not
reach replacement or resumed islands. Islands check the tree against
``--code-pack-digest`` before they start.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import REPO_ROOT

CODE_PACK_OFF = "off"
CODE_PACK_CACHE = Path("~/.cache/yeto/codepacks")


def code_workdir(args) -> str:
    """The directory shipped as the island workdir: the run pack tree, else the repo."""

    return str(getattr(args, "code_pack_tree", None) or REPO_ROOT)


def code_pack_island_flag(args) -> str:
    pack = getattr(args, "code_pack", None)
    return f" --code-pack-digest {pack.digest}" if pack is not None else ""


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
    No-op unless ``--rl-code-pack git-patch`` on an RL launch."""

    mode = getattr(args, "rl_code_pack", None) or CODE_PACK_OFF
    upgrade = getattr(args, "rl_code_pack_upgrade", None)
    if mode == CODE_PACK_OFF or getattr(args, "training_mode", "sft") != "rl":
        if upgrade is not None:
            raise ValueError("--rl-code-pack-upgrade requires --rl-code-pack git-patch")
        return
    from .. import codepack
    from ..provenance import source_tree_sha256

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
