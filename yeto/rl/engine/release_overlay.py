"""A servable checkpoint directory from a base checkpoint and a partial release.

A partial-experts release (``manifest.partial_experts``, the Kimi-K3.1 recipe) holds
only the trained experts' tensors under the base checkpoint's names, plus the base's
config / tokenizer files and an index that lists those tensors. An engine that starts
on a directory (the convertible node going back to sampling, a restart reload) needs
every weight, so :func:`overlay_release` builds a directory of symlinks: the base's
files, the release's tensor files under ``release-`` names, and a merged
``model.safetensors.index.json`` whose entries for the released tensors point at the
release files. Nothing is copied.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

INDEX_FILE = "model.safetensors.index.json"
RELEASE_PREFIX = "release-"
PARTIAL_EXPERTS = "partial_experts"


class ReleaseOverlayError(ValueError):
    pass


def is_partial_release(manifest: Mapping[str, Any] | None) -> bool:
    return bool((manifest or {}).get(PARTIAL_EXPERTS))


def _weight_map(directory: Path) -> dict[str, str]:
    path = directory / INDEX_FILE
    if not path.is_file():
        raise ReleaseOverlayError(f"{directory} has no {INDEX_FILE}")
    return dict(json.loads(path.read_text())["weight_map"])


def overlay_release(base_dir: str | Path, release_dir: str | Path, out_dir: str | Path) -> Path:
    """Build ``out_dir`` (replaced if present) = ``base_dir`` with the tensors of
    ``release_dir`` in place of the base's same-named ones. Every released tensor
    must exist in the base (a partial release never adds weights)."""

    base, release, out = Path(base_dir).resolve(), Path(release_dir).resolve(), Path(out_dir)
    base_map, release_map = _weight_map(base), _weight_map(release)
    unknown = sorted(set(release_map) - set(base_map))
    if unknown:
        raise ReleaseOverlayError(f"release {release} has {len(unknown)} tensors the base lacks, "
                                  f"e.g. {unknown[:3]}")
    if out.exists():
        for child in out.iterdir():
            child.unlink()
    out.mkdir(parents=True, exist_ok=True)
    for child in base.iterdir():
        if child.name != INDEX_FILE and child.is_file():
            os.symlink(child, out / child.name)
    for name in sorted(set(release_map.values())):
        os.symlink(release / name, out / f"{RELEASE_PREFIX}{name}")
    merged = {**base_map, **{tensor: f"{RELEASE_PREFIX}{file}" for tensor, file in release_map.items()}}
    record = json.loads((base / INDEX_FILE).read_text())
    record["weight_map"] = merged
    (out / INDEX_FILE).write_text(json.dumps(record, indent=1, sort_keys=True))
    return out
