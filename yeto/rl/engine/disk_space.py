"""Local disk preflight before a release lands on a node (download, delta rebuild).

A Kimi-K3.1 release is about 200 GB: a sampler's hot load downloads it into its
cache, a trainer's ``--rl-init-from-release`` downloads and rebuilds it on every
node. Without a check, a small disk fails with ENOSPC part way through and leaves a
release-sized partial directory behind. :func:`check_free` compares the free bytes
with what is about to be written (the release record's sizes times
:data:`HEADROOM`) and raises :class:`InsufficientDiskSpace` before the first byte.
"""

from __future__ import annotations

import math
import os
import shutil
from pathlib import Path

HEADROOM_ENV = "YETO_RL_DISK_HEADROOM"
HEADROOM = 1.05  # filesystem overhead and small files beside the recorded weights


class InsufficientDiskSpace(OSError):
    """Too little free space on the local disk for what is about to be written."""


def free_bytes(path: str | os.PathLike[str]) -> int:
    """Free bytes of the filesystem ``path`` is (or will be) on: its nearest existing parent."""
    probe = Path(path).expanduser().absolute()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return int(shutil.disk_usage(probe).free)


def headroom(environ=None) -> float:
    raw = (os.environ if environ is None else environ).get(HEADROOM_ENV, "").strip()
    try:
        value = float(raw) if raw else HEADROOM
    except ValueError:
        value = HEADROOM
    return value if math.isfinite(value) and value >= 1.0 else HEADROOM  # inf/nan/<1: default


def check_free(path: str | os.PathLike[str], need_bytes: int, what: str) -> None:
    """Raise :class:`InsufficientDiskSpace` when ``path``'s disk has less free space
    than ``need_bytes`` x :func:`headroom`."""
    need = int(int(need_bytes) * headroom())
    if need <= 0:
        return
    free = free_bytes(path)
    if free < need:
        raise InsufficientDiskSpace(
            f"{what}: needs about {need / 1e9:.2f} GB on {Path(path).expanduser()} "
            f"({int(need_bytes)} bytes x {headroom():g} headroom), only {free / 1e9:.2f} GB free; "
            f"free the disk or point the cache at a larger one before retrying")
