"""Adjacent marker for an authoritative terminal syncer checkpoint."""

from __future__ import annotations

import re
import struct
from pathlib import Path

_CHECKPOINT_MAGIC = 0xD170_5A7E
_CHECKPOINT_PREFIX = struct.Struct("<IQ")
_MARKER = re.compile(rb"YETO_FINAL_V1\nglobal_step=([0-9]+)\n\Z")


def final_marker_path(checkpoint_path: str | Path) -> Path:
    checkpoint = Path(checkpoint_path).expanduser()
    return Path(f"{checkpoint}.final")


def parse_final_marker(marker_path: str | Path) -> int:
    path = Path(marker_path).expanduser()
    match = _MARKER.fullmatch(path.read_bytes())
    if match is None:
        raise ValueError(f"{path}: malformed final checkpoint marker")
    global_step = int(match.group(1))
    if global_step > 0xFFFF_FFFF_FFFF_FFFF:
        raise ValueError(f"{path}: marker global_step does not fit u64")
    return global_step


def read_checkpoint_global_step(checkpoint_path: str | Path) -> int:
    """Read only the fixed checkpoint prefix needed to resume consolidation."""
    path = Path(checkpoint_path).expanduser()
    with path.open("rb") as handle:
        prefix = handle.read(_CHECKPOINT_PREFIX.size)
    if len(prefix) != _CHECKPOINT_PREFIX.size:
        raise ValueError(f"{path}: truncated checkpoint header")
    magic, global_step = _CHECKPOINT_PREFIX.unpack(prefix)
    if magic != _CHECKPOINT_MAGIC:
        raise ValueError(f"{path}: bad checkpoint magic 0x{magic:08X}")
    return global_step


def validate_final_checkpoint(checkpoint_path: str | Path) -> int:
    """Require a checkpoint and its adjacent marker to name the same step."""
    checkpoint = Path(checkpoint_path).expanduser()
    marker_step = parse_final_marker(final_marker_path(checkpoint))
    checkpoint_step = read_checkpoint_global_step(checkpoint)
    if checkpoint_step != marker_step:
        raise ValueError(
            f"{final_marker_path(checkpoint)}: global_step={marker_step} does not "
            f"match checkpoint global_step={checkpoint_step}"
        )
    return checkpoint_step
