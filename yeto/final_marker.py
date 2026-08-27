"""Adjacent marker for an authoritative terminal syncer checkpoint."""

from __future__ import annotations

import re
import struct
from pathlib import Path

_CHECKPOINT_MAGIC_V1 = 0xD170_5A7E
_CHECKPOINT_MAGIC_V2 = 0xD170_5A7F
_CHECKPOINT_MAGIC_V3 = 0xD170_5A80
_MAGIC = struct.Struct("<I")
_GLOBAL_STEP = struct.Struct("<Q")
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
    """Read the revision-aware prefix needed to resume consolidation."""
    path = Path(checkpoint_path).expanduser()
    with path.open("rb") as handle:
        # V3 has the longest prefix: magic + backend + fingerprint + step.
        prefix = handle.read(4 + 1 + 32 + 8)
    # Preserve the historical contract that a file too short to contain even
    # the V1 prefix is reported as truncated, regardless of its first bytes.
    if len(prefix) < 12:
        raise ValueError(f"{path}: truncated checkpoint header")
    (magic,) = _MAGIC.unpack(prefix[:4])
    if magic == _CHECKPOINT_MAGIC_V1:
        step_offset = 4
    elif magic == _CHECKPOINT_MAGIC_V2:
        if prefix[4] not in (0, 1):
            raise ValueError(
                f"{path}: checkpoint V2 has unknown ISO backend ID {prefix[4]}"
            )
        step_offset = 5
    elif magic == _CHECKPOINT_MAGIC_V3:
        if prefix[4] not in (0, 1):
            raise ValueError(
                f"{path}: checkpoint V3 has unknown ISO backend ID {prefix[4]}"
            )
        step_offset = 37
    else:
        raise ValueError(f"{path}: bad checkpoint magic 0x{magic:08X}")
    prefix_tail = prefix[step_offset : step_offset + _GLOBAL_STEP.size]
    if len(prefix_tail) != _GLOBAL_STEP.size:
        raise ValueError(f"{path}: truncated checkpoint header")
    (global_step,) = _GLOBAL_STEP.unpack(prefix_tail)
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
