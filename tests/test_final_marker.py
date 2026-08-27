import struct

import pytest

from yeto.export import parse_checkpoint
from yeto.export import CKPT_MAGIC_V1, CKPT_MAGIC_V2, CKPT_MAGIC_V3
from yeto.final_marker import (
    final_marker_path,
    parse_final_marker,
    read_checkpoint_global_step,
    validate_final_checkpoint,
)


def _checkpoint(path, global_step: int, magic: int = CKPT_MAGIC_V1) -> None:
    header = bytearray(struct.pack("<I", magic))
    if magic in (CKPT_MAGIC_V2, CKPT_MAGIC_V3):
        header += struct.pack("<B", 0)
    if magic == CKPT_MAGIC_V3:
        header += bytes(range(32))
    header += struct.pack("<QI", global_step, 0)
    header += struct.pack("<I", 0)
    path.write_bytes(header)


def test_valid_marker_is_strict_and_bound_to_checkpoint_step(tmp_path):
    checkpoint = tmp_path / "state.ckpt"
    _checkpoint(checkpoint, 17)
    marker = final_marker_path(checkpoint)
    marker.write_text("YETO_FINAL_V1\nglobal_step=17\n", encoding="utf-8")

    assert parse_final_marker(marker) == 17
    assert validate_final_checkpoint(checkpoint) == 17

    marker.write_text("YETO_FINAL_V1\nglobal_step=18\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match checkpoint"):
        validate_final_checkpoint(checkpoint)


@pytest.mark.parametrize(
    "content",
    [
        b"YETO_FINAL_V1\nglobal_step=17",
        b"YETO_FINAL_V1\nglobal_step=-1\n",
        b"YETO_FINAL_V1\nglobal_step=17\nextra\n",
        b"YETO_FINAL_V2\nglobal_step=17\n",
        b"\xff",
    ],
)
def test_malformed_marker_is_rejected(tmp_path, content):
    marker = tmp_path / "state.ckpt.final"
    marker.write_bytes(content)
    with pytest.raises(ValueError, match="malformed"):
        parse_final_marker(marker)


def test_unmarked_checkpoint_remains_a_generic_recovery_checkpoint(tmp_path):
    checkpoint = tmp_path / "state.ckpt"
    _checkpoint(checkpoint, 9)
    assert parse_checkpoint(checkpoint).global_step == 9
    with pytest.raises(FileNotFoundError):
        validate_final_checkpoint(checkpoint)


def test_checkpoint_step_reader_uses_only_the_valid_fixed_prefix(tmp_path):
    checkpoint = tmp_path / "state.ckpt"
    _checkpoint(checkpoint, 19)
    assert read_checkpoint_global_step(checkpoint) == 19

    checkpoint.write_bytes(b"short")
    with pytest.raises(ValueError, match="truncated checkpoint header"):
        read_checkpoint_global_step(checkpoint)

    checkpoint.write_bytes(struct.pack("<IQ", 0, 19))
    with pytest.raises(ValueError, match="bad checkpoint magic"):
        read_checkpoint_global_step(checkpoint)


@pytest.mark.parametrize("magic", [CKPT_MAGIC_V1, CKPT_MAGIC_V2, CKPT_MAGIC_V3])
def test_checkpoint_step_reader_understands_all_revisions(tmp_path, magic):
    checkpoint = tmp_path / "state.ckpt"
    _checkpoint(checkpoint, 123, magic)
    assert read_checkpoint_global_step(checkpoint) == 123
    assert parse_checkpoint(checkpoint).global_step == 123


def test_checkpoint_step_reader_rejects_unknown_versioned_backend(tmp_path):
    checkpoint = tmp_path / "state.ckpt"
    checkpoint.write_bytes(
        struct.pack("<IBQ", CKPT_MAGIC_V2, 99, 123) + struct.pack("<II", 0, 0)
    )
    with pytest.raises(ValueError, match="unknown ISO backend ID 99"):
        read_checkpoint_global_step(checkpoint)
