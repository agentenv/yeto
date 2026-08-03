import struct

import pytest

from yeto.export import parse_checkpoint
from yeto.final_marker import (
    final_marker_path,
    parse_final_marker,
    read_checkpoint_global_step,
    validate_final_checkpoint,
)


def _checkpoint(path, global_step: int) -> None:
    path.write_bytes(
        struct.pack("<IQI", 0xD1705A7E, global_step, 0)
        + struct.pack("<I", 0)
    )


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
