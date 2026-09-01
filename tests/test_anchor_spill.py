from __future__ import annotations

from pathlib import Path

import pytest
import torch

import yeto.megatron.anchor_spill as spill_module
from yeto.megatron.anchor_spill import BF16AnchorSpill


def _store(root: Path, fragment_numels=(8, 5)) -> BF16AnchorSpill:
    return BF16AnchorSpill(
        root,
        layout_fingerprint=bytes(range(32)),
        fragment_numels=fragment_numels,
    )


def test_bf16_anchor_round_trip_is_bit_exact_and_fragment_bounded(tmp_path: Path):
    store = _store(tmp_path / "anchors")
    first = torch.tensor(
        [0.0, -0.0, 1.0, -2.0, float("inf"), float("-inf"), float("nan"), 0.5],
        dtype=torch.bfloat16,
    )
    second = torch.arange(5, dtype=torch.bfloat16)
    store.write(0, 7, first)
    store.write(1, 9, second)

    seen_numels = []

    def consume(value: torch.Tensor) -> torch.Tensor:
        seen_numels.append(value.numel())
        return value.view(torch.uint16).clone()

    actual = store.read(0, 7, consume)
    assert torch.equal(actual, first.view(torch.uint16))
    assert seen_numels == [first.numel()]
    assert len(list((tmp_path / "anchors").glob("fragment-*.anchor"))) == 2
    store.close(successful=True)
    assert not (tmp_path / "anchors").exists()


def test_spill_owns_exactly_one_file_per_fragment_and_replaces_atomically(
    tmp_path: Path,
):
    store = _store(tmp_path / "anchors", fragment_numels=(2,) * 96)
    for fragment_id in range(96):
        store.write(
            fragment_id,
            fragment_id,
            torch.tensor([fragment_id, -fragment_id], dtype=torch.bfloat16),
        )
    files = sorted((tmp_path / "anchors").glob("fragment-*.anchor"))
    assert len(files) == 96
    assert not list((tmp_path / "anchors").glob("*.tmp"))

    replacement = torch.tensor([123.0, -456.0], dtype=torch.bfloat16)
    store.write(17, 100, replacement)
    actual = store.read(17, 100, lambda value: value.clone())
    assert torch.equal(actual, replacement)
    assert len(list((tmp_path / "anchors").glob("fragment-*.anchor"))) == 96
    store.close(successful=True)


@pytest.mark.parametrize("damage", ["missing", "truncated", "payload", "header"])
def test_spill_fails_closed_on_missing_or_corrupt_fragment(tmp_path: Path, damage: str):
    store = _store(tmp_path / "anchors", fragment_numels=(16,))
    store.write(0, 3, torch.arange(16, dtype=torch.bfloat16))
    path = tmp_path / "anchors" / "fragment-000.anchor"
    if damage == "missing":
        path.unlink()
    elif damage == "truncated":
        path.write_bytes(path.read_bytes()[:-1])
    elif damage == "payload":
        with path.open("r+b") as handle:
            handle.seek(-1, 2)
            byte = handle.read(1)
            handle.seek(-1, 2)
            handle.write(bytes([byte[0] ^ 0xFF]))
    else:
        with path.open("r+b") as handle:
            handle.write(b"not-json")

    with pytest.raises(RuntimeError):
        store.read(0, 3, lambda value: value.sum())
    store.close(successful=False)


def test_failed_atomic_replacement_preserves_previous_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = _store(tmp_path / "anchors", fragment_numels=(4,))
    original = torch.arange(4, dtype=torch.bfloat16)
    store.write(0, 2, original)
    path = tmp_path / "anchors" / "fragment-000.anchor"
    original_file = path.read_bytes()

    def reject_replace(source: Path, destination: Path) -> None:
        del source, destination
        raise OSError("injected atomic replacement failure")

    monkeypatch.setattr(spill_module.os, "replace", reject_replace)
    with pytest.raises(OSError, match="injected"):
        store.write(0, 3, torch.full((4,), 9.0, dtype=torch.bfloat16))

    assert path.read_bytes() == original_file
    assert not list((tmp_path / "anchors").glob("*.tmp"))
    actual = store.read(0, 2, lambda value: value.clone())
    assert torch.equal(actual, original)
    store.close(successful=True)


def test_failed_store_initialization_removes_only_its_empty_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "anchors"

    def reject_fsync(path: Path) -> None:
        del path
        raise OSError("injected directory fsync failure")

    monkeypatch.setattr(spill_module, "_fsync_directory", reject_fsync)
    with pytest.raises(OSError, match="injected"):
        _store(root)
    assert not root.exists()


def test_spill_rejects_stale_or_preexisting_run_root(tmp_path: Path):
    root = tmp_path / "anchors"
    root.mkdir()
    (root / "user-data").write_text("do not delete", encoding="utf-8")

    with pytest.raises(FileExistsError):
        _store(root)
    assert (root / "user-data").read_text(encoding="utf-8") == "do not delete"


def test_spill_validates_dtype_shape_layout_and_version(tmp_path: Path):
    store = _store(tmp_path / "anchors", fragment_numels=(4,))
    with pytest.raises(TypeError, match="bfloat16"):
        store.write(0, 0, torch.zeros(4))
    with pytest.raises(ValueError, match="shape/size"):
        store.write(0, 0, torch.zeros(5, dtype=torch.bfloat16))

    store.write(0, 2, torch.zeros(4, dtype=torch.bfloat16))
    with pytest.raises(RuntimeError, match="version mismatch"):
        store.read(0, 3, lambda value: value.sum())
    with pytest.raises(RuntimeError, match="retained the spill read buffer"):
        store.read(0, 2, lambda value: value)
    store.close(successful=True)
