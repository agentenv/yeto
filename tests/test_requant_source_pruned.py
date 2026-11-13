"""Seam: the trainer prunes a BF16 release while the re-quantizer still reads it.

The trainer's ``PolicyReleaseStore`` keeps the newest ``keep`` releases; a slow
re-quantization (minutes for a large model) can outlive two trainer releases, so
the BF16 directory it reads disappears mid-conversion. The releaser must skip that
release (no partial target release, thread alive) and convert the newer one.
"""

from __future__ import annotations

import threading
from pathlib import Path

from yeto.rl.engine.policy_release import PolicyReleaseStore
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.quant.release import RequantReleaser, start_scoped_releaser


def _bf16_write(tag: str):
    def write(d: Path) -> None:
        (d / "model.safetensors").write_text(tag)
    return write


def _stores(tmp_path: Path):
    store = CheckpointStore(tmp_path / "store")
    trainer = PolicyReleaseStore(store, keep=2)
    source = PolicyReleaseStore(store, keep=2)
    target = PolicyReleaseStore(store, keep=2, release_dir="policy-releases-w4a8")
    return trainer, source, target


def test_source_pruned_mid_conversion_is_skipped_then_newer_converted(tmp_path):
    trainer, source, target = _stores(tmp_path)
    trainer.release(1, "h1", _bf16_write("v1"))
    calls: list[str] = []

    def requantize(src: Path, out: Path) -> dict:
        if not calls:  # the trainer publishes two releases during this conversion
            trainer.release(2, "h2", _bf16_write("v2"))
            trainer.release(3, "h3", _bf16_write("v3"))
        calls.append(src.name)
        (out / "model.safetensors").write_text("q:" + (src / "model.safetensors").read_text())
        return {}

    releaser = RequantReleaser(source, target, requantize, log=lambda m: None)
    assert releaser.sync() is None  # r1 was pruned while it was read
    assert target.latest() is None
    assert not [p for p in target.root.iterdir() if p.is_dir()]  # no partial target release
    assert releaser.waiting and "r1-v1" in releaser.waiting

    record = releaser.sync()
    assert record is not None and record.version == 3
    assert (target.resolve(record) / "model.safetensors").read_text() == "q:v3"


def test_releaser_thread_survives_a_pruned_source(tmp_path):
    trainer, source, target = _stores(tmp_path)
    trainer.release(1, "h1", _bf16_write("v1"))
    first = threading.Event()

    def requantize(src: Path, out: Path) -> dict:
        if not first.is_set():
            first.set()
            trainer.release(2, "h2", _bf16_write("v2"))
            trainer.release(3, "h3", _bf16_write("v3"))
        (out / "model.safetensors").write_text("q:" + (src / "model.safetensors").read_text())
        return {}

    stop = threading.Event()
    thread = start_scoped_releaser(RequantReleaser(source, target, requantize, log=lambda m: None),
                                   stop, poll_s=0.01)
    try:
        for _ in range(500):
            latest = target.latest()
            if latest is not None and latest.version == 3:
                break
            stop.wait(0.01)
        assert thread.is_alive()
        assert target.latest() is not None and target.latest().version == 3
    finally:
        stop.set()
        thread.join(5)


def test_converter_error_on_the_live_release_still_stops(tmp_path):
    import pytest

    trainer, source, target = _stores(tmp_path)
    trainer.release(1, "h1", _bf16_write("v1"))

    def broken(src: Path, out: Path) -> dict:
        raise ValueError("bad layout")

    with pytest.raises(ValueError, match="bad layout"):
        RequantReleaser(source, target, broken, log=lambda m: None).sync()
