"""ModalVolumeStore: a reload never runs while a writer of the same Volume holds
a ``hold_writer()`` guard, in this process or another one of the container (S26
E6: rollout actors' reloads dropped a half-written tier 2 release)."""

from __future__ import annotations

import contextlib
import subprocess
import sys
import threading

import pytest

from yeto.cloud import modal_ckpt_store as mcs
from yeto.cloud.modal_ckpt_store import ModalVolumeStore
from yeto.rl.engine.policy_release import PolicyReleaseStore


class _Volume:
    def __init__(self) -> None:
        self.reloads = 0
        self.commits = 0

    def reload(self) -> None:
        self.reloads += 1

    def commit(self) -> None:
        self.commits += 1


@pytest.fixture(autouse=True)
def _guard_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(mcs.GUARD_DIR_ENV, str(tmp_path))


def _store(tmp_path, volume: _Volume, name: str = "v") -> ModalVolumeStore:
    return ModalVolumeStore(tmp_path, f"{name}-{id(volume)}", volume_factory=lambda _n: volume)


@contextlib.contextmanager
def _writing(store: ModalVolumeStore):
    guard = store.hold_writer()
    try:
        yield
    finally:
        guard.release()


def test_reload_is_skipped_while_a_writer_of_the_same_volume_holds_the_guard(tmp_path):
    volume = _Volume()
    writer, reader = _store(tmp_path, volume), _store(tmp_path, volume)  # two store objects, one Volume
    with _writing(writer):
        reader.reload()
    assert volume.reloads == 0
    reader.reload()
    assert volume.reloads == 1


def test_a_release_written_while_another_thread_reloads_is_never_reloaded_over(tmp_path):
    volume = _Volume()
    store = _store(tmp_path, volume)
    releases = PolicyReleaseStore(store)
    inside = threading.Event()
    seen: list[int] = []

    def write(directory):
        (directory / "model.safetensors").write_bytes(b"x")
        inside.set()
        before = volume.reloads
        reloader = threading.Thread(target=_store(tmp_path, volume).reload)
        reloader.start()
        reloader.join(5)
        seen.append(volume.reloads - before)

    record = releases.release(1, "h1", write)
    assert inside.is_set() and seen == [0]
    assert record is not None and (tmp_path / releases.release_dir / record.path / "model.safetensors").is_file()


def test_a_writer_waits_for_a_reload_in_flight(tmp_path):
    started, release = threading.Event(), threading.Event()

    class SlowVolume(_Volume):
        def reload(self) -> None:
            started.set()
            release.wait(5)
            super().reload()

    volume = SlowVolume()
    store = _store(tmp_path, volume)
    reloader = threading.Thread(target=store.reload)
    reloader.start()
    started.wait(5)
    entered = threading.Event()

    def writer():
        with _writing(store):
            entered.set()

    t = threading.Thread(target=writer)
    t.start()
    assert not entered.wait(0.2)  # blocked behind the reload
    release.set()
    reloader.join(5)
    assert entered.wait(5)
    t.join(5)


def test_a_writer_in_another_process_blocks_the_reload(tmp_path):
    volume = _Volume()
    store = _store(tmp_path, volume)
    code = ("import os, sys, time; from yeto.cloud.modal_ckpt_store import ModalVolumeStore\n"
            "s = ModalVolumeStore(sys.argv[1], sys.argv[2], volume_factory=lambda n: None)\n"
            "g = s.hold_writer(); print('held', flush=True); sys.stdin.readline(); g.release()\n")
    proc = subprocess.Popen([sys.executable, "-c", code, str(tmp_path), store.volume_name],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "held"
        store.reload()
        assert volume.reloads == 0
    finally:
        proc.communicate("\n", timeout=30)
    store.reload()
    assert volume.reloads == 1


# S27 G-CC E2: the trainer driver's release reload failed with "there are open files
# preventing the operation" while the buffer reader process read sample payloads.


def test_a_reader_in_another_process_blocks_the_reload(tmp_path):
    volume = _Volume()
    store = _store(tmp_path, volume)
    code = ("import sys; from yeto.cloud.modal_ckpt_store import ModalVolumeStore\n"
            "s = ModalVolumeStore(sys.argv[1], sys.argv[2], volume_factory=lambda n: None)\n"
            "g = s.hold_reader(); print('held', flush=True); sys.stdin.readline(); g.release()\n")
    proc = subprocess.Popen([sys.executable, "-c", code, str(tmp_path), store.volume_name],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "held"
        PolicyReleaseStore(store).latest()  # the driver's release read reloads the store
        assert volume.reloads == 0
    finally:
        proc.communicate("\n", timeout=30)
    store.reload()
    assert volume.reloads == 1


def test_buffer_index_and_payload_reads_hold_the_reader_guard(tmp_path, monkeypatch):
    from pathlib import Path

    from yeto.rl.engine import sample_buffer as sb

    volume = _Volume()
    reader = _store(tmp_path, volume)
    driver = _store(tmp_path, volume)  # another store object (process) of the same Volume
    index = sb.JsonlIndex(reader)
    ref = index.write_payload("g0.bin", b"payload")
    index.append(lambda _records: None, lambda: {"pos": 0})
    reloads_during: dict[str, int] = {}
    original = Path.read_bytes

    def read_bytes(path):
        before = volume.reloads
        driver.reload()  # a reload while the file is being read
        reloads_during[path.name] = volume.reloads - before
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    assert index.read_payload(ref) == b"payload"
    assert reloads_during == {"g0.bin": 0}
    monkeypatch.setattr(Path, "read_bytes", original)
    opened = []
    real_open = open

    def tracking_open(file, *args, **kwargs):
        if str(file) == str(index.path):
            before = volume.reloads
            driver.reload()
            opened.append(volume.reloads - before)
        return real_open(file, *args, **kwargs)

    fresh = sb.JsonlIndex(reader, create_index=False)  # a reader that has read nothing yet
    monkeypatch.setattr("builtins.open", tracking_open)
    reloads_before = volume.reloads
    assert fresh.read_new() == [{"pos": 0}]
    assert opened == [0]  # the index read held the guard; the reader's own reload ran before it
    assert volume.reloads == reloads_before + 1
    monkeypatch.setattr("builtins.open", real_open)
    driver.reload()  # no read open: the reload runs
    assert volume.reloads == reloads_before + 2


def test_local_and_duck_typed_stores_hold_nothing_for_readers(tmp_path):
    from yeto.rl.engine import sample_buffer as sb
    from yeto.rl.engine.resume import CheckpointStore

    guard = CheckpointStore(tmp_path).hold_reader()
    guard.release()
    guard.release()  # idempotent

    class Bare:  # a store object without hold_reader (test fakes)
        root = tmp_path / "bare"

        def reload(self):
            return None

        def commit(self):
            return 0.0

    index = sb.JsonlIndex(Bare())
    ref = index.write_payload("g1.bin", b"x")
    assert index.read_payload(ref) == b"x" and index.read_new() == []
