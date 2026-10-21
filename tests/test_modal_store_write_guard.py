"""ModalVolumeStore: a reload never runs while a writer of the same Volume holds
``writing()`` (S26 E6: a reload dropped a half-written tier 2 release)."""

from __future__ import annotations

import threading

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


def _store(tmp_path, volume: _Volume, name: str = "v") -> ModalVolumeStore:
    return ModalVolumeStore(tmp_path, f"{name}-{id(volume)}", volume_factory=lambda _n: volume)


def test_reload_is_skipped_while_a_writer_of_the_same_volume_holds_the_guard(tmp_path):
    volume = _Volume()
    writer, reader = _store(tmp_path, volume), _store(tmp_path, volume)  # two store objects, one Volume
    with writer.writing():
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
        with store.writing():
            entered.set()

    t = threading.Thread(target=writer)
    t.start()
    assert not entered.wait(0.2)  # blocked behind the reload
    release.set()
    reloader.join(5)
    assert entered.wait(5)
    t.join(5)
