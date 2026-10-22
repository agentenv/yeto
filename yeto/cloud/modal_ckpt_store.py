"""Cloud layer: Modal Volume checkpoint store for cross-launch resume
(rl-resume-from-checkpoint).

Moved out of ``yeto/rl/engine/resume.py`` (the neutral core must not import
``modal``; same pattern as ``yeto.cloud.modal_reward_env``).  The core only knows
``CheckpointStore`` and the ``YETO_RL_STORE_IMPL`` hook; the launcher exports
``YETO_RL_STORE_IMPL=yeto.cloud.modal_ckpt_store:store_from_env`` plus
``YETO_RL_STORE_MODAL_VOLUME=<volume name>`` on a Modal island with a
``modal-volume://`` store.  Behaviour is unchanged from the original location.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from collections.abc import Callable, Iterator, Mapping
from typing import Any

from yeto.rl.engine.resume import CheckpointStore

MODAL_VOLUME_ENV = "YETO_RL_STORE_MODAL_VOLUME"  # set by the launcher on a Modal island
MODAL_VOLUME_SCHEME = "modal-volume"
STORE_IMPL = "yeto.cloud.modal_ckpt_store:store_from_env"  # value of resume.STORE_IMPL_ENV


GUARD_DIR_ENV = "YETO_MODAL_VOLUME_GUARD_DIR"  # tests; default /tmp (container-local, not the Volume)


@contextlib.contextmanager
def _volume_lock(volume: str, mode: int) -> Iterator[bool]:
    """``flock`` on a container-local file per Volume, shared by every process of the
    container (the trainer driver, its Ray actors and threads each open their own
    descriptor). Yields whether the lock was taken (False only for a non-blocking try).

    A reload remounts the Volume for the whole container and drops local files not
    committed yet, so writers hold it shared and a reload tries it exclusive and is
    skipped while any writer holds it (reload is best effort). Seen on Modal
    (e6-t2-pair-a/-c, 2026-10-21): rollout actors' reloads removed the tier 2 releaser
    thread's half-written FP8 release (FileNotFoundError on RELEASE.json.tmp / LATEST.json.tmp)."""
    import fcntl
    import re

    root = os.environ.get(GUARD_DIR_ENV) or "/tmp"
    path = os.path.join(root, "yeto-modal-volume-" + re.sub(r"[^A-Za-z0-9_.-]", "_", volume) + ".lock")
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        try:
            fcntl.flock(fd, mode)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


class ModalVolumeStore(CheckpointStore):
    """A Modal Volume (v1) mounted into the island container. Writes become visible
    to another container only after ``commit()`` (Modal docs: background commits every
    few seconds + a final commit at exit); a cut counts as written only after we
    committed once ``LATEST`` is on disk."""

    kind = "modal-volume"

    def __init__(self, root: str | os.PathLike[str], volume_name: str,
                 volume_factory: Callable[[str], Any] | None = None) -> None:
        super().__init__(root)
        self.volume_name = volume_name
        self._factory = volume_factory

    def _volume(self) -> Any:
        if self._factory is not None:
            return self._factory(self.volume_name)
        import modal  # only inside a Modal container

        return modal.Volume.from_name(self.volume_name)

    def commit(self) -> float:
        """Explicit commit; a failing commit raises (the sync then reports store_synced
        False). In the island container the learner's Python cannot import the Modal
        client, so the commit runs in the Modal runner's interpreter
        (``YETO_MODAL_PYTHON`` / ``YETO_MODAL_SYSPATH``, exported by modal_runner)."""
        started = time.monotonic()
        if self._factory is None and os.environ.get("YETO_MODAL_PYTHON"):
            self._runner_call("commit")
        else:
            self._volume().commit()
        self.commits += 1
        return time.monotonic() - started

    def _runner_call(self, verb: str) -> None:
        import subprocess

        env = dict(os.environ)
        env["PYTHONPATH"] = os.environ.get("YETO_MODAL_SYSPATH", "")
        code = ("import sys, modal; modal.Volume.from_name(sys.argv[1])." + verb + "()")
        done = subprocess.run([os.environ["YETO_MODAL_PYTHON"], "-c", code, self.volume_name],
                              env=env, capture_output=True, text=True, timeout=600)
        if done.returncode != 0:
            raise RuntimeError(f"modal volume {verb} via the runner interpreter failed "
                               f"(rc {done.returncode}): {done.stderr[-2000:]}")

    commits = 0

    @contextlib.contextmanager
    def writing(self) -> Iterator[None]:
        import fcntl

        with _volume_lock(self.volume_name, fcntl.LOCK_SH):
            yield

    def reload(self) -> None:
        import fcntl

        with _volume_lock(self.volume_name, fcntl.LOCK_EX | fcntl.LOCK_NB) as taken:
            if not taken:  # a reload now would drop a writer's uncommitted files
                logging.getLogger(__name__).info("modal volume reload skipped: a writer holds %s",
                                                 self.volume_name)
                return
            try:
                if self._factory is None and os.environ.get("YETO_MODAL_PYTHON"):
                    self._runner_call("reload")
                else:
                    self._volume().reload()
            except Exception as exc:  # noqa: BLE001 - reload is best effort (open files block it)
                logging.getLogger(__name__).warning("modal volume reload failed: %r", exc)
            else:  # logged so a remount can be lined up with a writer's failure (e6-t2-pair-e)
                logging.getLogger(__name__).info("modal volume reloaded: %s", self.volume_name)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "volume": self.volume_name}


def store_from_env(path: str | os.PathLike[str], *, environ: Mapping[str, str],
                   volume_factory: Callable[[str], Any] | None = None) -> ModalVolumeStore:
    """``resume.store_for`` hook: the Volume name comes from :data:`MODAL_VOLUME_ENV`."""
    volume = environ.get(MODAL_VOLUME_ENV)
    if not volume:
        raise ValueError(f"modal-volume checkpoint store selected but {MODAL_VOLUME_ENV} is not set")
    return ModalVolumeStore(path, volume, volume_factory=volume_factory)


def parse_modal_volume_uri(value: str) -> tuple[str, str]:
    """``modal-volume://<name>[/<prefix>]`` -> (name, prefix). Raises ValueError."""
    scheme, sep, rest = str(value).partition("://")
    if scheme != MODAL_VOLUME_SCHEME or not sep:
        raise ValueError(f"{value!r} is not a {MODAL_VOLUME_SCHEME}:// URI")
    name, _, prefix = rest.strip("/").partition("/")
    if not name or not all(c.isalnum() or c in "-_." for c in name) or ".." in prefix.split("/"):
        raise ValueError(f"{value!r}: bad Modal Volume name or prefix")
    return name, prefix.strip("/")
