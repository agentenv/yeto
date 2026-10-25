"""Cloud layer: Modal Volume checkpoint store for cross-launch resume
(rl-resume-from-checkpoint).

Moved out of ``yeto/rl/engine/resume.py`` (the neutral core must not import
``modal``; same pattern as the other ``yeto.cloud.modal_*`` modules).  The core only knows
``CheckpointStore`` and the ``YETO_RL_STORE_IMPL`` hook; the launcher exports
``YETO_RL_STORE_IMPL=yeto.cloud.modal_ckpt_store:store_from_env`` plus
``YETO_RL_STORE_MODAL_VOLUME=<volume name>`` on a Modal island with a
``modal-volume://`` store.  Behaviour is unchanged from the original location.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from yeto.rl.engine.resume import CheckpointStore, StoreWriterGuard

MODAL_VOLUME_ENV = "YETO_RL_STORE_MODAL_VOLUME"  # set by the launcher on a Modal island
MODAL_VOLUME_SCHEME = "modal-volume"
STORE_IMPL = "yeto.cloud.modal_ckpt_store:store_from_env"  # value of resume.STORE_IMPL_ENV
GUARD_DIR_ENV = "YETO_RL_STORE_GUARD_DIR"  # where the per-Volume lock files live (default /tmp)
# where the island mounts the store Volume (yeto.launch.storage.MODAL_CHECKPOINT_STORE_MOUNT)
MOUNT_ENV = "YETO_RL_STORE_MODAL_MOUNT"
DEFAULT_MOUNT = "/root/yeto-checkpoint-store"
_UPLOAD_SCRIPT = """
import json, sys, modal
volume = modal.Volume.from_name(sys.argv[1])
with volume.batch_upload(force=True) as batch:
    for kind, local, remote in json.loads(sys.argv[2]):
        (batch.put_directory if kind == "dir" else batch.put_file)(local, remote)
"""


def _volume_has(volume, remote):
    """Whether the absolute Volume path ``remote`` exists. Each directory is listed
    from the Volume root down, so only directories known to exist are listed."""
    current = "/"
    for name in [part for part in remote.split("/") if part]:
        if name not in {entry.path.rstrip("/").rsplit("/", 1)[-1] for entry in volume.listdir(current)}:
            return False
        current = current.rstrip("/") + "/" + name
    return True


def _remove_if_present(volume, remote):
    """Remove ``remote`` recursively; a missing path is no error. Modal reports a
    missing path with several types (``FileNotFoundError``, ``NotFoundError``,
    ``InvalidError`` "No such file or directory") that also carry other errors, so a
    failure is not classified by its type or text: the path is looked up, and the
    error is raised only when the path is still there."""
    try:
        volume.remove_file(remote, recursive=True)
    except Exception:
        if _volume_has(volume, remote):
            raise


# the same removal in the Modal runner's interpreter (it has the Modal client, not yeto)
_REMOVE_SCRIPT = "\n".join([
    "import sys, modal",
    inspect.getsource(_volume_has),
    inspect.getsource(_remove_if_present),
    "_remove_if_present(modal.Volume.from_name(sys.argv[1]), sys.argv[2])",
])


def _guard_path(volume: str) -> str:
    import re

    root = os.environ.get(GUARD_DIR_ENV) or "/tmp"
    return os.path.join(root, "yeto-modal-volume-" + re.sub(r"[^A-Za-z0-9_.-]", "_", volume) + ".lock")


# A store commit through the runner interpreter is cut off after this many seconds,
# or after the uncommitted bytes of a commit_budget at COMMIT_MIN_BYTES_PER_S when that
# is longer (S27 G-CD: the commit of a 473 GB round cut timed out at 600 s, s27-bb-b1).
COMMIT_TIMEOUT_S = 600.0
COMMIT_MIN_BYTES_PER_S = 100e6


def commit_timeout_s(pending_bytes: int | None) -> float:
    return max(COMMIT_TIMEOUT_S, float(pending_bytes or 0) / COMMIT_MIN_BYTES_PER_S)


class _VolumeWriterGuard(StoreWriterGuard):
    """A shared ``flock`` on a container-local file per Volume. Every process
    and thread of the container opens its own descriptor, so writers share it
    and a reload, which needs it exclusive, waits for none of them: it is
    skipped while any writer holds the lock."""

    def __init__(self, volume: str) -> None:
        import fcntl

        self._fd: int | None = os.open(_guard_path(volume), os.O_RDWR | os.O_CREAT, 0o666)
        try:
            fcntl.flock(self._fd, fcntl.LOCK_SH)
        except BaseException:
            os.close(self._fd)
            self._fd = None
            raise

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is not None:
            os.close(fd)  # closing the descriptor drops its lock


class ModalVolumeStore(CheckpointStore):
    """A Modal Volume (v1) mounted into the island container. Writes become visible
    to another container only after ``commit()`` (Modal docs: background commits every
    few seconds + a final commit at exit); a cut counts as written only after we
    committed once ``LATEST`` is on disk."""

    kind = "modal-volume"

    def __init__(self, root: str | os.PathLike[str], volume_name: str,
                 volume_factory: Callable[[str], Any] | None = None,
                 mount: str | os.PathLike[str] | None = None) -> None:
        """``mount``: where the Volume root is mounted (the store root lies under it);
        None: no direct upload, :meth:`put_directory` writes through the mount."""
        super().__init__(root)
        self.volume_name = volume_name
        self._factory = volume_factory
        self.mount = None if mount is None else os.fspath(mount)

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
            pending = self._commit_budgets().get(threading.get_ident())
            self._runner_call("commit", timeout_s=commit_timeout_s(pending))
        else:
            self._volume().commit()
        self.commits += 1
        return time.monotonic() - started

    def _commit_budgets(self) -> dict[int, int]:
        """{thread id: uncommitted bytes} of the open :meth:`commit_budget` contexts."""
        return self.__dict__.setdefault("_budgets", {})

    @contextmanager
    def commit_budget(self, pending_bytes: int | None) -> Iterator[None]:
        """Commits on this thread may take :func:`commit_timeout_s` of ``pending_bytes``
        (``resume.commit_budget``: a round cut knows the bytes it wrote)."""
        budgets, key = self._commit_budgets(), threading.get_ident()
        previous = budgets.get(key)
        budgets[key] = int(pending_bytes or 0)
        try:
            yield
        finally:
            if previous is None:
                budgets.pop(key, None)
            else:
                budgets[key] = previous

    def _runner_call(self, verb: str, *, timeout_s: float = COMMIT_TIMEOUT_S) -> None:
        self._runner_script("import sys, modal; modal.Volume.from_name(sys.argv[1])." + verb + "()",
                            verb=verb, timeout_s=timeout_s)

    def _runner_script(self, code: str, *argv: str, verb: str, timeout_s: float = COMMIT_TIMEOUT_S) -> None:
        """Run ``code`` in the Modal runner's interpreter (it has the Modal client)."""
        import subprocess

        env = dict(os.environ)
        env["PYTHONPATH"] = os.environ.get("YETO_MODAL_SYSPATH", "")
        done = subprocess.run([os.environ["YETO_MODAL_PYTHON"], "-c", code, self.volume_name, *argv],
                              env=env, capture_output=True, text=True, timeout=timeout_s)
        if done.returncode != 0:
            raise RuntimeError(f"modal volume {verb} via the runner interpreter failed "
                               f"(rc {done.returncode}): {done.stderr[-2000:]}")

    # -- direct upload (policy releases staged on local disk) -------------------
    def _volume_path(self, relative: str) -> str | None:
        """``relative`` (under the store root) as a path in the Volume; None when the
        store root is not under the known mount (then the mount is written)."""
        if self.mount is None:
            return None
        try:
            inside = self.root.relative_to(self.mount)
        except ValueError:
            return None
        return "/" + (inside / relative).as_posix().lstrip("/")

    @property
    def puts_bypass_mount(self) -> bool:
        return self.mount is not None and self._volume_path("") is not None

    def _direct(self) -> bool:
        return self._factory is not None or bool(os.environ.get("YETO_MODAL_PYTHON"))

    def _upload(self, ops: list[tuple[str, str, str]]) -> None:
        """One ``batch_upload`` (all or nothing): the files are durable and visible
        to a reload elsewhere when it returns; this host's mount does not change."""
        if self._factory is not None:
            with self._volume().batch_upload(force=True) as batch:
                for kind, local, remote in ops:
                    (batch.put_directory if kind == "dir" else batch.put_file)(local, remote)
            return
        self._runner_script(_UPLOAD_SCRIPT, json.dumps(ops), verb="batch_upload", timeout_s=3600.0)

    def put_directory(self, local: str | os.PathLike[str], relative: str) -> None:
        remote = self._volume_path(relative)
        if remote is None or not self._direct():
            return super().put_directory(local, relative)
        self._upload([("dir", os.fspath(local), remote)])

    def put_files(self, files: Mapping[str, bytes]) -> None:
        """All of ``files`` in one upload (a reader sees all or none of them)."""
        remotes = {relative: self._volume_path(relative) for relative in files}
        if None in remotes.values() or not self._direct():
            return super().put_files(files)
        import tempfile

        with tempfile.TemporaryDirectory(prefix="yeto-put-") as tmp:
            ops = []
            for index, (relative, data) in enumerate(files.items()):
                local = os.path.join(tmp, str(index))
                with open(local, "wb") as handle:
                    handle.write(data)
                ops.append(("file", local, remotes[relative]))
            self._upload(ops)

    def remove_tree(self, relative: str) -> None:
        remote = self._volume_path(relative)
        if remote is None or not self._direct():
            return super().remove_tree(relative)
        if self._factory is not None:
            _remove_if_present(self._volume(), remote)
            return
        self._runner_script(_REMOVE_SCRIPT, remote, verb="remove_file")

    commits = 0

    def hold_writer(self) -> StoreWriterGuard:
        """A reload remounts the Volume for the whole container and drops local
        files not committed yet: a cut's shards written meanwhile fail (EPERM on
        makedirs of the store directory). Writers hold this until their commit."""
        return _VolumeWriterGuard(self.volume_name)

    def hold_reader(self) -> StoreWriterGuard:
        """A reload with a Volume file open in any process of the container fails
        (ConflictError "there are open files preventing the operation", S27 G-CC
        E2: the trainer driver's release reload hit the payload reads of the
        buffer reader). A reader holds the same shared lock as a writer while
        its file is open, so such a reload is skipped instead."""
        return _VolumeWriterGuard(self.volume_name)

    def reload(self) -> None:
        import fcntl

        log = logging.getLogger(__name__)
        fd = os.open(_guard_path(self.volume_name), os.O_RDWR | os.O_CREAT, 0o666)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:  # reload is best effort; a writer's files come first
                log.info("modal volume reload skipped: a writer holds %s", self.volume_name)
                return
            try:
                if self._factory is None and os.environ.get("YETO_MODAL_PYTHON"):
                    self._runner_call("reload")
                else:
                    self._volume().reload()
            except Exception as exc:  # noqa: BLE001 - reload is best effort (open files block it)
                log.warning("modal volume reload failed: %r", exc)
            else:  # logged so a remount can be lined up with a writer's failure
                log.info("modal volume reloaded: %s", self.volume_name)
        finally:
            os.close(fd)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "volume": self.volume_name}


def store_from_env(path: str | os.PathLike[str], *, environ: Mapping[str, str],
                   volume_factory: Callable[[str], Any] | None = None) -> ModalVolumeStore:
    """``resume.store_for`` hook: the Volume name comes from :data:`MODAL_VOLUME_ENV`."""
    volume = environ.get(MODAL_VOLUME_ENV)
    if not volume:
        raise ValueError(f"modal-volume checkpoint store selected but {MODAL_VOLUME_ENV} is not set")
    return ModalVolumeStore(path, volume, volume_factory=volume_factory,
                            mount=environ.get(MOUNT_ENV) or DEFAULT_MOUNT)


def parse_modal_volume_uri(value: str) -> tuple[str, str]:
    """``modal-volume://<name>[/<prefix>]`` -> (name, prefix). Raises ValueError."""
    scheme, sep, rest = str(value).partition("://")
    if scheme != MODAL_VOLUME_SCHEME or not sep:
        raise ValueError(f"{value!r} is not a {MODAL_VOLUME_SCHEME}:// URI")
    name, _, prefix = rest.strip("/").partition("/")
    if not name or not all(c.isalnum() or c in "-_." for c in name) or ".." in prefix.split("/"):
        raise ValueError(f"{value!r}: bad Modal Volume name or prefix")
    return name, prefix.strip("/")
