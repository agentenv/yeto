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

import logging
import os
import time
from collections.abc import Callable, Mapping
from typing import Any

from yeto.rl.engine.resume import CheckpointStore

MODAL_VOLUME_ENV = "YETO_RL_STORE_MODAL_VOLUME"  # set by the launcher on a Modal island
MODAL_VOLUME_SCHEME = "modal-volume"
STORE_IMPL = "yeto.cloud.modal_ckpt_store:store_from_env"  # value of resume.STORE_IMPL_ENV


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

    def reload(self) -> None:
        try:
            if self._factory is None and os.environ.get("YETO_MODAL_PYTHON"):
                self._runner_call("reload")
            else:
                self._volume().reload()
        except Exception as exc:  # noqa: BLE001 - reload is best effort (open files block it)
            logging.getLogger(__name__).warning("modal volume reload failed: %r", exc)

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
