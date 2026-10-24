"""The trained model of chosen rounds as HF directories (``--rl-export-hf-rounds``).

Plain rules:

* Values: round numbers and ``final``, comma-separated (``48,96`` or
  ``56,final``). Round ``r`` is the island's ``r``-th trained round (rollout id
  ``r - 1``). ``final`` is the round whose sync boundary stops the island: on a
  synced full-parameter island, the boundary after the island received the
  final manifest and drained its pushes; without sync, the last round.
* When: at the round's sync boundary, before the next publication. The final
  export therefore runs before the island ACKs the final manifest: the island
  is still a member, so the syncer waits for it (its final-ACK timeout, the
  launcher's ``--rl-sync-wait-timeout-s``, must cover the export) and the
  launch does not end and tear the island down during the export.
* What: the island's own model. With the ACK-only final cut (the async
  default) the island keeps its local model, so a synced island exports its
  local model, not the merged one.
* Where: ``<--rl-checkpoint-store>/hf/island<learner id>/round-<r>`` (the
  launcher forwards ``<store>/hf``; one directory per island, so two islands
  that export the same round do not overwrite each other). A final export is
  named by its actual round; ``island<k>/final.json`` points to it.
* How: the backend writes the model into a local staging directory and leaves
  its completion marker there (Miles: ``.complete``, written after every rank
  finished). The island takes the marker out, removes an older directory of the
  same round, puts the directory into the store (the put commits it), then
  puts the marker (and ``final.json``) in a second put that commits again. So a
  directory with the marker is complete and committed: readers (another
  container's mount) read only directories with the marker. The staging copy
  is deleted.
* Upload failures: removing a directory that is not there is no error (a
  first export). A failed upload (remove, put, marker) runs again after
  :data:`UPLOAD_RETRY_S` seconds, :data:`UPLOAD_ATTEMPTS` attempts in all.
  When the last attempt fails, the staging copy stays on the island's local
  disk with its marker put back (``staging_kept`` on the tape): one upload of
  that directory to ``path`` restores the export (a final export also needs
  ``final.json``). At most :data:`KEEP_FAILED` such copies stay (the oldest
  goes first; ``staging_dropped``): a Nano copy is about 63 GB. They live on
  the container's disk and go with the container.
* Tape: every export is one ``rl_hf_export`` (:func:`HfRoundExports.export`
  lists the fields; ``final`` marks the final one, ``final.json`` is the
  pointer in the store), ``ok`` false with ``error`` when it failed. A failed
  export does not stop the island; when the trainer ranks are out of step
  after a failed collective write, the next collective fails on its own.
* Known limit: a full-parameter island of a synced launch restarts its round
  counter at 0 in every process, so a relaunched island counts its rounds again.

This module is backend-neutral: the backend gives the function that writes the
model into a directory.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXPORT_HF_FLAG = "--rl-export-hf-rounds"
HF_EXPORT_EVENT = "rl_hf_export"
HF_EXPORT_DIR = "hf"
FINAL = "final"
# Miles' HF_EXPORT_COMPLETE_MARKER (miles.utils.hf_utils.config): the writer's
# marker in the staging directory, and the reader's marker in the store
COMPLETE_MARKER = ".complete"
FINAL_POINTER = "final.json"
UPLOAD_ATTEMPTS = 3  # one upload and two retries
UPLOAD_RETRY_S = (10.0, 30.0)  # the waits before the second and the third attempt
KEEP_FAILED = 2  # staging copies kept after failed uploads, per island process


@dataclass(frozen=True)
class ExportRounds:
    """The rounds of ``--rl-export-hf-rounds``: numbers (ascending) and ``final``."""

    numbers: tuple[int, ...] = ()
    final: bool = False

    @classmethod
    def parse(cls, text: str | None) -> ExportRounds:
        """``"96,48,final"`` -> numbers (48, 96) and final; empty for None or ""."""
        numbers, final = set(), False
        for item in (text or "").split(","):
            item = item.strip()
            if item == FINAL:
                final = True
            elif item.isdigit() and int(item) >= 1:
                numbers.add(int(item))
            elif item:
                raise ValueError(f"{EXPORT_HF_FLAG}: {item!r} is neither a round number >= 1 nor {FINAL!r}")
        return cls(tuple(sorted(numbers)), final)

    def __bool__(self) -> bool:
        return bool(self.numbers) or self.final

    def __str__(self) -> str:
        return ",".join([*(str(n) for n in self.numbers), *([FINAL] if self.final else [])])

    def due(self, rollout_id: int, *, final: bool) -> bool:
        """Round ``rollout_id + 1`` is exported (``final``: its boundary stops the island)."""
        return int(rollout_id) + 1 in self.numbers or (self.final and final)


def export_store_path(store: str, *, learner_id: int) -> str:
    """Where one island puts its exports, under the export store ``store``."""
    return f"{store.rstrip('/')}/island{int(learner_id)}"


def _take_marker(directory: Path) -> None:
    """Remove the writer's completion marker from ``directory``; raises when the
    writer left none (an unfinished or failed write)."""
    marker = directory / COMPLETE_MARKER
    if not marker.is_file():
        raise RuntimeError(f"the writer left no {COMPLETE_MARKER} in {directory}")
    marker.unlink()


def _directory_bytes(directory: Path) -> tuple[int, int]:
    """(files, bytes) under ``directory``."""
    sizes = [path.stat().st_size for path in directory.rglob("*") if path.is_file()]
    return len(sizes), sum(sizes)


def _json(fields: dict[str, Any]) -> bytes:
    return json.dumps(fields, sort_keys=True).encode("utf-8")


class HfRoundExports:
    """The exports of one island process (module doc).

    ``write(rollout_id, directory)`` writes the model into the new local
    ``directory``; ``store`` is a ``resume.CheckpointStore`` (``put_directory``
    commits it)."""

    def __init__(self, rounds: ExportRounds, store: Any, write: Callable[[int, Path], None], *,
                 staging: Path, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 upload_attempts: int = UPLOAD_ATTEMPTS,
                 retry_s: tuple[float, ...] = UPLOAD_RETRY_S, keep_failed: int = KEEP_FAILED) -> None:
        if int(upload_attempts) < 1:
            raise ValueError("upload_attempts must be >= 1")
        self.rounds = rounds
        self.store = store
        self.write = write
        self.staging = Path(staging)
        self.clock = clock
        self.sleep = sleep
        self.upload_attempts = int(upload_attempts)
        self.retry_s = tuple(float(s) for s in retry_s) or (0.0,)
        self.keep_failed = int(keep_failed)
        self._kept: list[Path] = []  # staging copies of failed uploads, oldest first

    def due(self, rollout_id: int, *, final: bool) -> bool:
        return self.rounds.due(rollout_id, final=final)

    def export(self, rollout_id: int, *, policy_version: int, final: bool) -> dict[str, Any]:
        """Write, put and time the export of round ``rollout_id + 1`` (module doc).
        Returns the fields of its ``rl_hf_export``: round, final, policy_version,
        path (the directory in the store), store_kind, ok, files, bytes (the
        model files, marker excluded), write_s, upload_s, seconds,
        upload_attempts, and error (``write: ...`` or ``upload: ...``) when not
        ok; upload_errors (the errors of attempts retried), staging_kept and
        staging_dropped (module doc) when they apply."""
        round_number = int(rollout_id) + 1
        relative = f"round-{round_number}"
        local = self.staging / relative
        record: dict[str, Any] = {"round": round_number, "final": bool(final),
                                  "policy_version": int(policy_version),
                                  "path": str(Path(self.store.root) / relative),
                                  "store_kind": self.store.kind}
        shutil.rmtree(local, ignore_errors=True)  # a torn earlier attempt
        started, written = self.clock(), None
        try:
            self.write(int(rollout_id), local)
            _take_marker(local)
            written = self.clock()
            record["files"], record["bytes"] = _directory_bytes(local)
            self._upload(local, relative, record)
        except Exception as error:  # noqa: BLE001 - the export is evidence (module doc)
            stage = "write" if written is None else "upload"
            record["error"] = f"{stage}: {type(error).__name__}: {error}"[:2000]
        finally:
            if written is not None and "error" in record:
                self._keep(local, relative, record)
            else:
                shutil.rmtree(local, ignore_errors=True)
        ended = self.clock()
        written = ended if written is None else written
        record.setdefault("files", 0)
        record.setdefault("bytes", 0)
        record.setdefault("upload_attempts", 0)
        record.update(ok="error" not in record, write_s=round(written - started, 3),
                      upload_s=round(ended - written, 3), seconds=round(ended - started, 3))
        return record

    def _upload(self, local: Path, relative: str, record: dict[str, Any]) -> None:
        """Remove an older directory of this round (a relaunch), put the directory,
        then its marker (and the final pointer); retried (module doc)."""
        for attempt in range(1, self.upload_attempts + 1):
            record["upload_attempts"] = attempt
            try:
                self.store.remove_tree(relative)
                self.store.put_directory(local, relative)
                self.store.put_files(self._completion(relative, record))
                return
            except Exception as error:  # noqa: BLE001 - retried, then raised
                if attempt == self.upload_attempts:
                    raise
                record.setdefault("upload_errors", []).append(f"{type(error).__name__}: {error}"[:500])
                self.sleep(self.retry_s[min(attempt, len(self.retry_s)) - 1])

    def _keep(self, local: Path, relative: str, record: dict[str, Any]) -> None:
        """Keep the staging copy of a failed upload, its marker put back, so one
        upload restores it; drop the oldest beyond ``keep_failed`` (0: keep none)."""
        if self.keep_failed < 1:
            shutil.rmtree(local, ignore_errors=True)
            return
        marker = self._completion(relative, record)[f"{relative}/{COMPLETE_MARKER}"]
        (local / COMPLETE_MARKER).write_bytes(marker)
        record["staging_kept"] = str(local)
        self._kept = [path for path in self._kept if path != local] + [local]
        while len(self._kept) > self.keep_failed:
            dropped = self._kept.pop(0)
            shutil.rmtree(dropped, ignore_errors=True)
            record.setdefault("staging_dropped", []).append(str(dropped))

    @staticmethod
    def _completion(relative: str, record: dict[str, Any]) -> dict[str, bytes]:
        """The second put of an export: its marker, then the final pointer."""
        marker = {k: record[k] for k in ("round", "final", "policy_version", "files", "bytes")}
        files = {f"{relative}/{COMPLETE_MARKER}": _json(marker)}
        if record["final"]:
            files[FINAL_POINTER] = _json({**marker, "path": relative})
        return files
