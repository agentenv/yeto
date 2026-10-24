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
                 staging: Path, clock: Callable[[], float] = time.monotonic) -> None:
        self.rounds = rounds
        self.store = store
        self.write = write
        self.staging = Path(staging)
        self.clock = clock

    def due(self, rollout_id: int, *, final: bool) -> bool:
        return self.rounds.due(rollout_id, final=final)

    def export(self, rollout_id: int, *, policy_version: int, final: bool) -> dict[str, Any]:
        """Write, put and time the export of round ``rollout_id + 1`` (module doc).
        Returns the fields of its ``rl_hf_export``: round, final, policy_version,
        path (the directory in the store), store_kind, ok, files, bytes (the
        model files, marker excluded), write_s, upload_s, seconds, and error
        (``write: ...`` or ``upload: ...``) when not ok."""
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
            self.store.remove_tree(relative)  # an older export of this round (a relaunch)
            self.store.put_directory(local, relative)
            self.store.put_files(self._completion(relative, record))
        except Exception as error:  # noqa: BLE001 - the export is evidence (module doc)
            stage = "write" if written is None else "upload"
            record["error"] = f"{stage}: {type(error).__name__}: {error}"[:2000]
        finally:
            shutil.rmtree(local, ignore_errors=True)
        ended = self.clock()
        written = ended if written is None else written
        record.setdefault("files", 0)
        record.setdefault("bytes", 0)
        record.update(ok="error" not in record, write_s=round(written - started, 3),
                      upload_s=round(ended - written, 3), seconds=round(ended - started, 3))
        return record

    @staticmethod
    def _completion(relative: str, record: dict[str, Any]) -> dict[str, bytes]:
        """The second put of an export: its marker, then the final pointer."""
        marker = {k: record[k] for k in ("round", "final", "policy_version", "files", "bytes")}
        files = {f"{relative}/{COMPLETE_MARKER}": _json(marker)}
        if record["final"]:
            files[FINAL_POINTER] = _json({**marker, "path": relative})
        return files
