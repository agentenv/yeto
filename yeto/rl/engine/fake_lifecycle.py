"""CPU fakes for the trainer lifecycle (rl-trainer-start-stop L3.1).

* :class:`FakeMembership` records JOIN / LEAVE on a tape;
* :class:`CuttableFakeTrainer` is the fake trainer with the ``CuttableTrainer``
  verbs ``save_cut`` / ``restore_cut`` (weights, optimizer moments, step on disk);
* :class:`FakeRoundCuts` is the round-cut holder ``LocalOnlySync`` uses
  (``save`` at the final safe point, ``resume`` at start). It writes every cut
  through :func:`recovery.save` and is also the lifecycle's :class:`CutLog`.

Not a GPU substitute.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from yeto.rl.elastic_client import LeaveReason

from . import recovery
from .fake import FakeEngine, FakeRolloutPool, FakeTrainerGroup
from .resume import CheckpointStore
from .trainer_lifecycle import CutRef, SampleIndexEntry

POINTER = "lifecycle-cut.json"


class FakeMembership:
    """``tape``: ``("JOIN", incarnation)`` / ``("LEAVE", incarnation, reason)``."""

    def __init__(self, base_version: int | None = None) -> None:
        self.base_version = base_version
        self.tape: list[tuple] = []

    def join(self, incarnation: int) -> int | None:
        self.tape.append(("JOIN", int(incarnation)))
        return self.base_version

    def leave(self, incarnation: int, reason: LeaveReason) -> None:
        self.tape.append(("LEAVE", int(incarnation), LeaveReason(reason)))


class CuttableFakeTrainer(FakeTrainerGroup):
    """The fake trainer with the cut verbs. A cut is one file per cut id."""

    def layout(self) -> dict[str, int]:
        return {"tp": 1, "pp": 1, "dp": 1}

    def actual_layout(self) -> dict[str, int]:
        return self.layout()

    def save_cut(self, *, epoch: int, context: Any) -> str:
        e = self.engine
        cut_id, root = context["cut_id"], Path(context["root"])
        root.mkdir(parents=True, exist_ok=True)
        e.calls.append(("save_cut", cut_id))
        torch.save({"epoch": int(epoch), "tensors": e.tensors, "moments": e.moments,
                    "step": e.scheduler_step}, root / f"{cut_id}.pt")
        return cut_id

    def restore_cut(self, cut_id: str, *, epoch: int, root: str, expect: Any,
                    shared_filesystem: bool = True) -> Any:
        e = self.engine
        e.calls.append(("restore_cut", cut_id))
        payload = torch.load(Path(root) / f"{cut_id}.pt", weights_only=True)
        if int(payload["epoch"]) != int(epoch):
            raise ValueError(f"cut {cut_id} is at epoch {payload['epoch']}, not {epoch}")
        if set(payload["tensors"]) != set(e.tensors):
            raise ValueError("layout mismatch")
        for name, value in payload["tensors"].items():
            e.tensors[name].copy_(value)
        for name, value in payload["moments"].items():
            e.moments[name].copy_(value)
        e.scheduler_step = int(payload["step"])
        return {"cut_id": cut_id, "local_step": e.scheduler_step}


def cuttable_fake_engine(**kwargs: Any) -> FakeEngine:
    """A fake engine (a freshly acquired trainer) whose trainer has the cut verbs."""
    engine = FakeEngine(**kwargs)
    engine.trainer = CuttableFakeTrainer(engine)
    return engine


class FakeRoundCuts:
    """The round-cut holder of ``LocalOnlySync`` for fakes, and a ``CutLog``.

    Cuts only at the end of an incarnation (``final=True``). The pointer holds the
    cut record (:data:`recovery.CUT_KEY`) and the published version at the cut; a
    later start restores that cut into its fresh trainer."""

    def __init__(self, root: str | os.PathLike[str], *,
                 watermark: Callable[[], int | None] | None = None) -> None:
        self.store = CheckpointStore(root)
        self.watermark = watermark  # the durable buffer's consumed watermark (D4)
        self.root = Path(root)
        self.saved: list[dict[str, Any]] = []
        self.resumed: dict[str, Any] | None = None

    @property
    def pointer_path(self) -> Path:
        return self.root / POINTER

    def pointer(self) -> dict[str, Any] | None:
        path = self.pointer_path
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def latest(self) -> CutRef | None:
        info = self.pointer()
        if info is None:
            return None
        cut = recovery.ConsistentCut.from_record(info[recovery.CUT_KEY])
        return CutRef(cut_id=cut.checkpoint_id, trigger=cut.trigger,
                      policy_version=int(info["policy_version"]),
                      watermark=cut.buffer_watermark)

    def save(self, driver: Any, *, rollout_id: int, final: bool = False,
             trigger: str | None = None) -> dict[str, Any] | None:
        if not final:
            return None
        current = self.pointer()
        if current is not None and int(current["policy_version"]) == int(rollout_id):
            return None  # nothing trained since this cut
        cut_id = f"r{int(rollout_id):06d}-n{len(self.saved)}"
        driver.trainer.save_cut(epoch=0, context={"cut_id": cut_id, "root": str(self.root)})
        cut = recovery.ConsistentCut(checkpoint_id=cut_id, trigger=trigger or recovery.MANUAL,
                                     outer_step=int(rollout_id), island_steps={"0": int(driver.local_step)},
                                     buffer_watermark=None if self.watermark is None else self.watermark())
        info = {"cut_id": cut_id, "policy_version": int(rollout_id), "local_step": int(driver.local_step)}
        saved = recovery.save(cut, store=self.store, write=lambda record: self._write(info, record))
        self.saved.append(saved)
        return {**info, recovery.CUT_KEY: saved[recovery.CUT_KEY]}

    def _write(self, info: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
        tmp = self.pointer_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({**info, recovery.CUT_KEY: record}, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.pointer_path)
        return {}

    def resume(self, driver: Any) -> dict[str, Any] | None:
        info = self.pointer()
        if info is None:
            return None
        driver.trainer.restore_cut(info["cut_id"], epoch=0, root=str(self.root), expect=None)
        driver.local_step = int(info["local_step"])
        self.resumed = info
        return {"next_rollout_id": int(info["policy_version"]), **info}


class BufferFedFakeRollout(FakeRolloutPool):
    """The fake rollout pool of a trainer that reads the durable buffer: each round
    selects one batch, and ``commit_round`` (called by the lifecycle after the
    round's boundary) commits its cursor. ``trained`` lists each round's sample
    ids and behavior versions."""

    def __init__(self, engine: FakeEngine, buffer: Any, *, consumer: str = "trainer") -> None:
        super().__init__(engine)
        self.buffer = buffer
        self.consumer = consumer
        self.batch = engine.groups * engine.samples_per_group
        self.trained: dict[int, list[tuple[str, int]]] = {}
        self._cursor: dict[int, int] = {}

    def generate(self, rollout_id: int, *, expected_policy_version: str | None = None) -> Any:
        selected = self.buffer.select(self.consumer, self.batch)
        if len(selected) < self.batch:
            raise RuntimeError(f"backlog has {len(selected)} samples, a batch needs {self.batch}")
        batch = super().generate(rollout_id, expected_policy_version=expected_policy_version)
        spg = self.engine.samples_per_group
        groups = tuple(replace(g, sample_ids=tuple(s.entry.sample_id for s in selected[i * spg:(i + 1) * spg]))
                       for i, g in enumerate(batch.groups))
        self.trained[rollout_id] = [(s.entry.sample_id, s.entry.behavior_version) for s in selected]
        self._cursor[rollout_id] = selected[-1].position + 1
        return replace(batch, groups=groups)

    def commit_round(self, rollout_id: int) -> None:
        self.buffer.commit_consumed(self._cursor.pop(rollout_id))


class FakeSampler:
    """A sampler that appends to the durable buffer with the version it loaded.
    ``load`` takes the newest published version (D5); while no trainer runs,
    ``latest_published`` returns None and the sampler keeps its version."""

    def __init__(self, buffer: Any, latest_published: Callable[[], int | None], *, name: str = "s0") -> None:
        self.buffer = buffer
        self.latest_published = latest_published
        self.name = name
        self.loaded_version = 0
        self.produced = 0

    def load(self) -> int:
        version = self.latest_published()
        if version is not None and version > self.loaded_version:
            self.loaded_version = int(version)
        return self.loaded_version

    def produce(self, n: int) -> None:
        for _ in range(n):
            sample_id = f"{self.name}-{self.produced:05d}"
            ref = self.buffer.write_payload(sample_id, f"{sample_id}@v{self.loaded_version}".encode())
            self.buffer.append(SampleIndexEntry(sample_id, self.loaded_version, ref, tokens=6))
            self.produced += 1
