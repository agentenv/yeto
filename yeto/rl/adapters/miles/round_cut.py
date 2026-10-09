"""rl-multinode-island M4: policy-weight continuation across a machine replacement.

Without an outer syncer (``--rl-single-island-no-sync``, :class:`~..bridges.LocalOnlySync`)
a restarted learner used to start again at rollout 0 from the startup weights: the
checkpoint store (Q4, C5) carried the journal/ledger, but no LoRA/optimizer state.

With ``--rl-elastic-checkpoint-store`` set this module keeps a 4.2 cut of the trainer
(every rank's LoRA + optimizer + scheduler + RNG shard, manifest committed last) at
every round-boundary safe point and resumes from the newest one at start:

* the cut root is ``<store>/round-cuts`` -- the store is visible on every node (sky
  Storage MOUNT of the bucket), so the shard of a rank on another node (PP across
  nodes) lands where the driver can check it (``shared_filesystem=True``);
* after the manifest is committed, ``round-cut.json`` (the pointer: cut id, next
  rollout id, local step, policy hash, writing incarnation) is written into the state
  dir and the state dir is synced to the store -- pointer, ledger and journal travel
  in ONE store snapshot (``STORE-MANIFEST.json`` last), so a restored pointer never
  names a cut newer than the restored ledger;
* at start (in-place restart or a fresh machine after the store restore) the cut the
  pointer names is verified (``verify_cut``: algorithm, layout, backend, step,
  policy version, every shard digest) and loaded on the fresh trainer
  (``MilesTrainerGroup.restore_cut``); the driver then rebases the ledger and seeks
  the data source to the pointer's rollout id. Any failure is fatal (no silent
  restart at 0 once a pointer exists).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROUND_CUTS = "round-cuts"  # under the checkpoint store (not under the state dir)
POINTER = "round-cut.json"  # under the state dir (synced with the ledger/journal)


class RoundCutError(RuntimeError):
    pass


class RoundCutCheckpoint:
    """``save`` at a safe point, ``resume`` at sync start (see the module doc)."""

    def __init__(self, *, controller: Any, source: Any, trainer: Any) -> None:
        if getattr(controller, "checkpoint_store", None) is None:
            raise ValueError("round cuts need the checkpoint store")
        self.controller = controller
        self.source = source  # rebuild_wiring.CutSource (cut_root is set here)
        self.trainer = trainer  # MilesTrainerGroup: save_cut / restore_cut / actual_layout
        self.root = str(Path(controller.checkpoint_store) / ROUND_CUTS)
        source.cut_root = self.root
        self.resumed: dict[str, Any] | None = None

    # ---------------------------------------------------------------- pointer
    @property
    def pointer_path(self) -> Path:
        return Path(self.controller.state_dir) / POINTER

    def pointer(self) -> dict[str, Any] | None:
        path = self.pointer_path
        if not path.is_file():
            return None
        return dict(json.loads(path.read_text(encoding="utf-8")))

    def _write_pointer(self, info: dict[str, Any]) -> None:
        tmp = self.pointer_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(info, sort_keys=True), encoding="utf-8")
        with open(tmp, "rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, self.pointer_path)

    def _epoch(self) -> int:
        return int(self.controller.journal.epochs.config_epoch)

    # ---------------------------------------------------------------- save
    def save(self, driver: Any, *, rollout_id: int) -> dict[str, Any] | None:
        """Cut the trainer at the safe point before rollout ``rollout_id`` (= the published
        version), point the state dir at it, sync the store. Returns the pointer, or None
        when nothing was cut (no publication yet, already cut at this rollout, island
        RECOVERY_REQUIRED). Raises on a refused/failed cut (the caller logs it)."""
        ctl = self.controller
        if ctl.recovery_required or driver.published_version is None:
            return None
        if int(driver.published_version) != int(rollout_id):
            raise RoundCutError(f"safe point {rollout_id} != published version {driver.published_version}")
        current = self.pointer()
        if current is not None and int(current["next_rollout_id"]) == int(rollout_id):
            return None  # just resumed from (or already cut) this point
        cut_id = f"r{int(rollout_id):06d}-{ctl.incarnation['id']}"
        context = self.source.context(cut_id)
        self.trainer.save_cut(epoch=self._epoch(), context=context)
        progress = context.progress
        info = {
            "cut_id": cut_id, "root": ROUND_CUTS, "epoch": self._epoch(),
            "next_rollout_id": int(progress.next_rollout_id),
            "local_step": int(progress.local_step),
            "policy_version": int(progress.policy_version),
            "policy_hash": progress.policy_hash,
            "incarnation": ctl.incarnation["id"],
        }
        self._write_pointer(info)
        info["store_synced"] = bool(ctl.sync_checkpoint_store("round_cut", skip_if_recovery=True))
        return info

    # ---------------------------------------------------------------- resume
    def resume(self, driver: Any) -> dict[str, Any] | None:
        """Load the pointed cut into the freshly built trainer and set the driver's local
        step. Returns the pointer (``next_rollout_id`` = where the run continues), or None
        when this island never cut (fresh run). Raises on any inconsistency."""
        from yeto.rl.engine.cut import RestoreExpectation

        info = self.pointer()
        if info is None:
            return None
        rid = int(info["next_rollout_id"])
        ledger = getattr(self.source, "ledger", None)
        if ledger is not None and rid > 0:
            last = ledger.batch(rid - 1)
            if last is None or last.get("state") != "outer_recorded":
                raise RoundCutError(f"round cut {info['cut_id']} continues at rollout {rid} but the "
                                    f"ledger has no outer-recorded rollout {rid - 1}: {last}")
        expect = RestoreExpectation(
            algorithm=self.source.identity, layout=dict(self.trainer.actual_layout()),
            backend_fingerprint=self.source.backend_fingerprint,
            local_step=int(info["local_step"]), policy_version=int(info["policy_version"]),
            epoch=self._epoch(),
        )
        manifest = self.trainer.restore_cut(info["cut_id"], epoch=self._epoch(), root=self.root,
                                            expect=expect, shared_filesystem=True)
        if manifest.progress.policy_hash != info["policy_hash"] or manifest.progress.next_rollout_id != rid:
            raise RoundCutError(f"round cut {info['cut_id']} manifest disagrees with its pointer")
        restored = driver.export_local().policy_tensor_hash()
        if restored != info["policy_hash"]:
            raise RoundCutError(f"restored trainer holds {restored}, round cut holds {info['policy_hash']}")
        driver.local_step = int(info["local_step"])
        self.resumed = dict(info)
        self.controller._record("round_cut", tx_id=None, action="restore", cut_id=info["cut_id"],
                                next_rollout_id=rid, local_step=int(info["local_step"]),
                                policy_hash=info["policy_hash"], cut_incarnation=info["incarnation"],
                                incarnation=self.controller.incarnation["id"])
        driver.emit("rl_round_cut_restored", rollout_id=rid, cut_id=info["cut_id"],
                    local_step=int(info["local_step"]), policy_hash=info["policy_hash"],
                    cut_incarnation=info["incarnation"])
        return info


def wire_round_cuts(driver: Any, *, controller: Any, source: Any) -> RoundCutCheckpoint | None:
    """Give a :class:`~..bridges.LocalOnlySync` driver round cuts when a checkpoint store is
    configured (other syncs own the policy: their syncer is authoritative)."""
    from yeto.rl.engine.bridges import LocalOnlySync

    if getattr(controller, "checkpoint_store", None) is None or not isinstance(driver.sync, LocalOnlySync):
        return None
    checkpoint = RoundCutCheckpoint(controller=controller, source=source, trainer=driver.trainer)
    driver.sync.round_cuts = checkpoint
    return checkpoint
