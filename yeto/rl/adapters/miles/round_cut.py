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
    """``save`` at a safe point, ``resume`` at sync start (see the module doc).

    rl-resume-from-checkpoint (S17 C4): ``every`` (``--rl-cut-every``) cuts only every N
    rounds (plus the last round and our own stop, ``final=True``); the pointer carries
    ``run_fingerprint`` / ``lr_at_next_round`` / ``consumed_prompt_ids_digest``; a resume
    checks the fingerprint (refused on a difference unless ``allow_config_change``), the
    consumed-group digest and, on the first resumed round, the learning rate."""

    def __init__(self, *, controller: Any, source: Any, trainer: Any, every: int = 1,
                 fingerprint: dict[str, Any] | None = None, allow_config_change: bool = False,
                 next_lr: Any = None) -> None:
        if getattr(controller, "checkpoint_store", None) is None:
            raise ValueError("round cuts need the checkpoint store")
        if int(every) < 1:
            raise ValueError("--rl-cut-every must be >= 1")
        self.controller = controller
        self.source = source  # rebuild_wiring.CutSource (cut_root is set here)
        self.trainer = trainer  # MilesTrainerGroup: save_cut / restore_cut / actual_layout
        self.root = str(Path(controller.checkpoint_store) / ROUND_CUTS)
        source.cut_root = self.root
        self.resumed: dict[str, Any] | None = None
        self.every = int(every)
        self.fingerprint = dict(fingerprint) if fingerprint is not None else None
        self.allow_config_change = bool(allow_config_change)
        self.next_lr = next_lr  # callable(driver) -> list[float] | None
        self.pending_lr_check: dict[str, Any] | None = None

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

    def _consumed(self) -> tuple[int, str | None]:
        ledger = getattr(self.source, "ledger", None)
        groups = getattr(ledger, "_consumed_groups", None)
        if groups is None:
            return 0, None
        from yeto.rl.engine.resume import consumed_digest

        return len(groups), consumed_digest(groups)

    # ---------------------------------------------------------------- save
    def due(self, rollout_id: int, *, final: bool = False) -> bool:
        from yeto.rl.engine.resume import should_cut

        return should_cut(int(rollout_id), every=self.every, final=final)

    def save(self, driver: Any, *, rollout_id: int, final: bool = False) -> dict[str, Any] | None:
        """Cut the trainer at the safe point before rollout ``rollout_id`` (= the published
        version), point the state dir at it, sync the store. Returns the pointer, or None
        when nothing was cut (not due, no publication yet, already cut at this rollout,
        island RECOVERY_REQUIRED). Raises on a refused/failed cut (the caller logs it)."""
        import time

        from yeto.rl.engine.resume import directory_bytes

        ctl = self.controller
        if ctl.recovery_required or driver.published_version is None:
            return None
        if not self.due(rollout_id, final=final):
            return None
        if int(driver.published_version) != int(rollout_id):
            raise RoundCutError(f"safe point {rollout_id} != published version {driver.published_version}")
        current = self.pointer()
        if current is not None and int(current["next_rollout_id"]) == int(rollout_id):
            return None  # just resumed from (or already cut) this point
        cut_id = f"r{int(rollout_id):06d}-{ctl.incarnation['id']}"
        started = time.monotonic()
        context = self.source.context(cut_id)
        # colocated + offloaded trainer (1 GPU): its weights/optimizer are not on the
        # device between publish and the next train step; wake it for the cut only.
        offloaded = bool(getattr(driver, "_trainer_offloaded", False))
        if offloaded:
            self.trainer.onload()
        try:
            self.trainer.save_cut(epoch=self._epoch(), context=context)
        finally:
            if offloaded:
                self.trainer.offload()
        save_s = time.monotonic() - started
        progress = context.progress
        consumed_n, consumed_sha = self._consumed()
        next_lr = self.next_lr(driver) if callable(self.next_lr) else None
        info = {
            "cut_id": cut_id, "root": ROUND_CUTS, "epoch": self._epoch(),
            "next_rollout_id": int(progress.next_rollout_id),
            "local_step": int(progress.local_step),
            "policy_version": int(progress.policy_version),
            "policy_hash": progress.policy_hash,
            "incarnation": ctl.incarnation["id"],
            "incarnation_index": ctl.incarnation.get("index"),
            "run_fingerprint": self.fingerprint,
            "lr_at_next_round": None if next_lr is None else [float(x) for x in next_lr],
            "consumed_groups": consumed_n,
            "consumed_prompt_ids_digest": consumed_sha,
            "outer_version": getattr(getattr(driver, "sync", None), "outer_version", None),
            "final": bool(final),
        }
        self._write_pointer(info)
        synced = ctl.sync_checkpoint_store("round_cut", skip_if_recovery=True)
        sync = dict(getattr(ctl, "last_store_sync", None) or {})
        cut_bytes = directory_bytes(Path(self.root) / cut_id)
        total_s = time.monotonic() - started
        info["store_synced"] = bool(synced)
        info["cut_bytes"] = cut_bytes
        info["save_s"] = round(save_s, 3)
        info["store_copy_s"] = sync.get("copy_s")
        info["store_commit_s"] = sync.get("commit_s")
        info["total_s"] = round(total_s, 3)
        info["write_bytes_per_s"] = (cut_bytes / save_s) if save_s > 0 else None
        info["pruned"] = sync.get("pruned")
        info["latest_seq"] = sync.get("seq")
        info["trainer_onloaded_for_cut"] = offloaded
        if not synced:
            info["store_error"] = sync.get("error")
        return info

    # ---------------------------------------------------------------- resume
    def resume(self, driver: Any) -> dict[str, Any] | None:
        """Load the pointed cut into the freshly built trainer and set the driver's local
        step. Returns the pointer (``next_rollout_id`` = where the run continues), or None
        when this island never cut (fresh run). Raises on any inconsistency."""
        import time

        from yeto.rl.engine.cut import RestoreExpectation
        from yeto.rl.engine.resume import check_fingerprint

        info = self.pointer()
        if info is None:
            return None
        started = time.monotonic()
        rid = int(info["next_rollout_id"])
        checks: dict[str, Any] = {}
        diff = {}
        if self.fingerprint is not None and info.get("run_fingerprint") is None:
            checks["run_fingerprint"] = "absent (pointer written before rl-resume-from-checkpoint)"
        elif self.fingerprint is not None:
            diff = check_fingerprint(info.get("run_fingerprint"), self.fingerprint,
                                     allow_change=self.allow_config_change)
            checks["run_fingerprint"] = "same" if not diff else "changed(allowed)"
        ledger = getattr(self.source, "ledger", None)
        if ledger is not None and rid > 0:
            last = ledger.batch(rid - 1)
            if last is None or last.get("state") != "outer_recorded":
                raise RoundCutError(f"round cut {info['cut_id']} continues at rollout {rid} but the "
                                    f"ledger has no outer-recorded rollout {rid - 1}: {last}")
            checks["ledger_last_round"] = "outer_recorded"
        if info.get("consumed_prompt_ids_digest") is not None:
            n, sha = self._consumed()
            if sha != info["consumed_prompt_ids_digest"] or n != int(info.get("consumed_groups", n)):
                raise RoundCutError(f"round cut {info['cut_id']}: the restored ledger's consumed groups "
                                    f"({n}, {sha}) differ from the pointer's "
                                    f"({info.get('consumed_groups')}, {info['consumed_prompt_ids_digest']})")
            checks["consumed_prompt_ids_digest"] = "same"
        expect = RestoreExpectation(
            algorithm=self.source.identity, layout=dict(self.trainer.actual_layout()),
            backend_fingerprint=self.source.backend_fingerprint,
            local_step=int(info["local_step"]), policy_version=int(info["policy_version"]),
            epoch=self._epoch(),
        )
        manifest = self.trainer.restore_cut(info["cut_id"], epoch=self._epoch(), root=self.root,
                                            expect=expect, shared_filesystem=True)
        checks["cut_shards_sha256"] = "ok"
        if manifest.progress.policy_hash != info["policy_hash"] or manifest.progress.next_rollout_id != rid:
            raise RoundCutError(f"round cut {info['cut_id']} manifest disagrees with its pointer")
        restored = driver.export_local().policy_tensor_hash()
        if restored != info["policy_hash"]:
            raise RoundCutError(f"restored trainer holds {restored}, round cut holds {info['policy_hash']}")
        checks["policy_hash"] = "same"
        checks["policy_version"] = int(info["policy_version"])
        driver.local_step = int(info["local_step"])
        self.resumed = dict(info)
        if info.get("lr_at_next_round") is not None:
            self.pending_lr_check = {"rollout_id": rid, "expected": list(info["lr_at_next_round"])}
        self.controller._record("round_cut", tx_id=None, action="restore", cut_id=info["cut_id"],
                                next_rollout_id=rid, local_step=int(info["local_step"]),
                                policy_hash=info["policy_hash"], cut_incarnation=info["incarnation"],
                                incarnation=self.controller.incarnation["id"])
        driver.emit("rl_round_cut_restored", rollout_id=rid, cut_id=info["cut_id"],
                    local_step=int(info["local_step"]), policy_hash=info["policy_hash"],
                    cut_incarnation=info["incarnation"])
        restored_store = getattr(self.controller, "restored", None) or {}
        cut_bytes = sum(int(f.bytes) for f in manifest.files)
        driver.emit(
            "rl_resume", rollout_id=rid, cut_id=info["cut_id"],
            incarnation=self.controller.incarnation.get("index"),
            incarnation_id=self.controller.incarnation["id"],
            cut_incarnation=info["incarnation"], cut_incarnation_index=info.get("incarnation_index"),
            policy_version=int(info["policy_version"]), policy_hash=info["policy_hash"],
            local_step=int(info["local_step"]), lr_at_next_round=info.get("lr_at_next_round"),
            checks=checks, config_diff=diff or None,
            discarded_rounds=list(getattr(self, "discarded_rounds", []) or []),
            restore_s=round(time.monotonic() - started, 3), cut_bytes=cut_bytes,
            state_restore=restored_store or None,
        )
        return info

    def check_first_round(self, driver: Any, *, rollout_id: int, applied_lrs: Any) -> None:
        """Design §4.5, on the boundary of the first resumed round (before it is recorded
        or published): the trained learning rate continues bitwise. Raises."""
        from yeto.rl.engine.resume import check_lr_continues

        pending = self.pending_lr_check
        if pending is None or int(rollout_id) != int(pending["rollout_id"]):
            return
        self.pending_lr_check = None
        problem = check_lr_continues(pending["expected"], None if applied_lrs is None else list(applied_lrs))
        driver.emit("rl_resume_check", rollout_id=int(rollout_id), check="lr_at_next_round",
                    expected=pending["expected"],
                    applied=None if applied_lrs is None else [float(x) for x in applied_lrs],
                    ok=problem is None)
        if problem is not None:
            raise RoundCutError(problem)


def wire_round_cuts(driver: Any, *, controller: Any, source: Any, **options: Any) -> RoundCutCheckpoint | None:
    """Give a :class:`~..bridges.LocalOnlySync` driver round cuts when a checkpoint store is
    configured (other syncs own the policy: their syncer is authoritative)."""
    from yeto.rl.engine.bridges import LocalOnlySync

    if getattr(controller, "checkpoint_store", None) is None or not isinstance(driver.sync, LocalOnlySync):
        return None
    checkpoint = RoundCutCheckpoint(controller=controller, source=source, trainer=driver.trainer, **options)
    driver.sync.round_cuts = checkpoint
    return checkpoint
