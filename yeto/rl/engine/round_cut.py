"""Round cuts of a run without an outer syncer: save at safe points, resume at start.

Neutral core (moved from the Miles adapter for the verl island, S25 N7.6b): the
backend gives a ``CuttableTrainer`` (``save_cut`` / ``restore_cut`` /
``actual_layout``) and a rollout pool with ``data_cursor`` / ``seek_data_cursor``;
:class:`CutSource` builds the cut context from the driver, and
:class:`RoundCutCheckpoint` is the cut controller wired into
:class:`~.bridges.LocalOnlySync` (:func:`wire_round_cuts`).

rl-multinode-island M4: policy-weight continuation across a machine replacement.


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
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from . import recovery
from .recovery import CUT_KEY, FINAL_TRIGGERS, MANUAL, PERIODIC, ConsistentCut
from .resume import ROUND_CUTS  # under the checkpoint store (not under the state dir)

ISLAND = "local"  # round cuts run without an outer syncer: one island, the local one
POINTER = "round-cut.json"  # under the state dir (synced with the ledger/journal)


class RoundCutError(RuntimeError):
    pass


def _algorithm_identity(algorithm: Any, *, ref_model: Mapping[str, Any] | None) -> Any:
    from yeto.rl.engine.cut import AlgorithmIdentity

    runtime_attrs = None
    to_attrs = getattr(algorithm, "to_legacy_runtime_attrs", None)
    if callable(to_attrs):
        runtime_attrs = dict(to_attrs()) or None
    return AlgorithmIdentity.from_spec(algorithm, runtime_attrs=runtime_attrs, ref_model=ref_model)


class CutSource:
    """What the driver contributes to a cut (E2 ``CutContext``) and what a
    restore must match (``RestoreExpectation``); shared by the 4.4 same-shape
    rebuilder and E3's ``MilesTrainerOps`` (4.7)."""

    # backend check of an in-flight export: problems refuse the cut (None: no check;
    # the Miles adapter overrides it with its export check)
    check_in_flight: Callable[[Mapping[str, Any]], list[str]] | None = None

    def __init__(self, *, driver: Any, trainer: Any, rollout: Any, ledger: Any, algorithm: Any,
                 backend_fingerprint: str, cut_root: str, global_batch_size: int,
                 ref_model: Mapping[str, Any] | None = None,
                 shared_filesystem: bool = True) -> None:
        self.driver_ref = driver  # the driver, or a zero-arg callable returning it
        self.trainer = trainer
        self.rollout = rollout
        self.ledger = ledger
        self.identity = _algorithm_identity(algorithm, ref_model=ref_model)
        self.backend_fingerprint = backend_fingerprint
        self.cut_root = cut_root
        self.global_batch_size = int(global_batch_size)
        self.shared_filesystem = shared_filesystem

    @property
    def driver(self) -> Any:
        ref = self.driver_ref
        return ref() if callable(ref) and not hasattr(ref, "published_state") else ref

    def context(self, cut_id: str) -> Any:
        from yeto.rl.engine.controller import RebuildRefused
        from yeto.rl.engine.cut import CutProgress
        from yeto.rl.engine.cut import CutContext

        driver = self.driver
        state = driver.published_state
        if state is None or driver.published_version is None:
            raise RebuildRefused("no published policy to cut")
        cursor = (self.rollout.data_cursor()
                  if callable(getattr(self.rollout, "data_cursor", None)) else None)
        if cursor is None:
            raise RebuildRefused("rollout data cursor unknown (needs the elastic metadata)")
        data = dict(cursor)
        summary = dict(self.ledger.cut_summary()) if self.ledger is not None else {}
        if summary.get("engine_buffer_length") is not None:
            data.setdefault("buffer_length", summary["engine_buffer_length"])
        in_flight: tuple = ()
        limit = int(getattr(getattr(driver, "profile", None), "max_policy_age", 0) or 0)
        if limit > 0:
            # agentic-rollout-utilization 3.3: the unfinished work the rollout
            # process holds NOW (buffered groups, suspended agentic trajectories).
            exporter = getattr(self.rollout, "export_in_flight", None)
            if not callable(exporter):
                raise RebuildRefused("the rollout pool cannot export its in-flight trajectories")
            exported = exporter()
            problems = list(self.check_in_flight(exported)) if self.check_in_flight else []
            if problems:
                raise RebuildRefused("; ".join(problems))
            in_flight = tuple(exported["entries"])
            data["buffer_length"] = int(exported["buffer_groups"])
            summary["carried_over"] = len(in_flight)
            summary["max_policy_age"] = limit
        progress = CutProgress(
            local_step=int(driver.local_step),
            scheduler_samples=int(driver.local_step) * self.global_batch_size,
            global_batch_size=self.global_batch_size,
            next_rollout_id=int(driver.published_version),
            policy_version=int(state.policy_version),
            policy_hash=state.policy_tensor_hash(),
        )
        return CutContext(
            root=self.cut_root, cut_id=cut_id, backend_fingerprint=self.backend_fingerprint,
            progress=progress, algorithm=self.identity, data=data, ledger=summary,
            # The safe point is after boundary() returned non-stop and every
            # member acknowledged the policy: the outer commit is settled.
            outer={"settled": bool(driver.at_safe_point), "policy_version": state.policy_version,
                   "policy_token": driver.expected_token},
            shared_filesystem=self.shared_filesystem,
            in_flight=in_flight,
        )

    def expectation(self, layout: Mapping[str, int], *, epoch: int) -> Any:
        from yeto.rl.engine.cut import RestoreExpectation

        driver = self.driver
        return RestoreExpectation(
            algorithm=self.identity, layout=dict(layout),
            backend_fingerprint=self.backend_fingerprint, local_step=int(driver.local_step),
            policy_version=int(driver.published_state.policy_version), epoch=epoch,
        )


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
        # set by resume(): where the rollout data source continues (the cut's cursor)
        self.resume_data_cursor: dict[str, int] | None = None
        # rl-trainer-start-stop D4: the durable sample buffer's consumed watermark,
        # written into the cut record (None: the run reads no sample buffer).
        # buffer_watermark(next_rollout_id) commits and returns the position after the
        # last trained round; buffer_resume(watermark) continues there after a restore
        # (buffer_rollout.wire_sample_buffer sets both).
        self.buffer_watermark: Any = None  # callable(int) -> int | None
        self.buffer_resume: Any = None  # callable(int) -> int

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

    def save(self, driver: Any, *, rollout_id: int, final: bool = False,
             trigger: str | None = None) -> dict[str, Any] | None:
        """Cut the trainer at the safe point before rollout ``rollout_id`` (= the published
        version), point the state dir at it, sync the store. Returns the pointer, or None
        when nothing was cut (not due, no publication yet, already cut at this rollout,
        island RECOVERY_REQUIRED). Raises on a refused/failed cut (the caller logs it).

        ``trigger`` names a final cut's trigger (default ``MANUAL``; ``SCHEDULED`` for
        a stop the trainer scheduler ordered); a periodic cut takes none."""
        import time

        if trigger is not None and (not final or trigger not in FINAL_TRIGGERS):
            raise RoundCutError(f"cut trigger {trigger!r} needs final=True and one of {FINAL_TRIGGERS}")
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
        # Hold the store's writer guard from the first shard to the commit: a Modal
        # Volume reload in between remounts the Volume and drops uncommitted files.
        # Seen on Modal (e6-t2-pair-e, 2026-10-21): a reload during the shard
        # snapshot made the rank's makedirs of the store dir fail with EPERM; the
        # cell died and the island exited 134.
        with ctl.store.writing():
            return self._save_guarded(driver, cut_id=cut_id, context=context, rollout_id=rollout_id,
                                      final=final, trigger=trigger, started=started, offloaded=offloaded)

    def _save_guarded(self, driver: Any, *, cut_id: str, context: Any, rollout_id: int, final: bool,
                      trigger: str | None, started: float, offloaded: bool) -> dict[str, Any]:
        """:meth:`save` under the store's writer guard: the trainer cut, the pointer, the store sync."""
        import time

        from yeto.rl.engine.resume import directory_bytes

        ctl = self.controller
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
        cut = ConsistentCut(
            checkpoint_id=cut_id, trigger=(trigger or MANUAL) if final else PERIODIC,
            outer_step=info["outer_version"], island_steps={ISLAND: info["local_step"]},
            completed_groups=consumed_n, completed_groups_digest=consumed_sha,
            buffer_watermark=(self.buffer_watermark(int(rollout_id)) if callable(self.buffer_watermark)
                              else None))
        saved = recovery.save(cut, store=ctl.store, write=lambda record: self._write_and_sync(info, record))
        info[CUT_KEY] = saved[CUT_KEY]
        synced = saved["synced"]
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

    def _write_and_sync(self, info: dict[str, Any], cut_record: dict[str, Any]) -> dict[str, Any]:
        """The round cut's ``recovery.save`` writer: the pointer (with the cut record)
        into the state dir, then the state dir into the store (which commits)."""
        self._write_pointer({**info, CUT_KEY: cut_record})
        synced = self.controller.sync_checkpoint_store("round_cut", skip_if_recovery=True)
        # the store sync commits itself (snapshot layout: in write_snapshot; flat layout:
        # in the controller, outside its lock), also when it failed: recovery.save must not
        return {"synced": bool(synced), "committed": True}

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
            if callable(self.buffer_resume):
                self.buffer_resume(0)
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
        backend_fingerprint = self.source.backend_fingerprint
        if self.allow_config_change:
            # The backend fingerprint hashes the whole Miles argv, so a longer run
            # (--num-rollout, --lr-decay-iters) changes it; allowed changes are recorded.
            from yeto.rl.engine.cut import load_manifest

            saved = load_manifest(self.root, info["cut_id"]).runtime.get("backend_fingerprint")
            if saved != backend_fingerprint:
                checks["backend_fingerprint"] = {"result": "changed(allowed)", "cut": saved,
                                                 "run": backend_fingerprint}
                backend_fingerprint = saved
        expect = RestoreExpectation(
            algorithm=self.source.identity, layout=dict(self.trainer.actual_layout()),
            backend_fingerprint=backend_fingerprint,
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
        in_flight = self._restore_in_flight(driver, manifest, rid)
        if in_flight is not None:
            checks["in_flight"] = in_flight
        self.resume_data_cursor = _cut_data_cursor(manifest)
        if callable(self.buffer_resume):
            watermark = (info.get(CUT_KEY) or {}).get("buffer_watermark")
            if watermark is None:
                raise RoundCutError(f"round cut {info['cut_id']} has no sample buffer watermark "
                                    "but this trainer reads the sample buffer")
            checks["buffer_watermark"] = int(self.buffer_resume(int(watermark)))
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

    def _restore_in_flight(self, driver: Any, manifest: Any, rollout_id: int) -> dict[str, Any] | None:
        """agentic-rollout-utilization 3.3: the cut's unfinished trajectories, by the
        CURRENT run's limit: within it and not already trained -> continued
        (buffered groups back into the data buffer); else discarded and reported.
        Suspended agentic trajectories never survive a restart (discarded).
        None when the cut carries none."""
        entries = tuple(getattr(manifest, "in_flight", ()) or ())
        if not entries:
            return None
        limit = int(getattr(getattr(driver, "profile", None), "max_policy_age", 0) or 0)
        ledger = getattr(self.source, "ledger", None)
        completed = sorted(getattr(ledger, "_consumed_groups", None) or ())
        rollout = getattr(self.source, "rollout", None)
        importer = getattr(rollout, "import_in_flight", None)
        if not callable(importer):
            raise RoundCutError("the cut carries in-flight trajectories but the rollout pool "
                                "cannot import them")
        report = importer(entries, max_policy_age=limit, current_version=int(rollout_id),
                          completed_group_ids=completed)
        driver.emit("rl_in_flight_restored", rollout_id=int(rollout_id), max_policy_age=limit,
                    entries=len(entries), **report)
        return report

    def check_first_round(self, driver: Any, *, rollout_id: int, applied_lrs: Any) -> None:
        """Design §4.5, on the boundary of the first resumed round (before it is recorded
        or published): the trained learning rate continues bitwise. Raises, unless
        ``allow_config_change`` (the event then records the difference)."""
        from yeto.rl.engine.resume import check_lr_continues

        pending = self.pending_lr_check
        if pending is None or int(rollout_id) != int(pending["rollout_id"]):
            return
        self.pending_lr_check = None
        problem = check_lr_continues(pending["expected"], None if applied_lrs is None else list(applied_lrs))
        driver.emit("rl_resume_check", rollout_id=int(rollout_id), check="lr_at_next_round",
                    expected=pending["expected"],
                    applied=None if applied_lrs is None else [float(x) for x in applied_lrs],
                    ok=problem is None, allowed=problem is not None and self.allow_config_change)
        if problem is not None and not self.allow_config_change:
            raise RoundCutError(problem)


def _cut_data_cursor(manifest: Any) -> dict[str, int] | None:
    """The data source position recorded at the cut (None when incomplete).

    Under fully_async the background worker keeps drawing prompts after the
    round's trained groups were reported, and the cut puts the untrained ones
    back into the data buffer. The position to resume from is therefore the
    one read at the cut, not the one the batch ledger recorded for the round."""
    from .cut import DATA_CURSOR_FIELDS

    data = dict(getattr(manifest, "data", None) or {})
    # a backend with fewer position fields (verl: sample_offset) names them in the cut
    fields = tuple(data.get("cursor_fields") or DATA_CURSOR_FIELDS)
    if not all(isinstance(data.get(f), int) for f in fields):
        return None
    return {f: int(data[f]) for f in fields}


def wire_round_cuts(driver: Any, *, controller: Any, source: Any, **options: Any) -> RoundCutCheckpoint | None:
    """Give a :class:`~..bridges.LocalOnlySync` driver round cuts when a checkpoint store is
    configured (other syncs own the policy: their syncer is authoritative)."""
    from .bridges import LocalOnlySync

    if getattr(controller, "checkpoint_store", None) is None or not isinstance(driver.sync, LocalOnlySync):
        return None
    checkpoint = RoundCutCheckpoint(controller=controller, source=source, trainer=driver.trainer, **options)
    driver.sync.round_cuts = checkpoint
    return checkpoint
