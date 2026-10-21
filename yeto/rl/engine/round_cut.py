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
from dataclasses import dataclass
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

    def context(self, cut_id: str, *, pending: tuple[int, Any] | None = None) -> Any:
        """The cut context at the safe point after the current publication.

        ``pending`` = ``(version, state)``: build it before that publication, for
        the state the driver is about to publish as ``version`` (the policy token
        is the one the publication will set). Only valid when nothing but the
        publication happens until the safe point (colocated, generation done);
        :meth:`RoundCutCheckpoint.save` keeps such a cut only if the publication
        then names the same policy."""
        from yeto.rl.engine.controller import RebuildRefused
        from yeto.rl.engine.cut import CutProgress
        from yeto.rl.engine.cut import CutContext
        from yeto.rl.engine.driver import policy_token

        driver = self.driver
        if pending is not None:
            # the token is read last (below): a deferred digest (E1) hashes meanwhile
            version, state = int(pending[0]), pending[1]
            settled, token = True, None
        else:
            version, state = driver.published_version, driver.published_state
            settled, token = bool(driver.at_safe_point), driver.expected_token
        if state is None or version is None:
            raise RebuildRefused("no published policy to cut")
        if getattr(state, "verified", True) is False:
            # E1: a cut carries the policy's content hash (resume checks it); a round
            # whose publication was not hashed is never committed as a cut
            raise RebuildRefused(f"the publication of policy v{version} was not verified "
                                 "(no content hash); see --rl-verify-every-round")
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
            if exported.get("data_cursor") is not None:
                # read together with the entries (a rollout that draws while the
                # cut is taken, e.g. Miles fully_async): keep cursor and entries consistent
                data.update(exported["data_cursor"])
            data["buffer_length"] = int(exported["buffer_groups"])
            summary["carried_over"] = len(in_flight)
            summary["max_policy_age"] = limit
        progress = CutProgress(
            local_step=int(driver.local_step),
            scheduler_samples=int(driver.local_step) * self.global_batch_size,
            global_batch_size=self.global_batch_size,
            next_rollout_id=int(version),
            policy_version=int(state.policy_version),
            policy_hash=state.policy_tensor_hash(),
        )
        if token is None:
            token = policy_token(version, progress.policy_hash)
        return CutContext(
            root=self.cut_root, cut_id=cut_id, backend_fingerprint=self.backend_fingerprint,
            progress=progress, algorithm=self.identity, data=data, ledger=summary,
            # The safe point is after boundary() returned non-stop and every
            # member acknowledged the policy: the outer commit is settled.
            outer={"settled": settled, "policy_version": state.policy_version,
                   "policy_token": token},
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


@dataclass(frozen=True)
class _PreparedCut:
    """The trainer part of a cut written before the publication (``prepare``)."""

    rollout_id: int
    context: Any  # CutContext the shards and manifest were written with
    save_s: float
    pending: Any = None  # the trainer's pending cut (start_cut); None: already written

    def matches(self, rollout_id: int, context: Any) -> bool:
        """The safe point's own context names the same cut: same id, progress
        (version, policy hash, steps), data cursor, ledger summary, outer commit
        (settled, policy token) and in-flight entries."""
        mine = self.context
        return (self.rollout_id == rollout_id and mine.cut_id == context.cut_id
                and mine.progress == context.progress and dict(mine.data) == dict(context.data)
                and dict(mine.ledger) == dict(context.ledger) and dict(mine.outer) == dict(context.outer)
                and tuple(mine.in_flight or ()) == tuple(context.in_flight or ()))


class RoundCutCheckpoint:
    """``save`` at a safe point, ``resume`` at sync start (see the module doc).

    rl-resume-from-checkpoint (S17 C4): ``every`` (``--rl-cut-every``) cuts only every N
    rounds (plus the last round and our own stop, ``final=True``); the pointer carries
    ``run_fingerprint`` / ``lr_at_next_round`` / ``consumed_prompt_ids_digest``; a resume
    checks the fingerprint (refused on a difference unless ``allow_config_change``), the
    consumed-group digest and, on the first resumed round, the learning rate."""

    def __init__(self, *, controller: Any, source: Any, trainer: Any, every: int = 1,
                 fingerprint: dict[str, Any] | None = None, allow_config_change: bool = False,
                 next_lr: Any = None, background_commit: bool | None = None) -> None:
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
        # E1: a cut prepared before the trainer offloads (see prepare()), consumed by save()
        self._prepared: _PreparedCut | None = None
        # E1: a periodic cut writes its shards and commits the store in the background
        # (the controller must offer commit_checkpoint_store); completed before the next
        # cut and at the final cut. YETO_RL_CUT_BACKGROUND_COMMIT=0 does it all inline.
        if background_commit is None:
            background_commit = os.environ.get("YETO_RL_CUT_BACKGROUND_COMMIT", "1").strip().lower() \
                not in {"0", "false", "no", "off"}
        self.background_commit = bool(background_commit) and callable(
            getattr(controller, "commit_checkpoint_store", None))
        self._commit: tuple[Any, dict[str, Any]] | None = None  # (thread, result box)
        # E1: a periodic cut captured at its safe point, completed by join_commit()
        self._deferred: dict[str, Any] | None = None
        self._commit_info: dict[str, Any] | None = None  # the running commit's cut
        self.skipped_busy = 0  # periodic cuts skipped while a commit was running
        # E1: prepared cuts a busy commit made the safe point skip; their trainer-side
        # writers are collected (and their files dropped) at the next cut, when done
        self._abandoned: list[_PreparedCut] = []
        self._declined: int | None = None  # rollout whose prepare met a running commit

    # ---------------------------------------------------------------- pointer
    @property
    def pointer_path(self) -> Path:
        return Path(self.controller.state_dir) / POINTER

    def pointer(self) -> dict[str, Any] | None:
        path = self.pointer_path
        if not path.is_file():
            return None
        return dict(json.loads(path.read_text(encoding="utf-8")))

    def _write_pointer(self, info: dict[str, Any], *, path: Path | None = None) -> None:
        target = self.pointer_path if path is None else Path(path)
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(info, sort_keys=True), encoding="utf-8")
        with open(tmp, "rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, target)

    def _stage(self, cut_id: str) -> Path:
        """E1: a copy of the state dir as it is now (small: ledger, journal, pointer),
        next to it, for a store sync that runs after later rounds."""
        from yeto.rl.engine.resume import stage_state_dir

        state_dir = Path(self.controller.state_dir)
        return stage_state_dir(state_dir, state_dir.parent / f".{state_dir.name}.stage-{cut_id}")

    @staticmethod
    def _unstage(staged: Path | None) -> None:
        if staged is not None:
            import shutil

            shutil.rmtree(staged, ignore_errors=True)

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

    def _skip(self, driver: Any, rollout_id: int, *, final: bool) -> bool:
        """No cut at this safe point: not due, island RECOVERY_REQUIRED, or just
        resumed from (or already cut at) this rollout."""
        if self.controller.recovery_required or not self.due(rollout_id, final=final):
            return True
        current = self.pointer()
        return current is not None and int(current["next_rollout_id"]) == int(rollout_id)

    def _cut_id(self, rollout_id: int) -> str:
        return f"r{int(rollout_id):06d}-{self.controller.incarnation['id']}"

    def _start(self, context: Any, *, background: bool) -> Any:
        """Start the trainer part of a cut: a pending cut whose shards are written in
        the background (trainers with ``start_cut``), else the finished cut (``None``)."""
        start = getattr(self.trainer, "start_cut", None)
        if background and callable(start):
            return start(epoch=self._epoch(), context=context, background=True)
        self.trainer.save_cut(epoch=self._epoch(), context=context)
        return None

    def prepare(self, driver: Any, *, rollout_id: int, state: Any, final: bool = False) -> bool:
        """E1: take the trainer part of the cut due at the safe point before rollout
        ``rollout_id`` NOW, while the trainer is still resident, for the ``state`` the
        driver is about to publish as ``rollout_id``. Saves the wake/sleep a cut of an
        offloaded colocated trainer costs. The pointer and the store sync stay at the
        safe point (:meth:`save`), which keeps the cut only if the publication names
        the same policy; otherwise the prepared files are never pointed at (dropped).
        Returns True when a cut was prepared. Raises like :meth:`save`."""
        import time

        self._prepared = None
        self._reap_abandoned()
        if driver.published_version is None or self._skip(driver, rollout_id, final=final):
            return False
        if not final and self._commit is not None and self._commit[0].is_alive():
            self._declined = int(rollout_id)  # save() skips it if still busy, else _unprepared_wake
            return False  # save() skips this periodic cut (_commit_busy)
        if int(state.policy_version) != int(rollout_id):
            raise RoundCutError(f"prepared cut {rollout_id} for policy v{state.policy_version}")
        self.join_commit(driver, wait=False)  # the previous cut is complete before this one starts
        cut_id = self._cut_id(rollout_id)
        self._drop_unpointed(cut_id)
        started = time.monotonic()
        context = self.source.context(cut_id, pending=(int(rollout_id), state))
        pending = self._start(context, background=self._defer(final))
        self._prepared = _PreparedCut(rollout_id=int(rollout_id), context=context,
                                      save_s=time.monotonic() - started, pending=pending)
        return True

    def _commit_busy(self, driver: Any, rollout_id: int, *, final: bool) -> bool:
        """E1: a periodic cut is skipped while the previous cut's store commit still
        runs (a slow store, e.g. a Modal volume commit of 2-10 s, would otherwise hold
        the round); the next safe point cuts again. The final cut waits for it."""
        if final or self._commit is None or not self._commit[0].is_alive():
            return False
        self.skipped_busy += 1
        emit = getattr(driver, "emit", None)
        if callable(emit):
            emit("rl_cut_skipped", rollout_id=int(rollout_id), reason="previous cut still committing",
                 committing=self._commit[1].get("cut_id"), skipped_total=self.skipped_busy)
        return True

    def _unprepared_wake(self, driver: Any, rollout_id: int, prepared: "_PreparedCut | None", *,
                         final: bool) -> bool:
        """E1: skip a periodic cut whose prepare was declined (it met a running commit)
        when it would wake the offloaded trainer
        (~2 s on the Modal A10G sync run e1a7) and a committed cut already exists; the
        next safe point cuts again. The final cut and the first cut are always taken."""
        declined, self._declined = self._declined, None
        if final or declined != int(rollout_id) or prepared is not None:
            return False
        if self.pointer() is None or not getattr(driver, "_trainer_offloaded", False):
            return False
        self.skipped_busy += 1
        emit = getattr(driver, "emit", None)
        if callable(emit):
            emit("rl_cut_skipped", rollout_id=int(rollout_id), reason="not prepared before the trainer offloaded",
                 skipped_total=self.skipped_busy)
        return True

    def _defer(self, final: bool) -> bool:
        """Write the trainer shards and commit the store in the background: periodic
        cuts only (the final cut is complete when the run ends), and not when a sample
        buffer commits the same store during the round (no two commits at once), and
        not before the first cut of the store is committed (a store holding cuts but
        no LATEST refuses to start)."""
        return (self.background_commit and not final and self.buffer_watermark is None
                and self.pointer() is not None)

    def _drop_unpointed(self, cut_id: str) -> None:
        import shutil

        from yeto.rl.engine.cut import cut_dir

        directory = cut_dir(self.root, cut_id)
        if directory.exists():
            shutil.rmtree(directory)

    def join_commit(self, driver: Any = None, *, wait: bool = True) -> dict[str, Any] | None:
        """Complete the previous background cut: wait for a running store commit,
        then collect the pending cut's shards, commit its manifest, point the state
        dir at it and start its store commit (``wait``: and wait for that too). Each
        finished commit is reported (``rl_cut_committed``); on any failure the
        previous pointer stays the committed one. Returns the last outcome (None:
        nothing was pending)."""
        result = self._join_running(driver)
        deferred, self._deferred = self._deferred, None
        if deferred is not None:
            done = self._complete_deferred(deferred)
            if not done.get("ok", True):
                self._report(driver, done)
                return done
            self._commit_info = done
            result = self._join_running(driver) if wait else done
        return result

    def _join_running(self, driver: Any) -> dict[str, Any] | None:
        if self._commit is None:
            return None
        thread, box = self._commit
        thread.join()
        self._commit = None
        result = {**(self._commit_info or {}), **dict(box.get("result") or {
            "ok": False, "error": "commit thread ended without a result"})}
        self._commit_info = None
        result.setdefault("cut_id", box.get("cut_id"))
        self._report(driver, result)
        return result

    @staticmethod
    def _report(driver: Any, result: Mapping[str, Any]) -> None:
        emit = getattr(driver, "emit", None)
        if callable(emit):
            emit("rl_cut_committed", **{k: result.get(k) for k in (
                "cut_id", "rollout_id", "ok", "finish_s", "commit_s", "pruned", "seq", "cut_bytes")},
                **({"writer": result["writer"]} if result.get("writer") else {}),
                **({"error": result["error"]} if result.get("error") else {}))

    def _complete_deferred(self, deferred: dict[str, Any]) -> dict[str, Any]:
        """Collect the pending cut's shards from the trainer (this thread: the trainer
        is driven from here), then commit its manifest, point the state dir at it and
        commit the store in the background (:meth:`_commit_in_background`)."""
        import time

        info = deferred["info"]
        started = time.monotonic()
        try:
            collect = getattr(self.trainer, "collect_cut", None)
            if callable(collect):
                commit_manifest = collect(deferred["pending"])
            else:
                self.trainer.finish_cut(deferred["pending"])
                commit_manifest = None
        except Exception as error:  # noqa: BLE001 - the previous cut stays the pointed one
            self._drop_unpointed(info["cut_id"])
            self._unstage(deferred.get("staged"))
            return {"cut_id": info["cut_id"], "rollout_id": deferred["rollout_id"], "ok": False,
                    "error": repr(error)[:2000]}
        self._commit_in_background(deferred, commit_manifest)
        writer = getattr(self.trainer, "last_cut_writer", None)
        return {"cut_id": info["cut_id"], "rollout_id": deferred["rollout_id"],
                "collect_s": round(time.monotonic() - started, 3), **({"writer": writer} if writer else {})}

    def _commit_in_background(self, deferred: dict[str, Any], commit_manifest: Any) -> None:
        """E1: off the round's critical path: the cut manifest (``commit_manifest``, None
        when the trainer committed it already), the pointer with the cut record, the
        state dir copied into the store, the store commit. Until the pointer moves the
        previous cut stays the pointed one; a failure before that drops the new cut's
        files. The outcome lands in the result box, reported by :meth:`join_commit`."""
        import threading
        import time

        from yeto.rl.engine.resume import directory_bytes

        info, cut = deferred["info"], deferred["cut"]
        cut_id = info["cut_id"]
        box: dict[str, Any] = {"cut_id": cut_id}

        def run() -> None:
            started = time.monotonic()
            try:
                if commit_manifest is not None:
                    commit_manifest()
                saved = recovery.save(cut, store=self.controller.store,
                                      write=lambda record: self._write_and_sync(
                                          info, record, defer=True, staged=deferred.get("staged")))
            except Exception as error:  # noqa: BLE001 - reported at the next join
                self._drop_unpointed(cut_id)
                box["result"] = {"ok": False, "error": repr(error)[:2000]}
                return
            finally:
                self._unstage(deferred.get("staged"))
            done = {"finish_s": round(time.monotonic() - started, 3),
                    "cut_bytes": directory_bytes(Path(self.root) / cut_id)}
            if not saved["synced"]:
                box["result"] = {**done, "ok": False, "error": (
                    getattr(self.controller, "last_store_sync", None) or {}).get("error")}
                return
            box["result"] = {**done, **dict(self.controller.commit_checkpoint_store() or {})}

        thread = threading.Thread(target=run, name=f"yeto-cut-commit-{cut_id}", daemon=True)
        self._commit = (thread, box)
        thread.start()

    def save(self, driver: Any, *, rollout_id: int, final: bool = False,
             trigger: str | None = None) -> dict[str, Any] | None:
        """Cut the trainer at the safe point before rollout ``rollout_id`` (= the published
        version), point the state dir at it, sync the store. Returns the pointer, or None
        when nothing was cut (not due, no publication yet, already cut at this rollout,
        island RECOVERY_REQUIRED). Raises on a refused/failed cut (the caller logs it).

        A periodic cut (:meth:`_defer`) only captures the state here: the shards are
        written, the pointer moved and the store committed in the background, completed
        by :meth:`join_commit` before the next cut (``store_commit`` "deferred"; the
        outcome is the ``rl_cut_committed`` event). Until then the previous cut stays
        the committed one.

        ``trigger`` names a final cut's trigger (default ``MANUAL``; ``SCHEDULED`` for
        a stop the trainer scheduler ordered); a periodic cut takes none."""
        import time

        from yeto.rl.engine.resume import directory_bytes

        if trigger is not None and (not final or trigger not in FINAL_TRIGGERS):
            raise RoundCutError(f"cut trigger {trigger!r} needs final=True and one of {FINAL_TRIGGERS}")
        ctl = self.controller
        prepared, self._prepared = self._prepared, None
        if driver.published_version is None or self._skip(driver, rollout_id, final=final):
            self._abandon(prepared)
            return None
        if self._commit_busy(driver, rollout_id, final=final):
            # its trainer may still be writing the shards: collect them at the next
            # cut instead of waiting for the writer here (E1: up to ~0.9 s per round)
            if prepared is not None:
                self._abandoned.append(prepared)
            return None
        self._reap_abandoned()
        if self._unprepared_wake(driver, rollout_id, prepared, final=final):
            self._abandon(prepared)
            return None
        if int(driver.published_version) != int(rollout_id):
            self._abandon(prepared)
            raise RoundCutError(f"safe point {rollout_id} != published version {driver.published_version}")
        self.join_commit(driver, wait=final)  # the final cut leaves nothing running
        started = time.monotonic()
        context = self.source.context(self._cut_id(rollout_id))
        progress = context.progress
        offloaded = False
        defer = self._defer(final)
        used_prepared = prepared is not None and prepared.matches(int(rollout_id), context)
        if used_prepared:
            save_s, pending = prepared.save_s, prepared.pending  # taken before the trainer offloaded
        else:
            self._abandon(prepared)
            # a failed prepare or an earlier attempt left files under this id; no pointer
            # names them (_skip), and cuts are immutable: drop them
            self._drop_unpointed(context.cut_id)
            # colocated + offloaded trainer (1 GPU): its weights/optimizer are not on the
            # device between publish and the next train step; wake it for the cut only.
            offloaded = bool(getattr(driver, "_trainer_offloaded", False))
            if offloaded:
                self.trainer.onload()
            try:
                pending = self._start(context, background=defer)
            finally:
                if offloaded:
                    self.trainer.offload()
            save_s = time.monotonic() - started
        if pending is None:
            defer = False  # the trainer wrote the cut already (no start_cut)
        cut_id = context.cut_id
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
        extra = {"prepared_before_offload": used_prepared, "trainer_onloaded_for_cut": offloaded,
                 "save_s": round(save_s, 3)}
        capture = pending.get("capture") if isinstance(pending, Mapping) else None
        if capture:
            # E1: per rank, device copies or the host path (and why: e.g. low device memory)
            extra["cut_capture"] = capture
        if defer:
            # the state dir (ledger, journal) as it is at this cut: its store sync runs
            # in the background after later rounds have changed it
            staged = self._stage(cut_id)
            self._deferred = {"info": info, "cut": cut, "pending": pending, "rollout_id": int(rollout_id),
                              "staged": staged}
            return {**info, **extra, "store_commit": "deferred", "store_synced": None,
                    "total_s": round(time.monotonic() - started, 3)}
        if pending is not None:
            self.trainer.finish_cut(pending)
        saved = recovery.save(cut, store=ctl.store, write=lambda record: self._write_and_sync(info, record))
        info[CUT_KEY] = saved[CUT_KEY]
        synced = saved["synced"]
        sync = dict(getattr(ctl, "last_store_sync", None) or {})
        cut_bytes = directory_bytes(Path(self.root) / cut_id)
        info.update(extra)
        info["store_synced"] = bool(synced)
        info["store_commit"] = "inline"
        info["cut_bytes"] = cut_bytes
        info["store_copy_s"] = sync.get("copy_s")
        info["store_commit_s"] = sync.get("commit_s")
        info["total_s"] = round(time.monotonic() - started, 3)
        info["write_bytes_per_s"] = (cut_bytes / save_s) if save_s > 0 else None
        info["pruned"] = sync.get("pruned")
        info["latest_seq"] = sync.get("seq")
        if not synced:
            info["store_error"] = sync.get("error")
        return info

    def _reap_abandoned(self) -> None:
        abandoned, self._abandoned = self._abandoned, []
        for prepared in abandoned:
            self._abandon(prepared)

    def _abandon(self, prepared: "_PreparedCut | None") -> None:
        """A prepared cut the safe point does not keep: wait for its writer, drop its files."""
        if prepared is None:
            return
        if prepared.pending is not None:
            try:
                self.trainer.finish_cut(prepared.pending)
            except Exception:  # noqa: BLE001 - dropped below either way
                pass
        self._drop_unpointed(prepared.context.cut_id)

    def _write_and_sync(self, info: dict[str, Any], cut_record: dict[str, Any], *,
                        defer: bool = False, staged: Path | None = None) -> dict[str, Any]:
        """The round cut's ``recovery.save`` writer: the pointer (with the cut record)
        into the state dir, then the state dir into the store (which commits, unless
        ``defer``: the caller commits in the background). ``staged``: the state dir as
        staged at the cut (:meth:`_stage`) is the one synced, with the same pointer."""
        pointer = {**info, CUT_KEY: cut_record}
        self._write_pointer(pointer)
        kwargs: dict[str, Any] = {"defer_commit": True} if defer else {}
        if staged is not None:
            self._write_pointer(pointer, path=Path(staged) / POINTER)
            kwargs["source_dir"] = Path(staged)
        synced = self.controller.sync_checkpoint_store("round_cut", skip_if_recovery=True, **kwargs)
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
