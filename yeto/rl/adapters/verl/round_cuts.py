"""verl island round cuts: when to cut, the pointer, stop triggers, resume (S25 N7.6b).

The verl sync island runs the same cut controller as the Miles island
(:mod:`yeto.rl.engine.round_cut`), wired the same way: with ``--rl-resume-store``
and no outer syncer (``--rl-single-island-no-sync``),

* a :class:`~yeto.rl.engine.resume.ResumeController` restores the store's
  ``LATEST`` snapshot into the state dir at start (pointer ``round-cut.json``,
  batch ledger, journal) and syncs the state dir back after every cut;
* a :class:`~yeto.rl.engine.round_cut.RoundCutCheckpoint` on the
  :class:`~yeto.rl.engine.bridges.LocalOnlySync` cuts the trainer
  (:class:`~.cut.VerlCutTrainer`: one yeto file per FSDP rank) every
  ``--rl-cut-every`` rounds, at the last round and at our own stop
  (``--rl-stop-after-rounds``), and resumes from the pointed cut at start;
* triggers of the final cut: ``MANUAL`` (last round, our own stop),
  ``SCHEDULED`` (the head's trainer scheduler stopped us:
  ``trainer_remote`` commands, :func:`run_driver`); a preemption signal only
  writes a marker (``install_preempt_handler`` on the main thread; in a Ray
  actor ``watch_preempt_marker`` turns the Modal runner's reclaim marker file
  into the same notice, S26 E9.1): the next launch resumes from
  the newest complete cut.

The resume assertions of :mod:`.cut` run on this path: the restored rank digests
and Adam steps at restore, the landed data cursor against the cut's data
watermark (driver ``_restore_data_cursor`` -> ``confirm_resumed_data_cursor``),
and the Adam/scheduler advance after the first resumed round.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

RESUME_FLAGS = ("rl_cut_every", "rl_cut_keep", "rl_stop_after_rounds")
DEFAULT_STATE_DIR = "~/yeto-rl/resume-state"
# VerlRunConfig fields that may change between a cut and the launch resuming it:
# paths of this container, run length (extending a run is the normal resume),
# and settings that do not change the trained numbers.
FINGERPRINT_EXCLUDED = ("out_dir", "train_file", "val_file", "total_rounds", "experiment_name",
                        "gpu_memory_utilization", "reward_path")


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def resume_config(args: Any) -> dict[str, Any] | None:
    """``plan["resume"]`` from the island flags (None: no resume store). Raises on
    a flag combination the cut controller cannot run."""
    store = getattr(args, "rl_resume_store", None)
    given = [_flag(n) for n in RESUME_FLAGS if getattr(args, n, None) is not None]
    if getattr(args, "rl_resume_allow_config_change", False):
        given.append("--rl-resume-allow-config-change")
    for name in RESUME_FLAGS:
        value = getattr(args, name, None)
        if value is not None and int(value) < 1:
            raise ValueError(f"{_flag(name)} must be >= 1")
    if not store:
        if given:
            raise ValueError(", ".join(given) + " need --rl-resume-store")
        return None
    from yeto.rl.engine import sample_roles

    if sample_roles.sample_role(args) == sample_roles.TRAINER:
        # a buffer trainer (S25 C2) names its sample buffer's store with
        # --rl-resume-store; round cuts are not wired on that path
        if given:
            raise ValueError(", ".join(given) + ": round cuts are not wired for a verl buffer "
                             "trainer (--rl-sample-role trainer)")
        return None
    if not getattr(args, "rl_single_island_no_sync", False):
        raise ValueError("--rl-resume-store needs --rl-single-island-no-sync (a synced island "
                         "resumes from the syncer)")
    if int(getattr(args, "rl_max_policy_age", 0) or 0):
        raise ValueError("--rl-resume-store is not wired for the verl fully_async path "
                         "(--rl-max-policy-age > 0)")
    return {
        "store": str(store),
        "state_dir": str(getattr(args, "rl_resume_state_dir", None) or DEFAULT_STATE_DIR),
        "every": int(getattr(args, "rl_cut_every", None) or 1),
        "keep": int(getattr(args, "rl_cut_keep", None) or 2),
        "allow_config_change": bool(getattr(args, "rl_resume_allow_config_change", False)),
        "stop_after": (None if getattr(args, "rl_stop_after_rounds", None) is None
                       else int(args.rl_stop_after_rounds)),
    }


def run_fingerprint_of(plan: Mapping[str, Any], algorithm: Any) -> dict[str, Any]:
    """What must be identical between a cut and the launch resuming it."""
    run = {k: v for k, v in dict(plan.get("run_config") or {}).items() if k not in FINGERPRINT_EXCLUDED}
    sha = getattr(algorithm, "sha256", None)
    return {**run, "model_revision": plan.get("model_revision"), "lora_targets": plan.get("lora_targets"),
            "algorithm_spec_sha256": sha() if callable(sha) else None}


@dataclass
class ResumeWiring:
    """The round-cut controller and batch ledger of one verl island run."""

    controller: Any  # ResumeController
    ledger: Any  # BatchLedger (in the state dir: restored and synced with the pointer)
    config: dict[str, Any]


def resume_wiring_for(plan: Mapping[str, Any], *, runtime_fingerprint: str,
                      environ: Mapping[str, str] | None = None) -> ResumeWiring | None:
    """Restore the store into the state dir (before the driver is built: the
    ledger the driver gets is the restored one). None without ``plan["resume"]``."""
    config = plan.get("resume")
    if not config:
        return None
    if plan.get("sync") != "none":
        raise ValueError("verl round cuts need sync 'none' (--rl-single-island-no-sync)")
    if not runtime_fingerprint:
        raise ValueError("verl round cuts need a runtime fingerprint (a cut without it is refused)")
    from yeto.rl.engine.ledger import BatchLedger
    from yeto.rl.engine.resume import ResumeController, store_for

    store = store_for(config["store"], environ=environ)
    controller = ResumeController(state_dir=os.path.expanduser(config["state_dir"]), store=store,
                                  runtime_fingerprint=runtime_fingerprint, keep=int(config.get("keep", 2)))
    return ResumeWiring(controller=controller, ledger=BatchLedger(controller.state_dir), config=dict(config))


def wire_round_cuts(driver: Any, resume: ResumeWiring | None, *, plan: Mapping[str, Any],
                    algorithm: Any, install_signals: bool = True) -> Any:
    """Give the driver's LocalOnlySync the cut controller (the Miles wiring, verl
    units: one scheduler step per round, so the cut's ``global_batch_size`` is 1).
    Returns the :class:`~yeto.rl.engine.round_cut.RoundCutCheckpoint` or None."""
    if resume is None:
        return None
    from yeto.rl.engine import round_cut
    from yeto.rl.engine.resume import install_preempt_handler, watch_preempt_marker

    controller = resume.controller
    source = round_cut.CutSource(
        driver=lambda: driver, trainer=driver.trainer, rollout=driver.rollout, ledger=resume.ledger,
        algorithm=algorithm, backend_fingerprint=controller.runtime_fingerprint, cut_root="",
        global_batch_size=1)
    checkpoint = round_cut.wire_round_cuts(
        driver, controller=controller, source=source, every=int(resume.config.get("every", 1)),
        fingerprint=run_fingerprint_of(plan, algorithm),
        allow_config_change=bool(resume.config.get("allow_config_change", False)))
    if checkpoint is None:
        raise ValueError("verl round cuts need the driver's sync to be LocalOnlySync")
    driver.sync.stop_after = resume.config.get("stop_after")
    # the handler needs the main thread (verl: the task runner actor's); [] when it is not
    signals = install_preempt_handler(controller, emit=driver.emit) if install_signals else []
    # S26 E9.1: the signal reaches the container's main process, not this actor; the
    # Modal runner writes the reclaim marker and this thread turns it into the notice
    watcher = watch_preempt_marker(controller) if install_signals else None
    driver.emit("rl_round_cuts_wired", backend="verl", every=checkpoint.every,
                keep=int(controller.keep), stop_after=driver.sync.stop_after,
                store=str(controller.checkpoint_store), preempt_signals=list(signals),
                preempt_marker_watch=watcher is not None,
                restored_store=controller.restored is not None)
    return checkpoint


def run_driver(driver: Any, resume: ResumeWiring | None, *, environ: Mapping[str, str] | None = None) -> Any:
    """Run the island: as one commanded incarnation when the head's trainer
    scheduler starts and stops it (``trainer_remote``: its stop cuts with the
    SCHEDULED trigger), else to the end of the run. Returns the final policy state."""
    from yeto.rl.engine import trainer_remote

    if not trainer_remote.commanded(environ):
        return driver.run()
    if resume is None:
        raise RuntimeError("a trainer started by the head's scheduler needs round cuts "
                           "(--rl-resume-store with --rl-single-island-no-sync)")
    trainer_remote.run_commanded_island(driver, store=resume.controller.store, environ=environ)
    return driver.published_state
