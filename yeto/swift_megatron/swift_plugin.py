"""ms-swift Megatron hook layer for yeto full-parameter DiLoCo.

Load this file with ms-swift's official plugin mechanism::

    megatron sft ... \\
        --external_plugins /path/to/yeto/swift_megatron/swift_plugin.py \\
        --callbacks yeto_diloco

Importing the module registers ``YetoDiLoCoCallback`` under the name
``yeto_diloco`` in ``swift.megatron.callbacks.megatron_callbacks_map``.
``BaseMegatronTrainer.__init__`` instantiates every name in ``args.callbacks``
after the model, optimizer and resume checkpoint are loaded, so the callback
sees the final model chunks, the (distributed / offloaded) optimizer and the
resumed iteration.

If adding ``--callbacks`` to the production command line is not possible, set
``YETO_DILOCO_AUTO_ATTACH=1``: the import then wraps
``BaseMegatronTrainer.__init__`` so the callback is added automatically, ahead
of ms-swift's own callbacks.

Call contract:
  * every rank calls ``sync.after_optimizer_step(iteration, target_tokens)``
    exactly once per ``optimizer.step()``, including skipped steps
    (loss-scale overflow / NaN grad norm), where ``target_tokens`` is 0;
  * ``iteration`` is the 1-based count of completed train steps
    (``trainer.state.iteration`` after ms-swift increments it), so it continues
    from the checkpoint iteration on resume;
  * ``target_tokens`` is the number of loss tokens of this step summed over
    all micro-batches and the whole data-parallel (+context-parallel) group,
    identical on every rank of the island;
  * ``sync.close()`` is called from ``on_train_end`` (and at interpreter exit
    as a fallback if training aborts).

The sync runs inside ``on_step_end``, i.e. before ms-swift's own
``default_flow`` callback decides to save, so a checkpoint written at that
step already contains the merged base. Once the syncer's final cut is
installed the callback ends training at that step (ms-swift then saves), so
later local steps cannot overwrite the final global parameters.

Configuration (environment variable / ``--yeto_*`` style is not accepted by
ms-swift's argument parser, so environment variables are the only channel):

  YETO_DILOCO_SYNCER              syncer address (required)
  YETO_DILOCO_LEARNER_ID          learner id (int, required)
  YETO_DILOCO_NUM_LEARNERS        number of learners (int, required)
  YETO_DILOCO_FRAGMENTS           fragment count (int, default 1)
  YETO_DILOCO_SYNC_INTERVAL_STEPS inner steps per round (int, required)
  YETO_DILOCO_WIRE_DTYPE          bf16 | fp32 | int8 | int4 | int4-log | push-bf16
                                  (default bf16)
  YETO_DILOCO_FINAL_CUT_PAYLOAD   full | manifest (default full; a large island
                                  should use manifest: the full final cut is
                                  every fragment in FP32 on rank 0's host)
  YETO_DILOCO_TRANSPORT           cpu | device: where the sync's collectives
                                  run (default cpu, gloo mirrors)
  YETO_DILOCO_HMAC_KEY_PATH       path to the HMAC key file (optional)
  YETO_DILOCO_DISABLE             set to 1 to make the callback a no-op
  YETO_DILOCO_INITIAL_CUT_TIMEOUT seconds to install the syncer's initial cut
                                  (or a resumed island's catch-up) fragment by
                                  fragment before training; default 3600.  0 =
                                  apply it at the first boundaries instead: the
                                  whole cut (4 bytes per parameter) then queues
                                  on rank 0 first, and a resumed island takes it
                                  only after one more step
  YETO_DILOCO_RESUME_DIR          checkpoint directory whose sync record to
                                  resume from (default: ``args.mcore_model``
                                  when it holds one)
  YETO_DILOCO_SAVE_ANCHORS        1 = also save every rank's anchor shards
                                  with each checkpoint (~2 bytes per local
                                  parameter per rank for bf16); default 0
  YETO_DILOCO_RESUME_ANCHOR_MMAP  1 (default) = map the saved anchor file on
                                  resume instead of reading it into anonymous
                                  memory.  Same values, but the shards the
                                  restart never needs stay out of RSS (120B
                                  pp0: ~30 GiB per rank, ~240 GiB per host).
                                  0 restores the eager read.
  YETO_DILOCO_ANCHOR_DTYPE        fp32 | bf16; default: the broadcast dtype
                                  (FP32 for an int4/int8 push session).  bf16
                                  halves the resident anchors but rounds the
                                  pushed delta and the resume shift -- only
                                  for an island that otherwise does not fit.
  YETO_DILOCO_INIT_PROBE_S        resumed learner 0 with an initial-cut wait:
                                  send INIT_PARAMS only if the syncer sends no
                                  catch-up within this many seconds (an
                                  initialized syncer ignores INIT); default 1800
  YETO_DILOCO_HEARTBEAT_S         seconds between rank 0's HEARTBEATs to the
                                  syncer, sent from a thread (independent of
                                  step ends) so its --island-lease-s can be on;
                                  default 30, 0 = none
  YETO_DILOCO_HEARTBEAT_STALL_S   stop heartbeating once the training loop
                                  made no progress for this long, so a hung
                                  island loses its lease; default 0 (never)
  YETO_DILOCO_LR_RESUME           auto (default) | off: rebuild the learning
                                  rate schedule of a resume that did not load
                                  it (``restore_lr_schedule``)
  YETO_DILOCO_STEP_LOG            path: sync rank 0 appends one JSON line per
                                  step (sync time, export / apply split,
                                  push / apply counts)
  YETO_DILOCO_HOST_MEM_CENSUS     1 = add this rank's and the node's host
                                  memory (``host_memory_census``) to each
                                  step log line; default 0

Resume: every checkpoint also gets the sync's record (``on_save`` ->
``MegatronDiLoCoSync.save_state``): each fragment's anchor version and
H-gate counters.  On a restart from that checkpoint the syncer's catch-up
no longer overwrites the island: a fragment still at its checkpointed
version keeps its local values (anchor only), a newer one becomes
``global_new + (local - anchor_old)`` if the anchors were saved and the
global otherwise; without a record (a cold start) the island takes the
syncer's cut as before.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def _ensure_repo_on_path() -> None:
    """Make ``import yeto`` work when ms-swift loads this file by path.

    ``--external_plugins`` imports the file directly, so the yeto checkout is
    not necessarily importable; add the repository root (two levels above
    this package) so no PYTHONPATH has to be forwarded to every node.
    """
    import sys

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if os.path.isdir(os.path.join(root, "yeto")) and root not in sys.path:
        sys.path.append(root)


_ensure_repo_on_path()

CALLBACK_NAME = "yeto_diloco"
ENV_PREFIX = "YETO_DILOCO_"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiLoCoHookConfig:
    syncer: str
    learner_id: int
    num_learners: int
    fragments: int
    sync_interval_steps: int
    wire_dtype: str = "bf16"
    hmac_key_path: Optional[str] = None
    final_cut_payload: str = "full"
    transport: str = "cpu"
    initial_cut_timeout: float = 3600.0
    resume_dir: Optional[str] = None
    save_anchors: bool = False
    init_probe_s: float = 1800.0
    heartbeat_s: float = 30.0
    heartbeat_stall_s: float = 0.0
    anchor_dtype: Optional[str] = None
    resume_anchor_mmap: bool = True

    @classmethod
    def from_env(cls, env=None) -> "DiLoCoHookConfig":
        env = os.environ if env is None else env

        def get(name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
            value = env.get(ENV_PREFIX + name)
            if value is None or value == "":
                if required:
                    raise ValueError(f"{ENV_PREFIX}{name} must be set to enable yeto DiLoCo sync")
                return default
            return value

        def get_int(name: str, default: Optional[int] = None, required: bool = False) -> int:
            raw = get(name, None if default is None else str(default), required)
            try:
                return int(raw)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                raise ValueError(f"{ENV_PREFIX}{name} must be an integer, got {raw!r}") from None

        def get_float(name: str, default: float) -> float:
            raw = get(name, str(default))
            try:
                return float(raw)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                raise ValueError(f"{ENV_PREFIX}{name} must be a number, got {raw!r}") from None

        cfg = cls(
            syncer=get("SYNCER", required=True),  # type: ignore[arg-type]
            learner_id=get_int("LEARNER_ID", required=True),
            num_learners=get_int("NUM_LEARNERS", required=True),
            fragments=get_int("FRAGMENTS", default=1),
            sync_interval_steps=get_int("SYNC_INTERVAL_STEPS", required=True),
            wire_dtype=get("WIRE_DTYPE", default="bf16"),  # type: ignore[arg-type]
            hmac_key_path=get("HMAC_KEY_PATH"),
            final_cut_payload=get("FINAL_CUT_PAYLOAD", default="full"),  # type: ignore[arg-type]
            transport=get("TRANSPORT", default="cpu"),  # type: ignore[arg-type]
            initial_cut_timeout=get_float("INITIAL_CUT_TIMEOUT", 3600.0),
            resume_dir=get("RESUME_DIR"),
            save_anchors=(get("SAVE_ANCHORS", default="0") or "").strip().lower() in {"1", "true", "yes", "on"},
            init_probe_s=get_float("INIT_PROBE_S", 1800.0),
            heartbeat_s=get_float("HEARTBEAT_S", 30.0),
            heartbeat_stall_s=get_float("HEARTBEAT_STALL_S", 0.0),
            anchor_dtype=get("ANCHOR_DTYPE"),
            resume_anchor_mmap=(get("RESUME_ANCHOR_MMAP", default="1") or "").strip().lower()
            in {"1", "true", "yes", "on"},
        )
        if cfg.initial_cut_timeout < 0:
            raise ValueError(f"{ENV_PREFIX}INITIAL_CUT_TIMEOUT must be >= 0")
        if cfg.num_learners < 1 or not 0 <= cfg.learner_id < cfg.num_learners:
            raise ValueError(f"invalid learner_id={cfg.learner_id} for num_learners={cfg.num_learners}")
        if cfg.fragments < 1 or cfg.sync_interval_steps < 1:
            raise ValueError("fragments and sync_interval_steps must be >= 1")
        if cfg.final_cut_payload not in ("full", "manifest"):
            raise ValueError(f"{ENV_PREFIX}FINAL_CUT_PAYLOAD must be full or manifest")
        if cfg.transport not in ("cpu", "device"):
            raise ValueError(f"{ENV_PREFIX}TRANSPORT must be cpu or device")
        if cfg.hmac_key_path is not None and not os.path.isfile(cfg.hmac_key_path):
            raise ValueError(f"{ENV_PREFIX}HMAC_KEY_PATH does not exist: {cfg.hmac_key_path}")
        return cfg


_PROC_GIB = 1024.0 * 1024.0  # /proc reports kB


def _proc_gib(path: str, keys) -> dict:
    out = {}
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                name, _, rest = line.partition(":")
                if name in keys:
                    out[name] = round(int(rest.split()[0]) / _PROC_GIB, 3)
    except (OSError, ValueError, IndexError):
        return {}
    return out


def _pinned_host_stats() -> dict:
    """The accelerator's host (pinned) allocator counters, in GiB.

    torch_npu's caching host allocator is where the CPU-offloaded optimizer's
    D2H/H2D staging lives.  Those bytes are driver memory and do not move with
    ``malloc_trim``, so they are worth watching -- but only two of these
    counters are gauges, and mistaking the others for gauges is exactly how
    this allocator produces a leak that is not there.  Measured on one card:

    * ``allocated_bytes.current`` IS a gauge: the bytes the allocator holds
      from the OS.  Freeing a tensor leaves it unchanged (the block is
      cached); ``host_empty_cache()`` is what brings it down.
    * ``active_bytes.current`` is NOT.  ``active_bytes.freed`` stays 0, so
      "current" is every byte ever handed out and rises for the life of the
      process by construction.  On a 120B island it rose +0.155 GiB per
      rank per step for 90 steps while the gauge sat at 28 GiB and the
      node's MemAvailable did not fall at all.

    So emit ``.current`` beside ``.freed`` and ``.peak`` for every byte
    counter: a reader can then tell a gauge from a running total by looking
    at whether ``.freed`` ever moves, instead of assuming.  No derived
    "idle": subtracting the running total from the gauge gives a number
    that goes negative and stays there.
    """

    try:
        import torch

        stats = dict(torch.npu.host_memory_stats())  # type: ignore[attr-defined]
    except Exception:
        return {}
    out = {}
    try:
        for key, value in stats.items():
            name = str(key)
            if not name.endswith((".current", ".freed", ".peak")) or "_bytes" not in name:
                continue
            if isinstance(value, (int, float)):
                out["pinned_" + name.replace(".", "_")] = round(value / 2**30, 3)
    except Exception:
        return {}
    return out


def host_memory_census() -> dict:
    """One flat dict of this rank's host-memory footprint and the node's.

    Costs three small /proc reads and one allocator query, so it is safe to
    call every step; ``YETO_DILOCO_HOST_MEM_CENSUS=1`` adds it to the step
    log.  Never raises: a missing file or a torch without the NPU host
    allocator just leaves those keys out.
    """

    status = _proc_gib("/proc/self/status", ("VmRSS", "VmHWM"))
    roll = _proc_gib("/proc/self/smaps_rollup", ("Anonymous", "Private_Dirty"))
    node = _proc_gib("/proc/meminfo", ("MemAvailable", "AnonPages"))
    out = {}
    if "VmRSS" in status:
        out["rss_gib"], out["hwm_gib"] = status["VmRSS"], status.get("VmHWM", 0.0)
    if "Anonymous" in roll:
        out["anon_gib"], out["private_dirty_gib"] = roll["Anonymous"], roll.get("Private_Dirty", 0.0)
    if "MemAvailable" in node:
        out["node_memavailable_gib"] = node["MemAvailable"]
        out["node_anonpages_gib"] = node.get("AnonPages", 0.0)
    out.update(_pinned_host_stats())
    return out


def _env_flag(name: str) -> bool:
    return os.environ.get(ENV_PREFIX + name, "").strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------


def local_loss_tokens(metrics) -> int:
    """Sum the loss-token counts of this rank's micro-batches.

    ms-swift's ``MegatronTrainer.loss_func`` returns ``{'loss': [sum, n_tokens]}``
    per micro-batch; ``forward_backward_func`` hands back that list on the last
    pipeline stage and an empty list elsewhere.
    """
    total = 0
    for mb in metrics or ():
        if not isinstance(mb, dict):
            continue
        loss = mb.get("loss")
        if loss is None:
            continue
        try:
            total += int(round(float(loss[1])))
        except (TypeError, IndexError):
            continue
    return total


def megatron_global_tokens(local_tokens: int) -> int:
    """Reduce per-rank loss tokens to the island-wide count on every rank.

    SUM over the data-parallel group including context parallel (each DP/CP
    rank sees a disjoint shard of the global batch), then MAX over the
    pipeline group to broadcast the last stage's count to earlier stages.
    TP ranks of one DP replica already hold identical counts.
    """
    import torch
    from megatron.core import parallel_state as mpu

    t = torch.tensor([int(local_tokens)], dtype=torch.long, device=torch.cuda.current_device())
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM,
                                 group=mpu.get_data_parallel_group(with_context_parallel=True))
    if mpu.get_pipeline_model_parallel_world_size() > 1:
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX,
                                     group=mpu.get_pipeline_model_parallel_group())
    return int(t.item())


# ---------------------------------------------------------------------------
# Sync construction
# ---------------------------------------------------------------------------


def trainable_filter(name: str, param) -> bool:
    """Only parameters that the optimizer updates take part in sync."""
    return bool(getattr(param, "requires_grad", False))


def default_sync_factory(model_chunks, optimizer, cfg: DiLoCoHookConfig, *,
                         resume_dir: Optional[str] = None, resume_iteration: Optional[int] = None):
    from yeto.swift_megatron.sync import MegatronDiLoCoSync

    return MegatronDiLoCoSync(
        model_chunks,
        optimizer,
        syncer=cfg.syncer,
        learner_id=cfg.learner_id,
        num_learners=cfg.num_learners,
        fragments=cfg.fragments,
        sync_interval_steps=cfg.sync_interval_steps,
        wire_dtype=cfg.wire_dtype,
        hmac_key_path=cfg.hmac_key_path,
        trainable_filter=trainable_filter,
        final_cut_payload=cfg.final_cut_payload,
        transport=cfg.transport,
        initial_cut_timeout=cfg.initial_cut_timeout,
        resume_dir=resume_dir,
        resume_iteration=resume_iteration,
        init_probe_s=cfg.init_probe_s,
        heartbeat_s=cfg.heartbeat_s,
        heartbeat_stall_s=cfg.heartbeat_stall_s,
        anchor_dtype=cfg.anchor_dtype,
        resume_anchor_mmap=cfg.resume_anchor_mmap,
    )


def resume_dir_for(cfg: DiLoCoHookConfig, args) -> Optional[str]:
    """The checkpoint directory whose sync record a resumed island reads:
    ``YETO_DILOCO_RESUME_DIR``, else the checkpoint ms-swift loaded
    (``--mcore_model``) when it holds a record.  None: a cold start."""
    from yeto.swift_megatron.sync import RESUME_STATE_FILE

    if cfg.resume_dir:
        return cfg.resume_dir
    loaded = getattr(args, "mcore_model", None)
    if isinstance(loaded, str) and os.path.isfile(os.path.join(loaded, RESUME_STATE_FILE)):
        return loaded
    return None


def save_dir_for(args, iteration: int, *call_args, **call_kwargs) -> Optional[str]:
    """The directory of the checkpoint ``on_save`` reports: its
    ``output_dir`` argument, else ``<args.output_dir>/checkpoint-<iteration>``."""
    for value in (call_kwargs.get("output_dir"), call_kwargs.get("save_dir"), *call_args):
        if isinstance(value, (str, os.PathLike)):
            return os.fspath(value)
    root = getattr(args, "output_dir", None) or getattr(args, "save", None)
    return os.path.join(root, f"checkpoint-{iteration}") if root else None


def restore_lr_schedule(trainer) -> Optional[int]:
    """Put a resumed run's LR schedule where an uninterrupted run would be.

    ms-swift loads the ``OptimizerParamScheduler`` state only together with
    the optimizer (``load_mcore_checkpoint``: ``not finetune and not
    no_load_optim``), and a checkpoint saved with ``--no_save_optim`` holds
    no scheduler state at all.  So an island that saves parameters only and
    resumes with ``--no_load_optim true --finetune false`` gets back its
    iteration and data position but restarts the schedule at step 0: warmup
    again, the whole decay shifted by the resumed iteration, silently.

    The scheduler counts samples (ms-swift steps it by ``global_batch_size``
    per successful update), so a fresh scheduler (``num_steps == 0``) at a
    resumed iteration ``i > 0`` is stepped by ``i * global_batch_size`` here,
    which sets every parameter group's lr and weight decay the way ``i``
    uninterrupted updates would.  A scheduler restored from the checkpoint
    (``num_steps > 0``) and a cold start are left alone.  Returns the samples
    stepped, or None.  Runs on every rank (each holds its own optimizer)."""
    if (os.environ.get(ENV_PREFIX + "LR_RESUME", "auto").strip().lower() or "auto") == "off":
        return None
    scheduler = getattr(trainer, "opt_param_scheduler", None)
    iteration = int(getattr(trainer.state, "iteration", 0) or 0)
    if scheduler is None or iteration <= 0 or int(getattr(scheduler, "num_steps", 0) or 0) != 0:
        return None
    samples = iteration * int(trainer.args.global_batch_size)
    scheduler.step(increment=samples)
    logger.warning("yeto DiLoCo: the checkpoint of iteration %d restored no LR schedule; stepped it by %d "
                   "samples to resume it in place", iteration, samples)
    return samples


def _visible_logs() -> None:
    """This module's and the sync core's WARNING/INFO lines would otherwise
    go to the root handler the Megatron entry point leaves at ERROR."""
    _ensure_repo_on_path()
    try:
        from yeto.swift_megatron.sync import ensure_visible_logs, log as sync_log
    except ImportError:  # pragma: no cover - the sync core needs torch
        return
    ensure_visible_logs(logger, sync_log)


def _model_chunks(trainer) -> list:
    # Pass the DDP-wrapped chunks: the sync core needs the grad/param buffers
    # and the DistributedOptimizer's mapping, both keyed on the wrapped models.
    chunks = getattr(trainer, "wrapped_models", None)
    if chunks is None:
        chunks = getattr(trainer, "unwrapped_models", None)
    if chunks is None:
        raise RuntimeError("ms-swift trainer exposes neither wrapped_models nor unwrapped_models")
    return list(chunks)


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

try:  # ms-swift >= 4.x Megatron callback base class
    from swift.megatron.callbacks import MegatronCallback as _CallbackBase
except Exception:  # pragma: no cover - lets unit tests run without ms-swift

    class _CallbackBase:  # type: ignore[no-redef]
        def __init__(self, trainer):
            self.trainer = trainer
            self.args = trainer.args
            self.state = trainer.state


class YetoDiLoCoCallback(_CallbackBase):
    """Drives ``MegatronDiLoCoSync`` from ms-swift's Megatron training loop."""

    # Overridable in tests.
    sync_factory: Callable[..., Any] = staticmethod(default_sync_factory)
    token_reducer: Callable[[int], int] = staticmethod(megatron_global_tokens)

    def __init__(self, trainer):
        super().__init__(trainer)
        self.enabled = not _env_flag("DISABLE")
        self.sync = None
        self._closed = False
        self._pending = None  # (iteration_before_step, local_tokens, update_successful)
        if not self.enabled:
            logger.warning("yeto DiLoCo callback disabled via %sDISABLE", ENV_PREFIX)
            return
        self.config = DiLoCoHookConfig.from_env()
        _visible_logs()
        restore_lr_schedule(trainer)
        self._wrap_train_step()
        # Build here (not in on_train_begin) so a bad config fails before data
        # loading; the trainer has already loaded the resume checkpoint, which
        # is what learner 0's INIT_PARAMS carries.
        start = int(self.state.iteration)
        resume_dir = resume_dir_for(self.config, self.args) if start > 0 else None
        self.sync = type(self).sync_factory(_model_chunks(trainer), trainer.optimizer, self.config,
                                            resume_dir=resume_dir, resume_iteration=start)
        atexit.register(self.close)
        logger.info("yeto DiLoCo sync attached: learner %d/%d, interval %d, fragments %d, start iteration %d",
                    self.config.learner_id, self.config.num_learners, self.config.sync_interval_steps,
                    self.config.fragments, int(self.state.iteration))

    def _wrap_train_step(self) -> None:
        """Capture train_step's metrics and update flag on this trainer instance.

        ``on_step_end`` receives no arguments, so the per-step result is
        stashed here. Only the instance attribute is replaced; the class and
        other trainers are untouched.
        """
        trainer = self.trainer
        original = trainer.train_step
        if getattr(original, "_yeto_diloco_wrapped", False):
            return

        def train_step(*args, **kwargs):
            iteration_before = int(self.state.iteration)
            metrics, grad_norm, update_successful = original(*args, **kwargs)
            self._pending = (iteration_before, local_loss_tokens(metrics), bool(update_successful))
            return metrics, grad_norm, update_successful

        train_step._yeto_diloco_wrapped = True  # type: ignore[attr-defined]
        trainer.train_step = train_step

    def on_step_end(self):
        if self.sync is None:
            return
        pending, self._pending = self._pending, None
        if pending is None:
            raise RuntimeError("yeto DiLoCo: on_step_end without a captured train_step result")
        iteration_before, local_tokens, update_successful = pending
        iteration = int(self.state.iteration)
        if iteration != iteration_before + 1:
            raise RuntimeError(f"yeto DiLoCo: unexpected iteration {iteration} after step from {iteration_before}")
        # The reduction is a collective, so it runs on every rank even when the
        # step was skipped; the skipped step then contributes zero tokens.
        global_tokens = int(type(self).token_reducer(local_tokens))
        target_tokens = global_tokens if update_successful else 0
        stats = getattr(self.sync, "stats", None) or {}
        before = {k: stats.get(k, 0) for k in ("applied", "pushed", "apply_s", "export_s")}
        started = time.perf_counter()
        self.sync.after_optimizer_step(iteration, target_tokens,
                                       update_successful=update_successful)
        self._log_step(iteration, target_tokens, time.perf_counter() - started, before, stats)
        if getattr(self.sync, "finished", False) and iteration < int(self.args.train_iters):
            # The final cut is installed: stop here (ms-swift's default_flow
            # saves at train_iters) instead of training past the global result.
            logger.warning("yeto DiLoCo: final cut installed at iteration %d; ending training (train_iters %d)",
                           iteration, int(self.args.train_iters))
            self.args.train_iters = iteration

    def _log_step(self, iteration: int, target_tokens: int, sync_s: float,
                  before: dict, stats: dict) -> None:
        """Append one JSON line per step (sync rank 0 only) to ``<prefix>STEP_LOG``.

        ``sync_s`` is the wall time the training loop spent inside the sync
        boundary; ``export_s`` / ``apply_s`` split out the fp32 master gather
        for a push and the write-back of a received broadcast."""
        path = os.environ.get(ENV_PREFIX + "STEP_LOG")
        if not path or getattr(self.sync, "rank", 0) != 0:
            return
        record = {"iteration": iteration, "time": time.time(), "target_tokens": target_tokens,
                  "sync_s": round(sync_s, 4),
                  "export_s": round(stats.get("export_s", 0) - before["export_s"], 4),
                  "apply_s": round(stats.get("apply_s", 0) - before["apply_s"], 4),
                  "pushes_sent": stats.get("pushed", 0) - before["pushed"],
                  "applies": stats.get("applied", 0) - before["applied"]}
        if _env_flag("HOST_MEM_CENSUS"):
            record.update(host_memory_census())
        try:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        except OSError as error:  # never let bookkeeping kill a training step
            logger.warning("yeto DiLoCo: cannot write step log %s: %s", path, error)

    def on_save(self, *args, **kwargs):
        """Record the sync state next to the checkpoint just written (every rank)."""
        if self.sync is None or self._closed or getattr(self.sync, "finished", False):
            return
        iteration = int(self.state.iteration)
        output_dir = save_dir_for(self.args, iteration, *args, **kwargs)
        if output_dir is None:
            logger.warning("yeto DiLoCo: on_save without a checkpoint directory; sync state not recorded")
            return
        self.sync.save_state(output_dir, iteration=iteration, save_anchors=self.config.save_anchors)

    def on_train_end(self):
        self.close()

    def close(self):
        if self._closed or self.sync is None:
            return
        self._closed = True
        try:
            self.sync.close()
        finally:
            try:
                atexit.unregister(self.close)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register(auto_attach: Optional[bool] = None) -> bool:
    """Register the callback with ms-swift. Returns False if ms-swift is absent."""
    try:
        from swift.megatron.callbacks import megatron_callbacks_map
    except Exception:
        return False
    megatron_callbacks_map[CALLBACK_NAME] = YetoDiLoCoCallback
    if auto_attach is None:
        auto_attach = _env_flag("AUTO_ATTACH")
    if auto_attach:
        _patch_trainer_init()
    return True


def _patch_trainer_init() -> None:
    from swift.megatron.trainers.base import BaseMegatronTrainer

    original = BaseMegatronTrainer.__init__
    if getattr(original, "_yeto_diloco_wrapped", False):
        return

    def __init__(self, args, *a, **kw):
        callbacks = list(getattr(args, "callbacks", None) or [])
        if CALLBACK_NAME not in callbacks:
            # Run before ms-swift's own callbacks so a save at this step sees
            # the merged base.
            args.callbacks = [CALLBACK_NAME] + callbacks
        original(self, args, *a, **kw)

    __init__._yeto_diloco_wrapped = True  # type: ignore[attr-defined]
    BaseMegatronTrainer.__init__ = __init__


register()
