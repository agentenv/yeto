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
"""

from __future__ import annotations

import atexit
import logging
import os
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
        )
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


def default_sync_factory(model_chunks, optimizer, cfg: DiLoCoHookConfig):
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
    )


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
        self._wrap_train_step()
        # Build here (not in on_train_begin) so a bad config fails before data
        # loading; the trainer has already loaded the resume checkpoint, which
        # is what learner 0's INIT_PARAMS carries.
        self.sync = type(self).sync_factory(_model_chunks(trainer), trainer.optimizer, self.config)
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
        self.sync.after_optimizer_step(iteration, target_tokens,
                                       update_successful=update_successful)
        if getattr(self.sync, "finished", False) and iteration < int(self.args.train_iters):
            # The final cut is installed: stop here (ms-swift's default_flow
            # saves at train_iters) instead of training past the global result.
            logger.warning("yeto DiLoCo: final cut installed at iteration %d; ending training (train_iters %d)",
                           iteration, int(self.args.train_iters))
            self.args.train_iters = iteration

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
