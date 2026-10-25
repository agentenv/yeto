"""verl trainer mode ``yeto_separate`` (S27 G-DC; runs inside verl's task runner).

verl's v1 ``separate_async`` trainer with the yeto driver in charge of the
round loop, for the Megatron island (:mod:`.megatron_engine`):

* ``hybrid_engine=False``: the rollout runs only on the standalone vLLM
  servers (their own cards); the trainer cards never host a rollout replica;
* the driver publishes (``MegatronPublisher`` -> the standalone NCCL checkpoint
  engine) and only then submits the next round's prompts, so verl's own
  hooks that publish or pre-submit are switched off here: ``on_init_end``
  (first publication = the driver's policy version 0), ``on_train_begin``
  (warm-up batches) and ``on_step_end`` (publication after every step);
* ``parameter_sync_step=1``: the trainer's current weights are pi_old for the
  single mini-batch of a round, so the old log-probs are computed directly.
  verl's version-aware path would copy the whole model to host memory every
  round (``save_model_to_cpu``) only to restore the same weights.

The yeto mismatch criteria are attached right after the old log-probs, as in
``trainer.YetoSyncTrainer`` (design D7).
"""

from __future__ import annotations

import time

from verl.trainer.ppo.v1.trainer_base import PPOTrainer, register_trainer
from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync

from . import advantage  # noqa: F401 - registers the yeto_grpo estimator in the trainer process
from .separate import TRAINER_MODE
from .ray_ports import spread_master_ports


@register_trainer(TRAINER_MODE)
class YetoSeparateTrainer(PPOTrainerSeparateAsync):
    yeto_criteria_sink = None

    def __init__(self, config):
        sync_step = int(config.trainer.v1.separate_async.parameter_sync_step)
        if sync_step != 1:
            raise ValueError(f"{TRAINER_MODE} needs parameter_sync_step=1 (one optimizer step per "
                             f"published policy), got {sync_step}")
        if bool(config.actor_rollout_ref.get("hybrid_engine", True)):
            raise ValueError(f"{TRAINER_MODE} needs actor_rollout_ref.hybrid_engine=False")
        spread_master_ports()  # the replicas' worker groups start at once (ray_ports)
        super().__init__(config)

    def on_init_end(self):  # the first publication is the driver's (policy version 0)
        return None

    def on_train_begin(self):  # no warm-up batch: the driver submits each round after its publication
        return None

    def on_step_end(self):  # publication happens after the outer boundary (driver)
        return None

    def prepare_step(self) -> dict:
        # verl's fit loop calls on_step_begin before each step (it resets the step's
        # sample-wait counters that on_sample_end adds to); the yeto rollout port
        # starts a round with prepare_step, so the step begins here
        self.on_step_begin()
        return super().prepare_step()

    def _timed(self, name: str, fn, *args):
        """verl's own loop wraps each stage in ``marked_timer``; the yeto ports call the
        stages directly, so the two the efficiency record needs are timed here
        (``timing_s/old_log_prob`` and ``timing_s/update_actor`` in the step metrics)."""
        started = time.perf_counter()
        try:
            return fn(*args)
        finally:
            timing = getattr(self, "timing_raw", None)
            if isinstance(timing, dict):
                timing[name] = timing.get(name, 0.0) + (time.perf_counter() - started)

    def _update_actor(self, batch, metrics):
        return self._timed("update_actor", super()._update_actor, batch, metrics)

    def _compute_old_log_prob(self, batch, metrics):
        batch = self._timed("old_log_prob", PPOTrainer._compute_old_log_prob, self, batch, metrics)
        sink = self.yeto_criteria_sink
        if sink is not None:
            from .ports_impl import criteria_from_trainer_batch

            criteria_from_trainer_batch(sink, batch)
            for key in ("abs_diff", "k3", "tis_clipfrac", "signed_mean"):
                value = sink.last_criteria.get(key)
                if isinstance(value, float):
                    metrics[f"yeto_mismatch/{key}"] = value
        return batch
