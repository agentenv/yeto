"""verl buffer trainer island, image side (S25 C2; runs inside the verl image).

The trainer island of a sampler/trainer split (``--rl-sample-role trainer``):
verl's ``FullyAsyncTrainer`` trains, yeto's driver runs the rounds over the
fully_async ports, and the queued samples come from the durable sample buffer
instead of verl's rollouter (:mod:`.buffer_feed`, :mod:`.buffer_ports`).

* :class:`YetoBufferTaskRunner` builds only the trainer (tokenizer, worker
  mapping, the trainer actor and its workers): no rollouter, no MessageQueue,
  no checkpoint engine, no initial weight push. Each publication is a store
  release instead (``BufferReleasePublisher``).
* :data:`YetoBufferTrainer` is the fully_async trainer actor
  (``_YetoTrainerMixin``: assemble, train step with the mismatch criteria and
  the per-source off-policy metrics, LoRA export) plus ``yeto_add_group`` (one
  kept group's rows as a verl ``RolloutSample``) and ``yeto_param_version``.
* A full-parameter trainer (S27 G-EW, VERL-PARITY #14): the same actor; its
  sync and policy state are the full-parameter island's
  (:mod:`.fully_async_full`) and each release is its own weights in the model's
  dtype (``buffer_ports.release_weights_for``). The samplers load full releases as before.

verl itself is not changed. Only exercised on a GPU; the logic it relies on is
in the CPU-tested :mod:`.buffer_feed` and :mod:`.buffer_ports`.
"""

from __future__ import annotations

import os

import numpy as np
import ray

from .fully_async_runner import YetoFullyAsyncTaskRunner, _TrainerBase, _YetoTrainerMixin

FLOAT_ROWS = ("rollout_log_probs", "rm_scores")
INT32_FIELDS = ("__num_turns__",)
SAMPLE_BUFFER_PLAN_KEY = "sample_buffer"  # plan section of a buffer trainer (island_entry)


def _object_array(values) -> np.ndarray:
    out = np.empty(len(values), dtype=object)
    out[:] = list(values)
    return out


def rows_to_dataproto(rows: dict):
    """The plain rows of :func:`.buffer_feed.group_rows` as a verl ``DataProto``."""
    import torch
    from tensordict import TensorDict
    from verl import DataProto

    tensors = rows["tensors"]
    n = len(tensors["prompts"])
    batch = TensorDict({name: torch.tensor(values, dtype=torch.float32 if name in FLOAT_ROWS else torch.long)
                        for name, values in tensors.items()}, batch_size=n)
    non_tensor = {name: (np.asarray(values, dtype=np.int32) if name in INT32_FIELDS else _object_array(values))
                  for name, values in rows["non_tensor"].items()}
    meta = {key: (list(value) if isinstance(value, list) else value) for key, value in rows["meta_info"].items()}
    return DataProto(batch=batch, non_tensor_batch=non_tensor, meta_info=meta)


class _YetoBufferTrainerMixin(_YetoTrainerMixin):
    def yeto_add_group(self, rows: dict) -> int:
        from verl.experimental.fully_async_policy.detach_utils import RolloutSample

        if getattr(self, "_yeto_kept", None) is None:
            self._yeto_kept = []
        proto = rows_to_dataproto(rows)
        self._yeto_kept.append(RolloutSample(full_batch=proto, sample_id=str(rows["non_tensor"]["uid"][0]),
                                             epoch=0, rollout_status={}))
        return len(self._yeto_kept)

    def yeto_param_version(self) -> int:
        return int(self.current_param_version)

    async def _yeto_queue_size(self):
        return None  # no MessageQueue: the buffer backlog is the samplers' report


YetoBufferTrainer = ray.remote(num_cpus=10)(
    type("YetoBufferTrainer", (_YetoBufferTrainerMixin, _TrainerBase), {}))


class YetoBufferTaskRunner(YetoFullyAsyncTaskRunner):
    """The trainer-only set-up, then the yeto driver over the buffer ports."""

    trainer_actor = YetoBufferTrainer

    def _initialize_trainer(self, config, total_train_steps: int) -> None:
        """verl's ``_initialize_components`` (fork acad9875, fully_async_main.py:51)
        up to the trainer, without the rollouter and the MessageQueue."""
        from omegaconf import OmegaConf
        from verl.experimental.separation.utils import create_role_worker_mapping
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.fs import copy_to_local

        OmegaConf.resolve(config)
        local_path = copy_to_local(config.actor_rollout_ref.model.path,
                                   use_shm=config.actor_rollout_ref.model.get("use_shm", False))
        trust_remote_code = config.data.get("trust_remote_code", False)
        self.components["tokenizer"] = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.components["processor"] = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        self.components["config"] = config
        mapping, group_cls = create_role_worker_mapping(config)
        self.components["role_worker_mapping"] = mapping
        self.components["ray_worker_group_cls"] = group_cls
        self._create_trainer(config)
        trainer = self.components["trainer"]
        ray.get(trainer.set_total_train_steps.remote(int(total_train_steps)))
        ray.get(trainer.load_checkpoint.remote())

    def run(self, config, plan: dict):
        from omegaconf import OmegaConf

        from yeto.island_credential_guard import check_island_credentials
        from yeto.rl.engine.policy_release import PolicyReleaseStore
        from yeto.rl.engine.resume import store_for
        from yeto.rl.engine.merged_release import BufferLane, lane_cursor_log
        from yeto.rl.engine.sample_buffer import open_trainer_buffer

        from .buffer_feed import RowShape
        from .buffer_ports import install_buffer_ports, release_weights_for
        from .trainer import run_fully_async_island

        check_island_credentials()  # secret-handling-hardening D4 (buffer trainer Ray actor)
        section = plan[SAMPLE_BUFFER_PLAN_KEY]
        self._initialize_trainer(config, int(plan["global_rounds"]))
        trainer = self.components["trainer"]
        tokenizer = self.components["tokenizer"]

        def call(name, *args, timeout=None):
            return ray.get(getattr(trainer, name).remote(*args), timeout=timeout)

        environ = dict(os.environ)
        store = store_for(section["store"], environ=environ)
        # S25 C3: a synced trainer reads its lane of the buffer (section "lane": "M/N")
        buffer = open_trainer_buffer(store, batch_size=1, create_index=False,
                                     lane=BufferLane.parse(section.get("lane")),
                                     shards=section.get("shards"))
        releases = PolicyReleaseStore(store_for(section.get("release_store") or section["store"],
                                                environ=environ))
        shape = RowShape(int(config.data.max_prompt_length), int(config.data.max_response_length),
                         int(tokenizer.pad_token_id))

        def ports(island):
            return install_buffer_ports(island, buffer=buffer, releases=releases, shape=shape,
                                        base_dir=plan["model_path"], tokenize=tokenizer.encode,
                                        wait_s=float(section.get("wait_s") or 3600.0),
                                        poll_s=float(section.get("poll_s") or 2.0),
                                        cursor_log=lane_cursor_log(buffer),
                                        weights=release_weights_for(island, plan))

        return run_fully_async_island(call, None, plan, OmegaConf.to_container(config, resolve=True),
                                      buffer_ports=ports)
