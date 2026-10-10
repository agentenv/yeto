"""verl fully_async island, image side (agentic-rollout-utilization 6.4b).

Runs inside the verl image (imports verl, Ray).  Design item 1 of 6.4b:
verl's two ``fit()`` loops are not run as a pair.  ``FullyAsyncRollouter``
stays the resident generator (MessageQueue, sample-count throttle,
partial-rollout continuation unchanged); the trainer actor is
:class:`YetoFullyAsyncTrainer`, whose ``yeto_*`` methods are the round pieces
the yeto ``IslandDriver`` calls through the ports of :mod:`.fully_async_ports`:

* ``yeto_pull_sample`` / ``yeto_keep_last`` / ``yeto_drop_last`` /
  ``yeto_assemble`` -- the queue draining of ``_get_samples_from_queue`` one
  sample at a time, so yeto can judge each one (6.4a (c));
* ``yeto_train_step`` -- ``fit_step`` from reward to actor update, without
  ``_fit_update_weights`` (publication is yeto's);
* ``yeto_push_weights`` -- ``checkpoint_manager.update_weights(global_steps=
  current_param_version)`` + the rollouter's ``reset_staleness``;
* ``yeto_export_lora`` / ``yeto_call_workers`` -- the LoRA hooks of
  ``ports_impl.VerlPolicyState`` on the trainer's actor worker group (6.4b item 2).

Only exercised on a GPU (rl-verl-backend 6.6); the logic it relies on is in the
CPU-tested :mod:`.fully_async_round` and :mod:`.fully_async_ports`.
"""

from __future__ import annotations

import asyncio
import time

import ray
from verl.experimental.fully_async_policy import fully_async_main as fa_main
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer

from .fully_async_round import RESUME_CALLS_KEY

_TrainerBase = getattr(FullyAsyncTrainer, "__ray_actor_class__", FullyAsyncTrainer)


def _ints(values) -> list:
    return [None if v is None else int(v) for v in values]


class _YetoTrainerMixin:
    _yeto_last = None
    _yeto_kept: list
    _yeto_batch = None

    async def yeto_pull_sample(self):
        sample, _queue_len = await self.message_queue_client.get_sample()
        if sample is None:
            return None
        rs = ray.cloudpickle.loads(sample)
        self._yeto_last = rs
        batch = rs.full_batch
        nt = batch.non_tensor_batch
        if "response_mask" in batch.batch.keys():
            mask = batch.batch["response_mask"]
        else:
            from verl.trainer.ppo.ray_trainer import compute_response_mask

            mask = compute_response_mask(batch)
        lengths = [int(x) for x in mask.sum(-1).tolist()]
        rewards = (batch.batch["rm_scores"].float().sum(-1).tolist()
                   if "rm_scores" in batch.batch.keys() else None)
        calls = nt.get(RESUME_CALLS_KEY)
        return {
            "uid": str(nt["uid"][0]),
            "min_steps": tuple(_ints(nt.get("min_global_steps", [None] * len(lengths)))),
            "max_steps": tuple(_ints(nt.get("max_global_steps", [None] * len(lengths)))),
            "calls": tuple(None if c is None else tuple((int(v), int(n)) for v, n in c)
                           for c in (calls if calls is not None else [None] * len(lengths))),
            "lengths": tuple(lengths),
            "rewards": tuple(float("nan") for _ in lengths) if rewards is None else tuple(rewards),
        }

    def yeto_keep_last(self):
        if not hasattr(self, "_yeto_kept") or self._yeto_kept is None:
            self._yeto_kept = []
        self._yeto_kept.append(self._yeto_last)
        self._yeto_last = None

    def yeto_drop_last(self):
        self._yeto_last = None

    async def yeto_assemble(self):
        from verl.experimental.fully_async_policy.detach_utils import assemble_batch_from_rollout_samples

        kept, self._yeto_kept = list(getattr(self, "_yeto_kept", None) or []), []
        balance = self._balance_batch if self.config.trainer.balance_batch else None
        batch = assemble_batch_from_rollout_samples(kept, self.tokenizer, self.config, balance)
        self.metrics = {"training/global_step": self.global_steps, "training/epoch": self.epoch}
        self.timing_raw = {}
        self.future_reward = None
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}
        self._collect_metrics_from_samples(batch, self.metrics)
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        self._yeto_batch = batch
        return {"samples": len(kept), "queue_size": int(await self.message_queue_client.get_queue_size())}

    def _yeto_criteria(self, batch) -> dict:
        from yeto.rl.engine import mismatch_criteria

        try:
            train_lp = batch.batch["old_log_probs"]
            infer_lp = batch.batch["rollout_log_probs"]
            mask = batch.batch["response_mask"]
            width = min(train_lp.shape[-1], infer_lp.shape[-1], mask.shape[-1])
            crit = mismatch_criteria.compute(train_lp[..., :width], infer_lp[..., :width], mask[..., :width],
                                             tis_upper=float(getattr(self, "yeto_tis_upper", 2.0)))
            key = tuple(getattr(self, "yeto_thresholds_key", ()))
            try:
                crit["alarms"] = mismatch_criteria.judge(crit, mismatch_criteria.thresholds_for(key))
                crit["thresholds_key"] = list(key)
            except mismatch_criteria.Uncalibrated as exc:
                crit["alarms"] = None
                crit["uncalibrated"] = str(exc)
        except Exception as exc:  # noqa: BLE001 - reported as a failed criterion
            crit = {"error": f"{type(exc).__name__}: {exc}", "alarms": ["criteria_unavailable"]}
        crit["param_version"] = int(self.current_param_version)
        return crit

    def yeto_configure(self, tis_upper: float, thresholds_key) -> None:
        self.yeto_tis_upper = float(tis_upper)
        self.yeto_thresholds_key = tuple(thresholds_key)

    def yeto_train_step(self):
        batch, self._yeto_batch = self._yeto_batch, None
        if batch is None:
            raise RuntimeError("yeto_train_step without an assembled batch")
        t0 = time.time()
        batch = self._fit_compute_reward(batch)
        batch = self._fit_compute_log_prob(batch)
        criteria = self._yeto_criteria(batch)  # 6.4b item 6: re-hooked after old log-probs
        batch = self._fit_compute_ref_log_prob(batch)
        batch = self._fit_compute_critic(batch)
        batch = self._fit_compute_advantage(batch)
        batch = self._fit_update_critic(batch)
        batch = self._fit_update_actor(batch)
        self._fit_update_local_step()  # trigger_parameter_sync_step=1: param version + 1
        self.timing_raw["step"] = time.time() - t0
        try:
            self._fit_collect_metrics(batch)
        except Exception as exc:  # noqa: BLE001 - metrics are archival
            self.metrics["yeto/collect_metrics_error"] = repr(exc)[:300]
        self.global_steps += 1
        numeric = {k: float(v) for k, v in self.metrics.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)}
        return {"metrics": numeric, "criteria": criteria, "param_version": int(self.current_param_version)}

    async def yeto_push_weights(self):
        t0 = time.time()
        await self.checkpoint_manager.update_weights(global_steps=self.current_param_version)
        update_s = time.time() - t0
        timing = await asyncio.wrap_future(self.rollouter.reset_staleness.remote().future())
        return {"param_version": int(self.current_param_version),
                "timing": {"update_weights_s": update_s, **{k: float(v) for k, v in timing.items()}}}

    async def yeto_call_workers(self, fn, *args):
        workers = self.actor_rollout_wg.workers
        return list(await asyncio.gather(*[w.__ray_call__.remote(fn, *args) for w in workers]))

    async def yeto_export_lora(self):
        from .ports_impl import remote_export_lora

        return (await self.yeto_call_workers(remote_export_lora))[0]


YetoFullyAsyncTrainer = ray.remote(num_cpus=10)(type("YetoFullyAsyncTrainer", (_YetoTrainerMixin, _TrainerBase), {}))


# verl decorates its task runner with @ray.remote too; Ray refuses subclasses of
# actor classes (found by the S19 Modal dry run), so subclass the plain class.
_TaskRunnerBase = getattr(fa_main.FullyAsyncTaskRunner, "__ray_actor_class__", fa_main.FullyAsyncTaskRunner)


class YetoFullyAsyncTaskRunner(_TaskRunnerBase):
    """verl's component set-up (trainer, rollouter, MessageQueue, initial weight
    sync); then the yeto driver instead of ``_run_training_loop``."""

    def run(self, config, plan: dict):
        from yeto.island_credential_guard import check_island_credentials

        check_island_credentials()  # secret-handling-hardening D4 (fully_async Ray actor)
        self._initialize_components(config)
        from omegaconf import OmegaConf

        from .trainer import run_fully_async_island

        trainer = self.components["trainer"]
        rollouter = self.components["rollouter"]
        state = {"future": None}

        def start_rollouter():
            state["future"] = rollouter.fit.remote()

        def call(name, *args, timeout=None):
            # timeout (s): S19 async5 hung 33 min in a push whose vLLM side had died;
            # a call that does not return in time raises TimeoutError.
            ref = getattr(trainer, name).remote(*args)
            deadline = None if timeout is None else time.monotonic() + float(timeout)
            while True:
                waits = [ref] if state["future"] is None else [ref, state["future"]]
                left = None if deadline is None else max(0.0, deadline - time.monotonic())
                ready, _ = ray.wait(waits, num_returns=1, timeout=left)
                if ref in ready:
                    return ray.get(ref)
                if not ready:
                    raise TimeoutError(f"trainer.{name} did not return within {timeout} s")
                ray.get(state["future"])  # raises the rollouter's error; a clean end puts None in the queue
                state["future"] = None

        try:
            return run_fully_async_island(call, start_rollouter, plan,
                                          OmegaConf.to_container(config, resolve=True))
        finally:
            if state["future"] is not None:
                ray.cancel(state["future"])
            asyncio.run(self.components["message_queue_client"].clear_queue())

    def _create_trainer(self, config) -> None:
        """verl's ``_create_trainer`` (fork acad9875, fully_async_main.py:138), building
        :data:`YetoFullyAsyncTrainer`.  Setting ``fa_main.FullyAsyncTrainer`` did not
        reach verl's method inside the Ray actor (s19-verl64b-async7-20261010a: the
        trainer was verl's own class, "no attribute yeto_configure"), so the method is
        overridden here instead of relying on that module global."""
        from verl.experimental.separation.utils import create_resource_pool_manager
        from verl.trainer.ppo.utils import Role

        print("[ASYNC MAIN] Starting create trainer (yeto)...")
        mapping = {role: cls for role, cls in self.components["role_worker_mapping"].items()
                   if role != Role.Rollout}
        trainer = YetoFullyAsyncTrainer.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=mapping,
            resource_pool_manager=create_resource_pool_manager(config, roles=list(mapping.keys())),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            device_name=config.trainer.device,
        )
        trainer.yeto_configure  # noqa: B018 - fail fast (AttributeError) if this is not our class
        ray.get(trainer.init_workers.remote())
        self.components["trainer"] = trainer
        print("[ASYNC MAIN] YetoFullyAsyncTrainer created and initialized successfully")
