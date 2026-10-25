"""The five engine ports over verl's v1 sync trainer (rl-verl-backend 2.1/2.2, design D3).

Runs inside verl's task-runner Ray actor (``verl_main``), next to the verl
trainer object; imports verl lazily.  yeto's neutral ``IslandDriver`` drives
the rounds and the outer sync (none / strict-avg / elastic) exactly as on the
Miles path:

* ``VerlRolloutPool.generate``  -- verl steps 1-3 of ``_step_once``: submit the
  prompts, sample the replay buffer (vLLM generates), rule reward, balance;
  the KVBatchMeta stays in verl (``payload``), yeto reads metadata only.
* ``VerlTrainerGroup.train_step`` -- steps 4-9: old log-prob recompute (+ the
  yeto mismatch criteria in the Miles convention, design D7), GRPO advantage,
  actor update; returns the receipt; ``step_metrics`` gives grad_norm/lr.
* ``VerlPolicyState`` -- export/apply the LoRA tensors of the FSDP2 PEFT model
  by canonical name (``param_names``), via ``__ray_call__`` on the actor
  workers; ``reset`` zeroes the Adam moments.
* ``VerlPublisher`` -- verl's colocated in-memory update (``update_weights``)
  plus the read-back checksum of what vLLM registered (``vllm_readback``).
* ``VerlPlacement`` -- colocated, one GPU.

Policy token of the groups: the published token.  This holds by construction
(sync colocated: the replicas sleep after sampling and only wake through the
next verified publication), it is not a per-sample engine report as on Miles
-- recorded in ``capabilities.extra``.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import prod
from pathlib import Path
from typing import Any

from yeto.rl.contracts import InferencePublicationManifest, LocalStepReceipt
from yeto.rl.engine import mismatch_criteria, offpolicy_source
from yeto.rl.engine.policy_digest import PolicyDigest, TrainerResidentState, is_resident
from yeto.rl.engine.ports import GroupMetadata, PlacementDescription, PublicationResult, RolloutBatchHandle
from yeto.rl.engine.trainable_state import LAYOUT_LORA, TrainableState

from . import publish as pub
from .cut import CutPorts
from .param_names import train_to_canonical

MEMBER = "vllm-0"


# ----------------------------------------------------------------------------- worker-side functions
def _inner_worker(worker_dict):
    inner = getattr(worker_dict, "worker_dict", None)
    if inner:
        for value in inner.values():
            if hasattr(value, "actor"):
                return value
    return worker_dict


def _peft_module(worker):
    module = worker.actor.engine.module
    return getattr(module, "_fsdp_wrapped_module", module)


def _lora_tensors(worker_dict) -> tuple[dict, set]:
    """({canonical name: full LoRA tensor on its device}, parameter dtypes); collective
    (``full_tensor``): every rank calls it."""
    peft = _peft_module(_inner_worker(worker_dict))
    out, dtypes = {}, set()
    for name, param in peft.named_parameters():
        if ".lora_" not in name:
            continue
        dtypes.add(str(param.dtype))
        full = param.full_tensor() if hasattr(param, "full_tensor") else param
        out[train_to_canonical(name)] = full.detach()
    return out, dtypes


def remote_export_lora(worker_dict):
    """(canonical f32 CPU tensors, info) from the actor's PEFT model."""
    import torch

    with torch.no_grad():
        tensors, dtypes = _lora_tensors(worker_dict)
        out = {name: value.to("cpu", torch.float32).clone() for name, value in tensors.items()}
    return out, {"param_dtypes": sorted(dtypes), "count": len(out)}


def digest_lora_tensors(tensors, *, base_model_revision: str, lora_config_hash: str,
                        layout_hash: str, expected_specs) -> dict:
    """rl-publish-fastpath, trainer side: validate ``tensors`` exactly as the driver's
    ``canonical_state`` would and return only what the driver reads -- the policy
    digests and the bf16 send checksum of :func:`publish.lora_checksum` (small dicts,
    no tensors).  Same functions as the full-export path, so the values are equal
    bit for bit."""

    import time

    from concurrent.futures import ThreadPoolExecutor

    from yeto.rl.core import canonical_state_from_owned_tensors
    from yeto.rl.engine.policy_digest import digest_canonical_tensors

    started = time.monotonic()
    # E1: ``tensors`` are this call's own copies (the trainer's export), so the
    # validation keeps them without a second clone; the values are the same
    state = canonical_state_from_owned_tensors(
        0, tensors, base_model_revision=base_model_revision, lora_config_hash=lora_config_hash,
        layout_hash=layout_hash, expected_specs=expected_specs)
    # E1: the bf16 send checksum runs in its own thread beside the two digest streams
    # (hashlib releases the GIL); each function is unchanged, so the values are too
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="yeto-lora-checksum") as pool:
        checksum = pool.submit(pub.lora_checksum, dict(state.tensors))
        digest = digest_canonical_tensors(state.tensors, base_model_revision=state.base_model_revision,
                                          lora_config_hash=state.lora_config_hash,
                                          layout_hash=state.layout_hash)
        lora_checksum = checksum.result()
    return {"digest": digest.to_wire(), "layout_hash": state.layout_hash,
            "lora_checksum": lora_checksum,
            "export_seconds": round(time.monotonic() - started, 3)}


def remote_export_digest(worker_dict, kwargs: dict):
    """All ranks gather the LoRA (``full_tensor`` is collective); rank 0 hashes it.

    E1: rank 0 stages the tensors in one reusable pinned buffer (one GPU
    synchronization, :func:`~yeto.rl.engine.policy_digest.stage_f32_on_host`); they
    are hashed here and dropped, never returned."""
    import torch
    import torch.distributed as dist

    from yeto.rl.engine.policy_digest import HostStaging, stage_f32_on_host

    with torch.no_grad():
        tensors, _dtypes = _lora_tensors(worker_dict)
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return None
        worker = _inner_worker(worker_dict)
        staging = getattr(worker, "_yeto_digest_staging", None)
        if staging is None:
            staging = HostStaging()
            setattr(worker, "_yeto_digest_staging", staging)
        tensors = stage_f32_on_host(tensors, staging)
    return digest_lora_tensors(tensors, **kwargs)


def remote_apply_lora(worker_dict, tensors, reset_optimizer: bool):
    import torch

    worker = _inner_worker(worker_dict)
    peft = _peft_module(worker)
    seen = set()
    with torch.no_grad():
        for name, param in peft.named_parameters():
            if ".lora_" not in name:
                continue
            canon = train_to_canonical(name)
            src = tensors[canon]
            if hasattr(param, "device_mesh"):
                from torch.distributed.tensor import distribute_tensor

                param.copy_(distribute_tensor(src.to(param.device, param.dtype), param.device_mesh,
                                              param.placements))
            else:
                param.copy_(src.to(param.device, param.dtype))
            seen.add(canon)
    missing = sorted(set(tensors) - seen)
    if missing:
        raise KeyError(f"canonical tensors not in the verl PEFT model: {missing[:4]} (+{len(missing)})")
    zeroed = 0
    if reset_optimizer:
        optimizer = getattr(worker.actor.engine, "optimizer", None)
        for state in (optimizer.state.values() if optimizer is not None else ()):
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state and hasattr(state[key], "zero_"):
                    state[key].zero_()
                    zeroed += 1
    return {"applied": len(seen), "moments_zeroed": zeroed}


# ----------------------------------------------------------------------------- helpers
def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _metric(metrics: dict, *suffixes: str):
    for suffix in suffixes:
        for key, value in metrics.items():
            if key == suffix or key.endswith("/" + suffix):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
    return None


class VerlIsland:
    """Shared state of the five ports for one verl island (one trainer)."""

    def __init__(self, trainer, *, learner_id: int, out_dir: str, base_model_revision: str,
                 lora_config_hash: str, layout_hash: str, expected_specs, tis_upper: float,
                 thresholds_key, emit=None):
        self.trainer = trainer
        self.learner_id = int(learner_id)
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.base_model_revision = base_model_revision
        self.lora_config_hash = lora_config_hash
        self.layout_hash = layout_hash
        self.expected_specs = tuple(expected_specs)
        self.tis_upper = float(tis_upper)
        self.thresholds_key = tuple(thresholds_key)
        self.emit = emit or (lambda event, **f: None)
        self.published: tuple[int, str] | None = None
        self.last_metrics: dict = {}
        self.last_criteria: dict = {}
        # A full-parameter island's trainer group as its role stream sees it
        # (``full_parameter_stream.VerlFullParameterGroup``); None on a LoRA island.
        self.full_group = None
        self.readback_dir = Path(os.environ.get("YETO_VERL_READBACK_DIR", str(self.out / "readback")))
        self.readback_dir.mkdir(parents=True, exist_ok=True)
        (self.readback_dir / "expected_shapes.json").write_text(
            json.dumps({s.name: list(s.shape) for s in self.expected_specs}))

    # worker calls
    def call_workers(self, fn, *args):
        import ray

        workers = self.trainer.actor_rollout_wg.workers
        return ray.get([w.__ray_call__.remote(fn, *args) for w in workers])

    def call_workers_shared(self, fn, *args):
        """:meth:`call_workers` with every argument put in the object store once,
        so a large argument (whole parameter tensors) is not serialized per worker."""
        import ray

        return self.call_workers(fn, *[ray.put(arg) for arg in args])

    def export_tensors(self) -> tuple[dict, dict]:
        results = self.call_workers(remote_export_lora)
        return results[0]

    def export_digest(self, kwargs: dict) -> dict:
        """rl-publish-fastpath: digests + send checksum computed on the workers."""
        return next(r for r in self.call_workers(remote_export_digest, kwargs) if r is not None)

    def append_jsonl(self, name: str, record: dict) -> None:
        with (self.out / name).open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


# ----------------------------------------------------------------------------- ports
@dataclass
class VerlPendingGeneration:
    """A batch whose prompts went to the rollout servers (:meth:`VerlRolloutPool.start_generate`):
    the round, the publication ``(version, hash)`` the servers held at submission,
    ``prepare_step``'s metrics and the submission time."""

    rollout_id: int
    published: tuple[int, str]
    prep_metrics: dict
    started: float
    expected_policy_version: str | None = None
    ahead: bool = False
    cancelled: bool = False


def _check_expected_token(rollout_id: int, published: tuple[int, str], expected: str | None) -> None:
    """IR-3 (as Miles' ``_begin_generation``): the driver's token and the publication the
    servers hold must be the same, before any prompt is submitted."""
    if expected is None:
        return
    token = f"yeto:{published[0]}:{published[1]}"
    if expected != token:
        from yeto.rl.engine.driver import PolicyIdentityError  # the driver never imports the adapter

        raise PolicyIdentityError(f"rollout {rollout_id}: driver expects {expected}, the servers hold {token}")


class VerlRolloutPool:
    def __init__(self, island: VerlIsland, dispense=None):
        from .data_cursor import VerlDataCursor

        self.island = island
        # S27 G-EU (VERL-PARITY #31): the rows come from the task dispenser
        # (``task_dispense.VerlDispensedBatches``, installed on the trainer already)
        self.dispense = dispense
        if dispense is not None:
            # the dispenser keeps the data position: nothing to seek (the driver's
            # restart and rejoin paths skip a pool without a callable seek)
            self.seek_data_cursor = None
        else:
            # S17 N16: lets the driver skip whole rounds when the island (re)starts at v > 0
            self._cursor = VerlDataCursor(island.trainer)

    def data_cursor(self) -> dict[str, int]:
        if self.dispense is not None:
            return self.dispense.cursor()
        return self._cursor.cursor()

    def commit_dispensed(self, rollout_id: int) -> dict | None:
        """S27 G-EU (driver, after the round's sync boundary): report the round's
        kept rows trained to the dispenser; None without one."""
        if self.dispense is None or getattr(self, "_dispense_closed", False):
            return None
        return self.dispense.commit(int(rollout_id))

    def close_dispensed(self) -> dict | None:
        """S27 G-EU (driver, finished run): release every row not trained yet."""
        if self.dispense is None or getattr(self, "_dispense_closed", False):
            return None
        self._dispense_closed = True
        return self.dispense.close()

    def seek_data_cursor(self, cursor) -> dict[str, int]:
        landed = self._cursor.seek(cursor)
        self.island.emit("rl_verl_data_seek", data_cursor=landed)
        return landed

    def generate(self, rollout_id: int, *, expected_policy_version: str | None = None) -> RolloutBatchHandle:
        from verl.utils.skip import SkipManager

        isl, t = self.island, self.island.trainer
        if isl.published is None or isl.published[0] != rollout_id:
            raise RuntimeError(f"rollout {rollout_id} requested without a verified publication")
        t.global_steps = rollout_id + 1
        SkipManager.set_step(t.global_steps)
        t.timing_raw = {}
        started = time.monotonic()
        prep = t.prepare_step()
        return self._collect(VerlPendingGeneration(rollout_id, isl.published, dict(prep or {}), started,
                                                   expected_policy_version))

    # -- rollout ahead (driver ``rollout_ahead``, ``yeto.rl.engine.rollout_ahead``; S27 G-EV) --
    def start_generate(self, rollout_id: int, *,
                       expected_policy_version: str | None = None) -> VerlPendingGeneration:
        """Submit the prompts of ``rollout_id`` to the rollout servers with the policy
        published now (``rollout_id - 1``) and return while they generate.

        Only the separate mode runs it (the vLLM servers have their own cards, so they
        generate while the trainer trains ``rollout_id - 1``). verl's ``prepare_step``
        hands the prompts to the agent loop without waiting (fork ``agent_loop_tq``);
        the prompts carry ``global_steps = rollout_id + 1``, as in :meth:`generate`. The
        trainer's own step counter goes back to the round it is about to train."""
        from verl.utils.skip import SkipManager

        isl, t = self.island, self.island.trainer
        if isl.published is None or isl.published[0] != rollout_id - 1:
            raise RuntimeError(f"rollout {rollout_id} started ahead without the publication of "
                               f"v{rollout_id - 1} (published: {isl.published})")
        _check_expected_token(rollout_id, isl.published, expected_policy_version)
        training_step = t.global_steps
        t.global_steps = rollout_id + 1
        SkipManager.set_step(t.global_steps)
        started = time.monotonic()
        try:
            prep = t.prepare_step()
        finally:
            t.global_steps = training_step
            SkipManager.set_step(training_step)
        isl.emit("rl_verl_generate_started", rollout_id=rollout_id, policy_version=isl.published[0])
        return VerlPendingGeneration(rollout_id, isl.published, dict(prep or {}), started,
                                     expected_policy_version, ahead=True)

    def finish_generate(self, pending: VerlPendingGeneration) -> RolloutBatchHandle:
        """Wait for a batch :meth:`start_generate` submitted (the driver calls it after the
        previous round trained, before the next publication); the servers are idle after it."""
        from verl.utils.skip import SkipManager

        if pending.cancelled:
            raise RuntimeError(f"rollout {pending.rollout_id}: the batch generated ahead was cancelled")
        t = self.island.trainer
        t.global_steps = pending.rollout_id + 1
        SkipManager.set_step(t.global_steps)
        t.timing_raw = {}
        return self._collect(pending)

    def cancel_generate(self, pending: VerlPendingGeneration) -> None:
        """A failed round leaves its batch generating: nothing will train it. verl has no
        per-batch abort on this path; the island stops after the round's error."""
        pending.cancelled = True

    def _collect(self, pending: VerlPendingGeneration) -> RolloutBatchHandle:
        """verl steps 1-3 of ``_step_once`` after the prompts were submitted: sample the
        replay buffer (waits for the batch), rule reward, balance; the batch handle."""
        import numpy as np
        import transfer_queue as tq

        isl, t = self.island, self.island.trainer
        rollout_id = pending.rollout_id
        version, digest = pending.published
        metrics: dict = dict(pending.prep_metrics)
        started = pending.started
        t.on_sample_begin()
        batch, off_policy = t.replay_buffer.sample(global_steps=t.global_steps, partition_id="train",
                                                   batch_size=t.config.data.train_batch_size)
        metrics.update(off_policy or {})
        batch.extra_info["temperature"] = t.config.actor_rollout_ref.rollout.temperature
        t.on_sample_end()  # sync: replicas sleep until the next publication
        if t.reward_loop_manager.reward_loop_worker_handles is None:
            batch = t._compute_reward_colocate(batch, metrics=metrics)
        batch = t._balance_batch(batch, metrics=metrics)
        gen_seconds = time.monotonic() - started
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id,
                               select_fields=["uid", "rm_scores", "response_mask", "prompts"])
        mask_nested = data["response_mask"]
        padded = data.to_padded_tensor()
        uids = [str(u) for u in padded.pop("uid").tolist()]
        prompt_sha = hashlib.sha256(padded["prompts"].cpu().numpy().tobytes()).hexdigest()
        # the same digest independent of arrival order (rows sorted): compares the prompts two
        # runs drew for one round (T-C4: a resumed island vs a continuous one)
        prompt_set_sha = hashlib.sha256(b"".join(sorted(
            row.tobytes() for row in padded["prompts"].cpu().numpy()))).hexdigest()
        gen_steps = sorted({int(tag.get("global_steps", -1)) for tag in batch.tags})
        rewards = padded["rm_scores"].float().sum(-1).tolist()
        lengths = padded["response_mask"].float().sum(-1).tolist()
        del mask_nested
        padding = [bool(tag.get("is_padding", False)) for tag in batch.tags]
        token = f"yeto:{version}:{digest}"
        groups: dict[str, list[int]] = {}
        for i, uid in enumerate(uids):
            if not padding[i]:
                groups.setdefault(uid, []).append(i)
        metas = []
        for gi, (uid, idx) in enumerate(sorted(groups.items())):
            values = np.array([rewards[i] for i in idx], dtype=np.float64)
            metas.append(GroupMetadata(
                group_id=f"r{rollout_id}-{uid}",
                sample_ids=tuple(f"r{rollout_id}-{uid}-{k}" for k in range(len(idx))),
                policy_token=token,
                reward_mean=float(values.mean()),
                reward_std=float(values.std()),
                token_count=int(sum(lengths[i] for i in idx)),
            ))
        isl.pending = {"batch": batch, "metrics": metrics, "gen_seconds": gen_seconds}
        # S27 G-EU: the dispensed rows of the groups the trainer gets wait for the commit
        data_rows = (None if self.dispense is None else
                     self.dispense.kept_round(rollout_id, {uid: len(idx) for uid, idx in groups.items()}))
        lens = sorted(lengths)
        summary = {"resp_len_mean": float(np.mean(lengths)) if lengths else None,
                   "resp_len_p95": float(lens[int(0.95 * (len(lens) - 1))]) if lens else None,
                   "reward_p50": float(np.median(rewards)) if rewards else None}
        isl.append_jsonl(f"verl-rollout-{isl.learner_id}.jsonl", {
            "rollout_id": rollout_id, "policy_token": token, "groups": len(metas),
            "samples": len(uids), "reward_mean": float(np.mean(rewards)) if rewards else None,
            "gen_seconds": gen_seconds, "expected_policy_version": pending.expected_policy_version,
            **({"ahead": True} if pending.ahead else {}),
            "prompts_sha256": prompt_sha, "prompt_set_sha256": prompt_set_sha,
            "sample_submit_global_steps": gen_steps,
            "trainer_global_step": t.global_steps, "rewards": rewards, "uids": uids,
            "data_cursor": self.data_cursor()})
        return RolloutBatchHandle(
            rollout_id=rollout_id, policy_version=version, policy_hash=digest,
            groups=tuple(metas), completed=len(metas), aborted=0, payload=batch,
            batch_summary=summary, data_cursor=self.data_cursor(),
            backend="verl", round_data_rows=data_rows,
        )

    def abort(self) -> None:
        return None

    def members(self) -> frozenset[str]:
        return frozenset({MEMBER})


def round_base_policy(island, batch: RolloutBatchHandle) -> tuple[int, str]:
    """(version, hash) of the policy a round's optimizer step starts from: the island's
    publication of that round. Serial rounds: the policy that generated the batch.
    Rollout ahead (S27 G-EV): the batch was generated one version earlier, so the
    receipt names the round's own publication (the full-parameter stream binds it to
    the round's anchor) and the batch evidence keeps the generating policy."""
    published = getattr(island, "published", None)
    if published is not None and int(published[0]) == int(batch.rollout_id) > int(batch.policy_version):
        return int(published[0]), str(published[1])
    return int(batch.policy_version), str(batch.policy_hash)


class VerlTrainerGroup(CutPorts):
    publish_offloaded = False

    def __init__(self, island: VerlIsland):
        self.island = island
        self._metrics = None
        self._cut_trainer = None

    # ---- cut verbs (S25 N7.6; .cut.CutPorts) ----
    def _cut_calls(self):
        return self.island.call_workers

    def _optimizer_steps_per_round(self) -> int:
        from .cut import optimizer_steps_per_round

        return optimizer_steps_per_round(self.island.trainer.config)

    def onload(self) -> None:
        return None

    def offload(self) -> None:
        return None

    def train_step(self, batch: RolloutBatchHandle) -> LocalStepReceipt:
        import transfer_queue as tq
        from verl.trainer.ppo.v1.replay_buffer import DAPO_FILTERED_REWARD_COUNTS_KEY
        from yeto.rl.engine.driver import TrainStepMetrics

        isl, t = self.island, self.island.trainer
        pending = isl.pending
        kv = batch.payload
        metrics = pending["metrics"]
        timing = t.timing_raw
        t.yeto_criteria_sink = isl  # read by YetoSyncTrainer._compute_old_log_prob
        kv = t._compute_old_log_prob(kv, metrics=metrics)
        if t.use_reference_policy:
            kv = t._compute_ref_log_prob(kv, metrics=metrics)
        kv = t._compute_advantage(kv, metrics=metrics)
        if getattr(isl, "debug_dump", False):  # S27 G-EU (VERL-PARITY #35): --rl-debug-dump
            debug_dump(isl, kv, batch.rollout_id)
        kv = t._update_actor(kv, metrics=metrics)
        try:
            t._compute_metrics(kv, metrics, timing, global_steps=t.global_steps, epoch=0)
        except Exception as exc:  # noqa: BLE001 - metrics are archival
            metrics["yeto/compute_metrics_error"] = repr(exc)[:300]
        tq.kv_clear(keys=kv.keys, partition_id=kv.partition_id)
        metrics.pop(DAPO_FILTERED_REWARD_COUNTS_KEY, None)
        try:
            t.logger.log(data={k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                         step=t.global_steps)
        except Exception:  # noqa: BLE001
            pass
        isl.last_metrics = {k: (v if isinstance(v, (int, float, str)) else repr(v))
                            for k, v in metrics.items()}
        isl.append_jsonl(f"verl-metrics-{isl.learner_id}.jsonl",
                         {"rollout_id": batch.rollout_id, "global_step": t.global_steps,
                          "metrics": isl.last_metrics})
        grad_norm = _metric(metrics, "grad_norm")
        lr = _metric(metrics, "lr")
        self._metrics = TrainStepMetrics(
            grad_norm=float("nan") if grad_norm is None else grad_norm,
            loss=_metric(metrics, "pg_loss"),
            pg_loss=_metric(metrics, "pg_loss"),
            lr=lr,
            clip_fraction=_metric(metrics, "pg_clipfrac"),
            train_step=batch.rollout_id,
            applied_lrs=None if lr is None else (lr,),
        )
        if grad_norm is not None:
            self._check_first_step_after_resume(batch.rollout_id, isl.emit)
        ids = tuple(s for g in batch.groups for s in g.sample_ids)
        base_version, base_hash = round_base_policy(isl, batch)
        receipt = LocalStepReceipt(
            algorithm="grpo", learner_id=isl.learner_id, learner_generation=0,
            base_policy_version=base_version, base_policy_hash=base_hash,
            input_batch_hash=_sha("|".join(ids)), trajectory_ids=ids,
            trained_tokens=sum(g.token_count for g in batch.groups), optimizer_steps=1,
            optimizer_step_succeeded=grad_norm is not None, parameter_layout_hash=isl.layout_hash,
        )
        if isl.full_group is not None:  # the full-parameter role stream binds the round to this
            from .full_parameter_stream import VerlRoundEvidence

            isl.full_group.record_round(receipt, VerlRoundEvidence(
                rollout_id=batch.rollout_id, behavior_policy_hash=batch.policy_hash,
                input_batch_hash=receipt.input_batch_hash, trained_tokens=receipt.trained_tokens,
                trajectory_ids=ids))
        return receipt

    def step_metrics(self):
        return self._metrics

    def algorithm_metrics(self) -> dict:
        """Mismatch criteria in the Miles convention (tape: rl_round_trained.mismatch)."""
        return dict(self.island.last_criteria)

    def round_metrics(self) -> dict:
        out = {}
        for key in ("critic/score/mean", "response_length/mean", "actor/entropy",
                    "rollout_corr/k3_kl", "rollout_corr/kl", "training/rollout_probs_diff_mean"):
            value = self.island.last_metrics.get(key)
            if isinstance(value, (int, float)):
                out[f"verl/{key}"] = float(value)
        return out


@dataclass(frozen=True)
class VerlResidentState(TrainerResidentState):
    """A verl policy kept in the trainer; carries the bf16 send checksum the
    publisher compares with the vLLM read-back (no tensors on the driver)."""

    lora_checksum: Mapping[str, Any] | None = field(default=None, compare=False)


def sent_checksum(state) -> tuple[dict, int]:
    """(send checksum, bf16 payload bytes) of the policy being published.  A resident
    state brings both from the trainer; any other state is hashed here as before."""

    if is_resident(state) and getattr(state, "lora_checksum", None) is not None:
        payload_bytes = sum(prod(shape) * 2 for _name, shape in state.digest.specs)
        return dict(state.lora_checksum), int(payload_bytes)
    tensors = dict(state.to_lora().tensors)
    return pub.lora_checksum(tensors), sum(int(t.numel()) * 2 for t in tensors.values())


class VerlPolicyState:
    def __init__(self, island: VerlIsland):
        self.island = island
        self.version = 0

    def export(self) -> TrainableState:
        from yeto.rl.core import canonical_state

        tensors, _info = self.island.export_tensors()
        state = canonical_state(self.version, tensors, base_model_revision=self.island.base_model_revision,
                                lora_config_hash=self.island.lora_config_hash,
                                layout_hash=self.island.layout_hash,
                                expected_specs=self.island.expected_specs)
        return TrainableState.from_lora(state)

    def export_digest(self, *, policy_version: int) -> VerlResidentState | None:
        """rl-publish-fastpath: the workers export and hash; only digests cross Ray.
        None when the island has no digest hook (the driver then calls :meth:`export`)."""

        hook = getattr(self.island, "export_digest", None)
        if not callable(hook):
            return None
        isl, started = self.island, time.monotonic()
        raw = hook({"base_model_revision": isl.base_model_revision,
                    "lora_config_hash": isl.lora_config_hash, "layout_hash": isl.layout_hash,
                    "expected_specs": tuple(isl.expected_specs)})
        digest = PolicyDigest.from_wire(raw["digest"])
        print(f"[rl] verl publish-fastpath digest export v{policy_version}: "
              f"total={time.monotonic() - started:.2f}s trainer={raw.get('export_seconds')}s "
              f"hash={digest.hash_seconds:.2f}s bytes={digest.payload_bytes}", flush=True)
        return VerlResidentState(
            layout=LAYOUT_LORA, base_model_revision=isl.base_model_revision,
            config_hash=isl.lora_config_hash, layout_hash=str(raw["layout_hash"]),
            policy_version=int(policy_version), digest=digest, materialize_fn=self._materialize,
            lora_checksum=raw["lora_checksum"])

    def _materialize(self, policy_version: int) -> TrainableState:
        from yeto.rl.core import canonical_state

        lora = self.export().to_lora()
        return TrainableState.from_lora(canonical_state(
            int(policy_version), lora.tensors, base_model_revision=lora.base_model_revision,
            lora_config_hash=lora.lora_config_hash, layout_hash=lora.layout_hash,
            expected_specs=lora.specs))

    def apply(self, state: TrainableState, *, optimizer: str, local_step: int) -> None:
        if optimizer not in ("reset", "preserve"):
            raise ValueError(optimizer)
        lora = state.to_lora()
        tensors = {name: lora.tensors[name] for name in lora.tensors}
        results = self.island.call_workers(remote_apply_lora, tensors, optimizer == "reset")
        self.version = int(state.policy_version)
        self.island.append_jsonl(f"verl-apply-{self.island.learner_id}.jsonl", {
            "policy_version": state.policy_version, "optimizer": optimizer, "local_step": local_step,
            "policy_tensor_hash": state.policy_tensor_hash(), "workers": results})


class PublicationFailed(RuntimeError):
    pass


class VerlPublisher:
    def __init__(self, island: VerlIsland, *, strict: bool = True, timeout_s: float = 120.0):
        self.island = island
        self.strict = strict
        self.timeout_s = timeout_s

    def _await_readback(self, version: int) -> dict | None:
        path = self.island.readback_dir / f"readback-v{version}.json"
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            if path.is_file():
                try:
                    return json.loads(path.read_text())
                except ValueError:
                    pass
            time.sleep(0.2)
        return None

    def publish(self, state: TrainableState) -> PublicationResult:
        from yeto.rl.engine.driver import PublicationError
        from yeto.rl.engine.ports import PublicationCause

        isl, t = self.island, self.island.trainer
        version = int(state.policy_version)
        sent, payload_bytes = sent_checksum(state)
        (isl.readback_dir / "expected_version").write_text(str(version))
        stale = isl.readback_dir / f"readback-v{version}.json"
        if stale.exists():
            stale.unlink()
        started = time.monotonic()
        t.checkpoint_manager.update_weights(t.global_steps)
        update_seconds = time.monotonic() - started
        readback = self._await_readback(version)
        check = pub.compare(version, sent, readback,
                            detail="" if readback else f"no readback within {self.timeout_s}s")
        record = {**check.to_event(), "update_seconds": update_seconds,
                  "readback_seconds": time.monotonic() - started - update_seconds,
                  "policy_tensor_hash": state.policy_tensor_hash(),
                  "readback_dtypes": (readback or {}).get("dtypes"),
                  "readback_note": (readback or {}).get("note"),
                  "readback_error": (readback or {}).get("error")}
        isl.append_jsonl(f"verl-publish-{isl.learner_id}.jsonl", record)
        isl.emit("rl_publication_check", **{k: v for k, v in record.items() if k != "event"})
        if check.status != pub.VERIFIED and self.strict:
            cause = (PublicationCause.LORA_UNVERIFIABLE if check.status == pub.UNVERIFIABLE
                     else PublicationCause.PAYLOAD_MISMATCH)
            raise PublicationError(f"verl publication v{version}: {check.status} {check.detail} "
                                   f"{list(check.mismatched[:3])}", cause=cause)
        digest = state.policy_tensor_hash()
        isl.published = (version, digest)
        manifest = InferencePublicationManifest(
            publication_mode="full", base_policy_version=None, target_policy_version=version,
            target_policy_hash=digest, target_manifest_hash=_sha(f"{version}:{digest}"),
            payload_hash=check.readback_checksum or sent["total"], payload_bytes=payload_bytes,
            complete=True)
        return PublicationResult(manifest, frozenset({MEMBER}))


class VerlFullParameterPublisher:
    """Publication of a full-parameter island: verl's colocated ``update_weights``
    of the whole model into vLLM.

    vLLM's copy is not read back: the LoRA read-back hook (``vllm_readback``)
    reads a registered adapter, and no full-parameter hook exists yet.  Every
    record says so (status ``UNVERIFIED``); nothing claims a verified engine copy."""

    STATUS = "UNVERIFIED"
    DETAIL = "full-parameter publication: vLLM's weights are not read back"

    def __init__(self, island: VerlIsland):
        self.island = island

    def publish(self, state) -> PublicationResult:
        isl, t = self.island, self.island.trainer
        version = int(state.policy_version)
        digest = state.policy_tensor_hash()
        started = time.monotonic()
        t.checkpoint_manager.update_weights(t.global_steps)
        record = {"status": self.STATUS, "detail": self.DETAIL, "policy_version": version,
                  "policy_tensor_hash": digest, "payload_hash": state.digest.payload_hash,
                  "payload_bytes": int(state.digest.payload_bytes),
                  "update_seconds": time.monotonic() - started}
        isl.append_jsonl(f"verl-publish-{isl.learner_id}.jsonl", record)
        isl.emit("rl_publication_check", **record)
        isl.published = (version, digest)
        manifest = InferencePublicationManifest(
            publication_mode="full", base_policy_version=None, target_policy_version=version,
            target_policy_hash=digest, target_manifest_hash=_sha(f"{version}:{digest}"),
            payload_hash=state.digest.payload_hash, payload_bytes=int(state.digest.payload_bytes),
            complete=True)
        return PublicationResult(manifest, frozenset({MEMBER}))


class VerlPlacement:
    def describe(self) -> PlacementDescription:
        return PlacementDescription("colocated", ("cuda:0",), ("cuda:0",),
                                    extra={"weight_transport": "verl-colocated-memory"})


DEBUG_DUMP_FIELDS = ("uid", "prompts", "responses", "response_mask", "rm_scores", "advantages")


def debug_dump(isl: VerlIsland, kv_batch, rollout_id: int, *, get=None, save=None) -> dict:
    """S27 G-EU (``--rl-debug-dump``, ``train_options.debug_plan``): the round's
    samples and train data after the advantage step, as
    ``verl-debug-<id>-r<round>.pt`` at the top of the output dir (the tape mirror
    picks it up). Archival: a field verl does not hold is skipped and named in the
    file, an error is recorded on the tape; training never fails on it."""
    out = isl.out / f"verl-debug-{isl.learner_id}-r{int(rollout_id):04d}.pt"
    record: dict = {"rollout_id": int(rollout_id), "global_step": int(getattr(isl.trainer, "global_steps", -1))}
    try:
        if get is None:
            import transfer_queue as tq

            def get(fields):
                return tq.kv_batch_get(keys=kv_batch.keys, partition_id=kv_batch.partition_id,
                                       select_fields=list(fields))
        if save is None:
            import torch

            save = torch.save
        tensors, missing = {}, []
        for name in DEBUG_DUMP_FIELDS:
            try:
                data = get([name])
                value = data.to_padded_tensor()[name] if hasattr(data, "to_padded_tensor") else data[name]
            except Exception as exc:  # noqa: BLE001 - archival: one field at a time
                missing.append(f"{name}: {type(exc).__name__}: {str(exc)[:120]}")
                continue
            tensors[name] = value.cpu() if hasattr(value, "cpu") else list(value)
        save({**record, **tensors, "missing": missing}, out)
        record.update(path=out.name, fields=sorted(tensors), missing=missing)
    except Exception as exc:  # noqa: BLE001 - archival, training continues
        record["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    isl.emit("rl_verl_debug_dump", **record)
    return record


def criteria_from_trainer_batch(isl: VerlIsland, kv_batch) -> None:
    """Called after verl recomputed old log-probs: Miles-convention criteria + raw dump."""
    import torch
    import transfer_queue as tq

    t = isl.trainer
    step = int(t.global_steps)
    try:
        data = tq.kv_batch_get(keys=kv_batch.keys, partition_id=kv_batch.partition_id,
                               select_fields=["old_log_probs", "rollout_log_probs", "response_mask",
                                              "responses"])
        padded = data.to_padded_tensor()
        train_lp, infer_lp = padded["old_log_probs"], padded["rollout_log_probs"]
        mask = padded["response_mask"]
        width = min(train_lp.shape[-1], infer_lp.shape[-1], mask.shape[-1])
        train_lp, infer_lp, mask = train_lp[..., :width], infer_lp[..., :width], mask[..., :width]
        crit = mismatch_criteria.compute(train_lp, infer_lp, mask, tis_upper=isl.tis_upper)
        try:
            limits = mismatch_criteria.thresholds_for(isl.thresholds_key)
            crit["alarms"] = mismatch_criteria.judge(crit, limits)
            crit["thresholds_key"] = list(isl.thresholds_key)
        except mismatch_criteria.Uncalibrated as exc:
            crit["alarms"] = None
            crit["uncalibrated"] = str(exc)
        torch.save({"step": step, "old_log_probs": train_lp.float().cpu(),
                    "rollout_log_probs": infer_lp.float().cpu(),
                    "response_mask": mask.cpu(), "responses": padded["responses"].cpu(),
                    "dtypes": [str(train_lp.dtype), str(infer_lp.dtype)]},
                   isl.out / f"verl-dump-{isl.learner_id}-step{step:03d}.pt")  # top level: synced by the tape
    except Exception as exc:  # noqa: BLE001 - reported as a failed criterion, training continues
        crit = {"error": f"{type(exc).__name__}: {exc}", "alarms": ["criteria_unavailable"]}
    crit["global_step"] = step
    isl.last_criteria = crit
    isl.append_jsonl(f"verl-criteria-{isl.learner_id}.jsonl", crit)
    isl.emit("rl_mismatch_criteria", **{k: v for k, v in crit.items()})
    if "error" not in crit:
        try:
            provenances, read = _row_field(kv_batch, "provenance")
            event = source_event(train_lp, infer_lp, mask, provenances, read,
                                 rollout_id=step, trainer_version=step, tis_upper=isl.tis_upper)
        except Exception as exc:  # noqa: BLE001 - archival, training continues
            event = source_error(exc, rollout_id=step, trainer_version=step)
        _emit_offpolicy_source(isl, event)


def _row_field(kv_batch, name: str) -> tuple[list | None, str]:
    """Per-row values of one ``extra_fields`` entry (``HarborAgentLoop`` puts
    ``provenance`` there); ``(None, reason)`` when the batch has none.

    verl's TransferQueue agent loop (``agent_loop_tq._agent_loop_postprocess``)
    stores ``AgentLoopOutput.as_dict()``, so the entry sits inside the
    ``extra_fields`` record of each row, not under its own name (the classic
    agent loop flattens it into ``non_tensor_batch``; the fully async runner
    reads that).  The batch is read by the record name."""
    import transfer_queue as tq

    try:
        data = tq.kv_batch_get(keys=kv_batch.keys, partition_id=kv_batch.partition_id,
                               select_fields=["extra_fields"])
        rows = extra_field_values(list(data["extra_fields"]), name)
    except Exception as exc:  # noqa: BLE001 - the rows then group under ``unknown``
        return None, f"missing: {type(exc).__name__}: {str(exc)[:200]}"
    if all(v is None for v in rows):
        return None, f"missing: no {name!r} in extra_fields"
    return rows, "ok"


def extra_field_values(records: list, name: str) -> list:
    """``record[name]`` of each row's ``extra_fields`` (``None`` where absent).  A
    TransferQueue read can return each record wrapped in a ``NonTensorData``."""
    values = []
    for record in records:
        record = getattr(record, "data", record)
        values.append(record.get(name) if isinstance(record, dict) else None)
    return values


def source_event(train_lp, infer_lp, mask, provenances, provenance_read: str, *, rollout_id: int,
                 trainer_version: int, tis_upper: float) -> dict:
    """The ``offpolicy_source`` event of one verl step (padded tensors, same width)."""
    event = offpolicy_source.offpolicy_source_event(
        offpolicy_source.masked_source_samples(train_lp.float().tolist(), infer_lp.float().tolist(),
                                               mask.tolist(), provenances=provenances),
        rollout_id=rollout_id, trainer_version=trainer_version, backend="verl", tis_upper=tis_upper)
    event["provenance_read"] = provenance_read
    return event


def source_error(exc: Exception, *, rollout_id: int, trainer_version: int) -> dict:
    return {"schema": offpolicy_source.SCHEMA, "rollout_id": int(rollout_id), "trainer_version": int(trainer_version),
            "backend": "verl", "error": f"{type(exc).__name__}: {str(exc)[:300]}", "sources": []}


def _emit_offpolicy_source(isl: VerlIsland, event: dict) -> None:
    isl.append_jsonl(f"verl-offpolicy-source-{isl.learner_id}.jsonl", event)
    isl.emit(offpolicy_source.OFFPOLICY_SOURCE_EVENT, **event)
