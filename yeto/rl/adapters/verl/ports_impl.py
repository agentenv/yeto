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
from pathlib import Path
from typing import Any

from yeto.rl.contracts import InferencePublicationManifest, LocalStepReceipt
from yeto.rl.engine import mismatch_criteria
from yeto.rl.engine.ports import GroupMetadata, PlacementDescription, PublicationResult, RolloutBatchHandle
from yeto.rl.engine.trainable_state import TrainableState

from . import publish as pub
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


def remote_export_lora(worker_dict):
    """(canonical f32 CPU tensors, info) from the actor's PEFT model."""
    import torch

    peft = _peft_module(_inner_worker(worker_dict))
    out, dtypes = {}, set()
    with torch.no_grad():
        for name, param in peft.named_parameters():
            if ".lora_" not in name:
                continue
            dtypes.add(str(param.dtype))
            full = param.full_tensor() if hasattr(param, "full_tensor") else param
            out[train_to_canonical(name)] = full.detach().to("cpu", torch.float32).clone()
    return out, {"param_dtypes": sorted(dtypes), "count": len(out)}


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
        self.readback_dir = Path(os.environ.get("YETO_VERL_READBACK_DIR", str(self.out / "readback")))
        self.readback_dir.mkdir(parents=True, exist_ok=True)
        (self.readback_dir / "expected_shapes.json").write_text(
            json.dumps({s.name: list(s.shape) for s in self.expected_specs}))

    # worker calls
    def call_workers(self, fn, *args):
        import ray

        workers = self.trainer.actor_rollout_wg.workers
        return ray.get([w.__ray_call__.remote(fn, *args) for w in workers])

    def export_tensors(self) -> tuple[dict, dict]:
        results = self.call_workers(remote_export_lora)
        return results[0]

    def append_jsonl(self, name: str, record: dict) -> None:
        with (self.out / name).open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


# ----------------------------------------------------------------------------- ports
class VerlRolloutPool:
    def __init__(self, island: VerlIsland):
        self.island = island

    def generate(self, rollout_id: int, *, expected_policy_version: str | None = None) -> RolloutBatchHandle:
        import numpy as np
        import transfer_queue as tq
        from verl.utils.skip import SkipManager

        isl, t = self.island, self.island.trainer
        if isl.published is None or isl.published[0] != rollout_id:
            raise RuntimeError(f"rollout {rollout_id} requested without a verified publication")
        version, digest = isl.published
        t.global_steps = rollout_id + 1
        SkipManager.set_step(t.global_steps)
        t.timing_raw = {}
        metrics: dict = {}
        started = time.monotonic()
        prep = t.prepare_step()
        if prep:
            metrics.update(prep)
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
        lens = sorted(lengths)
        summary = {"resp_len_mean": float(np.mean(lengths)) if lengths else None,
                   "resp_len_p95": float(lens[int(0.95 * (len(lens) - 1))]) if lens else None,
                   "reward_p50": float(np.median(rewards)) if rewards else None}
        isl.append_jsonl(f"verl-rollout-{isl.learner_id}.jsonl", {
            "rollout_id": rollout_id, "policy_token": token, "groups": len(metas),
            "samples": len(uids), "reward_mean": float(np.mean(rewards)) if rewards else None,
            "gen_seconds": gen_seconds, "expected_policy_version": expected_policy_version,
            "prompts_sha256": prompt_sha, "sample_submit_global_steps": gen_steps,
            "trainer_global_step": t.global_steps, "rewards": rewards, "uids": uids})
        return RolloutBatchHandle(
            rollout_id=rollout_id, policy_version=version, policy_hash=digest,
            groups=tuple(metas), completed=len(metas), aborted=0, payload=batch,
            batch_summary=summary,
        )

    def abort(self) -> None:
        return None

    def members(self) -> frozenset[str]:
        return frozenset({MEMBER})


class VerlTrainerGroup:
    publish_offloaded = False

    def __init__(self, island: VerlIsland):
        self.island = island
        self._metrics = None

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
        ids = tuple(s for g in batch.groups for s in g.sample_ids)
        return LocalStepReceipt(
            algorithm="grpo", learner_id=isl.learner_id, learner_generation=0,
            base_policy_version=batch.policy_version, base_policy_hash=batch.policy_hash,
            input_batch_hash=_sha("|".join(ids)), trajectory_ids=ids,
            trained_tokens=sum(g.token_count for g in batch.groups), optimizer_steps=1,
            optimizer_step_succeeded=grad_norm is not None, parameter_layout_hash=isl.layout_hash,
        )

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
        lora = state.to_lora()
        sent = pub.lora_checksum(dict(lora.tensors))
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
        payload_bytes = sum(int(t_.numel()) * 2 for t_ in lora.tensors.values())
        manifest = InferencePublicationManifest(
            publication_mode="full", base_policy_version=None, target_policy_version=version,
            target_policy_hash=digest, target_manifest_hash=_sha(f"{version}:{digest}"),
            payload_hash=check.readback_checksum or sent["total"], payload_bytes=payload_bytes,
            complete=True)
        return PublicationResult(manifest, frozenset({MEMBER}))


class VerlPlacement:
    def describe(self) -> PlacementDescription:
        return PlacementDescription("colocated", ("cuda:0",), ("cuda:0",),
                                    extra={"weight_transport": "verl-colocated-memory"})


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
