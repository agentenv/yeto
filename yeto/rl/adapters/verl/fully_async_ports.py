"""Engine ports of the verl fully_async path (agentic-rollout-utilization 6.4b).

The ports run in yeto's task runner and talk to the trainer actor
(:class:`.fully_async_runner.YetoFullyAsyncTrainer`) only through
``call(method, *args)`` (``ray.get(actor.method.remote(*args))`` in the image;
a fake in the unit tests).  The actor returns plain data, so everything the
driver sees -- version segments, discards, carry-over, publication read-back --
is decided here and in :mod:`.fully_async_round`.

* ``generate``  -- the actor drains the MessageQueue until the round is full
  (:class:`~.fully_async_round.RoundCollector`, judged against the
  :class:`~.fully_async_translate.VersionMap`), assembles the kept samples and
  returns their metadata; the rollouter's own ``fit`` loop is started once,
  after the first publication.
* ``train_step`` -- old log-prob -> advantage -> actor update on the actor
  (verl's ``fit_step`` without ``_fit_update_weights``); mismatch criteria in
  the Miles convention on the trainer's DataProto.
* policy state  -- ``ports_impl.VerlPolicyState`` unchanged: its two island hooks
  (``export_tensors`` / ``call_workers``) run on the trainer actor's worker group.
* ``publish``   -- the driver has applied the state; the actor pushes it
  through ``checkpoint_manager.update_weights(global_steps=param_version)``,
  resets the rollouter's staleness counter, and the pairing
  ``param_version <-> outer version`` is recorded; the read-back of the rollouter's
  vLLM replica is compared with the trainer export (``publish.compare``).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from yeto.rl.contracts import InferencePublicationManifest, LocalStepReceipt
from yeto.rl.engine.ports import PlacementDescription, PublicationResult, RolloutBatchHandle

from . import publish as pub
from .fully_async_round import (DISCARD_MECHANISM, FULLY_ASYNC_PLACEMENT, RoundCollector,
                                SampleMeta, group_metadata)
from .fully_async_translate import VersionMap
from .rollout_events import cutoff_fields

MEMBER = "vllm-rollouter-0"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class FullyAsyncIsland:
    """Shared state of the ports for one fully_async island."""

    def __init__(self, call: Callable[..., Any], *, learner_id: int, limit: int, required: int,
                 out_dir: str, readback_dir: str, emit=None, start_rollouter=None,
                 base_model_revision: str = "", lora_config_hash: str = "", layout_hash: str = "",
                 expected_specs=()):
        self.call = call
        self.base_model_revision = base_model_revision
        self.lora_config_hash = lora_config_hash
        self.layout_hash = layout_hash
        self.expected_specs = tuple(expected_specs)
        self.learner_id = int(learner_id)
        self.limit = int(limit)
        self.required = int(required)
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.readback_dir = Path(readback_dir)
        self.readback_dir.mkdir(parents=True, exist_ok=True)
        (self.readback_dir / "expected_shapes.json").write_text(
            json.dumps({s.name: list(s.shape) for s in self.expected_specs}))
        self.emit = emit or (lambda event, **f: None)
        self.start_rollouter = start_rollouter
        self.rollouter_started = False
        self.vmap = VersionMap()
        self.hashes: dict[int, str] = {}       # outer version -> published policy hash
        self.published: tuple[int, str] | None = None
        self.last_metrics: dict = {}
        self.last_criteria: dict = {}

    # hooks of ports_impl.VerlPolicyState (run on the trainer actor's worker group)
    def export_tensors(self) -> tuple[dict, dict]:
        return self.call("yeto_export_lora")

    def call_workers(self, fn, *args):
        return self.call("yeto_call_workers", fn, *args)

    def append_jsonl(self, name: str, record: dict) -> None:
        with (self.out / name).open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


class FullyAsyncRolloutPool:
    def __init__(self, island: FullyAsyncIsland, *, max_pulls_factor: int = 8):
        self.island = island
        self.max_pulls = max_pulls_factor * island.required

    def generate(self, rollout_id: int, *, expected_policy_version: str | None = None) -> RolloutBatchHandle:
        isl = self.island
        if isl.published is None or isl.published[0] != rollout_id:
            raise RuntimeError(f"rollout {rollout_id} requested without a verified publication")
        if not isl.rollouter_started:
            if isl.start_rollouter is not None:
                isl.start_rollouter()
            isl.rollouter_started = True
        started = time.monotonic()
        collector = RoundCollector(isl.vmap, rollout_id, isl.limit, isl.required)
        pulled = 0
        while not collector.full:
            if pulled >= self.max_pulls:
                raise RuntimeError(f"rollout {rollout_id}: {pulled} queued samples pulled, only "
                                   f"{len(collector.kept)}/{isl.required} within the policy-age limit")
            meta = isl.call("yeto_pull_sample")
            if meta is None:
                raise RuntimeError(f"rollout {rollout_id}: rollouter ended the queue")
            pulled += 1
            keep = collector.offer(SampleMeta(**meta))
            isl.call("yeto_keep_last" if keep else "yeto_drop_last")
        queue = isl.call("yeto_assemble")
        version, digest = isl.published
        groups = group_metadata(collector, rollout_id, isl.hashes)
        carry = {**collector.carry_over_fields(), "queue_size_after": queue.get("queue_size"),
                 "pulled_samples": pulled}
        discard = cutoff_fields(collector.discard_tally())
        record = {"rollout_id": rollout_id, "groups": len(groups), "pulled": pulled,
                  "gen_seconds": time.monotonic() - started, "carry_over": carry,
                  "versions": {g.group_id: list(g.policy_versions or (rollout_id,)) for g in groups},
                  "rewards": [g.reward_mean for g in groups], "queue": queue,
                  "version_map": isl.vmap.to_dict()}
        isl.append_jsonl(f"verl-fa-rollout-{isl.learner_id}.jsonl", record)
        isl.emit("rl_verl_fa_round", rollout_id=rollout_id, pulled=pulled, kept=len(groups),
                 discarded=len(collector.discarded), queue_size_after=queue.get("queue_size"))
        return RolloutBatchHandle(
            rollout_id=rollout_id, policy_version=version, policy_hash=digest, groups=groups,
            completed=len(groups), aborted=0, payload=None,
            submitted_groups=pulled, abort_mechanism=DISCARD_MECHANISM if collector.discarded else None,
            carry_over=carry, **(discard if collector.discarded else {}),
        )

    def abort(self) -> None:
        return None

    def members(self) -> frozenset[str]:
        return frozenset({MEMBER})


class FullyAsyncTrainerGroup:
    publish_offloaded = False

    def __init__(self, island: FullyAsyncIsland):
        self.island = island
        self._metrics = None

    def onload(self) -> None:
        return None

    def offload(self) -> None:
        return None

    def train_step(self, batch: RolloutBatchHandle) -> LocalStepReceipt:
        from yeto.rl.engine.driver import TrainStepMetrics

        isl = self.island
        out = isl.call("yeto_train_step")
        metrics = out.get("metrics") or {}
        isl.last_metrics = metrics
        isl.last_criteria = out.get("criteria") or {}
        isl.append_jsonl(f"verl-metrics-{isl.learner_id}.jsonl",
                         {"rollout_id": batch.rollout_id, "param_version": out.get("param_version"),
                          "metrics": metrics})
        if isl.last_criteria:
            isl.append_jsonl(f"verl-criteria-{isl.learner_id}.jsonl", isl.last_criteria)
            isl.emit("rl_mismatch_criteria", **isl.last_criteria)

        def metric(*suffixes):
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and any(key.endswith(s) for s in suffixes):
                    return float(value)
            return None

        grad_norm, lr = metric("grad_norm"), metric("actor/lr", "/lr")
        self._metrics = TrainStepMetrics(
            grad_norm=float("nan") if grad_norm is None else grad_norm, loss=metric("pg_loss"),
            pg_loss=metric("pg_loss"), lr=lr, clip_fraction=metric("pg_clipfrac"),
            train_step=batch.rollout_id, applied_lrs=None if lr is None else (lr,))
        ids = tuple(s for g in batch.groups for s in g.sample_ids)
        return LocalStepReceipt(
            algorithm="grpo", learner_id=isl.learner_id, learner_generation=0,
            base_policy_version=batch.policy_version, base_policy_hash=batch.policy_hash,
            input_batch_hash=_sha("|".join(ids)), trajectory_ids=ids,
            trained_tokens=sum(g.token_count for g in batch.groups), optimizer_steps=1,
            optimizer_step_succeeded=grad_norm is not None,
            parameter_layout_hash=isl.layout_hash)

    def step_metrics(self):
        return self._metrics

    def algorithm_metrics(self) -> dict:
        return dict(self.island.last_criteria)

    def round_metrics(self) -> dict:
        out = {}
        for key in ("critic/score/mean", "response_length/mean", "actor/entropy",
                    "fully_async/partial/partial_ratio", "fully_async/partial/max_partial_span"):
            value = self.island.last_metrics.get(key)
            if isinstance(value, (int, float)):
                out[f"verl/{key}"] = float(value)
        return out


class FullyAsyncPublisher:
    """yeto publication over verl's checkpoint engine (6.4b design item 2)."""

    def __init__(self, island: FullyAsyncIsland, *, strict: bool = True,
                 timeout_s: float = 180.0, sleep=time.sleep):
        self.island = island
        self.strict = strict
        self.timeout_s = timeout_s
        self.sleep = sleep

    def _await_readback(self, version: int) -> dict | None:
        path = self.island.readback_dir / f"readback-v{version}.json"
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            if path.is_file():
                try:
                    return json.loads(path.read_text())
                except ValueError:
                    pass
            self.sleep(0.2)
        return None

    def publish(self, state) -> PublicationResult:
        from yeto.rl.engine.driver import PublicationError
        from yeto.rl.engine.ports import PublicationCause

        isl = self.island
        version = int(state.policy_version)
        lora = state.to_lora()
        sent = pub.lora_checksum(dict(lora.tensors))
        (isl.readback_dir / "expected_version").write_text(str(version))
        stale = isl.readback_dir / f"readback-v{version}.json"
        if stale.exists():
            stale.unlink()
        started = time.monotonic()
        pushed = isl.call("yeto_push_weights")  # {"param_version", "timing"}
        param_version = int(pushed["param_version"])
        isl.vmap.record(param_version, version)  # raises on contradiction / regression
        readback = self._await_readback(version)
        check = pub.compare(version, sent, readback,
                            detail="" if readback else f"no readback within {self.timeout_s}s")
        record = {**check.to_event(), "param_version": param_version,
                  "update_seconds": time.monotonic() - started,
                  "rollouter_timing": pushed.get("timing"),
                  "policy_tensor_hash": state.policy_tensor_hash(),
                  "version_map": isl.vmap.to_dict()}
        isl.append_jsonl(f"verl-publish-{isl.learner_id}.jsonl", record)
        isl.emit("rl_publication_check", **{k: v for k, v in record.items() if k != "event"})
        isl.emit("rl_verl_version_map", param_version=param_version, outer_version=version)
        if check.status != pub.VERIFIED and self.strict:
            cause = (PublicationCause.LORA_UNVERIFIABLE if check.status == pub.UNVERIFIABLE
                     else PublicationCause.PAYLOAD_MISMATCH)
            raise PublicationError(f"verl fully_async publication v{version}: {check.status} "
                                   f"{check.detail}", cause=cause)
        digest = state.policy_tensor_hash()
        isl.published = (version, digest)
        isl.hashes[version] = digest
        payload_bytes = sum(int(t.numel()) * 2 for t in lora.tensors.values())
        manifest = InferencePublicationManifest(
            publication_mode="full", base_policy_version=None, target_policy_version=version,
            target_policy_hash=digest, target_manifest_hash=_sha(f"{version}:{digest}"),
            payload_hash=check.readback_checksum or sent["total"], payload_bytes=payload_bytes,
            complete=True)
        return PublicationResult(manifest, frozenset({MEMBER}))


class FullyAsyncPlacement:
    def describe(self) -> PlacementDescription:
        return PlacementDescription(FULLY_ASYNC_PLACEMENT, ("cuda:trainer-0",), ("cuda:rollouter-0",),
                                    extra={"engine": "verl-fully-async"})
