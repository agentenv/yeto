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
* policy state  -- ``ports_impl.VerlPolicyState`` unchanged: its island hooks
  (``export_tensors`` / ``export_digest`` / ``call_workers``) run on the trainer actor's worker group.
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
from yeto.rl.engine.host_memory import host_memory_mode
from yeto.rl.engine.offpolicy_source import OFFPOLICY_SOURCE_EVENT
from yeto.rl.engine.ports import PlacementDescription, PublicationResult, RolloutBatchHandle

from . import publish as pub
from .cut import CutPorts
from .fully_async_cut import FullyAsyncDataCursor
from .fully_async_round import (DISCARD_MECHANISM, FULLY_ASYNC_PLACEMENT, RoundCollector,
                                SampleMeta, group_metadata)
from .fully_async_translate import VersionMap
from .ports_impl import sent_checksum
from .rollout_events import cutoff_fields

MEMBER = "vllm-rollouter-0"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class FullyAsyncIsland:
    """Shared state of the ports for one fully_async island."""

    def __init__(self, call: Callable[..., Any], *, learner_id: int, limit: int, required: int,
                 out_dir: str, readback_dir: str, emit=None, start_rollouter=None,
                 base_model_revision: str = "", lora_config_hash: str = "", layout_hash: str = "",
                 expected_specs=(), seek_rollouter=None, optimizer_steps_per_round: int | None = None):
        """``seek_rollouter(prompts)``: skip that many prompts of the rollouter's data
        before it starts (``fully_async_cut.skip_rollouter_prompts``; round cuts, S26
        E9.2). ``optimizer_steps_per_round``: the resume first-step check."""
        self.call = call
        self.seek_rollouter = seek_rollouter
        self.optimizer_steps_per_round = optimizer_steps_per_round
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
        # A full-parameter island's trainer group (``full_parameter_stream.VerlFullParameterGroup``,
        # set by ``fully_async_full.full_parameter_ports``, S27 G-EW); None on a LoRA island.
        self.full_group = None

    # hooks of ports_impl.VerlPolicyState (run on the trainer actor's worker group)
    def export_tensors(self) -> tuple[dict, dict]:
        return self.call("yeto_export_lora")

    def export_digest(self, kwargs: dict) -> dict:
        """rl-publish-fastpath: digests + send checksum computed on the trainer actor."""
        return self.call("yeto_export_digest", kwargs)

    def call_workers(self, fn, *args):
        return self.call("yeto_call_workers", fn, *args)

    def call_workers_shared(self, fn, *args):
        """:meth:`call_workers` with every argument put in the trainer actor's object
        store once (``ports_impl.VerlIsland.call_workers_shared``; whole parameter
        tensors of a full-parameter apply are not serialized per worker)."""
        return self.call("yeto_call_workers_shared", fn, *args)

    def append_jsonl(self, name: str, record: dict) -> None:
        with (self.out / name).open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


class FullyAsyncRolloutPool:
    def __init__(self, island: FullyAsyncIsland, *, max_pulls_factor: int = 8,
                 queue_end_is_buffer_end: bool = False):
        """``queue_end_is_buffer_end``: the queue is the sample buffer (a buffer
        trainer): its end is the samplers' data end, so a round it cannot fill
        raises :class:`~yeto.rl.engine.buffer_payload.SampleBufferExhausted` and
        the run ends FINISHED (S26 E9.3), not a failure."""
        self.island = island
        self.max_pulls = max_pulls_factor * island.required
        self.queue_end_is_buffer_end = bool(queue_end_is_buffer_end)
        self._cursor = FullyAsyncDataCursor(island.seek_rollouter)

    def data_cursor(self) -> dict[str, int]:
        """Prompts the rounds took from the queue (kept or discarded; .fully_async_cut)."""
        return self._cursor.cursor()

    def export_in_flight(self) -> dict[str, Any]:
        """A round cut's in-flight work (``round_cut.CutSource``): none is carried.
        Every group a round pulled is kept for that round or dropped, so the
        driver holds no pulled group at the cut; the groups still queued in the
        rollouter are generated again after a resume, from the data cursor
        (:mod:`.fully_async_cut`)."""
        return {"entries": [], "buffer_groups": 0}

    def seek_data_cursor(self, cursor) -> dict[str, int]:
        self._cursor.started = self.island.rollouter_started
        landed = self._cursor.seek(cursor)
        self.island.emit("rl_verl_data_seek", data_cursor=landed, fully_async=True)
        return landed

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
                if self.queue_end_is_buffer_end:
                    from yeto.rl.engine.buffer_payload import SampleBufferExhausted

                    raise SampleBufferExhausted(
                        f"rollout {rollout_id}: the sample buffer ended with {len(collector.kept)}/"
                        f"{isl.required} trainable groups for this round")
                raise RuntimeError(f"rollout {rollout_id}: rollouter ended the queue")
            pulled += 1
            self._cursor.consume(1)
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
            carry_over=carry,
            backend="verl",
            # Same field and meaning as Miles: groups aborted in flight and NOT
            # trained.  verl's partial rollout continues interrupted requests
            # (counted as resumed_trajectories in carry_over, like Miles'
            # carried-over groups), so the only in-flight loss is yeto's
            # over-age discard; 0 when none (known, not None -- S19 async10
            # showed null on every round).
            **(discard if collector.discarded else {"aborted_in_flight_groups": 0}),
        )

    def abort(self) -> None:
        return None

    def members(self) -> frozenset[str]:
        return frozenset({MEMBER})


class FullyAsyncTrainerGroup(CutPorts):
    publish_offloaded = False

    def __init__(self, island: FullyAsyncIsland):
        self.island = island
        self._metrics = None
        self._cut_trainer = None

    # ---- cut verbs (S26 E9.2; .cut.CutPorts over the trainer actor's worker group) ----
    def _cut_calls(self):
        return self.island.call_workers

    def _optimizer_steps_per_round(self) -> int:
        if self.island.optimizer_steps_per_round is None:
            raise RuntimeError("this fully_async island was built without round cuts")
        return int(self.island.optimizer_steps_per_round)

    def onload(self) -> None:
        return None

    def offload(self) -> None:
        return None

    def train_step(self, batch: RolloutBatchHandle) -> LocalStepReceipt:
        from yeto.rl.engine.driver import TrainStepMetrics

        isl = self.island
        memory_mode = host_memory_mode()
        if memory_mode is not None:
            self._host_memory("before_step", batch.rollout_id, release=False)
        out = isl.call("yeto_train_step")
        if memory_mode is not None:
            self._host_memory("after_step", batch.rollout_id, release=memory_mode == "trim")
        metrics = out.get("metrics") or {}
        isl.last_metrics = metrics
        isl.last_criteria = out.get("criteria") or {}
        isl.append_jsonl(f"verl-metrics-{isl.learner_id}.jsonl",
                         {"rollout_id": batch.rollout_id, "param_version": out.get("param_version"),
                          "metrics": metrics})
        if isl.last_criteria:
            isl.append_jsonl(f"verl-criteria-{isl.learner_id}.jsonl", isl.last_criteria)
            isl.emit("rl_mismatch_criteria", **isl.last_criteria)
        if source := out.get("offpolicy_source"):
            source = {**source, "rollout_id": int(batch.rollout_id)}
            isl.append_jsonl(f"verl-offpolicy-source-{isl.learner_id}.jsonl", source)
            isl.emit(OFFPOLICY_SOURCE_EVENT, **source)

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
        if grad_norm is not None:
            self._check_first_step_after_resume(batch.rollout_id, isl.emit)
        ids = tuple(s for g in batch.groups for s in g.sample_ids)
        receipt = LocalStepReceipt(
            algorithm="grpo", learner_id=isl.learner_id, learner_generation=0,
            base_policy_version=batch.policy_version, base_policy_hash=batch.policy_hash,
            input_batch_hash=_sha("|".join(ids)), trajectory_ids=ids,
            trained_tokens=sum(g.token_count for g in batch.groups), optimizer_steps=1,
            optimizer_step_succeeded=grad_norm is not None,
            parameter_layout_hash=isl.layout_hash)
        if isl.full_group is not None:  # S27 G-EW: as ports_impl.VerlTrainerGroup.train_step
            from .full_parameter_stream import VerlRoundEvidence

            isl.full_group.record_round(receipt, VerlRoundEvidence(
                rollout_id=batch.rollout_id, behavior_policy_hash=batch.policy_hash,
                input_batch_hash=receipt.input_batch_hash, trained_tokens=receipt.trained_tokens,
                trajectory_ids=ids))
        return receipt

    def _host_memory(self, phase: str, rollout_id: int, *, release: bool) -> None:
        """Every worker's host memory around a step (``YETO_HOST_MEMORY``, S27
        G-EZ3; :func:`.full_parameter.remote_host_memory`): one jsonl record and
        one ``rl_host_memory`` event with the per-node summary. A failed read is
        recorded and training goes on."""
        from yeto.rl.engine.host_memory import process_memory, summarize_by_node

        from .full_parameter import remote_host_memory

        isl = self.island
        started = time.monotonic()
        try:
            summary = summarize_by_node(list(isl.call_workers(remote_host_memory, release)))
        except Exception as exc:  # noqa: BLE001 - a report, not a check
            summary = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        record = {"phase": phase, "rollout_id": int(rollout_id), "release": bool(release),
                  "driver": process_memory(), "seconds": round(time.monotonic() - started, 3), **summary}
        isl.append_jsonl(f"verl-host-memory-{isl.learner_id}.jsonl", record)
        isl.emit("rl_host_memory", phase=phase, rollout_id=int(rollout_id), release=bool(release),
                 nodes=record.get("nodes"), error=record.get("error"), seconds=record["seconds"])

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
                 timeout_s: float = 180.0, push_timeout_s: float = 600.0, sleep=time.sleep):
        self.island = island
        self.push_timeout_s = push_timeout_s
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
        sent, payload_bytes = sent_checksum(state)
        (isl.readback_dir / "expected_version").write_text(str(version))
        stale = isl.readback_dir / f"readback-v{version}.json"
        if stale.exists():
            stale.unlink()
        started = time.monotonic()
        try:  # S19 async5: a push that never returns must end the island, not hang it
            pushed = isl.call("yeto_push_weights", timeout=self.push_timeout_s)  # {"param_version", "timing"}
        except TimeoutError as exc:
            check = pub.compare(version, sent, None, detail=f"push timeout: {exc}")
            record = {**check.to_event(), "update_seconds": time.monotonic() - started}
            isl.append_jsonl(f"verl-publish-{isl.learner_id}.jsonl", record)
            isl.emit("rl_publication_check", **{k: v for k, v in record.items() if k != "event"})
            raise PublicationError(f"verl fully_async publication v{version}: {check.status} {check.detail}",
                                   cause=PublicationCause.LORA_UNVERIFIABLE) from exc
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
