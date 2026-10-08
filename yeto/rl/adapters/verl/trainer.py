"""verl v1 trainer subclass + the island loop (runs inside verl's task runner).

``yeto_sync`` = verl's ``PPOTrainerSync`` with the two publication hooks
removed (yeto publishes after the outer sync, through ``VerlPublisher``) and
the yeto mismatch criteria attached right after verl recomputes the old
log-probs (design D7: the criterion is computed by yeto in the Miles
convention; verl's ``rollout_corr/*`` are only archived).
"""

from __future__ import annotations

import json
import os
import time

from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync


@register_trainer("yeto_sync")
class YetoSyncTrainer(PPOTrainerSync):
    yeto_criteria_sink = None

    def on_init_end(self):  # the first publication is the driver's (policy version 0)
        return None

    def on_step_end(self):  # publication happens after the outer sync (driver)
        return None

    def _compute_old_log_prob(self, batch, metrics):
        batch = super()._compute_old_log_prob(batch, metrics)
        sink = self.yeto_criteria_sink
        if sink is not None:
            from .ports_impl import criteria_from_trainer_batch

            criteria_from_trainer_batch(sink, batch)
            for key in ("abs_diff", "k3", "tis_clipfrac", "signed_mean"):
                value = sink.last_criteria.get(key)
                if isinstance(value, float):
                    metrics[f"yeto_mismatch/{key}"] = value
        return batch


def _capabilities(fingerprint: str):
    from .entry import verl_capabilities

    return verl_capabilities(fingerprint)


def _fingerprint(cfg: dict) -> str:
    import hashlib

    from .pins import VERL_COMMIT

    text = json.dumps({"engine": "verl", "commit": VERL_COMMIT,
                       "versions": cfg.get("_versions", {})}, sort_keys=True)
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


class _TestExitPublisher:
    """TEST ONLY (V2 leave/rejoin): end this island's process right after it published
    ``version``.  Armed only on a first incarnation (its first publication is policy
    version 0); a relaunched island starts from the syncer's committed version > 0 and
    never exits again."""

    def __init__(self, inner, version: int | None, tape):
        self.inner, self.version, self.tape = inner, version, tape
        self.armed: bool | None = None if version is not None else False

    def publish(self, state):
        result = self.inner.publish(state)
        published = int(state.policy_version)
        if self.armed is None:
            self.armed = published == 0
            self.tape.append({"event": "rl_test_exit_armed", "armed": self.armed,
                              "first_published_version": published, "exit_after_version": self.version})
        if self.armed and published == self.version:
            self.tape.append({"event": "rl_test_island_exit", "policy_version": self.version,
                              "time_unix": time.time()})
            print(f"[yeto-verl] TEST exit after publishing v{self.version}", flush=True)
            os._exit(17)
        return result


def check_asserted(asserted: dict, cfg: dict) -> dict:
    """Initialisation assertions: each override yeto set must read back from verl's
    resolved config with the same value (string compare after normalisation)."""

    def lookup(path):
        node = cfg
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return "<missing>"
            node = node[part]
        return node

    def norm(value):
        text = str(value).strip().lower()
        try:
            return repr(float(text))
        except ValueError:
            return text

    report, bad = {}, []
    for key, want in sorted(asserted.items()):
        got = lookup(key)
        report[key] = {"want": want, "got": got}
        if want is not None and norm(got) != norm(want):
            bad.append(f"{key}: want {want!r} got {got!r}")
    if bad:
        raise RuntimeError("verl 初始化断言失败: " + "; ".join(bad))
    return report


def publish_selftest(island, publisher, policy_state, specs) -> dict:
    """rl-verl-backend 2.2 acceptance, before the first real publication:

    1. export -> apply -> export is bitwise equal;
    2. tamper: the trainer holds the exported tensors with one tensor shifted, the
       publisher claims the untampered state -> the read-back must report exactly
       that tensor (LORA_MISMATCH); the tensors are restored and the replicas put
       back to sleep (the driver's first publication expects them asleep).
    """
    import dataclasses

    import torch

    from yeto.rl.core import canonical_state
    from yeto.rl.engine.trainable_state import TrainableState

    t0 = time.time()
    first = policy_state.export()
    policy_state.apply(first, optimizer="preserve", local_step=0)
    again = policy_state.export()
    names = list(first.tensor_names)
    bitwise = all(torch.equal(first.tensors[n], again.tensors[n]) for n in names)
    target = names[len(names) // 2]
    tampered = {n: first.tensors[n].clone() for n in names}
    tampered[target] = tampered[target] + 0.01
    tamper_state = TrainableState.from_lora(canonical_state(
        0, tampered, base_model_revision=island.base_model_revision,
        lora_config_hash=island.lora_config_hash, layout_hash=island.layout_hash,
        expected_specs=island.expected_specs))
    policy_state.apply(tamper_state, optimizer="preserve", local_step=0)
    claimed = TrainableState.from_lora(dataclasses.replace(first.to_lora(), policy_version=999999))
    strict, publisher.strict = publisher.strict, False
    try:
        publisher.publish(claimed)
    finally:
        publisher.strict = strict
    record = {}
    path = island.out / f"verl-publish-{island.learner_id}.jsonl"
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        if rec.get("policy_version") == 999999:
            record = rec
    policy_state.apply(first, optimizer="preserve", local_step=0)
    island.trainer.checkpoint_manager.sleep_replicas()
    island.published = None
    detected = record.get("status") == "LORA_MISMATCH" and record.get("mismatched") == [target]
    return {"export_apply_export_bitwise": bitwise, "tampered_tensor": target,
            "tamper_status": record.get("status"), "tamper_mismatched": record.get("mismatched"),
            "tamper_detected": detected, "seconds": time.time() - t0}


def run_island(trainer, agent_loop_manager, plan: dict, cfg: dict) -> dict:
    from verl.utils.skip import SkipManager
    from verl.utils.tracking import Tracking

    from yeto.rl.core import canonical_layout_hash, canonical_lora_config_hash
    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.bridges import ElasticAvgSync, LocalOnlySync, StrictAvgSync
    from yeto.rl.engine.driver import EventTape, IslandDriver
    from yeto.rl.export import adapter_targets, derive_peft_lora_specs

    from .identity import backend_identity
    from .ports_impl import (VerlIsland, VerlPlacement, VerlPolicyState, VerlPublisher,
                             VerlRolloutPool, VerlTrainerGroup)

    t0 = time.time()
    trainer.agent_loop_manager = agent_loop_manager
    SkipManager.init(trainer.config)
    trainer.logger = Tracking(project_name=trainer.config.trainer.project_name,
                              experiment_name=trainer.config.trainer.experiment_name,
                              default_backend=trainer.config.trainer.logger, config=cfg)
    from verl.utils.tracking import DapoFilteredRewardTableLogger, ValidationGenerationsLogger

    names = dict(project_name=trainer.config.trainer.project_name,
                 experiment_name=trainer.config.trainer.experiment_name)
    trainer.validation_generations_logger = ValidationGenerationsLogger(**names)
    trainer.dapo_filtered_reward_logger = DapoFilteredRewardTableLogger(**names)
    trainer.global_steps = 1
    SkipManager.set_step(1)
    # attributes verl's fit() sets before its loop (profiling off on the yeto path)
    trainer.prev_step_profile = trainer.curr_step_profile = trainer.next_step_profile = False
    trainer.timing_raw = {}
    trainer._reissue_inflight_prompts()
    trainer.on_train_begin()

    learner_id = int(plan["learner_id"])
    asserted = check_asserted(plan.get("asserted") or {}, cfg)
    specs =derive_peft_lora_specs(plan["model_path"], None, rank=int(plan["lora_rank"]),
                                   targets=plan["lora_targets"])
    layout_hash = canonical_layout_hash(specs)
    lora_hash = canonical_lora_config_hash(rank=int(plan["lora_rank"]), target_modules=adapter_targets(specs))
    tape = EventTape(plan["event_tape"], learner_id)
    island = VerlIsland(trainer, learner_id=learner_id, out_dir=plan["out_dir"],
                        base_model_revision=plan["model_revision"], lora_config_hash=lora_hash,
                        layout_hash=layout_hash, expected_specs=specs,
                        tis_upper=float(plan.get("tis_upper", 2.0)),
                        thresholds_key=plan["thresholds_key"],
                        emit=lambda event, **f: tape.append({"event": event, **f}))
    identity = backend_identity()
    tape.append({"event": "rl_verl_island_start", "plan": plan, "backend_identity": identity.to_dict(),
                 "backend_identity_sha256": identity.sha256(), "layout_hash": layout_hash,
                 "lora_config_hash": lora_hash, "n_specs": len(specs), "asserted": asserted,
                 "init_seconds": time.time() - t0})
    mode = plan["sync"]
    rounds = int(plan["global_rounds"])
    if mode == "none":
        sync = LocalOnlySync(rounds)
    else:
        from yeto.rl.bridge import BridgeConfig

        host, _, port = plan["syncer"].rpartition(":")
        bridge = BridgeConfig(
            syncer_addr=(host, int(port)), learner_id=learner_id, global_rounds=rounds,
            groups_per_round=int(plan["groups_per_round"]),
            samples_per_group=int(plan["samples_per_group"]), local_optimizer_steps=1,
            expected_specs=tuple(specs), base_model_revision=plan["model_revision"],
            lora_config_hash=lora_hash, layout_hash=layout_hash, event_tape=plan["event_tape"],
            wan_streams=int(plan.get("wan_streams", 4)),
            backend_identity_sha256=identity.sha256())
        if mode == "strict":
            sync = StrictAvgSync(bridge, progress=None)
        elif mode == "elastic":
            sync = ElasticAvgSync(bridge, progress=None, syncer_epoch=int(plan.get("syncer_epoch", 0)))
        else:
            raise ValueError(f"unknown sync mode {mode!r}")
    spec = (AlgorithmSpec.from_dict(plan["algorithm_spec"]) if plan.get("algorithm_spec")
            else AlgorithmSpec())
    publisher = VerlPublisher(island, strict=bool(plan.get("publish_strict", True)))
    driver = IslandDriver(
        learner_id=learner_id, rollout=VerlRolloutPool(island), trainer=VerlTrainerGroup(island),
        policy_state=VerlPolicyState(island),
        publisher=_TestExitPublisher(publisher, plan.get("test_exit_after_version"), tape),
        placement=VerlPlacement(), capabilities=_capabilities(_fingerprint(cfg)), algorithm=spec,
        sync=sync, events=tape)
    if plan.get("publish_selftest"):
        tape.append({"event": "rl_verl_publish_selftest",
                     **publish_selftest(island, publisher, VerlPolicyState(island), specs)})
    state = driver.run()
    result = {"learner_id": learner_id, "rounds_completed": driver.rounds_completed,
              "final_policy_version": state.policy_version,
              "final_policy_tensor_hash": state.policy_tensor_hash(),
              "seconds": time.time() - t0}
    tape.append({"event": "rl_verl_island_done", **result})
    tape.append({"event": "rl_learner_finalized"})  # launcher: a complete island tape ends with this
    return result
