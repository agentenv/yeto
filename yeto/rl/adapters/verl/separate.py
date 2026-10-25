"""verl separate mode: trainer cards apart from the rollout cards (S27 G-DC, G-EK).

One verl island, two pools on its cards (``--rl-placement fixed-partition``):

* trainer pool: ``trainer_nodes`` x ``trainer_gpus_per_node`` cards of the
  training engine (FSDP2, or Megatron for TP/PP/EP > 1, ``megatron_engine``);
* rollout pool: ``rollout_nodes`` x ``rollout_gpus_per_node`` cards of
  standalone vLLM servers, ``rollout_tp`` cards each;
* trainer mode ``yeto_separate`` (:mod:`.separate_trainer`): verl's v1
  ``separate_async`` trainer with ``hybrid_engine=False``,
  ``parameter_sync_step=1`` and no warm-up batch; the yeto driver publishes
  policy r and only then submits round r (age 0, partitioned-serial);
* weights reach the rollout servers through verl's NCCL checkpoint engine
  (``standalone_checkpoint_manager.update_weights``): :class:`SeparatePublisher`.

Rollout ahead (S27 G-EV, VERL-PARITY #10; ``--rl-rollout-ahead 1``,
``yeto.rl.engine.rollout_ahead``): the driver submits round r+1 to the rollout
servers with the published policy r before the trainer trains round r, and
waits for that batch before it publishes r+1 (``VerlRolloutPool.start_generate``
/ ``finish_generate``). Every sample then lags one version; the profile carries
that age under the bounded-staleness contract (:func:`policy_age_of`), and the
trainer's TIS on the rollout log-probs corrects it. Off (the default), the
profile, the capabilities and the driver are those of the serial island.

Everything here is independent of the training engine. G-DC wrote it for the
Megatron island (s27-verl-megatron); G-EK moved the engine-neutral part here so
the FSDP2 full-parameter island runs the same mode (VERL-PARITY #13). The
Megatron-only keys stay in ``megatron_engine`` and ``megatron_ports``.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from typing import Any

TRAINER_MODE = "yeto_separate"
WEIGHT_TRANSPORT = "verl-nccl-checkpoint-engine"
READBACK_STATUS = "ENGINE_READBACK_NOT_IMPLEMENTED"
FIXED_PARTITION = "fixed-partition"


class SeparatePoolError(ValueError):
    """The island's cards do not split into a trainer pool and a rollout pool."""


@dataclass(frozen=True)
class SeparatePools:
    trainer_nodes: int = 1
    trainer_gpus_per_node: int = 8
    rollout_nodes: int = 1
    rollout_gpus_per_node: int = 8
    rollout_tp: int = 2
    gpu_memory_utilization: float = 0.8

    @property
    def trainer_gpus(self) -> int:
        return self.trainer_nodes * self.trainer_gpus_per_node

    @property
    def rollout_gpus(self) -> int:
        return self.rollout_nodes * self.rollout_gpus_per_node

    def to_dict(self) -> dict:
        out = asdict(self)
        out.update(trainer_gpus=self.trainer_gpus, rollout_gpus=self.rollout_gpus)
        return out


def validate_pools(pools: SeparatePools, error: type[ValueError] = SeparatePoolError) -> None:
    """Raise ``error`` naming the first violated pool rule."""
    for name in ("trainer_nodes", "trainer_gpus_per_node", "rollout_nodes", "rollout_gpus_per_node", "rollout_tp"):
        if int(getattr(pools, name)) < 1:
            raise error(f"{name} 必须 >= 1，得到 {getattr(pools, name)}")
    if pools.rollout_tp > pools.rollout_gpus_per_node or pools.rollout_gpus_per_node % pools.rollout_tp:
        raise error(f"rollout TP={pools.rollout_tp} 必须整除每个 rollout 节点的卡数 {pools.rollout_gpus_per_node}")
    if not 0.0 < float(pools.gpu_memory_utilization) < 1.0:
        raise error(f"gpu_memory_utilization 必须在 (0, 1)，得到 {pools.gpu_memory_utilization}")


def pools_from_args(args: Any, total_gpus: int, *, what: str = "verl 分离岛") -> SeparatePools:
    """The pools from the island flags the launcher computed (``launch.topology.
    rl_trainer_shape``: ``--actor-num-nodes``, ``--actor-num-gpus-per-node``,
    ``--rollout-num-gpus``, ``--rollout-num-gpus-per-engine``). The two pools must
    use every card the island-local Ray sees. The launcher's engine memory flag is
    spelled for SGLang (``--sglang-mem-fraction-static``); here it is vLLM's."""
    per_node = int(getattr(args, "rl_island_gpus_per_node", None) or getattr(args, "actor_num_gpus_per_node", None)
                   or total_gpus)
    rollout_gpus = int(getattr(args, "rollout_num_gpus", None) or 0)
    trainer_gpus_per_node = int(getattr(args, "actor_num_gpus_per_node", None) or per_node)
    trainer_gpus = int(getattr(args, "actor_num_nodes", 1) or 1) * trainer_gpus_per_node
    if rollout_gpus <= 0 or rollout_gpus % min(per_node, rollout_gpus):
        raise SystemExit(f"{what}：--rollout-num-gpus {rollout_gpus} 不能按每节点 {per_node} 卡排布")
    if trainer_gpus + rollout_gpus != total_gpus:
        raise SystemExit(f"{what}：训练 {trainer_gpus} + rollout {rollout_gpus} 卡 != 岛内 Ray 的 {total_gpus} 卡")
    rollout_per_node = min(per_node, rollout_gpus)
    return SeparatePools(
        trainer_nodes=int(getattr(args, "actor_num_nodes", 1) or 1), trainer_gpus_per_node=trainer_gpus_per_node,
        rollout_nodes=rollout_gpus // rollout_per_node, rollout_gpus_per_node=rollout_per_node,
        rollout_tp=int(getattr(args, "rollout_num_gpus_per_engine", None) or 1),
        gpu_memory_utilization=float(getattr(args, "sglang_mem_fraction_static", None) or 0.8))


# Keys of ``config.build_overrides`` (the colocated FSDP2 island) that the separate
# mode replaces: the trainer mode and the shape of the two pools.
POOL_PREFIXES = (
    "trainer.v1.trainer_mode=",
    "trainer.n_gpus_per_node=",
    "trainer.nnodes=",
    "actor_rollout_ref.rollout.tensor_model_parallel_size=",
    "actor_rollout_ref.rollout.gpu_memory_utilization=",
)


def separate_overrides(pools: SeparatePools) -> list[str]:
    """Hydra overrides of the separate mode (fork acad9875 ``ppo_trainer.yaml``):
    the trainer mode, the separate_async settings and both pools."""
    validate_pools(pools)
    return [
        f"trainer.v1.trainer_mode={TRAINER_MODE}",
        "actor_rollout_ref.hybrid_engine=False",
        "trainer.v1.separate_async.parameter_sync_step=1",
        "trainer.v1.separate_async.num_warmup_batches=0",
        "trainer.v1.separate_async.hybrid_rollout.enable_switch=False",
        f"trainer.n_gpus_per_node={pools.trainer_gpus_per_node}",
        f"trainer.nnodes={pools.trainer_nodes}",
        # rollout: standalone vLLM servers on their own cards, NCCL weight sync
        f"actor_rollout_ref.rollout.nnodes={pools.rollout_nodes}",
        f"actor_rollout_ref.rollout.n_gpus_per_node={pools.rollout_gpus_per_node}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={pools.rollout_tp}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={pools.gpu_memory_utilization}",
        "actor_rollout_ref.rollout.checkpoint_engine.backend=nccl",
    ]


def with_separate_pools(overrides: list[str], pools: SeparatePools) -> list[str]:
    """``overrides`` of a colocated island turned into the separate mode."""
    return [o for o in overrides if not o.startswith(POOL_PREFIXES)] + separate_overrides(pools)


# The island re-reads these after Hydra composition and asserts the values
# (``trainer.check_asserted``).
ASSERTED_KEYS = (
    "actor_rollout_ref.hybrid_engine",
    "trainer.v1.trainer_mode",
    "trainer.v1.separate_async.parameter_sync_step",
    "trainer.v1.separate_async.num_warmup_batches",
    "actor_rollout_ref.rollout.checkpoint_engine.backend",
)


# ----------------------------------------------------------------------------- ports (trainer process)
class SeparatePublisher:
    """Publication over verl's standalone NCCL checkpoint engine.

    There is no read-back of what vLLM loaded (the LoRA read-back of
    ``vllm_readback`` does not cover full weights); the train/rollout log-prob
    criteria of every round are the check that the rollout runs the published
    weights. ``state`` is a ``TrainerResidentState`` (``policy_digest``)."""

    def __init__(self, island: Any):
        self.island = island

    def publish(self, state):
        from yeto.rl.contracts import InferencePublicationManifest
        from yeto.rl.engine.ports import PublicationResult

        from .ports_impl import MEMBER

        isl, t = self.island, self.island.trainer
        version = int(state.policy_version)
        check = getattr(state, "check_holds", None)
        if callable(check):
            check()  # the trainer still holds the hashed weights
        started = time.monotonic()
        sync_metrics = dict(t.standalone_checkpoint_manager.update_weights(version) or {})
        update_seconds = time.monotonic() - started
        digest = state.policy_tensor_hash()
        payload_hash, payload_bytes = state.payload_digest
        record = {"policy_version": version, "status": READBACK_STATUS, "update_seconds": update_seconds,
                  "policy_tensor_hash": digest, "payload_bytes": int(payload_bytes),
                  "transport": WEIGHT_TRANSPORT,
                  "sync_metrics": {k: v for k, v in sync_metrics.items() if isinstance(v, (int, float, str))}}
        isl.append_jsonl(f"verl-publish-{isl.learner_id}.jsonl", record)
        isl.emit("rl_publication_check", **record)
        isl.published = (version, digest)
        manifest = InferencePublicationManifest(
            publication_mode="full", base_policy_version=None, target_policy_version=version,
            target_policy_hash=digest,
            target_manifest_hash=hashlib.sha256(f"{version}:{digest}".encode()).hexdigest(),
            payload_hash=payload_hash, payload_bytes=int(payload_bytes), complete=True)
        return PublicationResult(manifest, frozenset({MEMBER}))


class SeparatePlacement:
    """The fixed-partition placement of the island's two pools."""

    def __init__(self, shape: dict):
        self.shape = dict(shape)

    def describe(self):
        from yeto.rl.engine.ports import PlacementDescription

        trainer = tuple(f"trainer:{i}" for i in range(int(self.shape["trainer_gpus"])))
        rollout = tuple(f"rollout:{i}" for i in range(int(self.shape["rollout_gpus"])))
        return PlacementDescription(FIXED_PARTITION, trainer, rollout,
                                    extra={"weight_transport": WEIGHT_TRANSPORT, "shape": self.shape})


def policy_age_of(plan: dict) -> int:
    """The policy age a separate island's samples may have: the run's
    ``--rl-max-policy-age`` with rollout ahead (every sample lags one version, the
    launcher checked the limit is at least that), else 0 (serial rounds)."""
    return int(plan.get("max_policy_age") or 0) if int(plan.get("rollout_ahead") or 0) else 0


def execution_profile(spec, plan: dict, *, name: str, engine: str, outer_protocol: str = "none"):
    """The partitioned-serial profile of a separate island: on-policy and age 0, or
    with rollout ahead the bounded-staleness contract at :func:`policy_age_of`."""
    from yeto.rl.engine.execution_profile import ExecutionProfile
    from yeto.rl.engine.policy_age import BOUNDED_STALENESS_CONTRACT

    age = policy_age_of(plan)
    return ExecutionProfile(
        name=name, execution_mode="partitioned-serial", outer_protocol=outer_protocol,
        algorithm_contract=BOUNDED_STALENESS_CONTRACT if age else "on-policy", max_policy_age=age,
        groups_per_batch=int(plan["groups_per_round"]), samples_per_group=int(plan["samples_per_group"]),
        algorithm_spec_sha256=spec.sha256(), extra={"engine": engine})


def with_policy_age(capabilities, plan: dict):
    """``capabilities`` declaring the policy age of :func:`policy_age_of` (unchanged at 0)."""
    import dataclasses

    age = policy_age_of(plan)
    if not age:
        return capabilities
    return dataclasses.replace(capabilities, execution=dataclasses.replace(
        capabilities.execution, max_policy_staleness=age))


def wire_rollout_ahead(driver: Any, plan: dict) -> None:
    """The driver of a separate island generates ``plan["rollout_ahead"]`` batch ahead
    (0: serial rounds, nothing set). The driver checks its ports when it starts
    (``rollout_ahead.driver_problems``)."""
    ahead = int(plan.get("rollout_ahead") or 0)
    if ahead and not plan.get("separate") and not plan.get("megatron"):
        raise ValueError("rollout ahead needs verl's separate mode (trainer and rollout cards apart)")
    if ahead:
        driver.rollout_ahead = ahead


# ----------------------------------------------------------------------------- the FSDP2 island (S27 G-EK)
def full_parameter_publisher(island: Any, plan: dict, colocated_cls):
    """The publisher of a full-parameter FSDP2 island: :class:`SeparatePublisher` with
    ``plan["separate"]``, else ``colocated_cls`` (``ports_impl.VerlFullParameterPublisher``:
    verl's colocated ``update_weights``)."""
    return SeparatePublisher(island) if plan.get("separate") else colocated_cls(island)


def driver_options(plan: dict, spec, capabilities) -> dict:
    """``IslandDriver`` keywords of a full-parameter FSDP2 island in the separate mode:
    the fixed-partition placement, the partitioned-serial profile and the matching
    capabilities (as the Megatron island's); empty for a colocated island, whose
    driver keeps its defaults."""
    if not plan.get("separate"):
        return {}
    from .entry import with_partitioned_serial

    return {"placement": SeparatePlacement(plan["separate"]),
            "capabilities": with_policy_age(with_partitioned_serial(capabilities), plan),
            "profile": execution_profile(spec, plan, name="verl-fsdp2-separate", engine="verl-fsdp2",
                                         outer_protocol="none" if plan.get("sync") == "none" else "strict-avg")}
