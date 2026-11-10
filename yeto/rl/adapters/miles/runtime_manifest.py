"""Miles runtime-manifest declarations (yeto-framework-decoupling 2.5). Import-light.

The core :mod:`yeto.rl.engine.runtime_manifest` collects and checks a
manifest from a :class:`~yeto.rl.engine.runtime_manifest.RuntimeDescription`;
this module is the Miles one: the upstream / fork / Megatron interfaces it
probes, which capability needs which interface, the version modules, the
Miles/SGLang checkouts, the image build record and commit pins. (The critic
overlay record went with the overlay: the k31 lineage patches nothing after
the image build.)
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from yeto.rl.engine.runtime_manifest import RuntimeDescription

# interface name -> (module, attribute path). Probed with getattr only.
# Groups: the upstream-main surface the adapter calls (every one exists on the
# k31 upstream base 778794a8), the F1 fork additions (k31/f1; TODO(F1): align
# with test_k31_baseline_surface once it lands), the later k31 batches (B2,
# B3-B7) and the elastic / membership verbs (B4: on k31/b4+, the capabilities
# needing them are refused on an image without them).
INTERFACES: Mapping[str, tuple[str, str]] = {
    # upstream main
    "orchestration_startup_info": ("miles.utils.orchestration_utils", "ArgvOrchestratorStartupInfo"),
    "trainer_handles": ("miles.ray.placement_group", "create_trainer_handles"),
    "rollout_components": ("miles.ray.placement_group", "create_rollout_components"),
    "train_parallel_config": ("miles.utils.dp_schedule", "TrainParallelConfig"),
    "checksum_merge_ranks": ("miles.utils.audit_utils.checksum_utils", "merge_inference_engine_ranks"),
    "adapter_sampler_export": ("miles.backends.training_utils.checkpoint.adapter.adapter_dir", "write_sampler_export"),
    "adapter_peft_export": ("miles.backends.training_utils.checkpoint.adapter.adapter_dir", "write_peft_export"),
    "checkpoint_dir_write": ("miles.backends.training_utils.checkpoint.io", "write_checkpoint_dir"),
    "loss_hub_policy_loss": ("miles.backends.training_utils.loss.hub.losses", "policy_loss_function"),
    "loss_hub_icepop": ("miles.backends.training_utils.loss.hub.corrections", "icepop_function"),
    "context_parallel_gather": ("miles.backends.training_utils.data.context_parallel", "all_gather_with_cp"),
    "replay_register_sequential": ("miles.backends.training_utils.replay.data", "register_replay_list_sequential"),
    "model_companion_weight_version": (
        "miles.backends.training_utils.model_companion",
        "ModelCompanionWeightVersionUtils.bump_weight_version",
    ),
    "named_params_and_buffers": ("miles.backends.megatron_utils.named_weights", "named_params_and_buffers"),
    "router_choice": ("miles.utils.args.configs.cluster", "ClusterConfig"),
    # k31/f1 fork additions
    "run_plugin": ("miles.ray.train.group", "TrainerController.run_plugin"),
    "fork_b0_yeto_extension": ("miles.utils.args.configs.yeto_extension", "YetoExtensionConfig"),
    "rollout_runtime_stats": ("miles.rollout.rollout_runtime_stats", "publish"),
    # k31/b2 fork additions (B2: the full-parameter transport and the per-rank
    # state helpers; pins.K31_FEATURES "full_parameter_transport")
    "fork_b2_full_parameter_transport": ("miles.ray.full_parameter_transport", "apply_chunked_cut"),
    "fork_b2_full_parameter_controller": (
        "miles.ray.train.full_parameter_controller", "FullParameterTransportMixin.apply_full_parameter_chunked_cut"),
    "fork_b2_full_parameter_rank_state": (
        "miles.backends.megatron_utils.full_parameter_state", "full_parameter_rank_state"),
    "fork_b2_set_model_weight_version": (
        "miles.backends.megatron_utils.full_parameter_state", "set_model_weight_version"),
    # the ports-v1 full-parameter commits after the transport that B2 brought
    # along (one probe per pins.K31_FEATURES entry; registered on k31/b2):
    # distopt_full_parameter (fork 56190a5a8): the DP=1 distributed-optimizer
    # master writer; dp_replica_full_parameter (fork 921cde8c7): the replica
    # plan of a data-parallel replica rank; dp_sharded_master (fork d50fb04dd):
    # each rank's FP32 master ranges under the distributed optimizer;
    # verified_apply (fork 102672760): the ahead-of-apply chunk check;
    # shape_free_layout (fork 8d03f386a): the owner-map install verb. No
    # capability needs them: the DP>1 / offload rules read the pins features
    # (run_config_rules.full_parameter_dp_*_gap), and verified_apply /
    # shape_free_layout are negotiated per group through
    # full_parameter_transport.FULL_PARAMETER_TRANSPORT_FEATURES
    "fork_b2_distopt_full_parameter": (
        "miles.backends.megatron_utils.full_parameter_state", "_propagate_master_writes"),
    "fork_b2_dp_replicas": ("miles.backends.megatron_utils.full_parameter_replicas", "install_replica_plan"),
    "fork_b2_dp_sharded_masters": ("miles.backends.megatron_utils.full_parameter_sharded_masters", "master_ranges"),
    "fork_b2_verified_apply": (
        "miles.backends.megatron_utils.full_parameter_state", "verify_stored_fragment_chunks"),
    "fork_b2_shape_free_layout": (
        "miles.ray.train.full_parameter_controller", "FullParameterTransportMixin.install_full_parameter_owner_maps"),
    # k31/b3-b7 fork additions (one probe per pins.K31_FEATURES entry of that batch)
    # B3 engine_failure_isolation: the join failure class (an ExternalFailureError)
    "fork_b3_engine_failure_isolation": (
        "miles.backends.training_utils.weight_update.protocols.broadcast", "RolloutEngineJoinError"),
    # B5 abort_tally: the engine abort helper the tallying abort() calls (the
    # rollout_runtime_stats module itself is an F1 addition, probed above);
    # agentic_suspend: the suspend() the rollout runs at the cut-off. The typed
    # fields (RolloutConfig.agentic_suspend_*, DebugConfig.host_sync_free) are
    # pydantic model fields, not class attributes, so they have no getattr probe.
    "fork_b5_abort_tally": ("miles.rollout.inference_rollout.inference_rollout_train", "_abort_engines"),
    "fork_b5_agentic_suspend": ("miles.rollout.inference_rollout.inference_rollout_train", "suspend"),
    # B6 session_responses_api: the Responses -> chat translation; lora_per_token_loss
    # has no attribute of its own (one assignment inside the LoRA bridge setup), so
    # the probe names the function that carries it
    "fork_b6_session_responses_api": ("miles.rollout.session.responses_adapter", "to_chat_request"),
    "fork_b6_lora_per_token_loss": ("miles.backends.megatron_utils.lora.bridge", "_setup_lora_model_via_bridge"),
    # B7 bridge_moe_runtime_flags, host_sync_free_flag, nemotron_h_vl_bridge, replay_hybrid_layers
    "fork_b7_bridge_moe_runtime_flags": ("miles.utils.megatron_bridge_utils", "megatron_cli_defaults"),
    "fork_b7_bridge_moe_runtime_overrides": ("miles.utils.megatron_bridge_utils", "bridge_moe_runtime_overrides"),
    # host_sync_free_flag: sync_free.configure() is what the train actor calls with
    # the typed DebugConfig.host_sync_free (added by the same commit as the flag)
    "fork_b7_host_sync_free_flag": ("miles.utils.sync_free", "configure"),
    # the language-only VL bridge class is registered inside the plugin's install
    # function; the module constant names the architecture it serves
    "fork_b7_nemotron_h_vl_bridge": ("miles_plugins.megatron_bridge.nemotron_h", "NEMOTRON_H_OMNI_ARCHITECTURE"),
    "fork_b7_replay_hybrid_layers": ("miles.backends.megatron_utils.replay_utils", "_layer_ids_from_routers"),
    # Megatron private surface the vendored dp_invariant_state reads
    # (adapters/miles/dp_invariant_state.py; HybridDeviceOptimizer internals)
    "megatron_hdo": ("megatron.core.optimizer.cpu_offloading", "HybridDeviceOptimizer"),
    "megatron_hdo_sync_state": (
        "megatron.core.optimizer.cpu_offloading", "HybridDeviceOptimizer._sync_hdo_state_to_sub_optimizers"),
    "megatron_hdo_dummy_states": (
        "megatron.core.optimizer.cpu_offloading", "HybridDeviceOptimizer._init_optimizer_states_with_dummy_values"),
    "megatron_distopt_main_param_state": (
        "megatron.core.optimizer.distrib_optimizer", "DistributedOptimizer._get_main_param_and_optimizer_states"),
    "megatron_distopt_set_main_param_state": (
        "megatron.core.optimizer.distrib_optimizer", "DistributedOptimizer._set_main_param_and_optimizer_states"),
    "megatron_cuda_rng_tracker": ("megatron.core.tensor_parallel", "get_cuda_rng_tracker"),
    # elastic / membership verbs (k31/b4 batch, ported from ports-v1 M1-M6; one
    # probe per pins.K31_FEATURES entry of that batch). placement_map: the parser
    # lives in miles.ray.placement_map, re-exported by placement_group;
    # membership_epoch: InferenceController inherits start_cells from
    # InferenceMembershipMixin; router_cordon; member_publish: the member branch
    # of start_update_weights that update_member_weights drives; trainer_rebuild:
    # its own module (placement_group does not re-export it, an import cycle)
    "fork_m1_placement_map": ("miles.ray.placement_group", "parse_placement_map"),
    "fork_m2_start_stop_cells": ("miles.ray.rollout.inference_controller", "InferenceController.start_cells"),
    "fork_m3_router_cordon": ("miles.router.router", "MilesRouter.cordon_worker"),
    "fork_m4_member_publish": (
        "miles.ray.rollout.inference_controller",
        "InferenceController._start_member_update_weights",
    ),
    "fork_m6_rebuild_trainer": ("miles.ray.trainer_rebuild", "rebuild_training_models"),
    # pins.K31_FEATURES "workers_lost" (Miles k31/b4-lost efc83590, on k31/b8): the
    # cell state property that makes describe_cells report "workers_lost" with
    # "lost_workers". No capability needs it: without it the watchdog journals a
    # cell whose engine died as no_workers (elastic_wiring._lost_workers_of)
    "fork_b4_workers_lost": ("miles.utils.workers.ray_worker_manager", "_CellManager.state"),
}

# The upstream-main / F1 / Megatron interfaces every ports island needs.
PORTS_BASE_INTERFACES: tuple[str, ...] = (
    "run_plugin", "orchestration_startup_info", "trainer_handles", "rollout_components",
    "train_parallel_config", "checksum_merge_ranks", "model_companion_weight_version",
)

# Declared capability -> interfaces it needs. Missing any => refused.
CAPABILITY_REQUIREMENTS: Mapping[str, tuple[str, ...]] = {
    "ports-colocated-serial": PORTS_BASE_INTERFACES,
    "ports-partitioned-serial": PORTS_BASE_INTERFACES,
    "fixed-partition-standby": (*PORTS_BASE_INTERFACES, "fork_m1_placement_map"),
    "Publisher.publish(lora)": ("adapter_sampler_export", "adapter_peft_export", "checkpoint_dir_write"),
    "Cut.dp_invariant_state": ("megatron_hdo", "megatron_hdo_sync_state", "megatron_hdo_dummy_states",
                               "megatron_distopt_main_param_state", "megatron_distopt_set_main_param_state",
                               "megatron_cuda_rng_tracker"),
    "RolloutPool.add_engines": ("fork_m2_start_stop_cells", "fork_m3_router_cordon"),
    "RolloutPool.remove_engines": ("fork_m2_start_stop_cells", "fork_m3_router_cordon"),
    "RolloutPool.drain": ("fork_m3_router_cordon",),
    "Publisher.publish(members)": ("fork_m4_member_publish",),
    "Placement.reconfigure": ("fork_m1_placement_map", "fork_m6_rebuild_trainer"),
    # synced full-parameter trainer islands (K3.1 --tuning full behind the syncer)
    "Cut.full_parameter_transport": ("fork_b2_full_parameter_transport", "fork_b2_full_parameter_controller",
                                     "fork_b2_full_parameter_rank_state", "fork_b2_set_model_weight_version",
                                     "model_companion_weight_version"),
}

VERSION_MODULES = ("torch", "megatron.core", "torch_memory_saver", "peft", "sglang", "ray")
REQUIRED_VERSIONS = ("torch", "megatron.core", "cuda", "nccl", "torch_memory_saver", "peft")


def image_manifest() -> dict[str, Any]:
    try:
        from yeto.rl import MILES_NEXT_IMAGE_MANIFEST

        return json.loads(Path(MILES_NEXT_IMAGE_MANIFEST).read_text())
    except Exception:
        return {}


def expected_pins() -> dict[str, str]:
    from yeto.rl import MILES_NEXT_COMMIT, MILES_NEXT_IMAGE, SGLANG_NEXT_COMMIT

    return {"miles": MILES_NEXT_COMMIT, "sglang": SGLANG_NEXT_COMMIT, "image": MILES_NEXT_IMAGE}


MILES_RUNTIME = RuntimeDescription(
    interfaces=INTERFACES,
    capability_requirements=CAPABILITY_REQUIREMENTS,
    version_modules=VERSION_MODULES,
    required_versions=REQUIRED_VERSIONS,
    checkouts=("miles", "sglang"),
    nested_checkouts=("sglang",),  # sglang/python/sglang layout
    image_manifest=image_manifest,
    expected_pins=expected_pins,
)
