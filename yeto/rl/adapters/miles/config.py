"""RLRunConfig + AlgorithmSpec -> upstream Miles argv .

Rules enforced here, all *before* any GPU process exists:

* Every leaf of the engine-agnostic :class:`~yeto.rl.engine.run_config.RLRunConfig`
  must be explicitly handled by this translation. A leaf this module does not
  know (e.g. a field added to ``run_config`` later) raises
  :class:`UnmappedConfigError` naming the field; a known leaf whose *value* has
  no upstream equivalent (fork-only ports, DSV4 recipe, full-parameter mode,
  ...) is also rejected with its name. Nothing is silently dropped.
* Miles fault-tolerance semantics (indep-DP, healing, api server, mini FT
  controller, ...) are rejected, both in the generated/extra argv and on the
  parsed namespace (:func:`validate_parsed_args`), which also asserts that the
  actor ``TrainGroup`` will have exactly one cell.
* The requested placement is recorded so :mod:`.placement` can detect Miles'
  argument normalization rewriting it.

Pure Python; miles is imported lazily only by :func:`parse_miles_args`,
:func:`validate_parsed_args` (optional cell count) and the TITO parser lookup.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
from pathlib import Path
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from yeto.rl.adapters.miles.pins import (
    MILES_DISTOPT_FULL_PARAMETER_COMMIT,
    MILES_NEXT_COMMIT,
)
from yeto.rl.engine.algorithm import (
    AlgorithmSpecError,
    BOUNDED_NONZERO_STD_FILTER,
    STOCK_NONZERO_STD_FILTER,
    normalize_filter_name,
    AlgorithmSpec,
)
from yeto.rl.quant.tiers import FP8_SOURCE_ENGINE, TIER_ONLINE_FP8, sglang_online_argv
from .lr_schedule import LR_SCHEDULE_FLAGS
from .algorithm_flags import (
    OBJECTIVE_FLAGS,
    absorb_extra_argv,
    algorithm_argv,
    mapped_flags,
    objective_flags,
)
from .placement import PlacementRequest, check_placement_not_rewritten
from .run_config_rules import full_parameter_dp_replica_gap, full_parameter_dp_sharded_master_gap
from . import sampler_snapshot
from .binding import filter_argv

ROLLOUT_META_HOOK_PATH = (
    "yeto.rl.adapters.miles.rollout_meta_hook.extract_rollout_metadata"
)
TRAINED_GROUPS_HOOK_PATH = (
    "yeto.rl.adapters.miles.rollout_meta_hook.record_trained_groups"
)
POLICY_BUFFER_FILTER_PATH = (
    "yeto.rl.adapters.miles.rollout_meta_hook.policy_buffer_filter"
)
QWEN3_5_SPEC = ("miles_plugins.models.qwen3_5", "get_qwen3_5_spec")

# Miles fault-tolerance / independent-DP surface (spec: MUST NOT depend on it).
FAULT_TOLERANCE_FLAGS = frozenset(
    {
        "--indep-dp",
        "--delay-split-train-data-by-dp",
        "--use-fault-tolerance",
        "--ft-components",
        "--api-server-port",
        "--api-server-host",
        "--mini-ft-controller-enable",
        "--mini-ft-controller-poll-interval",
        "--mini-ft-controller-resume-delay",
        "--init-expected-num-cells",
        "--ci-ft-test-actions",
        "--ci-ft-test-actions-path",
        "--enable-witness",
        "--witness-buffer-size",
        "--save-local-weight-checksum",
    }
)
# Session server timeout for its upstream model calls (Miles --miles-router-timeout,
# default None: no limit). Without a limit a call to a dead engine hangs until the
# router's own limit (14400 s) instead of failing (see the router's timeout). 600 s is
# above one turn's longest generation (4096 tokens at about 70 tokens/s per sequence
# under a large batch is about 60 s, plus queueing) and below Codex's stream idle
# timeout (1800 s, codex_provider), so the session server answers 502 first and the
# caller retries. Extra argv may set its own value.
SESSION_UPSTREAM_TIMEOUT_S = 600

# Flags whose value this adapter owns; an override via extra argv would bypass
# placement/algorithm/driver invariants.
ADAPTER_OWNED_FLAGS = frozenset(
    {
        "--colocate",
        "--rollout-num-gpus",
        "--actor-num-nodes",
        "--actor-num-gpus-per-node",
        "--rollout-num-gpus-per-engine",
        "--advantage-estimator",
        "--kl-coef",
        "--dynamic-sampling-filter-path",
        "--rollout-all-samples-process-path",
        "--rollout-sample-filter-path",
        "--rollout-function-path",
        "--data-source-path",
        "--buffer-filter-path",
        "--fully-async",
        "--debug-train-only",
        "--debug-rollout-only",
        "--load-debug-rollout-data",
        "--rollout-external-engine-addrs",
        "--custom-inference-engine-provider-path",
        "--deploy-component",
        "--trainer-controller-addrs",
        "--eval-num-gpus",
        "--external-policy-sync-path",  # legacy fork only; the driver owns sync
        # Fixed partition: the transfer mode and the
        # fork-M1 role->bundle map are decided by the placement translation.
        "--update-weight-transfer-mode",
        "--yeto-placement-map",
        # LR schedule is decided by RLRunConfig.algorithm.lr_schedule
        # (legacy rejects the same LR_SCHEDULE_FLAGS in learner.py).
        *LR_SCHEDULE_FLAGS,
        # Every objective-changing algorithm flag (rl-algorithm-capabilities):
        # mapped ones only enter through absorption into AlgorithmSpec,
        # unmapped ones are refused.
        *OBJECTIVE_FLAGS,
        # agentic-rollout-utilization 2.4: derived from --rl-max-policy-age
        # (rows registered in algo_flag_rows, after this import-time constant).
        "--partial-rollout",
        "--mask-offpolicy-in-partial-rollout",
    }
)
# Parsed-namespace attributes that must stay off (post-normalization check).
_FT_NAMESPACE_OFF = (
    "indep_dp",
    "use_fault_tolerance",
    "ft_components",
    "api_server_port",
    "mini_ft_controller_enable",
    "enable_witness",
    "fully_async",
)


class MilesConfigError(ValueError):
    """The ports path cannot express this configuration; startup is refused."""


def _require_elastic_pin(k31_feature: str, what: str) -> None:
    """Refuse ``what`` on a pin without the elastic verbs it needs: the k31
    lineage from k31/b4 on (``k31_feature``), or the ports-v1 fork (M1-M6 since
    0af62f4d)."""
    from yeto.rl.adapters.miles import pins

    if pins.pin_has(k31_feature, MILES_NEXT_COMMIT) or pins.pin_has("elastic_m1_m6", MILES_NEXT_COMMIT):
        return
    raise MilesConfigError(
        f"{what} needs Miles k31/b4 {pins.MILES_K31_B4_COMMIT[:9]} or later ({k31_feature}); "
        f"the pinned image ships {MILES_NEXT_COMMIT[:9]}")


class UnmappedConfigError(MilesConfigError):
    def __init__(self, path: str, reason: str) -> None:
        super().__init__(f"ports engine cannot map config option {path!r}: {reason}")
        self.path = path


class FaultToleranceArgsError(MilesConfigError):
    """Miles FT/indep-DP semantics were requested; the ports path refuses them."""


class SingleCellError(MilesConfigError):
    """The actor TrainGroup would not consist of exactly one cell."""


@dataclass(frozen=True)
class MilesLaunchArgs:
    argv: tuple[str, ...]
    placement: PlacementRequest
    algorithm_sha256: str
    # Attributes to set on the parsed namespace (read by yeto code running in
    # Miles processes, e.g. the bounded dynamic-sampling filter).
    runtime_attrs: Mapping[str, Any] = field(default_factory=dict)
    # The spec after absorbing mapped extra-argv flags (its sha256 is
    # ``algorithm_sha256``) and the absorbed flags ({flag: argv value}).
    algorithm: AlgorithmSpec | None = None
    absorbed_flags: Mapping[str, Any] = field(default_factory=dict)
    # Overlap: generation keeps running while the trainer trains (Miles
    # fully_async, bounded by --max-weight-staleness = the policy-age limit).
    rollout_overlap: bool = False


# --------------------------------------------------------------------------
# Leaf policy table
# --------------------------------------------------------------------------

_Check = Callable[[Any, Any], "str | None"]  # (value, config) -> rejection reason


def _ok(_value, _config) -> None:
    return None


def _must_be_none(reason: str) -> _Check:
    def check(value, _config):
        return None if value is None else reason

    return check


def _must_be_false(reason: str) -> _Check:
    def check(value, _config):
        return reason if value else None

    return check


# IR-1: the only upstream generate function that reads the agent flags.
AGENTIC_TOOL_CALL_GENERATE = "miles.rollout.generate_hub.agentic_tool_call.generate"
# IR-1 (codex-harness 7.1): reward is attributed per trajectory; a per-segment
# scope would make sibling segments pose as independent GRPO samples.
from yeto.rl.harness.reward_scope import HARNESS_REWARD_SCOPES, reward_scope_problem  # noqa: E402
# Upstream rule the session-server / partial-rollout exclusion quotes.
SESSION_SERVER_PARTIAL_ROLLOUT_RULE = (
    'miles/utils/arguments.py:3240 "--use-session-server does not support --partial-rollout"'
)


AGENT_FLAG_READERS = frozenset({AGENTIC_TOOL_CALL_GENERATE})


def _requires_agentic_generate(flag: str) -> _Check:
    def check(value, config):
        if value is None:
            return None
        if config.agent.custom_generate_function_path not in AGENT_FLAG_READERS:
            return (
                f"{flag} requires custom_generate_function_path={AGENTIC_TOOL_CALL_GENERATE} "
                f"(got {config.agent.custom_generate_function_path!r})"
            )
        return None

    return check


def check_session_server_partial_rollout(use_session_server: Any, partial_rollout: Any) -> None:
    """IR-1: the session server and partial rollout are mutually exclusive upstream."""
    if use_session_server and partial_rollout:
        raise UnmappedConfigError(
            "agent.use_session_server",
            f"--use-session-server with --partial-rollout is refused upstream: "
            f"{SESSION_SERVER_PARTIAL_ROLLOUT_RULE}",
        )


def check_harness_reward_scope(scope: Any) -> None:
    """IR-1 (codex-harness 7.1): only ``trajectory`` (or unset) starts."""
    problem = reward_scope_problem(scope)  # rule in the core (decoupling 3.5)
    if problem is not None:
        raise MilesConfigError(problem)


def _check_parameter_mode(value, _config):
    return None if value in ("lora", "full") else f"parameter mode {value!r}"


def _check_expert_full(value, config):
    """Expert-full rides the full-parameter transport, which needs DP=1."""

    if value and config.parallel.data_parallel != 1:
        return "expert-full tuning (full-parameter LoRA hybrid) requires DP=1"
    return None


def _check_sglang_tp(value, config):
    if value is None or int(value) == config.parallel.rollout_num_gpus_per_engine:
        return None
    return (
        "upstream Miles overrides --sglang-tp-size with "
        "--rollout-num-gpus-per-engine; the requested value would be rewritten"
    )


_GEOMETRY_FIELDS = (
    "num_layers hidden_size num_attention_heads num_query_groups kv_channels "
    "ffn_hidden_size max_position_embeddings vocab_size normalization "
    "norm_epsilon position_embedding_type rope_type rotary_base rotary_percent "
    "mrope_section gated_linear_unit untie_embeddings_and_output_weights "
    "disable_bias_linear add_qkv_bias qk_layernorm multi_latent_attention "
    "mla_dims moe"
).split()

LEAF_POLICY: dict[str, _Check] = {
    "hf_checkpoint": _ok,
    "ref_load": _ok,
    "model_recipe.name": _ok,
    "model_recipe.provider_class": _ok,
    "model_recipe.training_attention_backend": _ok,
    "model_recipe.gdn.gated_delta_net": _ok,
    "model_recipe.gdn.qkv_format": _ok,
    "model_recipe.transformer_impl": _ok,
    "model_recipe.model_flags": _ok,
    "model_recipe.expert_selection": _ok,  # kimi_k31: --partial-experts-selection
    "model_recipe.routed_expert_qat": _ok,  # kimi_k31: --routed-expert-qat
    **{f"geometry.{name}": _ok for name in _GEOMETRY_FIELDS},
    **{
        f"geometry.moe.{name}": _ok
        for name in (
            "num_experts",
            "ffn_hidden_size",
            "router_topk",
            "layer_freq",
            "shared_expert_intermediate_size",
        )
    },
    "parallel.actor_num_nodes": _ok,
    "parallel.actor_num_gpus_per_node": _ok,
    "parallel.tensor_parallel": _ok,
    "parallel.pipeline_parallel": _ok,
    "parallel.expert_parallel": _ok,
    "parallel.context_parallel": _ok,  # --context-parallel-size (translate_run_config)
    # Derived by Megatron from world size / TP / PP; consumed, not passed.
    "parallel.data_parallel": _ok,
    "parallel.rollout_num_gpus_per_engine": _ok,
    "parallel.dedicated_rollout_gpus": _ok,
    "parallel.visible_gpus_per_node": _ok,
    "parallel.island_gpus_per_node": _ok,
    "parallel.bundle_map": _ok,
    "parallel.uneven_pipeline_layers": _ok,
    "parallel.standby_gpus": _ok,
    "parallel.rollout_cell_names": _ok,
    # placement-only opt-ins consumed by yeto (PlacementRequest), no Miles flag
    "parallel.allow_cross_node_tp": _ok,
    "parallel.allow_cross_node_engine_tp": _ok,
    "trainable.parameter_mode": _check_parameter_mode,
    "trainable.lora_rank": _ok,
    "trainable.lora_dropout": _ok,
    "trainable.lora_expert_rank": _ok,
    "trainable.lora_targets": _ok,
    "trainable.target_modules": _ok,
    "trainable.expert_full_count": _check_expert_full,
    "data.prompt_path": _ok,
    # Source-column names are consumed by yeto's prompt normalization
    # (prepare_prompt_data) before Miles sees the file (#59); engine-neutral.
    "data.columns.prompt_column": _ok,
    "data.columns.label_column": _ok,
    "data.columns.input_key": _ok,
    "data.columns.label_key": _ok,
    "data.columns.metadata_key": _ok,
    "data.apply_chat_template": _ok,
    "data.chat_template_kwargs": _ok,
    **{
        f"batch.{name}": _ok
        for name in (
            "seq_len",
            "global_rounds",
            "eval_only",
            "groups_per_round",
            "samples_per_group",
            "over_sampling_batch_size",
            "optimizer_steps",
            "global_batch",
            "rollout_max_response_len",
        )
    },
    "algorithm.advantage_estimator": _ok,  # cross-checked against AlgorithmSpec
    "algorithm.reward_function": _ok,
    "algorithm.lr": _ok,
    "algorithm.lr_schedule": _ok,
    "algorithm.lr_schedule.decay_style": _ok,
    "algorithm.lr_schedule.decay_iters": _ok,
    "algorithm.seed": _ok,
    "algorithm.observe_training_entropy": _ok,
    "algorithm.mamba_training_ssm_states_dtype": _ok,
    "algorithm.rollout_seed": _ok,
    # rl-algo-critic-family 2.4: translated with the AlgorithmSpec (critic_load_argv)
    "algorithm.critic": _ok,
    "algorithm.critic.critic_load": _ok,
    "algorithm.critic.init_sha256": _ok,
    "eval": _ok,
    **{
        f"eval.{name}": _ok
        for name in (
            "interval",
            "dataset_name",
            "prompt_path",
            "samples_per_prompt",
            "temperature",
            "top_p",
            "max_prompt_len",
            "max_response_len",
            "max_context_len",
        )
    },
    "serving.mem_fraction_static": _ok,
    "serving.deterministic_inference": _ok,
    "serving.offload_train": _ok,
    "serving.tp_size": _check_sglang_tp,
    "serving.dp_size": _ok,
    "serving.ep_size": _ok,
    "serving.attention_backend": _ok,
    "serving.page_size": _ok,
    "serving.max_running_requests": _ok,
    "serving.chunked_prefill_size": _ok,
    "serving.sampler_snapshot": _ok,  # gated in translate_run_config (sampler_snapshot)
    "serving.rollout_quant_tier": _ok,  # SGLang flags in translate_run_config
    "serving.rollout_quant_fp8_source": _ok,  # tier 1 engine vs trainer (Miles block FP8)
    "ports.rollout_engine_base_port": _must_be_none(
        "upstream Miles has no --rollout-engine-base-port (fork-only port isolation, task 1.4b)"
    ),
    "ports.sglang_router_port": _ok,
    "ports.sglang_router_prometheus_port": _must_be_none(
        "no verified upstream equivalent of --sglang-router-prometheus-port"
    ),
    "ports.train_master_base_port": _must_be_none(
        "upstream Miles has no --train-master-base-port (fork-only port isolation, task 1.4b)"
    ),
    "agent.custom_generate_function_path": _ok,
    # IR-1: upstream agentic_tool_call.generate owns --custom-agent-function-path
    # and --max-seq-len (miles/rollout/generate_hub/agentic_tool_call.py
    # add_arguments); any other generate function has no reader for them.
    "agent.custom_agent_function_path": _requires_agentic_generate(
        "--custom-agent-function-path"
    ),
    "agent.agent_max_seq_len": _ok,  # upstream #4699: a core RolloutConfig field, every generate reads it
    "agent.dynamic_sampling_filter_path": _ok,  # reconciled with AlgorithmSpec
    "agent.use_rollout_routing_replay": _ok,
    "agent.use_session_server": _ok,
    "agent.session_server_ip": _ok,
    "agent.session_server_port": _ok,
    "agent.session_server_workers": _ok,
    "agent.tito_model": _ok,
    "agent.tito_allowed_append_roles": _must_be_none(
        "upstream Miles has no --tito-allowed-append-roles; the append roles are "
        "decided by the --tito-model template (allowed_append_roles)"
    ),
    # The driver owns the loop and the outer sync; no Miles callback needed.
    "yeto_policy_sync": _ok,
    "distributed_timeout_minutes": _ok,
    "deterministic_trainer": _ok,
    "trainer_dp_edges": _ok,
    "use_miles_router": _ok,
    "update_weight_group_timeout_s": _ok,
    "host_sync_free": _ok,  # k31/b3-b7: --no-host-sync-free when off (pin gate below)
    # d2-wire: read by the yeto-side ElasticHook only; no Miles option.
    "edge_costs_path": _ok,
    "elastic_window_s": _ok,
    # The buffer rollout functions (translate_run_config); the island
    # environment and the sampler loop read the rest.
    "sample_buffer": _ok,
    # translated below (Megatron full recompute / Miles dynamic batching)
    "train_memory.activation_recompute": _ok,
    "train_memory.max_tokens_per_gpu": _ok,
    "train_memory.gradient_accumulation_fusion": _ok,
    "train_memory.optimizer_offload_fraction": _ok,
    "train_memory.dp_optimizer": _ok,  # _full_parameter_optimizer_attrs
    "train_memory.grad_reduce_bf16": _ok,  # base argv: --grad-reduce-in-bf16
    "train_memory.precision_aware_optimizer": _ok,  # train_memory_argv
    **{f"sample_buffer.{name}": _ok
       for name in ("role", "store", "max_backlog_groups", "wait_s", "min_round_s",
                    "fused_stage", "release_legacy", "release_inline",
                    "release_delta_full_every", "release_every", "release_skip_busy",
                    "until_trainers_end")},
}


def iter_config_leaves(value: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Yield ``(dotted_path, value)`` for every non-dataclass leaf field."""

    if not dataclasses.is_dataclass(value) or isinstance(value, type):
        yield prefix, value
        return
    for f in dataclasses.fields(value):
        path = f"{prefix}.{f.name}" if prefix else f.name
        yield from iter_config_leaves(getattr(value, f.name), path)


def check_config_mapped(config: Any) -> None:
    """Reject unknown leaves and known leaves whose value has no mapping."""

    for path, value in iter_config_leaves(config):
        check = LEAF_POLICY.get(path)
        if check is None:
            raise UnmappedConfigError(path, "no ports translation exists for this option")
        reason = check(value, config)
        if reason is not None:
            raise UnmappedConfigError(path, reason)


# --------------------------------------------------------------------------
# Translation
# --------------------------------------------------------------------------


def _dotted_callable(spec: str) -> str:
    module, sep, name = spec.partition(":")
    return f"{module}.{name}" if sep else spec


def _algorithm_filter(config, algorithm: AlgorithmSpec) -> str | None:
    requested = normalize_filter_name(config.agent.dynamic_sampling_filter_path)  # 4.4
    if requested == STOCK_NONZERO_STD_FILTER and (
        algorithm.dynamic_sampling_max_replacements is not None
    ):
        requested = BOUNDED_NONZERO_STD_FILTER  # launcher normalization (legacy)
    if requested is not None and requested != algorithm.dynamic_sampling_filter:
        raise UnmappedConfigError(
            "agent.dynamic_sampling_filter_path",
            f"{requested!r} disagrees with AlgorithmSpec filter "
            f"{algorithm.dynamic_sampling_filter!r}",
        )
    return algorithm.dynamic_sampling_filter


def placement_request(config) -> PlacementRequest:
    parallel = config.parallel
    trainer = parallel.actor_num_nodes * parallel.actor_num_gpus_per_node
    if parallel.colocated:
        return PlacementRequest(
            kind="colocated",
            trainer_gpus=trainer,
            rollout_gpus=trainer,
            gpus_per_engine=parallel.rollout_num_gpus_per_engine,
            # m5: a multi-node colocated island applies the node rules (engine opt-in /
            # whole-node replicas, trainer tp*cp in-node) like a fixed partition
            gpus_per_node=getattr(parallel, "island_gpus_per_node", None),
            model_parallel=int(parallel.tensor_parallel) * int(parallel.pipeline_parallel)
            * int(getattr(parallel, "context_parallel", 1) or 1),
            node_parallel=int(parallel.tensor_parallel) * int(getattr(parallel, "context_parallel", 1) or 1),
            expert_parallel=int(getattr(parallel, "expert_parallel", 1) or 1),
            allow_cross_node_tp=bool(getattr(parallel, "allow_cross_node_tp", False)),
            allow_cross_node_engine_tp=bool(getattr(parallel, "allow_cross_node_engine_tp", False)),
        )
    _cp = int(getattr(parallel, "context_parallel", 1) or 1)
    request = PlacementRequest(
        kind="fixed-partition",
        trainer_gpus=trainer,
        rollout_gpus=int(parallel.dedicated_rollout_gpus),
        gpus_per_engine=parallel.rollout_num_gpus_per_engine,
        standby_gpus=int(getattr(parallel, "standby_gpus", 0) or 0),
        rollout_cell_names=tuple(getattr(parallel, "rollout_cell_names", ()) or ()),
        gpus_per_node=getattr(parallel, "island_gpus_per_node", None),
        # in-node group = tp*cp (TP stays in a node);
        # EP/PP may span nodes. model_parallel keeps the dense world group.
        model_parallel=int(parallel.tensor_parallel) * int(parallel.pipeline_parallel) * _cp,
        expert_parallel=int(getattr(parallel, "expert_parallel", 1) or 1),
        node_parallel=int(parallel.tensor_parallel) * _cp,
        bundle_map=getattr(parallel, "bundle_map", None),
        allow_cross_node_tp=bool(getattr(parallel, "allow_cross_node_tp", False)),
        allow_cross_node_engine_tp=bool(getattr(parallel, "allow_cross_node_engine_tp", False)),
    )
    if request.topology is not None:
        # The trainer rectangle the placement rules derive
        # must be the actor shape Miles is told (fail closed before any Ray work).
        shape = request.trainer_shape()
        told = (int(parallel.actor_num_nodes), int(parallel.actor_num_gpus_per_node))
        if shape != told:
            raise MilesConfigError(
                f"placement trainer shape {shape} (nodes x GPUs per node) disagrees with "
                f"--actor-num-nodes/--actor-num-gpus-per-node {told}")
    return request


def _flags(tokens: Sequence[str]) -> Iterator[str]:
    for token in tokens:
        flag = token.split("=", 1)[0]
        if flag.startswith("--"):
            yield flag


PIPELINE_SPLIT_FLAGS = ("--decoder-first-pipeline-num-layers", "--decoder-last-pipeline-num-layers")


def _argv_int(tokens: Sequence[str], flag: str) -> int | None:
    """The last value of ``flag`` in ``tokens`` (``--flag N`` or ``--flag=N``), None if absent."""
    found = None
    for i, token in enumerate(tokens):
        name, eq, value = token.partition("=")
        if name != flag:
            continue
        raw = value if eq else (tokens[i + 1] if i + 1 < len(tokens) else "")
        try:
            found = int(raw)
        except ValueError as exc:
            raise MilesConfigError(f"{flag} needs an integer layer count, got {raw!r}") from exc
    return found


def suggest_pipeline_splits(num_layers: int, pipeline_parallel: int, limit: int = 3) -> list[str]:
    """Up to ``limit`` (first, last) stage sizes for which the middle ``PP-2`` stages hold
    equally many layers, closest to an even split first."""
    pp = pipeline_parallel
    if pp < 2:
        return []
    if pp == 2:
        half = num_layers // 2
        return [f"first {num_layers - half} / last {half}"] if half >= 1 else []
    out: list[str] = []
    base = num_layers // pp
    for middle in sorted(range(1, num_layers // (pp - 2) + 1), key=lambda k: (abs(k - base), -k)):
        rest = num_layers - middle * (pp - 2)
        if rest < 2:
            continue
        first, last = (rest + 1) // 2, rest // 2
        out.append(f"first {first} / last {last} ({pp - 2} middle stages x {middle})")
        if len(out) == limit:
            break
    return out


def check_pipeline_split(num_layers: int, pipeline_parallel: int,
                         uneven: tuple[int, int] | None, extra_argv: Sequence[str] = ()) -> None:
    """Refuse a pipeline split Megatron cannot build, before any GPU starts.

    The effective split is the run config's ``uneven_pipeline_layers`` or the
    ``--decoder-first/last-pipeline-num-layers`` of the extra argv (both set to different
    values is a conflict). Without either, PP>1 needs ``num_layers % PP == 0``. With
    them, the layers left after the given first / last stage must split evenly over the
    remaining stages, at least one layer each (Megatron's uneven-PP rule)."""
    pp = int(pipeline_parallel)
    given = {flag: _argv_int(extra_argv, flag) for flag in PIPELINE_SPLIT_FLAGS}
    first, last = given[PIPELINE_SPLIT_FLAGS[0]], given[PIPELINE_SPLIT_FLAGS[1]]
    if uneven is not None:
        if (first is not None and first != uneven[0]) or (last is not None and last != uneven[1]):
            raise MilesConfigError(
                f"pipeline split conflict: the run config splits first {uneven[0]} / last {uneven[1]}, "
                f"the extra argv passes first {first} / last {last}")
        first, last = uneven
    if first is None and last is None:
        if pp > 1 and num_layers % pp:
            raise MilesConfigError(
                f"{num_layers} layers do not split evenly over pipeline parallel {pp}: pass an uneven "
                f"split ({' or '.join(PIPELINE_SPLIT_FLAGS)}), e.g. "
                + "; ".join(suggest_pipeline_splits(num_layers, pp)))
        return
    if pp < 2:
        raise MilesConfigError(f"{' / '.join(PIPELINE_SPLIT_FLAGS)} need pipeline parallel > 1 (got {pp})")
    fixed = [n for n in (first, last) if n is not None]
    if any(n < 1 for n in fixed):
        raise MilesConfigError(f"pipeline stage layer counts must be >= 1 (first {first} / last {last})")
    rest_layers = num_layers - sum(fixed)
    rest_stages = pp - len(fixed)
    ok = (rest_layers == 0) if rest_stages == 0 else (rest_layers >= rest_stages and rest_layers % rest_stages == 0)
    if not ok:
        what = (f"first {first} + last {last}" if len(fixed) == 2
                else f"{'first' if first is not None else 'last'} {fixed[0]}")
        raise MilesConfigError(
            f"pipeline split does not add up: {num_layers} layers, PP {pp}, {what} leaves {rest_layers} "
            f"layers for {rest_stages} stage(s) (needs an equal share of >= 1 each); e.g. "
            + "; ".join(suggest_pipeline_splits(num_layers, pp)))


def reject_fault_tolerance_flags(tokens: Sequence[str]) -> None:
    for flag in _flags(tokens):
        if flag in FAULT_TOLERANCE_FLAGS:
            raise FaultToleranceArgsError(
                f"{flag} enables Miles fault-tolerance semantics, which the ports "
                "engine does not support"
            )


def _refuse_decoupled_ppo_breakers(tokens: Sequence[str]) -> None:
    """Flags that remove pi_prox or mask
    tokens by version are refused, with the flag named."""
    from .offpolicy_correction import refuse_forbidden_argv
    from yeto.rl.engine.offpolicy_correction import CorrectionConfigError

    try:
        refuse_forbidden_argv(tokens)
    except CorrectionConfigError as exc:
        raise MilesConfigError(str(exc)) from exc


def check_extra_argv(extra_argv: Sequence[str], algorithm: AlgorithmSpec | None = None):
    """Refuse FT/adapter-owned flags; absorb mapped algorithm flags.

    Mapped algorithm flags (``algorithm_flags.MAPPINGS``) are absorbed into
    ``algorithm`` (conflicts raise, naming flag and both values); other
    objective-changing flags are refused as not yet in the algorithm spec.
    With ``algorithm`` returns ``(spec, remaining_argv, absorbed)``.
    """

    reject_fault_tolerance_flags(extra_argv)
    _refuse_decoupled_ppo_breakers(extra_argv)
    mapped = mapped_flags()
    unmapped_objective = objective_flags() - mapped
    for flag in _flags(extra_argv):
        if flag in mapped:
            continue
        if flag in unmapped_objective:
            raise MilesConfigError(
                f"{flag} changes the training objective but is not part of the algorithm "
                "spec yet; it cannot be passed through extra argv"
            )
        if flag in ADAPTER_OWNED_FLAGS:
            raise MilesConfigError(f"{flag} is owned by the ports adapter and cannot be overridden")
    if algorithm is None:
        return None
    try:
        return absorb_extra_argv(algorithm, extra_argv)
    except AlgorithmSpecError as exc:
        raise MilesConfigError(str(exc)) from exc


def critic_load_argv(config: Any, algorithm: AlgorithmSpec) -> list[str]:
    """rl-algo-critic-family: ``--critic-load`` of the ports main stage.

    A copied critic with a warm-up (``critic.init='copy_actor_backbone'``,
    ``warmup_steps > 0``) loads the warm-up stage product given by the run
    config; without a warm-up Miles copies the actor checkpoint itself
    (``critic_load`` defaults to ``--load``, arguments.py:3607-3608).
    ``critic.init='load'`` is translated by the spec (``critic_argv``).
    """

    critic = getattr(config.algorithm, "critic", None)
    given = critic is not None and critic.critic_load is not None
    if not algorithm.execution.needs_critic:
        if given:
            raise MilesConfigError("algorithm.critic.critic_load is set but the algorithm has no critic")
        return []
    spec = algorithm.critic
    if spec.init == "copy_actor_backbone" and spec.warmup_steps:
        if not given:
            raise MilesConfigError(
                f"critic.warmup_steps={spec.warmup_steps}: the main stage loads the warm-up "
                "stage product (run config algorithm.critic.critic_load); run the warm-up "
                "stage first (rl-algo-critic-family)"
            )
        return ["--critic-load", critic.critic_load]
    if given:
        raise MilesConfigError(
            "algorithm.critic.critic_load is the warm-up product, used only with "
            f"critic.init='copy_actor_backbone' and warmup_steps > 0 (spec: init={spec.init!r}, "
            f"warmup_steps={spec.warmup_steps})"
        )
    return []


def _lora_argv(trainable: Any, *, colocated: bool) -> list[str]:
    """Upstream LoRA flags (miles/utils/lora/arguments.py); none in full-parameter mode.

    Expert-full trains base parameters too, so a restored base CPU backup would
    undo published cuts; Miles refuses the backup for the LoRA hybrid.
    """

    if trainable.parameter_mode != "lora":
        return []
    return [
        "--lora-rank", str(trainable.lora_rank),
        "--lora-alpha", str(trainable.lora_rank),
        # "0" (default argv unchanged) unless --rl-lora-dropout
        "--lora-dropout", (format(trainable.lora_dropout, "g")
                           if getattr(trainable, "lora_dropout", 0.0) else "0"),
        # DSV4 clone-only LoRA keeps one packed adapter per grouped-expert
        # linear (standard LoRA); canonical LoRA splits it per projection.
        "--lora-type", "lora" if trainable.routed_expert_lora else "canonical_lora",
        "--target-modules", ",".join(trainable.target_modules),
        # upstream applies the LoRA base CPU backup only under colocate
        # (utils/lora/utils.py:39-41); a LoRA fixed partition drops it.
        *(["--lora-base-cpu-backup"] if colocated and not trainable.expert_full else []),
        "--sglang-max-lora-rank", str(trainable.lora_rank),
    ]



def _leading_dense_layers(moe: Any) -> int:
    """``first_k_dense_replace`` of a MoE geometry: the leading zeros of its layer
    mask (an integer frequency means every layer is MoE)."""
    freq = None if moe is None else moe.layer_freq
    if not isinstance(freq, (list, tuple)):
        return 0
    dense = 0
    for value in freq:
        if int(value):
            break
        dense += 1
    return dense


def translate_run_config(
    config: Any,
    algorithm: AlgorithmSpec,
    *,
    extra_argv: Sequence[str] = (),
    tito_parser_resolver: Callable[[str], tuple[str | None, str | None]] | None = None,
    max_policy_age: int = 0,
    async_buffer_rounds: int | None = None,
    full_parameter_sync: bool = True,
    rollout_ahead: int = 0,
) -> MilesLaunchArgs:
    """Pure translation of an ``RLRunConfig`` into upstream Miles argv.

    ``max_policy_age`` (--rl-max-policy-age, agentic-rollout-utilization 4.1):
    0 adds nothing; N > 0 needs an algorithm tolerating N
    (``execution.max_policy_staleness``). A colocated island appends Miles'
    partial rollout (carry-over at the cut-off); a partitioned island overlaps
    rollout with training through Miles' fully_async, so the
    all-samples metadata hook and the carry-over buffer filter are not set:
    the trained-groups hook reports the round instead.

    ``async_buffer_rounds`` (--rl-async-buffer-rounds): K >= 1 sets the
    fully_async buffer to K rounds of groups (``--async-data-buffer-capacity-factor
    K``); None (default) uses K = ``max_policy_age``, not Miles' own default 2,
    which at age 1 dropped about half of the generated groups as too old
    Only with overlap.

    ``full_parameter_sync`` (default True): a full-parameter trainer feeds the
    ports full-parameter sync, which needs whole FP32 masters; False for a
    no-sync trainer island (``_full_parameter_optimizer_attrs``).

    ``rollout_ahead`` (--rl-rollout-ahead, ``yeto.rl.engine.rollout_ahead``):
    the driver generates whole batches ahead; Miles then sees one on-policy
    batch per publication, so neither fully_async nor partial rollout is set
    and the argv is the limit-0 argv (the limit binds only the driver)."""

    if not isinstance(algorithm, AlgorithmSpec):
        raise TypeError("algorithm must be an AlgorithmSpec")
    check_config_mapped(config)
    algorithm, extra_argv, absorbed = check_extra_argv(extra_argv, algorithm)
    check_pipeline_split(config.geometry.num_layers, config.parallel.pipeline_parallel,
                         config.parallel.uneven_pipeline_layers, extra_argv)
    problems = algorithm.rejections()
    if problems:
        raise MilesConfigError("algorithm spec rejected: " + "; ".join(problems))
    from yeto.rl.engine.algorithm import launch_problems

    problems = launch_problems(algorithm, {
        "rollout_batch_size": config.batch.groups_per_round,
        "rollout_max_response_len": config.batch.rollout_max_response_len,
        "context_parallel_size": int(getattr(config.parallel, "context_parallel", 1) or 1),
        "multi_lora": any(t.split("=", 1)[0] == "--multi-lora" for t in extra_argv),
        # rl-algo-critic-family 2.3: critic GPU counts / deploy-component / indep-dp
        "extra_argv": tuple(extra_argv),
        "reward_function": config.algorithm.reward_function,
        "actor_num_nodes": config.parallel.actor_num_nodes,
        "actor_num_gpus_per_node": config.parallel.actor_num_gpus_per_node,
        "elastic": bool(config.use_miles_router),  # set iff --rl-elastic (run_config)
        # critic_lr_warmup: the critic scheduler's decay length (Miles model.py)
        "num_rollout": 0 if config.batch.eval_only else config.batch.global_rounds,
        "n_samples_per_prompt": config.batch.samples_per_group,
        "global_batch_size": config.batch.global_batch,
        "lr_decay_iters": (config.algorithm.lr_schedule.decay_iters
                           if config.algorithm.lr_schedule is not None else None),
        "lr_decay_style": (config.algorithm.lr_schedule.decay_style
                           if config.algorithm.lr_schedule is not None else None),
    })
    if problems:
        raise MilesConfigError("algorithm spec rejected for this run: " + "; ".join(problems))
    requested_over = algorithm.sampling.over_sampling_batch_size
    if requested_over is not None and requested_over != config.batch.over_sampling_batch_size:
        raise UnmappedConfigError(
            "batch.over_sampling_batch_size",
            f"{config.batch.over_sampling_batch_size!r} disagrees with AlgorithmSpec "
            f"sampling.over_sampling_batch_size {requested_over!r}",
        )
    if config.algorithm.advantage_estimator != algorithm.advantage_estimator:
        raise UnmappedConfigError(
            "algorithm.advantage_estimator",
            f"{config.algorithm.advantage_estimator!r} disagrees with AlgorithmSpec "
            f"{algorithm.advantage_estimator!r}",
        )
    dynamic_filter = _algorithm_filter(config, algorithm)

    from yeto.rl.engine.run_config import (
        RECIPE_DEEPSEEK_V4_FLASH,
        RECIPE_GENERIC,
        RECIPE_GLM5_NEXT,
        RECIPE_KIMI_K31,
        RECIPE_NEMOTRON_H,
        RECIPE_QWEN3_5,
        RECIPE_QWEN3_8_NEXT,
    )
    from .lr_schedule import lr_schedule_argv
    from .models import glm5_next, nemotron_h
    from .models.deepseek_v4 import ports_recipe as deepseek_v4
    from .models.kimi_k31 import ports_recipe as kimi_k31_recipe
    from .models.kimi_k31 import release as kimi_k31_release

    geometry = config.geometry
    parallel = config.parallel
    trainable = config.trainable
    batch = config.batch
    recipe = config.model_recipe
    serving = config.serving
    agent = config.agent
    columns = config.data.columns
    request = placement_request(config)
    sample_buffer = getattr(config, "sample_buffer", None)
    from .buffer_rollout import CONVERTIBLE

    if sample_buffer is not None and sample_buffer.role == CONVERTIBLE:
        # Trainer and engines share the GPUs; the converted node loads its own
        # trained weights into its engines (no release through the store). LoRA
        # (the GPU-verified form) or full parameters: the handoff path is Miles'
        # colocated update_weights, which is the upstream default for every
        # parameter; the K3.1 recipe (partial experts, MXFP4 releases) is
        # full-parameter only, so the node trains and releases whole checkpoints.
        if trainable.parameter_mode not in ("lora", "full") or request.kind != "colocated":
            raise MilesConfigError("--rl-sample-role convertible needs LoRA or full parameters "
                                   "(--tuning lora|full) on a colocated island")
    elif (sample_buffer is not None and sample_buffer.role == "trainer"
            and trainable.parameter_mode == "lora" and request.kind == "colocated"
            and request.trainer_gpus != 1):
        # A colocated trainer is asleep (process groups destroyed) when it publishes;
        # release_plugin reloads them for the LoRA export as Miles' update_weights
        # does. GPU-verified on one rank only (T-LSMP i); more ranks stay refused.
        raise MilesConfigError("--rl-sample-role trainer with LoRA and a colocated placement "
                               "supports one trainer GPU only; use --rl-placement fixed-partition")
    # a buffer island selects its own rollout function, so never Miles' fully_async
    rollout_overlap = (max_policy_age > 0 and request.kind != "colocated" and sample_buffer is None
                       and not rollout_ahead)

    if request.kind == "colocated":
        placement_values = ["--colocate"]
    elif request.trainer_only:
        # Samples come from the store: Miles starts no engine and syncs no weights.
        placement_values = ["--debug-train-only"]
    else:
        placement_values = ["--rollout-num-gpus", str(request.rollout_gpus)]
        if trainable.parameter_mode == "lora":
            if serving.offload_train:
                # Conservative choice of this change, NOT an upstream
                # constraint: upstream train.py:139-151 also offloads outside
                # colocate. Refused until a partitioned GPU run shows that a
                # per-round onload + NCCL broadcast publish is safe/cheap.
                raise MilesConfigError(
                    "a LoRA fixed partition publishes over NCCL broadcast every round; "
                    "the trainer must stay resident (no offload_train)"
                )
            # upstream protocol.py:73-89: only broadcast (or colocate CUDA IPC)
            # supports LoRA; p2p/disk-delta assert no LoRA.
            placement_values += ["--update-weight-transfer-mode", "broadcast"]
        if request.placement_map_arg is not None:
            # fork-M1 (--yeto-placement-map): explicit role -> bundle map
            # (ClusterConfig.yeto_placement_map on k31/b4+).
            _require_elastic_pin("placement_map", "an explicit bundle map (--yeto-placement-map)")
            placement_values += [
                "--yeto-placement-map",
                json.dumps(request.placement_map_arg, sort_keys=True, separators=(",", ":")),
            ]

    model_recipe_values: list[str] = []
    if recipe.name == RECIPE_QWEN3_5:
        model_name = "qwen3_5"
        model_recipe_values = [
            "--spec",
            *QWEN3_5_SPEC,
            "--apply-layernorm-1p",
            "--attention-output-gate",
            "--attention-dropout",
            "0.0",
            "--hidden-dropout",
            "0.0",
        ]
    elif recipe.name == RECIPE_GLM5_NEXT:
        model_name = glm5_next.MODEL_NAME  # the model flags replace the geometry below
    elif recipe.name == RECIPE_KIMI_K31:
        # the plugin spec replaces the geometry below; a Kimi-K3 checkpoint (no DSA
        # indexer, e.g. the 4-layer smoke model) takes the K3 bridge and spec
        kimi_hf_model_type = _kimi_hf_model_type(config.hf_checkpoint)
        model_name = kimi_k31_recipe.model_name_for(kimi_hf_model_type)
    elif recipe.name == RECIPE_DEEPSEEK_V4_FLASH:
        model_name = deepseek_v4.MODEL_NAME
        model_recipe_values = deepseek_v4.recipe_argv(trainable)
    else:
        model_name = recipe.provider_class
        if recipe.name == RECIPE_GENERIC:
            # Megatron's default dropout is 0.1; the Miles recipes train RL with
            # none (the policy's train-time log-probs must match the rollout's)
            model_recipe_values = ["--attention-dropout", "0.0", "--hidden-dropout", "0.0"]

    values: list[str] = [
        "train.py",
        "--train-backend", "megatron",
        "--hf-checkpoint", config.hf_checkpoint,
        "--ref-load", config.ref_load,
        # glm5_next: Megatron-Bridge has no glm5_next bridge; Miles loads the torch_dist
        # checkpoint and exports through its own converters (raw). kimi_k31: the
        # partial-experts trainer loads the MXFP4 checkpoint through the Miles mbridge
        # plugin and publishes through the Direct iterator (raw).
        "--megatron-to-hf-mode", kimi_k31_release.megatron_to_hf_mode(recipe.name),
        "--model-name", model_name,
        *model_recipe_values,
        "--num-layers", str(geometry.num_layers),
        "--hidden-size", str(geometry.hidden_size),
        "--num-attention-heads", str(geometry.num_attention_heads),
        "--num-query-groups", str(geometry.num_query_groups),
        "--kv-channels", str(geometry.kv_channels),
        "--ffn-hidden-size", str(geometry.ffn_hidden_size),
        "--max-position-embeddings", str(geometry.max_position_embeddings),
        "--seq-length", str(batch.seq_len),
        "--normalization", geometry.normalization,
        "--norm-epsilon", str(geometry.norm_epsilon),
        "--position-embedding-type", geometry.position_embedding_type,
        "--rotary-base", str(geometry.rotary_base),
        "--rotary-percent", str(geometry.rotary_percent),
        "--vocab-size", str(geometry.vocab_size),
        *_lora_argv(trainable, colocated=request.kind == "colocated"),
        # placement
        "--actor-num-nodes", str(parallel.actor_num_nodes),
        "--actor-num-gpus-per-node", str(parallel.actor_num_gpus_per_node),
        "--num-gpus-per-node", str(parallel.visible_gpus_per_node),
        "--rollout-num-gpus-per-engine", str(parallel.rollout_num_gpus_per_engine),
        *placement_values,
        # Upstream colocate (LoRA base CPU backup, serial offload/onload in
        # the driver) requires torch_memory_saver offload of the trainer;
        # without it update_weights fails on the missing LD_PRELOAD hook.
        "--offload-train"
        if serving.offload_train or request.kind == "colocated"
        else "--no-offload-train",
        "--sglang-mem-fraction-static", str(serving.mem_fraction_static),
        "--tensor-model-parallel-size", str(parallel.tensor_parallel),
        "--pipeline-model-parallel-size", str(parallel.pipeline_parallel),
        "--context-parallel-size", str(int(getattr(parallel, "context_parallel", 1) or 1)),
        "--expert-model-parallel-size", str(parallel.expert_parallel),
        "--expert-tensor-parallel-size", "1",
        # data
        "--prompt-data", config.data.prompt_path,
        "--input-key", columns.input_key,
        "--label-key", columns.label_key,
        "--metadata-key", columns.metadata_key,
        "--rollout-seed", str(config.algorithm.rollout_seed),
        "--num-rollout", str(0 if batch.eval_only else batch.global_rounds),
        "--rollout-batch-size", str(batch.groups_per_round),
        "--n-samples-per-prompt", str(batch.samples_per_group),
        "--over-sampling-batch-size", str(batch.over_sampling_batch_size),
        "--num-steps-per-rollout", str(batch.optimizer_steps),
        "--global-batch-size", str(batch.global_batch),
        # DP-change certification refuses --balance-data (reshard.reshard_problems):
        # dropped only when trainer DP-change edges are enabled.
        *(() if getattr(config, "trainer_dp_edges", False) else ("--balance-data",)),
        # upstream #4699: --rollout-max-context-len became the core RolloutConfig
        # field --max-seq-len (the agentic generate reads the same field), emitted once
        "--max-seq-len", str(int(agent.agent_max_seq_len) if agent.agent_max_seq_len is not None
                             else batch.seq_len),
        "--rollout-max-response-len", str(batch.rollout_max_response_len),
        # metadata is extracted inside the rollout process
        "--rollout-sample-filter-path", TRAINED_GROUPS_HOOK_PATH,
        *(() if rollout_overlap else (
            "--rollout-all-samples-process-path", ROLLOUT_META_HOOK_PATH,
            "--buffer-filter-path", POLICY_BUFFER_FILTER_PATH,
        )),
        "--custom-rm-path", _dotted_callable(config.algorithm.reward_function),
        # algorithm: the only place AlgorithmSpec becomes Miles args
        "--advantage-estimator", algorithm.advantage_estimator,
        "--lr", str(config.algorithm.lr),
        # bf16 gradient buffers (--rl-grad-reduce-bf16, or the bf16 main grads of
        # --rl-precision-aware-optimizer): Megatron asserts that
        # --accumulate-allreduce-grads-in-fp32 is absent then
        *_gradient_dtype_argv(config.train_memory),
        "--attention-softmax-in-fp32",
        "--attention-backend", recipe.training_attention_backend,
        *(() if config.train_memory.gradient_accumulation_fusion else ("--no-gradient-accumulation-fusion",)),
        "--bf16",
        "--no-load-optim",
        "--no-load-rng",
        "--no-save-optim",
        "--no-save-rng",
        "--finetune",
        "--seed", str(config.algorithm.seed),
        "--pin-rollout-manager-to-head",
    ]
    # Explicit LR schedule: Miles' default horizon is --num-rollout (global
    # rounds), which under decoupled is not the island's local step count.
    values.extend(lr_schedule_argv(config.algorithm.lr_schedule))
    if config.algorithm.observe_training_entropy:
        values.append("--observe-training-entropy")
    if config.algorithm.mamba_training_ssm_states_dtype is not None:
        values.extend(("--mamba-training-ssm-states-dtype", config.algorithm.mamba_training_ssm_states_dtype))
    if algorithm.kl_coef is not None:
        values.extend(("--kl-coef", str(algorithm.kl_coef)))
    if dynamic_filter is not None:
        values.extend(filter_argv(dynamic_filter))  # 4.4: neutral name -> Miles path
    # Non-default AlgorithmSpec v2 fields (empty for every v1 spec, so the
    # default GRPO argv is byte-identical to R0).
    values.extend(algorithm_argv(algorithm))
    if max_policy_age:
        from yeto.rl.engine.execution_profile import algorithm_max_policy_staleness

        from .policy_age import SUPPORT, policy_age_argv

        SUPPORT.check(max_policy_age, buffer_trainer=sample_buffer is not None
                      and sample_buffer.role == "trainer")
        tolerated = algorithm_max_policy_staleness(algorithm)
        if tolerated < max_policy_age:
            raise MilesConfigError(
                f"--rl-max-policy-age {max_policy_age} needs an algorithm spec with "
                f"execution.max_policy_staleness >= {max_policy_age} (and a cross-version correction: "
                "correction.method tis, or score_centering_is tis / mis); "
                f"the run's spec tolerates {tolerated}")
        if not rollout_ahead:
            agentic = bool(getattr(config.agent, "custom_agent_function_path", None))
            if agentic and not rollout_overlap:
                # B5 (k31/b3-b7, ports-v1 ddce20992): --agentic-suspend-between-turns /
                # --agentic-suspend-max-rounds (RolloutConfig); older pins reject the flags.
                from yeto.rl.adapters.miles import pins

                if not pins.pin_has("agentic_suspend", MILES_NEXT_COMMIT):
                    raise MilesConfigError(
                        f"--rl-max-policy-age {max_policy_age} on an agentic task needs Miles with "
                        f"--agentic-suspend-between-turns (k31/b3-b7 {pins.MILES_K31_B3B7_COMMIT[:9]} or "
                        f"later); the pinned image ships {MILES_NEXT_COMMIT[:9]}")
            values.extend(policy_age_argv(max_policy_age, agentic=agentic, overlap=rollout_overlap))
    if async_buffer_rounds is None and rollout_overlap:
        async_buffer_rounds = max_policy_age  # default: buffer depth = A_hard
    if async_buffer_rounds is not None:
        if not rollout_overlap:
            raise MilesConfigError(
                "--rl-async-buffer-rounds needs generation that overlaps training "
                "(--rl-max-policy-age > 0 on a partitioned island without a sample buffer)")
        if int(async_buffer_rounds) < 1:
            raise MilesConfigError(
                f"--rl-async-buffer-rounds must be >= 1, got {async_buffer_rounds}")
        from .offpolicy_correction import BUFFER_CAPACITY_FLAG

        values.extend((BUFFER_CAPACITY_FLAG, str(int(async_buffer_rounds))))
    values.extend(critic_load_argv(config, algorithm))
    if sample_buffer is not None:
        values.extend(sample_buffer_argv(sample_buffer.role, max_policy_age=max_policy_age))

    evaluation = config.eval
    if evaluation is not None:
        # Upstream inherits the (stock) rollout function for eval.
        values.extend(
            (
                "--eval-prompt-data",
                evaluation.dataset_name,
                evaluation.prompt_path,
                "--eval-interval",
                str(evaluation.interval),
                "--skip-eval-before-train",
                "--n-samples-per-eval-prompt",
                str(evaluation.samples_per_prompt),
                "--log-passrate",
            )
        )
        for flag, value in (
            ("--eval-temperature", evaluation.temperature),
            ("--eval-top-p", evaluation.top_p),
            ("--eval-max-prompt-len", evaluation.max_prompt_len),
            ("--eval-max-response-len", evaluation.max_response_len),
            # --eval-max-context-len is gone upstream (#4699); eval context = --max-seq-len
        ):
            if value is not None:
                values.extend((flag, str(value)))
    if serving.deterministic_inference:
        values.append("--sglang-enable-deterministic-inference")
    if getattr(serving, "rollout_quant_tier", 0) == TIER_ONLINE_FP8 and \
            getattr(serving, "rollout_quant_fp8_source", FP8_SOURCE_ENGINE) == FP8_SOURCE_ENGINE:
        # tier 1: SGLang quantizes the in-scope linears itself; the ignore list
        # travels in the workers' environment (entry.connect_island_ray). The trainer
        # source needs no SGLang flag: the FP8 checkpoint's config carries the method.
        values.extend(sglang_online_argv())
    if getattr(serving, "sampler_snapshot", False):
        try:
            values.extend(sampler_snapshot.snapshot_argv(request, extra_argv=extra_argv))
        except sampler_snapshot.SamplerSnapshotRefused as exc:
            raise MilesConfigError(str(exc)) from exc
    if parallel.uneven_pipeline_layers is not None:
        first_layers, last_layers = parallel.uneven_pipeline_layers
        values.extend(
            (
                "--decoder-first-pipeline-num-layers", str(first_layers),
                "--decoder-last-pipeline-num-layers", str(last_layers),
            )
        )
    if config.data.apply_chat_template:
        values.append("--apply-chat-template")
    if parallel.tensor_parallel > 1:
        values.append("--sequence-parallel")
    values.extend(("--distributed-timeout-minutes", str(config.distributed_timeout_minutes)))
    if getattr(config, "use_miles_router", False):
        # --rl-elastic: cordon/drain/admit_cordoned need the Miles router (and its
        # cordon endpoints: k31/b4 router_cordon, ports-v1 M3)
        _require_elastic_pin("router_cordon", "--rl-elastic (cordon / drain / admit on the Miles router)")
        values.extend(("--router", "miles"))  # upstream #4322: --use-miles-router is this alias
    timeout_s = getattr(config, "update_weight_group_timeout_s", None)
    if timeout_s is not None:
        # A27: bounded weight-update group rendezvous. The upstream flag names
        # the per-engine request timeout; the k31/b3-b7 B3 batch (feature
        # engine_failure_isolation) bounds each engine's join by it and raises an
        # ExternalFailureError instead of losing the trainer (ports-v1: fork image-m3a27+
        # --update-weight-group-timeout-s, renamed here).
        values.extend(("--update-weight-engine-request-timeout", f"{float(timeout_s):g}"))
    if not getattr(config, "host_sync_free", True):
        # B7: the fork's MILES_HOST_SYNC_FREE=0 escape hatch is the typed
        # --no-host-sync-free (DebugConfig.host_sync_free); the environment
        # variable is not read any more. The default (on) emits nothing.
        from yeto.rl.adapters.miles import pins

        if not pins.pin_has("host_sync_free_flag", MILES_NEXT_COMMIT):
            raise MilesConfigError(
                "--no-host-sync-free needs Miles k31/b3-b7 "
                f"{pins.MILES_K31_B3B7_COMMIT[:9]} or later (DebugConfig.host_sync_free); "
                f"the pinned image ships {MILES_NEXT_COMMIT[:9]}")
        values.append("--no-host-sync-free")
    if config.train_memory.max_tokens_per_gpu is not None and recipe.gdn.gated_delta_net:
        # Miles miles_validate_args: dynamic batch size is not supported for bshd
        raise MilesConfigError(
            "--rl-max-tokens-per-gpu needs packed (thd) sequences; this gated-delta-net "
            f"model trains with --qkv-format {recipe.gdn.qkv_format}"
        )
    values.extend(train_memory_argv(config.train_memory))
    if getattr(config, "deterministic_trainer", False):
        values.append("--deterministic-mode")  # Megatron deterministic kernels
    if config.data.chat_template_kwargs:
        values.extend(
            (
                "--apply-chat-template-kwargs",
                json.dumps(config.data.chat_template_kwargs, sort_keys=True, separators=(",", ":")),
            )
        )
    router_port = config.ports.sglang_router_port
    if router_port is not None:
        # Co-resident islands (task 1.4b): upstream defaults the router
        # Prometheus listener (29000) and the dynamic worker port range
        # (20000) per host, so two islands on one machine collide. Derive
        # disjoint values from the island's router port.
        values.extend(
            (
                "--router-prometheus-port", str(int(router_port) + 1),
                "--worker-dynamic-port-start", str(int(router_port) + 100),
            )
        )
    for flag, value in (
        ("--sglang-router-port", config.ports.sglang_router_port),
        ("--sglang-tp-size", serving.tp_size),
        ("--sglang-dp-size", serving.dp_size),
        ("--sglang-ep-size", serving.ep_size),
        ("--sglang-attention-backend", serving.attention_backend),
        ("--sglang-page-size", serving.page_size),
        ("--sglang-max-running-requests", serving.max_running_requests),
        ("--sglang-chunked-prefill-size", serving.chunked_prefill_size),
    ):
        if value is not None:
            values.extend((flag, str(value)))
    if agent.use_rollout_routing_replay:
        values.append("--use-rollout-routing-replay")
    if agent.custom_generate_function_path:
        values.extend(("--custom-generate-function-path", agent.custom_generate_function_path))
    if agent.custom_agent_function_path:  # IR-1 (leaf policy proved agentic generate)
        values.extend(("--custom-agent-function-path", agent.custom_agent_function_path))
        values.extend(harbor_rollout_argv(agent.custom_agent_function_path, sample_buffer=sample_buffer))
    if agent.use_session_server:
        values.append("--use-session-server")
        if agent.session_server_ip:
            values.extend(("--session-server-ip", agent.session_server_ip))
        if agent.session_server_port:
            values.append("--session-server-port")
            values.extend(str(port) for port in agent.session_server_port)
        if agent.session_server_workers is not None:
            values.extend(("--session-server-workers", str(agent.session_server_workers)))
        if "--miles-router-timeout" not in _flags(extra_argv):
            values.extend(("--miles-router-timeout", str(SESSION_UPSTREAM_TIMEOUT_S)))
        if agent.tito_model:
            resolver = tito_parser_resolver or _resolve_tito_parsers
            values.extend(("--tito-model", agent.tito_model))
            reasoning_parser, tool_call_parser = resolver(agent.tito_model)
            if reasoning_parser is not None:
                values.extend(("--sglang-reasoning-parser", reasoning_parser))
            if tool_call_parser is not None:
                values.extend(("--sglang-tool-call-parser", tool_call_parser))
    elif agent.tito_model:
        raise UnmappedConfigError("agent.tito_model", "TITO requires the session server")
    if geometry.group_query_attention:
        values.append("--group-query-attention")
    if geometry.rope_type is not None:
        values.extend(("--rope-type", geometry.rope_type))
    if geometry.mrope_section is not None:
        values.append("--mrope-section")
        values.extend(str(v) for v in geometry.mrope_section)
    if geometry.gated_linear_unit:
        values.append("--swiglu")
    if geometry.untie_embeddings_and_output_weights:
        values.append("--untie-embeddings-and-output-weights")
    if geometry.disable_bias_linear:
        values.append("--disable-bias-linear")
    if geometry.add_qkv_bias:
        values.append("--add-qkv-bias")
    if geometry.qk_layernorm:
        values.append("--qk-layernorm")
    if recipe.gdn.gated_delta_net:
        values.extend(("--qkv-format", recipe.gdn.qkv_format))
    moe = geometry.moe
    if moe is not None:
        values.extend(
            (
                "--num-experts", str(moe.num_experts),
                "--moe-ffn-hidden-size", str(moe.ffn_hidden_size),
                "--moe-router-topk", str(moe.router_topk),
                "--moe-layer-freq", megatron_moe_layer_freq(moe.layer_freq),
            )
        )
        if moe.shared_expert_intermediate_size is not None:
            values.extend(
                ("--moe-shared-expert-intermediate-size", str(moe.shared_expert_intermediate_size))
            )
    # recipe-owned model flags read from the provider (nemotron_h: router, latent MoE)
    values.extend(getattr(recipe, "model_flags", ()))
    if recipe.name == RECIPE_NEMOTRON_H:
        values.extend(_nemotron_h_train_flags(nemotron_h, config))
    if geometry.multi_latent_attention:
        values.append("--multi-latent-attention")
        for name, value in geometry.mla_dims:
            values.extend((f"--{name.replace('_', '-')}", str(value)))
    if recipe.name == RECIPE_QWEN3_8_NEXT:
        # the native Flash-Next recipe replaces the Bridge-geometry model and
        # LoRA flags with the launcher's (profiles/qwen3_8_next.py).
        from yeto.rl.profiles import qwen3_8_next as q

        variant = {n: v for v, n in q.NUM_LAYERS.items()}.get(geometry.num_layers)
        if variant is None:
            raise MilesConfigError(f"no Flash-Next variant has {geometry.num_layers} layers")
        values = list(q.apply_ports_recipe(values, q.ports_recipe_argv(
            variant, lora_rank=trainable.lora_rank,
            lora_expert_rank=trainable.lora_expert_rank,
            lora_dropout=float(getattr(trainable, "lora_dropout", 0.0) or 0.0))))
        # run_qwen3_8_next.py: each SGLang engine is TP=EP=engine GPUs (TP8/EP8 full)
        engine = str(parallel.rollout_num_gpus_per_engine)
        for flag in ("--sglang-tp-size", "--sglang-ep-size"):
            if flag not in values:
                values.extend((flag, engine))
    if recipe.name == RECIPE_GLM5_NEXT:
        # the plugin spec replaces the Bridge-geometry model flags
        # experts and dense-layer count from the checkpoint (a cut-down model differs from GLM5.3-Flash)
        hf_config = Path(config.hf_checkpoint) / "config.json"
        shape = (glm5_next.checkpoint_geometry(json.loads(hf_config.read_text()))
                 if hf_config.is_file() else {})
        shape["num_layers"] = geometry.num_layers
        values = glm5_next.apply_recipe(values, glm5_next.model_argv(
            **shape, transformer_impl=recipe.transformer_impl))
    if recipe.name == RECIPE_KIMI_K31:
        # the plugin spec replaces the Bridge-geometry model flags (INTERFACES 2.5 / Y3
        # ports_recipe): raw bridge mode, frozen router bias, the selection file, routing
        # replay; the layer count follows the checkpoint (Kimi-K3-4layer smoke model)
        lora = trainable.parameter_mode == "lora"
        if not lora and not getattr(recipe, "expert_selection", None):
            raise MilesConfigError("the kimi_k31 recipe needs --rl-expert-selection PATH")
        if lora:
            values = _kimi_k31_lora_values(values, trainable, recipe)
        try:
            values = kimi_k31_recipe.apply_recipe(values, kimi_k31_recipe.recipe_argv(
                selection_path=None if lora else recipe.expert_selection, ep_size=parallel.expert_parallel,
                context_parallel_size=int(getattr(parallel, "context_parallel", 1) or 1),
                seq_length=batch.seq_len, extra_argv=extra_argv,
                num_layers=geometry.num_layers,
                first_k_dense_replace=_leading_dense_layers(geometry.moe),
                routed_expert_qat=getattr(recipe, "routed_expert_qat", None),
                hf_model_type=kimi_hf_model_type,
                num_experts=getattr(geometry.moe, "num_experts", None)))
        except kimi_k31_recipe.KimiK31RecipeError as error:
            raise MilesConfigError(f"kimi_k31 recipe: {error}") from None
        values = _full_recompute_for_mxfp4_experts(values)
    values.extend(extra_argv)
    train_extra = train_extra_argv(recipe.name, os.environ.get(TRAIN_EXTRA_ENV, ""))
    if any(flag in train_extra for flag in PIPELINE_SPLIT_FLAGS):
        check_pipeline_split(geometry.num_layers, parallel.pipeline_parallel,
                             parallel.uneven_pipeline_layers, (*extra_argv, *train_extra))
    values.extend(train_extra)
    reject_fault_tolerance_flags(values[1:])  # generated argv never carries FT flags
    _refuse_decoupled_ppo_breakers(values)  # also when the spec emitted the flag
    # IR-1: the final argv never pairs the session server with partial rollout
    # (also re-checked on the parsed namespace by validate_parsed_args).
    check_session_server_partial_rollout(agent.use_session_server, "--partial-rollout" in values)

    return MilesLaunchArgs(
        argv=tuple(values),
        placement=request,
        algorithm_sha256=algorithm.sha256(),
        rollout_overlap=rollout_overlap,
        runtime_attrs={
            **algorithm.to_legacy_runtime_attrs(),
            # only a quantized tier adds the attribute (tier 0 keeps the argv golden)
            **({"yeto_rollout_quant_tier": int(serving.rollout_quant_tier)}
               if getattr(serving, "rollout_quant_tier", 0) else {}),
            **({"yeto_rollout_quant_fp8_source": serving.rollout_quant_fp8_source}
               if getattr(serving, "rollout_quant_fp8_source", FP8_SOURCE_ENGINE) != FP8_SOURCE_ENGINE else {}),
            **_full_parameter_optimizer_attrs(trainable, config.train_memory,
                                              full_parameter_sync=full_parameter_sync,
                                              data_parallel=config.parallel.data_parallel,
                                              builds_trainer=not _is_sampler_island(sample_buffer)),
            # rl-algo-critic-family 4.1: receipt init source (warm-up product)
            **({"yeto_rl_critic_init_sha256": config.algorithm.critic.init_sha256}
               if getattr(config.algorithm, "critic", None) is not None
               and algorithm.execution.needs_critic else {}),
        },
        algorithm=algorithm,
        absorbed_flags=dict(absorbed),
    )


# Miles builds the HF snapshot exporter (actor.export_hf) only when one of its HF
# outputs is set. yeto never calls save_model, so this path is never written; the
# trainer's policy releases call export_hf with their own directory.
RELEASE_EXPORTER_SAVE_HF = "/tmp/yeto-rl-unused-save-hf/rollout_{rollout_id}"


SAMPLER_ONLY_FLAG = "--debug-rollout-only"


# Megatron's --optimizer-cpu-offload runs a HybridDeviceOptimizer under the
# distributed optimizer and requires the precision-aware optimizer path (its
# validate_args asserts it); the FP32 masters and moments stay FP32 (Megatron's
# default dtypes), which the cut's name-keyed optimizer export carries.
OPTIMIZER_OFFLOAD_ARGV = ("--use-distributed-optimizer", "--use-precision-aware-optimizer",
                          "--optimizer-cpu-offload", "--overlap-cpu-optimizer-d2h-h2d")
# --rl-precision-aware-optimizer (no-sync islands only): bf16 main grads and bf16
# Adam moments on the device; Megatron keeps the FP32 masters as 16-bit remainders
# of the bf16 params (store_param_remainders, its default). A measurement job
# ran exactly these flags on 8 x H100 80 GB (Nemotron-3 Nano TP1/EP8/DP8, 8192 tokens
# per GPU: PyTorch peak 64.6 GiB allocated, actor MFU 16.7%).
PRECISION_AWARE_ARGV = ("--use-precision-aware-optimizer", "--main-grads-dtype", "bf16",
                        "--exp-avg-dtype", "bf16", "--exp-avg-sq-dtype", "bf16")


def _gradient_dtype_argv(memory: Any) -> tuple[str, ...]:
    """fp32 gradient accumulation by default; none of it with bf16 main grads
    (precision-aware optimizer) and ``--grad-reduce-in-bf16`` with --rl-grad-reduce-bf16."""
    if getattr(memory, "precision_aware_optimizer", False):
        return ()  # PRECISION_AWARE_ARGV sets --main-grads-dtype bf16
    if getattr(memory, "grad_reduce_bf16", False):
        return ("--grad-reduce-in-bf16",)
    return ("--accumulate-allreduce-grads-in-fp32",)


def megatron_moe_layer_freq(layer_freq: Any) -> str:
    """Megatron's ``--moe-layer-freq`` value: an int, or a per-layer 0/1 mask.

    Megatron's ``moe_freq_type`` accepts a list only as a pattern of digits,
    commas, brackets, ``+``, ``*`` and parentheses (``_eval_pattern``): no
    spaces. ``str([0, 1])`` has spaces and is rejected (Nemotron-H
    120B: ``invalid moe_freq_type value``), so a mask is written ``[0,1,...]``.
    """
    if isinstance(layer_freq, (list, tuple)):
        return "[" + ",".join(str(int(value)) for value in layer_freq) + "]"
    return str(layer_freq)


# Trainer-speed flags for MFU measurement on the Nemotron-H recipe, from the
# island environment (launcher --modal-env). Only allow-listed Megatron flags pass;
# none of them changes the parallel layout (TP/EP/PP/CP/DP).
TRAIN_EXTRA_ENV = "YETO_MILES_TRAIN_EXTRA_ARGV"
TRAIN_EXTRA_FLAGS = frozenset({
    "--recompute-granularity", "--recompute-modules", "--recompute-method",
    "--recompute-num-layers", "--moe-token-dispatcher-type", "--moe-permute-fusion",
    "--cross-entropy-loss-fusion", "--timing-log-level", "--timing-log-option",
    "--moe-flex-dispatcher-backend",
    # Miles computes the loss log-probabilities over the sequence in chunks
    # of N tokens instead of one fp32 full-vocabulary block (a full block ran out of
    # memory there on a long agentic sample: 11.94 GiB for one block)
    "--log-probs-chunk-size",
    # torch profiler of the Miles train loop (Megatron --use-pytorch-profiler and its
    # step window, Miles --profile-target); the trace goes to --tensorboard-dir
    "--use-pytorch-profiler", "--profile-step-start", "--profile-step-end", "--profile-ranks",
    "--profile-target", "--tensorboard-dir",
    # YETO_RL_NAN_SKIP: Megatron's DDP check raises on a NaN grad; off, Miles skips that optimizer step
    "--no-check-for-nan-in-loss-and-grad",
    # kimi_k31 first_stage_grad (PP stage 0 expert grads under full recompute on
    # images before Miles k31/stage0-grad); chains another hook via its env var
    "--custom-megatron-before-train-step-hook-path",
    # an uneven pipeline split of a layer count PP divides (balance stage memory);
    # the caller checks the split with check_pipeline_split
    *PIPELINE_SPLIT_FLAGS,
})
# Recipes that take TRAIN_EXTRA_ENV.
TRAIN_EXTRA_RECIPES = ("nemotron_h", "kimi_k31")


def _nemotron_h_train_flags(recipe_module: Any, config: Any) -> tuple[str, ...]:
    """Nemotron-H training defaults (``models.nemotron_h.train_flags``); prints its launch warnings."""

    memory, parallel = config.train_memory, config.parallel
    for warning in recipe_module.train_warnings(
        samples_per_step=config.batch.global_batch,
        optimizer_offload_fraction=float(getattr(memory, "optimizer_offload_fraction", 0.0) or 0.0),
        trainer_gpus=parallel.actor_num_nodes * parallel.actor_num_gpus_per_node,
    ):
        print(f"[rl] WARNING: {warning}", flush=True)
    return recipe_module.train_flags(
        tuple(getattr(config.model_recipe, "model_flags", ())),
        activation_recompute=bool(memory.activation_recompute),
        max_tokens_per_gpu=memory.max_tokens_per_gpu,
        tensor_parallel=int(parallel.tensor_parallel),
        context_parallel=int(getattr(parallel, "context_parallel", 1) or 1),
        seq_len=int(config.batch.seq_len),
        deepep=recipe_module.deepep_available(),
    )


def train_extra_argv(recipe_name: str, raw: str) -> tuple[str, ...]:
    """Parse ``YETO_MILES_TRAIN_EXTRA_ARGV`` (space- or comma-separated); empty unless set.
    Only the :data:`TRAIN_EXTRA_RECIPES` recipes take it."""
    import shlex

    tokens = shlex.split((raw or "").replace(",", " "))  # commas: one shell word
    if not tokens:
        return ()
    if recipe_name not in TRAIN_EXTRA_RECIPES:
        raise MilesConfigError(f"{TRAIN_EXTRA_ENV} is only for the {', '.join(TRAIN_EXTRA_RECIPES)} recipes")
    bad = [t for t in tokens if t.startswith("--") and t not in TRAIN_EXTRA_FLAGS]
    if bad or not tokens[0].startswith("--"):
        raise MilesConfigError(f"{TRAIN_EXTRA_ENV}: flags not allowed: {bad or tokens[:1]}")
    return tuple(tokens)


def _kimi_k31_lora_values(values: list[str], trainable: Any, recipe: Any) -> list[str]:
    """kimi_k31 ``--tuning lora`` (``yeto.rl.engine.kimi_k31_lora``): the generic LoRA
    flags plus the recipe's. Routed-expert targets add ``--experts-shared-outer-loras``
    (the upstream recipe's expert layout). ``--lora-base-cpu-backup`` is dropped: Miles'
    16xB300 recipe has no host room for SGLang's base copy next to the trainer's offloaded
    weights and re-sends the base at every colocated sync instead."""
    from yeto.rl.engine import kimi_k31_lora

    try:
        kimi_k31_lora.check_tuning("lora", expert_selection=getattr(recipe, "expert_selection", None),
                                   routed_expert_qat=getattr(recipe, "routed_expert_qat", None),
                                   lora_targets=trainable.lora_targets)
    except kimi_k31_lora.KimiK31LoraError as error:
        raise MilesConfigError(f"kimi_k31 recipe: {error}") from None
    values = [v for v in values if v != "--lora-base-cpu-backup"]
    return [*values, *kimi_k31_lora.extra_lora_argv(trainable.lora_targets)]


FULL_RECOMPUTE_ARGV = ("--recompute-granularity", "full", "--recompute-method", "uniform",
                       "--recompute-num-layers", "1")


def _full_recompute_for_mxfp4_experts(values: list[str]) -> list[str]:
    """kimi_k31: the frozen routed experts stay MXFP4 and every GEMM dequantizes a
    whole layer of them to BF16 (Miles ``mxfp4_experts``). Without full recompute
    autograd keeps each layer's dequantized weights for the backward, about 7 GB per
    MoE layer per GPU at EP8, which no pipeline stage of the full model fits; full
    recompute frees them after each layer's backward. The recipe turns it on when the
    launch did not (as --rl-activation-recompute would) and refuses any other
    recompute granularity."""
    granularity = None
    for flag, value in zip(values, values[1:]):
        if flag == "--recompute-granularity":
            granularity = value
    if granularity is None:
        logging.getLogger(__name__).warning("kimi_k31: turning on full uniform activation recompute (the frozen MXFP4 "
                       "experts' dequantized weights would otherwise stay alive until the backward)")
        return [*values, *FULL_RECOMPUTE_ARGV]
    if granularity != "full":
        raise MilesConfigError(
            f"kimi_k31 recipe: --recompute-granularity {granularity} keeps the frozen MXFP4 experts' "
            "dequantized weights for the backward; use full recompute (--rl-activation-recompute)")
    return values


def train_memory_argv(memory: Any) -> tuple[str, ...]:
    """Megatron full uniform recompute (one layer per chunk), Miles dynamic
    batching (the same values the validated Flash-Next recipe uses) and the
    optimizer host offload."""

    values: list[str] = []
    if memory.activation_recompute:
        values.extend(FULL_RECOMPUTE_ARGV)
    if memory.max_tokens_per_gpu is not None:
        values.extend(("--use-dynamic-batch-size",
                       "--max-tokens-per-gpu", str(memory.max_tokens_per_gpu)))
    fraction = float(getattr(memory, "optimizer_offload_fraction", 0.0) or 0.0)
    if fraction > 0:
        values.extend((*OPTIMIZER_OFFLOAD_ARGV, "--optimizer-offload-fraction", repr(fraction)))
    if getattr(memory, "precision_aware_optimizer", False):
        values.extend(PRECISION_AWARE_ARGV)
    return tuple(values)


def harbor_rollout_argv(agent_function: str, *, sample_buffer: Any = None) -> tuple[str, ...]:
    """The Harbor agent's rollout function: the stock one plus boat VM cleanup at
    ``RolloutExecutor.dispose()`` (``harbor_rollout``). Without it the pool's idle
    VMs outlive the island until their boat TTL. ``()`` for any other agent."""
    from .harbor_agent import AGENT_FUNCTION, ROLLOUT_FUNCTION

    if agent_function != AGENT_FUNCTION:
        return ()
    if sample_buffer is not None:
        raise MilesConfigError("the Harbor agent's rollout function and the sample buffer rollout "
                               "both replace Miles' rollout function; use one of them")
    return ("--rollout-function-path", ROLLOUT_FUNCTION)


def sample_buffer_argv(role: str, *, max_policy_age: int) -> tuple[str, ...]:
    """A sampler writes the buffer and builds no trainer model
    (``--debug-rollout-only``: rollout GPUs only, engines serve the base weights);
    a trainer reads the buffer and can export HF snapshots for its releases; a
    convertible node does both with one switchable rollout function, and
    releases what it trains too (the other nodes load the releases)."""
    from .buffer_rollout import ROLLOUT_LOG_FUNCTION, SAMPLER, rollout_argv

    argv = rollout_argv(role, max_policy_age=max_policy_age)
    if role == SAMPLER:
        return (*argv, SAMPLER_ONLY_FLAG)
    # a reader pads short rounds: its rollout metrics leave the padding samples out
    return (*argv, "--save-hf", RELEASE_EXPORTER_SAVE_HF,
            "--custom-rollout-log-function-path", ROLLOUT_LOG_FUNCTION)


def _is_sampler_island(sample_buffer: Any) -> bool:
    from .buffer_rollout import SAMPLER

    return sample_buffer is not None and sample_buffer.role == SAMPLER


def _full_parameter_optimizer_attrs(trainable: Any, memory: Any, *, full_parameter_sync: bool,
                                    data_parallel: int = 1, builds_trainer: bool = True) -> dict[str, Any]:
    """Pick the optimizer layout the full-parameter sync reads its FP32 masters from.

    The ports full-parameter sync (presets ``full-parameter`` /
    ``sao-streaming-full``) reads one FP32 master per parameter. At DP>1 one
    rank of each data-parallel replica group owns (exports) a parameter and
    the other replicas apply its bytes; that needs a Miles pin with
    data-parallel replicas (``run_config_rules.full_parameter_dp_replica_gap``)
    and no optimizer offload. The optimizer at DP>1 follows
    ``--rl-dp-optimizer`` (:func:`_dp_optimizer_attrs`). A sampler island
    (``builds_trainer=False``) has no trainer and no such check.

    * No optimizer offload: the non-distributed optimizer, whose masters are
      whole tensors. Miles' ``set_default_megatron_args`` turns the
      distributed optimizer on for Adam regardless of argv, so the attribute
      is applied in ``preprocess_args``, after Miles' defaults and before
      Megatron's ``validate_args``.
    * ``--rl-optimizer-offload-fraction`` > 0: Megatron's distributed +
      precision-aware optimizer with a HybridDeviceOptimizer
      (``OPTIMIZER_OFFLOAD_ARGV``). At DP=1 each parameter's FP32 master is
      the hybrid optimizer's whole FP32 copy (host-pinned when offloaded);
      Miles reads it and, on a cut apply, writes both that copy and the BF16
      model view (``full_parameter_state._propagate_master_writes``). Needs
      a Miles pin with the pins feature ``distopt_full_parameter`` (ports-v1
      :data:`MILES_DISTOPT_FULL_PARAMETER_COMMIT` on, k31 from B2 on);
      DP=1 only (at DP>1 a distributed-optimizer master is a partial range).

    A no-sync trainer island (samples from the store) has no such reader:
    it keeps Miles' distributed optimizer, so DP replicas shard the optimizer
    state, and its cuts carry each rank's owned ranges (``cut_plugin``).
    """

    if trainable.parameter_mode != "full" or not full_parameter_sync:
        return {}
    if getattr(memory, "precision_aware_optimizer", False):
        raise MilesConfigError(
            "the full-parameter sync reads FP32 optimizer masters, which the precision-aware "
            "optimizer keeps only as remainders inside TE FusedAdam: drop --rl-precision-aware-optimizer "
            "(no-sync islands only)")
    offload = float(getattr(memory, "optimizer_offload_fraction", 0.0) or 0.0) > 0
    if data_parallel != 1 and builds_trainer:
        if (gap := full_parameter_dp_replica_gap()) is not None:
            raise MilesConfigError(f"the full-parameter sync at DP {data_parallel} needs {gap}")
        if offload:
            raise MilesConfigError(
                f"the full-parameter sync at DP {data_parallel} needs its FP32 masters on the device: "
                "drop --rl-optimizer-offload-fraction")
        return _dp_optimizer_attrs(getattr(memory, "dp_optimizer", "auto"), data_parallel)
    if offload:
        if not _miles_reads_distributed_optimizer_masters():
            raise MilesConfigError(
                "--rl-optimizer-offload-fraction with the full-parameter sync needs Miles "
                f"{MILES_DISTOPT_FULL_PARAMETER_COMMIT[:9]} or later (reads distributed-optimizer masters); "
                f"the pinned image ships {MILES_NEXT_COMMIT[:9]} (or use a no-sync trainer island)")
        return {}
    return {"use_distributed_optimizer": False}


def _dp_optimizer_attrs(choice: str, data_parallel: int) -> dict[str, Any]:
    """The optimizer of a full-parameter trainer at DP>1 (``--rl-dp-optimizer``).

    ``distributed``: Miles' default distributed optimizer; each rank keeps one
    FP32 range of every parameter and the sync reads and writes those ranges
    (needs ``run_config_rules.full_parameter_dp_sharded_master_gap`` to be
    None). ``replicated``: the non-distributed optimizer, whole FP32 masters
    on every rank (the replica path). ``auto``: distributed when the
    pin supports it. Both give the same synced values (Miles CPU tests); the
    distributed one keeps 1/DP of the masters and Adam moments per rank."""

    gap = full_parameter_dp_sharded_master_gap()
    if choice == "replicated" or (choice == "auto" and gap is not None):
        return {"use_distributed_optimizer": False}
    if gap is not None:
        raise MilesConfigError(f"--rl-dp-optimizer distributed at DP {data_parallel} needs {gap}")
    return {}


def _miles_reads_distributed_optimizer_masters() -> bool:
    """The pinned Miles carries the pins feature ``distopt_full_parameter``
    (ports-v1 56190a5a8 on, k31 from B2 on)."""
    from yeto.rl.adapters.miles import pins

    return pins.pin_has("distopt_full_parameter", MILES_NEXT_COMMIT)


def _resolve_tito_parsers(model: str) -> tuple[str | None, str | None]:
    from miles.utils.chat_template_utils import resolve_reasoning_and_tool_call_parser

    return resolve_reasoning_and_tool_call_parser(model)


# --------------------------------------------------------------------------
# Parse + post-normalization checks
# --------------------------------------------------------------------------


def apply_runtime_attrs(args: Any, launch: MilesLaunchArgs) -> Any:
    """Legacy carrier: every runtime attribute as one attribute on a mutable namespace."""
    for name, value in launch.runtime_attrs.items():
        setattr(args, name, value)
    return args


def split_runtime_attrs(launch: MilesLaunchArgs) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(miles_part, yeto_part)`` of ``launch.runtime_attrs``.

    The Miles part is set on the raw namespace in ``preprocess_args`` (before
    Miles validates ownership): Megatron fields such as
    ``use_distributed_optimizer`` and the MIS hook's own fields (``tis_*``,
    ``rs_*``, ``use_rs``; collected by Miles into the hook config, see
    :func:`mis_hook_config_class`). The yeto part goes to the ``IslandArgs``
    context after parsing. The table itself (and so the golden files) does
    not change.
    """
    from yeto.rl.adapters.miles.island_args import is_yeto_name

    miles_part = {k: v for k, v in launch.runtime_attrs.items() if not is_yeto_name(k)}
    yeto_part = {k: v for k, v in launch.runtime_attrs.items() if is_yeto_name(k)}
    return miles_part, yeto_part


def _preprocess_runtime_attrs(args: Any, launch: MilesLaunchArgs) -> Any:
    for name, value in split_runtime_attrs(launch)[0].items():
        setattr(args, name, value)
    return args


# yeto's MIS attributes (``mismatch_correction.mis_config`` keys): on the new Miles
# argument system a hook declares its own fields through ``config_class``.
MIS_HOOK_FIELDS: tuple[str, ...] = (
    "rs_level", "rs_lower_bound", "rs_upper_bound", "rs_veto_threshold",
    "tis_batch_normalize", "tis_level", "tis_lower_bound", "tis_mode", "tis_upper_bound", "use_rs",
)


def attach_mis_hook_config() -> None:
    """Declare the vendored MIS hook's fields to Miles (``config_class``) before ``parse_args``.

    The hook lives in the engine-neutral core, which may not import Miles, so
    the adapter attaches the class from here; without the Miles schema (the
    pinned fork) the attributes stay on the namespace and nothing is attached.
    """
    config_class = mis_hook_config_class()
    if config_class is None:
        return
    try:
        from yeto.rl.algos.vendor import miles_mis
    except ImportError:  # no torch on this host: the hook cannot be used here anyway
        return
    miles_mis.compute_mis_weights_with_cp.config_class = config_class  # type: ignore[attr-defined]


def mis_hook_config_class() -> type | None:
    """The ``config_class`` of the vendored MIS hook, or ``None`` on a Miles without the schema.

    Fields carry no ``Arg`` (no CLI flag): preprocess sets them on the namespace,
    ``resolve_custom_function_configs`` moves them into the hook config and the
    hook's view exposes them under the same names, so the vendored code reads
    ``args.tis_mode`` unchanged.
    """
    try:
        from miles.utils.args.schema import BaseConfig
    except ImportError:
        return None
    namespace: dict[str, Any] = {"__annotations__": {}, "__module__": __name__,
                                 "__doc__": "yeto MIS (mixed importance sampling) hook configuration."}
    for name in MIS_HOOK_FIELDS:
        namespace["__annotations__"][name] = Any
        namespace[name] = None
    return type("YetoMisConfig", (BaseConfig,), namespace)


def validate_parsed_args(
    args: Any,
    launch: MilesLaunchArgs,
    *,
    num_cells: Callable[[Any], int] | None = None,
) -> None:
    """Refuse FT semantics, multi-cell groups and rewritten placement."""

    if (
        launch.runtime_attrs.get("use_distributed_optimizer") is False
        and getattr(args, "use_distributed_optimizer", False)
    ):
        raise MilesConfigError(
            "Miles re-enabled the distributed optimizer; full-parameter mode needs whole FP32 masters"
        )
    if launch.rollout_overlap and not getattr(args, "fully_async", False):
        raise MilesConfigError("Miles did not enable fully_async for the overlapped rollout")
    allowed = {"fully_async"} if launch.rollout_overlap else set()
    enabled = [name for name in _FT_NAMESPACE_OFF
               if name not in allowed and getattr(args, name, None)]
    if enabled:
        raise FaultToleranceArgsError(
            f"Miles normalized the arguments into fault-tolerance mode ({enabled}); "
            "the ports engine refuses FT semantics"
        )
    # IR-1: Miles normalization may set either flag; refuse the pair here too.
    check_session_server_partial_rollout(
        getattr(args, "use_session_server", False), getattr(args, "partial_rollout", False)
    )
    check_harness_reward_scope(getattr(args, "yeto_harness_reward_scope", None))
    counter = num_cells or _upstream_num_cells
    cells = counter(args)
    if cells != 1:
        raise SingleCellError(f"actor TrainGroup would have {cells} cells; ports require exactly 1")
    if SAMPLER_ONLY_FLAG in launch.argv:
        # A sampler island asked for --debug-rollout-only: Miles places the
        # rollout engines only and builds no trainer, so the trainer partition of the
        # request is unused by design, not rewritten.
        return
    check_placement_not_rewritten(launch.placement, args)


def _kimi_hf_model_type(hf_checkpoint: Any) -> str:
    """``model_type`` of the checkpoint's config.json (``kimi_k31`` when unreadable:
    the K3.1 spec is the recipe's default)."""
    from yeto.rl.adapters.miles.models.kimi_k31 import ports_recipe as kimi_k31_recipe

    try:
        config = json.loads((Path(str(hf_checkpoint)) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return kimi_k31_recipe.MODEL_NAME
    return str(config.get("model_type") or kimi_k31_recipe.MODEL_NAME)


def _upstream_num_cells(args: Any) -> int:
    from yeto.rl.adapters.miles.island_args import miles_config

    try:
        from miles.ray.specs.train import ACTOR_ROLE, compute_trainer_num_cells
    except ImportError:
        return 1 if not getattr(args, "indep_dp", False) else -1
    return _call_num_cells(compute_trainer_num_cells, miles_config(args), ACTOR_ROLE)


def _call_num_cells(compute_trainer_num_cells, config: Any, role: Any) -> int:
    """Both pin generations of ``compute_trainer_num_cells``: the ports-v1 fork
    takes ``(config, *, role)``; upstream main (the k31 lineage,
    ``miles.utils.megatron_args_utils``) takes ``(args, *, total_gpus)`` with the
    trainer's GPU count from ``compute_trainer_total_gpus(config, role=)``."""
    import inspect

    params = inspect.signature(compute_trainer_num_cells).parameters
    if "total_gpus" in params:
        from miles.utils.args.trainer_utils import compute_trainer_total_gpus

        return compute_trainer_num_cells(config, total_gpus=compute_trainer_total_gpus(config, role=role))
    return compute_trainer_num_cells(config, role=role)


TEST_DEBUG_ROLLOUT_ENV = "YETO_TEST_MILES_LOAD_DEBUG_ROLLOUT_DATA"


def _apply_test_debug_rollout(args: Any) -> Any:
    """Test-only: train on a recorded rollout dump (Miles ``--load-debug-rollout-data``,
    which implies ``--debug-train-only``; no inference engines). Set before Miles
    validates the namespace so its derived placement flags follow. Unset: no change."""

    path = os.environ.get(TEST_DEBUG_ROLLOUT_ENV)
    if path:
        if "{rollout_id}" not in path:
            raise MilesConfigError(f"{TEST_DEBUG_ROLLOUT_ENV} must contain {{rollout_id}}")
        print(f"[rl] TEST ONLY: --load-debug-rollout-data {path} (debug_train_only)", flush=True)
        args.load_debug_rollout_data = path
        backend = os.environ.get("YETO_TEST_MILES_ATTENTION_BACKEND")
        if backend:  # e.g. "fused" on a pre-Hopper test GPU (flash_attn.cute needs sm90)
            from megatron.core.transformer.enums import AttnBackend

            args.attention_backend = AttnBackend[backend]
            print(f"[rl] TEST ONLY: attention_backend={backend}", flush=True)
    return args


def parse_miles_args(launch: MilesLaunchArgs) -> Any:
    """Run upstream ``parse_args`` on the translated argv, then validate it.

    Returns the driver's :class:`~yeto.rl.adapters.miles.island_args.IslandArgs`
    over the parsed Miles configuration: Miles' names read through, yeto's
    runtime attributes sit in its context (and, on a mutable namespace, also
    on the namespace itself).
    """
    from yeto.rl.adapters.miles.island_args import IslandArgs

    previous = sys.argv
    try:
        sys.argv = list(launch.argv)
        from miles.utils.arguments import parse_args

        attach_mis_hook_config()
        args = parse_args(
            preprocess_args=lambda ns: _apply_test_debug_rollout(_preprocess_runtime_attrs(ns, launch))
        )
    finally:
        sys.argv = previous
    island = IslandArgs(args)
    for name, value in split_runtime_attrs(launch)[1].items():
        setattr(island, name, value)
    validate_parsed_args(island, launch)
    return island
