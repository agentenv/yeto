"""RLRunConfig + AlgorithmSpec -> upstream Miles argv (task 3.1, design D5/D7/D8).

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
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..algorithm import (
    AlgorithmSpecError,
    BOUNDED_NONZERO_STD_FILTER,
    STOCK_NONZERO_STD_FILTER,
    AlgorithmSpec,
)
from ..run_config import LR_SCHEDULE_FLAGS
from .algorithm_flags import (
    OBJECTIVE_FLAGS,
    absorb_extra_argv,
    algorithm_argv,
    mapped_flags,
    objective_flags,
)
from .placement import PlacementRequest, check_placement_not_rewritten

ROLLOUT_META_HOOK_PATH = (
    "yeto.rl.engine.miles_adapter.rollout_meta_hook.extract_rollout_metadata"
)
TRAINED_GROUPS_HOOK_PATH = (
    "yeto.rl.engine.miles_adapter.rollout_meta_hook.record_trained_groups"
)
POLICY_BUFFER_FILTER_PATH = (
    "yeto.rl.engine.miles_adapter.rollout_meta_hook.policy_buffer_filter"
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
        # rl-infra-spec 2.1 (fixed partition): the transfer mode and the
        # fork-M1 role->bundle map are decided by the placement translation.
        "--update-weight-transfer-mode",
        "--yeto-placement-map",
        # LR schedule is decided by RLRunConfig.algorithm.lr_schedule
        # (legacy rejects the same LR_SCHEDULE_FLAGS in learner.py).
        *LR_SCHEDULE_FLAGS,
        # Every objective-changing algorithm flag (rl-algorithm-capabilities
        # D3): mapped ones only enter through absorption into AlgorithmSpec,
        # unmapped ones are refused.
        *OBJECTIVE_FLAGS,
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


def _check_parameter_mode(value, _config):
    return None if value == "lora" else "only the LoRA parameter layout is implemented"


def _check_lora_targets(value, _config):
    if value == "attention-routed-experts":
        return "routed-expert (DSV4 clone) LoRA is legacy-only"
    return None


def _check_recipe(value, _config):
    from ..run_config import RECIPE_DEEPSEEK_V4_FLASH

    if value == RECIPE_DEEPSEEK_V4_FLASH:
        return "the DeepSeek V4 recipe is legacy-only"
    return None


def _check_expert_full(value, _config):
    return "expert-full tuning is legacy-only" if value else None


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
    "model_recipe.name": _check_recipe,
    "model_recipe.provider_class": _ok,
    "model_recipe.training_attention_backend": _ok,
    "model_recipe.gdn.gated_delta_net": _ok,
    "model_recipe.gdn.qkv_format": _ok,
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
    # Derived by Megatron from world size / TP / PP; consumed, not passed.
    "parallel.data_parallel": _ok,
    "parallel.rollout_num_gpus_per_engine": _ok,
    "parallel.dedicated_rollout_gpus": _ok,
    "parallel.visible_gpus_per_node": _ok,
    "parallel.uneven_pipeline_layers": _ok,
    "parallel.standby_gpus": _ok,
    "parallel.rollout_cell_names": _ok,
    "trainable.parameter_mode": _check_parameter_mode,
    "trainable.lora_rank": _ok,
    "trainable.lora_dropout": _ok,
    "trainable.lora_targets": _check_lora_targets,
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
    "algorithm.rollout_seed": _ok,
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
    "agent.custom_agent_function_path": _must_be_none(
        "upstream Miles has no --custom-agent-function-path"
    ),
    "agent.agent_max_seq_len": _must_be_none(
        "--max-seq-len only accompanies the fork-only custom agent function"
    ),
    "agent.dynamic_sampling_filter_path": _ok,  # reconciled with AlgorithmSpec
    "agent.use_rollout_routing_replay": _ok,
    "agent.use_session_server": _ok,
    "agent.session_server_ip": _ok,
    "agent.session_server_port": _ok,
    "agent.tito_model": _ok,
    "agent.tito_allowed_append_roles": _must_be_none(
        "upstream Miles has no --tito-allowed-append-roles"
    ),
    # The driver owns the loop and the outer sync; no Miles callback needed.
    "yeto_policy_sync": _ok,
    "distributed_timeout_minutes": _ok,
    "deterministic_trainer": _ok,
    "trainer_dp_edges": _ok,
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
    requested = config.agent.dynamic_sampling_filter_path
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
        )
    return PlacementRequest(
        kind="fixed-partition",
        trainer_gpus=trainer,
        rollout_gpus=int(parallel.dedicated_rollout_gpus),
        gpus_per_engine=parallel.rollout_num_gpus_per_engine,
        standby_gpus=int(getattr(parallel, "standby_gpus", 0) or 0),
        rollout_cell_names=tuple(getattr(parallel, "rollout_cell_names", ()) or ()),
    )


def _flags(tokens: Sequence[str]) -> Iterator[str]:
    for token in tokens:
        flag = token.split("=", 1)[0]
        if flag.startswith("--"):
            yield flag


def reject_fault_tolerance_flags(tokens: Sequence[str]) -> None:
    for flag in _flags(tokens):
        if flag in FAULT_TOLERANCE_FLAGS:
            raise FaultToleranceArgsError(
                f"{flag} enables Miles fault-tolerance semantics, which the ports "
                "engine does not support"
            )


def check_extra_argv(extra_argv: Sequence[str], algorithm: AlgorithmSpec | None = None):
    """Refuse FT/adapter-owned flags; absorb mapped algorithm flags.

    Mapped algorithm flags (``algorithm_flags.MAPPINGS``) are absorbed into
    ``algorithm`` (conflicts raise, naming flag and both values); other
    objective-changing flags are refused as not yet in the algorithm spec.
    With ``algorithm`` returns ``(spec, remaining_argv, absorbed)``.
    """

    reject_fault_tolerance_flags(extra_argv)
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


def translate_run_config(
    config: Any,
    algorithm: AlgorithmSpec,
    *,
    extra_argv: Sequence[str] = (),
    tito_parser_resolver: Callable[[str], tuple[str | None, str | None]] | None = None,
) -> MilesLaunchArgs:
    """Pure translation of an ``RLRunConfig`` into upstream Miles argv."""

    if not isinstance(algorithm, AlgorithmSpec):
        raise TypeError("algorithm must be an AlgorithmSpec")
    check_config_mapped(config)
    algorithm, extra_argv, absorbed = check_extra_argv(extra_argv, algorithm)
    problems = algorithm.rejections()
    if problems:
        raise MilesConfigError("algorithm spec rejected: " + "; ".join(problems))
    from ..algorithm import launch_problems

    problems = launch_problems(algorithm, {
        "rollout_batch_size": config.batch.groups_per_round,
        "rollout_max_response_len": config.batch.rollout_max_response_len,
        "context_parallel_size": 1,  # ports emits --context-parallel-size 1
        "multi_lora": any(t.split("=", 1)[0] == "--multi-lora" for t in extra_argv),
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

    from ..run_config import RECIPE_QWEN3_5, lr_schedule_argv

    geometry = config.geometry
    parallel = config.parallel
    trainable = config.trainable
    batch = config.batch
    recipe = config.model_recipe
    serving = config.serving
    agent = config.agent
    columns = config.data.columns
    request = placement_request(config)

    if request.kind == "colocated":
        placement_values = ["--colocate"]
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
            # fork-M1 (--yeto-placement-map): explicit role -> bundle map.
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
    else:
        model_name = recipe.provider_class

    values: list[str] = [
        "train.py",
        "--train-backend", "megatron",
        "--hf-checkpoint", config.hf_checkpoint,
        "--ref-load", config.ref_load,
        "--megatron-to-hf-mode", "bridge",
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
        # LoRA (upstream miles/utils/lora/arguments.py)
        "--lora-rank", str(trainable.lora_rank),
        "--lora-alpha", str(trainable.lora_rank),
        # "0" (default argv unchanged) unless --rl-lora-dropout
        "--lora-dropout", (format(trainable.lora_dropout, "g")
                           if getattr(trainable, "lora_dropout", 0.0) else "0"),
        "--lora-type", "canonical_lora",
        "--target-modules", ",".join(trainable.target_modules),
        # upstream applies the LoRA base CPU backup only under colocate
        # (utils/lora/utils.py:39-41); a LoRA fixed partition drops it.
        *(
            []
            if request.kind != "colocated" and trainable.parameter_mode == "lora"
            else ["--lora-base-cpu-backup"]
        ),
        "--sglang-max-lora-rank", str(trainable.lora_rank),
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
        "--context-parallel-size", "1",
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
        # E3 DP certification refuses --balance-data (reshard.reshard_problems):
        # dropped only when trainer DP-change edges are enabled.
        *(() if getattr(config, "trainer_dp_edges", False) else ("--balance-data",)),
        "--rollout-max-context-len", str(batch.seq_len),
        "--rollout-max-response-len", str(batch.rollout_max_response_len),
        # D3: metadata is extracted inside the rollout process
        "--rollout-sample-filter-path", TRAINED_GROUPS_HOOK_PATH,
        "--rollout-all-samples-process-path", ROLLOUT_META_HOOK_PATH,
        "--buffer-filter-path", POLICY_BUFFER_FILTER_PATH,
        "--custom-rm-path", _dotted_callable(config.algorithm.reward_function),
        # algorithm (D7): the only place AlgorithmSpec becomes Miles args
        "--advantage-estimator", algorithm.advantage_estimator,
        "--lr", str(config.algorithm.lr),
        "--accumulate-allreduce-grads-in-fp32",
        "--attention-softmax-in-fp32",
        "--attention-backend", recipe.training_attention_backend,
        "--no-gradient-accumulation-fusion",
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
    if algorithm.kl_coef is not None:
        values.extend(("--kl-coef", str(algorithm.kl_coef)))
    if dynamic_filter is not None:
        values.extend(("--dynamic-sampling-filter-path", dynamic_filter))
    # Non-default AlgorithmSpec v2 fields (empty for every v1 spec, so the
    # default GRPO argv is byte-identical to R0).
    values.extend(algorithm_argv(algorithm))

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
            ("--eval-max-context-len", evaluation.max_context_len),
        ):
            if value is not None:
                values.extend((flag, str(value)))
    if serving.deterministic_inference:
        values.append("--sglang-enable-deterministic-inference")
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
    if getattr(config, "deterministic_trainer", False):
        values.append("--deterministic-mode")  # Megatron deterministic kernels (E2 plan-v2 §0)
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
    if agent.use_session_server:
        values.append("--use-session-server")
        if agent.session_server_ip:
            values.extend(("--session-server-ip", agent.session_server_ip))
        if agent.session_server_port:
            values.append("--session-server-port")
            values.extend(str(port) for port in agent.session_server_port)
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
                "--moe-layer-freq", str(moe.layer_freq),
            )
        )
        if moe.shared_expert_intermediate_size is not None:
            values.extend(
                ("--moe-shared-expert-intermediate-size", str(moe.shared_expert_intermediate_size))
            )
    if geometry.multi_latent_attention:
        values.append("--multi-latent-attention")
        for name, value in geometry.mla_dims:
            values.extend((f"--{name.replace('_', '-')}", str(value)))
    values.extend(extra_argv)
    reject_fault_tolerance_flags(values[1:])  # generated argv never carries FT flags

    return MilesLaunchArgs(
        argv=tuple(values),
        placement=request,
        algorithm_sha256=algorithm.sha256(),
        runtime_attrs=dict(algorithm.to_legacy_runtime_attrs()),
        algorithm=algorithm,
        absorbed_flags=dict(absorbed),
    )


def _resolve_tito_parsers(model: str) -> tuple[str | None, str | None]:
    from miles.utils.chat_template_utils import resolve_reasoning_and_tool_call_parser

    return resolve_reasoning_and_tool_call_parser(model)


# --------------------------------------------------------------------------
# Parse + post-normalization checks
# --------------------------------------------------------------------------


def apply_runtime_attrs(args: Any, launch: MilesLaunchArgs) -> Any:
    for name, value in launch.runtime_attrs.items():
        setattr(args, name, value)
    return args


def validate_parsed_args(
    args: Any,
    launch: MilesLaunchArgs,
    *,
    num_cells: Callable[[Any], int] | None = None,
) -> None:
    """Refuse FT semantics, multi-cell groups and rewritten placement."""

    enabled = [name for name in _FT_NAMESPACE_OFF if getattr(args, name, None)]
    if enabled:
        raise FaultToleranceArgsError(
            f"Miles normalized the arguments into fault-tolerance mode ({enabled}); "
            "the ports engine refuses FT semantics"
        )
    counter = num_cells or _upstream_num_cells
    cells = counter(args)
    if cells != 1:
        raise SingleCellError(f"actor TrainGroup would have {cells} cells; ports require exactly 1")
    check_placement_not_rewritten(launch.placement, args)


def _upstream_num_cells(args: Any) -> int:
    try:
        from miles.ray.specs.train import ACTOR_ROLE, compute_trainer_num_cells
    except ImportError:
        return 1 if not getattr(args, "indep_dp", False) else -1
    return compute_trainer_num_cells(args, role=ACTOR_ROLE)


def parse_miles_args(launch: MilesLaunchArgs) -> Any:
    """Run upstream ``parse_args`` on the translated argv, then validate it."""

    previous = sys.argv
    try:
        sys.argv = list(launch.argv)
        from miles.utils.arguments import parse_args

        args = parse_args(preprocess_args=lambda ns: apply_runtime_attrs(ns, launch))
    finally:
        sys.argv = previous
    validate_parsed_args(args, launch)
    return args
