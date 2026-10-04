"""Engine-agnostic RL run configuration (rl-engine-ports task 2.4, design D8).

``resolve_rl_run_config`` turns one parsed learner namespace plus the Bridge
model provider into an immutable :class:`RLRunConfig`.  All validation and
derivation that does not depend on a particular RL engine lives here; each
engine then owns a pure translation of the config into its own launch
arguments (legacy: ``yeto.rl.learner.build_miles_argv``; ports:
``miles_adapter/config.py``).  Both paths start from the same resolved config,
so equivalence runs are guaranteed identical inputs.

This module is import-light: it never imports torch, ray or miles.

Slots filled by merged PRs (see migration-ledger.md):

* #59 -- ``DatasetColumns.prompt_column`` / ``label_column`` (source dataset
  column names; ``None`` keeps today's ``messages``/``prompt``/``input`` and
  ``label`` lookup).  Normalized keys handed to the engine stay
  ``messages``/``label``/``metadata``.
* #65 -- ``GdnRecipe``: the provider's ``gated_delta_net`` capability.  It
  selects the ``bshd`` qkv format and the ``qwen3_5`` recipe
  (``select_gdn_recipe``), in addition to the pinned id+revision.
"""

from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path
from typing import Any

PARAMETER_MODES = frozenset({"lora", "full"})
RECIPE_GENERIC = "generic"
RECIPE_QWEN3_5 = "qwen3_5"
RECIPE_DEEPSEEK_V4_FLASH = "deepseek-v4-flash"
GATED_DELTA_NET = "gated_delta_net"


# --------------------------------------------------------------------------
# Provider readers (kept local so this module does not import the learner).
# --------------------------------------------------------------------------


def _provider_value(provider, *names: str):
    for name in names:
        value = getattr(provider, name, None)
        if value is not None:
            return value
    raise ValueError(
        f"Megatron-Bridge provider {type(provider).__name__} lacks {names[0]}"
    )


def _positive_int(provider, *names: str) -> int:
    value = _provider_value(provider, *names)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"Megatron-Bridge provider has invalid {names[0]}={value!r}")
    return value


def _text(value: Any) -> str:
    return str(getattr(value, "value", value))


# --------------------------------------------------------------------------
# Config sections
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MoeGeometry:
    num_experts: int
    ffn_hidden_size: int
    router_topk: int
    # Normalized: DeepSeek V4 Flash's per-layer mask collapses to scalar 1.
    layer_freq: Any
    shared_expert_intermediate_size: int | None


@dataclass(frozen=True)
class ModelGeometry:
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_query_groups: int
    kv_channels: int
    ffn_hidden_size: int
    max_position_embeddings: int
    vocab_size: int
    normalization: str
    norm_epsilon: float
    position_embedding_type: str
    rope_type: str | None
    rotary_base: int
    rotary_percent: float
    mrope_section: tuple[int, ...] | None
    gated_linear_unit: bool
    untie_embeddings_and_output_weights: bool
    disable_bias_linear: bool
    add_qkv_bias: bool
    qk_layernorm: bool
    multi_latent_attention: bool
    # (name, value) pairs of the MLA ranks/dims the provider exposes.
    mla_dims: tuple[tuple[str, int], ...]
    moe: MoeGeometry | None

    @property
    def group_query_attention(self) -> bool:
        # GQA and MLA are mutually exclusive; MLA carries its own KV geometry.
        return (
            self.num_query_groups < self.num_attention_heads
            and not self.multi_latent_attention
        )


@dataclass(frozen=True)
class GdnRecipe:
    """Gated-delta-net hybrid capability (slot for #65).

    ``gated_delta_net`` mirrors the provider's
    ``experimental_attention_variant``.  It selects the ``bshd`` qkv format
    and (#65) routes every such provider through the Qwen3.5 layer spec.
    """

    gated_delta_net: bool
    qkv_format: str = "bshd"


@dataclass(frozen=True)
class ModelRecipe:
    name: str  # RECIPE_GENERIC | RECIPE_QWEN3_5 | RECIPE_DEEPSEEK_V4_FLASH
    provider_class: str
    training_attention_backend: str
    gdn: GdnRecipe


@dataclass(frozen=True)
class ParallelLayout:
    actor_num_nodes: int
    actor_num_gpus_per_node: int
    tensor_parallel: int
    pipeline_parallel: int
    expert_parallel: int
    data_parallel: int
    rollout_num_gpus_per_engine: int
    # None means colocated rollout; otherwise dedicated rollout GPU count.
    dedicated_rollout_gpus: int | None
    visible_gpus_per_node: int
    # Uneven pipeline split: (first, last) stage layer counts, else None.
    uneven_pipeline_layers: tuple[int, int] | None
    # rl-infra-spec 2.1: reserved standby GPUs of a fixed partition (never
    # started by any role); a non-zero value needs the fork-M1 placement map.
    standby_gpus: int = 0
    # rl-infra-spec 3.x/4.7 (fork F-R1): yeto names of the rollout engine cells
    # declared to the fork (placement map "rollout_cells"); () = fork default.
    rollout_cell_names: tuple[str, ...] = ()
    # rl-multinode-island: GPUs per island node on a multi-node island; None on
    # a single node (every node rule off).
    island_gpus_per_node: int | None = None
    # rl-multinode-island Q2: explicit role -> logical bundle map of a multi-node
    # fixed partition whose trainer is not the leading bundles (the launcher
    # derives it from the elastic cfg placement); None = leading layout.
    bundle_map: dict[str, tuple[int, ...]] | None = None

    @property
    def colocated(self) -> bool:
        return self.dedicated_rollout_gpus is None


@dataclass(frozen=True)
class TrainableConfig:
    parameter_mode: str  # "lora" | "full"
    lora_rank: int
    lora_targets: str  # preset name, e.g. "attention"
    target_modules: tuple[str, ...]  # engine-resolved module names
    expert_full_count: int
    # Training-time LoRA dropout (ports only; --rl-lora-dropout). The exported
    # adapter / canonical LoRA config keep dropout 0 (inference is unaffected).
    lora_dropout: float = 0.0

    @property
    def routed_expert_lora(self) -> bool:
        return self.lora_targets == "attention-routed-experts"

    @property
    def expert_full(self) -> bool:
        return self.expert_full_count > 0


@dataclass(frozen=True)
class DatasetColumns:
    """Dataset column naming (slot for #59).

    ``prompt_column`` / ``label_column`` name the *source* dataset columns
    used during prompt normalization; ``None`` keeps today's lookup.  The
    ``*_key`` fields are the normalized record keys the engine reads.
    """

    prompt_column: str | None = None
    label_column: str | None = None
    input_key: str = "messages"
    label_key: str = "label"
    metadata_key: str = "metadata"


@dataclass(frozen=True)
class DataConfig:
    prompt_path: str
    columns: DatasetColumns
    apply_chat_template: bool
    chat_template_kwargs: dict[str, Any] | None


@dataclass(frozen=True)
class BatchConfig:
    seq_len: int
    global_rounds: int
    eval_only: bool
    groups_per_round: int
    samples_per_group: int
    over_sampling_batch_size: int
    optimizer_steps: int
    global_batch: int
    rollout_max_response_len: int


@dataclass(frozen=True)
class AlgorithmConfig:
    advantage_estimator: str
    reward_function: str  # "package.module:function"
    lr: Any
    seed: int
    rollout_seed: int
    # Explicit learning-rate schedule (never Miles' implicit default, which
    # derives its horizon from ``--num-rollout`` = global rounds).  ``None``
    # (eval-only) emits no schedule flags.
    lr_schedule: "LrSchedule | None" = None


LR_DECAY_STYLES = frozenset({"linear", "constant"})


@dataclass(frozen=True)
class LrSchedule:
    """Per-island optimizer LR schedule, in local optimizer steps.

    strict-avg: every island runs exactly ``global_rounds * optimizer_steps``
    local steps, so linear decay over that horizon is well defined (and is
    what Miles' implicit default already produced).  decoupled: islands run
    until the syncer stops them, so the local step count is not known up
    front and any finite linear horizon can reach zero mid-run; the LR is
    held constant.
    """

    decay_style: str
    decay_iters: int

    def __post_init__(self) -> None:
        if self.decay_style not in LR_DECAY_STYLES:
            raise ValueError(f"unsupported LR decay style {self.decay_style!r}")
        if type(self.decay_iters) is not int or self.decay_iters < 1:
            raise ValueError("LR decay horizon must be a positive step count")


LR_SCHEDULE_FLAGS = ("--lr-decay-style", "--lr-decay-iters", "--lr-warmup-iters", "--min-lr")


def lr_schedule_argv(schedule: "LrSchedule | None") -> tuple[str, ...]:
    """Miles/Megatron flags for ``schedule``; shared verbatim by both engines."""

    if schedule is None:
        return ()
    return (
        "--lr-decay-style", schedule.decay_style,
        "--lr-decay-iters", str(schedule.decay_iters),
        "--lr-warmup-iters", "0",
        "--min-lr", "0",
    )


def resolve_lr_schedule(
    *,
    sync_preset: str,
    eval_only: bool,
    global_rounds: int,
    optimizer_steps: int,
    rollout_batch_size: int,
    n_samples_per_prompt: int,
    global_batch: int,
) -> LrSchedule | None:
    """The island's LR schedule, decided by the sync mode (design D1-D3).

    Both engine translations (legacy ``_legacy_miles_argv`` and ports
    ``translate_run_config``) emit exactly this schedule.
    """

    if eval_only:
        return None
    horizon = global_rounds * optimizer_steps
    if sync_preset == "decoupled":
        # Run-until-stop: the local step count is unknown up front.
        # decay_iters only satisfies Megatron's ``lr_decay_steps > 0``; a
        # constant schedule never reads it.
        return LrSchedule("constant", horizon)
    # strict-avg / dense-full: one local round (optimizer_steps steps) per
    # global round.  The explicit linear horizon equals Miles' implicit
    # ``num_rollout * rollout_batch_size * n_samples / global_batch`` only
    # when every round is exactly ``optimizer_steps`` optimizer steps.
    if rollout_batch_size * n_samples_per_prompt != global_batch * optimizer_steps:
        raise ValueError(
            "strict LR schedule requires rollout_batch_size * n_samples_per_prompt "
            f"== global_batch * optimizer_steps (got {rollout_batch_size} * "
            f"{n_samples_per_prompt} != {global_batch} * {optimizer_steps})"
        )
    return LrSchedule("linear", horizon)


@dataclass(frozen=True)
class EvalConfig:
    interval: int
    dataset_name: str
    prompt_path: str
    samples_per_prompt: int
    temperature: Any = None
    top_p: Any = None
    max_prompt_len: Any = None
    max_response_len: Any = None
    max_context_len: Any = None


@dataclass(frozen=True)
class ServingConfig:
    mem_fraction_static: Any
    deterministic_inference: bool
    offload_train: bool
    tp_size: Any = None
    dp_size: Any = None
    ep_size: Any = None
    attention_backend: Any = None
    page_size: Any = None
    max_running_requests: Any = None
    chunked_prefill_size: Any = None


@dataclass(frozen=True)
class NetworkPorts:
    rollout_engine_base_port: Any = None
    sglang_router_port: Any = None
    sglang_router_prometheus_port: Any = None
    train_master_base_port: Any = None


@dataclass(frozen=True)
class AgentConfig:
    custom_generate_function_path: str | None
    custom_agent_function_path: str | None
    agent_max_seq_len: int | None
    dynamic_sampling_filter_path: str | None
    use_rollout_routing_replay: bool
    use_session_server: bool
    session_server_ip: str | None
    session_server_port: tuple[Any, ...] | None
    tito_model: str | None
    tito_allowed_append_roles: tuple[str, ...] | None


@dataclass(frozen=True)
class RLRunConfig:
    hf_checkpoint: str
    ref_load: str
    model_recipe: ModelRecipe
    geometry: ModelGeometry
    parallel: ParallelLayout
    trainable: TrainableConfig
    data: DataConfig
    batch: BatchConfig
    algorithm: AlgorithmConfig
    eval: EvalConfig | None
    serving: ServingConfig
    ports: NetworkPorts
    agent: AgentConfig
    yeto_policy_sync: bool
    distributed_timeout_minutes: int
    # E2 plan-v2 §0 (A6/A6b/A8): Megatron --deterministic-mode; the matching
    # NCCL/cuBLAS/TF32 environment is set by the learner (--rl-deterministic-trainer).
    deterministic_trainer: bool = False
    # rl-infra-spec 4.7 (E3): trainer DP-change edges enabled
    # (--rl-elastic-trainer-edges); the first certified DP profile refuses
    # --balance-data, so the translation drops it only then.
    trainer_dp_edges: bool = False
    # 3.x (--rl-elastic): the fork's cordon / drain / cordoned admission need the
    # Miles router (fork server_cell._assert_cordonable, admit_cordoned).
    use_miles_router: bool = False
    # A27 (--rl-elastic): bound the fork's trainer<->engine weight-update group
    # rendezvous (--update-weight-group-timeout-s) so a member dying mid-publish
    # fails the publish (abort -> REBUILD_OLD) instead of stalling the run.
    # None = flag not emitted (fork default: torch's process-group timeout).
    update_weight_group_timeout_s: float | None = None


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------


# A27: seconds the fork waits for the weight-update NCCL group during an
# elastic member publish (needs Miles >= image-m3a27; older forks reject the flag).
ELASTIC_UPDATE_WEIGHT_GROUP_TIMEOUT_S = 120.0


def select_gdn_recipe(provider) -> GdnRecipe:
    return GdnRecipe(
        gated_delta_net=(
            _text(getattr(provider, "experimental_attention_variant", None))
            == GATED_DELTA_NET
        )
    )


def _resolve_ref_load(args, model_path) -> str:
    configured = getattr(args, "megatron_ref_load", None)
    if configured is None:
        return str(model_path)
    configured_path = Path(configured).expanduser()
    if not configured_path.is_absolute():
        raise ValueError("--megatron-ref-load must be an absolute local path")
    if configured_path.is_symlink() or not configured_path.is_dir():
        raise ValueError("--megatron-ref-load must be a real local directory")
    release_marker = configured_path / "latest_checkpointed_iteration.txt"
    try:
        marker = release_marker.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ValueError("--megatron-ref-load has no readable release marker") from exc
    if release_marker.is_symlink() or marker != "release":
        raise ValueError("--megatron-ref-load is not a release checkpoint")
    return str(configured_path.resolve())


def _validate_callable_spec(spec: str) -> None:
    module, separator, function = spec.partition(":")
    if not separator or not module or not function.isidentifier():
        raise ValueError("RL callable must be package.module:function")


def resolve_rl_run_config(
    args,
    *,
    model_path: str | Path,
    rollout_model_path: str | Path | None = None,
    prompt_path: str | Path,
    eval_prompt_path: str | Path | None = None,
    provider,
    target_modules: list[str],
    yeto_policy_sync: bool = True,
) -> RLRunConfig:
    """Validate and resolve one RL run, independent of the engine.

    Validation order matches the pre-split ``build_miles_argv`` so the same
    invalid input raises the same error.
    """

    parameter_mode = getattr(args, "parameter_mode", "lora")
    if parameter_mode == "full" and int(getattr(args, "rl_standby_gpus", 0) or 0):
        # review F7: never drop a standby request silently
        raise ValueError("--rl-standby-gpus is not supported for full-parameter mode")
    if parameter_mode not in PARAMETER_MODES:
        raise ValueError("unsupported RL parameter mode")
    if parameter_mode == "full":
        if target_modules:
            raise ValueError("full-parameter Miles must not select LoRA targets")
    elif not target_modules:
        raise ValueError("PEFT selected no LoRA target modules")

    from ..codex_backend import QWEN35_MODEL, QWEN35_REVISION

    gdn = select_gdn_recipe(provider)
    # #65: Miles runs gated-delta-net hybrids (Qwen3.5 / Qwen3.6 dense)
    # through its own layer spec, not the generic GPT provider path; the
    # pinned Codex profile is one such model, and Bridge reports the
    # capability on the provider for every other checkpoint of the family.
    qwen35_recipe = gdn.gated_delta_net or (
        getattr(args, "model", None) == QWEN35_MODEL
        and getattr(args, "model_revision", None) == QWEN35_REVISION
    )

    hidden = _positive_int(provider, "hidden_size")
    heads = _positive_int(provider, "num_attention_heads")
    layers = _positive_int(provider, "num_layers")
    ffn = _positive_int(provider, "ffn_hidden_size")
    query_groups = int(getattr(provider, "num_query_groups", heads))
    multi_latent_attention = bool(getattr(provider, "multi_latent_attention", False))
    kv_channels = int(getattr(provider, "kv_channels", hidden // heads))
    max_positions = int(
        _provider_value(
            provider, "seq_length", "max_sequence_length", "max_position_embeddings"
        )
    )
    if args.seq_len > max_positions:
        raise ValueError(
            f"--seq-len {args.seq_len} exceeds model context limit {max_positions}"
        )
    vocab_size = _positive_int(provider, "vocab_size", "padded_vocab_size")
    normalization = _text(getattr(provider, "normalization", "RMSNorm"))
    epsilon = float(_provider_value(provider, "layernorm_epsilon", "norm_epsilon"))
    position_type = _text(getattr(provider, "position_embedding_type", "rope"))
    rope_type = None
    if position_type == "yarn":
        position_type, rope_type = "rope", "yarn"
    rotary_base = int(getattr(provider, "rotary_base", 10000))
    rotary_percent = float(getattr(provider, "rotary_percent", 1.0))

    actor_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
    tensor_parallel = getattr(args, "tensor_parallel", 1)
    pipeline_parallel = getattr(args, "pipeline_parallel", 1)
    model_parallel = tensor_parallel * pipeline_parallel
    if tensor_parallel <= 0 or pipeline_parallel <= 0 or actor_gpus % model_parallel:
        raise ValueError("Miles actor world must be divisible by TP*PP")
    is_moe = getattr(provider, "num_moe_experts", None) is not None
    expert_parallel = getattr(args, "expert_parallel", None) or (
        actor_gpus if is_moe else 1
    )
    if not is_moe and expert_parallel != 1:
        raise ValueError("EP>1 requires a MoE model")
    if actor_gpus % expert_parallel:
        raise ValueError("expert parallelism must divide Miles actor world size")
    if is_moe and expert_parallel > 1 and args.lora_targets == "all-linear":
        raise ValueError(
            "EP>1 requires replicated attention LoRA, not expert-sharded all-linear LoRA"
        )
    recipe = getattr(args, "rl_model_recipe", RECIPE_GENERIC)
    if args.lora_targets == "attention-routed-experts" and (
        not is_moe or recipe != RECIPE_DEEPSEEK_V4_FLASH
    ):
        raise ValueError(
            "attention-routed-experts is reserved for the expanded DeepSeek V4 recipe"
        )
    expert_full_count = int(getattr(args, "expert_full_count", 0) or 0)
    if not 0 <= expert_full_count <= 32:
        raise ValueError("expert-full count must be between 1 and 32 when enabled")
    expert_full = expert_full_count > 0
    if expert_full and (
        recipe != RECIPE_DEEPSEEK_V4_FLASH or args.lora_targets != "attention"
    ):
        raise ValueError(
            "expert-full tuning requires the DeepSeek V4 recipe and attention LoRA"
        )
    data_parallel = actor_gpus // model_parallel
    if parameter_mode == "full" and data_parallel != 1:
        raise ValueError("dense full-parameter GRPO requires DP=1")
    dedicated_rollout_gpus: int | None = None
    if parameter_mode == "full":
        rollout_gpus = getattr(args, "rollout_num_gpus", None)
        rollout_gpus_per_engine = getattr(args, "rollout_num_gpus_per_engine", None)
        if (
            args.actor_num_nodes != 1
            or type(rollout_gpus) is not int
            or rollout_gpus < 1
            or type(rollout_gpus_per_engine) is not int
            or rollout_gpus_per_engine < 1
            or rollout_gpus % rollout_gpus_per_engine
        ):
            raise ValueError(
                "Milestone-1 dense full-parameter mode requires one node and "
                "dedicated, evenly partitioned inference engines"
            )
        if qwen35_recipe and (
            rollout_gpus_per_engine != 1
            or getattr(args, "sglang_tp_size", None) not in {None, 1}
        ):
            raise ValueError("pinned Qwen3.5 requires TP1 SGLang inference engines")
        dedicated_rollout_gpus = rollout_gpus
        visible_gpus_per_node = args.actor_num_gpus_per_node + rollout_gpus
    elif getattr(args, "rl_placement", "colocated") == "fixed-partition":
        # rl-infra-spec 2.1: a LoRA fixed partition (ports engine only; the
        # legacy CLI never sets rl_placement, so its layout is unchanged).
        rollout_gpus = getattr(args, "rollout_num_gpus", None)
        per_engine = getattr(args, "rollout_num_gpus_per_engine", 1)
        island_gpus_per_node = getattr(args, "rl_island_gpus_per_node", None)
        if (
            (args.actor_num_nodes != 1 and island_gpus_per_node is None)
            or type(rollout_gpus) is not int
            or rollout_gpus < 1
            or rollout_gpus % per_engine
        ):
            raise ValueError(
                "a LoRA fixed partition needs one node (or --rl-island-gpus-per-node on a "
                "multi-node island) and --rollout-num-gpus as a positive multiple of "
                "--rollout-num-gpus-per-engine"
            )
        dedicated_rollout_gpus = rollout_gpus
        standby_gpus = int(getattr(args, "rl_standby_gpus", 0) or 0)
        if standby_gpus < 0:
            raise ValueError("--rl-standby-gpus must be non-negative")
        if island_gpus_per_node is None:
            visible_gpus_per_node = args.actor_num_gpus_per_node + rollout_gpus + standby_gpus
        else:
            # Multi-node island: every node exposes all its GPUs; the trainer
            # rectangle, rollout and standby are laid out over the whole island.
            visible_gpus_per_node = int(island_gpus_per_node)
    else:
        visible_gpus_per_node = args.actor_num_gpus_per_node
    requested_standby = int(getattr(args, "rl_standby_gpus", 0) or 0)
    if dedicated_rollout_gpus is None:
        if requested_standby:
            raise ValueError("--rl-standby-gpus needs --rl-placement fixed-partition")
        standby_gpus = 0
    elif parameter_mode == "full":
        standby_gpus = 0

    ref_load = _resolve_ref_load(args, model_path)
    global_batch = args.groups_per_round * args.samples_per_group // args.optimizer_steps
    if global_batch % data_parallel:
        raise ValueError("Miles global batch must divide evenly across DP ranks")

    if recipe == RECIPE_DEEPSEEK_V4_FLASH:
        expected_lora_targets = "attention" if expert_full else "attention-routed-experts"
        if args.lora_targets != expected_lora_targets:
            raise ValueError(
                f"expanded DeepSeek V4 recipe requires {expected_lora_targets}"
            )
        if (
            tensor_parallel != 8
            or expert_parallel != 8
            or getattr(args, "rollout_num_gpus_per_engine", 1) != 8
        ):
            raise ValueError(
                "expanded DeepSeek V4 requires TP8/EP8 pipeline stages and "
                "per-node eight-GPU rollout replicas"
            )
        if layers != 43 or not is_moe or not multi_latent_attention:
            raise ValueError(
                "DeepSeek V4 Flash recipe requires the 43-layer MoE/MLA provider"
            )
        recipe_name, attention_backend = RECIPE_DEEPSEEK_V4_FLASH, "flash"
    elif qwen35_recipe:
        recipe_name, attention_backend = RECIPE_QWEN3_5, "flash"
    else:
        recipe_name, attention_backend = RECIPE_GENERIC, "unfused"

    _validate_callable_spec(args.reward_function)

    eval_config = _resolve_eval(
        args,
        parameter_mode=parameter_mode,
        prompt_path=prompt_path,
        eval_prompt_path=eval_prompt_path,
        yeto_policy_sync=yeto_policy_sync,
    )

    uneven_pipeline_layers = None
    if pipeline_parallel > 1 and layers % pipeline_parallel:
        middle = layers // pipeline_parallel
        remainder = layers - middle * pipeline_parallel
        uneven_pipeline_layers = (
            middle + (remainder + 1) // 2,
            middle + remainder // 2,
        )

    custom_agent = getattr(args, "custom_agent_function_path", None) or None
    agent_max_seq_len = None
    if custom_agent:
        agent_max_seq_len = getattr(args, "agent_max_seq_len", None) or args.seq_len
        if eval_config is not None and eval_config.max_context_len is not None:
            agent_max_seq_len = eval_config.max_context_len

    mrope_section = None
    if position_type == "mrope":
        mrope_section = tuple(
            int(value) for value in _provider_value(provider, "mrope_section")
        )

    moe = None
    if is_moe:
        moe_layer_freq = _provider_value(provider, "moe_layer_freq")
        if recipe == RECIPE_DEEPSEEK_V4_FLASH:
            if isinstance(moe_layer_freq, (list, tuple)):
                mask = tuple(int(value) for value in moe_layer_freq)
                if len(mask) != layers or set(mask) != {1}:
                    raise ValueError(
                        "DeepSeek V4 Flash requires every one of its 43 layers "
                        "to be an MoE layer"
                    )
            elif int(moe_layer_freq) != 1:
                raise ValueError("DeepSeek V4 Flash requires moe_layer_freq=1")
            moe_layer_freq = 1
        shared = getattr(provider, "moe_shared_expert_intermediate_size", None)
        moe = MoeGeometry(
            num_experts=_positive_int(provider, "num_moe_experts"),
            ffn_hidden_size=_positive_int(provider, "moe_ffn_hidden_size"),
            router_topk=_positive_int(provider, "moe_router_topk"),
            layer_freq=moe_layer_freq,
            shared_expert_intermediate_size=None if shared is None else int(shared),
        )

    mla_dims: tuple[tuple[str, int], ...] = ()
    if multi_latent_attention:
        mla_dims = tuple(
            (name, int(getattr(provider, name)))
            for name in (
                "q_lora_rank",
                "kv_lora_rank",
                "qk_head_dim",
                "qk_pos_emb_head_dim",
                "v_head_dim",
            )
            if getattr(provider, name, None) is not None
        )

    geometry = ModelGeometry(
        num_layers=layers,
        hidden_size=hidden,
        num_attention_heads=heads,
        num_query_groups=query_groups,
        kv_channels=kv_channels,
        ffn_hidden_size=ffn,
        max_position_embeddings=max_positions,
        vocab_size=vocab_size,
        normalization=normalization,
        norm_epsilon=epsilon,
        position_embedding_type=position_type,
        rope_type=rope_type,
        rotary_base=rotary_base,
        rotary_percent=rotary_percent,
        mrope_section=mrope_section,
        gated_linear_unit=bool(getattr(provider, "gated_linear_unit", False)),
        untie_embeddings_and_output_weights=not bool(
            getattr(provider, "share_embeddings_and_output_weights", True)
        ),
        disable_bias_linear=getattr(provider, "add_bias_linear", None) is False,
        add_qkv_bias=bool(getattr(provider, "add_qkv_bias", False)),
        qk_layernorm=bool(getattr(provider, "qk_layernorm", False)),
        multi_latent_attention=multi_latent_attention,
        mla_dims=mla_dims,
        moe=moe,
    )
    # Session-server fields are only read (strictly, as before) when enabled.
    use_session_server = bool(args.use_session_server)
    session_ip = session_port = tito_model = append_roles = None
    if use_session_server:
        session_ip = args.session_server_ip or None
        session_port = args.session_server_port
        tito_model = args.tito_model or None
        append_roles = getattr(args, "tito_allowed_append_roles", None)
    return RLRunConfig(
        hf_checkpoint=str(rollout_model_path or model_path),
        ref_load=ref_load,
        model_recipe=ModelRecipe(
            name=recipe_name,
            provider_class=type(provider).__name__,
            training_attention_backend=attention_backend,
            gdn=gdn,
        ),
        geometry=geometry,
        parallel=ParallelLayout(
            actor_num_nodes=args.actor_num_nodes,
            actor_num_gpus_per_node=args.actor_num_gpus_per_node,
            tensor_parallel=tensor_parallel,
            pipeline_parallel=pipeline_parallel,
            expert_parallel=expert_parallel,
            data_parallel=data_parallel,
            rollout_num_gpus_per_engine=getattr(args, "rollout_num_gpus_per_engine", 1),
            dedicated_rollout_gpus=dedicated_rollout_gpus,
            visible_gpus_per_node=visible_gpus_per_node,
            uneven_pipeline_layers=uneven_pipeline_layers,
            standby_gpus=standby_gpus,
            rollout_cell_names=_rollout_cell_names(args, dedicated_rollout_gpus),
            island_gpus_per_node=(int(getattr(args, "rl_island_gpus_per_node", None))
                                  if getattr(args, "rl_island_gpus_per_node", None) is not None
                                  and dedicated_rollout_gpus is not None else None),
            bundle_map=_island_bundle_map(args, dedicated_rollout_gpus),
        ),
        trainable=TrainableConfig(
            parameter_mode=parameter_mode,
            lora_rank=args.lora_r,
            lora_dropout=_lora_dropout(args, parameter_mode),
            lora_targets=args.lora_targets,
            target_modules=tuple(target_modules),
            expert_full_count=expert_full_count,
        ),
        data=DataConfig(
            prompt_path=str(prompt_path),
            columns=DatasetColumns(
                prompt_column=getattr(args, "rl_prompt_column", None),
                label_column=getattr(args, "rl_label_column", None),
            ),
            # Agentic sessions receive the raw message list (the session
            # server renders it); pre-rendering would double-render.
            apply_chat_template=not custom_agent,
            chat_template_kwargs=getattr(args, "apply_chat_template_kwargs", None)
            or None,
        ),
        batch=BatchConfig(
            seq_len=args.seq_len,
            global_rounds=args.global_rounds,
            eval_only=bool(getattr(args, "eval_only", False)),
            groups_per_round=args.groups_per_round,
            samples_per_group=args.samples_per_group,
            over_sampling_batch_size=args.over_sampling_batch_size,
            optimizer_steps=args.optimizer_steps,
            global_batch=global_batch,
            rollout_max_response_len=args.rollout_max_response_len,
        ),
        algorithm=AlgorithmConfig(
            advantage_estimator="grpo",
            reward_function=args.reward_function,
            lr=args.inner_lr,
            lr_schedule=resolve_lr_schedule(
                sync_preset=getattr(args, "sync_preset", "strict-avg"),
                eval_only=bool(getattr(args, "eval_only", False)),
                global_rounds=args.global_rounds,
                optimizer_steps=args.optimizer_steps,
                rollout_batch_size=args.groups_per_round,
                n_samples_per_prompt=args.samples_per_group,
                global_batch=global_batch,
            ),
            seed=args.seed,
            rollout_seed=getattr(args, "rollout_seed", args.seed + args.learner_id),
        ),
        eval=eval_config,
        serving=ServingConfig(
            mem_fraction_static=getattr(args, "sglang_mem_fraction_static", 0.4),
            deterministic_inference=bool(
                getattr(args, "sglang_deterministic_inference", True)
            ),
            offload_train=bool(getattr(args, "rl_offload_train", False)),
            tp_size=getattr(args, "sglang_tp_size", None),
            dp_size=getattr(args, "sglang_dp_size", None),
            ep_size=getattr(args, "sglang_ep_size", None),
            attention_backend=getattr(args, "sglang_attention_backend", None),
            page_size=getattr(args, "sglang_page_size", None),
            max_running_requests=getattr(args, "sglang_max_running_requests", None),
            chunked_prefill_size=getattr(args, "sglang_chunked_prefill_size", None),
        ),
        ports=NetworkPorts(
            rollout_engine_base_port=getattr(args, "rollout_engine_base_port", None),
            sglang_router_port=getattr(args, "sglang_router_port", None),
            sglang_router_prometheus_port=getattr(
                args, "sglang_router_prometheus_port", None
            ),
            train_master_base_port=getattr(args, "train_master_base_port", None),
        ),
        agent=AgentConfig(
            custom_generate_function_path=args.custom_generate_function_path or None,
            custom_agent_function_path=custom_agent,
            agent_max_seq_len=agent_max_seq_len,
            dynamic_sampling_filter_path=(
                getattr(args, "dynamic_sampling_filter_path", None) or None
            ),
            use_rollout_routing_replay=bool(
                getattr(args, "use_rollout_routing_replay", False)
            ),
            use_session_server=use_session_server,
            session_server_ip=session_ip,
            session_server_port=tuple(session_port) if session_port else None,
            tito_model=tito_model,
            tito_allowed_append_roles=tuple(append_roles) if append_roles else None,
        ),
        yeto_policy_sync=yeto_policy_sync,
        distributed_timeout_minutes=getattr(args, "rl_distributed_timeout_minutes", 10),
        deterministic_trainer=bool(getattr(args, "rl_deterministic_trainer", False)),
        use_miles_router=bool(getattr(args, "rl_elastic", False)),
        update_weight_group_timeout_s=(
            ELASTIC_UPDATE_WEIGHT_GROUP_TIMEOUT_S if getattr(args, "rl_elastic", False) else None
        ),
        trainer_dp_edges=bool(getattr(args, "rl_elastic", False)
                              and getattr(args, "rl_elastic_trainer_edges", False)),
    )


def _island_bundle_map(args, dedicated_rollout_gpus) -> dict[str, tuple[int, ...]] | None:
    """``--rl-island-bundle-map`` (Q2): JSON text or a dict; needs a multi-node
    fixed partition (``--rl-island-gpus-per-node``). Validated against the role
    sizes by ``PlacementRequest``."""
    raw = getattr(args, "rl_island_bundle_map", None)
    if raw is None:
        return None
    if dedicated_rollout_gpus is None or getattr(args, "rl_island_gpus_per_node", None) is None:
        raise ValueError("--rl-island-bundle-map needs a multi-node fixed partition "
                         "(--rl-placement fixed-partition and --rl-island-gpus-per-node)")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError as exc:
            raise ValueError(f"--rl-island-bundle-map is not JSON: {exc}") from None
    if not isinstance(raw, dict) or set(raw) - {"trainer", "rollout", "standby"}:
        raise ValueError("--rl-island-bundle-map must map trainer/rollout/standby to bundle lists")
    out = {}
    for role in ("trainer", "rollout", "standby"):
        values = raw.get(role, [])
        if not isinstance(values, (list, tuple)) or any(
                isinstance(v, bool) or not isinstance(v, int) for v in values):
            raise ValueError(f"--rl-island-bundle-map {role} must be a list of ints")
        out[role] = tuple(values)
    return out


def _rollout_cell_names(args, dedicated_rollout_gpus) -> tuple[str, ...]:
    if not getattr(args, "rl_elastic_declare_cells", False):
        return ()
    if dedicated_rollout_gpus is None:
        raise ValueError("--rl-elastic-declare-cells needs --rl-placement fixed-partition")
    names = tuple(c.strip() for c in (getattr(args, "rl_elastic_cells", None) or "").split(",")
                  if c.strip())
    if not names:
        raise ValueError("--rl-elastic-declare-cells needs --rl-elastic-cells (the cell names)")
    return names


def _lora_dropout(args, parameter_mode: str) -> float:
    value = getattr(args, "rl_lora_dropout", None)
    if value is None:
        return 0.0
    value = float(value)
    if not 0.0 <= value < 1.0:
        raise ValueError("--rl-lora-dropout must be in [0, 1)")
    if getattr(args, "rl_engine", "ports") != "ports" or parameter_mode != "lora":
        raise ValueError("--rl-lora-dropout only applies to the ports LoRA engine")
    return value


def ports_training_eval(args, *, parameter_mode: str | None) -> bool:
    """Training-time heldout eval on a ports LoRA island (not eval-only, not dense).

    Shared by the learner's dataset-identity check and :func:`_resolve_eval`
    so both accept exactly the same runs.
    """

    return (
        not getattr(args, "eval_only", False)
        and getattr(args, "rl_engine", "ports") == "ports"
        and parameter_mode != "full"
    )


def _resolve_eval(
    args,
    *,
    parameter_mode: str,
    prompt_path,
    eval_prompt_path,
    yeto_policy_sync: bool,
) -> EvalConfig | None:
    eval_interval = getattr(args, "eval_interval", None)
    if eval_interval is None:
        return None
    if not yeto_policy_sync:
        raise ValueError("Yeto evaluation requires the external policy boundary")
    if eval_interval <= 0:
        raise ValueError("evaluation interval must be positive")
    eval_only = getattr(args, "eval_only", False)
    dense_train_eval = (
        not eval_only
        and parameter_mode == "full"
        and getattr(args, "sync_preset", None) == "dense-full"
    )
    ports_train_eval = ports_training_eval(args, parameter_mode=parameter_mode)
    if eval_only:
        if eval_interval != 1:
            raise ValueError("SSH evaluation must be one separate eval-only run")
        selected = prompt_path
    elif dense_train_eval:
        if eval_interval != args.global_rounds + 1:
            raise ValueError(
                "dense full evaluation interval must remain outside the "
                "training-round budget"
            )
        if eval_prompt_path is None:
            raise ValueError("dense full evaluation requires heldout prompt data")
        selected = eval_prompt_path
    elif ports_train_eval:
        # Ports LoRA island (rl-infra-spec 2.3 / launcher wiring): Miles runs
        # the heldout eval every ``interval`` rollouts through the driver's
        # evaluate port; a distinct heldout split is mandatory.
        if eval_prompt_path is None:
            raise ValueError("ports training-time evaluation requires heldout prompt data")
        selected = eval_prompt_path
    else:
        raise ValueError(
            "training-time evaluation is restricted to dense full mode or the ports LoRA engine"
        )
    eval_name = getattr(args, "eval_dataset_name", None)
    eval_samples = getattr(args, "eval_samples_per_prompt", None)
    if not eval_name or not isinstance(eval_samples, int) or eval_samples <= 0:
        raise ValueError("evaluation requires a dataset name and positive sample count")
    # Eval-only reuses its sole normalized prompt file.  Dense training binds a
    # separately normalized, immutable heldout split; its interval sits beyond
    # the rollout budget because the dense policy hook evaluates only the exact
    # initial and terminal published policies.
    return EvalConfig(
        interval=eval_interval,
        dataset_name=str(eval_name),
        prompt_path=str(selected),
        samples_per_prompt=eval_samples,
        temperature=getattr(args, "eval_temperature", None),
        top_p=getattr(args, "eval_top_p", None),
        max_prompt_len=getattr(args, "eval_max_prompt_len", None),
        max_response_len=getattr(args, "eval_max_response_len", None),
        max_context_len=getattr(args, "eval_max_context_len", None),
    )
