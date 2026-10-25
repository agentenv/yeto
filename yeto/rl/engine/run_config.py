"""Engine-agnostic RL run configuration (rl-engine-ports task 2.4, design D8).

``resolve_rl_run_config`` turns one parsed learner namespace plus the Bridge
model provider into an immutable :class:`RLRunConfig`.  All validation and
derivation that does not depend on a particular RL engine lives here; each
engine then owns a pure translation of the config into its own launch
arguments (Miles: ``yeto.rl.adapters.miles.config.translate_run_config``).

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

from . import sample_roles
from .sample_roles import SampleBufferConfig

PARAMETER_MODES = frozenset({"lora", "full"})
RECIPE_GENERIC = "generic"
RECIPE_QWEN3_5 = "qwen3_5"
RECIPE_DEEPSEEK_V4_FLASH = "deepseek-v4-flash"
# Qwen3.8-Flash-Next (qwen4_exp): GDN + QSA + 512-expert MoE through the native
# Miles plugin (miles_plugins/models/qwen3_8_next), never the Qwen3.5 spec.
RECIPE_QWEN3_8_NEXT = "qwen3_8_next"
# GLM5.3-Flash (glm5_next): KDA + DSA MLA + 288-expert MoE through the Miles
# plugin spec (adapters/miles/models/glm5_next.py), full-parameter.
RECIPE_GLM5_NEXT = "glm5_next"
# NVIDIA Nemotron-H family (Nemotron-3 Nano / Super, Nemotron-3.5 Super VL text
# model): Mamba-2 + MoE + attention hybrid built by the Megatron-Bridge
# HybridModelProvider from the HF checkpoint (Miles --megatron-to-hf-mode
# bridge). The recipe is driven by the provider's layer pattern and router
# fields, not by one checkpoint (adapters/miles/models/nemotron_h.py).
RECIPE_NEMOTRON_H = "nemotron_h"
TRANSFORMER_IMPLS = ("transformer_engine", "local")
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
    name: str  # RECIPE_GENERIC | RECIPE_QWEN3_5 | RECIPE_DEEPSEEK_V4_FLASH | RECIPE_QWEN3_8_NEXT | RECIPE_GLM5_NEXT
    provider_class: str
    training_attention_backend: str
    gdn: GdnRecipe
    # Megatron --transformer-impl of a recipe that chooses it (glm5_next: the
    # N7.5 TE-free fallback is "local"); the other recipes keep their own.
    transformer_impl: str = "transformer_engine"
    # Recipe-owned Miles model flags read from the provider (nemotron_h router,
    # latent MoE); empty for the other recipes.
    model_flags: tuple[str, ...] = ()


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
    # ruling 2026-10-04 v2: explicit cross-node TP opt-ins (--rl-allow-cross-node-tp /
    # --rl-allow-cross-node-engine-tp); default False = TP stays inside a node.
    allow_cross_node_tp: bool = False
    allow_cross_node_engine_tp: bool = False
    # S27 G-AC: Megatron context parallelism of the trainer (Miles splits each
    # packed micro batch of max_tokens_per_gpu * CP tokens across the CP ranks).
    context_parallel: int = 1

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
    # Routed-expert LoRA rank r_e (Flash-Next native recipe only; --rl-lora-expert-rank).
    # None outside that recipe; the recipe defaults it to the profile's value.
    lora_expert_rank: int | None = None

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


DP_OPTIMIZERS = ("auto", "distributed", "replicated")


@dataclass(frozen=True)
class TrainMemoryConfig:
    """Trainer memory knobs (--rl-activation-recompute, --rl-max-tokens-per-gpu,
    --rl-optimizer-offload-fraction). Defaults keep the historical argv: no
    recompute, fixed micro batches, optimizer state on the device."""

    # Recompute every layer's activations in backward (keep only layer inputs).
    activation_recompute: bool = False
    # Token budget of one packed micro batch per GPU (dynamic batching);
    # None = fixed micro batches of --micro-batch-size samples.
    max_tokens_per_gpu: int | None = None
    # Share of the optimizer state kept and updated on the host (0 = none).
    optimizer_offload_fraction: float = 0.0
    # Megatron gradient-accumulation fusion (fused wgrad into main_grad; needs
    # APEX/TE fused kernels). Off keeps the historical argv.
    gradient_accumulation_fusion: bool = False
    # Optimizer of a full-parameter trainer at DP>1 (--rl-dp-optimizer):
    # "distributed" (sharded FP32 masters), "replicated" (whole masters on
    # every DP rank) or "auto" (distributed when the Miles pin supports it).
    dp_optimizer: str = "auto"


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
    # rl-algo-critic-family 2.4/5.1: run-level critic state (None = no critic
    # state given). The algorithm itself (gamma, value_clip, init ...) is the
    # AlgorithmSpec; this carries only what one run resolves.
    critic: "CriticRunConfig | None" = None
    # Log the training-side entropy (metrics only while entropy_coef is 0).
    observe_training_entropy: bool = False
    # Mamba training inter-chunk SSM state dtype ("fp32"/"bf16"; None = Megatron default).
    mamba_training_ssm_states_dtype: str | None = None


@dataclass(frozen=True)
class CriticRunConfig:
    """Run-level critic inputs of the ports main stage (design D5).

    ``critic_load``: the warm-up stage product (critic checkpoint directory)
    the main stage loads with ``--critic-load``; ``init_sha256``: its content
    hash (``critic_warmup.checkpoint_sha256``), recorded in receipts.
    """

    critic_load: str | None = None
    init_sha256: str | None = None

    def __post_init__(self) -> None:
        if (self.critic_load is None) != (self.init_sha256 is None):
            raise ValueError("critic run config needs critic_load and init_sha256 together")
        if self.init_sha256 is not None and (
            len(self.init_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in self.init_sha256)
        ):
            raise ValueError("critic init_sha256 must be a lowercase hex SHA256")


def resolve_critic_run_config(args) -> "CriticRunConfig | None":
    """``--rl-critic-load`` / ``--rl-critic-init-sha256`` (None when absent)."""

    load = getattr(args, "rl_critic_load", None)
    digest = getattr(args, "rl_critic_init_sha256", None)
    if load is None and digest is None:
        return None
    return CriticRunConfig(critic_load=load, init_sha256=digest)


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


# The Miles/Megatron flags of a schedule (``--lr-decay-style`` ...) are the
# Miles adapter's translation: yeto.rl.adapters.miles.lr_schedule
# (decoupling 4.1). The schedule decision itself stays here.


LR_SCHEDULE_CHOICES = ("auto", "linear", "constant")


def resolve_lr_schedule(
    *,
    sync_preset: str,
    eval_only: bool,
    global_rounds: int,
    optimizer_steps: int,
    rollout_batch_size: int,
    n_samples_per_prompt: int,
    global_batch: int,
    island_scheduling: str = "legacy",
    requested: str = "auto",
    full_parameter_async: bool = False,
) -> LrSchedule | None:
    """The island's LR schedule (design D1-D3), ``--rl-lr-schedule`` (S17 N16).

    ``auto`` (default) keeps the mode rule: decoupled, island-pool and
    asynchronous full-parameter islands run until the syncer stops them (an
    island that re-JOINs needs local rounds past ``global_rounds``; S17 N5
    a73ab1b2; an asynchronous full-parameter island trains H local rounds per
    fragment round and, with a partial quorum, as many rounds as it is fast),
    so any finite linear horizon can reach zero mid-run -> constant;
    strict-avg / full-parameter sweep / single island -> linear over
    ``global_rounds * optimizer_steps``.
    ``constant`` holds the LR in every mode (e.g. a single island compared
    with a fixed-LR baseline; S17 N4 had to approximate it with a 20000-step
    linear horizon). ``linear`` is refused where the horizon is unknown.

    The Miles translation (``translate_run_config``) emits exactly this schedule.
    """

    if requested not in LR_SCHEDULE_CHOICES:
        raise ValueError(f"--rl-lr-schedule must be one of {LR_SCHEDULE_CHOICES}, got {requested!r}")
    if eval_only:
        return None
    horizon = global_rounds * optimizer_steps
    run_until_stop = (sync_preset == "decoupled" or island_scheduling == "elastic"
                      or full_parameter_async)
    if requested == "linear" and run_until_stop:
        raise ValueError(
            "--rl-lr-schedule linear needs a known local step count; decoupled, "
            "island-pool and asynchronous full-parameter islands run until the syncer "
            "stops them (use auto or constant)")
    if requested == "constant" or (requested == "auto" and run_until_stop):
        # decay_iters only satisfies Megatron's ``lr_decay_steps > 0``; a
        # constant schedule never reads it.
        return LrSchedule("constant", horizon)
    # strict-avg / full-parameter: one local round (optimizer_steps steps) per
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
    # rl-trainer-cold-start-cost L8.2: opt-in SGLang snapshot flags on the
    # sampler (rollout) engines; refused without a recorded compatibility pass.
    sampler_snapshot: bool = False
    # S26 E6: rollout quantization tier (yeto.rl.quant.tiers; 0 = BF16)
    rollout_quant_tier: int = 0
    # tier 1: who quantizes ("engine" online FP8, or "trainer": Miles block FP8 updates)
    rollout_quant_fp8_source: str = "engine"


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
    # last + defaulted: direct AgentConfig constructors in older tests/helpers
    # (keyword-style, without this field) keep working
    session_server_workers: int | None = None


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
    # D1/D2 wiring (6.5, d2-wire): 5.7 per-(profile_hash, source, target) cost
    # table read by ``elastic.ElasticHook`` (None = no costs -> recommend/auto hold),
    # and the load-window length for ``timeline.load_windows`` (None = no hook).
    edge_costs_path: str | None = None
    elastic_window_s: float | None = None
    # rl-trainer-start-stop L4.6 (--rl-sample-role sampler|trainer): the island
    # writes or reads the durable sample buffer. None = role both (unchanged).
    sample_buffer: SampleBufferConfig | None = None
    train_memory: TrainMemoryConfig = TrainMemoryConfig()


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------


# A27: seconds the fork waits for the weight-update NCCL group during an
# elastic member publish (needs Miles >= image-m3a27; older forks reject the flag).
ELASTIC_UPDATE_WEIGHT_GROUP_TIMEOUT_S = 120.0


def qwen3_8_next_variant(args, provider) -> str | None:
    """``"full"`` / ``"4layer"`` when the run is Qwen3.8-Flash-Next, else None.

    Fingerprint, not name: the HF model id (any path whose basename is a known
    Flash-Next repo) OR the provider shape (GDN + 512 routed experts + hidden
    2560 + 48/4 layers).  Both GDN, so without this the Qwen3.5 spec would be
    picked silently (G1)."""
    from yeto.rl.profiles import qwen3_8_next as q

    model = str(getattr(args, "model", None) or "").rstrip("/")
    base = model.rsplit("/", 1)[-1]
    for variant, name in q.MODEL_NAMES.items():
        if base == name:
            return variant
    if (
        _text(getattr(provider, "experimental_attention_variant", None)) == GATED_DELTA_NET
        and getattr(provider, "num_moe_experts", None) == 512
        and getattr(provider, "hidden_size", None) == 2560
    ):
        layers = getattr(provider, "num_layers", None)
        for variant, n in q.NUM_LAYERS.items():
            if layers == n:
                return variant
    return None


def _is_nemotron_h(provider) -> bool:
    """A Megatron-Bridge hybrid (Mamba) provider built from a Nemotron-H checkpoint."""
    return bool(getattr(provider, "is_hybrid_model", False)) and bool(
        getattr(provider, "hybrid_layer_pattern", None))


def _recipe_model_flags(recipe_name: str, provider) -> tuple[str, ...]:
    if recipe_name != RECIPE_NEMOTRON_H:
        return ()
    return _nemotron_h_recipe().model_flags(provider)


def select_gdn_recipe(provider) -> GdnRecipe:
    return GdnRecipe(
        gated_delta_net=(
            _text(getattr(provider, "experimental_attention_variant", None))
            == GATED_DELTA_NET
        )
    )


def _nemotron_h_recipe():
    """The backend's Nemotron-H recipe (Miles -> adapters/miles/models/nemotron_h)."""
    from . import backends

    return backends.module("nemotron_h_recipe", backends.DEFAULT_BACKEND)


def _rules():
    """Backend-specific checks of the run (decoupling 4.2): Miles -> adapters/miles/run_config_rules."""
    from . import backends

    return backends.module("run_config_rules", backends.DEFAULT_BACKEND)


def full_parameter_layout_problem(*, data_parallel: int, actor_num_nodes: int, rollout_gpus: Any,
                                  rollout_gpus_per_engine: Any, sample_role: str = sample_roles.BOTH,
                                  island_gpus: int | None = None,
                                  island_gpus_per_node: int | None = None) -> str | None:
    """Why the Miles full-parameter mode refuses this layout; None when it accepts it.

    Three layouts are accepted:

    * a sampler island (``--rl-sample-role sampler``): its own engine partition.
      It builds no trainer, so every island GPU (``island_gpus``) serves an
      engine and it loads each full-weight release from the store. No
      ``--rollout-num-gpus`` (the island size is the partition); the island GPUs
      split evenly into engines of ``--rollout-num-gpus-per-engine``;
    * samples from the store (``--rl-sample-role trainer``): a trainer-only
      island. Any node count and any DP; the island starts no inference engine
      (``--rollout-num-gpus`` absent or 0), so no card is shared with one;
    * engines on the island (Milestone 1): dedicated, evenly partitioned
      engines on one node, or on a multi-node island (``island_gpus_per_node``,
      ``--rl-island-gpus-per-node``: the trainer rectangle and the engines are
      laid out over the island's nodes, as for a LoRA fixed partition). A
      full-parameter trainer never shares its cards with engines (no colocated
      placement). DP>1 needs a Miles pin with data-parallel replicas
      (``run_config_rules.full_parameter_dp_replica_gap``).
    """
    if sample_role == sample_roles.SAMPLER:
        if rollout_gpus not in (None, 0):
            return ("a full-parameter sampler island is its own engine partition: every island "
                    "GPU serves an engine; drop --rollout-num-gpus and size the island with --gpu")
        per_engine = 1 if rollout_gpus_per_engine is None else rollout_gpus_per_engine
        if (type(per_engine) is not int or per_engine < 1
                or (island_gpus is not None and int(island_gpus) % per_engine)):
            return (f"a full-parameter sampler island splits its {island_gpus} GPU(s) evenly into "
                    f"engines: --rollout-num-gpus-per-engine {rollout_gpus_per_engine} must divide it")
        return None
    if sample_role == sample_roles.TRAINER:
        if rollout_gpus not in (None, 0):
            return ("a full-parameter trainer that reads samples from the store starts no "
                    "inference engine: --rollout-num-gpus must be 0 or absent")
        return None
    if data_parallel != 1 and (gap := _rules().full_parameter_dp_replica_gap()) is not None:
        return ("dense full-parameter GRPO with engines on the island requires DP=1 "
                f"with this Miles pin (DP>1 needs {gap})")
    if (
        (actor_num_nodes != 1 and island_gpus_per_node is None)
        or type(rollout_gpus) is not int
        or rollout_gpus < 1
        or type(rollout_gpus_per_engine) is not int
        or rollout_gpus_per_engine < 1
        or rollout_gpus % rollout_gpus_per_engine
    ):
        return ("full-parameter mode with engines on the island requires one node (or a "
                "multi-node island, --rl-island-gpus-per-node) and dedicated, evenly "
                "partitioned inference engines (never colocated); or read samples from the "
                "store (--rl-sample-role trainer)")
    return None


def _resolve_ref_load(args, model_path) -> str:
    """Reference-model path; the checkpoint-format rules belong to the backend (4.2)."""
    return _rules().resolve_ref_load(args, model_path)


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
    verify_ref_load: bool = True,
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
    context_parallel = int(getattr(args, "context_parallel", 1) or 1)
    model_parallel = tensor_parallel * pipeline_parallel * context_parallel
    rules = _rules()
    rules.check_trainer_parallel(actor_gpus, tensor_parallel, pipeline_parallel)
    if context_parallel < 1 or actor_gpus % model_parallel:
        raise ValueError("trainer world must be divisible by TP*PP*CP")
    is_moe = getattr(provider, "num_moe_experts", None) is not None
    expert_parallel = getattr(args, "expert_parallel", None) or (
        actor_gpus if is_moe else 1
    )
    if not is_moe and expert_parallel != 1:
        raise ValueError("EP>1 requires a MoE model")
    rules.check_expert_parallel(actor_gpus, expert_parallel)
    fn_variant = qwen3_8_next_variant(args, provider)
    # the native Flash-Next plugin shards routed-expert LoRA by EP itself
    if fn_variant is None and is_moe and expert_parallel > 1 and args.lora_targets == "all-linear":
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
    dedicated_rollout_gpus: int | None = None
    if parameter_mode == "full":
        rollout_gpus = getattr(args, "rollout_num_gpus", None)
        rollout_gpus_per_engine = getattr(args, "rollout_num_gpus_per_engine", None)
        role = sample_roles.sample_role(args)
        island_gpus_per_node = getattr(args, "rl_island_gpus_per_node", None)
        if problem := full_parameter_layout_problem(
                data_parallel=data_parallel, actor_num_nodes=args.actor_num_nodes,
                rollout_gpus=rollout_gpus, rollout_gpus_per_engine=rollout_gpus_per_engine,
                sample_role=role, island_gpus=actor_gpus, island_gpus_per_node=island_gpus_per_node):
            raise ValueError(problem)
        if role == sample_roles.TRAINER:
            rollout_gpus = 0  # trainer-only island (Miles --debug-train-only): no engine
        if qwen35_recipe and (
            rollout_gpus_per_engine != 1
            or getattr(args, "sglang_tp_size", None) not in {None, 1}
        ):
            raise ValueError("pinned Qwen3.5 requires TP1 SGLang inference engines")
        if role == sample_roles.SAMPLER:
            # Engines only (Miles --debug-rollout-only): the island GPUs are the
            # engine partition, as on a LoRA sampler island; no trainer GPU.
            visible_gpus_per_node = args.actor_num_gpus_per_node
        else:
            dedicated_rollout_gpus = rollout_gpus
            # A multi-node island exposes every GPU of each node; the trainer
            # rectangle and the engines are laid out over the island.
            visible_gpus_per_node = (args.actor_num_gpus_per_node + rollout_gpus
                                     if island_gpus_per_node is None else int(island_gpus_per_node))
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

    ref_load = (_resolve_ref_load(args, model_path) if verify_ref_load
                else (getattr(args, "megatron_ref_load", None) or str(model_path)))
    global_batch = args.groups_per_round * args.samples_per_group // args.optimizer_steps
    rules.check_global_batch(global_batch, data_parallel)

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
    elif recipe == RECIPE_GLM5_NEXT:
        if parameter_mode != "full":
            raise ValueError("the glm5_next recipe is full-parameter only")
        if getattr(args, "megatron_ref_load", None) is None:
            # Miles loads an HF checkpoint only through Megatron-Bridge, which has no
            # glm5_next bridge (Modal run-7): train from the converted torch_dist checkpoint
            raise ValueError(
                "the glm5_next recipe needs --megatron-ref-load <dir> (Miles "
                "tools/convert_hf_to_torch_dist.py output; Megatron-Bridge cannot load glm5_next)"
            )
        # Megatron picks the backend (Miles run_glm5_3_flash.py passes none)
        recipe_name, attention_backend = RECIPE_GLM5_NEXT, "auto"
    elif fn_variant is not None:
        if parameter_mode != "lora":
            raise ValueError("the Qwen3.8-Flash-Next recipe is LoRA-only")
        if getattr(args, "megatron_ref_load", None) is None:
            # --megatron-to-hf-mode raw loads a Megatron torch_dist checkpoint, not
            # the HF snapshot (run_qwen3_8_next.py: --ref-load <ckpt>/<type>_torch_dist).
            raise ValueError(
                "the Qwen3.8-Flash-Next recipe needs --megatron-ref-load "
                "<dir>/<megatron_model_type>_torch_dist (raw mode cannot load the HF snapshot)"
            )
        recipe_name, attention_backend = RECIPE_QWEN3_8_NEXT, "flash"
        if getattr(args, "sglang_deterministic_inference", True):
            # S11 try26: the recipe's GDN prefill backend is FlashInfer (Miles
            # run_qwen3_8_next.py: --sglang-linear-attn-prefill-backend flashinfer),
            # which SGLang rejects under --enable-deterministic-inference; Miles'
            # validated recipe runs non-deterministic inference.
            raise ValueError(
                "the Qwen3.8-Flash-Next recipe needs --no-sglang-deterministic-inference "
                "(FlashInfer GDN prefill is not supported with deterministic inference)"
            )
    elif _is_nemotron_h(provider):
        _nemotron_h_recipe().check_provider(provider, parameter_mode=parameter_mode)
        # Miles run_nemotron_3_nano.py: --attention-backend auto
        recipe_name, attention_backend = RECIPE_NEMOTRON_H, "auto"
    elif qwen35_recipe:
        recipe_name, attention_backend = RECIPE_QWEN3_5, "flash"
    else:
        recipe_name, attention_backend = RECIPE_GENERIC, "unfused"

    # rlf-codex-rl 5.8: --rl-training-attention-backend overrides the recipe's
    # default (e.g. flash -> unfused when the flash-attn cute JIT fails to
    # compile on a given GPU). Recipes whose kernels hard-require flash
    # attention refuse any other backend.
    backend_override = getattr(args, "rl_training_attention_backend", "auto") or "auto"
    if backend_override != "auto":
        if recipe_name in {RECIPE_DEEPSEEK_V4_FLASH, RECIPE_QWEN3_8_NEXT} and backend_override != "flash":
            raise ValueError(
                f"the {recipe_name} recipe requires --rl-training-attention-backend flash "
                "(its kernels have no unfused/fused path)"
            )
        attention_backend = backend_override

    transformer_impl = str(getattr(args, "rl_transformer_impl", None) or TRANSFORMER_IMPLS[0])
    if transformer_impl not in TRANSFORMER_IMPLS:
        raise ValueError(f"--rl-transformer-impl {transformer_impl!r} is not one of {TRANSFORMER_IMPLS}")
    if transformer_impl != TRANSFORMER_IMPLS[0] and recipe_name != RECIPE_GLM5_NEXT:
        raise ValueError(f"--rl-transformer-impl {transformer_impl} is supported by the "
                         f"{RECIPE_GLM5_NEXT} recipe only")

    _validate_callable_spec(args.reward_function)

    eval_config = _resolve_eval(
        args,
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
            transformer_impl=transformer_impl,
            model_flags=_recipe_model_flags(recipe_name, provider),
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
                                  # m5: a multi-node colocated island keeps it too (node rules)
                                  else None),
            bundle_map=_island_bundle_map(args, dedicated_rollout_gpus),
            allow_cross_node_tp=bool(getattr(args, "rl_allow_cross_node_tp", False)),
            allow_cross_node_engine_tp=bool(getattr(args, "rl_allow_cross_node_engine_tp", False)),
            context_parallel=context_parallel,
        ),
        trainable=TrainableConfig(
            parameter_mode=parameter_mode,
            lora_rank=args.lora_r,
            lora_dropout=_lora_dropout(args, parameter_mode),
            lora_expert_rank=_lora_expert_rank(args, recipe_name),
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
                island_scheduling="elastic" if getattr(args, "rl_island_pool", False) else "legacy",
                requested=getattr(args, "rl_lr_schedule", None) or "auto",
                full_parameter_async=getattr(args, "full_parameter_rounds", "sweep") == "async",
            ),
            observe_training_entropy=bool(getattr(args, "observe_training_entropy", False)),
            mamba_training_ssm_states_dtype=getattr(args, "mamba_training_ssm_states_dtype", None) or None,
            seed=args.seed,
            critic=resolve_critic_run_config(args),
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
            sampler_snapshot=bool(getattr(args, "rl_sampler_snapshot", False)),
            rollout_quant_tier=int(getattr(args, "rl_rollout_quant_tier", 0) or 0),
            rollout_quant_fp8_source=_fp8_source(args),
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
            session_server_workers=(
                int(args.session_server_workers)
                if getattr(args, "session_server_workers", None) is not None
                else None
            ),
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
        edge_costs_path=getattr(args, "rl_edge_costs_path", None) or None,
        elastic_window_s=(float(args.rl_elastic_window_s)
                          if getattr(args, "rl_elastic_window_s", None) else None),
        sample_buffer=sample_roles.resolve(args, groups_per_round=args.groups_per_round),
        train_memory=_train_memory(args),
    )


def _fp8_source(args) -> str:
    from ..quant.tiers import fp8_source

    return fp8_source(args)


def _train_memory(args) -> TrainMemoryConfig:
    max_tokens = getattr(args, "rl_max_tokens_per_gpu", None)
    cp = int(getattr(args, "context_parallel", 1) or 1)
    if max_tokens is not None and int(max_tokens) * cp < int(args.seq_len):
        # a sample longer than the budget would not fit in any micro batch
        # (Miles packs max_tokens_per_gpu * CP tokens per micro batch: dp_schedule.py)
        raise ValueError(
            f"--rl-max-tokens-per-gpu {max_tokens} x CP {cp} is below --seq-len {args.seq_len}"
        )
    offload = float(getattr(args, "rl_optimizer_offload_fraction", 0.0) or 0.0)
    if not 0.0 <= offload <= 1.0:
        raise ValueError(f"--rl-optimizer-offload-fraction {offload} is not in [0, 1]")
    dp_optimizer = getattr(args, "rl_dp_optimizer", None) or "auto"
    if dp_optimizer not in DP_OPTIMIZERS:
        raise ValueError(f"--rl-dp-optimizer {dp_optimizer!r} is not one of {', '.join(DP_OPTIMIZERS)}")
    return TrainMemoryConfig(
        activation_recompute=bool(getattr(args, "rl_activation_recompute", False)),
        max_tokens_per_gpu=int(max_tokens) if max_tokens is not None else None,
        optimizer_offload_fraction=offload,
        gradient_accumulation_fusion=bool(getattr(args, "rl_gradient_accumulation_fusion", False)),
        dp_optimizer=dp_optimizer,
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


def _lora_expert_rank(args, recipe_name: str) -> int | None:
    value = getattr(args, "rl_lora_expert_rank", None)
    if recipe_name != RECIPE_QWEN3_8_NEXT:
        if value is not None:
            raise ValueError("--rl-lora-expert-rank only applies to the Qwen3.8-Flash-Next recipe")
        return None
    if value is None:
        from yeto.rl.profiles.qwen3_8_next import Qwen38NextLoraProfile

        value = Qwen38NextLoraProfile.lora_expert_rank  # profile default (8)
    value = int(value)
    if not 0 <= value <= args.lora_r:
        raise ValueError(
            f"--rl-lora-expert-rank must satisfy 0 <= r_e <= --lora-r ({value} vs {args.lora_r})"
        )
    return value


def _lora_dropout(args, parameter_mode: str) -> float:
    value = getattr(args, "rl_lora_dropout", None)
    if value is None:
        return 0.0
    value = float(value)
    if not 0.0 <= value < 1.0:
        raise ValueError("--rl-lora-dropout must be in [0, 1)")
    if parameter_mode != "lora":
        raise ValueError("--rl-lora-dropout only applies to LoRA training")
    return value


def ports_training_eval(args) -> bool:
    """Training-time heldout eval on a ports island (not eval-only).

    Dense full-parameter islands use the same driver evaluation (rollout 0,
    every ``interval`` rollouts and the final one; P0 1.7b.1b).

    Shared by the learner's dataset-identity check and :func:`_resolve_eval`
    so both accept exactly the same runs.
    """

    return not getattr(args, "eval_only", False)


def _resolve_eval(
    args,
    *,
    prompt_path,
    eval_prompt_path,
    yeto_policy_sync: bool,
) -> EvalConfig | None:
    eval_interval = getattr(args, "eval_interval", None)
    if eval_interval is None:
        return None
    if not yeto_policy_sync and not getattr(args, "rl_single_island_no_sync", False):
        # A single island without outer sync owns its policy versions (it publishes
        # every round to its own engines), so the driver's training-time eval needs no
        # syncer (S27 G-DA: held-out agentic eval at round 0 and at the last round).
        raise ValueError("Yeto evaluation requires the external policy boundary")
    if eval_interval <= 0:
        raise ValueError("evaluation interval must be positive")
    eval_only = getattr(args, "eval_only", False)
    ports_train_eval = ports_training_eval(args)
    if eval_only:
        if eval_interval != 1:
            raise ValueError("SSH evaluation must be one separate eval-only run")
        selected = prompt_path
    elif ports_train_eval:
        # Ports LoRA island (rl-infra-spec 2.3 / launcher wiring): Miles runs
        # the heldout eval every ``interval`` rollouts through the driver's
        # evaluate port; a distinct heldout split is mandatory.
        if eval_prompt_path is None:
            raise ValueError("ports training-time evaluation requires heldout prompt data")
        selected = eval_prompt_path
    else:
        raise ValueError("training-time evaluation is restricted to the ports engine")
    eval_name = getattr(args, "eval_dataset_name", None)
    eval_samples = getattr(args, "eval_samples_per_prompt", None)
    if not eval_name or not isinstance(eval_samples, int) or eval_samples <= 0:
        raise ValueError("evaluation requires a dataset name and positive sample count")
    # Eval-only reuses its sole normalized prompt file; training binds a
    # separately normalized, immutable heldout split.
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
