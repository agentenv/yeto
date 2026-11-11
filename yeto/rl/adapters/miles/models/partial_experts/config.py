"""Per-model declaration of a partial-experts training contract.

One :class:`PartialExpertModelConfig` per model says what the generic layer
(:mod:`.freeze`, :mod:`.optimizer`, :mod:`.export`) may assume. The two
strategies:

``clone_prefix``
    the checkpoint is expanded with cloned experts and a prefix of the clones
    is trained (DeepSeek V4); the engine needs a routing patch;
``in_place_permuted``
    the original experts are trained in place; the trainer permutes experts
    into EP slots at load time from an ``expert_selection.json``, the engine
    sees the unmodified model (Kimi-K3.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

CLONE_PREFIX = "clone_prefix"
IN_PLACE_PERMUTED = "in_place_permuted"
STRATEGIES = (CLONE_PREFIX, IN_PLACE_PERMUTED)


@dataclass(frozen=True)
class PartialExpertModelConfig:
    model_type: str  # HF ``model_type``
    num_routed_experts: int  # experts the router scores (original ids)
    num_moe_layers: int
    trainable_strategy: str
    ep_size_default: int
    # Megatron ``--use-distributed-optimizer``: DSV4 streams complete FP32
    # masters per rank (DiLoCo) and refuses it; K3.1 GRPO allows it.
    allow_distributed_optimizer: bool
    # Whether the FP32 master of every trainable parameter must be complete
    # (not DP-sharded) on this rank; True whenever a caller streams masters.
    requires_complete_master: bool
    # Attribute the freeze step sets on trainable expert parameters; the
    # optimizer groups them with ``ParamKey(attr=...)``.
    trainable_param_attr: str
    # Whether the rollout engine needs a yeto patch (clone routing); the
    # in-place strategy never does.
    needs_engine_patch: bool
    # Default experts trained per layer (None: per-run count, DSV4).
    num_trainable_per_layer: int | None = None
    # Megatron parameter-name fragment of a trainable expert weight.
    trainable_param_fragment: str = ".mlp.experts."
    # Storage dtype of rollout routed-expert ids (uint8 only below 256 experts).
    routed_experts_dtype: str = "uint16"
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.trainable_strategy not in STRATEGIES:
            raise ValueError(f"trainable_strategy must be one of {STRATEGIES}")
        if self.num_routed_experts <= 0 or self.num_moe_layers <= 0 or self.ep_size_default <= 0:
            raise ValueError("expert, layer and EP counts must be positive")
        if self.num_trainable_per_layer is not None and not (
            0 < self.num_trainable_per_layer <= self.num_routed_experts
        ):
            raise ValueError("num_trainable_per_layer must be in (0, num_routed_experts]")
        if self.requires_complete_master and self.allow_distributed_optimizer:
            # A sharded master is never complete: the two flags cannot both hold.
            raise ValueError(
                "requires_complete_master and allow_distributed_optimizer are exclusive"
            )
        if self.trainable_strategy == IN_PLACE_PERMUTED and self.needs_engine_patch:
            raise ValueError("the in-place permuted strategy never patches the engine")

    def check_ep_size(self, ep_size: int) -> int:
        ep = int(ep_size)
        if ep <= 0 or self.num_routed_experts % ep:
            raise ValueError(f"EP {ep} must divide {self.num_routed_experts} experts")
        if self.num_trainable_per_layer is not None and self.num_trainable_per_layer % ep:
            raise ValueError(
                f"EP {ep} must divide the {self.num_trainable_per_layer} trainable experts per layer"
            )
        return ep


_REGISTRY: dict[str, PartialExpertModelConfig] = {}


def register_model(config: PartialExpertModelConfig) -> PartialExpertModelConfig:
    current = _REGISTRY.get(config.model_type)
    if current is not None and current != config:
        raise ValueError(f"partial-experts model {config.model_type!r} registered twice")
    _REGISTRY[config.model_type] = config
    return config


def model_config(model_type: str) -> PartialExpertModelConfig:
    known_model_configs()
    try:
        return _REGISTRY[model_type]
    except KeyError:
        raise KeyError(
            f"no partial-experts contract for {model_type!r}; known: {sorted(_REGISTRY)}"
        ) from None


def known_model_configs() -> dict[str, PartialExpertModelConfig]:
    """Every registered model (imports the model modules that register one)."""

    import importlib

    for module in (
        "yeto.rl.adapters.miles.models.deepseek_v4.expert_full_runtime",
        "yeto.rl.adapters.miles.models.kimi_k31.model",
    ):
        importlib.import_module(module)
    return dict(_REGISTRY)
