"""Dataclasses of the model registry (A21 section 3.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

STATUS_ORDER = {"verified": 0, "smoke-only": 1, "inferred": 2, "unsupported": 9}


@dataclass(frozen=True)
class ModelFacts:
    """What the registry reads from a ``config.json`` (no weights are read)."""

    architectures: tuple[str, ...]
    model_type: str
    text_model_type: str | None
    moe: bool
    num_experts: int
    topk: int
    first_k_dense: int
    total_params_b: float
    active_params_b: float
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    vocab: int
    max_positions: int
    attention: str          # gqa | mla | dsa | gdn-hybrid | mamba-hybrid
    checkpoint_quant: str   # bf16 | fp8 | mxfp4 | ...
    vl: bool
    breakdown: dict[str, float] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class HardwareFacts:
    device_family: str      # cuda | ascend (| rocm for a sampler-only island)
    arch: str               # hopper | blackwell | 910b | gfx950
    card: str
    hbm_gib: int
    cards_per_node: int = 8
    max_nodes: int = 64
    interconnect: str = "nvlink+ib"   # nvlink+ib | roce | gloo-only | none


@dataclass(frozen=True)
class Constraints:
    tuning: str | None = None           # lora | full
    sampler_engine: str | None = None   # sglang | vllm
    trainer_backend: str | None = None  # miles | verl
    max_cards: int | None = None
    context_len: int = 8192


@dataclass(frozen=True)
class QuantPlan:
    train_dtype: str
    sampler_weights: str
    release_format: str
    requant: str = "none"


@dataclass(frozen=True)
class BackendRecipe:
    name: str
    hardware: tuple[str, ...]           # device families of the trainer island
    trainer: str                        # miles | verl-fsdp2 | verl-megatron
    sampler: str                        # sglang | vllm | vllm-ascend
    rl_model_recipe: str | None         # a --rl-model-recipe value (None: unsupported)
    tunings: tuple[str, ...]
    size_band_b: tuple[float, float]
    trainer_image: str                  # "module:CONST" or ""
    sampler_image: str
    quant: QuantPlan
    status: str                         # verified | smoke-only | inferred | unsupported
    evidence: str
    hard_constraints: tuple[str, ...] = ()
    pitfalls: tuple[str, ...] = ()
    sampler_hardware: str | None = None  # a separate sampler island (e.g. rocm:gfx950)
    reason: str = ""                    # status unsupported: why
    alternative: str = ""               # status unsupported: what to do instead


@dataclass(frozen=True)
class FamilyEntry:
    family: str
    match_architectures: tuple[str, ...]
    match_model_types: tuple[str, ...]
    recipes: tuple[BackendRecipe, ...]
    pitfalls: tuple[str, ...] = ()


@dataclass
class Recommendation:
    facts: ModelFacts
    hardware: HardwareFacts
    family: str
    supported: bool
    recipe: BackendRecipe | None
    trainer_parallel: dict[str, Any]
    sampler_parallel: dict[str, Any]
    memory: dict[str, Any]
    pitfalls: list[str]
    rejected: list[tuple[str, str]]
    warnings: list[str]
    alternatives: list[str]

    def to_cli_flags(self) -> list[str]:
        from .render import cli_flags
        return cli_flags(self)
