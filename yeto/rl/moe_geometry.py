"""Routed-expert geometry of a mixture-of-experts checkpoint, from its config.

Which decoder layers are MoE layers and how many routed experts each has is
what partial-expert training, the selection tool and the full-parameter layout
checks need, for any MoE model rather than a list of model names. Two sources
give it and must agree:

* a Hugging Face ``config.json`` (:func:`from_hf_config`), read before any
  cloud work. The fields follow the transformers decoder layers:
  ``decoder_sparse_step`` / ``mlp_only_layers`` (``Qwen2MoeDecoderLayer``,
  ``Qwen3MoeDecoderLayer``), ``first_k_dense_replace`` / ``moe_layer_freq``
  (``DeepseekV2DecoderLayer``, ``DeepseekV3DecoderLayer``, GLM-4-MoE,
  Kimi-K3), ``moe_layers`` / ``interleave_moe_layer_step`` (Llama-4),
  ``hybrid_override_pattern`` ``E`` symbols (Nemotron-H); a config
  with routed experts and none of them is MoE in every layer (Mixtral, OLMoE,
  Phi-MoE). A multimodal config's ``text_config`` / ``llm_config`` is read;
* a Megatron-Bridge model provider (:func:`from_bridge_provider`), the view the
  trainer builds. ``moe_layer_freq`` has Megatron's meaning
  (``megatron/core/models/gpt/gpt_layer_specs.py``,
  ``get_gpt_decoder_block_spec``): an integer ``N`` makes layer ``i`` MoE when
  ``i % N == 0``, a list is the per-layer 0/1 mask.

Layer indices are 0-based HF decoder-layer indices, the same numbers Megatron
global parameter names carry (``decoder.layers.L``) and an
``expert_selection.json`` keys its layers by.

Import-light: no torch, transformers or megatron import.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# Routed-expert count and top-k under their transformers names, first found wins.
_EXPERT_COUNT_FIELDS = ("n_routed_experts", "num_experts", "num_local_experts", "moe_num_experts")
_TOPK_FIELDS = ("num_experts_per_tok", "num_experts_per_token", "moe_topk", "moe_router_topk")
# Nested language-model configs of multimodal checkpoints.
_TEXT_CONFIG_FIELDS = ("text_config", "llm_config", "language_config")

# One routed expert's weight in an HF checkpoint:
# ``model.layers.L.mlp.experts.E.gate_proj.weight`` (Qwen-MoE, DeepSeek, GLM, OLMoE),
# ``model.layers.L.block_sparse_moe.experts.E.w1.weight`` (Mixtral, Kimi). Shared
# experts (``shared_experts.gate_proj``, ``shared_expert.gate_proj``) carry no index.
_HF_EXPERT = re.compile(
    r"(?:^|\.)layers\.(\d+)\.(?:mlp|block_sparse_moe|moe|feed_forward|mixer)\.experts\.(\d+)\."
)


class MoeGeometryError(ValueError):
    """The config does not describe a routed-expert model this module understands."""


@dataclass(frozen=True)
class RoutedExpertGeometry:
    """Routed experts of one decoder: which layers, how many experts each."""

    num_layers: int
    moe_layers: tuple[int, ...]
    num_experts: int
    router_topk: int | None = None
    model_type: str = ""

    def __post_init__(self) -> None:
        if self.num_layers <= 0 or self.num_experts <= 0:
            raise MoeGeometryError("layer and expert counts must be positive")
        if not self.moe_layers:
            raise MoeGeometryError("a routed-expert model needs at least one MoE layer")
        if list(self.moe_layers) != sorted(set(self.moe_layers)):
            raise MoeGeometryError("MoE layers must be unique and ascending")
        if self.moe_layers[0] < 0 or self.moe_layers[-1] >= self.num_layers:
            raise MoeGeometryError(f"MoE layers must lie in [0, {self.num_layers})")
        if self.router_topk is not None and not 0 < self.router_topk <= self.num_experts:
            raise MoeGeometryError(f"router top-k {self.router_topk} outside (0, {self.num_experts}]")

    @property
    def num_moe_layers(self) -> int:
        return len(self.moe_layers)

    def check_ep_size(self, ep_size: int) -> int:
        ep = int(ep_size)
        if ep <= 0 or self.num_experts % ep:
            raise MoeGeometryError(f"expert parallel size {ep} must divide {self.num_experts} experts")
        return ep

    def same_as(self, other: "RoutedExpertGeometry") -> None:
        """Raise when two sources disagree on layers or expert count."""

        mine = (self.num_layers, self.moe_layers, self.num_experts)
        theirs = (other.num_layers, other.moe_layers, other.num_experts)
        if mine != theirs:
            raise MoeGeometryError(
                f"MoE geometry differs: {self.num_layers} layers / {self.num_experts} experts / "
                f"MoE {_span(self.moe_layers)} vs {other.num_layers} / {other.num_experts} / "
                f"{_span(other.moe_layers)}"
            )


def _span(layers: Sequence[int]) -> str:
    return f"{len(layers)} layers {list(layers[:3])}..{list(layers[-1:])}" if layers else "none"


def _field(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def text_config(config: Any) -> Any:
    """The language-model part of a (possibly multimodal) HF config."""

    for name in _TEXT_CONFIG_FIELDS:
        inner = _field(config, name)
        if isinstance(inner, Mapping) or (inner is not None and hasattr(inner, "num_hidden_layers")):
            return inner
    return config


def _first(config: Any, names: Sequence[str]) -> Any:
    for name in names:
        value = _field(config, name)
        if value is not None:
            return value
    return None


def megatron_moe_layer_mask(num_layers: int, moe_layer_freq: Any) -> tuple[int, ...]:
    """Megatron's per-layer MoE mask for ``--moe-layer-freq``.

    ``get_gpt_decoder_block_spec``: an integer ``N`` gives
    ``[1 if i % N == 0 else 0 for i in range(num_layers)]``; a list must have
    one 0/1 entry per layer.
    """

    if isinstance(moe_layer_freq, bool):
        raise MoeGeometryError(f"invalid moe_layer_freq {moe_layer_freq!r}")
    if isinstance(moe_layer_freq, int):
        if moe_layer_freq <= 0:
            raise MoeGeometryError(f"moe_layer_freq must be positive, got {moe_layer_freq}")
        return tuple(1 if i % moe_layer_freq == 0 else 0 for i in range(num_layers))
    if isinstance(moe_layer_freq, (list, tuple)):
        mask = tuple(int(value) for value in moe_layer_freq)
        if len(mask) != num_layers or set(mask) - {0, 1}:
            raise MoeGeometryError(
                f"moe_layer_freq list must hold one 0/1 entry per layer ({num_layers}), got {len(mask)}"
            )
        return mask
    raise MoeGeometryError(f"unsupported moe_layer_freq {moe_layer_freq!r}")


def _layers_of_mask(mask: Sequence[int]) -> tuple[int, ...]:
    return tuple(i for i, value in enumerate(mask) if value)


def _hf_moe_layers(text: Any, num_layers: int) -> tuple[int, ...]:
    pattern = _field(text, "hybrid_override_pattern")
    if isinstance(pattern, str) and pattern:
        # Nemotron-H: one symbol per layer, "E" is a MoE layer.
        if len(pattern) != num_layers:
            raise MoeGeometryError(
                f"hybrid_override_pattern has {len(pattern)} symbols for {num_layers} layers"
            )
        return tuple(i for i, symbol in enumerate(pattern) if symbol == "E")
    explicit = _field(text, "moe_layers")
    if isinstance(explicit, (list, tuple)) and explicit:
        # Llama-4: the config lists its MoE layers.
        return tuple(sorted({int(layer) for layer in explicit}))
    interleave = _field(text, "interleave_moe_layer_step")
    if interleave is not None:
        # Llama4TextDecoderLayer: (layer + 1) % interleave_moe_layer_step == 0.
        step = int(interleave)
        if step <= 0:
            raise MoeGeometryError(f"interleave_moe_layer_step must be positive, got {interleave}")
        return tuple(i for i in range(num_layers) if (i + 1) % step == 0)
    sparse_step = _field(text, "decoder_sparse_step")
    mlp_only = _field(text, "mlp_only_layers")
    first_dense = _field(text, "first_k_dense_replace")
    freq = _field(text, "moe_layer_freq")
    if sparse_step is not None or mlp_only is not None:
        # Qwen2MoeDecoderLayer / Qwen3MoeDecoderLayer.
        step = int(sparse_step or 1)
        if step <= 0:
            raise MoeGeometryError(f"decoder_sparse_step must be positive, got {sparse_step}")
        dense = {int(layer) for layer in (mlp_only or ())}
        return tuple(i for i in range(num_layers) if i not in dense and (i + 1) % step == 0)
    if first_dense is not None or freq is not None:
        # DeepseekV2/V3DecoderLayer, GLM-4-MoE, Kimi-K3: layer >= first_k_dense_replace and
        # layer % moe_layer_freq == 0; a list moe_layer_freq is Megatron's mask.
        dense = int(first_dense or 0)
        if isinstance(freq, (list, tuple)):
            mask = megatron_moe_layer_mask(num_layers, list(freq))
            return tuple(i for i in _layers_of_mask(mask) if i >= dense)
        every = int(freq or 1)
        if every <= 0:
            raise MoeGeometryError(f"moe_layer_freq must be positive, got {freq}")
        return tuple(i for i in range(dense, num_layers) if i % every == 0)
    # Mixtral, OLMoE, Phi-MoE, GraniteMoE: every decoder layer routes.
    return tuple(range(num_layers))


def from_hf_config(config: Any) -> RoutedExpertGeometry | None:
    """Routed-expert geometry of an HF config; ``None`` for a dense model."""

    text = text_config(config)
    experts = _first(text, _EXPERT_COUNT_FIELDS)
    if experts in (None, 0):
        return None
    layers = _field(text, "num_hidden_layers")
    if layers is None:
        raise MoeGeometryError("config has routed experts but no num_hidden_layers")
    topk = _first(text, _TOPK_FIELDS)
    model_type = str(_field(config, "model_type") or _field(text, "model_type") or "")
    return RoutedExpertGeometry(
        num_layers=int(layers),
        moe_layers=_hf_moe_layers(text, int(layers)),
        num_experts=int(experts),
        router_topk=None if topk is None else int(topk),
        model_type=model_type,
    )


def from_bridge_provider(provider: Any) -> RoutedExpertGeometry | None:
    """Routed-expert geometry of a Megatron-Bridge provider; ``None`` when dense.

    Reads ``num_layers``, ``num_moe_experts``, ``moe_layer_freq`` and
    ``moe_router_topk`` (``megatron.bridge.models.*Provider``, the fields
    ``AutoBridge.to_megatron_provider`` fills from the HF config).
    """

    experts = getattr(provider, "num_moe_experts", None)
    if not experts:
        return None
    layers = int(getattr(provider, "num_layers"))
    mask = megatron_moe_layer_mask(layers, getattr(provider, "moe_layer_freq", 1))
    topk = getattr(provider, "moe_router_topk", None)
    return RoutedExpertGeometry(
        num_layers=layers,
        moe_layers=_layers_of_mask(mask),
        num_experts=int(experts),
        router_topk=None if topk is None else int(topk),
        model_type=type(provider).__name__,
    )


def hf_expert_of(name: str) -> tuple[int, int] | None:
    """``(layer, expert)`` of a routed expert's HF weight name, else ``None``."""

    match = _HF_EXPERT.search(name)
    return None if match is None else (int(match.group(1)), int(match.group(2)))
