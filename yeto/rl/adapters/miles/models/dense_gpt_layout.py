"""Offline full-parameter fragment layout of a Miles dense Qwen3 actor.

From the HF ``config.json`` alone (no GPU, no weights) this module lists the
per-rank FP32 master specs a Miles Megatron ``GPTModel`` actor reports
(Megatron-Core TE layer spec: fused input norms in ``linear_qkv`` /
``linear_fc1``, QK norms ``q_layernorm`` / ``k_layernorm``, fused gate/up in
``linear_fc1``) in Miles' global form (``module.module.`` prefix, global layer
indices), and builds the island's owner-affine layout with the function the
island uses (:func:`yeto.rl.local_learner.fit_owner_affine_layout`). The
launcher uses it to check ``--fragments`` before any GPU starts: the island
requires its layout to have exactly ``--fragments`` fragments
(``MilesChunkedFullParameterAdapter.initialize``).

Checked against a measured island layout (tests): Qwen3-4B-Instruct-2507 at
TP2, ``--rl-sync-matrix-merge rda``, 2 GiB fragments gives 290 specs and
2,011,332,096 scalars per rank, 10 fragments and the island's layout hash.

Exact at PP=1, CP=1, DP=1 (every rank holds its own TP slice; replicated norms
appear on every TP rank; the vocabulary is not padded). At PP>1 the even layer
split and the last stage's output layer are an estimate (:attr:`exact` False):
Megatron may place layers unevenly.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from math import prod

_PREFIX = "module.module."
# model_type values whose Megatron parameter set this module lists.
SUPPORTED_MODEL_TYPES = frozenset({"qwen3"})


class DenseLayoutError(ValueError):
    """The model or the parallelism is outside the offline layout's scope."""


@dataclass(frozen=True)
class DenseGeometry:
    """The parameter-shaping fields of a dense Qwen3 language model."""

    hidden_size: int
    ffn_hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    tie_word_embeddings: bool
    attention_bias: bool

    @classmethod
    def from_hf_config(cls, config: Mapping) -> DenseGeometry:
        model_type = config.get("model_type")
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise DenseLayoutError(f"model_type {model_type!r} is not modelled "
                                   f"(supported: {sorted(SUPPORTED_MODEL_TYPES)})")
        try:
            hidden = int(config["hidden_size"])
            heads = int(config["num_attention_heads"])
            return cls(
                hidden_size=hidden,
                ffn_hidden_size=int(config["intermediate_size"]),
                num_layers=int(config["num_hidden_layers"]),
                num_attention_heads=heads,
                num_key_value_heads=int(config.get("num_key_value_heads") or heads),
                head_dim=int(config.get("head_dim") or hidden // heads),
                vocab_size=int(config["vocab_size"]),
                tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
                attention_bias=bool(config.get("attention_bias", False)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DenseLayoutError(f"config.json lacks a dense-model field: {exc}") from None


def _divide(value: int, parts: int, what: str) -> int:
    if value % parts:
        raise DenseLayoutError(f"{what} ({value}) is not divisible by {parts}")
    return value // parts


def _shard_id(tp_rank: int, tp: int, pp_rank: int, pp: int) -> str:
    return f"tp{tp_rank}-of-{tp}.pp{pp_rank}-of-{pp}.ep0-of-1.cp0-of-1.dp0-of-1"


def _layer_specs(g: DenseGeometry, layer: int, tp: int) -> list[tuple[str, tuple[int, ...]]]:
    h = g.hidden_size
    base = f"{_PREFIX}decoder.layers.{layer}."
    query = g.num_attention_heads * g.head_dim
    qkv = _divide(query + 2 * g.num_key_value_heads * g.head_dim, tp, "QKV width")
    ffn = _divide(g.ffn_hidden_size, tp, "MLP width")
    specs = [
        (base + "self_attention.linear_qkv.weight", (qkv, h)),
        (base + "self_attention.linear_qkv.layer_norm_weight", (h,)),
        (base + "self_attention.q_layernorm.weight", (g.head_dim,)),
        (base + "self_attention.k_layernorm.weight", (g.head_dim,)),
        (base + "self_attention.linear_proj.weight", (h, _divide(query, tp, "query width"))),
        (base + "mlp.linear_fc1.weight", (2 * ffn, h)),
        (base + "mlp.linear_fc1.layer_norm_weight", (h,)),
        (base + "mlp.linear_fc2.weight", (h, ffn)),
    ]
    if g.attention_bias:
        specs.append((base + "self_attention.linear_qkv.bias", (qkv,)))
    return specs


def rank_specs(geometry: DenseGeometry, *, tp: int, pp: int = 1, role: str = "actor"):
    """Every training rank's FP32 master specs (CP=1, DP=1), as
    :class:`~yeto.rl.local_learner.ParameterSpec` in (pp, tp) rank order."""
    from yeto.rl.local_learner import ParameterSpec

    if tp < 1 or pp < 1:
        raise DenseLayoutError("parallel sizes must be positive")
    g = geometry
    per_stage = _divide(g.num_layers, pp, "layers")
    vocab = _divide(g.vocab_size, tp, "vocabulary")
    out = []
    for pp_rank in range(pp):
        for tp_rank in range(tp):
            named: list[tuple[str, tuple[int, ...]]] = []
            if pp_rank == 0:
                named.append((_PREFIX + "embedding.word_embeddings.weight", (vocab, g.hidden_size)))
            for layer in range(pp_rank * per_stage, (pp_rank + 1) * per_stage):
                named += _layer_specs(g, layer, tp)
            if pp_rank == pp - 1:
                named.append((_PREFIX + "decoder.final_layernorm.weight", (g.hidden_size,)))
                # Tied at PP=1: the output layer is the embedding (no own parameter).
                if not g.tie_word_embeddings or pp > 1:
                    named.append((_PREFIX + "output_layer.weight", (vocab, g.hidden_size)))
            sid = _shard_id(tp_rank, tp, pp_rank, pp)
            out += [ParameterSpec(role, name, shape, "float32", prod(shape), sid)
                    for name, shape in sorted(named)]
    return tuple(out)


@dataclass(frozen=True)
class OfflineLayout:
    fragments: int
    layout_hash: str | None  # None without a commit revision and config hash
    fragment_bytes: tuple[int, ...]
    ranks: int
    exact: bool


def offline_layout(config: Mapping, *, tp: int, pp: int = 1, minimum_fragments: int,
                   max_fragment_bytes: int = 2 << 30, matrix_merge: str = "avg",
                   model_revision: str = "", config_sha256: str = "") -> OfflineLayout:
    """The layout a Miles dense Qwen3 island builds for ``--fragments
    minimum_fragments`` (shape-bound, GRPO actor). ``exact`` is False at PP>1.
    The fragments do not depend on the component identity; the layout hash does,
    so it is only given with a commit ``model_revision`` and ``config_sha256``."""
    from yeto.rl.local_learner import ComponentIdentity, fit_owner_affine_layout

    geometry = DenseGeometry.from_hf_config(config)
    specs = rank_specs(geometry, tp=tp, pp=pp)
    revision = model_revision.lower()
    identified = bool(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision)
                      and re.fullmatch(r"[0-9a-f]{64}", config_sha256))
    component = ComponentIdentity(role="actor", model_revision=revision if identified else "0" * 40,
                                  config_hash=config_sha256 if identified else "0" * 64)
    layout = fit_owner_affine_layout(
        algorithm="grpo", components=(component,), specs=specs,
        minimum_fragments=minimum_fragments, max_fragment_bytes=max_fragment_bytes,
        matrix_merge=matrix_merge)
    return OfflineLayout(
        fragments=layout.fragments.num_fragments,
        layout_hash=layout.layout_hash if identified else None,
        fragment_bytes=tuple(f.numel * 4 for f in layout.fragments.fragments),
        ranks=tp * pp,
        exact=pp == 1,
    )
