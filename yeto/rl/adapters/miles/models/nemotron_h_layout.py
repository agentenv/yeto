"""Offline full-parameter manifests and fragment layout of a Miles Nemotron-H actor.

From the HF ``config.json`` alone (no GPU, no weights) this module writes the
per-rank FP32 master specs a Miles Nemotron-H actor reports
(``miles/backends/megatron_utils/full_parameter_state.full_parameter_shard_manifest``):
Megatron-Core ``HybridModel`` parameter names in Miles' global form
(``named_weights._named_params_and_buffers_global``: ``module.module.`` prefix,
global layer and expert indices) and their per-rank shapes. The island's
owner-affine fragment layout then comes from the same function the island uses
(:func:`yeto.rl.local_learner.fit_owner_affine_layout`), so a planned
parallelism gets its fragment count and layout hash before any GPU starts.

The parameter set follows the Megatron-Core hybrid TE stack spec
(``megatron/core/models/hybrid/hybrid_layer_specs.py``: fused input norms in
``in_proj`` / ``linear_qkv``, ``pre_mlp_layernorm`` before the MoE, TE grouped
GEMM experts ``weight<E>``) and the Mamba mixer
(``megatron/core/ssm/mamba_mixer.py``: ``conv1d_weight``/``conv1d_bias``,
``dt_bias``, ``A_log``, ``D`` per head, gated RMSNorm ``norm.weight``). MoE
latent projections are duplicated on every TP rank
(``moe_layer.py``: ``parallel_mode="duplicated"``).

Checked against a measured island layout (tests): Nemotron-3 Nano 30B-A3B at
TP2/EP2 (S27 ``s27-gr-probe-b``) gives 3264 specs and 15,792,996,640 scalars
per rank, the two Miles per-rank manifest hashes, 60 fragments and the layout
hash ``0ec93ba8...``.

Scope (refused outside it): DP=1 and CP=1 with EP equal to TP, so every expert
has exactly one owner (:mod:`yeto.rl.moe_layout`); expert TP 1; untied
embeddings; no MTP parameters (the Miles Nemotron-H bridge drops the MTP block
unless ``MILES_NEMOTRONH_KEEP_MTP`` is set); the vision part of a VL
checkpoint is not built (Miles ``MilesNemotronHLanguageOnlyBridge``).

Import-light: no torch.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import prod

from yeto.rl.hybrid_params import hybrid_pattern, language_model_config
from yeto.rl.local_learner import (
    ComponentIdentity,
    ParameterLayout,
    ParameterSpec,
    fit_owner_affine_layout,
    layout_stream_role,
)

_PREFIX = "module.module."
_MILES_MANIFEST_SCHEMA = 1


class NemotronHLayoutError(ValueError):
    """The planned parallelism is outside the offline layout's scope."""


@dataclass(frozen=True)
class NemotronHGeometry:
    """The parameter-shaping fields of a NemotronH language model."""

    pattern: str
    hidden_size: int
    vocab_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    mamba_num_heads: int
    mamba_head_dim: int
    mamba_groups: int
    ssm_state_size: int
    conv_kernel: int
    ffn_hidden_size: int
    num_experts: int
    moe_ffn_hidden_size: int
    shared_expert_ffn_hidden_size: int
    moe_latent_size: int | None
    mtp_layers: int
    tie_word_embeddings: bool

    @classmethod
    def from_hf_config(cls, config: Mapping) -> NemotronHGeometry:
        llm = language_model_config(config)
        shared = int(llm.get("moe_shared_expert_intermediate_size") or 0) if llm.get("n_shared_experts") else 0
        latent = llm.get("moe_latent_size")
        return cls(
            pattern=hybrid_pattern(config),
            hidden_size=int(llm["hidden_size"]),
            vocab_size=int(llm["vocab_size"]),
            num_attention_heads=int(llm["num_attention_heads"]),
            num_key_value_heads=int(llm["num_key_value_heads"]),
            head_dim=int(llm["head_dim"]),
            mamba_num_heads=int(llm["mamba_num_heads"]),
            mamba_head_dim=int(llm["mamba_head_dim"]),
            mamba_groups=int(llm["n_groups"]),
            ssm_state_size=int(llm["ssm_state_size"]),
            conv_kernel=int(llm["conv_kernel"]),
            ffn_hidden_size=int(llm.get("intermediate_size") or 0),
            num_experts=int(llm.get("n_routed_experts") or 0),
            moe_ffn_hidden_size=int(llm.get("moe_intermediate_size") or 0),
            shared_expert_ffn_hidden_size=shared,
            moe_latent_size=None if latent is None else int(latent),
            mtp_layers=int(llm.get("num_nextn_predict_layers") or 0),
            tie_word_embeddings=bool(llm.get("tie_word_embeddings", config.get("tie_word_embeddings", False))),
        )

    @property
    def mamba_inner(self) -> int:
        return self.mamba_num_heads * self.mamba_head_dim

    def layer_counts(self) -> dict[str, int]:
        return {symbol: self.pattern.count(symbol) for symbol in "ME*-"}


@dataclass(frozen=True)
class RankManifest:
    """One training rank's FP32 master specs: ``(Miles global name, shape)`` in name order."""

    tp_rank: int
    tp_size: int
    pp_rank: int
    pp_size: int
    ep_rank: int
    ep_size: int
    specs: tuple[tuple[str, tuple[int, ...]], ...]

    @property
    def topology(self) -> dict[str, int]:
        # Field order of Miles ``FullParameterTopology`` (asdict).
        return {
            "tp_rank": self.tp_rank, "tp_size": self.tp_size,
            "pp_rank": self.pp_rank, "pp_size": self.pp_size,
            "ep_rank": self.ep_rank, "ep_size": self.ep_size,
            "cp_rank": 0, "cp_size": 1,
            "dp_rank": 0, "dp_size": 1,
        }

    @property
    def shard_id(self) -> str:
        t = self.topology
        return ".".join(f"{axis}{t[f'{axis}_rank']}-of-{t[f'{axis}_size']}" for axis in ("tp", "pp", "ep", "cp", "dp"))

    @property
    def numel(self) -> int:
        return sum(prod(shape) for _, shape in self.specs)

    def parameter_specs(self, role: str = "actor") -> tuple[ParameterSpec, ...]:
        return tuple(ParameterSpec(role, name, shape, "float32", prod(shape), self.shard_id)
                     for name, shape in self.specs)


def _divide(value: int, parts: int, what: str) -> int:
    if value % parts:
        raise NemotronHLayoutError(f"{what} ({value}) is not divisible by {parts}")
    return value // parts


def _padded_vocab(vocab: int, divisor: int, tp: int) -> int:
    multiple = divisor * tp
    return -(-vocab // multiple) * multiple


def _layer_specs(g: NemotronHGeometry, symbol: str, layer: int, *, tp: int, ep: int,
                 ep_rank: int) -> list[tuple[str, tuple[int, ...]]]:
    h = g.hidden_size
    base = f"{_PREFIX}decoder.layers.{layer}."
    if symbol == "M":
        inner = _divide(g.mamba_inner, tp, "Mamba inner size")
        heads = _divide(g.mamba_num_heads, tp, "Mamba heads")
        bc = 2 * _divide(g.mamba_groups, tp, "Mamba groups") * g.ssm_state_size
        conv = inner + bc
        return [
            (base + "mixer.in_proj.weight", (2 * inner + bc + heads, h)),
            (base + "mixer.in_proj.layer_norm_weight", (h,)),
            (base + "mixer.conv1d_weight", (conv, 1, g.conv_kernel)),
            (base + "mixer.conv1d_bias", (conv,)),
            (base + "mixer.dt_bias", (heads,)),
            (base + "mixer.A_log", (heads,)),
            (base + "mixer.D", (heads,)),
            (base + "mixer.norm.weight", (inner,)),
            (base + "mixer.out_proj.weight", (h, inner)),
        ]
    if symbol == "*":
        query = g.num_attention_heads * g.head_dim
        kv = g.num_key_value_heads * g.head_dim
        return [
            (base + "self_attention.linear_qkv.weight", (_divide(query + 2 * kv, tp, "QKV width"), h)),
            (base + "self_attention.linear_qkv.layer_norm_weight", (h,)),
            (base + "self_attention.linear_proj.weight", (h, _divide(query, tp, "query width"))),
        ]
    if symbol == "-":
        ffn = _divide(g.ffn_hidden_size, tp, "MLP width")
        return [
            (base + "mlp.linear_fc1.weight", (ffn, h)),
            (base + "mlp.linear_fc1.layer_norm_weight", (h,)),
            (base + "mlp.linear_fc2.weight", (h, ffn)),
        ]
    if symbol == "E":
        if not g.num_experts:
            raise NemotronHLayoutError("MoE layer in the pattern but the config has no experts")
        width = g.moe_latent_size or h
        local = _divide(g.num_experts, ep, "experts")
        specs = [
            (base + "pre_mlp_layernorm.weight", (h,)),
            (base + "mlp.router.weight", (g.num_experts, h)),
        ]
        if g.moe_latent_size:
            specs += [
                (base + "mlp.fc1_latent_proj.weight", (g.moe_latent_size, h)),
                (base + "mlp.fc2_latent_proj.weight", (h, g.moe_latent_size)),
            ]
        for expert in range(ep_rank * local, (ep_rank + 1) * local):
            specs += [
                (base + f"mlp.experts.linear_fc1.weight{expert}", (g.moe_ffn_hidden_size, width)),
                (base + f"mlp.experts.linear_fc2.weight{expert}", (width, g.moe_ffn_hidden_size)),
            ]
        if g.shared_expert_ffn_hidden_size:
            shared = _divide(g.shared_expert_ffn_hidden_size, tp, "shared expert width")
            specs += [
                (base + "mlp.shared_experts.linear_fc1.weight", (shared, h)),
                (base + "mlp.shared_experts.linear_fc2.weight", (h, shared)),
            ]
        return specs
    raise NemotronHLayoutError(f"unknown layer symbol {symbol!r}")


def rank_manifests(geometry: NemotronHGeometry, *, tp: int, ep: int, pp: int = 1,
                   vocab_divisor: int = 128) -> tuple[RankManifest, ...]:
    """Every training rank's specs at DP=1, CP=1 (module scope), in rank order (pp, tp)."""
    if tp < 1 or ep < 1 or pp < 1:
        raise NemotronHLayoutError("parallel sizes must be positive")
    if geometry.num_experts and ep != tp:
        raise NemotronHLayoutError(
            f"EP ({ep}) must equal TP x CP ({tp}) at DP=1: otherwise experts are replicated "
            "on several ranks (refused by yeto.rl.moe_layout) or EP does not fit the ranks")
    if not geometry.num_experts and ep != 1:
        raise NemotronHLayoutError("a dense model has EP 1")
    if geometry.tie_word_embeddings:
        raise NemotronHLayoutError("tied embeddings are not modelled")
    layers = len(geometry.pattern)
    per_stage = _divide(layers, pp, "layers")
    vocab = _divide(_padded_vocab(geometry.vocab_size, vocab_divisor, tp), tp, "padded vocabulary")
    manifests = []
    for pp_rank in range(pp):
        for tp_rank in range(tp):
            ep_rank = tp_rank if geometry.num_experts else 0
            specs: list[tuple[str, tuple[int, ...]]] = []
            if pp_rank == 0:
                specs.append((_PREFIX + "embedding.word_embeddings.weight", (vocab, geometry.hidden_size)))
            for layer in range(pp_rank * per_stage, (pp_rank + 1) * per_stage):
                specs += _layer_specs(geometry, geometry.pattern[layer], layer, tp=tp, ep=ep, ep_rank=ep_rank)
            if pp_rank == pp - 1:
                specs.append((_PREFIX + "decoder.final_norm.weight", (geometry.hidden_size,)))
                specs.append((_PREFIX + "output_layer.weight", (vocab, geometry.hidden_size)))
            manifests.append(RankManifest(tp_rank, tp, pp_rank, pp, ep_rank, ep if geometry.num_experts else 1,
                                          tuple(sorted(specs))))
    return tuple(manifests)


def miles_source_layout_sha256(manifest: RankManifest, role: str = "actor") -> str:
    """The rank's manifest hash as Miles reports it (``source_layout_sha256``;
    mirrors ``full_parameter_state._layout_hash`` schema 1), for comparing a probe."""
    payload = {
        "schema": _MILES_MANIFEST_SCHEMA,
        "topology": manifest.topology,
        "specs": [
            {"role": role, "shard_id": manifest.shard_id, "name": name, "shape": list(shape),
             "dtype": "float32", "numel": prod(shape)}
            for name, shape in manifest.specs
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def offline_layout(manifests: Sequence[RankManifest], *, model_revision: str, config_sha256: str,
                   matrix_merge: str = "avg", max_fragment_bytes: int = 2 << 30,
                   minimum_fragments: int = 1, algorithm: str = "grpo",
                   role: str = "actor") -> ParameterLayout:
    """The island's fragment layout for these manifests (``MilesChunkedFullParameterAdapter.create``
    with the run's ``--rl-sync-matrix-merge`` and fragment byte bound)."""
    from yeto.rl.moe_layout import check_expert_partition

    specs = tuple(spec for manifest in manifests for spec in manifest.parameter_specs(role))
    check_expert_partition(specs)
    return fit_owner_affine_layout(
        algorithm=algorithm,
        components=(ComponentIdentity(role=role, model_revision=model_revision.lower(),
                                      config_hash=config_sha256),),
        specs=specs,
        minimum_fragments=minimum_fragments,
        max_fragment_bytes=max_fragment_bytes,
        stream_role=layout_stream_role(algorithm, role),
        matrix_merge=matrix_merge,
    )
