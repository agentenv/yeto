"""Which linear modules a quantized rollout may quantize (S26 E6 scope rule).

The rule is fixed (GOAL §5 E6):

* a dense model quantizes only its FFN (MLP) linears:
  ``mlp.gate_proj``, ``mlp.up_proj``, ``mlp.down_proj`` (and the fused
  ``mlp.gate_up_proj``);
* a MoE model quantizes only its routed experts:
  ``mlp.experts.<e>.{gate_proj,up_proj,down_proj,w1,w2,w3}`` (and the stacked
  ``mlp.experts.gate_up_proj`` / ``mlp.experts.down_proj``); its router, its
  shared experts and any dense FFN layer stay BF16;
* the router, ``lm_head``, the embeddings, every norm and every attention
  module always stay BF16.

Names are module names as an HF checkpoint stores them (``<module>.weight``).
Anything the rule does not name as in scope is out of scope, so a new module
kind (a vision tower, an MTP head) stays BF16 until the rule names it.

This module does not import torch: the launcher, the island entries and the
sampler's post-load check use it.

* :func:`model_kind`: ``dense`` or ``moe`` from an HF ``config.json``;
* :func:`in_scope` / :func:`check_quantized`: the rule and its refusal
  (:class:`QuantScopeError`, raised before any engine starts);
* :func:`linear_modules` / :func:`ignored_modules`: the out-of-scope linear
  modules of a checkpoint (the engines' ignore lists);
* :func:`engine_ignore_list` / :func:`online_quantized`: the SGLang online FP8
  ignore list and what SGLang would quantize with it (simulated, see
  :func:`path_match`).
"""

from __future__ import annotations

import json
import re
import struct
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

DENSE = "dense"
MOE = "moe"
KINDS = (DENSE, MOE)

_FFN_PROJ = r"(?:gate_proj|up_proj|down_proj|gate_up_proj)"
_DENSE_FFN = re.compile(rf"(?:^|\.)mlp\.{_FFN_PROJ}$")
_ROUTED_EXPERT = re.compile(
    r"(?:^|\.)mlp\.experts\.(?:\d+\.(?:gate_proj|up_proj|down_proj|w1|w2|w3)|gate_up_proj|down_proj)$")

# Out-of-scope reasons, checked in this order (the first match names the reason).
_KEEP_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("lm_head", re.compile(r"(?:^|\.)lm_head$")),
    ("embedding", re.compile(r"(?:^|\.)(?:embed_tokens|embeddings?|wte|word_embeddings)(?:$|\.)")),
    ("norm", re.compile(r"(?:norm|layernorm|ln_\w*)$", re.IGNORECASE)),
    ("attention", re.compile(r"(?:^|\.)(?:self_attn|linear_attn|attention|attn|self_attention)(?:$|\.)")),
    ("shared_expert", re.compile(r"(?:^|\.)mlp\.shared_experts?(?:_gate)?(?:$|\.)")),
    ("router", re.compile(r"(?:^|\.)mlp\.(?:gate|router)(?:$|\.)")),
)

# Config keys that give the routed-expert count of a MoE model.
_EXPERT_KEYS = ("num_experts", "n_routed_experts", "num_local_experts", "moe_num_experts")

# SGLang online FP8 (``--quantization fp8``) reads its ignore list from
# SGLANG_FP8_IGNORED_LAYERS (sglang 2fa88018 ``layers/quantization/fp8.py``
# Fp8Config.__init__). These path segments are added to the explicit names, so
# a module whose SGLang prefix differs from its checkpoint name still matches.
_ENGINE_SEGMENTS = {
    DENSE: ("self_attn", "linear_attn", "lm_head", "embed_tokens", "mlp.shared_expert",
            "mlp.shared_experts", "mlp.shared_expert_gate"),
    MOE: ("self_attn", "linear_attn", "lm_head", "embed_tokens", "mlp.shared_expert",
          "mlp.shared_experts", "mlp.shared_expert_gate", "mlp.gate", "mlp.gate_proj",
          "mlp.up_proj", "mlp.down_proj"),
}


class QuantScopeError(ValueError):
    """A quantization plan or checkpoint quantizes a module outside the scope rule."""


def model_kind(config: Mapping[str, Any]) -> str:
    """``moe`` when the HF config declares routed experts (also in ``text_config``), else ``dense``."""
    for section in (config, config.get("text_config") or {}):
        for key in _EXPERT_KEYS:
            value = section.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return MOE
    return DENSE


def model_kind_of(checkpoint: Path) -> str:
    return model_kind(json.loads((Path(checkpoint) / "config.json").read_text()))


def keep_reason(module: str) -> str | None:
    """Why ``module`` must stay BF16 by name (``None``: the name gives no reason)."""
    for reason, pattern in _KEEP_RULES:
        if pattern.search(module):
            return reason
    return None


def in_scope(module: str, kind: str) -> bool:
    """True when the scope rule lets ``kind`` models quantize ``module``."""
    if kind not in KINDS:
        raise ValueError(f"unknown model kind {kind!r}; known: {', '.join(KINDS)}")
    if kind == MOE:
        return bool(_ROUTED_EXPERT.search(module))
    return bool(_DENSE_FFN.search(module)) and keep_reason(module) is None


def violations(modules: Iterable[str], kind: str) -> dict[str, str]:
    """Out-of-scope modules of ``modules`` and why (``not_ffn`` / ``not_routed_expert`` when no name rule fits)."""
    other = "not_routed_expert" if kind == MOE else "not_ffn"
    return {m: keep_reason(m) or other for m in sorted(set(modules)) if not in_scope(m, kind)}


def check_quantized(modules: Iterable[str], kind: str, *, what: str) -> None:
    """Raise :class:`QuantScopeError` when ``modules`` holds an out-of-scope module."""
    bad = violations(modules, kind)
    if bad:
        sample = ", ".join(f"{m} ({r})" for m, r in list(bad.items())[:5])
        raise QuantScopeError(
            f"{what} quantizes {len(bad)} module(s) outside the {kind} scope (only "
            f"{'routed experts' if kind == MOE else 'FFN linears'} may be quantized; router, "
            f"lm_head, embeddings, norms and attention stay BF16), e.g. {sample}")


def linear_modules(tensors: Mapping[str, Sequence[int]]) -> list[str]:
    """Modules with a 2-D ``<module>.weight`` (``tensors``: name -> shape)."""
    return sorted(n[: -len(".weight")] for n, shape in tensors.items()
                  if n.endswith(".weight") and len(shape) == 2)


def ignored_modules(tensors: Mapping[str, Sequence[int]], kind: str) -> list[str]:
    """The out-of-scope modules of ``linear_modules(tensors)``."""
    return [m for m in linear_modules(tensors) if not in_scope(m, kind)]


# Fused engine modules and their checkpoint shards (SGLang / vLLM ``packed_modules_mapping``).
FUSED_SHARDS: dict[str, tuple[str, ...]] = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
}


def with_fused_aliases(modules: Iterable[str]) -> list[str]:
    """``modules`` plus the fused engine name of every group whose shards are all listed
    (``...self_attn.q/k/v_proj`` -> ``...self_attn.qkv_proj``).

    SGLang 2fa88018 compressed-tensors matches its ``ignore`` list against the fused
    module name and falls back to the shards only through the model class's
    ``packed_modules_mapping``, which Qwen3ForCausalLM does not set: without the
    alias ``qkv_proj`` was given the FP8 scheme on the card (e6-gate-t2, 10-21)."""
    listed = set(modules)
    aliases = set()
    for module in listed:
        parent, _, leaf = module.rpartition(".")
        for fused, shards in FUSED_SHARDS.items():
            if leaf in shards and all(f"{parent}.{shard}" in listed for shard in shards):
                aliases.add(f"{parent}.{fused}" if parent else fused)
    return sorted(listed | aliases)


def checkpoint_shapes(checkpoint: Path) -> dict[str, list[int]]:
    """Tensor name -> shape of a safetensors checkpoint, from the shard headers only."""
    checkpoint = Path(checkpoint)
    index = checkpoint / "model.safetensors.index.json"
    shards = (sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.is_file()
              else sorted(p.name for p in checkpoint.glob("*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"{checkpoint}: no safetensors shards")
    shapes: dict[str, list[int]] = {}
    for shard in shards:
        with open(checkpoint / shard, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(n))
        header.pop("__metadata__", None)
        shapes.update({name: list(info["shape"]) for name, info in header.items()})
    return shapes


# ---------------------------------------------------------------- engine ignore lists
def engine_ignore_list(tensors: Mapping[str, Sequence[int]], kind: str) -> list[str]:
    """SGLang online FP8 ignore list: the explicit out-of-scope names plus the path segments."""
    return sorted(set(ignored_modules(tensors, kind)) | set(_ENGINE_SEGMENTS[kind]))


def path_match(ignored: str, prefix: str) -> bool:
    """SGLang's ``_module_path_match`` (sglang 2fa88018 ``layers/quantization/utils.py``):
    equal, a dotted prefix, or a dotted segment run of ``prefix``."""
    ignored, prefix = ignored.rstrip("."), prefix.rstrip(".")
    return ignored == prefix or prefix.startswith(ignored + ".") or f".{ignored}." in f".{prefix}."


def online_quantized(modules: Iterable[str], ignore: Sequence[str]) -> list[str]:
    """The modules an online FP8 engine quantizes with ``ignore`` (every linear not matched).
    Embeddings and ``lm_head`` are not linear layers in SGLang or vLLM, so they are never
    quantized there; they are still listed in the ignore lists."""
    return sorted(m for m in modules
                  if keep_reason(m) not in ("embedding", "lm_head")
                  and not any(path_match(i, m) for i in ignore))


def check_online_plan(tensors: Mapping[str, Sequence[int]], kind: str, ignore: Sequence[str]) -> list[str]:
    """Refuse an ignore list that lets the engine quantize an out-of-scope module, or that
    keeps every in-scope module BF16; returns the modules the engine would quantize."""
    quantized = online_quantized(linear_modules(tensors), ignore)
    check_quantized(quantized, kind, what="the online FP8 plan")
    if not quantized:
        raise QuantScopeError(f"the online FP8 plan quantizes nothing ({kind} model; no in-scope linear)")
    return quantized
