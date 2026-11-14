"""``config.json`` -> :class:`ModelFacts`, and the family entry that matches it."""

from __future__ import annotations

import json
import os
from pathlib import Path

from yeto.memory_estimate import param_breakdown

from .schema import FamilyEntry, ModelFacts


def load_config(model: str) -> tuple[dict, str]:
    """(config dict, source) of a local dir / config.json file / HF id in the local HF
    cache. Never downloads: the dry run reads what is on disk."""
    path = Path(os.path.expanduser(model))
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8")), str(path)
    if (path / "config.json").is_file():
        return json.loads((path / "config.json").read_text(encoding="utf-8")), str(path / "config.json")
    from yeto.memory_estimate import _hf_cache_configs
    found = _hf_cache_configs(model)
    if found:
        return json.loads(Path(found[-1]).read_text(encoding="utf-8")), found[-1]
    raise FileNotFoundError(
        f"no config.json for {model!r}: pass a local model dir or config.json "
        "(the dry run does not download; e.g. `huggingface-cli download <id> config.json`)")


def _attention(t: dict) -> str:
    if t.get("index_topk") or t.get("index_n_heads"):
        return "dsa"
    if t.get("kv_lora_rank"):
        return "mla"
    if t.get("hybrid_override_pattern") or t.get("mamba_num_heads") or t.get("hybrid_layer_pattern"):
        return "mamba-hybrid"
    layer_types = t.get("layer_types") or ()
    if "linear_attention" in layer_types or t.get("linear_conv_kernel_dim"):
        return "gdn-hybrid"
    return "gqa"


def _quant(cfg: dict, t: dict) -> str:
    q = cfg.get("quantization_config") or t.get("quantization_config") or {}
    if q:
        fmt = str(q.get("quant_method") or q.get("format") or "quantized").lower()
        if "mxfp4" in json.dumps(q).lower():
            return "mxfp4"
        return fmt
    return str(cfg.get("torch_dtype") or t.get("torch_dtype") or t.get("dtype") or "bfloat16").replace(
        "bfloat16", "bf16")


def facts_from_config(cfg: dict) -> ModelFacts:
    t = cfg.get("text_config") or cfg
    b = param_breakdown(cfg)
    experts = int(t.get("num_experts") or t.get("n_routed_experts") or 0)
    nh = int(t.get("num_attention_heads") or 1)
    return ModelFacts(
        architectures=tuple(cfg.get("architectures") or t.get("architectures") or ()),
        model_type=str(cfg.get("model_type") or t.get("model_type") or ""),
        text_model_type=(str(t["model_type"]) if t is not cfg and t.get("model_type") else None),
        moe=experts > 0,
        num_experts=experts,
        topk=int(t.get("num_experts_per_tok") or t.get("num_experts_per_token") or 0),
        first_k_dense=int(b["dense_layers"]) if experts else 0,
        total_params_b=b["total"] / 1e9,
        active_params_b=b["active"] / 1e9,
        hidden=int(t["hidden_size"]),
        layers=int(t["num_hidden_layers"]),
        heads=nh,
        kv_heads=int(t.get("num_key_value_heads") or nh),
        vocab=int(t["vocab_size"]),
        max_positions=int(t.get("max_position_embeddings") or 0),
        attention=_attention(t),
        checkpoint_quant=_quant(cfg, t),
        vl="vision_config" in cfg,
        breakdown=b,
    )


def match_family(facts: ModelFacts, entries: tuple[FamilyEntry, ...] | None = None) -> tuple[FamilyEntry, str]:
    """(entry, how): architectures, then model_type / text_config.model_type, then
    shape features, then the generic-dense / generic-moe fallback."""
    from .families import FAMILIES, GENERIC_DENSE, GENERIC_MOE
    entries = FAMILIES if entries is None else entries
    for e in entries:
        if set(facts.architectures) & set(e.match_architectures):
            return e, "architectures"
    types = {facts.model_type, facts.text_model_type} - {None, ""}
    for e in entries:
        if types & set(e.match_model_types):
            return e, "model_type"
    if facts.attention == "mamba-hybrid":
        for e in entries:
            if e.family == "nemotron-h":
                return e, "shape:mamba-hybrid"
    if facts.attention == "mla" and facts.moe:
        for e in entries:
            if e.family == "deepseek-v3":
                return e, "shape:mla-moe"
    return (GENERIC_MOE if facts.moe else GENERIC_DENSE), "fallback"
