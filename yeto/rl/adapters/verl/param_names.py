"""verl LoRA tensor names <-> yeto canonical names (rl-verl-backend 1.5, design D2).

yeto canonical LoRA names are the PEFT state-dict names
``base_model.model.<module>.lora_{A,B}.weight`` (``yeto.rl.core``), f32.

* FSDP2 training side (verl ``FSDPEngine`` wraps a HF PEFT model): parameters
  are named ``base_model.model.<module>.lora_{A,B}.<adapter>.weight`` with
  adapter ``default``; dropping the adapter segment gives the canonical name.
  ``peft.get_peft_model_state_dict`` (what verl sends to vLLM) already drops it.
* vLLM inference side: a registered ``LoRAModel`` keys its layers by module
  name without the ``base_model.model.`` prefix, and packs fused modules
  (``qkv_proj`` = q, k, v; ``gate_up_proj`` = gate, up) into one entry whose
  ``lora_a`` / ``lora_b`` are lists in that order.

The map itself is data (``PARAM_MAP``) and is hashed into the backend identity,
so a change here changes the identity.
"""

from __future__ import annotations

import re

CANONICAL_PREFIX = "base_model.model."
ADAPTER_NAME = "default"
VLLM_PACKED_MODULES = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
}
PARAM_MAP = {
    "schema": "yeto-param-map-v1",
    "backend": "verl",
    "train_backend": "fsdp2",
    "kind": "peft-adapter-segment",
    "train_rule": "drop '.<adapter>' before '.weight' in '.lora_{A,B}.<adapter>.weight'",
    "adapter": ADAPTER_NAME,
    "infer_rule": "prefix 'base_model.model.'; unpack vLLM packed modules in listed order",
    "vllm_packed_modules": {k: list(v) for k, v in VLLM_PACKED_MODULES.items()},
}

_CANON = re.compile(r"^base_model\.model\.(?P<module>.+)\.lora_(?P<ab>[AB])\.weight$")
_TRAIN = re.compile(r"^(?P<head>base_model\.model\..+\.lora_[AB])\.(?P<adapter>[^.]+)\.weight$")


class ParamNameError(ValueError):
    """A name does not follow the expected verl / canonical pattern."""


def is_canonical(name: str) -> bool:
    return _CANON.match(name) is not None


def train_to_canonical(name: str) -> str:
    """FSDP2 PEFT parameter name -> canonical name (identity on canonical names)."""
    if is_canonical(name):
        return name
    match = _TRAIN.match(name)
    if match is None:
        raise ParamNameError(f"not a verl FSDP2 LoRA parameter name: {name!r}")
    if match["adapter"] != ADAPTER_NAME:
        raise ParamNameError(f"unexpected PEFT adapter {match['adapter']!r} in {name!r}")
    return f"{match['head']}.weight"


def canonical_to_train(name: str) -> str:
    match = _CANON.match(name)
    if match is None:
        raise ParamNameError(f"not a canonical LoRA name: {name!r}")
    return f"{CANONICAL_PREFIX}{match['module']}.lora_{match['ab']}.{ADAPTER_NAME}.weight"


def vllm_to_canonical(module_name: str, ab: str, packed_index: int | None = None) -> str:
    """vLLM ``LoRAModel`` entry (module, A/B, slot in a packed module) -> canonical name."""
    if ab not in ("A", "B"):
        raise ParamNameError(f"lora side must be A or B, got {ab!r}")
    module_name = module_name.removeprefix(CANONICAL_PREFIX)
    parent, _, leaf = module_name.rpartition(".")
    if leaf in VLLM_PACKED_MODULES:
        if packed_index is None:
            raise ParamNameError(f"packed vLLM module {module_name!r} needs a slot index")
        parts = VLLM_PACKED_MODULES[leaf]
        if not 0 <= packed_index < len(parts):
            raise ParamNameError(f"slot {packed_index} out of range for {module_name!r}")
        leaf = parts[packed_index]
    elif packed_index not in (None, 0):
        raise ParamNameError(f"unpacked vLLM module {module_name!r} has no slot {packed_index}")
    module = f"{parent}.{leaf}" if parent else leaf
    return f"{CANONICAL_PREFIX}{module}.lora_{ab}.weight"


def expected_qwen3_names(num_layers: int, targets=("q_proj", "k_proj", "v_proj", "o_proj",
                                                    "gate_proj", "up_proj", "down_proj")) -> list[str]:
    """Canonical LoRA names of a Qwen3 decoder with ``all-linear`` targets (lm_head excluded)."""
    attn = {"q_proj", "k_proj", "v_proj", "o_proj"}
    names = []
    for layer in range(num_layers):
        for target in targets:
            block = "self_attn" if target in attn else "mlp"
            for ab in ("A", "B"):
                names.append(f"{CANONICAL_PREFIX}model.layers.{layer}.{block}.{target}.lora_{ab}.weight")
    return sorted(names)
