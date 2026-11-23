"""What the pinned upstream trainer stack can load and publish, per HF architecture.

Read once from the pinned sources (no import of them here; stdlib only):

* Megatron-Bridge ``a0c6d6e09b0bd4d208c2d936608319d5ab456d87`` (the commit the
  pinned Miles image installs, ``docker/Dockerfile`` of Miles
  ``9fa600b9`` and its base ``778794a807``): every ``@MegatronModelBridge.register_bridge(source=...)``
  under ``src/megatron/bridge/models``. Miles builds its Megatron model and
  maps LoRA adapters through ``AutoBridge``, so an architecture without a
  bridge does not load on the Miles trainer.
* the pinned Miles fork (``MILES_NEXT_COMMIT``, ``MILES_COMMIT`` below)
  ``miles/backends/megatron_utils/megatron_to_hf/__init__.py``
  ``_convert_to_hf_core``: the Megatron -> HF converters ``update_weights`` uses
  to publish full-parameter weights to the rollout engine. It dispatches on
  ``type(hf_config).__name__.lower()`` (``converter_model_name``) unless the
  recipe sets ``--model-name``, by substring in the order below, and raises
  ``Unsupported model`` otherwise. LoRA publication goes through the bridge
  instead and does not need a converter.

The fork's dispatch is the upstream base's (``778794a807``): between the two,
the fork commits that touch ``megatron_to_hf`` change converters, not the
dispatch (``02b43a22`` kimi_k3 slot permutation, ``88358095`` MXFP4 packed rows,
``329565db`` the Kimi-K3.1 DSA indexer). Re-read both tables when
``MILES_NEXT_COMMIT`` moves; ``tests/test_model_sync_profile.py`` fails until
``MILES_COMMIT`` names the new pin.
"""

from __future__ import annotations

MEGATRON_BRIDGE_COMMIT = "a0c6d6e09b0bd4d208c2d936608319d5ab456d87"
MILES_COMMIT = "9fa600b9ae9d8c4024f3c2e4cd23a8ed433b2b54"  # the fork pin the tables were read at
MILES_UPSTREAM_COMMIT = "778794a807866a4f51f34d5e87dda3e55d9e85ad"  # its upstream base

MEGATRON_BRIDGE_ARCHITECTURES = frozenset({
    "BailingMoeV2ForCausalLM", "DeciLMForCausalLM", "DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM",
    "DeepseekV4ForCausalLM", "Exaone4ForCausalLM", "ExaoneMoeForCausalLM", "FalconH1ForCausalLM",
    "Gemma2ForCausalLM", "Gemma3ForCausalLM", "Gemma3ForConditionalGeneration", "Gemma4ForCausalLM",
    "Gemma4ForConditionalGeneration", "GemmaForCausalLM", "Glm4MoeForCausalLM", "Glm4MoeLiteForCausalLM",
    "Glm4vMoeForConditionalGeneration", "GlmMoeDsaForCausalLM", "GptOssForCausalLM", "HYV3ForCausalLM",
    "KimiK25ForConditionalGeneration", "KimiK2ForCausalLM", "KimiK3ForConditionalGeneration",
    "LlamaForCausalLM", "MiMoForCausalLM", "MiMoV2FlashForCausalLM", "MiniMaxM2ForCausalLM",
    "MiniMaxM3SparseForConditionalGeneration", "Mistral3ForConditionalGeneration", "MistralForCausalLM",
    "NemotronForCausalLM", "NemotronHForCausalLM", "OlmoeForCausalLM", "Qwen2ForCausalLM",
    "Qwen2_5_VLForConditionalGeneration", "Qwen3ForCausalLM", "Qwen3MoeForCausalLM",
    "Qwen3NextForCausalLM", "Qwen3VLForConditionalGeneration", "Qwen3VLMoeForConditionalGeneration",
    "Qwen3_5ForCausalLM", "Qwen3_5MoeForCausalLM", "SarvamMLAForCausalLM", "SarvamMoEForCausalLM",
    "Step3p5ForCausalLM",
})

# _convert_to_hf_core, in dispatch order: (substrings of the lowered config class name, converter).
_MILES_CONVERTERS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("glm5_next", "glm5next"), "glm5_next"),
    (("glm4moelite", "deepseekv3", "glmmoedsa", "glm_moe_dsa"), "deepseekv3"),
    (("glm4moe",), "glm4moe"),
    (("glm4",), "glm4"),
    (("qwen3moe",), "qwen3moe"),
    (("qwen3next",), "qwen3_next"),
    (("qwen3_8_next", "qwen4_exp", "qwen3.8_next"), "qwen3_8_next"),
    (("qwen3_5", "qwen3_6"), "qwen3_5"),
    (("qwen2", "qwen3"), "qwen2"),
    (("deepseekv41",), "deepseekv4_1"),
    (("deepseekv4",), "deepseekv4"),
    (("inkling",), "inkling"),
    (("llama",), "llama"),
    (("minimax_m3",), "minimax_m3"),
    (("mimo",), "mimo"),
    (("kimi_k3",), "kimi_k3"),
    (("kimivl",), "kimivl"),
    (("kimi_k25", "kimik25"), "kimi_k25"),
)


def config_class_name(architecture: str) -> str:
    """The transformers config class an architecture name pairs with
    (``Qwen3MoeForCausalLM`` -> ``Qwen3MoeConfig``)."""

    for suffix in ("ForCausalLM", "ForConditionalGeneration"):
        if architecture.endswith(suffix):
            return architecture[: -len(suffix)] + "Config"
    return architecture + "Config"


def miles_full_parameter_converter(config_class: str) -> str | None:
    """The Miles Megatron -> HF converter ``update_weights`` picks for a config
    class (generic recipe, no ``--model-name``); None: ``Unsupported model``."""

    name = config_class.lower()
    for needles, converter in _MILES_CONVERTERS:
        if any(needle in name for needle in needles):
            return converter
    return None


def has_megatron_bridge(architectures: tuple[str, ...]) -> bool:
    return any(a in MEGATRON_BRIDGE_ARCHITECTURES for a in architectures)
