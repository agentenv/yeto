"""Minimal test copy of Miles c35702e miles/utils/hf_utils/config.py.

Verbatim except: only the two qwen4_exp aliases are kept (the deepseek/glm
entries need torch AutoModel mappings) and load_hf_config/helpers are dropped.
tests/test_rl_fn_provider_view.py checks these entries against a real c35702e
checkout when one is available."""

import importlib
from dataclasses import dataclass
from pathlib import Path

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES


@dataclass(frozen=True)
class _HFConfigAlias:
    model_type: str
    base_module: str
    base_class: str
    compat_class_name: str
    auto_model_classes: tuple = (AutoModelForCausalLM,)
    # Set True to override transformers' native config.
    override_hf_native: bool = False


_CONFIG_ALIASES: tuple[_HFConfigAlias, ...] = (
    # Qwen3.8-Flash-Next: the composite config resolves text_config by its nested
    # model_type, so both levels need an alias; extra fields survive as attributes
    _HFConfigAlias(
        model_type="qwen4_exp_text",
        base_module="transformers.models.qwen3_5_moe.configuration_qwen3_5_moe",
        base_class="Qwen3_5MoeTextConfig",
        compat_class_name="Qwen4ExpTextConfig",
        auto_model_classes=(),
    ),
    _HFConfigAlias(
        model_type="qwen4_exp",
        base_module="transformers.models.qwen3_5_moe.configuration_qwen3_5_moe",
        base_class="Qwen3_5MoeConfig",
        compat_class_name="Qwen4ExpConfig",
        auto_model_classes=(),
    ),
)

_REGISTERED_ALIASES: set[str] = set()


def register_hf_config_aliases() -> None:
    """Register miles model_type aliases with transformers. Idempotent.

    Already called inside `load_hf_config` and `load_tokenizer`. Only call
    directly before a third-party entry point that won't go through either
    (e.g. megatron's `_build_tokenizer`).
    """
    for alias in _CONFIG_ALIASES:
        if alias.model_type in _REGISTERED_ALIASES:
            continue
        if alias.model_type in CONFIG_MAPPING_NAMES and not alias.override_hf_native:
            raise RuntimeError(
                f"transformers now natively supports model_type={alias.model_type!r}; "
                f"set override_hf_native=True to override."
            )
        module = importlib.import_module(alias.base_module)
        base_config = getattr(module, alias.base_class)
        compat_config = type(
            alias.compat_class_name,
            (base_config,),
            {"model_type": alias.model_type, "__module__": __name__},
        )
        AutoConfig.register(alias.model_type, compat_config, exist_ok=alias.override_hf_native)
        for auto_cls in alias.auto_model_classes:
            base_model_cls = auto_cls._model_mapping[base_config]
            compat_model_cls = type(
                base_model_cls.__name__, (base_model_cls,), {"config_class": compat_config, "__module__": __name__}
            )
            auto_cls.register(compat_config, compat_model_cls, exist_ok=alias.override_hf_native)
        _REGISTERED_ALIASES.add(alias.model_type)

    try:
        import sglang.srt.configs.inkling  # noqa: F401
    except ImportError:
        pass
