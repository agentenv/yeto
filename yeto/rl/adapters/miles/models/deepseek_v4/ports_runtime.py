"""DeepSeek V4 runtime patches for the ports Miles layout (P0 1.7b.2/1.7b.2b).

Expert-full (:func:`install`) trains attention LoRA plus a prefix of the E288
clone experts.  On ports, Miles itself streams that hybrid state through the
full-parameter transport (``--full-parameter-lora-hybrid``) and publishes base
weights on every weight update.  Yeto only adds what is specific to this
checkpoint:

* the LoRA transform unfreezes the selected clone experts
  (:func:`~.expert_full.wrap_attention_lora_with_expert_full`);
* the optimizer gives those experts their own learning rate
  (:func:`~.expert_full_runtime.wrap_expert_full_optimizer`);
* a publish exports only the selected clone experts, through the same local
  expert offsets on every EP rank so the Bridge export collectives align.

Clone-only expert LoRA (:func:`install_clone_only`) trains attention LoRA plus
packed routed-expert LoRA whose original-expert slots stay zero
(:func:`~.clone_lora.wrap_clone_only_lora`).  Its LoRA state sync carries the
clone experts only, through the ports state plugin's packed-adapter policy
(:class:`~.clone_lora.ClonePackedAdapterPolicy`).

The legacy Miles layout keeps :mod:`.expert_full_runtime` and
:func:`.clone_lora.install` until P0 1.7b.4 removes them; this module touches
only ports module names.
"""

from __future__ import annotations

import itertools
from types import ModuleType

from yeto.rl.adapters.miles.models.deepseek_v4.expert_full_runtime import (
    configured_expert_count,
    export_selected_expert_weights,
    install_or_defer,
    weight_iterator_task_bridge,
    wrap_expert_full_optimizer,
)

ENV_FLAG = "YETO_DSV4_EXPERT_FULL_PORTS"
CLONE_ONLY_ENV_FLAG = "YETO_DSV4_CLONE_ONLY_LORA_PORTS"

_LORA_BRIDGE = "miles.backends.megatron_utils.lora.bridge"
_MODEL = "miles.backends.megatron_utils.model"
_WEIGHT_ITERATOR = "miles.backends.megatron_utils.update_weight.hf_weight_iterator_bridge"


def _install_once(module: ModuleType, patch) -> None:
    marker = f"_yeto_dsv4_ports__{patch.__name__.lstrip('_')}__installed"
    if getattr(module, marker, False):
        return
    patch(module)
    setattr(module, marker, True)


def _wrap_lora_factory(module: ModuleType, wrap) -> None:
    original = module.create_lora_instance

    def create_lora_instance(args, *, target_modules):
        return wrap(original(args, target_modules=target_modules))

    module.create_lora_instance = create_lora_instance


def _patch_lora_bridge(module: ModuleType) -> None:
    from yeto.rl.adapters.miles.models.deepseek_v4.expert_full import (
        wrap_attention_lora_with_expert_full,
    )

    _wrap_lora_factory(module, lambda lora: wrap_attention_lora_with_expert_full(
        lora, configure_kwargs={"expert_count": configured_expert_count()}
    ))


def _patch_lora_bridge_clone_only(module: ModuleType) -> None:
    from yeto.rl.adapters.miles.models.deepseek_v4.clone_lora import wrap_clone_only_lora

    _wrap_lora_factory(module, wrap_clone_only_lora)


def _patch_model(module: ModuleType) -> None:
    module.get_megatron_optimizer = wrap_expert_full_optimizer(module.get_megatron_optimizer)


def _patch_weight_iterator(module: ModuleType) -> None:
    """Replace the base-weight units with the selected clone experts only.

    DeepSeek V4 has no multimodal tower, so the stock trailing tower units
    are not needed.
    """

    def iter_selected_expert_units(self, weights, *, materialize):
        named_weights = export_selected_expert_weights(
            module,
            self,
            weights,
            export_kwargs=self._source_name_kwargs(weight_iterator_task_bridge(self).export_hf_weights),
            materialize=materialize,
        )
        for _sources, group in itertools.groupby(named_weights, key=lambda item: item[2]):
            yield [(hf_name, weight) for hf_name, weight, _megatron_names in group]

    module.HfWeightIteratorBridge._iter_hf_param_units = iter_selected_expert_units


def install_on_lora_bridge(module: ModuleType) -> None:
    _install_once(module, _patch_lora_bridge)


def install_clone_only_on_lora_bridge(module: ModuleType) -> None:
    _install_once(module, _patch_lora_bridge_clone_only)


def install_on_model(module: ModuleType) -> None:
    _install_once(module, _patch_model)


def install_on_weight_iterator(module: ModuleType) -> None:
    _install_once(module, _patch_weight_iterator)


def install() -> None:
    """Patch the imported ports Miles modules or arm lazy import hooks."""

    for fullname, installer in (
        (_LORA_BRIDGE, install_on_lora_bridge),
        (_MODEL, install_on_model),
        (_WEIGHT_ITERATOR, install_on_weight_iterator),
    ):
        install_or_defer(fullname, installer)


def install_clone_only() -> None:
    """Register the clone-only state policy and patch (or arm) the LoRA factory."""

    from yeto.rl.adapters.miles.models.deepseek_v4.clone_lora import ClonePackedAdapterPolicy
    from yeto.rl.adapters.miles.state_plugin import register_packed_adapter_policy

    register_packed_adapter_policy(ClonePackedAdapterPolicy())
    install_or_defer(_LORA_BRIDGE, install_clone_only_on_lora_bridge)
