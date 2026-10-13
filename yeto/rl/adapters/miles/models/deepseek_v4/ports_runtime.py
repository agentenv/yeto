"""DeepSeek V4 expert-full patches for the ports Miles layout (P0 1.7b.2).

Expert-full trains attention LoRA plus a prefix of the E288 clone experts.
On ports, Miles itself streams that hybrid state through the full-parameter
transport (``--full-parameter-lora-hybrid``) and publishes base weights on
every weight update.  Yeto only adds what is specific to this checkpoint:

* the LoRA transform unfreezes the selected clone experts
  (:func:`~.expert_full.wrap_attention_lora_with_expert_full`);
* the optimizer gives those experts their own learning rate
  (:func:`~.expert_full_runtime.wrap_expert_full_optimizer`);
* a publish exports only the selected clone experts, through the same local
  expert offsets on every EP rank so the Bridge export collectives align.

The legacy Miles layout keeps :mod:`.expert_full_runtime` until P0 1.7b.4
removes it; :func:`install` touches only ports module names.
"""

from __future__ import annotations

import itertools
from types import ModuleType

from yeto.rl.adapters.miles.models.deepseek_v4.expert_clone import (
    CLONES_PER_EXPERT_RANK,
    NUM_LAYERS,
)
from yeto.rl.adapters.miles.models.deepseek_v4.expert_full_runtime import (
    configured_expert_count,
    filter_collective_expert_tasks,
    install_or_defer,
    selected_expert_hf_name,
    weight_iterator_task_bridge,
    wrap_expert_full_optimizer,
)

ENV_FLAG = "YETO_DSV4_EXPERT_FULL_PORTS"

_LORA_BRIDGE = "miles.backends.megatron_utils.lora.bridge"
_MODEL = "miles.backends.megatron_utils.model"
_WEIGHT_ITERATOR = "miles.backends.megatron_utils.update_weight.hf_weight_iterator_bridge"
_INSTALLED = "_yeto_dsv4_expert_full_ports_installed"


def _install_once(module: ModuleType, patch) -> None:
    if getattr(module, _INSTALLED, False):
        return
    patch(module)
    setattr(module, _INSTALLED, True)


def _patch_lora_bridge(module: ModuleType) -> None:
    from yeto.rl.adapters.miles.models.deepseek_v4.expert_full import (
        wrap_attention_lora_with_expert_full,
    )

    original = module.create_lora_instance

    def create_lora_instance(args, *, target_modules):
        return wrap_attention_lora_with_expert_full(
            original(args, target_modules=target_modules),
            configure_kwargs={"expert_count": configured_expert_count()},
        )

    module.create_lora_instance = create_lora_instance


def _patch_model(module: ModuleType) -> None:
    module.get_megatron_optimizer = wrap_expert_full_optimizer(module.get_megatron_optimizer)


def expected_expert_task_count(expert_count: int) -> int:
    """Selected clone-expert conversion tasks: per layer, fc1 and fc2 per local offset."""

    return NUM_LAYERS * min(expert_count, CLONES_PER_EXPERT_RANK) * 2


def _patch_weight_iterator(module: ModuleType) -> None:
    """Replace the base-weight units with the selected clone experts only.

    The task Bridge reads the trainer checkpoint (``ref_load``), not a possibly
    quantized rollout checkpoint.  DeepSeek V4 has no multimodal tower, so the
    stock trailing tower units are not needed.
    """

    def iter_selected_expert_units(self, weights, *, materialize):
        expert_count = configured_expert_count()
        task_bridge = weight_iterator_task_bridge(self)
        renamed = {module.strip_param_name_prefix(name): value for name, value in weights.items()}
        with module.megatron_bridge_utils.patch_megatron_model(self.model):
            tasks = filter_collective_expert_tasks(
                task_bridge.get_conversion_tasks(self.model), expert_count=expert_count
            )
            expected = expected_expert_task_count(expert_count)
            if len(tasks) != expected:
                raise RuntimeError(
                    f"expert-full publish found {len(tasks)} conversion tasks, expected {expected}"
                )
            named_weights = task_bridge.export_hf_weights(
                self.model,
                cpu=False,
                conversion_tasks=module._process_conversion_tasks(tasks, renamed),
                merge_adapter_weights=False,
                **self._source_name_kwargs(task_bridge.export_hf_weights),
            )
            if not materialize:
                for _ in named_weights:  # every rank still runs the export collectives
                    pass
                return
            selected = (
                item for item in named_weights
                if selected_expert_hf_name(item[0], expert_count=expert_count)
            )
            processed = self._postprocess_and_quantize(selected, "base")
            for _sources, group in itertools.groupby(processed, key=lambda item: item[2]):
                unit = [
                    (hf_name, weight)
                    for hf_name, weight, _megatron_names in group
                    if not module.is_lora_weight_name(hf_name)
                ]
                if unit:
                    yield unit

    module.HfWeightIteratorBridge._iter_hf_param_units = iter_selected_expert_units


def install_on_lora_bridge(module: ModuleType) -> None:
    _install_once(module, _patch_lora_bridge)


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
