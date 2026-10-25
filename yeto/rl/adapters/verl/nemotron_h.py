"""verl recipe for the NVIDIA Nemotron-H family (role ``nemotron_h_recipe``; S27 G-DC, V10).

On verl the Megatron engine builds the model itself from the Megatron-Bridge
provider of the HF checkpoint (``use_mbridge``), so there is no Megatron argv
to write: :func:`model_flags` is empty. The training knobs that the Miles
recipe passes as flags are Hydra overrides of the Megatron island
(``megatron_engine.build_overrides``: fp32 router, grouped GEMM, routing
statistics fixed with bias update rate 0 and aux loss 0).

:func:`check_provider` refuses what the verl island does not handle:

* a vision-language checkpoint (verl's ``freeze_vision_tower`` is read by no
  code in fork acad9875, so the vision part would train);
* an MTP block in full-parameter mode (no MTP loss in RL; the block would only
  decay and be published);
* a hybrid pattern with MoE layers but no experts.
"""

from __future__ import annotations

from typing import Any

VL_ARCHITECTURES = ("NemotronH_Omni_Reasoning_V3",)


class NemotronHRecipeError(ValueError):
    pass


def _value(provider: Any, name: str, default: Any = None) -> Any:
    value = getattr(provider, name, default)
    return default if value is None else value


def check_provider(provider: Any, *, parameter_mode: str) -> None:
    pattern = str(_value(provider, "hybrid_layer_pattern", ""))
    if not pattern:
        raise NemotronHRecipeError("Nemotron-H provider has no hybrid_layer_pattern")
    if "E" in pattern and _value(provider, "num_moe_experts") is None:
        raise NemotronHRecipeError("hybrid pattern has MoE layers (E) but the provider has no experts")
    architectures = tuple(_value(provider, "architectures", ()) or ())
    if set(architectures) & set(VL_ARCHITECTURES):
        raise NemotronHRecipeError("verl 不冻结视觉部分（freeze_vision_tower 无代码读取）；VL 模型请用 Miles")
    if int(_value(provider, "mtp_num_layers", 0) or 0) and parameter_mode == "full":
        raise NemotronHRecipeError(
            "the Nemotron-H provider has an MTP block: the verl Megatron island would train it "
            "without an MTP loss and publish it")


def model_flags(provider: Any) -> tuple[str, ...]:
    """No Megatron argv on verl (module doc)."""
    return ()
