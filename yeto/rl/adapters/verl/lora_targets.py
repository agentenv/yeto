"""``--lora-targets`` -> verl's ``actor_rollout_ref.model.target_modules``.

The island derives its canonical LoRA specs from ``--lora-targets``
(``yeto.rl.export.derive_peft_lora_specs``, ``trainer.run_island``), and verl builds its
PEFT model from ``target_modules`` (fork ``FSDPEngine._build_lora_module``,
``peft.LoraConfig``). Both must name the same modules, or the first export fails the
canonical spec check. yeto resolves a choice to ``all-linear`` or to a module-name
regex ``.*\\.(a|b|...)$`` (``yeto.rl.export.target_modules``); the regex is handed to
verl as the equivalent name list ``[a,b,...]`` (PEFT matches a list entry as the last
module-name component, the regex's own rule), which a Hydra override can carry.

``all-linear`` on a mixture-of-experts model is refused (:func:`moe_all_linear_problem`):
the island's PEFT 0.19.1 cannot build it (Transformers 5's MoE target conversion turns
the string into a set of its characters, so ``get_peft_model`` raises on Qwen3-MoE,
OLMoE, Mixtral, DeepSeek-V3, GLM-4.5 ...); a PEFT without that bug adapts the router
``gate`` and wraps each fused ``experts`` tensor as one stacked parameter adapter, which
is neither the per-expert layout ``yeto model check`` describes nor one vLLM 0.29 loads
into a 2D fused-MoE layer (it expects ``experts.<e>.{gate,up,down}_proj`` unless the
engine starts with ``enable_mixed_moe_lora_format``). Routed-expert LoRA runs on Miles.
"""

from __future__ import annotations

import re
from typing import Any

from .config import VerlConfigError

CHOICES = ("auto", "attention", "all-linear")
_NAME_ALTERNATION = re.compile(r"\.\*\\\.\((?P<names>[A-Za-z0-9_]+(?:\|[A-Za-z0-9_]+)*)\)\$")


def moe_all_linear_problem(choice: str, config: Any) -> str | None:
    """Why ``choice`` cannot train on the verl backend for this model, or ``None``:
    ``all-linear`` on a mixture-of-experts model (routed-expert LoRA, see the module doc)."""
    from yeto.rl.lora_layout import adapts_routed_experts

    if not adapts_routed_experts(choice, config):
        return None
    return ("--lora-targets all-linear on a mixture-of-experts model is refused on the verl backend: its "
            "PEFT 0.19.1 cannot build routed-expert LoRA on Transformers 5 fused experts, and the "
            "per-expert adapters yeto syncs are not what verl trains or vLLM loads "
            "(yeto/rl/adapters/verl/lora_targets.py); use --lora-targets attention, or --rl-backend miles "
            "for routed-expert LoRA")


def peft_targets(choice: str, config: Any) -> str:
    """The PEFT ``target_modules`` yeto's spec derivation uses for ``choice``."""
    if choice not in CHOICES:
        raise VerlConfigError(f"--lora-targets {choice} is not mapped on the verl backend (it maps "
                              + ", ".join(CHOICES) + "; attention-routed-experts is the Miles "
                              "DeepSeek V4 recipe)")
    if (problem := moe_all_linear_problem(choice, config)) is not None:
        raise VerlConfigError(problem)
    from yeto.rl.export import target_modules

    return target_modules(choice, config)


def verl_target_modules(choice: str, config: Any) -> str:
    """The ``target_modules`` override value for ``choice``: ``all-linear`` or ``[a,b,...]``."""
    targets = peft_targets(choice, config)
    if targets == "all-linear":
        return targets
    match = _NAME_ALTERNATION.fullmatch(targets)
    if match is None:
        raise VerlConfigError(f"--lora-targets {choice} resolves to {targets!r}, which has no verl "
                              "target_modules list form")
    return "[" + ",".join(match["names"].split("|")) + "]"


def target_list(value: str) -> str | list[str]:
    """Inverse of the override form: ``[a,b]`` -> ``["a", "b"]``; ``all-linear`` unchanged."""
    if value.startswith("[") and value.endswith("]"):
        return [name for name in value[1:-1].split(",") if name]
    return value


def island_target_modules(choice: str, model_path: str, *, trust_remote_code: bool = False) -> str:
    """:func:`verl_target_modules` for the island's model (its ``config.json`` only)."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=trust_remote_code)
    return verl_target_modules(choice, config)
