"""``--lora-targets`` -> verl's ``actor_rollout_ref.model.target_modules``.

The island derives its canonical LoRA specs from ``--lora-targets``
(``yeto.rl.export.derive_peft_lora_specs``, ``trainer.run_island``), and verl builds its
PEFT model from ``target_modules`` (fork ``FSDPEngine._build_lora_module``,
``peft.LoraConfig``). Both must name the same modules, or the first export fails the
canonical spec check. yeto resolves a choice to ``all-linear`` or to a module-name
regex ``.*\\.(a|b|...)$`` (``yeto.rl.export.target_modules``); the regex is handed to
verl as the equivalent name list ``[a,b,...]`` (PEFT matches a list entry as the last
module-name component, the regex's own rule), which a Hydra override can carry.
"""

from __future__ import annotations

import re
from typing import Any

from .config import VerlConfigError

CHOICES = ("auto", "attention", "all-linear")
_NAME_ALTERNATION = re.compile(r"\.\*\\\.\((?P<names>[A-Za-z0-9_]+(?:\|[A-Za-z0-9_]+)*)\)\$")


def peft_targets(choice: str, config: Any) -> str:
    """The PEFT ``target_modules`` yeto's spec derivation uses for ``choice``."""
    if choice not in CHOICES:
        raise VerlConfigError(f"--lora-targets {choice} is not mapped on the verl backend (it maps "
                              + ", ".join(CHOICES) + "; attention-routed-experts is the Miles "
                              "DeepSeek V4 recipe)")
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
