"""Release-manifest fields of a Kimi-K3.1 partial-experts run.

A release ships only the trainable experts (original ids, MXFP4 g32); its
manifest must name the selection the trainer ran with so a sampler never
mixes experts of two selections. ``expert_selection_sha256`` is the
INTERFACES 2.3 ``sha256`` of the selection file.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from yeto.rl.adapters.miles.models.kimi_k31.model import MODEL_TYPE, NUM_EXPERTS, NUM_TRAINABLE, moe_layers
from yeto.rl.adapters.miles.models.kimi_k31.ports_recipe import MEGATRON_TO_HF_MODE
from yeto.rl.adapters.miles.models.partial_experts.attest import require_sha256
from yeto.rl.adapters.miles.models.partial_experts.selection import ExpertSelection

# Recipe names that export through Miles' own converters (raw) instead of
# Megatron-Bridge; config.translate_run_config consults this for
# ``--megatron-to-hf-mode``.
RAW_HF_MODE_RECIPES = frozenset({"glm5_next", "kimi_k31"})


def uses_raw_hf_mode(recipe_name: str) -> bool:
    return recipe_name in RAW_HF_MODE_RECIPES


def megatron_to_hf_mode(recipe_name: str) -> str:
    return MEGATRON_TO_HF_MODE if uses_raw_hf_mode(recipe_name) else "bridge"


def check_selection(
    selection: ExpertSelection, *, layers: Iterable[int] | None = None, num_experts: int | None = None
) -> ExpertSelection:
    """The selection must be a Kimi-K3.1 one: 896 experts, a per-layer trainable
    count that splits evenly over the selection's EP ranks (the recipe's default
    is 128; a smaller count, e.g. 32, is a bring-up layout that fits fewer nodes),
    and exactly the model's MoE layers (``layers``; default Kimi-K3.1-Pro's 1..92;
    a cut-down checkpoint passes its own, e.g. ``range(1, 4)``). ``num_experts``
    is the checkpoint's routed-expert count (default Kimi-K3.1-Pro's 896; a
    cut-down checkpoint passes its own, e.g. 64)."""

    total = NUM_EXPERTS if num_experts is None else int(num_experts)
    if selection.model != MODEL_TYPE:
        raise ValueError(f"selection is for {selection.model!r}, not {MODEL_TYPE!r}")
    ep = int(selection.ep_size)
    if (selection.num_experts != total or not 0 < selection.num_trainable <= total
            or ep <= 0 or selection.num_trainable % ep):
        raise ValueError(
            f"selection trains {selection.num_trainable}/{selection.num_experts} experts over EP {ep}; "
            f"the checkpoint has {total} experts and the trainable count must split evenly over EP "
            f"(default {NUM_TRAINABLE})"
        )
    selection.check_layers(moe_layers() if layers is None else layers)
    return selection


def release_manifest_fields(
    selection: ExpertSelection, *, layers: Iterable[int] | None = None, num_experts: int | None = None
) -> dict[str, Any]:
    """Fields the release manifest carries for a partial-experts release
    (``PolicyRelease.manifest``); ``layers`` and ``num_experts`` as in
    :func:`check_selection`."""

    check_selection(selection, layers=layers, num_experts=num_experts)
    return {
        "expert_selection_sha256": selection.sha256,
        "partial_experts": {
            "model": MODEL_TYPE,
            "num_experts": selection.num_experts,
            "num_trainable": selection.num_trainable,
            "ep_size": selection.ep_size,
            "layers": len(selection.trainable),
            "trainable_sha256": selection.trainable_sha256,
            "source": dict(selection.source),
        },
    }


def check_release_manifest(manifest: Mapping[str, Any], *, expected_selection_sha256: str) -> None:
    """A sampler-side check that a release belongs to the running selection."""

    actual = manifest.get("expert_selection_sha256")
    wanted = require_sha256(expected_selection_sha256, "expected selection SHA256")
    if actual is None:
        raise ValueError("release manifest lacks expert_selection_sha256")
    if require_sha256(actual, "release expert_selection_sha256") != wanted:
        raise ValueError(f"release expert selection {actual} != running selection {wanted}")
