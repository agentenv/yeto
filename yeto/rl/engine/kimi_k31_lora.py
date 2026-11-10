"""Kimi-K3.1 LoRA on ports Miles (``--tuning lora --rl-model-recipe kimi_k31``).

The recipe has two trainable forms, picked by ``--tuning``:

* ``full``: the partial-experts form (128 of 896 routed experts per MoE layer
  train in place, ``--rl-expert-selection``); unchanged by this module;
* ``lora``: a BF16 adapter over the frozen checkpoint, as Miles'
  ``scripts/kimi_k31_w4a16_lora_common.py`` trains it (``docs/models/kimi/kimi-k31.md``
  5.5). The Miles fork keeps the frozen routed experts as the checkpoint's MXFP4
  bytes (``miles_plugins.models.kimi_k3.mxfp4_experts``; ``frozen_experts_are_mxfp4``
  allows them for LoRA or partial experts only) and resolves the HF targets itself
  (``miles.utils.lora.hf_lora_targets._kimi_k3_targets``).

Targets (yeto ``--lora-targets`` -> Miles ``--target-modules``):

* ``attention``: the ``attn`` group, ``self_attn.{q_a_proj, kv_a_proj_with_mqa, o_proj}``
  of every layer (KDA layers have ``o_proj`` only). Every adapted weight is BF16 in
  the checkpoint, so a sampler without LoRA support can merge the adapter exactly
  (up to BF16 rounding) and keep the MXFP4 experts byte-identical;
* ``auto``: ``attention`` (as ``--help`` states for MoE: router and routed experts
  stay frozen), so the default adapter is one every sampler can merge;
* ``all-linear`` (explicit only): the upstream recipe, ``attn`` + ``mlp``: also the
  dense MLP of layer 0, the shared expert and every routed expert, the latter with
  shared outer factors (``--experts-shared-outer-loras``). The routed-expert factors
  can only be served as a bypass next to the MXFP4 experts (SGLang
  ``sglang-miles-kimi-k31``: MoE LoRA on the Marlin MXFP4 runner); merging them would
  re-quantize every expert to MXFP4. A vLLM A-card sampler (MXFP4 experts, merged
  BF16 release) cannot serve them, so a trainer-role launch that names
  ``--rl-sampler-engine vllm`` (or names no engine) is refused at dry-run time
  (:func:`check_sampler_engine`).

``--rl-expert-selection`` and ``--rl-routed-expert-qat`` belong to the full form;
a LoRA run refuses them.

Trainable state (yeto's export/apply, the sync and the publication digests): Miles
trains K3.1 in raw bridge mode, so the adapter lives in Miles' native adapter slots
(``NativeAdapterSlots``), not in Megatron-Bridge adapter conversion tasks; the island
marks the run (``yeto_rl_native_lora_export = "kimi_k31"``) and the state plugin
reads and writes the fp32 masters through the slots (``state_plugin._export_native_slots``).
The LoRA layout is not predictable on the CPU (AutoBridge cannot read the checkpoint):
it is learned from the first export, on every island alike, and the syncer's session
contract (layout + island contract) refuses islands whose layouts differ.

Policy sync (synced trainer islands behind the Rust syncer, decoupled DiLoCo):
``--lora-targets attention`` only, on the round presets whose syncer layout is built
from the first export (:data:`SYNC_PRESETS`: ``strict-avg``, also as the island pool).
r=16 attention is about 35 M parameters (141 MB f32) for the 93-layer checkpoint, so
the syncer's traffic and memory are small. Routed-expert targets are refused there:
their adapter (tens of GB) is not a delta the syncer should carry, and an A-card
sampler cannot merge it anyway. ``decoupled`` (fragment streaming) needs the layout
before the first export (its fragment plan is fixed at start-up) and is refused.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

MODEL_TYPE = "kimi_k31"  # = adapters/miles/models/kimi_k31/model.MODEL_TYPE (neutral core: no adapter import)

FULL, LORA = "full", "lora"
TUNINGS = (FULL, LORA)

# yeto --lora-targets -> Miles --target-modules (comma list of HF groups)
TARGET_MODULES: dict[str, tuple[str, ...]] = {
    "attention": ("attn",),
    "all-linear": ("all-linear",),
    "auto": ("attn",),
}
ROUTED_EXPERT_TARGETS = frozenset({"all-linear"})
# --rl-sampler-engine values that can serve routed-expert LoRA factors next to MXFP4 experts
ROUTED_EXPERT_LORA_ENGINES = frozenset({"sglang"})
SHARED_OUTER_FLAG = "--experts-shared-outer-loras"
# policy sync: the presets whose syncer layout follows the first export, and the targets
SYNC_PRESETS = ("strict-avg",)
SYNC_TARGETS = ("attention",)
# yeto_rl_native_lora_export: the state plugin's native-slots path (module doc)
NATIVE_EXPORT = MODEL_TYPE
# the release manifest's LoRA block (PolicyRelease.manifest["lora"])
MANIFEST_KEY = "lora"
MANIFEST_SCHEMA = "yeto-k31-lora-v1"


class KimiK31LoraError(ValueError):
    """A K3.1 tuning request outside the recipe's two forms."""


def target_modules(lora_targets: str | None) -> tuple[str, ...]:
    """The Miles ``--target-modules`` of a yeto ``--lora-targets`` preset."""
    try:
        return TARGET_MODULES[str(lora_targets)]
    except KeyError:
        raise KimiK31LoraError(
            f"--lora-targets {lora_targets} is not a Kimi-K3.1 LoRA preset; "
            f"use one of {sorted(TARGET_MODULES)}") from None


def adapts_routed_experts(lora_targets: str | None) -> bool:
    return str(lora_targets) in ROUTED_EXPERT_TARGETS


def check_tuning(tuning: str, *, expert_selection: Any = None, routed_expert_qat: Any = None,
                 lora_targets: str | None = None, policy_sync: bool = False,
                 sync_preset: str | None = None) -> str:
    """Validate one K3.1 request; returns the tuning. The full form keeps its own
    checks at each call site (its messages are unchanged). ``policy_sync``: the
    island runs behind the syncer (``sync_preset``: its preset; None = not known
    at this call site, the island checks it)."""
    tuning = str(tuning)
    if tuning not in TUNINGS:
        raise KimiK31LoraError(f"--tuning {tuning}: the kimi_k31 recipe trains {TUNINGS}")
    if tuning == FULL:
        return tuning
    if expert_selection:
        raise KimiK31LoraError("--rl-expert-selection belongs to the full-parameter kimi_k31 form "
                               "(--tuning full); a LoRA run never trains routed experts in place")
    if routed_expert_qat:
        raise KimiK31LoraError("--rl-routed-expert-qat belongs to the full-parameter kimi_k31 form "
                               "(--tuning full); a LoRA run never trains the MXFP4 experts")
    target_modules(lora_targets)
    if policy_sync:
        check_sync(lora_targets, sync_preset)
    return tuning


def check_sync(lora_targets: str | None, sync_preset: str | None = None) -> None:
    """A K3.1 LoRA island behind the syncer (module doc, policy sync)."""
    if target_modules(lora_targets) not in {TARGET_MODULES[t] for t in SYNC_TARGETS}:
        raise KimiK31LoraError(
            f"kimi_k31 LoRA behind the syncer takes --lora-targets {'/'.join(SYNC_TARGETS)} "
            f"(got {lora_targets}): routed-expert factors are tens of GB per learner and no A-card "
            "sampler can merge them; train them on one island (--rl-single-island-no-sync)")
    if sync_preset is not None and str(sync_preset) not in SYNC_PRESETS:
        raise KimiK31LoraError(
            f"kimi_k31 LoRA behind the syncer runs --rl-sync-preset {'/'.join(SYNC_PRESETS)} "
            f"(got {sync_preset}): the LoRA layout is learned from the first export, and "
            "the decoupled fragment plan needs it before start-up")


def check_sampler_engine(lora_targets: str | None, *, sample_role: str | None,
                         sampler_engine: str | None) -> None:
    """Refuse a K3.1 LoRA release the separate samplers cannot serve.

    Only a trainer-role island hands its adapter to samplers it does not run.
    A vLLM A-card sampler keeps the routed experts as MXFP4 checkpoint bytes and
    applies each release by merging the adapter into BF16 weights: attention (and
    the BF16 dense / shared modules) merge exactly, routed-expert factors would
    re-quantize every expert. Routed-expert targets therefore need a sampler engine
    that serves MoE LoRA as a bypass (``--rl-sampler-engine sglang``).
    """
    if str(sample_role) != "trainer" or not adapts_routed_experts(lora_targets):
        return
    if sampler_engine in ROUTED_EXPERT_LORA_ENGINES:
        return
    if sampler_engine is None:
        raise KimiK31LoraError(
            f"kimi_k31 --lora-targets {lora_targets} adapts the MXFP4 routed experts; a "
            "--rl-sample-role trainer release must be served by samplers that run MoE LoRA "
            "as a bypass. Pass --rl-sampler-engine sglang if they do, or use "
            "--lora-targets attention (the default) for vLLM A-card samplers")
    raise KimiK31LoraError(
        f"kimi_k31 --lora-targets {lora_targets} is not servable by --rl-sampler-engine "
        f"{sampler_engine}: the vLLM A-card sampler keeps the routed experts as MXFP4 bytes "
        "and merges each release into BF16 weights, so it can apply attention (and BF16 "
        "dense/shared) adapters only; merging routed-expert factors would re-quantize every "
        "expert. Use --lora-targets attention")


def extra_lora_argv(lora_targets: str | None) -> list[str]:
    """Miles flags a K3.1 LoRA run adds beyond the generic LoRA flags.

    Every native kimi_k3 LoRA declares the shared-outer expert layout: Miles'
    ``validate_native_lora_targets`` asserts it for any spec with experts, whatever
    the targets (an attention-only adapter attaches nothing to the experts; the
    flag only fixes the layout the adapter config records)."""
    del lora_targets
    return [SHARED_OUTER_FLAG]


def manifest_fields(*, rank: int, lora_targets: str | None) -> dict[str, Any]:
    """``PolicyRelease.manifest`` of a K3.1 LoRA release: what a sampler needs
    to decide, before it touches the engine, how it can serve the adapter."""
    if int(rank) <= 0:
        raise KimiK31LoraError(f"LoRA rank {rank} must be positive")
    return {MANIFEST_KEY: {
        "schema": MANIFEST_SCHEMA,
        "model": MODEL_TYPE,
        "rank": int(rank),
        "lora_targets": str(lora_targets),
        "target_modules": list(target_modules(lora_targets)),
        "routed_experts": adapts_routed_experts(lora_targets),
        "experts_shared_outer": True,  # extra_lora_argv: every native kimi_k3 LoRA
    }}


def lora_manifest(manifest: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    block = (manifest or {}).get(MANIFEST_KEY)
    return block if isinstance(block, Mapping) else None


__all__ = ["FULL", "LORA", "TUNINGS", "TARGET_MODULES", "SHARED_OUTER_FLAG", "MANIFEST_KEY",
           "NATIVE_EXPORT", "SYNC_PRESETS", "SYNC_TARGETS",
           "KimiK31LoraError", "adapts_routed_experts", "check_sampler_engine", "check_sync", "check_tuning", "extra_lora_argv",
           "lora_manifest", "manifest_fields", "target_modules"]
