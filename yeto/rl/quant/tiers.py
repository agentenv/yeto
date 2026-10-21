"""Rollout quantization tiers: one backend-neutral setting, two engine plans (S26 E6).

``--rl-rollout-quant-tier N`` picks the precision the sampler generates with.
The trainer always trains in its own precision (BF16 weights); the off-policy
gap the tier adds is corrected by the run's TIS (``--rl-algorithm-spec``).

======  ===============  =====================================================
tier    name             what changes
======  ===============  =====================================================
0       ``bf16``         nothing (default)
1       ``online-fp8``   the sampler engine quantizes the in-scope linears to
                         FP8 itself at load (SGLang ``--quantization fp8``;
                         vLLM ``quantization=fp8``); KV cache stays BF16:
                         an FP8 KV cache without calibrated scales failed
                         the gate on Qwen3-4B (p99 |diff| 0.82 against 0.46
                         with a BF16 KV cache, limit 0.5; e6-gate-t1-r3 /
                         e6-gate-t1-bf16kv, 2026-10-21)
2       ``trainer-fp8``  the trainer side converts each BF16 release to FP8
                         W8A8 per channel on the in-scope linears
                         (:data:`~.schemes.FP8_FFN_CHANNEL`) and the sampler
                         island loads that checkpoint; KV cache stays BF16
3       ``4bit``         interface only (large models); refused at launch
======  ===============  =====================================================

Scope (every tier, :mod:`.scope`): a dense model quantizes only its FFN
linears, a MoE model only its routed experts; router, ``lm_head``,
embeddings, norms and attention stay BF16. An engine plan that would quantize
anything else is refused before the engine starts.

Where each tier runs:

* tier 1, colocated or sampler islands, both backends. Miles/SGLang: the
  ignore list reaches SGLang through ``SGLANG_FP8_IGNORED_LAYERS`` (sglang
  2fa88018 reads it in ``Fp8Config``); verl/vLLM: the FP8 config of verl's own
  FP8 rollout (fork acad9875 ``vllm_async_server._apply_quantization``) with
  the out-of-scope linears added to ``ignored_layers``.
* tier 2, store-coupled sampler/trainer islands only
  (``--rl-island-sample-role``, full-parameter): a colocated island has no
  checkpoint transfer to make smaller, and a LoRA release carries no base
  weights to convert.

FP8 needs compute capability 8.9 or newer (Ada, Hopper, Blackwell); the launcher
accepts only the cards of :data:`FP8_CARDS` for tiers 1 and 2.

Status: CPU tests only (tests/test_rl_quant_tiers.py). NOT VERIFIED on a GPU:
SGLang online FP8 with an in-place BF16 weight update (Miles colocated), the
vLLM FP8 config through ``engine_kwargs.vllm.hf_overrides`` (verl), the FP8
compressed-tensors checkpoint through ``update_weights_from_disk``.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import scope
from .schemes import BF16, FP8_FFN_CHANNEL

TIER_BF16 = 0
TIER_ONLINE_FP8 = 1
TIER_TRAINER_FP8 = 2
TIER_4BIT = 3
TIER_NAMES = {TIER_BF16: "bf16", TIER_ONLINE_FP8: "online-fp8", TIER_TRAINER_FP8: "trainer-fp8",
              TIER_4BIT: "4bit"}
TIERS = tuple(TIER_NAMES)

FLAG = "--rl-rollout-quant-tier"
SOURCE_FLAG = "--rl-rollout-quant-fp8-source"
# Tier 1, who quantizes: the engine at load (online FP8), or the trainer on every weight
# update (Miles quantize_params_fp8, 128x128 blocks) with the engines started on Miles'
# block-FP8 conversion of the base. Both keep the scope rule's modules in BF16.
FP8_SOURCE_ENGINE = "engine"
FP8_SOURCE_TRAINER = "trainer"
NATIVE_BLOCK = [128, 128]
SGLANG_IGNORE_ENV = "SGLANG_FP8_IGNORED_LAYERS"
# verl's own FP8 rollout config (fork acad9875 vllm_async_server.py:1373); the
# ignore list is replaced by ours, which keeps verl's ``mlp.gate`` entries.
VLLM_FP8_CONFIG = {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
                   "weight_block_size": [128, 128]}

# Cards with FP8 tensor cores (compute capability >= 8.9), by canonical name
# (``yeto.gpu_spec``). Any other card is refused, also NPUs and PPUs (not checked).
FP8_CARDS = frozenset({"L4", "L40S", "RTX-6000-Ada", "RTX-PRO-6000", "H100", "H200", "B200"})


class QuantTierError(ValueError):
    """The tier cannot run with this launch (refused before any cloud spend)."""


def tier_of(args: Any) -> int:
    tier = getattr(args, "rl_rollout_quant_tier", TIER_BF16)
    tier = TIER_BF16 if tier is None else int(tier)
    if tier not in TIERS:
        raise QuantTierError(f"{FLAG} {tier}: expected one of {TIERS}")
    return tier


def release_scheme(tier: int) -> str:
    """The release scheme the sampler of ``tier`` serves (tier 2: the FP8 releases)."""
    return FP8_FFN_CHANNEL if tier == TIER_TRAINER_FP8 else BF16


VERL_BACKEND = "verl"
# H100 runs 2026-10-21 (SESSION26/logs/E6): vLLM 0.29 given verl's FP8 config
# (hf_overrides, quant_method fp8) on a BF16 checkpoint treats the weights as FP8
# serialized and generates garbage (512-token answers, NaN log-probs) until verl's
# weight sync re-quantizes the base; yeto verl is LoRA-only and a LoRA sync never sends
# the base (e6-t1-verl-a-r2, e6-vllm-fp8-probe). Plain vLLM online FP8 works with LoRA
# but has no ignore list: it quantizes attention, which failed the gate as SGLang
# online FP8 of every linear (p99 |diff| 0.60, limit 0.5).
VERL_TIER1_REFUSAL = (f"{FLAG} 1 is not available on verl: verl's FP8 rollout needs a base-weight "
                      "sync to quantize the base, and yeto verl islands are LoRA-only (the engine "
                      "would generate garbage); use tier 0 on verl")


# Backends whose tier 1 defaults to the trainer FP8 source (framework-native block FP8
# with the ignore list): its H100 gate margin is larger than the engine's online FP8
# (p99 |diff| 0.43 vs 0.46-0.49, limit 0.5; SESSION26 E6 round 2).
TRAINER_SOURCE_BACKENDS = frozenset({"miles"})


def default_fp8_source(args: Any) -> str:
    """The FP8 source when ``--rl-rollout-quant-fp8-source`` is not given."""
    backend = getattr(args, "rl_backend", None) or "miles"
    if tier_of(args) == TIER_ONLINE_FP8 and backend in TRAINER_SOURCE_BACKENDS:
        return FP8_SOURCE_TRAINER
    return FP8_SOURCE_ENGINE


def fp8_source(args: Any) -> str:
    return getattr(args, "rl_rollout_quant_fp8_source", None) or default_fp8_source(args)


def validate_launch(args: Any) -> int:
    """Refuse a tier the launch cannot run (launcher side, no checkpoint needed); returns the tier."""
    tier = tier_of(args)
    if getattr(args, "rl_rollout_quant_fp8_source", None) is None and hasattr(args, "__dict__"):
        # Resolved once here so the island flags and the run config carry it explicitly.
        args.rl_rollout_quant_fp8_source = default_fp8_source(args)
    if tier == TIER_BF16:
        if fp8_source(args) != FP8_SOURCE_ENGINE:
            raise QuantTierError(f"{SOURCE_FLAG} {fp8_source(args)} needs {FLAG} 1")
        return tier
    if tier == TIER_4BIT:
        raise QuantTierError(f"{FLAG} 3 (4-bit) is an interface only in this version (large models; "
                             "H100 has no FP4 units); use tier 0, 1 or 2")
    cards = _cards(getattr(args, "gpu", None))
    if lacking := sorted(cards - FP8_CARDS):
        raise QuantTierError(f"{FLAG} {tier} needs FP8 tensor cores (compute capability >= 8.9: "
                             f"{', '.join(sorted(FP8_CARDS))}); --gpu names {', '.join(lacking)}")
    if tier == TIER_ONLINE_FP8 and getattr(args, "rl_backend", None) == VERL_BACKEND:
        raise QuantTierError(VERL_TIER1_REFUSAL)
    if fp8_source(args) == FP8_SOURCE_TRAINER and tier != TIER_ONLINE_FP8:
        raise QuantTierError(f"{SOURCE_FLAG} trainer applies to tier 1 only (got {FLAG} {tier})")
    if tier == TIER_TRAINER_FP8:
        role = getattr(args, "rl_sample_role", None) or "both"
        if not getattr(args, "rl_island_sample_role", None) and role not in ("sampler", "trainer"):
            raise QuantTierError(
                f"{FLAG} 2 converts each trainer release and ships it to a sampler island: it needs "
                "store-coupled islands (--rl-island-sample-role, or one launch per role with "
                "--rl-sample-role). Colocated islands use tier 1")
        if getattr(args, "tuning", "lora") != "full":
            raise QuantTierError(f"{FLAG} 2 needs --tuning full: a LoRA release carries no base weights "
                                 "to convert (tier 1 quantizes the base once in the engine)")
    return tier


def _cards(gpu: Any) -> set[str]:
    """Canonical card names of a ``--gpu`` spec; empty when there is none (auto fleet)
    or it does not parse (the launcher reports that itself)."""
    if not gpu:
        return set()
    from yeto.gpu_spec import parse_gpu_spec

    try:
        return {cluster.gpu for cluster in parse_gpu_spec(str(gpu))}
    except ValueError:
        return set()


# ---------------------------------------------------------------- tier 1 engine plans
@dataclass(frozen=True)
class OnlinePlan:
    """What an online FP8 engine gets (tier 1), checked against the checkpoint.

    ``explicit``: every out-of-scope linear of the checkpoint by name;
    ``segments``: path segments SGLang also matches (:func:`~.scope.engine_ignore_list`);
    ``quantized``: the modules the engine will quantize (simulated, in scope by check)."""

    kind: str
    explicit: tuple[str, ...]
    segments: tuple[str, ...]
    quantized: tuple[str, ...]

    def summary(self) -> dict[str, Any]:
        return {"tier": TIER_ONLINE_FP8, "kind": self.kind, "quantized_modules": len(self.quantized),
                "ignored_modules": len(self.explicit), "example_quantized": list(self.quantized[:3])}


def online_plan(checkpoint: Path) -> OnlinePlan:
    """The tier 1 plan for the BF16 checkpoint ``checkpoint``; raises
    :class:`~.scope.QuantScopeError` when the engine would quantize an out-of-scope module
    (with either the SGLang list or the explicit vLLM list)."""
    kind = scope.model_kind_of(checkpoint)
    shapes = scope.checkpoint_shapes(checkpoint)
    explicit = scope.ignored_modules(shapes, kind)
    segments = sorted(set(scope.engine_ignore_list(shapes, kind)) - set(explicit))
    quantized = scope.check_online_plan(shapes, kind, [*explicit, *segments])
    if scope.check_online_plan(shapes, kind, explicit) != quantized:
        raise scope.QuantScopeError("the SGLang and vLLM ignore lists quantize different modules")
    return OnlinePlan(kind, tuple(explicit), tuple(segments), tuple(quantized))


def sglang_online_argv() -> list[str]:
    """Miles spellings of the SGLang tier 1 server flags (Miles prefixes SGLang's own)."""
    return ["--sglang-quantization", "fp8"]


def sglang_online_env(plan: OnlinePlan) -> dict[str, str]:
    """The SGLang worker environment of a tier 1 plan."""
    entries = [*plan.explicit, *plan.segments]
    if any("," in entry for entry in entries):
        raise scope.QuantScopeError("an ignore entry holds a comma; SGLANG_FP8_IGNORED_LAYERS splits on commas")
    return {SGLANG_IGNORE_ENV: ",".join(entries)}


def native_ignore_list(plan: OnlinePlan) -> list[str]:
    """Tier 1 trainer source: the modules Miles' converter and ``quantize_params_fp8``
    keep in BF16 (the converter records them in ``modules_to_not_convert``, which
    SGLang and Miles both read; SGLang path match, as the online ignore list)."""
    return [*plan.explicit, *plan.segments]


def check_native_base(directory: Path) -> int:
    """Tier 1 trainer source: Miles' block-FP8 conversion must be ``quant_method fp8``
    with 128x128 blocks and hold FP8 weights only in scope. Returns the FP8 module count."""
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    q = config.get("quantization_config") or {}
    if q.get("quant_method") != "fp8" or list(q.get("weight_block_size") or ()) != NATIVE_BLOCK:
        raise scope.QuantScopeError(f"{directory}: quantization_config is {q.get('quant_method')!r} "
                                    f"blocks {q.get('weight_block_size')!r}, expected fp8 {NATIVE_BLOCK}")
    dtypes = _checkpoint_dtypes(directory)
    quantized = [n[: -len(".weight")] for n, d in dtypes.items() if d == "F8_E4M3" and n.endswith(".weight")]
    if not quantized:
        raise scope.QuantScopeError(f"{directory}: an FP8 base without FP8 weights")
    scope.check_quantized(quantized, scope.model_kind(config), what=f"FP8 base {directory.name}")
    return len(quantized)


def vllm_online_overrides(plan: OnlinePlan, num_layers: int) -> list[str]:
    """verl Hydra overrides of a tier 1 plan.

    ``rollout.quantization=fp8`` keeps verl's FP8 load path (weights quantized on each
    sync, only for modules vLLM built as FP8: ``vllm_quant_utils.is_fp8_weight``).
    ``engine_kwargs.vllm.hf_overrides`` replaces verl's FP8 config (``**engine_kwargs``
    comes after it in the vLLM arguments) with the same config whose ``ignored_layers``
    are the plan's explicit names plus verl's ``model.layers.<i>.mlp.gate`` entries.
    vLLM matches ``ignored_layers`` by exact name. The value is a JSON string, which
    verl passes unchanged as ``--hf-overrides``.
    """
    gates = [f"model.layers.{i}.mlp.gate" for i in range(int(num_layers))]
    config = {**VLLM_FP8_CONFIG, "ignored_layers": sorted(set(plan.explicit) | set(gates))}
    hf_overrides = json.dumps({"quantization_config": config}, separators=(",", ":"))
    if "'" in hf_overrides:
        raise scope.QuantScopeError("a module name holds a single quote; it cannot be a Hydra quoted value")
    return ["actor_rollout_ref.rollout.quantization=fp8",
            f"++actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides='{hf_overrides}'"]


def num_hidden_layers(checkpoint: Path) -> int:
    config = json.loads((Path(checkpoint) / "config.json").read_text())
    section = config if "num_hidden_layers" in config else (config.get("text_config") or {})
    return int(section["num_hidden_layers"])


# ---------------------------------------------------------------- tier 2 release side
def check_release_scope(directory: Path) -> int:
    """Post-load check (before the engine is touched) of an FP8 release: the release
    must declare the FP8 format and every FP8 weight must be in scope. Returns the
    quantized module count. Reads ``config.json`` and the shard headers only."""
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    q = config.get("quantization_config") or {}
    if q.get("quant_method") != "compressed-tensors" or q.get("format") != "float-quantized":
        raise scope.QuantScopeError(f"{directory}: quantization_config is {q.get('quant_method')!r}/"
                                    f"{q.get('format')!r}, expected compressed-tensors/float-quantized")
    kind = scope.model_kind(config)
    dtypes = _checkpoint_dtypes(directory)
    quantized = [n[: -len(".weight")] for n, d in dtypes.items() if d == "F8_E4M3" and n.endswith(".weight")]
    scope.check_quantized(quantized, kind, what=f"release {directory.name}")
    if not quantized:
        raise scope.QuantScopeError(f"{directory}: an FP8 release without FP8 weights")
    unlisted = [m for m in scope.ignored_modules(scope.checkpoint_shapes(directory), kind)
                if m not in set(q.get("ignore") or ())]
    if unlisted:
        raise scope.QuantScopeError(f"{directory}: {len(unlisted)} BF16 linears are not in the config "
                                    f"ignore list (the engine would expect FP8 weights), e.g. {unlisted[:3]}")
    return len(quantized)


def _checkpoint_dtypes(directory: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for shard in sorted(Path(directory).glob("*.safetensors")):
        with open(shard, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(n))
        header.pop("__metadata__", None)
        out.update({name: info["dtype"] for name, info in header.items()})
    return out


def describe(tier: int) -> str:
    return f"rollout quant tier {tier} ({TIER_NAMES[tier]})"
