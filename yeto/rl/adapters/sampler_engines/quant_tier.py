"""Rollout quantization tier of an engine sampler island (S26 E6; adapter layer).

The launch layer writes ``"rollout_quant": {"tier": N, "apply": APPLY}`` into the
sampler spec (:func:`spec_section`); the engine sampler entry calls ``APPLY`` on
the sampler node before it builds anything (the checkpoint is only there). The
function returns a new spec with the engine settings of the tier filled in:

* tier 1, ``sglang-server`` only: ``--quantization fp8`` (KV cache BF16) and ``SGLANG_FP8_IGNORED_LAYERS`` from the checked online plan of
  ``base_dir`` (:func:`yeto.rl.quant.tiers.online_plan`). ``vllm-server`` is
  refused: stand-alone vLLM online FP8 has no ignore list, so it would quantize
  attention (verl passes one through its own FP8 path instead);
* tier 2, both adapters: ``base_dir`` is converted to the FP8 scheme in
  ``work_dir`` (the engine must start on the layout later releases load into);
  ``quant_scheme`` / ``expected_format`` / ``release_dir`` follow the FP8
  scheme; ``checks.expected_expert_matrices`` is the routed-expert count of the
  layout (0 for a dense model, which the post-load check accepts as "no routed
  experts" once the release index also counts 0).

A spec that already sets a quantized scheme, an engine ``--quantization`` flag
or another expected expert count is refused (one source of truth per setting).
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from yeto.rl.quant import tiers
from yeto.rl.quant.schemes import BF16, FP8_FFN_CHANNEL, release_dir_for

APPLY = "yeto.rl.adapters.sampler_engines.quant_tier:apply"
FP8_FORMAT = "float-quantized"
SGLANG = "sglang-server"
VLLM = "vllm-server"


class QuantTierSpecError(ValueError):
    """The sampler spec cannot carry the requested tier."""


def spec_section(tier: int) -> dict[str, Any] | None:
    """What the launch layer writes into a sampler spec (None for tier 0)."""
    if tier == tiers.TIER_BF16:
        return None
    if tier not in (tiers.TIER_ONLINE_FP8, tiers.TIER_TRAINER_FP8):
        raise QuantTierSpecError(f"{tiers.FLAG} {tier}: engine sampler islands run tier 0, 1 or 2")
    return {"tier": int(tier), "apply": APPLY}


def apply(spec: Mapping[str, Any], *, work_dir: Path | None = None) -> dict[str, Any]:
    """The spec with the tier's engine settings (module doc)."""
    tier = int((spec.get("rollout_quant") or {}).get("tier", tiers.TIER_BF16))
    out = copy.deepcopy(dict(spec))
    if tier == tiers.TIER_BF16:
        return out
    engine = out["engine"]
    server = engine["server"]
    adapter = str(engine.get("adapter"))
    if server.get("quant_scheme", BF16) != BF16 or "--quantization" in (server.get("extra_args") or ()):
        raise QuantTierSpecError(f"{tiers.FLAG} {tier}: the spec already sets quant_scheme "
                                 f"{server.get('quant_scheme')!r} / extra_args {server.get('extra_args')}")
    base = Path(server["base_dir"])
    if tier == tiers.TIER_ONLINE_FP8:
        if adapter != SGLANG:
            raise QuantTierSpecError(f"{tiers.FLAG} 1 on {adapter}: stand-alone vLLM online FP8 has no ignore "
                                     "list and would quantize attention; use sglang-server or tier 2")
        plan = tiers.online_plan(base)
        server["extra_args"] = [*(server.get("extra_args") or ()), "--quantization", "fp8"]
        server["extra_env"] = {**(server.get("extra_env") or {}), **tiers.sglang_online_env(plan)}
        server["precision"] = "fp8"  # provenance of the samples; the release scheme stays bf16
        out["rollout_quant"] = {**out["rollout_quant"], "plan": plan.summary()}
        return out
    from yeto.rl.quant.awq import AwqSmoothing  # noqa: PLC0415 - torch only for tier 2
    from yeto.rl.quant.release import routed_expert_matrices  # noqa: PLC0415
    from yeto.rl.quant.requant import requantize_checkpoint, save_scoped_template  # noqa: PLC0415

    work = Path(work_dir) if work_dir is not None else Path(str(out.get("event_tape", "."))).expanduser().parent
    template, converted = work / "fp8-template", work / "fp8-base"
    layout = save_scoped_template(base, template, FP8_FFN_CHANNEL)
    requantize_checkpoint(base, template, converted, AwqSmoothing.none())
    tiers.check_release_scope(converted)
    experts = routed_expert_matrices(layout.quantized_modules)
    checks = dict(out.get("checks") or {})
    if "expected_expert_matrices" in checks and checks["expected_expert_matrices"] != experts:
        raise QuantTierSpecError(f"checks.expected_expert_matrices {checks['expected_expert_matrices']} but the "
                                 f"FP8 layout has {experts} routed-expert matrices")
    checks["expected_expert_matrices"] = experts
    server.update(base_dir=str(converted), quant_scheme=FP8_FFN_CHANNEL, expected_format=FP8_FORMAT,
                  precision="fp8")
    out.update(base_dir=str(converted), release_dir=release_dir_for(FP8_FFN_CHANNEL), checks=checks)
    out["rollout_quant"] = {**out["rollout_quant"], "bf16_base": str(base), "routed_expert_matrices": experts}
    return out
