"""Pick a backend recipe for (model, hardware, constraints) and derive the parallel
layout and memory budget (A21 sections 3.2-3.5). Every number is an estimate from the
config shape; the rules are inferred and need a smoke run to calibrate."""

from __future__ import annotations

import math
from dataclasses import replace

from .families import MILES_K31
from .match import match_family
from .schema import (STATUS_ORDER, BackendRecipe, Constraints, FamilyEntry, HardwareFacts, ModelFacts,
                     Recommendation)

GIB = float(1 << 30)
USABLE_FRACTION = 0.6      # model state per card <= 60% HBM; the rest: activations, logits, KV
LARGE_MOE_B = 300.0        # >= this many params (B), a MoE gets LoRA / partial experts only
MXFP4_BYTES = 0.53         # bytes per param of MXFP4 (4-bit + e8m0 scale per 32)
K31_TRAINABLE_PER_LAYER = 128

# card -> (device family, arch, HBM GiB); the trainer-island cards of each --hardware
CARDS = {
    "nvidia": {"H100": ("cuda", "hopper", 80), "H200": ("cuda", "hopper", 141),
               "B200": ("cuda", "blackwell", 180), "B300": ("cuda", "blackwell", 288),
               "A100-80GB": ("cuda", "ampere", 80)},
    "ascend": {"910B": ("ascend", "910b", 64)},
}
DEFAULT_CARD = {"nvidia": "H100", "ascend": "910B"}
SAMPLER_ISLANDS = {"rocm:gfx950": HardwareFacts("rocm", "gfx950", "MI355X", 288, 8, 8, "roce")}
BITS = {"bf16": 16, "fp8": 8, "w8a8": 8, "mxfp4": MXFP4_BYTES * 8}


def hardware_facts(hardware: str, card: str | None = None, *, cards_per_node: int = 8,
                   max_nodes: int = 64, interconnect: str | None = None) -> HardwareFacts:
    if hardware not in CARDS:
        raise ValueError(f"--hardware must be one of {sorted(CARDS)}, not {hardware!r}")
    name = (card or DEFAULT_CARD[hardware]).upper()
    table = {k.upper(): (k, v) for k, v in CARDS[hardware].items()}
    if name not in table:
        raise ValueError(f"--card {card!r} is not a {hardware} card; known: {sorted(CARDS[hardware])}")
    canon, (fam, arch, hbm) = table[name]
    return HardwareFacts(fam, arch, canon, hbm, cards_per_node, max_nodes,
                         interconnect or ("roce" if hardware == "ascend" else "nvlink+ib"))


def _default_tuning(facts: ModelFacts, recipe: BackendRecipe) -> str:
    if recipe.rl_model_recipe == "kimi_k31":
        return "full"   # partial experts: the delivered K3.1 form
    if "lora" in recipe.tunings:
        return "lora"
    return recipe.tunings[0]


def _pow2_upto(n: int) -> list[int]:
    out, v = [], 1
    while v <= n:
        out.append(v)
        v *= 2
    return out


def _hard(recipe: BackendRecipe, key: str) -> int | None:
    for c in recipe.hard_constraints:
        if c.startswith(key + "=="):
            return int(c.split("==", 1)[1])
    return None


def plan_trainer(facts: ModelFacts, hw: HardwareFacts, recipe: BackendRecipe, tuning: str,
                 *, max_cards: int | None = None) -> dict:
    """Smallest world whose per-card model state fits; TP first (inside a node, divides
    the heads), EP for MoE (divides num_experts), PP for the rest, DP = world/(TP*PP)."""
    b = facts.breakdown
    routed = b.get("routed_experts", 0.0)
    dense = b["total"] - routed
    partial = recipe.rl_model_recipe == "kimi_k31" and tuning == "full"
    frac = K31_TRAINABLE_PER_LAYER / facts.num_experts if partial and facts.num_experts else 0.0
    frozen_routed_bytes = MXFP4_BYTES if facts.checkpoint_quant == "mxfp4" else 2.0
    fixed_tp, fixed_ep = _hard(recipe, "tp"), _hard(recipe, "ep")
    budget = USABLE_FRACTION * hw.hbm_gib * GIB
    worlds = [w for w in _pow2_upto(hw.cards_per_node)] + [
        n * hw.cards_per_node for n in range(2, hw.max_nodes + 1)]
    if max_cards:
        worlds = [w for w in worlds if w <= max_cards]
    if hw.interconnect in ("none", "gloo-only"):
        worlds = [w for w in worlds if w <= hw.cards_per_node]
    for world in worlds:
        for pp in [p for p in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32) if p <= facts.layers]:
            for tp in [t for t in _pow2_upto(min(hw.cards_per_node, world))
                       if facts.heads % t == 0 and (fixed_tp is None or t == fixed_tp)]:
                if world % (tp * pp):
                    continue
                dp = world // (tp * pp)
                ep, edp = 1, dp
                if facts.moe:
                    cands = [e for e in range(1, min(world // pp, facts.num_experts, 64) + 1)
                             if facts.num_experts % e == 0 and (world // pp) % e == 0]
                    ep = fixed_ep if fixed_ep and fixed_ep in cands else (max(cands) if cands else 1)
                    if fixed_ep and ep != fixed_ep:
                        continue
                    edp = world // (pp * ep)
                per = dense * 2.0 / (tp * pp)
                if tuning == "full" and not partial:
                    per += dense * (4.0 + 12.0 / dp) / (tp * pp)
                    per += routed * (2.0 + 4.0 + 12.0 / edp) / (ep * pp)
                elif partial:
                    per += routed * frac * (2.0 + 4.0 + 12.0 / edp) / (ep * pp)
                    per += routed * (1 - frac) * frozen_routed_bytes / (ep * pp)
                else:  # LoRA: frozen base; the adapter is <1% of the base
                    per += routed * (frozen_routed_bytes if facts.moe else 2.0) / (ep * pp)
                if per <= budget:
                    return {"world": world, "nodes": max(1, math.ceil(world / hw.cards_per_node)),
                            "tensor_parallel": tp, "pipeline_parallel": pp, "expert_parallel": ep,
                            "data_parallel": dp, "expert_data_parallel": edp,
                            "state_gib_per_card": round(per / GIB, 1), "card": hw.card,
                            "budget_gib_per_card": round(budget / GIB, 1), "fits": True,
                            "engine": ("megatron" if recipe.trainer == "verl" and max(tp, pp, ep) > 1
                                       else ("fsdp2" if recipe.trainer == "verl" else "megatron"))}
    return {"fits": False, "card": hw.card, "max_world": worlds[-1] if worlds else 0,
            "budget_gib_per_card": round(budget / GIB, 1)}


def plan_sampler(facts: ModelFacts, hw: HardwareFacts, weights: str) -> dict:
    """Weights at ``weights`` precision within 60% of the cards' HBM (the rest is KV);
    one node when it fits, else a cross-node EP/TP engine (RoCE/IB only)."""
    nbytes = facts.total_params_b * 1e9 * BITS[weights] / 8
    need = max(1, math.ceil(nbytes / (USABLE_FRACTION * hw.hbm_gib * GIB)))
    cards = next((c for c in _pow2_upto(hw.cards_per_node) if c >= need), None)
    out = {"card": hw.card, "weights": weights, "weight_gib": round(nbytes / GIB, 1)}
    if cards is not None:
        tp = cards
        while tp > 1 and facts.heads % tp:
            tp //= 2
        out.update(nodes=1, gpus_per_engine=cards, tensor_parallel=tp,
                   expert_parallel=cards if facts.moe and cards > 1 else 1, fits=True)
        return out
    nodes = math.ceil(need / hw.cards_per_node)
    ok = hw.interconnect not in ("none", "gloo-only") and nodes <= hw.max_nodes
    out.update(nodes=nodes, gpus_per_engine=nodes * hw.cards_per_node, tensor_parallel=hw.cards_per_node,
               expert_parallel=nodes * hw.cards_per_node if facts.moe else 1, fits=ok,
               note=("cross-node engine: needs RoCE/IB between the sampler nodes" if ok else
                     f"needs {nodes} nodes linked by RoCE/IB; interconnect {hw.interconnect} cannot"))
    return out


def _sampler_weights(facts: ModelFacts, recipe: BackendRecipe, hw: HardwareFacts) -> str:
    if recipe.quant.sampler_weights.startswith("mxfp4"):
        return "mxfp4"
    if hw.device_family == "ascend":
        single = facts.total_params_b * 2e9 <= USABLE_FRACTION * hw.hbm_gib * GIB * hw.cards_per_node
        return "bf16" if single else "w8a8"
    return "bf16"


def _reject(facts: ModelFacts, hw: HardwareFacts, c: Constraints, r: BackendRecipe) -> str | None:
    if hw.device_family not in r.hardware:
        return f"hardware {hw.device_family} not in {list(r.hardware)}"
    lo, hi = r.size_band_b
    if not lo <= facts.total_params_b < hi:
        return f"size {facts.total_params_b:.1f}B outside [{lo}, {hi})B"
    if c.trainer_backend and r.status != "unsupported" and r.trainer != c.trainer_backend:
        return f"trainer {r.trainer} != requested {c.trainer_backend}"
    if c.sampler_engine and r.status != "unsupported" and not r.sampler.startswith(c.sampler_engine):
        return f"sampler {r.sampler} != requested {c.sampler_engine}"
    if c.tuning and r.status != "unsupported" and c.tuning not in r.tunings:
        return f"tuning {c.tuning} not in {list(r.tunings)}"
    return None


def recommend(facts: ModelFacts, hw: HardwareFacts, constraints: Constraints | None = None,
              *, entry: FamilyEntry | None = None, card_explicit: bool = True) -> Recommendation:
    """``card_explicit=False``: ``hw.card`` is the --hardware default, so a recipe verified
    on one card (``card:B300``) is sized on that card instead."""
    c = constraints or Constraints()
    how = "given"
    if entry is None:
        entry, how = match_family(facts)
    warnings: list[str] = []
    if how == "fallback":
        warnings.append(f"no family entry matched architectures={list(facts.architectures)} "
                        f"model_type={facts.model_type!r}: falling back to {entry.family} (inferred)")
    rejected: list[tuple[str, str]] = []
    ok: list[BackendRecipe] = []
    for r in entry.recipes:
        why = _reject(facts, hw, c, r)
        if why:
            rejected.append((r.name, why))
        else:
            ok.append(r)
    ok.sort(key=lambda r: STATUS_ORDER.get(r.status, 5))
    pitfalls = list(entry.pitfalls)
    alternatives: list[str] = []
    rec = Recommendation(facts=facts, hardware=hw, family=entry.family, supported=False, recipe=None,
                         trainer_parallel={}, sampler_parallel={}, memory={}, pitfalls=pitfalls,
                         rejected=rejected, warnings=warnings, alternatives=alternatives)
    if not ok:
        rec.warnings.append("no recipe of this family fits the hardware/constraints (see rejected)")
        alternatives.extend(_other_hardware_hint(entry, facts, hw))
        return rec
    recipe = ok[0]
    if recipe.status == "unsupported":
        rec.recipe = recipe
        rec.rejected.append((recipe.name, "unsupported: " + recipe.reason))
        alternatives.append(recipe.alternative)
        return rec
    tuning = c.tuning or _default_tuning(facts, recipe)
    if facts.moe and facts.total_params_b >= LARGE_MOE_B and tuning == "full" \
            and recipe.rl_model_recipe != "kimi_k31":
        rec.recipe = recipe
        rec.rejected.append((recipe.name, f"full-parameter on a {facts.total_params_b:.0f}B MoE: optimizer "
                             f"state ~{facts.total_params_b * 16 / 1000:.1f} TB (16 B/param)"))
        alternatives.append("--tuning lora (frozen base), or a partial-experts recipe like kimi_k31")
        return rec
    if recipe.rl_model_recipe == "kimi_k31" and tuning == "lora":
        recipe = replace(recipe, trainer_image=MILES_K31)
    need_card = next((x.split(":", 1)[1] for x in recipe.hard_constraints if x.startswith("card:")), None)
    if need_card and hw.card != need_card and not card_explicit:
        for family, cards in CARDS.items():
            if need_card in cards and cards[need_card][0] == hw.device_family:
                hw = replace(hw, card=need_card, arch=cards[need_card][1], hbm_gib=cards[need_card][2])
                rec.hardware = hw
    if need_card and hw.card != need_card:
        warnings.append(f"{recipe.name} is verified on {need_card}; sizing on {hw.card} instead")
    trainer = plan_trainer(facts, hw, recipe, tuning, max_cards=c.max_cards)
    s_hw = SAMPLER_ISLANDS.get(recipe.sampler_hardware or "", hw)
    sampler = plan_sampler(facts, s_hw, _sampler_weights(facts, recipe, s_hw))
    rec.recipe, rec.trainer_parallel, rec.sampler_parallel = recipe, trainer, sampler
    rec.trainer_parallel["tuning"] = tuning
    pitfalls.extend(recipe.pitfalls)
    if recipe.status != "verified":
        warnings.append(f"recipe status {recipe.status}: {recipe.evidence}; smoke a small/4-layer copy first")
    if not trainer["fits"]:
        warnings.append(f"trainer state does not fit {trainer['max_world']} x {hw.card} at "
                        f"{USABLE_FRACTION:.0%} HBM; use LoRA, more nodes (--max-nodes), or a larger card")
    if not sampler["fits"]:
        warnings.append("sampler: " + sampler.get("note", "does not fit"))
    rec.supported = bool(trainer["fits"] and sampler["fits"])
    if not rec.supported:
        alternatives.extend(_other_hardware_hint(entry, facts, hw))
    rec.memory = memory_budget(facts, trainer, sampler, recipe, tuning)
    return rec


def memory_budget(facts: ModelFacts, trainer: dict, sampler: dict, recipe: BackendRecipe, tuning: str) -> dict:
    """Host RAM, /dev/shm and disk (GiB) of the steps that hold the whole model."""
    p = facts.total_params_b * 1e9
    nodes = max(1, trainer.get("nodes", 1))
    out = {
        "bf16_weights_gib": round(p * 2 / GIB, 1),
        "release_gib": sampler.get("weight_gib"),
        "dev_shm_gib_per_sampler_node": round((sampler.get("weight_gib") or 0) / max(1, sampler.get("nodes", 1)), 1),
        "host_ram_gib_per_trainer_node_load": round(p * 2 / GIB / nodes, 1),
    }
    if tuning == "full":
        trainable = p if recipe.rl_model_recipe != "kimi_k31" else (
            facts.breakdown.get("routed_experts", 0) * K31_TRAINABLE_PER_LAYER / max(1, facts.num_experts))
        out["checkpoint_disk_gib"] = round(trainable * 16 / GIB, 1)
        out["syncer_f32_master_gib"] = round(trainable * 4 / GIB, 1)
        out["syncer_f32_master_x3_gib"] = round(trainable * 12 / GIB, 1)
    if recipe.trainer == "verl":
        out["verl_rank0_fp32_load_gib_unpatched"] = round(p * 4 / GIB, 1)
    return out


def _other_hardware_hint(entry: FamilyEntry, facts: ModelFacts, hw: HardwareFacts) -> list[str]:
    out = []
    for r in entry.recipes:
        if r.status != "unsupported" and hw.device_family not in r.hardware \
                and r.size_band_b[0] <= facts.total_params_b < r.size_band_b[1]:
            out.append(f"{r.name} ({r.trainer} + {r.sampler}, status {r.status})")
    return out
