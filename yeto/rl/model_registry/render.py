"""Render a :class:`Recommendation`: existing CLI flags, a text report, a run-config draft."""

from __future__ import annotations

import json
from typing import Any

from .schema import Recommendation

BACKEND = {"miles": "miles", "verl": "verl"}


def cli_flags(rec: Recommendation, model: str = "<model>") -> list[str]:
    """``yeto launch`` flags that exist today (``--rl-model-recipe`` choices, parallel
    sizes, ``--tuning``); images are left to the backend default (``resolve_default_rl_image``)."""
    r = rec.recipe
    if not rec.supported or r is None or r.rl_model_recipe is None:
        return []
    t = rec.trainer_parallel
    flags = ["--training-mode", "rl", "--model", model, "--rl-backend", BACKEND[r.trainer],
             "--rl-model-recipe", r.rl_model_recipe, "--tuning", t["tuning"],
             "--tensor-parallel", str(t["tensor_parallel"]),
             "--pipeline-parallel", str(t["pipeline_parallel"])]
    if rec.facts.moe:
        flags += ["--expert-parallel", str(t["expert_parallel"])]
    if r.rl_model_recipe == "kimi_k31":
        flags += ["--rl-sampler-engine", r.sampler]
        if t["tuning"] == "full":
            flags += ["--rl-expert-selection", "<expert_selection.json>"]
    if "needs:megatron_ref_load" in r.hard_constraints:
        flags += ["--megatron-ref-load", "<torch_dist dir>"]
    if "rollout_gpus_per_engine==8" in r.hard_constraints:
        flags += ["--rollout-num-gpus-per-engine", "8"]
    return flags


def as_dict(rec: Recommendation, model: str = "<model>") -> dict[str, Any]:
    f, r = rec.facts, rec.recipe
    return {
        "model": model, "family": rec.family, "supported": rec.supported,
        "facts": {"architectures": list(f.architectures), "model_type": f.model_type, "moe": f.moe,
                  "num_experts": f.num_experts, "topk": f.topk, "first_k_dense": f.first_k_dense,
                  "total_params_b": round(f.total_params_b, 2), "active_params_b": round(f.active_params_b, 2),
                  "attention": f.attention, "checkpoint_quant": f.checkpoint_quant, "layers": f.layers},
        "hardware": {"family": rec.hardware.device_family, "card": rec.hardware.card,
                     "hbm_gib": rec.hardware.hbm_gib, "cards_per_node": rec.hardware.cards_per_node},
        "recipe": None if r is None else {
            "name": r.name, "status": r.status, "trainer": r.trainer, "sampler": r.sampler,
            "rl_model_recipe": r.rl_model_recipe, "trainer_image": r.trainer_image,
            "sampler_image": r.sampler_image, "sampler_hardware": r.sampler_hardware,
            "quant": vars(r.quant), "evidence": r.evidence, "hard_constraints": list(r.hard_constraints)},
        "trainer_parallel": rec.trainer_parallel, "sampler_parallel": rec.sampler_parallel,
        "memory": rec.memory, "pitfalls": rec.pitfalls, "warnings": rec.warnings,
        "rejected": [list(x) for x in rec.rejected], "alternatives": rec.alternatives,
        "cli_flags": cli_flags(rec, model),
    }


def config_draft(rec: Recommendation, model: str = "<model>") -> dict[str, Any] | None:
    """A ``yeto run up`` run-config draft (yeto/rl/up.py); placeholders in <...>."""
    flags = cli_flags(rec, model)
    if not flags:
        return None
    return {
        "run": "<run>", "store": "s3://<bucket>/runs/<run>",
        "pool": {"config": "<pool.json>"},
        "trainer": {"argv": flags + ["--data", "<data>", "--reward-function", "<pkg.module:function>"]},
        "budget": {"usd": "<usd>", "wall_h": "<hours>"},
        "_registry": {"family": rec.family, "recipe": rec.recipe.name, "status": rec.recipe.status,
                      "trainer_image_pin": rec.recipe.trainer_image,
                      "sampler_image_pin": rec.recipe.sampler_image,
                      "sampler": {"engine": rec.recipe.sampler, **rec.sampler_parallel}},
    }


def render_text(rec: Recommendation, model: str = "<model>") -> str:
    d = as_dict(rec, model)
    f, out = d["facts"], []
    out.append(f"model      {model}  family={rec.family}  type={f['model_type']}  attention={f['attention']}")
    out.append(f"params     total {f['total_params_b']}B  active {f['active_params_b']}B  "
               f"moe={f['moe']} experts={f['num_experts']} topk={f['topk']} dense_first={f['first_k_dense']}  "
               f"ckpt={f['checkpoint_quant']}")
    out.append(f"hardware   {rec.hardware.device_family} {rec.hardware.card} {rec.hardware.hbm_gib} GiB "
               f"x {rec.hardware.cards_per_node}/node")
    r = d["recipe"]
    out.append(f"verdict    {'SUPPORTED' if rec.supported else 'NOT SUPPORTED'}"
               + (f"  recipe={r['name']} status={r['status']}" if r else ""))
    if r and r["rl_model_recipe"]:
        out.append(f"trainer    {r['trainer']}  image pin {r['trainer_image']}")
        out.append(f"sampler    {r['sampler']}"
                   + (f" on {r['sampler_hardware']}" if r["sampler_hardware"] else "")
                   + f"  image pin {r['sampler_image']}")
        out.append(f"quant      {r['quant']}")
        out.append(f"trainer // {json.dumps(d['trainer_parallel'])}")
        out.append(f"sampler // {json.dumps(d['sampler_parallel'])}")
        out.append(f"memory     {json.dumps(d['memory'])}")
    for title, items in (("warning", rec.warnings), ("pitfall", rec.pitfalls),
                         ("alternative", rec.alternatives)):
        out += [f"{title:<10} {x}" for x in items]
    out += [f"rejected   {n}: {why}" for n, why in rec.rejected]
    if d["cli_flags"]:
        out.append("flags      " + " ".join(d["cli_flags"]))
    out.append("(advisory; estimates from config.json shape, inferred rules -- smoke before a real run)")
    return "\n".join(out)
