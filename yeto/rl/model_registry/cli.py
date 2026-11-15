"""``yeto adapt <model> --hardware nvidia|ascend``: a dry run that reads config.json and
prints the registry's recommendation (and a run-config draft); and the ``--rl-model-auto``
launch hook, which prints the same advice and never changes an argument."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .match import facts_from_config, load_config
from .recommend import CARDS, hardware_facts, recommend
from .render import as_dict, config_draft, render_text
from .schema import Constraints


def add_parser(sub: Any) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "adapt", help="dry run: recommend backend, images, parallel layout for a model (reads config.json only)",
        description="Read a model's config.json and recommend trainer backend, sampler engine, image pins, "
                    "parallel layout, quantization and known pitfalls; print a run-config draft. Nothing launches.")
    p.add_argument("model", help="local model dir, config.json path, or HF id already in the local HF cache")
    p.add_argument("--hardware", required=True, choices=sorted(CARDS))
    p.add_argument("--card", default=None, help="trainer card (nvidia: H100/H200/B200/B300; ascend: 910B)")
    p.add_argument("--cards-per-node", type=int, default=8)
    p.add_argument("--max-nodes", type=int, default=64)
    p.add_argument("--interconnect", choices=["nvlink+ib", "roce", "gloo-only", "none"], default=None)
    p.add_argument("--tuning", choices=["lora", "full"], default=None)
    p.add_argument("--trainer-backend", choices=["miles", "verl"], default=None)
    p.add_argument("--sampler-engine", choices=["sglang", "vllm"], default=None)
    p.add_argument("--max-cards", type=int, default=None)
    p.add_argument("--json", action="store_true", help="print one JSON object (recommendation + draft)")
    p.add_argument("--write-config", default=None, metavar="PATH", help="write the run-config draft here")
    return p


def cmd_adapt(args: argparse.Namespace) -> int:
    try:
        cfg, source = load_config(args.model)
        hw = hardware_facts(args.hardware, args.card, cards_per_node=args.cards_per_node,
                            max_nodes=args.max_nodes, interconnect=args.interconnect)
        facts = facts_from_config(cfg)
    except (FileNotFoundError, ValueError) as exc:
        print(f"yeto adapt: {exc}", file=sys.stderr)
        return 2
    except (KeyError, TypeError, AttributeError) as exc:  # malformed config.json: not "unsupported" (1)
        print(f"yeto adapt: cannot read {source}: malformed config ({type(exc).__name__}: {exc})",
              file=sys.stderr)
        return 2
    rec = recommend(facts, hw, Constraints(tuning=args.tuning, sampler_engine=args.sampler_engine,
                                           trainer_backend=args.trainer_backend, max_cards=args.max_cards),
                    card_explicit=args.card is not None)
    draft = config_draft(rec, args.model)
    if args.json:
        print(json.dumps({"config_source": source, **as_dict(rec, args.model), "config_draft": draft}, indent=2))
    else:
        print(render_text(rec, args.model))
        if draft:
            print("\nrun-config draft (yeto run up):")
            print(json.dumps(draft, indent=2))
    if args.write_config and draft:
        try:
            with open(args.write_config, "w", encoding="utf-8") as fh:
                json.dump(draft, fh, indent=2)
        except OSError as exc:
            print(f"yeto adapt: cannot write --write-config {args.write_config}: {exc}", file=sys.stderr)
            return 2
    return 0 if rec.supported else 1


def advise_launch(args: Any, *, stream=None) -> None:
    """``--rl-model-auto``: print the recommendation for ``args.model``; read-only on
    ``args`` and never raises (a missing config.json is a printed note)."""
    stream = stream or sys.stderr
    try:
        cfg, _ = load_config(str(getattr(args, "model", "") or ""))
        hardware = "ascend" if "910" in str(getattr(args, "gpu", "") or "").upper() else "nvidia"
        rec = recommend(facts_from_config(cfg), hardware_facts(hardware),
                        Constraints(tuning=getattr(args, "tuning", None)), card_explicit=False)
        print("[rl-model-auto] " + render_text(rec, str(args.model)).replace("\n", "\n[rl-model-auto] "),
              file=stream)
        given = getattr(args, "rl_model_recipe", None)
        if rec.recipe and rec.recipe.rl_model_recipe and given and given != rec.recipe.rl_model_recipe:
            print(f"[rl-model-auto] WARNING: --rl-model-recipe {given} differs from the registry's "
                  f"{rec.recipe.rl_model_recipe}; the explicit flag is used", file=stream)
    except Exception as exc:  # advisory only: never block a launch
        print(f"[rl-model-auto] no recommendation: {exc}", file=stream)
