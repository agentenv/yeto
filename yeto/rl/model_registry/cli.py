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


# ----------------------------------------------------------------------------- yeto model check
def add_model_parser(sub: Any) -> argparse.ArgumentParser:
    from .check import BACKENDS, FORMS, PRESETS

    p = sub.add_parser(
        "model", help="model onboarding: `yeto model check <model>` (reads config.json only)",
        description="Model onboarding tools. `check`: can this model train in this form, sync preset and "
                    "backend; per-item verdicts, the recommended DiLoCo settings and memory, the yeto launch "
                    "flags, and the templates a new model needs. Never downloads weights.")
    model_sub = p.add_subparsers(dest="model_command", metavar="{check}", required=True)
    c = model_sub.add_parser("check", help="check a model against a trainable form, sync preset and backend")
    c.add_argument("model", help="HF id, local model dir, or config.json path")
    c.add_argument("--revision", default=None, help="HF revision for the config fetch")
    c.add_argument("--no-fetch", action="store_true",
                   help="read config.json from disk or the local HF cache only (no network)")
    c.add_argument("--index", default=None, metavar="PATH|auto",
                   help="model.safetensors.index.json to verify the routed-expert names (auto: fetch it; "
                        "a few MB at most, never weights)")
    c.add_argument("--form", choices=FORMS, default="lora")
    c.add_argument("--preset", choices=PRESETS, default="decoupled")
    c.add_argument("--backend", choices=BACKENDS, default="miles")
    c.add_argument("--recipe", default=None, help="--rl-model-recipe (default: the model registry's family recipe)")
    c.add_argument("--tensor-parallel", type=_positive, default=1)
    c.add_argument("--pipeline-parallel", type=_positive, default=1)
    c.add_argument("--expert-parallel", type=_positive, default=None, help="default: recommended")
    c.add_argument("--lora-targets", choices=["attention", "all-linear"], default="attention")
    c.add_argument("--lora-rank", type=_positive, default=16)
    c.add_argument("--islands", type=_positive, default=2)
    c.add_argument("--gpus-per-island", type=_positive, default=8)
    c.add_argument("--card", default="H100", help="H100, H200, B200, B300, A100-80GB")
    c.add_argument("--seq-len", type=_positive, default=4096)
    c.add_argument("--json", action="store_true", help="print one JSON object")
    return p


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, not {text}")
    return value


def _known_recipes() -> list[str]:
    from yeto.rl.stage1 import _miles_recipes

    return _miles_recipes()


def _fetch(model: str, filename: str, revision: str | None) -> str:
    """One small file of an HF repo (config.json, the safetensors index), never weights."""

    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(model, filename, revision=revision)
    except Exception as exc:  # network, auth, proxy, missing repo: one line, exit 2
        raise FileNotFoundError(f"cannot fetch {filename} of {model!r}: {type(exc).__name__}: {exc}") from exc


def _check_config(args: argparse.Namespace) -> tuple[dict, str]:
    try:
        return load_config(args.model)
    except FileNotFoundError:
        if args.no_fetch:
            raise
    path = _fetch(args.model, "config.json", args.revision)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh), f"{args.model}@{args.revision or 'main'}:config.json"


def _weight_names(args: argparse.Namespace, source: str) -> list[str] | None:
    import os

    if args.index is None:
        return None
    path = args.index
    if path == "auto":
        local = os.path.join(os.path.dirname(source), "model.safetensors.index.json")
        path = local if os.path.isfile(local) else _fetch(args.model, "model.safetensors.index.json", args.revision)
    with open(path, encoding="utf-8") as fh:
        return sorted(json.load(fh)["weight_map"])


def cmd_model(args: argparse.Namespace) -> int:
    from .check import CheckRequest, check_model, render_text, report_dict

    if args.recipe is not None and args.recipe not in (known := _known_recipes()):
        print(f"yeto model check: --recipe {args.recipe} is not a --rl-model-recipe ({', '.join(known)})",
              file=sys.stderr)
        return 2
    try:
        cfg, source = _check_config(args)
        names = _weight_names(args, source)
    except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
        print(f"yeto model check: {exc}", file=sys.stderr)
        return 2
    request = CheckRequest(
        form=args.form, preset=args.preset, backend=args.backend, recipe=args.recipe,
        tensor_parallel=args.tensor_parallel, pipeline_parallel=args.pipeline_parallel,
        expert_parallel=args.expert_parallel, lora_targets=args.lora_targets, lora_rank=args.lora_rank,
        islands=args.islands, gpus_per_island=args.gpus_per_island, card=args.card, seq_len=args.seq_len)
    report = check_model(cfg, request, model=args.model, weight_names=names)
    if args.json:
        print(json.dumps({"config_source": source, **report_dict(report)}, indent=2, default=str))
    else:
        print(f"config   {source}")
        print(render_text(report))
    return 0 if report.runnable else 1
