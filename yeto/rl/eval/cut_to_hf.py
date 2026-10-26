"""Materialize a full-parameter Miles round cut as a Hugging Face model directory (S27 R-5).

The offline evaluator (:mod:`yeto.rl.eval.offline_cuts`) serves each cut with
SGLang, which loads a Hugging Face directory. A Miles cut shard
(:mod:`yeto.rl.adapters.miles.cut_plugin`, ``torch.save`` of a dict) keeps the
trainer's model copy under ``"adapter"``, keyed by Megatron parameter name.
This module turns that copy into ``model.safetensors`` next to the base
model's config and tokenizer files.

Scope (refused otherwise, never guessed):

* full-parameter cuts only (a LoRA cut holds adapters, not a model);
* TP = PP = EP = 1: the DP-rank-0 shard holds the whole model copy;
* TP/PP/EP > 1: every shard must carry the ``param_layout`` the writer records
  (:mod:`yeto.rl.engine.param_layout`: global names, TP split, expert or
  dense); one shard per (PP, TP, EP) coordinate is loaded and the slices are
  merged. A cut written without that record is refused; CP > 1 is refused;
* the HF tensor names produced must equal the base model's weight names.

Name conversion reuses Miles' ``megatron_to_hf`` (``_convert_to_hf_core`` plus
vocab padding removal), imported lazily; tests inject a converter.

Use from the CLI as ``--materialize yeto.rl.eval.cut_to_hf:materialize`` with
``YETO_EVAL_BASE_MODEL`` (base HF dir), ``YETO_EVAL_HF_OUT`` (cache dir) and
optionally ``YETO_EVAL_MODEL_NAME`` (Miles model family, default from the
base config's ``model_type``).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence

from yeto.rl.engine.cut import load_manifest
from yeto.rl.engine.param_layout import ParamLayoutError, merge_shards, required_coverage

DONE = "YETO-MATERIALIZED.json"
WEIGHT_FILE = "model.safetensors"
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")
_MEGATRON_PREFIX = "module.module."
_LORA_MARKERS = ("lora_", "adapter.")

# (args, megatron name, tensor) -> [(hf name, tensor), ...]
Converter = Callable[[Any, str, Any], Iterable[tuple[str, Any]]]


class MaterializeError(RuntimeError):
    pass


def megatron_args(config: Mapping[str, Any]) -> SimpleNamespace:
    """The Megatron-side fields Miles' converters read, from an HF ``config.json``."""
    heads = int(config["num_attention_heads"])
    return SimpleNamespace(
        hidden_size=int(config["hidden_size"]), num_attention_heads=heads,
        num_query_groups=int(config.get("num_key_value_heads") or heads),
        kv_channels=config.get("head_dim"), vocab_size=int(config["vocab_size"]),
        num_layers=int(config["num_hidden_layers"]))


def miles_converter(model_name: str) -> Converter:
    """Miles' per-family Megatron -> HF converter (needs ``miles`` importable)."""
    from miles.backends.megatron_utils.megatron_to_hf import _convert_to_hf_core
    from miles.backends.megatron_utils.megatron_to_hf.processors import remove_padding

    def convert(args: Any, name: str, tensor: Any) -> Iterable[tuple[str, Any]]:
        return _convert_to_hf_core(args, model_name, name, remove_padding(name, tensor, args.vocab_size))

    return convert


def canonical_name(name: str) -> str:
    """Megatron name as Miles' converters expect it (``module.module.<path>``)."""
    if name.startswith("chunk"):
        raise MaterializeError(f"{name}: virtual-pipeline chunks are not supported")
    while name.startswith("module."):
        name = name[len("module."):]
    return _MEGATRON_PREFIX + name


def _base_weight_names(base: Path) -> set[str] | None:
    index = base / "model.safetensors.index.json"
    if index.is_file():
        return set(json.loads(index.read_text())["weight_map"])
    single = base / WEIGHT_FILE
    if single.is_file():
        from safetensors import safe_open

        with safe_open(str(single), framework="pt") as fh:
            return set(fh.keys())
    return None


def _coord(f: Any) -> tuple[int, int, int, int]:
    c = f.coord
    return int(c.get("pp", 0)), int(c.get("tp", 0)), int(c.get("ep", 0)), int(c.get("dp", 0))


def _model_shards(cut_path: Path) -> list[Path]:
    """One shard per (PP, TP, EP) coordinate (the lowest DP rank): enough to hold every slice."""
    manifest = load_manifest(cut_path.parent, cut_path.name)
    if int((manifest.runtime.get("layout") or {}).get("cp", 1) or 1) != 1:
        raise MaterializeError(f"cut {manifest.cut_id}: CP > 1 is not supported")
    chosen: dict[tuple[int, int, int], Any] = {}
    for f in manifest.files:
        pp, tp, ep, dp = _coord(f)
        if (pp, tp, ep) not in chosen or dp < _coord(chosen[(pp, tp, ep)])[3]:
            chosen[(pp, tp, ep)] = f
    if not chosen:
        raise MaterializeError(f"cut {manifest.cut_id}: no shard in the manifest")
    return [cut_path / chosen[k].path for k in sorted(chosen)]


def _check_full_parameter(model: Mapping[str, Any]) -> None:
    lora = sorted(n for n in model if any(m in n for m in _LORA_MARKERS))
    if lora:
        raise MaterializeError(f"LoRA cut ({lora[0]} ...): materialize a full-parameter cut only")


def merged_model(shards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The whole Megatron model copy, by ``module.module.<global name>``, from cut shards."""
    for shard in shards:
        model = shard.get("adapter")
        if not isinstance(model, Mapping) or not model:
            raise MaterializeError("cut shard holds no model copy ('adapter')")
        _check_full_parameter(model)
    layouts = [shard.get("param_layout") for shard in shards]
    if len(shards) == 1 and layouts[0] is None:  # TP=PP=EP=1 cut from before the layout record
        return {canonical_name(n): t for n, t in shards[0]["adapter"].items()}
    for layout in layouts:
        if layout is None:
            raise MaterializeError("a TP/PP/EP-sharded cut without param_layout cannot be merged "
                                   "(written before the layout record)")
        if "error" in layout:
            raise MaterializeError(f"the writer could not record the param layout: {layout['error']}")
    missing = required_coverage(layouts)
    if missing:
        raise MaterializeError(f"cut shards do not cover {missing[:6]}")
    try:
        merged = merge_shards([(lay, shard["adapter"]) for lay, shard in zip(layouts, shards)])
    except ParamLayoutError as error:
        raise MaterializeError(str(error)) from error
    return {canonical_name(n): t for n, t in merged.items()}


def convert_model(model: Mapping[str, Any], args: Any, convert: Converter, *, dtype: Any = None) -> dict[str, Any]:
    """HF tensors from a whole Megatron model copy (names ``module.module.<global name>``)."""
    out: dict[str, Any] = {}
    for name, tensor in model.items():
        for hf_name, hf_tensor in convert(args, name, tensor):
            if hf_name in out:
                raise MaterializeError(f"two Megatron tensors map to {hf_name}")
            t = hf_tensor.detach().contiguous()
            out[hf_name] = t.to(dtype) if dtype is not None else t
    return out


def materialize_cut(cut_path: str | os.PathLike[str], *, base_model: str | os.PathLike[str],
                    out_root: str | os.PathLike[str], convert: Converter | None = None,
                    model_name: str | None = None, dtype: Any = None) -> str:
    """Write ``<out_root>/<cut_id>`` (config/tokenizer from ``base_model`` + converted weights).

    Reuses an existing directory whose marker names the same manifest. Returns its path."""
    import torch
    from safetensors.torch import save_file

    cut_path, base = Path(cut_path), Path(base_model)
    manifest_sha = json.loads((cut_path / "manifest.json").read_text())["manifest_sha256"]
    target = Path(out_root) / cut_path.name
    marker = target / DONE
    if marker.is_file() and json.loads(marker.read_text()).get("manifest_sha256") == manifest_sha:
        return str(target)

    config = json.loads((base / "config.json").read_text())
    if convert is None:
        convert = miles_converter(model_name or str(config["model_type"]))
    shards = [torch.load(p, map_location="cpu", weights_only=True) for p in _model_shards(cut_path)]
    if dtype is None:  # the base model's dtype (transformers 5 writes "dtype", 4.x "torch_dtype")
        name = config.get("dtype") or config.get("torch_dtype")
        dtype = getattr(torch, str(name), None) if name else None
    tensors = convert_model(merged_model(shards), megatron_args(config), convert, dtype=dtype)
    del shards
    expected = _base_weight_names(base)
    if expected is not None:
        if config.get("tie_word_embeddings"):
            expected.discard("lm_head.weight")
            tensors.pop("lm_head.weight", None)
        missing, extra = sorted(expected - set(tensors)), sorted(set(tensors) - expected)
        if missing or extra:
            raise MaterializeError(f"cut {cut_path.name}: HF names differ from the base model "
                                   f"(missing {missing[:5]}, extra {extra[:5]})")

    staging = target.with_name(f".{target.name}.tmp")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    for item in base.iterdir():
        if item.is_file() and not item.name.endswith(_WEIGHT_SUFFIXES) and item.name != "model.safetensors.index.json":
            shutil.copy2(item, staging / item.name)
    save_file(tensors, str(staging / WEIGHT_FILE), metadata={"format": "pt"})
    (staging / DONE).write_text(json.dumps({"cut_id": cut_path.name, "manifest_sha256": manifest_sha,
                                            "base_model": str(base), "tensors": len(tensors)}))
    shutil.rmtree(target, ignore_errors=True)
    os.replace(staging, target)
    return str(target)


def materialize(cut: Any) -> str:
    """``--materialize`` entry for :mod:`yeto.rl.eval.offline_cuts` (configured by environment)."""
    try:
        base, out = os.environ["YETO_EVAL_BASE_MODEL"], os.environ["YETO_EVAL_HF_OUT"]
    except KeyError as exc:
        raise MaterializeError(f"{exc.args[0]} is not set") from None
    return materialize_cut(cut.path, base_model=base, out_root=out,
                           model_name=os.environ.get("YETO_EVAL_MODEL_NAME"))
