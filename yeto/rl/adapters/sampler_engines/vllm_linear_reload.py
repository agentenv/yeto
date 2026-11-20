"""In-place reload of merged attention projections in a running vLLM server
(DeepSeek-V4.1-Flash on ROCm: FP8 dense layers with an e8m0 scale per 32 x 32
block, MXFP4 routed experts), loaded as a vLLM worker extension
(``--worker-extension-cls`` :data:`EXTENSION`) and called through
``POST /collective_rpc`` (``VLLM_SERVER_DEV_MODE=1``).

A LoRA release on this model trains attention only, and the sampler serves it
merged into the FP8 base (``lora_release_merge``: only ``layers.<i>.attn.{wq_a,
wkv, wq_b, wo_a, wo_b}.{weight, scale}`` change). vLLM's own ``reload_weights``
re-runs the loader over the whole checkpoint through its layerwise reload
(``model_loader/reload``), and on this model that breaks the served weights.
Measured on 8 x MI355X (vLLM 0.32.1rc1.dev36+ge68efc685, TP8, breakable CUDA
graphs), fingerprinting every parameter and buffer against a cold start:
``reload_weights`` of the unchanged base checkpoint leaves the MXFP4 routed
experts of all 40 layers (``w13`` / ``w2`` weights and scales, every rank) with
bytes a cold start does not have, attention and every other tensor equal; the
server then answers NaN log-probs. (The "Failed to load weights" warnings of the
rotary embeddings and the Engram hash state are harmless: those modules get
their served tensors back unchanged.) This extension touches only the modules a
release changes, with vLLM's own code for each:

* every released tensor must be an attention projection of the merge's naming
  (:data:`RELEASE_TENSOR`); a release with anything else is refused before any
  write, and the caller restarts instead;
* the target modules are the served ``fused_wqa_wkv`` (wq_a + wkv), ``wq_b``,
  ``wo_a`` and ``wo_b`` of the touched layers; each is staged whole: all of its
  checkpoint tensors are read from the release directory (the overlay also lists
  the unchanged ones), so a fused module always gets both halves;
* per module, the parameters it had at construction (vLLM's
  ``record_metadata_for_reloading``: checkpoint shape, dtype, ``weight_loader``)
  are materialized as zero tensors, the tensors go through the model's own
  ``load_weights`` (name mapping, fusion, TP narrowing, the per-row expansion of
  the block scale), then the module's own ``quant_method.process_weights_after_
  loading`` (the ROCm MXFP8 kernel compacts the scale back to blocks);
* the processed tensors are copied byte for byte (``uint8`` views) into the
  served tensors, which are put back: no parameter object is replaced, so
  captured CUDA graphs keep their pointers. A processed tensor whose name, shape
  or dtype differs from the served one, or a new tensor attribute, refuses the
  module before anything of it is written. The module's registry is restored
  exactly, ``None`` entries included (``bias=None`` is read by the forward).

Measured there: pause + reload of all 160 projections (40 layers) + resume takes
0.8-1.0 s (0.72 s on each rank, 955 MB written per rank), staging peaks at about
10 MB per GPU, and no GPU holds more memory afterwards. After base -> release v0
-> v1, v1 -> base and base -> v1, every one of the 1264 tensors per rank is
byte-equal to a cold start of the target; 32 requests in flight across a reload
all finish with finite log-probs.

Nothing else is read or written: routed experts, the shared expert, the Engram
tables and hash state, norms, rotary caches and the KV cache stay as they are.
``/pause`` clears the prefix cache (vLLM's default), so no KV of the old weights
is reused after ``/resume``.

:meth:`LinearReloadExtension.yeto_param_fingerprint` returns a per-tensor
fingerprint of the served model (the cold-vs-hot comparison).

This module must not import torch / vLLM at import time (CPU tests).
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .vllm_expert_reload import fingerprint, release_cached_memory

EXTENSION = "yeto.rl.adapters.sampler_engines.vllm_linear_reload.LinearReloadExtension"
RELOAD_METHOD = "yeto_reload_linears"
FINGERPRINT_METHOD = "yeto_param_fingerprint"
INDEX_FILE = "model.safetensors.index.json"
MERGE_RECORD = "yeto_lora_merge.json"  # lora_release_merge.MERGE_RECORD
RELEASE_PREFIX = "release-"  # release_overlay.RELEASE_PREFIX
RELEASE_TENSOR = re.compile(r"^layers\.(?P<layer>\d+)\.attn\.(?P<proj>wq_a|wkv|wq_b|wo_a|wo_b)\."
                            r"(?P<kind>weight|scale)$")
# checkpoint projection -> served module (DeepseekV4Model.load_weights stacked_params_mapping)
SERVED_MODULE = {"wq_a": "fused_wqa_wkv", "wkv": "fused_wqa_wkv", "wq_b": "wq_b", "wo_a": "wo_a", "wo_b": "wo_b"}
MODULE_PROJS = {"fused_wqa_wkv": ("wq_a", "wkv"), "wq_b": ("wq_b",), "wo_a": ("wo_a",), "wo_b": ("wo_b",)}
KINDS = ("weight", "scale")


class LinearReloadError(RuntimeError):
    pass


# ------------------------------------------------------------------ plan (CPU)
def released_tensors(weights_path: str | Path) -> list[str]:
    """Names of the release's own tensors in a merged overlay directory: those the
    index maps to a ``release-*`` file (``release_overlay``), else the merge
    record's list (``lora_release_merge``)."""
    root = Path(weights_path)
    index = dict(json.loads((root / INDEX_FILE).read_text())["weight_map"])
    names = sorted(name for name, file in index.items() if file.startswith(RELEASE_PREFIX))
    if names:
        return names
    record = root / MERGE_RECORD
    if record.is_file():
        return sorted(json.loads(record.read_text()).get("merged") or [])
    return []


def plan_modules(released: Iterable[str], weight_map: Mapping[str, str]) -> dict[tuple[int, str], list[str]]:
    """``(layer, served module) -> every checkpoint tensor of that module`` for the
    modules ``released`` touches. Refuses a released tensor that is not an
    attention projection, and a module whose checkpoint tensors are not all in
    ``weight_map`` (a fused module is staged whole)."""
    touched: set[tuple[int, str]] = set()
    for name in released:
        match = RELEASE_TENSOR.match(name)
        if match is None:
            raise LinearReloadError(f"{name} is not an attention projection; this path reloads "
                                    "layers.<i>.attn.{wq_a,wkv,wq_b,wo_a,wo_b} only")
        touched.add((int(match["layer"]), SERVED_MODULE[match["proj"]]))
    plan: dict[tuple[int, str], list[str]] = {}
    for layer, module in sorted(touched):
        names = [f"layers.{layer}.attn.{proj}.{kind}" for proj in MODULE_PROJS[module] for kind in KINDS]
        missing = [name for name in names if name not in weight_map]
        if missing:
            raise LinearReloadError(f"layer {layer} {module}: the release directory lacks {missing}")
        plan[(layer, module)] = names
    return plan


def _weight_map(directory: Path) -> dict[str, str]:
    return dict(json.loads((directory / INDEX_FILE).read_text())["weight_map"])


def overlay_names(model_dir: str | Path, base_dir: str | Path) -> list[str] | None:
    """The release tensors of ``model_dir`` when it is an overlay of ``base_dir``
    (every other tensor is the base's, in the base's own file), ``[]`` for the base
    itself, None for anything else."""
    model, base = Path(model_dir), Path(base_dir)
    if model.resolve() == base.resolve():
        return []
    try:
        index, base_map = _weight_map(model), _weight_map(base)
    except (OSError, ValueError, KeyError):
        return None
    released = sorted(name for name, file in index.items() if file.startswith(RELEASE_PREFIX))
    if not released or set(index) != set(base_map):
        return None
    files = set()
    for name, file in index.items():
        if file.startswith(RELEASE_PREFIX):
            continue
        if file != base_map[name]:
            return None
        files.add(file)
    for file in files:
        if (model / file).resolve() != (base / file).resolve():
            return None
    return released


def update_plan(model_dir: str | Path, served_dir: str | Path, base_dir: str | Path) -> tuple[list[str] | None, str]:
    """``(names, reason)``: the checkpoint tensors an in-place reload of ``model_dir``
    stages, given the server serves ``served_dir``: the new release's tensors plus
    the served release's (a module the served release changed and the new one does
    not goes back to the base), all attention projections. ``names`` is None and
    ``reason`` says why when the reload must restart instead."""
    new = overlay_names(model_dir, base_dir)
    if new is None:
        return None, "not_an_overlay"
    served = overlay_names(served_dir, base_dir)
    if served is None:
        return None, "served_not_an_overlay"
    names = sorted(set(new) | set(served))
    try:
        plan_modules(names, _weight_map(Path(model_dir)))
    except LinearReloadError:
        return None, "not_attention_only"
    return names, "attention_overlay"


def summarize(plan: Mapping[tuple[int, str], Sequence[str]]) -> dict[str, Any]:
    layers = sorted({layer for layer, _ in plan})
    return {"modules": len(plan), "layers": len(layers), "tensors": sum(len(v) for v in plan.values())}


# ------------------------------------------------------------------ on the worker
def _served_modules(model: Any) -> dict[tuple[int, str], Any]:
    found: dict[tuple[int, str], Any] = {}
    pattern = re.compile(r"(?:^|\.)layers\.(\d+)\.attn\.(fused_wqa_wkv|wq_b|wo_a|wo_b)$")
    for name, module in model.named_modules():
        match = pattern.search(name)
        if match is None:
            continue
        key = (int(match[1]), match[2])
        if key in found:
            raise LinearReloadError(f"two modules for layer {key[0]} {key[1]}")
        found[key] = module
    return found


def _tensor_attrs(module: Any, torch: Any) -> dict[str, Any]:
    """Plain tensor attributes (not parameters / buffers) a quant method may derive."""
    return {key: value for key, value in vars(module).items()
            if isinstance(value, torch.Tensor) and not key.startswith("_")}


def _u8(tensor: Any, torch: Any) -> Any:
    data = tensor.data if hasattr(tensor, "data") else tensor
    if not data.is_contiguous():
        raise LinearReloadError(f"tensor of shape {tuple(data.shape)} is not contiguous")
    return data.view(-1).view(torch.uint8)


def reload_module(model: Any, module: Any, tensors: Sequence[tuple[str, Any]], torch: Any) -> int:
    """Stage ``module`` in its checkpoint format, load ``tensors`` through the
    model's own loader, run the module's post-load processing and copy the result
    into the served tensors (module doc). Returns the bytes written."""
    from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info  # noqa: PLC0415
    from vllm.model_executor.model_loader.reload.meta import (  # noqa: PLC0415
        materialize_layer,
        restore_layer_on_meta,
    )
    from vllm.model_executor.model_loader.reload.utils import get_layer_params_buffers  # noqa: PLC0415

    info = get_layerwise_info(module)
    if not info.restore_metadata[0]:
        raise LinearReloadError(f"{type(module).__name__}: no recorded checkpoint metadata "
                                "(record_metadata_for_reloading)")
    if info.can_load():
        raise LinearReloadError(f"{type(module).__name__}: a layerwise reload is in progress")
    method = getattr(module, "quant_method", None)
    if method is None or not hasattr(method, "process_weights_after_loading"):
        raise LinearReloadError(f"{type(module).__name__}: no quant method to process the staged weights")
    saved = _snapshot(module, torch)
    served_params, served_buffers = get_layer_params_buffers(module)
    served_attrs = saved[3]
    try:
        restore_layer_on_meta(module, info)
        materialize_layer(module, info)
        for tensor in list(module._parameters.values()) + list(module._buffers.values()):
            if tensor is not None:
                _u8(tensor, torch).zero_()
        loaded = model.load_weights(tensors)
        if hasattr(module, "_already_called_process_weights_after_loading"):
            delattr(module, "_already_called_process_weights_after_loading")
        method.process_weights_after_loading(module)
        new_params, new_buffers = get_layer_params_buffers(module)
        new_attrs = _tensor_attrs(module, torch)
        pairs = []
        for kind, served, new in (("parameter", served_params, new_params), ("buffer", served_buffers, new_buffers),
                                  ("attribute", served_attrs, new_attrs)):
            if set(served) != set(new):
                raise LinearReloadError(f"{kind}s {sorted(served)} served, {sorted(new)} after processing")
            for name, old in served.items():
                fresh = new[name]
                if fresh is old:
                    continue
                if tuple(fresh.shape) != tuple(old.shape) or fresh.dtype != old.dtype:
                    raise LinearReloadError(f"{kind} {name}: processed {tuple(fresh.shape)} {fresh.dtype}, "
                                            f"served {tuple(old.shape)} {old.dtype}")
                pairs.append((old, fresh.contiguous()))
        if not loaded:
            raise LinearReloadError("load_weights loaded nothing")
    except BaseException:
        _put_back(module, saved)
        raise
    written = 0
    for old, fresh in pairs:
        _u8(old, torch).copy_(_u8(fresh, torch))
        written += fresh.numel() * fresh.element_size()
    _put_back(module, saved)
    return written


def _snapshot(module: Any, torch: Any) -> tuple[dict[str, Any], dict[str, Any], set[str], dict[str, Any]]:
    """The module's exact registry: parameters and buffers including ``None``
    entries (``bias=None`` is a registered parameter the forward reads), the
    non-persistent buffer names and the plain tensor attributes."""
    return (dict(module._parameters), dict(module._buffers), set(module._non_persistent_buffers_set),
            _tensor_attrs(module, torch))


def _put_back(module: Any, saved: tuple[dict[str, Any], dict[str, Any], set[str], dict[str, Any]]) -> None:
    """Restore the registry :func:`_snapshot` saved, objects and order unchanged."""
    params, buffers, non_persistent, attrs = saved
    module._parameters.clear()
    module._parameters.update(params)
    module._buffers.clear()
    module._buffers.update(buffers)
    module._non_persistent_buffers_set.clear()
    module._non_persistent_buffers_set.update(non_persistent)
    for name, value in attrs.items():
        object.__setattr__(module, name, value)


class LinearReloadExtension:
    """Mixed into vLLM's GPU worker (``--worker-extension-cls``); methods are
    called on every rank by ``collective_rpc`` (str arguments only)."""

    def yeto_reload_linears(self, weights_path: str, names: str | None = None) -> dict[str, Any]:
        """Reload the modules of ``names`` (a JSON list of checkpoint tensor names,
        :func:`update_plan`; default: the release tensors of ``weights_path``) from
        ``weights_path``."""
        import torch  # noqa: PLC0415
        from safetensors import safe_open  # noqa: PLC0415
        from vllm.config import set_current_vllm_config  # noqa: PLC0415

        began = time.perf_counter()
        root = Path(weights_path)
        weight_map = dict(json.loads((root / INDEX_FILE).read_text())["weight_map"])
        plan = plan_modules(json.loads(names) if names else released_tensors(root), weight_map)
        if not plan:
            return {**summarize(plan), "bytes": 0, "seconds": 0.0}
        model = self.model_runner.get_model()
        served = _served_modules(model)
        unknown = sorted(set(plan) - set(served))
        if unknown:
            raise LinearReloadError(f"modules {unknown[:4]} are not served on this rank")
        handles: dict[str, Any] = {}
        written = 0
        read_s = 0.0
        torch.cuda.reset_peak_memory_stats()
        base_alloc = torch.cuda.memory_allocated()
        try:
            with torch.no_grad(), set_current_vllm_config(self.vllm_config):
                for key, names in plan.items():
                    t0 = time.perf_counter()
                    named = []
                    for name in names:
                        file = weight_map[name]
                        if file not in handles:
                            handles[file] = safe_open(str(root / file), framework="pt", device="cpu")
                        named.append((name, handles[file].get_tensor(name)))
                    read_s += time.perf_counter() - t0
                    written += reload_module(model, served[key], named, torch)
            torch.cuda.synchronize()
        finally:
            handles.clear()
            staging_peak = torch.cuda.max_memory_allocated() - base_alloc
            release_cached_memory(torch)
        for reset in ("reset_encoder_cache", "reset_mm_cache"):
            hook = getattr(self.model_runner, reset, None)
            if callable(hook):
                hook()
        return {**summarize(plan), "bytes": written, "read_s": round(read_s, 3),
                "staging_peak_bytes": int(staging_peak), "seconds": round(time.perf_counter() - began, 3)}

    def yeto_param_fingerprint(self, pattern: str = "") -> dict[str, str]:
        """``name -> fingerprint`` of every parameter and buffer of the served model
        whose name matches ``pattern`` (all when empty), on this rank."""
        import torch  # noqa: PLC0415

        model = self.model_runner.get_model()
        regex = re.compile(pattern) if pattern else None
        out: dict[str, str] = {}
        with torch.no_grad():
            for kind, items in (("p", model.named_parameters()), ("b", model.named_buffers())):
                for name, tensor in items:
                    if regex is not None and not regex.search(name):
                        continue
                    out[f"{kind}:{name}"] = "meta" if tensor.is_meta else fingerprint(tensor)
        return out


__all__ = ["EXTENSION", "FINGERPRINT_METHOD", "LinearReloadError", "LinearReloadExtension", "RELOAD_METHOD",
           "plan_modules", "released_tensors", "reload_module", "summarize"]
