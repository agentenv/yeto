"""In-place reload of trained MXFP4 routed experts in a running vLLM server
(Kimi-K3.1 on ROCm, the ``vllm/vllm-openai-rocm:kimi-k31`` image), loaded as a
vLLM worker extension (``--worker-extension-cls`` :data:`EXTENSION`) and called
through ``POST /collective_rpc`` (``VLLM_SERVER_DEV_MODE=1``).

Why not vLLM's own ``reload_weights`` (``model_loader/reload``, layerwise):

* it processes a layer only once every element of the layer was loaded; a
  partial-experts release (128 of 896 experts per layer) never completes a
  layer, so ``finalize_layerwise_processing`` runs the post-load transform on
  freshly materialized (uninitialised) tensors holding 128 loaded experts and
  768 garbage ones ("Delayed processing"), and copies that over the served
  weights;
* the AITER MXFP4 W4A16 transform (``convert_weight_to_mxfp4_moe_kernel_format``,
  backend ``AITER_MXFP4_BF16``) leaves the expert weights as
  ``torch.float4_e2m1fn_x2``, for which torch has no ``copy_`` kernel, so the
  copy back into the served storage raises; the layer is then left on its
  staging tensors and every later request fails.

What this extension does instead, per MoE layer the release touches (all of it
vLLM's own code except the expert-slot bookkeeping):

* the layer's checkpoint-format parameters as vLLM recorded them at
  construction (``record_metadata_for_reloading``: shape, dtype, the
  ``weight_loader`` of ``create_weights``) give zero-filled staging parameters
  with one row per released expert (``create_weights`` fills zeros too);
* the released tensors go through the model's own ``load_weights`` (name
  mapping, w1/w3 fusion, TP narrowing, padding, the MXFP4 scale encoding) into
  those staging parameters; only the expert id is moved to its staging row;
* the staged experts go through the same kernel-format conversion as the cold
  start (``convert_weight_to_mxfp4_moe_kernel_format`` with the arguments of
  ``Mxfp4MoEMethod._setup_kernel``); every step of it (gate/up
  de-interleave, ``shuffle_weight_a16w4``, ``shuffle_scale_a16w4``) is a
  permutation within one expert, so converting k experts gives the bytes the
  cold start gives those experts among all 896;
* the converted rows are written into the served parameters' storage
  (``index_copy_`` on ``uint8`` views: FP4 has no ``copy_``); no parameter is
  replaced, so captured graphs and the kernel's references stay valid.

Refused (``ExpertReloadError``, nothing written): an expert-parallel or
remapped expert placement, expert biases, any MXFP4 backend other than
``AITER_MXFP4_BF16`` on gfx950, a tensor that is not a routed expert of the
K3.1 naming. :meth:`K31ExpertReloadExtension.k31_param_fingerprint` returns a
per-parameter fingerprint of the served weights (the cold-vs-hot comparison).

This module must not import torch / vLLM at import time (CPU tests).
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

EXTENSION = "yeto.rl.adapters.sampler_engines.vllm_expert_reload.K31ExpertReloadExtension"
RELOAD_METHOD = "k31_reload_experts"
FINGERPRINT_METHOD = "k31_param_fingerprint"
RELEASE_CACHE_METHOD = "k31_release_cached_memory"
INDEX_FILE = "model.safetensors.index.json"
RELEASE_PREFIX = "release-"  # yeto.rl.engine.release_overlay
EXPERT_TENSOR = re.compile(r"^(?:language_model\.)?(?:model\.)?layers\.(?P<layer>\d+)\.block_sparse_moe\.experts\."
                           r"(?P<expert>\d+)\.(?P<proj>w[123])\.(?P<kind>weight_packed|weight_scale)$")
EXPERT_PARAMS = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
SUPPORTED_BACKENDS = frozenset({"AITER_MXFP4_BF16"})
FINGERPRINT_CHUNK = 1 << 26  # int64 elements per fingerprint step


class ExpertReloadError(RuntimeError):
    pass


# ------------------------------------------------------------------ plan (CPU)
def released_tensors(weights_path: str | Path, files: Sequence[str] | None = None) -> dict[str, str]:
    """``tensor -> file`` of the tensors to reload: those of ``files`` when given,
    else the overlay's ``release-*`` files (``release_overlay``)."""
    root = Path(weights_path)
    index = dict(json.loads((root / INDEX_FILE).read_text())["weight_map"])
    if files is not None:
        wanted = set(files)
        return {name: file for name, file in index.items() if file in wanted}
    return {name: file for name, file in index.items() if file.startswith(RELEASE_PREFIX)}


def group_by_layer(tensors: Mapping[str, str]) -> dict[int, dict[str, Any]]:
    """``layer -> {"experts": sorted ids, "tensors": {name: file}}``; every tensor
    must be a routed-expert tensor and every expert must come whole (the six
    tensors w1/w2/w3 x packed/scale)."""
    layers: dict[int, dict[str, Any]] = {}
    pieces: dict[tuple[int, int], set[tuple[str, str]]] = {}
    for name, file in tensors.items():
        match = EXPERT_TENSOR.match(name)
        if match is None:
            raise ExpertReloadError(f"{name} is not a routed-expert tensor; this path reloads experts only")
        layer, expert = int(match["layer"]), int(match["expert"])
        entry = layers.setdefault(layer, {"experts": set(), "tensors": {}})
        entry["experts"].add(expert)
        entry["tensors"][name] = file
        pieces.setdefault((layer, expert), set()).add((match["proj"], match["kind"]))
    whole = {(p, k) for p in ("w1", "w2", "w3") for k in ("weight_packed", "weight_scale")}
    partial = sorted(key for key, got in pieces.items() if got != whole)
    if partial:
        raise ExpertReloadError(f"{len(partial)} experts come partly, e.g. layer/expert {partial[0]}")
    return {layer: {"experts": sorted(entry["experts"]), "tensors": dict(sorted(entry["tensors"].items()))}
            for layer, entry in sorted(layers.items())}


# ------------------------------------------------------------------ on the worker
def _routed_experts_by_layer(model: Any) -> dict[int, tuple[str, Any]]:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts  # noqa: PLC0415

    found: dict[int, tuple[str, Any]] = {}
    for name, module in model.named_modules():
        if isinstance(module, RoutedExperts):
            match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
            if match is None:
                continue
            layer = int(match[1])
            if layer in found:
                raise ExpertReloadError(f"two routed-expert modules in layer {layer}: {found[layer][0]}, {name}")
            found[layer] = (name, module)
    return found


def _check_layer(module: Any, experts: Sequence[int]) -> None:
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend  # noqa: PLC0415
    from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod  # noqa: PLC0415

    method = module.quant_method
    if not isinstance(method, Mxfp4MoEMethod):
        raise ExpertReloadError(f"{type(method).__name__}: only the MXFP4 MoE method is supported")
    backend = getattr(method.mxfp4_backend, "name", str(method.mxfp4_backend))
    if backend not in SUPPORTED_BACKENDS or method.is_k3_situ_int4_gfx942:
        raise ExpertReloadError(f"MXFP4 backend {backend} is not supported (only {sorted(SUPPORTED_BACKENDS)})")
    from vllm.platforms import current_platform  # noqa: PLC0415

    if current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx1250  # noqa: PLC0415

        if on_gfx1250():
            raise ExpertReloadError("gfx1250 keeps AITER MXFP4 weights in the triton format; not supported")
    if method.mxfp4_backend == Mxfp4MoeBackend.NONE:
        raise ExpertReloadError("the layer has no MXFP4 kernel backend")
    if getattr(method.moe, "has_bias", False):
        raise ExpertReloadError("expert biases are not reloadable here")
    if getattr(module, "expert_substitution", None) is not None:
        raise ExpertReloadError("expert substitution is not supported")
    local = int(module.w13_weight.shape[0])
    for expert in experts:
        if module._map_global_expert_id_to_local_expert_id(expert) != expert or expert >= local:
            raise ExpertReloadError("experts are placed by EP / EPLB; only the identity placement is supported")


def _staging(module: Any, rows: int, slot_of: Mapping[int, int]) -> dict[str, Any]:
    """Zero staging parameters in the checkpoint format of ``create_weights``,
    ``rows`` experts deep, whose loader writes global expert e into row slot_of[e]."""
    from types import MethodType  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from vllm.model_executor.model_loader.reload.layerwise import LAYERWISE_INFO  # noqa: PLC0415
    from vllm.model_executor.model_loader.reload.sanitize import layer_ref_sentinel  # noqa: PLC0415

    info = LAYERWISE_INFO.get(module)
    params = info.restore_metadata[0] if info is not None else {}
    missing = [name for name in EXPERT_PARAMS if name not in params]
    if missing:
        raise ExpertReloadError(f"no recorded checkpoint metadata for {missing} (record_metadata_for_reloading)")
    device = module.w13_weight.device
    staged: dict[str, Any] = {}
    for name in EXPERT_PARAMS:
        meta = params[name]  # read only: vLLM's own reload keeps using it
        tensor = torch.zeros((rows, *meta.shape[1:]), dtype=meta.dtype, device=device)
        param = torch.nn.Parameter(tensor, requires_grad=False)
        for key, value in meta.__dict__.items():
            if isinstance(value, MethodType) and value.__self__ is layer_ref_sentinel:
                value = value.__func__.__get__(module)  # sanitize.restore_layer_refs, on a copy
            setattr(param, key, value)
        loader = param.weight_loader

        def slot_loader(param, loaded_weight, weight_name, shard_id, expert_id, return_success=False,
                        _loader=loader):
            return _loader(param, loaded_weight, weight_name, shard_id=shard_id, expert_id=slot_of[int(expert_id)],
                           return_success=return_success)

        param.weight_loader = slot_loader
        staged[name] = param
    return staged


def _convert(module: Any, staged: Mapping[str, Any]) -> tuple[Any, Any, Any, Any]:
    """The cold start's kernel-format conversion (``Mxfp4MoEMethod._setup_kernel``)."""
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (  # noqa: PLC0415
        convert_weight_to_mxfp4_moe_kernel_format,
    )

    method = module.quant_method
    w13, w2, s13, s2, _, _ = convert_weight_to_mxfp4_moe_kernel_format(
        mxfp4_backend=method.mxfp4_backend, layer=module,
        w13_weight=staged["w13_weight"], w2_weight=staged["w2_weight"],
        w13_weight_scale=staged["w13_weight_scale"], w2_weight_scale=staged["w2_weight_scale"],
        w13_bias=None, w2_bias=None, _cache_permute_indices=method._cache_permute_indices,
        activation=method.moe.activation, use_separated_a4w4=method.moe.use_mxfp4_w4a4_dsv4,
        activation_key=method.activation_key)
    return w13, s13, w2, s2


def _rows_u8(tensor: Any, rows: int) -> Any:
    import torch  # noqa: PLC0415

    data = tensor.data if hasattr(tensor, "data") else tensor
    if not data.is_contiguous():
        raise ExpertReloadError(f"tensor of shape {tuple(data.shape)} is not contiguous")
    flat = data.view(torch.uint8) if data.element_size() == 1 else data.view(-1).view(torch.uint8)
    if flat.numel() % rows:
        raise ExpertReloadError(f"{flat.numel()} bytes do not split into {rows} experts")
    return flat.reshape(rows, -1)


def reload_layer(model: Any, module: Any, experts: Sequence[int], tensors: Iterable[tuple[str, Any]]) -> int:
    """Reload ``experts`` of one routed-expert module from checkpoint ``tensors``;
    returns the bytes written into the served parameters."""
    import torch  # noqa: PLC0415

    _check_layer(module, experts)
    slot_of = {int(e): i for i, e in enumerate(experts)}
    staged = _staging(module, len(experts), slot_of)
    served = {name: module._parameters[name] for name in EXPERT_PARAMS}
    try:
        for name in EXPERT_PARAMS:
            module._parameters[name] = staged[name]
        model.load_weights(tensors)
        converted = _convert(module, {name: module._parameters[name] for name in EXPERT_PARAMS})
    finally:
        for name in EXPERT_PARAMS:
            module._parameters[name] = served[name]
    total = int(served["w13_weight"].shape[0])
    ids = torch.tensor(list(experts), dtype=torch.long, device=served["w13_weight"].device)
    written = 0
    for name, new in zip(EXPERT_PARAMS, converted):
        dst = _rows_u8(served[name], total)
        src = _rows_u8(new.contiguous(), len(experts))
        if src.shape[1] != dst.shape[1]:
            raise ExpertReloadError(f"{name}: {src.shape[1]} bytes per converted expert, "
                                    f"{dst.shape[1]} per served expert")
        dst.index_copy_(0, ids, src)
        written += src.numel()
    return written


def fingerprint(tensor: Any) -> str:
    """Order-sensitive 64-bit fingerprint of a tensor's bytes (equal bytes give
    equal fingerprints; computed on the tensor's device)."""
    import torch  # noqa: PLC0415

    data = tensor.data if hasattr(tensor, "data") else tensor
    flat = data.contiguous().view(-1).view(torch.uint8)
    pad = (-flat.numel()) % 8
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    words = flat.view(torch.int64)
    acc_sum = torch.zeros((), dtype=torch.int64, device=words.device)
    acc_xor = torch.zeros((), dtype=torch.int64, device=words.device)
    for start in range(0, words.numel(), FINGERPRINT_CHUNK):
        part = words[start:start + FINGERPRINT_CHUNK]
        weights = torch.arange(start, start + part.numel(), dtype=torch.int64, device=words.device)
        weights = weights * 6364136223846793005 + 1442695040888963407
        acc_sum += (part * weights).sum()
        acc_xor ^= torch.bitwise_xor(part, weights).sum()
    return f"{int(acc_sum) & (2**64 - 1):016x}{int(acc_xor) & (2**64 - 1):016x}:{data.numel()}"


def expert_chunks(entry: Mapping[str, Any], max_experts: int | None) -> list[tuple[list[int], dict[str, str]]]:
    """One layer's released experts (``group_by_layer`` entry) cut into chunks of at
    most ``max_experts`` (None / 0: one chunk), each with its own tensors: the
    in-place reload's memory guard stages one chunk at a time."""
    experts = list(entry["experts"])
    size = len(experts) if not max_experts or max_experts <= 0 else int(max_experts)
    out = []
    for i in range(0, len(experts), size):
        part = set(experts[i:i + size])
        tensors = {name: file for name, file in entry["tensors"].items()
                   if int(EXPERT_TENSOR.match(name)["expert"]) in part}
        out.append((sorted(part), tensors))
    return out


def release_cached_memory(torch: Any) -> None:
    """Hand the reload's freed staging / conversion blocks back to the device: they
    stay in the caching allocator otherwise, outside vLLM's budget, and the serving
    path then has that much less room to grow (a TP8 MI355X sampler serves at ~95%
    of HBM at --gpu-memory-utilization 0.88)."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


class K31ExpertReloadExtension:
    """Mixed into vLLM's GPU worker (``--worker-extension-cls``); methods are
    called on every rank by ``collective_rpc``. Only str arguments arrive over
    HTTP, so ``files`` is a JSON list."""

    def k31_release_cached_memory(self) -> dict[str, Any]:
        """Empty this rank's caching allocator (the reload memory guard's first step)."""
        import torch  # noqa: PLC0415

        release_cached_memory(torch)
        free, total = torch.cuda.mem_get_info()
        return {"free_bytes": int(free), "total_bytes": int(total)}

    def k31_reload_experts(self, weights_path: str, files: str | None = None,
                           max_experts: str | None = None) -> dict[str, Any]:
        import torch  # noqa: PLC0415
        from safetensors import safe_open  # noqa: PLC0415
        from vllm.config import set_current_vllm_config  # noqa: PLC0415

        began = time.perf_counter()
        wanted = json.loads(files) if files else None
        plan = group_by_layer(released_tensors(weights_path, wanted))
        if not plan:
            return {"layers": 0, "experts": 0, "bytes": 0, "seconds": 0.0}
        model = self.model_runner.get_model()
        modules = _routed_experts_by_layer(model)
        unknown = sorted(set(plan) - set(modules))
        if unknown:
            raise ExpertReloadError(f"layers {unknown[:4]} have no routed-expert module")
        for layer, entry in plan.items():  # check every layer before writing any
            _check_layer(modules[layer][1], entry["experts"])
        root = Path(weights_path)
        handles: dict[str, Any] = {}
        written = experts = 0
        read_s = 0.0
        try:
            with torch.no_grad(), set_current_vllm_config(self.vllm_config):
                chunk = int(max_experts) if max_experts else None
                for layer, entry in plan.items():
                    pieces = expert_chunks(entry, chunk)
                    for part, tensors in pieces:
                        t0 = time.perf_counter()
                        named = []
                        for name, file in tensors.items():
                            if file not in handles:
                                handles[file] = safe_open(str(root / file), framework="pt", device="cpu")
                            named.append((name, handles[file].get_tensor(name)))
                        read_s += time.perf_counter() - t0
                        written += reload_layer(model, modules[layer][1], part, named)
                        experts += len(part)
                        if len(pieces) > 1:
                            del named
                            release_cached_memory(torch)  # the chunk's staging goes back first
        finally:
            handles.clear()
            release_cached_memory(torch)
        for reset in ("reset_encoder_cache", "reset_mm_cache"):
            hook = getattr(self.model_runner, reset, None)
            if callable(hook):
                hook()
        return {"layers": len(plan), "experts": experts, "bytes": written, "max_experts": max_experts,
                "read_s": round(read_s, 3), "seconds": round(time.perf_counter() - began, 3)}

    def k31_param_fingerprint(self, pattern: str = "") -> dict[str, str]:
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
                    if tensor.is_meta:
                        out[f"{kind}:{name}"] = "meta"
                        continue
                    out[f"{kind}:{name}"] = fingerprint(tensor)
        return out
