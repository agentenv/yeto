"""In-place reload of MXFP4 routed experts on SGLang's SM100 FlashInfer path
(trtllm-gen, ``moe_runner_backend=flashinfer_mxfp4``: Kimi-K3 / K3.1 on
B200 / B300), installed in every SGLang process by ``sitecustomize`` when
:data:`ENV` is ``1`` (the ``sglang-server`` adapter sets it for MXFP4 on CUDA).

The problem (``sglang-miles-kimi-k31@35708f76``, unchanged on upstream main
2026-11-10): ``Mxfp4MoEMethod.process_weights_after_loading`` reorders the
w1/w3 rows, permutes every expert for the transposed-MMA epilogue
(``_get_flashinfer_mxfp4_device_permute_indices``), interleaves the block
scales (``nvfp4_block_scale_interleave``) and replaces the four expert
parameters with new ``Parameter`` objects that carry no ``weight_loader``.
A second ``load_weights`` (``update_weights_from_tensor`` /
``update_weights_from_disk``) dies on ``'Parameter' object has no attribute
'weight_loader'``; and ``end_weight_update`` / the disk path run
``process_weights_after_loading`` again, which would transform the
already-transformed bytes a second time. ``restore_weights_before_loading``
is only implemented for the Marlin backend (and it zeroes every expert, so
it is a full-reload contract, not a partial one).

The fix, per FusedMoE layer on that path, after its first post-load
transform:

* the four runtime parameters get a ``weight_loader`` again, the layer's own
  ``FusedMoE.weight_loader`` (so global -> local expert mapping, EP and EPLB
  placement stay SGLang's); the layer's ``_weight_loader_impl`` is shadowed
  so each (expert, matrix) lands in a small per-expert staging buffer in the
  checkpoint layout, initialised as ``create_weights`` initialises the
  parameters (zero bytes, UE8M0 one for the scales), through SGLang's own
  ``_weight_loader_impl`` (TP narrowing, padding);
* once an expert has all six pieces (w1, w3, w2 data and scales) it goes
  through the same per-expert transform as the cold start and is copied into
  the expert's slot of the runtime parameters (same storage: CUDA graphs and
  published addresses stay valid). The bytes equal a cold start on the new
  checkpoint because the cold transform is the same per-expert byte
  permutation of the same staged bytes;
* ``process_weights_after_loading`` on an armed layer does not transform
  again; it fails when an expert was only partly sent (a half-written expert
  would be served otherwise).

Partial updates are the point: a release that changes 128 of 896 experts per
layer sends only those experts, everything else keeps its bytes. Expert
biases (absent in K3 / K3.1 checkpoints) are not reloadable here and raise.

This module must not import SGLang at import time (CPU tests use fakes).
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
import threading
from types import ModuleType
from typing import Any

ENV = "YETO_SGLANG_MXFP4_INPLACE"
TARGET = "sglang.srt.layers.quantization.mxfp4"
UE8M0_ONE = 127  # mxfp4.py ``_UE8M0_ONE``: the scale ``create_weights`` fills with
EPILOGUE_TILE_M = 128  # mxfp4.py: ``epilogue_tile_m = 128`` of the trtllm-gen shuffle
SF_BLOCK = 32
# runtime parameter -> (staging kind, the shards that fill it)
PARAMS = {"w13_weight": ("w13", ("w1", "w3")), "w13_weight_scale": ("s13", ("w1", "w3")),
          "w2_weight": ("w2", ("w2",)), "w2_weight_scale": ("s2", ("w2",))}
PIECES = frozenset((kind, shard) for kind, shards in PARAMS.values() for shard in shards)

_lock = threading.Lock()


class PartialExpertUpdate(RuntimeError):
    """An expert reached the post-load step with only some of its matrices."""


def is_trtllm_sm100(method: Any) -> bool:
    """The branch of ``process_weights_after_loading`` this module handles."""
    if getattr(method, "use_marlin", False) and not getattr(method, "use_mega_moe", False):
        return False
    if getattr(method, "use_deep_gemm", False) or getattr(method, "use_mega_moe", False):
        return False
    return bool(getattr(method, "use_flashinfer", False)) and getattr(method, "_fi_kernel", None) == "trtllm_sm100"


def staging_shapes(method: Any) -> dict[str, tuple[int, int]]:
    """Per-expert checkpoint-layout shapes (``create_weights`` of the same method)."""
    inter, hidden = int(method.intermediate_size_per_partition), int(method.hidden_size)
    return {"w13": (2 * inter, hidden // 2), "s13": (2 * inter, hidden // SF_BLOCK),
            "w2": (hidden, inter // 2), "s2": (hidden, inter // SF_BLOCK)}


def _swap_every_two_rows(x: Any, axis: int) -> Any:
    """mxfp4.py ``swap_every_two_rows``."""
    shape = list(x.shape)
    axis = axis % len(shape)
    pairs = shape[:axis] + [shape[axis] // 2, 2] + shape[axis + 1:]
    return x.reshape(*pairs).flip(axis + 1).reshape(*shape)


def transform_expert(mx: ModuleType, layer: Any, staged: dict[str, Any]) -> dict[str, Any]:
    """One expert's checkpoint-layout bytes -> the trtllm-gen runtime bytes, the
    per-expert body of ``Mxfp4MoEMethod.process_weights_after_loading``
    (``staged`` values are ``[1, rows, cols]`` uint8)."""
    import torch  # noqa: PLC0415

    w13, s13, w2, s2 = staged["w13"], staged["s13"], staged["w2"], staged["s2"]
    if getattr(layer.moe_runner_config, "gate_up_interleaved", True):
        s13 = _swap_every_two_rows(s13, -2)
        w13 = _swap_every_two_rows(w13, -2)
    else:
        half = w13.shape[-2] // 2
        pair_idx = torch.empty(2 * half, dtype=torch.long)
        pair_idx[0::2] = torch.arange(half) + half
        pair_idx[1::2] = torch.arange(half)
        w13 = w13[..., pair_idx, :].contiguous()
        s13 = s13[..., pair_idx, :].contiguous()
    permute = mx._get_flashinfer_mxfp4_device_permute_indices
    w13 = w13[0].view(torch.uint8)
    s13 = s13[0].view(torch.uint8)
    w2 = w2[0].view(torch.uint8)
    s2 = s2[0].view(torch.uint8)
    out = {
        "w13_weight": w13[permute(w13, EPILOGUE_TILE_M)].contiguous(),
        "w13_weight_scale": mx.nvfp4_block_scale_interleave(
            s13[permute(s13, EPILOGUE_TILE_M, num_elts_per_sf=16)].contiguous()),
        "w2_weight": w2[permute(w2, EPILOGUE_TILE_M)].contiguous(),
        "w2_weight_scale": mx.nvfp4_block_scale_interleave(
            s2[permute(s2, EPILOGUE_TILE_M, num_elts_per_sf=16)].contiguous()),
    }
    return out


def _write_expert(mx: ModuleType, layer: Any, slot: int, staged: dict[str, Any]) -> None:
    import torch  # noqa: PLC0415

    for name, value in transform_expert(mx, layer, staged).items():
        target = getattr(layer, name).data[slot].view(torch.uint8)
        target.copy_(value.reshape(target.shape))


def _stage(mx: ModuleType, method: Any, layer: Any, original_impl: Any, *, param: Any, loaded_weight: Any,
           weight_name: str, shard_id: str, expert_id: int) -> None:
    """The layer's ``_weight_loader_impl`` once armed: ``expert_id`` is local here."""
    import torch  # noqa: PLC0415

    name = next((n for n in PARAMS if getattr(layer, n, None) is param), None)
    if name is None:
        raise RuntimeError(f"{type(layer).__name__} layer {getattr(layer, 'layer_id', '?')}: "
                           f"{weight_name} cannot be reloaded in place (only MXFP4 expert weights and scales)")
    kind, shards = PARAMS[name]
    if shard_id not in shards:
        raise RuntimeError(f"{weight_name}: shard {shard_id} does not belong to {name}")
    with _lock:
        pending = layer._yeto_mxfp4_pending
        entry = pending.setdefault(int(expert_id), {"tensors": {}, "pieces": set()})
        staged = entry["tensors"].get(kind)
        if staged is None:
            rows, cols = staging_shapes(method)[kind]
            fill = UE8M0_ONE if kind.startswith("s") else 0
            staged = torch.nn.Parameter(torch.full((1, rows, cols), fill, dtype=torch.uint8, device=param.device),
                                        requires_grad=False)
            if kind.startswith("s"):
                staged.quant_method = "group"  # create_weights: ``w*_weight_scale.quant_method = "group"``
            entry["tensors"][kind] = staged
        original_impl(param=staged, loaded_weight=loaded_weight, weight_name=weight_name, shard_id=shard_id,
                      expert_id=0)
        entry["pieces"].add((kind, shard_id))
        if entry["pieces"] == PIECES:
            _write_expert(mx, layer, int(expert_id), entry["tensors"])
            del pending[int(expert_id)]
            layer._yeto_mxfp4_reloaded += 1


def arm(mx: ModuleType, method: Any, layer: Any) -> None:
    """Make an already-transformed layer reloadable (module doc)."""
    original_impl = type(layer)._weight_loader_impl.__get__(layer)
    loader = type(layer).weight_loader.__get__(layer)

    def impl(param: Any, loaded_weight: Any, weight_name: str, shard_id: str, expert_id: int) -> None:
        _stage(mx, method, layer, original_impl, param=param, loaded_weight=loaded_weight,
               weight_name=weight_name, shard_id=shard_id, expert_id=expert_id)

    layer._weight_loader_impl = impl
    for name in PARAMS:
        getattr(layer, name).weight_loader = loader
    layer._yeto_mxfp4_pending = {}
    layer._yeto_mxfp4_reloaded = 0
    layer._yeto_mxfp4_armed = True


def finish(layer: Any) -> int:
    """The post-load step of an armed layer: no second transform; raise on a
    partly sent expert. Returns the experts written since the last call."""
    with _lock:
        pending = layer._yeto_mxfp4_pending
        if pending:
            detail = {e: sorted(map(list, v["pieces"])) for e, v in list(pending.items())[:4]}
            pending.clear()
            raise PartialExpertUpdate(f"layer {getattr(layer, 'layer_id', '?')}: {len(detail)}+ expert(s) were "
                                      f"sent incompletely (local slot -> pieces received): {detail}; the served "
                                      "weights of those experts are unchanged but the update is incomplete")
        written, layer._yeto_mxfp4_reloaded = layer._yeto_mxfp4_reloaded, 0
        return written


def install_on_module(mx: ModuleType) -> None:
    cls = getattr(mx, "Mxfp4MoEMethod", None)
    if cls is None:
        raise RuntimeError(f"{TARGET} has no Mxfp4MoEMethod")
    if getattr(cls, "_yeto_mxfp4_inplace", False):
        return
    for attr in ("_get_flashinfer_mxfp4_device_permute_indices",):
        if not hasattr(mx, attr):
            raise RuntimeError(f"{TARGET} has no {attr}: the pinned trtllm-gen transform changed")
    original = cls.process_weights_after_loading

    def process_weights_after_loading(self: Any, layer: Any) -> Any:
        if getattr(layer, "_yeto_mxfp4_armed", False):
            finish(layer)
            return None
        result = original(self, layer)
        if is_trtllm_sm100(self):
            arm(mx, self, layer)
        return result

    cls.process_weights_after_loading = process_weights_after_loading
    cls._yeto_mxfp4_inplace = True
    cls._yeto_mxfp4_original = original


class _Loader(importlib.abc.Loader):
    def __init__(self, wrapped: Any, installer: Any) -> None:
        self.wrapped, self.installer = wrapped, installer

    def create_module(self, spec: Any) -> Any:
        return self.wrapped.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self.wrapped.exec_module(module)
        self.installer(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname != TARGET:
            return None
        sys.meta_path.remove(self)
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot locate {TARGET}")
        spec.loader = _Loader(spec.loader, install_on_module)
        return spec


def install() -> None:
    """Patch the loaded module, or arm a one-shot import hook for it."""
    loaded = sys.modules.get(TARGET)
    if loaded is not None:
        install_on_module(loaded)
    elif not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


# ------------------------------------------------------------- rank-local file reads
# ``/update_weights_from_tensor`` with tensors in the body pickles and base64-encodes
# every byte once per rank (4-layer K3.1, 256 experts: 129 s). The file loader below
# takes only file names; every rank reads the changed expert tensors from its own
# disk (page cache) and runs the model's ``load_weights`` on them, the same naming
# path as the tensor update, the arming above doing the in-place write.
FILE_LOADER = f"{__name__}.load_files"  # the --custom-weight-loader / load_format value
FILE_PREFIX = "yeto-mxfp4-files."


def files_command_name(model_dir: str, files: list[str]) -> str:
    import base64  # noqa: PLC0415
    import json  # noqa: PLC0415

    payload = json.dumps({"dir": str(model_dir), "files": list(files)}, separators=(",", ":")).encode()
    return FILE_PREFIX + base64.urlsafe_b64encode(payload).decode("ascii")


def parse_files_command(name: str) -> dict[str, Any] | None:
    import base64  # noqa: PLC0415
    import json  # noqa: PLC0415

    if not name.startswith(FILE_PREFIX):
        return None
    return json.loads(base64.urlsafe_b64decode(name[len(FILE_PREFIX):].encode()))


def local_expert_filter(model: Any):
    """``(layer, expert) -> bool``: the expert has a slot on this rank (EP), so
    a rank reads only its own experts' bytes. None (read everything) when a layer
    places experts through EPLB metadata or the model shows no FusedMoE layers."""
    layers = {}
    for _, module in (model.named_modules() if hasattr(model, "named_modules") else ()):
        mapper = getattr(module, "_map_global_expert_id_to_local_expert_id", None)
        if mapper is None or getattr(module, "layer_id", None) is None:
            continue
        if getattr(module, "uses_expert_location", False):
            try:
                from sglang.srt.eplb.expert_location import get_global_expert_location_metadata  # noqa: PLC0415
            except ImportError:
                return None
            if get_global_expert_location_metadata() is not None:
                return None
        layers[int(module.layer_id)] = mapper
    if not layers:
        return None
    return lambda layer, expert: layer not in layers or layers[layer](expert) >= 0


def expert_tensors_of_files(model_dir: str, files: list[str], keep: Any = None):
    """``(name, CPU tensor)`` of every K3.1 routed-expert tensor in ``files``
    (only those ``keep(layer, expert)`` accepts, when given)."""
    from pathlib import Path  # noqa: PLC0415

    from safetensors import safe_open  # noqa: PLC0415

    from yeto.rl.adapters.sampler_engines.k31_release import parse_expert_tensor  # noqa: PLC0415

    for file in files:
        with safe_open(str(Path(model_dir) / file), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                parsed = parse_expert_tensor(name)
                if parsed is not None and (keep is None or keep(parsed[0], parsed[1])):
                    yield name, handle.get_tensor(name)


def load_files(model: Any, named_tensors: list[tuple[str, Any]]) -> None:
    """SGLang custom weight loader (``load_format`` = :data:`FILE_LOADER`)."""
    for name, _ in named_tensors:
        command = parse_files_command(name)
        if command is None:
            raise ValueError(f"{FILE_LOADER}: {name[:80]} is not a file command")
        model.load_weights(expert_tensors_of_files(command["dir"], command["files"], local_expert_filter(model)))
