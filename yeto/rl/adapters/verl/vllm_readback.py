"""Runs inside the vLLM worker process (rl-verl-backend 2.2, design D5).

Called by the one-line hook the image adds after verl's ``add_lora`` in
``verl/workers/rollout/vllm_rollout/utils.py`` (``image.PATCHES``).  It reads
back the adapter vLLM *registered* (``LoRAModel`` held by the worker's LoRA
manager), maps its entries to canonical names (``param_names``) and writes the
bf16 checksum to ``$YETO_VERL_READBACK_DIR/readback-v<version>.json``; the
version is the one the island driver announced in ``expected_version`` just
before publishing (single-container island: shared filesystem).

Never raises into vLLM: failures are written as ``{"error": ...}`` and the
driver reports LORA_UNVERIFIABLE.
"""

from __future__ import annotations

import json
import os
import time

READBACK_ENV = "YETO_VERL_READBACK_DIR"


def _registered_lora_model(worker, lora_int_id: int):
    manager = worker.model_runner.lora_manager
    adapter_manager = getattr(manager, "_adapter_manager", manager)
    getter = getattr(adapter_manager, "get_adapter", None)
    model = getter(lora_int_id) if getter is not None else None
    if model is None:
        model = getattr(adapter_manager, "_registered_adapters", {}).get(lora_int_id)
    if model is None:
        raise LookupError(f"no registered LoRA with id {lora_int_id}")
    return model


def canonical_tensors_from_lora_model(lora_model) -> tuple[dict, dict]:
    """{canonical name: tensor} from a vLLM ``LoRAModel`` (+ a note dict)."""
    from .param_names import vllm_to_canonical

    out, transposed = {}, 0
    for module_name, layer in lora_model.loras.items():
        a, b = layer.lora_a, layer.lora_b
        if isinstance(a, (list, tuple)):
            pairs = [(i, a[i], b[i]) for i in range(len(a)) if a[i] is not None]
        else:
            pairs = [(None, a, b)]
        for slot, la, lb in pairs:
            out[vllm_to_canonical(module_name, "A", slot)] = la
            out[vllm_to_canonical(module_name, "B", slot)] = lb
    return out, {"entries": len(lora_model.loras), "transposed": transposed}


def after_add_lora(worker, lora_int_id: int, expected_shapes: dict | None = None) -> None:
    root = os.environ.get(READBACK_ENV)
    if not root:
        return
    t0 = time.time()
    try:
        version = open(os.path.join(root, "expected_version")).read().strip()
    except OSError:
        version = "unknown"
    path = os.path.join(root, f"readback-v{version}.json")
    if expected_shapes is None:
        try:
            expected_shapes = json.load(open(os.path.join(root, "expected_shapes.json")))
        except (OSError, ValueError):
            expected_shapes = None
    try:
        from .publish import lora_checksum

        tensors, note = canonical_tensors_from_lora_model(_registered_lora_model(worker, lora_int_id))
        if expected_shapes:
            # vLLM may keep lora_a as (in, rank) on some versions; record, do not guess silently.
            for name, tensor in list(tensors.items()):
                want = expected_shapes.get(name)
                if want and list(tensor.shape) != list(want) and list(tensor.shape)[::-1] == list(want):
                    tensors[name] = tensor.t()
                    note["transposed"] += 1
        result = lora_checksum(tensors)
        result["note"] = json.dumps(note, sort_keys=True)
        result["dtypes"] = sorted({str(t.dtype) for t in tensors.values()})
    except Exception as exc:  # noqa: BLE001 - reported, never raised into vLLM
        import traceback

        result = {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-2000:]}
    result.update({"version": version, "pid": os.getpid(), "seconds": round(time.time() - t0, 3)})
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(result, handle)
    os.replace(tmp, path)
