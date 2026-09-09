"""Compare checkpoint samples in the same FSDP rank-local parameter layout.

Use only between completed train/validation iterations, after checkpoint save.
Validation can retain gathered FSDP parameters; reshard before sampling and again
after reload so a global row is never compared to a different rank's local row.
Only live memory is perturbed. Reload must precede any further optimizer update
or checkpoint save. This is a sampled restoration check, not a full byte audit.
"""
from __future__ import annotations


def reshard_model_parts(model_parts):
    """Invoke public, non-recursive FSDPModule.reshard on every unique wrapper."""
    from torch.distributed.fsdp import FSDPModule

    if not isinstance(model_parts, (list, tuple)) or not model_parts:
        raise ValueError("Expected NeMo's nonempty model_parts list")
    seen = set()
    count = 0
    for part in model_parts:
        # Children first; the public operation is explicitly not recursive.
        for module in reversed(list(part.modules())):
            if id(module) not in seen and isinstance(module, FSDPModule):
                module.reshard()
                seen.add(id(module))
                count += 1
    return count


def _local(parameter):
    return parameter.to_local() if hasattr(parameter, "to_local") else parameter


def _layout(parameter, local):
    return {"global_shape": list(parameter.shape), "local_shape": list(local.shape),
            "local_stride": list(local.stride()), "dtype": str(local.dtype),
            "placements": [str(p) for p in getattr(parameter, "placements", ())]}


def perturb_model_state(model_parts, *, sample_values=8):
    """Normalize FSDP layout, snapshot and change one local trainable sample."""
    import torch

    if type(sample_values) is not int or not 1 <= sample_values <= 64:
        raise ValueError("Checkpoint probe sample count must be 1..64")
    wrappers = reshard_model_parts(model_parts)
    for part_index, part in enumerate(model_parts):
        for name, parameter in part.named_parameters():
            local = _local(parameter)
            if not parameter.requires_grad or not local.numel():
                continue
            if not local.is_contiguous() or not local.is_floating_point():
                raise RuntimeError("Model probe requires a contiguous floating point local parameter")
            sample = local.view(-1)[:sample_values]
            if not torch.isfinite(sample).all().item():
                raise RuntimeError("Model checkpoint sample is nonfinite")
            probe = {"part_index": part_index, "parameter_name": name,
                     "layout": _layout(parameter, local), "saved": sample.detach().clone(),
                     "fsdp_wrappers_resharded": wrappers}
            with torch.no_grad():
                sample.add_(1)
            if torch.equal(sample, probe["saved"]):
                raise RuntimeError("Model probe perturbation did not change the sampled state")
            return probe
    raise RuntimeError("No local trainable parameter available for checkpoint probe")


def verify_model_state_restored(model_parts, probe):
    """Re-resolve loaded parameters; require matching layout and exact values.

    The caller must all-reduce the result across every participating rank before
    accepting a checkpoint receipt. A missing reload or changed shard fails.
    """
    import torch

    wrappers = reshard_model_parts(model_parts)
    current_layout = None
    restored = False
    try:
        parameter = dict(model_parts[probe["part_index"]].named_parameters())[probe["parameter_name"]]
        local = _local(parameter)
        current_layout = _layout(parameter, local)
        saved = probe["saved"]
        restored = (current_layout == probe["layout"] and local.is_contiguous()
                    and local.numel() >= saved.numel()
                    and torch.equal(local.view(-1)[:saved.numel()], saved.to(local.device)))
    except (KeyError, IndexError, TypeError):
        restored = False
    return bool(restored), {"sampled_parameters": 1, "sampled_values": int(probe["saved"].numel()),
                            "parameter_name": probe["parameter_name"],
                            "saved_layout": probe["layout"], "restored_layout": current_layout,
                            "fsdp_wrappers_resharded_before": probe["fsdp_wrappers_resharded"],
                            "fsdp_wrappers_resharded_after": wrappers,
                            "all_model_bytes_compared": False}
