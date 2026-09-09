"""Small rank-local optimizer corruption probes for checkpoint reload smokes.

Call only after a completed checkpoint, then reload before any optimizer step or
checkpoint save. This mutates live memory, never checkpoint files. It proves the
sampled state/step restoration; it does not claim to compare every optimizer byte.
"""
from __future__ import annotations

import copy


def _local_tensor(value):
    return value.to_local() if hasattr(value, "to_local") else value


def _resolve(optimizers, location):
    optimizer_index, group_index, parameter_index, scope, key = location
    optimizer = optimizers[optimizer_index]
    group = optimizer.param_groups[group_index]
    container = group if scope == "group" else optimizer.state[group["params"][parameter_index]]
    return container, key


def perturb_optimizer_state(optimizers, *, sample_values=8):
    """Snapshot and alter both Adam moments and a true step counter per optimizer.

    TE FusedAdam stores its step counter in the parameter group, while ordinary
    Adam stores it per parameter. DTensor moments are sampled via local shards;
    no full parameter/state gathering or state_dict materialization is needed.
    """
    import torch

    if type(sample_values) is not int or not 1 <= sample_values <= 64:
        raise ValueError("Checkpoint probe sample count must be 1..64")
    if not isinstance(optimizers, (list, tuple)) or not optimizers:
        raise ValueError("Expected NeMo's nonempty optimizer list")
    probes = []
    # Identify and snapshot every probe before mutating anything.
    for oi, optimizer in enumerate(optimizers):
        selected = None
        for gi, group in enumerate(optimizer.param_groups):
            for pi, parameter in enumerate(group["params"]):
                state = optimizer.state.get(parameter, {})
                if not {"exp_avg", "exp_avg_sq"} <= state.keys():
                    continue
                tensors = [_local_tensor(state[key]) for key in ("exp_avg", "exp_avg_sq")]
                if not all(isinstance(t, torch.Tensor) and t.numel() and t.is_contiguous() for t in tensors):
                    continue
                if "step" in group:
                    step_scope, step = "group", group["step"]
                elif "step" in state:
                    step_scope, step = "state", state["step"]
                else:
                    continue
                selected = (gi, pi, state, step_scope, step)
                break
            if selected:
                break
        if selected is None:
            raise RuntimeError("No initialized rank-local Adam moments and step counter to probe")
        gi, pi, state, scope, step = selected
        for key in ("exp_avg", "exp_avg_sq", "master_param"):
            if key not in state:
                continue
            local = _local_tensor(state[key])
            if not isinstance(local, torch.Tensor) or not local.numel() or not local.is_contiguous():
                raise RuntimeError("Optimizer state probe requires a nonempty contiguous local tensor")
            saved = local.view(-1)[:sample_values].detach().clone()
            probes.append({"location": (oi, gi, pi, "state", key), "saved": saved, "tensor": True})
        local_step = _local_tensor(step)
        if isinstance(local_step, torch.Tensor):
            if local_step.numel() != 1 or not local_step.is_contiguous() or local_step.item() <= 0:
                raise RuntimeError("Optimizer step counter is not an initialized scalar")
            saved_step = local_step.view(-1).detach().clone()
            probes.append({"location": (oi, gi, pi, scope, "step"), "saved": saved_step, "tensor": True})
        else:
            if type(step) not in {int, float} or step <= 0:
                raise RuntimeError("Optimizer step counter is not an initialized numeric value")
            probes.append({"location": (oi, gi, pi, scope, "step"), "saved": copy.deepcopy(step), "tensor": False})
    with torch.no_grad():
        for probe in probes:
            container, key = _resolve(optimizers, probe["location"])
            if probe["tensor"]:
                current = _local_tensor(container[key]).view(-1)[:probe["saved"].numel()]
                # Negation alone would not change zero-initialized moments.
                current.add_(7 if key == "step" else 1)
                if torch.equal(current, probe["saved"]):
                    raise RuntimeError("Optimizer probe perturbation did not change the sampled state")
            else:
                container[key] = probe["saved"] + 7
    return probes


def verify_optimizer_state_restored(optimizers, probes):
    """Resolve the current loaded state afresh, compare sampled bytes and counters.

    Returns local bool plus numeric coverage. Caller must all-reduce the bool
    across all eight ranks before writing a successful smoke receipt.
    """
    import torch

    if not probes:
        raise ValueError("No optimizer probes were recorded")
    restored = True
    for probe in probes:
        try:
            container, key = _resolve(optimizers, probe["location"])
            if probe["tensor"]:
                current = _local_tensor(container[key])
                saved = probe["saved"]
                matches = (isinstance(current, torch.Tensor) and current.is_contiguous()
                           and current.numel() >= saved.numel() and current.dtype == saved.dtype
                           and torch.equal(current.view(-1)[:saved.numel()], saved.to(current.device)))
            else:
                matches = type(container[key]) is type(probe["saved"]) and container[key] == probe["saved"]
        except (KeyError, IndexError, TypeError):
            matches = False
        restored = restored and bool(matches)
    return restored, {"optimizer_count": len({p["location"][0] for p in probes}),
                      "sampled_state_tensors": sum(p["tensor"] and p["location"][-1] != "step" for p in probes),
                      "step_counters": sum(p["location"][-1] == "step" for p in probes),
                      "all_optimizer_bytes_compared": False}
