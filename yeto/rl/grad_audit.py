"""Pre-optimizer LoRA gradient audit for the legacy-vs-ports teacher-forcing check.

rl-engine-ports tier 2 (design D12) gates the teacher-forcing step on the LoRA
*gradient*, not on the Adam update: Adam's first step is ~``lr * sign(g)``, so
bf16 noise flips the sign of near-zero gradient elements and inflates the update
distance even when the gradients agree. Both engines therefore export, per
optimizer step, the gradient every LoRA tensor carries when ``optimizer.step()``
is entered:

* after the pipeline forward/backward and Megatron's ``finalize_model_grads``
  (DP all-reduce / reduce-scatter of ``main_grad`` has run; with
  ``--accumulate-allreduce-grads-in-fp32`` -- set on both paths -- ``main_grad``
  is the FP32 DDP buffer; ``param.grad`` is only a fallback);
* divided by the optimizer loss scale (1.0 for bf16 without a grad scaler);
* BEFORE ``clip_grad_norm``: the raw gradient. The clip coefficient Megatron
  will apply (``min(1, clip_grad / (||g|| + 1e-6))``) is recorded next to it,
  together with the grad norm ``optimizer.step()`` returns.

Scope: DP-replicated gradients only. Tensor/pipeline/expert model parallel and
a sharded distributed optimizer with DP > 1 are refused (each rank would hold a
partial gradient); the teacher-forcing runs use one GPU per island.

Files (next to the round audit ``round-<n>.{base,delta}.f32``, same
``<n> = rollout_id + 1``; optimizer steps after the first get ``.step-<k>``):

* ``round-<n>.grad.f32``  -- little-endian f32, tensors concatenated in sorted
  canonical PEFT name order, each in canonical HF shape;
* ``round-<n>.grad.json`` -- index (name/shape/offset/numel), sha256, norms,
  clip coefficient and provenance.

Enabled only when ``YETO_RL_AUDIT_GRADS=1``. Wiring:

* ports: ``state_plugin.install_grad_norm_recorder`` wraps upstream
  ``train_one_step`` and arms :func:`arm` for every step;
* legacy: ``yeto.rl.adapters.miles.island_entry`` sets Miles'
  ``--custom-megatron-before-train-step-hook-path`` to :data:`HOOK_PATH`
  (fork code is untouched; the hook exists in both Miles trees).

The directory comes from ``args.yeto_rl_grad_audit_dir`` (set by the learner
from the island's ``audit_dir``). torch / megatron are imported lazily.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GRAD_AUDIT_ENV = "YETO_RL_AUDIT_GRADS"
GRAD_AUDIT_DIR_ATTR = "yeto_rl_grad_audit_dir"
HOOK_PATH = "yeto.rl.grad_audit.before_train_step"
CANONICAL_PREFIX = "base_model.model."
LORA_SUFFIXES = (".lora_A.weight", ".lora_B.weight")
SCHEMA = 1
CLIP_EPS = 1.0e-6  # megatron.core.optimizer.clip_grads.clip_grad_by_total_norm_fp32


class GradAuditError(RuntimeError):
    pass


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return (os.environ if environ is None else environ).get(GRAD_AUDIT_ENV) == "1"


def canonical_grad_name(name: str) -> str:
    """Canonical PEFT name shared by both engines (``base_model.model.*.lora_[AB].weight``)."""

    name = name if name.startswith(CANONICAL_PREFIX) else CANONICAL_PREFIX + name
    # PEFT's in-memory adapter segment never appears in saved/canonical names.
    name = name.replace(".lora_A.default.weight", ".lora_A.weight").replace(
        ".lora_B.default.weight", ".lora_B.weight"
    )
    if not name.endswith(LORA_SUFFIXES):
        raise GradAuditError(f"not a canonical PEFT LoRA tensor name: {name!r}")
    return name


def stem(rollout_id: int, step_id: int) -> str:
    base = f"round-{int(rollout_id) + 1:08d}"
    return base if int(step_id) == 0 else f"{base}.step-{int(step_id):03d}"


@dataclass(frozen=True)
class GradBinding:
    """One trainable LoRA tensor: canonical name, registered Parameter, Megatron->HF layout."""

    name: str
    parameter: Any
    to_hf: Callable[[Any], Any]


# --------------------------------------------------------------------------
# Gradient capture (CPU-testable with plain torch modules)
# --------------------------------------------------------------------------


def raw_grad(parameter: Any) -> tuple[Any, str]:
    """The reduced gradient of one parameter: Megatron ``main_grad`` else ``.grad``."""

    grad = getattr(parameter, "main_grad", None)
    if grad is not None:
        return grad, "main_grad"
    if parameter.grad is not None:
        return parameter.grad, "grad"
    raise GradAuditError("LoRA parameter has no gradient at optimizer.step()")


def collect(bindings: Sequence[GradBinding], *, loss_scale: float = 1.0) -> tuple[dict[str, Any], dict[str, Any]]:
    """Canonical f32 CPU gradients (loss scale removed) and capture provenance."""

    import torch

    names = [b.name for b in bindings]
    if not names or len(names) != len(set(names)):
        raise GradAuditError("empty or duplicate canonical LoRA gradient names")
    out: dict[str, Any] = {}
    sources, dtypes = set(), set()
    with torch.no_grad():
        for b in bindings:
            grad, source = raw_grad(b.parameter)
            sources.add(source)
            dtypes.add(str(grad.dtype).replace("torch.", ""))
            value = b.to_hf(grad.detach().view(b.parameter.shape))
            value = value.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
            if loss_scale != 1.0:
                value.div_(float(loss_scale))
            if not torch.isfinite(value).all().item():
                raise GradAuditError(f"gradient of {b.name!r} contains NaN or Inf")
            out[b.name] = value
    return out, {"grad_source": sorted(sources), "grad_dtype": sorted(dtypes)}


def l2_norm(tensors: Iterable[Any]) -> float:
    import math

    return math.sqrt(sum(float(t.double().pow(2).sum().item()) for t in tensors))


def clip_coefficient(norm: float, clip_grad: float | None) -> float:
    """Megatron's ``clip_grad_by_total_norm_fp32`` coefficient (1.0 when not clipping)."""

    if not clip_grad or clip_grad <= 0:
        return 1.0
    coeff = float(clip_grad) / (float(norm) + CLIP_EPS)
    return coeff if coeff < 1.0 else 1.0


def write(directory: str | Path, name_stem: str, tensors: Mapping[str, Any], meta: Mapping[str, Any]) -> Path:
    """Atomically write ``<stem>.grad.f32`` (sorted names) and ``<stem>.grad.json``."""

    import torch

    root = Path(directory).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    data_path = root / f"{name_stem}.grad.f32"
    index_path = root / f"{name_stem}.grad.json"
    specs, offset = [], 0
    digest = hashlib.sha256()
    tmp = data_path.with_name(f".{data_path.name}.tmp-{os.getpid()}")
    try:
        with tmp.open("wb") as handle:
            for name in sorted(tensors):
                value = tensors[name].detach().to(device="cpu", dtype=torch.float32).contiguous()
                raw = value.numpy().astype("<f4", copy=False).tobytes()
                handle.write(raw)
                digest.update(raw)
                specs.append({"name": name, "shape": list(value.shape), "offset": offset,
                              "numel": int(value.numel())})
                offset += int(value.numel())
        os.replace(tmp, data_path)
    finally:
        tmp.unlink(missing_ok=True)
    index = {"schema": SCHEMA, **dict(meta), "semantics": "pre-clip, post-DP-reduce, loss-scale removed",
             "numel": offset, "specs": specs, "grad_f32_sha256": digest.hexdigest()}
    tmp = index_path.with_name(f".{index_path.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(json.dumps(index, sort_keys=True, indent=1) + "\n", encoding="utf-8")
        os.replace(tmp, index_path)
    finally:
        tmp.unlink(missing_ok=True)
    return data_path


def read(data_path: str | Path) -> tuple[dict[str, Any], Any]:
    """(index, flat little-endian f32 numpy array) of one ``*.grad.f32`` file."""

    import numpy as np

    data_path = Path(data_path)
    index = json.loads(data_path.with_name(data_path.name[: -len(".f32")] + ".json").read_text())
    flat = np.fromfile(str(data_path), dtype="<f4")
    if flat.size != index["numel"]:
        raise GradAuditError(f"{data_path}: {flat.size} values, index says {index['numel']}")
    return index, flat


# --------------------------------------------------------------------------
# Optimizer wrapping
# --------------------------------------------------------------------------


def _world(group_getter: str) -> int:
    try:
        from megatron.core import parallel_state as ps
    except ImportError:
        return 1
    try:
        return int(getattr(ps, group_getter)())
    except Exception:  # noqa: BLE001 - uninitialized parallel state
        return 1


def check_supported(args: Any, *, dp_world: int | None = None) -> None:
    for attr in ("tensor_model_parallel_size", "pipeline_model_parallel_size", "expert_model_parallel_size"):
        if int(getattr(args, attr, 1) or 1) > 1:
            raise GradAuditError(f"gradient audit needs {attr}=1 (per-rank gradients are partial)")
    dp = _world("get_data_parallel_world_size") if dp_world is None else dp_world
    if dp > 1 and getattr(args, "use_distributed_optimizer", False):
        raise GradAuditError("gradient audit needs DP=1 with --use-distributed-optimizer "
                             "(reduce-scattered main_grad is only valid on the local shard)")


def _is_writer() -> bool:
    try:
        import torch.distributed as dist
    except ImportError:  # pragma: no cover
        return True
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def _loss_scale(optimizer: Any) -> float:
    getter = getattr(optimizer, "get_loss_scale", None)
    if not callable(getter):
        return 1.0
    try:
        value = getter()
    except Exception:  # noqa: BLE001 - chained/stub optimizers
        return 1.0
    return float(value.item() if hasattr(value, "item") else value)


def arm(
    args: Any,
    rollout_id: int,
    step_id: int,
    optimizer: Any,
    bindings: Callable[[], Sequence[GradBinding]],
    *,
    engine: str,
    environ: Mapping[str, str] | None = None,
    is_writer: Callable[[], bool] = _is_writer,
    dp_world: int | None = None,
) -> bool:
    """Capture the LoRA gradients at the next ``optimizer.step()`` (one shot).

    Returns ``False`` (nothing armed) when the audit is disabled.
    """

    if not enabled(environ) or optimizer is None:
        return False
    directory = getattr(args, GRAD_AUDIT_DIR_ATTR, None)
    if not directory:
        raise GradAuditError(f"{GRAD_AUDIT_ENV}=1 but args.{GRAD_AUDIT_DIR_ATTR} is not set")
    check_supported(args, dp_world=dp_world)
    if getattr(optimizer, "_yeto_grad_audit_armed", False):
        return True
    original = optimizer.step
    writer = is_writer()

    def step(*a: Any, **kw: Any):
        optimizer.step = original
        optimizer._yeto_grad_audit_armed = False
        captured = None
        if writer:
            tensors, provenance = collect(bindings(), loss_scale=_loss_scale(optimizer))
            norm = l2_norm(tensors.values())
            clip_grad = getattr(args, "clip_grad", None)
            captured = (tensors, {
                "engine": engine, "rollout_id": int(rollout_id), "step_id": int(step_id),
                "grad_norm_l2": norm, "clip_grad": clip_grad,
                "clip_coefficient": clip_coefficient(norm, clip_grad),
                "loss_scale": _loss_scale(optimizer),
                "check_for_nan_in_loss_and_grad": getattr(args, "check_for_nan_in_loss_and_grad", None),
                **provenance,
            })
        result = original(*a, **kw)
        if captured is not None:
            tensors, meta = captured
            try:
                meta["optimizer_grad_norm"] = float(result[1].item() if hasattr(result[1], "item") else result[1])
            except (TypeError, IndexError, ValueError):
                meta["optimizer_grad_norm"] = None
            write(directory, stem(rollout_id, step_id), tensors, meta)
        return result

    optimizer.step = step
    optimizer._yeto_grad_audit_armed = True
    return True


# --------------------------------------------------------------------------
# Legacy entry point (Miles --custom-megatron-before-train-step-hook-path)
# --------------------------------------------------------------------------

_LEGACY_BINDINGS: dict[int, tuple[GradBinding, ...]] = {}


def legacy_bindings(args: Any, model: Sequence[Any]) -> tuple[GradBinding, ...]:
    """Megatron-Bridge adapter conversion tasks -> canonical names (same rule as the fork's export)."""

    key = id(model[0]) if model else 0
    cached = _LEGACY_BINDINGS.get(key)
    if cached is not None:
        return cached
    from megatron.bridge import AutoBridge

    # The run's own remote-code decision (the fork's export forces it; names do not depend on it).
    bridge = AutoBridge.from_hf_pretrained(
        args.hf_checkpoint, trust_remote_code=bool(getattr(args, "trust_remote_code", False))
    )
    build = getattr(getattr(bridge, "_model_bridge", None), "build_adapter_conversion_tasks", None)
    if build is None:
        raise GradAuditError("Megatron-Bridge lacks adapter conversion tasks")
    tasks_by_base = build(model)
    found = []
    for base in sorted(tasks_by_base):
        for task in sorted(tasks_by_base[base], key=lambda t: t.adapter_key or ""):
            for side in (task.linear_in_task, task.linear_out_task):
                if side.param_weight is None:
                    continue
                converted = side.mapping.megatron_to_hf(side.param_weight.detach(), side.megatron_module)
                if len(converted) != 1:
                    raise GradAuditError(f"ambiguous LoRA mapping for {side.param_name!r}")

                def to_hf(tensor, _side=side):
                    return next(iter(_side.mapping.megatron_to_hf(tensor, _side.megatron_module).values()))

                found.append(GradBinding(canonical_grad_name(next(iter(converted))), side.param_weight, to_hf))
    trainable = {id(p) for chunk in model for p in chunk.parameters() if p.requires_grad}
    if {id(b.parameter) for b in found} != trainable:
        raise GradAuditError("adapter conversion does not cover every trainable parameter")
    result = tuple(sorted(found, key=lambda b: b.name))
    _LEGACY_BINDINGS[key] = result
    return result


def before_train_step(args, rollout_id, step_id, model, optimizer, opt_param_scheduler) -> None:
    """Miles before-train-step hook (legacy path): arm the one-shot gradient capture."""

    del opt_param_scheduler
    arm(args, rollout_id, step_id, optimizer, lambda: legacy_bindings(args, model), engine="legacy")
