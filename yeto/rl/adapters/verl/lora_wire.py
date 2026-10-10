"""LoRA over verl's disaggregated checkpoint engine (6.4b, S19 async5 fix).

The fork (acad9875) pushes weights trainer -> rollouter through
``checkpoint_engine.send_weights`` and drops the PEFT config on both ends
(``engine_workers.py`` ``per_tensor_param, _ = get_per_tensor_param()``;
``CheckpointEngineWorker.update_weights`` calls the rollout without it).  With
``lora.merge=False`` the rollout vLLM then loads ``...base_layer.weight``
names as plain weights and fails (s19-verl64b-async5-20261009a).  Only the
colocated ("naive") path passes ``peft_config``/``base_sync_done`` and so
reaches ``add_lora``.

This module carries both through the tensor stream itself (the checkpoint
engine moves any named CUDA tensor):

* first tensor ``HEADER``: uint8 JSON ``{"peft_config": ..., "phases": [...]}``;
* each phase ends with a ``PHASE_END`` marker tensor;
* phase ``"base"`` (only when the rollout came up on dummy weights, verl's own
  rule ``base_sync_done = "dummy" not in load_format``; sent once) is loaded
  with ``base_sync_done=False``; phase ``"adapter"`` with ``base_sync_done=True``,
  i.e. ``add_lora`` -- where yeto's read-back hook runs (F3).

Two build-time patches (``patch_verl.py``) call :func:`sender_stream` and
:func:`receive_phases`.  A stream without ``HEADER`` (no LoRA, or merged LoRA)
passes through unchanged.
"""

from __future__ import annotations

import json
from typing import Any

HEADER = "__yeto_lora_wire__"
PHASE_END = "__yeto_lora_phase_end__"


def encode_header(meta: dict, device) -> Any:
    import torch

    raw = json.dumps(meta, sort_keys=True, default=_jsonable).encode()
    return torch.tensor(list(raw), dtype=torch.uint8, device=device)


def decode_header(tensor) -> dict:
    return json.loads(bytes(tensor.detach().to("cpu").tolist()).decode())


def _jsonable(value):
    if isinstance(value, (set, frozenset, tuple)):
        return sorted(value) if isinstance(value, (set, frozenset)) else list(value)
    if hasattr(value, "value"):  # enums (peft TaskType)
        return value.value
    return str(value)


def plan_phases(*, lora: bool, merge: bool, load_format: str, base_sent: bool) -> list[str] | None:
    """None = plain stream (no wire header); otherwise the phases to send."""
    if not lora or merge:
        return None
    need_base = "dummy" in str(load_format) and not base_sent
    return (["base"] if need_base else []) + ["adapter"]


def _to_device(tensor, device):
    if hasattr(tensor, "full_tensor"):
        tensor = tensor.full_tensor()
    return tensor.detach().to(device).contiguous()


def sender_stream(worker, device=None):
    """Replacement for ``self.actor.engine.get_per_tensor_param()`` in
    ``ActorRolloutRefWorker.update_weights`` (checkpoint-engine branch)."""
    import torch

    engine = worker.actor.engine
    lora_cfg = engine.model_config.lora
    merge = bool(lora_cfg.get("merge", False))
    rollout_cfg = worker.config.rollout
    layered = bool(rollout_cfg.get("layered_summon", False))
    adapter, peft_config = engine.get_per_tensor_param(layered_summon=layered, base_sync_done=True)
    phases = plan_phases(lora=peft_config is not None, merge=merge,
                         load_format=rollout_cfg.load_format,
                         base_sent=getattr(worker, "_yeto_base_sent", False))
    if phases is None:
        per_tensor_param, _ = engine.get_per_tensor_param()
        return per_tensor_param
    base = None
    if "base" in phases:
        base, _ = engine.get_per_tensor_param(layered_summon=layered, base_sync_done=False)
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    worker._yeto_base_sent = True

    def gen():
        yield HEADER, encode_header({"peft_config": peft_config, "phases": phases}, device)
        for phase in phases:
            for name, tensor in (base if phase == "base" else adapter):
                yield name, _to_device(tensor, device)
            yield PHASE_END, torch.zeros(1, dtype=torch.uint8, device=device)

    return gen()


async def receive_phases(weights):
    """Split the received stream: yields ``(sub_stream, rollout_kwargs)`` per
    phase.  Each sub-stream must be consumed before the next one is taken."""
    it = weights.__aiter__()
    try:
        first = await it.__anext__()
    except StopAsyncIteration:
        return
    if first[0] != HEADER:
        async def plain():
            yield first
            async for item in it:
                yield item

        yield plain(), {}
        return
    meta = decode_header(first[1])

    async def sub():
        async for name, tensor in it:
            if name == PHASE_END:
                return
            yield name, tensor

    for phase in meta["phases"]:
        yield sub(), {"peft_config": meta["peft_config"], "base_sync_done": phase == "adapter"}
    async for _ in it:  # drain: lets the engine finish its receive loop
        pass
