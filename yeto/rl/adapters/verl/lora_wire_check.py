"""CPU check of the patched verl LoRA push path, run inside the verl image
(Modal dry run, no GPU).  Prints one JSON line.

Drives the *patched* ``CheckpointEngineWorker.update_weights`` (fork
``checkpoint_engine/base.py``) with a stream built by :func:`lora_wire.sender_stream`
from a real ``peft.LoraConfig`` (as the FSDP engine returns it), forwards each
phase to the real ``vLLMColocateWorkerExtension._update_weights`` and records
whether it reached ``add_lora`` with a ``TensorLoRARequest`` that vLLM's
``PEFTHelper.from_dict`` accepts.  The vLLM engine itself (GPU) is faked.

Usage: python -m yeto.rl.adapters.verl.lora_wire_check
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import traceback
from types import SimpleNamespace


def run() -> dict:
    import torch
    from peft import LoraConfig

    from verl.checkpoint_engine.base import CheckpointEngineWorker
    from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension

    from . import lora_wire

    out: dict = {"patched_sender": False, "patched_receiver": False}
    import inspect

    import verl.workers.engine_workers as ew

    out["patched_sender"] = "_yeto_sender_stream(self)" in inspect.getsource(ew)
    out["patched_receiver"] = "_yeto_receive_phases(weights)" in inspect.getsource(CheckpointEngineWorker)

    peft = LoraConfig(r=32, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM").to_dict()
    names = ["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
             "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"]

    class Engine:
        model_config = SimpleNamespace(lora={"merge": False})

        def get_per_tensor_param(self, layered_summon=False, base_sync_done=False, **_):
            if base_sync_done:
                return iter([(names[0], torch.ones(32, 1024)), (names[1], torch.ones(1024, 32))]), dict(peft)
            return iter([("base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight",
                          torch.zeros(64, 64))]), dict(peft)

    class Rollout(dict):
        __getattr__ = dict.__getitem__

    trainer_worker = SimpleNamespace(actor=SimpleNamespace(engine=Engine()),
                                     config=SimpleNamespace(rollout=Rollout(load_format="safetensors",
                                                                            layered_summon=False)))
    added = []
    readback_dir = tempfile.mkdtemp()
    os.environ["YETO_VERL_READBACK_DIR"] = readback_dir
    with open(os.path.join(readback_dir, "expected_version"), "w") as f:
        f.write("7")

    ext = SimpleNamespace(add_lora=lambda req: added.append(req) or True)
    calls = []

    class Adapter:
        async def update_weights(self, weights, global_steps=None, wire_format="named_tensors", **kw):
            items = [(n, t) async for n, t in weights]
            calls.append({"names": [n for n, _ in items], "kw": sorted(kw), "base_sync_done": kw.get("base_sync_done")})
            vLLMColocateWorkerExtension._update_weights(ext, items, peft_config=kw.get("peft_config"),
                                                        base_sync_done=kw.get("base_sync_done", False))

    from verl.checkpoint_engine.base import merge_weight_chunks, split_weight_chunks

    bucket = 1 << 20

    class Engine2:
        """Packs the stream into one byte bucket the way the NCCL engine does
        (real split/merge helpers): catches dtype misalignment (async6)."""

        async def receive_weights(self, global_steps=None):
            buf = torch.empty(bucket, dtype=torch.uint8)

            async def chunks():
                offset, pending = 0, []
                async for meta, chunk in split_weight_chunks(
                        lora_wire.sender_stream(trainer_worker, device="cpu"), bucket):
                    assert offset + meta.chunk_size <= bucket, "check data must fit one bucket"
                    buf[offset:offset + meta.chunk_size] = chunk
                    pending.append((meta, offset))
                    offset += meta.chunk_size
                for meta, at in pending:
                    yield meta, buf[at:at + meta.chunk_size]

            async for item in merge_weight_chunks(chunks(), bucket):
                yield item

    fake_self = SimpleNamespace(checkpoint_engine=Engine2(), server_adapter=Adapter())
    fn = CheckpointEngineWorker.update_weights
    fn = getattr(fn, "__wrapped__", fn)
    asyncio.run(fn(fake_self, global_steps=3))
    out["rollout_calls"] = calls
    out["add_lora_calls"] = len(added)
    if added:
        req = added[0]
        out["lora_tensor_names"] = sorted(req.lora_tensors)
        out["values_ok"] = all(bool((t == 1).all()) and t.dtype == torch.float32
                               for t in req.lora_tensors.values())
        from vllm.lora.peft_helper import PEFTHelper

        helper = PEFTHelper.from_dict(req.peft_config)
        out["peft_helper"] = {"r": helper.r, "lora_alpha": helper.lora_alpha,
                              "target_modules": sorted(helper.target_modules)
                              if isinstance(helper.target_modules, (list, set)) else helper.target_modules}
    rb = os.path.join(readback_dir, "readback-v7.json")
    # the read-back hook ran (it reports an error here: the fake worker has no LoRA manager)
    out["readback_hook_ran"] = os.path.isfile(rb)
    out["pass"] = (out["patched_sender"] and out["patched_receiver"] and len(calls) == 1
                   and calls[0]["base_sync_done"] is True and out["add_lora_calls"] == 1
                   and out["lora_tensor_names"] == sorted(names) and out["peft_helper"]["r"] == 32 and out["values_ok"]
                   and out["readback_hook_ran"])
    return out


def main() -> None:
    try:
        out = run()
    except Exception as exc:  # noqa: BLE001 - reported, the dry run fails on it
        out = {"pass": False, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-3000:]}
    print("YETO_LORA_WIRE_CHECK " + json.dumps(out, default=str))


if __name__ == "__main__":
    main()
