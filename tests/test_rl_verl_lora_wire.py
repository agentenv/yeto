"""S19 6.4b async5 fix: LoRA config + phases over verl's checkpoint engine."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import torch

from yeto.rl.adapters.verl import lora_wire as lw

PEFT = {"r": 32, "lora_alpha": 32, "target_modules": {"q_proj", "v_proj"}, "task_type": "CAUSAL_LM"}


class _Cfg(dict):
    def __getattr__(self, k):
        return self[k]


class _Engine:
    def __init__(self, peft=PEFT, merge=False):
        self.model_config = SimpleNamespace(lora=_Cfg(merge=merge))
        self.peft = peft
        self.calls = []

    def get_per_tensor_param(self, layered_summon=False, base_sync_done=False, **_):
        self.calls.append(base_sync_done)
        if self.peft is None:
            return iter([("model.w", torch.ones(2))]), None
        if base_sync_done:
            return iter([("m.q_proj.lora_A.weight", torch.full((2, 3), 1.0)),
                         ("m.q_proj.lora_B.weight", torch.full((3, 2), 2.0))]), dict(self.peft)
        return iter([("m.q_proj.base_layer.weight", torch.zeros(3, 3))]), dict(self.peft)


def _worker(engine, load_format="safetensors"):
    return SimpleNamespace(actor=SimpleNamespace(engine=engine),
                           config=SimpleNamespace(rollout=_Cfg(load_format=load_format, layered_summon=False)))


async def _agen(items):
    for x in items:
        yield x


def _receive(stream):
    async def run():
        out = []
        async for sub, kw in lw.receive_phases(_agen(list(stream))):
            out.append((kw, [(n, t.clone()) async for n, t in sub]))
        return out
    return asyncio.run(run())


def test_plan_phases_follows_verl_base_sync_rule():
    assert lw.plan_phases(lora=False, merge=False, load_format="safetensors", base_sent=False) is None
    assert lw.plan_phases(lora=True, merge=True, load_format="dummy", base_sent=False) is None
    assert lw.plan_phases(lora=True, merge=False, load_format="safetensors", base_sent=False) == ["adapter"]
    assert lw.plan_phases(lora=True, merge=False, load_format="dummy", base_sent=False) == ["base", "adapter"]
    assert lw.plan_phases(lora=True, merge=False, load_format="dummy", base_sent=True) == ["adapter"]


def test_adapter_only_stream_reaches_rollout_with_peft_config_and_base_sync_done():
    worker = _worker(_Engine())
    phases = _receive(lw.sender_stream(worker, device="cpu"))
    assert len(phases) == 1
    kw, tensors = phases[0]
    assert kw["base_sync_done"] is True  # -> verl's add_lora branch
    assert kw["peft_config"]["r"] == 32 and kw["peft_config"]["target_modules"] == ["q_proj", "v_proj"]
    assert [n for n, _ in tensors] == ["m.q_proj.lora_A.weight", "m.q_proj.lora_B.weight"]
    assert torch.equal(tensors[1][1], torch.full((3, 2), 2.0))


def test_dummy_load_format_sends_base_once_then_adapter():
    engine = _Engine()
    worker = _worker(engine, load_format="dummy")
    first = _receive(lw.sender_stream(worker, device="cpu"))
    assert [kw["base_sync_done"] for kw, _ in first] == [False, True]
    assert [n for n, _ in first[0][1]] == ["m.q_proj.base_layer.weight"]
    second = _receive(lw.sender_stream(worker, device="cpu"))
    assert [kw["base_sync_done"] for kw, _ in second] == [True]


def test_no_lora_or_merged_lora_passes_through_unchanged():
    for engine in (_Engine(peft=None), _Engine(merge=True)):
        phases = _receive(lw.sender_stream(_worker(engine), device="cpu"))
        assert len(phases) == 1 and phases[0][0] == {}
        assert all(not n.startswith("__yeto") for n, _ in phases[0][1])


def test_header_round_trip():
    meta = {"peft_config": PEFT, "phases": ["adapter"]}
    back = lw.decode_header(lw.encode_header(meta, "cpu"))
    assert back["phases"] == ["adapter"] and back["peft_config"]["target_modules"] == ["q_proj", "v_proj"]


def test_marker_tensors_keep_later_tensors_aligned():
    """s19-verl64b-async6-20261010a: a 1113-byte header shifted the next fp32
    tensor to an odd bucket offset; markers are padded to ALIGN bytes."""
    stream = list(lw.sender_stream(_worker(_Engine()), device="cpu"))
    offset = 0
    for name, tensor in stream:
        assert offset % tensor.element_size() == 0, (name, offset)
        offset += tensor.numel() * tensor.element_size()
    assert stream[0][1].numel() % lw.ALIGN == 0 and stream[-1][1].numel() == lw.ALIGN
