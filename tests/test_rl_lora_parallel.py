"""Canonical LoRA under tensor and pipeline parallelism (Miles state plugin).

The CPU tests check the two Megatron-Bridge adapter helpers of
``yeto.rl.adapters.miles.state_plugin``: binding names come from Bridge's
collective-free ``local_hf_param_specs`` (a PP stage never enters a PP
collective while it resolves only its own adapters), and apply takes each
rank's shard of the FP32 canonical tensor locally, in FP32 (Bridge's
``hf_to_megatron`` casts to the bf16 model dtype before its scatter).

The real-stack test spawns gloo ranks over Megatron-Core, Megatron-Bridge
``CanonicalLoRA`` and the state plugin (``tests/megatron_lora_roundtrip.py``)
when ``YETO_MEGATRON_PYTHON`` / ``_LM_PATH`` / ``_BRIDGE_PATH`` are set.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from yeto.rl.adapters.miles import state_plugin as sp

REPO = Path(__file__).resolve().parents[1]


class _Mapping:
    """The members of a Bridge ``ColumnParallelMapping`` / ``RowParallelMapping`` /
    ``ReplicatedMapping`` the helpers read. ``megatron_to_hf`` broadcasts over the
    PP group in Bridge, so binding resolution must not call it."""

    def __init__(self, hf_param: str, dim: int | None, tp_size: int = 1, tp_rank: int = 0):
        self.hf_param, self._dim, self.tp_size, self.tp_rank = hf_param, dim, tp_size, tp_rank
        self.ep_size = 1

    def local_hf_param_specs(self):
        return (SimpleNamespace(name=self.hf_param, split_count=1),)

    def _local_hf_sharding(self, weight, module):
        if self._dim is None:
            return "replicated", None, 1
        return "tp", self._dim, self.tp_size

    def megatron_to_hf(self, *_args):
        raise AssertionError("binding resolution entered a PP collective")


@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("tp_rank", [0, 1])
def test_apply_takes_the_rank_shard_in_fp32(dim, tp_rank):
    full = torch.randn(8, 6, dtype=torch.float32) * 1e-3 + 1.0
    local = torch.nn.Parameter(torch.zeros(full.chunk(2, dim=dim)[0].shape, dtype=torch.bfloat16))
    mapping = _Mapping("model.layers.0.self_attn.q_proj.lora_A.weight", dim, tp_size=2, tp_rank=tp_rank)

    shard = sp._bridge_local_shard(mapping, local, full)

    assert shard.dtype == torch.float32 and shard.shape == local.shape
    assert torch.equal(shard, full.chunk(2, dim=dim)[tp_rank])
    # the bf16 cast Bridge's scatter applies would lose these low bits
    assert not torch.equal(shard, shard.to(torch.bfloat16).float())


def test_apply_keeps_a_replicated_tensor_whole():
    full = torch.randn(4, 4)
    mapping = _Mapping("model.layers.0.mlp.experts.0.down_proj.lora_B.weight", None, tp_size=2, tp_rank=1)
    assert sp._bridge_local_shard(mapping, torch.zeros(4, 4), full) is full


def test_apply_refuses_a_tensor_that_does_not_split():
    mapping = _Mapping("x.lora_B.weight", 0, tp_size=2)
    with pytest.raises(sp.StatePluginError, match="does not split into 2 TP shards"):
        sp._bridge_local_shard(mapping, torch.zeros(3, 4), torch.zeros(7, 4))


def test_an_ambiguous_bridge_view_is_refused():
    mapping = SimpleNamespace(hf_param={"q": "a", "k": "b"},
                              local_hf_param_specs=lambda: (SimpleNamespace(name="a", split_count=2),))
    with pytest.raises(sp.StatePluginError, match="ambiguous LoRA mapping"):
        sp._bridge_hf_name(mapping)


def _actor_with_stage(monkeypatch, layers: range, *, tp_rank: int = 0):
    """One PP stage's adapters: its local bindings only, as Bridge's tasks give them."""

    model = torch.nn.Module()
    tasks = {}
    for layer in layers:
        a = torch.nn.Parameter(torch.zeros(4, 3, dtype=torch.bfloat16))  # row-parallel linear_in, TP 2
        b = torch.nn.Parameter(torch.zeros(16, 4, dtype=torch.bfloat16))  # replicated linear_out
        model.register_parameter(f"a{layer}", a)
        model.register_parameter(f"b{layer}", b)
        prefix = f"model.layers.{layer}.self_attn.o_proj"

        def side(param, name, dim):
            return SimpleNamespace(param_weight=param, param_name=name, megatron_module=None,
                                   mapping=_Mapping(name, dim, tp_size=2, tp_rank=tp_rank))

        tasks[prefix] = [SimpleNamespace(adapter_key=None,
                                         linear_in_task=side(a, f"{prefix}.lora_A.weight", 1),
                                         linear_out_task=side(b, f"{prefix}.lora_B.weight", None))]
    bridge = SimpleNamespace(_model_bridge=SimpleNamespace(build_adapter_conversion_tasks=lambda _m: tasks))
    hf_export = ModuleType("miles.backends.megatron_utils.hf_export")
    hf_export._get_hf_bridge = lambda _path: bridge
    monkeypatch.setitem(sys.modules, hf_export.__name__, hf_export)
    return SimpleNamespace(model=[model], args=SimpleNamespace(hf_checkpoint="/ckpt"))


def test_a_pipeline_stage_names_its_bindings_without_a_collective(monkeypatch):
    # the last stage of an uneven split holds two layers, the first stage one
    actor = _actor_with_stage(monkeypatch, range(1, 3), tp_rank=1)

    bindings = {b.name: b for b in sp.adapter_bindings(actor)}

    assert sorted(bindings) == [f"base_model.model.model.layers.{i}.self_attn.o_proj.lora_{s}.weight"
                                for i in (1, 2) for s in "AB"]
    full_a = torch.arange(24, dtype=torch.float32).reshape(4, 6) / 7
    a = bindings["base_model.model.model.layers.1.self_attn.o_proj.lora_A.weight"]
    assert torch.equal(a.from_hf(full_a), full_a[:, 3:])


class _ExpertBridge:
    """Bridge's per-expert name resolution: expert ``g`` of a grouped linear is
    ``<prefix>.weight<g>``; Qwen3-MoE names it ``model.layers.L.mlp.experts.g.*``."""

    def __init__(self, tasks, *, packed: bool = False):
        self._tasks, self._packed = tasks, packed

    def build_adapter_conversion_tasks(self, _model):
        return self._tasks

    def mapping_registry(self):
        return "registry"

    def _resolve_hf_adapter_param_name(self, registry, prefix, suffix, base_suffix, key):
        assert registry == "registry"
        layer = prefix.split(".")[2]
        expert = "" if self._packed else f".{base_suffix.removeprefix('.weight')}"
        proj = {"adapter_gate": "gate_proj", "adapter_up": "up_proj", None: "down_proj"}[key]
        side = "lora_A" if suffix == ".linear_in.weight" else "lora_B"
        return f"model.layers.{layer}.mlp.experts{expert}.{proj}.{side}.weight"


def _expert_actor(monkeypatch, *, ep_rank: int, num_local: int = 2, packed: bool = False, etp: int = 1):
    model = torch.nn.Module()
    prefix = "decoder.layers.0.mlp.experts.linear_fc2"
    a = torch.nn.Parameter(torch.zeros(num_local, 4, 8, dtype=torch.bfloat16))
    b = torch.nn.Parameter(torch.zeros(num_local, 6, 4, dtype=torch.bfloat16))
    model.a, model.b = a, b

    def side(param):
        mapping = SimpleNamespace(hf_param="unused", tp_size=etp, ep_rank=ep_rank, ep_size=2)
        return SimpleNamespace(param_weight=param, param_name="p", megatron_module=None, mapping=mapping)

    task = SimpleNamespace(global_base_prefix=prefix, adapter_key=None, requires_expert_splits=True,
                           linear_in_task=side(a), linear_out_task=side(b))
    bridge = SimpleNamespace(_model_bridge=_ExpertBridge({prefix: [task]}, packed=packed))
    hf_export = ModuleType("miles.backends.megatron_utils.hf_export")
    hf_export._get_hf_bridge = lambda _path: bridge
    monkeypatch.setitem(sys.modules, hf_export.__name__, hf_export)
    return SimpleNamespace(model=[model], args=SimpleNamespace(hf_checkpoint="/ckpt")), a, b


def test_grouped_expert_bindings_name_each_global_expert(monkeypatch):
    actor, a, b = _expert_actor(monkeypatch, ep_rank=1)

    bindings = sp.adapter_bindings(actor)

    names = [(b_.name.removeprefix("base_model.model.model.layers.0.mlp.experts."), b_.slot) for b_ in bindings]
    # EP rank 1 holds global experts 2 and 3
    assert names == [("2.down_proj.lora_A.weight", 0), ("2.down_proj.lora_B.weight", 0),
                     ("3.down_proj.lora_A.weight", 1), ("3.down_proj.lora_B.weight", 1)]
    assert {id(x.parameter) for x in bindings} == {id(a), id(b)}
    with pytest.raises(sp.StatePluginError, match="Bridge collective export"):
        bindings[0].to_hf(a)


def test_grouped_expert_apply_stacks_the_slots_in_fp32(monkeypatch):
    actor, a, _b = _expert_actor(monkeypatch, ep_rank=0)
    bindings = [x for x in sp.adapter_bindings(actor) if x.parameter is a]
    values = {x.name: torch.full((4, 8), 0.1 + x.slot, dtype=torch.float32) for x in bindings}

    targets = sp._apply_targets(bindings, values)

    assert targets[id(a)].dtype == torch.float32 and targets[id(a)].shape == a.shape
    assert torch.equal(targets[id(a)][1], values[bindings[1].name])
    with pytest.raises(sp.StatePluginError, match=r"expects \(4, 8\)"):
        bindings[0].from_hf(torch.zeros(8, 4))
    with pytest.raises(sp.StatePluginError, match="do not cover 2 experts"):
        sp._apply_targets(bindings[:1], values)


def test_grouped_expert_lora_refuses_stacked_names_and_expert_tp(monkeypatch):
    actor, _a, _b = _expert_actor(monkeypatch, ep_rank=0, packed=True)
    with pytest.raises(sp.StatePluginError, match="no per-expert HF names"):
        sp.adapter_bindings(actor)
    actor, _a, _b = _expert_actor(monkeypatch, ep_rank=0, etp=2)
    with pytest.raises(sp.StatePluginError, match="expert tensor parallel size 1"):
        sp.adapter_bindings(actor)


def test_sequential_expert_lora_under_ep_is_refused(monkeypatch):
    actor = _actor_with_stage(monkeypatch, range(1))
    (task,) = next(iter(_actor_tasks(actor).values()))
    task.global_base_prefix = "decoder.layers.0.mlp.experts.local_experts.0.linear_fc2"
    task.linear_in_task.mapping.ep_size = 2
    with pytest.raises(sp.StatePluginError, match="SequentialMLP experts"):
        sp.adapter_bindings(actor)


def _actor_tasks(actor):
    from miles.backends.megatron_utils.hf_export import _get_hf_bridge

    return _get_hf_bridge("/ckpt")._model_bridge.build_adapter_conversion_tasks(actor.model)


# ----------------------------------------------------------------------------- real Megatron-Core and Bridge
_REAL = {key: os.environ.get(key) for key in ("YETO_MEGATRON_PYTHON", "YETO_MEGATRON_LM_PATH",
                                              "YETO_MEGATRON_BRIDGE_PATH")}


def _spawn(tmp_path, tp: int, pp: int, ep: int, dp: int, *extra: str, timeout: int = 900):
    out = tmp_path / f"lora-{tp}-{pp}-{ep}-{dp}{''.join(extra)}.json"
    env = {**os.environ, "OMP_NUM_THREADS": "2"}
    proc = subprocess.run([_REAL["YETO_MEGATRON_PYTHON"], "-I", str(REPO / "tests/megatron_lora_roundtrip.py"),
                           str(tp), str(pp), str(ep), str(dp), str(out), *extra], env=env, capture_output=True,
                          text=True, timeout=timeout)
    return proc, out


def _run(tmp_path, tp: int, pp: int, ep: int, dp: int) -> dict:
    proc, out = _spawn(tmp_path, tp, pp, ep, dp)
    assert proc.returncode == 0, proc.stderr[-4000:]
    return json.loads(out.read_text())


@pytest.mark.skipif(not all(_REAL.values()), reason="needs YETO_MEGATRON_PYTHON / _LM_PATH / _BRIDGE_PATH")
def test_real_canonical_lora_round_trip_matches_across_tp_and_pp(tmp_path):
    """TP2, PP2 (uneven stages), TP2xPP2 and TP2xDP2 export the single-rank
    names and shapes, and apply-then-export is bit-exact FP32 on each."""

    reference = _run(tmp_path, 1, 1, 0, 1)
    assert reference["ok"] and reference["names"] == 3 * 7 * 2
    for tp, pp, dp in ((2, 1, 1), (1, 2, 1), (2, 2, 1), (2, 1, 2)):
        result = _run(tmp_path, tp, pp, 0, dp)
        assert result["ok"] and result["ranks"] == tp * pp * dp
        assert result["shapes"] == reference["shapes"], (tp, pp, dp)


@pytest.mark.skipif(not all(_REAL.values()), reason="needs YETO_MEGATRON_PYTHON / _LM_PATH / _BRIDGE_PATH")
def test_real_grouped_expert_lora_round_trip_matches_across_ep_tp_and_pp(tmp_path):
    """Grouped-expert LoRA (Bridge ``GroupedExpertLinearAdapter``, one pair per
    expert): EP2 alone and with TP2 / PP2 export every expert of every layer under
    its own PEFT name with the EP1 shapes, and apply-then-export is bit-exact."""

    reference = _run(tmp_path, 1, 1, 1, 1)
    experts = 3 * 4 * 6  # layers x experts x (gate, up, down) x (A, B)
    assert reference["ok"] and reference["grouped_adapters"] and reference["names"] == 3 * 4 * 2 + experts
    for tp, pp, ep, dp in ((1, 1, 2, 2), (2, 1, 2, 1), (1, 2, 2, 2), (2, 2, 2, 1)):
        result = _run(tmp_path, tp, pp, ep, dp)
        assert result["ok"] and result["grouped_adapters"], (tp, pp, ep, dp)
        assert result["shapes"] == reference["shapes"], (tp, pp, ep, dp)


@pytest.mark.skipif(not all(_REAL.values()), reason="needs YETO_MEGATRON_PYTHON / _LM_PATH / _BRIDGE_PATH")
def test_real_sequential_expert_lora_under_ep_is_refused(tmp_path):
    """Bridge exports SequentialMLP expert adapters without an EP gather, so the
    main rank would miss half the experts: the bindings refuse it by name."""

    proc, _out = _spawn(tmp_path, 1, 1, 2, 2, "sequential")
    assert proc.returncode != 0
    assert "SequentialMLP experts" in proc.stderr and "--moe-grouped-gemm" in proc.stderr
