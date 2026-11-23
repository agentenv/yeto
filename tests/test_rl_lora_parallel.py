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


# ----------------------------------------------------------------------------- real Megatron-Core and Bridge
_REAL = {key: os.environ.get(key) for key in ("YETO_MEGATRON_PYTHON", "YETO_MEGATRON_LM_PATH",
                                              "YETO_MEGATRON_BRIDGE_PATH")}


def _run(tmp_path, tp: int, pp: int, ep: int, dp: int, *, timeout: int = 900) -> dict:
    out = tmp_path / f"lora-{tp}-{pp}-{ep}-{dp}.json"
    env = {**os.environ, "OMP_NUM_THREADS": "2"}
    proc = subprocess.run([_REAL["YETO_MEGATRON_PYTHON"], "-I", str(REPO / "tests/megatron_lora_roundtrip.py"),
                           str(tp), str(pp), str(ep), str(dp), str(out)], env=env, capture_output=True, text=True,
                          timeout=timeout)
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
