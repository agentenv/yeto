"""The sparse-update probe on an SGLang-style Qwen2-MoE rank: stacked experts
(``w13``/``w2`` FusedMoE parameters), fused q/k/v and gate/up, TP slices,
expert parallelism and a transposed copy; the fast probe must give the same
element map as the exact one, element by element."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import torch

from yeto.rl.adapters.sampler_engines import sglang_sparse_loader as loader
from yeto.rl.adapters.sampler_engines import sparse_update


@dataclass(frozen=True)
class MoeConfig:
    layers: int = 2
    hidden: int = 32
    heads: int = 4
    kv_heads: int = 2
    head_dim: int = 8
    experts: int = 8
    moe_inter: int = 16
    shared_inter: int = 24
    vocab: int = 40
    tied: bool = False


def checkpoint_shapes(c: MoeConfig) -> dict[str, tuple[int, ...]]:
    """HF Qwen2-MoE checkpoint tensors."""
    out: dict[str, tuple[int, ...]] = {"model.embed_tokens.weight": (c.vocab, c.hidden),
                                       "model.norm.weight": (c.hidden,)}
    if not c.tied:
        out["lm_head.weight"] = (c.vocab, c.hidden)
    q, kv = c.heads * c.head_dim, c.kv_heads * c.head_dim
    for i in range(c.layers):
        p = f"model.layers.{i}."
        out.update({p + "input_layernorm.weight": (c.hidden,), p + "post_attention_layernorm.weight": (c.hidden,),
                    p + "self_attn.q_proj.weight": (q, c.hidden), p + "self_attn.q_proj.bias": (q,),
                    p + "self_attn.k_proj.weight": (kv, c.hidden), p + "self_attn.k_proj.bias": (kv,),
                    p + "self_attn.v_proj.weight": (kv, c.hidden), p + "self_attn.v_proj.bias": (kv,),
                    p + "self_attn.o_proj.weight": (c.hidden, q),
                    p + "mlp.gate.weight": (c.experts, c.hidden),
                    p + "mlp.shared_expert_gate.weight": (1, c.hidden),
                    p + "mlp.shared_expert.gate_proj.weight": (c.shared_inter, c.hidden),
                    p + "mlp.shared_expert.up_proj.weight": (c.shared_inter, c.hidden),
                    p + "mlp.shared_expert.down_proj.weight": (c.hidden, c.shared_inter)})
        for e in range(c.experts):
            for kind, shape in (("gate_proj", (c.moe_inter, c.hidden)), ("up_proj", (c.moe_inter, c.hidden)),
                                ("down_proj", (c.hidden, c.moe_inter))):
                out[p + f"mlp.experts.{e}.{kind}.weight"] = shape
    return out


class MoeRank(torch.nn.Module):
    """One TP rank (``ep``: experts split over the ranks instead of sliced) of an
    SGLang-style Qwen2-MoE, with its ``load_weights`` dispatch (stacked params
    mapping, FusedMoE weight loader). ``transpose_w2``: w2 stored transposed."""

    def __init__(self, c: MoeConfig, rank: int = 0, tp: int = 1, *, ep: bool = False,
                 transpose_w2: bool = False, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.c, self.rank, self.tp, self.ep, self.transpose_w2 = c, rank, tp, ep, transpose_w2
        q, kv = c.heads * c.head_dim // tp, c.kv_heads * c.head_dim // tp
        self.local_experts = c.experts // tp if ep else c.experts
        inter = c.moe_inter if ep else c.moe_inter // tp
        self._params: dict[str, torch.nn.Parameter] = {}

        def add(name: str, *shape: int) -> None:
            self._params[name] = torch.nn.Parameter(torch.zeros(*shape, dtype=dtype), requires_grad=False)

        vocab = -(-c.vocab // tp)
        add("model.embed_tokens.weight", vocab, c.hidden)
        if not c.tied:
            add("lm_head.weight", vocab, c.hidden)
        add("model.norm.weight", c.hidden)
        for i in range(c.layers):
            p = f"model.layers.{i}."
            add(p + "input_layernorm.weight", c.hidden)
            add(p + "post_attention_layernorm.weight", c.hidden)
            add(p + "self_attn.qkv_proj.weight", q + 2 * kv, c.hidden)
            add(p + "self_attn.qkv_proj.bias", q + 2 * kv)
            add(p + "self_attn.o_proj.weight", c.hidden, q)
            add(p + "mlp.gate.weight", c.experts, c.hidden)
            add(p + "mlp.shared_expert_gate.weight", 1, c.hidden)
            add(p + "mlp.shared_expert.gate_up_proj.weight", 2 * c.shared_inter // tp, c.hidden)
            add(p + "mlp.shared_expert.down_proj.weight", c.hidden, c.shared_inter // tp)
            add(p + "mlp.experts.w13_weight", self.local_experts, 2 * inter, c.hidden)
            if transpose_w2:
                add(p + "mlp.experts.w2_weight", self.local_experts, inter, c.hidden)
            else:
                add(p + "mlp.experts.w2_weight", self.local_experts, c.hidden, inter)
        for n, t in self._params.items():
            self.register_parameter(n.replace(".", "__"), t)

    def named_parameters(self, *args: Any, **kw: Any):
        yield from self._params.items()

    def _expert(self, param: torch.Tensor, w: torch.Tensor, expert: int, shard: str) -> None:
        if self.ep:
            per = self.local_experts
            if not self.rank * per <= expert < (self.rank + 1) * per:
                return
            local, sub = expert - self.rank * per, w
        else:
            local = expert
            dim = 0 if shard in ("w1", "w3") else 1
            size = w.shape[dim] // self.tp
            sub = w.narrow(dim, self.rank * size, size)
        dst = param.data[local]
        if shard == "w2":
            dst.copy_(sub.t() if self.transpose_w2 else sub)
        else:
            half = dst.shape[0] // 2
            dst.narrow(0, 0 if shard == "w1" else half, half).copy_(sub)

    def load_weights(self, weights) -> None:
        c, r, tp = self.c, self.rank, self.tp
        q, kv = c.heads * c.head_dim, c.kv_heads * c.head_dim
        stacked = [("qkv_proj", "q_proj", 0, q), ("qkv_proj", "k_proj", q // tp, kv),
                   ("qkv_proj", "v_proj", (q + kv) // tp, kv),
                   ("gate_up_proj", "gate_proj", 0, c.shared_inter),
                   ("gate_up_proj", "up_proj", c.shared_inter // tp, c.shared_inter)]
        experts = [(f"experts.{e}.{w}.", e, s) for e in range(c.experts)
                   for w, s in (("gate_proj", "w1"), ("down_proj", "w2"), ("up_proj", "w3"))]
        params = self._params
        for name, w in weights:
            for stacked_name, short, offset, full in stacked:
                if f".{short}." not in name or "mlp.experts." in name:
                    continue
                param = params[name.replace(short, stacked_name)]
                rows = full // tp
                param.data.narrow(0, offset, rows).copy_(w.narrow(0, r * rows, rows))
                break
            else:
                for key, e, shard in experts:
                    if key in name:
                        prefix = name.split("mlp.experts.")[0] + "mlp.experts."
                        self._expert(params[prefix + ("w2_weight" if shard == "w2" else "w13_weight")],
                                     w, e, shard)
                        break
                else:
                    if name.endswith("o_proj.weight") or name.endswith("shared_expert.down_proj.weight"):
                        cols = w.shape[1] // tp
                        params[name].data.copy_(w.narrow(1, r * cols, cols))
                    elif name in ("model.embed_tokens.weight", "lm_head.weight"):
                        per = params[name].shape[0]
                        rows = min(per, c.vocab - r * per)
                        params[name].data.narrow(0, 0, rows).copy_(w.narrow(0, r * per, rows))
                        if c.tied and name == "model.embed_tokens.weight" and "lm_head.weight" in params:
                            params["lm_head.weight"].data.copy_(params[name].data)
                    else:
                        params[name].data.copy_(w)


def _probe(model: torch.nn.Module, shapes: dict[str, tuple[int, ...]], tmp_path, mode: str,
           exact_groups: list[str] | None = None) -> dict[str, Any]:
    out, result = tmp_path / f"{mode}.pkl", tmp_path / f"{mode}.json"
    command = {"op": "prepare", "rank": 0, "out": str(out), "result": str(result), "probe": mode,
               "tensors": {n: list(s) for n, s in shapes.items()}}
    loader.load(model, [(loader.command_name(command), torch.zeros(1))])
    report = json.loads(result.read_text())
    assert report["ok"], report
    if mode == "fast":
        assert report["exact_groups"] == (exact_groups or []), report
    return sparse_update.load_map(out)


def _same_map(a: dict[str, Any], b: dict[str, Any]) -> None:
    assert a["params"] == b["params"]
    assert sorted(a["unmapped"]) == sorted(b["unmapped"])
    assert a["tensors"].keys() == b["tensors"].keys()
    for name, targets in a["tensors"].items():
        other = b["tensors"][name]
        assert len(targets) == len(other), name
        for x, y in zip(targets, other):
            assert x["param"] == y["param"] and x["dims"] == y["dims"], name
            assert {int(k): int(v) for k, v in x["select"].items()} == \
                {int(k): int(v) for k, v in y["select"].items()}, name
            assert list(x["maps"]) == list(y["maps"]), name
            for d in x["maps"]:
                assert np.array_equal(np.asarray(x["maps"][d]), np.asarray(y["maps"][d])), (name, d)


def _placed(model: MoeRank, mapping: dict[str, Any], shapes: dict[str, tuple[int, ...]]) -> None:
    """The map places every element where ``load_weights`` puts it (random values)."""
    g = torch.Generator().manual_seed(1)
    weights = {n: torch.randn(s, generator=g, dtype=torch.float32).to(torch.bfloat16) for n, s in shapes.items()}
    model.load_weights(weights.items())
    params = dict(model.named_parameters())
    for name, targets in mapping["tensors"].items():
        index = np.arange(int(np.prod(shapes[name])), dtype=np.int64)
        src = weights[name].reshape(-1).view(torch.int16).numpy()
        for target in targets:
            param = params[target["param"]]
            pos, valid = sparse_update.target_positions(target, shapes[name], param.shape, index)
            got = param.data.reshape(-1).view(torch.int16).numpy()[pos]
            assert np.array_equal(got, src[valid]), name


LAYOUTS = {
    "tp1": dict(tp=1), "tp2-rank1": dict(tp=2, rank=1), "tp4-rank3": dict(tp=4, rank=3),
    "ep2-rank1": dict(tp=2, rank=1, ep=True), "w2-transposed-tp2": dict(tp=2, rank=0, transpose_w2=True),
}


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
@pytest.mark.parametrize("tied", [False, True])
def test_fast_probe_gives_the_exact_probe_map_on_a_moe(tmp_path, layout, tied):
    c = MoeConfig(experts=12, vocab=42, tied=tied)
    shapes = checkpoint_shapes(c)
    kw = LAYOUTS[layout]
    fast = _probe(MoeRank(c, **kw), shapes, tmp_path, "fast")
    exact = _probe(MoeRank(c, **kw), shapes, tmp_path, "exact")
    _same_map(fast, exact)
    assert fast["unmapped"] == []
    _placed(MoeRank(c, **kw), fast, shapes)


class _OddRank(torch.nn.Module):
    """Placements the fast probe must not get wrong: an interleaved copy (strided
    rows), a tensor in the middle of rows (no seed at a row end), a tensor
    reshaped over two dimensions, a duplicated copy and a transformed one."""

    def __init__(self, scaled: bool) -> None:
        super().__init__()
        self.scaled = scaled
        z = lambda *s: torch.nn.Parameter(torch.zeros(*s, dtype=torch.bfloat16), requires_grad=False)  # noqa: E731
        self.p = {"model.layers.0.inter": z(8, 6), "model.layers.0.mid": z(4, 10),
                  "model.layers.1.reshaped": z(3, 4), "model.layers.1.dup": z(2, 5),
                  "model.layers.2.scaled": z(5), "model.layers.2.plain": z(5)}

    def named_parameters(self, *args: Any, **kw: Any):
        yield from self.p.items()

    def load_weights(self, weights) -> None:
        p = self.p
        for name, w in weights:
            if name == "model.layers.0.a":
                p["model.layers.0.inter"].data[0::2].copy_(w)
            elif name == "model.layers.0.b":
                p["model.layers.0.inter"].data[1::2].copy_(w.flip(0))
            elif name == "model.layers.0.m":
                p["model.layers.0.mid"].data[:, 3:7].copy_(w)
            elif name == "model.layers.1.r":
                p["model.layers.1.reshaped"].data.copy_(w.reshape(3, 4))
            elif name == "model.layers.1.d":
                p["model.layers.1.dup"].data[0].copy_(w)
                p["model.layers.1.dup"].data[1].copy_(w)
            elif name == "model.layers.2.s":
                p["model.layers.2.scaled"].data.copy_(w * 2 if self.scaled else w)
            elif name == "model.layers.2.t":
                p["model.layers.2.plain"].data.copy_(w)


@pytest.mark.parametrize("scaled", [False, True])
def test_fast_probe_falls_back_to_the_exact_probe_for_odd_placements(tmp_path, scaled):
    shapes = {"model.layers.0.a": (4, 6), "model.layers.0.b": (4, 6), "model.layers.0.m": (4, 4),
              "model.layers.1.r": (12,), "model.layers.1.d": (5,), "model.layers.2.s": (5,),
              "model.layers.2.t": (5,)}
    fast = _probe(_OddRank(scaled), shapes, tmp_path, "fast",
                  ["model.layers.1."] + (["model.layers.2."] if scaled else []))
    _same_map(fast, _probe(_OddRank(scaled), shapes, tmp_path, "exact"))
    assert sorted(fast["unmapped"]) == ["model.layers.1.d", "model.layers.1.r"] + \
        (["model.layers.2.s", "model.layers.2.t"] if scaled else [])
    assert [t["param"] for t in fast["tensors"]["model.layers.0.b"]] == ["model.layers.0.inter"]


def test_fast_probe_scales_with_the_expert_count(tmp_path):
    """A regression guard for the probe cost: 128 experts in well under the
    exact probe's time, and the time per element does not grow with the experts."""
    seconds = {}
    for experts in (16, 128):
        c = MoeConfig(layers=2, hidden=128, heads=4, kv_heads=2, head_dim=32, experts=experts,
                      moe_inter=64, shared_inter=256, vocab=1000)
        shapes = checkpoint_shapes(c)
        elements = sum(int(np.prod(s)) for s in shapes.values())
        began = time.monotonic()
        mapping = _probe(MoeRank(c), shapes, tmp_path, "fast")
        seconds[experts] = (time.monotonic() - began) / elements
        assert not mapping["unmapped"]
    assert seconds[128] < 4 * seconds[16] + 1e-7, seconds
    assert seconds[128] * elements < 20, seconds
