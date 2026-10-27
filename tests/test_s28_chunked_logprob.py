"""S28 FCE: chunked vocab-parallel log-probs (``yeto.rl.adapters.verl.chunked_logprob``).

Numerics on CPU, against "full logits, then log_softmax" (the quantity verl's
non-fused Megatron path computes):

* fp32: log-prob and entropy within 1e-5 absolute; gradients within 1e-5 of the
  reference's largest entry (the weight gradient is a sum over tokens, so its scale
  grows with T; the threshold scales with it);
* float64 ``gradcheck`` (log-prob and entropy together, and each alone: the backward
  takes the ``None`` gradient branches then);
* bf16: no worse than the current path. At temperature 1 both take the same bf16
  logits into fp32, so their log-probs agree to 1e-5 (measured: 1.9e-6). Otherwise
  each is held to the fp64 truth: the chunked error may not exceed the current path's
  (entropy: verl computes it in bf16, so the chunked one is closer);
* gloo TP2/TP4 (``tests/chunked_logprob_tp_worker.py``): vocabulary shards and the
  real all-reduces, forward and backward;
* memory: peak live tensor bytes (a dispatch mode that counts every storage created
  inside it) grow with ``chunk * V``, not ``T * V``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import weakref
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from torch.utils._python_dispatch import TorchDispatchMode

from yeto.rl.adapters.verl import chunked_logprob as cl

REPO = Path(__file__).resolve().parents[1]
FP32_ABS = 1e-5


def _inputs(T, H, V, dtype=torch.float32, scale=2.0, seed=0):
    gen = torch.Generator().manual_seed(seed)
    hidden = torch.randn(T, H, generator=gen, dtype=torch.float64)
    weight = torch.randn(V, H, generator=gen, dtype=torch.float64) * (scale / H ** 0.5)
    labels = torch.randint(0, V, (T,), generator=gen)
    return hidden.to(dtype), weight.to(dtype), labels


def _full(hidden, weight, labels, temperature):
    logp = torch.log_softmax((hidden @ weight.t()).to(torch.promote_types(hidden.dtype, torch.float32))
                             / temperature, -1)
    return logp.gather(1, labels.unsqueeze(-1)).squeeze(-1), -(logp.exp() * logp).sum(-1)


def _grads(fn, hidden, weight, labels, temperature, a, b):
    h = hidden.detach().clone().requires_grad_()
    w = weight.detach().clone().requires_grad_()
    lp, ent = fn(h, w, labels, temperature)
    (lp.double() * a).sum().add((ent.double() * b).sum()).backward()
    return lp.detach(), ent.detach(), h.grad, w.grad


def _chunked(chunk):
    return lambda h, w, y, t: cl.ChunkedVocabParallelLogprob.apply(h, w, y, t, chunk, None)


def _gap(a, b):
    return float((a.double() - b.double()).abs().max())


@pytest.mark.parametrize("chunk", [1, 7, 64, 300, 1000])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_fp32_matches_full_logits(chunk, temperature):
    hidden, weight, labels = _inputs(300, 32, 512)
    gen = torch.Generator().manual_seed(7)
    a, b = torch.randn(300, generator=gen, dtype=torch.float64), torch.randn(300, generator=gen, dtype=torch.float64)
    lp, ent, dh, dw = _grads(_chunked(chunk), hidden, weight, labels, temperature, a, b)
    rlp, rent, rdh, rdw = _grads(_full, hidden, weight, labels, temperature, a, b)
    assert lp.dtype == ent.dtype == torch.float32 and lp.shape == ent.shape == (300,)
    assert _gap(lp, rlp) <= FP32_ABS and _gap(ent, rent) <= FP32_ABS
    assert _gap(dh, rdh) <= FP32_ABS * max(1.0, float(rdh.abs().max()))
    assert _gap(dw, rdw) <= FP32_ABS * max(1.0, float(rdw.abs().max()))
    assert dh.dtype == dw.dtype == torch.float32


@pytest.mark.parametrize("outputs", ["both", "logprob", "entropy"])
def test_float64_gradcheck(outputs):
    hidden, weight, labels = _inputs(20, 8, 40, dtype=torch.float64, seed=3)
    pick = {"both": lambda r: r, "logprob": lambda r: r[0], "entropy": lambda r: r[1]}[outputs]
    fn = lambda h, w: pick(cl.ChunkedVocabParallelLogprob.apply(h, w, labels, 0.9, 7, None))
    assert torch.autograd.gradcheck(fn, (hidden.requires_grad_(), weight.requires_grad_()))


def _verl_like_bf16(h, w, labels, temperature):
    """What verl's non-fused path does with bf16: the output layer's bf16 logits, the
    temperature divided in bf16 (``logits.div_``), Megatron's cross entropy in fp32
    (``vocab_parallel_logits.float()``) and verl's ``_VocabParallelEntropy`` in bf16."""
    logits = (h @ w.t()) / temperature
    lp = torch.log_softmax(logits.float(), -1).gather(1, labels.unsqueeze(-1)).squeeze(-1)
    z = logits.detach()
    m = z.max(-1, keepdim=True).values
    e = (z - m).exp()
    s = e.sum(-1, keepdim=True)
    ent = (m + s.log() - ((e / s) * z).sum(-1, keepdim=True)).squeeze(-1).float()
    return lp, ent


@pytest.mark.parametrize("T,H,V,scale,temperature", [(1024, 256, 4096, 1.0, 1.0), (1024, 256, 4096, 3.0, 1.0),
                                                       (1024, 512, 8192, 2.0, 0.7)])
def test_bf16_is_no_worse_than_the_current_path(T, H, V, scale, temperature):
    hidden, weight, labels = _inputs(T, H, V, dtype=torch.bfloat16, scale=scale, seed=11)
    a = torch.randn(T, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    zero = torch.zeros(T, dtype=torch.float64)
    clp, cent, cdh, cdw = _grads(_chunked(128), hidden, weight, labels, temperature, a, zero)
    vlp, vent, vdh, vdw = _grads(_verl_like_bf16, hidden, weight, labels, temperature, a, zero)
    tlp, tent, tdh, tdw = _grads(_full, hidden.double(), weight.double(), labels, temperature, a, zero)

    def rel(x, ref):
        return float((x.double() - ref.double()).norm() / ref.double().norm())

    if temperature == 1.0:
        assert _gap(clp, vlp) <= 1e-5
    assert _gap(clp, tlp) <= _gap(vlp, tlp) + 1e-5
    assert _gap(cent, tent) <= _gap(vent, tent) + 1e-5
    # the weight gradient is formed chunk by chunk from bf16 GEMMs (each partial rounded
    # to bf16 once), the current path in one GEMM: measured ratio up to 1.22
    assert rel(cdh, tdh) <= 1.5 * rel(vdh, tdh)
    assert rel(cdw, tdw) <= 1.5 * rel(vdw, tdw)
    assert cdh.dtype == cdw.dtype == torch.bfloat16


@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("dtype,bound", [("float32", FP32_ABS), ("float64", 1e-12)])
def test_tensor_parallel_gloo(tp, dtype, bound, tmp_path):
    out = tmp_path / f"tp{tp}-{dtype}.json"
    proc = subprocess.run([sys.executable, "-I", str(REPO / "tests/chunked_logprob_tp_worker.py"), str(tp), dtype,
                           str(out)], capture_output=True, text=True, timeout=600,
                          env={"OMP_NUM_THREADS": "1", "PATH": "/usr/bin:/bin"}, check=False)
    assert proc.returncode == 0, proc.stderr[-3000:]
    result = json.loads(out.read_text())
    assert result["ranks_identical"] and result["partial_d_hidden_differs_from_full"]
    for key, gap in result["max_abs"].items():
        assert gap <= bound * max(1.0, result["ref_max_abs"][key]), (key, gap)


class _LiveBytes(TorchDispatchMode):
    """Peak bytes of the tensor storages created inside the mode that are still referenced."""

    def __init__(self):
        super().__init__()
        self.live = self.peak = 0
        self.refs: dict[int, list[int]] = {}

    def _track(self, t):
        storage = t.untyped_storage()
        key = storage.data_ptr()
        if key == 0:
            return
        if key not in self.refs:
            self.refs[key] = [0, storage.nbytes()]
            self.live += storage.nbytes()
            self.peak = max(self.peak, self.live)
        self.refs[key][0] += 1
        weakref.finalize(t, self._drop, key)

    def _drop(self, key):
        entry = self.refs.get(key)
        if entry is not None:
            entry[0] -= 1
            if entry[0] == 0:
                self.live -= entry[1]
                del self.refs[key]

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in out if isinstance(out, (tuple, list)) else (out,):
            if isinstance(t, torch.Tensor):
                self._track(t)
        return out


def _peak(T, V, H, chunk, full):
    hidden, weight, labels = _inputs(T, H, V, scale=0.5)
    hidden.requires_grad_()
    weight.requires_grad_()
    a = torch.randn(T)
    with _LiveBytes() as mode:
        lp, ent = _full(hidden, weight, labels, 1.0) if full else \
            cl.ChunkedVocabParallelLogprob.apply(hidden, weight, labels, 1.0, chunk, None)
        ((lp * a).sum() + ent.sum()).backward()
    return mode.peak


def test_peak_memory_follows_the_chunk_not_the_sequence():
    H, chunk, f32 = 64, 256, 4
    peaks = {}
    for T in (4096, 16384):
        for V in (1024, 4096, 16384):
            peaks[(T, V)] = _peak(T, V, H, chunk, full=False)
    for V in (1024, 16384):
        full = _peak(4096, V, H, chunk, full=True)
        # the tracker does see the [T, V] logits when they exist
        assert full >= 4096 * V * f32
        assert peaks[(4096, V)] < full / 10
    for (T, V), peak in peaks.items():
        # a few [chunk, V] fp32 buffers, the fp32 weight-gradient accumulator, per-token rows
        assert peak <= 4 * chunk * V * f32 + 3 * V * H * f32 + T * (3 * H * f32 + 64), (T, V, peak)
    for V in (1024, 4096, 16384):
        # four times the tokens: the growth is per-token rows only, no V term per token
        growth = peaks[(16384, V)] - peaks[(4096, V)]
        assert growth <= (16384 - 4096) * (3 * H * f32 + 64), (V, growth)


def test_linear_cross_entropy_takes_megatron_shapes_and_refuses_the_rest(monkeypatch):
    hidden, weight, labels = _inputs(50, 16, 64)
    lp2, ent2 = cl.linear_cross_entropy(hidden, weight, labels, 1.0, "none", None)
    lp3, ent3 = cl.linear_cross_entropy(hidden.unsqueeze(1), weight, labels.unsqueeze(0), 1.0, "none", None)
    assert torch.equal(lp2, lp3) and torch.equal(ent2, ent3) and lp3.shape == (50,)
    with pytest.raises(ValueError, match="batch != 1"):
        cl.linear_cross_entropy(hidden.view(25, 2, 16), weight, labels.view(2, 25), 1.0)
    with pytest.raises(ValueError, match="reduction"):
        cl.linear_cross_entropy(hidden, weight, labels, 1.0, "mean")
    with pytest.raises(TypeError):
        cl.linear_cross_entropy(hidden, weight, labels, 1)
    with pytest.raises(ValueError, match="labels for"):
        cl.linear_cross_entropy(hidden, weight, labels[:-1], 1.0)
    monkeypatch.setenv(cl.CHUNK_ENV, "16")
    assert cl.chunk_tokens() == 16
    lp16, _ = cl.linear_cross_entropy(hidden, weight, labels, 1.0)
    assert _gap(lp16, lp2) <= FP32_ABS
    monkeypatch.setenv(cl.CHUNK_ENV, "0")
    with pytest.raises(ValueError):
        cl.chunk_tokens()
    monkeypatch.delenv(cl.CHUNK_ENV)
    assert cl.chunk_tokens() == cl.DEFAULT_CHUNK_TOKENS


def test_no_grad_forward_keeps_working():
    hidden, weight, labels = _inputs(40, 8, 32)
    with torch.no_grad():
        lp, ent = cl.linear_cross_entropy(hidden.requires_grad_(), weight, labels, 1.0)
    rlp, rent = _full(hidden.detach(), weight, labels, 1.0)
    assert _gap(lp, rlp) <= FP32_ABS and _gap(ent, rent) <= FP32_ABS and not lp.requires_grad


# ----------------------------------------------------------------------------- real HybridModel + verl
_HYBRID = {key: __import__("os").environ.get(key) for key in ("YETO_MEGATRON_PYTHON", "YETO_MEGATRON_LM_PATH",
                                                               "YETO_VERL_FORK_PATH")}


@pytest.mark.skipif(not all(_HYBRID.values()), reason="needs YETO_MEGATRON_PYTHON / _LM_PATH / YETO_VERL_FORK_PATH")
@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_hybrid_model_through_verls_fused_forward(tp, dtype, tmp_path):
    """``tests/chunked_logprob_hybrid_worker.py``: the fused path on a real Megatron-Core
    HybridModel against verl's non-fused path, every parameter gradient included. fp32:
    1e-5 (scaled by the reference's largest entry). bf16 is a consistency check: the
    log-probs come from the same bf16 logits (1e-5); verl computes the entropy in bf16
    (measured gap 0.035, the pure-function test bounds it against the truth) and the
    gradients carry that entropy term (measured 0.0076 scaled)."""
    import os

    out = tmp_path / f"hybrid-{tp}-{dtype}.json"
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    proc = subprocess.run([_HYBRID["YETO_MEGATRON_PYTHON"], "-I", str(REPO / "tests/chunked_logprob_hybrid_worker.py"),
                           str(tp), dtype, str(out)], env=env, capture_output=True, text=True, timeout=900,
                          check=False)
    assert proc.returncode == 0, proc.stderr[-4000:]
    result = json.loads(out.read_text())
    # without the fix verl leaves the model unpatched and its legacy call fails
    assert result["unpatched_mode"] is None and result["legacy_call_before"]
    assert "unexpected keyword argument 'temperature'" in result["before_error"]
    assert result["checked"].endswith("HybridModel:hook:last") and result["plain_unchanged"]
    assert result["logits_shape"] == [1, 64, 256 // tp]
    lp_bound, ent_bound, grad_bound = (FP32_ABS, FP32_ABS, FP32_ABS) if dtype == "float32" else (1e-5, 0.1, 0.02)
    assert len(result["ranks"]) == tp
    for rank in result["ranks"]:
        assert rank["logprob"][0] <= lp_bound and rank["entropy"][0] <= ent_bound
        assert len(rank["grads"]) == 9  # embedding, 2 x (norm, fc1, fc2), final norm, output layer
        for name, (gap, scale) in rank["grads"].items():
            assert gap <= grad_bound * max(1.0, scale), (name, gap)
        # pipeline stages: a first stage hands back the plain hidden states bit for bit; a
        # last stage (input from set_input_tensor) matches, its input gradient included
        assert rank["first_stage_same"]
        stage = rank["last_stage"]
        assert stage["checked"].endswith("HybridModel:hook:mid,HybridModel:hook:last")
        assert stage["logprob"][0] <= lp_bound and stage["entropy"][0] <= ent_bound
        assert "input_tensor" in stage["grads"] and "embedding.word_embeddings.weight" not in stage["grads"]
        for name, (gap, scale) in stage["grads"].items():
            assert gap <= grad_bound * max(1.0, scale), ("last stage", name, gap)
