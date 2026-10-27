"""yeto.rl.ascend_vllm_mamba2: torch Mamba2 ops for vLLM on Ascend, checked on CPU against a
step-by-step recurrence and torch's conv1d."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from yeto.rl import ascend_vllm_mamba2 as m2  # noqa: E402

H, P, N, G, W, DIM = 4, 3, 5, 2, 4, 6


def _recurrence(x, dt, A, B, C, D, init):
    """x (L,H,P), dt (L,H) after softplus, B/C (L,G,N). Returns y (L,H,P), final state (H,P,N)."""
    rep = H // G
    s = init.clone()
    ys = []
    for t in range(x.shape[0]):
        Bh = B[t].repeat_interleave(rep, 0)
        Ch = C[t].repeat_interleave(rep, 0)
        s = s * torch.exp(dt[t] * A)[:, None, None] + (dt[t][:, None] * x[t])[..., None] * Bh[:, None, :]
        ys.append((s * Ch[:, None, :]).sum(-1) + D[:, None] * x[t])
    return torch.stack(ys), s


def _inputs(T, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, H, P, generator=g, dtype=torch.float64)
    dt = torch.randn(T, H, generator=g, dtype=torch.float64)
    A = -torch.rand(H, generator=g, dtype=torch.float64) - 0.1
    B = torch.randn(T, G, N, generator=g, dtype=torch.float64)
    C = torch.randn(T, G, N, generator=g, dtype=torch.float64)
    D = torch.randn(H, generator=g, dtype=torch.float64)
    bias = torch.randn(H, generator=g, dtype=torch.float64) - 1.0
    return x, dt, A, B, C, D, bias


@pytest.mark.parametrize("chunk", [4, 7, 64])
def test_ssd_prefill_matches_recurrence_varlen_with_initial_states(chunk):
    lens = [1, 9, 16, 5]
    T = sum(lens)
    x, dt, A, B, C, D, bias = _inputs(T)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)))
    init = torch.randn(len(lens), H, P, N, dtype=torch.float64)
    init[1] = 0
    out = torch.zeros(T, H, P, dtype=torch.float64)
    st = m2.ssd_prefill(x, dt, A, B, C, chunk, cu, out=out, D=D, dt_bias=bias, initial_states=init,
                        dt_softplus=True)
    dts = torch.nn.functional.softplus(dt + bias)
    for i in range(len(lens)):
        a, b = int(cu[i]), int(cu[i + 1])
        y_ref, s_ref = _recurrence(x[a:b], dts[a:b], A, B[a:b], C[a:b], D, init[i])
        assert torch.allclose(out[a:b], y_ref, atol=1e-6, rtol=1e-6)
        assert torch.allclose(st[i].double(), s_ref, atol=1e-6, rtol=1e-6)


def test_ssd_prefill_then_decode_equals_full_recurrence():
    T = 12
    x, dt, A, B, C, D, bias = _inputs(T, seed=1)
    pre = 8
    out = torch.zeros(pre, H, P, dtype=torch.float64)
    st = m2.ssd_prefill(x[:pre], dt[:pre], A, B[:pre], C[:pre], 4, torch.tensor([0, pre]), out=out, D=D,
                        dt_bias=bias, dt_softplus=True, state_dtype=torch.float64)
    cache = torch.zeros(3, H, P, N, dtype=torch.float64)
    cache[2] = st[0]
    ys = []
    for t in range(pre, T):  # decode in the shapes mamba_mixer2 passes
        o = torch.zeros(2, H, P, dtype=torch.float64)
        m2.ssd_decode(cache, x[t:t + 1].expand(2, -1, -1).clone(), dt[t][None, :, None].expand(2, -1, P),
                      A[:, None, None].expand(-1, P, N), B[t:t + 1].expand(2, -1, -1), C[t:t + 1].expand(2, -1, -1),
                      D[:, None].expand(-1, P), dt_bias=bias[:, None].expand(-1, P), dt_softplus=True,
                      state_batch_indices=torch.tensor([2, m2.PAD_SLOT_ID]), out=o)
        assert torch.all(o[1] == 0)  # padded row untouched
        ys.append(o[0])
    dts = torch.nn.functional.softplus(dt + bias)
    y_ref, s_ref = _recurrence(x, dts, A, B, C, D, torch.zeros(H, P, N, dtype=torch.float64))
    assert torch.allclose(torch.stack(ys), y_ref[pre:], atol=1e-6)
    assert torch.allclose(cache[2], s_ref, atol=1e-6)
    assert torch.all(cache[:2] == 0)


def test_conv_prefill_and_decode_match_conv1d():
    g = torch.Generator().manual_seed(3)
    T = 10
    x = torch.randn(DIM, T, generator=g, dtype=torch.float64)
    w = torch.randn(DIM, W, generator=g, dtype=torch.float64)
    bias = torch.randn(DIM, generator=g, dtype=torch.float64)
    base = torch.randn(4, W - 1, DIM, dtype=torch.float64)  # vLLM layout: (slots, width-1, dim), viewed transposed
    states = base.transpose(-1, -2)
    init1 = states[1].clone()
    lens = [6, 4]
    qsl = torch.tensor([0, 6, 10])
    out = m2.conv_prefill(x, w, bias, activation="silu", conv_states=states, has_initial_state=torch.tensor([False, True]),
                          cache_indices=torch.tensor([0, 1]), query_start_loc=qsl)

    def ref(xs, init):
        xx = torch.cat([init, xs], -1)
        return torch.nn.functional.silu(torch.nn.functional.conv1d(xx[None], w[:, None, :], bias, groups=DIM)[0]), xx

    y0, xx0 = ref(x[:, :6], torch.zeros(DIM, W - 1, dtype=torch.float64))
    y1, xx1 = ref(x[:, 6:], init1)
    assert torch.allclose(out[:, :6], y0) and torch.allclose(out[:, 6:], y1)
    assert torch.allclose(states[0], xx0[:, -(W - 1):]) and torch.allclose(states[1], xx1[:, -(W - 1):])
    # decode one token for slot 1 (row 0) plus a padded row
    xt = torch.randn(2, DIM, generator=g, dtype=torch.float64)
    yd = m2.conv_decode(xt, states, w, bias, "silu", conv_state_indices=torch.tensor([1, m2.PAD_SLOT_ID]))
    full = torch.cat([x[:, 6:], xt[0][:, None]], -1)
    yfull, xxf = ref(full, init1)
    assert torch.allclose(yd[0], yfull[:, -1]) and torch.all(yd[1] == 0)
    assert torch.allclose(states[1], xxf[:, -(W - 1):])


def test_relu2_mlp_passthrough_for_other_activations():
    calls = []

    def orig(**kw):
        calls.append(kw["activation"])
        return "orig"

    f = m2.make_relu2_unquant_apply_mlp(orig)
    assert f(None, None, None, None, activation="silu") == "orig" and calls == ["silu"]
    assert m2._is_relu2_no_mul("relu2_no_mul")

    class E:
        value = "relu2_no_mul"

    assert m2._is_relu2_no_mul(E()) and not m2._is_relu2_no_mul("relu2")


def test_plugin_entry_point_file_names_register():
    import configparser
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    ep = root / "tools/ascend_vllm_mamba2/site/yeto_ascend_mamba2-0.1.dist-info/entry_points.txt"
    cp = configparser.ConfigParser()
    cp.read(ep)
    target = cp["vllm.general_plugins"]["yeto_ascend_mamba2"]
    mod, fn = target.split(":")
    import importlib

    assert callable(getattr(importlib.import_module(mod), fn))


def test_gateless_platform_proxy_only_flips_for_the_named_caller():
    class Real:
        name = "npu"

        def is_cuda_alike(self):
            return False

    def guarded(platform):
        return platform.is_cuda_alike()

    def other(platform):
        return platform.is_cuda_alike()

    proxy = m2._GatelessMoEPlatform(Real(), guarded.__code__)
    assert guarded(proxy) is True
    assert other(proxy) is False
    assert proxy.name == "npu"


def test_ssd_decode_per_head_fast_path_equals_general_path():
    g = torch.Generator().manual_seed(5)
    b = 3
    state = torch.randn(4, H, P, N, generator=g, dtype=torch.float64)
    x = torch.randn(b, H, P, generator=g, dtype=torch.float64)
    dt_h = torch.randn(b, H, generator=g, dtype=torch.float64)
    A_h = -torch.rand(H, generator=g, dtype=torch.float64)
    B = torch.randn(b, G, N, generator=g, dtype=torch.float64)
    C = torch.randn(b, G, N, generator=g, dtype=torch.float64)
    D = torch.randn(H, generator=g, dtype=torch.float64)[:, None].expand(-1, P)
    bias = torch.randn(H, generator=g, dtype=torch.float64)[:, None].expand(-1, P)
    idx = torch.tensor([3, 0, 2])
    outs, states = [], []
    for expand in (True, False):
        dt = dt_h[:, :, None].expand(-1, -1, P)
        A = A_h[:, None, None].expand(-1, P, N)
        bb = bias
        if not expand:
            dt, A, bb = dt.contiguous(), A.contiguous(), bias.contiguous()
        st = state.clone()
        o = torch.zeros(b, H, P, dtype=torch.float64)
        m2.ssd_decode(st, x, dt, A, B, C, D, dt_bias=bb, dt_softplus=True, state_batch_indices=idx, out=o)
        outs.append(o)
        states.append(st)
    assert torch.allclose(outs[0], outs[1]) and torch.allclose(states[0], states[1])


def test_restore_moe_loaders_handles_both_layouts():
    import types

    E, I, Hd = 2, 3, 4

    class Moe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.local_num_experts, self.intermediate_size_per_partition, self.hidden_size = E, I, Hd
            self.moe_config = types.SimpleNamespace(is_act_and_mul=False)
            # vllm_ascend's kernel layout, loaders dropped
            self.w13_weight = torch.nn.Parameter(torch.zeros(E, Hd, I), requires_grad=False)
            self.w2_weight = torch.nn.Parameter(torch.zeros(E, I, Hd), requires_grad=False)

        def weight_loader(self, param, loaded, expert_id):
            param.data[expert_id].copy_(loaded)
            return True

    m = torch.nn.Module()
    m.moe = Moe()
    assert m2.restore_moe_loaders(m) == 2 and m2.restore_moe_loaders(m) == 0
    w = torch.randn(I, Hd)
    m.moe.w13_weight.weight_loader(m.moe.w13_weight, w, expert_id=1)
    assert torch.equal(m.moe.w13_weight.data[1], w.t())
    staged = torch.nn.Parameter(torch.zeros(E, I, Hd), requires_grad=False)  # verl's staging view
    m.moe.w13_weight.weight_loader(staged, w, expert_id=0)
    assert torch.equal(staged.data[0], w)
    with pytest.raises(RuntimeError):
        m.moe.w13_weight.weight_loader(torch.nn.Parameter(torch.zeros(5, 5, 5)), w, expert_id=0)
