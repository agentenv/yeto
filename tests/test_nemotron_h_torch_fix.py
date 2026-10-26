"""yeto.rl.nemotron_h_torch_fix: transformers' NemotronH Mamba2 torch path against a
step-by-step recurrence (CPU, a tiny random mixer)."""

from __future__ import annotations

import importlib

import pytest

torch = pytest.importorskip("torch")
nh = pytest.importorskip("transformers.models.nemotron_h.modeling_nemotron_h")

from yeto.rl import nemotron_h_torch_fix as fix  # noqa: E402


def _mixer(chunk_size: int = 4, time_step_min: float = 0.001):
    from transformers import NemotronHConfig

    config = NemotronHConfig(hidden_size=32, mamba_num_heads=4, mamba_head_dim=8, ssm_state_size=6, n_groups=2,
                             chunk_size=chunk_size, conv_kernel=4, mamba_hidden_act="silu", use_conv_bias=True,
                             time_step_limit=(0.0, float("inf")), time_step_min=time_step_min,
                             layer_norm_epsilon=1e-5, vocab_size=64)
    torch.manual_seed(0)
    mixer = nh.NemotronHMamba2Mixer(config, layer_idx=0).float().eval()
    with torch.no_grad():
        for p in mixer.parameters():
            p.copy_(torch.randn_like(p) * 0.3)
        mixer.dt_bias.copy_(torch.randn_like(mixer.dt_bias) - 2.0)  # small dt, some below time_step_min
    return mixer


def _reference(mixer, x: torch.Tensor) -> torch.Tensor:
    """The selective scan one step at a time (kernel semantics: dt clamped by time_step_limit)."""
    b, t, _ = x.shape
    nheads, hd, n, ng = mixer.num_heads, mixer.head_dim, mixer.ssm_state_size, mixer.n_groups
    proj = mixer.in_proj(x)
    gate, xbc, dt = torch.split(proj, [mixer.intermediate_size, mixer.conv_dim, nheads], dim=-1)
    xbc = mixer.act(mixer.conv1d(xbc.transpose(1, 2))[..., :t].transpose(1, 2))
    xs, B, C = torch.split(xbc, [mixer.intermediate_size, ng * n, ng * n], dim=-1)
    dt = torch.nn.functional.softplus(dt + mixer.dt_bias)
    lo, hi = mixer.time_step_limit
    dt = torch.clamp(dt, lo, hi)
    A = -torch.exp(mixer.A_log.float())
    xs = xs.reshape(b, t, nheads, hd)
    B = B.reshape(b, t, ng, n).repeat_interleave(nheads // ng, dim=2)
    C = C.reshape(b, t, ng, n).repeat_interleave(nheads // ng, dim=2)
    h = torch.zeros(b, nheads, hd, n)
    ys = []
    for i in range(t):
        da = torch.exp(dt[:, i] * A)  # (b, h)
        h = h * da[:, :, None, None] + (dt[:, i, :, None, None] * xs[:, i, :, :, None] * B[:, i, :, None, :])
        y = (h * C[:, i, :, None, :]).sum(-1) + mixer.D[None, :, None] * xs[:, i]
        ys.append(y.reshape(b, -1))
    y = torch.stack(ys, dim=1)
    return mixer.out_proj(mixer.norm(y, gate))


@pytest.fixture(autouse=True)
def _fresh_module():
    importlib.reload(nh)  # each test starts from the unpatched class
    yield
    importlib.reload(nh)


def test_unpatched_torch_path_is_wrong_past_one_chunk_and_exact_within_one():
    mixer = _mixer(time_step_min=0.0)  # isolate the scan bug from the clamp difference
    with torch.no_grad():
        short = torch.randn(1, 4, 32)  # one chunk
        long = torch.randn(1, 13, 32)  # four chunks (padded)
        assert torch.allclose(mixer.torch_forward(short), _reference(mixer, short), atol=1e-4)
        assert (mixer.torch_forward(long) - _reference(mixer, long)).abs().max() > 1e-2


def test_patch_matches_the_recurrence_on_several_chunks():
    info = fix.apply(module=nh)
    assert info == {"already": False, "module": nh.__name__, "scan_fix": True, "align_dt_clamp": True}
    mixer = _mixer()
    with torch.no_grad():
        for t in (3, 4, 9, 13, 17):
            x = torch.randn(2, t, 32)
            assert torch.allclose(mixer.torch_forward(x), _reference(mixer, x), atol=1e-4), t


def test_scan_fix_alone_still_differs_by_the_dt_clamp():
    fix.apply(module=nh, align_dt_clamp=False)
    mixer = _mixer(time_step_min=0.05)  # some softplus(dt + bias) values fall below it
    with torch.no_grad():
        x = torch.randn(1, 13, 32)
        assert (mixer.torch_forward(x) - _reference(mixer, x)).abs().max() > 1e-4


def test_apply_is_idempotent_and_refuses_a_missing_anchor():
    first = fix.apply(module=nh)
    second = fix.apply(module=nh)
    assert first["already"] is False and second["already"] is True
    with pytest.raises(fix.NemotronHTorchFixError):
        fix.patched_source("def torch_forward(self):\n    return 1\n")
