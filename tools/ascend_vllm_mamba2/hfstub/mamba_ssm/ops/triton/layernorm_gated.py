"""G-M2 stub: torch reference of mamba_ssm's gated RMSNorm (rms_norm_ref semantics), for the HF CPU reference."""
import torch


def rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-6, group_size=None, norm_before_gate=True, upcast=True):
    dtype = x.dtype
    xf = x.float()
    w = weight.float()
    if z is not None and not norm_before_gate:
        xf = xf * torch.nn.functional.silu(z.float())
    gs = group_size or xf.shape[-1]
    xg = xf.view(*xf.shape[:-1], xf.shape[-1] // gs, gs)
    xg = xg * torch.rsqrt(xg.pow(2).mean(-1, keepdim=True) + eps)
    out = xg.view_as(xf) * w
    if bias is not None:
        out = out + bias.float()
    if z is not None and norm_before_gate:
        out = out * torch.nn.functional.silu(z.float())
    return out.to(dtype)
