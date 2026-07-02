"""Pack/unpack fragments between torch tensors and wire bytes."""

from __future__ import annotations

import torch

from .fragments import Fragment
from .protocol import DTYPE_BF16, DTYPE_F32

_WIRE_TORCH = {DTYPE_F32: torch.float32, DTYPE_BF16: torch.bfloat16}


def pack_fragment(frag: Fragment, params: dict[str, torch.Tensor], dtype: int) -> bytes:
    """Concatenate the fragment's tensors (layout order) into wire bytes."""
    flat = torch.cat([params[name].detach().reshape(-1).float() for name, _ in frag.tensors])
    wire = flat.to(_WIRE_TORCH[dtype]).contiguous().cpu()
    return wire.view(torch.uint8).numpy().tobytes()


def unpack_fragment(frag: Fragment, data: bytes, dtype: int) -> torch.Tensor:
    """Decode wire bytes into a flat f32 tensor of the fragment's numel."""
    raw = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    flat = raw.view(_WIRE_TORCH[dtype]).float()
    if flat.numel() != frag.numel:
        raise ValueError(f"fragment payload has {flat.numel()} values, expected {frag.numel}")
    return flat


def apply_fragment(frag: Fragment, flat: torch.Tensor, params: dict[str, torch.Tensor]) -> None:
    """Overwrite the fragment's tensors from a flat f32 tensor (α = 0)."""
    off = 0
    with torch.no_grad():
        for name, numel in frag.tensors:
            p = params[name]
            p.copy_(flat[off : off + numel].view_as(p).to(p.dtype))
            off += numel
