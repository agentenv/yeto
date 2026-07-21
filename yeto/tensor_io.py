"""Pack/unpack fragments between torch tensors and wire bytes.

Every PUSH_FRAGMENT is a base-relative learner delta. Q4 wire format
(DTYPE_Q4) encodes those values in blocks of 256; each block is
serialized as an f32 absmax scale followed by 128 bytes of packed nibbles
(two values per byte, low nibble first). A nibble is 1 sign bit (bit 3) + a
3-bit magnitude level: level 0 is
exactly zero, level L in 1..7 decodes to sign * 2^(L-7) * scale (E3M0 —
sign, 3 exponent bits, no mantissa). The last block is zero-padded; the
decoder truncates to the fragment's numel.
"""

from __future__ import annotations

import struct

import torch

from .fragments import Fragment
from .protocol import DTYPE_BF16, DTYPE_F32

_WIRE_TORCH = {DTYPE_F32: torch.float32, DTYPE_BF16: torch.bfloat16}

Q4_BLOCK = 256  # values per scale block; 4.125 bits/value on the wire


def fragment_flat(frag: Fragment, params: dict[str, torch.Tensor]) -> torch.Tensor:
    """The fragment's tensors (layout order) as one flat f32 tensor,
    on the params' device."""
    return torch.cat([params[name].detach().reshape(-1).float() for name, _ in frag.tensors])


def pack_fragment(frag: Fragment, params: dict[str, torch.Tensor], dtype: int) -> bytes:
    """Concatenate the fragment's tensors (layout order) into wire bytes."""
    return pack_tensor(fragment_flat(frag, params), dtype)


def pack_tensor(flat: torch.Tensor, dtype: int) -> bytes:
    """Encode one flat f32 tensor in an unquantized wire dtype."""
    wire = flat.detach().float().to(_WIRE_TORCH[dtype]).contiguous().cpu()
    return wire.view(torch.uint8).numpy().tobytes()


def unpack_fragment(frag: Fragment, data: bytes, dtype: int) -> torch.Tensor:
    """Decode wire bytes into a flat f32 tensor of the fragment's numel."""
    raw = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    flat = raw.view(_WIRE_TORCH[dtype]).float()
    if flat.numel() != frag.numel:
        raise ValueError(f"fragment payload has {flat.numel()} values, expected {frag.numel}")
    return flat


def apply_fragment(frag: Fragment, flat: torch.Tensor, params: dict[str, torch.Tensor]) -> None:
    """Overwrite the fragment's tensors from a flat f32 tensor."""
    off = 0
    with torch.no_grad():
        for name, numel in frag.tensors:
            p = params[name]
            p.copy_(flat[off : off + numel].view_as(p).to(p.dtype))
            off += numel


def quantize_q4(flat: torch.Tensor) -> bytes:
    """Encode a flat f32 tensor as blockwise E3M0 (see module docstring)."""
    flat = flat.detach().float().cpu().contiguous()
    numel = flat.numel()
    blocks = -(-numel // Q4_BLOCK)
    padded = torch.zeros(blocks * Q4_BLOCK, dtype=torch.float32)
    padded[:numel] = flat
    padded = padded.view(blocks, Q4_BLOCK)

    scale = padded.abs().amax(dim=1)
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    normalized = padded.abs() / safe.unsqueeze(1)
    # Nearest level in log space; below 2^-6.5 rounds to exact zero.
    level = torch.where(
        normalized > 0,
        (normalized.log2().round() + 7).clamp(0, 7),
        torch.zeros_like(normalized),
    ).to(torch.uint8)
    sign = ((padded < 0) & (level > 0)).to(torch.uint8)
    nibbles = sign * 8 + level
    packed = (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).contiguous()

    out = bytearray()
    scale_bytes = scale.numpy().tobytes()  # little-endian f32 on all supported platforms
    packed_bytes = packed.numpy().tobytes()
    row = Q4_BLOCK // 2
    for b in range(blocks):
        out += scale_bytes[b * 4 : b * 4 + 4]
        out += packed_bytes[b * row : (b + 1) * row]
    return bytes(out)


_Q4_LUT = torch.tensor([0.0] + [2.0 ** (level - 7) for level in range(1, 8)], dtype=torch.float32)


def dequantize_q4(data: bytes, numel: int) -> torch.Tensor:
    """Decode q4 bytes into a flat f32 tensor of `numel` values."""
    blocks = -(-numel // Q4_BLOCK)
    row = Q4_BLOCK // 2
    if len(data) != blocks * (4 + row):
        raise ValueError(f"q4 payload has {len(data)} bytes, expected {blocks * (4 + row)}")
    scales = torch.tensor(
        struct.unpack_from(f"<{blocks}f", b"".join(data[b * (4 + row) : b * (4 + row) + 4] for b in range(blocks))),
        dtype=torch.float32,
    )
    packed = torch.frombuffer(
        bytearray(b"".join(data[b * (4 + row) + 4 : (b + 1) * (4 + row)] for b in range(blocks))),
        dtype=torch.uint8,
    ).view(blocks, row)
    nibbles = torch.empty(blocks, Q4_BLOCK, dtype=torch.uint8)
    nibbles[:, 0::2] = packed & 0x0F
    nibbles[:, 1::2] = packed >> 4
    magnitude = _Q4_LUT[(nibbles & 0x07).long()] * scales.unsqueeze(1)
    values = torch.where(nibbles >= 8, -magnitude, magnitude)
    return values.reshape(-1)[:numel]
