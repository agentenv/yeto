"""Overlay for the SGLang fork 2fa88018 in the Miles ports image (S26 E6 probe only).

Qwen2 MLP and Qwen3 attention cast the activation to the projection's weight dtype
when they differ (true-on-policy: fp32 RMSNorm output -> bf16 weights). With an FP8
projection (online FP8 or a compressed-tensors FP8 checkpoint) the weight dtype is
float8_e4m3fn, so the BF16 activation is cast to FP8 and ``apply_fp8_linear`` then
asserts on a "pre-quantized" input without a scale (e6-gate-t1 / e6-gate-t2,
2026-10-21). The overlay limits the cast to weights of two or more bytes.

Usage (in the image): ``python sglang_fp8_cast_overlay.py /sgl-workspace/sglang/python``.
Refuses to run when a pattern is not found exactly once (another SGLang commit).
"""

import sys
from pathlib import Path

EDITS = {
    "sglang/srt/models/qwen2.py": [
        ("or x.dtype != self.gate_up_proj.weight.dtype\n",
         "or (x.dtype != self.gate_up_proj.weight.dtype and self.gate_up_proj.weight.dtype.itemsize > 1)\n"),
    ],
    "sglang/srt/models/qwen3.py": [
        ("or hidden_states.dtype != self.qkv_proj.weight.dtype\n",
         "or (hidden_states.dtype != self.qkv_proj.weight.dtype and self.qkv_proj.weight.dtype.itemsize > 1)\n"),
    ],
}


def main(root: str) -> None:
    for rel, edits in EDITS.items():
        path = Path(root) / rel
        text = path.read_text()
        for old, new in edits:
            if text.count(old) != 1:
                raise SystemExit(f"{path}: expected one {old!r}, found {text.count(old)}")
            text = text.replace(old, new)
        path.write_text(text)
        print(f"patched {path}")


if __name__ == "__main__":
    main(sys.argv[1])
