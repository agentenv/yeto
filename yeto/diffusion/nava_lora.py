"""Small LoRA patcher for NAVA's plain torch modules."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

PRESET_PATTERNS = {
    "auto": r"^backbone\.(double|single|double_final)_blocks\.\d+\..*",
    "attention": r"^backbone\.(double|single|double_final)_blocks\.\d+\..*(self_attn|cross_attn|attn).*",
    "all-linear": r"^backbone\.(double|single|double_final)_blocks\.\d+\..*",
}


@dataclass
class LoRAConfig:
    r: int
    alpha: int
    target: str
    patched_modules: list[str]


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: int):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be positive")
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.scaling = float(alpha) / float(r)
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.lora_A.to(device=base.weight.device, dtype=torch.float32)
        self.lora_B.to(device=base.weight.device, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        a = self.lora_A.weight.to(device=x.device, dtype=x.dtype)
        b = self.lora_B.weight.to(device=x.device, dtype=x.dtype)
        update = F.linear(F.linear(x, a), b) * self.scaling
        return out + update.to(out.dtype)


def _parent_and_attr(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parent = root
    parts = module_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def patch_lora(model: nn.Module, *, r: int, alpha: int, target: str = "auto") -> LoRAConfig:
    for p in model.parameters():
        p.requires_grad_(False)
    pattern = PRESET_PATTERNS.get(target, target)
    rx = re.compile(pattern)
    modules = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and rx.search(n)]
    if not modules:
        raise ValueError(f"NAVA LoRA target {target!r} matched no nn.Linear modules")
    patched = []
    for name, module in modules:
        parent, attr = _parent_and_attr(model, name)
        setattr(parent, attr, LoRALinear(module, r=r, alpha=alpha))
        patched.append(name)
    return LoRAConfig(r=r, alpha=alpha, target=target, patched_modules=patched)
