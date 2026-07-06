"""LoRA utilities for NAVA-style arbitrary torch modules.

This intentionally does not depend on PEFT: NAVA is not a CausalLM wrapper,
and the trainable adapters must have stable module-path names for Yeto's
fragment layout and checkpoint export.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


PRESET_PATTERNS = {
    # All Linear modules in the MMDiT transformer blocks. This is the default
    # domain-adaptation path: small adapter payload, broad coverage.
    "mmdit-all-linear": r"^backbone\.(double|single|double_final)_blocks\.\d+\..*",
    # Attention-only adapters for lower VRAM / faster iteration.
    "attention-only": r"^backbone\.(double|single|double_final)_blocks\.\d+\..*(self_attn|cross_attn).*",
    # Prompt/context alignment path.
    "cross-attn-only": r"^backbone\.(double|single|double_final)_blocks\.\d+\..*cross_attn.*",
    # FFN-only is useful for style/domain shifts while touching fewer routing paths.
    "ffn-only": r"^backbone\.(double|single|double_final)_blocks\.\d+\..*ffn.*",
}

_WRAPPER_PREFIXES = ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "module.")


def _clean_param_name(name: str) -> str:
    for prefix in _WRAPPER_PREFIXES:
        name = name.replace(prefix, "")
    return name


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target: str = "mmdit-all-linear"
    patched_modules: list[str] | None = None

    @property
    def scaling(self) -> float:
        return self.alpha / max(1, self.r)


class LoRALinear(nn.Module):
    """Drop-in ``nn.Linear`` replacement with a frozen base and trainable LoRA.

    Adapter params are named ``lora_A.weight`` and ``lora_B.weight`` so they
    remain easy to filter and export. The base module is kept as ``base``.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be positive")
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = float(alpha) / float(r)
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)
        self.reset_parameters()
        self.lora_A.to(device=base.weight.device, dtype=torch.float32)
        self.lora_B.to(device=base.weight.device, dtype=torch.float32)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        # Keep adapter math in the activation dtype at runtime; adapter storage
        # is fp32 for stable AdamW state, then cast on the fly.
        a = self.lora_A.weight.to(dtype=x.dtype, device=x.device)
        b = self.lora_B.weight.to(dtype=x.dtype, device=x.device)
        dropped = self.dropout(x)
        update = F.linear(F.linear(dropped, a), b) * self.scaling
        return base_out + update.to(base_out.dtype)

    def merged_weight_bias(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        w = self.base.weight.detach().float().clone()
        delta = self.lora_B.weight.detach().float() @ self.lora_A.weight.detach().float()
        w.add_(delta, alpha=self.scaling)
        b = self.base.bias.detach().float().clone() if self.base.bias is not None else None
        return w, b


def _matcher(target: str) -> Callable[[str, nn.Module], bool]:
    pattern = PRESET_PATTERNS.get(target, target)
    rx = re.compile(pattern)

    def match(name: str, module: nn.Module) -> bool:
        return isinstance(module, nn.Linear) and bool(rx.search(name))

    return match


def _parent_and_attr(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parent = root
    parts = module_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def patch_lora(
    model: nn.Module,
    *,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.0,
    target: str = "mmdit-all-linear",
    verbose: bool = False,
) -> LoRAConfig:
    """Patch matching ``nn.Linear`` modules in-place and freeze everything else."""
    for p in model.parameters():
        p.requires_grad_(False)

    match = _matcher(target)
    to_patch = [(name, mod) for name, mod in model.named_modules() if match(name, mod)]
    if not to_patch:
        raise ValueError(f"LoRA target {target!r} matched no nn.Linear modules")

    patched = []
    for name, module in to_patch:
        parent, attr = _parent_and_attr(model, name)
        setattr(parent, attr, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
        patched.append(name)
        if verbose:
            print(f"[nava-lora] patched {name}")
    return LoRAConfig(r=r, alpha=alpha, dropout=dropout, target=target, patched_modules=patched)


def trainable_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        _clean_param_name(name): p.detach().cpu()
        for name, p in model.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    }


def save_lora_adapter(model: nn.Module, config: LoRAConfig, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        from safetensors.torch import save_file

        save_file(trainable_lora_state_dict(model), str(out / "adapter.safetensors"))
    except Exception:
        torch.save(trainable_lora_state_dict(model), out / "adapter.pt")
    (out / "adapter_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


def load_lora_adapter(model: nn.Module, adapter_dir: str | Path, *, strict: bool = False):
    adapter_dir = Path(adapter_dir)
    cfg = LoRAConfig(**json.loads((adapter_dir / "adapter_config.json").read_text()))
    patch_lora(model, r=cfg.r, alpha=cfg.alpha, dropout=cfg.dropout, target=cfg.target)
    sf = adapter_dir / "adapter.safetensors"
    if sf.exists():
        from safetensors.torch import load_file

        sd = load_file(str(sf), device="cpu")
    else:
        sd = torch.load(adapter_dir / "adapter.pt", map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    return cfg, list(missing), list(unexpected)


def merge_lora_inplace(model: nn.Module) -> int:
    """Replace every ``LoRALinear`` with a merged frozen ``nn.Linear``."""
    merged = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue
        parent, attr = _parent_and_attr(model, name)
        weight, bias = module.merged_weight_bias()
        lin = nn.Linear(module.in_features, module.out_features, bias=bias is not None)
        lin.weight.data.copy_(weight.to(lin.weight.dtype))
        if bias is not None:
            lin.bias.data.copy_(bias.to(lin.bias.dtype))
        lin.to(device=module.base.weight.device, dtype=module.base.weight.dtype)
        setattr(parent, attr, lin)
        merged += 1
    return merged
