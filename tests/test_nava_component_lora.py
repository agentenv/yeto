import torch
import torch.nn as nn

from yeto.components.nava.lora import LoRALinear, patch_lora, trainable_lora_state_dict


class TinyNavaLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.double_blocks = nn.ModuleList([
            nn.ModuleDict({"self_attn": nn.ModuleDict({"q": nn.Linear(4, 4)})})
        ])
        self.other = nn.Linear(4, 4)

    def forward(self, x):
        return self.backbone.double_blocks[0]["self_attn"]["q"](x) + self.other(x)


def test_patch_lora_only_targets_nava_backbone_blocks():
    model = TinyNavaLike()
    cfg = patch_lora(model, r=2, alpha=4, target="mmdit-all-linear")

    patched = model.backbone.double_blocks[0]["self_attn"]["q"]
    assert isinstance(patched, LoRALinear)
    assert cfg.patched_modules == ["backbone.double_blocks.0.self_attn.q"]
    assert not model.other.weight.requires_grad
    assert patched.lora_A.weight.requires_grad
    assert patched.lora_B.weight.requires_grad
    assert sorted(trainable_lora_state_dict(model)) == [
        "backbone.double_blocks.0.self_attn.q.lora_A.weight",
        "backbone.double_blocks.0.self_attn.q.lora_B.weight",
    ]
    assert model(torch.randn(2, 4)).shape == (2, 4)
