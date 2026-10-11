"""Self-checks for tools/eval_sft_heldout.py (no network, tiny fake model).

Run: PYTHONPATH=. python -m pytest tools/test_eval_sft_heldout.py
Optional real check: set RECOVERED_ADAPTER=<dir> to verify a recovered PEFT
adapter loads onto Qwen/Qwen3-0.6B with no missing/unexpected keys.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from eval_sft_heldout import weighted_ce  # noqa: E402


class _Fixed(torch.nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab

    def forward(self, input_ids):
        from types import SimpleNamespace

        return SimpleNamespace(logits=torch.zeros(*input_ids.shape, self.vocab))


def test_weighted_ce_uses_shifted_weights():
    vocab = 7
    loss, tw = weighted_ce(_Fixed(vocab), [[1, 2, 3, 4]], [[1.0, 0.0, 1.0, 1.0]])
    # weights[1:] = [0, 1, 1] -> 2 targets, uniform logits -> CE = log(vocab)
    assert tw == 2.0
    assert loss / tw == pytest.approx(torch.log(torch.tensor(float(vocab))).item())


@pytest.mark.skipif(not os.environ.get("RECOVERED_ADAPTER"), reason="set RECOVERED_ADAPTER")
def test_recovered_adapter_loads_cleanly():
    from eval_sft_heldout import load_model

    _, info = load_model(
        "Qwen/Qwen3-0.6B",
        "c1899de289a04d12100db370d81485cdf75e47ca",
        os.environ["RECOVERED_ADAPTER"],
        torch.float32,
    )
    assert info["adapter_tensors"] == 392
    assert not info["missing_lora_keys"] and not info["unexpected_keys"]
