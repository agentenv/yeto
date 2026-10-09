import torch
import pytest

from yeto.rl.adapters.miles.state import MilesPolicyState


def test_export_layout_mismatch_names_the_exported_tensors():
    exporter = MilesPolicyState(actor_model=object(), base_model_revision="r" * 40,
                                      config_hash="c" * 64, expected_layout_hash="0" * 64)
    tensors = {"layers.0.q.lora_A.weight": torch.zeros(16, 8), "layers.0.q.lora_B.weight": torch.zeros(8, 16)}
    with pytest.raises(ValueError, match=r"layout hash changed: exported 2 tensors .*layers\.0\.q\.lora_A\.weight\[16, 8\]"):
        exporter._export_result(0, [{"policy_version": 0, "tensors": tensors}])
