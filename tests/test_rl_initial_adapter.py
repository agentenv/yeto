from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from yeto.adapter_lifecycle import directory_sha256
from yeto.rl.core import (
    canonical_layout_hash,
    canonical_lora_config_hash,
    canonical_state,
    policy_tensor_hash,
)
from yeto.rl.export import write_peft_adapter
from yeto.rl.initial_adapter import load_initial_adapter

MODEL = "org/model"
MODEL_REVISION = "a" * 40
RANK = 2
TARGETS = ["q_proj"]
TENSORS = {
    "base_model.model.layer.q_proj.lora_A.weight": torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    ),
    "base_model.model.layer.q_proj.lora_B.weight": torch.tensor(
        [[0.5, 1.0], [1.5, 2.0], [2.5, 3.0], [3.5, 4.0]],
    ),
}


def _expected_state():
    return canonical_state(
        0,
        {name: torch.zeros_like(tensor) for name, tensor in TENSORS.items()},
        base_model_revision=MODEL_REVISION,
        lora_config_hash=canonical_lora_config_hash(
            rank=RANK,
            target_modules=TARGETS,
        ),
        layout_hash=canonical_layout_hash(
            canonical_state(
                0,
                TENSORS,
                base_model_revision=MODEL_REVISION,
                lora_config_hash=canonical_lora_config_hash(
                    rank=RANK,
                    target_modules=TARGETS,
                ),
            ).specs
        ),
    )


def _write_adapter(root: Path) -> tuple[Path, str]:
    loaded = canonical_state(
        0,
        TENSORS,
        base_model_revision=MODEL_REVISION,
        lora_config_hash=canonical_lora_config_hash(
            rank=RANK,
            target_modules=TARGETS,
        ),
    )
    write_peft_adapter(
        loaded,
        root,
        base_model=MODEL,
        model_revision=MODEL_REVISION,
        rank=RANK,
    )
    (root / "yeto_rl_provenance.json").write_text(
        json.dumps(
            {
                "sync_preset": "decoupled",
                "policy_hash": policy_tensor_hash(loaded),
            }
        ),
        encoding="utf-8",
    )
    return root, directory_sha256(root)


def _load(root: Path, digest: str):
    return load_initial_adapter(
        root,
        digest,
        model=MODEL,
        expected=_expected_state(),
    )


def test_load_initial_adapter_returns_exact_version_zero_policy(tmp_path):
    root, digest = _write_adapter(tmp_path / "adapter")

    loaded = _load(root, digest)

    assert loaded.policy_version == 0
    assert loaded.specs == _expected_state().specs
    assert policy_tensor_hash(loaded) == json.loads(
        (root / "yeto_rl_provenance.json").read_text(encoding="utf-8")
    )["policy_hash"]
    for name, tensor in TENSORS.items():
        torch.testing.assert_close(loaded.tensors[name], tensor)


def test_load_initial_adapter_rejects_directory_digest_mismatch(tmp_path):
    root, _digest = _write_adapter(tmp_path / "adapter")

    with pytest.raises(ValueError, match="adapter SHA256 mismatch"):
        _load(root, "0" * 64)


def test_load_initial_adapter_rejects_model_revision_mismatch(tmp_path):
    root, _digest = _write_adapter(tmp_path / "adapter")
    config_path = root / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["revision"] = "b" * 40
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="model revision"):
        _load(root, directory_sha256(root))


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("peft_type", "IA3", "peft_type=LORA"),
        ("task_type", "SEQ_CLS", "task_type=CAUSAL_LM"),
        ("base_model_name_or_path", "org/other", "base model"),
        ("lora_alpha", RANK + 1, "LoRA alpha"),
        ("lora_dropout", 0.1, "LoRA dropout"),
        ("bias", "all", "LoRA bias"),
        ("target_modules", ["v_proj"], "target modules"),
        ("target_modules", ["q_proj", "q_proj"], "target modules"),
        ("use_rslora", True, "use_rslora"),
        ("fan_in_fan_out", True, "fan_in_fan_out"),
        ("use_dora", True, "use_dora"),
        ("rank_pattern", {"q_proj": RANK + 1}, "rank_pattern"),
        ("alpha_pattern", {"q_proj": RANK + 1}, "alpha_pattern"),
    ],
)
def test_load_initial_adapter_rejects_lora_contract_drift(
    tmp_path, field, value, match
):
    root, _digest = _write_adapter(tmp_path / "adapter")
    config_path = root / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config[field] = value
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        _load(root, directory_sha256(root))


def test_load_initial_adapter_rejects_lora_rank_drift(tmp_path):
    root, _digest = _write_adapter(tmp_path / "adapter")
    config_path = root / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["r"] = RANK + 1
    config["lora_alpha"] = RANK + 1
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="LoRA rank"):
        _load(root, directory_sha256(root))


def test_load_initial_adapter_rejects_tensor_contract_mismatch(tmp_path):
    root, _digest = _write_adapter(tmp_path / "adapter")
    save_file(
        {name: tensor for name, tensor in TENSORS.items() if "lora_B" not in name},
        root / "adapter_model.safetensors",
    )

    with pytest.raises(ValueError, match="names, shapes, or dtypes"):
        _load(root, directory_sha256(root))


def test_load_initial_adapter_rejects_nonfinite_tensors(tmp_path):
    root, _digest = _write_adapter(tmp_path / "adapter")
    tensors = dict(TENSORS)
    tensors[next(iter(tensors))] = tensors[next(iter(tensors))].clone()
    tensors[next(iter(tensors))][0, 0] = float("nan")
    save_file(tensors, root / "adapter_model.safetensors")

    with pytest.raises(ValueError, match="NaN or Inf"):
        _load(root, directory_sha256(root))


def test_load_initial_adapter_rejects_provenance_policy_mismatch(tmp_path):
    root, _digest = _write_adapter(tmp_path / "adapter")
    provenance_path = root / "yeto_rl_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["policy_hash"] = "f" * 64
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="provenance policy hash"):
        _load(root, directory_sha256(root))


def test_load_initial_adapter_requires_decoupled_export_provenance(tmp_path):
    root, _digest = _write_adapter(tmp_path / "adapter")
    provenance_path = root / "yeto_rl_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["sync_preset"] = "strict-avg"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="Decoupled RL export"):
        _load(root, directory_sha256(root))
