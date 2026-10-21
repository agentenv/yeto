"""rl-publish-fastpath for verl: the trainer-side digest equals the driver-side
full-export values bit for bit, and the publishers use it without tensors."""

from __future__ import annotations

import torch

from yeto.rl.adapters.verl import publish as pub
from yeto.rl.adapters.verl.ports_impl import (VerlPolicyState, VerlResidentState, digest_lora_tensors,
                                              sent_checksum)
from yeto.rl.core import CanonicalTensorSpec, canonical_layout_hash, canonical_state, policy_tensor_hash
from yeto.rl.engine.policy_digest import is_resident
from yeto.rl.engine.trainable_state import TrainableState

NAMES = ("model.layers.0.q_proj.lora_A.weight", "model.layers.0.q_proj.lora_B.weight")
SPECS = (CanonicalTensorSpec(NAMES[0], (4, 8), "float32", 32),
         CanonicalTensorSpec(NAMES[1], (8, 4), "float32", 32))
IDS = {"base_model_revision": "a" * 40, "lora_config_hash": "b" * 64, "layout_hash": canonical_layout_hash(SPECS)}


def _tensors(seed: int = 0) -> dict:
    gen = torch.Generator().manual_seed(seed)
    # bf16 originals upcast to f32, as remote_export_lora returns them
    return {s.name: torch.randn(s.shape, generator=gen).to(torch.bfloat16).float() for s in SPECS}


class _Island:
    def __init__(self, tensors):
        self.tensors = tensors
        self.base_model_revision, self.lora_config_hash = IDS["base_model_revision"], IDS["lora_config_hash"]
        self.layout_hash, self.expected_specs = IDS["layout_hash"], SPECS
        self.exports = 0

    def export_tensors(self):
        self.exports += 1
        return {k: v.clone() for k, v in self.tensors.items()}, {}

    def export_digest(self, kwargs):
        return digest_lora_tensors(dict(self.tensors), **kwargs)


def test_trainer_digest_equals_the_full_export_values():
    tensors = _tensors()
    raw = digest_lora_tensors(tensors, **IDS, expected_specs=SPECS)
    full = canonical_state(3, tensors, **IDS, expected_specs=SPECS)
    assert raw["digest"]["policy_tensor_hash"] == policy_tensor_hash(full)
    assert raw["lora_checksum"] == pub.lora_checksum(dict(full.tensors))


def test_resident_state_crosses_no_tensors_and_publishes_the_same_checksum():
    island = _Island(_tensors(1))
    ps = VerlPolicyState(island)
    resident = ps.export_digest(policy_version=5)
    assert isinstance(resident, VerlResidentState) and is_resident(resident)
    assert island.exports == 0
    full = ps.export()
    assert resident.policy_tensor_hash() == full.policy_tensor_hash()
    assert sent_checksum(resident) == sent_checksum(full)
    assert island.exports == 1
    # a re-versioned handle keeps the checksum (bridges.with_version)
    assert sent_checksum(resident.with_version(6))[0] == sent_checksum(full)[0]


def test_materialize_falls_back_to_the_full_export_at_the_handle_version():
    island = _Island(_tensors(2))
    resident = VerlPolicyState(island).export_digest(policy_version=7)
    state = resident.materialize()
    assert isinstance(state, TrainableState) and state.policy_version == 7
    assert state.policy_tensor_hash() == resident.policy_tensor_hash()
    assert island.exports == 1


def test_island_without_a_digest_hook_falls_back():
    class Plain:
        pass

    assert VerlPolicyState(Plain()).export_digest(policy_version=0) is None


def test_driver_switch_turns_the_fast_path_off(monkeypatch):
    from yeto.rl.engine.driver import IslandDriver

    driver = IslandDriver.__new__(IslandDriver)
    driver.policy_state = VerlPolicyState(_Island(_tensors()))
    assert is_resident(driver.export_local_resident(policy_version=0))
    monkeypatch.setenv("YETO_RL_PUBLISH_FASTPATH", "0")
    assert driver.export_local_resident(policy_version=0) is None
