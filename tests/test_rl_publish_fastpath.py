"""CPU tests for rl-publish-fastpath: digests computed on the trainer are
byte-identical to the driver-side ones (tape values unchanged), only digests
cross the plugin boundary, and the "trainer still holds what is published"
check keeps its meaning."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from test_rl_miles_adapter_state import (  # noqa: E402
    CFG,
    REV,
    _generic_optimizer_reset,  # noqa: F401  (autouse fixture)
    make,
)
from yeto.rl.core import canonical_layout_hash, canonical_specs, canonical_state, policy_tensor_hash  # noqa: E402
from yeto.rl.engine import bridges  # noqa: E402
from yeto.rl.engine.driver import IslandDriver  # noqa: E402
from yeto.rl.engine.miles_adapter import state_plugin as sp  # noqa: E402
from yeto.rl.engine.miles_adapter.publish import (  # noqa: E402
    MilesPublisher,
    PublicationError,
    _payload_of,
    payload_digest,
)
from yeto.rl.engine.policy_digest import (  # noqa: E402
    PolicyDigest,
    PolicyDigestError,
    TrainerResidentState,
    digest_canonical_tensors,
    is_resident,
    layout_hash_of,
)
from yeto.rl.engine.trainable_state import TrainableState  # noqa: E402

NAMES = [
    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
    "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
    "base_model.model.model.layers.1.mlp.down_proj.lora_A.weight",
    "base_model.model.model.layers.1.mlp.down_proj.lora_B.weight",
]


def _random_state(seed=0, version=3):
    g = torch.Generator().manual_seed(seed)
    shapes = [(4, 16), (32, 4), (4, 64), (16, 4)]
    tensors = {n: torch.randn(*s, generator=g) for n, s in zip(NAMES, shapes)}
    layout = canonical_layout_hash(canonical_specs(tensors))
    return TrainableState.from_lora(canonical_state(
        version, tensors, base_model_revision=REV, lora_config_hash=CFG, layout_hash=layout))


@pytest.mark.parametrize("parallel", [True, False])
def test_digest_is_byte_identical_to_driver_side_hashes(parallel):
    state = _random_state()
    digest = digest_canonical_tensors(
        state.tensors, base_model_revision=REV, lora_config_hash=CFG,
        layout_hash=state.layout_hash, parallel=parallel)
    assert digest.policy_tensor_hash == policy_tensor_hash(state.to_lora())
    assert (digest.payload_hash, digest.payload_bytes) == payload_digest(state)
    assert layout_hash_of(digest.specs) == state.layout_hash
    assert PolicyDigest.from_wire(digest.to_wire()) == digest


def test_digest_depends_on_values_not_version():
    a, b = _random_state(version=1), _random_state(version=7)
    kw = dict(base_model_revision=REV, lora_config_hash=CFG, layout_hash=a.layout_hash)
    assert digest_canonical_tensors(a.tensors, **kw).policy_tensor_hash == \
        digest_canonical_tensors(b.tensors, **kw).policy_tensor_hash
    c = _random_state(seed=1)
    assert digest_canonical_tensors(c.tensors, **kw).payload_hash != \
        digest_canonical_tensors(a.tensors, **kw).payload_hash


def test_export_digest_matches_full_export_and_ships_no_tensors():
    ranks, group, ps = make()
    for r in ranks:
        r.train_step()
    resident = ps.export_digest(policy_version=2)
    assert group.calls == [sp.EXPORT_DIGEST]
    assert is_resident(resident) and resident.policy_version == 2
    full = ps.export(policy_version=2)
    assert resident.policy_tensor_hash() == full.policy_tensor_hash()
    assert resident.layout_hash == full.layout_hash
    # the manifest / tape inputs are the same values as before
    assert _payload_of(resident) == payload_digest(full)
    assert _payload_of(full) == payload_digest(full)
    assert resident.tensor_names == full.tensor_names


def test_resident_versions_and_lazy_materialize():
    ranks, group, ps = make()
    resident = ps.export_digest(policy_version=0).with_version(5)
    assert resident.policy_version == 5
    group.calls.clear()
    full = resident.materialize()
    assert group.calls == [sp.EXPORT_STATE]
    assert full.policy_version == 5 and full.policy_tensor_hash() == resident.policy_tensor_hash()
    assert resident.policy_hash() == full.policy_hash()  # cached, no second export
    assert group.calls == [sp.EXPORT_STATE]
    stale = ps.export_digest(policy_version=0)
    for r in ranks:
        r.train_step()
    with pytest.raises(PolicyDigestError):
        stale.materialize()


def _publisher(ps):
    return MilesPublisher(args=SimpleNamespace(), actor_model=None, rollout_executor=None,
                          inference_controller=None, export_trainer_state=ps.export)


def test_publish_check_still_refuses_changed_trainer_weights():
    ranks, group, ps = make()
    pub = _publisher(ps)
    resident = ps.export_digest(policy_version=1)
    group.calls.clear()
    assert pub._check_trainer_holds(resident) == resident.policy_tensor_hash()
    assert group.calls == [sp.EXPORT_DIGEST]  # re-hashed in the trainer, no full export
    for r in ranks:
        r.train_step()
    with pytest.raises(PublicationError, match="trainer weights differ"):
        pub._check_trainer_holds(resident)
    # the full-state path is unchanged
    full = ps.export(policy_version=1)
    assert pub._check_trainer_holds(full) == full.policy_tensor_hash()


class _Driver:
    def __init__(self, ps, fast=True):
        self.policy_state = ps
        self.events = []
        if fast:
            self.export_local_resident = lambda *, policy_version: IslandDriver.export_local_resident(
                self, policy_version=policy_version)

    def export_local(self):
        return self.policy_state.export()

    def emit(self, *a, **k):
        self.events.append((a, k))


def test_local_only_sync_uses_resident_state(monkeypatch):
    ranks, group, ps = make()
    sync = bridges.LocalOnlySync(num_rollout=3)
    start = sync.start(_Driver(ps))
    assert is_resident(start.state) and start.state.policy_version == 0
    stats = SimpleNamespace()
    monkeypatch.setattr(bridges, "asdict", lambda s: {})
    boundary = sync.boundary(_Driver(ps), rollout_id=0, stats=stats)
    assert is_resident(boundary.state) and boundary.state.policy_version == 1
    assert sp.EXPORT_STATE not in group.calls
    slow = sync.boundary(_Driver(ps, fast=False), rollout_id=0, stats=stats)
    assert isinstance(slow, bridges.SyncBoundary) and not is_resident(slow.state)
    assert slow.state.policy_tensor_hash() == boundary.state.policy_tensor_hash()
    assert slow.state.policy_version == 1


def test_fastpath_switch_off_falls_back(monkeypatch):
    ranks, group, ps = make()
    monkeypatch.setenv("YETO_RL_PUBLISH_FASTPATH", "0")
    assert IslandDriver.export_local_resident(_Driver(ps, fast=False), policy_version=0) is None
    monkeypatch.delenv("YETO_RL_PUBLISH_FASTPATH")
    assert is_resident(IslandDriver.export_local_resident(_Driver(ps, fast=False), policy_version=0))
    assert IslandDriver.export_local_resident(SimpleNamespace(policy_state=object()), policy_version=0) is None


def test_resident_state_type_guards():
    d = PolicyDigest("a" * 64, "b" * 64, 8, ((NAMES[0], (2, 1)),))
    with pytest.raises(ValueError):
        TrainerResidentState("full", REV, CFG, "c" * 64, 0, d, lambda v: None, lambda: d)
