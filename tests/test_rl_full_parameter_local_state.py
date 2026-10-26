"""CPU tests for the no-sync full-parameter policy state (T-FPUB, L5.4 run i).

GPU run s24-l54-pair-20261014i: a full-parameter trainer island without an outer
sync published through the LoRA digest export and failed with "Megatron produced
an empty or duplicate LoRA mapping".  A full-parameter island now gets
``MilesFullParameterLocalPolicyState``: a trainer-side content digest of every
model parameter, no LoRA export.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from test_rl_miles_adapter_state import FakeTrainGroup  # noqa: E402
from yeto.rl.adapters.miles import state_plugin as sp  # noqa: E402
from yeto.rl.adapters.miles.entry import _policy_state_for  # noqa: E402
from yeto.rl.adapters.miles.publish import MilesPublisher, PublicationError, _payload_of  # noqa: E402
from yeto.rl.adapters.miles.state import (  # noqa: E402
    MilesFullParameterLocalPolicyState,
    MilesPolicyState,
    PolicyStateError,
)
from yeto.rl.engine import bridges  # noqa: E402
from yeto.rl.engine.driver import IslandDriver  # noqa: E402
from yeto.rl.engine.policy_digest import UnsupportedLayoutError, is_resident  # noqa: E402
from yeto.rl.engine.trainable_state import LAYOUT_FRAGMENTS  # noqa: E402

REV = "r" * 40
CFG = "c" * 64
LAYOUT = "d" * 64


class FullRank:
    """A full-parameter Megatron rank: plain bf16 weights, no LoRA adapters."""

    def __init__(self, main: bool, seed: int = 0) -> None:
        torch.manual_seed(seed)
        self.model = [torch.nn.Linear(4, 3).to(torch.bfloat16)]
        self.args = SimpleNamespace(tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
                                    expert_model_parallel_size=1, hf_checkpoint="/model")
        self._is_first_replica_megatron_main_rank = main

    def train_step(self) -> None:
        with torch.no_grad():
            self.model[0].weight.add_(1.0)
        sp.bump_weights_version()


def _ranks(n=2):
    return [FullRank(main=(i == 0), seed=i) for i in range(n)]


def _state(ranks):
    group = FakeTrainGroup(ranks)
    return group, MilesFullParameterLocalPolicyState(
        actor_model=group, base_model_revision=REV, config_hash=CFG, expected_layout_hash=LAYOUT)


class _Driver:
    def __init__(self, ps):
        self.policy_state = ps
        self.events = []

    def export_local_resident(self, *, policy_version):
        return IslandDriver.export_local_resident(self, policy_version=policy_version)

    def export_local(self):
        return self.policy_state.export()

    def emit(self, *a, **k):
        self.events.append((a, k))


def test_lora_state_on_a_full_parameter_trainer_reproduces_the_l54_failure(monkeypatch):
    """Before the fix the no-sync island used the LoRA state: the bridge finds no
    adapter on a full-parameter model and the digest export fails."""

    hf_export = ModuleType("miles.backends.megatron_utils.hf_export")
    no_adapters = SimpleNamespace(build_adapter_conversion_tasks=lambda model: {})
    hf_export._get_hf_bridge = lambda path: SimpleNamespace(_model_bridge=no_adapters)
    monkeypatch.setitem(sys.modules, "miles.backends.megatron_utils.hf_export", hf_export)
    group = FakeTrainGroup(_ranks())
    lora = MilesPolicyState(actor_model=group, base_model_revision=REV, config_hash=CFG)
    with pytest.raises(sp.StatePluginError, match="empty or duplicate LoRA mapping"):
        IslandDriver.export_local_resident(_Driver(lora), policy_version=0)


def test_policy_state_selection_follows_the_parameter_mode():
    group = FakeTrainGroup(_ranks())
    kwargs = dict(actor_model=group, base_model_revision=REV, config_hash=CFG,
                  expected_layout_hash=LAYOUT, runner=None)
    assert isinstance(_policy_state_for(None, full_parameter=True, **kwargs),
                      MilesFullParameterLocalPolicyState)
    lora = _policy_state_for(None, **kwargs)
    assert type(lora) is MilesPolicyState


def test_full_parameter_digest_is_resident_and_content_addressed():
    ranks = _ranks()
    group, ps = _state(ranks)
    first = ps.export_digest(policy_version=2)
    assert group.calls == [sp.EXPORT_FULL_DIGEST]
    assert is_resident(first) and first.layout == LAYOUT_FRAGMENTS
    assert first.policy_version == 2 and first.layout_hash == LAYOUT == ps.layout_hash
    assert first.weights_mark == sp.current_weights_version()
    # payload bytes: every rank's bf16 weight (3x4) and bias (3)
    assert _payload_of(first)[1] == 2 * (12 + 3) * 2
    # same weights -> same hash, whatever the version; a rebuilt state agrees
    assert ps.export(policy_version=5).policy_tensor_hash() == first.policy_tensor_hash()
    assert _state(_ranks())[1].export_digest(policy_version=0).policy_tensor_hash() == \
        first.policy_tensor_hash()
    # rank order is part of the identity
    assert _state(list(reversed(_ranks())))[1].export_digest(policy_version=0).policy_tensor_hash() != \
        first.policy_tensor_hash()
    ranks[1].train_step()
    assert ps.export_digest(policy_version=2).policy_tensor_hash() != first.policy_tensor_hash()


def test_publisher_holds_check_and_refusals():
    ranks = _ranks()
    group, ps = _state(ranks)
    publisher = MilesPublisher(args=SimpleNamespace(), actor_model=None, rollout_executor=None,
                               inference_controller=None, export_trainer_state=ps.export)
    resident = ps.export_digest(policy_version=1)
    group.calls.clear()
    assert publisher._check_trainer_holds(resident) == resident.policy_tensor_hash()
    assert group.calls == [sp.WEIGHTS_VERSION]
    ranks[0].train_step()
    with pytest.raises(PublicationError, match="trainer weights differ"):
        publisher._check_trainer_holds(resident)
    with pytest.raises(UnsupportedLayoutError):
        resident.materialize()
    with pytest.raises(PolicyStateError, match="never applies"):
        ps.apply(resident, optimizer="preserve", local_step=0)
    with pytest.raises(PolicyStateError, match="bound layout hash"):
        MilesFullParameterLocalPolicyState(actor_model=group, base_model_revision=REV,
                                           config_hash=CFG, expected_layout_hash=None)


@pytest.mark.parametrize("fastpath", ["1", "0"])
def test_local_only_sync_publishes_a_full_parameter_state(monkeypatch, fastpath):
    monkeypatch.setenv("YETO_RL_PUBLISH_FASTPATH", fastpath)
    _group, ps = _state(_ranks())
    sync = bridges.LocalOnlySync(num_rollout=3)
    start = sync.start(_Driver(ps))
    assert is_resident(start.state) and start.state.policy_version == 0
    monkeypatch.setattr(bridges, "asdict", lambda s: {})
    boundary = sync.boundary(_Driver(ps), rollout_id=0, stats=SimpleNamespace())
    assert is_resident(boundary.state) and boundary.state.policy_version == 1
    assert boundary.state.policy_tensor_hash() == start.state.policy_tensor_hash()
