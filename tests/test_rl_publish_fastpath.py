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
from yeto.rl.adapters.miles import state_plugin as sp  # noqa: E402
from yeto.rl.adapters.miles.publish import (  # noqa: E402
    MilesPublisher,
    PublicationError,
    _payload_of,
    payload_digest,
)
from yeto.rl.engine.policy_digest import (  # noqa: E402
    PolicyDigest,
    PolicyDigestError,
    WeightsChanged,
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


def _step(ranks):
    """Fake ranks bypass Miles' train_one_step; bump the weights version as the
    installed wrapper does after every optimizer step."""

    for r in ranks:
        r.train_step()
    sp.bump_weights_version()


def _publisher(ps):
    return MilesPublisher(args=SimpleNamespace(), actor_model=None, rollout_executor=None,
                          inference_controller=None, export_trainer_state=ps.export)


def test_publish_check_compares_weights_version_without_export():
    ranks, group, ps = make()
    pub = _publisher(ps)
    resident = ps.export_digest(policy_version=1)
    assert resident.weights_mark == sp.current_weights_version()
    group.calls.clear()
    assert pub._check_trainer_holds(resident) == resident.policy_tensor_hash()
    assert group.calls == [sp.WEIGHTS_VERSION]  # no export, no hashing
    # with_version keeps the mark (same weights, next policy version)
    assert pub._check_trainer_holds(resident.with_version(2)) == resident.policy_tensor_hash()
    _step(ranks)
    with pytest.raises(PublicationError, match="trainer weights differ.*weights version"):
        pub._check_trainer_holds(resident)
    # the full-state path is unchanged (content comparison)
    full = ps.export(policy_version=1)
    assert pub._check_trainer_holds(full) == full.policy_tensor_hash()


def test_apply_and_restore_paths_bump_the_version():
    ranks, group, ps = make()
    resident = ps.export_digest(policy_version=0)
    before = sp.current_weights_version()["version"]
    ps.apply(ps.export(policy_version=0), optimizer="preserve", local_step=0)
    assert sp.current_weights_version()["version"] == before + 2  # one per rank
    with pytest.raises(WeightsChanged, match="weights version"):
        resident.check_holds()
    import inspect

    from yeto.rl.adapters.miles import cut_plugin

    for fn in (cut_plugin.restore_cut_shard, cut_plugin.restore_resharded_shard, sp.apply_state):
        assert "bump_weights_version()" in inspect.getsource(fn)
    assert "bump_weights_version()" in inspect.getsource(sp.install_grad_norm_recorder)


def test_wrapped_train_one_step_bumps_even_when_it_raises(monkeypatch):
    import sys
    import types

    calls = []

    def original(*a, **k):
        calls.append(1)
        if k.get("fail"):
            raise RuntimeError("boom")
        return (0.0, 1.0)

    fake = types.ModuleType("miles.backends.megatron_utils.model")
    fake.train_one_step = original
    for name in ("miles", "miles.backends", "miles.backends.megatron_utils"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "miles.backends.megatron_utils.model", fake)
    monkeypatch.setattr(sys.modules["miles.backends.megatron_utils"], "model", fake, raising=False)
    monkeypatch.setattr(sp, "_RECORDER_INSTALLED", False)
    monkeypatch.setattr(sp, "_record_applied_lr", lambda *a: None)
    monkeypatch.setattr(sp, "_arm_grad_audit", lambda *a: None)
    monkeypatch.setattr(sp, "_record_step_losses", lambda *a: None)
    assert sp.install_grad_norm_recorder()
    v0 = sp.current_weights_version()["version"]
    fake.train_one_step()
    assert sp.current_weights_version()["version"] == v0 + 1
    with pytest.raises(RuntimeError):
        fake.train_one_step(fail=True)
    assert sp.current_weights_version()["version"] == v0 + 2


def test_replaced_trainer_process_falls_back_to_content_check(monkeypatch):
    ranks, group, ps = make()
    resident = ps.export_digest(policy_version=3)
    monkeypatch.setattr(sp, "_WEIGHTS_PROCESS_ID", "rebuilt-process")
    monkeypatch.setattr(sp, "_WEIGHTS_VERSION", 0)
    group.calls.clear()
    resident.check_holds()  # same content: accepted after a trainer-side re-hash
    assert group.calls == [sp.WEIGHTS_VERSION, sp.EXPORT_DIGEST]
    for r in ranks:
        r.train_step()
    with pytest.raises(WeightsChanged, match="rebuilt trainer holds"):
        resident.check_holds()


def test_missing_mark_is_refused():
    ranks, group, ps = make()
    from dataclasses import replace

    resident = replace(ps.export_digest(policy_version=0), weights_mark=None)
    with pytest.raises(PublicationError, match="no trainer weights version"):
        _publisher(ps)._check_trainer_holds(resident)


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
        TrainerResidentState("full", REV, CFG, "c" * 64, 0, d, lambda v: None)


# E1: the digest path stages the tensors in one reusable host buffer
def _mixed_tensors(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "b.bf16": torch.randn(8, 3, generator=g).to(torch.bfloat16),
        "a.f32_transposed": torch.randn(5, 7, generator=g).t(),
        "c.f16": torch.randn(4, generator=g).to(torch.float16),
        "d.scalar": torch.tensor(1.5),
    }


def test_staged_copies_are_the_per_tensor_copies_bit_for_bit():
    from yeto.rl.engine.policy_digest import HostStaging, stage_f32_on_host

    tensors = _mixed_tensors()
    plain = stage_f32_on_host(tensors)
    staging = HostStaging(pin=False)
    staged = stage_f32_on_host(tensors, staging)
    assert list(staged) == sorted(tensors) == list(plain)
    for name, value in tensors.items():
        want = value.detach().to("cpu", torch.float32).contiguous()
        for got in (plain[name], staged[name]):
            assert got.dtype == torch.float32 and got.is_contiguous() and got.shape == want.shape
            assert torch.equal(got.view(torch.int32) if got.dim() else got, want.view(torch.int32) if want.dim() else want)
    kw = dict(base_model_revision=REV, lora_config_hash=CFG, layout_hash="0" * 64)
    a, b = digest_canonical_tensors(staged, **kw), digest_canonical_tensors(plain, **kw)
    assert (a.policy_tensor_hash, a.payload_hash, a.payload_bytes, a.specs) == \
        (b.policy_tensor_hash, b.payload_hash, b.payload_bytes, b.specs)
    # the buffer is reused (and grown) by the next call; each call's values are its own
    bigger = {**_mixed_tensors(1), "e.more": torch.randn(64)}
    again = stage_f32_on_host(bigger, staging)
    assert all(torch.equal(again[n], stage_f32_on_host(bigger)[n]) for n in bigger)


@pytest.mark.parametrize("staged", [False, True])
def test_staging_names_the_first_non_finite_tensor(staged):
    from yeto.rl.engine.policy_digest import HostStaging, NonFiniteTensor, stage_f32_on_host

    tensors = _mixed_tensors()
    tensors["c.f16"][1] = float("inf")
    tensors["b.bf16"][0, 0] = float("nan")
    with pytest.raises(NonFiniteTensor, match="'b.bf16' contains NaN or Inf"):
        stage_f32_on_host(tensors, HostStaging(pin=False) if staged else None)


def test_export_digest_is_stable_across_rounds_with_the_reused_buffer():
    ranks, group, ps = make()
    for r in ranks:
        r.train_step()
    first = ps.export_digest(policy_version=2).policy_tensor_hash()
    assert ps.export_digest(policy_version=2).policy_tensor_hash() == first == \
        ps.export(policy_version=2).policy_tensor_hash()


# E1: sync colocated mode hashes the staged policy while the trainer offloads
def test_deferred_digest_is_the_eager_digest_and_is_collected_once():
    ranks, group, ps = make()
    for r in ranks:
        r.train_step()
    eager = ps.export_digest(policy_version=2)
    group.calls.clear()
    deferred = ps.export_digest(policy_version=2, deferred=True)
    assert group.calls == [sp.EXPORT_DIGEST] and not deferred.digest.is_resolved
    assert deferred.tensor_names == eager.tensor_names and deferred.layout_hash == eager.layout_hash
    for r in ranks:  # the trainer trains on (or offloads): the staged copy is hashed
        r.train_step()
    assert deferred.policy_tensor_hash() == eager.policy_tensor_hash()
    assert _payload_of(deferred) == _payload_of(eager)
    assert deferred.digest.to_wire() == {**eager.digest.to_wire(), "hash_seconds": deferred.digest.hash_seconds}
    assert group.calls == [sp.EXPORT_DIGEST, sp.COLLECT_DIGEST]
    deferred.policy_tensor_hash()
    assert group.calls == [sp.EXPORT_DIGEST, sp.COLLECT_DIGEST]  # resolved once


def test_a_new_export_waits_for_the_deferred_hash_of_the_reused_buffer():
    ranks, group, ps = make()
    first = ps.export_digest(policy_version=1)
    deferred = ps.export_digest(policy_version=1, deferred=True)
    for r in ranks:
        r.train_step()
    other = ps.export_digest(policy_version=2)  # overwrites the staging buffer
    assert other.policy_tensor_hash() != first.policy_tensor_hash()
    assert deferred.policy_tensor_hash() == first.policy_tensor_hash()


def test_collecting_another_version_is_refused():
    ranks, group, ps = make()
    ps.export_digest(policy_version=1, deferred=True)
    with pytest.raises(Exception, match="v1, not v3"):
        ps._collect_digest(3)


def test_the_driver_defers_only_for_a_colocated_offloading_trainer(monkeypatch):
    ranks, group, ps = make()
    driver = _Driver(ps, fast=False)
    from yeto.rl.engine.driver import defer_digest

    assert defer_digest(driver) is False
    driver.colocated = driver.publish_offloaded = True
    state = IslandDriver.export_local_resident(driver, policy_version=0)
    assert not state.digest.is_resolved
    monkeypatch.setenv("YETO_RL_DEFERRED_DIGEST", "0")
    assert defer_digest(driver) is False


# E1: which publications are verified (policy hash + engine read-back)
def _verify_driver(ps, *, every, published, cut_due=lambda v: False, all_rounds=False):
    from yeto.rl.engine import driver as driver_mod

    sync = SimpleNamespace(round_cuts=SimpleNamespace(every=every), cut_due=cut_due)
    return SimpleNamespace(policy_state=ps, sync=sync, published_version=published,
                           verify_every_round=all_rounds), driver_mod.verify_publication


def test_cut_first_and_every_k_rounds_are_verified_and_the_switch_verifies_all():
    ranks, group, ps = make()
    d, verify = _verify_driver(ps, every=4, published=None)
    assert verify(d, 1)  # the first publication
    d, verify = _verify_driver(ps, every=4, published=1, cut_due=lambda v: v == 6)
    assert [v for v in range(2, 10) if verify(d, v)] == [4, 6, 8]  # every 4th and a due cut
    d, verify = _verify_driver(ps, every=4, published=1, all_rounds=True)
    assert all(verify(d, v) for v in range(2, 10))  # --rl-verify-every-round
    d, verify = _verify_driver(ps, every=1, published=1)
    assert all(verify(d, v) for v in range(2, 10))  # --rl-cut-every 1: every round
    d, verify = _verify_driver(SimpleNamespace(), every=4, published=1)
    assert verify(d, 3)  # a policy state that cannot skip: every round


def test_an_unverified_round_still_checks_the_trainer_holds_its_policy():
    ranks, group, ps = make()
    for r in ranks:
        r.train_step()
    verified = ps.export_digest(policy_version=2)
    group.calls.clear()
    unverified = ps.export_identity(policy_version=3)
    assert sp.EXPORT_DIGEST not in group.calls  # no export, no hash
    assert unverified.verified is False and unverified.tensor_names == verified.tensor_names
    assert unverified.policy_tensor_hash() != verified.policy_tensor_hash()
    assert len(unverified.policy_tensor_hash()) == 64
    unverified.check_holds()
    _step(ranks)
    with pytest.raises(WeightsChanged):
        unverified.check_holds()


def test_an_unverified_publication_is_never_committed_as_a_cut():
    from yeto.rl.engine.controller import RebuildRefused
    from yeto.rl.engine.round_cut import CutSource

    ranks, group, ps = make()
    ps.export_digest(policy_version=1)
    unverified = ps.export_identity(policy_version=2)
    source = CutSource.__new__(CutSource)
    source.driver_ref = SimpleNamespace(published_version=2, published_state=unverified, at_safe_point=True,
                                        expected_token="yeto:2:" + "0" * 64)
    with pytest.raises(RebuildRefused, match="not verified"):
        source.context("r000002-x")
    with pytest.raises(RebuildRefused, match="not verified"):
        source.context("r000002-x", pending=(2, unverified))
