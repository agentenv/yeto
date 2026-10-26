"""S27 G-EC: lazy capture and the numpy transport of a verl full-parameter island.

* A local round's capture is lazy (the engine stream's deferred + lazy
  capture, ``yeto.rl.engine.full_parameter_stream``): it reads nothing from
  the trainer; materialize reads the fragments a push needs, while the trainer
  still holds them.  Every pushed byte equals what a whole capture would have
  pushed (bit for bit, also through the real syncer).
* A fragment is read on the workers (rank 0 builds the flat payload, checks it
  finite, hashes it) and moves as a numpy array; an apply sends numpy arrays
  and proves itself with a worker-side hash.  The bytes equal the old
  per-tensor export + driver flatten.

CPU only; the real-syncer test needs cargo.
"""
from __future__ import annotations

import asyncio
import hashlib
import threading

import numpy as np
import pytest
import torch

from test_s27_verl_full_parameter_sync import (  # noqa: F401 - syncer_binary is a fixture
    SHAPES, _init, _Island, _joined, _layout, _Model, _plan, _run_island, _start_syncer, needs_cargo,
    syncer_binary,
)
from yeto.rl.adapters.verl import full_parameter as fp
from yeto.rl.adapters.verl import full_parameter_stream as vfs
from yeto.rl.contracts import LocalStepReceipt
from yeto.rl.engine.full_parameter_stream import (
    DEFER_CAPTURE_ENV, FULL_CHECK_EVERY_ENV, LAZY_CAPTURE_ENV, capture_switch_on,
)

run = asyncio.run


def _train(island: _Island, group: vfs.VerlFullParameterGroup, amount: float, rollout_id: int) -> None:
    """One local step on the fake trainer, recorded like the trainer port does."""
    with torch.no_grad():
        for parameter in island.model.parameters():
            parameter.add_(amount)
    group.record_round(
        LocalStepReceipt(algorithm="grpo", learner_id=0, learner_generation=0, base_policy_version=0,
                         base_policy_hash="0" * 64, input_batch_hash="1" * 64, trajectory_ids=("t",),
                         trained_tokens=1, optimizer_steps=1, optimizer_step_succeeded=True,
                         parameter_layout_hash="2" * 64),
        vfs.VerlRoundEvidence(rollout_id, "0" * 64, "1" * 64, 1, ("t",)))


def _old_fragment(adapter, island, fragment_id) -> torch.Tensor:
    """The 7bfa57e6 read: per-tensor export, driver flatten."""
    values = fp.remote_export_for_reader(island.worker, list(adapter._names[fragment_id]))["tensors"]
    return adapter.flatten(fragment_id, values)


# ----------------------------------------------------------------------------- the worker read
def test_the_worker_read_equals_the_old_export_and_flatten_bit_for_bit():
    island = _Island(_Model(seed=3))
    _group, adapter, cut = _init(island)
    assert adapter.layout.has_split_parameters
    for fragment_id in range(adapter.layout.fragments.num_fragments):
        old = _old_fragment(adapter, island, fragment_id)
        assert cut.fragments[fragment_id].numpy().tobytes() == old.numpy().tobytes()
        assert cut.fragment_hashes[fragment_id] == hashlib.sha256(old.numpy().tobytes()).hexdigest()
        read = fp.remote_export_flat(island.worker, adapter._units[fragment_id],
                                     adapter.fragment_numel(fragment_id), False, True)
        assert "flat" not in read and read["sha256"] == cut.fragment_hashes[fragment_id] and read["finite"]


def test_a_non_finite_value_is_refused_at_capture_and_at_materialize():
    island = _Island(_Model())
    group, adapter, _cut = _init(island)
    lazy = run(adapter.capture(group, policy_version=0, local_step_generation=1, defer_payload=True, lazy=True))
    with torch.no_grad():
        island.model.model.norm.weight[1] = float("nan")
    bad = next(f for f in range(adapter.layout.fragments.num_fragments)
               if "model.norm.weight" in adapter._names[f])
    with pytest.raises(ValueError, match="NaN/Inf"):
        run(adapter.materialize(group, lazy, [bad]))
    with pytest.raises(ValueError, match="NaN/Inf"):
        run(adapter.capture(group, policy_version=0, local_step_generation=1))


def test_numpy_transport_round_trips_read_only_arrays():
    island = _Island(_Model())
    array = np.arange(8, dtype=np.float32)
    array.setflags(write=False)  # what Ray hands out of its object store
    owned = vfs._owned(array)
    assert owned.numpy().flags.writeable and not np.shares_memory(owned.numpy(), array)
    result = fp.remote_apply_fragment(island.worker, {"model.norm.weight": array[:4]}, False)
    assert result["applied"] == 1
    assert torch.equal(island.model.model.norm.weight.detach(), torch.arange(4, dtype=torch.float32))


# ----------------------------------------------------------------------------- lazy capture
def test_a_lazy_capture_reads_nothing_and_materialize_reads_the_trainer_bytes():
    island = _Island(_Model())
    group, adapter, _cut = _init(island)
    _train(island, group, 0.5, 0)
    calls = []
    original = island.call_workers
    island.call_workers = lambda fn, *args: (calls.append(fn), original(fn, *args))[1]
    lazy = run(adapter.capture(group, policy_version=0, local_step_generation=1, defer_payload=True, lazy=True))
    assert calls == []  # nothing read
    assert adapter.deferred_fragment_ids(lazy) == frozenset(range(adapter.layout.fragments.num_fragments))
    island.call_workers = original
    whole = run(adapter.capture(group, policy_version=0, local_step_generation=1))
    # Labels: unique per capture, never a content hash; the cut identity differs.
    again = run(adapter.capture(group, policy_version=0, local_step_generation=1, defer_payload=True, lazy=True))
    assert set(lazy.fragment_hashes).isdisjoint(whole.fragment_hashes)
    assert set(lazy.fragment_hashes).isdisjoint(again.fragment_hashes)
    assert lazy.policy_hash != whole.policy_hash != again.policy_hash
    described = run(adapter.materialize(group, again, [2, 0]))
    assert sorted(described) == [0, 2]
    for fragment_id in (0, 2):
        assert torch.equal(again.fragments[fragment_id], whole.fragments[fragment_id])
        assert described[fragment_id] == vfs.VerlFragmentContent(
            fragment_id, adapter.fragment_numel(fragment_id), whole.fragment_hashes[fragment_id])
    assert adapter.deferred_fragment_ids(again) == frozenset(range(adapter.layout.fragments.num_fragments)) - {0, 2}
    # The cut identity does not change when fragments are read.
    assert again.fragment_hashes[0] != whole.fragment_hashes[0]


def test_a_lazy_push_equals_the_whole_capture_push_and_needs_its_descriptor():
    island = _Island(_Model())
    group, adapter, initial = _init(island)
    anchor = adapter.fragment_sink(initial).adopt(initial, 1, 0, initial.fragment_hashes[1])
    _train(island, group, 0.125, 0)
    whole = run(adapter.capture(group, policy_version=0, local_step_generation=1))
    lazy = run(adapter.capture(group, policy_version=0, local_step_generation=1, defer_payload=True, lazy=True))
    with pytest.raises(ValueError, match="not materialized"):
        adapter.delta_parts_from_authoritative(anchor, lazy, 1)
    descriptor = run(adapter.materialize(group, lazy, [1]))[1]
    with pytest.raises(ValueError, match="needs its content descriptor"):
        adapter.delta_parts_from_authoritative(anchor, lazy, 1)
    with pytest.raises(ValueError, match="differs from its content descriptor"):
        adapter.delta_parts_from_authoritative(anchor, lazy, 1, local_descriptor=vfs.VerlFragmentContent(
            1, descriptor.numel, "0" * 64))
    pushed = _joined(adapter.delta_parts_from_authoritative(anchor, lazy, 1, local_descriptor=descriptor)())
    assert pushed == _joined(adapter.delta_parts_from_authoritative(anchor, whole, 1)())
    with pytest.raises(ValueError, match="never read"):
        adapter.fragment_parts(run(adapter.capture(group, policy_version=0, local_step_generation=1,
                                                   defer_payload=True, lazy=True)), 0)


def test_materialize_refuses_once_the_trainer_moved():
    island = _Island(_Model())
    group, adapter, initial = _init(island)
    _train(island, group, 0.25, 0)
    lazy = run(adapter.capture(group, policy_version=0, local_step_generation=1, defer_payload=True, lazy=True))
    # An apply of fragment 3 moves only fragment 3 (and fragments sharing its parameters).
    stored = adapter.fragment_sink(initial)(3, 1, torch.zeros(adapter.fragment_numel(3)).numpy().tobytes())
    run(adapter.apply(group, adapter.assemble_mixed_target(lazy, target_policy_version=1,
                                                           authoritative_fragments=[stored])))
    with pytest.raises(RuntimeError, match="no longer holds fragment 3"):
        run(adapter.materialize(group, lazy, [3]))
    untouched = next(f for f in range(adapter.layout.fragments.num_fragments)
                     if f != 3 and f not in adapter._sharing_fragments([3]))
    run(adapter.materialize(group, lazy, [untouched]))  # the trainer still holds it
    other = next(f for f in range(adapter.layout.fragments.num_fragments)
                 if f not in (3, untouched) and f not in adapter._sharing_fragments([3]))
    _train(island, group, 0.25, 1)  # a local round: nothing of the old cut may be read now
    with pytest.raises(RuntimeError, match="no longer holds fragment"):
        run(adapter.materialize(group, lazy, [other]))


def test_an_apply_on_a_lazy_resident_writes_the_same_bits_as_on_a_whole_one():
    """A split parameter is written whole: the slices outside the replaced
    fragment come from the trainer (lazy) or from the cut (whole); the
    trainers end bit-identical."""
    results = []
    for lazy in (False, True):
        island = _Island(_Model())
        group, adapter, initial = _init(island)
        _train(island, group, 0.375, 0)
        cut = run(adapter.capture(group, policy_version=0, local_step_generation=1,
                                  defer_payload=lazy, lazy=lazy))
        split = next(f for f in range(adapter.layout.fragments.num_fragments) if adapter._sharing_fragments([f]))
        new = torch.linspace(-2, 2, adapter.fragment_numel(split))
        stored = adapter.fragment_sink(initial)(split, 1, new.numpy().tobytes())
        target = adapter.assemble_mixed_target(cut, target_policy_version=1, authoritative_fragments=[stored])
        written = run(adapter.apply(group, target))
        assert written == len(fp.fragment_tensor_names(adapter.layout, split))
        assert adapter.resident is target
        results.append({name: value.numpy().tobytes() for name, value in island.model.values().items()})
    assert results[0] == results[1]


def test_the_switches_default_on_and_turn_off(monkeypatch):
    stream = vfs.VerlFullParameterRoleStream.__new__(vfs.VerlFullParameterRoleStream)
    assert stream._supports_deferred_capture() and stream._supports_lazy_capture()
    monkeypatch.setenv(LAZY_CAPTURE_ENV, "0")
    assert not capture_switch_on(LAZY_CAPTURE_ENV) and not stream._supports_lazy_capture()
    assert stream._supports_deferred_capture()
    monkeypatch.setenv(DEFER_CAPTURE_ENV, "0")
    assert not stream._supports_deferred_capture()


# ----------------------------------------------------------------------------- through the real syncer
@needs_cargo
@pytest.mark.parametrize("switches", [{}, {FULL_CHECK_EVERY_ENV: "3"}, {LAZY_CAPTURE_ENV: "0"}],
                         ids=["lazy", "lazy-full-check-3", "whole"])
def test_asynchronous_rounds_push_the_whole_capture_bytes(syncer_binary, tmp_path, monkeypatch, switches):
    """Two verl islands, the launcher's async defaults: every push from a lazy
    cut reads exactly the bytes the trainer held at that cut's capture."""
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "verl-lazy-capture-test-key")
    monkeypatch.setenv("TOKIO_WORKER_THREADS", "4")
    for key, value in switches.items():
        monkeypatch.setenv(key, value)
    expected: dict[int, list[bytes]] = {}
    checked: list[tuple[int, bool]] = []
    lock = threading.Lock()
    capture, delta = vfs.VerlChunkedFullParameterAdapter.capture, vfs.VerlChunkedFullParameterAdapter.delta_parts_from_authoritative

    async def recording_capture(self, group, **kwargs):
        cut = await capture(self, group, **kwargs)
        held = {name: value.detach().clone() for name, value in group.island.model.named_parameters()}
        with lock:
            expected[id(cut)] = [self.flatten(f, held).numpy().tobytes()
                                 for f in range(self.layout.fragments.num_fragments)]
        return cut

    def checking_delta(self, stored, local, local_fragment_id, **kwargs):
        parts = delta(self, stored, local, local_fragment_id, **kwargs)
        with lock:
            assert local.fragments[local_fragment_id].numpy().tobytes() == expected[id(local)][local_fragment_id]
            checked.append((local_fragment_id, bool(local.lazy_ids)))
        return parts

    monkeypatch.setattr(vfs.VerlChunkedFullParameterAdapter, "capture", recording_capture)
    monkeypatch.setattr(vfs.VerlChunkedFullParameterAdapter, "delta_parts_from_authoritative", checking_delta)
    rounds, fragments = 3, len(_layout().fragments.fragments)
    process, port, profile = _start_syncer(syncer_binary, tmp_path, fragments=fragments, rounds=rounds, extra=(
        "--rl-sync-full-parameter-rounds", "async", "--rl-sync-final-cut", "download"))
    results: dict = {}
    try:
        threads = [threading.Thread(target=_run_island, args=(
            i, _plan(i, port, profile, rounds=rounds, fragments=fragments, async_rounds=True,
                     matrix_merge="rda", pipeline=2, final_cut="download"), 20 * rounds, results))
            for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
        assert process.wait(timeout=30) == 0
    finally:
        if process.poll() is None:
            process.terminate()
        log = process.communicate(timeout=10)[0]
    for learner_id in range(2):
        assert not isinstance(results.get(learner_id), BaseException), (results.get(learner_id), log[-3000:])
    for name in SHAPES:  # both downloaded the same final cut
        assert torch.equal(results[0]["values"][name], results[1]["values"][name]), name
    lazy_pushes = sum(1 for _fragment, lazy in checked if lazy)
    rounds_lazy = [fields.get("sync/capture_lazy", False) for learner_id in range(2)
                   for event, fields in results[learner_id]["events"] if event == "rl_full_parameter_round"]
    assert checked and len(checked) >= rounds * fragments
    if switches.get(LAZY_CAPTURE_ENV) == "0":
        assert lazy_pushes == 0 and not any(rounds_lazy)
    else:
        assert lazy_pushes > 0 and any(rounds_lazy)
        if FULL_CHECK_EVERY_ENV in switches:
            assert not all(rounds_lazy)  # every third local round reads and hashes everything
