"""S27 G-CG verify-ahead on the real Miles role stream.

A broadcast's stored chunks are checked with Miles' own prepare check
(``verify_stored_fragment_chunks``) on a background thread as soon as the
fragment is stored; the apply at the next boundary names the checked
fragments (``verified_fragments``) so the trainer's prepare reads none of
their chunks.  The run with verify-ahead must give bit-identical trainer
values, syncer values and pushed deltas to the run without it, and every
claim must name a replaced fragment with its exact payload hash.

Needs Miles with ``verify_stored_fragment_chunks`` (``importorskip``).
"""

from __future__ import annotations

# ruff: noqa: I001 -- the Miles test helpers install a bridge stub before import.

import shlex
from concurrent.futures import Future

import pytest

pytest.importorskip("miles")
_state = pytest.importorskip("miles.backends.megatron_utils.full_parameter_state")
if not hasattr(_state, "verify_stored_fragment_chunks"):
    pytest.skip("needs a Miles with the verified apply (yeto/s27-sync-crit)", allow_module_level=True)

import test_s27_sync_fast_miles_role_stream as fast_run  # noqa: E402
from test_s27_sync_fast_miles_role_stream import _bits  # noqa: E402

from yeto.rl.adapters.miles.full_parameter.role_stream import RoleStreamConfig  # noqa: E402
from yeto.rl.adapters.miles.models import chunked_full_parameter as adapter_module  # noqa: E402
from yeto.rl.adapters.miles.models.chunked_full_parameter import StoredAuthoritativeFragment  # noqa: E402


class _VerifiedContractTrainer(fast_run._MilesContractTrainer):
    """The Miles contract trainer whose apply takes ``verified_fragments``:
    a claimed fragment must be one the cut replaces, with its payload hash;
    Miles' prepare then reads none of its chunks (its commit still copies)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.claimed = 0
        self.checked_in_prepare = 0

    async def apply_full_parameter_chunked_cut(self, cut, *, commit_token=None, max_commit_attempts=3,
                                               skip_resident_fragments=False, verified_fragments=None):
        claims = dict(verified_fragments or {})
        for shard in cut.shards:
            for descriptor in shard.fragments:
                resident = self.resident is not None and \
                    self.resident.get(descriptor.fragment_id) == descriptor.payload_hash
                if descriptor.fragment_id in claims:
                    assert not resident, "claimed a fragment the masters already hold"
                    assert claims.pop(descriptor.fragment_id) == descriptor.payload_hash
                    self.claimed += 1
                elif not resident:
                    self.checked_in_prepare += 1
        assert not claims, f"claimed fragments outside the cut: {claims}"
        return await super().apply_full_parameter_chunked_cut(
            cut, commit_token=commit_token, max_commit_attempts=max_commit_attempts,
            skip_resident_fragments=skip_resident_fragments)


def _run(monkeypatch, *, verify_ahead: bool):
    monkeypatch.setattr(fast_run, "_MilesContractTrainer",
                        _VerifiedContractTrainer if verify_ahead else fast_run._MilesContractTrainer)
    monkeypatch.setenv(adapter_module.VERIFY_AHEAD_ENV, "1" if verify_ahead else "0")
    return fast_run._run(monkeypatch, fast=True)


def test_verify_ahead_keeps_every_value_bit_identical_and_claims_only_checked_fragments(monkeypatch):
    checked = _run(monkeypatch, verify_ahead=False)
    ahead = _run(monkeypatch, verify_ahead=True)

    for run in (checked, ahead):
        assert run.syncer.committed == list(range(1, fast_run.ROUNDS * fast_run.P + 1))
        assert run.stream.final_applied and run.stream.final_acknowledged
    assert _bits(ahead.trainer.values) == _bits(checked.trainer.values)
    assert _bits(ahead.syncer.global_) == _bits(checked.syncer.global_)
    assert [(s, a, d.view(fast_run.torch.int32).tolist()) for s, a, d in ahead.syncer.pushed] == [
        (s, a, d.view(fast_run.torch.int32).tolist()) for s, a, d in checked.syncer.pushed]

    counters = ahead.stream.verify_ahead_counters
    replaced = ahead.trainer.claimed + ahead.trainer.checked_in_prepare
    print(f"verify-ahead: {counters}; trainer claims {ahead.trainer.claimed}, "
          f"prepare checks {ahead.trainer.checked_in_prepare} of {replaced}")
    assert ahead.trainer.claimed == counters["applied"] >= 1
    # A fragment whose check had not ended is checked by the trainer as before;
    # the final cut's fragments are claimed like any broadcast.
    assert counters["pending"] <= ahead.trainer.checked_in_prepare
    assert checked.stream.verify_ahead_counters == {"applied": 0, "pending": 0}
    # The boundary events report the claims of their boundary's applies.
    rounds = [fields for event, fields in ahead.driver.events if event == "rl_full_parameter_round"]
    assert rounds and all("sync/apply_verified_ahead" in fields for fields in rounds)
    assert 1 <= sum(fields["sync/apply_verified_ahead"] for fields in rounds) <= counters["applied"]


_DESCRIPTOR = type("Descriptor", (), {"payload_hash": "h" * 64})()


def _stored(future):
    return StoredAuthoritativeFragment(
        fragment_id=3, version=7, parameter_layout_hash="l" * 64, topology=None, plan_hash="p" * 64,
        descriptor=_DESCRIPTOR, refs=[object()], wire_payload_hash="w" * 64, verification=future)


def test_a_running_check_leaves_the_fragment_to_the_trainer_and_a_failed_one_fails_the_apply():
    from yeto.rl.adapters.miles.full_parameter.role_stream import MilesFullParameterRoleStream

    stream = MilesFullParameterRoleStream.__new__(MilesFullParameterRoleStream)
    stream.config = RoleStreamConfig.__new__(RoleStreamConfig)
    object.__setattr__(stream.config, "role", "actor")
    stream.verify_ahead_counters = {"applied": 0, "pending": 0}

    running = Future()
    done = Future()
    done.set_result(None)
    assert stream._verified_ahead({3: _stored(running), 4: _stored(None), 5: _stored(done)}) == {5: "h" * 64}
    assert stream.verify_ahead_counters == {"applied": 1, "pending": 1}

    failed = Future()
    failed.set_exception(ValueError("full-parameter fragment hash changed"))
    with pytest.raises(RuntimeError, match="failed its off-trainer chunk check") as caught:
        stream._verified_ahead({3: _stored(failed)})
    assert isinstance(caught.value.__cause__, ValueError)


def test_the_stored_fragment_equality_ignores_its_check():
    done = Future()
    done.set_result(None)
    a, b = _stored(done), _stored(None)
    b.refs[:] = a.refs
    assert a == b


def _plan(extra=()):
    from test_rl_full_parameter_async import ASYNC
    from test_rl_full_parameter_async import _plan as plan

    return plan(ASYNC + tuple(extra))


def _flag(command: str, flag: str):
    tokens = shlex.split(command)
    return tokens[tokens.index(flag) + 1] if flag in tokens else None


@pytest.mark.parametrize(("extra", "island"), [((), None), (("--rl-sync-verify-ahead", "off"), "off")])
def test_the_launcher_turns_verify_ahead_off_only_on_request(extra, island):
    _args, plan = _plan(extra)
    for request in plan["island_requests"]:
        assert _flag(request["learner_command"], "--sweep-verify-ahead") == island


def test_the_streaming_inbound_sink_starts_the_check_of_each_new_fragment():
    """The real client streams a broadcast through ``MilesRayInboundChunkSink``;
    the stored fragment carries a check that passes on the stored bytes and
    names its payload hash, and a replay starts no second check."""
    import torch
    from test_rl_miles_chunked_full_parameter import FakeRay, _cut, component, manifest

    from yeto.protocol import MSG_BCAST_FRAGMENT
    from yeto.rl.adapters.miles.full_parameter.inbound_sink import MilesRayInboundChunkSink
    from yeto.rl.adapters.miles.models.chunked_full_parameter import (
        MilesChunkedFullParameterAdapter,
        StoredFragmentVerifier,
    )

    ray = FakeRay()
    adapter = MilesChunkedFullParameterAdapter.create(
        (manifest(0, (3,)), manifest(1, (2,))), algorithm="grpo", components=(component(),),
        minimum_fragments=2, max_fragment_bytes=64, max_chunk_bytes=16)
    anchor = adapter.validate_cut(_cut(adapter, {0: 0, 1: 0}, policy_version=0, generation=0, ray=ray), 0, 0)
    sink = adapter.fragment_sink(anchor, ray_module=ray)
    sink.verifier = StoredFragmentVerifier(ray_module=ray)
    streaming = MilesRayInboundChunkSink(sink)
    numel = next(d.numel for shard in anchor.transport_cut.shards for d in shard.fragments if d.fragment_id == 0)
    payload = torch.arange(numel, dtype=torch.float32).numpy().tobytes()
    try:
        transaction = streaming.begin_message(1, len(payload))
        streaming.bind_fragment(transaction, MSG_BCAST_FRAGMENT, 0, 1, len(payload))
        streaming.consume_chunk(transaction, 0, payload)
        stored = streaming.finish_message(transaction).data
        assert stored is sink._stored[(0, 1)]
        stored.verification.result(timeout=30)
        assert stored.verified_payload_hash() == stored.descriptor.payload_hash
        replay = streaming.begin_message(2, len(payload))
        streaming.bind_fragment(replay, MSG_BCAST_FRAGMENT, 0, 1, len(payload))
        streaming.consume_chunk(replay, 0, payload)
        assert streaming.finish_message(replay).data is stored
    finally:
        sink.verifier.close()
