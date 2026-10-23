"""S27 G-AG: pushes computed ahead on worker threads are byte-identical.

``YETO_FULL_PARAMETER_PUSH_WORKERS`` (default 3) lets the pairing compute the
next pushes' deltas and wire encodings (``prepare_push``) on worker threads
while the pushing thread enqueues the current one, in plan order.  Covered:
the client's encode-ahead frame equals the streamed frame for every
deterministic codec (with and without tensor segments); the role stream's
prepared push sends the same bytes and keeps the same bookkeeping; the
pairing pushes in plan order, prefetches at most ``workers`` pushes, and
falls back to the streamed push when a role cannot encode ahead (error
feedback, stochastic rounding) or the variable is 1; and the per-push time
split reaches the round event.
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_rl_full_parameter_sync import _PairedRole, _role_stream_config

from yeto.fragments import MERGE_AVG, Fragment, FragmentLayout
from yeto.protocol import (
    DTYPE_F32_PUSH_I4,
    DTYPE_F32_PUSH_I8,
    DTYPE_F32_PUSH_Q4,
    DTYPE_FLAG_TENSOR_BLOCKS,
    PullRequest,
    SyncerClient,
)
from yeto.rl.adapters.miles.full_parameter.role_stream import (
    MilesFullParameterRoleStream,
    RoleSubmissionCandidate,
)
from yeto.rl.engine import full_parameter_sync as sync_module
from yeto.rl.engine.full_parameter_sync import FullParameterSyncProfile, RolePairing

SIZES = (700, 1300, 37)


def _layout():
    return FragmentLayout([Fragment(MERGE_AVG, [(f"w{f}.{i}", n) for i, n in enumerate(SIZES)])
                           for f in range(2)])


def _client(dtype: int, **kwargs) -> SyncerClient:
    client = SyncerClient(("127.0.0.1", 9), 0, _layout(), dtype=dtype, num_streams=1, connect_timeout=1,
                          max_reconnects=0, **kwargs)
    client.sent = []

    def capture(msg_type, prefix, parts, size, *, label, before_last_enqueue=None):
        data = b"".join(bytes(part) for part in parts)
        assert len(data) == size, label
        client.sent.append((msg_type, bytes(prefix), data))
        return True

    client._send_large_parts = capture
    return client


def _delta(seed: int) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal(sum(SIZES)) * 0.01).astype(np.float32)


def _parts(delta: np.ndarray):
    raw = delta.tobytes()
    return [raw[:1000], raw[1000:5001], raw[5001:]]  # unaligned to the codec blocks


@pytest.mark.parametrize("dtype", [DTYPE_F32_PUSH_I4, DTYPE_F32_PUSH_I8, DTYPE_F32_PUSH_Q4,
                                   DTYPE_F32_PUSH_I4 | DTYPE_FLAG_TENSOR_BLOCKS])
def test_an_encoded_ahead_push_frame_equals_the_streamed_frame(dtype):
    client = _client(dtype)
    assert client.can_encode_ahead
    delta = _delta(1)
    timings: dict[str, float] = {}
    assert client.push_fragment_parts(1, 3, 1, 1, 2, 2, 9, _parts(delta), timings=timings) is True
    assert {"produce_s", "push_s"} <= set(timings)
    encoded = client.encode_push_parts(1, _parts(delta), timings={})
    assert client.push_fragment_parts(1, 3, 1, 1, 2, 2, 9, (), encoded=encoded) is True
    assert client.sent[0] == client.sent[1]
    with pytest.raises(ValueError, match="another fragment"):
        client.push_fragment_parts(0, 2, 1, 0, 2, 2, 9, (), encoded=encoded)


def test_stochastic_rounding_and_f32_are_not_encoded_ahead():
    assert not _client(DTYPE_F32_PUSH_I4, push_i4_stochastic_seed=3).can_encode_ahead
    from yeto.protocol import DTYPE_F32

    plain = _client(DTYPE_F32)
    assert not plain.can_encode_ahead
    with pytest.raises(ValueError, match="cannot be encoded ahead"):
        plain.encode_push_parts(0, _parts(_delta(0)))


def _stream(dtype_name="int4", **config):
    """A Miles role stream with a fake adapter over a real (unconnected) client."""
    config = replace(_role_stream_config("actor", 30100), push_dtype=dtype_name,
                     expected_fragments=2, total_fragment_steps=8, **config)
    stream = MilesFullParameterRoleStream(config)
    stream.client = _client(config.session_dtype)
    deltas = {0: _delta(10), 1: _delta(11)}
    threads = []

    def delta_parts(anchor, cut, fragment_id, **_kwargs):
        def parts():
            threads.append(threading.current_thread().name)
            return iter(_parts(deltas[fragment_id] + np.float32(cut.shift)))
        return parts

    stream.adapter = SimpleNamespace(layout=SimpleNamespace(fragments=_layout()),
                                     delta_parts_from_authoritative=delta_parts)
    stream.threads = threads
    return stream


def _candidate(stream, step: int, cut, anchor):
    permit = PullRequest((step - 1) % 2, step, 1)
    stream._pending[permit.fragment_id] = permit
    stream._pinned_cuts[id(cut)] = (cut, 1)
    stream._pinned_anchors[id(anchor)] = (anchor, 1)
    return RoleSubmissionCandidate("actor", permit, max(0, step - 2), 1, 1, 7, sum(SIZES) * 4, cut, anchor)


def test_the_role_stream_sends_the_same_bytes_with_a_prepared_push():
    streamed, prepared = _stream(), _stream()
    cut, anchor = SimpleNamespace(shift=0.5), object()
    assert prepared.encodes_ahead
    first = streamed.push_candidate(_candidate(streamed, 1, cut, anchor))
    candidate = _candidate(prepared, 1, cut, anchor)
    ahead = prepared.prepare_push(candidate)
    second = prepared.push_candidate(candidate, ahead)
    assert streamed.client.sent == prepared.client.sent
    assert replace(first, pull_to_push_seconds=0) == replace(second, pull_to_push_seconds=0)
    assert prepared._pending == streamed._pending == {}
    for stream in (streamed, prepared):
        assert stream.push_timing_totals["count"] == 1
        assert {"delta_s", "encode_s", "enqueue_s", "total_s"} <= set(stream.push_timing_totals)
    with pytest.raises(RuntimeError, match="another candidate"):
        other = _candidate(prepared, 2, SimpleNamespace(shift=0.0), anchor)
        prepared.push_candidate(other, ahead)


def test_error_feedback_keeps_the_streamed_push():
    stream = _stream(push_int4_error_feedback="bf16")
    assert stream.error_feedback is not None and not stream.encodes_ahead


# -- the pairing ---------------------------------------------------------------------


class _PreparingRole(_PairedRole):
    """Records the thread and the order of prepares and pushes."""

    encodes_ahead = True

    def __init__(self, *args, delay=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay = delay
        self.prepared_on: dict[int, str] = {}
        self.pushed: list[tuple[int, object]] = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def prepare_push(self, candidate):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(self.delay)
        with self.lock:
            self.active -= 1
        self.prepared_on[candidate.permit.global_step] = threading.current_thread().name
        return ("bytes", candidate.permit.global_step)

    def push_candidate(self, candidate, prepared=None):
        self.pushed.append((candidate.permit.global_step, prepared))
        return super().push_candidate(candidate)


def _candidates(steps):
    return [RoleSubmissionCandidate("actor", PullRequest((s - 1) % 8, s, 1), 0, 1, 1, 7, 8, object(), object())
            for s in steps]


def _pairing(role, depth=8):
    role.fragment_versions = (0,) * 8
    profile = FullParameterSyncProfile(roles=("actor",), total_fragment_steps=64, fragments=8, local_horizon=1,
                                       pipeline_depth=depth, wait_timeout=900.0, poll_seconds=0.001,
                                       partial_quorum=True)
    return RolePairing({"actor": role}, profile)


def test_the_pairing_pushes_prepared_bytes_in_plan_order_with_bounded_lookahead(monkeypatch):
    monkeypatch.setenv(sync_module.PUSH_WORKERS_ENV, "3")
    role = _PreparingRole("actor", _candidates(range(1, 9)), [], release_broadcasts=False, delay=0.02)
    progress = _pairing(role).submit_ready()
    assert [value.global_step for value in progress["actor"].submissions] == list(range(1, 9))
    assert role.pushed == [(step, ("bytes", step)) for step in range(1, 9)]
    assert all(name.startswith("yeto-push-prepare") for name in role.prepared_on.values())
    assert 1 < role.max_active <= 3


@pytest.mark.parametrize("workers", ["1", "bad"])
def test_one_worker_or_a_bad_value_keeps_the_streamed_push(monkeypatch, workers):
    monkeypatch.setenv(sync_module.PUSH_WORKERS_ENV, workers)
    role = _PreparingRole("actor", _candidates(range(1, 4)), [], release_broadcasts=False)
    _pairing(role).submit_ready()
    assert [prepared for _, prepared in role.pushed] == [None, None, None] and role.prepared_on == {}


def test_a_failed_prepare_fail_stops_and_leaves_no_worker_running(monkeypatch):
    monkeypatch.setenv(sync_module.PUSH_WORKERS_ENV, "3")

    class _Failing(_PreparingRole):
        def prepare_push(self, candidate):
            if candidate.permit.global_step == 2:
                raise ValueError("encode failed")
            return super().prepare_push(candidate)

    role = _Failing("actor", _candidates(range(1, 7)), [], release_broadcasts=False, delay=0.01)
    pairing = _pairing(role)
    with pytest.raises(ValueError, match="encode failed"):
        pairing.submit_ready()
    assert pairing.failure is not None and role.closed
    assert [step for step, _ in role.pushed] == [1]
    assert role.active == 0
