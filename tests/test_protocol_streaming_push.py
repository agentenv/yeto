"""Bounded outbound PUSH_FRAGMENT framing tests."""

import itertools
import queue
import struct

import pytest

from yeto.fragments import MERGE_RDA, Fragment, FragmentLayout
from yeto.protocol import (
    _CHUNK_HEAD,
    _HEADER,
    CHUNK_SIZE,
    DTYPE_F32,
    MAGIC,
    MSG_CHUNK,
    MSG_PULL_REQ,
    PartialMessageGenerationLost,
    SyncerClient,
)


def _layout(tensor_bytes: int) -> FragmentLayout:
    assert tensor_bytes % 4 == 0
    return FragmentLayout(
        [Fragment(MERGE_RDA, [("model.body.weight", tensor_bytes // 4)])]
    )


def _client(tensor_bytes: int, *, streams: int = 2) -> SyncerClient:
    client = SyncerClient(
        ("unused", 0),
        7,
        _layout(tensor_bytes),
        dtype=DTYPE_F32,
        num_streams=streams,
    )
    client._gen = 4
    client._connected.set()
    client._queues = [queue.Queue() for _ in range(streams + 1)]
    return client


def _push_kwargs() -> dict:
    return {
        "fragment_id": 0,
        "global_step": 11,
        "round_attempt": 2,
        "base_version": 9,
        "local_step": 17,
        "c_steps": 3,
        "c_tokens": 1234,
    }


def _queued_inner(client: SyncerClient) -> tuple[bytes, list[int], list[int]]:
    chunks = []
    streams = []
    envelope_sizes = []
    for stream, outbound in enumerate(client._queues):
        while not outbound.empty():
            item = outbound.get_nowait()
            wire = item.data
            magic, msg_type, payload_len = _HEADER.unpack_from(wire)
            assert (magic, msg_type) == (MAGIC, MSG_CHUNK)
            assert payload_len == len(wire) - _HEADER.size
            msg_id, total, offset = _CHUNK_HEAD.unpack_from(wire, _HEADER.size)
            data = bytes(wire[_HEADER.size + _CHUNK_HEAD.size :])
            assert len(data) <= CHUNK_SIZE
            chunks.append((msg_id, total, offset, data))
            streams.append(stream)
            envelope_sizes.append(len(wire))
    assert len({msg_id for msg_id, *_ in chunks}) == 1
    assert len({total for _, total, *_ in chunks}) == 1
    total = chunks[0][1]
    inner = bytearray(total)
    for _, _, offset, data in chunks:
        inner[offset : offset + len(data)] = data
    return bytes(inner), streams, envelope_sizes


def test_streaming_push_reassembles_to_exact_legacy_inner_frame():
    tensor = bytes(range(16))
    legacy = _client(len(tensor))
    streamed = _client(len(tensor))
    tail = bytearray(tensor[11:])

    legacy.push_fragment(**_push_kwargs(), tensor_bytes=tensor)
    assert streamed.push_fragment_parts(
        **_push_kwargs(),
        tensor_parts=[tensor[:3], memoryview(tensor)[3:11], tail],
    )
    tail[:] = b"z" * len(tail)

    legacy_inner, _, _ = _queued_inner(legacy)
    streamed_inner, _, envelope_sizes = _queued_inner(streamed)
    assert streamed_inner == legacy_inner
    assert max(envelope_sizes) <= CHUNK_SIZE + _HEADER.size + _CHUNK_HEAD.size


def test_streaming_init_reassembles_to_exact_legacy_inner_frame():
    tensor = bytes(range(16))
    legacy = _client(len(tensor))
    streamed = _client(len(tensor))
    tail = bytearray(tensor[11:])

    legacy.send_init(0, tensor)
    assert streamed.send_init_parts(
        0,
        [tensor[:3], memoryview(tensor)[3:11], tail],
    )
    tail[:] = b"z" * len(tail)

    legacy_inner, _, _ = _queued_inner(legacy)
    streamed_inner, _, envelope_sizes = _queued_inner(streamed)
    assert streamed_inner == legacy_inner
    assert max(envelope_sizes) <= CHUNK_SIZE + _HEADER.size + _CHUNK_HEAD.size


def test_streaming_push_consumes_parts_incrementally_and_stripes_bounded_chunks():
    tensor_bytes = 2 * CHUNK_SIZE
    client = _client(tensor_bytes, streams=3)
    part = b"x" * (CHUNK_SIZE // 4)
    total_parts = tensor_bytes // len(part)
    yielded = 0
    exhausted = False

    def parts():
        nonlocal yielded, exhausted
        for _ in range(total_parts):
            yielded += 1
            yield part
        exhausted = True

    calls = []

    def enqueue(stream, data, gen=None, sent=None):
        del sent
        offset = _CHUNK_HEAD.unpack_from(data, _HEADER.size)[2]
        calls.append((stream, len(data), gen, yielded, exhausted, offset))
        return True

    client._enqueue = enqueue
    assert client.push_fragment_parts(**_push_kwargs(), tensor_parts=parts())

    assert len(calls) == 3
    assert calls[0][3] < total_parts
    assert not calls[0][4]
    assert [call[0] for call in calls] == [1, 2, 3]
    assert all(call[1] <= CHUNK_SIZE + _HEADER.size + _CHUNK_HEAD.size for call in calls)
    assert all(call[2] == 4 for call in calls)
    assert [call[5] for call in calls] == [0, CHUNK_SIZE, 2 * CHUNK_SIZE]
    assert yielded == total_parts
    assert exhausted


def test_streaming_push_runs_release_barrier_before_last_envelope_is_visible():
    tensor_bytes = 2 * CHUNK_SIZE
    client = _client(tensor_bytes)
    barrier_ran = False
    observed = []

    def barrier():
        nonlocal barrier_ran
        assert not barrier_ran
        barrier_ran = True

    def enqueue(_stream, data, gen=None, sent=None):
        del gen, sent
        _, total, offset = _CHUNK_HEAD.unpack_from(data, _HEADER.size)
        observed.append((offset, barrier_ran, total))
        return True

    client._enqueue = enqueue
    part = b"x" * (CHUNK_SIZE // 4)
    parts = itertools.repeat(part, tensor_bytes // len(part))

    assert client.push_fragment_parts(
        **_push_kwargs(),
        tensor_parts=parts,
        before_last_enqueue=barrier,
    )

    assert barrier_ran
    assert len(observed) == 3
    assert [barrier for _, barrier, _ in observed] == [False, False, True]


def test_streaming_push_release_barrier_failure_poisons_partial_generation():
    tensor_bytes = 2 * CHUNK_SIZE
    client = _client(tensor_bytes)
    queued = []
    client._enqueue = lambda _stream, data, **_kwargs: queued.append(data) or True
    part = b"x" * (CHUNK_SIZE // 4)
    parts = itertools.repeat(part, tensor_bytes // len(part))

    def fail_release():
        raise RuntimeError("base release failed")

    with pytest.raises(RuntimeError, match="base release failed"):
        client.push_fragment_parts(
            **_push_kwargs(),
            tensor_parts=parts,
            before_last_enqueue=fail_release,
        )

    assert len(queued) == 2
    assert not client._connected.is_set()
    assert client._failure.is_set()


@pytest.mark.parametrize(
    ("parts", "error"),
    [
        ([b"x" * 12], "has 12 delta bytes, expected 16"),
        ([b"x" * 20], "exceeds expected 16 delta bytes"),
        ([memoryview(bytearray(32))[::2]], "not C-contiguous"),
        ([object()], "not bytes-like"),
    ],
)
def test_streaming_push_rejects_invalid_part_streams_without_a_complete_frame(
    parts, error
):
    client = _client(16)
    with pytest.raises((TypeError, ValueError), match=error):
        client.push_fragment_parts(**_push_kwargs(), tensor_parts=parts)
    assert all(outbound.empty() for outbound in client._queues)


def test_streaming_push_drop_does_not_consume_an_outage_or_post_drop_parts():
    disconnected = _client(16)
    disconnected._connected.clear()

    def must_not_iterate():
        raise AssertionError("disconnected push consumed its parts")
        yield b""

    assert not disconnected.push_fragment_parts(
        **_push_kwargs(), tensor_parts=must_not_iterate()
    )

    tensor_bytes = 2 * CHUNK_SIZE
    dropped = _client(tensor_bytes)
    part = b"x" * (CHUNK_SIZE // 4)
    total_parts = tensor_bytes // len(part)
    yielded = 0

    def tracking_parts():
        nonlocal yielded
        for _ in range(total_parts):
            yielded += 1
            yield part

    calls = []

    def drop_first(stream, data, gen=None, sent=None):
        del stream, data, sent
        calls.append(gen)
        return False

    dropped._enqueue = drop_first
    assert not dropped.push_fragment_parts(
        **_push_kwargs(), tensor_parts=tracking_parts()
    )

    assert calls == [4]
    assert yielded < total_parts
    assert dropped._connected.is_set()
    assert not dropped._failure.is_set()


def test_streaming_push_late_producer_failure_poisons_partial_generation():
    tensor_bytes = 2 * CHUNK_SIZE
    client = _client(tensor_bytes)
    queued = []

    def enqueue(stream, data, gen=None, sent=None):
        del stream, sent
        queued.append((gen, len(data)))
        return True

    client._enqueue = enqueue

    class ProducerFailure(RuntimeError):
        pass

    def failing_parts():
        yield b"x" * CHUNK_SIZE
        raise ProducerFailure("late producer failure")

    with pytest.raises(ProducerFailure, match="late producer failure") as caught:
        client.push_fragment_parts(**_push_kwargs(), tensor_parts=failing_parts())

    assert not isinstance(caught.value, PartialMessageGenerationLost)
    assert queued == [(4, CHUNK_SIZE + _HEADER.size + _CHUNK_HEAD.size)]
    assert not client._connected.is_set()
    assert client._failure.is_set()
    assert isinstance(client._last_err, ProducerFailure)


def test_streaming_push_late_size_validation_remains_non_transient():
    tensor_bytes = 2 * CHUNK_SIZE
    client = _client(tensor_bytes)
    client._enqueue = lambda *_args, **_kwargs: True

    with pytest.raises(ValueError, match="delta bytes, expected") as caught:
        client.push_fragment_parts(
            **_push_kwargs(),
            tensor_parts=[b"x" * CHUNK_SIZE],
        )

    assert not isinstance(caught.value, PartialMessageGenerationLost)
    assert not client._connected.is_set()
    assert client._failure.is_set()


def test_streaming_push_mid_message_enqueue_drop_poisons_and_raises():
    tensor_bytes = 2 * CHUNK_SIZE
    client = _client(tensor_bytes)
    client.connection_generation = 4004
    calls = 0

    def enqueue(stream, data, gen=None, sent=None):
        nonlocal calls
        del stream, data, gen, sent
        calls += 1
        return calls == 1

    client._enqueue = enqueue
    part = b"x" * (CHUNK_SIZE // 4)
    parts = itertools.repeat(part, tensor_bytes // len(part))

    with pytest.raises(
        PartialMessageGenerationLost,
        match="dropped after 1 .* chunks were queued",
    ) as caught:
        client.push_fragment_parts(**_push_kwargs(), tensor_parts=parts)

    assert caught.value.connection_generation == 4004
    assert caught.value.connection_epoch == 4
    assert caught.value.queued_chunks == 1
    assert caught.value.operation == "PUSH_FRAGMENT fragment 0"
    assert calls == 2
    assert not client._connected.is_set()
    assert client._failure.is_set()


def test_streaming_push_generation_loss_can_retry_only_replayed_pull():
    tensor_bytes = 2 * CHUNK_SIZE
    client = _client(tensor_bytes)
    client.connection_generation = 4004
    calls = 0

    def drop_second(_stream, _data, *, gen=None, sent=None):
        nonlocal calls
        del gen, sent
        calls += 1
        return calls == 1

    client._enqueue = drop_second
    part = b"x" * (CHUNK_SIZE // 4)
    tensor_parts = lambda: itertools.repeat(part, tensor_bytes // len(part))

    with pytest.raises(PartialMessageGenerationLost):
        client.push_fragment_parts(**_push_kwargs(), tensor_parts=tensor_parts())

    # Model the supervisor's successful reconnect.  The peer, not the caller,
    # re-authorizes the operation by replaying its PULL on the new generation.
    client._gen = 6
    client.connection_generation = 6006
    client._failure.clear()
    client._connected.set()
    client._dispatch(6, MSG_PULL_REQ, struct.pack("<IQI", 0, 11, 2))
    replayed = client.drain_pulls()
    assert len(replayed) == 1
    permit = replayed[0]

    retry_generations = []

    def accept_retry(_stream, _data, *, gen=None, sent=None):
        del sent
        retry_generations.append(gen)
        return True

    client._enqueue = accept_retry
    assert client.push_fragment_parts(
        fragment_id=permit.fragment_id,
        global_step=permit.global_step,
        round_attempt=permit.round_attempt,
        base_version=9,
        local_step=17,
        c_steps=3,
        c_tokens=1234,
        tensor_parts=tensor_parts(),
    )
    assert retry_generations
    assert set(retry_generations) == {6}
