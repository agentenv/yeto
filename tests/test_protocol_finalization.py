"""Versioned final-artifact protocol and raw-overwrite unit tests."""

import queue
import socket
import struct
import threading

import pytest
import torch

from yeto.finalization import finalize_torch_island
from yeto.fragments import build_layout
from yeto.protocol import (
    DTYPE_F32,
    FINALIZATION_REVISION,
    MSG_ERROR,
    MSG_BCAST_FRAGMENT,
    MSG_FINAL_ACK,
    MSG_FINAL_FRAGMENT,
    MSG_FINAL_MANIFEST,
    MSG_SHUTDOWN,
    FinalFragment,
    FinalManifest,
    SyncerClient,
    decode_final_manifest,
    encode_final_manifest,
    read_frame,
)
from yeto.tensor_io import pack_fragment


def _layout(count=2):
    return build_layout([(f"model.layer.{i}.weight", 4) for i in range(count)], count)


def _bcast_payload(fid: int, version: int, values: list[float]) -> bytes:
    return struct.pack("<IQ", fid, version) + struct.pack(f"<{len(values)}f", *values)


def test_final_manifest_golden_shape_and_revision_validation():
    assert MSG_ERROR == 10
    assert MSG_FINAL_MANIFEST == 11
    assert MSG_FINAL_ACK == 12
    assert MSG_FINAL_FRAGMENT == 13
    payload = encode_final_manifest(17, [15, 16, 17])
    expected = struct.pack(
        "<HQI3Q",
        FINALIZATION_REVISION,
        17,
        3,
        15,
        16,
        17,
    )
    assert payload == expected
    assert decode_final_manifest(payload, 3) == FinalManifest(17, (15, 16, 17))

    bad = bytearray(payload)
    bad[0:2] = struct.pack("<H", FINALIZATION_REVISION + 1)
    with pytest.raises(ValueError, match="unsupported finalization revision"):
        decode_final_manifest(bytes(bad), 3)
    with pytest.raises(ValueError, match="has 3 fragments, expected 2"):
        decode_final_manifest(payload, 2)


def test_manifest_and_data_stream_reordering_preserves_lossless_final_fragments():
    client = SyncerClient(("unused", 0), 0, _layout(), dtype=DTYPE_F32)
    client._gen = 4

    # An ordinary broadcast at the final version cannot satisfy terminal
    # delivery: its session dtype may be lossy. The dedicated f32 fragment 0
    # then arrives before the manifest, while the manifest overtakes fragment 1.
    client._dispatch(4, MSG_BCAST_FRAGMENT, _bcast_payload(0, 8, [0.5] * 4))
    assert len(client.drain_updates()) == 1
    client._dispatch(4, MSG_FINAL_FRAGMENT, _bcast_payload(0, 8, [1.0001] * 4))
    client._dispatch(4, MSG_FINAL_MANIFEST, encode_final_manifest(9, [8, 9]))
    client._dispatch(4, MSG_FINAL_FRAGMENT, _bcast_payload(1, 9, [2.0001] * 4))

    manifest, fragments = client.wait_for_final_fragments(timeout=0.1)
    assert manifest == FinalManifest(9, (8, 9))
    assert [(item.fragment_id, item.version) for item in fragments] == [(0, 8), (1, 9)]
    assert fragments[0].data == struct.pack("<4f", *([1.0001] * 4))


def test_missing_manifest_version_fails_with_bounded_diagnostic():
    client = SyncerClient(("unused", 0), 0, _layout(), dtype=DTYPE_F32)
    client._gen = 1
    client._dispatch(1, MSG_FINAL_MANIFEST, encode_final_manifest(9, [8, 9]))
    client._dispatch(1, MSG_FINAL_FRAGMENT, _bcast_payload(0, 8, [1.0] * 4))
    with pytest.raises(TimeoutError, match=r"missing fragment versions.*\(1, 9, None\)"):
        client.wait_for_final_fragments(timeout=0.01)


def test_final_fragment_requires_exact_f32_payload_size():
    client = SyncerClient(("unused", 0), 0, _layout(1), dtype=DTYPE_F32)
    client._gen = 1
    client._dispatch(1, MSG_FINAL_FRAGMENT, _bcast_payload(0, 4, [1.0, 2.0]))
    with pytest.raises(RuntimeError, match="expected 16 f32 bytes"):
        client.check_health()


def test_stale_generation_final_messages_are_ignored_and_legacy_shutdown_fails():
    client = SyncerClient(("unused", 0), 0, _layout(1), dtype=DTYPE_F32)
    client._gen = 3
    client._dispatch(2, MSG_FINAL_FRAGMENT, _bcast_payload(0, 7, [3.0] * 4))
    client._dispatch(2, MSG_FINAL_MANIFEST, encode_final_manifest(7, [7]))
    client._dispatch(2, MSG_SHUTDOWN, b"")
    assert not client._final_fragments
    assert not client.finalizing.is_set()
    assert not client.shutdown.is_set()
    client.check_health()

    client._dispatch(3, MSG_SHUTDOWN, b"")
    assert not client.shutdown.is_set()
    with pytest.raises(RuntimeError, match="legacy SHUTDOWN"):
        client.check_health()


def test_final_ack_waits_until_sender_thread_has_written_bytes():
    client_sock, server_sock = socket.socketpair()
    client = SyncerClient(("unused", 0), 0, _layout(1), dtype=DTYPE_F32)
    client._gen = 1
    client._connected.set()
    client._queues = [queue.Queue()]
    sender = threading.Thread(
        target=client._send_loop,
        args=(1, client_sock, client._queues[0]),
        daemon=True,
    )
    sender.start()
    try:
        manifest = FinalManifest(5, (5,))
        client._final_manifest = manifest
        client.finalizing.set()
        client.acknowledge_finalization(manifest, timeout=1.0)
        msg_type, payload = read_frame(server_sock)
        assert msg_type == MSG_FINAL_ACK
        assert payload == struct.pack("<HQ", FINALIZATION_REVISION, 5)
        assert client.finalized.is_set()

        client._dispatch(1, MSG_SHUTDOWN, b"")
        assert client.shutdown.is_set()
    finally:
        client.close()
        server_sock.close()
        sender.join(timeout=1)


def test_torch_finalization_overwrites_blended_local_values_with_raw_global():
    layout = _layout(2)
    params = {name: torch.full((numel,), 9.0) for name, numel in [
        ("model.layer.0.weight", 4),
        ("model.layer.1.weight", 4),
    ]}
    authoritative = {
        "model.layer.0.weight": torch.full((4,), 1.25),
        "model.layer.1.weight": torch.full((4,), -2.5),
    }
    manifest = FinalManifest(12, (11, 12))
    broadcasts = [
        FinalFragment(
            fid,
            manifest.versions[fid],
            pack_fragment(fragment, authoritative, DTYPE_F32),
        )
        for fid, fragment in enumerate(layout.fragments)
    ]

    class FakeClient:
        dtype = DTYPE_F32

        def __init__(self):
            self.acknowledged = None

        def wait_for_final_fragments(self):
            return manifest, broadcasts

        def acknowledge_finalization(self, value):
            self.acknowledged = value

    client = FakeClient()
    got = finalize_torch_island(
        client,
        layout,
        params,
        rank=0,
        world=1,
        device=torch.device("cpu"),
    )
    assert got == manifest
    assert client.acknowledged == manifest
    for name in params:
        assert torch.equal(params[name], authoritative[name])
