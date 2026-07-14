import hashlib
import struct

import pytest

from yeto.fragments import MERGE_RDA, Fragment, FragmentLayout
from yeto.protocol import (
    AUDIT_PUSH_VERSION,
    MSG_PUSH_FRAGMENT,
    MSG_PUSH_FRAGMENT_AUDIT,
    PushAudit,
    SyncerClient,
)


def _client() -> SyncerClient:
    layout = FragmentLayout([Fragment(MERGE_RDA, [("model.body.weight", 4)])])
    return SyncerClient(("127.0.0.1", 1), 7, layout, num_streams=0)


def _push(client: SyncerClient, tensor: bytes, **extra):
    return client.push_fragment(
        fragment_id=3,
        global_step=11,
        base_version=7,
        local_step=101,
        c_steps=4,
        c_tokens=4096,
        tensor_bytes=tensor,
        **extra,
    )


def test_legacy_push_envelope_remains_byte_exact():
    client = _client()
    sent = []
    client._send_large = lambda kind, payload: sent.append((kind, payload)) or True
    tensor = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)

    assert _push(client, tensor) is True
    assert sent == [
        (
            MSG_PUSH_FRAGMENT,
            struct.pack("<IIQQQIQ", 7, 3, 11, 7, 101, 4, 4096) + tensor,
        )
    ]


def test_audited_push_carries_exact_window_attempt_and_wire_digest():
    client = _client()
    sent = []
    client._send_large = lambda kind, payload: sent.append((kind, payload)) or True
    tensor = struct.pack("<4f", -1.0, 0.25, 3.5, 8.0)
    window = bytes.fromhex("00112233445566778899aabbccddeeff")
    digest = hashlib.sha256(tensor).digest()

    assert (
        _push(
            client,
            tensor,
            audit=PushAudit(
                window_uuid=window, attempt_serial=23, payload_sha256=digest
            ),
        )
        is True
    )
    kind, payload = sent[0]
    assert kind == MSG_PUSH_FRAGMENT_AUDIT
    assert struct.unpack_from("<IIQQQIQ", payload) == (7, 3, 11, 7, 101, 4, 4096)
    version, got_window, attempt, got_digest = struct.unpack_from(
        "<B16sQ32s", payload, 44
    )
    assert (version, got_window, attempt, got_digest) == (
        AUDIT_PUSH_VERSION,
        window,
        23,
        digest,
    )
    assert payload[101:] == tensor
    assert hashlib.sha256(payload[101:]).digest() == digest


def test_audited_push_fails_before_enqueue_on_digest_mismatch():
    client = _client()
    client._send_large = lambda *_args: pytest.fail("mismatched payload was enqueued")
    with pytest.raises(ValueError, match="does not match"):
        _push(
            client,
            b"tensor",
            audit=PushAudit(
                window_uuid=b"w" * 16,
                attempt_serial=1,
                payload_sha256=b"d" * 32,
            ),
        )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"window_uuid": b"\0" * 16}, "nonzero"),
        ({"attempt_serial": 0}, "attempt_serial"),
        ({"payload_sha256": b"x" * 31}, "32 bytes"),
    ],
)
def test_push_audit_rejects_malformed_identity(kwargs, message):
    values = {
        "window_uuid": b"w" * 16,
        "attempt_serial": 1,
        "payload_sha256": b"d" * 32,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        PushAudit(**values)


def test_send_large_reports_drop_while_disconnected():
    client = _client()
    assert _push(client, b"\0" * 16) is False
