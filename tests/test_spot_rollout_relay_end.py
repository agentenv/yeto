"""S27 G-EM: the relay names each sample's cut, and the clients end with the run.

* a relay item carries the cut of every engine version in its group; the
  trainer's pump writes each sample's cut into its metadata and rejects an item
  without it;
* relay producers and consumers give up once the head is gone (``gone_s``) and
  a stopping sampler ends its retries; the sampler buffer then stops quietly;
* a subscriber keeps the syncer's end-of-run notice even when it arrives on a
  connection generation the client is already tearing down.
"""

from __future__ import annotations

import socket
import time

import pytest

from tests.test_spot_rollout_syncer import layout
from yeto.protocol import MSG_ERROR, SUBSCRIBER_ID_BIT
from yeto.rl.engine.buffer_payload import decode_group, encode_group
from yeto.rl.engine.external_policy import CUT_METADATA_KEY, EXTERNAL_MERGED_CUT
from yeto.rl.engine.sample_buffer import SampleIndexEntry
from yeto.rl.spot_rollout.relay import (
    PUT_STORED,
    Depth,
    PutAck,
    RelayConsumer,
    RelayGone,
    RelayProducer,
    RelayStopped,
)
from yeto.rl.spot_rollout.relay_buffer import (
    RelayItemError,
    RelaySampleBuffer,
    decode_item,
    with_cut_metadata,
)
from yeto.rl.spot_rollout.subscriber import RUN_FINISHED, SubscriberClient


def dead_addr() -> tuple[str, int]:
    """A local port nobody listens on (connection refused at once)."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()


def _samples(versions):
    return [{"index": i, "response_length": 2, "rollout_log_probs": [-0.1, -0.2],
             "metadata": {"weight_version": str(v)}} for i, v in enumerate(versions)]


def _cut(version):
    return {"merged_version": version, "digest": f"{version:x}".rjust(64, "a"),
            "fragment_versions": [2 * version, 2 * version - 1]}


def test_producer_gives_up_when_the_head_is_gone():
    producer = RelayProducer(dead_addr(), 0, 1, key=b"k", retry_s=0.02, gone_s=0.3, timeout=1.0)
    started = time.monotonic()
    with pytest.raises(RelayGone, match="the head is gone"):
        producer.put(b"x")
    assert time.monotonic() - started < 5.0


def test_a_stopping_producer_ends_its_retries():
    producer = RelayProducer(dead_addr(), 0, 1, key=b"k", retry_s=0.02, gone_s=None, timeout=1.0,
                             should_stop=lambda: True)
    with pytest.raises(RelayStopped):
        producer.depth()


def test_consumer_gives_up_when_the_head_is_gone():
    consumer = RelayConsumer(dead_addr(), 0, key=b"k", retry_s=0.02, gone_s=0.3, timeout=1.0)
    with pytest.raises(RelayGone):
        consumer.take(wait_s=0.0)


def test_sampler_buffer_stops_quietly_when_the_relay_is_gone():
    producer = RelayProducer(dead_addr(), 0, 1, key=b"k", retry_s=0.02, gone_s=0.3, timeout=1.0)
    buffer = RelaySampleBuffer(producer)
    assert buffer.refresh() == 0 and buffer.gone
    buffer.write_payload("s1", encode_group(_samples([3])))
    buffer.append(SampleIndexEntry(sample_id="s1", behavior_version=3, payload_ref="relay/s1", tokens=2))
    assert buffer.dropped == 1 and buffer.written == 0


class _Producer:
    def __init__(self):
        self.items = []
        self.should_stop = lambda: False

    def put_blocking(self, data, **_):
        self.items.append(data)
        return PutAck(len(self.items), PUT_STORED, len(self.items), Depth(len(self.items), 0, 0, len(self.items) + 1))


def test_sampler_items_carry_the_cut_of_every_version():
    producer = _Producer()
    buffer = RelaySampleBuffer(producer, cut_of=lambda v: _cut(v) if v in (3, 4) else None)
    for sid, versions in (("a", [3, 4]), ("b", [2, 3])):
        buffer.write_payload(sid, encode_group(_samples(versions)))
        buffer.append(SampleIndexEntry(sample_id=sid, behavior_version=min(versions),
                                       payload_ref=f"relay/{sid}", tokens=4))
    assert buffer.written == 1 and buffer.uncut == 1  # version 2 has no loaded cut: not sent
    header, _ = decode_item(producer.items[0])
    assert header["cuts"] == {"3": _cut(3), "4": _cut(4)}
    assert header["fragment_versions"] == _cut(3)["fragment_versions"]


def test_pump_writes_each_samples_cut_and_rejects_items_without_one():
    header = {"behavior_version": 3, "cuts": {"3": _cut(3), "4": _cut(4)}}
    samples = decode_group(with_cut_metadata(header, encode_group(_samples([3, 4]))))
    assert [s["metadata"][CUT_METADATA_KEY] for s in samples] == [
        {"kind": EXTERNAL_MERGED_CUT, "merged_version": 3, "digest": _cut(3)["digest"]},
        {"kind": EXTERNAL_MERGED_CUT, "merged_version": 4, "digest": _cut(4)["digest"]}]
    with pytest.raises(RelayItemError, match="names no merged cut"):
        with_cut_metadata({"behavior_version": 3}, encode_group(_samples([3])))
    with pytest.raises(RelayItemError, match="engine version 4"):
        with_cut_metadata({"behavior_version": 3, "cuts": {"3": _cut(3)}}, encode_group(_samples([3, 4])))
    with pytest.raises(RelayItemError, match="behavior version 2"):
        with_cut_metadata({"behavior_version": 2, "cuts": {"3": _cut(3)}}, encode_group(_samples([3])))


def test_subscriber_keeps_the_end_of_run_notice_of_an_old_generation():
    client = SubscriberClient(dead_addr(), SUBSCRIBER_ID_BIT | 5, layout(), hmac_key=b"k")
    client._gen = 7  # the client already tore that link down and redials
    assert not client.run_finished()
    client._dispatch(6, MSG_ERROR, RUN_FINISHED.encode())
    assert client.run_finished()
    assert client._closed.is_set()  # no more redials
    other = SubscriberClient(dead_addr(), SUBSCRIBER_ID_BIT | 6, layout(), hmac_key=b"k")
    other._gen = 7
    other._dispatch(6, MSG_ERROR, b"subscriber refused: something else")
    assert not other.run_finished() and not other._closed.is_set()


def test_the_base_version_is_named_by_its_checkpoint_digest(tmp_path):
    from yeto.rl.engine.external_policy import ExternalCutError, base_cut_digest
    from yeto.rl.spot_rollout.release_store import SubscriberReleaseStore

    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "config.json").write_text("{}")
    store = SubscriberReleaseStore(object(), lambda cut: tmp_path, base_dir=tmp_path)
    base = store.cut_identity(0)
    assert base == {"merged_version": 0, "digest": base_cut_digest(tmp_path), "fragment_versions": []}
    assert store.cut_identity(1) is None
    (tmp_path / "model.safetensors").write_bytes(b"other")
    assert base_cut_digest(tmp_path) != base["digest"]
    assert SubscriberReleaseStore(object(), lambda cut: tmp_path).cut_identity(0) is None
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ExternalCutError, match="no weight file"):
        base_cut_digest(empty)
