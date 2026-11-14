"""A relay item with a malformed header is rejected and acked, never stops the pump step."""

import json
import struct

from yeto.rl.spot_rollout.relay import Item
from yeto.rl.spot_rollout.relay_buffer import HEADER_SCHEMA, MAGIC, RelayPump


class _Buffer:
    read_cursor = 0

    def entries(self):
        return []

    def refresh(self):
        pass

    def __len__(self):
        return 0


class _Consumer:
    def __init__(self, items):
        self.items = items
        self.acked = []

    def take(self, *, max_items, wait_s):
        items, self.items = self.items, []
        return items

    def ack(self, item_id):
        self.acked.append(item_id)


def _raw(header_json: bytes) -> bytes:
    return MAGIC + struct.pack("<I", len(header_json)) + header_json + b"payload"


def test_malformed_headers_are_rejected_and_acked():
    consumer = _Consumer([
        Item(1, 7, _raw(b"[1, 2]")),  # not an object
        Item(2, 7, _raw(json.dumps({"schema": HEADER_SCHEMA}).encode())),  # no sample_id
        Item(3, 7, _raw(b"\"text\"")),
    ])
    events = []
    pump = RelayPump(consumer, _Buffer(), emit=lambda name, **kw: events.append(name))
    assert pump.step() == 0
    assert pump.rejected == 3
    assert consumer.acked == [1, 2, 3]
    assert events == ["rl_relay_item_rejected"] * 3
