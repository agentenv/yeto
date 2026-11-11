"""The per-fragment received log (SC2: two islands' merged fragment content)."""

from __future__ import annotations

import json


def test_received_fragment_line_names_fragment_version_and_hashes(capsys):
    from yeto.rl.adapters.miles.full_parameter.inbound_sink import FRAGMENT_RECEIVED_TAG, log_received_fragment

    line = log_received_fragment(3, 7, "a" * 64, "b" * 64)
    err = capsys.readouterr().err.strip()
    assert err == line and line.startswith(FRAGMENT_RECEIVED_TAG + " ")
    body = json.loads(line[len(FRAGMENT_RECEIVED_TAG) + 1:])
    assert body == {"fragment": 3, "version": 7, "payload": "a" * 64, "wire": "b" * 64}


def test_finish_message_logs_each_new_authoritative_fragment_once():
    import inspect

    from yeto.rl.adapters.miles.full_parameter import inbound_sink

    source = inspect.getsource(inbound_sink.MilesRayInboundChunkSink.finish_message)
    # logged on the newly stored path only (not on a replay of known bytes)
    assert source.count("log_received_fragment(") == 1
    assert source.index("self.sink._stored[key] = stored") < source.index("log_received_fragment(")
