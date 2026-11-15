"""A29 (D4): opt-in scaling of the SyncerClient connect/finalization timeouts.

Unset, every client keeps the fixed 900 s defaults (0.6B to K3.1 unchanged);
set, a timeout not passed explicitly grows with the layout's f32 bytes but
never drops below the default, and an explicit value always wins.
"""

from __future__ import annotations

import pytest

from yeto import protocol
from yeto.fragments import build_layout


def _client(**kwargs):
    layout = build_layout([("w", 1 << 20)], 1)
    return protocol.SyncerClient(("127.0.0.1", 1), 0, layout, protocol.DTYPE_F32, 1, **kwargs)


def test_defaults_are_unchanged_without_the_env(monkeypatch):
    monkeypatch.delenv(protocol.SYNC_TIMEOUT_BANDWIDTH_ENV, raising=False)
    assert protocol.scaled_sync_timeout(900.0, 11 * 10**12) == 900.0
    client = _client()
    assert (client.connect_timeout, client.finalization_timeout) == (900.0, 900.0)


def test_opt_in_scales_with_bytes_and_keeps_the_floor(monkeypatch):
    monkeypatch.setenv(protocol.SYNC_TIMEOUT_BANDWIDTH_ENV, str(10**9))
    # 0.6B f32 (2.4 GB) and even a 120B f32 layout (480 GB -> 960 s) sit near the floor.
    assert protocol.scaled_sync_timeout(900.0, 2_400_000_000) == 900.0
    assert protocol.scaled_sync_timeout(900.0, 480 * 10**9) == pytest.approx(960.0)
    # 2.8T f32 = 11.2 TB at 1 GB/s: 2 x 11200 s.
    assert protocol.scaled_sync_timeout(900.0, 11_200 * 10**9) == pytest.approx(22_400.0)
    client = _client(connect_timeout=30.0, finalization_timeout=60.0)
    assert (client.connect_timeout, client.finalization_timeout) == (30.0, 60.0)
    assert _client().finalization_timeout == 900.0


@pytest.mark.parametrize("raw", ["0", "-5", "inf", "nan"])
def test_bad_bandwidth_is_refused(monkeypatch, raw):
    monkeypatch.setenv(protocol.SYNC_TIMEOUT_BANDWIDTH_ENV, raw)
    with pytest.raises(ValueError, match="positive bytes/s"):
        protocol.scaled_sync_timeout(900.0, 1)
