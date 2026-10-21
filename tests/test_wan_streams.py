"""S26 E3b: per-stream accounting, stall watchdog and rate-aware striping."""

from __future__ import annotations

import queue
import socket
import threading
import time

import pytest

from yeto.protocol import SyncerClient
from yeto.rl.engine.sync_timing import SyncTiming, upload_fields
from yeto.wan_streams import (
    StreamMeter,
    StreamSnapshot,
    UploadWindow,
    pick_stream,
    upload_window,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _meter_with_rate(clock: _Clock, rate_bps: float, backlog: int = 0) -> StreamMeter:
    meter = StreamMeter(clock)
    meter.enqueued(int(rate_bps))
    meter.begin()
    clock.now += 1.0
    meter.progress(int(rate_bps))
    meter.end()
    if backlog:
        meter.enqueued(backlog)
    return meter


def test_round_robin_ignores_rates():
    clock = _Clock()
    meters = [_meter_with_rate(clock, 1e6), _meter_with_rate(clock, 1e3)]
    assert [pick_stream("rr", meters, ticket) for ticket in range(4)] == [0, 1, 0, 1]


def test_adaptive_spreads_an_idle_start_over_every_stream():
    meters = [StreamMeter() for _ in range(4)]
    picked = []
    for ticket in range(4):
        index = pick_stream("adaptive", meters, ticket)
        meters[index].enqueued(4 << 20)
        picked.append(index)
    assert sorted(picked) == [0, 1, 2, 3]


def test_adaptive_gives_a_slow_stream_fewer_chunks_in_proportion_to_its_rate():
    clock = _Clock()
    fast, slow = 40e6, 5e6
    meters = [_meter_with_rate(clock, fast), _meter_with_rate(clock, slow)]
    counts = [0, 0]
    for ticket in range(90):
        index = pick_stream("adaptive", meters, ticket)
        meters[index].enqueued(4 << 20)
        counts[index] += 1
    # Equal estimated finish time: chunk counts follow the 8:1 rate ratio.
    assert counts[1] == pytest.approx(90 / 9, abs=1)
    assert counts[0] == pytest.approx(90 * 8 / 9, abs=1)


def test_adaptive_starves_a_stream_whose_backlog_is_stuck():
    clock = _Clock()
    meters = [_meter_with_rate(clock, 10e6), _meter_with_rate(clock, 10e6, backlog=64 << 20)]
    assert {pick_stream("adaptive", meters, ticket) for ticket in range(10)} == {0}


def test_unknown_schedule_is_refused():
    with pytest.raises(ValueError, match="schedule"):
        pick_stream("random", [StreamMeter()], 0)
    with pytest.raises(ValueError, match="stream_schedule"):
        SyncerClient(("127.0.0.1", 1), 0, _layout(), stream_schedule="random")
    with pytest.raises(ValueError, match="stall_timeout"):
        SyncerClient(("127.0.0.1", 1), 0, _layout(), stream_stall_timeout=0)


def test_meter_reports_a_stall_only_while_bytes_wait():
    clock = _Clock()
    meter = StreamMeter(clock)
    clock.now += 50
    assert meter.stalled_for() == 0.0  # idle is not stalled
    meter.enqueued(10)
    meter.begin()
    meter.progress(4)
    clock.now += 7
    assert meter.stalled_for() == pytest.approx(7)
    meter.progress(6)
    meter.end()
    clock.now += 30
    assert meter.stalled_for() == 0.0 and meter.idle
    meter.enqueued(5)
    meter.drop_pending()
    assert meter.idle


def test_upload_window_and_fields_report_rates_and_the_wait_after_upload():
    before = (StreamSnapshot(100, 1.0, 0, None), StreamSnapshot(0, 0.0, 0, None))
    after = (StreamSnapshot(100 + 40_000_000, 3.0, 0, 2e7), StreamSnapshot(5_000_000, 1.0, 0, 5e6))
    window = upload_window(before, after)
    assert window == UploadWindow((40_000_000, 5_000_000), (2.0, 1.0))
    assert window.stream_mbps() == [20.0, 5.0]
    fields = upload_fields(window, drained_ts=1010.0, timing=SyncTiming(1000.0, 1025.0), stalls=1)
    assert fields["sync/upload_mbps"] == pytest.approx(22.5)
    assert fields["sync/upload_min_stream_mbps"] == 5.0
    assert fields["sync/upload_active_streams"] == 2
    assert fields["sync/wait_after_upload_seconds"] == pytest.approx(15.0)
    assert fields["sync/stream_stalls"] == 1
    # The upload drained before this boundary's submit: only the submit counts.
    early = upload_fields(window, drained_ts=990.0, timing=SyncTiming(1000.0, 1025.0), stalls=0)
    assert early["sync/wait_after_upload_seconds"] == pytest.approx(25.0)
    unknown = upload_fields(window, drained_ts=None, timing=SyncTiming(1000.0, None), stalls=0)
    assert unknown["sync/wait_after_upload_seconds"] is None


def test_sync_round_diagnostics_reach_wandb_on_the_policy_axis():
    from yeto.rl.wandb_rl import event_metrics

    metrics = event_metrics({
        "event": "rl_full_parameter_round", "policy_version": 3,
        "sync/sync_wait_seconds": 40.0, "sync/upload_mbps": 22.5,
        "sync/upload_min_stream_mbps": 5.0, "sync/upload_stream_mbps": [20.0, 5.0],
        "sync/wait_after_upload_seconds": 15.0, "sync/global_policy_hash": "ab",
    })
    assert metrics == {"rl/step": 3, "sync/sync_wait_seconds": 40.0, "sync/upload_mbps": 22.5,
                       "sync/upload_min_stream_mbps": 5.0, "sync/wait_after_upload_seconds": 15.0}
    assert event_metrics({"event": "rl_full_parameter_round", "policy_version": 3}) is None


# -- socket-level behaviour ---------------------------------------------------


def _layout():
    from yeto.fragments import MERGE_RDA, Fragment, FragmentLayout

    return FragmentLayout([Fragment(MERGE_RDA, [("w", 4)])])


def test_send_loop_meters_bytes_and_marks_the_upload_drained():
    left, right = socket.socketpair()
    client = SyncerClient(("127.0.0.1", 1), 0, _layout(), num_streams=1)
    q: queue.Queue = queue.Queue()
    meter = client._meters[1]
    worker = threading.Thread(target=client._send_loop, args=(client._gen, left, q, meter))
    worker.start()
    try:
        meter.enqueued(3 << 20)
        q.put(type("Item", (), {"data": b"x" * (3 << 20), "sent": None})())
        received = 0
        while received < 3 << 20:
            received += len(right.recv(1 << 20))
        q.put(None)
        worker.join(timeout=5)
        snap = client.stream_snapshots()[0]
        assert snap.bytes_sent == 3 << 20 and snap.backlog_bytes == 0
        assert client.upload_drained_at is not None
    finally:
        left.close()
        right.close()


def test_watchdog_fails_the_group_when_a_stream_stops_moving():
    client = SyncerClient(("127.0.0.1", 1), 0, _layout(), num_streams=2, stream_stall_timeout=0.2)
    a, b = socket.socketpair()
    try:
        client._socks = [a]
        client._connected.set()
        meter = client._meters[2]
        meter.enqueued(10)
        meter.begin()
        assert client._stalled_stream() is None
        time.sleep(0.3)
        assert "outbound stream 2" in client._stalled_stream()
        client._watchdog = threading.Thread(target=client._watch_stalls, daemon=True)
        client._watchdog.start()
        assert client._failure.wait(3), "the watchdog must fail the live group"
        assert client.send_progress().stalls == 1
        assert "no progress" in str(client._last_err)
    finally:
        client._closed.set()
        a.close()
        b.close()


def test_watchdog_notices_an_inbound_message_that_stops_growing():
    client = SyncerClient(("127.0.0.1", 1), 0, _layout(), num_streams=1, stream_stall_timeout=0.2)
    client._reasm_touched[5] = time.monotonic() - 1.0
    assert "inbound message 5" in client._stalled_stream()
    client._clear_partial_reassemblies()
    assert client._stalled_stream() is None


def test_stall_timeout_sets_tcp_user_timeout():
    if not hasattr(socket, "TCP_USER_TIMEOUT"):
        pytest.skip("Linux-only socket option")
    from yeto.protocol import set_tcp_user_timeout

    with socket.socket() as sock:
        set_tcp_user_timeout(sock, 2.5)
        assert sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_USER_TIMEOUT) == 2500


def test_transport_flags_reach_the_island_command_only_when_set():
    from argparse import Namespace

    from yeto.launch.rl_island_task import _wan_transport_flags

    assert _wan_transport_flags(Namespace()) == ""
    assert _wan_transport_flags(Namespace(wan_stream_schedule="adaptive", wan_stall_timeout=60.0)) == (
        " --wan-stream-schedule adaptive --wan-stall-timeout 60.0")
    with pytest.raises(ValueError, match="wan-stall-timeout"):
        _wan_transport_flags(Namespace(wan_stall_timeout=-1.0))


def test_island_parser_accepts_the_transport_flags():
    from yeto.rl.engine.run_flags import ISLAND, rl_option

    assert rl_option("wan_stream_schedule", ISLAND) == "--wan-stream-schedule"
    assert rl_option("wan_stall_timeout", ISLAND) == "--wan-stall-timeout"


def test_tcp_user_timeout_waits_for_the_initial_cut():
    if not hasattr(socket, "TCP_USER_TIMEOUT"):
        pytest.skip("Linux-only socket option")
    with socket.create_server(("127.0.0.1", 0)) as server:
        addr = server.getsockname()
        client = SyncerClient(addr, 0, _layout(), num_streams=1, stream_stall_timeout=2.0)
        early = client._dial(5)
        try:
            assert early.getsockopt(socket.IPPROTO_TCP, socket.TCP_USER_TIMEOUT) == 0
            client._socks = [early]
            client.mark_initial_cut_received()
            assert early.getsockopt(socket.IPPROTO_TCP, socket.TCP_USER_TIMEOUT) == 2000
            late = client._dial(5)
            assert late.getsockopt(socket.IPPROTO_TCP, socket.TCP_USER_TIMEOUT) == 2000
            late.close()
        finally:
            early.close()


def test_init_replay_is_due_only_on_a_new_group_before_the_initial_cut():
    client = SyncerClient(("127.0.0.1", 1), 0, _layout(), num_streams=1)
    client._connected.set()
    client.connection_generation = 11
    assert client.init_replay_due()
    client._init_generation = 11
    assert not client.init_replay_due()
    client.connection_generation = 12
    assert client.init_replay_due()
    client.mark_initial_cut_received()
    assert not client.init_replay_due()
