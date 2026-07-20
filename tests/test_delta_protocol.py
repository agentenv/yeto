"""Golden-wire and raw-socket integration tests for protocol v4 invariants."""

import socket
import struct
import subprocess
import time
from pathlib import Path

import pytest
import torch

from yeto.fragments import MERGE_RDA, Fragment, FragmentLayout
from yeto.protocol import (
    DTYPE_BF16,
    DTYPE_F32,
    DTYPE_Q4,
    MAGIC,
    MSG_ERROR,
    MSG_FINAL_ACK,
    MSG_FINAL_FRAGMENT,
    MSG_FINAL_MANIFEST,
    MSG_HELLO,
    MSG_INIT_PARAMS,
    MSG_PULL_REQ,
    MSG_PUSH_FRAGMENT,
    PROTOCOL_VERSION,
    _HEADER,
    bulk_dtype,
    encode_hello,
    layout_fingerprint,
    read_frame,
    write_frame,
)
from yeto.tensor_io import pack_tensor, quantize_q4


ROOT = Path(__file__).resolve().parent.parent
PUSH_HEAD = struct.Struct("<IIQIQQIQ")
PULL = struct.Struct("<IQI")


@pytest.fixture(scope="module")
def syncer_binary() -> Path:
    subprocess.run(["cargo", "build", "-q"], cwd=ROOT / "syncer", check=True)
    binary = ROOT / "syncer/target/debug/yeto-syncer"
    assert binary.is_file()
    return binary


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def connect(port: int) -> socket.socket:
    deadline = time.monotonic() + 10
    while True:
        try:
            return socket.create_connection(("127.0.0.1", port), timeout=1)
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.02)


def one_value_layout(numel: int = 1) -> FragmentLayout:
    return FragmentLayout(
        [Fragment(MERGE_RDA, [("model.body.weight", numel)])]
    )


class RawLearner:
    def __init__(
        self,
        port: int,
        learner_id: int,
        generation: int,
        layout: FragmentLayout,
        dtype: int = DTYPE_F32,
    ):
        self.sock = connect(port)
        self.sock.settimeout(10)
        self.learner_id = learner_id
        self.generation = generation
        self.layout = layout
        self.dtype = dtype
        write_frame(
            self.sock,
            MSG_HELLO,
            encode_hello(learner_id, dtype, layout, 0, generation),
        )

    def close(self) -> None:
        self.sock.close()

    def recv(self) -> tuple[int, bytes]:
        return read_frame(self.sock)

    def recv_pull(self, step: int) -> tuple[int, int, int]:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            msg_type, payload = self.recv()
            if msg_type == MSG_ERROR:
                raise AssertionError(payload.decode("utf-8", errors="replace"))
            if msg_type == MSG_PULL_REQ:
                pull = PULL.unpack(payload)
                if pull[1] == step:
                    return pull
        raise TimeoutError(f"no pull for step {step}")

    def send_init(self, values: torch.Tensor, fragment_id: int = 0) -> None:
        payload = struct.pack("<I", fragment_id) + pack_tensor(
            values, bulk_dtype(self.dtype)
        )
        write_frame(self.sock, MSG_INIT_PARAMS, payload)

    def send_delta(
        self,
        delta: torch.Tensor,
        *,
        step: int,
        attempt: int,
        base_version: int,
        fragment_id: int = 0,
        payload_learner_id: int | None = None,
    ) -> None:
        body = quantize_q4(delta) if self.dtype == DTYPE_Q4 else pack_tensor(delta, self.dtype)
        header = PUSH_HEAD.pack(
            self.learner_id if payload_learner_id is None else payload_learner_id,
            fragment_id,
            step,
            attempt,
            base_version,
            10,
            1,
            100,
        )
        write_frame(self.sock, MSG_PUSH_FRAGMENT, header + body)


def launch_syncer(
    binary: Path,
    tmp_path: Path,
    *,
    learners: int = 1,
    quorum: int = 1,
    total_steps: int = 1,
    quorum_timeout_s: int = 2,
) -> tuple[subprocess.Popen, Path]:
    port = free_port()
    final_state = tmp_path / f"state-{port}.bin"
    proc = subprocess.Popen(
        [
            str(binary),
            "--port",
            str(port),
            "--learners",
            str(learners),
            "--quorum",
            str(quorum),
            "--grace-ms",
            "20",
            "--quorum-timeout-s",
            str(quorum_timeout_s),
            "--total-steps",
            str(total_steps),
            "--pipeline",
            "1",
            "--sync-interval-steps",
            "0",
            "--delta-correction",
            "none",
            "--outer-lr",
            "1",
            "--outer-momentum",
            "0",
            "--final-state",
            str(final_state),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    proc.port = port
    return proc, final_state


def stop_process(proc: subprocess.Popen) -> str:
    if proc.poll() is None:
        proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    return proc.stdout.read() if proc.stdout else ""


def read_final_scalar(path: Path) -> float:
    payload = path.read_bytes()
    fragments = struct.unpack_from("<I", payload, 0)[0]
    assert fragments == 1
    numel = struct.unpack_from("<Q", payload, 4)[0]
    assert numel == 1
    return struct.unpack_from("<f", payload, 12)[0]


def receive_error(sock: socket.socket) -> str:
    msg_type, payload = read_frame(sock)
    assert msg_type == MSG_ERROR
    return payload.decode("utf-8", errors="replace")


def test_versioned_hello_golden_bytes_and_reserved_ids():
    layout = one_value_layout(4)
    generation = 0x0102_0304_0506_0708
    encoded = encode_hello(7, DTYPE_F32, layout, 3, generation)
    expected = (
        struct.pack("<HIQBI", PROTOCOL_VERSION, 7, generation, DTYPE_F32, 1)
        + struct.pack("<BIQ", MERGE_RDA, 1, 4)
        + layout_fingerprint(layout)
        + struct.pack("<H", 3)
    )
    assert encoded == expected
    assert (
        MSG_ERROR,
        MSG_FINAL_MANIFEST,
        MSG_FINAL_ACK,
        MSG_FINAL_FRAGMENT,
    ) == (10, 11, 12, 13)


@pytest.mark.parametrize("dtype", [DTYPE_F32, DTYPE_BF16, DTYPE_Q4])
@pytest.mark.timeout(30)
def test_fresh_and_one_version_stale_deltas_ignore_global_drift(
    syncer_binary, tmp_path, dtype
):
    proc, final_state = launch_syncer(
        syncer_binary, tmp_path, total_steps=2
    )
    learner = None
    try:
        learner = RawLearner(proc.port, 0, 101, one_value_layout(), dtype)
        learner.send_init(torch.tensor([10.0]))
        fragment, step, attempt = learner.recv_pull(1)
        assert (fragment, step, attempt) == (0, 1, 1)
        learner.send_delta(
            torch.tensor([-2.0]), step=1, attempt=attempt, base_version=0
        )

        fragment, step, attempt = learner.recv_pull(2)
        assert (fragment, step, attempt) == (0, 2, 1)
        # This response is still anchored at version 0. Its local update is
        # -1, so the outer gradient is +1 and must be applied to current 8 -> 7.
        learner.send_delta(
            torch.tensor([-1.0]), step=2, attempt=attempt, base_version=0
        )
        assert proc.wait(timeout=10) == 0
        assert read_final_scalar(final_state) == pytest.approx(7.0, abs=1e-5)
    finally:
        if learner is not None:
            learner.close()
        output = stop_process(proc)
        if dtype == DTYPE_Q4:
            assert "stale base-relative delta admitted" in output


@pytest.mark.timeout(30)
def test_mid_round_join_cannot_count_toward_frozen_membership(
    syncer_binary, tmp_path
):
    proc, final_state = launch_syncer(
        syncer_binary, tmp_path, learners=2, quorum=2, total_steps=2
    )
    first = departing = joined = None
    try:
        first = RawLearner(proc.port, 0, 101, one_value_layout())
        departing = RawLearner(proc.port, 1, 201, one_value_layout())
        first.send_init(torch.tensor([0.0]))
        _, step, attempt = first.recv_pull(1)
        assert departing.recv_pull(1) == (0, step, attempt)
        departing.send_delta(
            torch.tensor([-1.0]), step=step, attempt=attempt, base_version=0
        )
        departing.close()
        departing = None
        # Let the old generation's PUSH and disconnect reach the scheduler
        # before the second response completes round 1.
        time.sleep(0.2)
        first.send_delta(
            torch.tensor([-1.0]), step=step, attempt=attempt, base_version=0
        )

        _, step, attempt = first.recv_pull(2)
        joined = RawLearner(proc.port, 1, 202, one_value_layout())
        # Learner 1 joined after attempt 1 of step 2 launched. Guessing its
        # fields cannot add it to the frozen member set.
        joined.send_delta(
            torch.tensor([-100.0]), step=step, attempt=attempt, base_version=1
        )
        first.send_delta(
            torch.tensor([-1.0]), step=step, attempt=attempt, base_version=1
        )
        assert proc.wait(timeout=10) == 0
        assert read_final_scalar(final_state) == pytest.approx(-2.0)
    finally:
        if first is not None:
            first.close()
        if departing is not None:
            departing.close()
        if joined is not None:
            joined.close()
        output = stop_process(proc)
        assert "UnexpectedMember" in output


@pytest.mark.timeout(30)
def test_reconnect_generation_cannot_inject_or_be_erased_by_old_disconnect(
    syncer_binary, tmp_path
):
    proc, final_state = launch_syncer(
        syncer_binary, tmp_path, total_steps=2
    )
    old = new = None
    try:
        old = RawLearner(proc.port, 0, 101, one_value_layout())
        old.send_init(torch.tensor([0.0]))
        _, step, attempt = old.recv_pull(1)
        new = RawLearner(proc.port, 0, 202, one_value_layout())
        new.send_delta(
            torch.tensor([-100.0]), step=step, attempt=attempt, base_version=0
        )
        old.send_delta(
            torch.tensor([-1.0]), step=step, attempt=attempt, base_version=0
        )
        old.close()
        old = None

        _, step, attempt = new.recv_pull(2)
        new.send_delta(
            torch.tensor([-2.0]), step=step, attempt=attempt, base_version=1
        )
        assert proc.wait(timeout=10) == 0
        assert read_final_scalar(final_state) == pytest.approx(-3.0)
    finally:
        if old is not None:
            old.close()
        if new is not None:
            new.close()
        output = stop_process(proc)
        assert "UnexpectedMember" in output


@pytest.mark.timeout(30)
def test_below_quorum_timeout_discards_partial_attempt(
    syncer_binary, tmp_path
):
    proc, final_state = launch_syncer(
        syncer_binary,
        tmp_path,
        learners=2,
        quorum=2,
        quorum_timeout_s=1,
    )
    first = second = None
    try:
        first = RawLearner(proc.port, 0, 101, one_value_layout())
        second = RawLearner(proc.port, 1, 201, one_value_layout())
        first.send_init(torch.tensor([0.0]))
        _, step, attempt = first.recv_pull(1)
        assert second.recv_pull(1) == (0, step, attempt)
        first.send_delta(
            torch.tensor([-100.0]), step=step, attempt=attempt, base_version=0
        )

        # K=2 was not reached. The retry must use a new attempt token and
        # discard the first response instead of merging it at timeout.
        _, retry_step, retry_attempt = first.recv_pull(1)
        assert second.recv_pull(1) == (0, retry_step, retry_attempt)
        assert retry_attempt == attempt + 1
        for learner in (first, second):
            learner.send_delta(
                torch.tensor([-1.0]),
                step=retry_step,
                attempt=retry_attempt,
                base_version=0,
            )
        assert proc.wait(timeout=10) == 0
        assert read_final_scalar(final_state) == pytest.approx(-1.0)
    finally:
        if first is not None:
            first.close()
        if second is not None:
            second.close()
        output = stop_process(proc)
        assert "new frozen-membership attempt" in output


@pytest.mark.timeout(30)
def test_version_layout_and_dtype_mismatches_fail_clearly(
    syncer_binary, tmp_path
):
    proc, _ = launch_syncer(syncer_binary, tmp_path)
    accepted = None
    sockets = []
    try:
        accepted = RawLearner(proc.port, 0, 101, one_value_layout(), DTYPE_F32)

        wrong_version = connect(proc.port)
        sockets.append(wrong_version)
        hello = bytearray(encode_hello(0, DTYPE_F32, one_value_layout(), 0, 102))
        struct.pack_into("<H", hello, 0, PROTOCOL_VERSION + 1)
        write_frame(wrong_version, MSG_HELLO, bytes(hello))
        assert "protocol version mismatch" in receive_error(wrong_version)

        wrong_dtype = connect(proc.port)
        sockets.append(wrong_dtype)
        write_frame(
            wrong_dtype,
            MSG_HELLO,
            encode_hello(0, DTYPE_BF16, one_value_layout(), 0, 103),
        )
        assert "session mismatch" in receive_error(wrong_dtype)

        wrong_layout = connect(proc.port)
        sockets.append(wrong_layout)
        write_frame(
            wrong_layout,
            MSG_HELLO,
            encode_hello(0, DTYPE_F32, one_value_layout(2), 0, 104),
        )
        assert "session mismatch" in receive_error(wrong_layout)

        wrong_semantics = connect(proc.port)
        sockets.append(wrong_semantics)
        same_sizes_different_tensor = FragmentLayout(
            [Fragment(MERGE_RDA, [("model.other.weight", 1)])]
        )
        write_frame(
            wrong_semantics,
            MSG_HELLO,
            encode_hello(
                0, DTYPE_F32, same_sizes_different_tensor, 0, 105
            ),
        )
        assert "session mismatch" in receive_error(wrong_semantics)

        malformed = connect(proc.port)
        sockets.append(malformed)
        write_frame(
            malformed,
            MSG_HELLO,
            struct.pack("<HIQBI", PROTOCOL_VERSION, 0, 106, DTYPE_F32, 0),
        )
        assert "layout must contain at least one fragment" in receive_error(malformed)
    finally:
        if accepted is not None:
            accepted.close()
        for sock in sockets:
            sock.close()
        stop_process(proc)


@pytest.mark.timeout(30)
def test_out_of_range_learner_ids_are_rejected_at_startup_and_after_init(
    syncer_binary, tmp_path
):
    proc, final_state = launch_syncer(syncer_binary, tmp_path)
    valid = None
    invalid_sockets = []
    try:
        for generation in (101,):
            sock = connect(proc.port)
            invalid_sockets.append(sock)
            write_frame(
                sock,
                MSG_HELLO,
                encode_hello(1, DTYPE_F32, one_value_layout(), 0, generation),
            )
            assert "outside configured range" in receive_error(sock)

        valid = RawLearner(proc.port, 0, 201, one_value_layout())
        valid.send_init(torch.tensor([0.0]))
        _, step, attempt = valid.recv_pull(1)

        sock = connect(proc.port)
        invalid_sockets.append(sock)
        write_frame(
            sock,
            MSG_HELLO,
            encode_hello(1, DTYPE_F32, one_value_layout(), 0, 202),
        )
        assert "outside configured range" in receive_error(sock)

        valid.send_delta(
            torch.tensor([-1.0]), step=step, attempt=attempt, base_version=0
        )
        assert proc.wait(timeout=10) == 0
        assert read_final_scalar(final_state) == pytest.approx(-1.0)
    finally:
        if valid is not None:
            valid.close()
        for sock in invalid_sockets:
            sock.close()
        stop_process(proc)


@pytest.mark.timeout(30)
def test_malicious_payload_ids_fragments_and_lengths_are_rejected(
    syncer_binary, tmp_path
):
    proc, _ = launch_syncer(syncer_binary, tmp_path)
    sockets = []
    try:
        cases = [
            PUSH_HEAD.pack(9, 0, 1, 1, 0, 1, 1, 1) + struct.pack("<f", -1.0),
            PUSH_HEAD.pack(0, 99, 1, 1, 0, 1, 1, 1) + struct.pack("<f", -1.0),
            PUSH_HEAD.pack(0, 0, 1, 1, 0, 1, 1, 1) + b"\x00",
        ]
        expected = ["does not match connected group", "unknown fragment", "expected 4"]
        for index, (payload, message) in enumerate(zip(cases, expected), start=1):
            learner = RawLearner(proc.port, 0, 300 + index, one_value_layout())
            sockets.append(learner.sock)
            write_frame(learner.sock, MSG_PUSH_FRAGMENT, payload)
            assert message in receive_error(learner.sock)

        learner = RawLearner(proc.port, 0, 400, one_value_layout())
        sockets.append(learner.sock)
        write_frame(
            learner.sock,
            MSG_INIT_PARAMS,
            struct.pack("<I", 99) + struct.pack("<f", 0.0),
        )
        assert "unknown fragment" in receive_error(learner.sock)
    finally:
        for sock in sockets:
            sock.close()
        stop_process(proc)


@pytest.mark.timeout(30)
def test_huge_first_frame_header_is_rejected_without_payload(
    syncer_binary, tmp_path
):
    proc, _ = launch_syncer(syncer_binary, tmp_path)
    sock = None
    try:
        sock = connect(proc.port)
        sock.settimeout(2)
        sock.sendall(_HEADER.pack(MAGIC, MSG_HELLO, 1 << 40))
        started = time.monotonic()
        message = receive_error(sock)
        assert time.monotonic() - started < 1.0
        assert "exceeds limit" in message
    finally:
        if sock is not None:
            sock.close()
        stop_process(proc)
