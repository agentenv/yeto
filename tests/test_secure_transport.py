"""TLS 1.3 mutual-authentication and protocol-v5 run-identity regressions."""

from __future__ import annotations

import socket
import ssl
import subprocess
import time
from pathlib import Path

import pytest

from yeto.fragments import MERGE_RDA, Fragment, FragmentLayout
from yeto.protocol import (
    DTYPE_F32,
    MSG_DATA_HELLO,
    MSG_ERROR,
    MSG_HELLO,
    encode_data_hello,
    encode_hello,
    read_frame,
    write_frame,
)
from yeto.security import RunSecurity, prepare_run_security

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def syncer_binary() -> Path:
    subprocess.run(["cargo", "build", "-q"], cwd=ROOT / "syncer", check=True)
    return ROOT / "syncer/target/debug/yeto-syncer"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _layout() -> FragmentLayout:
    return FragmentLayout([Fragment(MERGE_RDA, [("model.weight", 1)])])


def _launch(syncer_binary: Path, bundle: RunSecurity, port: int) -> subprocess.Popen:
    return subprocess.Popen(
        [
            str(syncer_binary),
            "--bind-address",
            "127.0.0.1",
            "--port",
            str(port),
            "--learners",
            "2",
            "--quorum",
            "1",
            "--total-steps",
            "1",
            "--run-id-file",
            str(bundle.run_id_file),
            "--tls-cert",
            str(bundle.server_cert),
            "--tls-key",
            str(bundle.server_key),
            "--tls-client-ca",
            str(bundle.ca_cert),
            "--tls-client-fingerprints",
            str(bundle.fingerprint_allowlist),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _context(
    ca: Path,
    cert: Path | None = None,
    key: Path | None = None,
) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.maximum_version = ssl.TLSVersion.TLSv1_3
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_verify_locations(cafile=str(ca))
    if cert is not None and key is not None:
        context.load_cert_chain(str(cert), str(key))
    return context


def _connect(
    port: int,
    context: ssl.SSLContext,
    server_name: str,
) -> ssl.SSLSocket:
    deadline = time.monotonic() + 10
    while True:
        try:
            raw = socket.create_connection(("127.0.0.1", port), timeout=1)
            return context.wrap_socket(raw, server_hostname=server_name)
        except ConnectionRefusedError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.02)


def _error(sock: ssl.SSLSocket) -> str:
    msg_type, payload = read_frame(sock)
    assert msg_type == MSG_ERROR
    return payload.decode("utf-8", errors="replace")


@pytest.mark.timeout(30)
def test_tls_rejects_wrong_ca_wrong_san_and_missing_client_certificate(
    syncer_binary,
    tmp_path,
):
    bundle = prepare_run_security(tmp_path / "correct", 2, "syncer.secure.test")
    wrong = prepare_run_security(tmp_path / "wrong", 2, "wrong.secure.test")
    port = _free_port()
    proc = _launch(syncer_binary, bundle, port)
    try:
        with pytest.raises(ssl.SSLCertVerificationError):
            _connect(
                port,
                _context(wrong.ca_cert, bundle.learner_cert(0), bundle.learner_key(0)),
                bundle.server_name,
            )
        with pytest.raises(ssl.SSLCertVerificationError, match="Hostname mismatch"):
            _connect(
                port,
                _context(bundle.ca_cert, bundle.learner_cert(0), bundle.learner_key(0)),
                "different.secure.test",
            )
        without_client = _context(bundle.ca_cert)
        with pytest.raises(ssl.SSLError):
            sock = _connect(port, without_client, bundle.server_name)
            try:
                write_frame(
                    sock,
                    MSG_HELLO,
                    encode_hello(
                        0, DTYPE_F32, _layout(), 0, 1, bundle.run_id_file.read_bytes()
                    ),
                )
                read_frame(sock)
            finally:
                sock.close()
    finally:
        proc.kill()
        proc.wait(timeout=5)


@pytest.mark.timeout(30)
def test_tls_binds_learner_id_and_every_data_stream_to_the_allowlisted_certificate(
    syncer_binary,
    tmp_path,
):
    bundle = prepare_run_security(tmp_path / "identity", 2, "syncer.identity.test")
    run_id = bundle.run_id_file.read_bytes()
    port = _free_port()
    proc = _launch(syncer_binary, bundle, port)
    sockets = []
    try:
        swapped = _connect(
            port,
            _context(bundle.ca_cert, bundle.learner_cert(1), bundle.learner_key(1)),
            bundle.server_name,
        )
        sockets.append(swapped)
        write_frame(
            swapped, MSG_HELLO, encode_hello(0, DTYPE_F32, _layout(), 1, 11, run_id)
        )
        assert "not authorized for claimed learner ID" in _error(swapped)

        control = _connect(
            port,
            _context(bundle.ca_cert, bundle.learner_cert(0), bundle.learner_key(0)),
            bundle.server_name,
        )
        sockets.append(control)
        write_frame(
            control, MSG_HELLO, encode_hello(0, DTYPE_F32, _layout(), 1, 12, run_id)
        )

        swapped_reconnect = _connect(
            port,
            _context(bundle.ca_cert, bundle.learner_cert(1), bundle.learner_key(1)),
            bundle.server_name,
        )
        sockets.append(swapped_reconnect)
        write_frame(
            swapped_reconnect,
            MSG_HELLO,
            encode_hello(0, DTYPE_F32, _layout(), 1, 13, run_id),
        )
        assert "not authorized for claimed learner ID" in _error(swapped_reconnect)

        mismatched_data = _connect(
            port,
            _context(bundle.ca_cert, bundle.learner_cert(1), bundle.learner_key(1)),
            bundle.server_name,
        )
        sockets.append(mismatched_data)
        write_frame(
            mismatched_data,
            MSG_DATA_HELLO,
            encode_data_hello(0, 12, 0, run_id),
        )
        assert "not authorized for claimed learner ID" in _error(mismatched_data)
    finally:
        for sock in sockets:
            sock.close()
        proc.kill()
        proc.wait(timeout=5)


@pytest.mark.timeout(30)
def test_protocol_v5_rejects_wrong_run_id_on_control_and_data_streams(
    syncer_binary,
    tmp_path,
):
    bundle = prepare_run_security(tmp_path / "run-id", 2, "syncer.run-id.test")
    run_id = bundle.run_id_file.read_bytes()
    wrong_run_id = bytes(value ^ 0xFF for value in run_id)
    context = _context(bundle.ca_cert, bundle.learner_cert(0), bundle.learner_key(0))
    port = _free_port()
    proc = _launch(syncer_binary, bundle, port)
    sockets = []
    try:
        wrong_control = _connect(port, context, bundle.server_name)
        sockets.append(wrong_control)
        write_frame(
            wrong_control,
            MSG_HELLO,
            encode_hello(0, DTYPE_F32, _layout(), 1, 21, wrong_run_id),
        )
        assert "run ID does not match" in _error(wrong_control)

        control = _connect(port, context, bundle.server_name)
        sockets.append(control)
        write_frame(
            control, MSG_HELLO, encode_hello(0, DTYPE_F32, _layout(), 1, 22, run_id)
        )
        wrong_data = _connect(port, context, bundle.server_name)
        sockets.append(wrong_data)
        write_frame(
            wrong_data,
            MSG_DATA_HELLO,
            encode_data_hello(0, 22, 0, wrong_run_id),
        )
        assert "run ID does not match" in _error(wrong_data)
    finally:
        for sock in sockets:
            sock.close()
        proc.kill()
        proc.wait(timeout=5)


def test_non_loopback_plaintext_and_partial_tls_configuration_fail_closed(
    syncer_binary,
    tmp_path,
):
    run_id = tmp_path / "run-id.bin"
    run_id.write_bytes(bytes(range(32)))
    run_id.chmod(0o600)
    base = [
        str(syncer_binary),
        "--bind-address",
        "0.0.0.0",
        "--learners",
        "1",
        "--quorum",
        "1",
        "--total-steps",
        "1",
        "--run-id-file",
        str(run_id),
    ]
    plaintext = subprocess.run(
        [*base, "--allow-insecure-loopback"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert plaintext.returncode != 0
    assert "non-loopback listeners require TLS" in plaintext.stderr

    partial = subprocess.run(
        [*base, "--tls-cert", str(tmp_path / "missing.crt")],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert partial.returncode != 0
    assert "TLS requires all" in partial.stderr
