"""Unit coverage for durable transport identities and fail-closed clients."""

from __future__ import annotations

import argparse
import json
import ssl
from pathlib import Path

import pytest

import yeto.protocol as protocol
from yeto.fragments import MERGE_RDA, Fragment, FragmentLayout
from yeto.protocol import (
    DTYPE_F32,
    ProtocolError,
    SyncerClient,
    SyncerTlsConfig,
    syncer_security_from_args,
)
from yeto.security import load_run_security, prepare_run_security


def _layout() -> FragmentLayout:
    return FragmentLayout([Fragment(MERGE_RDA, [("model.weight", 1)])])


def _mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_generated_bundle_is_private_distinct_and_durably_reused(tmp_path):
    root = tmp_path / "bundle"
    bundle = prepare_run_security(root, 3, "syncer.bundle.test")

    assert _mode(root) == 0o700
    assert all(
        not path.is_symlink() and _mode(path) == 0o600 for path in root.iterdir()
    )
    assert len(bundle.run_id_file.read_bytes()) == 32
    manifest = json.loads((root / "manifest.json").read_text())
    assert set(manifest["fingerprints"]) == {"0", "1", "2"}
    assert len(set(manifest["fingerprints"].values())) == 3
    assert (root / "ca.key").is_file()

    first_run_id = bundle.run_id_file.read_bytes()
    first_server_certificate = bundle.server_cert.read_bytes()
    reused = prepare_run_security(root, 3, "syncer.bundle.test")
    assert reused.run_id_file.read_bytes() == first_run_id
    assert reused.server_cert.read_bytes() == first_server_certificate

    with pytest.raises(ValueError, match="learner count/server name"):
        prepare_run_security(root, 2, "syncer.bundle.test")


def test_bundle_reuse_rejects_tampering_and_unsafe_modes(tmp_path):
    identity = prepare_run_security(tmp_path / "identity", 1, "syncer.identity.test")
    identity.learner_key(0).write_bytes(identity.server_key.read_bytes())
    with pytest.raises(ValueError, match="do not match"):
        load_run_security(identity.root)

    permissions = prepare_run_security(
        tmp_path / "permissions", 1, "syncer.permissions.test"
    )
    permissions.server_key.chmod(0o640)
    with pytest.raises(ValueError, match="unsafe permissions"):
        load_run_security(permissions.root)


def test_network_client_profiles_fail_closed_before_dial(tmp_path):
    run_id_file = tmp_path / "run-id.bin"
    run_id_file.write_bytes(bytes(range(32)))
    run_id_file.chmod(0o600)

    incomplete = argparse.Namespace(
        sync_run_id_file=str(run_id_file),
        sync_tls_ca=str(tmp_path / "ca.crt"),
        sync_tls_cert=None,
        sync_tls_key=None,
        sync_server_name=None,
        allow_insecure_loopback=False,
    )
    with pytest.raises(ValueError, match="requires --sync-tls-ca"):
        syncer_security_from_args(incomplete)

    with pytest.raises(ValueError, match="explicit allow_insecure_loopback"):
        SyncerClient(
            ("127.0.0.1", 29400),
            0,
            _layout(),
            DTYPE_F32,
            run_id=run_id_file.read_bytes(),
        )

    plaintext = SyncerClient(
        ("203.0.113.7", 29400),
        0,
        _layout(),
        DTYPE_F32,
        run_id=run_id_file.read_bytes(),
        allow_insecure_loopback=True,
    )
    with pytest.raises(ProtocolError, match="restricted to loopback"):
        plaintext._require_loopback_destination()


def test_reconnect_tls_authentication_error_is_immediately_fatal(
    tmp_path,
    monkeypatch,
):
    bundle = prepare_run_security(tmp_path / "tls", 1, "syncer.reconnect.test")
    client = SyncerClient(
        ("127.0.0.1", 29400),
        0,
        _layout(),
        DTYPE_F32,
        run_id=bundle.run_id_file.read_bytes(),
        tls=SyncerTlsConfig(
            bundle.ca_cert,
            bundle.learner_cert(0),
            bundle.learner_key(0),
            bundle.server_name,
        ),
    )

    def reject(*, patient):
        assert patient is False
        raise ssl.SSLCertVerificationError(1, "certificate verify failed")

    monkeypatch.setattr(client, "_connect_group", reject)
    monkeypatch.setattr(protocol, "RECONNECT_BACKOFF_START", 0.0)
    monkeypatch.setattr(protocol, "RECONNECT_BACKOFF_CAP", 0.0)
    client._failure.set()
    client._supervise()

    assert client._closed.is_set()
    assert client._reconnects_used == 1
    with pytest.raises(RuntimeError, match="TLS authentication failed"):
        client.check_health()
