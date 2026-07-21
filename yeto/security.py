"""Per-run TLS identities for the syncer WAN transport.

Issuance uses the system OpenSSL command and never prints private-key bytes.
The CA private key remains on the controller; remote tasks receive only their
own key, certificate, the CA certificate, and the non-secret run ID.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import secrets
import shutil
import ssl
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

SECURITY_BUNDLE_VERSION = 1
CERTIFICATE_VALIDITY_DAYS = 365
_SAFE_SERVER_NAME = re.compile(r"^[A-Za-z0-9.-]+$")


@dataclass(frozen=True)
class RunSecurity:
    root: Path
    server_name: str
    learners: int
    run_id_file: Path
    ca_cert: Path
    server_cert: Path
    server_key: Path
    fingerprint_allowlist: Path
    learner_certs: tuple[Path, ...]
    learner_keys: tuple[Path, ...]

    def learner_cert(self, learner_id: int) -> Path:
        return self.learner_certs[learner_id]

    def learner_key(self, learner_id: int) -> Path:
        return self.learner_keys[learner_id]


def _private_write(path: Path, value: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _openssl_output(arguments: list[str], *, cwd: Path) -> str:
    executable = shutil.which("openssl")
    if executable is None:
        raise RuntimeError(
            "OpenSSL is required for automatic per-run certificate issuance; "
            "install it or provision a validated security bundle explicitly"
        )
    completed = subprocess.run(
        [executable, *arguments],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit status {completed.returncode}"
        raise RuntimeError(f"OpenSSL certificate provisioning failed: {detail}")
    return completed.stdout


def _run_openssl(arguments: list[str], *, cwd: Path) -> None:
    _openssl_output(arguments, cwd=cwd)


def _certificate_fingerprint(path: Path) -> str:
    der = ssl.PEM_cert_to_DER_cert(path.read_text())
    return hashlib.sha256(der).hexdigest()


def _validate_private_mode(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"security bundle file is missing: {path}")
    if os.name == "posix" and path.stat().st_mode & 0o077:
        raise ValueError(f"security bundle file has unsafe permissions: {path}")


def _validate_key_pair(certificate: Path, private_key: Path, label: str) -> None:
    try:
        certificate_public_key = _openssl_output(
            ["x509", "-in", certificate.name, "-pubkey", "-noout"],
            cwd=certificate.parent,
        )
        private_public_key = _openssl_output(
            ["pkey", "-in", private_key.name, "-pubout"],
            cwd=private_key.parent,
        )
    except RuntimeError as exc:
        raise ValueError(f"security bundle {label} key pair is invalid") from exc
    if certificate_public_key != private_public_key:
        raise ValueError(
            f"security bundle {label} certificate and private key do not match"
        )


def _validate_certificate_profiles(bundle: RunSecurity) -> None:
    try:
        _run_openssl(
            [
                "verify",
                "-CAfile",
                bundle.ca_cert.name,
                "-purpose",
                "sslserver",
                bundle.server_cert.name,
            ],
            cwd=bundle.root,
        )
        try:
            ipaddress.ip_address(bundle.server_name)
        except ValueError:
            identity_check = "-checkhost"
        else:
            identity_check = "-checkip"
        _run_openssl(
            [
                "x509",
                "-in",
                bundle.server_cert.name,
                "-noout",
                identity_check,
                bundle.server_name,
            ],
            cwd=bundle.root,
        )
        for certificate in bundle.learner_certs:
            _run_openssl(
                [
                    "verify",
                    "-CAfile",
                    bundle.ca_cert.name,
                    "-purpose",
                    "sslclient",
                    certificate.name,
                ],
                cwd=bundle.root,
            )
    except RuntimeError as exc:
        raise ValueError(
            "security bundle certificate chain, lifetime, purpose, or server identity is invalid"
        ) from exc

    _validate_key_pair(bundle.server_cert, bundle.server_key, "server")
    for learner_id, (certificate, private_key) in enumerate(
        zip(bundle.learner_certs, bundle.learner_keys, strict=True)
    ):
        _validate_key_pair(certificate, private_key, f"learner {learner_id}")


def _bundle_from_manifest(root: Path, manifest: dict) -> RunSecurity:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(
            f"security bundle directory is missing or is a symlink: {root}"
        )
    if os.name == "posix" and root.stat().st_mode & 0o077:
        raise ValueError(f"security bundle directory has unsafe permissions: {root}")
    learners = manifest.get("learners")
    server_name = manifest.get("server_name")
    if isinstance(learners, bool) or not isinstance(learners, int) or learners < 1:
        raise ValueError("security bundle manifest has an invalid learner count")
    if not isinstance(server_name, str) or not server_name:
        raise ValueError("security bundle manifest has an invalid server name")
    bundle = RunSecurity(
        root=root,
        server_name=server_name,
        learners=learners,
        run_id_file=root / "run-id.bin",
        ca_cert=root / "ca.crt",
        server_cert=root / "server.crt",
        server_key=root / "server.key",
        fingerprint_allowlist=root / "client-fingerprints.txt",
        learner_certs=tuple(root / f"learner-{index}.crt" for index in range(learners)),
        learner_keys=tuple(root / f"learner-{index}.key" for index in range(learners)),
    )
    for path in (
        bundle.run_id_file,
        bundle.ca_cert,
        bundle.server_cert,
        bundle.server_key,
        bundle.fingerprint_allowlist,
        *bundle.learner_certs,
        *bundle.learner_keys,
    ):
        _validate_private_mode(path)
    if len(bundle.run_id_file.read_bytes()) != 32:
        raise ValueError("security bundle run ID must contain exactly 32 raw bytes")
    manifest_fingerprints = manifest.get("fingerprints")
    if not isinstance(manifest_fingerprints, dict):
        raise ValueError("security bundle manifest has no learner fingerprints")
    expected = {
        str(key): str(value).lower() for key, value in manifest_fingerprints.items()
    }
    expected_ids = {str(index) for index in range(learners)}
    if set(expected) != expected_ids or any(
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in expected.values()
    ):
        raise ValueError("security bundle manifest has invalid learner fingerprints")
    actual = {
        str(index): _certificate_fingerprint(path)
        for index, path in enumerate(bundle.learner_certs)
    }
    if actual != expected or len(set(actual.values())) != learners:
        raise ValueError(
            "security bundle learner certificate fingerprints do not match manifest"
        )
    allowlisted: dict[str, str] = {}
    for line in bundle.fingerprint_allowlist.read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError("security bundle fingerprint allowlist is malformed")
        learner_id, fingerprint = fields
        if learner_id in allowlisted:
            raise ValueError(
                "security bundle fingerprint allowlist has a duplicate learner ID"
            )
        allowlisted[learner_id] = fingerprint.lower()
    if allowlisted != expected:
        raise ValueError(
            "security bundle fingerprint allowlist does not match certificates"
        )
    if manifest.get("ca_fingerprint") != _certificate_fingerprint(bundle.ca_cert):
        raise ValueError("security bundle CA certificate does not match manifest")
    if manifest.get("server_fingerprint") != _certificate_fingerprint(
        bundle.server_cert
    ):
        raise ValueError("security bundle server certificate does not match manifest")
    _validate_certificate_profiles(bundle)
    return bundle


def load_run_security(root: str | os.PathLike[str]) -> RunSecurity:
    root = Path(root).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"security bundle manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != SECURITY_BUNDLE_VERSION:
        raise ValueError("security bundle version is unsupported")
    return _bundle_from_manifest(root, manifest)


def prepare_run_security(
    root: str | os.PathLike[str],
    learners: int,
    server_name: str,
) -> RunSecurity:
    """Create or strictly reuse one durable, per-run certificate bundle."""
    if learners < 1:
        raise ValueError("security bundle requires at least one learner")
    if not server_name or not _SAFE_SERVER_NAME.fullmatch(server_name):
        raise ValueError(
            "server name must be a DNS name or IP address without shell metacharacters"
        )
    try:
        ipaddress.ip_address(server_name)
    except ValueError:
        san = f"DNS:{server_name}"
    else:
        san = f"IP:{server_name}"

    root = Path(root).expanduser().resolve()
    if (root / "manifest.json").exists():
        existing = load_run_security(root)
        if existing.learners != learners or existing.server_name != server_name:
            raise ValueError(
                "existing security bundle identity does not match learner count/server name"
            )
        return existing
    if root.exists() and any(root.iterdir()):
        raise ValueError(
            f"refusing to issue into nonempty security directory without manifest: {root}"
        )

    root.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent)
    )
    os.chmod(temporary, 0o700)
    try:
        _private_write(temporary / "run-id.bin", secrets.token_bytes(32))
        _run_openssl(
            [
                "genpkey",
                "-algorithm",
                "EC",
                "-pkeyopt",
                "ec_paramgen_curve:P-256",
                "-out",
                "ca.key",
            ],
            cwd=temporary,
        )
        _run_openssl(
            [
                "req",
                "-x509",
                "-new",
                "-sha256",
                "-key",
                "ca.key",
                "-days",
                str(CERTIFICATE_VALIDITY_DAYS),
                "-subj",
                "/CN=Yeto per-run transport CA",
                "-addext",
                "basicConstraints=critical,CA:TRUE",
                "-addext",
                "keyUsage=critical,keyCertSign,cRLSign",
                "-out",
                "ca.crt",
            ],
            cwd=temporary,
        )

        def issue(name: str, common_name: str, extensions: str) -> None:
            _run_openssl(
                [
                    "genpkey",
                    "-algorithm",
                    "EC",
                    "-pkeyopt",
                    "ec_paramgen_curve:P-256",
                    "-out",
                    f"{name}.key",
                ],
                cwd=temporary,
            )
            _run_openssl(
                [
                    "req",
                    "-new",
                    "-key",
                    f"{name}.key",
                    "-subj",
                    f"/CN={common_name}",
                    "-out",
                    f"{name}.csr",
                ],
                cwd=temporary,
            )
            extension_path = temporary / f"{name}.ext"
            _private_write(extension_path, extensions.encode())
            _run_openssl(
                [
                    "x509",
                    "-req",
                    "-sha256",
                    "-days",
                    str(CERTIFICATE_VALIDITY_DAYS),
                    "-in",
                    f"{name}.csr",
                    "-CA",
                    "ca.crt",
                    "-CAkey",
                    "ca.key",
                    "-CAcreateserial",
                    "-extfile",
                    f"{name}.ext",
                    "-out",
                    f"{name}.crt",
                ],
                cwd=temporary,
            )
            (temporary / f"{name}.csr").unlink()
            extension_path.unlink()

        issue(
            "server",
            server_name,
            f"basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature\nextendedKeyUsage=serverAuth\nsubjectAltName={san}\n",
        )
        fingerprints = {}
        for learner_id in range(learners):
            name = f"learner-{learner_id}"
            issue(
                name,
                f"yeto-learner-{learner_id}",
                "basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature\nextendedKeyUsage=clientAuth\n",
            )
            fingerprints[str(learner_id)] = _certificate_fingerprint(
                temporary / f"{name}.crt"
            )
        if len(set(fingerprints.values())) != learners:
            raise RuntimeError(
                "certificate issuance produced duplicate learner identities"
            )
        allowlist = "".join(
            f"{learner_id} {fingerprints[str(learner_id)]}\n"
            for learner_id in range(learners)
        )
        _private_write(temporary / "client-fingerprints.txt", allowlist.encode())
        manifest = {
            "version": SECURITY_BUNDLE_VERSION,
            "server_name": server_name,
            "learners": learners,
            "fingerprints": fingerprints,
            "ca_fingerprint": _certificate_fingerprint(temporary / "ca.crt"),
            "server_fingerprint": _certificate_fingerprint(temporary / "server.crt"),
        }
        _private_write(
            temporary / "manifest.json",
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
        )
        (temporary / "ca.srl").unlink(missing_ok=True)
        for path in temporary.iterdir():
            os.chmod(path, 0o600)
        if root.exists():
            root.rmdir()
        os.replace(temporary, root)
        os.chmod(root, 0o700)
        temporary = None
        if os.name == "posix":
            directory_descriptor = os.open(root.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        return load_run_security(root)
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
