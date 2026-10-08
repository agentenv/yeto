"""Backend identity, hashed beside the algorithm and contract hashes
(yeto-framework-decoupling phase 5, tasks 6.1 / 6.2, design D7, audit E23).

``AlgorithmSpec.sha256()`` and ``ExecutionProfile.contract_hash`` say *what*
is trained; they do not say *which training stack* runs it.  Two islands with
the same algorithm but different engines (Miles vs verl), engine commits,
device families or parameter-name maps must not average their deltas.

* :class:`BackendIdentity` -- ``{engine, engine_commit, device_family,
  param_map_sha256}`` with its own hash (the old two hashes are unchanged);
  each adapter supplies its value (Miles: ``yeto.rl.adapters.miles.identity``).
* :func:`session_contract_hash` -- what an RL island sends as the syncer HELLO
  session contract: the tensor-layout fingerprint bound to the identity hash.
  The syncer only admits learners whose contract is byte-identical, so a Miles
  island and a verl island (or two different Miles pins) are refused at the
  handshake, while two islands of the same backend agree as before.
* :func:`check_identity_match` -- the same comparison where both identities
  are known (launcher, island status), naming both sides in the error.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping

IDENTITY_SCHEMA = "yeto-backend-identity-v1"
SESSION_CONTRACT_DOMAIN = b"yeto-rl-session-contract-v2\0"


class BackendIdentityMismatch(ValueError):
    """Two islands run different training backends; they must not be merged."""


def _sha256_hex(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase sha256 hex digest")
    return value


@dataclass(frozen=True)
class BackendIdentity:
    engine: str  # registry name: "miles", "verl", ...
    engine_commit: str  # pinned engine source commit
    device_family: str  # "nvidia", "ascend", ... (yeto/hw/device.py later)
    param_map_sha256: str  # hash of the backend's parameter-name map to canonical names

    def __post_init__(self) -> None:
        for name in ("engine", "engine_commit", "device_family"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"backend identity {name} must be a non-empty string")
        _sha256_hex(self.param_map_sha256, "param_map_sha256")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": IDENTITY_SCHEMA, **asdict(self)}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "BackendIdentity":
        if raw.get("schema") != IDENTITY_SCHEMA:
            raise ValueError(f"unknown backend identity schema {raw.get('schema')!r}")
        return cls(str(raw["engine"]), str(raw["engine_commit"]), str(raw["device_family"]),
                   str(raw["param_map_sha256"]))

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


def param_map_sha256(param_map: Mapping[str, Any]) -> str:
    """Hash of a parameter-name map (canonical JSON)."""
    text = json.dumps(dict(param_map), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def session_contract_hash(layout_fingerprint: bytes, identity_sha256: str) -> bytes:
    """32-byte syncer HELLO session contract: layout fingerprint bound to the backend identity."""
    if not isinstance(layout_fingerprint, bytes) or len(layout_fingerprint) != 32:
        raise ValueError("layout fingerprint must be 32 bytes")
    digest = bytes.fromhex(_sha256_hex(identity_sha256, "backend identity sha256"))
    return hashlib.sha256(SESSION_CONTRACT_DOMAIN + layout_fingerprint + digest).digest()


def check_identity_match(local: BackendIdentity, peer: BackendIdentity) -> None:
    if local.sha256() != peer.sha256():
        raise BackendIdentityMismatch(
            "backend identity differs, islands cannot be merged: "
            f"local {local.to_dict()} ({local.sha256()[:12]}) vs "
            f"peer {peer.to_dict()} ({peer.sha256()[:12]})")
