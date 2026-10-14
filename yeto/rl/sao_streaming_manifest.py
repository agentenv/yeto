"""The final SAO streaming contract bundle, as the launcher reads it.

``tools/probes/build_tbench21_sao_streaming_contracts.py finalize`` writes one
runtime contract and one layout attestation per island plus a manifest that
names them and the frozen syncer semantic profile.  The launcher reads the
manifest to start one syncer per role with exactly that profile and to send
every island its own contract (P0-1.7b.3).

Import-light: no torch, ray or miles.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yeto.syncer_profile import SyncerSemanticProfile

MANIFEST_SCHEMA = "yeto.tbench21-sao-streaming-contracts.v1"
_HEX = frozenset("0123456789abcdef")


class SaoManifestError(ValueError):
    """The SAO streaming manifest is not a final, self-consistent bundle."""


@dataclass(frozen=True)
class ShippedFile:
    """A host file and the absolute path the island reads it from."""

    host_path: Path
    container_path: str
    sha256: str


@dataclass(frozen=True)
class SaoIslandContract:
    island_id: int
    context_sha256: str
    runtime: ShippedFile
    layout_attestation: ShippedFile
    # The syncer port of each role the runtime contract names; it is part of
    # every role session's identity, so the launch must serve exactly these.
    role_ports: tuple[tuple[str, int], ...]
    # Local rounds per sync (H); both roles share it.
    local_horizon: int


@dataclass(frozen=True)
class SaoStreamingManifest:
    profile: SyncerSemanticProfile
    expected_fragments: int
    islands: tuple[SaoIslandContract, ...]

    def island(self, learner_id: int) -> SaoIslandContract:
        return self.islands[learner_id]


def _sha256(name: str, value: Any) -> str:
    if type(value) is not str or len(value) != 64 or not set(value) <= _HEX:
        raise SaoManifestError(f"{name} must be a lowercase SHA256")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _shipped(name: str, record: Any) -> ShippedFile:
    if not isinstance(record, dict):
        raise SaoManifestError(f"{name} record is missing")
    host = Path(str(record.get("host_path", "")))
    container = record.get("container_path")
    sha256 = _sha256(f"{name} sha256", record.get("sha256"))
    if type(container) is not str or not container.startswith("/"):
        raise SaoManifestError(f"{name} container path must be absolute")
    if host.is_symlink() or not host.is_file() or _file_sha256(host) != sha256:
        raise SaoManifestError(f"{name} host file is missing or changed")
    return ShippedFile(host, container, sha256)


def _launch_fields(runtime: ShippedFile) -> tuple[tuple[tuple[str, int], ...], int]:
    """The role syncer ports and the shared local horizon the runtime names."""
    try:
        payload = json.loads(runtime.host_path.read_text(encoding="utf-8"))
        roles = [payload[role] for role in ("actor", "critic")]
        ports = tuple(
            (role, record["syncer"]["port"])
            for role, record in zip(("actor", "critic"), roles)
        )
        horizons = {record["local_horizon"] for record in roles}
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise SaoManifestError(
            "SAO streaming runtime names no role syncer ports or local horizon"
        ) from error
    if any(type(port) is not int or not 0 < port < 65536 for _role, port in ports):
        raise SaoManifestError("SAO streaming runtime role syncer port is invalid")
    if len(horizons) != 1 or any(type(h) is not int or h < 1 for h in horizons):
        raise SaoManifestError("SAO streaming runtime roles need one local horizon >= 1")
    return ports, horizons.pop()


def load_sao_streaming_manifest(path: str | Path) -> SaoStreamingManifest:
    """Load a ``finalize`` manifest; every shipped file must match its hash."""

    source = Path(path).expanduser()
    if source.is_symlink() or not source.is_file():
        raise SaoManifestError("SAO streaming manifest is not a regular file")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SaoManifestError("SAO streaming manifest is malformed") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != MANIFEST_SCHEMA
        or payload.get("phase") != "final"
    ):
        raise SaoManifestError("SAO streaming manifest is not a final contract bundle")
    record = payload.get("syncer_profile")
    if not isinstance(record, dict):
        raise SaoManifestError("SAO streaming manifest has no syncer profile")
    profile = SyncerSemanticProfile.from_mapping(record.get("profile"))
    profile.validate_sao()
    if profile.sha256 != record.get("semantic_sha256"):
        raise SaoManifestError("SAO syncer profile hash does not match its profile")
    fragments = payload.get("expected_fragments")
    if type(fragments) is not int or fragments < 1 or profile.total_steps != fragments:
        raise SaoManifestError("SAO syncer profile differs from the expected fragments")
    raw_islands = payload.get("islands")
    if not isinstance(raw_islands, list) or len(raw_islands) != profile.learners:
        raise SaoManifestError("SAO manifest must name one contract per learner")
    islands = []
    for island_id, raw in enumerate(raw_islands):
        if not isinstance(raw, dict) or raw.get("island_id") != island_id:
            raise SaoManifestError("SAO manifest island order changed")
        context = raw.get("sao_context")
        runtime = _shipped("SAO streaming runtime", raw.get("streaming_context"))
        role_ports, local_horizon = _launch_fields(runtime)
        islands.append(
            SaoIslandContract(
                island_id=island_id,
                context_sha256=_sha256(
                    "SAO context sha256",
                    context.get("sha256") if isinstance(context, dict) else None,
                ),
                runtime=runtime,
                layout_attestation=_shipped(
                    "SAO layout attestation", raw.get("layout_attestation")
                ),
                role_ports=role_ports,
                local_horizon=local_horizon,
            )
        )
    return SaoStreamingManifest(profile, fragments, tuple(islands))
