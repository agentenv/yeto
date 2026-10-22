"""Wire format of peer recovery and of the HELLO extension trailer.

Mirrors ``syncer/src/recovery.rs`` (all little-endian). The flow is described
there and in :mod:`yeto.rl.peer_recovery`, which drives it on the island side.

HELLO extension: the last 13 bytes of the HELLO body (before the HMAC) may be
``b"YEX1" | flags u8 | syncer_epoch u64``. flags bit 0: the epoch is set;
bit 1: the island asks for peer recovery. A client that sets neither sends no
trailer, so its HELLO bytes are unchanged.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

MSG_RECOVERY_CLOCK = 40
MSG_PEER_SNAPSHOT_REQ = 41
MSG_PEER_SNAPSHOT_READY = 42
MSG_RECOVERY_SOURCE = 43
MSG_RECOVERY_RESTORED = 44
MSG_RECOVERY_FAILED = 45
MSG_RECOVERY_DONE = 46

NO_DONOR = 0xFFFF_FFFF
SOURCE_PEER = 1
SOURCE_DEGRADED = 2
MAX_RECOVERY_FRAME = 1 << 20
MAX_TEXT = 4096

HELLO_EXT_MAGIC = b"YEX1"
_HELLO_EXT = struct.Struct("<4sBQ")
_FLAG_HAS_EPOCH = 1
_FLAG_RECOVER = 2
# Every PUSH_FRAGMENT of the session carries per-expert routed-token tags
# (yeto.protocol.encode_expert_tags); the syncer must run with expert token
# weights.
_FLAG_EXPERT_TOKENS = 4

# MSG_ERROR texts the client acts on (syncer/src/server.rs, membership.rs).
LEASE_EXPIRED_PREFIX = "island lease expired"
SYNCER_EPOCH_FENCE_PREFIX = "fenced: syncer_epoch"


def encode_hello_ext(syncer_epoch: int | None, recover: bool, expert_tokens: bool = False) -> bytes:
    """The trailer, or b"" when no field is set (legacy bytes)."""
    if syncer_epoch is None and not recover and not expert_tokens:
        return b""
    if syncer_epoch is not None and not 0 <= syncer_epoch <= 0xFFFF_FFFF_FFFF_FFFF:
        raise ValueError(f"syncer_epoch must fit u64, got {syncer_epoch}")
    flags = (
        (_FLAG_HAS_EPOCH if syncer_epoch is not None else 0)
        | (_FLAG_RECOVER if recover else 0)
        | (_FLAG_EXPERT_TOKENS if expert_tokens else 0)
    )
    return _HELLO_EXT.pack(HELLO_EXT_MAGIC, flags, syncer_epoch or 0)


class _Reader:
    def __init__(self, data: bytes | memoryview) -> None:
        self._data = bytes(data)
        self._pos = 0

    def take(self, n: int) -> bytes:
        if self._pos + n > len(self._data):
            raise ValueError("recovery message is truncated")
        out = self._data[self._pos:self._pos + n]
        self._pos += n
        return out

    def unpack(self, fmt: str) -> tuple:
        s = struct.Struct("<" + fmt)
        return s.unpack(self.take(s.size))

    def versions(self, fragments: int) -> tuple[int, ...]:
        (n,) = self.unpack("I")
        if n != fragments:
            raise ValueError(f"recovery message carries {n} fragment versions, layout has {fragments}")
        return self.unpack(f"{n}Q")

    def text(self) -> str:
        (n,) = self.unpack("I")
        if n > MAX_TEXT:
            raise ValueError(f"recovery text of {n} bytes exceeds {MAX_TEXT}")
        return self.take(n).decode("utf-8")

    def end(self) -> None:
        if self._pos != len(self._data):
            raise ValueError(f"recovery message has {len(self._data) - self._pos} trailing bytes")


def _versions(values) -> bytes:
    values = tuple(int(v) for v in values)
    return struct.pack(f"<I{len(values)}Q", len(values), *values)


def _text(text: str) -> bytes:
    raw = text.encode("utf-8")[:MAX_TEXT]
    return struct.pack("<I", len(raw)) + raw


@dataclass(frozen=True)
class RecoveryClock:
    """Syncer clock at the start of a recovery: ``t_s`` and per-fragment ``V_s``."""

    t_s: int
    versions: tuple[int, ...]
    donor: int | None

    @classmethod
    def decode(cls, payload, fragments: int) -> "RecoveryClock":
        r = _Reader(payload)
        (t_s,) = r.unpack("Q")
        versions = r.versions(fragments)
        (donor,) = r.unpack("I")
        r.end()
        return cls(t_s, versions, None if donor == NO_DONOR else donor)


@dataclass(frozen=True)
class SnapshotRequest:
    """Syncer -> donor: snapshot once every fragment version is >= ``versions``."""

    recovering_id: int
    recovering_generation: int
    t_s: int
    versions: tuple[int, ...]

    @classmethod
    def decode(cls, payload, fragments: int) -> "SnapshotRequest":
        r = _Reader(payload)
        recovering_id, generation, t_s = r.unpack("IQQ")
        versions = r.versions(fragments)
        r.end()
        return cls(recovering_id, generation, t_s, versions)


def encode_snapshot_ready(request: SnapshotRequest, versions, sha256: bytes, locator: str) -> bytes:
    if len(sha256) != 32:
        raise ValueError("snapshot sha256 must be 32 bytes")
    if not locator:
        raise ValueError("snapshot locator must not be empty")
    return (
        struct.pack("<IQ", request.recovering_id, request.recovering_generation)
        + _versions(versions)
        + sha256
        + _text(locator)
    )


@dataclass(frozen=True)
class RecoverySource:
    """Where the recovering island restores from.

    ``kind`` SOURCE_PEER: the donor snapshot at ``locator`` with ``sha256``,
    holding ``versions``. SOURCE_DEGRADED: no donor; ``text`` is the reason.
    """

    kind: int
    donor: int | None
    versions: tuple[int, ...]
    sha256: bytes
    text: str

    @property
    def is_peer(self) -> bool:
        return self.kind == SOURCE_PEER

    @classmethod
    def decode(cls, payload, fragments: int) -> "RecoverySource":
        r = _Reader(payload)
        (kind, donor) = r.unpack("BI")
        if kind not in (SOURCE_PEER, SOURCE_DEGRADED):
            raise ValueError(f"unknown recovery source kind {kind}")
        versions = r.versions(fragments)
        sha256 = r.take(32)
        text = r.text()
        r.end()
        return cls(kind, None if donor == NO_DONOR else donor, versions, sha256, text)


def encode_restored(versions) -> bytes:
    """``versions`` held after the restore, or None for "resend everything"."""
    if versions is None:
        return b"\x00" + _versions(())
    return b"\x01" + _versions(versions)


def encode_failed(reason: str) -> bytes:
    return _text(reason)


@dataclass(frozen=True)
class RecoveryDone:
    """The island is back on the roster. It holds ``versions`` once the
    replayed fragments (sent on the data streams) have arrived."""

    t_s: int
    done_step: int
    replayed: int
    versions: tuple[int, ...]

    @property
    def steps(self) -> int:
        return max(0, self.done_step - self.t_s)

    @classmethod
    def decode(cls, payload, fragments: int) -> "RecoveryDone":
        r = _Reader(payload)
        t_s, done_step, replayed = r.unpack("QQI")
        versions = r.versions(fragments)
        r.end()
        return cls(t_s, done_step, replayed, versions)
