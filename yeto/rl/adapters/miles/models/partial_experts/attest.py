"""Selection-file identity: SHA256 of files and of canonical JSON.

Every partial-experts run pins *which* experts it trains by a SHA256 that
travels with the checkpoint metadata, the release manifest and the trainer
start-up check. Two identities are used:

* the raw file digest (:func:`sha256_file`) -- the file as it was audited;
* a canonical-JSON digest (:func:`canonical_sha256`) of the fields that
  define the trainable set -- stable across re-serialization.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_HEX = frozenset("0123456789abcdef")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    """Compact, key-sorted JSON (the form every selection SHA256 is taken over)."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def require_sha256(value: Any, name: str) -> str:
    """``value`` as a lowercase 64-hex SHA256 string, or ``ValueError``."""

    if not isinstance(value, str):
        raise ValueError(f"{name} must be a SHA256 string")
    normalized = value.lower()
    if len(normalized) != 64 or any(character not in _HEX for character in normalized):
        raise ValueError(f"{name} must be a lowercase SHA256 string")
    return normalized


def verify_file_sha256(path: str | Path, expected: str | None, *, name: str = "selection") -> str:
    """The file's SHA256; raises when ``expected`` is given and differs."""

    actual = sha256_file(path)
    if expected is not None:
        wanted = require_sha256(expected, f"expected {name} SHA256")
        if actual != wanted:
            raise ValueError(f"{name} SHA256 mismatch: expected {wanted}, got {actual}")
    return actual
