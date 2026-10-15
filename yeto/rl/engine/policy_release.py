"""Policy releases: trainer -> sampler weight hand-off through the store (L4.6, D5).

A trainer island whose samplers run elsewhere releases every published policy
version into the checkpoint store; a sampler island loads the newest release
into its inference engines. The store is the one both islands already mount for
the sample buffer (local directory, bucket mount or Modal Volume), so the
hand-off needs no new network path. Layout under ``policy-releases/``:

* ``r<seq>-v<version>/``: the weights in the engine's load format, written by
  the engine adapter, then ``RELEASE.json`` in it (the release record). ``seq``
  counts the releases of the store (1, 2, ...). ``weights`` names the format:
  ``full`` (a whole HF checkpoint) or ``lora`` (an HF/PEFT adapter directory the
  sampler hot-swaps over its base weights);
* ``LATEST.json``: a copy of the newest release record, written last. It is the
  commit point: a reader never sees a release whose weights are incomplete.

The record carries the policy version (the engine weight version the sampler
stamps), the version-independent policy hash, the trainer's buffer read
position when it released (``buffer_selected``; the sampler's backlog is the
buffer length minus this position), ``sha256`` (the digest of the weight files,
:func:`directory_digest`; the sampler checks it before it loads; LoRA releases
only: reading a whole checkpoint back is as slow as loading it) and, when the
trainer has one, ``engine_checksum`` (the digest of the trainer engines' weight
read-back after this publish; a sampler that reads its own engines back after
the load must get the same value).

A trainer that publishes the same version and hash again (a restart) releases
nothing. A trainer restored from an older cut publishes an older version with
other weights (a new lineage): that is a new release with a higher ``seq``.

:class:`PolicyFollower` is the sampler side: it loads a release when its ``seq``
differs from the one its engines serve. When the trainer stops, no new release
appears and the sampler keeps the last version.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .resume import CheckpointStore, _write_json_atomic

SCHEMA = "yeto.rl.policy-release/v1"
RELEASE_DIR = "policy-releases"
LATEST = "LATEST.json"
RECORD = "RELEASE.json"
_RELEASE_NAME = re.compile(r"r(\d+)-v\d+")
FULL = "full"
LORA = "lora"
WEIGHT_FORMATS = (FULL, LORA)
_DIGEST_CHUNK = 1 << 20


class PolicyReleaseError(RuntimeError):
    pass


@dataclass(frozen=True)
class PolicyRelease:
    seq: int
    version: int
    policy_hash: str
    path: str  # relative to the release root: "r<seq>-v<version>"
    buffer_selected: int | None = None
    released_at: float = 0.0
    weights: str = FULL
    sha256: str | None = None  # None: not digested (a full checkpoint)
    engine_checksum: str | None = None
    schema: str = SCHEMA

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "PolicyRelease":
        if record.get("schema") != SCHEMA:
            raise PolicyReleaseError(f"release schema {record.get('schema')!r} != {SCHEMA}")
        return cls(**{k: record[k] for k in record if k in cls.__dataclass_fields__})


def directory_digest(directory: Path) -> str:
    """sha256 over every file under ``directory`` except the release record:
    relative path, size and bytes, in sorted path order."""
    digest = hashlib.sha256()
    root = Path(directory)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.relative_to(root) != Path(RECORD))
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big") + relative)
        digest.update(os.path.getsize(path).to_bytes(8, "big"))
        with path.open("rb") as handle:
            while chunk := handle.read(_DIGEST_CHUNK):
                digest.update(chunk)
    return digest.hexdigest()


def check_release_files(release: PolicyRelease, directory: Path) -> None:
    """Raise when the weight files differ from the ones the trainer released."""
    if release.sha256 is None:
        return
    actual = directory_digest(directory)
    if actual != release.sha256:
        raise PolicyReleaseError(f"release r{release.seq}-v{release.version}: weight files "
                                 f"sha256 {actual} != released {release.sha256}")


class PolicyReleaseStore:
    """Writes (trainer) and reads (sampler) the releases in one store."""

    def __init__(self, store: CheckpointStore, *, keep: int = 2,
                 clock: Callable[[], float] = time.time) -> None:
        if int(keep) < 1:
            raise ValueError("keep must be >= 1")
        self.store = store
        self.keep = int(keep)
        self.clock = clock
        self.root = Path(store.root) / RELEASE_DIR

    def resolve(self, release: PolicyRelease) -> Path:
        """The release's weight directory as this island mounts the store."""
        return self.root / release.path

    def latest(self) -> PolicyRelease | None:
        self.store.reload()
        path = self.root / LATEST
        if not path.is_file():
            return None
        return PolicyRelease.from_record(json.loads(path.read_text(encoding="utf-8")))

    def release(self, version: int, policy_hash: str, write: Callable[[Path], None], *,
                buffer_selected: int | None = None, weights: str = FULL,
                engine_checksum: str | None = None) -> PolicyRelease | None:
        """Write the weights with ``write(directory)``, then the record, then LATEST.

        Returns None (writes nothing) when the latest release is this version and hash."""
        version = int(version)
        if not policy_hash:
            raise ValueError("policy_hash must be non-empty")
        if weights not in WEIGHT_FORMATS:
            raise ValueError(f"weights must be one of {WEIGHT_FORMATS}, not {weights!r}")
        latest = self.latest()
        if latest is not None and (latest.version, latest.policy_hash) == (version, policy_hash):
            return None
        seq = 1 if latest is None else latest.seq + 1
        name = f"r{seq}-v{version}"
        directory = self.root / name
        if directory.exists():  # a partial release of an earlier attempt (no LATEST names it)
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
        write(directory)
        record = PolicyRelease(seq=seq, version=version, policy_hash=str(policy_hash), path=name,
                               buffer_selected=None if buffer_selected is None else int(buffer_selected),
                               released_at=round(self.clock(), 3), weights=weights,
                               sha256=directory_digest(directory) if weights == LORA else None,
                               engine_checksum=engine_checksum or None)
        _write_json_atomic(directory / RECORD, asdict(record))
        self.store.commit()  # the weights and the record are durable before LATEST names them
        _write_json_atomic(self.root / LATEST, asdict(record))
        self.store.commit()
        self._prune()
        return record

    def _prune(self) -> list[str]:
        """Delete all but the newest ``keep`` releases (a sampler that is loading
        one of them fails the load and retries with the next LATEST)."""
        named = sorted((int(m.group(1)), p.name) for p in self.root.iterdir()
                       if p.is_dir() and (m := _RELEASE_NAME.fullmatch(p.name)))
        removed = [name for _, name in named[:-self.keep]]
        for name in removed:
            shutil.rmtree(self.root / name, ignore_errors=True)
        if removed:
            self.store.commit()
        return removed


class PolicyFollower:
    """Sampler side: load each new release into the engines.

    ``load(release, directory)`` loads the weights and stamps ``release.version``
    on every engine (raises on any failure; the served release is then unchanged).
    The weight files are checked against the record's ``sha256`` first.
    ``version`` is the version the engines serve before any release (the base
    policy, 0)."""

    def __init__(self, releases: PolicyReleaseStore,
                 load: Callable[[PolicyRelease, Path], None], *, version: int = 0) -> None:
        self.releases = releases
        self._load = load
        self.version = int(version)
        self.release: PolicyRelease | None = None  # the release the engines serve
        self.last_seen: PolicyRelease | None = None  # the newest release read

    def refresh(self) -> PolicyRelease | None:
        """The release loaded by this call, or None when there is no new one."""
        latest = self.releases.latest()
        self.last_seen = latest
        if latest is None or (self.release is not None and latest.seq == self.release.seq):
            return None
        directory = self.releases.resolve(latest)
        check_release_files(latest, directory)
        self._load(latest, directory)
        self.version = latest.version
        self.release = latest
        return latest
