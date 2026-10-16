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
the load must get the same value). ``files`` lists every weight file with its
size; ``buffer_shards`` is the trainer's read position per sampler shard when
the buffer is sharded (``sample_buffer`` module doc).

Complete releases only (rl-cross-site-sample-store D4): a reader loads a release
only when every file in ``files`` exists with that size, in the mount or as an
object. Otherwise (an upload still running, a store whose overwrite is not yet
visible) :class:`PolicyFollower` keeps the release it serves and tries again on
the next refresh. The check does not depend on the store's consistency model.
Records without ``files`` (written before it existed) are not checked.

A store with an object API (``CheckpointStore.objects``: an S3 bucket whose
mount may serve a stale copy of a replaced object) is read through that API:
``LATEST.json`` by key, and a release's weight files downloaded into a local
cache directory before the load (streamed to disk: a release may be hundreds of
GB). The trainer still writes through the mount
(each file is new and written once).

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

from .object_store import ObjectStore, download_file, object_size
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
DEFAULT_CACHE_DIR = "~/.cache/yeto/policy-releases"


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
    files: dict[str, int] | None = None  # weight file (relative path) -> size in bytes
    buffer_shards: dict[str, int] | None = None  # sampler shard -> trainer read position
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


class ReleaseIncomplete(PolicyReleaseError):
    """A release whose files are not all present with their recorded sizes yet."""


def file_sizes(directory: Path) -> dict[str, int]:
    """{relative path: size} of every file under ``directory`` but the record."""
    root = Path(directory)
    return {p.relative_to(root).as_posix(): p.stat().st_size for p in sorted(root.rglob("*"))
            if p.is_file() and p.relative_to(root) != Path(RECORD)}


def _incomplete(release: PolicyRelease, sizes: dict[str, int | None]) -> ReleaseIncomplete | None:
    """The error for files of ``release.files`` that ``sizes`` lacks or sizes differently."""
    bad = sorted(name for name, size in (release.files or {}).items() if sizes.get(name) != size)
    if not bad:
        return None
    return ReleaseIncomplete(f"release r{release.seq}-v{release.version}: {len(bad)} of "
                             f"{len(release.files or {})} files missing or of another size (first: {bad[0]})")


def check_complete(release: PolicyRelease, directory: Path) -> None:
    """Raise :class:`ReleaseIncomplete` unless every recorded file is in ``directory``
    with its recorded size (no check for a record without ``files``)."""
    if release.files is None:
        return
    root = Path(directory)
    sizes = {name: (root / name).stat().st_size if (root / name).is_file() else None for name in release.files}
    if (error := _incomplete(release, sizes)) is not None:
        raise error


def check_release_files(release: PolicyRelease, directory: Path) -> None:
    """Raise when the weight files differ from the ones the trainer released."""
    if release.sha256 is None:
        return
    actual = directory_digest(directory)
    if actual != release.sha256:
        raise PolicyReleaseError(f"release r{release.seq}-v{release.version}: weight files "
                                 f"sha256 {actual} != released {release.sha256}")


class PolicyReleaseStore:
    """Writes (trainer) and reads (sampler) the releases in one store.
    ``cache_dir``: where a store with an object API downloads a release to."""

    def __init__(self, store: CheckpointStore, *, keep: int = 2,
                 clock: Callable[[], float] = time.time,
                 cache_dir: str | os.PathLike[str] = DEFAULT_CACHE_DIR) -> None:
        if int(keep) < 1:
            raise ValueError("keep must be >= 1")
        self.store = store
        self.keep = int(keep)
        self.clock = clock
        self.root = Path(store.root) / RELEASE_DIR
        self.cache_dir = Path(cache_dir).expanduser()

    def resolve(self, release: PolicyRelease) -> Path:
        """The release's weight directory on this island: in the store mount, or
        downloaded from the store's object API. Raises :class:`ReleaseIncomplete`
        when a recorded file is missing or of another size."""
        objects = self.store.objects
        if objects is None:
            directory = self.root / release.path
            check_complete(release, directory)
            return directory
        return self._download(objects, release)

    def _download(self, objects: ObjectStore, release: PolicyRelease) -> Path:
        """Download every object of the release into ``cache_dir/<path>`` (once:
        a complete download is renamed into place) and drop older downloads.
        The recorded files are checked against the object sizes first."""
        target = self.cache_dir / release.path
        if not target.is_dir():
            prefix = f"{RELEASE_DIR}/{release.path}/"
            keys = objects.list(prefix)
            if not keys:
                raise ReleaseIncomplete(f"release r{release.seq}-v{release.version}: no objects under {prefix}")
            if release.files is not None:
                listed = {key[len(prefix):] for key in keys}
                sizes = {name: object_size(objects, prefix + name) if name in listed else None
                         for name in release.files}
                if (error := _incomplete(release, sizes)) is not None:
                    raise error
            staging = self.cache_dir / f".{release.path}.partial"
            shutil.rmtree(staging, ignore_errors=True)
            for key in keys:
                path = staging / key[len(prefix):]
                path.parent.mkdir(parents=True, exist_ok=True)
                if not download_file(objects, key, path):
                    raise ReleaseIncomplete(f"release object {key} disappeared during the download")
            check_complete(release, staging)
            staging.replace(target)
        for old in self.cache_dir.iterdir():
            if old.is_dir() and old.name != release.path:
                shutil.rmtree(old, ignore_errors=True)
        return target

    def latest(self) -> PolicyRelease | None:
        objects = self.store.objects
        if objects is not None:
            raw = objects.read(f"{RELEASE_DIR}/{LATEST}")
            return None if raw is None else PolicyRelease.from_record(json.loads(raw))
        self.store.reload()
        path = self.root / LATEST
        if not path.is_file():
            return None
        return PolicyRelease.from_record(json.loads(path.read_text(encoding="utf-8")))

    def release(self, version: int, policy_hash: str, write: Callable[[Path], None], *,
                buffer_selected: int | None = None, weights: str = FULL,
                engine_checksum: str | None = None,
                buffer_shards: dict[str, int] | None = None) -> PolicyRelease | None:
        """Write the weights with ``write(directory)``, then the record (it lists the
        weight files and their sizes), then LATEST.

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
                               engine_checksum=engine_checksum or None, files=file_sizes(directory),
                               buffer_shards=None if buffer_shards is None else
                               {str(k): int(v) for k, v in sorted(buffer_shards.items())})
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
        self.incomplete: str | None = None  # why the newest release is not loaded yet

    def refresh(self) -> PolicyRelease | None:
        """The release loaded by this call, or None when there is no new one, or
        when the newest one is not complete yet (``incomplete`` says why; the
        engines keep the release they serve)."""
        latest = self.releases.latest()
        self.last_seen = latest
        if latest is None or (self.release is not None and latest.seq == self.release.seq):
            return None
        try:
            directory = self.releases.resolve(latest)
        except ReleaseIncomplete as error:
            self.incomplete = str(error)
            return None
        self.incomplete = None
        check_release_files(latest, directory)
        self._load(latest, directory)
        self.version = latest.version
        self.release = latest
        return latest
