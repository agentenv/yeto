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

Synced trainers (several islands merged by a syncer) release through one
publisher, with the merged version: :mod:`.merged_release`.

A trainer that publishes the same version and hash again (a restart) releases
nothing. A trainer restored from an older cut publishes an older version with
other weights (a new lineage): that is a new release with a higher ``seq``.

:class:`PolicyFollower` is the sampler side: it loads a release when its ``seq``
differs from the one its engines serve. When the trainer stops, no new release
appears and the sampler keeps the last version.

Latency phases (S26 E5): the trainer side (:meth:`PolicyReleaseStore.release`)
and the sampler side (:meth:`PolicyFollower.refresh`) time their phases with
:class:`PhaseTimer` (``last_timings``, seconds); the islands write them in their
release and load events. They only measure: no phase changes its order or work.
Writing with a known LATEST (``cached_latest``, default on): the store has one
writer per release directory (one trainer, or the publisher of synced trainers,
:mod:`.merged_release`; a re-quantizer writes its own directory), so the writer
remembers the release it wrote last and does not reload the store (a Modal
Volume reload takes seconds) to read LATEST again. It still reloads and reads
LATEST for its first release in a process (a resumed trainer continues the
store's ``seq``), after a failed release, and when the LATEST it sees without a
reload is not the one it wrote (another writer). The weights, the record and
LATEST then go to the store in one commit: a reader never loads a release
whose files are not all there with their recorded sizes (``check_complete``),
so LATEST becoming visible with or before the weights only delays the load.
``cached_latest=False`` is the earlier behavior (reload before every release,
one commit for the weights and record, a second one for LATEST).

A load hook that checks the weight files itself while it copies them
(``verifies_files``, :func:`copy_with_digest`) reads a LoRA release once instead
of twice; the follower then skips its own check.

Staged releases (S27 G-AT): the trainer exports a full checkpoint into a
directory on its local disk, and :meth:`PolicyReleaseStore.release_staged`
publishes that directory: the weight files in one store upload
(``CheckpointStore.put_directory``; a Modal Volume takes it in one
``batch_upload``, without the trainer's mount), then the record and LATEST in
one more (``put_files``). A reader still sees a release only once LATEST names
it and its files are complete. The upload runs
on a :class:`BackgroundReleaser` thread, so the trainer trains its next step
while the samplers' release is written; releases stay in version order (the next
one waits for the upload of the previous one).

Delta releases (S27 G-BG, ``--rl-release-delta-full-every N``): a staged full
release may be uploaded as the bf16 delta against the trainer's previous release
(:class:`DeltaBase`, its files kept on the trainer's local disk;
:mod:`.release_delta`). The record's ``delta`` names the base and the digest of
every rebuilt file; ``files`` lists the delta files in the store. A reader with a
:class:`~.release_delta.DeltaMirror` rebuilds the release on its local disk
(:meth:`PolicyReleaseStore.resolve`) and loads that copy. Pruning keeps every
base a kept release needs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from yeto.rl.quant.schemes import BF16, RELEASE_DIR, W4A8_INT4_CHANNEL, release_dir_for

from .object_store import ObjectStore, download_file, object_size
from .release_delta import DeltaMirror, ReleaseDeltaError, encode_directory
from .resume import CheckpointStore, _write_json_atomic

SCHEMA = "yeto.rl.policy-release/v1"
# Releases re-quantized from the BF16 ones (yeto.rl.quant.release) live in
# release_dir_for(scheme); a sampler follows the directory of the scheme it serves.
W4A8_RELEASE_DIR = release_dir_for(W4A8_INT4_CHANNEL)
LATEST = "LATEST.json"
RECORD = "RELEASE.json"
_RELEASE_NAME = re.compile(r"r(\d+)-v\d+")
FULL = "full"
LORA = "lora"
WEIGHT_FORMATS = (FULL, LORA)
_DIGEST_CHUNK = 1 << 20
DEFAULT_CACHE_DIR = "~/.cache/yeto/policy-releases"
# a delta larger than this share of the full release is not worth it: release in full
MAX_DELTA_RATIO = 0.5
# a sampler stops after this many delta rebuilds in a row fail their digest check
# (PolicyFollower): the first from its mirror, the next ones from the full release
# in the store, downloaded again
MAX_REBUILD_FAILURES = 3


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
    # A release made from another one (a W4A8 re-quantization of a BF16 release):
    # {"release_dir", "seq", "version", "policy_hash", "scheme", ...}; None: trained weights.
    derived_from: dict[str, Any] | None = None
    # Who released it when the trainers sync (S25 C3, yeto.rl.engine.merged_release):
    # {"kind": "merged", "learner_id", "num_learners", "local_version"}; None: one trainer.
    origin: dict[str, Any] | None = None
    # A bf16 delta release (yeto.rl.engine.release_delta): {"base_seq", "base_path",
    # "base_version", "ratio", "files": {name: {"mode", "size", "sha256"}}}; the
    # directory holds the deltas, ``files`` above lists them. None: the weights themselves.
    delta: dict[str, Any] | None = None
    schema: str = SCHEMA

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "PolicyRelease":
        if record.get("schema") != SCHEMA:
            raise PolicyReleaseError(f"release schema {record.get('schema')!r} != {SCHEMA}")
        return cls(**{k: record[k] for k in record if k in cls.__dataclass_fields__})


def release_scheme(release: PolicyRelease) -> str:
    """The weight scheme of a release: ``derived_from.scheme`` of a derived
    release, else :data:`~yeto.rl.quant.schemes.BF16` (trained weights)."""
    return str((release.derived_from or {}).get("scheme") or BF16)


class SchemeMismatch(PolicyReleaseError):
    """A release whose weight scheme is not the one the reader serves."""


def check_release_scheme(release: PolicyRelease, expected: str) -> None:
    """Raise :class:`SchemeMismatch` when ``release`` is not of scheme ``expected``."""
    actual = release_scheme(release)
    if actual != expected:
        raise SchemeMismatch(f"release r{release.seq}-v{release.version} has weight scheme {actual!r}; "
                             f"this reader serves {expected!r} (it follows the wrong release directory?)")


class PhaseTimer:
    """Wall seconds per named phase (``phase(name)`` adds to the name's total).
    ``as_dict()``: the totals in milliseconds precision, in first-use order."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self.clock = clock
        self._totals: dict[str, float] = {}

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        started = self.clock()
        try:
            yield
        finally:
            self._totals[name] = self._totals.get(name, 0.0) + (self.clock() - started)

    def add(self, name: str, seconds: float) -> None:
        self._totals[name] = self._totals.get(name, 0.0) + float(seconds)

    def as_dict(self) -> dict[str, float]:
        return {name: round(value, 3) for name, value in self._totals.items()}


def _digested_files(root: Path) -> list[Path]:
    """The files :func:`directory_digest` covers, in its order (all but the record)."""
    return sorted(p for p in root.rglob("*") if p.is_file() and p.relative_to(root) != Path(RECORD))


def _digest_header(digest: Any, root: Path, path: Path) -> None:
    relative = path.relative_to(root).as_posix().encode("utf-8")
    digest.update(len(relative).to_bytes(8, "big") + relative)
    digest.update(os.path.getsize(path).to_bytes(8, "big"))


def directory_digest(directory: Path) -> str:
    """sha256 over every file under ``directory`` except the release record:
    relative path, size and bytes, in sorted path order."""
    digest = hashlib.sha256()
    root = Path(directory)
    for path in _digested_files(root):
        _digest_header(digest, root, path)
        with path.open("rb") as handle:
            while chunk := handle.read(_DIGEST_CHUNK):
                digest.update(chunk)
    return digest.hexdigest()


def copy_with_digest(source: Path, target: Path) -> str:
    """Copy every file under ``source`` into ``target`` (created; existing files are
    overwritten) and return :func:`directory_digest` of ``source``, reading each
    file once. The release record is copied but not digested, as there."""
    root, out = Path(source), Path(target)
    out.mkdir(parents=True, exist_ok=True)
    digested = set(_digested_files(root))
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        destination = out / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        hashed = path in digested
        if hashed:
            _digest_header(digest, root, path)
        with path.open("rb") as reader, destination.open("wb") as writer:
            while chunk := reader.read(_DIGEST_CHUNK):
                if hashed:
                    digest.update(chunk)
                writer.write(chunk)
    return digest.hexdigest()


class ReleaseIncomplete(PolicyReleaseError):
    """A release whose files are not all present with their recorded sizes yet."""


class ReleaseNotLoaded(PolicyReleaseError):
    """The load hook did not take this release, and the sampler may go on: the
    release scheme is not the one the engine serves, or a post-load check
    failed (``yeto.rl.engine.sampler_checks``). The follower keeps its served
    release and tries the newest one again on the next refresh. ``reason`` is
    a short machine-readable cause; ``fields``: more event fields (hashable values)."""

    def __init__(self, message: str, *, reason: str, fields: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.reason = reason
        self.fields = dict(fields or {})


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
    if release.sha256 is not None:
        check_release_digest(release, directory_digest(directory))


def check_release_digest(release: PolicyRelease, actual: str) -> None:
    """Raise when ``actual`` (a :func:`directory_digest`) is not the released one
    (no check for a release without ``sha256``)."""
    if release.sha256 is not None and actual != release.sha256:
        raise PolicyReleaseError(f"release r{release.seq}-v{release.version}: weight files "
                                 f"sha256 {actual} != released {release.sha256}")


@dataclass(frozen=True)
class DeltaBase:
    """The previous release of this trainer and its full weight files on local
    disk: the base of the next delta release (``release_delta`` module doc)."""

    release: PolicyRelease
    directory: Path


class DeltaChain:
    """Trainer side of delta releases: which earlier release the next one is a
    delta of. Every ``full_every``-th release is full (a sampler that starts or
    falls behind rebuilds at most ``full_every - 1`` deltas); the others are
    deltas against the release before them. After each upload the staged
    directory becomes the next base (:meth:`advance`) and the older base's files
    are deleted, so the trainer's disk holds at most two staged releases."""

    def __init__(self, full_every: int) -> None:
        if int(full_every) < 2:
            raise ValueError("a delta chain needs full_every >= 2")
        self.full_every = int(full_every)
        self._base: DeltaBase | None = None
        self._since_full = 0  # deltas since the last full release

    def base(self) -> DeltaBase | None:
        """The base of the next release; None: the next release is full."""
        if self._base is None or self._since_full + 1 >= self.full_every:
            return None
        return self._base

    def advance(self, release: PolicyRelease | None, staged: Path) -> bool:
        """Record the upload of ``staged`` (``release``: what the store now holds;
        None when nothing was released). True: ``staged`` is kept as the next base."""
        if release is None:
            return False
        previous, self._base = self._base, DeltaBase(release, Path(staged))
        self._since_full = 0 if release.delta is None else self._since_full + 1
        if previous is not None and previous.directory != Path(staged):
            shutil.rmtree(previous.directory, ignore_errors=True)
        return True


class PolicyReleaseStore:
    """Writes (trainer) and reads (sampler) the releases in one store.
    ``cache_dir``: where a store with an object API downloads a release to.
    ``release_dir``: the release directory in the store
    (:func:`~yeto.rl.quant.schemes.release_dir_for` of the scheme)."""

    def __init__(self, store: CheckpointStore, *, keep: int = 2,
                 clock: Callable[[], float] = time.time,
                 cache_dir: str | os.PathLike[str] = DEFAULT_CACHE_DIR,
                 release_dir: str = RELEASE_DIR,
                 timer: Callable[[], float] = time.perf_counter,
                 cached_latest: bool = True, mirror: DeltaMirror | None = None) -> None:
        if int(keep) < 1:
            raise ValueError("keep must be >= 1")
        self.store = store
        self.mirror = mirror  # sampler side: where delta releases are rebuilt
        self._bases: dict[str, str] = {}  # delta release name -> its base (this writer)
        self.keep = int(keep)
        self.clock = clock
        self.release_dir = str(release_dir)
        self.root = Path(store.root) / self.release_dir
        self.timer = timer
        self.cached_latest = bool(cached_latest)
        self._written: PolicyRelease | None = None  # the release this process wrote last
        # release directory names in the store, oldest first (staged releases prune by it:
        # the mount of a store that bypasses it does not list them); None: not read yet
        self._names: list[str] | None = None
        # how the last release() found LATEST: "cached", "reloaded" (first release,
        # after a failure, or cached_latest off) or "other_writer"
        self.last_latest_source: str | None = None
        # seconds per phase of the last release() that wrote a release (module doc)
        self.last_timings: dict[str, float] = {}
        self.last_visible_at: float | None = None  # wall time LATEST named the last staged release
        self.cache_dir = Path(cache_dir).expanduser()
        if self.release_dir != RELEASE_DIR:  # its own cache: the prune below drops other directories
            self.cache_dir /= self.release_dir

    def resolve(self, release: PolicyRelease, *, from_store: bool = False) -> Path:
        """The release's weight directory on this island: in the store mount, or
        downloaded from the store's object API; a delta release is rebuilt in the
        mirror (``release_delta.DeltaMirror``). Raises :class:`ReleaseIncomplete`
        when a recorded file (or a base record) is missing or of another size, and
        ``ReleaseDeltaError`` when a rebuilt file is not the released one.
        ``from_store``: rebuild a delta release from the full release in the store,
        downloaded again (not from a rebuilt copy or an earlier download)."""
        if release.delta is None:
            return self._stored(release)
        if self.mirror is None:
            raise PolicyReleaseError(f"release r{release.seq}-v{release.version} is a delta "
                                     "release: this reader has no mirror to rebuild it in")
        if from_store and self.store.objects is not None:
            self._prune_cache(keep=())
        directory = self.mirror.materialize(release, record_of=self._record_at, from_store=from_store,
                                            weights_dir=lambda r: self._stored(r, prune_cache=False))
        if self.store.objects is not None:
            self._prune_cache(keep=())
        return directory

    def _stored(self, release: PolicyRelease, *, prune_cache: bool = True) -> Path:
        """The release directory as stored (for a delta release: the deltas)."""
        objects = self.store.objects
        if objects is None:
            directory = self.root / release.path
            check_complete(release, directory)
            return directory
        return self._download(objects, release, prune_cache=prune_cache)

    def _record_at(self, path: str) -> PolicyRelease:
        """The record in release directory ``path`` (a delta's base)."""
        objects = self.store.objects
        if objects is not None:
            raw = objects.read(f"{self.release_dir}/{path}/{RECORD}")
        else:
            record = self.root / path / RECORD
            raw = record.read_bytes() if record.is_file() else None
        if raw is None:
            raise ReleaseIncomplete(f"release {path}: no record (pruned, or not yet visible)")
        return PolicyRelease.from_record(json.loads(raw))

    def _download(self, objects: ObjectStore, release: PolicyRelease, *,
                  prune_cache: bool = True) -> Path:
        """Download every object of the release into ``cache_dir/<path>`` (once:
        a complete download is renamed into place) and drop older downloads
        (unless ``prune_cache`` is false: a delta chain needs its bases).
        The recorded files are checked against the object sizes first."""
        target = self.cache_dir / release.path
        if not target.is_dir():
            prefix = f"{self.release_dir}/{release.path}/"
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
        if prune_cache:
            self._prune_cache(keep=(release.path,))
        return target

    def _prune_cache(self, keep: tuple[str, ...]) -> None:
        for old in self.cache_dir.iterdir() if self.cache_dir.is_dir() else ():
            # The cache of another release directory (e.g. W4A8_RELEASE_DIR under the
            # default one) belongs to another store: never drop it here.
            if old.is_dir() and old.name not in keep and not old.name.startswith(RELEASE_DIR):
                shutil.rmtree(old, ignore_errors=True)

    def latest(self) -> PolicyRelease | None:
        objects = self.store.objects
        if objects is not None:
            raw = objects.read(f"{self.release_dir}/{LATEST}")
            return None if raw is None else PolicyRelease.from_record(json.loads(raw))
        self.store.reload()
        path = self.root / LATEST
        if not path.is_file():
            return None
        return PolicyRelease.from_record(json.loads(path.read_text(encoding="utf-8")))

    def records(self) -> list[PolicyRelease]:
        """The release records still in the store (pruning keeps the newest
        ``keep``) and LATEST, in ``seq`` order. A restarted trainer restores its
        published versions from them (S27 L11)."""
        found: dict[int, PolicyRelease] = {}
        raws: list[bytes | str] = []
        objects = self.store.objects
        if objects is not None:
            keys = [k for k in objects.list(f"{self.release_dir}/")
                    if k.endswith(f"/{RECORD}") or k == f"{self.release_dir}/{LATEST}"]
            raws = [raw for raw in (objects.read(k) for k in keys) if raw is not None]
        else:
            self.store.reload()
            if self.root.is_dir():
                paths = [p / RECORD for p in self.root.iterdir()
                         if p.is_dir() and _RELEASE_NAME.fullmatch(p.name)] + [self.root / LATEST]
                raws = [p.read_bytes() for p in paths if p.is_file()]
        for raw in raws:
            try:
                record = PolicyRelease.from_record(json.loads(raw))
            except (ValueError, TypeError, PolicyReleaseError):
                continue  # a torn or foreign record: not a release this trainer made
            found[record.seq] = record
        return [found[seq] for seq in sorted(found)]

    def release(self, version: int, policy_hash: str, write: Callable[[Path], None], *,
                buffer_selected: int | None = None, weights: str = FULL,
                engine_checksum: str | None = None,
                buffer_shards: dict[str, int] | None = None,
                derived_from: dict[str, Any] | None = None,
                origin: dict[str, Any] | None = None) -> PolicyRelease | None:
        """Write the weights with ``write(directory)``, then the record (it lists the
        weight files and their sizes), then LATEST.

        Returns None (writes nothing) when the latest release is this version and hash."""
        version = int(version)
        if not policy_hash:
            raise ValueError("policy_hash must be non-empty")
        if weights not in WEIGHT_FORMATS:
            raise ValueError(f"weights must be one of {WEIGHT_FORMATS}, not {weights!r}")
        timer = PhaseTimer(self.timer)
        with timer.phase("read_latest_s"):
            latest = self._latest_for_release()
        if latest is not None and (latest.version, latest.policy_hash) == (version, policy_hash):
            self._written = latest  # LATEST is known: the next release need not read it again
            return None
        self._written = None  # until this release is committed
        guard = self.store.hold_writer()  # no Volume reload drops the files before the commit
        try:
            record = self._write_release(timer, latest, version, policy_hash, write, weights=weights,
                                         buffer_selected=buffer_selected,
                                         engine_checksum=engine_checksum,
                                         buffer_shards=buffer_shards, derived_from=derived_from,
                                         origin=origin)
        finally:
            guard.release()
        self._written = record
        self.last_timings = timer.as_dict()
        return record

    def release_staged(self, version: int, policy_hash: str, staged: Path, *,
                       position: Callable[[], tuple[int | None, dict[str, int] | None]],
                       weights: str = FULL, engine_checksum: str | None = None,
                       origin: dict[str, Any] | None = None,
                       delta_base: DeltaBase | None = None) -> PolicyRelease | None:
        """Publish the weights already written into the local directory ``staged``
        (module doc, staged releases): the weight files go into the store first;
        then the record, in the release directory and as LATEST, in one put.
        ``position()`` is called between the two: ``(buffer_selected,
        buffer_shards)`` then include the rounds the trainer took while the weights
        uploaded, so a sampler's backlog cap lifts with this release. ``staged``
        keeps its files.

        ``delta_base``: upload the bf16 delta of ``staged`` against this earlier
        release instead of the weights (``release_delta``), unless the delta is
        above :data:`MAX_DELTA_RATIO` of the weights; the record's ``delta`` names
        the base. The base stays in the store while a kept release needs it.

        Returns None (puts nothing) when the latest release is this version and hash."""
        version = int(version)
        if not policy_hash:
            raise ValueError("policy_hash must be non-empty")
        if weights not in WEIGHT_FORMATS:
            raise ValueError(f"weights must be one of {WEIGHT_FORMATS}, not {weights!r}")
        staged = Path(staged)
        timer = PhaseTimer(self.timer)
        with timer.phase("read_latest_s"):
            latest = self._latest_for_release()
        if latest is not None and (latest.version, latest.policy_hash) == (version, policy_hash):
            self._written = latest
            return None
        self._written = None  # until this release is in the store
        if self._names is None:  # the mount was just reloaded (first release of this process)
            self._names = self._listed_names()
        seq = 1 if latest is None else latest.seq + 1
        name = f"r{seq}-v{version}"
        upload, delta = staged, None
        if delta_base is not None and weights == FULL:
            with timer.phase("delta_s"):
                upload, delta = self._stage_delta(staged, delta_base)
        try:
            with timer.phase("digest_s"):
                sha256 = directory_digest(staged) if weights == LORA else None
                files = file_sizes(upload)
            with timer.phase("upload_s"):
                self.store.put_directory(upload, f"{self.release_dir}/{name}")
        finally:
            if upload != staged:
                shutil.rmtree(upload, ignore_errors=True)
        buffer_selected, buffer_shards = position()
        record = PolicyRelease(seq=seq, version=version, policy_hash=str(policy_hash), path=name,
                               buffer_selected=None if buffer_selected is None else int(buffer_selected),
                               released_at=round(self.clock(), 3), weights=weights, sha256=sha256,
                               engine_checksum=engine_checksum or None, files=files,
                               buffer_shards=None if buffer_shards is None else
                               {str(k): int(v) for k, v in sorted(buffer_shards.items())},
                               origin=None if origin is None else dict(origin), delta=delta)
        text = json.dumps(asdict(record), sort_keys=True).encode("utf-8")
        with timer.phase("latest_s"):
            self.store.put_files({f"{self.release_dir}/{name}/{RECORD}": text,
                                  f"{self.release_dir}/{LATEST}": text})
        self._names = [n for n in self._names if n != name] + [name]
        if delta is not None:
            self._bases[name] = str(delta["base_path"])
        with timer.phase("prune_s"):
            kept = self._kept_names()
            for old in self._names:
                if old not in kept:
                    self.store.remove_tree(f"{self.release_dir}/{old}")
                    self._bases.pop(old, None)
            self._names = [n for n in self._names if n in kept]
        self._written = record
        self.last_timings = timer.as_dict()
        self.last_visible_at = round(self.clock(), 3)  # LATEST names it
        return record

    def _stage_delta(self, staged: Path, base: DeltaBase) -> tuple[Path, dict[str, Any] | None]:
        """(directory to upload, the record's ``delta`` field): the delta of
        ``staged`` against ``base`` when its files are still there and the delta
        is small enough, else ``staged``."""
        if not Path(base.directory).is_dir():
            return staged, None
        out = staged.parent / f".{staged.name}.delta"
        shutil.rmtree(out, ignore_errors=True)
        try:
            result = encode_directory(base.directory, staged, out, skip=lambda name: name == RECORD)
        except BaseException:
            shutil.rmtree(out, ignore_errors=True)
            raise
        if result.ratio > MAX_DELTA_RATIO:
            shutil.rmtree(out, ignore_errors=True)
            return staged, None
        return out, {"base_seq": base.release.seq, "base_path": base.release.path,
                     "base_version": base.release.version, "ratio": round(result.ratio, 6),
                     "files": {n: e.to_dict() for n, e in sorted(result.entries.items())}}

    def _kept_names(self) -> set[str]:
        """The newest ``keep`` releases and every base their deltas need."""
        kept = set(self._names[-self.keep:])
        pending = list(kept)
        while pending:
            base = self._bases.get(pending.pop())
            if base is not None and base not in kept:
                kept.add(base)
                pending.append(base)
        return kept

    def _listed_names(self) -> list[str]:
        """The release directories under the root, in ``seq`` order."""
        if not self.root.is_dir():
            return []
        named = sorted((int(m.group(1)), p.name) for p in self.root.iterdir()
                       if p.is_dir() and (m := _RELEASE_NAME.fullmatch(p.name)))
        return [name for _, name in named]

    def _latest_without_reload(self) -> PolicyRelease | None:
        path = self.root / LATEST
        if not path.is_file():
            return None
        return PolicyRelease.from_record(json.loads(path.read_text(encoding="utf-8")))

    def _latest_for_release(self) -> PolicyRelease | None:
        """LATEST as the writer needs it (module doc, ``cached_latest``)."""
        known = self._written
        if not self.cached_latest or known is None or self.store.objects is not None:
            self.last_latest_source = "reloaded"
            return self.latest()
        if self.store.puts_bypass_mount:  # the mount does not show what this process put
            self.last_latest_source = "cached"
            return known
        seen = self._latest_without_reload()
        if seen is None or (seen.seq, seen.path, seen.policy_hash) != (known.seq, known.path, known.policy_hash):
            self.last_latest_source = "other_writer"
            return self.latest()
        self.last_latest_source = "cached"
        return known

    def _write_release(self, timer: PhaseTimer, latest: PolicyRelease | None, version: int,
                       policy_hash: str, write: Callable[[Path], None], *, weights: str,
                       buffer_selected: int | None, engine_checksum: str | None,
                       buffer_shards: dict[str, int] | None, derived_from: dict[str, Any] | None,
                       origin: dict[str, Any] | None) -> PolicyRelease:
        seq = 1 if latest is None else latest.seq + 1
        name = f"r{seq}-v{version}"
        directory = self.root / name
        with timer.phase("export_s"):
            if directory.exists():  # a partial release of an earlier attempt (no LATEST names it)
                shutil.rmtree(directory)
            directory.mkdir(parents=True)
            write(directory)
        with timer.phase("digest_s"):
            sha256 = directory_digest(directory) if weights == LORA else None
            files = file_sizes(directory)
        record = PolicyRelease(seq=seq, version=version, policy_hash=str(policy_hash), path=name,
                               buffer_selected=None if buffer_selected is None else int(buffer_selected),
                               released_at=round(self.clock(), 3), weights=weights, sha256=sha256,
                               engine_checksum=engine_checksum or None, files=files,
                               buffer_shards=None if buffer_shards is None else
                               {str(k): int(v) for k, v in sorted(buffer_shards.items())},
                               derived_from=None if derived_from is None else dict(derived_from),
                               origin=None if origin is None else dict(origin))
        if self.cached_latest:  # one commit (module doc)
            with timer.phase("commit_s"):
                _write_json_atomic(directory / RECORD, asdict(record))
                _write_json_atomic(self.root / LATEST, asdict(record))
                self.store.commit()
        else:
            with timer.phase("commit_weights_s"):
                _write_json_atomic(directory / RECORD, asdict(record))
                self.store.commit()  # the weights and the record are durable before LATEST names them
            with timer.phase("commit_latest_s"):
                _write_json_atomic(self.root / LATEST, asdict(record))
                self.store.commit()
        with timer.phase("prune_s"):
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


class BackgroundReleaser:
    """Runs release uploads on one background thread, one at a time and in order.

    :meth:`submit` waits for the previous upload first (at most one in flight),
    so releases reach the store in submission order and the caller's local
    staging holds at most two releases (the one uploading and the next one).
    An upload's error is raised by the next :meth:`wait`, :meth:`submit` or
    :meth:`drain` on the caller's thread: a failed release stops the trainer as
    an inline one does."""

    def __init__(self, name: str = "policy-release-upload",
                 timer: Callable[[], float] = time.perf_counter) -> None:
        self.name = name
        self.timer = timer
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def wait(self) -> float:
        """Wait for the upload in flight; returns the seconds waited. Raises its error."""
        started = self.timer()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        waited = self.timer() - started
        error, self._error = self._error, None
        if error is not None:
            raise PolicyReleaseError(f"background policy release failed: {type(error).__name__}: "
                                     f"{error}") from error
        return waited

    def submit(self, job: Callable[[], None]) -> float:
        """Start ``job`` once the previous one ended; returns the seconds waited for it."""
        waited = self.wait()

        def run() -> None:
            try:
                job()
            except BaseException as error:  # noqa: BLE001 - re-raised on the caller's thread
                self._error = error

        self._thread = threading.Thread(target=run, name=self.name, daemon=True)
        self._thread.start()
        return waited

    def drain(self) -> float:
        """Wait for the last upload (the end of a run); raises its error."""
        return self.wait()


@dataclass
class FollowerCounters:
    """What a sampler's follower did with the releases it read (staleness accounting).

    ``loaded``: releases loaded into the engines; ``superseded``: releases the
    engines never served because a newer one was already committed when the
    follower read the store (seq gap); ``stale_reads``: reads of a ``LATEST.json``
    older than one already seen (a store mount that serves an old copy; nothing
    is loaded); ``lineage_resets``: loaded releases whose version is below the
    served one (a trainer restored from an older cut); ``deferred_in_flight``:
    refreshes refused because a round was generating on engines that cannot
    update their weights mid-generation; ``rebuild_failures``: delta releases
    whose rebuilt files were not the released ones (nothing is loaded)."""

    loaded: int = 0
    superseded: int = 0
    stale_reads: int = 0
    lineage_resets: int = 0
    deferred_in_flight: int = 0
    rebuild_failures: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class PolicyFollower:
    """Sampler side: load each new release into the engines.

    ``load(release, directory)`` loads the weights and stamps ``release.version``
    on every engine (raises on any failure; the served release is then unchanged).
    The weight files are checked against the record's ``sha256`` first, unless
    the hook has a true ``verifies_files`` attribute: it then checks them itself
    (with :func:`check_release_digest`) before any engine sees them.
    ``version`` is the version the engines serve before any release (the base
    policy, 0). ``scheme``: the weight scheme the engines serve; a release of
    another scheme raises :class:`SchemeMismatch` before any file is read
    (None: no check, for callers that check in ``load``).

    Order: releases are loaded in increasing ``seq`` only. A read that returns a
    release at or below the newest seq already seen loads nothing and keeps
    ``last_seen`` (the backlog reference) at the newest one. The version may go
    down with a higher seq (a new lineage); that release is loaded and counted.

    In-flight rule: while the caller marks a round as generating
    (:meth:`generating`), :meth:`refresh` loads nothing unless ``live_update``
    (the engines take new weights during a generation and stamp each sample with
    the version that produced it). A refused refresh is counted and the release
    is loaded at the next refresh outside the round.

    Rebuild failures: a delta release whose rebuilt files fail their digest
    check (``ReleaseDeltaError``) is not loaded; the engines keep the release
    they serve and :meth:`refresh` raises :class:`ReleaseNotLoaded` (reason
    ``delta_rebuild_failed``). Each later try rebuilds from the full release in
    the store (``resolve(from_store=True)``) until one release is resolved.
    The ``max_rebuild_failures``-th failure in a row raises the
    ``ReleaseDeltaError`` (the island stops).

    ``last_timings``: seconds per phase of the last refresh that loaded a
    release; ``seen_at``: the wall time this follower first read the newest
    release (``last_seen``) in ``LATEST``."""

    def __init__(self, releases: PolicyReleaseStore,
                 load: Callable[[PolicyRelease, Path], None], *, version: int = 0,
                 scheme: str | None = None, live_update: bool = False,
                 timer: Callable[[], float] = time.perf_counter,
                 wall: Callable[[], float] = time.time,
                 max_rebuild_failures: int = MAX_REBUILD_FAILURES) -> None:
        if int(max_rebuild_failures) < 1:
            raise ValueError("max_rebuild_failures must be >= 1")
        self.max_rebuild_failures = int(max_rebuild_failures)
        self.rebuild_failures = 0  # in a row, since a release was last resolved
        self.releases = releases
        self.scheme = scheme
        self._load = load
        self.version = int(version)
        self.live_update = bool(live_update)
        self.release: PolicyRelease | None = None  # the release the engines serve
        self.last_seen: PolicyRelease | None = None  # the newest release read
        self.incomplete: str | None = None  # why the newest release is not loaded yet
        self.counters = FollowerCounters()
        self._in_flight = False
        self.timer = timer
        self.wall = wall
        self.seen_at: float | None = None
        self.last_timings: dict[str, float] = {}

    @contextmanager
    def generating(self) -> Iterator[None]:
        """Mark a round as generating on the engines (the in-flight rule, class doc)."""
        if self._in_flight:
            raise RuntimeError("a round is already generating")
        self._in_flight = True
        try:
            yield
        finally:
            self._in_flight = False

    def version_lag(self) -> int:
        """Trainer versions between the newest release seen and the served version
        (0 before any release, and when a new lineage is below the served one)."""
        if self.last_seen is None:
            return 0
        return max(0, int(self.last_seen.version) - int(self.version))

    def refresh(self) -> PolicyRelease | None:
        """The release loaded by this call, or None when there is no new one, when
        the newest one is not complete yet (``incomplete`` says why; the engines
        keep the release they serve), or when the in-flight rule defers it."""
        timer = PhaseTimer(self.timer)
        with timer.phase("read_latest_s"):
            latest = self.releases.latest()
        if latest is None:
            return None
        if self.last_seen is not None and latest.seq < self.last_seen.seq:
            self.counters.stale_reads += 1
            latest = self.last_seen
        if self.last_seen is None or latest.seq != self.last_seen.seq:
            self.seen_at = round(self.wall(), 3)
        self.last_seen = latest
        if self.release is not None and latest.seq <= self.release.seq:
            return None
        if self._in_flight and not self.live_update:
            self.counters.deferred_in_flight += 1
            return None
        if self.scheme is not None:
            check_release_scheme(latest, self.scheme)
        try:
            with timer.phase("resolve_s"):
                directory = self.releases.resolve(latest, from_store=self.rebuild_failures > 0)
        except ReleaseIncomplete as error:
            self.incomplete = str(error)
            return None
        except ReleaseDeltaError as error:
            self._rebuild_failed(latest, error)
        self.rebuild_failures = 0
        self.incomplete = None
        if not getattr(self._load, "verifies_files", False):
            with timer.phase("verify_s"):
                check_release_files(latest, directory)
        with timer.phase("load_s"):
            self._load(latest, directory)
        previous = self.release
        self.counters.loaded += 1
        if previous is not None:
            self.counters.superseded += latest.seq - previous.seq - 1
            if latest.version < self.version:
                self.counters.lineage_resets += 1
        self.version = latest.version
        self.release = latest
        self.last_timings = timer.as_dict()
        return latest

    def _rebuild_failed(self, release: PolicyRelease, error: ReleaseDeltaError) -> None:
        """Count a failed rebuild of ``release`` and raise (class doc)."""
        self.rebuild_failures += 1
        self.counters.rebuild_failures += 1
        failures = self.rebuild_failures
        if failures >= self.max_rebuild_failures:
            raise ReleaseDeltaError(f"r{release.seq}-v{release.version}: {failures} delta rebuilds "
                                    f"in a row failed (last: {error})", file=error.file) from error
        raise ReleaseNotLoaded(
            f"r{release.seq}-v{release.version}: delta rebuild failed ({error}); the engines keep "
            f"v{self.version}, the next try rebuilds from the full release in the store",
            reason="delta_rebuild_failed",
            fields={"release_version": release.version, "file": error.file,
                    "rebuild_failures": failures, "max_rebuild_failures": self.max_rebuild_failures},
        ) from error
