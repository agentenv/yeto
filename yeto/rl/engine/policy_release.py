"""Policy releases: trainer -> sampler weight hand-off through the store.

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

Complete releases only (rl-cross-site-sample-store): a reader loads a release
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

Latency phases: the trainer side (:meth:`PolicyReleaseStore.release`)
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

Staged releases: the trainer exports a full checkpoint into a
directory on its local disk, and :meth:`PolicyReleaseStore.release_staged`
publishes that directory: the weight files in one store put
(``CheckpointStore.put_directory``: through the trainer's mount and one commit;
a Modal Volume store that is not mounted, or that is asked to, takes it in one
``batch_upload`` instead, ``ModalVolumeStore.put_path``), then the record and
LATEST in one more (``put_files``). A reader still sees a release only once LATEST names
it and its files are complete. The upload runs
on a :class:`BackgroundReleaser` thread, so the trainer trains its next step
while the samplers' release is written; releases stay in version order (the next
one waits for the upload of the previous one).

Delta releases (``--rl-release-delta-full-every N``): a staged full
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
import random
import re
import shutil
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from yeto.rl.quant.schemes import BF16, RELEASE_DIR, W4A8_INT4_CHANNEL, release_dir_for

from .object_store import DOWNLOAD_SUFFIX, ObjectStore, download_file, is_fatal_store_error, object_size
from .release_delta import DeltaMirror, ReleaseDeltaError, encode_directory, file_sha256
from .resume import CheckpointStore, _write_json_atomic, commit_paths

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
# the yeto cache root that replaces ``~/.cache/yeto`` in the default (e.g. a process
# whose HOME points elsewhere shares the node's cache); unset: the default as is
CACHE_DIR_ENV = "YETO_CACHE_DIR"


def default_cache_dir(environ: Mapping[str, str] | None = None) -> Path:
    """:data:`DEFAULT_CACHE_DIR`, or ``$YETO_CACHE_DIR/policy-releases`` when that is set."""
    root = (os.environ if environ is None else environ).get(CACHE_DIR_ENV, "").strip()
    return Path(root).expanduser() / "policy-releases" if root else Path(DEFAULT_CACHE_DIR).expanduser()
# a delta larger than this share of the full release is not worth it: release in full
MAX_DELTA_RATIO = 0.5
# a sampler stops after this many releases in a row fail their digest checks
# (PolicyFollower: a delta rebuilt from its mirror, then rebuilt again from the full
# release in the store, downloaded again; a full release's file_sha256)
MAX_REBUILD_FAILURES = 3
DIGEST_WORKERS = 8  # files hashed at a time (file_digests)


class PolicyReleaseError(RuntimeError):
    pass


@dataclass(frozen=True)
class PolicyRelease:
    seq: int
    version: int
    policy_hash: str
    path: str  # relative to the release root: "r<seq>-v<version>"
    buffer_selected: int | None = None
    # The buffer position these weights are trained up to: every group before it was
    # selected by a round the weights include (or passed over), none at or after it.
    # Read on the trainer's thread at the publication, not when the record is written:
    # ``buffer_selected`` of a staged release is the read position when its upload
    # ended, rounds later. A relaunch without a round cut reads the buffer on from
    # here. None: not recorded (an earlier yeto, or nothing was read yet).
    buffer_trained: int | None = None
    released_at: float = 0.0
    weights: str = FULL
    sha256: str | None = None  # None: not digested (a full checkpoint)
    engine_checksum: str | None = None
    files: dict[str, int] | None = None  # weight file (relative path) -> size in bytes
    buffer_shards: dict[str, int] | None = None  # sampler shard -> trainer read position
    # A release made from another one (a W4A8 re-quantization of a BF16 release):
    # {"release_dir", "seq", "version", "policy_hash", "scheme", ...}; None: trained weights.
    derived_from: dict[str, Any] | None = None
    # Who released it when the trainers sync (yeto.rl.engine.merged_release):
    # {"kind": "merged", "learner_id", "num_learners", "local_version"}; None: one trainer.
    origin: dict[str, Any] | None = None
    # A bf16 delta release (yeto.rl.engine.release_delta): {"base_seq", "base_path",
    # "base_version", "ratio", "files": {name: {"mode", "size", "sha256"}}}; the
    # directory holds the deltas, ``files`` above lists them. None: the weights themselves.
    delta: dict[str, Any] | None = None
    # sha256 of each weight file of a full release (``file_digests``: the full
    # releases of a delta chain); None: not digested file by file
    file_sha256: dict[str, str] | None = None
    # What the trainer declares about the weights beyond their bytes, e.g. the
    # partial-experts selection a Kimi-K3.1 release was trained with
    # (``{"expert_selection_sha256": ..., "partial_experts": {...}}``, INTERFACES 2.3);
    # a sampler bound to a selection refuses a release of another one. None: nothing declared.
    manifest: dict[str, Any] | None = None
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


def _parsed_records(raws: Any) -> list[PolicyRelease]:
    found: dict[int, PolicyRelease] = {}
    for raw in raws:
        try:
            record = PolicyRelease.from_record(json.loads(raw))
        except (ValueError, TypeError, PolicyReleaseError):
            continue  # a torn or foreign record: not a release this trainer made
        found[record.seq] = record
    return [found[seq] for seq in sorted(found)]


def records_in_objects(objects: Any, release_dir: str = RELEASE_DIR) -> list[PolicyRelease]:
    """The release records (every ``RELEASE.json`` and ``LATEST``) in an object store, in
    ``seq`` order (:meth:`PolicyReleaseStore.records`; also the launcher's view)."""
    keys = [k for k in objects.list(f"{release_dir}/")
            if k.endswith(f"/{RECORD}") or k == f"{release_dir}/{LATEST}"]
    return _parsed_records(raw for raw in (objects.read(k) for k in keys) if raw is not None)


def trained_releases(records: Any) -> list[PolicyRelease]:
    """The policy versions a trainer published (not derived, with a policy hash): what
    ``merged_release.resumed_releases`` continues the numbering at and what
    ``--rl-init-from-release latest`` selects from."""
    return [r for r in records if r.derived_from is None and r.policy_hash]


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


class ReleaseDigestMismatch(ReleaseDeltaError):
    """A full release whose weight files are not the released ones (``file_sha256``);
    the follower treats it like a failed delta rebuild (``PolicyFollower``)."""


LATEST_UNREAD = "latest_unread"  # ReleaseNotLoaded.reason: the poll of LATEST failed (no release judged)
# LATEST was read, but it is no release record
_DAMAGED_RECORD = (ValueError, TypeError, AttributeError, KeyError, PolicyReleaseError)


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


def file_digests(directory: Path, names: Iterable[str] | None = None, *,
                 workers: int = DIGEST_WORKERS) -> dict[str, str]:
    """{relative path: sha256} of ``names`` (default: every file but the record)
    under ``directory``, ``workers`` files at a time."""
    root = Path(directory)
    names = sorted(file_sizes(root)) if names is None else sorted(names)
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        return dict(zip(names, pool.map(lambda name: file_sha256(root / name), names)))


def check_release_files(release: PolicyRelease, directory: Path) -> None:
    """Raise when the weight files differ from the ones the trainer released:
    the release digest (``sha256``) or each file's (``file_sha256``)."""
    if release.sha256 is not None:
        check_release_digest(release, directory_digest(directory))
    if release.file_sha256:
        actual = file_digests(directory, release.file_sha256)
        bad = sorted(name for name, digest in release.file_sha256.items() if actual[name] != digest)
        if bad:
            raise ReleaseDigestMismatch(f"release r{release.seq}-v{release.version}: {len(bad)} of "
                                        f"{len(actual)} weight files differ from the released sha256 "
                                        f"(first: {bad[0]})", file=bad[0])


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
        self._adopted: Path | None = None  # an adopted base: not this chain's to delete
        self._since_full = 0  # deltas since the last full release

    def base(self) -> DeltaBase | None:
        """The base of the next release; None: the next release is full."""
        if self._base is None or self._since_full + 1 >= self.full_every:
            return None
        return self._base

    def adopt(self, release: PolicyRelease, directory: str | os.PathLike[str]) -> None:
        """Start the chain from a release that is already in the store and whose
        files are readable at ``directory``: the trainer's first
        release of the process is then a delta against it instead of a full
        upload. The usual case is the base policy (version 0) that both sides
        hold -- the sampler's own copy of the base checkpoint -- so the first
        cross-cloud update sends only the words the first optimizer step moved.
        ``advance`` deletes a base this chain staged itself, never an adopted
        directory."""
        if self._base is not None:
            raise ValueError("the delta chain already has a base")
        self._base = DeltaBase(release, Path(directory))
        self._adopted = Path(directory)
        self._since_full = 0

    def advance(self, release: PolicyRelease | None, staged: Path) -> bool:
        """Record the upload of ``staged`` (``release``: what the store now holds;
        None when nothing was released). True: ``staged`` is kept as the next base."""
        if release is None:
            return False
        previous, self._base = self._base, DeltaBase(release, Path(staged))
        self._since_full = 0 if release.delta is None else self._since_full + 1
        if (previous is not None and previous.directory != Path(staged)
                and previous.directory != self._adopted):
            shutil.rmtree(previous.directory, ignore_errors=True)
        return True


def store_cache_key(store: Any) -> str:
    """The sub-directory of :data:`DEFAULT_CACHE_DIR` a store's downloads live in:
    a short digest of the store's identity (its URI for an object store, else its
    resolved root). Release names (``r<seq>-v<version>``) repeat from run to run, so a
    node that serves several runs must never reuse another run's download."""
    identity = getattr(store, "uri", None) or str(Path(getattr(store, "root", ".")).expanduser().resolve())
    return hashlib.sha256(str(identity).encode("utf-8")).hexdigest()[:16]


class PolicyReleaseStore:
    """Writes (trainer) and reads (sampler) the releases in one store.
    ``cache_dir``: where a store with an object API downloads a release to; the
    default is :func:`default_cache_dir` (``$YETO_CACHE_DIR``) keyed by the store (:func:`store_cache_key`).
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
        # the release the engines serve (``mark_served``): a download never drops it,
        # its files are what a restart or a whole reload reads
        self.serving: str | None = None
        self.resumed_files = 0  # files the last download found whole from an earlier, failed one
        self.cache_dir = Path(cache_dir).expanduser()
        if str(cache_dir) == DEFAULT_CACHE_DIR:  # the shared default: one sub-directory per store
            self.cache_dir = default_cache_dir() / store_cache_key(store)
        if self.release_dir != RELEASE_DIR:  # its own cache: the prune below drops other directories
            self.cache_dir /= self.release_dir

    def resolve(self, release: PolicyRelease, *, from_store: bool = False) -> Path:
        """The release's weight directory on this island: in the store mount, or
        downloaded from the store's object API; a delta release is rebuilt in the
        mirror (``release_delta.DeltaMirror``). Raises :class:`ReleaseIncomplete`
        when a recorded file (or a base record) is missing or of another size, and
        ``ReleaseDeltaError`` when a rebuilt file is not the released one.
        ``from_store``: download again (not from an earlier download), and rebuild a
        delta release from the full release in the store (not from a rebuilt copy)."""
        if from_store and self.store.objects is not None:
            self._prune_cache(keep=())
        if release.delta is None:
            return self._stored(release)
        if self.mirror is None:
            raise PolicyReleaseError(f"release r{release.seq}-v{release.version} is a delta "
                                     "release: this reader has no mirror to rebuild it in")
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
        a complete download is renamed into place) after dropping older downloads
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
            if prune_cache:  # free older downloads first (two 207 GB Kimi-K3.1 releases fill a
                self._prune_cache(keep=(staging.name,))  # sampler's disk); never the served one
            # a download that failed part way is resumed: a file at its final name was
            # renamed there whole (object_store.DOWNLOAD_SUFFIX), and with the record's
            # sizes it is kept only at its recorded size; a 207 GB release whose last
            # file failed is not fetched again from the start (16 samplers would pull
            # 16 x 207 GB again from the store)
            keys = self._still_needed(release, prefix, keys, staging)

            def fetch(key: str) -> None:
                path = staging / key[len(prefix):]
                path.parent.mkdir(parents=True, exist_ok=True)
                part = Path(f"{path}{DOWNLOAD_SUFFIX}")
                if not download_file(objects, key, part):
                    raise ReleaseIncomplete(f"release object {key} disappeared during the download")
                part.replace(path)

            # several files at once: one file tops out near 100 MB/s on R2
            # (TransferSettings), a 207 GB Kimi-K3.1 release took 28 minutes file by file;
            # an object API with read_files (s3_store) also spreads them over processes
            transfer = getattr(objects, "transfer", None)
            workers = max(1, min(int(getattr(transfer, "file_concurrency", 1) or 1), len(keys)))
            read_files = getattr(objects, "read_files", None)
            if read_files is not None:
                items = []
                for key in keys:
                    path = staging / key[len(prefix):]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    items.append((key, path))
                missing = read_files(items)
                if missing:
                    raise ReleaseIncomplete(f"release object {missing[0]} disappeared during the download")
            elif workers <= 1:
                for key in keys:
                    fetch(key)
            else:
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(workers, thread_name_prefix="yeto-release-get") as pool:
                    for future in [pool.submit(fetch, key) for key in keys]:
                        future.result()
            check_complete(release, staging)
            staging.replace(target)
        if prune_cache:
            self._prune_cache(keep=(release.path,))
        return target

    def _still_needed(self, release: PolicyRelease, prefix: str, keys: list[str], staging: Path) -> list[str]:
        """The keys of ``release`` whose file is not in ``staging`` already (an earlier,
        failed download); leftovers of interrupted files are removed."""
        if not staging.is_dir():
            return keys
        for leftover in staging.rglob(f"*{DOWNLOAD_SUFFIX}"):
            leftover.unlink(missing_ok=True)
        sizes = release.files or {}
        need = []
        for key in keys:
            path = staging / key[len(prefix):]
            name = key[len(prefix):]
            if path.is_file() and (name not in sizes or path.stat().st_size == sizes[name]) and name != RECORD:
                continue
            need.append(key)
        self.resumed_files = len(keys) - len(need)
        return need

    def mark_served(self, release: PolicyRelease) -> None:
        """The engines serve ``release`` now (a load succeeded): drop the other
        downloads and the overlays of releases no longer served."""
        self.serving = release.path
        if self.store.objects is None or not self.cache_dir.is_dir():
            return
        self._prune_cache(keep=(release.path,))
        for old in self.cache_dir.iterdir():
            if (old.is_dir() and old.name.endswith(".overlay")
                    and old.name[:-len(".overlay")] != release.path):
                shutil.rmtree(old, ignore_errors=True)

    def _prune_cache(self, keep: tuple[str, ...]) -> None:
        """Drop the cached downloads except ``keep`` and the served release
        (``serving``): until a new release is loaded, a restart or a whole reload
        of the engines reads the served one's files."""
        keep = (*keep, self.serving) if self.serving else keep
        for old in self.cache_dir.iterdir() if self.cache_dir.is_dir() else ():
            # The cache of another release directory (e.g. W4A8_RELEASE_DIR under the
            # default one) belongs to another store: never drop it here.
            # An overlay (``release_overlay``: symlinks and an index) stays: the engine
            # compares the next release with the overlay it serves.
            if (old.is_dir() and old.name not in keep and not old.name.startswith(RELEASE_DIR)
                    and not old.name.endswith(".overlay")):
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
        published versions from them."""
        objects = self.store.objects
        if objects is not None:
            return records_in_objects(objects, self.release_dir)
        raws: list[bytes | str] = []
        self.store.reload()
        if self.root.is_dir():
            paths = [p / RECORD for p in self.root.iterdir()
                     if p.is_dir() and _RELEASE_NAME.fullmatch(p.name)] + [self.root / LATEST]
            raws = [p.read_bytes() for p in paths if p.is_file()]
        return _parsed_records(raws)

    def release(self, version: int, policy_hash: str, write: Callable[[Path], None], *,
                buffer_selected: int | None = None, buffer_trained: int | None = None,
                weights: str = FULL,
                engine_checksum: str | None = None,
                buffer_shards: dict[str, int] | None = None,
                derived_from: dict[str, Any] | None = None,
                origin: dict[str, Any] | None = None,
                manifest: dict[str, Any] | None = None) -> PolicyRelease | None:
        """Write the weights with ``write(directory)``, then the record (it lists the
        weight files and their sizes), then LATEST. ``manifest``: the record's
        ``manifest`` (class doc).

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
                                         buffer_trained=buffer_trained,
                                         engine_checksum=engine_checksum,
                                         buffer_shards=buffer_shards, derived_from=derived_from,
                                         origin=origin, manifest=manifest)
        finally:
            guard.release()
        self._written = record
        self.last_timings = timer.as_dict()
        return record

    def release_staged(self, version: int, policy_hash: str, staged: Path, *,
                       position: Callable[[], tuple[int | None, dict[str, int] | None]],
                       buffer_trained: int | None = None,
                       weights: str = FULL, engine_checksum: str | None = None,
                       origin: dict[str, Any] | None = None,
                       delta_base: DeltaBase | None = None,
                       digest_full: bool = False,
                       manifest: dict[str, Any] | None = None) -> PolicyRelease | None:
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

        ``digest_full``: a full release records the sha256 of each weight file
        (``file_sha256``), which the follower checks before it loads the release.

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
        # the mount was just reloaded (first release of this process, or after a failed one)
        if self._names is None:
            names = self._listed_names()
            # LATEST again after the listing (one read per process): a release another
            # writer published between the first read and the listing is in the listing
            # but above that read's seq; taken for a partial upload, it would be removed
            # while LATEST names it
            with timer.phase("read_latest_s"):
                latest = self._latest_for_release()
            if latest is not None and (latest.version, latest.policy_hash) == (version, policy_hash):
                self._written = latest
                return None
        else:
            names = self._names
        self._names = None  # until this release is in the store: a failed one lists again
        seq = 1 if latest is None else latest.seq + 1
        name = f"r{seq}-v{version}"
        if name in names:  # a partial upload of a killed or failed attempt (no LATEST names it):
            # its files are not all replaced by this upload (a staged store's push deletes
            # only what its own stage uploaded), and a reader downloads every object under it
            self.store.remove_tree(f"{self.release_dir}/{name}")
        upload, delta = staged, None
        if delta_base is not None and weights == FULL:
            with timer.phase("delta_s"):
                upload, delta = self._stage_delta(staged, delta_base)
        try:
            with timer.phase("digest_s"):
                sha256 = directory_digest(staged) if weights == LORA else None
                files = file_sizes(upload)
                digests = file_digests(staged, files) if digest_full and weights == FULL and delta is None else None
            with timer.phase("upload_s"):
                self.store.put_directory(upload, f"{self.release_dir}/{name}")
                self.store.evict_local(f"{self.release_dir}/{name}")
        finally:
            if upload != staged:
                shutil.rmtree(upload, ignore_errors=True)
        buffer_selected, buffer_shards = position()
        record = PolicyRelease(seq=seq, version=version, policy_hash=str(policy_hash), path=name,
                               buffer_selected=None if buffer_selected is None else int(buffer_selected),
                               buffer_trained=None if buffer_trained is None else int(buffer_trained),
                               released_at=round(self.clock(), 3), weights=weights, sha256=sha256,
                               engine_checksum=engine_checksum or None, files=files,
                               buffer_shards=None if buffer_shards is None else
                               {str(k): int(v) for k, v in sorted(buffer_shards.items())},
                               origin=None if origin is None else dict(origin), delta=delta,
                               file_sha256=digests, manifest=None if manifest is None else dict(manifest))
        text = json.dumps(asdict(record), sort_keys=True).encode("utf-8")
        with timer.phase("latest_s"):
            self.store.put_files({f"{self.release_dir}/{name}/{RECORD}": text,
                                  f"{self.release_dir}/{LATEST}": text})
        # a directory at this seq or above is a partial upload no LATEST named (the
        # store's seq continues from LATEST): oldest, so it never takes a kept place
        def orphan(n: str) -> bool:
            return n != name and int(_RELEASE_NAME.fullmatch(n).group(1)) >= seq
        names = [n for n in names if orphan(n)] + [n for n in names if n != name and not orphan(n)] + [name]
        if delta is not None:
            self._bases[name] = str(delta["base_path"])
        with timer.phase("prune_s"):
            kept = self._kept_names(names)
            for old in names:
                if old not in kept:
                    self.store.remove_tree(f"{self.release_dir}/{old}")
                    self._bases.pop(old, None)
            self._names = [n for n in names if n in kept]
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

    def _kept_names(self, names: list[str]) -> set[str]:
        """The newest ``keep`` releases and every base their deltas need.

        A base is read from this writer's own deltas (``_bases``) and, for a
        release this process did not write, from the release's record in the
        store: a restarted trainer starts with an empty ``_bases``,
        and without the record it deleted the base its kept delta needs -- after
        which no sampler could rebuild that delta (measured on npu1: a restarted
        island pruned ``r1-v0`` while keeping the delta ``r2-v1``). An unreadable
        record keeps the release rather than dropping a base that may be needed.
        """
        kept = set(names[-self.keep:])
        pending = list(kept)
        while pending:
            name = pending.pop()
            base = self._bases.get(name)
            if base is None:
                base = self._recorded_base(name)
            if base is not None and base not in kept:
                kept.add(base)
                pending.append(base)
        return kept

    def _recorded_base(self, name: str) -> str | None:
        """The base a stored release's record names (None: a full release, or no
        readable record). Remembered, so a prune reads each record once."""
        if name in self._bases:
            return self._bases[name]
        try:
            record = self._record_at(f"{self.release_dir}/{name}")
        except Exception:  # noqa: BLE001 -- unreadable: treat as "may have a base"
            return None
        base = None if record.delta is None else str(record.delta.get("base_path") or "") or None
        if base is not None:
            self._bases[name] = base
        return base

    def _listed_names(self) -> list[str]:
        """The release directories under the root and, for a store with an object
        API, in the store, in ``seq`` order. A staged store's root does not hold
        the releases of an earlier process (each upload is evicted from the stage,
        and a restarted trainer may run on another node): listed from the stage
        alone, a restarted trainer never pruned them, nor knew a partial upload
        of a killed one under the name it writes next."""
        found = set()
        if self.root.is_dir():
            found |= {p.name for p in self.root.iterdir() if p.is_dir() and _RELEASE_NAME.fullmatch(p.name)}
        objects = self.store.objects
        if objects is not None:
            prefix = f"{self.release_dir}/"
            for key in objects.list(prefix):
                head = key[len(prefix):].split("/", 1)
                if len(head) == 2 and _RELEASE_NAME.fullmatch(head[0]):
                    found.add(head[0])
        named = sorted((int(m.group(1)), name) for name in found if (m := _RELEASE_NAME.fullmatch(name)))
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
                       origin: dict[str, Any] | None,
                       manifest: dict[str, Any] | None = None,
                       buffer_trained: int | None = None) -> PolicyRelease:
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
                               buffer_trained=None if buffer_trained is None else int(buffer_trained),
                               released_at=round(self.clock(), 3), weights=weights, sha256=sha256,
                               engine_checksum=engine_checksum or None, files=files,
                               buffer_shards=None if buffer_shards is None else
                               {str(k): int(v) for k, v in sorted(buffer_shards.items())},
                               derived_from=None if derived_from is None else dict(derived_from),
                               origin=None if origin is None else dict(origin),
                               manifest=None if manifest is None else dict(manifest))
        if self.cached_latest:  # one commit (module doc)
            with timer.phase("commit_s"):
                _write_json_atomic(directory / RECORD, asdict(record))
                _write_json_atomic(self.root / LATEST, asdict(record))
                commit_paths(self.store, [f"{self.release_dir}/{name}", f"{self.release_dir}/{LATEST}"])
        else:
            with timer.phase("commit_weights_s"):
                _write_json_atomic(directory / RECORD, asdict(record))
                # the weights and the record are durable before LATEST names them
                commit_paths(self.store, [f"{self.release_dir}/{name}"])
            with timer.phase("commit_latest_s"):
                _write_json_atomic(self.root / LATEST, asdict(record))
                commit_paths(self.store, [f"{self.release_dir}/{LATEST}"])
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
            commit_paths(self.store, [f"{self.release_dir}/{name}" for name in removed])
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
    update their weights mid-generation; ``rebuild_failures``: tries whose
    rebuilt (delta) or downloaded (full, ``file_sha256``) files were not the
    released ones (nothing is loaded); ``forced_loads``: loads made while
    ``hold()`` was True because the newest release had been deferred for
    ``max_defer_s`` (the in-flight rule's bound, :class:`PolicyFollower`)."""

    loaded: int = 0
    superseded: int = 0
    stale_reads: int = 0
    lineage_resets: int = 0
    deferred_in_flight: int = 0
    rebuild_failures: int = 0
    forced_loads: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


DOWNLOAD_RETRY_S = 60.0  # a failed background download is tried again after this
# A background download starts after a delay of 0..DOWNLOAD_SPREAD_S: samplers that see
# the same LATEST together do not open their range reads on the store in the same second.
# The delay is fixed per sampler (:func:`download_delay_s`), so two samplers keep their
# distance at every release. It postpones the release on that sampler by at most the
# spread: keep the spread well below the trainer's release interval, or a sampler stays
# one more version behind (the number of versions behind stays bounded either way: the
# delay is the same constant for every release, and only the newest release is fetched).
DOWNLOAD_SPREAD_S = 5.0
DOWNLOAD_SPREAD_ENV = "YETO_RL_RELEASE_SPREAD_S"  # the spread in seconds (0: no delay)
DOWNLOAD_SPREAD_KEY_ENV = "YETO_RL_RELEASE_SPREAD_KEY"  # what tells this sampler apart (default: the host name)
DOWNLOAD_WAIT_S = 5.0  # refresh waits this long for a download it starts before going on


def download_delay_s(environ: Mapping[str, str] | None = None) -> float:
    """This sampler's delay before a background download: a fixed point of
    ``0..spread`` taken from its key (:data:`DOWNLOAD_SPREAD_KEY_ENV`, else the
    host name), ``spread`` from :data:`DOWNLOAD_SPREAD_ENV`, else
    :data:`DOWNLOAD_SPREAD_S`."""
    import socket  # noqa: PLC0415

    env = os.environ if environ is None else environ
    raw = env.get(DOWNLOAD_SPREAD_ENV)
    spread = max(0.0, float(raw)) if raw not in (None, "") else float(DOWNLOAD_SPREAD_S)
    if spread <= 0:
        return 0.0
    key = env.get(DOWNLOAD_SPREAD_KEY_ENV) or socket.gethostname()
    point = int.from_bytes(hashlib.sha256(str(key).encode("utf-8", "replace")).digest()[:8], "big") / 2 ** 64
    return spread * point


def _download_failed(release: PolicyRelease, error: BaseException, served: int) -> ReleaseNotLoaded:
    return ReleaseNotLoaded(
        f"r{release.seq}-v{release.version}: download failed ({type(error).__name__}: {error}); "
        f"the engines keep v{served} and the download is tried again",
        reason="download_failed",
        fields={"release_version": release.version, "error": f"{type(error).__name__}: {str(error)[:200]}"})


class _Fetch:
    """One background download of a release (``PolicyFollower(background_download=True)``)."""

    def __init__(self, release: PolicyRelease) -> None:
        self.release = release
        self.done = threading.Event()
        self.directory: Path | None = None
        self.error: BaseException | None = None
        self.seconds = 0.0
        self.reported = False
        self.retry_at: float | None = None

    def start(self, releases: "PolicyReleaseStore", *, from_store: bool, timer: Callable[[], float]) -> None:
        def run() -> None:
            time.sleep(download_delay_s())
            began = timer()
            try:
                self.directory = releases.resolve(self.release, from_store=from_store)
            except BaseException as error:  # noqa: BLE001 - reported by PolicyFollower.refresh
                self.error = error
            finally:
                self.seconds = round(timer() - began, 3)
                self.done.set()

        threading.Thread(target=run, name=f"release-download-r{self.release.seq}", daemon=True).start()


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
    is loaded at the next refresh outside the round. ``hold()`` extends the rule
    past the round (a streaming agentic round keeps trials running between
    the loop's rounds; while it returns True and ``live_update`` is off, a
    refresh loads nothing either).

    Deferral bound (``max_defer_s``): a streaming agentic round may always hold
    a group whose trials run for tens of minutes, so ``hold()`` alone can defer
    a release forever. Once a ready release has been deferred by ``hold()`` for
    ``max_defer_s`` seconds (``timer``), :meth:`refresh` loads it anyway
    (``last_forced`` True, ``last_deferred_s`` the wait; ``counters.forced_loads``).
    The trials still running are then served by two versions; the round keeps
    the one-version rule of its groups (``drain``: such a trial is lost as
    ``served_by_other_version``, a turn a reload fell inside is ABORTED by the
    records proxy). A round marked by :meth:`generating` is never forced.
    None: no bound.

    Rebuild failures: a delta release whose rebuilt files fail their digest
    check (``ReleaseDeltaError``), or a full release whose files fail their
    ``file_sha256`` (:class:`ReleaseDigestMismatch`), is not loaded; the engines
    keep the release they serve and :meth:`refresh` raises :class:`ReleaseNotLoaded`
    (reason ``delta_rebuild_failed`` or ``release_digest_mismatch``). Each later
    try downloads again and rebuilds from the full release in the store
    (``resolve(from_store=True)``) until one release passes its checks.
    The ``max_rebuild_failures``-th failure in a row raises the error (the
    island stops).

    Background download (``background_download``): :meth:`refresh` starts a
    thread that fetches the newest release and returns None until its files are
    on disk, so the engines go on generating on the served weights (also during
    a round: the download is not an engine load). One download runs at a time;
    a newer release is fetched when it ends. A failed download (any error but
    :class:`ReleaseIncomplete`) raises :class:`ReleaseNotLoaded` (reason
    ``download_failed``) once and is tried again ``download_retry_s`` later.
    :meth:`prefetch` starts and polls the same download outside a refresh; the
    release it reports on disk is the one the next refresh loads.
    Without it, a failed download raises the same :class:`ReleaseNotLoaded`
    on the refresh that tried it.

    ``last_timings``: seconds per phase of the last refresh that loaded a
    release; ``seen_at``: the wall time this follower first read the newest
    release (``last_seen``) in ``LATEST``."""

    def __init__(self, releases: PolicyReleaseStore,
                 load: Callable[[PolicyRelease, Path], None], *, version: int = 0,
                 scheme: str | None = None, live_update: bool = False,
                 timer: Callable[[], float] = time.perf_counter,
                 wall: Callable[[], float] = time.time,
                 max_rebuild_failures: int = MAX_REBUILD_FAILURES,
                 background_download: bool = False,
                 download_retry_s: float = DOWNLOAD_RETRY_S,
                 max_defer_s: float | None = None) -> None:
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
        self.hold: Callable[[], bool] = lambda: False  # generation in flight outside a round
        self.timer = timer
        self.wall = wall
        self.seen_at: float | None = None
        self.last_timings: dict[str, float] = {}
        # Background download (class doc): the files of the newest release are fetched
        # by a thread while the engines keep generating on the served weights.
        self.background_download = bool(background_download)
        self.download_retry_s = float(download_retry_s)
        self._fetch: _Fetch | None = None
        self._ready: _Fetch | None = None  # the download prefetch() reported on disk
        if max_defer_s is not None and float(max_defer_s) <= 0:
            raise ValueError("max_defer_s must be > 0 (None: no bound)")
        self.max_defer_s = None if max_defer_s is None else float(max_defer_s)
        self._deferred_since: float | None = None  # first hold() deferral of the pending release
        self.last_forced = False  # the last load ignored hold() (deferral bound)
        self.latest_read_failures = 0  # polls of LATEST the store did not answer
        self.last_deferred_s: float | None = None  # how long the last load was deferred

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

    def _served_seq(self) -> int:
        return -1 if self.release is None else int(self.release.seq)

    def _newest(self, latest: PolicyRelease | None) -> PolicyRelease | None:
        """``latest`` unless it is a stale read (below the newest release seen): then
        the newest one seen."""
        if latest is not None and self.last_seen is not None and latest.seq < self.last_seen.seq:
            return self.last_seen
        return latest

    def _ready_fetch(self) -> "_Fetch | None":
        """The download :meth:`prefetch` reported on disk, while it is still newer
        than the served release."""
        fetch = self._ready
        if fetch is None or fetch is not self._fetch or fetch.release.seq <= self._served_seq():
            self._ready = None
            return None
        return fetch

    def pending_seq(self) -> int | None:
        """The seq of a release newer than the served one in the store (one read
        of ``LATEST``; nothing is loaded), else None. Once :meth:`prefetch` has
        reported a release on disk it is that release's seq (the one the next
        :meth:`refresh` loads), also when the store has a newer one by then."""
        ready = self._ready_fetch()
        if ready is not None:
            return int(ready.release.seq)
        try:
            latest = self._newest(self.releases.latest())
        except Exception as error:  # noqa: BLE001 - LATEST not read: unknown now, read again at the next poll
            if is_fatal_store_error(error):
                raise
            self.latest_read_failures += 1
            return None
        if latest is None or latest.seq <= self._served_seq():
            return None
        return int(latest.seq)

    def prefetch(self) -> int | bool:
        """Start (or poll) the background download of the newest release, when it is
        newer than the served one. Returns the seq of the release that is on local
        disk and that the next :meth:`refresh` loads; True when there is nothing to
        fetch (no newer release, no background download); False while it downloads
        or cannot be fetched. Nothing is loaded.

        A release reported here stays the one to load until it is loaded: a newer
        release published meanwhile is fetched after that load. (A caller that
        drains its trials for the reported release then always gets a load for the
        drain; without this a release published during the drain made the refresh
        drop the downloaded one and load nothing.)

        The same checks as :meth:`refresh` apply before any byte is fetched: a stale
        read of ``LATEST`` does not replace a newer release already seen, and a
        release of another scheme is not downloaded. A failed download, a failed
        rebuild and a scheme mismatch read False and stay unreported here: the next
        :meth:`refresh` raises them (``rl_sampler_release_not_loaded``)."""
        if not self.background_download:
            return True
        ready = self._ready_fetch()
        if ready is not None:
            return int(ready.release.seq)
        try:
            latest = self._newest(self.releases.latest())
        except Exception as error:  # noqa: BLE001 - LATEST not read: nothing on disk to report; read again at the next poll
            if is_fatal_store_error(error):
                raise
            self.latest_read_failures += 1
            return False
        if latest is None or latest.seq <= self._served_seq():
            return True
        if self.last_seen is None or latest.seq != self.last_seen.seq:
            self.seen_at = round(self.wall(), 3)
        self.last_seen = latest
        if self.scheme is not None:
            try:
                check_release_scheme(latest, self.scheme)
            except SchemeMismatch:
                return False  # raised by refresh; never downloaded
        fetch = self._fetched(latest, report=False)
        if fetch is None:
            return False
        self._ready = fetch
        return int(fetch.release.seq)

    def refresh(self) -> PolicyRelease | None:
        """The release loaded by this call, or None when there is no new one, when
        the newest one is not complete yet (``incomplete`` says why; the engines
        keep the release they serve), or when the in-flight rule defers it."""
        timer = PhaseTimer(self.timer)
        try:
            with timer.phase("read_latest_s"):
                latest = self.releases.latest()
        except Exception as error:  # noqa: BLE001 - a poll that failed, not a release that failed
            # the engines keep the release they serve and the next refresh reads again;
            # "damaged": the record is there and does not parse (the caller bounds those);
            # refused credentials / no such bucket are no outage: raised as they are
            if is_fatal_store_error(error):
                raise
            self.latest_read_failures += 1
            damaged = isinstance(error, _DAMAGED_RECORD) and not isinstance(error, OSError)
            raise ReleaseNotLoaded(f"the newest release could not be read: {type(error).__name__}: {error}",
                                   reason=LATEST_UNREAD, fields={"damaged": damaged}) from error
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
        if self.scheme is not None:
            check_release_scheme(latest, self.scheme)
        fetched = None
        if self.background_download:
            fetched = self._ready_fetch()  # reported on disk by prefetch: this one is loaded
            if fetched is None:
                fetched = self._fetched(latest)
            if fetched is None:  # still downloading, or waiting to retry: keep generating
                return None
            latest = fetched.release  # may be older than ``last_seen``: the newest is fetched next
        forced = False
        if (self._in_flight or self.hold()) and not self.live_update:
            now = self.timer()
            if self._deferred_since is None:
                self._deferred_since = now
            if (self._in_flight or self.max_defer_s is None
                    or now - self._deferred_since < self.max_defer_s):
                self.counters.deferred_in_flight += 1
                return None
            forced = True  # deferral bound reached (class doc)
        if fetched is not None:
            directory = fetched.directory
            self._fetch = self._ready = None
            timer.add("resolve_s", fetched.seconds)
        else:
            try:
                with timer.phase("resolve_s"):
                    directory = self.releases.resolve(latest, from_store=self.rebuild_failures > 0)
            except ReleaseIncomplete as error:
                self.incomplete = str(error)
                return None
            except ReleaseDeltaError as error:
                self._rebuild_failed(latest, error)
            except Exception as error:  # noqa: BLE001 - a failed transfer: keep the served weights
                raise _download_failed(latest, error, self.version) from error
        self.incomplete = None
        if not getattr(self._load, "verifies_files", False):
            try:
                with timer.phase("verify_s"):
                    check_release_files(latest, directory)
            except ReleaseDigestMismatch as error:
                self._rebuild_failed(latest, error)
        self.rebuild_failures = 0
        with timer.phase("load_s"):
            self._load(latest, directory)
        mark = getattr(self.releases, "mark_served", None)
        if callable(mark):  # loaded: the older downloads may go now
            mark(latest)
        previous = self.release
        self.counters.loaded += 1
        self.last_forced = forced
        self.last_deferred_s = (None if self._deferred_since is None
                                else round(self.timer() - self._deferred_since, 3))
        self._deferred_since = None
        if forced:
            self.counters.forced_loads += 1
        if previous is not None:
            self.counters.superseded += latest.seq - previous.seq - 1
            if latest.version < self.version:
                self.counters.lineage_resets += 1
        self.version = latest.version
        self.release = latest
        self.last_timings = timer.as_dict()
        return latest

    def _fetched(self, latest: PolicyRelease, *, report: bool = True) -> "_Fetch | None":
        """The finished background download of ``latest``, else None after starting
        or polling it. A failed download raises :class:`ReleaseNotLoaded` once
        (reason ``download_failed``) and is tried again ``download_retry_s`` later;
        the engines keep the release they serve meanwhile. ``report`` False
        (:meth:`prefetch`): a failure not raised yet is left for the next call with
        ``report`` True, which raises it and starts the retry delay."""
        fetch = self._fetch
        if fetch is not None and fetch.release.seq != latest.seq and not fetch.done.is_set():
            self.incomplete = f"downloading r{fetch.release.seq}-v{fetch.release.version}"
            return None  # one download at a time: the newer release follows this one
        if (fetch is not None and fetch.release.seq != latest.seq and fetch.error is None
                and fetch.release.seq > self._served_seq()):
            # finished while a newer release appeared: load it, the newer one is
            # fetched next. Dropped, a sampler whose download is slower than the
            # release interval restarted it at every release and never loaded one.
            return fetch
        if fetch is None or fetch.release.seq != latest.seq:
            self._fetch = fetch = _Fetch(latest)
            fetch.start(self.releases, from_store=self.rebuild_failures > 0, timer=self.timer)
            fetch.done.wait(DOWNLOAD_WAIT_S)  # a small release is there at once
        if not fetch.done.is_set():
            self.incomplete = f"downloading r{latest.seq}-v{latest.version}"
            return None
        if fetch.error is None:
            return fetch
        error = fetch.error
        if isinstance(error, ReleaseIncomplete):  # not visible yet: read it again next time
            self._fetch = None
            self.incomplete = str(error)
            return None
        if not fetch.reported:
            if not report:
                return None
            fetch.reported = True
            # jittered: samplers that failed together (a store refusal) do not retry together
            fetch.retry_at = self.timer() + self.download_retry_s * random.uniform(0.75, 1.0)
            if isinstance(error, ReleaseDeltaError):
                self._fetch = None
                self._rebuild_failed(latest, error)
            raise _download_failed(latest, error, self.version) from error
        if self.timer() >= (fetch.retry_at or 0.0):
            self._fetch = None  # the next refresh starts a new download
        return None

    def _rebuild_failed(self, release: PolicyRelease, error: ReleaseDeltaError) -> None:
        """Count a failed rebuild of ``release`` and raise (class doc)."""
        self.rebuild_failures += 1
        self.counters.rebuild_failures += 1
        failures = self.rebuild_failures
        if failures >= self.max_rebuild_failures:
            raise type(error)(f"r{release.seq}-v{release.version}: {failures} delta rebuilds or "
                              f"digest checks in a row failed (last: {error})", file=error.file) from error
        full = isinstance(error, ReleaseDigestMismatch)
        raise ReleaseNotLoaded(
            f"r{release.seq}-v{release.version}: {'digest check' if full else 'delta rebuild'} failed "
            f"({error}); the engines keep v{self.version}, the next try rebuilds from the full "
            "release in the store",
            reason="release_digest_mismatch" if full else "delta_rebuild_failed",
            fields={"release_version": release.version, "file": error.file,
                    "rebuild_failures": failures, "max_rebuild_failures": self.max_rebuild_failures},
        ) from error
