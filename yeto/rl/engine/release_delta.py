"""bf16 delta releases: send the sampler only the weight words that changed (S27 G-BG).

A full-weight release is a whole HF checkpoint (Nemotron Nano: 63 GB). Between two
policy versions most bf16 words do not change: at lr 1e-6 an Adam step moves a
weight by about 1e-6, below half a bf16 step for most weights (CPU simulation,
``s1-runs/s27-bg/delta_sim.py``: 0.9-1.5 % of the words per version). A *delta
release* therefore stores, for each weight file, the words that differ from the
same file of the previous release; the sampler rebuilds the new file from its
copy of the previous one. Lossless: the rebuilt file must have the sha256 the
trainer recorded for the new file, or the load fails.

File format (one ``<name>.ydelta`` per weight file; ``<name>`` is the target's
relative path). The target is read as little-endian 16-bit words (a trailing odd
byte is kept in the footer) in chunks of ``chunk_words`` words; per chunk the
payload holds the words that differ, in the smaller of two encodings:

* ``idx``: uint32 word indexes within the chunk, then the uint16 new words;
* ``bits``: a bitmap (``numpy.packbits``, one bit per word of the chunk), then
  the uint16 new words;
* ``same``: nothing (no word differs).

After the payload: the footer JSON (``target_size``, ``base_size``,
``chunk_words``, ``tail``, ``chunks`` = [[kind, changed words], ...]), its
length as uint64 and :data:`MAGIC`. Writing and reading are sequential, chunk by
chunk; the sha256 of the target is computed while it is read (trainer) or
written (sampler), so no file is read twice.

A directory delta (:func:`encode_directory`) covers every file of the target
release: a ``*.safetensors`` file with a same-size file of the same name in the
base gets a delta; every other file is copied whole. Its entries
(:class:`FileEntry`: mode, size, sha256 of the target) go into the release
record; :func:`apply_directory` rebuilds the target directory from a base
directory and checks every digest.

:class:`DeltaMirror` is the sampler side: a local directory that keeps the last
rebuilt releases, so the next delta applies to a local copy; a delta whose base
is not there is rebuilt from the newest full release along its chain.

Two engines make and apply the same delta files (``YETO_RL_RELEASE_DELTA_ENGINE``):

* the file engine (``files``, S27 G-BG; :func:`encode_directory`,
  :func:`apply_directory`): one thread per file, the base read from disk;
* the resident engine (``resident``, the default; :class:`DeltaEngine`, S27
  G-CJ): the base release's weight files stay in memory as 16-bit words
  (:class:`ResidentRelease`, about the size of the weights), so no base file is
  read from disk; each file is cut into its chunks and the chunks of all files
  run on ``YETO_RL_RELEASE_DELTA_WORKERS`` threads (default: the CPUs, at most
  16); per file, one thread at a time hashes the chunks in order (and writes the
  delta chunks in order). The trainer's copy becomes each release it encodes
  (or, for a full release, hashes); the sampler's copy becomes each release it
  rebuilds, so the next delta applies in memory. After a run that did not
  finish, the copy is no release and the next run reads its base from disk.
  Why (s27-cc-e2-a): with 13 weight files of 5 GB and 8 threads the file engine
  ran in two waves of one 5 GB file per thread (encode 57-108 s per release on
  the trainer, rebuild 57-100 s on the sampler), while hashing the same staged
  files took 8-11 s.

This module is engine-neutral and imports only numpy.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import shutil
import struct
import tempfile
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

MIRROR_ENV = "YETO_RL_RELEASE_MIRROR"  # sampler: local directory of rebuilt delta releases
MAGIC = b"YDELTA1\n"
DELTA_SUFFIX = ".ydelta"
CHUNK_WORDS = 1 << 24  # 32 MiB of 16-bit words per chunk
MIN_DELTA_BYTES = 1 << 20  # smaller files are copied whole
DEFAULT_WORKERS = 8  # the file engine's threads (one file each)
ENGINE_ENV = "YETO_RL_RELEASE_DELTA_ENGINE"  # "resident" (default) or "files" (S27 G-CJ)
WORKERS_ENV = "YETO_RL_RELEASE_DELTA_WORKERS"  # threads of the resident engine and the digests
RESIDENT, FILES = "resident", "files"
MAX_WORKERS = 16  # default thread cap of the resident engine
DELTA, FULL_COPY = "delta", "full"
_IDX, _BITS, _SAME = "idx", "bits", "same"
_FOOTER = struct.Struct("<Q")


class ReleaseDeltaError(RuntimeError):
    """A delta that cannot be applied: a malformed file, another base, or a
    rebuilt file whose digest is not the recorded one. ``file``: the release
    file it is about, when known."""

    def __init__(self, message: str, *, file: str | None = None) -> None:
        super().__init__(message)
        self.file = file


@dataclass(frozen=True)
class FileEntry:
    """One file of a delta release: ``mode`` (:data:`DELTA` or :data:`FULL_COPY`),
    the target's ``size`` and ``sha256``."""

    mode: str
    size: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FileEntry":
        return cls(mode=str(value["mode"]), size=int(value["size"]), sha256=str(value["sha256"]))


@dataclass(frozen=True)
class DirectoryDelta:
    """What :func:`encode_directory` wrote: the entries and the bytes involved."""

    entries: dict[str, FileEntry]
    delta_bytes: int  # bytes written into the delta directory
    target_bytes: int  # bytes of the target release

    @property
    def ratio(self) -> float:
        return self.delta_bytes / self.target_bytes if self.target_bytes else 1.0


# ---------------------------------------------------------------- one file
def _diff_chunk(base: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[str, int, bytes]:
    """(kind, changed words, payload) of one chunk; ``mask`` is ``base != target``.
    The changed words of ``target`` are also written into ``base`` (the resident
    engine keeps its copy current in place; the file engine discards ``base``)."""
    changed = int(np.count_nonzero(mask))
    if changed == 0:
        return _SAME, 0, b""
    if 4 * changed < (target.size + 7) // 8:
        index = np.flatnonzero(mask).astype("<u4")
        values = target[index]
        base[index] = values
        return _IDX, changed, index.tobytes() + values.tobytes()
    values = target[mask]
    base[mask] = values
    return _BITS, changed, np.packbits(mask).tobytes() + values.tobytes()


def _payload_size(kind: str, changed: int, count: int) -> int:
    """Bytes of one chunk record of ``count`` words in a delta file."""
    if kind == _IDX:
        return 6 * changed
    if kind == _BITS:
        return (count + 7) // 8 + 2 * changed
    return 0


def _read_words(handle: Any, count: int) -> np.ndarray:
    """``count`` little-endian 16-bit words from ``handle`` into a new writable array."""
    words = np.empty(count, dtype="<u2")
    got = handle.readinto(memoryview(words).cast("B"))
    if got != 2 * count:
        raise ReleaseDeltaError(f"short read: {got} of {2 * count} bytes")
    return words


def encode_file(base: Path, target: Path, out: Path, *, chunk_words: int = CHUNK_WORDS) -> tuple[int, str]:
    """Write the delta ``base`` -> ``target`` into ``out``; the two files must have
    the same size. Returns (changed words, sha256 of ``target``)."""
    size = os.path.getsize(target)
    if os.path.getsize(base) != size:
        raise ReleaseDeltaError(f"{target.name}: base and target sizes differ")
    words, tail = divmod(size, 2)
    digest = hashlib.sha256()
    chunks: list[list[Any]] = []
    changed_total = 0
    with base.open("rb") as old, target.open("rb") as new, out.open("wb") as sink:
        for start in range(0, words, chunk_words):
            count = min(chunk_words, words - start)
            a, b = _read_words(old, count), _read_words(new, count)
            digest.update(b.data)
            kind, changed, payload = _diff_chunk(a, b, a != b)
            changed_total += changed
            sink.write(payload)
            chunks.append([kind, changed])
        tail_byte = None
        if tail:
            old.read(1)
            last = new.read(1)
            digest.update(last)
            tail_byte = last[0]
        footer = json.dumps({"target_size": size, "base_size": size, "chunk_words": chunk_words,
                             "tail": tail_byte, "chunks": chunks}, separators=(",", ":")).encode()
        sink.write(footer + _FOOTER.pack(len(footer)) + MAGIC)
    return changed_total, digest.hexdigest()


def _footer(handle: Any, path: Path) -> dict[str, Any]:
    end = handle.seek(0, os.SEEK_END)
    trailer = _FOOTER.size + len(MAGIC)
    if end < trailer:
        raise ReleaseDeltaError(f"{path.name}: not a delta file")
    handle.seek(end - trailer)
    raw = handle.read(trailer)
    if raw[_FOOTER.size:] != MAGIC:
        raise ReleaseDeltaError(f"{path.name}: not a delta file (magic)")
    (length,) = _FOOTER.unpack(raw[:_FOOTER.size])
    handle.seek(end - trailer - length)
    footer = json.loads(handle.read(length))
    handle.seek(0)
    return footer


def _patch_chunk(words: np.ndarray, kind: str, changed: int, payload: bytes, name: str) -> None:
    """Apply one chunk record of a delta file (its ``payload`` bytes) to ``words``."""
    if kind not in (_IDX, _BITS, _SAME):
        raise ReleaseDeltaError(f"{name}: unknown chunk kind {kind!r}")
    if len(payload) != _payload_size(kind, changed, words.size):
        raise ReleaseDeltaError(f"{name}: truncated delta chunk")
    if kind == _IDX:
        words[np.frombuffer(payload, dtype="<u4", count=changed)] = np.frombuffer(
            payload, dtype="<u2", offset=4 * changed)
    elif kind == _BITS:
        bitmap = (words.size + 7) // 8
        mask = np.unpackbits(np.frombuffer(payload, dtype=np.uint8, count=bitmap), count=words.size).view(bool)
        words[mask] = np.frombuffer(payload, dtype="<u2", offset=bitmap)


def _apply_chunk(words: np.ndarray, kind: str, changed: int, patch: Any, name: str) -> None:
    """Apply one chunk record of a delta file (read from ``patch``) to ``words``."""
    size = _payload_size(kind, changed, words.size) if kind in (_IDX, _BITS, _SAME) else 0
    _patch_chunk(words, kind, changed, patch.read(size), name)


def apply_chain(base: Path, deltas: Sequence[Path], out: Path) -> str:
    """Write ``base`` with every delta of ``deltas`` applied in order into ``out``,
    in one pass (each chunk is read once, patched by every delta, written once);
    returns the sha256 of ``out``."""
    if not deltas:
        raise ValueError("apply_chain needs at least one delta")
    with ExitStack() as stack:
        patches = [stack.enter_context(Path(d).open("rb")) for d in deltas]
        footers = [_footer(patch, Path(d)) for patch, d in zip(patches, deltas)]
        size = os.path.getsize(base)
        for footer, delta in zip(footers, deltas):
            if int(footer["base_size"]) != size or int(footer["chunk_words"]) != int(footers[0]["chunk_words"]):
                raise ReleaseDeltaError(f"{Path(delta).name}: the base has another size or chunking")
            size = int(footer["target_size"])
        chunk_words, words = int(footers[0]["chunk_words"]), size // 2
        digest = hashlib.sha256()
        old, sink = stack.enter_context(Path(base).open("rb")), stack.enter_context(Path(out).open("wb"))
        for number in range(len(footers[0]["chunks"])):
            chunk = _read_words(old, min(chunk_words, words - number * chunk_words))
            for footer, patch, delta in zip(footers, patches, deltas):
                kind, changed = footer["chunks"][number]
                _apply_chunk(chunk, kind, int(changed), patch, Path(delta).name)
            digest.update(chunk.data)
            sink.write(chunk.data)
        tail = footers[-1]["tail"]
        if tail is not None:
            digest.update(bytes([int(tail)]))
            sink.write(bytes([int(tail)]))
    return digest.hexdigest()


def apply_file(base: Path, delta: Path, out: Path) -> str:
    """Write ``base`` with ``delta`` applied into ``out``; returns its sha256."""
    return apply_chain(base, [delta], out)


def file_sha256(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------- directories
def _files(root: Path, skip: Callable[[str], bool] = lambda name: False) -> dict[str, Path]:
    return {p.relative_to(root).as_posix(): p for p in sorted(root.rglob("*"))
            if p.is_file() and not skip(p.relative_to(root).as_posix())}


def _delta_mode(name: str, target: Path, base: Path | None) -> str:
    if (base is None or not name.endswith(".safetensors")
            or os.path.getsize(target) < MIN_DELTA_BYTES
            or os.path.getsize(base) != os.path.getsize(target)):
        return FULL_COPY
    return DELTA


def encode_directory(base: Path, target: Path, out: Path, *, skip: Callable[[str], bool] = lambda n: False,
                     workers: int = DEFAULT_WORKERS) -> DirectoryDelta:
    """Write the delta of every file of ``target`` against ``base`` into ``out``
    (created; module doc). ``skip(name)``: files left out of the release (its
    record). Files run in parallel threads (numpy and hashlib release the GIL)."""
    base, target, out = Path(base), Path(target), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    bases = _files(base, skip)

    def one(item: tuple[str, Path]) -> tuple[str, FileEntry, int]:
        name, path = item
        mode = _delta_mode(name, path, bases.get(name))
        destination = out / (name + DELTA_SUFFIX if mode == DELTA else name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if mode == DELTA:
            _changed, sha256 = encode_file(bases[name], path, destination)
        else:
            shutil.copyfile(path, destination)
            sha256 = file_sha256(path)
        return name, FileEntry(mode, path.stat().st_size, sha256), destination.stat().st_size

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        results = list(pool.map(one, _files(target, skip).items()))
    return DirectoryDelta(entries={name: entry for name, entry, _ in results},
                          delta_bytes=sum(size for _, _, size in results),
                          target_bytes=sum(entry.size for _, entry, _ in results))


def _file_chain(name: str, base: Path | None, links: Sequence[tuple[Path, Mapping[str, FileEntry]]]
                ) -> tuple[Path | None, list[Path]]:
    """(start file, deltas after it) of ``name`` along ``links`` (oldest first): the
    newest link that holds the file whole, else ``base`` (None: no base)."""
    start, deltas = (None if base is None else base / name), []
    for directory, entries in links:
        if name not in entries or entries[name].mode != DELTA:
            start, deltas = directory / name, []
        else:
            deltas.append(directory / (name + DELTA_SUFFIX))
    return start, deltas


def _named(name: str, run: Callable[[], Any]) -> Any:
    """``run()``; a :class:`ReleaseDeltaError` without a file names ``name``."""
    try:
        return run()
    except ReleaseDeltaError as error:
        if error.file is not None:
            raise
        raise ReleaseDeltaError(f"{name}: {error}", file=name) from error


def _check_rebuilt(name: str, entry: FileEntry, actual: str, size: int) -> None:
    if actual != entry.sha256 or size != entry.size:
        raise ReleaseDeltaError(f"{name}: rebuilt sha256 {actual} != released {entry.sha256}", file=name)


def _rebuild_file(name: str, entry: FileEntry, start: Path | None, deltas: Sequence[Path],
                  destination: Path) -> None:
    """Rebuild one file sequentially (``start`` with ``deltas`` applied) and check it."""
    if start is None:
        raise ReleaseDeltaError(f"{name}: no base copy to rebuild from", file=name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if deltas:
        actual = apply_chain(start, deltas, destination)
    else:
        shutil.copyfile(start, destination)
        actual = file_sha256(destination)
    _check_rebuilt(name, entry, actual, destination.stat().st_size)


def apply_directory(base: Path, links: Sequence[tuple[Path, Mapping[str, FileEntry]]], out: Path, *,
                    workers: int = DEFAULT_WORKERS) -> None:
    """Rebuild into ``out`` every file of the last of ``links`` (delta directory,
    entries; oldest first), starting from ``base``: a file starts from the newest
    link that holds it whole (else from ``base``) and takes the later deltas in
    one pass. Raises :class:`ReleaseDeltaError` (with the file's name) on a
    digest mismatch or a delta that cannot be applied."""
    base, out = Path(base), Path(out)
    links = [(Path(directory), entries) for directory, entries in links]

    def one(item: tuple[str, FileEntry]) -> None:
        name, entry = item
        _named(name, lambda: _rebuild_file(name, entry, *_file_chain(name, base, links), out / name))

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        list(pool.map(one, links[-1][1].items()))


def entries_of(record: Mapping[str, Any]) -> dict[str, FileEntry]:
    """The :class:`FileEntry` map of a release record's ``delta`` field."""
    return {name: FileEntry.from_dict(value) for name, value in record["files"].items()}


# ---------------------------------------------------------------- resident engine (S27 G-CJ)
def delta_workers(environ: Mapping[str, str] | None = None) -> int:
    """Threads of the resident engine and of the release digests:
    :data:`WORKERS_ENV`, else this host's CPUs, at most :data:`MAX_WORKERS`."""
    env = os.environ if environ is None else environ
    raw = str(env.get(WORKERS_ENV, "") or "").strip()
    if raw:
        if not raw.isdigit() or int(raw) < 1:
            raise ValueError(f"{WORKERS_ENV} must be an integer >= 1, not {raw!r}")
        return int(raw)
    return max(1, min(MAX_WORKERS, os.cpu_count() or DEFAULT_WORKERS))


def delta_engine_kind(environ: Mapping[str, str] | None = None) -> str:
    """:data:`ENGINE_ENV`: :data:`RESIDENT` (default) or :data:`FILES` (G-BG's
    engine: one thread per file, the base read from disk)."""
    env = os.environ if environ is None else environ
    kind = str(env.get(ENGINE_ENV, "") or RESIDENT).strip().lower()
    if kind not in (RESIDENT, FILES):
        raise ValueError(f"{ENGINE_ENV} must be {RESIDENT!r} or {FILES!r}, not {kind!r}")
    return kind


def delta_engine(environ: Mapping[str, str] | None = None) -> "DeltaEngine | None":
    """The engine the environment selects: a :class:`DeltaEngine`, or None for
    the file engine (:func:`encode_directory`, :func:`apply_directory`)."""
    if delta_engine_kind(environ) == FILES:
        return None
    return DeltaEngine(workers=delta_workers(environ))


def _resident_file(name: str, size: int) -> bool:
    """A file the resident copy holds: a weight file large enough for a delta."""
    return name.endswith(".safetensors") and size >= MIN_DELTA_BYTES


def _pread_into(fd: int, words: np.ndarray, offset: int) -> None:
    view, got = memoryview(words).cast("B"), 0
    while got < len(view):
        count = os.preadv(fd, [view[got:]], offset + got)
        if count <= 0:
            raise ReleaseDeltaError(f"short read: {got} of {len(view)} bytes at {offset}")
        got += count


def _pread_exact(fd: int, size: int, offset: int) -> bytes:
    data = os.pread(fd, size, offset)
    while len(data) < size:
        more = os.pread(fd, size - len(data), offset + len(data))
        if not more:
            raise ReleaseDeltaError(f"short read: {len(data)} of {size} bytes at {offset}")
        data += more
    return data


def _pwrite_all(fd: int, data: Any, offset: int) -> None:
    view, done = memoryview(data).cast("B"), 0
    while done < len(view):
        done += os.pwrite(fd, view[done:], offset + done)


class ResidentRelease:
    """One release's weight files in memory, as 16-bit words: the base of the
    resident engine (:class:`DeltaEngine`). ``path``: the release (its store
    directory name) the words are; None while they are no release (never filled,
    or an encode or rebuild that did not finish)."""

    def __init__(self) -> None:
        self.path: str | None = None
        self._words: dict[str, np.ndarray] = {}
        self._tails: dict[str, int | None] = {}

    def holds(self, path: str | None) -> bool:
        return path is not None and self.path == path

    def size(self, name: str) -> int | None:
        """Bytes of ``name`` in this copy (None: not held)."""
        words = self._words.get(name)
        return None if words is None else 2 * words.size + (self._tails.get(name) is not None)

    def file_bytes(self, name: str) -> bytes:
        """The held file ``name`` (tests and checks)."""
        tail = self._tails.get(name)
        return self._words[name].tobytes() + (b"" if tail is None else bytes([tail]))

    @property
    def nbytes(self) -> int:
        return sum(words.nbytes for words in self._words.values())

    def _array(self, name: str, size: int) -> np.ndarray:
        """The words of ``name`` for a file of ``size`` bytes (an array of that
        length is reused as it is)."""
        words = self._words.get(name)
        if words is None or words.size != size // 2:
            words = self._words[name] = np.empty(size // 2, dtype="<u2")
        return words

    def _settle(self, path: str, tails: Mapping[str, int | None]) -> None:
        """The run that filled the files of ``tails`` (name: last byte of an odd
        size) finished: they are release ``path``; other files are dropped."""
        for name in [n for n in self._words if n not in tails]:
            del self._words[name]
        self._tails = dict(tails)
        self.path = path


class _Ordered:
    """The pieces of one file finish in any order; ``drain(index, result)`` runs
    for each in piece order, on one thread at a time, then ``close()`` once."""

    def __init__(self, pieces: int, drain: Callable[[int, Any], None], close: Callable[[], None]) -> None:
        self.pieces = pieces
        self._drain, self._close = drain, close
        self._lock = threading.Lock()
        self._ready: dict[int, Any] = {}
        self._next = 0
        self._draining = False

    def done(self, index: int, result: Any) -> None:
        with self._lock:
            self._ready[index] = result
            if self._draining:
                return
            self._draining = True
        while True:
            with self._lock:
                if self._next not in self._ready:
                    self._draining = False
                    return
                index, self._next = self._next, self._next + 1
                result = self._ready.pop(index)
            self._drain(index, result)
            if index == self.pieces - 1:
                self._close()


class _Scratch:
    """One thread's buffers (allocated on first use, reused for every piece)."""

    def __init__(self, chunk_words: int) -> None:
        self.chunk_words = chunk_words
        self._words: np.ndarray | None = None
        self._mask: np.ndarray | None = None

    def words(self, count: int) -> np.ndarray:
        if self._words is None:
            self._words = np.empty(self.chunk_words, dtype="<u2")
        return self._words[:count]

    def mask(self, count: int) -> np.ndarray:
        if self._mask is None:
            self._mask = np.empty(self.chunk_words, dtype=bool)
        return self._mask[:count]


def _interleaved(groups: Sequence[Sequence[Callable[[_Scratch], None]]]) -> list[Callable[[_Scratch], None]]:
    """The tasks of every group, one of each group in turn (each group keeps its
    order): concurrent pieces then mostly read and write different files."""
    return [task for row in itertools.zip_longest(*groups) for task in row if task is not None]


def _run_tasks(tasks: Sequence[Callable[[_Scratch], None]], workers: int, chunk_words: int) -> None:
    """Run ``tasks`` on ``workers`` threads, started in list order; the first
    error stops the start of further tasks and is raised."""
    lock, errors, position = threading.Lock(), [], [0]

    def work() -> None:
        scratch = _Scratch(chunk_words)
        while True:
            with lock:
                if errors or position[0] >= len(tasks):
                    return
                task = tasks[position[0]]
                position[0] += 1
            try:
                task(scratch)
            except BaseException as error:  # noqa: BLE001 - raised on the caller's thread
                with lock:
                    errors.append(error)
                return

    threads = [threading.Thread(target=work, name=f"release-delta-{i}", daemon=True)
               for i in range(max(1, min(int(workers), len(tasks))))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]


class _PieceFile:
    """One resident file of a run: its pieces (format chunks) and its digest."""

    def __init__(self, name: str, size: int, words: np.ndarray, chunk_words: int) -> None:
        self.name, self.size, self.words, self.chunk_words = name, size, words, chunk_words
        self.digest = hashlib.sha256()
        self.sha256: str | None = None
        self.tail_byte: int | None = None
        pieces = (words.size + chunk_words - 1) // chunk_words
        self.ordered = _Ordered(pieces, self._drain, lambda: _named(self.name, self._finish))

    def span(self, index: int) -> tuple[int, int]:
        start = index * self.chunk_words
        return start, min(self.chunk_words, self.words.size - start)

    def tasks(self) -> list[Callable[[_Scratch], None]]:
        if self.ordered.pieces == 0:
            return [lambda scratch: _named(self.name, self._finish)]
        return [partial(self._piece, index) for index in range(self.ordered.pieces)]

    def _piece(self, index: int, scratch: _Scratch) -> None:
        result = _named(self.name, lambda: self.piece(index, scratch))
        self.ordered.done(index, result)

    def _drain(self, index: int, result: Any) -> None:
        start, count = self.span(index)
        self.digest.update(self.words[start:start + count])
        self.drain(index, result)

    def _finish(self) -> None:
        tail = self.tail_byte = self.tail()
        if tail is not None:
            self.digest.update(bytes([tail]))
        self.sha256 = self.digest.hexdigest()
        self.finish(tail)

    # per kind
    def piece(self, index: int, scratch: _Scratch) -> Any:
        raise NotImplementedError

    def drain(self, index: int, result: Any) -> None:
        pass

    def tail(self) -> int | None:
        raise NotImplementedError

    def finish(self, tail: int | None) -> None:
        pass


class _EncodeFile(_PieceFile):
    """Trainer side: diff ``target`` against the base words (resident, or first
    read from ``base`` on disk) into a delta file (``delta``), or copy it whole
    (``copy``; None with ``delta``: only hash it); the words become ``target``'s."""

    def __init__(self, name: str, target: Path, words: np.ndarray, chunk_words: int, stack: ExitStack, *,
                 base: Path | None, delta: Path | None, copy: Path | None) -> None:
        super().__init__(name, os.path.getsize(target), words, chunk_words)
        self.target = stack.enter_context(_fd(target, os.O_RDONLY))
        self.base = None if base is None else stack.enter_context(_fd(base, os.O_RDONLY))
        self.copy = None if copy is None else stack.enter_context(
            _fd(copy, os.O_WRONLY | os.O_CREAT | os.O_TRUNC))
        self.sink = None if delta is None else stack.enter_context(Path(delta).open("wb"))
        self.chunks: list[list[Any]] = []

    def piece(self, index: int, scratch: _Scratch) -> Any:
        start, count = self.span(index)
        words = self.words[start:start + count]
        if self.sink is None:
            _pread_into(self.target, words, 2 * start)
            if self.copy is not None:
                _pwrite_all(self.copy, words, 2 * start)
            return None
        if self.base is not None:
            _pread_into(self.base, words, 2 * start)
        target = scratch.words(count)
        _pread_into(self.target, target, 2 * start)
        return _diff_chunk(words, target, np.not_equal(words, target, out=scratch.mask(count)))

    def drain(self, index: int, result: Any) -> None:
        if self.sink is not None:
            kind, changed, payload = result
            self.sink.write(payload)
            self.chunks.append([kind, changed])

    def tail(self) -> int | None:
        if not self.size % 2:
            return None
        last = _pread_exact(self.target, 1, self.size - 1)
        if self.copy is not None:
            os.pwrite(self.copy, last, self.size - 1)
        return last[0]

    def finish(self, tail: int | None) -> None:
        if self.sink is not None:
            footer = json.dumps({"target_size": self.size, "base_size": self.size,
                                 "chunk_words": self.chunk_words, "tail": tail, "chunks": self.chunks},
                                separators=(",", ":")).encode()
            self.sink.write(footer + _FOOTER.pack(len(footer)) + MAGIC)
            self.sink.flush()


class _RebuildFile(_PieceFile):
    """Sampler side: rebuild ``entry`` into ``out``: the words (resident, or first
    read from ``start`` on disk) with ``deltas`` applied, chunk by chunk."""

    def __init__(self, name: str, entry: FileEntry, words: np.ndarray, chunk_words: int, stack: ExitStack, *,
                 start: Path | None, start_size: int, deltas: Sequence[Path], out: Path) -> None:
        self.entry = entry
        self.start = None if start is None else stack.enter_context(_fd(start, os.O_RDONLY))
        self.links: list[tuple[int, list[Any], list[int], str]] = []
        size, footers = start_size, []
        for delta in deltas:
            handle = stack.enter_context(Path(delta).open("rb", buffering=0))
            footer = _footer(handle, Path(delta))
            if int(footer["base_size"]) != size or (footers and int(footer["chunk_words"]) != chunk_words):
                raise ReleaseDeltaError(f"{Path(delta).name}: the base has another size or chunking", file=name)
            chunk_words, size = int(footer["chunk_words"]), int(footer["target_size"])
            footers.append(footer)
            self.links.append((handle.fileno(), footer["chunks"], [], Path(delta).name))
        if size != entry.size:
            raise ReleaseDeltaError(f"{name}: the rebuilt size {size} is not the released {entry.size}", file=name)
        super().__init__(name, size, words, chunk_words)
        for _, chunks, offsets, delta_name in self.links:
            if len(chunks) != self.ordered.pieces:
                raise ReleaseDeltaError(f"{delta_name}: {len(chunks)} chunks for {self.ordered.pieces}", file=name)
            offset = 0
            for index, (kind, changed) in enumerate(chunks):
                offsets.append(offset)
                offset += _payload_size(kind, int(changed), self.span(index)[1])
        self.footer_tail = footers[-1]["tail"] if footers else None
        self.out = stack.enter_context(_fd(out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC))

    def piece(self, index: int, scratch: _Scratch) -> Any:
        start, count = self.span(index)
        words = self.words[start:start + count]
        if self.start is not None:
            _pread_into(self.start, words, 2 * start)
        for fd, chunks, offsets, delta_name in self.links:
            kind, changed = chunks[index]
            size = _payload_size(kind, int(changed), count)
            _patch_chunk(words, kind, int(changed), _pread_exact(fd, size, offsets[index]) if size else b"",
                         delta_name)
        _pwrite_all(self.out, words, 2 * start)
        return None

    def tail(self) -> int | None:
        if self.links:
            tail = self.footer_tail
        elif self.size % 2:
            if self.start is None:
                raise ReleaseDeltaError(f"{self.name}: no base copy of the last byte", file=self.name)
            tail = _pread_exact(self.start, 1, self.size - 1)[0]
        else:
            tail = None
        if tail is not None:
            os.pwrite(self.out, bytes([int(tail)]), self.size - 1)
        return None if tail is None else int(tail)

    def finish(self, tail: int | None) -> None:
        _check_rebuilt(self.name, self.entry, str(self.sha256), os.fstat(self.out).st_size)


@contextmanager
def _fd(path: Path, flags: int) -> Iterator[int]:
    fd = os.open(path, flags, 0o644)
    try:
        yield fd
    finally:
        os.close(fd)


class DeltaEngine:
    """The resident engine (module doc, resident engine): encodes and rebuilds
    delta releases chunk by chunk on ``workers`` threads, with the base release's
    weight files in memory (:attr:`resident`). Same file format and results as
    :func:`encode_directory` and :func:`apply_directory`."""

    def __init__(self, *, workers: int | None = None, chunk_words: int = CHUNK_WORDS,
                 resident: ResidentRelease | None = None) -> None:
        self.workers = delta_workers() if workers is None else max(1, int(workers))
        self.chunk_words = int(chunk_words)
        self.resident = ResidentRelease() if resident is None else resident

    def encode(self, base: Path | None, base_path: str | None, target: Path, target_path: str,
               out: Path | None, *, skip: Callable[[str], bool] = lambda n: False) -> DirectoryDelta:
        """:func:`encode_directory` of ``target`` into ``out`` against release
        ``base_path``: its resident copy when the engine holds it, else its files
        in ``base`` (None: no base, every file goes whole). ``out`` None: write
        nothing, only hash ``target`` (a full release's ``file_sha256``) and keep
        it resident. Afterwards the resident copy is ``target_path``."""
        target = Path(target)
        resident, from_resident = self.resident, self.resident.holds(base_path)
        bases = {} if base is None or not Path(base).is_dir() else _files(Path(base), skip)
        resident.path = None  # filled in place from here on
        if out is not None:
            Path(out).mkdir(parents=True, exist_ok=True)
        entries: dict[str, FileEntry] = {}
        outputs: dict[str, Path] = {}
        files: list[_EncodeFile] = []
        small: list[Callable[[_Scratch], None]] = []
        with ExitStack() as stack:
            for name, path in sorted(_files(target, skip).items(), key=lambda item: -item[1].stat().st_size):
                size = path.stat().st_size
                if not _resident_file(name, size):
                    small.append(partial(self._encode_small, name, path, out, entries, outputs))
                    continue
                if out is not None and from_resident and resident.size(name) == size:
                    mode, base_file = DELTA, None
                elif out is not None and name in bases and bases[name].stat().st_size == size:
                    mode, base_file = DELTA, bases[name]
                else:
                    mode, base_file = FULL_COPY, None
                destination = None if out is None else Path(out) / (name + DELTA_SUFFIX if mode == DELTA else name)
                if destination is not None:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    outputs[name] = destination
                piece = _EncodeFile(name, path, resident._array(name, size), self.chunk_words, stack,
                                    base=base_file, delta=destination if mode == DELTA else None,
                                    copy=destination if mode == FULL_COPY else None)
                entries[name] = FileEntry(mode, size, "")
                files.append(piece)
            _run_tasks(_interleaved([piece.tasks() for piece in files]) + small, self.workers, self.chunk_words)
        for piece in files:
            entries[piece.name] = FileEntry(entries[piece.name].mode, piece.size, str(piece.sha256))
        resident._settle(target_path, {piece.name: piece.tail_byte for piece in files})
        return DirectoryDelta(entries=dict(sorted(entries.items())),
                              delta_bytes=sum(path.stat().st_size for path in outputs.values()),
                              target_bytes=sum(entry.size for entry in entries.values()))

    @staticmethod
    def _encode_small(name: str, path: Path, out: Path | None, entries: dict[str, FileEntry],
                      outputs: dict[str, Path], _scratch: _Scratch) -> None:
        if out is not None:
            destination = Path(out) / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            outputs[name] = destination
        entries[name] = FileEntry(FULL_COPY, path.stat().st_size, file_sha256(path))

    def rebuild(self, base: Path | None, base_path: str | None,
                links: Sequence[tuple[Path, Mapping[str, FileEntry]]], out: Path, target_path: str) -> None:
        """:func:`apply_directory` into ``out``: a file whose chain starts at release
        ``base_path`` starts from its resident copy when the engine holds it, else
        from its file in ``base`` (None: no base on disk). Afterwards the resident
        copy is ``target_path``; after an error it is no release."""
        resident, from_resident = self.resident, self.resident.holds(base_path)
        resident.path = None  # patched in place from here on
        base, out = (None if base is None else Path(base)), Path(out)
        links = [(Path(directory), entries) for directory, entries in links]
        out.mkdir(parents=True, exist_ok=True)
        files: list[_RebuildFile] = []
        small: list[Callable[[_Scratch], None]] = []
        with ExitStack() as stack:
            for name, entry in sorted(links[-1][1].items(), key=lambda item: -item[1].size):
                start, deltas = _file_chain(name, base, links)
                destination = out / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not _resident_file(name, entry.size):
                    small.append(lambda _s, n=name, e=entry, st=start, d=deltas, o=destination:
                                 _named(n, lambda: _rebuild_file(n, e, st, d, o)))
                    continue
                at_base = start is None or (base is not None and start == base / name)
                if at_base and from_resident and resident.size(name) is not None:
                    start, start_size = None, int(resident.size(name))
                elif start is None:
                    raise ReleaseDeltaError(f"{name}: no base copy to rebuild from", file=name)
                else:
                    start_size = os.path.getsize(start)
                piece = _RebuildFile(name, entry, resident._array(name, entry.size), self.chunk_words, stack,
                                     start=start, start_size=start_size, deltas=deltas, out=destination)
                files.append(piece)
            _run_tasks(_interleaved([piece.tasks() for piece in files]) + small, self.workers, self.chunk_words)
        resident._settle(target_path, {piece.name: piece.tail_byte for piece in files})


# ---------------------------------------------------------------- sampler side
class DeltaMirror:
    """Rebuilt releases on the sampler's local disk (module doc).

    ``root/<release path>/`` holds the full files of a rebuilt release; a
    directory appears only complete (rebuilt into ``.partial`` first and checked).
    :meth:`materialize` rebuilds a delta release: it follows the release's bases
    (``record_of(path)``) back to one in the mirror (or held by the engine's
    resident copy) or to a full release in the store, then applies all the deltas
    after it in one pass per file. ``weights_dir(release)`` is a
    release's directory in the store (mount or download). ``from_store``: do not
    start from a rebuilt copy, follow the bases back to a full release in the
    store (a retry after a rebuild failed its digest check). The newest ``keep``
    rebuilt releases stay (the engines may still read the one they serve).
    ``engine``: the resident engine (:class:`DeltaEngine`); None: the file
    engine (:func:`apply_directory` on ``workers`` threads)."""

    def __init__(self, root: str | os.PathLike[str], *, keep: int = 2,
                 workers: int = DEFAULT_WORKERS, engine: DeltaEngine | None = None) -> None:
        self.root = Path(root).expanduser()
        self.keep = max(1, int(keep))
        self.workers = int(workers)
        self.engine = engine

    def holds(self, path: str) -> bool:
        return (self.root / path).is_dir()

    def materialize(self, release: Any, *, record_of: Callable[[str], Any],
                    weights_dir: Callable[[Any], Path], from_store: bool = False) -> Path:
        def held(path: str) -> bool:
            return not from_store and self.holds(path)

        def resident(path: str) -> bool:
            return not from_store and self.engine is not None and self.engine.resident.holds(path)

        chain = []
        link = release
        while link.delta is not None and not held(link.path) and not (chain and resident(link.path)):
            chain.append(link)
            link = record_of(str(link.delta["base_path"]))
        if held(link.path):
            base: Path | None = self.root / link.path
        else:
            base = weights_dir(link) if link.delta is None else None  # None: resident only
        if not chain:
            return base
        self.root.mkdir(parents=True, exist_ok=True)
        partial_dir = self.root / f".{release.path}.partial"
        shutil.rmtree(partial_dir, ignore_errors=True)
        links = [(weights_dir(step), entries_of(step.delta)) for step in reversed(chain)]
        if self.engine is None:
            apply_directory(base, links, partial_dir, workers=self.workers)
        else:
            self.engine.rebuild(base, None if from_store else link.path, links, partial_dir, release.path)
        partial_dir.replace(self.root / release.path)
        self._prune(newest=release.path)
        return self.root / release.path

    def _prune(self, newest: str) -> None:
        def seq(path: Path) -> int:
            head = path.name.split("-", 1)[0]
            return int(head[1:]) if head[1:].isdigit() else -1

        kept = sorted((p for p in self.root.iterdir() if p.is_dir() and not p.name.startswith(".")),
                      key=seq)[-self.keep:]
        for path in self.root.iterdir():
            if path.is_dir() and path not in kept and path.name != newest:
                shutil.rmtree(path, ignore_errors=True)


def default_mirror(environ: Mapping[str, str] | None = None) -> DeltaMirror:
    """A sampler's mirror: :data:`MIRROR_ENV`, else under the system temp
    directory; the engine the environment selects (:func:`delta_engine`)."""
    env = os.environ if environ is None else environ
    return DeltaMirror(env.get(MIRROR_ENV) or Path(tempfile.gettempdir()) / "yeto-release-mirror",
                       workers=delta_workers(env), engine=delta_engine(env))
