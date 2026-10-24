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

This module is engine-neutral and imports only numpy.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import tempfile
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

MIRROR_ENV = "YETO_RL_RELEASE_MIRROR"  # sampler: local directory of rebuilt delta releases
MAGIC = b"YDELTA1\n"
DELTA_SUFFIX = ".ydelta"
CHUNK_WORDS = 1 << 24  # 32 MiB of 16-bit words per chunk
MIN_DELTA_BYTES = 1 << 20  # smaller files are copied whole
DEFAULT_WORKERS = 8
DELTA, FULL_COPY = "delta", "full"
_IDX, _BITS, _SAME = "idx", "bits", "same"
_FOOTER = struct.Struct("<Q")


class ReleaseDeltaError(RuntimeError):
    """A delta that cannot be applied: a malformed file, another base, or a
    rebuilt file whose digest is not the recorded one."""


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
            mask = a != b
            changed = int(np.count_nonzero(mask))
            changed_total += changed
            if changed == 0:
                chunks.append([_SAME, 0])
                continue
            bitmap_bytes = (count + 7) // 8
            if 4 * changed < bitmap_bytes:
                index = np.flatnonzero(mask).astype("<u4")
                sink.write(index.tobytes())
                sink.write(b[index].tobytes())
                chunks.append([_IDX, changed])
            else:
                sink.write(np.packbits(mask).tobytes())
                sink.write(b[mask].tobytes())
                chunks.append([_BITS, changed])
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


def _apply_chunk(words: np.ndarray, kind: str, changed: int, patch: Any, name: str) -> None:
    """Apply one chunk record of a delta file (read from ``patch``) to ``words``."""
    if kind == _IDX:
        index = np.frombuffer(patch.read(4 * changed), dtype="<u4")
        words[index] = np.frombuffer(patch.read(2 * changed), dtype="<u2")
    elif kind == _BITS:
        bits = np.frombuffer(patch.read((words.size + 7) // 8), dtype=np.uint8)
        mask = np.unpackbits(bits, count=words.size).view(bool)
        words[mask] = np.frombuffer(patch.read(2 * changed), dtype="<u2")
    elif kind != _SAME:
        raise ReleaseDeltaError(f"{name}: unknown chunk kind {kind!r}")


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


def apply_directory(base: Path, links: Sequence[tuple[Path, Mapping[str, FileEntry]]], out: Path, *,
                    workers: int = DEFAULT_WORKERS) -> None:
    """Rebuild into ``out`` every file of the last of ``links`` (delta directory,
    entries; oldest first), starting from ``base``: a file starts from the newest
    link that holds it whole (else from ``base``) and takes the later deltas in
    one pass. Raises :class:`ReleaseDeltaError` on a digest mismatch."""
    base, out = Path(base), Path(out)
    links = [(Path(directory), entries) for directory, entries in links]

    def one(item: tuple[str, FileEntry]) -> None:
        name, entry = item
        start, deltas = base / name, []
        for directory, entries in links:
            if name not in entries or entries[name].mode != DELTA:
                start, deltas = directory / name, []
            else:
                deltas.append(directory / (name + DELTA_SUFFIX))
        destination = out / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if deltas:
            actual = apply_chain(start, deltas, destination)
        else:
            shutil.copyfile(start, destination)
            actual = file_sha256(destination)
        if actual != entry.sha256 or destination.stat().st_size != entry.size:
            raise ReleaseDeltaError(f"{name}: rebuilt sha256 {actual} != released {entry.sha256}")

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        list(pool.map(one, links[-1][1].items()))


def entries_of(record: Mapping[str, Any]) -> dict[str, FileEntry]:
    """The :class:`FileEntry` map of a release record's ``delta`` field."""
    return {name: FileEntry.from_dict(value) for name, value in record["files"].items()}


# ---------------------------------------------------------------- sampler side
class DeltaMirror:
    """Rebuilt releases on the sampler's local disk (module doc).

    ``root/<release path>/`` holds the full files of a rebuilt release; a
    directory appears only complete (rebuilt into ``.partial`` first and checked).
    :meth:`materialize` rebuilds a delta release: it follows the release's bases
    (``record_of(path)``) back to one in the mirror or to a full release in the
    store, then applies all the deltas after it in one pass per file. ``weights_dir(release)`` is a
    release's directory in the store (mount or download). The newest ``keep``
    rebuilt releases stay (the engines may still read the one they serve)."""

    def __init__(self, root: str | os.PathLike[str], *, keep: int = 2,
                 workers: int = DEFAULT_WORKERS) -> None:
        self.root = Path(root).expanduser()
        self.keep = max(1, int(keep))
        self.workers = int(workers)

    def holds(self, path: str) -> bool:
        return (self.root / path).is_dir()

    def materialize(self, release: Any, *, record_of: Callable[[str], Any],
                    weights_dir: Callable[[Any], Path]) -> Path:
        chain = []
        link = release
        while link.delta is not None and not self.holds(link.path):
            chain.append(link)
            link = record_of(str(link.delta["base_path"]))
        base = self.root / link.path if self.holds(link.path) else weights_dir(link)
        if not chain:
            return base
        self.root.mkdir(parents=True, exist_ok=True)
        partial = self.root / f".{release.path}.partial"
        shutil.rmtree(partial, ignore_errors=True)
        apply_directory(base, [(weights_dir(step), entries_of(step.delta)) for step in reversed(chain)],
                        partial, workers=self.workers)
        partial.replace(self.root / release.path)
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
    """A sampler's mirror: :data:`MIRROR_ENV`, else under the system temp directory."""
    env = os.environ if environ is None else environ
    return DeltaMirror(env.get(MIRROR_ENV) or Path(tempfile.gettempdir()) / "yeto-release-mirror")
