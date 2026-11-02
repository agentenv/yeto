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

Format v2 (:data:`MAGIC_V2`, S30 P0-b; ``YETO_RL_DELTA_FORMAT=v2``): every chunk
that changed is one ``zstd`` frame of the changed word indexes as gaps (uint32,
first gap from 0), then the changed words XOR the base word (uint16), each array
byte-plane shuffled (all low bytes, then the next byte plane, ...), so the many
small gaps and the near-identical high bytes compress; the footer's chunk
records are [kind, changed words, frame bytes] (kind ``z`` or ``same``) and it
carries ``"format": 2``. A v2 chunk applies to the chunk as rebuilt so far (the
XOR base is the delta's base), so :func:`apply_chain` mixes v1 and v2 deltas
freely; the target's sha256 is the same for both formats. ``zstd`` is the
standard library's ``compression.zstd`` (Python >= 3.14) or the ``zstandard``
package; only v2 files need it.

A directory delta (:func:`encode_directory`) covers every file of the target
release: a ``*.safetensors`` file with a same-size file of the same name in the
base gets a delta; every other file is copied whole. Its entries
(:class:`FileEntry`: mode, size, sha256 of the target) go into the release
record; :func:`apply_directory` rebuilds the target directory from a base
directory and checks every digest.

:class:`DeltaMirror` is the sampler side: a local directory that keeps the last
rebuilt releases, so the next delta applies to a local copy; a delta whose base
is not there is rebuilt from the newest full release along its chain.

This module is engine-neutral and imports only numpy (plus zstd for v2).
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
MAGIC_V2 = b"YDELTA2\n"
FORMAT_ENV = "YETO_RL_DELTA_FORMAT"  # trainer: "v1" (default) or "v2" for new delta files
ZSTD_LEVEL = 3
DELTA_SUFFIX = ".ydelta"
CHUNK_WORDS = 1 << 24  # 32 MiB of 16-bit words per chunk
MIN_DELTA_BYTES = 1 << 20  # smaller files are copied whole
DEFAULT_WORKERS = 8
DELTA, FULL_COPY = "delta", "full"
_IDX, _BITS, _SAME, _ZST = "idx", "bits", "same", "z"
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


# ---------------------------------------------------------------- zstd (v2)
def _zstd() -> tuple[Callable[[bytes, int], bytes], Callable[[bytes], bytes]]:
    """(compress(data, level), decompress(data)): ``compression.zstd`` or ``zstandard``."""
    try:
        from compression import zstd  # type: ignore[import-not-found]  # Python >= 3.14

        return (lambda data, level: zstd.compress(data, level=level)), zstd.decompress
    except ImportError:
        pass
    try:
        import zstandard
    except ImportError as error:
        raise ReleaseDeltaError("delta format v2 needs zstd: Python >= 3.14 or `pip install zstandard`") from error
    return ((lambda data, level: zstandard.ZstdCompressor(level=level).compress(data)),
            (lambda data: zstandard.ZstdDecompressor().decompress(data)))


def zstd_available() -> bool:
    try:
        _zstd()
    except ReleaseDeltaError:
        return False
    return True


def delta_format(environ: Mapping[str, str] | None = None) -> int:
    """The delta format new delta files get: :data:`FORMAT_ENV` (``v1``/``v2``), default 1."""
    value = (os.environ if environ is None else environ).get(FORMAT_ENV, "v1").strip().lower().lstrip("v")
    if value not in ("1", "2"):
        raise ValueError(f"{FORMAT_ENV}={value!r}: expected v1 or v2")
    return int(value)


def _shuffle(values: np.ndarray) -> bytes:
    """Byte planes of ``values`` (little-endian): every byte 0, then every byte 1, ..."""
    return values.view(np.uint8).reshape(-1, values.itemsize).T.tobytes()


def _unshuffle(raw: bytes, dtype: str, count: int) -> np.ndarray:
    size = np.dtype(dtype).itemsize
    planes = np.frombuffer(raw, dtype=np.uint8, count=size * count).reshape(size, count)
    return np.ascontiguousarray(planes.T).view(dtype).reshape(count)


def _v2_frame(a: np.ndarray, b: np.ndarray, index: np.ndarray, compress: Callable[[bytes, int], bytes],
              level: int) -> bytes:
    gaps = np.diff(index, prepend=np.uint32(0)).astype("<u4")
    values = (a[index] ^ b[index]).astype("<u2")
    return compress(_shuffle(gaps) + _shuffle(values), level)


# ---------------------------------------------------------------- one file
def _read_words(handle: Any, count: int) -> np.ndarray:
    """``count`` little-endian 16-bit words from ``handle`` into a new writable array."""
    words = np.empty(count, dtype="<u2")
    got = handle.readinto(memoryview(words).cast("B"))
    if got != 2 * count:
        raise ReleaseDeltaError(f"short read: {got} of {2 * count} bytes")
    return words


def encode_file(base: Path, target: Path, out: Path, *, chunk_words: int = CHUNK_WORDS,
                fmt: int = 1, level: int = ZSTD_LEVEL) -> tuple[int, str]:
    """Write the delta ``base`` -> ``target`` into ``out`` in format ``fmt`` (1 or
    2, module doc); the two files must have the same size. Returns (changed
    words, sha256 of ``target``)."""
    if fmt not in (1, 2):
        raise ValueError(f"unknown delta format {fmt}")
    compress = _zstd()[0] if fmt == 2 else None
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
                chunks.append([_SAME, 0] if fmt == 1 else [_SAME, 0, 0])
                continue
            if compress is not None:
                frame = _v2_frame(a, b, np.flatnonzero(mask).astype("<u4"), compress, level)
                sink.write(frame)
                chunks.append([_ZST, changed, len(frame)])
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
        fields: dict[str, Any] = {"target_size": size, "base_size": size, "chunk_words": chunk_words,
                                  "tail": tail_byte, "chunks": chunks}
        if fmt == 2:
            fields.update({"format": 2, "codec": "zstd", "xor": True})
        footer = json.dumps(fields, separators=(",", ":")).encode()
        sink.write(footer + _FOOTER.pack(len(footer)) + (MAGIC_V2 if fmt == 2 else MAGIC))
    return changed_total, digest.hexdigest()


def _footer(handle: Any, path: Path) -> dict[str, Any]:
    end = handle.seek(0, os.SEEK_END)
    trailer = _FOOTER.size + len(MAGIC)
    if end < trailer:
        raise ReleaseDeltaError(f"{path.name}: not a delta file")
    handle.seek(end - trailer)
    raw = handle.read(trailer)
    magic = raw[_FOOTER.size:]
    if magic not in (MAGIC, MAGIC_V2):
        raise ReleaseDeltaError(f"{path.name}: not a delta file (magic)")
    (length,) = _FOOTER.unpack(raw[:_FOOTER.size])
    handle.seek(end - trailer - length)
    footer = json.loads(handle.read(length))
    if int(footer.get("format", 1)) != (2 if magic == MAGIC_V2 else 1):
        raise ReleaseDeltaError(f"{path.name}: footer format does not match the magic")
    handle.seek(0)
    return footer


def _apply_chunk(words: np.ndarray, record: Sequence[Any], patch: Any, name: str) -> None:
    """Apply one chunk record of a delta file (read from ``patch``) to ``words``."""
    kind, changed = record[0], int(record[1])
    if kind == _ZST:
        frame = patch.read(int(record[2]))
        try:
            raw = _zstd()[1](frame)
        except ReleaseDeltaError:
            raise
        except Exception as error:  # a corrupt frame: the codec's own error type
            raise ReleaseDeltaError(f"{name}: bad zstd frame: {error}") from error
        if len(raw) != 6 * changed:
            raise ReleaseDeltaError(f"{name}: zstd frame holds {len(raw)} bytes, expected {6 * changed}")
        index = np.cumsum(_unshuffle(raw[:4 * changed], "<u4", changed), dtype=np.int64)
        if changed and (index[-1] >= words.size or np.any(np.diff(index) <= 0)):
            raise ReleaseDeltaError(f"{name}: word index out of the chunk")
        words[index] ^= _unshuffle(raw[4 * changed:], "<u2", changed)
    elif kind == _IDX:
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
                _apply_chunk(chunk, footer["chunks"][number], patch, Path(delta).name)
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
                     workers: int = DEFAULT_WORKERS, fmt: int | None = None) -> DirectoryDelta:
    """Write the delta of every file of ``target`` against ``base`` into ``out``
    (created; module doc). ``skip(name)``: files left out of the release (its
    record). ``fmt``: the delta file format (default :func:`delta_format`). Files
    run in parallel threads (numpy, hashlib and zstd release the GIL)."""
    base, target, out = Path(base), Path(target), Path(out)
    fmt = delta_format() if fmt is None else int(fmt)
    out.mkdir(parents=True, exist_ok=True)
    bases = _files(base, skip)

    def one(item: tuple[str, Path]) -> tuple[str, FileEntry, int]:
        name, path = item
        mode = _delta_mode(name, path, bases.get(name))
        destination = out / (name + DELTA_SUFFIX if mode == DELTA else name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if mode == DELTA:
            _changed, sha256 = encode_file(bases[name], path, destination, fmt=fmt)
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
    one pass. Raises :class:`ReleaseDeltaError` (with the file's name) on a
    digest mismatch or a delta that cannot be applied."""
    base, out = Path(base), Path(out)
    links = [(Path(directory), entries) for directory, entries in links]

    def one(item: tuple[str, FileEntry]) -> None:
        try:
            rebuild(*item)
        except ReleaseDeltaError as error:
            if error.file is not None:
                raise
            raise ReleaseDeltaError(f"{item[0]}: {error}", file=item[0]) from error

    def rebuild(name: str, entry: FileEntry) -> None:
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
            raise ReleaseDeltaError(f"{name}: rebuilt sha256 {actual} != released {entry.sha256}",
                                    file=name)

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        list(pool.map(one, links[-1][1].items()))


def entries_of(record: Mapping[str, Any]) -> dict[str, FileEntry]:
    """The :class:`FileEntry` map of a release record's ``delta`` field."""
    return {name: FileEntry.from_dict(value) for name, value in record["files"].items()}


# ---------------------------------------------------------------- sampler side
MIRROR_STAMP = ".yeto-mirror.json"  # in a rebuilt release's directory: which release it is


def mirror_stamp(release: Any) -> dict[str, Any]:
    """What identifies a release in a mirror (S30 D34): its name alone does not
    (every run's store has an ``r7-v12``); its sequence, version, policy hash and
    release time together do."""
    return {"path": str(release.path), "seq": int(release.seq), "version": int(release.version),
            "policy_hash": str(release.policy_hash),
            "released_at": None if getattr(release, "released_at", None) is None
            else float(release.released_at)}


class DeltaMirror:
    """Rebuilt releases on the sampler's local disk (module doc).

    ``root/<release path>/`` holds the full files of a rebuilt release; a
    directory appears only complete (rebuilt into ``.partial`` first and checked).
    :meth:`materialize` rebuilds a delta release: it follows the release's bases
    (``record_of(path)``) back to one in the mirror or to a full release in the
    store, then applies all the deltas after it in one pass per file. ``weights_dir(release)`` is a
    release's directory in the store (mount or download). ``from_store``: do not
    start from a rebuilt copy, follow the bases back to a full release in the
    store (a retry after a rebuild failed its digest check). The newest ``keep``
    rebuilt releases stay (the engines may still read the one they serve)."""

    def __init__(self, root: str | os.PathLike[str], *, keep: int = 2,
                 workers: int = DEFAULT_WORKERS) -> None:
        self.root = Path(root).expanduser()
        self.keep = max(1, int(keep))
        self.workers = int(workers)

    def holds(self, path: str, release: Any = None) -> bool:
        """The mirror holds ``path``; with ``release``: a copy rebuilt for that very
        release (its stamp, :data:`MIRROR_STAMP`, names the same record). S30 D34:
        the mirror root is shared by every run on a host (default under the
        system temp directory) and keyed by release name only, so ``r7-v12`` of
        another store or an earlier run is not this store's ``r7-v12``; a
        directory without a matching stamp is not used as a base."""
        directory = self.root / path
        if not directory.is_dir():
            return False
        if release is None:
            return True
        try:
            stamp = json.loads((directory / MIRROR_STAMP).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return stamp == mirror_stamp(release)

    def materialize(self, release: Any, *, record_of: Callable[[str], Any],
                    weights_dir: Callable[[Any], Path], from_store: bool = False) -> Path:
        def held(link: Any) -> bool:
            return not from_store and self.holds(link.path, link)

        chain = []
        link = release
        while link.delta is not None and not held(link):
            chain.append(link)
            link = record_of(str(link.delta["base_path"]))
        base = self.root / link.path if held(link) else weights_dir(link)
        if not chain:
            return base
        self.root.mkdir(parents=True, exist_ok=True)
        partial = self.root / f".{release.path}.partial"
        shutil.rmtree(partial, ignore_errors=True)
        try:
            apply_directory(base, [(weights_dir(step), entries_of(step.delta)) for step in reversed(chain)],
                            partial, workers=self.workers)
            (partial / MIRROR_STAMP).write_text(json.dumps(mirror_stamp(release), sort_keys=True),
                                                encoding="utf-8")
        except BaseException:
            shutil.rmtree(partial, ignore_errors=True)
            raise
        target = self.root / release.path
        if target.exists():  # a stale copy of another release with this name (D34)
            shutil.rmtree(target, ignore_errors=True)
        partial.replace(target)
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
