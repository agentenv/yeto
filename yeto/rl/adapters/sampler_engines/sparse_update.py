"""Sparse weight updates: send a running engine only the weight elements that
changed between the release it serves and the next one (numpy only; the
engine-side half is :mod:`.sglang_sparse_loader`).

A delta release rebuilt with change tracking (``release_delta.DeltaMirror``
``track_changes``) records, per weight file, the 16-bit words that differ from
the release it was rebuilt from. This module turns those words into elements
of named checkpoint tensors (:func:`changed_elements`, from the safetensors
headers), and the elements into per-rank patches of the engine's parameters
(:func:`rank_patches`) with the *element map* each engine rank reported.

Element map (one per tensor-parallel rank, written by the engine-side probe):
``{"params": {param: {"shape", "itemsize"}}, "tensors": {name: [target, ...]},
"unmapped": [name, ...]}``. A target places a checkpoint tensor into one rank
parameter as a separable index map: ``dims`` gives, for every parameter
dimension, ``["hf", d]`` (the coordinate comes from checkpoint dimension ``d``
through ``maps[d]``, an int64 array; -1 = not on this rank) or ``["const", c]``;
``select`` = {d: c}: only the checkpoint elements with coordinate ``c`` in
dimension ``d`` (a dimension no parameter dimension follows, e.g. one expert
of a stacked checkpoint tensor) are on this rank.
TP slices, fused projections (q/k/v, gate/up), stacked experts and transposes
are all of that form; a tensor the loader transforms in value or scatters in
another way is ``unmapped``, and a change to it makes the update fall back to
a full reload.

Patch (one directory per rank): ``patch.json`` (``params``: [[param, first,
count, itemsize, value byte offset]], ``checks``: the same for the check
samples), ``pos.bin`` (int64 flat positions), ``val.bin`` (raw element bits),
``check_pos.bin`` and ``check_val.bin`` (positions and the bits expected there
after the update; sampled from changed and unchanged elements).
"""

from __future__ import annotations

import json
import pickle
import shutil
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# safetensors dtype -> (unsigned view of its bits, item size)
_DTYPES = {"BF16": ("<u2", 2), "F16": ("<u2", 2), "F32": ("<u4", 4), "F64": ("<u8", 8),
           "I64": ("<u8", 8), "I32": ("<u4", 4), "I16": ("<u2", 2), "U16": ("<u2", 2),
           "I8": ("u1", 1), "U8": ("u1", 1), "BOOL": ("u1", 1),
           "F8_E4M3": ("u1", 1), "F8_E5M2": ("u1", 1)}
PATCH_FILE = "patch.json"


class SparseUnavailable(RuntimeError):
    """This update cannot be sparse; ``reason`` is a short code for the load record."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


@dataclass(frozen=True)
class Slot:
    """One tensor of a safetensors file: absolute byte range of its data."""

    name: str
    file: Path
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int

    @property
    def view(self) -> str:
        return _DTYPES[self.dtype][0]

    @property
    def itemsize(self) -> int:
        return _DTYPES[self.dtype][1]

    @property
    def numel(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) if self.shape else 1

    def values(self, index: np.ndarray | None = None) -> np.ndarray:
        """The raw bits of the elements ``index`` (all when None)."""
        data = np.memmap(self.file, dtype=self.view, mode="r", offset=self.start, shape=(self.numel,))
        return np.asarray(data if index is None else data[index])


def file_slots(path: Path) -> tuple[int, list[Slot]]:
    """(byte where the data begins, the tensors of ``path`` by data offset)."""
    path = Path(path)
    with path.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
    header.pop("__metadata__", None)
    begin = 8 + length
    slots = []
    for name, entry in header.items():
        if entry["dtype"] not in _DTYPES:
            raise SparseUnavailable("dtype", f"{path.name}: {name} has dtype {entry['dtype']}")
        first, last = (int(v) for v in entry["data_offsets"])
        slots.append(Slot(name, path, str(entry["dtype"]), tuple(int(d) for d in entry["shape"]),
                          begin + first, begin + last))
    return begin, sorted(slots, key=lambda s: s.start)


def release_slots(release_dir: Path) -> dict[str, Slot]:
    """Every tensor of the ``*.safetensors`` files directly in ``release_dir``."""
    out: dict[str, Slot] = {}
    for path in sorted(Path(release_dir).glob("*.safetensors")):
        for slot in file_slots(path)[1]:
            out[slot.name] = slot
    return out


def changed_elements(release_dir: Path, files: Mapping[str, Path | None]) -> dict[str, np.ndarray]:
    """{tensor: ascending int64 element indexes} of the elements that differ from
    the base, from the changed-word files of a change manifest
    (``release_delta.read_changes(...)["files"]``). Raises
    :class:`SparseUnavailable` when a change is not in tensor data (a header, the
    padding, a file that is not a weight file) or is unknown."""
    release_dir = Path(release_dir)
    out: dict[str, np.ndarray] = {}
    for name, words_path in sorted(files.items()):
        if words_path is None:
            raise SparseUnavailable("unknown_changes", name)
        words = np.fromfile(words_path, dtype="<u8").astype(np.int64)
        if not words.size:
            continue
        if not name.endswith(".safetensors") or "/" in name:
            raise SparseUnavailable("non_weight_file_changed", name)
        begin, slots = file_slots(release_dir / name)
        byte = words * 2
        if byte[0] < begin:
            raise SparseUnavailable("header_changed", name)
        starts = np.array([s.start for s in slots], dtype=np.int64)
        ends = np.array([s.end for s in slots], dtype=np.int64)
        which = np.searchsorted(starts, byte, side="right") - 1
        if np.any(which < 0) or np.any(byte + 1 >= ends[which]):
            raise SparseUnavailable("padding_changed", name)
        cut = np.flatnonzero(np.diff(which)) + 1
        for group in np.split(np.arange(which.size), cut):
            slot = slots[int(which[group[0]])]
            # a word spans two elements of a 1-byte tensor (or of an odd-aligned one)
            element = np.union1d((byte[group] - slot.start) // slot.itemsize,
                                 (byte[group] + 1 - slot.start) // slot.itemsize)
            out[slot.name] = element if slot.name not in out else np.union1d(out[slot.name], element)
    return out


# ---------------------------------------------------------------- element maps
def load_map(path: Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)  # noqa: S301 - written by the engine this island started


def target_positions(target: Mapping[str, Any], shape: Sequence[int], param_shape: Sequence[int],
                     index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(flat positions in the parameter, which of ``index`` land on this rank)."""
    coords = np.unravel_index(index, tuple(shape)) if len(shape) else ()
    valid = np.ones(index.size, dtype=bool)
    for dim, coordinate in (target.get("select") or {}).items():  # a slice that is on this rank
        valid &= coords[int(dim)] == int(coordinate)
    parts = []
    for kind, value in target["dims"]:
        if kind == "const":
            parts.append(np.full(index.size, int(value), dtype=np.int64))
        else:
            mapped = np.asarray(target["maps"][int(value)], dtype=np.int64)[coords[int(value)]]
            valid &= mapped >= 0
            parts.append(mapped)
    if not parts:  # a 0-d parameter
        return np.zeros(int(valid.sum()), dtype=np.int64), valid
    pos = np.ravel_multi_index(tuple(p[valid] for p in parts), tuple(param_shape))
    return pos.astype(np.int64), valid


@dataclass
class _Writes:
    pos: list[np.ndarray]
    val: list[np.ndarray]
    itemsize: int


def _add(table: dict[str, _Writes], param: str, itemsize: int, pos: np.ndarray, val: np.ndarray) -> None:
    if not pos.size:
        return
    entry = table.setdefault(param, _Writes([], [], itemsize))
    entry.pos.append(pos)
    entry.val.append(val)


def _tensor_writes(table: dict[str, _Writes], mapping: Mapping[str, Any], slot: Slot, index: np.ndarray,
                   values: np.ndarray) -> None:
    targets = mapping["tensors"].get(slot.name)
    if targets is None:
        raise SparseUnavailable("unmapped", slot.name)
    for target in targets:
        param = mapping["params"][target["param"]]
        if int(param["itemsize"]) != slot.itemsize:
            raise SparseUnavailable("dtype", f"{slot.name} -> {target['param']}")
        pos, valid = target_positions(target, slot.shape, param["shape"], index)
        _add(table, target["param"], slot.itemsize, pos, values[valid])


def _write(directory: Path, prefix: str, table: Mapping[str, _Writes]) -> tuple[list[list[Any]], int]:
    records, first, offset, count = [], 0, 0, 0
    with (directory / f"{prefix}pos.bin").open("wb") as pos_file, \
            (directory / f"{prefix}val.bin").open("wb") as val_file:
        for param, entry in sorted(table.items()):
            pos = np.concatenate(entry.pos).astype("<i8")
            val = np.concatenate(entry.val)
            pos_file.write(pos.tobytes())
            val_file.write(val.tobytes())
            records.append([param, first, int(pos.size), entry.itemsize, offset])
            first += int(pos.size)
            offset += int(val.nbytes)
            count += int(pos.size)
    return records, count


def rank_patches(out: Path, maps: Sequence[Mapping[str, Any]], slots: Mapping[str, Slot],
                 changes: Mapping[str, np.ndarray], checks: Mapping[str, np.ndarray],
                 meta: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Write one patch directory per rank under ``out`` (module doc): the new bits
    of every changed element, and the check samples ``checks`` ({tensor: element
    indexes}). Returns per rank {"dir", "elements", "params", "checks"}."""
    out = Path(out)
    shutil.rmtree(out, ignore_errors=True)
    values = {name: slots[name].values(index) for name, index in changes.items()}
    expected = {name: slots[name].values(index) for name, index in checks.items()}
    summary = []
    for rank, mapping in enumerate(maps):
        writes: dict[str, _Writes] = {}
        samples: dict[str, _Writes] = {}
        for name, index in changes.items():
            _tensor_writes(writes, mapping, slots[name], index, values[name])
        for name, index in checks.items():
            _tensor_writes(samples, mapping, slots[name], index, expected[name])
        directory = out / f"rank{rank}"
        directory.mkdir(parents=True)
        params, elements = _write(directory, "", writes)
        check_records, checked = _write(directory, "check_", samples)
        (directory / PATCH_FILE).write_text(json.dumps({**dict(meta), "rank": rank, "params": params,
                                                        "checks": check_records}), encoding="utf-8")
        summary.append({"dir": str(directory), "elements": elements, "params": len(params), "checks": checked})
    return summary


def pick_checks(rng: np.random.Generator, slots: Mapping[str, Slot], mapped: Iterable[str],
                changes: Mapping[str, np.ndarray], *, tensors: int, per_tensor: int) -> dict[str, np.ndarray]:
    """Check samples: up to ``tensors`` changed tensors (their changed elements)
    and as many mapped tensors (any element), ``per_tensor`` elements each."""
    out: dict[str, np.ndarray] = {}
    changed = sorted(changes)
    for name in rng.permutation(changed)[:tensors] if changed else ():
        index = changes[str(name)]
        out[str(name)] = np.unique(rng.choice(index, size=min(per_tensor, index.size), replace=False))
    pool = sorted(set(mapped) & set(slots))
    for name in rng.permutation(pool)[:tensors] if pool else ():
        numel = slots[str(name)].numel
        drawn = rng.integers(0, numel, size=min(per_tensor, numel), dtype=np.int64)
        out[str(name)] = np.unique(np.concatenate([out.get(str(name), drawn[:0]), drawn]))
    return out
