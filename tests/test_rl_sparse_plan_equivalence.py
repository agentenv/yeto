"""The vectorised sparse plan (``sparse_update.changed_elements`` and
``rank_patches``) writes byte for byte the patches of the straightforward
per-tensor implementation it replaced (kept here as the reference), on probed
MoE maps (multi-rank TP, EP, transposed w2, padded vocabulary) and on synthetic
maps and files (selects, constants, permuted and transposed maps, 1/4-byte and
odd-aligned tensors, unsorted word files); plus a timing regression."""

from __future__ import annotations

import json
import struct
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from yeto.rl.adapters.sampler_engines import sparse_update as su

from test_rl_sparse_probe_moe import MoeConfig, MoeRank, _probe, checkpoint_shapes


# ---------------------------------------------------------------- reference (the replaced implementation)
def ref_changed_elements(release_dir: Path, files: Mapping[str, Path | None]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, words_path in sorted(files.items()):
        if words_path is None:
            raise su.SparseUnavailable("unknown_changes", name)
        words = np.fromfile(words_path, dtype="<u8").astype(np.int64)
        if not words.size:
            continue
        if not name.endswith(".safetensors") or "/" in name:
            raise su.SparseUnavailable("non_weight_file_changed", name)
        begin, slots = su.file_slots(Path(release_dir) / name)
        byte = words * 2
        if byte[0] < begin:
            raise su.SparseUnavailable("header_changed", name)
        starts = np.array([s.start for s in slots], dtype=np.int64)
        ends = np.array([s.end for s in slots], dtype=np.int64)
        which = np.searchsorted(starts, byte, side="right") - 1
        if np.any(which < 0) or np.any(byte + 1 >= ends[which]):
            raise su.SparseUnavailable("padding_changed", name)
        cut = np.flatnonzero(np.diff(which)) + 1
        for group in np.split(np.arange(which.size), cut):
            slot = slots[int(which[group[0]])]
            element = np.union1d((byte[group] - slot.start) // slot.itemsize,
                                 (byte[group] + 1 - slot.start) // slot.itemsize)
            out[slot.name] = element if slot.name not in out else np.union1d(out[slot.name], element)
    return out


def _ref_tensor_writes(table: dict, mapping: Mapping[str, Any], slot: su.Slot, index: np.ndarray,
                       values: np.ndarray) -> None:
    targets = mapping["tensors"].get(slot.name)
    if targets is None:
        raise su.SparseUnavailable("unmapped", slot.name)
    for target in targets:
        param = mapping["params"][target["param"]]
        if int(param["itemsize"]) != slot.itemsize:
            raise su.SparseUnavailable("dtype", f"{slot.name} -> {target['param']}")
        pos, valid = su.target_positions(target, slot.shape, param["shape"], index)
        su._add(table, target["param"], slot.itemsize, pos, values[valid])


def ref_rank_patches(out: Path, maps: Sequence[Mapping[str, Any]], slots: Mapping[str, su.Slot],
                     changes: Mapping[str, np.ndarray], checks: Mapping[str, np.ndarray],
                     meta: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = {name: slots[name].values(index) for name, index in changes.items()}
    expected = {name: slots[name].values(index) for name, index in checks.items()}
    summary = []
    for rank, mapping in enumerate(maps):
        writes: dict = {}
        samples: dict = {}
        for name, index in changes.items():
            _ref_tensor_writes(writes, mapping, slots[name], index, values[name])
        for name, index in checks.items():
            _ref_tensor_writes(samples, mapping, slots[name], index, expected[name])
        directory = Path(out) / f"rank{rank}"
        directory.mkdir(parents=True)
        params, elements = su._write(directory, "", writes)
        check_records, checked = su._write(directory, "check_", samples)
        (directory / su.PATCH_FILE).write_text(json.dumps({**dict(meta), "rank": rank, "params": params,
                                                           "checks": check_records}), encoding="utf-8")
        summary.append({"dir": str(directory), "elements": elements, "params": len(params), "checks": checked})
    return summary


# ---------------------------------------------------------------- helpers
_SIZE = {"BF16": 2, "F32": 4, "U8": 1, "I64": 8}


def _write_release(path: Path, tensors: Sequence[tuple[str, str, tuple[int, ...]]], *, seed: int = 0,
                   gap: int = 0, sparse: bool = False) -> Path:
    """A safetensors file in this order (``gap`` padding bytes between tensors,
    so a tensor can start at an odd byte); random data or a hole (``sparse``)."""
    header, offset = {}, 0
    for name, dtype, shape in tensors:
        nbytes = int(np.prod(shape, dtype=np.int64)) * _SIZE[dtype]
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + nbytes]}
        offset += nbytes + gap
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(raw)) + raw)
        if sparse:
            handle.truncate(8 + len(raw) + offset)
        else:
            handle.write(np.random.default_rng(seed).integers(0, 256, offset, dtype=np.uint8).tobytes())
    return path


def _random_words(rng: np.random.Generator, slots: Sequence[su.Slot], fraction: float) -> np.ndarray:
    """Changed words inside tensor data (word w changes bytes 2w and 2w + 1)."""
    out = []
    for slot in slots:
        first, last = -(-slot.start // 2), slot.end // 2  # start <= 2w, 2w + 1 < end
        if last > first:
            count = max(1, int((last - first) * fraction))
            out.append(rng.integers(first, last, count, dtype=np.int64))
    return np.unique(np.concatenate(out))


def _words_file(path: Path, words: np.ndarray) -> Path:
    words.astype("<u8").tofile(path)
    return path


def _plan_both(tmp_path: Path, release: Path, files: Mapping[str, Path], maps: Sequence[Mapping[str, Any]],
               seed: int = 3) -> int:
    """Run both plans; assert identical changes, summaries and patch bytes."""
    slots = su.release_slots(release)
    ref_changes = ref_changed_elements(release, files)
    changes = su.changed_elements(release, files)
    assert list(changes) == list(ref_changes)
    for name in changes:
        assert changes[name].dtype == ref_changes[name].dtype and np.array_equal(changes[name], ref_changes[name])
    mapped = set.intersection(*(set(m["tensors"]) for m in maps))
    checks = su.pick_checks(np.random.default_rng(seed), slots, mapped, changes, tensors=6, per_tensor=50)
    a, b = tmp_path / "ref", tmp_path / "new"
    ref_summary = ref_rank_patches(a, maps, slots, ref_changes, checks, {"version": 7})
    summary = su.rank_patches(b, maps, slots, changes, checks, {"version": 7})
    assert [{**s, "dir": Path(s["dir"]).name} for s in summary] == \
        [{**s, "dir": Path(s["dir"]).name} for s in ref_summary]
    files_a = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file())
    assert files_a == sorted(p.relative_to(b) for p in b.rglob("*") if p.is_file())
    for rel in files_a:
        assert (a / rel).read_bytes() == (b / rel).read_bytes(), rel
    return sum(int(s["elements"]) for s in summary)


# ---------------------------------------------------------------- probed MoE maps
MOE_LAYOUTS = {"tp2": dict(tp=2), "tp4": dict(tp=4), "ep2": dict(tp=2, ep=True),
               "w2-transposed-tp2": dict(tp=2, transpose_w2=True)}


@pytest.mark.parametrize("layout", sorted(MOE_LAYOUTS))
@pytest.mark.parametrize("fraction", [0.02, 0.4])
def test_plan_matches_the_reference_on_probed_moe_ranks(tmp_path, layout, fraction):
    c = MoeConfig(experts=8, vocab=42)  # 42 rows over 4 ranks: a padded vocabulary
    shapes = checkpoint_shapes(c)
    kw = dict(MOE_LAYOUTS[layout])
    tp = kw.pop("tp")
    maps = [_probe(MoeRank(c, rank=r, tp=tp, **kw), shapes, tmp_path / f"probe{r}", "fast") for r in range(tp)]
    assert all(m["unmapped"] == [] for m in maps)
    release = _write_release(tmp_path / "release" / "model.safetensors",
                             [(n, "BF16", s) for n, s in shapes.items()])
    _, slots = su.file_slots(release)
    words = _random_words(np.random.default_rng(1), slots, fraction)
    files = {"model.safetensors": _words_file(tmp_path / "words.bin", words)}
    assert _plan_both(tmp_path, release.parent, files, maps) > 0


# ---------------------------------------------------------------- synthetic maps
def _shard(n: int, rank: int, per: int, offset: int = 0) -> np.ndarray:
    table = np.full(n, -1, dtype=np.int64)
    rows = np.arange(rank * per, min((rank + 1) * per, n))
    table[rows] = np.arange(rows.size) + offset
    return table


def _synthetic(tp: int) -> tuple[list[tuple[str, str, tuple[int, ...]]], list[dict[str, Any]]]:
    V, PV, H, E, I = 23, 24, 6, 3, 4
    tensors = [("embed", "BF16", (V, H)), ("norm", "F32", (H,)), ("bytes", "U8", (5, 3)),
               ("stacked", "BF16", (E, I, H)), ("w2", "BF16", (H, I)), ("perm", "BF16", (H, H)),
               ("scalar", "BF16", ()), ("big", "I64", (3, 2)), ("absent", "BF16", (2, 2))]
    rng = np.random.default_rng(5)
    maps = []
    for r in range(tp):
        params = {"embed": {"shape": [PV // tp, H], "itemsize": 2}, "norm": {"shape": [H], "itemsize": 4},
                  "bytes": {"shape": [3, 5], "itemsize": 1}, "experts": {"shape": [E, I // tp, H], "itemsize": 2},
                  "w2t": {"shape": [E, I // tp, H], "itemsize": 2}, "perm": {"shape": [H, H], "itemsize": 2},
                  "scalar": {"shape": [], "itemsize": 2}, "big": {"shape": [2, 3], "itemsize": 8},
                  "absent": {"shape": [2, 2], "itemsize": 2}}
        row_perm = rng.permutation(H)
        tensors_map = {
            "embed": [{"param": "embed", "dims": [["hf", 0], ["hf", 1]],
                       "maps": {0: _shard(V, r, PV // tp), 1: np.arange(H)}, "select": {}}],
            "norm": [{"param": "norm", "dims": [["hf", 0]], "maps": {0: np.arange(H)}, "select": {}}],
            "bytes": [{"param": "bytes", "dims": [["hf", 1], ["hf", 0]],  # transposed
                       "maps": {0: np.arange(5), 1: np.arange(3)}, "select": {}}],
            # one expert of a stacked checkpoint tensor per target (select), sliced rows
            "stacked": [{"param": "experts", "dims": [["const", e], ["hf", 1], ["hf", 2]],
                         "maps": {1: _shard(I, r, I // tp), 2: np.arange(H)}, "select": {0: e}} for e in range(E)],
            # down projection stored transposed and sliced in its input dimension, copied into two experts
            "w2": [{"param": "w2t", "dims": [["const", e], ["hf", 1], ["hf", 0]],
                    "maps": {0: np.arange(H), 1: _shard(I, r, I // tp)}, "select": {}} for e in (0, 2)],
            # rows permuted, odd columns only
            "perm": [{"param": "perm", "dims": [["hf", 0], ["hf", 1]],
                      "maps": {0: row_perm, 1: np.where(np.arange(H) % 2 == 1, np.arange(H), -1)}, "select": {}}],
            "scalar": [{"param": "scalar", "dims": [], "maps": {}, "select": {}}],
            "big": [{"param": "big", "dims": [["hf", 1], ["hf", 0]], "maps": {0: np.arange(3), 1: np.arange(2)},
                     "select": {}}],
            "absent": [] if r else [{"param": "absent", "dims": [["hf", 0], ["hf", 1]],
                                     "maps": {0: np.arange(2), 1: np.arange(2)}, "select": {}}],
        }
        maps.append({"params": params, "tensors": tensors_map, "unmapped": []})
    return tensors, maps


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("gap", [0, 1])  # 1: tensors at odd bytes, words spanning two tensors' elements
@pytest.mark.parametrize("order", ["sorted", "shuffled"])
def test_plan_matches_the_reference_on_synthetic_maps(tmp_path, tp, gap, order):
    tensors, maps = _synthetic(tp)
    release = _write_release(tmp_path / "release" / "model.safetensors", tensors, gap=gap)
    _, slots = su.file_slots(release)
    rng = np.random.default_rng(tp + 10 * gap)
    words = _random_words(rng, slots, 0.5)
    if order == "shuffled":  # not what the mirror writes, but must still give the same patch
        words = rng.permutation(np.concatenate([words, words[:5]]))
    files = {"model.safetensors": _words_file(tmp_path / "words.bin", words), "config.json": tmp_path / "none.bin"}
    (tmp_path / "none.bin").write_bytes(b"")
    assert _plan_both(tmp_path, release.parent, files, maps) > 0


def test_plan_refuses_what_the_reference_refuses(tmp_path):
    tensors, maps = _synthetic(2)
    release = _write_release(tmp_path / "release" / "model.safetensors", tensors, gap=4)
    begin, slots = su.file_slots(release)
    good = _random_words(np.random.default_rng(0), slots, 0.3)
    cases = {"header_changed": np.concatenate([[begin // 2 - 1], good]),
             "padding_changed": np.unique(np.concatenate([good, [slots[0].end // 2 + 1]])),
             "padding_changed_end": np.concatenate([good, [(slots[-1].end + 8) // 2]])}
    for case, words in cases.items():
        files = {"model.safetensors": _words_file(tmp_path / f"{case}.bin", words)}
        with pytest.raises(su.SparseUnavailable) as ref_error:
            ref_changed_elements(release.parent, files)
        with pytest.raises(su.SparseUnavailable) as error:
            su.changed_elements(release.parent, files)
        assert (error.value.reason, str(error.value)) == (ref_error.value.reason, str(ref_error.value)), case
    slot_map = su.release_slots(release.parent)
    changes = su.changed_elements(release.parent, {"model.safetensors": _words_file(tmp_path / "w.bin", good)})
    for broken, fix in (("unmapped", lambda m: m["tensors"].pop("perm")),
                        ("dtype", lambda m: m["params"]["norm"].update(itemsize=2))):
        bad = [maps[0], {**maps[1], "tensors": dict(maps[1]["tensors"]),
                         "params": {k: dict(v) for k, v in maps[1]["params"].items()}}]
        fix(bad[1])
        with pytest.raises(su.SparseUnavailable) as ref_error:
            ref_rank_patches(tmp_path / f"r-{broken}", bad, slot_map, changes, {}, {})
        with pytest.raises(su.SparseUnavailable) as error:
            su.rank_patches(tmp_path / f"n-{broken}", bad, slot_map, changes, {}, {})
        assert (error.value.reason, str(error.value)) == (ref_error.value.reason, str(ref_error.value)) and \
            error.value.reason == broken


# ---------------------------------------------------------------- timing regression
def test_plan_time_per_changed_element(tmp_path):
    """~2.4M changed elements over TP 4 (sliced rows and columns, stacked experts):
    the plan stays well under the reference's ~170 ns per element (measured
    ~8 ns at 100M elements, 18 threads; generous bounds for a shared CI box)."""
    H, I, E, tp = 1024, 384, 8, 4
    tensors = [("embed", "BF16", (8000, H))]
    for layer in range(2):
        tensors += [(f"{layer}.q", "BF16", (H, H)), (f"{layer}.o", "BF16", (H, H))]
        tensors += [(f"{layer}.e{e}.{k}", "BF16", (I, H)) for e in range(E) for k in ("gate", "up")]
    release = _write_release(tmp_path / "release" / "model.safetensors", tensors, sparse=True)
    maps = []
    for r in range(tp):
        params, tmap = {"embed": {"shape": [8000 // tp, H], "itemsize": 2}}, {}
        tmap["embed"] = [{"param": "embed", "dims": [["hf", 0], ["hf", 1]],
                          "maps": {0: _shard(8000, r, 8000 // tp), 1: np.arange(H)}, "select": {}}]
        for layer in range(2):
            params[f"{layer}.q"] = {"shape": [H // tp, H], "itemsize": 2}
            params[f"{layer}.o"] = {"shape": [H, H // tp], "itemsize": 2}
            params[f"{layer}.w13"] = {"shape": [E, 2 * I // tp, H], "itemsize": 2}
            tmap[f"{layer}.q"] = [{"param": f"{layer}.q", "dims": [["hf", 0], ["hf", 1]],
                                   "maps": {0: _shard(H, r, H // tp), 1: np.arange(H)}, "select": {}}]
            tmap[f"{layer}.o"] = [{"param": f"{layer}.o", "dims": [["hf", 0], ["hf", 1]],
                                   "maps": {0: np.arange(H), 1: _shard(H, r, H // tp)}, "select": {}}]
            for e in range(E):
                for k, off in (("gate", 0), ("up", I // tp)):
                    tmap[f"{layer}.e{e}.{k}"] = [{"param": f"{layer}.w13", "dims": [["const", e], ["hf", 0], ["hf", 1]],
                                                  "maps": {0: _shard(I, r, I // tp, off), 1: np.arange(H)},
                                                  "select": {}}]
        maps.append({"params": params, "tensors": tmap, "unmapped": []})
    _, slots = su.file_slots(release)
    words = _random_words(np.random.default_rng(0), slots, 0.1)
    files = {"model.safetensors": _words_file(tmp_path / "words.bin", words)}
    slot_map = su.release_slots(release.parent)

    def plan(changed, patches) -> tuple[float, int]:
        began = time.perf_counter()
        changes = changed(release.parent, files)
        patches(tmp_path / f"out-{patches.__name__}", maps, slot_map, changes, {}, {"version": 1})
        return time.perf_counter() - began, sum(int(i.size) for i in changes.values())

    ref_s, n = plan(ref_changed_elements, ref_rank_patches)
    new_s = min(plan(su.changed_elements, su.rank_patches)[0] for _ in range(2))
    assert n == words.size
    print(f"plan: reference {ref_s / n * 1e9:.1f} ns, now {new_s / n * 1e9:.1f} ns per changed element ({n})")
    assert new_s * 4 < ref_s, (new_s, ref_s)
    assert new_s / n * 1e9 < 60, f"{new_s / n * 1e9:.1f} ns per changed element"


def test_rank_patches_matches_the_reference_on_unsorted_indexes(tmp_path):
    tensors, maps = _synthetic(2)
    release = _write_release(tmp_path / "release" / "model.safetensors", tensors)
    slots = su.release_slots(release.parent)
    rng = np.random.default_rng(2)
    changes = {name: rng.permutation(slots[name].numel)[: max(1, slots[name].numel // 2)].astype(np.int64)
               for name in ("embed", "stacked", "perm", "bytes")}
    ref_rank_patches(tmp_path / "ref", maps, slots, changes, {}, {})
    su.rank_patches(tmp_path / "new", maps, slots, changes, {}, {})
    for path in sorted((tmp_path / "ref").rglob("*.*")):
        assert path.read_bytes() == (tmp_path / "new" / path.relative_to(tmp_path / "ref")).read_bytes(), path
