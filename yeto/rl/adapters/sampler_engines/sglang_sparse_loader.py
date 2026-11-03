"""The engine-side half of sparse weight updates (:mod:`.sparse_update`): a
custom weight loader that SGLang runs on every tensor-parallel rank.

SGLang calls ``load(model, named_tensors)`` for ``POST
/update_weights_from_tensor`` with ``load_format`` naming this function, when
the server was started with ``--custom-weight-loader`` naming it
(``ModelRunner.update_weights_from_tensor``; SGLang deserializes each rank's
payload with its restricted unpickler, so the payload is plain tensors). The
island sends one command per rank in the tensor *name*, ``yeto-sparse.<base64 json>``
(urlsafe base64, as SGLang treats a name with ``:`` as a LoRA tensor; the tensor itself is a placeholder), so no weight data passes over HTTP:

* ``prepare``: the probe. For each group of checkpoint tensors (one decoder
  layer, or everything outside the layers) the group's parameters are swapped
  for float64 zeros on ``probe_device`` and the model's own ``load_weights``
  loads probe tensors whose element values are ``offset + flat index + 1``
  (unique within the group); every nonzero parameter element then names the
  checkpoint element the loader put there, and the placement is fitted as a
  separable index map (:mod:`.sparse_update` module doc). A second pass with
  every value shifted must move every element by the shift; a group where the
  loader transformed any value stays unmapped as a whole. The map is fitted
  from one seed element per tensor and the parameter lines through it, then
  verified against every parameter element in both passes (cost: a few
  passes over the memory, not a sort per parameter); a group that fit cannot
  explain is probed exactly, element by element (``exact_groups`` of the
  result), so the map is the same either way. ``"probe": "exact"`` forces
  the exact probe. The real parameters
  are swapped back afterwards, but ``load_weights`` may have side effects
  (derived buffers, tied weights); the island reloads the served release from
  disk after a probe. Writes the element map to ``out``;
* ``apply``: writes a rank patch into the parameters (element bits, by flat
  position), then compares the check samples;
* ``check``: only the comparison.

SGLang ignores the loader's return value and a raised exception would end the
scheduler, so every command catches its errors and reports through the JSON
file ``result``; the island reads the files after the call.

Imports torch only (and numpy); SGLang is not imported.
"""

from __future__ import annotations

import base64
import json
import pickle
import re
import time
import traceback
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

# no ":" in the name: the pinned SGLang fork routes a streamed tensor whose name
# holds ":" (``{lora_name}:...``) to the LoRA stash, not to the loader
PREFIX = "yeto-sparse."
LOADER = f"{__name__}.load"  # the --custom-weight-loader / load_format value
_LAYER = re.compile(r"^(.*?\blayers\.\d+\.)")
_SIGNED = {1: "u1", 2: "<i2", 4: "<i4", 8: "<i8"}


def command_name(command: dict[str, Any]) -> str:
    """The tensor name carrying ``command``: :data:`PREFIX` + urlsafe base64 of its JSON."""
    return PREFIX + base64.urlsafe_b64encode(json.dumps(command, separators=(",", ":")).encode()).decode("ascii")


def parse_command(name: str) -> dict[str, Any] | None:
    """The command of a :func:`command_name` tensor name, or None for any other name."""
    if not str(name).startswith(PREFIX):
        return None
    return json.loads(base64.urlsafe_b64decode(str(name)[len(PREFIX):]))


def group_of(name: str) -> str:
    """The probe group of a parameter or checkpoint tensor name (module doc)."""
    match = _LAYER.match(name)
    return match.group(1) if match else ""


def load(model: Any, named_tensors: Iterable[tuple[str, Any]]) -> None:
    for name, _tensor in named_tensors:
        command = parse_command(name)
        if command is None:
            continue
        began = time.monotonic()
        try:
            result = _OPS[command["op"]](model, command)
            result = {"ok": True, **result}
        except Exception as error:  # noqa: BLE001 - reported to the island, never raised into SGLang
            result = {"ok": False, "error": f"{type(error).__name__}: {error}",
                      "trace": traceback.format_exc()[-2000:]}
        result["seconds"] = round(time.monotonic() - began, 4)
        Path(command["result"]).write_text(json.dumps(result), encoding="utf-8")


# ---------------------------------------------------------------- probe
def _infer(pos: Any, h: Any, shape: Sequence[int], param_shape: Sequence[int]) -> dict[str, Any] | None:
    """The separable index map that sends checkpoint elements ``h`` to parameter
    positions ``pos`` (both flat, int64 torch tensors), or None."""
    import torch

    def unravel(flat: Any, dims: Sequence[int]) -> list[Any]:
        out = []
        for size in reversed(list(dims)):
            out.append(flat % int(size))
            flat = flat // int(size)
        return list(reversed(out))

    pc, hc = unravel(pos, param_shape), unravel(h, shape)
    dims: list[list[Any]] = []
    maps: dict[int, Any] = {}
    for k in range(len(param_shape)):
        column = pc[k]
        if bool((column == column[0]).all()):  # e.g. the expert index of a stacked parameter
            dims.append(["const", int(column[0])])
            continue
        found = None
        for d in range(len(shape)):
            if d in maps:
                continue
            if _is_function(hc[d], column, int(shape[d])):
                found = d
                break
        if found is None:
            return None
        table = torch.full((int(shape[found]),), -1, dtype=torch.int64)
        table[hc[found]] = column
        maps[found] = table
        dims.append(["hf", found])
    select: dict[int, int] = {}
    expected = 1
    for d in range(len(shape)):
        if d in maps:
            expected *= int((maps[d] >= 0).sum())
        elif bool((hc[d] != hc[d][0]).any()):
            return None  # two checkpoint elements on one position
        elif int(shape[d]) > 1:
            select[d] = int(hc[d][0])
    if expected != int(pos.numel()):
        return None
    return {"dims": dims, "maps": {d: t.numpy() for d, t in maps.items()}, "select": select}


def _is_function(source: Any, image: Any, size: int) -> bool:
    """``image`` is a function of ``source`` (one image value per source value)."""
    import torch

    table = torch.full((size,), -1, dtype=torch.int64)
    table[source] = image
    return bool(torch.equal(table[source], image))


def _prepare(model: Any, command: dict[str, Any]) -> dict[str, Any]:
    import torch

    shapes = {str(n): tuple(int(d) for d in s) for n, s in command["tensors"].items()}
    device = torch.device(str(command.get("probe_device") or "cpu"))
    batch_elements = int(command.get("batch_elements") or (1 << 28))
    params = dict(model.named_parameters())
    mapping: dict[str, Any] = {
        "params": {n: {"shape": list(p.shape), "itemsize": int(p.element_size()), "dtype": str(p.dtype)}
                   for n, p in params.items()},
        "tensors": {}, "unmapped": []}
    exact_groups: list[str] = []
    if command.get("probe") == "exact":
        probe = _probe_group_exact
    else:
        def probe(*args: Any) -> dict[str, list[Any]]:
            return _probe_group(*args, exact_groups=exact_groups)
    groups: dict[str, list[str]] = {}
    for name in shapes:
        groups.setdefault(group_of(name), []).append(name)
    # The analysis runs on the CPU over every element of a group, and the SGLang
    # scheduler runs torch with one CPU thread: use the cores this rank may claim.
    threads = torch.get_num_threads()
    torch.set_num_threads(max(threads, int(command.get("threads") or 1)))
    try:
        for key, names in sorted(groups.items()):
            swapped = [n for n in params if group_of(n) == key]
            found = probe(model, [params[n] for n in swapped], swapped, names, shapes, device,
                          batch_elements) if swapped else {}
            for name in names:
                targets = found.get(name)
                if targets is not None:
                    mapping["tensors"][name] = targets
                else:
                    mapping["unmapped"].append(name)
    finally:
        torch.set_num_threads(threads)
    out = Path(command["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as handle:
        pickle.dump(mapping, handle)
    return {"mapped": len(mapping["tensors"]), "unmapped": len(mapping["unmapped"]),
            "exact_groups": exact_groups}


def _numel(shape: Sequence[int]) -> int:
    return int(np.prod(shape, dtype=np.int64)) if len(shape) else 1


def _offsets(names: list[str], shapes: dict[str, tuple[int, ...]]) -> tuple[list[int], int]:
    """Probe value offset of every tensor of a group, and the group's element count."""
    offsets, total = [], 0
    for name in names:
        offsets.append(total)
        total += _numel(shapes[name])
    if 2 * total + 1 >= 1 << 53:
        raise ValueError(f"probe group {group_of(names[0])!r} has too many elements for float64")
    return offsets, total


def _load_probes(model: Any, tensors: list[Any], names: list[str], shapes: dict[str, tuple[int, ...]],
                 offsets: list[int], shift: int, device: Any, batch_elements: int) -> None:
    """Swap ``tensors`` for float64 zeros and load the probes, element values
    ``shift + offset + flat index + 1``."""
    import torch

    for t in tensors:
        t.data = torch.zeros(t.shape, dtype=torch.float64, device=device)
    batch: list[tuple[str, Any]] = []
    size = 0
    with torch.no_grad():
        for name, offset in zip(names, offsets):
            n = _numel(shapes[name])
            first = float(shift + offset + 1)
            batch.append((name, (torch.arange(n, dtype=torch.float64, device=device) + first).reshape(shapes[name])))
            size += n
            if size >= batch_elements:
                model.load_weights(batch)
                batch, size = [], 0
        if batch:
            model.load_weights(batch)


# The fast probe. A separable index map places a checkpoint tensor on a
# Cartesian product of parameter coordinates (one coordinate list per
# parameter dimension), so one probe element of a tensor (a seed) and the
# parameter lines through it along every dimension give the whole map; the
# map is then verified against every element of the parameter (the regions
# must hold exactly the predicted probe values and nothing else may be
# nonzero), in both passes. Anything else (a transformed value, a tensor
# split over two dimensions, a seed not found) sends the group to the exact
# probe, which also decides what stays unmapped: the element map is the same.
_CHUNK = 1 << 24


class _NotSeparable(Exception):
    """The fast probe cannot explain a parameter: run the exact probe."""


class _Layout:
    """Decodes probe values of one group into (tensor, flat element index)."""

    def __init__(self, names: list[str], shapes: dict[str, tuple[int, ...]], offsets: list[int]) -> None:
        self.shapes = [tuple(shapes[n]) for n in names]
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.sizes = np.asarray([_numel(s) for s in self.shapes], dtype=np.int64)
        self.strides = [np.asarray([_numel(s[d + 1:]) for d in range(len(s))], dtype=np.int64)
                        for s in self.shapes]

    def decode(self, values: np.ndarray, shift: int) -> tuple[np.ndarray, np.ndarray]:
        """(tensor, element) of nonzero probe values; raises for a value no probe holds."""
        x = values - float(shift + 1)
        if not bool(np.all(x == np.floor(x))):
            raise _NotSeparable("a value that is not a probe value")
        x = x.astype(np.int64)
        which = np.searchsorted(self.offsets, x, side="right") - 1
        if bool(np.any(which < 0)):
            raise _NotSeparable("a value below every probe")
        h = x - self.offsets[which]
        if bool(np.any(h >= self.sizes[which])):
            raise _NotSeparable("a value beyond its tensor")
        return which, h

    def unravel(self, i: int, h: np.ndarray) -> np.ndarray:
        """Coordinates (rows) of flat elements ``h`` of tensor ``i``."""
        shape = self.shapes[i]
        if not shape:
            return np.zeros((h.size, 0), dtype=np.int64)
        return np.stack(np.unravel_index(h, shape), axis=1).astype(np.int64)


class _Region:
    """Where one tensor lands in one parameter (a product of coordinate lists)."""

    def __init__(self, tensor: int, seed: tuple[int, ...], seed_hc: np.ndarray) -> None:
        self.tensor, self.seed, self.seed_hc = tensor, seed, seed_hc
        self.coords: list[np.ndarray] = []  # per parameter dimension
        self.source: list[int | None] = []  # checkpoint dimension per parameter dimension (None: const)
        self.hc: list[np.ndarray | None] = []  # checkpoint coordinate at each coordinate of coords

    @property
    def count(self) -> int:
        return int(np.prod([c.size for c in self.coords], dtype=np.int64)) if self.coords else 1

    def target(self, layout: _Layout) -> dict[str, Any]:
        """The element map of :func:`_infer` for this region."""
        shape = layout.shapes[self.tensor]
        dims: list[list[Any]] = []
        maps: dict[int, np.ndarray] = {}
        for k, d in enumerate(self.source):
            if d is None:
                dims.append(["const", int(self.seed[k])])
            else:
                table = np.full(int(shape[d]), -1, dtype=np.int64)
                table[self.hc[k]] = self.coords[k]
                maps[d] = table
                dims.append(["hf", d])
        select = {d: int(self.seed_hc[d]) for d in range(len(shape)) if d not in maps and int(shape[d]) > 1}
        return {"dims": dims, "maps": maps, "select": select}


def _line(data: Any, seed: tuple[int, ...], k: int) -> np.ndarray:
    index = tuple(slice(None) if j == k else c for j, c in enumerate(seed))
    return data[index].to("cpu").numpy()


def _region(data: Any, layout: _Layout, i: int, seed_flat: int, seed_h: int) -> _Region:
    shape = data.shape
    seed = tuple(int(c) for c in np.unravel_index(seed_flat, tuple(shape))) if len(shape) else ()
    hc_seed = layout.unravel(i, np.asarray([seed_h], dtype=np.int64))[0]
    region = _Region(i, seed, hc_seed)
    used: set[int] = set()
    for k in range(len(shape)):
        line = _line(data, seed, k)
        nz = np.flatnonzero(line)
        which, h = layout.decode(line[nz], 0)
        js = nz[which == i]
        hcs = layout.unravel(i, h[which == i])
        varying = [d for d in range(hcs.shape[1]) if bool(np.any(hcs[:, d] != hcs[0, d]))]
        if js.size == 1:
            region.coords.append(js)
            region.source.append(None)
            region.hc.append(None)
            continue
        if len(varying) != 1 or varying[0] in used:
            raise _NotSeparable("a parameter line that is not one checkpoint dimension")
        d = varying[0]
        if np.unique(hcs[:, d]).size != js.size:
            raise _NotSeparable("two positions of a line hold one element")
        used.add(d)
        region.coords.append(js)
        region.source.append(d)
        region.hc.append(hcs[:, d])
    return region


def _matches(data: Any, layout: _Layout, region: _Region, shift: int) -> bool:
    """The region of ``data`` holds exactly the probe values of its tensor."""
    import torch

    i = region.tensor
    strides = layout.strides[i]
    base = float(shift + 1 + int(layout.offsets[i]))
    for d in range(len(layout.shapes[i])):
        if d not in region.source:
            base += float(strides[d] * region.seed_hc[d])
    if not region.coords:
        return float(data.item()) == base
    rank = len(region.coords)
    sizes = [c.size for c in region.coords]
    axis = int(np.argmax(sizes))
    rest = max(1, region.count // sizes[axis])
    step = max(1, _CHUNK // rest)
    device = data.device
    for start in range(0, sizes[axis], step):
        index, expected = [], torch.full((1,) * rank, base, dtype=torch.float64, device=device)
        for k in range(rank):
            coords, hc = region.coords[k], region.hc[k]
            if k == axis:
                coords = coords[start:start + step]
                hc = hc[start:start + step] if hc is not None else None
            view = [1] * rank
            view[k] = coords.size
            index.append(coords)
            if hc is not None:
                d = region.source[k]
                expected = expected + torch.from_numpy(hc * strides[d]).to(device=device, dtype=torch.float64).view(view)
        if all(c[-1] - c[0] + 1 == c.size for c in index):  # a box: compare a view
            got = data[tuple(slice(int(c[0]), int(c[-1]) + 1) for c in index)]
        else:
            got = data[tuple(torch.from_numpy(c).to(device).view([-1 if j == k else 1 for j in range(rank)])
                             for k, c in enumerate(index))]
        if not torch.equal(got, expected.expand(got.shape)):
            return False
    return True


def _seeds(flat: Any, layout: _Layout, row: int, full: bool) -> dict[int, tuple[int, int]]:
    """{tensor: (flat position, element)} for every tensor seen: in the first
    and last element of every row (``full``: every element)."""
    import torch

    seeds: dict[int, tuple[int, int]] = {}
    n = int(flat.numel())

    def take(pos: np.ndarray, values: np.ndarray) -> None:
        nz = np.flatnonzero(values)
        if not nz.size:
            return
        which, h = layout.decode(values[nz], 0)
        starts = np.flatnonzero(np.r_[True, which[1:] != which[:-1]])
        for s in starts:
            seeds.setdefault(int(which[s]), (int(pos[nz[s]]), int(h[s])))

    if not full:
        row = max(1, row)
        firsts = np.arange(0, n, row, dtype=np.int64)
        pos = np.stack([firsts, firsts + row - 1], axis=1).reshape(-1) if row > 1 else firsts
        take(pos, flat[torch.from_numpy(pos).to(flat.device)].to("cpu").numpy())
        return seeds
    for start in range(0, n, _CHUNK):
        chunk = flat[start:start + _CHUNK]
        nz = torch.nonzero(chunk).flatten()
        if nz.numel():
            take((nz + start).to("cpu").numpy(), chunk[nz].to("cpu").numpy())
    return seeds


def _fit(data: Any, layout: _Layout) -> list[_Region]:
    """The regions of every tensor in one parameter (probe pass 1), verified."""
    import torch

    flat = data.reshape(-1)
    nonzero = int(torch.count_nonzero(flat))
    if not nonzero:
        return []
    row = int(data.shape[-1]) if data.dim() else 1
    for full in (False, True):
        regions = [_region(data, layout, i, p, h) for i, (p, h) in sorted(_seeds(flat, layout, row, full).items())]
        if sum(r.count for r in regions) == nonzero and all(_matches(data, layout, r, 0) for r in regions):
            return regions
    raise _NotSeparable("a parameter the regions do not explain")


def _probe_group(model: Any, tensors: list[Any], param_names: list[str], names: list[str],
                 shapes: dict[str, tuple[int, ...]], device: Any, batch_elements: int,
                 exact_groups: list[str] | None = None) -> dict[str, list[Any]]:
    """The fast probe (comment above); the exact probe when it cannot explain the
    group (its key is added to ``exact_groups``)."""
    import torch

    offsets, total = _offsets(names, shapes)
    layout = _Layout(names, shapes, offsets)
    originals = [t.data for t in tensors]
    try:
        try:
            _load_probes(model, tensors, names, shapes, offsets, 0, device, batch_elements)
            fitted = [_fit(t.data, layout) for t in tensors]
            _load_probes(model, tensors, names, shapes, offsets, total, device, batch_elements)
            for t, regions in zip(tensors, fitted):
                if int(torch.count_nonzero(t.data)) != sum(r.count for r in regions) or \
                        not all(_matches(t.data, layout, r, total) for r in regions):
                    raise _NotSeparable("the second pass moved a value by other than the shift")
        except _NotSeparable:
            fitted = None
    finally:
        for t, data in zip(tensors, originals):
            t.data = data
    if fitted is None:
        if exact_groups is not None:
            exact_groups.append(group_of(names[0]))
        return _probe_group_exact(model, tensors, param_names, names, shapes, device, batch_elements)
    found: dict[str, list[Any]] = {name: [] for name in names}
    for param_name, regions in zip(param_names, fitted):
        for region in regions:
            found[names[region.tensor]].append({"param": param_name, **region.target(layout)})
    return found


def _probe_group_exact(model: Any, tensors: list[Any], param_names: list[str], names: list[str],
                 shapes: dict[str, tuple[int, ...]], device: Any, batch_elements: int) -> dict[str, list[Any]]:
    import torch

    offsets, total = _offsets(names, shapes)
    originals = [t.data for t in tensors]
    found: dict[str, list[Any]] = {}
    bad: set[str] = set()

    def run(shift: int) -> list[tuple[Any, Any]]:
        """Load the probes; per parameter the nonzero positions and their values."""
        _load_probes(model, tensors, names, shapes, offsets, shift, device, batch_elements)
        out = []
        for t in tensors:
            flat = t.data.reshape(-1).to("cpu")
            pos = torch.nonzero(flat).flatten()
            out.append((pos, flat[pos]))
        return out

    try:
        # Two passes with shifted values: a loader that copies moves every value by
        # the shift; one that transforms values (scales, adds) does not.
        first_pass, second_pass = run(0), run(total)
    finally:
        for t, data in zip(tensors, originals):
            t.data = data
    starts = torch.tensor(offsets, dtype=torch.float64)
    sizes = torch.tensor([_numel(shapes[n]) for n in names], dtype=torch.float64)
    any_tainted = False
    for param_name, t, (pos, value), (pos2, value2) in zip(param_names, tensors, first_pass, second_pass):
        if not pos.numel() and not pos2.numel():
            continue
        value = value - 1.0
        which = torch.searchsorted(starts, value, right=True) - 1
        h = value - starts[which.clamp(min=0)]
        valid = (which >= 0) & (h == torch.floor(h)) & (h < sizes[which.clamp(min=0)])
        if torch.equal(pos, pos2):
            valid &= (value2 - 1.0 - value) == float(total)
        else:
            valid &= False
        if not bool(valid.all()):  # values not copied: their source cannot be told
            any_tainted = True
            break
        # one pass per parameter, not one per tensor: a stacked-expert parameter
        # holds hundreds of checkpoint tensors and a boolean mask per tensor over
        # all its elements made the probe quadratic in the expert count
        order = torch.argsort(which, stable=True)
        which, pos, h = which[order], pos[order], h[order]
        counts = torch.bincount(which, minlength=len(names)).tolist()
        ends = np.cumsum(counts).tolist()
        for i, count in enumerate(counts):
            if not count:
                continue
            name = names[i]
            sel = slice(int(ends[i]) - int(count), int(ends[i]))
            target = _infer(pos[sel], h[sel].to(torch.int64), shapes[name], list(t.shape))
            if target is None:
                bad.add(name)
                continue
            found.setdefault(name, []).append({"param": param_name, **target})
    if any_tainted:
        # a value the loader transformed has no known source, and that source may
        # also be copied elsewhere: no tensor of the group is patched by element
        return {}
    for name in bad:
        found.pop(name, None)
    for name in names:  # loaded nowhere on this rank (e.g. another rank's expert): nothing to write
        if name not in bad:
            found.setdefault(name, [])
    return found


# ---------------------------------------------------------------- apply / check
def _records(directory: Path, prefix: str, records: list[list[Any]], params: dict[str, Any]) -> list[tuple]:
    """(param, positions, values) per record; checks every record before any write."""
    import torch

    pos = np.fromfile(directory / f"{prefix}pos.bin", dtype="<i8")
    raw = (directory / f"{prefix}val.bin").read_bytes()
    out = []
    for name, first, count, itemsize, offset in records:
        param = params.get(name)
        if param is None:
            raise KeyError(f"no parameter {name}")
        if int(param.element_size()) != int(itemsize) or not param.data.is_contiguous():
            raise ValueError(f"{name}: item size {param.element_size()} (patch {itemsize}) or not contiguous")
        index = pos[int(first):int(first) + int(count)]
        if index.size and (index.min() < 0 or index.max() >= param.numel()):
            raise IndexError(f"{name}: position out of range")
        values = np.frombuffer(raw, dtype=_SIGNED[int(itemsize)], count=int(count), offset=int(offset))
        out.append((param, torch.from_numpy(index.copy()), torch.from_numpy(values.copy())))
    return out


def _bits(param: Any) -> Any:
    import torch

    kind = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[int(param.element_size())]
    return param.data.view(-1).view(kind)


def _sync() -> None:
    import torch

    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


def _compare(directory: Path, patch: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    checked, mismatches, examples = 0, 0, []
    for param, index, values in _records(directory, "check_", patch.get("checks") or [], params):
        bits = _bits(param)
        got = bits[index.to(bits.device)].to("cpu")
        wrong = got != values
        checked += int(index.numel())
        mismatches += int(wrong.sum())
        if bool(wrong.any()) and len(examples) < 5:
            name = next(n for n, p in params.items() if p is param)
            examples.append({"param": name, "position": int(index[wrong][0])})
    return {"checked": checked, "mismatches": mismatches, "examples": examples}


def _apply(model: Any, command: dict[str, Any]) -> dict[str, Any]:
    import torch

    directory = Path(command["patch"])
    patch = json.loads((directory / "patch.json").read_text(encoding="utf-8"))
    params = dict(model.named_parameters())
    records = _records(directory, "", patch["params"], params)
    began = time.monotonic()
    elements = 0
    with torch.no_grad():
        for param, index, values in records:
            bits = _bits(param)
            bits[index.to(bits.device)] = values.to(bits.device)
            elements += int(index.numel())
    _sync()
    return {"elements": elements, "write_s": round(time.monotonic() - began, 4),
            **_compare(directory, patch, params)}


def _check(model: Any, command: dict[str, Any]) -> dict[str, Any]:
    directory = Path(command["patch"])
    patch = json.loads((directory / "patch.json").read_text(encoding="utf-8"))
    return _compare(directory, patch, dict(model.named_parameters()))


_OPS = {"prepare": _prepare, "apply": _apply, "check": _check}
