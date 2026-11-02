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
  loader transformed any value stays unmapped as a whole. The real parameters
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
            found = _probe_group(model, [params[n] for n in swapped], swapped, names, shapes, device,
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
    return {"mapped": len(mapping["tensors"]), "unmapped": len(mapping["unmapped"])}


def _numel(shape: Sequence[int]) -> int:
    return int(np.prod(shape, dtype=np.int64)) if len(shape) else 1


def _probe_group(model: Any, tensors: list[Any], param_names: list[str], names: list[str],
                 shapes: dict[str, tuple[int, ...]], device: Any, batch_elements: int) -> dict[str, list[Any]]:
    import torch

    offsets, total = [], 0
    for name in names:
        offsets.append(total)
        total += _numel(shapes[name])
    if 2 * total + 1 >= 1 << 53:
        raise ValueError(f"probe group {group_of(names[0])!r} has too many elements for float64")
    originals = [t.data for t in tensors]
    found: dict[str, list[Any]] = {}
    bad: set[str] = set()

    def run(shift: int) -> list[tuple[Any, Any]]:
        """Load the probes (values ``shift + offset + index + 1``); per parameter
        the nonzero positions and their values."""
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
