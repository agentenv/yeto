"""Checkpoint tensor names <-> the HF model's own state names (S27 G-EZ2, G-EQ3).

``transformers`` does not load a checkpoint by matching tensor names to the
model's ``named_parameters()``.  Between the two sits a registered conversion
table (``transformers.conversion_mapping.get_model_conversion_mapping``): for
``nemotron_h`` it renames ``backbone.`` to ``model.`` and stacks the 512
per-expert tensors ``mixer.experts.<e>.{up,down}_proj.weight`` of a MoE layer
into one ``mixer.experts.{up,down}_proj`` tensor.  A name-for-name comparison
therefore reports a correct checkpoint as unloadable (S27 G-EZ, EZ-7.5..7.9),
and an export that writes the model's own names does not reproduce the base
checkpoint's layout.

:func:`checkpoint_key_map` runs the loader's own key loop
(``core_model_loading.convert_and_load_state_dict_in_model``: the same
renamings, converters, base-model prefix rule and natural sort order) over a
list of checkpoint names, without reading any tensor.  It gives, for each
checkpoint name, the model state name it loads into and, for a stacked
tensor, its position along dim 0.  The pre-step correctness check and the
verl HF export (:mod:`.hf_export`) both use it, so the check tests the names
the export writes.

Only the two conversions this family needs are supported: a rename, and a
``MergeModulelist(dim=0)`` with one source pattern.  Anything else raises
:class:`CheckpointKeyError` instead of guessing a reverse.

``state_names`` are the model's ``state_dict`` names: parameters *and*
persistent buffers.  A checkpoint tensor can load into a buffer
(``mixer.gate.e_score_correction_bias`` is one: an f32 router bias that the
loader fills and no optimizer updates), so a check that counts only
parameters is short by exactly those tensors.

The reverse (S27 G-EQ3) is the same map read backwards: a checkpoint tensor
is its model tensor (renamed) or row ``index`` of a stacked one
(:func:`checkpoint_tensor`).  :func:`base_export` plans an export that writes
the *base checkpoint's own* files -- the same tensor names, the same per-file
grouping, the same dtypes -- from the model's tensors, so an unchanged model
exports byte-identical files and a ``release_delta`` against the base stays
small (the verl writer, :mod:`.hf_export`).  It reads the base's
``model.safetensors.index.json`` (or its single ``model.safetensors``) and
the safetensors headers, never a tensor.  The conversion rules come from a
meta-device build of the model (:func:`model_conversion`), so nothing here
names a model family.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class CheckpointKeyError(ValueError):
    """The checkpoint names cannot be mapped to the model's state names."""


@dataclass(frozen=True)
class CheckpointKey:
    """Where one checkpoint tensor goes: ``model_name`` (a ``state_dict`` name)
    and, for a stacked model tensor, ``index`` along dim 0 (None: the whole tensor)."""

    model_name: str
    index: int | None = None


@dataclass(frozen=True)
class KeyMap:
    """:func:`checkpoint_key_map`'s result."""

    keys: dict[str, CheckpointKey]  # checkpoint name -> where it loads
    unexpected: tuple[str, ...]  # checkpoint names the model has no state for
    rules: tuple[str, ...]  # the conversions applied (repr), for the report

    def targets(self) -> dict[str, tuple[str, ...]]:
        """model state name -> its checkpoint names, in stacking order."""
        out: dict[str, list[tuple[int, str]]] = {}
        for name, key in self.keys.items():
            out.setdefault(key.model_name, []).append((-1 if key.index is None else key.index, name))
        return {model: tuple(name for _i, name in sorted(items)) for model, items in out.items()}


def natural_key(name: str) -> list[tuple[int, Any]]:
    """``transformers.core_model_loading.dot_natural_key``: digits sort as numbers."""
    return [(0, int(part)) if part.isdigit() else (1, part) for part in name.split(".")]


def conversion_rules(model: Any) -> list[Any]:
    """The conversions ``from_pretrained`` applies to ``model`` (no key mapping,
    no quantizer), from ``transformers`` itself."""
    from transformers.conversion_mapping import get_model_conversion_mapping

    return list(get_model_conversion_mapping(model, None, None))


def checkpoint_key_map(names: Iterable[str], state_names: Iterable[str], *, rules: Sequence[Any],
                       prefix: str | None) -> KeyMap:
    """Map checkpoint ``names`` onto the model's ``state_names`` with ``rules``
    (:func:`conversion_rules`) and the model's ``base_model_prefix``, the way
    the HF loader does (module doc).  Raises :class:`CheckpointKeyError` on a
    conversion it cannot reverse or on a stacked tensor whose sources are not
    one complete run of indices."""
    from transformers.core_model_loading import WeightConverter, WeightRenaming, rename_source_key

    renamings = [rule for rule in rules if isinstance(rule, WeightRenaming)]
    converters = [rule for rule in rules if isinstance(rule, WeightConverter)]
    by_pattern = {pattern: converter for converter in converters for pattern in converter.source_patterns}
    state = {name: True for name in state_names}
    renamed: dict[str, tuple[str, str | None]] = {}
    unexpected = []
    for name in sorted(names, key=natural_key):  # the loader's order
        target, pattern = rename_source_key(name, renamings, converters, prefix, state)
        if target not in state and name in state:
            target, pattern = rename_source_key(name, [], [], prefix, state)
        if target not in state:
            unexpected.append(name)
            continue
        renamed[name] = (target, pattern)
    keys: dict[str, CheckpointKey] = {}
    stacked: dict[str, list[str]] = {}
    for name, (target, pattern) in renamed.items():
        if pattern is None:
            keys[name] = CheckpointKey(target)
        else:
            stacked.setdefault(target, []).append(name)
    for pattern in {p for _t, p in renamed.values() if p is not None}:
        _check_supported(by_pattern[pattern])  # only the conversions that match a name
    for target, sources in stacked.items():  # torch.stack in the loader's (natural) order
        for index, name in enumerate(sources):
            keys[name] = CheckpointKey(target, index)
    plain = [t for t, _p in renamed.values() if _p is None]
    if len(set(plain)) != len(plain) or set(plain) & set(stacked):
        raise CheckpointKeyError("two checkpoint tensors load into the same model tensor")
    return KeyMap(keys, tuple(unexpected), tuple(repr(rule) for rule in rules))


def model_key_map(model: Any, names: Iterable[str]) -> KeyMap:
    """:func:`checkpoint_key_map` for a built (or meta) HF ``model``."""
    return checkpoint_key_map(names, model.state_dict().keys(), rules=conversion_rules(model),
                              prefix=getattr(model, "base_model_prefix", None))


def check_stacked_shapes(key_map: KeyMap, checkpoint_shapes: Mapping[str, Sequence[int]],
                         model_shapes: Mapping[str, Sequence[int]]) -> list[str]:
    """Shape problems between the checkpoint and the model, as messages (empty: none).

    A whole tensor must have the checkpoint's shape; a stacked tensor must be
    ``(len(sources), *source_shape)`` with every source the same shape."""
    problems = []
    for model_name, sources in key_map.targets().items():
        have = tuple(int(d) for d in model_shapes[model_name])
        first = key_map.keys[sources[0]]
        if first.index is None:
            want = tuple(int(d) for d in checkpoint_shapes[sources[0]])
        else:
            shapes = {tuple(int(d) for d in checkpoint_shapes[s]) for s in sources}
            if len(shapes) != 1:
                problems.append(f"{model_name}: its {len(sources)} sources have {len(shapes)} shapes")
                continue
            want = (len(sources), *shapes.pop())
        if have != want:
            problems.append(f"{model_name}: model {have}, checkpoint gives {want}")
    return problems


def checkpoint_tensor(model_tensor: Any, key: CheckpointKey) -> Any:
    """The checkpoint tensor ``key`` names, cut out of its model tensor."""
    return model_tensor if key.index is None else model_tensor[key.index]


def _check_supported(converter: Any) -> None:
    from transformers.core_model_loading import MergeModulelist

    operations = list(getattr(converter, "operations", []) or [])
    if (len(operations) != 1 or not isinstance(operations[0], MergeModulelist)
            or int(getattr(operations[0], "dim", -1)) != 0
            or len(converter.source_patterns) != 1 or len(converter.target_patterns) != 1):
        raise CheckpointKeyError(
            f"unsupported checkpoint conversion {converter!r}: only renames and a one-pattern "
            "MergeModulelist(dim=0) have a known reverse here")


# ----------------------------------------------------------------------------- export in the base's layout
INDEX_FILE = "model.safetensors.index.json"
SINGLE_FILE = "model.safetensors"
# safetensors dtype tags an export can cast an FP32 master to
FLOAT_TAGS = {"BF16": "bfloat16", "F16": "float16", "F32": "float32", "F64": "float64"}


def model_conversion(model_path: str | Path) -> tuple[list[Any], str | None]:
    """``(rules, base_model_prefix)`` that ``from_pretrained`` uses for the
    checkpoint at ``model_path``: the model is built on the meta device from its
    ``config.json`` (no weights read), with the class its ``architectures`` names
    when ``transformers`` has it, else ``AutoModelForCausalLM``.  Native classes
    only: a ``trust_remote_code`` class gets no registered conversion from the
    loader either, so its checkpoint names are its state names (the identity
    case of :func:`base_export`)."""
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(str(model_path))
    architecture = (getattr(config, "architectures", None) or [None])[0]
    cls = getattr(transformers, architecture, None) if architecture else None
    with torch.device("meta"):
        model = cls._from_config(config) if cls is not None else AutoModelForCausalLM.from_config(config)
    return conversion_rules(model), getattr(model, "base_model_prefix", None)


def read_safetensors_header(path: str | Path) -> tuple[bytes, dict[str, Any]]:
    """The raw header bytes of a safetensors file and the parsed header."""
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        raw = handle.read(length)
    if len(raw) != length:
        raise CheckpointKeyError(f"{path}: truncated safetensors header")
    return raw, json.loads(raw)


@dataclass(frozen=True)
class BaseExport:
    """An export that rewrites the base checkpoint's own files (:func:`base_export`).

    ``files``: the base's weight files, sorted; ``names``: each file's
    checkpoint tensor names, in the file's data order; ``key_map``: where each
    checkpoint tensor comes from; ``dtypes``: each checkpoint tensor's
    safetensors dtype tag; ``headers``: each file's raw header bytes (the
    export writes them verbatim, :func:`.hf_export.write_like`)."""

    model_path: Path
    files: tuple[str, ...]
    names: Mapping[str, tuple[str, ...]]
    key_map: KeyMap
    dtypes: Mapping[str, str]
    headers: Mapping[str, bytes] = field(repr=False)
    index: bool  # the base has model.safetensors.index.json (the export copies it)

    def model_names(self, file_name: str) -> tuple[str, ...]:
        """The model tensors one file needs, each once, in first-use order."""
        return tuple(dict.fromkeys(self.key_map.keys[name].model_name for name in self.names[file_name]))

    def last_use(self) -> dict[str, int]:
        """model tensor name -> index in ``files`` of the last file that needs it."""
        out: dict[str, int] = {}
        for position, file_name in enumerate(self.files):
            for model_name in self.model_names(file_name):
                out[model_name] = position
        return out


def base_export(model_path: str | Path, state_shapes: Mapping[str, Sequence[int]], *,
                conversion: Any = model_conversion) -> tuple[BaseExport | None, str]:
    """Plan an export of ``state_shapes`` (the model's parameters and persistent
    buffers, by state name) into the base checkpoint's own files; ``(plan, why)``
    with ``plan`` None and ``why`` the reason when the base's layout cannot be
    reproduced.  The names match as-is (identity), else through the loader's
    conversion rules (``conversion(model_path) -> (rules, prefix)``)."""
    model_path = Path(model_path)
    index = model_path / INDEX_FILE
    if index.is_file():
        try:
            weight_map = {str(k): str(v) for k, v in json.loads(index.read_text())["weight_map"].items()}
        except (ValueError, KeyError, AttributeError):
            return None, f"{INDEX_FILE} has no usable weight_map"
    elif (model_path / SINGLE_FILE).is_file():
        _raw, header = read_safetensors_header(model_path / SINGLE_FILE)
        weight_map = {name: SINGLE_FILE for name in header if name != "__metadata__"}
    else:
        return None, "the base has neither model.safetensors.index.json nor model.safetensors"
    files = tuple(sorted(set(weight_map.values())))
    if not all(name.endswith(".safetensors") for name in files):
        return None, "the base index names a weight file that is not safetensors"
    headers: dict[str, bytes] = {}
    names: dict[str, tuple[str, ...]] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    dtypes: dict[str, str] = {}
    for file_name in files:
        if not (model_path / file_name).is_file():
            return None, f"the base index names {file_name}, which is not there"
        raw, header = read_safetensors_header(model_path / file_name)
        entries = {name: entry for name, entry in header.items() if name != "__metadata__"}
        if sorted(entries) != sorted(name for name, f in weight_map.items() if f == file_name):
            return None, f"{file_name}: its header and the index name different tensors"
        headers[file_name] = raw
        names[file_name] = tuple(sorted(entries, key=lambda name: entries[name]["data_offsets"][0]))
        for name, entry in entries.items():
            shapes[name] = tuple(int(d) for d in entry["shape"])
            dtypes[name] = str(entry["dtype"])
    odd = sorted({tag for tag in dtypes.values() if tag not in FLOAT_TAGS})
    if odd:
        return None, f"the base holds tensors of dtype {odd}; an export writes floating tensors only"
    state = {str(k): tuple(int(d) for d in v) for k, v in state_shapes.items()}
    if sorted(weight_map) == sorted(state):
        key_map = KeyMap({name: CheckpointKey(name) for name in weight_map}, (), ())
    else:
        try:
            rules, prefix = conversion(model_path)
            key_map = checkpoint_key_map(weight_map, state, rules=rules, prefix=prefix)
        except Exception as error:  # noqa: BLE001 -- any failure means: no reproducible layout
            return None, f"no conversion from the base's names to the model's: {type(error).__name__}: {error}"
    if key_map.unexpected:
        return None, (f"{len(key_map.unexpected)} base tensors load into no model tensor, "
                      f"e.g. {list(key_map.unexpected[:3])}")
    missing = sorted(set(state) - set(key_map.targets()))
    if missing:
        return None, f"{len(missing)} model tensors are not in the base, e.g. {missing[:3]}"
    problems = check_stacked_shapes(key_map, shapes, state)
    if problems:
        return None, f"shapes differ: {problems[:3]}"
    return BaseExport(model_path, files, names, key_map, dtypes, headers, index.is_file()), (
        f"{len(weight_map)} base tensors from {len(state)} model tensors in {len(files)} files"
        + ("" if not key_map.rules else f" ({len(key_map.rules)} conversion rules)"))
