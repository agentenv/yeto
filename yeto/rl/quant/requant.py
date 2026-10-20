"""BF16 checkpoint -> quantized checkpoint in the layout of a reference artefact (S25 N8.1, N8.5).

The target layout is taken from the reference checkpoint, not hard-coded:
:class:`CheckpointLayout` records every tensor's name, dtype, shape and output
shard. The ``format`` of its ``quantization_config`` selects the scheme:

* ``int-quantized`` -> :data:`W4A8_INT4_CHANNEL` (N8.1, :mod:`.int4`): a module
  is quantized exactly when the layout has ``<module>.weight`` as ``I8`` with a
  ``<module>.weight_scale`` next to it.
* ``mxfp4-pack-quantized`` -> :data:`MXFP4_G32_RTN` (N8.5, :mod:`.mxfp4`): a
  module is quantized exactly when the layout has ``<module>.weight_packed`` as
  ``U8`` with a ``<module>.weight_scale``; the BF16 source provides
  ``<module>.weight``. Use it without AWQ smoothing (RTN).

Every other tensor is cast to its layout dtype (the W4A8 artefact changes some
dtypes, e.g. routers BF16 -> F32 and ``e_score_correction_bias`` F32 -> BF16;
the MXFP4 artefact keeps them). AWQ smoothing (:mod:`.awq`) is applied first.
The result therefore has the same file names, tensor names, dtypes and shapes
as the reference, and the same metadata files.

A layout can be saved as a small template directory (``layout.json`` plus the
non-weight files such as ``config.json`` and ``recipe.yaml``), so later rounds
do not need the 300 GB original.

Work split: :func:`requantize_checkpoint` takes a subset of output shards, so
several workers can each write their own shards. :func:`transform_tensor`
works on any row slice of a 2-D weight with the same result rows, which is the
basis for quantizing on the trainer ranks (D7, task N8.2, :mod:`.shards`).
The source may be per-expert (HF checkpoint) or the trainer's stacked native
layout (:mod:`.stacked`, task N8.5b).

Status: W4A8 transform rules checked bit for bit against the original artefact
on layer 30 (router, shared experts, norm, one routed expert); MXFP4 rules
checked bit for bit against ``yetalabs/apex-flash-1-mxfp4-experts@70e86a0`` on
layer 10 routed experts and five non-expert tensors of layers 9-10 (CPU,
2026-10-19). The full-model comparisons (:func:`compare_checkpoints` over every
tensor) are not run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .awq import AwqSmoothing
from .int4 import QMAX, QMIN, quantize_per_channel
from .mxfp4 import GROUP_SIZE, quantize_mxfp4, unpack_fp4
from .schemes import MXFP4_G32_RTN as MXFP4_NAME
from .schemes import W4A8_INT4_CHANNEL as W4A8_NAME
from .stacked import StackedSourceIndex

INDEX_FILE = "model.safetensors.index.json"
LAYOUT_FILE = "layout.json"
LAYOUT_SCHEMA = "yeto.rl.quant.layout/v1"
SCALE_SUFFIX = ".weight_scale"
WEIGHT_SUFFIX = ".weight"
PACKED_SUFFIX = ".weight_packed"

_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I8": torch.int8,
    "I32": torch.int32,
    "I64": torch.int64,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}
_DTYPE_NAMES = {v: k for k, v in _DTYPES.items()}


class LayoutError(ValueError):
    """The source or output does not match the target layout."""


@dataclass(frozen=True)
class TensorSpec:
    dtype: str
    shape: tuple[int, ...]
    shard: str


class QuantScheme:
    """How one compressed-tensors format stores a quantized linear ``<module>``.

    The quantized data lives in ``<module><data_suffix>`` (dtype ``data_dtype``)
    and its scale in ``<module>.weight_scale``; the BF16 source tensor is
    ``<module>.weight``. ``name`` is the scheme tag written into release records
    (N8.3 readers compare it).
    """

    name: str
    format: str
    rtn_only: bool  # True: the scheme refuses AWQ smoothing
    data_suffix: str
    data_dtype: str
    weight_scheme: dict[str, Any]

    def check_weights(self, group: str, weights: Mapping[str, Any]) -> None:
        got = {k: weights.get(k) for k in self.weight_scheme}
        if got != self.weight_scheme:
            raise LayoutError(f"config group {group}: weight scheme {got} is not supported by {self.name}")

    def check_pair(self, module: str, data: TensorSpec, scale: TensorSpec) -> None:
        raise NotImplementedError

    def quantize(self, weight: torch.Tensor, scale: TensorSpec) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def check_values(self, name: str, data: torch.Tensor) -> None:
        """Reject impossible stored values (default: none)."""


class _Int4Channel(QuantScheme):
    name = W4A8_NAME
    rtn_only = False
    format = "int-quantized"
    data_suffix = WEIGHT_SUFFIX
    data_dtype = "I8"
    weight_scheme = {"num_bits": 4, "type": "int", "symmetric": True, "strategy": "channel", "group_size": None, "dynamic": False}

    def check_weights(self, group: str, weights: Mapping[str, Any]) -> None:
        super().check_weights(group, weights)
        if weights.get("observer") not in ("memoryless_minmax", "minmax"):
            raise LayoutError(f"config group {group}: observer {weights.get('observer')!r} is not supported by {self.name}")

    def check_pair(self, module: str, data: TensorSpec, scale: TensorSpec) -> None:
        if scale.shape != (data.shape[0], 1):
            raise LayoutError(f"{module}: scale shape {scale.shape} is not per-channel [{data.shape[0]}, 1]")

    def quantize(self, weight: torch.Tensor, scale: TensorSpec) -> tuple[torch.Tensor, torch.Tensor]:
        return quantize_per_channel(weight, scale_dtype=_DTYPES[scale.dtype])

    def check_values(self, name: str, data: torch.Tensor) -> None:
        if data.min() < QMIN or data.max() > QMAX:
            raise LayoutError(f"{name}: int4 values outside [{QMIN}, {QMAX}]")


class _Mxfp4Group32(QuantScheme):
    name = MXFP4_NAME
    rtn_only = True
    format = "mxfp4-pack-quantized"
    data_suffix = PACKED_SUFFIX
    data_dtype = "U8"
    weight_scheme = {"num_bits": 4, "type": "float", "symmetric": True, "strategy": "group",
                     "group_size": GROUP_SIZE, "dynamic": False, "scale_dtype": "torch.uint8"}

    def check_pair(self, module: str, data: TensorSpec, scale: TensorSpec) -> None:
        rows, cols = data.shape[0], 2 * data.shape[1]
        if scale.dtype != "U8" or scale.shape != (rows, cols // GROUP_SIZE) or cols % GROUP_SIZE:
            raise LayoutError(f"{module}: scale {scale.dtype}{scale.shape} is not E8M0 per {GROUP_SIZE} of [{rows}, {cols}]")

    def quantize(self, weight: torch.Tensor, scale: TensorSpec) -> tuple[torch.Tensor, torch.Tensor]:
        return quantize_mxfp4(weight)


W4A8_INT4_CHANNEL = _Int4Channel()
MXFP4_G32_RTN = _Mxfp4Group32()
SCHEMES = {s.format: s for s in (W4A8_INT4_CHANNEL, MXFP4_G32_RTN)}


@dataclass
class CheckpointLayout:
    """Tensor specs of the target checkpoint, plus the quantization config it declares."""

    tensors: dict[str, TensorSpec]
    quantization_config: dict[str, Any]

    # -- construction ---------------------------------------------------------
    @classmethod
    def from_checkpoint(cls, ckpt_dir: Path) -> "CheckpointLayout":
        """Read the index, every shard header (not the data) and ``config.json``."""
        ckpt_dir = Path(ckpt_dir)
        weight_map = json.loads((ckpt_dir / INDEX_FILE).read_text())["weight_map"]
        tensors: dict[str, TensorSpec] = {}
        for shard in sorted(set(weight_map.values())):
            for name, info in read_header(ckpt_dir / shard).items():
                if weight_map.get(name) != shard:
                    raise LayoutError(f"{shard}: tensor {name} not mapped to this shard in the index")
                tensors[name] = TensorSpec(info["dtype"], tuple(info["shape"]), shard)
        if missing := set(weight_map) - set(tensors):
            raise LayoutError(f"index names {len(missing)} tensors absent from shards, e.g. {sorted(missing)[:3]}")
        config = json.loads((ckpt_dir / "config.json").read_text())
        layout = cls(tensors, config.get("quantization_config") or {})
        layout.validate()
        return layout

    @classmethod
    def load(cls, template_dir: Path) -> "CheckpointLayout":
        """The layout of a template directory (``layout.json``) or, when it has
        none, of a reference checkpoint itself (index + shard headers; stage 1
        passes the round-0 MXFP4 export as the template)."""
        if not (Path(template_dir) / LAYOUT_FILE).is_file() and (Path(template_dir) / INDEX_FILE).is_file():
            return cls.from_checkpoint(Path(template_dir))
        raw = json.loads((Path(template_dir) / LAYOUT_FILE).read_text())
        if raw.get("schema") != LAYOUT_SCHEMA:
            raise LayoutError(f"{template_dir}: unknown layout schema {raw.get('schema')!r}")
        tensors = {n: TensorSpec(t["dtype"], tuple(t["shape"]), t["shard"]) for n, t in raw["tensors"].items()}
        layout = cls(tensors, raw["quantization_config"])
        layout.validate()
        return layout

    def save_template(self, ckpt_dir: Path, template_dir: Path) -> None:
        """Write ``layout.json`` and copy the non-weight files of ``ckpt_dir`` into ``template_dir``."""
        template_dir = Path(template_dir)
        template_dir.mkdir(parents=True, exist_ok=True)
        for f in metadata_files(Path(ckpt_dir)):
            shutil.copy2(f, template_dir / f.name)
        payload = {
            "schema": LAYOUT_SCHEMA,
            "quantization_config": self.quantization_config,
            "tensors": {n: {"dtype": t.dtype, "shape": list(t.shape), "shard": t.shard} for n, t in sorted(self.tensors.items())},
        }
        _write_text_atomic(template_dir / LAYOUT_FILE, json.dumps(payload, sort_keys=True))

    # -- queries --------------------------------------------------------------
    @property
    def scheme(self) -> QuantScheme:
        q = self.quantization_config
        scheme = SCHEMES.get(q.get("format")) if q.get("quant_method") == "compressed-tensors" else None
        if scheme is None:
            raise LayoutError(f"unsupported quantization format: {q.get('quant_method')!r} / {q.get('format')!r}")
        return scheme

    def validate(self) -> None:
        """Check that the quantization config is one the schemes implement, and the data/scale pairs."""
        scheme = self.scheme
        for group, cfg in (self.quantization_config.get("config_groups") or {}).items():
            if cfg.get("format", scheme.format) != scheme.format:
                raise LayoutError(f"config group {group}: format {cfg.get('format')!r} differs from {scheme.format!r}")
            scheme.check_weights(group, cfg.get("weights") or {})
        modules = self.quantized_modules
        for module in modules:
            scheme.check_pair(module, self.tensors[module + scheme.data_suffix], self.tensors[module + SCALE_SUFFIX])
        bare = [n for n, t in self.tensors.items()
                if t.dtype == scheme.data_dtype and n.endswith(scheme.data_suffix) and n[: -len(scheme.data_suffix)] not in modules]
        if bare:
            raise LayoutError(f"{len(bare)} {scheme.data_dtype} weights without a weight_scale, e.g. {bare[:3]}")
        if clash := [m for m in modules if m + WEIGHT_SUFFIX in self.tensors and scheme.data_suffix != WEIGHT_SUFFIX]:
            raise LayoutError(f"{len(clash)} quantized modules also keep a plain weight, e.g. {clash[:3]}")

    @cached_property
    def quantized_modules(self) -> frozenset[str]:
        """Modules ``m`` stored as ``m<data_suffix>`` (scheme data dtype) plus ``m.weight_scale``."""
        scheme = self.scheme
        return frozenset(
            n[: -len(scheme.data_suffix)] for n, t in self.tensors.items()
            if t.dtype == scheme.data_dtype and n.endswith(scheme.data_suffix)
            and n[: -len(scheme.data_suffix)] + SCALE_SUFFIX in self.tensors
        )

    def quantized_weights(self) -> list[str]:
        """Names of the quantized data tensors (``.weight`` for W4A8, ``.weight_packed`` for MXFP4)."""
        return sorted(m + self.scheme.data_suffix for m in self.quantized_modules)

    def source_of(self, name: str) -> str:
        """The source tensor a target tensor is computed from (``<module>.weight`` for data and scale)."""
        module = _quantized_module_of(name, self, (self.scheme.data_suffix, SCALE_SUFFIX))
        return name if module is None else module + WEIGHT_SUFFIX

    def source_names(self) -> set[str]:
        """Tensor names the BF16 source must provide."""
        return {self.source_of(n) for n in self.tensors}

    def shards(self) -> dict[str, list[str]]:
        """Output shard -> its tensor names (sorted); do not modify the result."""
        return self._shards

    @cached_property
    def _shards(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for name, spec in self.tensors.items():
            out[spec.shard].append(name)
        return {s: sorted(ns) for s, ns in sorted(out.items())}


def read_header(path: Path) -> dict[str, dict[str, Any]]:
    """Return the tensor entries of a safetensors header without reading tensor data."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header


def metadata_files(ckpt_dir: Path) -> list[Path]:
    """Non-weight files of a checkpoint directory (config, recipe, tokenizer, ...); the index is rebuilt."""
    return sorted(
        p for p in ckpt_dir.iterdir()
        if p.is_file() and p.suffix != ".safetensors" and p.name not in (INDEX_FILE, LAYOUT_FILE)
    )


def transform_tensor(
    name: str, tensor: torch.Tensor, layout: CheckpointLayout, smoothing: AwqSmoothing
) -> dict[str, torch.Tensor]:
    """Map one source tensor (or a row slice of it) to its target tensors.

    A quantized weight yields ``{<module><data_suffix>: data, <module>.weight_scale:
    scale}``; any other tensor yields ``{name: tensor in the layout dtype}``.
    Shapes are not checked here, so row slices pass; :func:`requantize_checkpoint`
    checks whole tensors.
    """
    scheme = layout.scheme
    module = _quantized_module_of(name, layout, (WEIGHT_SUFFIX,))
    if module is None:
        spec = layout.tensors.get(name)
        if spec is None or layout.source_of(name) != name:
            raise LayoutError(f"{name}: not a source tensor of the target layout")
        return {name: smoothing.apply(name, tensor, _DTYPES[spec.dtype])}
    # The W4A8 original smoothed the expert in FP64 and stored it in the source dtype before quantizing.
    smoothed = smoothing.apply(name, tensor, tensor.dtype)
    data, scale = scheme.quantize(smoothed, layout.tensors[module + SCALE_SUFFIX])
    return {module + scheme.data_suffix: data, module + SCALE_SUFFIX: scale}


def check_smoothing(layout: CheckpointLayout, smoothing: AwqSmoothing) -> None:
    """Refuse AWQ smoothing for a round-to-nearest-only scheme (MXFP4)."""
    if layout.scheme.rtn_only and smoothing.enabled:
        raise LayoutError(f"{layout.scheme.name} is round-to-nearest only; use no AWQ smoothing")


class _SourceReader:
    """Lazy reader over a sharded safetensors source checkpoint.

    A source in the trainer's native stacked layout (``mlp.experts.gate_up_proj``
    ``[E, 2I, H]`` and ``mlp.experts.down_proj`` ``[E, H, I]``, :mod:`.stacked`)
    is read under the per-expert names of the target layout; each per-expert
    matrix is sliced from the stacked tensor on read.
    """

    def __init__(self, source_dir: Path) -> None:
        self.dir = Path(source_dir)
        index = self.dir / INDEX_FILE
        if index.is_file():
            stored: dict[str, str] = json.loads(index.read_text())["weight_map"]
        else:
            stored = {n: p.name for p in sorted(self.dir.glob("*.safetensors")) for n in read_header(p)}
        headers: dict[str, dict[str, Any]] = {}

        def shape_of(name: str) -> list[int]:
            shard = stored[name]
            if shard not in headers:
                headers[shard] = read_header(self.dir / shard)
            return headers[shard][name]["shape"]

        self.stacked = StackedSourceIndex.from_names(stored, shape_of)
        self.stored = stored
        self.weight_map: dict[str, str] = {n: s for n, s in stored.items() if n not in self.stacked.stacked_names}
        for name, view in self.stacked.views.items():
            self.weight_map[name] = stored[view.stored]
        self._open: dict[str, Any] = {}

    def _file(self, shard: str) -> Any:
        if shard not in self._open:
            self._open[shard] = safe_open(str(self.dir / shard), framework="pt", device="cpu")
        return self._open[shard]

    def get(self, name: str) -> torch.Tensor:
        view = self.stacked.views.get(name)
        if view is None:
            return self._file(self.weight_map[name]).get_tensor(name)
        return view.read(self._file(self.stored[view.stored]).get_slice(view.stored))

    def close(self) -> None:
        self._open.clear()


_COMPOSITE_PREFIX, _TEXT_PREFIX = "model.language_model.", "model."


def requantize_checkpoint(
    source_dir: Path,
    template_dir: Path,
    out_dir: Path,
    smoothing: AwqSmoothing,
    shards: Iterable[str] | None = None,
    frozen: tuple[Path, str] | None = None,
) -> dict[str, Any]:
    """Write the target checkpoint (or the given output shards of it) into ``out_dir``.

    ``frozen``: ``(base_dir, pattern)``. A layout tensor that the source lacks and
    whose source name matches the regular expression ``pattern`` is read from the
    BF16 checkpoint ``base_dir`` (weights the trainer does not train and does not
    export, e.g. the vision tower of a text-only Miles run). Every other missing
    tensor still fails; ``stats["frozen_from_base"]`` counts the filled ones.

    ``template_dir`` holds ``layout.json`` and the metadata files
    (:meth:`CheckpointLayout.save_template`). Without ``shards`` the whole
    checkpoint is written, then the index and the metadata files, so a
    complete ``out_dir`` is loadable; with ``shards`` only those files are
    written and :func:`finalize_checkpoint` must run after all workers.
    The source must provide exactly the layout's source tensors.
    """
    layout = CheckpointLayout.load(template_dir)
    check_smoothing(layout, smoothing)
    source = _SourceReader(source_dir)
    expected = layout.source_names()
    missing, extra = expected - set(source.weight_map), set(source.weight_map) - expected
    # Miles' raw glm5_next export names the text model "model.layers..."; the HF
    # composite checkpoint (and so the layout) has "model.language_model.layers..."
    # (Modal run-14). Only an exact one-to-one prefix match is taken as the same tensor.
    alias = {n: n.replace(_COMPOSITE_PREFIX, _TEXT_PREFIX, 1) for n in missing
             if n.startswith(_COMPOSITE_PREFIX) and n.replace(_COMPOSITE_PREFIX, _TEXT_PREFIX, 1) in extra}
    missing, extra = missing - set(alias), extra - set(alias.values())
    base, filled = None, set()
    if frozen is not None and missing:
        matcher = re.compile(frozen[1])
        filled = {n for n in missing if matcher.search(n)}
        if filled:
            base = _SourceReader(Path(frozen[0]))
            if absent := filled - set(base.weight_map):
                raise LayoutError(f"frozen base {frozen[0]} lacks {len(absent)} tensors, e.g. {sorted(absent)[:3]}")
            missing = missing - filled
    if missing or extra:
        raise LayoutError(
            f"source tensors differ from the target layout: {len(missing)} missing (e.g. {sorted(missing)[:3]}), "
            f"{len(extra)} extra (e.g. {sorted(extra)[:3]})"
        )
    all_shards = layout.shards()
    selected = list(all_shards) if shards is None else list(shards)
    if unknown := set(selected) - set(all_shards):
        raise LayoutError(f"unknown output shards {sorted(unknown)}")
    quantized = set(layout.quantized_weights())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"shards": 0, "tensors": 0, "quantized": 0, "frozen_from_base": len(filled),
             "renamed_from_text_prefix": len(alias)}

    def read(name: str) -> torch.Tensor:
        return base.get(name) if name in filled else source.get(alias.get(name, name))

    try:
        for shard in selected:
            produced: dict[str, torch.Tensor] = {}
            # A scale may live in another shard than its data (seen in the W4A8 original);
            # its module is then quantized again in that shard.
            for name in all_shards[shard]:
                if name not in produced:
                    src = layout.source_of(name)
                    produced.update(transform_tensor(src, read(src), layout, smoothing))
            tensors = {n: produced[n] for n in all_shards[shard]}
            write_output_shard(out_dir, shard, tensors, layout)
            stats["shards"] += 1
            stats["quantized"] += sum(n in quantized for n in tensors)
            stats["tensors"] += len(tensors)
    finally:
        source.close()
        if base is not None:
            base.close()
    if shards is None:
        finalize_checkpoint(template_dir, out_dir)
    return stats


def write_output_shard(out_dir: Path, shard: str, tensors: Mapping[str, torch.Tensor], layout: CheckpointLayout) -> None:
    """Check ``tensors`` (exactly the layout's tensors of ``shard``) and write the shard file atomically.

    The single-process converter and the merge of per-rank parts
    (:mod:`.shards`) both write through here, so equal tensors give equal bytes.
    """
    expected = layout.shards().get(shard)
    if expected is None or set(tensors) != set(expected):
        raise LayoutError(f"{shard}: tensors differ from the layout's tensors of this shard")
    tensors = {n: t.contiguous() for n, t in tensors.items()}
    _check_against_layout(tensors, layout)
    tmp = Path(out_dir) / (shard + ".tmp")
    save_file(tensors, str(tmp), metadata={"format": "pt"})
    os.replace(tmp, Path(out_dir) / shard)


def finalize_checkpoint(template_dir: Path, out_dir: Path) -> None:
    """Check that every output shard exists, then write the metadata files and, last, the index."""
    layout = CheckpointLayout.load(template_dir)
    out_dir = Path(out_dir)
    for shard in layout.shards():
        if not (out_dir / shard).is_file():
            raise LayoutError(f"output shard {shard} missing in {out_dir}")
    for f in metadata_files(Path(template_dir)):
        shutil.copy2(f, out_dir / f.name)
    total = sum(_spec_bytes(t) for t in layout.tensors.values())
    index = {"metadata": {"total_size": total}, "weight_map": {n: t.shard for n, t in sorted(layout.tensors.items())}}
    _write_text_atomic(out_dir / INDEX_FILE, json.dumps(index, indent=2, sort_keys=True))


def compare_checkpoints(a_dir: Path, b_dir: Path, names: Iterable[str] | None = None) -> dict[str, Any]:
    """Compare two checkpoints of one layout tensor by tensor (N8.1 and N8.5 acceptance).

    Reports the share of equal int values over all I8 tensors, the largest
    relative error of the float weight scales, the share of equal FP4 codes
    (``U8 .weight_packed``) and of equal E8M0 codes (``U8 .weight_scale``), and
    the other tensors that are not bit-equal. No threshold is applied here.
    """
    ra, rb = _SourceReader(a_dir), _SourceReader(b_dir)
    todo = sorted(names) if names is not None else sorted(ra.weight_map)
    if missing := [n for n in todo if n not in rb.weight_map]:
        raise LayoutError(f"{len(missing)} tensors missing in {b_dir}, e.g. {missing[:3]}")
    int_total = int_equal = fp4_total = fp4_equal = e8m0_total = e8m0_equal = 0
    scale_rel = 0.0
    unequal: list[str] = []
    try:
        for name in todo:
            a, b = ra.get(name), rb.get(name)
            if a.dtype != b.dtype or a.shape != b.shape:
                unequal.append(f"{name}: {a.dtype}{tuple(a.shape)} vs {b.dtype}{tuple(b.shape)}")
            elif a.dtype == torch.int8:
                int_total += a.numel()
                int_equal += int((a == b).sum())
            elif a.dtype == torch.uint8 and name.endswith(PACKED_SUFFIX):
                fp4_total += 2 * a.numel()
                fp4_equal += int((unpack_fp4(a) == unpack_fp4(b)).sum())
            elif a.dtype == torch.uint8 and name.endswith(SCALE_SUFFIX):
                e8m0_total += a.numel()
                e8m0_equal += int((a == b).sum())
            elif name.endswith(SCALE_SUFFIX):
                rel = ((a.double() - b.double()).abs() / b.double().abs().clamp_min(1e-30)).max().item()
                scale_rel = max(scale_rel, rel)
            elif not torch.equal(a, b):
                unequal.append(name)
    finally:
        ra.close()
        rb.close()
    return {
        "tensors": len(todo),
        "int_values": int_total,
        "int_match_share": int_equal / int_total if int_total else None,
        "scale_max_rel_err": scale_rel,
        "fp4_values": fp4_total,
        "fp4_match_share": fp4_equal / fp4_total if fp4_total else None,
        "e8m0_match_share": e8m0_equal / e8m0_total if e8m0_total else None,
        "unequal_tensors": len(unequal),
        "unequal_examples": unequal[:20],
    }


def _quantized_module_of(name: str, layout: CheckpointLayout, suffixes: Iterable[str]) -> str | None:
    """``<module>`` if ``name`` is ``<module><suffix>`` for a quantized module, else ``None``."""
    modules = layout.quantized_modules
    for suffix in suffixes:
        if name.endswith(suffix) and name[: -len(suffix)] in modules:
            return name[: -len(suffix)]
    return None


def _check_against_layout(tensors: Mapping[str, torch.Tensor], layout: CheckpointLayout) -> None:
    scheme = layout.scheme
    for name, t in tensors.items():
        spec = layout.tensors[name]
        if _DTYPE_NAMES.get(t.dtype) != spec.dtype or tuple(t.shape) != spec.shape:
            raise LayoutError(f"{name}: produced {t.dtype}{tuple(t.shape)}, layout wants {spec.dtype}{spec.shape}")
        if name.endswith(scheme.data_suffix) and spec.dtype == scheme.data_dtype:
            scheme.check_values(name, t)


def _spec_bytes(spec: TensorSpec) -> int:
    n = 1
    for d in spec.shape:
        n *= d
    return n * _DTYPES[spec.dtype].itemsize


def _write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _smoothing_from_args(args: argparse.Namespace) -> AwqSmoothing:
    if args.no_awq:
        return AwqSmoothing.none()
    if not (args.recipe and args.awq_scales):
        raise SystemExit("give --recipe and --awq-scales (option R2) or --no-awq (option R1)")
    return AwqSmoothing.from_artefact(Path(args.recipe), Path(args.awq_scales))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m yeto.rl.quant.requant", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("template", help="record the layout of a reference quantized checkpoint (W4A8 or MXFP4)")
    t.add_argument("checkpoint")
    t.add_argument("template_dir")
    c = sub.add_parser("convert", help="BF16 checkpoint -> quantized checkpoint in the template layout")
    c.add_argument("--source", required=True)
    c.add_argument("--template", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--recipe")
    c.add_argument("--awq-scales")
    c.add_argument("--no-awq", action="store_true")
    c.add_argument("--shards", help="comma-separated output shard files (worker split); then run 'finalize'")
    f = sub.add_parser("finalize", help="write index and metadata after a split conversion")
    f.add_argument("--template", required=True)
    f.add_argument("--out", required=True)
    m = sub.add_parser("compare", help="tensor-by-tensor comparison of two checkpoints")
    m.add_argument("a")
    m.add_argument("b")
    m.add_argument("--match", help="regex: compare only matching tensor names")
    args = p.parse_args(argv)

    if args.cmd == "template":
        CheckpointLayout.from_checkpoint(Path(args.checkpoint)).save_template(Path(args.checkpoint), Path(args.template_dir))
        result: Any = {"template": args.template_dir}
    elif args.cmd == "convert":
        shards = args.shards.split(",") if args.shards else None
        result = requantize_checkpoint(Path(args.source), Path(args.template), Path(args.out), _smoothing_from_args(args), shards)
    elif args.cmd == "finalize":
        finalize_checkpoint(Path(args.template), Path(args.out))
        result = {"finalized": args.out}
    else:
        names = None
        if args.match:
            pat = re.compile(args.match)
            names = [n for n in _SourceReader(Path(args.a)).weight_map if pat.search(n)]
        result = compare_checkpoints(Path(args.a), Path(args.b), names)
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
