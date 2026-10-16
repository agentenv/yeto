"""BF16 checkpoint -> W4A8 checkpoint in the layout of the original artefact (S25 N8.1).

The target layout is taken from the original W4A8 checkpoint, not hard-coded:
:class:`CheckpointLayout` records every tensor's name, dtype, shape and output
shard. A tensor is quantized exactly when the layout has it as ``I8`` with a
``<module>.weight_scale`` next to it; every other tensor is cast to its layout
dtype (the original artefact changes some dtypes, e.g. routers BF16 -> F32 and
``e_score_correction_bias`` F32 -> BF16). AWQ smoothing (:mod:`.awq`) is
applied first. The result therefore has the same file names, tensor names,
dtypes and shapes as the original, and the same metadata files.

A layout can be saved as a small template directory (``layout.json`` plus the
non-weight files such as ``config.json`` and ``recipe.yaml``), so later rounds
do not need the 300 GB original.

Work split: :func:`requantize_checkpoint` takes a subset of output shards, so
several workers can each write their own shards. :func:`transform_tensor`
works on any row slice of a 2-D weight with the same result rows, which is the
basis for quantizing on the trainer ranks (D7, task N8.2; not done here).

Status: transform rules checked bit for bit against the original artefact on
layer 30 (router, shared experts, norm, one routed expert); the full-model
comparison (:func:`compare_checkpoints` over all 74169 tensors) is not run.
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
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .awq import AwqSmoothing
from .int4 import QMAX, QMIN, quantize_per_channel

INDEX_FILE = "model.safetensors.index.json"
LAYOUT_FILE = "layout.json"
LAYOUT_SCHEMA = "yeto.rl.quant.layout/v1"
SCALE_SUFFIX = ".weight_scale"
WEIGHT_SUFFIX = ".weight"

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
    def validate(self) -> None:
        """Check that the quantization config is the one :mod:`.int4` implements, and the I8/scale pairs."""
        q = self.quantization_config
        if q.get("quant_method") != "compressed-tensors" or q.get("format") != "int-quantized":
            raise LayoutError(f"unsupported quantization format: {q.get('quant_method')!r} / {q.get('format')!r}")
        for group, cfg in (q.get("config_groups") or {}).items():
            w = cfg.get("weights") or {}
            want = {"num_bits": 4, "type": "int", "symmetric": True, "strategy": "channel", "group_size": None, "dynamic": False}
            got = {k: w.get(k) for k in want}
            if got != want or w.get("observer") not in ("memoryless_minmax", "minmax"):
                raise LayoutError(f"config group {group}: weight scheme {got} (observer {w.get('observer')!r}) is not supported")
        quantized = set(self.quantized_weights())
        for name in quantized:
            scale = self.tensors[name[: -len(WEIGHT_SUFFIX)] + SCALE_SUFFIX]
            rows = self.tensors[name].shape[0]
            if scale.shape != (rows, 1):
                raise LayoutError(f"{name}: scale shape {scale.shape} is not per-channel [{rows}, 1]")
        bare = [n for n, t in self.tensors.items() if t.dtype == "I8" and n.endswith(WEIGHT_SUFFIX) and n not in quantized]
        if bare:
            raise LayoutError(f"{len(bare)} I8 weights without a weight_scale, e.g. {bare[:3]}")

    def quantized_weights(self) -> list[str]:
        return sorted(
            n for n, t in self.tensors.items()
            if t.dtype == "I8" and n.endswith(WEIGHT_SUFFIX) and n[: -len(WEIGHT_SUFFIX)] + SCALE_SUFFIX in self.tensors
        )

    def source_names(self) -> set[str]:
        """Tensor names the BF16 source must provide: every target tensor except the scales."""
        return {n for n in self.tensors if not _is_quant_scale(n, self)}

    def shards(self) -> dict[str, list[str]]:
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

    A quantized weight yields ``{name: int8, <module>.weight_scale: scale}``;
    any other tensor yields ``{name: tensor in the layout dtype}``. Shapes are
    not checked here, so row slices pass; :func:`requantize_checkpoint` checks
    whole tensors.
    """
    spec = layout.tensors.get(name)
    if spec is None or _is_quant_scale(name, layout):
        raise LayoutError(f"{name}: not a source tensor of the target layout")
    if spec.dtype != "I8":
        return {name: smoothing.apply(name, tensor, _DTYPES[spec.dtype])}
    module = name[: -len(WEIGHT_SUFFIX)]
    scale_spec = layout.tensors[module + SCALE_SUFFIX]
    # The original smoothed the expert in FP64 and stored it in the source dtype before quantizing.
    smoothed = smoothing.apply(name, tensor, tensor.dtype)
    q, scale = quantize_per_channel(smoothed, scale_dtype=_DTYPES[scale_spec.dtype])
    return {name: q, module + SCALE_SUFFIX: scale}


class _SourceReader:
    """Lazy reader over a sharded safetensors source checkpoint."""

    def __init__(self, source_dir: Path) -> None:
        self.dir = Path(source_dir)
        index = self.dir / INDEX_FILE
        if index.is_file():
            self.weight_map: dict[str, str] = json.loads(index.read_text())["weight_map"]
        else:
            self.weight_map = {n: p.name for p in sorted(self.dir.glob("*.safetensors")) for n in read_header(p)}
        self._open: dict[str, Any] = {}

    def get(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        if shard not in self._open:
            self._open[shard] = safe_open(str(self.dir / shard), framework="pt", device="cpu")
        return self._open[shard].get_tensor(name)

    def close(self) -> None:
        self._open.clear()


def requantize_checkpoint(
    source_dir: Path,
    template_dir: Path,
    out_dir: Path,
    smoothing: AwqSmoothing,
    shards: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Write the target checkpoint (or the given output shards of it) into ``out_dir``.

    ``template_dir`` holds ``layout.json`` and the metadata files
    (:meth:`CheckpointLayout.save_template`). Without ``shards`` the whole
    checkpoint is written, then the index and the metadata files, so a
    complete ``out_dir`` is loadable; with ``shards`` only those files are
    written and :func:`finalize_checkpoint` must run after all workers.
    The source must provide exactly the layout's source tensors.
    """
    layout = CheckpointLayout.load(template_dir)
    source = _SourceReader(source_dir)
    expected = layout.source_names()
    missing, extra = expected - set(source.weight_map), set(source.weight_map) - expected
    if missing or extra:
        raise LayoutError(
            f"source tensors differ from the target layout: {len(missing)} missing (e.g. {sorted(missing)[:3]}), "
            f"{len(extra)} extra (e.g. {sorted(extra)[:3]})"
        )
    all_shards = layout.shards()
    selected = list(all_shards) if shards is None else list(shards)
    if unknown := set(selected) - set(all_shards):
        raise LayoutError(f"unknown output shards {sorted(unknown)}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"shards": 0, "tensors": 0, "quantized": 0}
    try:
        for shard in selected:
            produced: dict[str, torch.Tensor] = {}
            for name in all_shards[shard]:
                if name in expected:
                    for out_name, t in transform_tensor(name, source.get(name), layout, smoothing).items():
                        produced[out_name] = t
                    stats["quantized"] += layout.tensors[name].dtype == "I8"
            # Scales may live in another shard than their weight (seen in the original); recompute them there.
            for name in all_shards[shard]:
                if name not in produced:
                    weight = name[: -len(SCALE_SUFFIX)] + WEIGHT_SUFFIX
                    produced[name] = transform_tensor(weight, source.get(weight), layout, smoothing)[name]
            tensors = {n: produced[n].contiguous() for n in all_shards[shard]}
            _check_against_layout(tensors, layout)
            tmp = out_dir / (shard + ".tmp")
            save_file(tensors, str(tmp), metadata={"format": "pt"})
            os.replace(tmp, out_dir / shard)
            stats["shards"] += 1
            stats["tensors"] += len(tensors)
    finally:
        source.close()
    if shards is None:
        finalize_checkpoint(template_dir, out_dir)
    return stats


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
    """Compare two checkpoints of one layout tensor by tensor (N8.1 acceptance).

    Reports the share of equal int values over all I8 tensors, the largest
    relative error of the weight scales, and the high-precision tensors that
    are not bit-equal. No threshold is applied here.
    """
    ra, rb = _SourceReader(a_dir), _SourceReader(b_dir)
    todo = sorted(names) if names is not None else sorted(ra.weight_map)
    if missing := [n for n in todo if n not in rb.weight_map]:
        raise LayoutError(f"{len(missing)} tensors missing in {b_dir}, e.g. {missing[:3]}")
    int_total = int_equal = 0
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
        "unequal_tensors": len(unequal),
        "unequal_examples": unequal[:20],
    }


def _is_quant_scale(name: str, layout: CheckpointLayout) -> bool:
    if not name.endswith(SCALE_SUFFIX):
        return False
    weight = layout.tensors.get(name[: -len(SCALE_SUFFIX)] + WEIGHT_SUFFIX)
    return weight is not None and weight.dtype == "I8"


def _check_against_layout(tensors: Mapping[str, torch.Tensor], layout: CheckpointLayout) -> None:
    for name, t in tensors.items():
        spec = layout.tensors[name]
        if _DTYPE_NAMES.get(t.dtype) != spec.dtype or tuple(t.shape) != spec.shape:
            raise LayoutError(f"{name}: produced {t.dtype}{tuple(t.shape)}, layout wants {spec.dtype}{spec.shape}")
        if t.dtype == torch.int8 and name.endswith(WEIGHT_SUFFIX) and (t.min() < QMIN or t.max() > QMAX):
            raise LayoutError(f"{name}: int4 values outside [{QMIN}, {QMAX}]")


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
    t = sub.add_parser("template", help="record the layout of an original W4A8 checkpoint")
    t.add_argument("checkpoint")
    t.add_argument("template_dir")
    c = sub.add_parser("convert", help="BF16 checkpoint -> W4A8 checkpoint in the template layout")
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
