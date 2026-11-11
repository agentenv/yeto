"""Export a syncer-held DiLoCo global state of an ms-swift Megatron island to HF.

With ``--final-cut-payload manifest`` (what a 120B island must use: a full
payload holds every fragment in FP32, 480 GB, on rank 0's host) the final
global parameters never reach the island.  They live in the syncer's
``--checkpoint-path`` and ``--final-state``.  This module turns either of
those into HuggingFace safetensors in the same layout as a
``swift export --mcore_model ... --to_hf true`` run, so the result is
comparable with an ms-swift export of the same recipe under the same
evaluation.

Pipeline
    1. read the syncer artifact -> one FP32 buffer per fragment;
    2. rebuild the island's fragment layout (``yeto.fragments.build_layout``
       over the sorted global trainable names, the same call
       ``MegatronDiLoCoSync`` makes) and cut each fragment back into named
       global tensors;
    3. load a *whole* mcore model from a Megatron checkpoint of the same run
       (TP=PP=EP=1, ``use_cpu_initialization``, which is what
       ``swift/megatron/convert.py`` already does), overwrite the synchronized
       tensors, and let ``mcore_bridge``'s ``NemotronHBridge.save_weights``
       write the HF files.

Why step 3 loads a Megatron checkpoint instead of building the name mapping
here: DiLoCo synchronizes only the *trainable* parameters -- the frozen vision
tower and the aligner are not in the layout -- so a complete HF model needs
those weights from somewhere anyway, and the Megatron checkpoint of the run
is the only artifact that has them next to the trained ones.  Taking the
HF-name mapping from the bridge at the same time means this module contains no
hand-written name table, which is the one thing a 42101-tensor hybrid
Mamba/MoE/attention model must not have duplicated.

The two places where this module cannot avoid Megatron-specific knowledge:

Expert and layer indices
    The sync names a parameter by its *global* Megatron name (global layer
    index, global expert index).  An export model with PP=1 and EP=1 holds
    every layer and every expert locally, so its own local names *are* the
    global ones -- except that the local expert index must be renumbered,
    which :func:`yeto.swift_megatron.sync.global_parameter_name` already does
    (with ``local_expert_indices == range(num_experts)`` it is the identity).
    So the lookup reuses that function, not a copy of it.

TP-fused tensors
    A full tensor in the sync namespace is the concatenation of the island's
    ``tp_size`` shards along ``partition_dim``.  For a plain row/column
    parallel tensor that equals the single-process tensor.  For a *fused*
    tensor whose single-process layout is a sequence of chunks that Megatron
    shards chunk-wise (a gated ``linear_fc1`` = [gate, up]; Mamba ``in_proj``
    = [z, x, B, C, dt]; Mamba ``conv1d`` = [x, B, C]), rank r holds the r-th
    slice *of every chunk*, so the concatenation interleaves them and has to be
    permuted back.  :func:`tp_deinterleave` does that from the chunk sizes
    alone, and :func:`fused_chunk_sizes` derives the chunk sizes from the mcore
    ``TransformerConfig`` of the model being written into -- no name table.
    With ``tp_size == 1`` (and ``ETP == 1`` for experts, the usual setting) the
    permutation is the identity and none of this runs.

Commands::

    # on a node of the island, inside the training process is not needed: any
    # process that can build the island's model works.  Writes the name/shape
    # table the export needs (it is the island's layout contract, not ours).
    python -m yeto.swift_megatron.export_hf dump-table --out table.json

    # offline, from the syncer's artifacts
    python -m yeto.swift_megatron.export_hf inspect \
        --checkpoint <syncer-dir>/syncer.ckpt --table table.json

    python -m yeto.swift_megatron.export_hf to-hf \
        --final-state <syncer-dir>/final_state.f32 \
        --table table.json \
        --mcore-model <run-dir>/checkpoint-<step> \
        --model <hf-base-model-dir> \
        --model-type nemotron_h_omni \
        --output-dir <out>/diloco-hf

    # LLM-only twin of the full VL export, with the key layout of an
    # ms-swift LLM-only export (its config is copied from --config-from)
    python -m yeto.swift_megatron.export_hf strip-vision \
        --hf-dir <out>/diloco-hf \
        --output-dir <out>/diloco-llm \
        --config-from <llm-only-export-dir>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import struct
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Sequence

log = logging.getLogger("yeto-export-hf")

# ``state.rs``: CKPT_MAGIC_V3, then iso_backend u8, layout fingerprint 32 B,
# global_step u64, fragment count u32, then per fragment version u64, numel
# u64, numel f32 params, numel f32 momentum.
CKPT_MAGIC_V3 = 0xD170_5A80
CKPT_MAGIC_V2 = 0xD170_5A7F
CKPT_MAGIC_V1 = 0xD170_5A7E
TABLE_SCHEMA = "yeto.swift-megatron.global-table/v1"


class ExportError(RuntimeError):
    pass


# --------------------------------------------------------------------- reading
def _read_exact(handle, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise ExportError(f"truncated syncer artifact: wanted {size} bytes, got {len(data)}")
    return data


def _f32(handle, numel: int):
    """``numel`` f32 values read straight into one buffer (no second copy:
    a 120B fragment is ~15 GB)."""
    import torch

    buf = bytearray(4 * numel)
    view, got = memoryview(buf), 0
    while got < len(buf):
        n = handle.readinto(view[got:])
        if not n:
            raise ExportError(f"truncated syncer artifact: wanted {len(buf)} bytes, got {got}")
        got += n
    return torch.frombuffer(buf, dtype=torch.float32, count=numel)


def _skip(handle, size: int) -> None:
    """Seek past ``size`` bytes that must exist (the momentum is never read)."""
    handle.seek(size, os.SEEK_CUR)
    if handle.tell() > os.fstat(handle.fileno()).st_size:
        raise ExportError("truncated syncer artifact: the momentum runs past the end of the file")


@dataclass
class SyncerState:
    """Per-fragment FP32 parameters of a syncer artifact, in fragment order."""

    fragments: list[Any]
    global_step: int | None = None
    versions: list[int] = field(default_factory=list)
    layout_fingerprint: bytes | None = None
    source: str = ""

    @property
    def numels(self) -> list[int]:
        return [int(f.numel()) for f in self.fragments]


def read_final_state(path: str) -> SyncerState:
    """``--final-state``: u32 fragment count, then (u64 numel, numel f32)*."""

    with open(path, "rb") as handle:
        (count,) = struct.unpack("<I", _read_exact(handle, 4))
        fragments = []
        for _ in range(count):
            (numel,) = struct.unpack("<Q", _read_exact(handle, 8))
            fragments.append(_f32(handle, numel))
        if handle.read(1):
            raise ExportError(f"{path}: trailing bytes after {count} fragments")
    return SyncerState(fragments, source=f"final-state:{path}")


def read_syncer_checkpoint(path: str) -> SyncerState:
    """``--checkpoint-path`` (V3). Momentum is read past, never kept."""

    with open(path, "rb") as handle:
        (magic,) = struct.unpack("<I", _read_exact(handle, 4))
        if magic in (CKPT_MAGIC_V1, CKPT_MAGIC_V2):
            raise ExportError(
                f"{path}: checkpoint V{1 if magic == CKPT_MAGIC_V1 else 2} carries no semantic "
                "layout fingerprint; re-run the syncer or export from --final-state")
        if magic != CKPT_MAGIC_V3:
            raise ExportError(f"{path}: bad checkpoint magic {magic:#x}")
        _read_exact(handle, 1)  # iso backend
        fingerprint = _read_exact(handle, 32)
        (global_step,) = struct.unpack("<Q", _read_exact(handle, 8))
        (count,) = struct.unpack("<I", _read_exact(handle, 4))
        fragments, versions = [], []
        for _ in range(count):
            (version,) = struct.unpack("<Q", _read_exact(handle, 8))
            (numel,) = struct.unpack("<Q", _read_exact(handle, 8))
            versions.append(int(version))
            fragments.append(_f32(handle, numel))
            _skip(handle, 4 * numel)  # momentum
    return SyncerState(fragments, global_step=int(global_step), versions=versions,
                       layout_fingerprint=fingerprint, source=f"checkpoint:{path}")


def read_syncer_state(*, checkpoint: str | None = None, final_state: str | None = None) -> SyncerState:
    if bool(checkpoint) == bool(final_state):
        raise ExportError("give exactly one of --checkpoint / --final-state")
    return read_syncer_checkpoint(checkpoint) if checkpoint else read_final_state(final_state)


# ----------------------------------------------------------------- name table
@dataclass
class GlobalTableFile:
    """The island's layout contract: every global trainable tensor's shape,
    its TP partition dim, and the parallel sizes the layout was built with."""

    shapes: dict[str, tuple[int, ...]]
    part_dims: dict[str, int]
    expert: dict[str, bool]
    tp_size: int
    etp_size: int
    fragments: int
    fragment_pattern: str

    @classmethod
    def load(cls, path: str) -> GlobalTableFile:
        with open(path) as handle:
            body = json.load(handle)
        if body.get("schema") != TABLE_SCHEMA:
            raise ExportError(f"{path}: not a {TABLE_SCHEMA} table")
        tensors = body["tensors"]
        return cls(
            shapes={n: tuple(int(d) for d in t["shape"]) for n, t in tensors.items()},
            part_dims={n: int(t.get("partition_dim", -1)) for n, t in tensors.items()},
            expert={n: bool(t.get("expert", False)) for n, t in tensors.items()},
            tp_size=int(body["tp_size"]),
            etp_size=int(body["etp_size"]),
            fragments=int(body["fragments"]),
            fragment_pattern=str(body.get("fragment_pattern", "binpack")),
        )

    def dump(self, path: str) -> None:
        body = {
            "schema": TABLE_SCHEMA,
            "tp_size": self.tp_size,
            "etp_size": self.etp_size,
            "fragments": self.fragments,
            "fragment_pattern": self.fragment_pattern,
            "tensors": {
                name: {"shape": list(shape),
                       "partition_dim": self.part_dims.get(name, -1),
                       "expert": bool(self.expert.get(name, False))}
                for name, shape in sorted(self.shapes.items())
            },
        }
        with open(path, "w") as handle:
            json.dump(body, handle, indent=1, sort_keys=True)

    def numel(self, name: str) -> int:
        out = 1
        for dim in self.shapes[name]:
            out *= int(dim)
        return out

    def tp_of(self, name: str) -> int:
        return self.etp_size if self.expert.get(name, False) else self.tp_size


def build_layout_from_table(table: GlobalTableFile):
    """The layout ``MegatronDiLoCoSync`` builds, from the table alone."""

    from yeto.fragments import build_layout

    from .sync import OUTPUT_LAYER_AVG_REGEX

    names = sorted(table.shapes)
    if not names:
        raise ExportError("the table has no tensors")
    return build_layout(
        [(name, table.numel(name)) for name in names], table.fragments, table.fragment_pattern,
        avg_name_regex=OUTPUT_LAYER_AVG_REGEX,
        named_shapes={name: table.shapes[name] for name in names})


def named_tensors(state: SyncerState, table: GlobalTableFile) -> dict[str, Any]:
    """Cut every fragment of ``state`` back into named global FP32 tensors."""

    from yeto.protocol import layout_fingerprint

    layout = build_layout_from_table(table)
    if layout.num_fragments != len(state.fragments):
        raise ExportError(
            f"{state.source} has {len(state.fragments)} fragments, the table's layout has "
            f"{layout.num_fragments} (fragments={table.fragments}, "
            f"pattern={table.fragment_pattern!r}): the table does not describe this run")
    if state.layout_fingerprint is not None:
        expected = layout_fingerprint(layout)
        if bytes(state.layout_fingerprint) != bytes(expected):
            raise ExportError(
                "the checkpoint's semantic layout fingerprint does not match the table's layout; "
                "the table is from a different island shape or a different trainable set")
    out: dict[str, Any] = {}
    for index, (fragment, buffer) in enumerate(zip(layout.fragments, state.fragments)):
        offset = 0
        for name, numel in fragment.tensors:
            if numel != table.numel(name):
                raise ExportError(f"{name}: layout numel {numel} != table numel {table.numel(name)}")
            out[name] = buffer[offset:offset + numel].view(table.shapes[name])
            offset += numel
        if offset != int(buffer.numel()):
            raise ExportError(
                f"fragment {index}: layout covers {offset} of {int(buffer.numel())} elements")
    return out


# ----------------------------------------------------------- TP-fused tensors
def tp_deinterleave(tensor: Any, part_dim: int, tp_size: int, chunk_sizes: Sequence[int]) -> Any:
    """The single-process tensor, from the concatenation of ``tp_size`` shards.

    Megatron shards a fused tensor chunk-wise: rank ``r`` holds, along
    ``part_dim`` and in this order, ``chunk_sizes[j] / tp_size`` rows of every
    chunk ``j``.  The sync's full tensor is ``cat(shard_0 .. shard_{t-1})``, so
    the chunks arrive interleaved and this reassembles them.
    """

    import torch

    total = int(tensor.shape[part_dim])
    if sum(chunk_sizes) != total:
        raise ExportError(f"chunk sizes {tuple(chunk_sizes)} do not sum to dim {part_dim} = {total}")
    if tp_size <= 1 or len(chunk_sizes) <= 1:
        return tensor
    shard = total // tp_size
    if shard * tp_size != total:
        raise ExportError(f"dim {part_dim} = {total} is not divisible by tp_size {tp_size}")
    local = []
    for size in chunk_sizes:
        if size % tp_size:
            raise ExportError(f"chunk of {size} rows is not divisible by tp_size {tp_size}")
        local.append(size // tp_size)
    pieces = []
    for chunk_index, size in enumerate(local):
        start = sum(local[:chunk_index])
        for rank in range(tp_size):
            base = rank * shard + start
            pieces.append(tensor.narrow(part_dim, base, size))
    return torch.cat(pieces, dim=part_dim)


# The fused families, keyed by the tail of the Megatron name.  Each entry
# returns the chunk sizes of the *single-process* tensor along its partition
# dim, from the mcore config -- so there is no hard-coded shape anywhere.
def _gated_fc1_chunks(name: str, shape: Sequence[int], config: Any) -> list[int] | None:
    if not getattr(config, "gated_linear_unit", False):
        return None
    rows = int(shape[0])
    if rows % 2:
        raise ExportError(f"{name}: gated linear_fc1 has an odd first dim {rows}")
    return [rows // 2, rows // 2]


def _mamba_in_proj_chunks(name: str, shape: Sequence[int], config: Any) -> list[int] | None:
    heads = getattr(config, "mamba_num_heads", None)
    head_dim = getattr(config, "mamba_head_dim", None)
    groups = getattr(config, "mamba_num_groups", None)
    state = getattr(config, "mamba_state_dim", None)
    if None in (heads, head_dim, groups, state):
        raise ExportError(f"{name}: the mcore config has no mamba_* sizes, cannot de-interleave")
    inner = int(heads) * int(head_dim)
    bc = int(groups) * int(state)
    chunks = [inner, inner, bc, bc, int(heads)]  # z, x, B, C, dt
    if sum(chunks) != int(shape[0]):
        raise ExportError(
            f"{name}: mamba in_proj rows {int(shape[0])} != z+x+B+C+dt = {sum(chunks)} "
            f"(heads={heads}, head_dim={head_dim}, groups={groups}, state={state})")
    return chunks


def _mamba_conv1d_chunks(name: str, shape: Sequence[int], config: Any) -> list[int] | None:
    heads = getattr(config, "mamba_num_heads", None)
    head_dim = getattr(config, "mamba_head_dim", None)
    groups = getattr(config, "mamba_num_groups", None)
    state = getattr(config, "mamba_state_dim", None)
    if None in (heads, head_dim, groups, state):
        raise ExportError(f"{name}: the mcore config has no mamba_* sizes, cannot de-interleave")
    inner = int(heads) * int(head_dim)
    bc = int(groups) * int(state)
    chunks = [inner, bc, bc]  # x, B, C
    if sum(chunks) != int(shape[0]):
        raise ExportError(
            f"{name}: mamba conv1d rows {int(shape[0])} != x+B+C = {sum(chunks)}")
    return chunks


FUSED_CHUNKS: tuple[tuple[re.Pattern, Callable[[str, Sequence[int], Any], list[int] | None]], ...] = (
    (re.compile(r"(^|\.)linear_fc1\.(weight|bias)\d*$"), _gated_fc1_chunks),
    (re.compile(r"(^|\.)in_proj\.(weight|bias)$"), _mamba_in_proj_chunks),
    (re.compile(r"(^|\.)conv1d\.(weight|bias)$"), _mamba_conv1d_chunks),
)


def fused_chunk_sizes(name: str, shape: Sequence[int], config: Any) -> list[int] | None:
    """Chunk sizes of a fused tensor, or ``None`` when it is not fused."""

    for pattern, resolve in FUSED_CHUNKS:
        if pattern.search(name):
            return resolve(name, shape, config)
    return None


def to_single_process(name: str, tensor: Any, part_dim: int, tp_size: int, config: Any) -> Any:
    """Undo the island's TP concatenation for ``name``."""

    if tp_size <= 1 or part_dim < 0:
        return tensor
    chunks = fused_chunk_sizes(name, tuple(tensor.shape), config)
    if chunks is None:
        return tensor
    return tp_deinterleave(tensor, part_dim, tp_size, chunks)


# ------------------------------------------------------------- mcore plumbing
def export_name_map(model_chunks: Sequence[Any]) -> dict[str, Any]:
    """``{global Megatron name: parameter}`` of a whole (TP=PP=EP=1) model."""

    from .sync import _expert_modules, _global_layer_prefixes, global_parameter_name, unwrap_chunk

    out: dict[str, Any] = {}
    seen: set[int] = set()
    for wrapped in model_chunks:
        chunk = unwrap_chunk(wrapped)
        layers = _global_layer_prefixes(chunk)
        experts = _expert_modules(chunk)
        for local, param in chunk.named_parameters():
            if id(param) in seen:
                continue
            seen.add(id(param))
            name = global_parameter_name(local, layers, experts)
            if name in out:
                raise ExportError(f"two parameters map to the global name {name!r}")
            out[name] = param
    return out


def apply_to_mcore(model_chunks: Sequence[Any], tensors: dict[str, Any], table: GlobalTableFile,
                   *, strict: bool = True) -> dict[str, int]:
    """Write every synchronized tensor into a whole mcore model, in place."""

    import torch

    from .sync import unwrap_chunk

    params = export_name_map(model_chunks)
    config = unwrap_chunk(model_chunks[0]).config if model_chunks else None
    stats = {"written": 0, "missing_in_model": 0, "not_synchronized": len(set(params) - set(tensors))}
    for name, value in tensors.items():
        param = params.get(name)
        if param is None:
            stats["missing_in_model"] += 1
            if strict:
                raise ExportError(
                    f"{name} is in the syncer state but not in the export model: the export model "
                    "has a different architecture or parallel shape")
            continue
        value = to_single_process(name, value, table.part_dims.get(name, -1),
                                  table.tp_of(name), config)
        if tuple(param.shape) != tuple(value.shape):
            raise ExportError(f"{name}: model shape {tuple(param.shape)} != exported {tuple(value.shape)}")
        if not torch.isfinite(value).all():
            raise ExportError(f"{name}: the syncer state has non-finite values")
        with torch.no_grad():
            param.copy_(value.to(dtype=param.dtype))
        stats["written"] += 1
    log.info("wrote %d tensors; %d model tensors were not synchronized (frozen vision / aligner)",
             stats["written"], stats["not_synchronized"])
    return stats


# ------------------------------------------------------------------- commands
def _megatron_export_model(args):
    """Build a whole (TP=PP=EP=1, CPU) mcore model and load ``--mcore-model``.

    This is ``swift/megatron/convert.py::convert_mcore2hf`` up to the point
    where it would call ``bridge.save_weights``, reusing its ``convert_kwargs``
    and its argument plumbing verbatim.
    """

    from swift.arguments import ExportArguments
    from swift.megatron.arguments import MegatronArguments
    from swift.megatron.convert import convert_kwargs
    from swift.megatron.model import get_mcore_model
    from swift.megatron.utils import load_mcore_checkpoint
    from swift.pipelines import prepare_model_template

    export_args = ExportArguments(model=args.model, model_type=args.model_type,
                                  mcore_model=args.mcore_model, to_hf=True,
                                  output_dir=args.output_dir)
    export_args.experts_impl = "eager"
    _, template = prepare_model_template(export_args, load_model=False)
    processor = template.processor
    kwargs = dict(convert_kwargs)
    if export_args.model_info.is_moe_model:
        kwargs["moe_grouped_gemm"] = True
    kwargs.update(MegatronArguments.load_args_config(args.mcore_model))
    kwargs["mcore_model"] = args.mcore_model
    megatron_args = MegatronArguments(model=args.model, model_type=args.model_type,
                                      **kwargs, output_dir=None,
                                      torch_dtype=export_args.torch_dtype)
    model = get_mcore_model(megatron_args, processor.model_info.config)[0]
    load_mcore_checkpoint(megatron_args, [model], load_arg="mcore_model")
    return model, megatron_args, processor


_REGISTERED_CHUNKS: list[Any] = []


def register_model_chunks(model_chunks: Sequence[Any]) -> None:
    """Make an island's model chunks visible to ``dump-table``.

    The table is the island's layout contract, so it has to be written by a
    process that holds the island's model.  A three-line external plugin is
    enough (it never changes training; it only writes a JSON file once)::

        from swift.megatron.callbacks import MegatronCallback, megatron_callbacks_map
        from yeto.swift_megatron import export_hf

        class DumpTable(MegatronCallback):
            def on_train_begin(self):
                export_hf.register_model_chunks(self.trainer.wrapped_models)
                export_hf.main(["dump-table", "--out", "<out>/table.json",
                                "--fragments", "32"])

        megatron_callbacks_map["yeto_dump_table"] = DumpTable
    """

    _REGISTERED_CHUNKS[:] = list(model_chunks)


def registered_model_chunks() -> Sequence[Any]:
    if not _REGISTERED_CHUNKS:
        raise ExportError(
            "no model chunks registered: call export_hf.register_model_chunks(trainer."
            "wrapped_models) first, or pass --chunks-getter module:attribute")
    return list(_REGISTERED_CHUNKS)


def cmd_dump_table(args) -> int:
    """Write the island's name/shape table. Run inside an island process (or
    any process that can build the island's model with the same TP/EP)."""

    import torch.distributed as dist

    from .sync import GlobalTable, ParallelContext, local_shards

    if not dist.is_initialized():
        raise ExportError("torch.distributed must be initialized: run this inside the island")
    chunks = _resolve_chunks(args.chunks_getter)
    ctx = ParallelContext.from_megatron()
    shards = local_shards(chunks, ctx)
    table = GlobalTable.build(shards)
    part_dims: dict[str, int] = {}
    expert: dict[str, bool] = {}
    rows = [(s.name, s.part_dim, s.expert) for s in shards]
    gathered: list[Any] = [None] * int(dist.get_world_size())
    dist.all_gather_object(gathered, rows)
    for held in gathered:
        for name, part_dim, is_expert in held:
            part_dims[name] = int(part_dim)
            expert[name] = bool(is_expert)
    out = GlobalTableFile(shapes=table.shapes, part_dims=part_dims, expert=expert,
                          tp_size=ctx.tp_size, etp_size=ctx.etp_size,
                          fragments=args.fragments, fragment_pattern=args.fragment_pattern)
    if int(dist.get_rank()) == 0:
        out.dump(args.out)
        print(f"wrote {args.out}: {len(out.shapes)} tensors, tp={out.tp_size} etp={out.etp_size}")
    return 0


def _resolve_chunks(spec: str) -> Sequence[Any]:
    module, _, attribute = spec.rpartition(":")
    import importlib

    value = getattr(importlib.import_module(module), attribute)
    return value() if callable(value) else value


def cmd_inspect(args) -> int:
    state = read_syncer_state(checkpoint=args.checkpoint, final_state=args.final_state)
    print(f"source          {state.source}")
    print(f"global_step     {state.global_step}")
    print(f"fragments       {len(state.fragments)}")
    print(f"versions        {state.versions}")
    total = sum(state.numels)
    print(f"parameters      {total} ({total / 1e9:.3f} B); numels {state.numels}")
    if not args.table:
        return 0
    table = GlobalTableFile.load(args.table)
    tensors = named_tensors(state, table)
    print(f"table           {len(table.shapes)} tensors, tp={table.tp_size} etp={table.etp_size}, "
          f"fragments={table.fragments} pattern={table.fragment_pattern}")
    import torch

    bad = [n for n, t in tensors.items() if not torch.isfinite(t).all()]
    print(f"non-finite      {len(bad)} tensors {bad[:5]}")
    for name in sorted(tensors)[: args.show]:
        tensor = tensors[name]
        print(f"  {name:70} {tuple(tensor.shape)!s:24} "
              f"mean {tensor.mean().item():+.3e} rms {tensor.float().pow(2).mean().sqrt().item():.3e}")
    return 0


def cmd_to_hf(args) -> int:
    state = read_syncer_state(checkpoint=args.checkpoint, final_state=args.final_state)
    table = GlobalTableFile.load(args.table)
    tensors = named_tensors(state, table)
    log.info("%s: %d fragments, %d named tensors", state.source, len(state.fragments), len(tensors))
    model, megatron_args, processor = _megatron_export_model(args)
    apply_to_mcore([model], tensors, table, strict=not args.allow_missing)
    bridge = model.config.bridge
    log.info("saving HF weights to %s", args.output_dir)
    bridge.save_weights([model], args.output_dir, args=megatron_args, processor=processor)
    source = os.path.join(args.mcore_model, "args.json")
    if os.path.exists(source):
        import shutil

        shutil.copy(source, os.path.join(args.output_dir, "args.json"))
    print(f"wrote {args.output_dir}")
    return 0


VISION_PREFIXES = ("mlp1.", "vision_model.", "vision_projector.")
LLM_PREFIX = "language_model."


def llm_only_key(key: str) -> str | None:
    """The LLM-only name of a full-VL export key, or ``None`` to drop it.

    Verified against an ms-swift VL / LLM-only export pair of the 120B model:
    the VL export has 42101 keys under ``language_model / mlp1 / vision_model /
    vision_projector``, the LLM-only one 41643 under ``backbone / lm_head``,
    and 42101 - 41643 = 458 is exactly the non-``language_model`` set.
    """

    if key.startswith(VISION_PREFIXES):
        return None
    if key.startswith(LLM_PREFIX):
        return key[len(LLM_PREFIX):]
    return key


def cmd_strip_vision(args) -> int:
    import shutil

    from safetensors import safe_open
    from safetensors.torch import save_file

    os.makedirs(args.output_dir, exist_ok=True)
    index_path = os.path.join(args.hf_dir, "model.safetensors.index.json")
    with open(index_path) as handle:
        weight_map = json.load(handle)["weight_map"]
    files: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        files.setdefault(shard, []).append(key)
    new_map: dict[str, str] = {}
    total_size = 0
    for shard in sorted(files):
        kept = {}
        with safe_open(os.path.join(args.hf_dir, shard), framework="pt") as handle:
            for key in sorted(files[shard]):
                new_key = llm_only_key(key)
                if new_key is None:
                    continue
                tensor = handle.get_tensor(key)
                kept[new_key] = tensor
                total_size += tensor.numel() * tensor.element_size()
        if not kept:
            continue
        save_file(kept, os.path.join(args.output_dir, shard), metadata={"format": "pt"})
        for key in kept:
            new_map[key] = shard
    with open(os.path.join(args.output_dir, "model.safetensors.index.json"), "w") as handle:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": new_map}, handle, indent=1)
    if args.config_from:
        for name in ("config.json", "generation_config.json", "tokenizer.json",
                     "tokenizer_config.json", "chat_template.jinja"):
            source = os.path.join(args.config_from, name)
            if os.path.exists(source):
                shutil.copy(source, os.path.join(args.output_dir, name))
    print(f"wrote {args.output_dir}: {len(new_map)} tensors "
          f"(dropped {len(weight_map) - len(new_map)} vision tensors)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="yeto.swift_megatron.export_hf", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    dump = sub.add_parser("dump-table", help="write the island's global name/shape table")
    dump.add_argument("--out", required=True)
    dump.add_argument("--fragments", type=int, required=True)
    dump.add_argument("--fragment-pattern", default="binpack")
    dump.add_argument("--chunks-getter",
                      default="yeto.swift_megatron.export_hf:registered_model_chunks",
                      help="module:attribute returning the island's model chunks")
    dump.set_defaults(func=cmd_dump_table)

    def source_args(parser):
        parser.add_argument("--checkpoint", help="the syncer's --checkpoint-path file")
        parser.add_argument("--final-state", help="the syncer's --final-state file")

    inspect = sub.add_parser("inspect", help="read a syncer artifact and check it against a table")
    source_args(inspect)
    inspect.add_argument("--table")
    inspect.add_argument("--show", type=int, default=10)
    inspect.set_defaults(func=cmd_inspect)

    to_hf = sub.add_parser("to-hf", help="write HF safetensors through mcore_bridge")
    source_args(to_hf)
    to_hf.add_argument("--table", required=True)
    to_hf.add_argument("--mcore-model", required=True,
                       help="a Megatron checkpoint of the run (supplies the frozen vision weights)")
    to_hf.add_argument("--model", required=True, help="the HF model dir the recipe trained from")
    to_hf.add_argument("--model-type", default="nemotron_h_omni")
    to_hf.add_argument("--output-dir", required=True)
    to_hf.add_argument("--allow-missing", action="store_true")
    to_hf.set_defaults(func=cmd_to_hf)

    strip = sub.add_parser("strip-vision", help="LLM-only twin of a full VL HF export")
    strip.add_argument("--hf-dir", required=True)
    strip.add_argument("--output-dir", required=True)
    strip.add_argument("--config-from", help="copy config/tokenizer from this HF dir")
    strip.set_defaults(func=cmd_strip_vision)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
