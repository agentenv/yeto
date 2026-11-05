"""Full-parameter DiLoCo sync of an ms-swift Megatron island.

ms-swift trains the model with Megatron-Core (TP/EP/PP, DistributedOptimizer,
optionally CPU optimizer offload through ``HybridDeviceOptimizer``).  This
module lets such an island speak the torch learner's DiLoCo protocol to the
unchanged Rust syncer (``yeto/learner.py`` + ``yeto/diloco_sync.py``):

* the same fragment layout semantics (``yeto.fragments.build_layout`` over the
  island's *global* trainable parameter set: whole tensors, size-balanced or
  strided, embeddings in the direct-averaged fragment);
* the same PUSH: ``local - raw_anchor`` per fragment, where ``raw_anchor`` is
  the exact global value of the last accepted broadcast (before any
  ``merge_alpha`` blend), answered only once the fragment has been anchored
  and trained on;
* the same BCAST handling: broadcasts are applied before pulls are answered,
  reset the fragment's counters and adopt the syncer's version;
* the same INIT (learner 0 seeds every fragment), finalization (raw final cut,
  then FINAL_ACK) and HELLO (layout, HMAC key).

What differs is only how a fragment is read and written, because no rank holds
a whole tensor:

Parameter namespace
    Every trainable Megatron parameter is named by its *global* Megatron name:
    the pipeline-local layer index is replaced by the global one
    (``layer_number - 1``) and an expert's local index by its global index
    (``local_expert_indices``).  So the name set and every tensor's full shape
    are independent of PP and EP, and the TP shards of one parameter
    concatenate along ``partition_dim`` into the full tensor.  Parameters the
    trainer does not train (``trainable_filter``: frozen vision tower and
    aligner) and buffers (router ``expert_bias``) are not part of the layout.

    The names are Megatron's, not the HF checkpoint's: the syncer is
    name-agnostic, and Megatron-Bridge (which ``yeto.rl.megatron_hf_boundary``
    builds on) is not what ms-swift uses (it uses ``mcore_bridge``, whose
    export is a whole-model pass with no per-parameter selection).  Fused
    Megatron tensors whose element order depends on the TP size (Mamba
    ``in_proj``, gated ``linear_fc1``) make the layout TP-specific, so the
    session contract hash binds the TP / expert-TP sizes and the syncer
    refuses an island of a different parallel shape.

Read (PUSH, INIT)
    Collective, one bounded chunk of tensors at a time: every rank gathers the
    FP32 optimizer masters of its local parameters over the data-parallel group
    (``yeto.rl.megatron_masters.full_masters``: DistributedOptimizer shards; with
    the precision-aware optimizer + CPU offload of the production recipe that is
    ``HybridDeviceOptimizer.param_to_fp32_param[bf16 shard]``, a pinned host
    tensor, looked up afresh on every call because HDO rebuilds its maps on
    ``load_state_dict``), subtracts its local anchor shard, all-gathers TP
    shards, and the lowest owning rank sends the full FP32 tensor
    point-to-point to global rank 0, which packs it into the wire buffer.  The
    bf16 model copy is never read: with ``overlap_param_gather`` its non-owned
    ranges are stale after a step.  The PUSH carries ``local - anchor``, the
    torch learner's sign (the syncer's outer gradient is its negation, ``base -
    local``).

Write (BCAST, final cut)
    Rank 0 broadcasts the chunk over the island in the wire dtype (bf16 by
    default, lossless for a bf16 session); every rank slices its TP shard,
    records it as its anchor, blends with its current master when
    ``merge_alpha > 0``, and writes the FP32 master shard *and* the bf16 model
    copy (``yeto.rl.megatron_masters.write_masters``: the HDO FP32 tensor, which
    under offload is also the host copy the CPU Adam steps, and the parameter's
    storage inside the DDP ``param_data`` buffer, from which the next forward's
    param all-gather reads).  The Adam moments are kept, exactly as the torch
    learner keeps its AdamW state when ``apply_fragment`` overwrites a fragment.

Only global rank 0 talks to the syncer.  Every boundary starts with one
``broadcast_object_list`` of rank 0's plan (fragments to apply, fragments to
push, finalization, shutdown, error), so every rank runs the same collectives
in the same order and an error on rank 0 is raised on every rank.

Memory (N = trainable parameters of the island, P = fragments, n_r = the
elements of rank r's local parameter shards, C = ``chunk_numel``, t = TP size):

* device: nothing with the default ``transport="cpu"`` (the collectives run
  on host buffers over gloo mirrors of the island's groups; the only device
  access is the in-place host-to-device copy into the existing bf16
  parameters).  The production pp0 stage has ~0.4 GiB of HBM headroom, so
  ``transport="device"`` (chunk buffers on the accelerator, peak about
  ``4 * C * (2 + t)`` bytes) is only for islands with room to spare;
* every rank, host: one anchor per local shard in the anchor dtype (bf16 for a
  bf16 session), ``2 * n_r`` bytes, plus one chunk in flight, about
  ``4 * C * (2 + t)`` bytes (C = 2**28 -> ~6 GiB at t = 4; a single tensor
  larger than C is a chunk of its own).  For the TP4/EP8/PP3 (world 24)
  Nemotron-3.5-Super recipe (~120B, MoE-dominated, ETP 1) n_r is about
  120e9 / 24 = 5e9: ~10 GB of anchors per rank, ~80 GB plus ~50 GB of chunks
  per 8-card host, beside the ~450 GB the offloaded masters and Adam moments
  take of its ~2 TB;
* rank 0, host additionally: the wire bytes of one pushed fragment
  (``2 * N / P`` for bf16) and the client's received broadcasts (each
  ``2 * N / P``; at most the syncer's ``--pipeline`` rounds in flight).  For
  N = 120e9 and P = 16 that is ~15 GB per fragment, ~45-60 GB peak.  A
  full-payload final cut holds every fragment in FP32 (``4 * N`` = 480 GB) on
  rank 0's host: run the syncer with ``--final-cut-payload manifest`` and
  ``final_cut_payload="manifest"`` for a 120B island.

Torch, Megatron-Core and the syncer client are imported lazily by the
objects that need them.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("swift-megatron-sync")

# Megatron's tied/untied output projection is an embedding-like tensor for the
# direct-averaged fragment, as ``lm_head`` is for the torch learner.
OUTPUT_LAYER_AVG_REGEX = r"(^|\.)output_layer\.weight$"
DEFAULT_CHUNK_NUMEL = 1 << 28
CONTRACT_DOMAIN = b"yeto-swift-megatron-native-v1\0"
WIRE_DTYPES = ("bf16", "f32", "fp32", "q4")
FINAL_CUT_PAYLOADS = ("full", "manifest")

_LAYER_MODULE_RE = re.compile(r"(^|\.)decoder\.layers\.(\d+)$")
_GROUPED_EXPERT_RE = re.compile(r"^(.*?)(weight|bias)(\d+)$")


class SwiftMegatronSyncError(RuntimeError):
    """The Megatron island cannot run the full-parameter DiLoCo sync as configured."""


# ----------------------------------------------------------------------------- parallel context
@dataclass(frozen=True)
class ParallelContext:
    """The process groups the sync needs, decoupled from ``parallel_state``
    so the sync runs on any group layout (tests build gloo subgroups).

    ``tp_*`` is the tensor-parallel group of dense parameters, ``etp_*`` the
    expert-tensor-parallel group of expert parameters (``param.allreduce is
    False``).  The data-parallel groups come from the optimizer
    (``data_parallel_group`` of each DistributedOptimizer leaf)."""

    rank: int
    world: int
    tp_group: Any
    tp_rank: int
    tp_size: int
    etp_group: Any
    etp_rank: int
    etp_size: int

    @classmethod
    def from_megatron(cls) -> ParallelContext:
        import torch.distributed as dist
        from megatron.core import parallel_state as mpu

        def expert_tp(name: str, fallback: Callable[[], Any]) -> Any:
            getter = getattr(mpu, name, None)
            return getter() if getter is not None else fallback()

        return cls(
            rank=int(dist.get_rank()),
            world=int(dist.get_world_size()),
            tp_group=mpu.get_tensor_model_parallel_group(),
            tp_rank=int(mpu.get_tensor_model_parallel_rank()),
            tp_size=int(mpu.get_tensor_model_parallel_world_size()),
            etp_group=expert_tp("get_expert_tensor_parallel_group", mpu.get_tensor_model_parallel_group),
            etp_rank=int(expert_tp("get_expert_tensor_parallel_rank", mpu.get_tensor_model_parallel_rank)),
            etp_size=int(expert_tp("get_expert_tensor_parallel_world_size",
                                   mpu.get_tensor_model_parallel_world_size)),
        )


# ----------------------------------------------------------------------------- global names
@dataclass
class LocalShard:
    """One local trainable parameter and where it sits in the global tensor."""

    name: str  # global Megatron name
    param: Any
    expert: bool
    part_dim: int  # TP partition dim, -1 when replicated over the TP group
    tp_rank: int
    tp_size: int
    tp_group: Any = field(repr=False, default=None)

    @property
    def local_shape(self) -> tuple[int, ...]:
        return tuple(int(d) for d in self.param.shape)

    def full_shape(self) -> tuple[int, ...]:
        shape = list(self.local_shape)
        if self.part_dim >= 0:
            shape[self.part_dim] *= self.tp_size
        return tuple(shape)


def unwrap_chunk(chunk: Any) -> Any:
    """The bare Megatron model of a (DDP / Float16Module wrapped) model chunk."""

    seen = 0
    while hasattr(chunk, "module") and seen < 8:
        chunk = chunk.module
        seen += 1
    return chunk


def _global_layer_prefixes(chunk: Any) -> list[tuple[str, str]]:
    """``(local prefix, global prefix)`` for every decoder layer of ``chunk``,
    longest first.  Megatron numbers ``decoder.layers`` per pipeline stage (and
    virtual stage); ``layer_number`` is the 1-based global index."""

    out = []
    for name, module in chunk.named_modules():
        match = _LAYER_MODULE_RE.search(name)
        if match is None or not hasattr(module, "layer_number"):
            continue
        global_name = name[: match.start(2)] + str(int(module.layer_number) - 1)
        out.append((name + ".", global_name + "."))
    return sorted(out, key=lambda item: -len(item[0]))


def _expert_modules(chunk: Any) -> list[tuple[str, list[int]]]:
    """``(prefix, local_expert_indices)`` of every expert container (the
    ``experts`` child of a MoE layer; the MoE layer itself also carries
    ``local_expert_indices`` but holds the router and shared experts),
    longest first."""

    out = []
    for name, module in chunk.named_modules():
        indices = getattr(module, "local_expert_indices", None)
        if indices is None or name.rsplit(".", 1)[-1] != "experts":
            continue
        out.append((name + ".", [int(i) for i in indices]))
    return sorted(out, key=lambda item: -len(item[0]))


def _globalize_expert(rest: str, indices: Sequence[int], name: str) -> str:
    """The global form of ``rest`` (the name below an expert container)."""

    head, _, tail = rest.partition(".")
    if head == "local_experts":  # SequentialMLP: local_experts.<e>.<...>
        local, _, below = tail.partition(".")
        if not local.isdigit():
            raise SwiftMegatronSyncError(f"unrecognised expert parameter {name!r}")
        return f"local_experts.{indices[int(local)]}.{below}"
    match = _GROUPED_EXPERT_RE.match(rest)
    if match is not None:  # TEGroupedMLP: linear_fc1.weight<e>
        return f"{match.group(1)}{match.group(2)}{indices[int(match.group(3))]}"
    raise SwiftMegatronSyncError(
        f"expert parameter {name!r} stacks several experts in one tensor (legacy GroupedMLP); "
        "use TEGroupedMLP or SequentialMLP")


def global_parameter_name(local_name: str, layers: Sequence[tuple[str, str]],
                          experts: Sequence[tuple[str, list[int]]]) -> str:
    """``local_name`` with its layer and expert indices made global."""

    name = local_name
    for prefix, indices in experts:
        if name.startswith(prefix):
            name = prefix + _globalize_expert(name[len(prefix):], indices, local_name)
            break
    for local, global_prefix in layers:
        if name.startswith(local):
            name = global_prefix + name[len(local):]
            break
    return name


def default_trainable_filter(name: str, param: Any) -> bool:
    return bool(getattr(param, "requires_grad", False))


def local_shards(model_chunks: Sequence[Any], ctx: ParallelContext,
                 trainable_filter: Callable[[str, Any], bool] | None = None) -> list[LocalShard]:
    """This rank's trainable parameters under their global names (sorted)."""

    keep = trainable_filter or default_trainable_filter
    out: dict[str, LocalShard] = {}
    seen: set[int] = set()
    for wrapped in model_chunks:
        chunk = unwrap_chunk(wrapped)
        layers = _global_layer_prefixes(chunk)
        experts = _expert_modules(chunk)
        for local_name, param in chunk.named_parameters():
            if id(param) in seen or not keep(local_name, param):
                continue
            seen.add(id(param))
            name = global_parameter_name(local_name, layers, experts)
            if name in out:
                raise SwiftMegatronSyncError(f"two local parameters map to the global name {name!r}")
            expert = getattr(param, "allreduce", True) is False
            tp_rank, tp_size, group = ((ctx.etp_rank, ctx.etp_size, ctx.etp_group) if expert
                                       else (ctx.tp_rank, ctx.tp_size, ctx.tp_group))
            part_dim = -1
            if getattr(param, "tensor_model_parallel", False) and tp_size > 1:
                if int(getattr(param, "partition_stride", 1)) != 1:
                    raise SwiftMegatronSyncError(f"{name}: strided TP partitions are not supported")
                part_dim = int(param.partition_dim)
            out[name] = LocalShard(name, param, expert, part_dim, tp_rank, tp_size, group)
    return [out[name] for name in sorted(out)]


@dataclass(frozen=True)
class GlobalTable:
    """Every global trainable tensor: its full shape and the rank that sends it
    (the lowest rank holding it at TP rank 0; pipeline copies of a tied
    embedding hold equal values, so any holder will do)."""

    shapes: dict[str, tuple[int, ...]]
    owners: dict[str, int]

    @classmethod
    def build(cls, shards: Sequence[LocalShard], group: Any = None) -> GlobalTable:
        import torch.distributed as dist

        rows = [(s.name, s.full_shape(), s.tp_rank) for s in shards]
        world = int(dist.get_world_size()) if dist.is_initialized() else 1
        if world > 1:
            gathered: list[Any] = [None] * world
            dist.all_gather_object(gathered, rows, group=group)
        else:
            gathered = [rows]
        shapes: dict[str, tuple[int, ...]] = {}
        owners: dict[str, int] = {}
        for holder, held in enumerate(gathered):
            for name, shape, tp_rank in held:
                shape = tuple(int(d) for d in shape)
                if shapes.setdefault(name, shape) != shape:
                    raise SwiftMegatronSyncError(
                        f"{name}: ranks disagree on the full shape ({shapes[name]} vs {shape})")
                if int(tp_rank) == 0 and (name not in owners or holder < owners[name]):
                    owners[name] = holder
        missing = sorted(set(shapes) - set(owners))
        if missing:
            raise SwiftMegatronSyncError(f"no TP-rank-0 holder for {missing[:4]}")
        return cls(shapes, owners)

    def numel(self, name: str) -> int:
        n = 1
        for d in self.shapes[name]:
            n *= int(d)
        return n


# ----------------------------------------------------------------------------- transport
TRANSPORTS = ("cpu", "device")


class Transport:
    """Where the sync's collectives run.

    ``"cpu"`` (default): host buffers over gloo mirrors of the island's groups
    (world, TP, expert-TP and every DistributedOptimizer leaf's data-parallel
    group), so the sync allocates nothing on the accelerator: the FP32
    masters already live on the host under CPU offload, and the only device
    traffic is the in-place host-to-device copy into the bf16 parameters.
    ``"device"``: the island's own (HCCL/NCCL) groups with accelerator buffers
    of at most one chunk, for islands with HBM to spare.

    Mirrors are created collectively (every rank creates every group, in one
    sorted order).  A gloo island needs no mirror."""

    def __init__(self, kind: str, ctx: ParallelContext, optimizer: Any, device: Any) -> None:
        import torch
        import torch.distributed as dist

        if kind not in TRANSPORTS:
            raise ValueError(f"transport must be one of {TRANSPORTS}, got {kind!r}")
        self.kind = kind
        self.buffer_device = torch.device("cpu") if kind == "cpu" else device
        self.mirrors: dict[tuple[int, ...], Any] | None = None
        self.world_ranks = tuple(range(ctx.world))
        if kind == "cpu" and dist.get_backend() != "gloo":
            from yeto.rl.megatron_masters import _optimizer_leaves

            groups = [None, ctx.tp_group, ctx.etp_group]
            groups += [getattr(leaf, "data_parallel_group", None) for leaf in _optimizer_leaves(optimizer)]
            mine = sorted({self._ranks(group) for group in groups})
            gathered: list[Any] = [None] * ctx.world
            dist.all_gather_object(gathered, mine)
            self.mirrors = {}
            for ranks in sorted({tuple(r) for held in gathered for r in held}):
                group = dist.new_group(list(ranks), backend="gloo")
                if ctx.rank in ranks:
                    self.mirrors[ranks] = group

    def _ranks(self, group: Any) -> tuple[int, ...]:
        import torch.distributed as dist

        if group is None or group is dist.GroupMember.WORLD:
            return self.world_ranks
        return tuple(int(r) for r in dist.get_process_group_ranks(group))

    def group(self, group: Any) -> Any:
        """The group to run a collective of ``group`` on."""

        if self.mirrors is None:
            return group
        return self.mirrors[self._ranks(group)]

    @property
    def world(self) -> Any:
        return self.group(None)

    def dp_reduce(self) -> Callable[[Any, Any], None] | None:
        """``full_masters``' data-parallel reduce on this transport (None: its own)."""

        if self.kind == "device":
            return None
        import torch.distributed as dist

        def reduce(flat: Any, leaf: Any) -> None:
            group = self.group(getattr(leaf, "data_parallel_group", None))
            if len(self._ranks(getattr(leaf, "data_parallel_group", None))) > 1:
                dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group)

        return reduce


# ----------------------------------------------------------------------------- helpers
def _wire_code(wire_dtype: str) -> int:
    from yeto.protocol import DTYPE_BF16, DTYPE_F32, DTYPE_Q4

    try:
        return {"bf16": DTYPE_BF16, "f32": DTYPE_F32, "fp32": DTYPE_F32, "q4": DTYPE_Q4}[wire_dtype]
    except KeyError:
        raise ValueError(f"wire_dtype must be one of {WIRE_DTYPES}, got {wire_dtype!r}") from None


def _torch_wire(code: int) -> Any:
    import torch

    from yeto.protocol import DTYPE_BF16, DTYPE_F32

    return {DTYPE_F32: torch.float32, DTYPE_BF16: torch.bfloat16}[code]


def _syncer_addrs(spec: str) -> list[tuple[str, int]]:
    from yeto.syncer_shards import parse_syncer_addrs

    items = [item.strip() for item in spec.split(",")]
    return parse_syncer_addrs(",".join(item.split("://", 1)[-1] for item in items))


def _read_hmac_key(path: str | None) -> bytes | None:
    """The island HMAC key file, with the encoding of ``YETO_ISLAND_HMAC_KEY``."""

    if path is None:
        return None
    with open(path, encoding="utf-8") as handle:
        value = handle.read().strip()
    if not value:
        raise SwiftMegatronSyncError(f"HMAC key file {path} is empty")
    return value.encode()


def chunk_names(names: Sequence[str], numel: Callable[[str], int], budget: int) -> list[list[str]]:
    """``names`` in order, cut into runs of at most ``budget`` elements (a
    larger tensor is a run of its own).  Pure, so every rank cuts alike."""

    out: list[list[str]] = []
    run: list[str] = []
    total = 0
    for name in names:
        n = numel(name)
        if run and total + n > budget:
            out.append(run)
            run, total = [], 0
        run.append(name)
        total += n
    if run:
        out.append(run)
    return out


@dataclass
class _Plan:
    """Rank 0's decision for one boundary, broadcast to every rank."""

    applies: list[tuple[int, int]] = field(default_factory=list)  # (fragment, version)
    pushes: list[tuple[int, int, int, int]] = field(default_factory=list)  # (fragment, step, attempt, c_units)
    finalizing: bool = False
    shutdown: bool = False
    error: str | None = None


# ----------------------------------------------------------------------------- the sync
class MegatronDiLoCoSync:
    """DiLoCo full-parameter sync of one ms-swift Megatron island (module doc).

    Construct it on every rank after the model, optimizer and resume checkpoint
    are loaded; it connects rank 0 to the syncer, and learner 0 seeds every
    fragment from its current FP32 masters (INIT).  Then call
    :meth:`after_optimizer_step` on every rank after every ``optimizer.step()``.

    ``model_chunks``: the model chunks (DDP-wrapped or bare; several with
    virtual pipeline parallelism).  ``optimizer``: the Megatron optimizer
    (DistributedOptimizer, ChainedOptimizer of dense + expert, precision-aware,
    CPU offload).  ``trainable_filter(local_name, param)``: which parameters
    take part (default ``requires_grad``).

    ``sync_interval_steps`` (H): a pull for a fragment is answered once the
    island has taken H optimizer steps since that fragment was last merged
    (H = 1 is the torch learner's self-clock: at least one step).  The steps
    are counted from the absolute ``iteration`` the trainer passes, so a
    resumed run keeps its rounds.

    ``merge_alpha``: local weight of a broadcast, as the torch learner's
    ``--merge-alpha`` (0 overwrites; the anchor is always the raw global).
    ``final_cut_payload``: what the syncer's ``--final-cut-payload`` sends;
    ``"full"`` overwrites the island with the final global, ``"manifest"``
    only acknowledges it (the final parameters stay with the syncer).
    """

    def __init__(self, model_chunks, optimizer, *, syncer: str, learner_id: int,
                 num_learners: int, fragments: int, sync_interval_steps: int,
                 wire_dtype: str = "bf16", hmac_key_path: str | None = None,
                 trainable_filter=None, merge_alpha: float = 0.0,
                 fragment_pattern: str = "binpack", wan_streams: int = 4,
                 final_cut_payload: str = "full", chunk_numel: int = DEFAULT_CHUNK_NUMEL,
                 transport: str = "cpu", parallel_context: ParallelContext | None = None,
                 client_factory=None) -> None:
        import torch
        import torch.distributed as dist

        from yeto.fragments import build_layout
        from yeto.protocol import bulk_dtype, layout_fingerprint

        if not 0 <= merge_alpha < 1:
            raise ValueError("merge_alpha must be in [0, 1)")
        if sync_interval_steps < 1 or fragments < 1 or num_learners < 1:
            raise ValueError("fragments, num_learners and sync_interval_steps must be >= 1")
        if not 0 <= learner_id < num_learners:
            raise ValueError(f"learner_id {learner_id} outside [0, {num_learners})")
        if final_cut_payload not in FINAL_CUT_PAYLOADS:
            raise ValueError(f"final_cut_payload must be one of {FINAL_CUT_PAYLOADS}")
        if not dist.is_initialized():
            raise SwiftMegatronSyncError("torch.distributed must be initialized (Megatron does this)")
        self.ctx = parallel_context or ParallelContext.from_megatron()
        self.rank, self.world = self.ctx.rank, self.ctx.world
        self.chunks = list(model_chunks)
        self.modules = [unwrap_chunk(chunk) for chunk in self.chunks]
        self.optimizer = optimizer
        self.learner_id = int(learner_id)
        self.interval = int(sync_interval_steps)
        self.merge_alpha = float(merge_alpha)
        self.final_cut_payload = final_cut_payload
        self.chunk_numel = int(chunk_numel)
        self.wire = _wire_code(wire_dtype)
        self.bulk = bulk_dtype(self.wire)
        self.bulk_torch = _torch_wire(self.bulk)
        self.anchor_dtype = self.bulk_torch  # the raw global is exact in the broadcast dtype

        self.shards = local_shards(self.chunks, self.ctx, trainable_filter)
        self.by_name: dict[str, list[LocalShard]] = {}
        for shard in self.shards:
            self.by_name.setdefault(shard.name, []).append(shard)
        devices = {shard.param.device for shard in self.shards}
        self.device = next(iter(devices)) if devices else torch.device("cpu")
        if dist.get_backend() != "gloo" and self.device.type == "cpu":
            # A rank with no trainable shard still joins device collectives.
            self.device = _current_accelerator()
        self.comm = Transport(transport, self.ctx, optimizer, self.device)
        self.table = GlobalTable.build(self.shards, self.comm.world)
        names = sorted(self.table.shapes)
        if not names:
            raise SwiftMegatronSyncError("no trainable parameter to synchronize")
        self.layout = build_layout(
            [(name, self.table.numel(name)) for name in names], int(fragments), fragment_pattern,
            avg_name_regex=OUTPUT_LAYER_AVG_REGEX,
            named_shapes={name: self.table.shapes[name] for name in names})
        self.num_fragments = self.layout.num_fragments
        self.fragment_chunks = [
            chunk_names([name for name, _ in frag.tensors], self.table.numel, self.chunk_numel)
            for frag in self.layout.fragments]
        self.contract = hashlib.sha256(
            CONTRACT_DOMAIN + layout_fingerprint(self.layout)
            + f"tp={self.ctx.tp_size};etp={self.ctx.etp_size}".encode()).digest()

        self.anchors: dict[int, Any] = {}  # id(param) -> raw global shard (host)
        self.anchored = [False] * self.num_fragments
        self.versions = [0] * self.num_fragments
        self.steps_at_reset: list[int] | None = None
        self.units_at_reset = [0] * self.num_fragments
        self.units_total = 0
        self.iteration = 0
        self.global_step = 0
        self.pending_pulls: list[Any] = []
        self._pull_meta: dict[tuple[int, int, int], int] = {}  # rank 0: c_steps of planned pushes
        self._updates: dict[int, Any] = {}  # rank 0: broadcasts drained for the current plan
        self._export_error: str | None = None  # rank 0: a non-finite value in the last export
        self.finished = False  # the final cut is installed and acknowledged
        self.shutdown = False
        self._closed = False
        self.client = None
        self.stats = {"applied": 0, "pushed": 0, "push_bytes": 0, "dropped_pushes": 0}
        # (iteration, "apply" | "push", fragment, version | pull step), same on every rank.
        self.events: deque[tuple[int, str, int, int]] = deque(maxlen=4096)

        if self.rank == 0:
            log.info("swift-megatron sync: %d trainable tensors (%.2f B elements) -> %d fragments",
                     len(names), sum(self.table.numel(n) for n in names) / 1e9, self.num_fragments)
        error = None
        if self.rank == 0:
            try:
                self.client = self._connect(syncer, hmac_key_path, wan_streams, client_factory)
            except Exception as exc:  # raised on every rank below
                error = f"{type(exc).__name__}: {exc}"
        self._raise_if(self._bcast_object(error))
        if self.learner_id == 0:
            self._send_init()

    # ------------------------------------------------------------------ setup
    def _connect(self, syncer: str, hmac_key_path: str | None, wan_streams: int, client_factory):
        from yeto.protocol import SyncerClient
        from yeto.syncer_shards import ShardedSyncerClient

        addrs = _syncer_addrs(syncer)
        kwargs = {"session_contract_hash": self.contract}
        key = _read_hmac_key(hmac_key_path)
        if key is not None:
            kwargs["hmac_key"] = key
        if client_factory is not None:
            client = client_factory(addrs, self.learner_id, self.layout, self.wire, wan_streams, **kwargs)
        elif len(addrs) > 1:
            client = ShardedSyncerClient(addrs, self.learner_id, self.layout, self.wire, wan_streams, **kwargs)
        else:
            client = SyncerClient(addrs[0], self.learner_id, self.layout, self.wire, wan_streams, **kwargs)
        client.start()
        log.info("connected to syncer at %s as learner %d", syncer, self.learner_id)
        return client

    def _send_init(self) -> None:
        """INIT_PARAMS of every fragment from this island's current masters."""

        error = None
        for fid in range(self.num_fragments):
            parts = self._export_fragment(fid, delta=False)
            if self.rank == 0 and error is None:
                try:
                    if self._export_error is not None:
                        raise SwiftMegatronSyncError(self._export_error)
                    self.client.send_init_parts(fid, parts)
                except Exception as exc:
                    error = f"INIT fragment {fid}: {type(exc).__name__}: {exc}"
        self._raise_if(self._bcast_object(error))
        if self.rank == 0:
            log.info("sent INIT_PARAMS for %d fragments", self.num_fragments)

    # ------------------------------------------------------------------ collectives
    def _bcast_object(self, value: Any) -> Any:
        import torch.distributed as dist

        if self.world == 1:
            return value
        box = [value]
        dist.broadcast_object_list(box, src=0, group=self.comm.world)
        return box[0]

    @staticmethod
    def _raise_if(error: str | None) -> None:
        if error:
            raise SwiftMegatronSyncError(f"syncer side failed on rank 0: {error}")

    def _masters(self, shards: Sequence[LocalShard]) -> list[Any]:
        from yeto.rl.megatron_masters import full_masters

        return full_masters(self.optimizer, [s.param for s in shards], device=self.comm.buffer_device,
                            reduce=self.comm.dp_reduce())

    def _export_chunk(self, names: Sequence[str], *, delta: bool) -> list[Any] | None:
        """Full FP32 values (``delta``: minus the anchor) of ``names``; the CPU
        tensors on rank 0, None elsewhere.  Collective (module doc)."""

        import torch
        import torch.distributed as dist

        wanted = set(names)
        local = [s for s in self.shards if s.name in wanted]
        masters = self._masters(local)
        values: dict[int, Any] = {}
        for i, (shard, master) in enumerate(zip(local, masters, strict=True)):
            value = master.detach().to(dtype=torch.float32)
            if delta:
                anchor = self.anchors.get(id(shard.param))
                if anchor is None:
                    raise SwiftMegatronSyncError(f"{shard.name} has no anchor yet")
                value = value - anchor.to(device=value.device, dtype=torch.float32)
            values[i] = value
        # TP gathers: dense first, then expert, each in name order, so no two
        # ranks wait on two groups in opposite orders.
        full: dict[str, Any] = {}
        for expert in (False, True):
            for i, shard in enumerate(local):
                if shard.expert is not expert:
                    continue
                value = values[i]
                if shard.part_dim >= 0:
                    parts = [torch.empty_like(value) for _ in range(shard.tp_size)]
                    dist.all_gather(parts, value.contiguous(), group=self.comm.group(shard.tp_group))
                    value = torch.cat(parts, dim=shard.part_dim) if shard.tp_rank == 0 else None
                    del parts
                if value is not None and self.table.owners[shard.name] == self.rank:
                    full.setdefault(shard.name, value)
        del values, masters
        out: list[Any] | None = [] if self.rank == 0 else None
        for name in names:
            owner = self.table.owners[name]
            if self.rank == 0:
                if owner == 0:
                    tensor = full.pop(name)
                else:
                    tensor = torch.empty(self.table.shapes[name], dtype=torch.float32,
                                         device=self.comm.buffer_device)
                    dist.recv(tensor, src=owner, group=self.comm.world)
                out.append(tensor.detach().to("cpu"))
            elif owner == self.rank:
                dist.send(full.pop(name).contiguous(), dst=0, group=self.comm.world)
        return out

    def _export_fragment(self, fid: int, *, delta: bool) -> list[Any] | None:
        """The fragment's values on rank 0 as wire byte parts (INIT, or PUSH
        for bf16/f32 sessions) or, for a q4 push, one f32 tensor."""

        from yeto.protocol import DTYPE_Q4
        from yeto.tensor_io import pack_tensor

        parts: list[Any] | None = [] if self.rank == 0 else None
        self._export_error = None
        q4 = delta and self.wire == DTYPE_Q4
        for names in self.fragment_chunks[fid]:
            tensors = self._export_chunk(names, delta=delta)
            if parts is None:
                continue
            for tensor in tensors:
                if self._export_error is None and not bool(tensor.isfinite().all()):
                    # Raised on every rank by the caller's error broadcast.
                    self._export_error = f"fragment {fid} holds a non-finite value"
                parts.append(tensor.reshape(-1) if q4 else
                             pack_tensor(tensor.reshape(-1), self.bulk if not delta else self._push_dtype()))
        return parts

    def _push_dtype(self) -> int:
        from yeto.protocol import push_wire_dtype

        return push_wire_dtype(self.wire)

    def _apply_chunk(self, names: Sequence[str], flat: Any | None, dtype: Any, *, blend: bool) -> None:
        """Install the raw global values of ``names`` (``flat`` on rank 0, in the
        broadcast dtype) as every rank's anchor, master and model copy."""

        import torch
        import torch.distributed as dist

        from yeto.rl.megatron_masters import has_complete_master, write_masters

        total = sum(self.table.numel(name) for name in names)
        if self.rank == 0:
            buffer = flat.to(device=self.comm.buffer_device)
        else:
            buffer = torch.empty(total, dtype=dtype, device=self.comm.buffer_device)
        if self.world > 1:
            dist.broadcast(buffer, src=0, group=self.comm.world)
        offsets, offset = {}, 0
        for name in names:
            offsets[name] = offset
            offset += self.table.numel(name)
        wanted = set(names)
        local = [s for s in self.shards if s.name in wanted]
        targets = []
        for shard in local:
            start = offsets[shard.name]
            whole = buffer[start:start + self.table.numel(shard.name)].view(self.table.shapes[shard.name])
            if shard.part_dim >= 0:
                size = shard.local_shape[shard.part_dim]
                whole = whole.narrow(shard.part_dim, shard.tp_rank * size, size)
            raw = whole.to(dtype=torch.float32).contiguous()
            self.anchors[id(shard.param)] = whole.to(device="cpu", dtype=self.anchor_dtype).contiguous().clone()
            targets.append(raw)
        if blend and self.merge_alpha > 0:
            masters = self._masters(local)
            targets = [self.merge_alpha * m.to(torch.float32) + (1.0 - self.merge_alpha) * t
                       for m, t in zip(masters, targets, strict=True)]
            del masters
        params = [s.param for s in local]
        with torch.no_grad():
            write_masters(self.optimizer, params, targets)
            for param, target in zip(params, targets, strict=True):
                if has_complete_master(param) and param.dtype != torch.float32:
                    param.data.copy_(target.to(dtype=param.dtype))

    def _apply_fragment(self, fid: int, data: Any, dtype: Any, *, blend: bool) -> None:
        """Apply one fragment's payload (``data`` on rank 0) chunk by chunk."""

        import torch

        view = memoryview(data).cast("B") if self.rank == 0 else None
        itemsize = torch.empty((), dtype=dtype).element_size()
        expected = self.layout.fragments[fid].numel * itemsize
        if view is not None and len(view) != expected:
            raise SwiftMegatronSyncError(f"fragment {fid} payload has {len(view)} bytes, expected {expected}")
        offset = 0
        for names in self.fragment_chunks[fid]:
            nbytes = sum(self.table.numel(name) for name in names) * itemsize
            flat = None
            if view is not None:
                flat = torch.frombuffer(bytearray(view[offset:offset + nbytes]), dtype=torch.uint8).view(dtype)
            self._apply_chunk(names, flat, dtype, blend=blend)
            offset += nbytes

    # ------------------------------------------------------------------ boundary
    def after_optimizer_step(self, iteration: int, target_tokens: int, *,
                             update_successful: bool = True) -> None:
        """Run one non-blocking sync boundary.  Every rank calls it after every
        ``optimizer.step()``, with the same arguments.

        ``iteration``: the trainer's absolute 1-based iteration (continues on
        resume); the H-step gate counts it.  ``target_tokens``: the loss
        tokens of this step over the whole island (the PUSH ``c_tokens``
        weight).  A skipped step (loss-scale overflow / NaN,
        ``update_successful=False``, ``target_tokens`` 0) still counts as an
        inner step, so rounds stay aligned with ``iteration``, but adds no
        token weight; its parameters did not move, so the next push carries
        the same delta it would have carried anyway."""

        if self.finished or self._closed:
            return
        iteration = int(iteration)
        if self.steps_at_reset is None:
            base = iteration - 1
            self.steps_at_reset = [base] * self.num_fragments
        self.iteration = iteration
        if update_successful:
            self.units_total += max(0, int(target_tokens))
        plan = self._plan() if self.rank == 0 else None
        plan = self._bcast_object(plan)
        self._raise_if(plan.error)
        if plan.finalizing:
            self._finalize()
            return
        self.shutdown = plan.shutdown
        updates = self._take_updates(plan) if self.rank == 0 else {}
        for fid, version in plan.applies:
            data = updates.get(fid) if self.rank == 0 else None
            self._apply_fragment(fid, data, self.bulk_torch, blend=True)
            self.steps_at_reset[fid] = iteration
            self.units_at_reset[fid] = self.units_total
            self.versions[fid] = version
            self.anchored[fid] = True
            self.global_step = max(self.global_step, version)
            self.stats["applied"] += 1
            self.events.append((iteration, "apply", fid, version))
        error = None
        for fid, step, attempt, c_units in plan.pushes:
            parts = self._export_fragment(fid, delta=True)
            self.events.append((iteration, "push", fid, step))
            if self.rank == 0 and error is None:
                error = self._push(fid, step, attempt, c_units, parts)
        if plan.pushes:
            self._raise_if(self._bcast_object(error))

    def _plan(self) -> _Plan:
        """Rank 0: drain the client and decide this boundary (never blocks)."""

        plan = _Plan()
        client = self.client
        try:
            client.check_health()
            plan.finalizing = bool(client.finalizing.is_set())
            if plan.finalizing:
                return plan
            self._updates = {}
            for update in client.drain_updates():
                # A later broadcast of the same fragment supersedes an earlier one.
                self._updates[update.fragment_id] = update
            plan.applies = sorted((fid, int(u.version)) for fid, u in self._updates.items())
            plan.shutdown = bool(client.shutdown.is_set())
            self.pending_pulls.extend(client.drain_pulls())
        except Exception as exc:
            plan.error = f"{type(exc).__name__}: {exc}"
            return plan
        reset = {fid for fid, _ in plan.applies}
        anchored = [self.anchored[f] or f in reset for f in range(self.num_fragments)]
        still = []
        for pull in self.pending_pulls:
            fid = int(pull.fragment_id)
            at_reset = self.iteration if fid in reset else self.steps_at_reset[fid]
            c_steps = self.iteration - at_reset
            if not anchored[fid] or c_steps < self.interval or any(p[0] == fid for p in plan.pushes):
                still.append(pull)
                continue
            units_at = self.units_total if fid in reset else self.units_at_reset[fid]
            plan.pushes.append((fid, int(pull.global_step), int(pull.round_attempt), self.units_total - units_at))
            self._pull_meta[(fid, int(pull.global_step), int(pull.round_attempt))] = c_steps
        self.pending_pulls = still
        return plan

    def _take_updates(self, plan: _Plan) -> dict[int, Any]:
        updates, self._updates = self._updates, {}
        return {fid: updates[fid].data for fid, _ in plan.applies}

    def _push(self, fid: int, step: int, attempt: int, c_units: int, parts: list[Any]) -> str | None:
        """Rank 0: send one PUSH_FRAGMENT.  A lost connection drops the push
        (the syncer re-requests it after the redial); anything else is fatal."""

        import torch

        from yeto.protocol import DTYPE_Q4
        from yeto.tensor_io import quantize_q4

        c_steps = self._pull_meta.pop((fid, step, attempt))
        if self._export_error is not None:
            return f"PUSH fragment {fid}: {self._export_error}"
        try:
            if self.wire == DTYPE_Q4:
                payload = quantize_q4(torch.cat(parts))
                self.client.push_fragment(fid, step, attempt, self.versions[fid], self.iteration,
                                          c_steps, c_units, payload)
                nbytes = len(payload)
            else:
                self.client.push_fragment_parts(fid, step, attempt, self.versions[fid], self.iteration,
                                                c_steps, c_units, parts)
                nbytes = sum(len(part) for part in parts)
        except (ConnectionError, OSError) as exc:
            self.stats["dropped_pushes"] += 1
            log.warning("push of fragment %d dropped (%s); the syncer re-requests it", fid, exc)
            return None
        except Exception as exc:
            return f"PUSH fragment {fid}: {type(exc).__name__}: {exc}"
        self.stats["pushed"] += 1
        self.stats["push_bytes"] += nbytes
        return None

    def _finalize(self) -> None:
        """Install the coordinator's final cut raw (``full``) or only confirm it
        (``manifest``), then FINAL_ACK; every rank returns together."""

        import torch
        import torch.distributed as dist

        from yeto.protocol import FinalManifest

        error, manifest, finals = None, None, {}
        if self.rank == 0:
            try:
                if self.final_cut_payload == "full":
                    manifest, broadcasts = self.client.wait_for_final_fragments()
                    finals = {update.fragment_id: update for update in broadcasts}
                else:
                    manifest = self.client.wait_for_final_manifest()
            except Exception as exc:
                error = f"finalization: {type(exc).__name__}: {exc}"
        box = self._bcast_object((error, None if manifest is None else (manifest.global_step,
                                                                         tuple(manifest.versions))))
        self._raise_if(box[0])
        global_step, versions = box[1]
        manifest = FinalManifest(global_step, tuple(versions))
        if self.final_cut_payload == "full":
            for fid in range(self.num_fragments):
                update = finals.get(fid) if self.rank == 0 else None
                if self.rank == 0 and update is not None and update.version != manifest.versions[fid]:
                    raise SwiftMegatronSyncError(f"final fragment {fid} has version {update.version}, "
                                                 f"expected {manifest.versions[fid]}")
                self._apply_fragment(fid, None if update is None else update.data, torch.float32, blend=False)
                self.versions[fid] = manifest.versions[fid]
        if self.world > 1:
            dist.barrier(group=self.comm.world)
        error = None
        if self.rank == 0:
            try:
                self.client.acknowledge_finalization(manifest)
            except Exception as exc:
                error = f"FINAL_ACK: {type(exc).__name__}: {exc}"
        self._raise_if(self._bcast_object(error))
        self.global_step = max(self.global_step, manifest.global_step)
        self.finished = True
        self.shutdown = True
        if self.rank == 0:
            log.info("final cut at global step %d installed and acknowledged", manifest.global_step)

    # ------------------------------------------------------------------ teardown
    def close(self) -> None:
        """Close the syncer connection (rank 0).  Idempotent, collective-free,
        safe from ``atexit`` after a crash."""

        if self._closed:
            return
        self._closed = True
        client, self.client = self.client, None
        if client is not None:
            try:
                client.close()
            except Exception as exc:  # teardown must not mask the original error
                log.warning("closing the syncer client failed: %s", exc)


def _current_accelerator() -> Any:
    import torch

    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            return torch.device("npu", torch.npu.current_device())
    except ImportError:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


__all__: Iterable[str] = (
    "GlobalTable",
    "LocalShard",
    "MegatronDiLoCoSync",
    "ParallelContext",
    "SwiftMegatronSyncError",
    "chunk_names",
    "global_parameter_name",
    "local_shards",
)
