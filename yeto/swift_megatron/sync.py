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

    The names are Megatron's, not the HF checkpoint's (the syncer is
    name-agnostic).  ms-swift's own converter, ``mcore_bridge``
    (``model.config.bridge``, NemotronHBridge), is deliberately not used for
    the gather and write: ``export_weights`` is a whole-model pass with no
    per-parameter selection, gathers every tensor over TP/EP/PP into
    accelerator buffers on every rank (``_all_gather_tp`` /
    ``_broadcast_ep_pp``) before ``target_device`` moves it, and reads the
    bf16 model copy, which is not the FP32 master and is stale outside the
    owned range under ``overlap_param_gather``; ``load_weights`` reads only a
    safetensors directory and writes the bf16 copy, not the optimizer master.
    With ~0.4 GiB of HBM headroom on the production pp0 stage none of that
    fits.  Megatron-Bridge (``yeto.rl.megatron_hf_boundary``) is not installed
    with ms-swift.  Everything protocol-side is the torch learner's own code
    (``yeto.diloco_sync``: session state, pull readiness, push encoding,
    metrics, client connection; ``yeto.fragments``; ``SyncerClient``).  Fused
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

Decoupled DiLoCo (no island waits for another)
    The island never waits for a merge or for another island: a boundary
    only drains what the client already received (broadcasts are applied at
    the next boundary after they arrive), the PUSH bytes are handed to a
    background sender thread on rank 0 (``_PushSender``; the client encodes
    int4/int8 with tensor blocks there), and while a PUSH is still streaming
    (slow link or syncer, a redial) pulls stay pending instead of blocking.
    One fragment is gathered per boundary.  A pull is not answered from an
    anchor already spent on another round (the syncer's late carry refuses
    that as ``anchor_reused``); a dropped PUSH un-spends its anchor.  The
    syncer side is the RL async full-parameter preset (quorum 1, adaptive
    grace, ``--max-base-lag 1``, tokens2-over-steps, two fragment rounds in
    flight, rda); ``wire_dtype="int4"`` is its int4 tensor-block push.

    What still blocks a step is the collective gather of one fragment's FP32
    masters (and the in-island broadcast of a received one): host copies,
    gloo all-reduce over DP and all-gather over TP, then point-to-point to
    rank 0.  For the 120B / 24-card recipe and P = 16 one fragment is
    ~7.5e9 elements, ~0.3e9 per rank (~1.2 GB FP32), 30 GB arriving at rank
    0: roughly 10-20 s over TCP gloo, against a ~570 s step, i.e. ~2-5 % at
    one push and one apply per step.  It is not overlapped with training: the
    HDO masters change under the next step, so overlapping needs a host
    snapshot of the local shards plus a second set of gloo groups driven by a
    background thread in the same order on every rank.

Memory (N = trainable parameters of the island, P = fragments, n_r = the
elements of rank r's local parameter shards, C = ``chunk_numel``, t = TP size):

* device: nothing with the default ``transport="cpu"`` (the collectives run
  on host buffers over gloo mirrors of the island's groups; the only device
  access is the in-place host-to-device copy into the existing bf16
  parameters).  The production pp0 stage has ~0.4 GiB of HBM headroom, so
  ``transport="device"`` (chunk buffers on the accelerator, peak about
  ``4 * C * (2 + t)`` bytes) is only for islands with room to spare;
* every rank, host: one anchor per local shard in the broadcast dtype (bf16
  for a bf16 session, FP32 for the int4 / int8 push sessions, whose
  broadcasts are FP32), ``2 * n_r`` or ``4 * n_r`` bytes, plus one chunk in
  flight, about
  ``4 * C * (2 + t)`` bytes (C = 2**28 -> ~6 GiB at t = 4; a single tensor
  larger than C is a chunk of its own).  For the TP4/EP8/PP3 (world 24)
  Nemotron-3.5-Super recipe (~120B, MoE-dominated, ETP 1) n_r is about
  120e9 / 24 = 5e9: ~10 GB (int4: ~20 GB) of anchors per rank, ~80 GB
  (int4: ~160 GB) plus ~50 GB of chunks
  per 8-card host, beside the ~450 GB the offloaded masters and Adam moments
  take of its ~2 TB;
* rank 0, host additionally: the wire bytes of one pushed fragment
  (``2 * N / P`` for bf16) and the client's received broadcasts (each
  ``2 * N / P``; at most the syncer's ``--pipeline`` rounds in flight).  For
  N = 120e9 and P = 16 that is ~15 GB per fragment, ~45-60 GB peak (int4:
  the FP32 PUSH parts and FP32 broadcasts, ~30 GB each, ~90-120 GB).  A
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
    ``experts`` child of a MoE layer), longest first.

    Older Megatron-Core put ``local_expert_indices`` on the expert container
    itself; since 0.1x only the MoE layer (its parent, which also holds the
    router and shared experts) carries it, so fall back to the parent."""

    modules = dict(chunk.named_modules())
    out = []
    for name, module in modules.items():
        if name.rsplit(".", 1)[-1] != "experts":
            continue
        indices = getattr(module, "local_expert_indices", None)
        if indices is None and "." in name:
            indices = getattr(modules.get(name.rsplit(".", 1)[0]), "local_expert_indices", None)
        if indices is None:
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
            if expert and ".experts." in local_name and not any(
                    local_name.startswith(prefix) for prefix, _ in experts):
                # Without its global expert index the tensor would alias the
                # same-named tensor of every other expert-parallel rank.
                raise SwiftMegatronSyncError(
                    f"{local_name}: expert parameter without local_expert_indices on its container")
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
def _torch_wire(code: int) -> Any:
    import torch

    from yeto.protocol import DTYPE_BF16, DTYPE_F32

    return {DTYPE_F32: torch.float32, DTYPE_BF16: torch.bfloat16}[code]


def _syncer_spec(spec: str) -> str:
    """``host:port[,host:port...]`` without ``tcp://`` schemes (the hook's
    ``YETO_DILOCO_SYNCER`` may carry one)."""

    return ",".join(item.strip().split("://", 1)[-1] for item in spec.split(","))


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
                 client_factory=None, observer: Callable[[dict], None] | None = None) -> None:
        import torch
        import torch.distributed as dist

        from yeto.diloco_sync import DiLoCoSyncState
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
        self.wire = session_dtype(wire_dtype)
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
        # (base version, round step) of each fragment's last sent PUSH: a
        # decoupled syncer refuses a second delta from a spent anchor for
        # another round (late carry, ``anchor_reused``), so such a pull waits
        # for a newer broadcast instead of a gather that would be refused.  A
        # restarted attempt of the same round may reuse it.
        self.spent: list[tuple[int, int] | None] = [None] * self.num_fragments
        # The torch learner's per-fragment counters and versions, kept alike on
        # every rank (anchors live in ``self.anchors``, sharded, not here).
        self.state = DiLoCoSyncState.create(self.num_fragments, track_anchors=False)
        self._counters_started = False
        self.units_total = 0
        self.iteration = 0
        self.observer = observer
        self._updates: dict[int, Any] = {}  # rank 0: broadcasts drained for the current plan
        self._export_error: str | None = None  # rank 0: a non-finite value in the last export
        self.finished = False  # the final cut is installed and acknowledged
        self._closed = False
        self.client = None
        self.stats = {"applied": 0, "pushed": 0, "push_bytes": 0, "dropped_pushes": 0,
                      "deferred_pushes": 0}
        self.sender: _PushSender | None = None  # rank 0: background PUSH streaming
        # (iteration, "apply" | "push", fragment, version | pull step), same on every rank.
        self.events: deque[tuple[int, str, int, int]] = deque(maxlen=4096)

        if self.rank == 0:
            log.info("swift-megatron sync: %d trainable tensors (%.2f B elements) -> %d fragments",
                     len(names), sum(self.table.numel(n) for n in names) / 1e9, self.num_fragments)
        error = None
        if self.rank == 0:
            try:
                self.client = self._connect(syncer, hmac_key_path, wan_streams, client_factory)
                self.sender = _PushSender(self.client, self.stats, self._push_dropped)
            except Exception as exc:  # raised on every rank below
                error = f"{type(exc).__name__}: {exc}"
        self._raise_if(self._bcast_object(error))
        if self.learner_id == 0:
            self._send_init()

    # ------------------------------------------------------------------ setup
    def _connect(self, syncer: str, hmac_key_path: str | None, wan_streams: int, client_factory):
        from yeto.diloco_sync import connect_syncer

        kwargs = {"session_contract_hash": self.contract}
        key = _read_hmac_key(hmac_key_path)
        if key is not None:
            kwargs["hmac_key"] = key
        connect = client_factory or connect_syncer
        client = connect(_syncer_spec(syncer), self.learner_id, self.layout, self.wire, wan_streams, **kwargs)
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
                             pack_tensor(tensor.reshape(-1), self.bulk if not delta else self._push_part_dtype()))
        return parts

    def _push_part_dtype(self) -> int:
        """The dtype of the PUSH parts handed to the client: f32 on the
        f32-bulk sessions (the client encodes bf16 / int8 / int4 itself, with
        tensor blocks), else the session's push wire dtype."""

        from yeto.protocol import DTYPE_F32, bulk_dtype, push_wire_dtype

        return DTYPE_F32 if bulk_dtype(self.wire) == DTYPE_F32 else push_wire_dtype(self.wire)

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

        from yeto.diloco_sync import boundary_metrics

        if self.finished or self._closed:
            return
        state = self.state
        iteration = int(iteration)
        if not self._counters_started:  # count from the (possibly resumed) first step
            state.steps_at_reset = [iteration - 1] * self.num_fragments
            self._counters_started = True
        self.iteration = iteration
        if update_successful:
            self.units_total += max(0, int(target_tokens))
        plan = self._plan() if self.rank == 0 else None
        plan = self._bcast_object(plan)
        self._raise_if(plan.error)
        if plan.finalizing:
            self._finalize()
            return
        state.shutdown = plan.shutdown
        # Apply broadcasts before answering pulls (as sync_diloco_boundary):
        # a next-round pull can overtake the previous broadcast.
        updates = self._take_updates(plan) if self.rank == 0 else {}
        for fid, version in plan.applies:
            data = updates.get(fid) if self.rank == 0 else None
            self._apply_fragment(fid, data, self.bulk_torch, blend=True)
            state.steps_at_reset[fid] = iteration
            state.units_at_reset[fid] = self.units_total
            state.fragment_versions[fid] = version
            self.anchored[fid] = True
            state.global_step = max(state.global_step, version)
            self.stats["applied"] += 1
            self.events.append((iteration, "apply", fid, version))
        pushes = self._ready_pushes() if self.rank == 0 else None
        pushes = self._bcast_object(pushes)
        if isinstance(pushes, str):
            self._raise_if(pushes)
        error = None
        for fid, step, attempt, c_steps, c_units in pushes:
            parts = self._export_fragment(fid, delta=True)
            self.events.append((iteration, "push", fid, step))
            if self.rank == 0 and error is None:
                error = self._push(fid, step, attempt, c_steps, c_units, parts)
        if pushes:
            self._raise_if(self._bcast_object(error))
        if self.observer is not None and self.rank == 0 and (plan.applies or pushes):
            self.observer(boundary_metrics(state, merges=len(plan.applies), pushes=len(pushes),
                                           push_bytes=self.stats["push_bytes"], push_delta_norm=0.0,
                                           steps_total=iteration))

    @property
    def shutdown(self) -> bool:
        return self.state.shutdown

    @property
    def global_step(self) -> int:
        return self.state.global_step

    def _plan(self) -> _Plan:
        """Rank 0: drain the client's broadcasts and flags (never blocks)."""

        plan = _Plan()
        client = self.client
        try:
            client.check_health()
            if self.sender is not None:
                self.sender.raise_error()
            plan.finalizing = bool(client.finalizing.is_set())
            if plan.finalizing:
                return plan
            self._updates = {}
            for update in client.drain_updates():
                # A later broadcast of the same fragment supersedes an earlier one.
                self._updates[update.fragment_id] = update
            plan.applies = sorted((fid, int(u.version)) for fid, u in self._updates.items())
            plan.shutdown = bool(client.shutdown.is_set())
        except Exception as exc:
            plan.error = f"{type(exc).__name__}: {exc}"
        return plan

    def _ready_pushes(self) -> list[tuple[int, int, int, int, int]] | str:
        """Rank 0: the pulls answered now, by the torch learner's rule
        (``take_ready_pulls``) with the H-step gate; one push per fragment
        per boundary, a second pull of it waits."""

        from yeto.diloco_sync import take_ready_pulls

        state = self.state
        try:
            state.pending_pulls.extend(self.client.drain_pulls())
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"
        if self.sender is not None and self.sender.busy():
            # Decoupled: the previous PUSH is still streaming (slow link or
            # syncer); every pull waits rather than the training step.
            if state.pending_pulls:
                self.stats["deferred_pushes"] += 1
            return []
        ready = take_ready_pulls(state, steps_total=self.iteration, units_total=self.units_total,
                                 anchored=self.anchored.__getitem__, min_steps=self.interval)
        # One fragment per boundary: its PUSH streams in the background while
        # training goes on, and the gather of a second would only queue
        # behind it on the host.
        out = []
        for pull, c_steps, c_units in ready:
            if out or not self._pushable(pull):
                state.pending_pulls.append(pull)
                continue
            out.append((int(pull.fragment_id), int(pull.global_step), int(pull.round_attempt), c_steps, c_units))
        return out

    def _pushable(self, pull: Any) -> bool:
        fid = int(pull.fragment_id)
        spent = self.spent[fid]
        return spent is None or spent[0] != self.state.fragment_versions[fid] or spent[1] == int(pull.global_step)

    def _push_dropped(self, fid: int, base: int, step: int) -> None:
        """Sender thread: a dropped PUSH never reached the syncer, so its anchor is not spent."""
        if self.spent[fid] == (base, step):
            self.spent[fid] = None

    def _take_updates(self, plan: _Plan) -> dict[int, Any]:
        updates, self._updates = self._updates, {}
        return {fid: updates[fid].data for fid, _ in plan.applies}

    def _push(self, fid: int, step: int, attempt: int, c_steps: int, c_units: int,
              parts: list[Any]) -> str | None:
        """Rank 0: hand one PUSH_FRAGMENT to the background sender and return
        at once (an error only for a fragment that cannot be pushed at all)."""

        import torch

        from yeto.diloco_sync import encode_push_delta
        from yeto.protocol import DTYPE_Q4

        if self._export_error is not None:
            return f"PUSH fragment {fid}: {self._export_error}"
        head = (fid, step, attempt, self.state.fragment_versions[fid], self.iteration, c_steps, c_units)
        self.spent[fid] = (self.state.fragment_versions[fid], step)
        if self.wire == DTYPE_Q4:
            # q4 blocks span tensors: the shared encoder takes the whole delta.
            self.sender.submit(head, payload=encode_push_delta(self.client, torch.cat(parts)))
        else:
            self.sender.submit(head, parts=parts)
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
                self.state.fragment_versions[fid] = manifest.versions[fid]
        if self.world > 1:
            dist.barrier(group=self.comm.world)
        error = None
        if self.rank == 0:
            try:
                self.client.acknowledge_finalization(manifest)
            except Exception as exc:
                error = f"FINAL_ACK: {type(exc).__name__}: {exc}"
        self._raise_if(self._bcast_object(error))
        self.state.global_step = max(self.state.global_step, manifest.global_step)
        self.finished = True
        self.state.shutdown = True
        if self.rank == 0:
            log.info("final cut at global step %d installed and acknowledged", manifest.global_step)

    # ------------------------------------------------------------------ teardown
    def close(self) -> None:
        """Close the syncer connection (rank 0).  Idempotent, collective-free,
        safe from ``atexit`` after a crash."""

        if self._closed:
            return
        self._closed = True
        if self.sender is not None:
            self.sender.stop()
        client, self.client = self.client, None
        if client is not None:
            try:
                client.close()
            except Exception as exc:  # teardown must not mask the original error
                log.warning("closing the syncer client failed: %s", exc)


def session_dtype(name: str) -> int:
    """The syncer session dtype of a wire name: the torch learner's
    ``--wire-dtype`` names (bf16, f32, q4), or the full-parameter push names
    of the RL island (``int4``, ``int8``, ``int4-log``, ``push-bf16``: f32
    INIT/BCAST, quantized PUSH deltas with blocks restarting at tensor
    boundaries, as the RL async full-parameter preset sends them)."""

    from yeto.diloco_sync import WIRE_DTYPES, wire_dtype_code
    from yeto.protocol import DTYPE_FLAG_TENSOR_BLOCKS, FULL_PARAMETER_PUSH_DTYPES

    if name in WIRE_DTYPES or name == "fp32":
        return wire_dtype_code(name)
    key = "bf16" if name == "push-bf16" else name
    if key not in FULL_PARAMETER_PUSH_DTYPES or key == "f32":
        raise ValueError(f"unknown wire dtype {name!r}: one of {sorted(WIRE_DTYPES)} or "
                         "int4, int8, int4-log, push-bf16")
    code = FULL_PARAMETER_PUSH_DTYPES[key]
    return code | DTYPE_FLAG_TENSOR_BLOCKS if key != "bf16" else code


class _PushSender:
    """Rank 0's background PUSH streaming: one PUSH in flight at a time, so a
    slow link or syncer never holds a training step (decoupled DiLoCo, as the
    RL full-parameter island's background push).  A lost connection drops the
    push (the syncer re-requests it after the redial); any other failure is
    re-raised by the next boundary on every rank."""

    def __init__(self, client: Any, stats: dict, on_drop: Callable[[int, int, int], None]) -> None:
        import queue
        import threading

        self.client, self.stats, self.on_drop = client, stats, on_drop
        self.jobs: queue.Queue = queue.Queue()
        self.idle = threading.Event()
        self.idle.set()
        self.error: str | None = None
        self.thread = threading.Thread(target=self._run, name="yeto-swift-push", daemon=True)
        self.thread.start()

    def busy(self) -> bool:
        return not self.idle.is_set()

    def submit(self, head: tuple, *, parts: list[Any] | None = None, payload: bytes | None = None) -> None:
        self.idle.clear()
        self.jobs.put((head, parts, payload))

    def raise_error(self) -> None:
        if self.error is not None:
            error, self.error = self.error, None
            raise SwiftMegatronSyncError(error)

    def wait_idle(self, timeout: float | None = None) -> bool:
        return self.idle.wait(timeout)

    def stop(self) -> None:
        self.jobs.put(None)
        self.thread.join(timeout=5.0)

    def _run(self) -> None:
        while True:
            job = self.jobs.get()
            if job is None:
                return
            head, parts, payload = job
            fid = head[0]
            try:
                if payload is not None:
                    self.client.push_fragment(*head, payload)
                    nbytes = len(payload)
                else:
                    self.client.push_fragment_parts(*head, parts)
                    nbytes = sum(len(part) for part in parts)
                self.stats["pushed"] += 1
                self.stats["push_bytes"] += nbytes
            except (ConnectionError, OSError) as exc:
                self.stats["dropped_pushes"] += 1
                self.on_drop(fid, head[3], head[1])
                log.warning("push of fragment %d dropped (%s); the syncer re-requests it", fid, exc)
            except Exception as exc:
                self.error = f"PUSH fragment {fid}: {type(exc).__name__}: {exc}"
            finally:
                del parts, payload, job
                if self.jobs.empty():
                    self.idle.set()


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
